from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch
from torch import Tensor, nn

from next_return_dataset import HISTORY_RETURN_COUNT
from normalized_glu_next_return import NormalizedGluNextReturn
from next_return_sequence import SUMMARY_NAMES
from return_oracle_ce import (
    BRANCH_NORMALIZATION_INITIAL_RADIUS,
    LearnableCenteringNorm,
    dropout_gate_probability,
    fused_glu,
)


MOVING_AVERAGE_WINDOWS = (
    ("1m", 60),
    ("1h", 3_600),
    ("1d", 86_400),
    ("1w", 604_800),
    ("1M", 2_592_000),
    ("3M", 7_776_000),
)
WINDOW_SECONDS = dict(MOVING_AVERAGE_WINDOWS)
MAX_WINDOW_LABELS = tuple(label for label, _seconds in MOVING_AVERAGE_WINDOWS)
MODEL_VARIANTS = ("separate-components", "joint-input")
ARCHITECTURE_CONTRACT = (
    "telescoping-return-moving-average-two-layer-normalized-glu-v1"
)
MINIMUM_STD = 1e-10


def window_path_slug(max_window: str) -> str:
    """Return a Windows-safe slug that distinguishes minute from month."""
    if max_window not in WINDOW_SECONDS:
        raise ValueError(f"unsupported maximum moving-average window: {max_window}")
    return "1mo" if max_window == "1M" else max_window


def selected_windows(max_window: str) -> tuple[tuple[str, int], ...]:
    if max_window not in WINDOW_SECONDS:
        raise ValueError(f"unsupported maximum moving-average window: {max_window}")
    index = MAX_WINDOW_LABELS.index(max_window)
    return tuple(reversed(MOVING_AVERAGE_WINDOWS[:index + 1]))


def component_labels(max_window: str) -> tuple[str, ...]:
    windows = selected_windows(max_window)
    labels = [f"ma_{windows[0][0]}"]
    for larger, smaller in zip(windows, windows[1:], strict=False):
        labels.append(f"ma_{smaller[0]}_minus_ma_{larger[0]}")
    labels.append("return_minus_ma_1m")
    return tuple(labels)


def telescoping_components(
    raw_returns: np.ndarray,
    moving_averages: dict[str, np.ndarray],
    max_window: str,
) -> np.ndarray:
    if raw_returns.ndim != 1 or not np.isfinite(raw_returns).all():
        raise ValueError("raw returns must be one finite vector")
    windows = selected_windows(max_window)
    values: list[np.ndarray] = []
    largest = moving_averages[windows[0][0]]
    values.append(largest)
    for larger, smaller in zip(windows, windows[1:], strict=False):
        values.append(
            moving_averages[smaller[0]] - moving_averages[larger[0]]
        )
    values.append(raw_returns - moving_averages["1m"])
    if any(value.shape != raw_returns.shape for value in values):
        raise ValueError("moving-average components are misaligned")
    result = np.stack(values, axis=1).astype(np.float32, copy=False)
    # Define the stored high-frequency residual after rounding all coarser
    # bands. This preserves the telescoping identity in the actual float32
    # training representation instead of only in the intermediate float64
    # calculation.
    raw_float32 = raw_returns.astype(np.float32)
    result[:, -1] = raw_float32 - result[:, :-1].sum(
        axis=1, dtype=np.float32
    )
    reconstruction_error = np.abs(
        result.sum(axis=1, dtype=np.float32) - raw_float32
    )
    rounding_tolerance = (
        2 * np.finfo(np.float32).eps
        * np.abs(result).sum(axis=1, dtype=np.float32)
        + 1e-12
    )
    if bool((reconstruction_error > rounding_tolerance).any()):
        raise RuntimeError("telescoping components do not reconstruct returns")
    return result


@dataclass(frozen=True)
class MultiscaleNormalization:
    feature_mean: np.ndarray
    feature_std: np.ndarray
    component_target_mean: np.ndarray
    component_target_std: np.ndarray
    component_cumulative_mean: np.ndarray
    component_cumulative_std: np.ndarray
    raw_target_mean: np.ndarray
    raw_target_std: np.ndarray
    raw_summary_mean: np.ndarray
    raw_summary_std: np.ndarray

    @property
    def component_count(self) -> int:
        return int(self.feature_mean.shape[0])

    @property
    def horizon(self) -> int:
        return int(self.raw_target_mean.shape[0])

    def validate(self) -> None:
        components = self.component_count
        horizon = self.horizon
        expected = {
            "feature_mean": (components, HISTORY_RETURN_COUNT),
            "feature_std": (components, HISTORY_RETURN_COUNT),
            "component_target_mean": (components, horizon),
            "component_target_std": (components, horizon),
            "component_cumulative_mean": (components,),
            "component_cumulative_std": (components,),
            "raw_target_mean": (horizon,),
            "raw_target_std": (horizon,),
            "raw_summary_mean": (len(SUMMARY_NAMES),),
            "raw_summary_std": (len(SUMMARY_NAMES),),
        }
        for name, shape in expected.items():
            values = getattr(self, name)
            if values.shape != shape or not np.isfinite(values).all():
                raise ValueError(f"invalid multiscale normalization: {name}")
        for values in (
            self.feature_std,
            self.component_target_std,
            self.component_cumulative_std,
            self.raw_target_std,
            self.raw_summary_std,
        ):
            if bool((values <= 0).any()):
                raise ValueError("multiscale normalization scales must be positive")


class TrainingPositionGlu(nn.Module):
    """Normalized GLU for an arbitrary flat training-position input."""

    def __init__(
        self,
        feature_mean: Tensor,
        feature_std: Tensor,
        target_mean: Tensor,
        target_std: Tensor,
        *,
        widths: tuple[int, ...] = (512, 512),
        dropout: float = 0.05,
        dropout_rate: float = 0.5,
        initial_radius: float = BRANCH_NORMALIZATION_INITIAL_RADIUS,
        minimum_radius: float = 1e-4,
    ) -> None:
        super().__init__()
        if feature_mean.ndim != 1 or feature_std.shape != feature_mean.shape \
                or target_mean.ndim != 1 or target_std.shape != target_mean.shape \
                or not widths or any(width < 2 for width in widths):
            raise ValueError("training-position GLU shapes are invalid")
        for values in (feature_mean, feature_std, target_mean, target_std):
            if not bool(torch.isfinite(values).all()):
                raise ValueError("training-position GLU scales must be finite")
        if bool((feature_std <= 0).any()) or bool((target_std <= 0).any()):
            raise ValueError("training-position GLU scales must be positive")
        if not 0 <= dropout < 1 or not 0 <= dropout_rate <= 1:
            raise ValueError("dropout settings are invalid")
        self.widths = tuple(int(value) for value in widths)
        self.dropout_rate = float(dropout_rate)
        self.dropout_gate_probability = dropout_gate_probability(dropout_rate)
        self.register_buffer("feature_mean", feature_mean.float().clone())
        self.register_buffer("feature_std", feature_std.float().clone())
        self.register_buffer("target_mean", target_mean.float().clone())
        self.register_buffer("target_std", target_std.float().clone())
        all_widths = (feature_mean.numel(), *self.widths)
        self.layers = nn.ModuleList([
            nn.Linear(input_width, 2 * output_width)
            for input_width, output_width in zip(
                all_widths[:-1], all_widths[1:], strict=True
            )
        ])
        self.value_centering_normalizers = nn.ModuleList([
            LearnableCenteringNorm(
                width,
                denominator_family="sqrt",
                initial_scale=initial_radius,
                minimum_scale=minimum_radius,
            )
            for width in self.widths
        ])
        self.gate_centering_normalizers = nn.ModuleList([
            LearnableCenteringNorm(
                width,
                denominator_family="sqrt",
                initial_scale=initial_radius,
                minimum_scale=minimum_radius,
            )
            for width in self.widths
        ])
        self.value_norm_biases = nn.ParameterList([
            nn.Parameter(torch.zeros(width)) for width in self.widths
        ])
        self.gate_norm_biases = nn.ParameterList([
            nn.Parameter(torch.zeros(width)) for width in self.widths
        ])
        self.value_transforms = nn.ModuleList([
            nn.Linear(width, width, bias=False) for width in self.widths
        ])
        self.gate_transforms = nn.ModuleList([
            nn.Linear(width, width, bias=False) for width in self.widths
        ])
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(self.widths[-1], target_mean.numel())
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for layer in self.layers:
            nn.init.kaiming_normal_(layer.weight, nonlinearity="linear")
            nn.init.zeros_(layer.bias)
        for normalizer in (
            *self.value_centering_normalizers,
            *self.gate_centering_normalizers,
        ):
            normalizer.reset_parameters()
        for bias in (*self.value_norm_biases, *self.gate_norm_biases):
            nn.init.zeros_(bias)
        for transform in (*self.value_transforms, *self.gate_transforms):
            nn.init.eye_(transform.weight)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def muon_parameters(self) -> tuple[Tensor, ...]:
        return (
            *(layer.weight for layer in self.layers),
            *(layer.weight for layer in self.value_transforms),
            *(layer.weight for layer in self.gate_transforms),
        )

    def forward_standardized(self, features: Tensor) -> Tensor:
        if features.ndim != 2 or features.shape[1] != self.feature_mean.numel():
            raise ValueError("training-position GLU input shape is invalid")
        hidden = (features.float() - self.feature_mean) / self.feature_std
        intermittent_dropout = (
            self.training
            and self.dropout.p > 0
            and 0 < self.dropout_rate < 1
        )
        pass_gate = (
            torch.rand((), device=hidden.device) < self.dropout_gate_probability
            if intermittent_dropout else None
        )
        for values in zip(
            self.layers,
            self.value_centering_normalizers,
            self.gate_centering_normalizers,
            self.value_norm_biases,
            self.gate_norm_biases,
            self.value_transforms,
            self.gate_transforms,
            strict=True,
        ):
            hidden, _raw_value, _raw_gate = fused_glu(values[0](hidden), *values[1:])
            if self.training and self.dropout.p > 0:
                if self.dropout_rate >= 1:
                    hidden = self.dropout(hidden)
                elif intermittent_dropout:
                    layer_gate = (
                        torch.rand((), device=hidden.device)
                        < self.dropout_gate_probability
                    )
                    hidden = torch.where(
                        pass_gate & layer_gate, self.dropout(hidden), hidden
                    )
        return self.output(hidden)

    def forward(self, features: Tensor) -> Tensor:
        return self.forward_standardized(features) * self.target_std \
            + self.target_mean


class SeparateComponentGlu(nn.Module):
    def __init__(
        self,
        normalization: MultiscaleNormalization,
        **glu_options,
    ) -> None:
        super().__init__()
        normalization.validate()
        self.component_count = normalization.component_count
        self.horizon = normalization.horizon
        self.branches = nn.ModuleList([
            NormalizedGluNextReturn(
                torch.from_numpy(normalization.feature_mean[index]),
                torch.from_numpy(normalization.feature_std[index]),
                torch.from_numpy(normalization.component_target_mean[index]),
                torch.from_numpy(normalization.component_target_std[index]),
                input_normalization="training-position",
                **glu_options,
            )
            for index in range(self.component_count)
        ])

    def muon_parameters(self) -> tuple[Tensor, ...]:
        return tuple(
            parameter
            for branch in self.branches
            for parameter in branch.muon_parameters()
        )

    def forward_components(self, features: Tensor) -> Tensor:
        if features.ndim != 3 \
                or features.shape[1:] != (
                    self.component_count, HISTORY_RETURN_COUNT
                ):
            raise ValueError("separate-component input shape is invalid")
        return torch.stack([
            branch(features[:, index, :])
            for index, branch in enumerate(self.branches)
        ], dim=1)

    def forward(self, features: Tensor) -> Tensor:
        return self.forward_components(features).sum(dim=1)


class JointInputGlu(nn.Module):
    def __init__(
        self,
        normalization: MultiscaleNormalization,
        **glu_options,
    ) -> None:
        super().__init__()
        normalization.validate()
        self.component_count = normalization.component_count
        self.horizon = normalization.horizon
        self.predictor = TrainingPositionGlu(
            torch.from_numpy(normalization.feature_mean.reshape(-1)),
            torch.from_numpy(normalization.feature_std.reshape(-1)),
            torch.from_numpy(normalization.raw_target_mean),
            torch.from_numpy(normalization.raw_target_std),
            **glu_options,
        )

    def muon_parameters(self) -> tuple[Tensor, ...]:
        return self.predictor.muon_parameters()

    def forward(self, features: Tensor) -> Tensor:
        if features.ndim != 3 \
                or features.shape[1:] != (
                    self.component_count, HISTORY_RETURN_COUNT
                ):
            raise ValueError("joint multiscale input shape is invalid")
        return self.predictor(features.flatten(start_dim=1))


def optimizer_parameter_groups(
    model: SeparateComponentGlu | JointInputGlu,
) -> tuple[tuple[Tensor, ...], tuple[Tensor, ...]]:
    muon = tuple(model.muon_parameters())
    muon_ids = {id(parameter) for parameter in muon}
    trainable = tuple(
        parameter for parameter in model.parameters() if parameter.requires_grad
    )
    adamw = tuple(
        parameter for parameter in trainable if id(parameter) not in muon_ids
    )
    if not muon or not adamw or len(muon_ids) != len(muon):
        raise RuntimeError("invalid multiscale optimizer routing")
    if {id(value) for value in (*muon, *adamw)} != {
        id(value) for value in trainable
    } or any(parameter.ndim != 2 for parameter in muon):
        raise RuntimeError("multiscale optimizer routing is incomplete")
    return muon, adamw


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
