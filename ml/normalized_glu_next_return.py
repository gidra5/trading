from __future__ import annotations

import torch
from torch import Tensor, nn

from next_return_dataset import HISTORY_RETURN_COUNT
from return_oracle_ce import (
    BRANCH_NORMALIZATION_INITIAL_RADIUS,
    LearnableCenteringNorm,
    dropout_gate_probability,
    fused_glu,
)


DEFAULT_WIDTHS = (512,)
ARCHITECTURE_CONTRACT = (
    "next-return-fused-glu-stack-independent-branch-centering-sqrt-"
    "learned-radius-full-a-post-bias-v2"
)


class NormalizedGluNextReturn(nn.Module):
    """A learned-radius normalized GLU stack adapted to T-return regression."""

    architecture_contract = ARCHITECTURE_CONTRACT

    def __init__(
        self,
        feature_mean: Tensor,
        feature_std: Tensor,
        target_mean: Tensor,
        target_std: Tensor,
        *,
        widths: tuple[int, ...] = DEFAULT_WIDTHS,
        dropout: float = 0.05,
        dropout_rate: float = 0.5,
        initial_radius: float = BRANCH_NORMALIZATION_INITIAL_RADIUS,
        minimum_radius: float = 1e-4,
    ) -> None:
        super().__init__()
        if feature_mean.shape != (HISTORY_RETURN_COUNT,) \
                or feature_std.shape != feature_mean.shape \
                or not bool(torch.isfinite(feature_mean).all()) \
                or not bool(torch.isfinite(feature_std).all()) \
                or bool((feature_std <= 0).any()):
            raise ValueError("feature normalization must contain 120 finite values")
        if target_mean.ndim > 1 or target_std.shape != target_mean.shape \
                or target_mean.numel() < 1 \
                or not bool(torch.isfinite(target_mean).all()) \
                or not bool(torch.isfinite(target_std).all()) \
                or bool((target_std <= 0).any()):
            raise ValueError("target normalization must contain T finite scales")
        if not widths or any(width < 2 for width in widths):
            raise ValueError(
                "next-return predictor requires one or more GLU layers "
                "with width at least two"
            )
        if not 0 <= dropout < 1 or not 0 <= dropout_rate <= 1:
            raise ValueError("dropout settings are invalid")
        self.widths = tuple(int(width) for width in widths)
        self.horizon_return_count = int(target_mean.numel())
        self.dropout_rate = float(dropout_rate)
        self.dropout_gate_probability = dropout_gate_probability(dropout_rate)
        self.register_buffer("feature_mean", feature_mean.float().clone())
        self.register_buffer("feature_std", feature_std.float().clone())
        target_shape = () if self.horizon_return_count == 1 else (-1,)
        self.register_buffer(
            "target_mean", target_mean.float().reshape(target_shape).clone()
        )
        self.register_buffer(
            "target_std", target_std.float().reshape(target_shape).clone()
        )

        all_widths = (HISTORY_RETURN_COUNT, *self.widths)
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
        self.output = nn.Linear(
            self.widths[-1], self.horizon_return_count
        )
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
        # Begin at the training-mean baseline. The sequence head learns on the
        # first update and then propagates task gradients into the GLU stack.
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def muon_parameters(self) -> tuple[Tensor, ...]:
        return (
            *(layer.weight for layer in self.layers),
            *(layer.weight for layer in self.value_transforms),
            *(layer.weight for layer in self.gate_transforms),
        )

    def forward_standardized(self, features: Tensor) -> Tensor:
        if features.ndim != 2 or features.shape[-1] != HISTORY_RETURN_COUNT:
            raise ValueError("normalized GLU expects [example, 120] returns")
        hidden = (features.float() - self.feature_mean) / self.feature_std
        intermittent_dropout = (
            self.training
            and self.dropout.p > 0
            and 0 < self.dropout_rate < 1
        )
        pass_gate = (
            torch.rand((), device=hidden.device) < self.dropout_gate_probability
            if intermittent_dropout
            else None
        )
        for (
            layer,
            value_normalizer,
            gate_normalizer,
            value_bias,
            gate_bias,
            value_transform,
            gate_transform,
        ) in zip(
            self.layers,
            self.value_centering_normalizers,
            self.gate_centering_normalizers,
            self.value_norm_biases,
            self.gate_norm_biases,
            self.value_transforms,
            self.gate_transforms,
            strict=True,
        ):
            hidden, _raw_value, _raw_gate = fused_glu(
                layer(hidden),
                value_normalizer,
                gate_normalizer,
                value_bias,
                gate_bias,
                value_transform,
                gate_transform,
            )
            if self.training and self.dropout.p > 0:
                if self.dropout_rate >= 1:
                    hidden = self.dropout(hidden)
                elif intermittent_dropout:
                    layer_gate = (
                        torch.rand((), device=hidden.device)
                        < self.dropout_gate_probability
                    )
                    hidden = torch.where(
                        pass_gate & layer_gate,
                        self.dropout(hidden),
                        hidden,
                    )
        result = self.output(hidden)
        return result.squeeze(-1) \
            if self.horizon_return_count == 1 else result

    def forward(self, features: Tensor) -> Tensor:
        standardized = self.forward_standardized(features)
        return standardized * self.target_std + self.target_mean


def optimizer_parameter_groups(
    model: NormalizedGluNextReturn,
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
        raise RuntimeError("invalid normalized GLU optimizer routing")
    if {id(value) for value in (*muon, *adamw)} != {
        id(value) for value in trainable
    } or any(parameter.ndim != 2 for parameter in muon):
        raise RuntimeError("normalized GLU optimizer routing is incomplete")
    return muon, adamw


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
