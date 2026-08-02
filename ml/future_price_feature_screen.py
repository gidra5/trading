from __future__ import annotations

from dataclasses import dataclass
import json

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as functional


HISTORY_MINUTES = 360
FORECAST_MINUTES = 60
BASE_CHANNEL_NAMES = (
    "closeLogReturn",
    "openGapLogReturn",
    "upperWickLogRange",
    "lowerWickLogRange",
    "log1pVolume",
    "volumeZeroMask",
)
BASE_CHANNEL_COUNT = len(BASE_CHANNEL_NAMES)
BASE_COMPONENT_CONTRACT = "completed-minute-reconstructable-ohlcv-geometry-v1"


@dataclass(frozen=True)
class FeatureSpec:
    format: str
    history_minutes: int = HISTORY_MINUTES
    ma_periods: tuple[int, ...] = (5, 15, 60)
    volume_lookback: int = 60
    volume_epsilon: float = 1e-6

    @classmethod
    def from_config(cls, value: dict) -> FeatureSpec:
        allowed = {
            "format",
            "historyMinutes",
            "maPeriods",
            "volumeLookback",
            "volumeEpsilon",
        }
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"unknown feature-format settings: {sorted(unknown)}")
        result = cls(
            format=str(value.get("format", "")),
            history_minutes=int(value.get("historyMinutes", HISTORY_MINUTES)),
            ma_periods=tuple(int(period) for period in value.get(
                "maPeriods",
                (5, 15, 60),
            )),
            volume_lookback=int(value.get("volumeLookback", 60)),
            volume_epsilon=float(value.get("volumeEpsilon", 1e-6)),
        )
        result.validate()
        return result

    def validate(self) -> None:
        if self.history_minutes != HISTORY_MINUTES:
            raise ValueError("feature screen history must be exactly 360 minutes")
        if self.format not in {
            "close",
            "close_ma_bands",
            "close_ohlc_geometry",
            "close_volume_residual",
        }:
            raise ValueError(f"unsupported feature format: {self.format}")
        if tuple(sorted(set(self.ma_periods))) != self.ma_periods \
                or not self.ma_periods \
                or self.ma_periods[-1] > self.history_minutes \
                or self.ma_periods[0] < 1:
            raise ValueError("MA periods must be unique, increasing, and causal")
        if self.format == "close_ma_bands" \
                and self.ma_periods != (5, 15, 60):
            raise ValueError("the v1 additive MA screen uses periods 5/15/60")
        if self.volume_lookback < 2 \
                or self.volume_lookback > self.history_minutes \
                or self.volume_epsilon <= 0:
            raise ValueError("volume normalization settings are invalid")

    @property
    def required_base_minutes(self) -> int:
        if self.format == "close_ma_bands":
            return self.history_minutes + self.ma_periods[-1] - 1
        if self.format == "close_volume_residual":
            return self.history_minutes + self.volume_lookback
        return self.history_minutes

    @property
    def channel_names(self) -> tuple[str, ...]:
        if self.format == "close":
            return ("closeLogReturn",)
        if self.format == "close_ma_bands":
            return (
                "closeLogReturn",
                "slowMa60Anchored",
                "ma15MinusMa60",
                "ma5MinusMa15",
                "anchoredLogCloseMinusMa5",
            )
        if self.format == "close_ohlc_geometry":
            return BASE_CHANNEL_NAMES[:4]
        if self.format == "close_volume_residual":
            return (
                "closeLogReturn",
                "logVolumePrior60ZResidual",
                "volumeZeroMask",
            )
        raise AssertionError("feature format was not validated")

    @property
    def channel_count(self) -> int:
        return len(self.channel_names)

    @property
    def identity_normalized_channels(self) -> tuple[str, ...]:
        # A binary outage flag must remain 0/1. In this corpus training has no
        # zero-volume minutes while validation does; dividing by a clamped
        # training std would turn a valid validation flag into an artificial
        # million-scale value.
        return (
            ("volumeZeroMask",)
            if self.format == "close_volume_residual"
            else ()
        )

    @property
    def contract(self) -> str:
        settings = {
            "format": self.format,
            "historyMinutes": self.history_minutes,
            "maPeriods": list(self.ma_periods),
            "volumeLookback": self.volume_lookback,
            "volumeEpsilon": self.volume_epsilon,
            "channels": list(self.channel_names),
        }
        return "six-hour-causal-feature-screen-v1:" + json.dumps(
            settings,
            sort_keys=True,
            separators=(",", ":"),
        )


def _rolling_path_mean(
    path: np.ndarray,
    period: int,
    desired_start: int,
) -> np.ndarray:
    prefix = np.concatenate((
        np.zeros((path.shape[0], 1), dtype=np.float64),
        np.cumsum(path, axis=1, dtype=np.float64),
    ), axis=1)
    rolling = (prefix[:, period:] - prefix[:, :-period]) / period
    offset = desired_start - period + 1
    return rolling[:, offset:offset + HISTORY_MINUTES]


def build_causal_features(base_minutes: np.ndarray, spec: FeatureSpec) -> np.ndarray:
    """Build one causal [example, 360, channel] feature representation."""
    spec.validate()
    values = np.asarray(base_minutes, dtype=np.float32)
    expected = (spec.required_base_minutes, BASE_CHANNEL_COUNT)
    if values.ndim != 3 or values.shape[1:] != expected:
        raise ValueError(
            "base feature window must have shape "
            f"[example, {expected[0]}, {expected[1]}]"
        )
    if not np.isfinite(values).all():
        raise ValueError("base feature window contains non-finite values")
    start = values.shape[1] - spec.history_minutes
    close = values[:, start:, 0]
    if spec.format == "close":
        return close[:, :, None]
    if spec.format == "close_ohlc_geometry":
        return values[:, start:, :4]
    if spec.format == "close_volume_residual":
        lookback = spec.volume_lookback
        log_volume = values[:, :, 4].astype(np.float64, copy=False)
        prefix = np.concatenate((
            np.zeros((values.shape[0], 1), dtype=np.float64),
            np.cumsum(log_volume, axis=1, dtype=np.float64),
        ), axis=1)
        square_prefix = np.concatenate((
            np.zeros((values.shape[0], 1), dtype=np.float64),
            np.cumsum(np.square(log_volume), axis=1, dtype=np.float64),
        ), axis=1)
        indices = np.arange(start, values.shape[1], dtype=np.int64)
        prior_sum = prefix[:, indices] - prefix[:, indices - lookback]
        prior_square_sum = (
            square_prefix[:, indices] - square_prefix[:, indices - lookback]
        )
        prior_mean = prior_sum / lookback
        prior_variance = np.maximum(
            0,
            prior_square_sum / lookback - np.square(prior_mean),
        )
        residual = (
            log_volume[:, start:] - prior_mean
        ) / np.sqrt(prior_variance + spec.volume_epsilon)
        return np.stack((
            close,
            residual.astype(np.float32),
            values[:, start:, 5],
        ), axis=-1)
    if spec.format == "close_ma_bands":
        path = np.cumsum(values[:, :, 0], axis=1, dtype=np.float64)
        slow = _rolling_path_mean(path, 60, start)
        middle = _rolling_path_mean(path, 15, start)
        fast = _rolling_path_mean(path, 5, start)
        anchor = path[:, start - 1:start]
        desired_path = path[:, start:]
        bands = (
            slow - anchor,
            middle - slow,
            fast - middle,
            desired_path - fast,
        )
        return np.stack((
            close,
            *(band.astype(np.float32) for band in bands),
        ), axis=-1)
    raise AssertionError("feature format was not validated")


@dataclass(frozen=True)
class FeatureNormalization:
    input_mean: Tensor
    input_std: Tensor
    target_mean: Tensor
    target_std: Tensor

    def validate(self, spec: FeatureSpec) -> None:
        input_shape = (spec.history_minutes, spec.channel_count)
        if self.input_mean.shape != input_shape \
                or self.input_std.shape != input_shape:
            raise ValueError(f"input normalization must have shape {input_shape}")
        if self.target_mean.shape != (FORECAST_MINUTES,) \
                or self.target_std.shape != (FORECAST_MINUTES,):
            raise ValueError("target normalization must have shape [60]")
        for value in (
            self.input_mean,
            self.input_std,
            self.target_mean,
            self.target_std,
        ):
            if not bool(torch.isfinite(value).all()):
                raise ValueError("feature normalization must be finite")
        if bool((self.input_std <= 0).any()) \
                or bool((self.target_std <= 0).any()):
            raise ValueError("feature normalization scales must be positive")


def _causal_moving_average(values: Tensor, kernel_size: int) -> Tensor:
    if values.ndim != 3:
        raise ValueError("causal moving average expects [example, time, channel]")
    channels_first = values.transpose(1, 2)
    padded = functional.pad(
        channels_first,
        (kernel_size - 1, 0),
        mode="replicate",
    )
    return functional.avg_pool1d(
        padded,
        kernel_size=kernel_size,
        stride=1,
    ).transpose(1, 2)


class MultiFeatureRLinearCore(nn.Module):
    def __init__(
        self,
        channel_count: int,
        *,
        moving_average_kernel: int = 15,
    ) -> None:
        super().__init__()
        if moving_average_kernel < 1 \
                or moving_average_kernel > HISTORY_MINUTES:
            raise ValueError("linear moving-average kernel is invalid")
        self.channel_count = int(channel_count)
        self.moving_average_kernel = int(moving_average_kernel)
        flattened = HISTORY_MINUTES * channel_count
        self.trend_projection = nn.Linear(flattened, FORECAST_MINUTES)
        self.residual_projection = nn.Linear(flattened, FORECAST_MINUTES)
        self.summary_projection = nn.Linear(2 * channel_count, FORECAST_MINUTES)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for layer in (
            self.trend_projection,
            self.residual_projection,
            self.summary_projection,
        ):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, standardized_history: Tensor) -> Tensor:
        expected = (HISTORY_MINUTES, self.channel_count)
        if standardized_history.ndim != 3 \
                or standardized_history.shape[1:] != expected:
            raise ValueError(f"multifeature RLinear expects [example, {expected}]")
        local_mean = standardized_history.mean(dim=1, keepdim=True).detach()
        local_scale = standardized_history.var(
            dim=1,
            correction=0,
            keepdim=True,
        ).add(1e-5).sqrt().detach()
        normalized = (standardized_history - local_mean) / local_scale
        trend = _causal_moving_average(
            normalized,
            self.moving_average_kernel,
        )
        residual = normalized - trend
        summary = torch.cat((
            local_mean.squeeze(1),
            local_scale.log().squeeze(1),
        ), dim=-1)
        return (
            self.trend_projection(trend.flatten(1))
            + self.residual_projection(residual.flatten(1))
            + self.summary_projection(summary)
        )


class FeaturePatchGluBlock(nn.Module):
    def __init__(
        self,
        width: int,
        *,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.left_padding = (kernel_size - 1) * dilation
        self.normalization = nn.LayerNorm(width)
        self.temporal = nn.Conv1d(
            width,
            width,
            kernel_size=kernel_size,
            dilation=dilation,
            groups=width,
        )
        self.fused_value_gate = nn.Conv1d(width, 2 * width, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens: Tensor) -> Tensor:
        hidden = self.normalization(tokens.transpose(1, 2)).transpose(1, 2)
        hidden = self.temporal(functional.pad(hidden, (self.left_padding, 0)))
        values, gates = self.fused_value_gate(hidden).chunk(2, dim=1)
        return tokens + self.dropout(values * torch.sigmoid(gates))


class MultiFeaturePatchTcnCore(nn.Module):
    def __init__(
        self,
        channel_count: int,
        *,
        patch_size: int = 10,
        width: int = 48,
        dilations: tuple[int, ...] = (1, 2, 4, 8, 4),
        kernel_size: int = 3,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        if HISTORY_MINUTES % patch_size != 0:
            raise ValueError("patch size must divide the 360-minute history")
        if channel_count < 1 or width < 2 or not dilations:
            raise ValueError("patch TCN dimensions must be positive")
        self.channel_count = int(channel_count)
        self.patch_size = int(patch_size)
        self.width = int(width)
        self.token_count = HISTORY_MINUTES // patch_size
        self.patch_projection = nn.Conv1d(
            channel_count,
            width,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.position = nn.Parameter(torch.zeros(1, width, self.token_count))
        self.blocks = nn.ModuleList([
            FeaturePatchGluBlock(
                width,
                kernel_size=kernel_size,
                dilation=int(dilation),
                dropout=dropout,
            )
            for dilation in dilations
        ])
        self.output_norm = nn.LayerNorm(2 * width)
        self.output_projection = nn.Linear(2 * width, FORECAST_MINUTES)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.position, std=0.02)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def encode_tokens(self, standardized_history: Tensor) -> Tensor:
        expected = (HISTORY_MINUTES, self.channel_count)
        if standardized_history.ndim != 3 \
                or standardized_history.shape[1:] != expected:
            raise ValueError(f"feature patch TCN expects [example, {expected}]")
        tokens = self.patch_projection(standardized_history.transpose(1, 2))
        tokens = tokens + self.position.to(dtype=tokens.dtype)
        for block in self.blocks:
            tokens = block(tokens)
        return tokens

    def forward(self, standardized_history: Tensor) -> Tensor:
        tokens = self.encode_tokens(standardized_history)
        pooled = torch.cat((tokens[:, :, -1], tokens.mean(dim=-1)), dim=-1)
        return self.output_projection(self.output_norm(pooled))


class FeatureDenseGluBlock(nn.Module):
    def __init__(self, width: int, *, dropout: float) -> None:
        super().__init__()
        self.normalization = nn.LayerNorm(width)
        self.fused_value_gate = nn.Linear(width, 2 * width)
        self.output_projection = nn.Linear(width, width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden: Tensor) -> Tensor:
        values, gates = self.fused_value_gate(
            self.normalization(hidden)
        ).chunk(2, dim=-1)
        update = self.output_projection(values * torch.sigmoid(gates))
        return hidden + self.dropout(update)


class MultiFeaturePatchTideCore(nn.Module):
    """Position-preserving patch aggregation with a modest TiDE-style MLP."""

    def __init__(
        self,
        channel_count: int,
        *,
        patch_size: int = 10,
        width: int = 24,
        hidden_width: int = 96,
        depth: int = 2,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        if HISTORY_MINUTES % patch_size != 0:
            raise ValueError("patch size must divide the 360-minute history")
        if channel_count < 1 or width < 2 or hidden_width < 2 or depth < 1:
            raise ValueError("patch TiDE dimensions must be positive")
        if not 0 <= dropout < 1:
            raise ValueError("patch TiDE dropout must be in [0, 1)")
        self.channel_count = int(channel_count)
        self.patch_size = int(patch_size)
        self.width = int(width)
        self.token_count = HISTORY_MINUTES // patch_size
        self.patch_projection = nn.Conv1d(
            channel_count,
            width,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.position = nn.Parameter(torch.zeros(1, width, self.token_count))
        self.input_normalization = nn.LayerNorm(width * self.token_count)
        self.fused_input_value_gate = nn.Linear(
            width * self.token_count,
            2 * hidden_width,
        )
        self.blocks = nn.ModuleList([
            FeatureDenseGluBlock(hidden_width, dropout=dropout)
            for _ in range(depth)
        ])
        self.output_normalization = nn.LayerNorm(hidden_width)
        self.output_projection = nn.Linear(hidden_width, FORECAST_MINUTES)
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.position, std=0.02)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def encode_patches(self, standardized_history: Tensor) -> Tensor:
        expected = (HISTORY_MINUTES, self.channel_count)
        if standardized_history.ndim != 3 \
                or standardized_history.shape[1:] != expected:
            raise ValueError(f"feature patch TiDE expects [example, {expected}]")
        patches = self.patch_projection(standardized_history.transpose(1, 2))
        return patches + self.position.to(dtype=patches.dtype)

    def forward(self, standardized_history: Tensor) -> Tensor:
        # Flattening preserves the identity of every past patch. Unlike terminal
        # and mean pooling, the dense encoder can express arbitrary interactions
        # between coarse positions before decoding the complete future path.
        patches = self.encode_patches(standardized_history)
        flattened = patches.flatten(1)
        values, gates = self.fused_input_value_gate(
            self.input_normalization(flattened)
        ).chunk(2, dim=-1)
        hidden = self.dropout(values * torch.sigmoid(gates))
        for block in self.blocks:
            hidden = block(hidden)
        return self.output_projection(self.output_normalization(hidden))


class FeatureScreenPredictor(nn.Module):
    def __init__(
        self,
        core: nn.Module,
        spec: FeatureSpec,
        normalization: FeatureNormalization,
    ) -> None:
        super().__init__()
        normalization.validate(spec)
        self.core = core
        self.spec = spec
        self.register_buffer("input_mean", normalization.input_mean.float().clone())
        self.register_buffer("input_std", normalization.input_std.float().clone())
        self.register_buffer("target_mean", normalization.target_mean.float().clone())
        self.register_buffer("target_std", normalization.target_std.float().clone())

    def forward(self, history_features: Tensor) -> Tensor:
        expected = (self.spec.history_minutes, self.spec.channel_count)
        if history_features.ndim != 3 or history_features.shape[1:] != expected:
            raise ValueError(f"feature predictor expects [example, {expected}]")
        standardized = (history_features - self.input_mean) / self.input_std
        normalized_forecast = self.core(standardized)
        return normalized_forecast * self.target_std + self.target_mean


def build_feature_predictor(
    architecture: str,
    architecture_config: dict,
    spec: FeatureSpec,
    normalization: FeatureNormalization,
) -> FeatureScreenPredictor:
    if architecture == "rlinear_dlinear":
        unknown = set(architecture_config) - {"movingAverageKernel"}
        if unknown:
            raise ValueError(f"unknown feature RLinear settings: {sorted(unknown)}")
        core: nn.Module = MultiFeatureRLinearCore(
            spec.channel_count,
            moving_average_kernel=int(
                architecture_config.get("movingAverageKernel", 15)
            ),
        )
    elif architecture == "causal_patch_tcn":
        unknown = set(architecture_config) - {
            "patchSize",
            "width",
            "dilations",
            "kernelSize",
            "dropout",
        }
        if unknown:
            raise ValueError(f"unknown feature patch-TCN settings: {sorted(unknown)}")
        core = MultiFeaturePatchTcnCore(
            spec.channel_count,
            patch_size=int(architecture_config.get("patchSize", 10)),
            width=int(architecture_config.get("width", 48)),
            dilations=tuple(int(value) for value in architecture_config.get(
                "dilations",
                (1, 2, 4, 8, 4),
            )),
            kernel_size=int(architecture_config.get("kernelSize", 3)),
            dropout=float(architecture_config.get("dropout", 0.05)),
        )
    elif architecture == "patch_tide":
        unknown = set(architecture_config) - {
            "patchSize",
            "width",
            "hiddenWidth",
            "depth",
            "dropout",
        }
        if unknown:
            raise ValueError(f"unknown feature patch-TiDE settings: {sorted(unknown)}")
        core = MultiFeaturePatchTideCore(
            spec.channel_count,
            patch_size=int(architecture_config.get("patchSize", 10)),
            width=int(architecture_config.get("width", 24)),
            hidden_width=int(architecture_config.get("hiddenWidth", 96)),
            depth=int(architecture_config.get("depth", 2)),
            dropout=float(architecture_config.get("dropout", 0.05)),
        )
    else:
        raise ValueError(f"unsupported feature predictor: {architecture}")
    return FeatureScreenPredictor(core, spec, normalization)


def architecture_contract(
    architecture: str,
    architecture_config: dict,
    spec: FeatureSpec,
) -> str:
    if architecture == "rlinear_dlinear":
        model = (
            "channelwise-revin-causal-dlinear-"
            f"ma{int(architecture_config.get('movingAverageKernel', 15))}"
        )
    elif architecture == "causal_patch_tcn":
        dilations = "-".join(str(int(value)) for value in architecture_config.get(
            "dilations",
            (1, 2, 4, 8, 4),
        ))
        model = (
            f"multichannel-patch{int(architecture_config.get('patchSize', 10))}-"
            f"causal-fused-glu-tcn-w{int(architecture_config.get('width', 48))}-"
            f"d{dilations}"
        )
    elif architecture == "patch_tide":
        model = (
            f"multichannel-patch{int(architecture_config.get('patchSize', 10))}-"
            f"position-preserving-tide-w{int(architecture_config.get('width', 24))}-"
            f"h{int(architecture_config.get('hiddenWidth', 96))}-"
            f"depth{int(architecture_config.get('depth', 2))}-"
            f"dropout{float(architecture_config.get('dropout', 0.05)):.8g}-"
            "fused-glu"
        )
    else:
        raise ValueError(f"unsupported feature predictor: {architecture}")
    return f"{model}:{spec.contract}"


def parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())
