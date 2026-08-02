from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
import torch.nn.functional as functional


HISTORY_RETURN_COUNT = 60
FORECAST_RETURN_COUNT = 60
FEATURE_CONTRACT = (
    "past-60-completed-minute-close-log-returns-to-next-60-completed-"
    "minute-close-log-returns-v1"
)


@dataclass(frozen=True)
class ForecastNormalization:
    input_mean: Tensor
    input_std: Tensor
    target_mean: Tensor
    target_std: Tensor

    def validate(self) -> None:
        expected = (HISTORY_RETURN_COUNT,)
        for name, value in (
            ("input_mean", self.input_mean),
            ("input_std", self.input_std),
            ("target_mean", self.target_mean),
            ("target_std", self.target_std),
        ):
            if value.shape != expected or not bool(torch.isfinite(value).all()):
                raise ValueError(f"{name} must contain 60 finite values")
        if bool((self.input_std <= 0).any()) or bool((self.target_std <= 0).any()):
            raise ValueError("forecast normalization standard deviations must be positive")


def _moving_average(values: Tensor, kernel_size: int) -> Tensor:
    if values.ndim != 2:
        raise ValueError("moving average expects [example, time]")
    if kernel_size < 1 or kernel_size % 2 != 1:
        raise ValueError("moving-average kernel must be a positive odd integer")
    padding = kernel_size // 2
    padded = functional.pad(
        values.unsqueeze(1),
        (padding, padding),
        mode="replicate",
    )
    return functional.avg_pool1d(
        padded,
        kernel_size=kernel_size,
        stride=1,
    ).squeeze(1)


class RLinearDLinearCore(nn.Module):
    """A fast RevIN-style DLinear baseline for one return channel."""

    def __init__(self, moving_average_kernel: int = 15) -> None:
        super().__init__()
        if moving_average_kernel > HISTORY_RETURN_COUNT:
            raise ValueError("moving-average kernel exceeds the history length")
        self.moving_average_kernel = int(moving_average_kernel)
        self.trend_projection = nn.Linear(
            HISTORY_RETURN_COUNT,
            FORECAST_RETURN_COUNT,
        )
        self.residual_projection = nn.Linear(
            HISTORY_RETURN_COUNT,
            FORECAST_RETURN_COUNT,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # DLinear's average initialization begins as a stable local-level
        # forecast instead of a high-variance random extrapolation.
        initial = 1.0 / HISTORY_RETURN_COUNT
        nn.init.constant_(self.trend_projection.weight, initial)
        nn.init.constant_(self.residual_projection.weight, initial)
        nn.init.zeros_(self.trend_projection.bias)
        nn.init.zeros_(self.residual_projection.bias)

    def forward(self, standardized_history: Tensor) -> Tensor:
        if standardized_history.ndim != 2 \
                or standardized_history.shape[-1] != HISTORY_RETURN_COUNT:
            raise ValueError("RLinear/DLinear expects [example, 60] history")
        local_mean = standardized_history.mean(dim=-1, keepdim=True).detach()
        local_scale = standardized_history.var(
            dim=-1,
            correction=0,
            keepdim=True,
        ).add(1e-5).sqrt().detach()
        normalized = (standardized_history - local_mean) / local_scale
        trend = _moving_average(normalized, self.moving_average_kernel)
        residual = normalized - trend
        normalized_forecast = (
            self.trend_projection(trend)
            + self.residual_projection(residual)
        )
        return normalized_forecast * local_scale + local_mean


class CausalPatchGluBlock(nn.Module):
    def __init__(
        self,
        width: int,
        *,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if kernel_size < 2 or dilation < 1:
            raise ValueError("causal block kernel and dilation must be positive")
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


class CausalPatchTcnCore(nn.Module):
    """Learned non-overlapping aggregates followed by an exact causal TCN."""

    def __init__(
        self,
        *,
        patch_size: int = 5,
        width: int = 24,
        dilations: tuple[int, ...] = (1, 2, 4, 4),
        kernel_size: int = 3,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        if HISTORY_RETURN_COUNT % patch_size != 0:
            raise ValueError("patch size must divide the 60-return history")
        if width < 2 or not dilations:
            raise ValueError("patch TCN width and depth must be positive")
        self.patch_size = int(patch_size)
        self.width = int(width)
        self.token_count = HISTORY_RETURN_COUNT // patch_size
        self.patch_projection = nn.Conv1d(
            1,
            width,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.position = nn.Parameter(torch.zeros(1, width, self.token_count))
        self.blocks = nn.ModuleList([
            CausalPatchGluBlock(
                width,
                kernel_size=kernel_size,
                dilation=int(dilation),
                dropout=dropout,
            )
            for dilation in dilations
        ])
        self.output_norm = nn.LayerNorm(2 * width)
        self.output_projection = nn.Linear(2 * width, FORECAST_RETURN_COUNT)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.position, std=0.02)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def encode_tokens(self, standardized_history: Tensor) -> Tensor:
        if standardized_history.ndim != 2 \
                or standardized_history.shape[-1] != HISTORY_RETURN_COUNT:
            raise ValueError("causal patch TCN expects [example, 60] history")
        tokens = self.patch_projection(standardized_history.unsqueeze(1))
        tokens = tokens + self.position.to(dtype=tokens.dtype)
        for block in self.blocks:
            tokens = block(tokens)
        return tokens

    def forward(self, standardized_history: Tensor) -> Tensor:
        tokens = self.encode_tokens(standardized_history)
        pooled = torch.cat((tokens[:, :, -1], tokens.mean(dim=-1)), dim=-1)
        return self.output_projection(self.output_norm(pooled))


class CloseReturnPredictor(nn.Module):
    """Apply immutable training-only scaling around a predictor core."""

    def __init__(self, core: nn.Module, normalization: ForecastNormalization) -> None:
        super().__init__()
        normalization.validate()
        self.core = core
        self.register_buffer(
            "input_mean",
            normalization.input_mean.detach().float().clone(),
        )
        self.register_buffer(
            "input_std",
            normalization.input_std.detach().float().clone(),
        )
        self.register_buffer(
            "target_mean",
            normalization.target_mean.detach().float().clone(),
        )
        self.register_buffer(
            "target_std",
            normalization.target_std.detach().float().clone(),
        )

    def forward(self, history_log_returns: Tensor) -> Tensor:
        if history_log_returns.ndim != 2 \
                or history_log_returns.shape[-1] != HISTORY_RETURN_COUNT:
            raise ValueError("close-return predictor expects [example, 60] history")
        standardized = (
            history_log_returns - self.input_mean
        ) / self.input_std
        normalized_forecast = self.core(standardized)
        return normalized_forecast * self.target_std + self.target_mean


def build_close_return_predictor(
    architecture: str,
    normalization: ForecastNormalization,
    config: dict,
) -> CloseReturnPredictor:
    if architecture == "rlinear_dlinear":
        allowed = {"movingAverageKernel"}
        unknown = set(config) - allowed
        if unknown:
            raise ValueError(f"unknown RLinear/DLinear settings: {sorted(unknown)}")
        core: nn.Module = RLinearDLinearCore(
            moving_average_kernel=int(config.get("movingAverageKernel", 15)),
        )
    elif architecture == "causal_patch_tcn":
        allowed = {"patchSize", "width", "dilations", "kernelSize", "dropout"}
        unknown = set(config) - allowed
        if unknown:
            raise ValueError(f"unknown patch-TCN settings: {sorted(unknown)}")
        core = CausalPatchTcnCore(
            patch_size=int(config.get("patchSize", 5)),
            width=int(config.get("width", 24)),
            dilations=tuple(int(value) for value in config.get(
                "dilations",
                (1, 2, 4, 4),
            )),
            kernel_size=int(config.get("kernelSize", 3)),
            dropout=float(config.get("dropout", 0.05)),
        )
    else:
        raise ValueError(f"unsupported close-return predictor: {architecture}")
    return CloseReturnPredictor(core, normalization)


def normalized_forecast_loss(
    prediction: Tensor,
    target: Tensor,
    example_weights: Tensor,
    target_std: Tensor,
    *,
    objective: str,
    huber_delta: float,
) -> Tensor:
    if prediction.shape != target.shape \
            or prediction.ndim != 2 \
            or prediction.shape[-1] != FORECAST_RETURN_COUNT:
        raise ValueError("prediction and target must have shape [example, 60]")
    if example_weights.shape != (prediction.shape[0],):
        raise ValueError("forecast example weights must have shape [example]")
    if target_std.shape != (FORECAST_RETURN_COUNT,) \
            or bool((target_std <= 0).any()):
        raise ValueError("target standard deviations must contain 60 positive values")
    normalized_error = (prediction - target) / target_std
    if objective == "mse":
        element_loss = normalized_error.square()
    elif objective == "huber":
        if huber_delta <= 0:
            raise ValueError("Huber delta must be positive")
        element_loss = functional.huber_loss(
            normalized_error,
            torch.zeros_like(normalized_error),
            reduction="none",
            delta=huber_delta,
        )
    else:
        raise ValueError(f"unsupported forecast objective: {objective}")
    weights = example_weights.to(dtype=element_loss.dtype).unsqueeze(-1)
    denominator = weights.sum() * FORECAST_RETURN_COUNT
    if not bool(torch.isfinite(denominator)) or float(denominator) <= 0:
        raise ValueError("forecast example weights must have positive finite sum")
    return (element_loss * weights).sum() / denominator


def parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def architecture_contract(architecture: str, config: dict) -> str:
    if architecture == "rlinear_dlinear":
        return (
            "train-position-scaled-revin-dlinear-trend-residual-"
            f"ma{int(config.get('movingAverageKernel', 15))}-v1"
        )
    if architecture == "causal_patch_tcn":
        dilations = "-".join(str(int(value)) for value in config.get(
            "dilations",
            (1, 2, 4, 4),
        ))
        return (
            "train-position-scaled-learned-patch-causal-depthwise-fused-glu-"
            f"tcn-p{int(config.get('patchSize', 5))}-"
            f"w{int(config.get('width', 24))}-d{dilations}-v1"
        )
    raise ValueError(f"unsupported close-return predictor: {architecture}")
