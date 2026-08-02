from __future__ import annotations

from dataclasses import dataclass
import json

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as functional


SOURCE_FUTURE_MINUTES = 60


@dataclass(frozen=True)
class ResolutionSpec:
    history_minutes: int
    candle_minutes: int
    target_steps: int

    @classmethod
    def from_config(cls, config: dict) -> ResolutionSpec:
        allowed = {"historyMinutes", "candleMinutes", "targetSteps"}
        unknown = set(config) - allowed
        if unknown:
            raise ValueError(f"unknown resolution settings: {sorted(unknown)}")
        spec = cls(
            history_minutes=int(config.get("historyMinutes", 0)),
            candle_minutes=int(config.get("candleMinutes", 0)),
            target_steps=int(config.get("targetSteps", 0)),
        )
        spec.validate()
        return spec

    @property
    def history_steps(self) -> int:
        return self.history_minutes // self.candle_minutes

    @property
    def target_minutes(self) -> int:
        return self.target_steps * self.candle_minutes

    @property
    def example_span_minutes(self) -> int:
        return self.history_minutes + self.target_minutes

    @property
    def contract(self) -> str:
        return "causal-aggregated-close-resolution-v1:" + json.dumps(
            {
                "aggregation": "sum-consecutive-completed-minute-log-returns",
                "candleMinutes": self.candle_minutes,
                "historyMinutes": self.history_minutes,
                "historySteps": self.history_steps,
                "targetMinutes": self.target_minutes,
                "targetSteps": self.target_steps,
            },
            sort_keys=True,
            separators=(",", ":"),
        )

    def validate(self) -> None:
        if self.history_minutes < 1 \
                or self.candle_minutes < 1 \
                or self.target_steps < 1:
            raise ValueError("resolution dimensions must be positive")
        if self.history_minutes % self.candle_minutes != 0:
            raise ValueError("history must contain complete target-resolution candles")
        if self.target_minutes > SOURCE_FUTURE_MINUTES:
            raise ValueError("resolution target escapes the assigned future hour")


@dataclass(frozen=True)
class ResolutionNormalization:
    input_mean: Tensor
    input_std: Tensor
    target_mean: Tensor
    target_std: Tensor

    def validate(self, spec: ResolutionSpec) -> None:
        expected = {
            "input_mean": (spec.history_steps,),
            "input_std": (spec.history_steps,),
            "target_mean": (spec.target_steps,),
            "target_std": (spec.target_steps,),
        }
        for name, shape in expected.items():
            value = getattr(self, name)
            if value.shape != shape \
                    or not bool(torch.isfinite(value).all()):
                raise ValueError(f"resolution normalization {name} must be {shape}")
        if bool((self.input_std <= 0).any()) \
                or bool((self.target_std <= 0).any()):
            raise ValueError("resolution normalization scales must be positive")


def aggregate_minute_log_returns(
    minute_returns: np.ndarray,
    candle_minutes: int,
) -> np.ndarray:
    if minute_returns.ndim < 1 \
            or candle_minutes < 1 \
            or minute_returns.shape[-1] % candle_minutes != 0:
        raise ValueError("minute returns do not form complete aggregate candles")
    if not np.isfinite(minute_returns).all():
        raise ValueError("minute returns must be finite")
    shape = (
        *minute_returns.shape[:-1],
        minute_returns.shape[-1] // candle_minutes,
        candle_minutes,
    )
    return minute_returns.reshape(shape).sum(axis=-1, dtype=np.float64).astype(
        np.float32
    )


def build_resolution_examples(
    history_minute_returns: np.ndarray,
    assigned_future_hour_returns: np.ndarray,
    spec: ResolutionSpec,
) -> tuple[np.ndarray, np.ndarray]:
    spec.validate()
    if history_minute_returns.ndim != 2 \
            or history_minute_returns.shape[-1] != spec.history_minutes:
        raise ValueError("resolution history minute window is misaligned")
    if assigned_future_hour_returns.shape != (
        history_minute_returns.shape[0],
        SOURCE_FUTURE_MINUTES,
    ):
        raise ValueError("resolution assigned future hour is misaligned")
    history = aggregate_minute_log_returns(
        history_minute_returns,
        spec.candle_minutes,
    )
    target = aggregate_minute_log_returns(
        assigned_future_hour_returns[:, :spec.target_minutes],
        spec.candle_minutes,
    )
    if history.shape != (history_minute_returns.shape[0], spec.history_steps) \
            or target.shape != (
                history_minute_returns.shape[0],
                spec.target_steps,
            ):
        raise RuntimeError("resolution aggregation produced an invalid shape")
    return history, target


class ResolutionPatchGluBlock(nn.Module):
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


class ResolutionPatchTcnCore(nn.Module):
    def __init__(
        self,
        spec: ResolutionSpec,
        *,
        patch_size: int,
        width: int,
        dilations: tuple[int, ...],
        kernel_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if spec.history_steps % patch_size != 0:
            raise ValueError("patch size must divide aggregate history steps")
        if width < 2 or not dilations or kernel_size < 1:
            raise ValueError("resolution patch-TCN dimensions are invalid")
        if not 0 <= dropout < 1:
            raise ValueError("resolution patch-TCN dropout is invalid")
        self.spec = spec
        self.patch_size = int(patch_size)
        self.width = int(width)
        self.token_count = spec.history_steps // patch_size
        self.patch_projection = nn.Conv1d(
            1,
            width,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.position = nn.Parameter(torch.zeros(1, width, self.token_count))
        self.blocks = nn.ModuleList([
            ResolutionPatchGluBlock(
                width,
                kernel_size=kernel_size,
                dilation=int(dilation),
                dropout=dropout,
            )
            for dilation in dilations
        ])
        self.output_normalization = nn.LayerNorm(2 * width)
        self.output_projection = nn.Linear(2 * width, spec.target_steps)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.position, std=0.02)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def encode_tokens(self, history: Tensor) -> Tensor:
        if history.ndim != 2 \
                or history.shape[-1] != self.spec.history_steps:
            raise ValueError("resolution patch-TCN history shape changed")
        tokens = self.patch_projection(history.unsqueeze(1))
        tokens = tokens + self.position.to(dtype=tokens.dtype)
        for block in self.blocks:
            tokens = block(tokens)
        return tokens

    def forward(self, history: Tensor) -> Tensor:
        tokens = self.encode_tokens(history)
        pooled = torch.cat((tokens[:, :, -1], tokens.mean(dim=-1)), dim=-1)
        return self.output_projection(self.output_normalization(pooled))


class ResolutionDenseGluBlock(nn.Module):
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
        return hidden + self.dropout(
            self.output_projection(values * torch.sigmoid(gates))
        )


class ResolutionPatchTideCore(nn.Module):
    def __init__(
        self,
        spec: ResolutionSpec,
        *,
        patch_size: int,
        width: int,
        hidden_width: int,
        depth: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if spec.history_steps % patch_size != 0:
            raise ValueError("patch size must divide aggregate history steps")
        if width < 2 or hidden_width < 2 or depth < 1:
            raise ValueError("resolution patch-TiDE dimensions are invalid")
        if not 0 <= dropout < 1:
            raise ValueError("resolution patch-TiDE dropout is invalid")
        self.spec = spec
        self.patch_size = int(patch_size)
        self.width = int(width)
        self.token_count = spec.history_steps // patch_size
        self.patch_projection = nn.Conv1d(
            1,
            width,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.position = nn.Parameter(torch.zeros(1, width, self.token_count))
        flattened_width = width * self.token_count
        self.input_normalization = nn.LayerNorm(flattened_width)
        self.fused_input_value_gate = nn.Linear(
            flattened_width,
            2 * hidden_width,
        )
        self.blocks = nn.ModuleList([
            ResolutionDenseGluBlock(hidden_width, dropout=dropout)
            for _ in range(depth)
        ])
        self.output_normalization = nn.LayerNorm(hidden_width)
        self.output_projection = nn.Linear(hidden_width, spec.target_steps)
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.position, std=0.02)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, history: Tensor) -> Tensor:
        if history.ndim != 2 \
                or history.shape[-1] != self.spec.history_steps:
            raise ValueError("resolution patch-TiDE history shape changed")
        patches = self.patch_projection(history.unsqueeze(1))
        patches = patches + self.position.to(dtype=patches.dtype)
        flattened = patches.flatten(1)
        values, gates = self.fused_input_value_gate(
            self.input_normalization(flattened)
        ).chunk(2, dim=-1)
        hidden = self.dropout(values * torch.sigmoid(gates))
        for block in self.blocks:
            hidden = block(hidden)
        return self.output_projection(self.output_normalization(hidden))


class ResolutionPredictor(nn.Module):
    def __init__(
        self,
        core: nn.Module,
        spec: ResolutionSpec,
        normalization: ResolutionNormalization,
    ) -> None:
        super().__init__()
        normalization.validate(spec)
        self.core = core
        self.spec = spec
        self.register_buffer("input_mean", normalization.input_mean.float().clone())
        self.register_buffer("input_std", normalization.input_std.float().clone())
        self.register_buffer("target_mean", normalization.target_mean.float().clone())
        self.register_buffer("target_std", normalization.target_std.float().clone())

    def forward(self, history: Tensor) -> Tensor:
        if history.ndim != 2 or history.shape[-1] != self.spec.history_steps:
            raise ValueError("resolution predictor history shape changed")
        standardized = (history - self.input_mean) / self.input_std
        normalized_target = self.core(standardized)
        return normalized_target * self.target_std + self.target_mean


def build_resolution_predictor(
    architecture: str,
    config: dict,
    spec: ResolutionSpec,
    normalization: ResolutionNormalization,
) -> ResolutionPredictor:
    if architecture == "causal_patch_tcn":
        allowed = {"patchSize", "width", "dilations", "kernelSize", "dropout"}
        unknown = set(config) - allowed
        if unknown:
            raise ValueError(f"unknown resolution patch-TCN settings: {sorted(unknown)}")
        core: nn.Module = ResolutionPatchTcnCore(
            spec,
            patch_size=int(config.get("patchSize", 3)),
            width=int(config.get("width", 48)),
            dilations=tuple(int(value) for value in config.get(
                "dilations",
                (1, 2, 4, 8, 4),
            )),
            kernel_size=int(config.get("kernelSize", 3)),
            dropout=float(config.get("dropout", 0.05)),
        )
    elif architecture == "patch_tide":
        allowed = {"patchSize", "width", "hiddenWidth", "depth", "dropout"}
        unknown = set(config) - allowed
        if unknown:
            raise ValueError(f"unknown resolution patch-TiDE settings: {sorted(unknown)}")
        core = ResolutionPatchTideCore(
            spec,
            patch_size=int(config.get("patchSize", 3)),
            width=int(config.get("width", 16)),
            hidden_width=int(config.get("hiddenWidth", 64)),
            depth=int(config.get("depth", 2)),
            dropout=float(config.get("dropout", 0.05)),
        )
    else:
        raise ValueError(f"unsupported resolution predictor: {architecture}")
    return ResolutionPredictor(core, spec, normalization)


def architecture_contract(
    architecture: str,
    config: dict,
    spec: ResolutionSpec,
) -> str:
    if architecture == "causal_patch_tcn":
        dilations = "-".join(str(int(value)) for value in config.get(
            "dilations",
            (1, 2, 4, 8, 4),
        ))
        model = (
            f"aggregate-patch{int(config.get('patchSize', 3))}-"
            f"causal-fused-glu-tcn-w{int(config.get('width', 48))}-"
            f"d{dilations}-k{int(config.get('kernelSize', 3))}-"
            f"dropout{float(config.get('dropout', 0.05)):.8g}"
        )
    elif architecture == "patch_tide":
        model = (
            f"aggregate-patch{int(config.get('patchSize', 3))}-"
            f"position-preserving-tide-w{int(config.get('width', 16))}-"
            f"h{int(config.get('hiddenWidth', 64))}-"
            f"depth{int(config.get('depth', 2))}-"
            f"dropout{float(config.get('dropout', 0.05)):.8g}-fused-glu"
        )
    else:
        raise ValueError(f"unsupported resolution predictor: {architecture}")
    return f"{model}:{spec.contract}"


def parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())
