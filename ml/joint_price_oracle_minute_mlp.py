from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn


CONTEXT_SECONDS = 3_600
INPUT_CLOSE_COUNT = CONTEXT_SECONDS + 1
SECONDS_PER_MINUTE = 60
MINUTE_RETURN_COUNT = CONTEXT_SECONDS // SECONDS_PER_MINUTE
OUTPUT_ACTION_COUNT = 101
FIXED_MINUTE_RETURN_SCALE = 1e-3
SCALE_FEATURE_NAMES = (
    "mean",
    "rms",
    "standard_deviation",
    "mean_absolute",
    "total_over_sqrt_count",
    "last_minute",
    "recent_5m_mean",
    "recent_15m_mean",
)
SCALE_FEATURE_COUNT = len(SCALE_FEATURE_NAMES)
INPUT_FEATURE_COUNT = MINUTE_RETURN_COUNT + SCALE_FEATURE_COUNT
ARCHITECTURE_CONTRACT = (
    "causal-3600s-to-60-completed-minute-fixed-scale-return-"
    "fused-glu-residual-mlp-direct-101-policy-v1"
)


def _is_graph_capture() -> bool:
    """Return whether data-dependent Python validation must be skipped."""
    if torch.jit.is_scripting() or torch.jit.is_tracing():
        return True
    if torch.onnx.is_in_onnx_export():
        return True
    compiler = getattr(torch, "compiler", None)
    return bool(compiler is not None and compiler.is_compiling())


def _validate_closes(closes: Tensor) -> None:
    if closes.ndim != 3 \
            or closes.shape[1] != INPUT_CLOSE_COUNT \
            or closes.shape[2] != 1:
        raise ValueError(
            f"closes must have shape [batch, {INPUT_CLOSE_COUNT}, 1]"
        )
    if not closes.is_floating_point():
        raise TypeError("closes must use a floating-point dtype")
    # These checks provide a clear failure in eager training and inference.
    # They are skipped while capturing an ONNX/compiled graph because valid
    # production inputs are already contract-checked by the caller and a
    # tensor-to-Python branch would otherwise be frozen into the export.
    # CUDA batches come from the trainer's already-validated immutable close
    # store.  Converting these reductions to Python booleans would otherwise
    # introduce two device synchronizations on every training step.  Invalid
    # CUDA values still propagate through log/softmax as non-finite loss and
    # fail the trainer's metric checks; eager CPU callers retain clear errors.
    if not _is_graph_capture() and closes.device.type == "cpu":
        if not bool(torch.isfinite(closes).all()):
            raise ValueError("closes must be finite")
        if not bool((closes > 0).all()):
            raise ValueError("closes must be strictly positive")


def causal_minute_log_returns(closes: Tensor) -> Tensor:
    """Aggregate the preceding hour into 60 completed-minute log returns.

    ``closes[:, 0]`` is the boundary close immediately before the first
    second and ``closes[:, -1]`` is the decision-time close.  Adjacent log
    returns are grouped as ``[0:60], [60:120], ..., [3540:3600]``.  Therefore
    every output uses only closes at or before decision time, and a change in
    the final close can affect only the final minute.
    """
    _validate_closes(closes)
    log_closes = closes.float().squeeze(-1).log()
    one_second_returns = log_closes[:, 1:] - log_closes[:, :-1]
    return one_second_returns.reshape(
        closes.shape[0],
        MINUTE_RETURN_COUNT,
        SECONDS_PER_MINUTE,
    ).sum(dim=-1)


def fixed_scale_minute_features(
    minute_log_returns: Tensor,
) -> tuple[Tensor, Tensor]:
    """Return fixed-scale minute inputs and explicit trend/scale statistics."""
    if minute_log_returns.ndim != 2 \
            or minute_log_returns.shape[1] != MINUTE_RETURN_COUNT:
        raise ValueError(
            "minute_log_returns must have shape "
            f"[batch, {MINUTE_RETURN_COUNT}]"
        )
    if not minute_log_returns.is_floating_point():
        raise TypeError("minute_log_returns must use a floating-point dtype")

    scaled = minute_log_returns.float() / FIXED_MINUTE_RETURN_SCALE
    mean = scaled.mean(dim=1)
    centered = scaled - mean.unsqueeze(1)
    rms = scaled.square().mean(dim=1).clamp_min(1e-12).sqrt()
    standard_deviation = (
        centered.square().mean(dim=1).clamp_min(1e-12).sqrt()
    )
    scale_features = torch.stack((
        mean,
        rms,
        standard_deviation,
        scaled.abs().mean(dim=1),
        scaled.sum(dim=1) / math.sqrt(MINUTE_RETURN_COUNT),
        scaled[:, -1],
        scaled[:, -5:].mean(dim=1),
        scaled[:, -15:].mean(dim=1),
    ), dim=1)
    return scaled, scale_features


def causal_minute_mlp_features(closes: Tensor) -> tuple[Tensor, Tensor]:
    """Build the dense MLP input and its explicit scale/trend bypass."""
    scaled, scale_features = fixed_scale_minute_features(
        causal_minute_log_returns(closes)
    )
    return torch.cat((scaled, scale_features), dim=1), scale_features


def _reset_fused_projection(projection: nn.Linear) -> None:
    output_width = projection.out_features // 2
    with torch.no_grad():
        nn.init.kaiming_normal_(
            projection.weight[:output_width],
            nonlinearity="linear",
        )
        nn.init.xavier_uniform_(
            projection.weight[output_width:],
            gain=0.5,
        )
        projection.bias[:output_width].zero_()
        projection.bias[output_width:].fill_(1.0)


class FusedValueGate(nn.Module):
    """One dense affine split into value and sigmoid-gate branches."""

    def __init__(self, input_width: int, output_width: int) -> None:
        super().__init__()
        self.input_width = int(input_width)
        self.output_width = int(output_width)
        self.projection = nn.Linear(input_width, 2 * output_width)
        _reset_fused_projection(self.projection)

    def forward(self, values: Tensor) -> Tensor:
        value, gate = self.projection(values).chunk(2, dim=-1)
        return value * torch.sigmoid(gate)


class FusedGluResidualBlock(nn.Module):
    """Pre-LayerNorm fused GLU with a variance-stable residual merge."""

    def __init__(
        self,
        width: int,
        dropout: float,
        residual_gain: float,
    ) -> None:
        super().__init__()
        self.normalization = nn.LayerNorm(width)
        self.value_gate = FusedValueGate(width, width)
        self.dropout = nn.Dropout(dropout)
        self.residual_gain = float(residual_gain)
        self.merge_scale = 1.0 / math.sqrt(1.0 + residual_gain ** 2)

    def forward(self, hidden: Tensor) -> Tensor:
        update = self.value_gate(self.normalization(hidden))
        return (
            hidden + self.residual_gain * self.dropout(update)
        ) * self.merge_scale


@dataclass(frozen=True)
class MinuteReturnMlpConfig:
    hidden_width: int = 512
    layer_count: int = 8
    dropout: float = 0.05
    residual_gain: float = 1.0

    def __post_init__(self) -> None:
        if isinstance(self.hidden_width, bool) \
                or not isinstance(self.hidden_width, int) \
                or self.hidden_width < 1:
            raise ValueError("hidden_width must be a positive integer")
        if isinstance(self.layer_count, bool) \
                or not isinstance(self.layer_count, int) \
                or self.layer_count < 1:
            raise ValueError("layer_count must be a positive integer")
        if not math.isfinite(self.dropout) or not 0 <= self.dropout < 1:
            raise ValueError("dropout must be finite and in [0, 1)")
        if not math.isfinite(self.residual_gain) \
                or self.residual_gain <= 0:
            raise ValueError("residual_gain must be finite and positive")


class MinuteReturnOracleMlp(nn.Module):
    """Fast causal minute-return MLP producing only 101 policy logits.

    The main path uses fixed-scale minute returns without per-example input
    normalization.  Eight explicit scale/trend features also reach the logits
    through a separate linear bypass, so the pre-normalized residual blocks
    cannot erase volatility magnitude relevant to fixed trading friction.
    """

    architecture_contract = ARCHITECTURE_CONTRACT
    context_length = INPUT_CLOSE_COUNT
    forecast_horizon = CONTEXT_SECONDS
    variable_count = 1
    action_count = OUTPUT_ACTION_COUNT

    def __init__(
        self,
        config: MinuteReturnMlpConfig | None = None,
        *,
        hidden_width: int | None = None,
        layer_count: int | None = None,
        dropout: float | None = None,
        residual_gain: float | None = None,
    ) -> None:
        super().__init__()
        if config is not None and any(
            value is not None
            for value in (
                hidden_width,
                layer_count,
                dropout,
                residual_gain,
            )
        ):
            raise ValueError(
                "pass either config or keyword overrides, not both"
            )
        if config is None:
            defaults = MinuteReturnMlpConfig()
            config = MinuteReturnMlpConfig(
                hidden_width=(
                    defaults.hidden_width
                    if hidden_width is None else hidden_width
                ),
                layer_count=(
                    defaults.layer_count
                    if layer_count is None else layer_count
                ),
                dropout=defaults.dropout if dropout is None else dropout,
                residual_gain=(
                    defaults.residual_gain
                    if residual_gain is None else residual_gain
                ),
            )
        self.config = config
        self.input_value_gate = FusedValueGate(
            INPUT_FEATURE_COUNT,
            config.hidden_width,
        )
        self.blocks = nn.ModuleList([
            FusedGluResidualBlock(
                config.hidden_width,
                config.dropout,
                config.residual_gain,
            )
            for _ in range(config.layer_count - 1)
        ])
        self.output_normalization = nn.LayerNorm(config.hidden_width)
        self.policy_head = nn.Linear(
            config.hidden_width,
            OUTPUT_ACTION_COUNT,
        )
        self.scale_bypass = nn.Linear(
            SCALE_FEATURE_COUNT,
            OUTPUT_ACTION_COUNT,
            bias=False,
        )
        nn.init.normal_(self.policy_head.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.policy_head.bias)
        nn.init.zeros_(self.scale_bypass.weight)

    def extract_features(self, closes: Tensor) -> tuple[Tensor, Tensor]:
        return causal_minute_mlp_features(closes)

    def forward(self, closes: Tensor) -> Tensor:
        features, scale_features = self.extract_features(closes)
        hidden = self.input_value_gate(features)
        for block in self.blocks:
            hidden = block(hidden)
        return (
            self.policy_head(self.output_normalization(hidden))
            + self.scale_bypass(scale_features)
        )

    def forward_policy_logits(self, closes: Tensor) -> Tensor:
        """Explicit forecast-free trainer alias for the logits-only model."""
        return self.forward(closes)

    def parameter_count(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )
