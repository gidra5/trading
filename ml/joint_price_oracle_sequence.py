from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as functional

from joint_price_oracle_minute_mlp import (
    FIXED_MINUTE_RETURN_SCALE,
    MINUTE_RETURN_COUNT as HOUR_SCALE_WINDOW_MINUTES,
    SCALE_FEATURE_COUNT,
    fixed_scale_minute_features,
)


SECONDS_PER_MINUTE = 60
SECOND_MS = 1_000
DECISION_INTERVAL_MS = 60_000
DECISION_PHASE_MS = 999
ACTION_COUNT = 101
DEFAULT_RECEPTIVE_FIELD_MINUTES = 60
LONG_CONTEXT_RECEPTIVE_FIELD_MINUTES = 360
INPUT_CLOSE_COUNT = DEFAULT_RECEPTIVE_FIELD_MINUTES * SECONDS_PER_MINUTE
ORACLE_FORECAST_HORIZON = 3_600
CAUSAL_MINUTE_FEATURE_COUNT = 5
FIXED_RETURN_SCALE = 10_000.0
FIXED_PATH_SCALE = 100.0
RMS_REFERENCE = 1e-4
SEQUENCE_ARCHITECTURE_CONTRACT = (
    "causal-patch-local-minute-token-exact-rf-tcn-direct-policy-v1"
)
BOUNDARY_INPUT_CLOSE_COUNT = INPUT_CLOSE_COUNT + 1
LONG_CONTEXT_BOUNDARY_INPUT_CLOSE_COUNT = (
    LONG_CONTEXT_RECEPTIVE_FIELD_MINUTES * SECONDS_PER_MINUTE + 1
)
BOUNDARY_SEQUENCE_ARCHITECTURE_CONTRACT = (
    "causal-boundary-complete-minute-token-exact-rf-tcn-"
    "hour-scale-bypass-direct-policy-v1"
)
BOUNDARY_LONG_SEQUENCE_ARCHITECTURE_CONTRACT = (
    "causal-boundary-complete-minute-token-exact-360m-rf-tcn-"
    "last-60m-scale-bypass-direct-policy-v1"
)
BOUNDARY_SEQUENCE_POLICY_DROPOUT_ARCHITECTURE_CONTRACT = (
    "causal-boundary-complete-minute-token-exact-rf-tcn-"
    "hour-scale-bypass-independent-policy-dropout-direct-policy-v2"
)
BOUNDARY_SEQUENCE_PROTOTYPE_MIXTURE_ARCHITECTURE_CONTRACT = (
    "causal-boundary-complete-minute-token-exact-rf-tcn-"
    "hour-scale-bypass-fixed-train-forward-kl-k16-soft-mixture-policy-v1"
)
BOUNDARY_MA_SEQUENCE_ARCHITECTURE_CONTRACT = (
    "causal-boundary-complete-minute-token-exact-rf-tcn-"
    "additive-ma-band-residual-hour-scale-bypass-direct-policy-v1"
)
ADDITIVE_MA_WINDOWS = (60, 15, 5)
ADDITIVE_MA_BAND_COUNT = 4
ADDITIVE_MA_FEATURE_COUNT = ADDITIVE_MA_BAND_COUNT * 2
SHAPE_INVARIANT_LINEAR_CHUNK_ROWS = 1_024


def _is_graph_capture() -> bool:
    if torch.jit.is_scripting() or torch.jit.is_tracing():
        return True
    if torch.onnx.is_in_onnx_export():
        return True
    compiler = getattr(torch, "compiler", None)
    return bool(compiler is not None and compiler.is_compiling())


def _fixed_chunk_affine(
    values: Tensor,
    weight: Tensor,
    bias: Tensor | None,
) -> Tensor:
    """Apply one affine map with shape-independent fixed-size kernels.

    GEMM implementations can change their floating-point accumulation path
    when the number of rows changes.  A reused sequence and its overlapping
    fixed windows have different outer shapes, so the six-hour exact-reuse
    contract evaluates both as padded 1,024-row chunks.  The extra padded
    rows are discarded and never enter either values or gradients.
    """
    if values.ndim < 1 or values.shape[-1] != weight.shape[1]:
        raise ValueError("fixed-chunk affine input width is incompatible")
    # Graph exporters need a shape-polymorphic operation.  Deployment uses a
    # single fixed window rather than sequence-core reuse, so the ordinary
    # affine is both semantically equivalent and safe for dynamic batches.
    if _is_graph_capture():
        return functional.linear(values, weight, bias)
    flat = values.reshape(-1, values.shape[-1])
    outputs: list[Tensor] = []
    for start in range(0, flat.shape[0], SHAPE_INVARIANT_LINEAR_CHUNK_ROWS):
        chunk = flat[start:start + SHAPE_INVARIANT_LINEAR_CHUNK_ROWS]
        count = chunk.shape[0]
        if count < SHAPE_INVARIANT_LINEAR_CHUNK_ROWS:
            chunk = functional.pad(
                chunk,
                (0, 0, 0, SHAPE_INVARIANT_LINEAR_CHUNK_ROWS - count),
            )
        outputs.append(functional.linear(
            chunk,
            weight,
            bias,
        )[:count])
    return torch.cat(outputs, dim=0).reshape(
        *values.shape[:-1],
        weight.shape[0],
    )


@dataclass(frozen=True)
class SequenceCoreAlignment:
    """Timestamp and tensor alignment for one supervised sequence core."""

    core_prediction_time_start: int
    core_prediction_time_end: int
    core_count: int
    receptive_field_minutes: int
    halo_minutes: int
    token_count: int
    close_count: int
    input_close_time_start: int
    input_close_time_end: int

    def prediction_time(self, core_index: int) -> int:
        if core_index < 0 or core_index >= self.core_count:
            raise IndexError("sequence core target index is out of range")
        return (
            self.core_prediction_time_start
            + core_index * DECISION_INTERVAL_MS
        )

    def sequence_index(self, core_index: int) -> int:
        """Return the sequence-logit row aligned with one target row."""
        self.prediction_time(core_index)
        return self.halo_minutes + core_index


def sequence_core_alignment(
    core_prediction_time_start: int,
    core_count: int,
    receptive_field_minutes: int,
) -> SequenceCoreAlignment:
    """Describe an exact minute-token halo ending at every target timestamp.

    A minute token ending at ``t`` contains the 60 one-second closes from
    ``t - 59s`` through ``t``.  Consequently, ``L`` tokens contain exactly
    the legacy ``L * 60`` closes ending at the same decision timestamp.
    """
    if isinstance(core_prediction_time_start, bool) \
            or not isinstance(core_prediction_time_start, int) \
            or core_prediction_time_start % DECISION_INTERVAL_MS \
            != DECISION_PHASE_MS:
        raise ValueError(
            "sequence cores must start on the verified minute phase"
        )
    if isinstance(core_count, bool) or core_count < 1:
        raise ValueError("sequence core count must be positive")
    if isinstance(receptive_field_minutes, bool) \
            or receptive_field_minutes < 1:
        raise ValueError("sequence receptive field must be positive")
    halo = receptive_field_minutes - 1
    token_count = halo + core_count
    core_end = (
        core_prediction_time_start
        + (core_count - 1) * DECISION_INTERVAL_MS
    )
    input_start = (
        core_prediction_time_start
        - (
            halo * SECONDS_PER_MINUTE
            + SECONDS_PER_MINUTE - 1
        ) * SECOND_MS
    )
    close_count = token_count * SECONDS_PER_MINUTE
    if input_start + (close_count - 1) * SECOND_MS != core_end:
        raise RuntimeError("sequence close and target alignment is inconsistent")
    return SequenceCoreAlignment(
        core_prediction_time_start=core_prediction_time_start,
        core_prediction_time_end=core_end,
        core_count=core_count,
        receptive_field_minutes=receptive_field_minutes,
        halo_minutes=halo,
        token_count=token_count,
        close_count=close_count,
        input_close_time_start=input_start,
        input_close_time_end=core_end,
    )


def boundary_sequence_core_alignment(
    core_prediction_time_start: int,
    core_count: int,
    receptive_field_minutes: int,
) -> SequenceCoreAlignment:
    """Describe a corrected core with one shared pre-return boundary close.

    Compared with the legacy 3,600-close alignment, this starts one second
    earlier and contains one additional close.  Its ``L``-minute first target
    therefore has exactly ``L * 60`` adjacent returns through prediction time.
    """
    legacy = sequence_core_alignment(
        core_prediction_time_start,
        core_count,
        receptive_field_minutes,
    )
    corrected_start = legacy.input_close_time_start - SECOND_MS
    corrected_count = legacy.close_count + 1
    if corrected_start + (corrected_count - 1) * SECOND_MS \
            != legacy.input_close_time_end:
        raise RuntimeError(
            "boundary-complete sequence alignment is inconsistent"
        )
    return SequenceCoreAlignment(
        core_prediction_time_start=legacy.core_prediction_time_start,
        core_prediction_time_end=legacy.core_prediction_time_end,
        core_count=legacy.core_count,
        receptive_field_minutes=legacy.receptive_field_minutes,
        halo_minutes=legacy.halo_minutes,
        token_count=legacy.token_count,
        close_count=corrected_count,
        input_close_time_start=corrected_start,
        input_close_time_end=legacy.input_close_time_end,
    )


def exact_receptive_field_dilations(
    receptive_field_minutes: int,
) -> tuple[int, ...]:
    """Return kernel-two dilations whose receptive field is exactly ``L``."""
    if isinstance(receptive_field_minutes, bool) \
            or receptive_field_minutes < 1:
        raise ValueError("sequence receptive field must be positive")
    remaining = receptive_field_minutes - 1
    dilation = 1
    result: list[int] = []
    while remaining > 0:
        selected = min(dilation, remaining)
        result.append(selected)
        remaining -= selected
        dilation *= 2
    if 1 + sum(result) != receptive_field_minutes:
        raise RuntimeError("exact receptive-field dilation construction failed")
    return tuple(result)


class PatchLocalMinuteTokenizer(nn.Module):
    """Encode each completed minute without depending on another patch.

    Prefix statistics reset independently inside every 60-close patch.  The
    token for a decision timestamp therefore has the same value in a long
    chronological sequence and in a single fixed-context inference window.
    """

    def __init__(self, token_width: int, epsilon: float = 1e-8) -> None:
        super().__init__()
        if token_width < 1:
            raise ValueError("minute token width must be positive")
        if epsilon <= 0 or not math.isfinite(epsilon):
            raise ValueError("minute token epsilon must be finite and positive")
        self.token_width = int(token_width)
        self.epsilon = float(epsilon)
        self.patch_projection = nn.Conv1d(
            CAUSAL_MINUTE_FEATURE_COUNT,
            token_width,
            kernel_size=SECONDS_PER_MINUTE,
        )
        self.output_norm = nn.LayerNorm(token_width)

    def forward(self, closes: Tensor) -> Tensor:
        if closes.ndim != 3 or closes.shape[2] != 1 \
                or closes.shape[1] < SECONDS_PER_MINUTE \
                or closes.shape[1] % SECONDS_PER_MINUTE != 0:
            raise ValueError(
                "minute tokenizer expects [batch, minutes * 60, 1] closes"
            )
        batch_size = closes.shape[0]
        minute_count = closes.shape[1] // SECONDS_PER_MINUTE
        log_patches = closes.float().clamp_min(self.epsilon).log().reshape(
            batch_size,
            minute_count,
            SECONDS_PER_MINUTE,
        )
        log_returns = torch.cat((
            torch.zeros_like(log_patches[:, :, :1]),
            log_patches[:, :, 1:] - log_patches[:, :, :-1],
        ), dim=-1)
        count = torch.arange(
            1,
            SECONDS_PER_MINUTE + 1,
            device=closes.device,
            dtype=torch.float32,
        ).view(1, 1, -1)
        cumulative_rms = (
            log_returns.square().cumsum(dim=-1) / count
            + self.epsilon
        ).sqrt()
        relative_path = log_patches - log_patches[:, :, :1]
        normalized_returns = log_returns / cumulative_rms
        normalized_path = relative_path / (
            cumulative_rms * count.sqrt()
        ).clamp_min(self.epsilon)
        features = torch.stack((
            torch.tanh(normalized_returns / 8.0) * 8.0,
            torch.tanh(normalized_path / 8.0) * 8.0,
            torch.tanh(
                log_returns * FIXED_RETURN_SCALE / 8.0
            ) * 8.0,
            torch.tanh(
                relative_path * FIXED_PATH_SCALE / 8.0
            ) * 8.0,
            torch.tanh(
                torch.log(
                    cumulative_rms.clamp_min(self.epsilon)
                    / RMS_REFERENCE
                ) / 4.0
            ) * 4.0,
        ), dim=2)
        projected = self.patch_projection(features.reshape(
            batch_size * minute_count,
            CAUSAL_MINUTE_FEATURE_COUNT,
            SECONDS_PER_MINUTE,
        )).squeeze(-1)
        tokens = functional.gelu(self.output_norm(projected))
        return tokens.reshape(batch_size, minute_count, self.token_width)


class ExactReceptiveFieldCausalBlock(nn.Module):
    """One gated causal kernel-two block with a known RF increment."""

    def __init__(self, width: int, dilation: int, dropout: float) -> None:
        super().__init__()
        if min(width, dilation) < 1:
            raise ValueError("causal block dimensions must be positive")
        if dropout < 0 or dropout >= 1 or not math.isfinite(dropout):
            raise ValueError("causal block dropout must be in [0, 1)")
        self.dilation = int(dilation)
        self.norm = nn.LayerNorm(width)
        self.depthwise = nn.Conv1d(
            width,
            width,
            kernel_size=2,
            dilation=dilation,
            groups=width,
        )
        self.value_gate = nn.Linear(width, width * 2)
        self.output_projection = nn.Linear(width, width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: Tensor) -> Tensor:
        if values.ndim != 3:
            raise ValueError("causal block expects [batch, minute, width]")
        hidden = self.norm(values).transpose(1, 2)
        hidden = self.depthwise(functional.pad(
            hidden,
            (self.dilation, 0),
        )).transpose(1, 2)
        value, gate = self.value_gate(hidden).chunk(2, dim=-1)
        hidden = functional.gelu(value) * torch.sigmoid(gate)
        hidden = self.output_projection(self.dropout(hidden))
        return (values + self.dropout(hidden)) / math.sqrt(2.0)


class ShapeInvariantExactReceptiveFieldCausalBlock(
    ExactReceptiveFieldCausalBlock
):
    """The same gated TCN block with exact cross-shape reuse arithmetic."""

    def forward(self, values: Tensor) -> Tensor:
        if values.ndim != 3:
            raise ValueError("causal block expects [batch, minute, width]")
        normalized = self.norm(values)
        prior = functional.pad(
            normalized,
            (0, 0, self.dilation, 0),
        )[:, :-self.dilation, :]
        kernel = self.depthwise.weight[:, 0, :]
        hidden = (
            prior * kernel[:, 0]
            + normalized * kernel[:, 1]
            + self.depthwise.bias
        )
        value, gate = _fixed_chunk_affine(
            hidden,
            self.value_gate.weight,
            self.value_gate.bias,
        ).chunk(2, dim=-1)
        hidden = functional.gelu(value) * torch.sigmoid(gate)
        hidden = _fixed_chunk_affine(
            self.dropout(hidden),
            self.output_projection.weight,
            self.output_projection.bias,
        )
        return (values + self.dropout(hidden)) / math.sqrt(2.0)


class ChronologicalMinutePolicyModel(nn.Module):
    """Emit raw 101-action logits once per completed causal minute."""

    architecture_contract = SEQUENCE_ARCHITECTURE_CONTRACT

    def __init__(
        self,
        receptive_field_minutes: int = DEFAULT_RECEPTIVE_FIELD_MINUTES,
        token_width: int = 128,
        policy_hidden_width: int = 192,
        action_count: int = ACTION_COUNT,
        dropout: float = 0.05,
        feature_epsilon: float = 1e-8,
        dilations: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        if min(
            receptive_field_minutes,
            token_width,
            policy_hidden_width,
            action_count,
        ) < 1:
            raise ValueError("sequence model dimensions must be positive")
        selected_dilations = (
            exact_receptive_field_dilations(receptive_field_minutes)
            if dilations is None
            else tuple(int(value) for value in dilations)
        )
        if any(value < 1 for value in selected_dilations) \
                or 1 + sum(selected_dilations) \
                != receptive_field_minutes:
            raise ValueError(
                "kernel-two dilations must exactly cover the receptive field"
            )
        self.receptive_field_minutes = int(receptive_field_minutes)
        self.context_length = (
            self.receptive_field_minutes * SECONDS_PER_MINUTE
        )
        # These are deployment/training metadata only.  The module deliberately
        # has no forecast decoder or future-close path.
        self.forecast_horizon = ORACLE_FORECAST_HORIZON
        self.variable_count = 1
        self.action_count = int(action_count)
        self.dilations = selected_dilations
        self.tokenizer = PatchLocalMinuteTokenizer(
            token_width,
            feature_epsilon,
        )
        self.blocks = nn.ModuleList([
            ExactReceptiveFieldCausalBlock(
                token_width,
                dilation,
                dropout,
            )
            for dilation in selected_dilations
        ])
        self.final_norm = nn.LayerNorm(token_width)
        self.policy_head = nn.Sequential(
            nn.Linear(token_width, policy_hidden_width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(policy_hidden_width, action_count),
        )

    def encode_sequence(self, closes: Tensor) -> Tensor:
        values = self.tokenizer(closes)
        for block in self.blocks:
            values = block(values)
        return self.final_norm(values)

    def forward_sequence(self, closes: Tensor) -> Tensor:
        """Return one logit row for every complete minute in ``closes``."""
        return self.policy_head(self.encode_sequence(closes))

    def forward_single(self, closes: Tensor) -> Tensor:
        """Return the final row for one exact fixed-receptive-field window."""
        if closes.ndim != 3 \
                or closes.shape[1] != self.context_length \
                or closes.shape[2] != 1:
            raise ValueError(
                "single decision input must match the exact context length"
            )
        return self.forward_sequence(closes)[:, -1, :]

    def forward_policy_logits(self, closes: Tensor) -> Tensor:
        """Explicit forecast-free entry point used by policy-only training."""
        return self.forward_single(closes)

    def forward(self, closes: Tensor) -> Tensor:
        """ONNX-facing single-decision forward."""
        return self.forward_single(closes)


def boundary_complete_minute_features(
    closes: Tensor,
    epsilon: float = 1e-8,
) -> tuple[Tensor, Tensor]:
    """Return exact 60-transition minute features and minute log returns.

    The first close is the boundary immediately preceding the first second.
    Differencing before reshaping therefore retains all 60 one-second returns
    in every completed minute, including the transition at each minute patch
    boundary.  The returned feature tensor is ``[B, minute, 5, 60]`` and the
    return tensor is ``[B, minute]``.
    """
    if closes.ndim != 3 or closes.shape[2] != 1 \
            or closes.shape[1] < SECONDS_PER_MINUTE + 1 \
            or (closes.shape[1] - 1) % SECONDS_PER_MINUTE != 0:
        raise ValueError(
            "boundary-complete minute features expect "
            "[batch, minutes * 60 + 1, 1] closes"
        )
    if not closes.is_floating_point():
        raise TypeError("closes must use a floating-point dtype")
    if epsilon <= 0 or not math.isfinite(epsilon):
        raise ValueError("minute feature epsilon must be finite and positive")
    batch_size = closes.shape[0]
    minute_count = (closes.shape[1] - 1) // SECONDS_PER_MINUTE
    log_closes = closes.float().clamp_min(epsilon).log().squeeze(-1)
    log_returns = (
        log_closes[:, 1:] - log_closes[:, :-1]
    ).reshape(batch_size, minute_count, SECONDS_PER_MINUTE)
    count = torch.arange(
        1,
        SECONDS_PER_MINUTE + 1,
        device=closes.device,
        dtype=torch.float32,
    ).view(1, 1, -1)
    cumulative_rms = (
        log_returns.square().cumsum(dim=-1) / count
        + epsilon
    ).sqrt()
    relative_path = log_returns.cumsum(dim=-1)
    normalized_returns = log_returns / cumulative_rms
    normalized_path = relative_path / (
        cumulative_rms * count.sqrt()
    ).clamp_min(epsilon)
    features = torch.stack((
        torch.tanh(normalized_returns / 8.0) * 8.0,
        torch.tanh(normalized_path / 8.0) * 8.0,
        torch.tanh(
            log_returns * FIXED_RETURN_SCALE / 8.0
        ) * 8.0,
        torch.tanh(
            relative_path * FIXED_PATH_SCALE / 8.0
        ) * 8.0,
        torch.tanh(
            torch.log(
                cumulative_rms.clamp_min(epsilon) / RMS_REFERENCE
            ) / 4.0
        ) * 4.0,
    ), dim=2)
    return features, log_returns.sum(dim=-1)


def last_hour_scale_minute_features(
    minute_log_returns: Tensor,
) -> tuple[Tensor, Tensor]:
    """Return the existing eight scale features from the final hour only.

    Long-context sequence encoders may use every causal minute token, while
    this bypass deliberately retains the v18 feature definition and scale.
    Keeping the suffix selection here prevents a longer receptive field from
    silently changing the meaning of those eight inputs.
    """
    if minute_log_returns.ndim != 2 \
            or minute_log_returns.shape[1] < HOUR_SCALE_WINDOW_MINUTES:
        raise ValueError(
            "hour-scale features require at least 60 minute returns"
        )
    if not minute_log_returns.is_floating_point():
        raise TypeError(
            "minute_log_returns must use a floating-point dtype"
        )
    return fixed_scale_minute_features(
        minute_log_returns[:, -HOUR_SCALE_WINDOW_MINUTES:].contiguous()
    )


def causal_sequence_moving_average(
    values: Tensor,
    window: int,
) -> Tensor:
    """Return a trailing causal average with prefix-sized early windows."""
    if values.ndim != 2 or not values.is_floating_point():
        raise ValueError("moving-average values must be floating [batch, time]")
    if isinstance(window, bool) or window < 1:
        raise ValueError("moving-average window must be positive")
    length = values.shape[1]
    prefix = torch.cat((
        torch.zeros_like(values[:, :1]),
        values.cumsum(dim=1),
    ), dim=1)
    prefix_count = min(length, window - 1)
    pieces: list[Tensor] = []
    if prefix_count:
        counts = torch.arange(
            1,
            prefix_count + 1,
            device=values.device,
            dtype=values.dtype,
        ).view(1, -1)
        pieces.append(prefix[:, 1:prefix_count + 1] / counts)
    if length >= window:
        pieces.append(
            (prefix[:, window:] - prefix[:, :-window]) / float(window)
        )
    if not pieces:
        return values[:, :0]
    return torch.cat(pieces, dim=1)


def causal_additive_minute_path_bands(
    minute_log_returns: Tensor,
) -> tuple[Tensor, Tensor]:
    """Decompose the causal minute path into exact additive MA bands.

    The four bands are ``MA60``, ``MA15 - MA60``, ``MA5 - MA15``, and
    ``path - MA5``.  Their sum is exactly the cumulative log-price path.
    The second result contains first differences of every band, with the
    first row measured from the zero path at the input boundary.
    """
    if minute_log_returns.ndim != 2 \
            or not minute_log_returns.is_floating_point():
        raise ValueError(
            "minute_log_returns must be floating [batch, minute]"
        )
    path = minute_log_returns.float().cumsum(dim=1)
    slow = causal_sequence_moving_average(path, ADDITIVE_MA_WINDOWS[0])
    middle = causal_sequence_moving_average(path, ADDITIVE_MA_WINDOWS[1])
    fast = causal_sequence_moving_average(path, ADDITIVE_MA_WINDOWS[2])
    bands = torch.stack((
        slow,
        middle - slow,
        fast - middle,
        path - fast,
    ), dim=-1)
    deltas = torch.cat((
        bands[:, :1, :],
        bands[:, 1:, :] - bands[:, :-1, :],
    ), dim=1)
    return bands, deltas


def causal_additive_minute_path_band_features(
    minute_log_returns: Tensor,
) -> Tensor:
    """Return fixed-scale exact bands and their deltas for token residuals."""
    bands, deltas = causal_additive_minute_path_bands(minute_log_returns)
    return torch.cat((bands, deltas), dim=-1) * FIXED_PATH_SCALE


class BoundaryCompleteMinuteTokenizer(nn.Module):
    """Encode completed minutes while retaining their boundary transition."""

    def __init__(
        self,
        token_width: int,
        epsilon: float = 1e-8,
        *,
        shape_invariant_projection: bool = False,
    ) -> None:
        super().__init__()
        if token_width < 1:
            raise ValueError("minute token width must be positive")
        if epsilon <= 0 or not math.isfinite(epsilon):
            raise ValueError("minute token epsilon must be finite and positive")
        self.token_width = int(token_width)
        self.epsilon = float(epsilon)
        self.shape_invariant_projection = bool(shape_invariant_projection)
        self.patch_projection = nn.Conv1d(
            CAUSAL_MINUTE_FEATURE_COUNT,
            token_width,
            kernel_size=SECONDS_PER_MINUTE,
        )
        self.output_norm = nn.LayerNorm(token_width)

    def forward_with_minute_returns(
        self,
        closes: Tensor,
    ) -> tuple[Tensor, Tensor]:
        features, minute_returns = boundary_complete_minute_features(
            closes,
            self.epsilon,
        )
        batch_size, minute_count = features.shape[:2]
        projection_input = features.reshape(
            batch_size * minute_count,
            CAUSAL_MINUTE_FEATURE_COUNT,
            SECONDS_PER_MINUTE,
        )
        projected = (
            _fixed_chunk_affine(
                projection_input.flatten(1),
                self.patch_projection.weight.flatten(1),
                self.patch_projection.bias,
            )
            if self.shape_invariant_projection
            else self.patch_projection(projection_input).squeeze(-1)
        )
        tokens = functional.gelu(self.output_norm(projected)).reshape(
            batch_size,
            minute_count,
            self.token_width,
        )
        return tokens, minute_returns

    def forward(self, closes: Tensor) -> Tensor:
        return self.forward_with_minute_returns(closes)[0]


class _BoundaryCompleteMinutePolicyModel(nn.Module):
    """Shared causal TCN for immutable boundary-complete contracts."""

    architecture_contract: str

    def __init__(
        self,
        *,
        architecture_contract: str,
        add_ma_band_residual: bool,
        receptive_field_minutes: int = DEFAULT_RECEPTIVE_FIELD_MINUTES,
        token_width: int = 128,
        policy_hidden_width: int = 192,
        action_count: int = ACTION_COUNT,
        dropout: float = 0.05,
        policy_dropout: float | None = None,
        scale_window_minutes: int = HOUR_SCALE_WINDOW_MINUTES,
        shape_invariant_reuse: bool = False,
        feature_epsilon: float = 1e-8,
        dilations: Sequence[int] | None = None,
        policy_output_width: int | None = None,
    ) -> None:
        super().__init__()
        if min(
            receptive_field_minutes,
            token_width,
            policy_hidden_width,
            action_count,
        ) < 1:
            raise ValueError("sequence model dimensions must be positive")
        selected_policy_dropout = (
            float(dropout)
            if policy_dropout is None
            else float(policy_dropout)
        )
        if not 0 <= selected_policy_dropout < 1 \
                or not math.isfinite(selected_policy_dropout):
            raise ValueError("policy dropout must be finite and in [0, 1)")
        if isinstance(scale_window_minutes, bool) \
                or not isinstance(scale_window_minutes, int) \
                or scale_window_minutes != HOUR_SCALE_WINDOW_MINUTES:
            raise ValueError(
                "the hour-scale bypass requires exactly 60 minutes"
            )
        if receptive_field_minutes < int(scale_window_minutes):
            raise ValueError(
                "the receptive field cannot be shorter than the 60-minute "
                "scale window"
            )
        selected_dilations = (
            exact_receptive_field_dilations(receptive_field_minutes)
            if dilations is None
            else tuple(int(value) for value in dilations)
        )
        if any(value < 1 for value in selected_dilations) \
                or 1 + sum(selected_dilations) \
                != receptive_field_minutes:
            raise ValueError(
                "kernel-two dilations must exactly cover the receptive field"
            )
        self.architecture_contract = architecture_contract
        self.supports_sequence_core_reuse = not add_ma_band_residual
        self.receptive_field_minutes = int(receptive_field_minutes)
        self.context_length = (
            self.receptive_field_minutes * SECONDS_PER_MINUTE + 1
        )
        self.forecast_horizon = ORACLE_FORECAST_HORIZON
        self.variable_count = 1
        self.action_count = int(action_count)
        selected_policy_output_width = (
            self.action_count
            if policy_output_width is None
            else int(policy_output_width)
        )
        if selected_policy_output_width < 1:
            raise ValueError("policy output width must be positive")
        self.policy_output_width = selected_policy_output_width
        self.dilations = selected_dilations
        self.core_dropout = float(dropout)
        self.policy_dropout = selected_policy_dropout
        self.scale_window_minutes = int(scale_window_minutes)
        self.shape_invariant_reuse = bool(shape_invariant_reuse)
        self.tokenizer = BoundaryCompleteMinuteTokenizer(
            token_width,
            feature_epsilon,
            shape_invariant_projection=self.shape_invariant_reuse,
        )
        self.band_projection = (
            nn.Linear(ADDITIVE_MA_FEATURE_COUNT, token_width, bias=False)
            if add_ma_band_residual
            else None
        )
        block_type = (
            ShapeInvariantExactReceptiveFieldCausalBlock
            if self.shape_invariant_reuse
            else ExactReceptiveFieldCausalBlock
        )
        self.blocks = nn.ModuleList([
            block_type(
                token_width,
                dilation,
                dropout,
            )
            for dilation in selected_dilations
        ])
        self.final_norm = nn.LayerNorm(token_width)
        self.policy_head = nn.Sequential(
            nn.Linear(token_width, policy_hidden_width),
            nn.GELU(),
            nn.Dropout(selected_policy_dropout),
            nn.Linear(policy_hidden_width, selected_policy_output_width),
        )
        self.hour_scale_bypass = nn.Linear(
            SCALE_FEATURE_COUNT,
            selected_policy_output_width,
        )
        nn.init.zeros_(self.hour_scale_bypass.weight)
        nn.init.zeros_(self.hour_scale_bypass.bias)

    def _validate_input(self, closes: Tensor) -> None:
        if closes.ndim != 3 \
                or closes.shape[1] != self.context_length \
                or closes.shape[2] != 1:
            raise ValueError(
                "corrected minute TCN expects exactly "
                f"[batch, {self.context_length}, 1] closes"
            )

    def encode_history(self, closes: Tensor) -> tuple[Tensor, Tensor]:
        self._validate_input(closes)
        values, minute_returns = self._encode_token_sequence(closes)
        encoded = values[:, -1, :]
        _scaled, hour_scale = last_hour_scale_minute_features(minute_returns)
        return encoded, hour_scale

    def _encode_token_sequence(
        self,
        closes: Tensor,
    ) -> tuple[Tensor, Tensor]:
        values, minute_returns = self.tokenizer.forward_with_minute_returns(
            closes
        )
        if self.band_projection is not None:
            band_features = causal_additive_minute_path_band_features(
                minute_returns
            ).to(dtype=values.dtype)
            values = values + self.band_projection(band_features)
        for block in self.blocks:
            values = block(values)
        return self.final_norm(values), minute_returns

    def forward_sequence_core(self, closes: Tensor) -> Tensor:
        """Return logits for contiguous targets after one RF-sized halo.

        This path is exactly equivalent to evaluating every overlapping
        fixed-receptive-field window when core dropout is zero.  Optional
        policy dropout remains target-local because it is applied only after
        the deterministic reused core.  Window-anchored MA bands cannot
        share tokens across cores, so that variant rejects a multi-row core
        instead of silently changing its representation.
        """
        if closes.ndim != 3 or closes.shape[2] != 1 \
                or closes.shape[1] < self.context_length \
                or (closes.shape[1] - 1) % SECONDS_PER_MINUTE != 0:
            raise ValueError(
                "sequence core expects [batch, tokens * 60 + 1, 1] closes "
                "covering at least one complete receptive field"
            )
        token_count = (closes.shape[1] - 1) // SECONDS_PER_MINUTE
        core_count = token_count - self.receptive_field_minutes + 1
        if not self.supports_sequence_core_reuse and core_count != 1:
            raise ValueError(
                "window-anchored additive MA bands cannot reuse tokens "
                "across multiple fixed-window targets"
            )
        encoded, minute_returns = self._encode_token_sequence(closes)
        core_encoded = encoded[
            :,
            self.receptive_field_minutes - 1:,
            :,
        ]
        minute_windows = minute_returns.unfold(
            1,
            self.receptive_field_minutes,
            1,
        )
        batch_size = minute_windows.shape[0]
        _scaled, hour_scale = last_hour_scale_minute_features(
            minute_windows[..., -self.scale_window_minutes:]
            .contiguous().reshape(
                batch_size * core_count,
                self.scale_window_minutes,
            )
        )
        hour_scale = hour_scale.reshape(
            batch_size,
            core_count,
            SCALE_FEATURE_COUNT,
        )
        return self._project_policy_output(
            self._apply_policy_head(core_encoded)
            + self._apply_scale_bypass(hour_scale)
        )

    def forward_policy_logits(self, closes: Tensor) -> Tensor:
        encoded, hour_scale = self.encode_history(closes)
        return self._project_policy_output(
            self._apply_policy_head(encoded)
            + self._apply_scale_bypass(hour_scale)
        )

    def _apply_policy_head(self, encoded: Tensor) -> Tensor:
        if not self.shape_invariant_reuse:
            return self.policy_head(encoded)
        hidden = _fixed_chunk_affine(
            encoded,
            self.policy_head[0].weight,
            self.policy_head[0].bias,
        )
        hidden = self.policy_head[2](functional.gelu(hidden))
        return _fixed_chunk_affine(
            hidden,
            self.policy_head[3].weight,
            self.policy_head[3].bias,
        )

    def _apply_scale_bypass(self, scale_features: Tensor) -> Tensor:
        if not self.shape_invariant_reuse:
            return self.hour_scale_bypass(scale_features)
        return _fixed_chunk_affine(
            scale_features,
            self.hour_scale_bypass.weight,
            self.hour_scale_bypass.bias,
        )

    def _project_policy_output(self, policy_output: Tensor) -> Tensor:
        """Map the target-local head output to action logits."""
        return policy_output

    def forward_single(self, closes: Tensor) -> Tensor:
        return self.forward_policy_logits(closes)

    def forward(self, closes: Tensor) -> Tensor:
        return self.forward_policy_logits(closes)


class BoundaryCompleteMinutePolicyModel(_BoundaryCompleteMinutePolicyModel):
    """Corrected v17 topology with exact returns and hour-scale bypass."""

    architecture_contract = BOUNDARY_SEQUENCE_ARCHITECTURE_CONTRACT

    def __init__(self, **kwargs) -> None:
        architecture_contract = (
            BOUNDARY_SEQUENCE_POLICY_DROPOUT_ARCHITECTURE_CONTRACT
            if kwargs.get("policy_dropout") is not None
            else self.architecture_contract
        )
        super().__init__(
            architecture_contract=architecture_contract,
            add_ma_band_residual=False,
            **kwargs,
        )


class BoundaryCompleteLongContextMinutePolicyModel(
    _BoundaryCompleteMinutePolicyModel
):
    """Six-hour v18-style TCN with a last-hour-only scale bypass."""

    architecture_contract = BOUNDARY_LONG_SEQUENCE_ARCHITECTURE_CONTRACT

    def __init__(self, **kwargs) -> None:
        receptive_field_minutes = kwargs.pop(
            "receptive_field_minutes",
            LONG_CONTEXT_RECEPTIVE_FIELD_MINUTES,
        )
        if isinstance(receptive_field_minutes, bool) \
                or not isinstance(receptive_field_minutes, int) \
                or receptive_field_minutes \
                != LONG_CONTEXT_RECEPTIVE_FIELD_MINUTES:
            raise ValueError(
                "long-context boundary TCN requires exactly 360 minutes"
            )
        super().__init__(
            architecture_contract=self.architecture_contract,
            add_ma_band_residual=False,
            receptive_field_minutes=LONG_CONTEXT_RECEPTIVE_FIELD_MINUTES,
            shape_invariant_reuse=True,
            **kwargs,
        )


class BoundaryPrototypeMixtureMinutePolicyModel(
    _BoundaryCompleteMinutePolicyModel
):
    """v18 causal encoder with a fixed train-only soft-policy dictionary.

    The learned head emits continuous simplex weights.  Its returned action
    logits are normalized log probabilities of ``weights @ prototypes``, so
    the ordinary soft-target cross-entropy remains the exact training
    objective.  The prototype matrix is a persistent, gradient-free buffer;
    hard prototype assignments are never produced or supervised.
    """

    architecture_contract = (
        BOUNDARY_SEQUENCE_PROTOTYPE_MIXTURE_ARCHITECTURE_CONTRACT
    )

    def __init__(
        self,
        *,
        policy_prototypes: Tensor,
        train_mixture_weights: Tensor,
        **kwargs,
    ) -> None:
        prototypes = torch.as_tensor(
            policy_prototypes,
            dtype=torch.float32,
        ).detach().clone()
        weights = torch.as_tensor(
            train_mixture_weights,
            dtype=torch.float32,
        ).detach().clone()
        if prototypes.ndim != 2:
            raise ValueError("policy prototypes must have shape [K, action]")
        prototype_count, prototype_action_count = prototypes.shape
        configured_action_count = int(kwargs.get("action_count", ACTION_COUNT))
        if prototype_count < 2 \
                or prototype_action_count != configured_action_count:
            raise ValueError(
                "policy prototype dimensions do not match the action grid"
            )
        if weights.shape != (prototype_count,):
            raise ValueError(
                "train prototype mixture weights must have shape [K]"
            )
        if not bool(torch.isfinite(prototypes).all()) \
                or not bool((prototypes > 0).all()) \
                or not bool(torch.isfinite(weights).all()) \
                or not bool((weights > 0).all()):
            raise ValueError(
                "policy prototypes and train mixture weights must be finite "
                "and strictly positive"
            )
        if not bool(torch.allclose(
            prototypes.sum(dim=-1),
            torch.ones(prototype_count),
            atol=2e-6,
            rtol=2e-6,
        )) or not bool(torch.allclose(
            weights.sum(),
            torch.ones(()),
            atol=2e-6,
            rtol=2e-6,
        )):
            raise ValueError(
                "policy prototypes and train mixture weights must be normalized"
            )
        # Remove only harmless float32 summation drift.  The installed matrix
        # remains fixed thereafter and is serialized in every checkpoint.
        prototypes = prototypes / prototypes.sum(dim=-1, keepdim=True)
        weights = weights / weights.sum()
        super().__init__(
            architecture_contract=self.architecture_contract,
            add_ma_band_residual=False,
            policy_output_width=prototype_count,
            **kwargs,
        )
        self.prototype_count = int(prototype_count)
        self.register_buffer(
            "policy_prototypes",
            prototypes,
            persistent=True,
        )
        self.register_buffer(
            "initial_train_mixture_weights",
            weights,
            persistent=True,
        )
        final_projection = self.policy_head[-1]
        if not isinstance(final_projection, nn.Linear):
            raise RuntimeError("prototype mixture head projection is invalid")
        nn.init.zeros_(final_projection.weight)
        with torch.no_grad():
            final_projection.bias.copy_(weights.log())
        nn.init.zeros_(self.hour_scale_bypass.weight)
        nn.init.zeros_(self.hour_scale_bypass.bias)

    def policy_mixture_weights(self, mixture_logits: Tensor) -> Tensor:
        """Return continuous normalized weights for the fixed dictionary."""
        if mixture_logits.shape[-1] != self.prototype_count:
            raise ValueError(
                "prototype mixture logits have the wrong final dimension"
            )
        return torch.softmax(mixture_logits.float(), dim=-1).to(
            dtype=mixture_logits.dtype
        )

    def policy_probabilities_from_mixture_logits(
        self,
        mixture_logits: Tensor,
    ) -> Tensor:
        weights = self.policy_mixture_weights(mixture_logits)
        return weights @ self.policy_prototypes.to(dtype=weights.dtype)

    def _project_policy_output(self, policy_output: Tensor) -> Tensor:
        # Work in log space so tiny but non-zero prototype tails remain stable
        # under bfloat16 autocast.  Each basis row is normalized, hence these
        # are already normalized action log probabilities.
        log_weights = torch.log_softmax(policy_output.float(), dim=-1)
        log_prototypes = self.policy_prototypes.float().log()
        return torch.logsumexp(
            log_weights.unsqueeze(-1) + log_prototypes,
            dim=-2,
        )


class BoundaryMaMinutePolicyModel(_BoundaryCompleteMinutePolicyModel):
    """Corrected topology with exact additive path-band token residuals."""

    architecture_contract = BOUNDARY_MA_SEQUENCE_ARCHITECTURE_CONTRACT

    def __init__(self, **kwargs) -> None:
        super().__init__(
            architecture_contract=self.architecture_contract,
            add_ma_band_residual=True,
            **kwargs,
        )


def masked_soft_target_policy_objective(
    logits: Tensor,
    target_probabilities: Tensor,
    valid_mask: Tensor | None = None,
) -> dict[str, Tensor]:
    """Direct soft-target CE/KL over valid chronological target rows."""
    if logits.ndim != 3 or target_probabilities.shape != logits.shape:
        raise ValueError(
            "sequence logits and target probabilities must be [B, T, A]"
        )
    if valid_mask is None:
        selected = torch.ones(
            logits.shape[:2],
            device=logits.device,
            dtype=torch.bool,
        )
    else:
        if valid_mask.shape != logits.shape[:2]:
            raise ValueError("sequence valid mask must be [B, T]")
        selected = valid_mask.to(device=logits.device, dtype=torch.bool)
    if not bool(selected.any()):
        raise ValueError("sequence objective requires at least one valid row")
    target = target_probabilities.float()
    target = target / target.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    predicted_log = torch.log_softmax(logits.float(), dim=-1)
    cross_entropy_rows = -(target * predicted_log).sum(dim=-1)
    target_log = torch.where(
        target > 0,
        target.clamp_min(1e-12).log(),
        torch.zeros_like(target),
    )
    kl_rows = (target * (target_log - predicted_log)).sum(dim=-1)
    cross_entropy = cross_entropy_rows.masked_select(selected).mean()
    kl_divergence = kl_rows.masked_select(selected).mean()
    return {
        "loss": cross_entropy,
        "crossEntropy": cross_entropy,
        "klDivergence": kl_divergence,
        "validRows": selected.sum(),
    }
