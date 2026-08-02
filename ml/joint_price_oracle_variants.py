from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as functional

from joint_price_oracle import (
    OUTPUT_ACTION_COUNT,
    FixedDctOrdinalLogitProjection,
    JointPriceOracleOutput,
    architecture_contract_with_policy_decoder,
    causal_moving_average,
)
from joint_price_oracle_minute_mlp import (
    ARCHITECTURE_CONTRACT as MINUTE_RETURN_MLP_CONTRACT,
    INPUT_CLOSE_COUNT as MINUTE_RETURN_INPUT_CLOSE_COUNT,
    MinuteReturnMlpConfig,
    MinuteReturnOracleMlp,
)
from joint_price_oracle_sequence import (
    BOUNDARY_INPUT_CLOSE_COUNT as MINUTE_SEQUENCE_BOUNDARY_INPUT_CLOSE_COUNT,
    LONG_CONTEXT_BOUNDARY_INPUT_CLOSE_COUNT,
    BOUNDARY_LONG_SEQUENCE_ARCHITECTURE_CONTRACT,
    BOUNDARY_MA_SEQUENCE_ARCHITECTURE_CONTRACT,
    BOUNDARY_SEQUENCE_PROTOTYPE_MIXTURE_ARCHITECTURE_CONTRACT,
    BOUNDARY_SEQUENCE_ARCHITECTURE_CONTRACT,
    BOUNDARY_SEQUENCE_POLICY_DROPOUT_ARCHITECTURE_CONTRACT,
    DEFAULT_RECEPTIVE_FIELD_MINUTES as MINUTE_SEQUENCE_RECEPTIVE_FIELD,
    LONG_CONTEXT_RECEPTIVE_FIELD_MINUTES,
    INPUT_CLOSE_COUNT as MINUTE_SEQUENCE_INPUT_CLOSE_COUNT,
    SEQUENCE_ARCHITECTURE_CONTRACT as MINUTE_SEQUENCE_TCN_CONTRACT,
    BoundaryCompleteMinutePolicyModel,
    BoundaryCompleteLongContextMinutePolicyModel,
    BoundaryMaMinutePolicyModel,
    BoundaryPrototypeMixtureMinutePolicyModel,
    ChronologicalMinutePolicyModel,
)
from trading_storage import read_shard_array


MINUTE_SEQUENCE_BOUNDARY_TCN_CONTRACT = (
    BOUNDARY_SEQUENCE_ARCHITECTURE_CONTRACT
)
MINUTE_SEQUENCE_BOUNDARY_LONG_INPUT_CLOSE_COUNT = (
    LONG_CONTEXT_BOUNDARY_INPUT_CLOSE_COUNT
)
MINUTE_SEQUENCE_LONG_RECEPTIVE_FIELD = (
    LONG_CONTEXT_RECEPTIVE_FIELD_MINUTES
)
MINUTE_SEQUENCE_BOUNDARY_LONG_TCN_CONTRACT = (
    BOUNDARY_LONG_SEQUENCE_ARCHITECTURE_CONTRACT
)
MINUTE_SEQUENCE_BOUNDARY_TCN_POLICY_DROPOUT_CONTRACT = (
    BOUNDARY_SEQUENCE_POLICY_DROPOUT_ARCHITECTURE_CONTRACT
)
MINUTE_SEQUENCE_BOUNDARY_PROTOTYPE_MIXTURE_CONTRACT = (
    BOUNDARY_SEQUENCE_PROTOTYPE_MIXTURE_ARCHITECTURE_CONTRACT
)
MINUTE_SEQUENCE_BOUNDARY_MA_TCN_CONTRACT = (
    BOUNDARY_MA_SEQUENCE_ARCHITECTURE_CONTRACT
)


PATCH_TRANSFORMER_CONTRACT = (
    "causal-multiresolution-patch-token-transformer-direct-policy-v2"
)
TCN_CONTRACT = "causal-multiscale-dilated-tcn-direct-policy-v3"
RESIDUAL_MIXER_CONTRACT = (
    "causal-multiscale-learned-trend-residual-patch-mixer-direct-policy-v1"
)
LONG_CONTEXT_PATCH_MIXER_CONTRACT = (
    "causal-long-context-multiscale-patch-tcn-linear-query-pool-"
    "direct-policy-v1"
)
LEARNED_AGGREGATE_PATCH_MIXER_CONTRACT = (
    "causal-multiscale-learned-aggregate-patch-tcn-linear-query-pool-"
    "direct-policy-v1"
)
CAUSAL_FEATURE_COUNT = 5
FIXED_RETURN_SCALE = 10_000.0
FIXED_PATH_SCALE = 100.0
RMS_REFERENCE = 1e-4
PROTOTYPE_BASIS_CONTRACT = "fixed-soft-oracle-policy-prototype-basis-v1"
PROTOTYPE_BASIS_COUNT = 16


def load_policy_prototype_basis(
    config: dict[str, Any],
    *,
    action_count: int,
) -> tuple[Tensor, Tensor]:
    """Load and verify the immutable train-only v27 policy dictionary."""

    configured_path = config.get("prototypeBasisFile")
    if not isinstance(configured_path, str) or not configured_path.strip():
        raise ValueError("prototypeBasisFile must be a non-empty path")
    prototype_count = _config_int(config, "prototypeCount")
    if prototype_count != PROTOTYPE_BASIS_COUNT:
        raise ValueError(
            f"prototypeCount must be exactly {PROTOTYPE_BASIS_COUNT} for "
            "the v27 architecture contract"
        )
    expected_content_hash = _config_sha256(
        config,
        "prototypeBasisContentSha256",
    )
    expected_reference_hash = _config_sha256(
        config,
        "prototypeBasisReferenceSha256",
    )
    expected_source_fingerprint = _config_sha256(
        config,
        "prototypeSourceTrainFingerprintSha256",
    )
    repo_root = Path(__file__).resolve().parents[1]
    reference_root = (
        repo_root
        / "data/training/immutable/refs/models/joint-price-oracle/"
        "prototype-basis"
    ).resolve()
    reference_file = Path(configured_path)
    if not reference_file.is_absolute():
        reference_file = repo_root / reference_file
    reference_file = reference_file.resolve()
    try:
        reference_file.relative_to(reference_root)
    except ValueError as error:
        raise ValueError(
            "prototypeBasisFile must remain under the immutable joint "
            "price-oracle prototype-basis reference directory"
        ) from error
    if hashlib.sha256(reference_file.read_bytes()).hexdigest() \
            != expected_reference_hash:
        raise ValueError(
            "prototype basis reference hash does not match the model contract"
        )

    shard, raw = read_shard_array(
        reference_file,
        "<f4",
        (prototype_count, action_count),
    )
    reference = shard.reference
    stored = reference["object"]
    layout = reference["layout"]
    metadata = reference.get("metadata")
    if stored.get("contentHash") != expected_content_hash:
        raise ValueError(
            "prototype basis content hash does not match the model contract"
        )
    if not isinstance(metadata, dict) \
            or metadata.get("artifactContract") != PROTOTYPE_BASIS_CONTRACT \
            or metadata.get("fitSplit") != "train" \
            or metadata.get("validationUsedForFit") is not False \
            or metadata.get("testReferencesOpened") != 0 \
            or metadata.get("testPayloadsOpened") != 0 \
            or metadata.get("hardAssignmentsUsedForSupervision") is not False \
            or metadata.get("modelUse") \
            != "continuous-simplex-weights-times-fixed-prototypes":
        raise ValueError(
            "prototype basis is not a verified train-only soft-mixture artifact"
        )
    if metadata.get("sourceTrainFingerprintSha256") \
            != expected_source_fingerprint:
        raise ValueError(
            "prototype basis source-train fingerprint does not match the "
            "model contract"
        )
    if metadata.get("matrixSha256") != expected_content_hash:
        raise ValueError("prototype basis matrix fingerprint is inconsistent")
    if layout.get("encoding") != "raw-row-major" \
            or layout.get("dtype") != "float32-le" \
            or layout.get("rows") != prototype_count \
            or layout.get("columns") != action_count \
            or layout.get("rowMeaning") \
            != "soft-oracle-action-probability-prototype":
        raise ValueError("prototype basis tensor layout is incompatible")

    prototypes = np.asarray(raw, dtype=np.float32)
    raw_weights = metadata.get("trainPrototypeMixtureWeights")
    weights = np.asarray(raw_weights, dtype=np.float32)
    if not np.isfinite(prototypes).all() \
            or bool((prototypes <= 0).any()) \
            or not np.allclose(
                prototypes.sum(axis=1),
                1,
                atol=2e-6,
                rtol=2e-6,
            ):
        raise ValueError(
            "prototype basis must contain finite, positive normalized rows"
        )
    if weights.shape != (prototype_count,) \
            or not np.isfinite(weights).all() \
            or bool((weights <= 0).any()) \
            or not np.isclose(weights.sum(), 1, atol=2e-6, rtol=2e-6):
        raise ValueError(
            "prototype basis train mixture weights are invalid"
        )
    return (
        torch.from_numpy(prototypes.copy()),
        torch.from_numpy(weights.copy()),
    )


def causal_standardized_close_features(
    closes: Tensor,
    epsilon: float = 1e-8,
) -> Tensor:
    """Build five causal shape-and-scale features as ``[batch, 5, time]``.

    Every feature at index ``t`` depends only on closes through ``t``.  The
    cumulative RMS avoids the suffix leakage that a whole-window RevIN scale
    would introduce into intermediate patch or TCN states.  Channels are:

    0. return divided by causal RMS (local shape),
    1. path from the first close divided by Brownian RMS scale (path shape),
    2. raw log return in fixed 1e-4 / basis-point-like units,
    3. raw log path in fixed percent-like units, and
    4. log causal RMS relative to a fixed 1e-4 reference.

    The last three channels deliberately preserve absolute scale, which is
    required to distinguish movements below and above fixed trading friction.
    """
    if closes.ndim != 3 or closes.shape[2] != 1 or closes.shape[1] < 2:
        raise ValueError("closes must have shape [batch, time, 1]")
    if epsilon <= 0 or not math.isfinite(epsilon):
        raise ValueError("feature epsilon must be finite and positive")
    log_closes = closes.float().clamp_min(epsilon).log().squeeze(-1)
    log_returns = torch.cat((
        torch.zeros_like(log_closes[:, :1]),
        log_closes[:, 1:] - log_closes[:, :-1],
    ), dim=1)
    count = torch.arange(
        1,
        log_returns.shape[1] + 1,
        device=log_returns.device,
        dtype=torch.float32,
    ).view(1, -1)
    cumulative_rms = (
        log_returns.float().square().cumsum(dim=1) / count
        + epsilon
    ).sqrt()
    normalized_returns = log_returns / cumulative_rms
    relative_path = log_closes - log_closes[:, :1]
    normalized_path = relative_path / (
        cumulative_rms * count.sqrt()
    ).clamp_min(epsilon)
    fixed_scale_returns = log_returns * FIXED_RETURN_SCALE
    fixed_scale_path = relative_path * FIXED_PATH_SCALE
    log_rms_scale = torch.log(
        cumulative_rms.clamp_min(epsilon) / RMS_REFERENCE
    )
    # A smooth bound protects mixed-precision training from isolated bad ticks
    # without introducing a data-dependent branch into the exported graph.
    return torch.stack((
        torch.tanh(normalized_returns / 8.0) * 8.0,
        torch.tanh(normalized_path / 8.0) * 8.0,
        torch.tanh(fixed_scale_returns / 8.0) * 8.0,
        torch.tanh(fixed_scale_path / 8.0) * 8.0,
        torch.tanh(log_rms_scale / 4.0) * 4.0,
    ), dim=1)


def _representation_regularizers(values: Tensor) -> tuple[Tensor, Tensor]:
    values_float = values.float()
    mean = values_float.mean(dim=-1, keepdim=True)
    variance = (values_float - mean).square().mean(dim=-1)
    return mean.squeeze(-1).square().mean(), (variance - 1.0).square().mean()


class CoarseToFineForecastHead(nn.Module):
    """Produce every future second from minute-scale anchors plus a low rank residual."""

    def __init__(
        self,
        input_width: int,
        forecast_horizon: int,
        coarse_steps: int,
        residual_rank: int,
        maximum_log_movement: float,
    ) -> None:
        super().__init__()
        if min(input_width, forecast_horizon, coarse_steps, residual_rank) < 1:
            raise ValueError("forecast dimensions must be positive")
        if forecast_horizon % coarse_steps != 0:
            raise ValueError("forecast horizon must divide into coarse steps")
        if residual_rank > min(input_width, forecast_horizon):
            raise ValueError("forecast residual rank is too large")
        if maximum_log_movement <= 0 or not math.isfinite(maximum_log_movement):
            raise ValueError("maximum log movement must be finite and positive")
        self.forecast_horizon = int(forecast_horizon)
        self.coarse_steps = int(coarse_steps)
        self.maximum_log_movement = float(maximum_log_movement)
        self.coarse_projection = nn.Linear(input_width, coarse_steps)
        self.residual_projection = nn.Linear(input_width, residual_rank)
        self.residual_basis = nn.Parameter(torch.empty(
            residual_rank,
            forecast_horizon,
        ))
        self.register_buffer(
            "interpolation",
            self._interpolation_matrix(forecast_horizon, coarse_steps),
            persistent=True,
        )
        self.reset_parameters()

    @staticmethod
    def _interpolation_matrix(
        forecast_horizon: int,
        coarse_steps: int,
    ) -> Tensor:
        interval = forecast_horizon // coarse_steps
        result = torch.zeros(coarse_steps, forecast_horizon)
        for horizon_index in range(forecast_horizon):
            position = (horizon_index + 1) / interval
            upper = min(coarse_steps - 1, int(math.ceil(position)) - 1)
            lower = upper - 1
            upper_horizon = (upper + 1) * interval
            if lower < 0:
                result[upper, horizon_index] = (
                    (horizon_index + 1) / upper_horizon
                )
            else:
                lower_horizon = (lower + 1) * interval
                fraction = (
                    (horizon_index + 1 - lower_horizon)
                    / (upper_horizon - lower_horizon)
                )
                result[lower, horizon_index] = 1.0 - fraction
                result[upper, horizon_index] = fraction
        return result

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.coarse_projection.weight)
        nn.init.zeros_(self.coarse_projection.bias)
        nn.init.zeros_(self.residual_projection.weight)
        nn.init.zeros_(self.residual_projection.bias)
        nn.init.normal_(self.residual_basis, mean=0.0, std=0.01)

    def forward(self, hidden: Tensor) -> Tensor:
        coarse = self.coarse_projection(hidden)
        residual = self.residual_projection(hidden) @ self.residual_basis
        raw = coarse @ self.interpolation + residual
        return torch.tanh(raw) * self.maximum_log_movement


class DirectPolicyHead(nn.Module):
    def __init__(
        self,
        input_width: int,
        hidden_width: int,
        action_count: int,
        dropout: float,
        policy_logit_rank: int | None = None,
    ) -> None:
        super().__init__()
        if min(input_width, hidden_width, action_count) < 1:
            raise ValueError("policy dimensions must be positive")
        output = (
            nn.Linear(hidden_width, action_count)
            if policy_logit_rank is None
            else FixedDctOrdinalLogitProjection(
                hidden_width,
                action_count,
                policy_logit_rank,
            )
        )
        self.layers = nn.Sequential(
            nn.LayerNorm(input_width),
            nn.Linear(input_width, hidden_width),
            nn.GELU(),
            nn.Dropout(dropout),
            output,
        )
        if policy_logit_rank is None:
            nn.init.normal_(self.layers[-1].weight, mean=0.0, std=0.01)
            nn.init.zeros_(self.layers[-1].bias)

    def forward(self, hidden: Tensor) -> Tensor:
        return self.layers(hidden)


class OnnxFriendlyCausalSelfAttention(nn.Module):
    def __init__(
        self,
        width: int,
        head_count: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if width < 1 or head_count < 1 or width % head_count != 0:
            raise ValueError("attention width must divide into its heads")
        self.width = int(width)
        self.head_count = int(head_count)
        self.head_width = width // head_count
        self.qkv = nn.Linear(width, width * 3)
        self.output = nn.Linear(width, width)
        self.attention_dropout = nn.Dropout(dropout)

    def forward(self, values: Tensor, allowed: Tensor) -> Tensor:
        batch, token_count, _width = values.shape
        qkv = self.qkv(values).reshape(
            batch,
            token_count,
            3,
            self.head_count,
            self.head_width,
        ).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(dim=0)
        scores = (query @ key.transpose(-1, -2)) / math.sqrt(self.head_width)
        scores = scores.masked_fill(~allowed, -10_000.0)
        probabilities = self.attention_dropout(torch.softmax(
            scores.float(),
            dim=-1,
        ).to(dtype=scores.dtype))
        attended = (probabilities @ value).transpose(1, 2).reshape(
            batch,
            token_count,
            self.width,
        )
        return self.output(attended)


class CausalTransformerBlock(nn.Module):
    def __init__(
        self,
        width: int,
        head_count: int,
        feed_forward_width: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(width)
        self.attention = OnnxFriendlyCausalSelfAttention(
            width,
            head_count,
            dropout,
        )
        self.feed_forward_norm = nn.LayerNorm(width)
        self.feed_forward = nn.Sequential(
            nn.Linear(width, feed_forward_width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feed_forward_width, width),
            nn.Dropout(dropout),
        )

    def forward(self, values: Tensor, allowed: Tensor) -> Tensor:
        values = values + self.attention(self.attention_norm(values), allowed)
        return values + self.feed_forward(self.feed_forward_norm(values))


class MultiResolutionPatchTransformer(nn.Module):
    """Strictly causal patch-token Transformer with a direct policy query."""

    architecture_contract = PATCH_TRANSFORMER_CONTRACT

    def __init__(
        self,
        context_length: int = 3_600,
        forecast_horizon: int = 3_600,
        action_count: int = OUTPUT_ACTION_COUNT,
        patch_sizes: Sequence[int] = (15, 60, 300),
        model_width: int = 128,
        attention_heads: int = 4,
        layer_count: int = 4,
        feed_forward_width: int = 384,
        policy_hidden_width: int = 192,
        policy_logit_rank: int | None = None,
        dropout: float = 0.05,
        forecast_coarse_steps: int = 60,
        forecast_rank: int = 32,
        maximum_log_movement: float = 1.0,
        feature_epsilon: float = 1e-8,
    ) -> None:
        super().__init__()
        sizes = _validated_int_sequence(
            patch_sizes,
            "patch_sizes",
            minimum=1,
        )
        if min(
            context_length,
            forecast_horizon,
            action_count,
            model_width,
            attention_heads,
            layer_count,
            feed_forward_width,
            policy_hidden_width,
        ) < 1:
            raise ValueError("Transformer dimensions must be positive")
        if any(context_length % patch_size != 0 for patch_size in sizes):
            raise ValueError("each patch size must divide the context length")
        if model_width % attention_heads != 0:
            raise ValueError("model width must divide into attention heads")
        if feed_forward_width < model_width:
            raise ValueError("feed-forward width cannot be smaller than model width")
        _validate_dropout(dropout)
        self.context_length = int(context_length)
        self.forecast_horizon = int(forecast_horizon)
        self.variable_count = 1
        self.action_count = int(action_count)
        self.architecture_contract = architecture_contract_with_policy_decoder(
            PATCH_TRANSFORMER_CONTRACT,
            policy_logit_rank,
        )
        self.patch_sizes = sizes
        self.feature_epsilon = float(feature_epsilon)
        self.patch_embeddings = nn.ModuleList([
            nn.Conv1d(
                CAUSAL_FEATURE_COUNT,
                model_width,
                kernel_size=patch_size,
                stride=patch_size,
            )
            for patch_size in sizes
        ])

        metadata: list[tuple[int, int]] = []
        for scale_index, patch_size in enumerate(sizes):
            metadata.extend(
                (end, scale_index)
                for end in range(patch_size - 1, context_length, patch_size)
            )
        order = sorted(
            range(len(metadata)),
            key=lambda index: (metadata[index][0], metadata[index][1]),
        )
        ordered_end_times = [metadata[index][0] for index in order]
        ordered_scales = [metadata[index][1] for index in order]
        token_count = len(order)
        if token_count > 1_024:
            raise ValueError("patch configuration creates too many attention tokens")
        self.register_buffer(
            "token_order",
            torch.tensor(order, dtype=torch.long),
            persistent=True,
        )
        self.register_buffer(
            "patch_end_times",
            torch.tensor(ordered_end_times, dtype=torch.long),
            persistent=True,
        )
        self.register_buffer(
            "patch_scale_indexes",
            torch.tensor(ordered_scales, dtype=torch.long),
            persistent=True,
        )
        all_times = torch.tensor(
            ordered_end_times + [context_length],
            dtype=torch.long,
        )
        allowed = all_times.view(-1, 1) >= all_times.view(1, -1)
        self.register_buffer(
            "attention_allowed",
            allowed.view(1, 1, token_count + 1, token_count + 1),
            persistent=True,
        )
        self.position_embedding = nn.Parameter(torch.empty(
            1,
            token_count,
            model_width,
        ))
        self.scale_embedding = nn.Parameter(torch.empty(
            len(sizes),
            model_width,
        ))
        self.policy_query = nn.Parameter(torch.empty(1, 1, model_width))
        self.blocks = nn.ModuleList([
            CausalTransformerBlock(
                model_width,
                attention_heads,
                feed_forward_width,
                dropout,
            )
            for _ in range(layer_count)
        ])
        self.final_norm = nn.LayerNorm(model_width)
        self.policy_head = DirectPolicyHead(
            model_width,
            policy_hidden_width,
            action_count,
            dropout,
            policy_logit_rank,
        )
        self.forecast_head = CoarseToFineForecastHead(
            model_width,
            forecast_horizon,
            forecast_coarse_steps,
            forecast_rank,
            maximum_log_movement,
        )
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)
        nn.init.normal_(self.scale_embedding, mean=0.0, std=0.02)
        nn.init.normal_(self.policy_query, mean=0.0, std=0.02)

    def encode_history(self, closes: Tensor) -> Tensor:
        self._validate_input(closes)
        features = causal_standardized_close_features(
            closes,
            self.feature_epsilon,
        )
        grouped = torch.cat([
            embedding(features).transpose(1, 2)
            for embedding in self.patch_embeddings
        ], dim=1)
        tokens = grouped.index_select(1, self.token_order)
        tokens = (
            tokens
            + self.position_embedding
            + self.scale_embedding.index_select(0, self.patch_scale_indexes)
                .unsqueeze(0)
        )
        query = self.policy_query.expand(tokens.shape[0], -1, -1)
        encoded = torch.cat((tokens, query), dim=1)
        for block in self.blocks:
            encoded = block(encoded, self.attention_allowed)
        return self.final_norm(encoded)

    def forward_with_forecast(self, closes: Tensor) -> JointPriceOracleOutput:
        encoded = self.encode_history(closes)
        summary = encoded[:, -1, :]
        policy_logits = self.policy_head(summary)
        log_movements = self.forecast_head(summary).unsqueeze(-1)
        last_close = closes.float()[:, -1:, :].clamp_min(self.feature_epsilon)
        predicted_closes = last_close * torch.exp(log_movements)
        mean_penalty, variance_penalty = _representation_regularizers(summary)
        return JointPriceOracleOutput(
            policy_logits=policy_logits,
            predicted_closes=predicted_closes,
            predicted_log_movements=log_movements,
            predicted_movements=torch.expm1(log_movements),
            soft_layer_norm_mean=mean_penalty,
            soft_layer_norm_variance=variance_penalty,
        )

    def forward(self, closes: Tensor) -> Tensor:
        return self.forward_with_forecast(closes).policy_logits

    def _validate_input(self, closes: Tensor) -> None:
        if closes.ndim != 3 or closes.shape[1] != self.context_length \
                or closes.shape[2] != 1:
            raise ValueError(
                "patch Transformer expects [batch, context_length, 1] closes"
            )


class LearnedMultiScaleTrendResidual(nn.Module):
    """Create exact trend/residual pairs from causal learned aggregates.

    Every learned kernel starts as the corresponding trailing moving average.
    A learned mixture can then move away from that baseline without sacrificing
    the exact ``trend + residual == input`` decomposition.
    """

    def __init__(self, scales: Sequence[int]) -> None:
        super().__init__()
        self.scales = _validated_int_sequence(
            scales,
            "aggregate_scales",
            minimum=1,
        )
        self.weight_logits = nn.ParameterList([
            nn.Parameter(torch.zeros(scale))
            for scale in self.scales
        ])
        self.learned_mix_logits = nn.Parameter(torch.zeros(len(self.scales)))

    def learned_weights(self) -> tuple[Tensor, ...]:
        return tuple(
            torch.softmax(logits.float(), dim=0)
            for logits in self.weight_logits
        )

    def forward(
        self,
        values: Tensor,
    ) -> tuple[tuple[Tensor, ...], tuple[Tensor, ...]]:
        if values.ndim != 2 or values.shape[1] < max(self.scales):
            raise ValueError(
                "trend decomposition expects [batch, time] values covering "
                "every aggregate scale"
            )
        channel_first = values.unsqueeze(1)
        trends: list[Tensor] = []
        residuals: list[Tensor] = []
        for index, (scale, weights) in enumerate(zip(
            self.scales,
            self.learned_weights(),
            strict=True,
        )):
            moving_average = causal_moving_average(
                values.unsqueeze(-1),
                scale,
            ).squeeze(-1)
            padded = functional.pad(
                channel_first,
                (scale - 1, 0),
                mode="replicate",
            )
            learned_aggregate = functional.conv1d(
                padded,
                weights.to(dtype=padded.dtype).view(1, 1, scale),
            ).squeeze(1)
            learned_mix = torch.sigmoid(
                self.learned_mix_logits[index]
            ).to(dtype=values.dtype)
            trend = (
                moving_average * (1.0 - learned_mix)
                + learned_aggregate * learned_mix
            )
            trends.append(trend)
            residuals.append(values - trend)
        return tuple(trends), tuple(residuals)


class CausalReversibleStreamNormalizer(nn.Module):
    """RLinear-style normalization with a separate state at every prefix."""

    def __init__(self, epsilon: float = 1e-8) -> None:
        super().__init__()
        if epsilon <= 0 or not math.isfinite(epsilon):
            raise ValueError("normalization epsilon must be finite and positive")
        self.epsilon = float(epsilon)

    def normalize(self, values: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if values.ndim != 2 or values.shape[1] < 1:
            raise ValueError(
                "causal reversible normalization expects [batch, time] values"
            )
        values_float = values.float()
        # Subtracting the first value before accumulating moments avoids the
        # precision loss of E[x^2] - E[x]^2 for log prices around 10-12.
        origin = values_float[:, :1]
        shifted = values_float - origin
        count = torch.arange(
            1,
            values.shape[1] + 1,
            device=values.device,
            dtype=torch.float32,
        ).view(1, -1)
        shifted_mean = shifted.cumsum(dim=1) / count
        variance = (
            shifted.square().cumsum(dim=1) / count
            - shifted_mean.square()
        ).clamp_min(0.0)
        mean = (origin + shifted_mean).detach()
        standard_deviation = (variance + self.epsilon).sqrt().detach()
        normalized = (values_float - mean) / standard_deviation
        return normalized, mean, standard_deviation

    @staticmethod
    def denormalize(
        normalized: Tensor,
        mean: Tensor,
        standard_deviation: Tensor,
    ) -> Tensor:
        if normalized.ndim != 2 \
                or mean.shape != normalized.shape \
                or standard_deviation.shape != normalized.shape:
            raise ValueError("causal normalization state is incompatible")
        return normalized * standard_deviation + mean


class CausalTokenMixerBlock(nn.Module):
    """Cheap PatchTST-like temporal mixing with an explicit causal mask."""

    def __init__(
        self,
        token_count: int,
        width: int,
        feed_forward_width: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if min(token_count, width, feed_forward_width) < 1:
            raise ValueError("token mixer dimensions must be positive")
        if feed_forward_width < width:
            raise ValueError(
                "stream feed-forward width cannot be smaller than stream width"
            )
        self.token_count = int(token_count)
        self.temporal_norm = nn.LayerNorm(width)
        self.temporal_logits = nn.Parameter(torch.zeros(
            token_count,
            token_count,
        ))
        self.register_buffer(
            "causal_mask",
            torch.ones(token_count, token_count, dtype=torch.bool).tril(),
            persistent=True,
        )
        self.temporal_projection = nn.Linear(width, width)
        self.temporal_gain_logit = nn.Parameter(torch.tensor(-2.0))
        self.channel_norm = nn.LayerNorm(width)
        self.channel_mixer = nn.Sequential(
            nn.Linear(width, feed_forward_width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feed_forward_width, width),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def temporal_weights(self) -> Tensor:
        return torch.softmax(
            self.temporal_logits.float().masked_fill(
                ~self.causal_mask,
                -10_000.0,
            ),
            dim=-1,
        )

    def forward(self, values: Tensor) -> Tensor:
        if values.ndim != 3 or values.shape[1] != self.token_count:
            raise ValueError("token mixer received an incompatible sequence")
        weights = self.temporal_weights().to(dtype=values.dtype)
        mixed = weights.unsqueeze(0) @ self.temporal_norm(values)
        mixed = self.temporal_projection(mixed)
        gain = torch.sigmoid(self.temporal_gain_logit).to(dtype=values.dtype)
        values = values + gain * self.dropout(mixed)
        return values + self.channel_mixer(self.channel_norm(values))


class ResidualStreamEncoder(nn.Module):
    """Patch one normalized stream, then mix its small causal token sequence."""

    def __init__(
        self,
        context_length: int,
        patch_size: int,
        width: int,
        layer_count: int,
        feed_forward_width: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if context_length % patch_size != 0:
            raise ValueError("stream patch size must divide the context length")
        token_count = context_length // patch_size
        self.patch_size = int(patch_size)
        self.token_count = int(token_count)
        self.patch_embedding = nn.Conv1d(
            1,
            width,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.position_embedding = nn.Parameter(torch.empty(
            1,
            token_count,
            width,
        ))
        self.blocks = nn.ModuleList([
            CausalTokenMixerBlock(
                token_count,
                width,
                feed_forward_width,
                dropout,
            )
            for _ in range(layer_count)
        ])
        self.final_norm = nn.LayerNorm(width)
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)

    def forward(self, values: Tensor) -> Tensor:
        if values.ndim != 2:
            raise ValueError("stream encoder expects [batch, time] values")
        tokens = self.patch_embedding(values.unsqueeze(1)).transpose(1, 2)
        tokens = tokens + self.position_embedding
        for block in self.blocks:
            tokens = block(tokens)
        return self.final_norm(tokens)


class GatedStreamFusion(nn.Module):
    """Fuse stream values and gates in one projection, plus raw scale state."""

    def __init__(
        self,
        stream_count: int,
        stream_summary_width: int,
        fusion_width: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if min(stream_count, stream_summary_width, fusion_width) < 1:
            raise ValueError("stream fusion dimensions must be positive")
        self.stream_count = int(stream_count)
        fused_input_width = stream_summary_width + 2
        self.stream_norm = nn.LayerNorm(fused_input_width)
        self.stream_embedding = nn.Parameter(torch.empty(
            1,
            stream_count,
            fused_input_width,
        ))
        # Value and gate are intentionally one fused projection.  This keeps
        # gating coupled to the representation it is selecting.
        self.fused_value_gate_projection = nn.Linear(
            fused_input_width,
            fusion_width + 1,
        )
        scale_width = stream_count * 2 + 3
        self.scale_projection = nn.Sequential(
            nn.LayerNorm(scale_width),
            nn.Linear(scale_width, fusion_width),
            nn.GELU(),
        )
        self.output = nn.Sequential(
            nn.LayerNorm(fusion_width),
            nn.Linear(fusion_width, fusion_width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_width, fusion_width),
            nn.GELU(),
        )
        nn.init.normal_(self.stream_embedding, mean=0.0, std=0.02)

    def forward(
        self,
        stream_summaries: Tensor,
        stream_scale_states: Tensor,
        global_scale_state: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if stream_summaries.ndim != 3 \
                or stream_summaries.shape[1] != self.stream_count \
                or stream_scale_states.shape != (
                    stream_summaries.shape[0],
                    self.stream_count,
                    2,
                ) \
                or global_scale_state.shape != (
                    stream_summaries.shape[0],
                    3,
                ):
            raise ValueError("stream fusion inputs are incompatible")
        fused_input = torch.cat((
            stream_summaries,
            stream_scale_states,
        ), dim=-1)
        projected = self.fused_value_gate_projection(
            self.stream_norm(fused_input) + self.stream_embedding
        )
        values = functional.gelu(projected[:, :, :-1])
        gates = torch.softmax(projected[:, :, -1].float(), dim=1).to(
            dtype=values.dtype
        )
        selected = (values * gates.unsqueeze(-1)).sum(dim=1)
        scale_input = torch.cat((
            stream_scale_states.flatten(start_dim=1),
            global_scale_state,
        ), dim=-1)
        hidden = selected + self.scale_projection(scale_input)
        return self.output(hidden), gates


class MultiScaleResidualMixer(nn.Module):
    """Causal learned trend/residual streams with scale-aware gated fusion."""

    architecture_contract = RESIDUAL_MIXER_CONTRACT

    def __init__(
        self,
        context_length: int = 3_600,
        forecast_horizon: int = 3_600,
        action_count: int = OUTPUT_ACTION_COUNT,
        aggregate_scales: Sequence[int] = (60, 300, 900),
        stream_width: int = 64,
        stream_mixer_layers: int = 2,
        stream_feed_forward_width: int = 128,
        fusion_width: int = 192,
        policy_hidden_width: int = 192,
        policy_logit_rank: int | None = None,
        dropout: float = 0.05,
        forecast_coarse_steps: int = 60,
        forecast_rank: int = 32,
        maximum_log_movement: float = 1.0,
        feature_epsilon: float = 1e-8,
    ) -> None:
        super().__init__()
        scales = _validated_int_sequence(
            aggregate_scales,
            "aggregate_scales",
            minimum=1,
        )
        if min(
            context_length,
            forecast_horizon,
            action_count,
            stream_width,
            stream_mixer_layers,
            stream_feed_forward_width,
            fusion_width,
            policy_hidden_width,
        ) < 1:
            raise ValueError("residual mixer dimensions must be positive")
        if stream_feed_forward_width < stream_width:
            raise ValueError(
                "stream feed-forward width cannot be smaller than stream width"
            )
        if any(context_length % scale != 0 for scale in scales):
            raise ValueError("each aggregate scale must divide the context length")
        if any(context_length // scale > 256 for scale in scales):
            raise ValueError(
                "aggregate scales must produce at most 256 tokens per stream"
            )
        if feature_epsilon <= 0 or not math.isfinite(feature_epsilon):
            raise ValueError("feature epsilon must be finite and positive")
        _validate_dropout(dropout)
        self.context_length = int(context_length)
        self.forecast_horizon = int(forecast_horizon)
        self.variable_count = 1
        self.action_count = int(action_count)
        self.architecture_contract = architecture_contract_with_policy_decoder(
            RESIDUAL_MIXER_CONTRACT,
            policy_logit_rank,
        )
        self.aggregate_scales = scales
        self.stream_scales = tuple(
            scale
            for scale in scales
            for _stream_kind in range(2)
        )
        self.feature_epsilon = float(feature_epsilon)
        self.decomposition = LearnedMultiScaleTrendResidual(scales)
        self.normalizer = CausalReversibleStreamNormalizer(feature_epsilon)
        self.stream_encoders = nn.ModuleList([
            ResidualStreamEncoder(
                context_length,
                scale,
                stream_width,
                stream_mixer_layers,
                stream_feed_forward_width,
                dropout,
            )
            for scale in scales
            for _stream_kind in range(2)
        ])
        self.fusion = GatedStreamFusion(
            len(self.stream_scales),
            stream_width * 2,
            fusion_width,
            dropout,
        )
        self.policy_head = DirectPolicyHead(
            fusion_width,
            policy_hidden_width,
            action_count,
            dropout,
            policy_logit_rank,
        )
        self.forecast_head = CoarseToFineForecastHead(
            fusion_width,
            forecast_horizon,
            forecast_coarse_steps,
            forecast_rank,
            maximum_log_movement,
        )

    def _normalized_streams(
        self,
        closes: Tensor,
    ) -> tuple[tuple[Tensor, ...], Tensor, Tensor]:
        self._validate_input(closes)
        log_closes = closes.float().clamp_min(self.feature_epsilon).log()
        relative_log_path = (
            log_closes - log_closes[:, :1, :]
        ).squeeze(-1)
        trends, residuals = self.decomposition(relative_log_path)
        raw_streams = tuple(
            stream
            for pair in zip(trends, residuals, strict=True)
            for stream in pair
        )
        normalized_streams: list[Tensor] = []
        scale_states: list[Tensor] = []
        for stream in raw_streams:
            normalized, _mean, standard_deviation = self.normalizer.normalize(
                stream
            )
            normalized_streams.append(normalized)
            raw_path = (stream[:, -1] - stream[:, 0]) * FIXED_PATH_SCALE
            raw_path = torch.tanh(raw_path / 8.0) * 8.0
            log_scale = torch.log(
                standard_deviation[:, -1].clamp_min(self.feature_epsilon)
                / RMS_REFERENCE
            )
            log_scale = torch.tanh(log_scale / 4.0) * 4.0
            scale_states.append(torch.stack((raw_path, log_scale), dim=-1))
        global_scale_state = causal_standardized_close_features(
            closes,
            self.feature_epsilon,
        )[:, 2:, -1]
        return (
            tuple(normalized_streams),
            torch.stack(scale_states, dim=1),
            global_scale_state,
        )

    def encode_streams(self, closes: Tensor) -> tuple[Tensor, ...]:
        normalized, _scale_states, _global_scale_state = (
            self._normalized_streams(closes)
        )
        return tuple(
            encoder(stream)
            for encoder, stream in zip(
                self.stream_encoders,
                normalized,
                strict=True,
            )
        )

    def _encode_history_with_gates(
        self,
        closes: Tensor,
    ) -> tuple[Tensor, Tensor]:
        normalized, scale_states, global_scale_state = (
            self._normalized_streams(closes)
        )
        encoded = tuple(
            encoder(stream)
            for encoder, stream in zip(
                self.stream_encoders,
                normalized,
                strict=True,
            )
        )
        summaries = torch.stack([
            torch.cat((stream[:, -1], stream.mean(dim=1)), dim=-1)
            for stream in encoded
        ], dim=1)
        return self.fusion(summaries, scale_states, global_scale_state)

    def encode_history(self, closes: Tensor) -> Tensor:
        hidden, _gates = self._encode_history_with_gates(closes)
        return hidden

    def forward_with_forecast(self, closes: Tensor) -> JointPriceOracleOutput:
        summary = self.encode_history(closes)
        policy_logits = self.policy_head(summary)
        log_movements = self.forecast_head(summary).unsqueeze(-1)
        last_close = closes.float()[:, -1:, :].clamp_min(
            self.feature_epsilon
        )
        predicted_closes = last_close * torch.exp(log_movements)
        mean_penalty, variance_penalty = _representation_regularizers(summary)
        return JointPriceOracleOutput(
            policy_logits=policy_logits,
            predicted_closes=predicted_closes,
            predicted_log_movements=log_movements,
            predicted_movements=torch.expm1(log_movements),
            soft_layer_norm_mean=mean_penalty,
            soft_layer_norm_variance=variance_penalty,
        )

    def forward(self, closes: Tensor) -> Tensor:
        return self.forward_with_forecast(closes).policy_logits

    def _validate_input(self, closes: Tensor) -> None:
        if closes.ndim != 3 or closes.shape[1] != self.context_length \
                or closes.shape[2] != 1:
            raise ValueError(
                "residual mixer expects [batch, context_length, 1] closes"
            )


class CausalDepthwiseTcnBlock(nn.Module):
    def __init__(
        self,
        width: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if min(width, kernel_size, dilation) < 1:
            raise ValueError("TCN block dimensions must be positive")
        if kernel_size < 2:
            raise ValueError("TCN kernel must contain at least two values")
        self.left_padding = (kernel_size - 1) * dilation
        self.norm = nn.LayerNorm(width)
        self.depthwise = nn.Conv1d(
            width,
            width,
            kernel_size=kernel_size,
            dilation=dilation,
            groups=width,
        )
        self.gate_projection = nn.Conv1d(width, width * 2, kernel_size=1)
        self.output_projection = nn.Conv1d(width, width, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: Tensor) -> Tensor:
        hidden = self.norm(values.transpose(1, 2)).transpose(1, 2)
        hidden = self.depthwise(functional.pad(
            hidden,
            (self.left_padding, 0),
        ))
        value, gate = self.gate_projection(hidden).chunk(2, dim=1)
        hidden = functional.gelu(value) * torch.sigmoid(gate)
        hidden = self.output_projection(self.dropout(hidden))
        return (values + hidden) / math.sqrt(2.0)


class TcnScaleBranch(nn.Module):
    def __init__(
        self,
        scale: int,
        width: int,
        kernel_size: int,
        dilations: Sequence[int],
        dropout: float,
    ) -> None:
        super().__init__()
        self.scale = int(scale)
        self.dilations = tuple(int(value) for value in dilations)
        self.receptive_field = 1 + (kernel_size - 1) * sum(self.dilations)
        self.stem = nn.Conv1d(
            CAUSAL_FEATURE_COUNT,
            width,
            kernel_size=scale,
            stride=scale,
        )
        self.blocks = nn.ModuleList([
            CausalDepthwiseTcnBlock(
                width,
                kernel_size,
                dilation,
                dropout,
            )
            for dilation in self.dilations
        ])
        self.final_norm = nn.LayerNorm(width)

    def forward(self, features: Tensor) -> Tensor:
        values = self.stem(features)
        for block in self.blocks:
            values = block(values)
        return self.final_norm(values.transpose(1, 2)).transpose(1, 2)


class MultiScaleDilatedTcn(nn.Module):
    """Causal multi-scale TCN whose policy bypasses its auxiliary forecast."""

    architecture_contract = TCN_CONTRACT

    def __init__(
        self,
        context_length: int = 3_600,
        forecast_horizon: int = 3_600,
        action_count: int = OUTPUT_ACTION_COUNT,
        scales: Sequence[int] = (5, 30, 60),
        tcn_width: int = 96,
        kernel_size: int = 3,
        dilations: Sequence[int] = (1, 2, 4, 8, 16, 32, 64, 128, 256),
        fusion_width: int = 192,
        policy_hidden_width: int = 192,
        policy_logit_rank: int | None = None,
        dropout: float = 0.05,
        forecast_coarse_steps: int = 60,
        forecast_rank: int = 32,
        maximum_log_movement: float = 1.0,
        feature_epsilon: float = 1e-8,
    ) -> None:
        super().__init__()
        scales_value = _validated_int_sequence(scales, "scales", minimum=1)
        dilations_value = _validated_int_sequence(
            dilations,
            "dilations",
            minimum=1,
        )
        if min(
            context_length,
            forecast_horizon,
            action_count,
            tcn_width,
            kernel_size,
            fusion_width,
            policy_hidden_width,
        ) < 1:
            raise ValueError("TCN dimensions must be positive")
        if any(context_length % scale != 0 for scale in scales_value):
            raise ValueError("each TCN scale must divide the context length")
        if tuple(sorted(dilations_value)) != dilations_value:
            raise ValueError("TCN dilations must be strictly increasing")
        if kernel_size < 2:
            raise ValueError("TCN kernel must contain at least two values")
        _validate_dropout(dropout)
        self.context_length = int(context_length)
        self.forecast_horizon = int(forecast_horizon)
        self.variable_count = 1
        self.action_count = int(action_count)
        self.architecture_contract = architecture_contract_with_policy_decoder(
            TCN_CONTRACT,
            policy_logit_rank,
        )
        self.scales = scales_value
        self.feature_epsilon = float(feature_epsilon)
        branch_dilations = tuple(
            _covering_dilation_prefix(
                context_length // scale,
                kernel_size,
                dilations_value,
            )
            for scale in scales_value
        )
        self.branches = nn.ModuleList([
            TcnScaleBranch(
                scale,
                tcn_width,
                kernel_size,
                selected_dilations,
                dropout,
            )
            for scale, selected_dilations in zip(
                scales_value,
                branch_dilations,
                strict=True,
            )
        ])
        fusion_input_width = len(scales_value) * tcn_width * 2
        self.fusion = nn.Sequential(
            nn.LayerNorm(fusion_input_width),
            nn.Linear(fusion_input_width, fusion_width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_width, fusion_width),
            nn.GELU(),
        )
        self.policy_head = DirectPolicyHead(
            fusion_width,
            policy_hidden_width,
            action_count,
            dropout,
            policy_logit_rank,
        )
        self.forecast_head = CoarseToFineForecastHead(
            fusion_width,
            forecast_horizon,
            forecast_coarse_steps,
            forecast_rank,
            maximum_log_movement,
        )

    def encode_branches(self, closes: Tensor) -> tuple[Tensor, ...]:
        self._validate_input(closes)
        features = causal_standardized_close_features(
            closes,
            self.feature_epsilon,
        )
        return tuple(branch(features) for branch in self.branches)

    def encode_history(self, closes: Tensor) -> Tensor:
        branches = self.encode_branches(closes)
        summaries = torch.cat([
            value
            for branch in branches
            for value in (branch[:, :, -1], branch.mean(dim=-1))
        ], dim=-1)
        return self.fusion(summaries)

    def forward_with_forecast(self, closes: Tensor) -> JointPriceOracleOutput:
        summary = self.encode_history(closes)
        policy_logits = self.policy_head(summary)
        log_movements = self.forecast_head(summary).unsqueeze(-1)
        last_close = closes.float()[:, -1:, :].clamp_min(self.feature_epsilon)
        predicted_closes = last_close * torch.exp(log_movements)
        mean_penalty, variance_penalty = _representation_regularizers(summary)
        return JointPriceOracleOutput(
            policy_logits=policy_logits,
            predicted_closes=predicted_closes,
            predicted_log_movements=log_movements,
            predicted_movements=torch.expm1(log_movements),
            soft_layer_norm_mean=mean_penalty,
            soft_layer_norm_variance=variance_penalty,
        )

    def forward(self, closes: Tensor) -> Tensor:
        return self.forward_with_forecast(closes).policy_logits

    def _validate_input(self, closes: Tensor) -> None:
        if closes.ndim != 3 or closes.shape[1] != self.context_length \
                or closes.shape[2] != 1:
            raise ValueError("TCN expects [batch, context_length, 1] closes")


class LinearQueryPool(nn.Module):
    """Summarize a token sequence with O(tokens * queries) attention.

    This deliberately is not self-attention: a small, fixed set of learned
    queries reads the causal token encoder once.  Extending the input history
    therefore increases work and activations linearly rather than
    quadratically.
    """

    def __init__(self, width: int, query_count: int) -> None:
        super().__init__()
        if min(width, query_count) < 1:
            raise ValueError("query pool dimensions must be positive")
        self.width = int(width)
        self.query_count = int(query_count)
        self.norm = nn.LayerNorm(width)
        self.key_projection = nn.Linear(width, width)
        self.value_projection = nn.Linear(width, width)
        self.queries = nn.Parameter(torch.empty(query_count, width))
        nn.init.normal_(self.queries, mean=0.0, std=0.02)

    def forward(self, values: Tensor) -> Tensor:
        if values.ndim != 3 or values.shape[2] != self.width:
            raise ValueError("query pool expects [batch, token, width]")
        normalized = self.norm(values)
        keys = self.key_projection(normalized)
        pooled_values = self.value_projection(normalized)
        scores = torch.matmul(keys, self.queries.transpose(0, 1))
        scores = scores / math.sqrt(self.width)
        weights = torch.softmax(scores.float(), dim=1).to(dtype=values.dtype)
        return weights.transpose(1, 2) @ pooled_values


class LongContextPatchBranch(nn.Module):
    """Learn patches, mix them causally, then read them with linear queries."""

    def __init__(
        self,
        context_length: int,
        patch_size: int,
        width: int,
        kernel_size: int,
        dilations: Sequence[int],
        query_count: int,
        dropout: float,
        dilation_coverage: str = "conservative-span",
    ) -> None:
        super().__init__()
        if context_length % patch_size != 0:
            raise ValueError("long-context patch size must divide the context")
        self.patch_size = int(patch_size)
        self.token_count = context_length // patch_size
        if dilation_coverage == "conservative-span":
            self.dilations = _covering_dilation_prefix(
                self.token_count,
                kernel_size,
                dilations,
            )
        elif dilation_coverage == "exact-stacked-receptive-field":
            self.dilations = _exact_covering_dilation_prefix(
                self.token_count,
                kernel_size,
                dilations,
            )
        else:
            raise ValueError("unknown patch TCN dilation coverage contract")
        self.receptive_field = 1 + (kernel_size - 1) * sum(self.dilations)
        self.patch_projection = nn.Conv1d(
            CAUSAL_FEATURE_COUNT,
            width,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.position_embedding = nn.Parameter(torch.empty(
            1,
            self.token_count,
            width,
        ))
        self.blocks = nn.ModuleList([
            CausalDepthwiseTcnBlock(
                width,
                kernel_size,
                dilation,
                dropout,
            )
            for dilation in self.dilations
        ])
        self.final_norm = nn.LayerNorm(width)
        self.query_pool = LinearQueryPool(width, query_count)
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)

    def encode(self, features: Tensor) -> Tensor:
        values = self.patch_projection(features).transpose(1, 2)
        values = values + self.position_embedding
        values = values.transpose(1, 2)
        for block in self.blocks:
            values = block(values)
        return self.final_norm(values.transpose(1, 2))

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor]:
        encoded = self.encode(features)
        return encoded, self.query_pool(encoded)


class LongContextMultiScalePatchMixer(nn.Module):
    """Six-hour-capable PatchTST-like encoder with linear history cost.

    Learned non-overlapping patches retain within-minute and five-minute
    structure.  Strictly causal depthwise TCN blocks give each patch stream a
    full-history receptive field, while a fixed number of query pools extracts
    several regimes without the O(tokens**2) cost of self-attention.  The
    policy head is independent of the auxiliary forecast head and can be
    trained with raw oracle cross-entropy alone.
    """

    architecture_contract = LONG_CONTEXT_PATCH_MIXER_CONTRACT

    def __init__(
        self,
        context_length: int = 21_600,
        forecast_horizon: int = 3_600,
        action_count: int = OUTPUT_ACTION_COUNT,
        patch_sizes: Sequence[int] = (60, 300),
        encoder_width: int = 96,
        kernel_size: int = 3,
        dilations: Sequence[int] = (1, 2, 4, 8, 16, 32, 64, 128),
        query_count: int = 4,
        fusion_width: int = 256,
        policy_hidden_width: int = 256,
        policy_logit_rank: int | None = None,
        dropout: float = 0.05,
        forecast_coarse_steps: int = 60,
        forecast_rank: int = 32,
        maximum_log_movement: float = 1.0,
        feature_epsilon: float = 1e-8,
        dilation_coverage: str = "conservative-span",
        base_architecture_contract: str = LONG_CONTEXT_PATCH_MIXER_CONTRACT,
    ) -> None:
        super().__init__()
        sizes = _validated_int_sequence(
            patch_sizes,
            "patch_sizes",
            minimum=1,
        )
        dilation_values = _validated_int_sequence(
            dilations,
            "dilations",
            minimum=1,
        )
        if min(
            context_length,
            forecast_horizon,
            action_count,
            encoder_width,
            kernel_size,
            query_count,
            fusion_width,
            policy_hidden_width,
        ) < 1:
            raise ValueError("long-context model dimensions must be positive")
        if context_length < 3_600 or context_length > 86_400:
            raise ValueError(
                "long-context history must cover between one hour and one day"
            )
        if any(context_length % patch_size != 0 for patch_size in sizes):
            raise ValueError(
                "each long-context patch size must divide the context length"
            )
        if tuple(sorted(dilation_values)) != dilation_values:
            raise ValueError(
                "long-context dilations must be strictly increasing"
            )
        if kernel_size < 2:
            raise ValueError("long-context kernel must contain at least two values")
        if feature_epsilon <= 0 or not math.isfinite(feature_epsilon):
            raise ValueError("feature epsilon must be finite and positive")
        _validate_dropout(dropout)
        self.context_length = int(context_length)
        self.forecast_horizon = int(forecast_horizon)
        self.variable_count = 1
        self.action_count = int(action_count)
        self.patch_sizes = sizes
        self.feature_epsilon = float(feature_epsilon)
        self.architecture_contract = architecture_contract_with_policy_decoder(
            base_architecture_contract,
            policy_logit_rank,
        )
        self.branches = nn.ModuleList([
            LongContextPatchBranch(
                context_length,
                patch_size,
                encoder_width,
                kernel_size,
                dilation_values,
                query_count,
                dropout,
                dilation_coverage,
            )
            for patch_size in sizes
        ])
        fusion_input_width = (
            len(sizes) * (query_count + 1) * encoder_width
            + 3
        )
        self.fusion = nn.Sequential(
            nn.LayerNorm(fusion_input_width),
            nn.Linear(fusion_input_width, fusion_width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_width, fusion_width),
            nn.GELU(),
        )
        self.policy_head = DirectPolicyHead(
            fusion_width,
            policy_hidden_width,
            action_count,
            dropout,
            policy_logit_rank,
        )
        self.forecast_head = CoarseToFineForecastHead(
            fusion_width,
            forecast_horizon,
            forecast_coarse_steps,
            forecast_rank,
            maximum_log_movement,
        )

    def encode_branches(self, closes: Tensor) -> tuple[Tensor, ...]:
        self._validate_input(closes)
        features = causal_standardized_close_features(
            closes,
            self.feature_epsilon,
        )
        return tuple(branch.encode(features) for branch in self.branches)

    def encode_history(self, closes: Tensor) -> Tensor:
        self._validate_input(closes)
        features = causal_standardized_close_features(
            closes,
            self.feature_epsilon,
        )
        summaries: list[Tensor] = []
        for branch in self.branches:
            encoded, pooled = branch(features)
            summaries.extend((
                encoded[:, -1, :],
                pooled.flatten(start_dim=1),
            ))
        # Fixed-scale path and RMS state lets the policy retain friction-
        # relevant magnitude after normalized patch encoding.
        summaries.append(features[:, 2:, -1])
        return self.fusion(torch.cat(summaries, dim=-1))

    def forward_with_forecast(self, closes: Tensor) -> JointPriceOracleOutput:
        summary = self.encode_history(closes)
        policy_logits = self.policy_head(summary)
        log_movements = self.forecast_head(summary).unsqueeze(-1)
        last_close = closes.float()[:, -1:, :].clamp_min(
            self.feature_epsilon
        )
        predicted_closes = last_close * torch.exp(log_movements)
        mean_penalty, variance_penalty = _representation_regularizers(summary)
        return JointPriceOracleOutput(
            policy_logits=policy_logits,
            predicted_closes=predicted_closes,
            predicted_log_movements=log_movements,
            predicted_movements=torch.expm1(log_movements),
            soft_layer_norm_mean=mean_penalty,
            soft_layer_norm_variance=variance_penalty,
        )

    def forward_policy_logits(self, closes: Tensor) -> Tensor:
        """Run only the causal encoder and policy head."""
        return self.policy_head(self.encode_history(closes))

    def forward(self, closes: Tensor) -> Tensor:
        return self.forward_with_forecast(closes).policy_logits

    def _validate_input(self, closes: Tensor) -> None:
        if closes.ndim != 3 or closes.shape[1] != self.context_length \
                or closes.shape[2] != 1:
            raise ValueError(
                "long-context patch mixer expects "
                "[batch, context_length, 1] closes"
            )


def build_variant_model(config: dict[str, Any]) -> nn.Module:
    """Build one current-contract architecture variant from a plan-style config."""
    if not isinstance(config, dict):
        raise ValueError("variant model config must be an object")
    variant = config.get("variant")
    if variant not in {
        "patch_transformer",
        "dilated_tcn",
        "multiscale_residual_mixer",
        "long_context_patch_mixer",
        "learned_aggregate_patch_mixer",
        "minute_return_mlp",
        "minute_sequence_tcn",
        "minute_sequence_boundary_tcn",
        "minute_sequence_boundary_long_tcn",
        "minute_sequence_boundary_prototype_mixture_tcn",
        "minute_sequence_boundary_ma_tcn",
    }:
        raise ValueError(
            "variant must be 'patch_transformer', 'dilated_tcn', or "
            "'multiscale_residual_mixer', 'long_context_patch_mixer', or "
            "'learned_aggregate_patch_mixer', or "
            "'minute_return_mlp', 'minute_sequence_tcn', "
            "'minute_sequence_boundary_tcn', or "
            "'minute_sequence_boundary_long_tcn', or "
            "'minute_sequence_boundary_prototype_mixture_tcn', or "
            "'minute_sequence_boundary_ma_tcn'"
        )
    context_length = _config_int(config, "contextLength")
    forecast_horizon = _config_int(config, "forecastHorizon")
    action_count = _config_int(config, "actionCount")
    variable_count = _config_int(config, "variableCount", default=1)
    if forecast_horizon != 3_600:
        raise ValueError("oracle variants require 3,600 forecast steps")
    if variant == "minute_return_mlp" \
            and context_length != MINUTE_RETURN_INPUT_CLOSE_COUNT:
        raise ValueError(
            "minute-return MLP requires 3,601 closes for exactly 3,600 "
            "one-second returns"
        )
    if variant == "minute_sequence_tcn" \
            and context_length != MINUTE_SEQUENCE_INPUT_CLOSE_COUNT:
        raise ValueError(
            "minute-sequence TCN requires exactly 3,600 closes ending at "
            "the current decision"
        )
    corrected_sequence_variants = {
        "minute_sequence_boundary_tcn",
        "minute_sequence_boundary_long_tcn",
        "minute_sequence_boundary_prototype_mixture_tcn",
        "minute_sequence_boundary_ma_tcn",
    }
    if variant in corrected_sequence_variants:
        expected_context_length = (
            MINUTE_SEQUENCE_BOUNDARY_LONG_INPUT_CLOSE_COUNT
            if variant == "minute_sequence_boundary_long_tcn"
            else MINUTE_SEQUENCE_BOUNDARY_INPUT_CLOSE_COUNT
        )
        if context_length != expected_context_length:
            raise ValueError(
                "boundary-complete minute-sequence TCN requires exactly "
                f"{expected_context_length:,} closes for all "
                f"{expected_context_length - 1:,} one-second returns"
            )
    if variant not in {
        "long_context_patch_mixer",
        "learned_aggregate_patch_mixer",
        "minute_return_mlp",
        "minute_sequence_tcn",
        *corrected_sequence_variants,
    } \
            and context_length != 3_600:
        raise ValueError("short-context oracle variants require 3,600 input steps")
    if action_count != OUTPUT_ACTION_COUNT or variable_count != 1:
        raise ValueError("oracle variants require one close and 101 actions")
    if variant == "minute_return_mlp":
        if config.get("policyLogitRank") is not None:
            raise ValueError(
                "minute-return MLP emits direct 101 logits and does not "
                "support policyLogitRank"
            )
        return MinuteReturnOracleMlp(MinuteReturnMlpConfig(
            hidden_width=_config_int(
                config,
                "hiddenWidth",
                default=512,
            ),
            layer_count=_config_int(
                config,
                "layerCount",
                default=8,
            ),
            dropout=_config_float(config, "dropout", default=0.05),
            residual_gain=_config_float(
                config,
                "residualGain",
                default=1.0,
            ),
        ))
    if variant == "minute_sequence_tcn":
        if config.get("policyLogitRank") is not None:
            raise ValueError(
                "minute-sequence TCN emits direct 101 logits and does not "
                "support policyLogitRank"
            )
        receptive_field_minutes = _config_int(
            config,
            "receptiveFieldMinutes",
            default=MINUTE_SEQUENCE_RECEPTIVE_FIELD,
        )
        if receptive_field_minutes != MINUTE_SEQUENCE_RECEPTIVE_FIELD:
            raise ValueError(
                "minute-sequence TCN window variant requires an exact "
                "60-minute receptive field"
            )
        return ChronologicalMinutePolicyModel(
            receptive_field_minutes=receptive_field_minutes,
            token_width=_config_int(config, "tokenWidth", default=128),
            policy_hidden_width=_config_int(
                config,
                "policyHiddenWidth",
                default=192,
            ),
            action_count=action_count,
            dropout=_config_float(config, "dropout", default=0.05),
            feature_epsilon=_config_float(
                config,
                "featureEpsilon",
                default=1e-8,
            ),
        )
    if variant in corrected_sequence_variants:
        if config.get("policyLogitRank") is not None:
            raise ValueError(
                "boundary-complete minute-sequence TCN emits direct 101 "
                "logits and does not support policyLogitRank"
            )
        receptive_field_minutes = _config_int(
            config,
            "receptiveFieldMinutes",
            default=(
                MINUTE_SEQUENCE_LONG_RECEPTIVE_FIELD
                if variant == "minute_sequence_boundary_long_tcn"
                else MINUTE_SEQUENCE_RECEPTIVE_FIELD
            ),
        )
        expected_receptive_field = (
            MINUTE_SEQUENCE_LONG_RECEPTIVE_FIELD
            if variant == "minute_sequence_boundary_long_tcn"
            else MINUTE_SEQUENCE_RECEPTIVE_FIELD
        )
        if receptive_field_minutes != expected_receptive_field:
            raise ValueError(
                "boundary-complete minute-sequence TCN requires an exact "
                f"{expected_receptive_field}-minute receptive field"
            )
        if "policyDropout" in config \
                and variant != "minute_sequence_boundary_tcn":
            raise ValueError(
                "policyDropout is supported only by the boundary-complete "
                "non-MA minute TCN"
            )
        common_sequence = dict(
            receptive_field_minutes=receptive_field_minutes,
            token_width=_config_int(config, "tokenWidth", default=128),
            policy_hidden_width=_config_int(
                config,
                "policyHiddenWidth",
                default=192,
            ),
            action_count=action_count,
            dropout=_config_float(config, "dropout", default=0.05),
            policy_dropout=(
                _config_float(config, "policyDropout", default=0.0)
                if "policyDropout" in config
                else None
            ),
            scale_window_minutes=_config_int(
                config,
                "scaleWindowMinutes",
                default=60,
            ),
            feature_epsilon=_config_float(
                config,
                "featureEpsilon",
                default=1e-8,
            ),
        )
        if variant == "minute_sequence_boundary_prototype_mixture_tcn":
            prototypes, train_weights = load_policy_prototype_basis(
                config,
                action_count=action_count,
            )
            return BoundaryPrototypeMixtureMinutePolicyModel(
                policy_prototypes=prototypes,
                train_mixture_weights=train_weights,
                **common_sequence,
            )
        model_class = (
            BoundaryMaMinutePolicyModel
            if variant == "minute_sequence_boundary_ma_tcn"
            else (
                BoundaryCompleteLongContextMinutePolicyModel
                if variant == "minute_sequence_boundary_long_tcn"
                else BoundaryCompleteMinutePolicyModel
            )
        )
        return model_class(**common_sequence)
    common = dict(
        context_length=context_length,
        forecast_horizon=forecast_horizon,
        action_count=action_count,
        policy_hidden_width=_config_int(
            config,
            "policyHiddenWidth",
            default=192,
        ),
        policy_logit_rank=_config_optional_int(
            config,
            "policyLogitRank",
        ),
        dropout=_config_float(config, "dropout", default=0.05),
        forecast_coarse_steps=_config_int(
            config,
            "forecastCoarseSteps",
            default=60,
        ),
        forecast_rank=_config_int(config, "forecastRank", default=32),
        maximum_log_movement=_config_float(
            config,
            "maximumLogMovement",
            default=1.0,
        ),
        feature_epsilon=_config_float(
            config,
            "featureEpsilon",
            default=1e-8,
        ),
    )
    if variant == "patch_transformer":
        return MultiResolutionPatchTransformer(
            patch_sizes=_config_int_sequence(
                config,
                "patchSizes",
                default=(15, 60, 300),
            ),
            model_width=_config_int(config, "modelWidth", default=128),
            attention_heads=_config_int(
                config,
                "attentionHeads",
                default=4,
            ),
            layer_count=_config_int(
                config,
                "transformerLayers",
                default=4,
            ),
            feed_forward_width=_config_int(
                config,
                "feedForwardWidth",
                default=384,
            ),
            **common,
        )
    if variant == "multiscale_residual_mixer":
        return MultiScaleResidualMixer(
            aggregate_scales=_config_int_sequence(
                config,
                "aggregateScales",
                default=(60, 300, 900),
            ),
            stream_width=_config_int(config, "streamWidth", default=64),
            stream_mixer_layers=_config_int(
                config,
                "streamMixerLayers",
                default=2,
            ),
            stream_feed_forward_width=_config_int(
                config,
                "streamFeedForwardWidth",
                default=128,
            ),
            fusion_width=_config_int(config, "fusionWidth", default=192),
            **common,
        )
    if variant in {
        "long_context_patch_mixer",
        "learned_aggregate_patch_mixer",
    }:
        learned_aggregate = variant == "learned_aggregate_patch_mixer"
        return LongContextMultiScalePatchMixer(
            patch_sizes=_config_int_sequence(
                config,
                "patchSizes",
                default=(60, 300),
            ),
            encoder_width=_config_int(
                config,
                "encoderWidth",
                default=96,
            ),
            kernel_size=_config_int(config, "kernelSize", default=3),
            dilations=_config_int_sequence(
                config,
                "dilations",
                default=(1, 2, 4, 8, 16, 32, 64, 128),
            ),
            query_count=_config_int(config, "queryCount", default=4),
            fusion_width=_config_int(config, "fusionWidth", default=256),
            dilation_coverage=(
                "exact-stacked-receptive-field"
                if learned_aggregate else "conservative-span"
            ),
            base_architecture_contract=(
                LEARNED_AGGREGATE_PATCH_MIXER_CONTRACT
                if learned_aggregate else LONG_CONTEXT_PATCH_MIXER_CONTRACT
            ),
            **common,
        )
    return MultiScaleDilatedTcn(
        scales=_config_int_sequence(
            config,
            "scales",
            default=(5, 30, 60),
        ),
        tcn_width=_config_int(config, "tcnWidth", default=96),
        kernel_size=_config_int(config, "kernelSize", default=3),
        dilations=_config_int_sequence(
            config,
            "dilations",
            default=(1, 2, 4, 8, 16, 32, 64, 128, 256),
        ),
        fusion_width=_config_int(config, "fusionWidth", default=192),
        **common,
    )


def _covering_dilation_prefix(
    sequence_length: int,
    kernel_size: int,
    dilations: Sequence[int],
) -> tuple[int, ...]:
    """Return the shortest conservative prefix covering one scale sequence.

    ``kernel_size * largest_dilation`` is deliberately conservative compared
    with the exact stacked receptive field.  For the default 3-wide octave
    schedule this retains dilations through 256, 64, and 32 for the 5s, 30s,
    and 60s branches respectively.  We additionally assert the exact stacked
    receptive field so custom schedules cannot pass on span alone.
    """
    selected: tuple[int, ...] | None = None
    for index, dilation in enumerate(dilations):
        if kernel_size * dilation >= sequence_length:
            selected = tuple(dilations[:index + 1])
            break
    if selected is None:
        raise ValueError(
            "TCN dilation schedule does not cover every branch sequence"
        )
    exact_receptive_field = 1 + (kernel_size - 1) * sum(selected)
    if exact_receptive_field < sequence_length:
        raise ValueError(
            "TCN dilation prefix has an insufficient exact receptive field"
        )
    return selected


def _exact_covering_dilation_prefix(
    sequence_length: int,
    kernel_size: int,
    dilations: Sequence[int],
) -> tuple[int, ...]:
    """Return the shortest prefix whose stacked causal field covers tokens."""
    for index in range(len(dilations)):
        selected = tuple(dilations[:index + 1])
        receptive_field = 1 + (kernel_size - 1) * sum(selected)
        if receptive_field >= sequence_length:
            return selected
    raise ValueError(
        "exact TCN dilation schedule does not cover every patch sequence"
    )


def _validate_dropout(value: float) -> None:
    if not 0 <= value < 1 or not math.isfinite(value):
        raise ValueError("dropout must be finite and in [0, 1)")


def _validated_int_sequence(
    value: Sequence[int],
    name: str,
    *,
    minimum: int,
) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be an integer sequence")
    result = tuple(value)
    if not result or any(
        isinstance(item, bool) or not isinstance(item, int) or item < minimum
        for item in result
    ):
        raise ValueError(f"{name} must contain integers >= {minimum}")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} cannot contain duplicates")
    return result


def _config_int(
    config: dict[str, Any],
    key: str,
    *,
    default: int | None = None,
) -> int:
    value = config.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{key} must be a positive integer")
    return value


def _config_float(
    config: dict[str, Any],
    key: str,
    *,
    default: float,
) -> float:
    value = config.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)) \
            or not math.isfinite(float(value)):
        raise ValueError(f"{key} must be finite")
    return float(value)


def _config_optional_int(
    config: dict[str, Any],
    key: str,
) -> int | None:
    value = config.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{key} must be a positive integer when configured")
    return value


def _config_sha256(
    config: dict[str, Any],
    key: str,
) -> str:
    value = config.get(key)
    if not isinstance(value, str) \
            or len(value) != 64 \
            or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{key} must be a lowercase SHA-256 digest")
    return value


def _config_int_sequence(
    config: dict[str, Any],
    key: str,
    *,
    default: Sequence[int],
) -> tuple[int, ...]:
    return _validated_int_sequence(
        config.get(key, default),
        key,
        minimum=1,
    )
