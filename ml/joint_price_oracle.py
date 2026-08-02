from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple

import torch
from torch import Tensor, nn
import torch.nn.functional as functional

from return_oracle_ce import (
    BRANCH_NORMALIZATION_INITIAL_RADIUS,
    LearnableCenteringNorm,
    fused_glu,
    soft_layer_norm_components,
)


OUTPUT_ACTION_COUNT = 101
ARCHITECTURE_CONTRACT = (
    "causal-log-close-hybrid-patch-trend-rlinear-low-rank-dlinear-dual-tide-"
    "forecast-to-verified-1h-1m-1m-oracle-glu-policy-v2"
)
FIXED_DCT_ORDINAL_DECODER_CONTRACT = "fixed-dct-ordinal-logit-decoder-v1"


def architecture_contract_with_policy_decoder(
    base_contract: str,
    policy_logit_rank: int | None,
) -> str:
    """Bind an opt-in policy decoder shape into the checkpoint contract."""
    if policy_logit_rank is None:
        return base_contract
    if isinstance(policy_logit_rank, bool) \
            or not isinstance(policy_logit_rank, int) \
            or policy_logit_rank < 1:
        raise ValueError("policy logit rank must be a positive integer")
    return (
        f"{base_contract}-{FIXED_DCT_ORDINAL_DECODER_CONTRACT}-"
        f"rank{policy_logit_rank}"
    )


def fixed_dct_ordinal_logit_basis(
    action_count: int,
    rank: int,
) -> Tensor:
    """Return the zero-mean orthonormal DCT-II action-logit basis.

    The constant DCT component is deliberately omitted because adding one
    constant to every action logit cannot change a softmax distribution.
    """
    if isinstance(action_count, bool) or not isinstance(action_count, int) \
            or isinstance(rank, bool) or not isinstance(rank, int) \
            or action_count < 2 or not 1 <= rank < action_count:
        raise ValueError(
            "DCT policy rank must be positive and smaller than action count"
        )
    positions = torch.arange(action_count, dtype=torch.float32) + 0.5
    frequencies = torch.arange(1, rank + 1, dtype=torch.float32)
    return (
        torch.cos(
            math.pi
            * frequencies.unsqueeze(1)
            * positions.unsqueeze(0)
            / action_count
        )
        * math.sqrt(2.0 / action_count)
    )


class FixedDctOrdinalLogitProjection(nn.Module):
    """Predict smooth ordered-action logits through a fixed DCT basis."""

    def __init__(self, input_width: int, action_count: int, rank: int) -> None:
        super().__init__()
        if isinstance(input_width, bool) or not isinstance(input_width, int) \
                or input_width < 1:
            raise ValueError("policy projection input width must be positive")
        self.input_width = int(input_width)
        self.action_count = int(action_count)
        self.rank = int(rank)
        self.coefficient_projection = nn.Linear(input_width, rank)
        self.action_bias = nn.Parameter(torch.zeros(action_count))
        self.register_buffer(
            "basis",
            fixed_dct_ordinal_logit_basis(action_count, rank),
            persistent=True,
        )
        nn.init.normal_(
            self.coefficient_projection.weight,
            mean=0.0,
            std=0.01,
        )
        nn.init.zeros_(self.coefficient_projection.bias)

    def forward(self, hidden: Tensor) -> Tensor:
        if hidden.ndim != 2 or hidden.shape[1] != self.input_width:
            raise ValueError("policy projection input width is incompatible")
        coefficients = self.coefficient_projection(hidden)
        return coefficients @ self.basis + self.action_bias


@dataclass(frozen=True)
class JointLossWeights:
    policy_cross_entropy: float = 1.0
    conditioned_policy_cross_entropy: float = 0.0
    forecast: float = 1.0
    soft_layer_norm: float = 0.01


class JointPriceOracleOutput(NamedTuple):
    policy_logits: Tensor
    predicted_closes: Tensor
    predicted_log_movements: Tensor
    predicted_movements: Tensor
    soft_layer_norm_mean: Tensor
    soft_layer_norm_variance: Tensor


def causal_moving_average(values: Tensor, window: int) -> Tensor:
    """Return a left-padded trailing average without reading future values."""
    if values.ndim != 3:
        raise ValueError(
            "causal moving average expects [example, time, variable] values"
        )
    if window < 1:
        raise ValueError("moving-average window must be positive")
    if values.shape[1] < 1 or values.shape[2] < 1:
        raise ValueError("moving-average input cannot have an empty axis")
    if window == 1:
        return values
    channel_first = values.transpose(1, 2)
    padded = functional.pad(
        channel_first,
        (window - 1, 0),
        mode="replicate",
    )
    return functional.avg_pool1d(
        padded,
        kernel_size=window,
        stride=1,
    ).transpose(1, 2)


class CausalPatchAggregate(nn.Module):
    """Learn one trailing-patch aggregate per market variable.

    The overlapping, channel-independent patches follow PatchTST's useful
    inductive bias while retaining an explicitly causal aggregation.  Uniform
    initialization is exactly the matching trailing moving average.
    """

    def __init__(self, variable_count: int, patch_length: int) -> None:
        super().__init__()
        if variable_count < 1 or patch_length < 1:
            raise ValueError("patch aggregation dimensions must be positive")
        self.variable_count = int(variable_count)
        self.patch_length = int(patch_length)
        self.weight_logits = nn.Parameter(
            torch.zeros(variable_count, patch_length)
        )

    def weights(self) -> Tensor:
        return torch.softmax(self.weight_logits.float(), dim=-1)

    def forward(self, values: Tensor) -> Tensor:
        if values.ndim != 3 \
                or values.shape[2] != self.variable_count:
            raise ValueError(
                "patch aggregation expects matching "
                "[example, time, variable] values"
            )
        channel_first = values.transpose(1, 2)
        padded = functional.pad(
            channel_first,
            (self.patch_length - 1, 0),
            mode="replicate",
        )
        aggregate = functional.conv1d(
            padded,
            self.weights().to(dtype=padded.dtype).unsqueeze(1),
            groups=self.variable_count,
        )
        return aggregate.transpose(1, 2)


class HybridTrendDecomposition(nn.Module):
    """Split a series into learned/fixed trend and exact residual streams."""

    def __init__(
        self,
        variable_count: int,
        moving_average_window: int,
        patch_length: int,
        aggregate_mode: str = "hybrid",
    ) -> None:
        super().__init__()
        if aggregate_mode not in {
            "moving_average",
            "learned_patch",
            "hybrid",
        }:
            raise ValueError(f"unsupported aggregate mode: {aggregate_mode}")
        self.variable_count = int(variable_count)
        self.moving_average_window = int(moving_average_window)
        self.aggregate_mode = aggregate_mode
        self.patch_aggregate = CausalPatchAggregate(
            variable_count,
            patch_length,
        )
        self.patch_mix_logits = nn.Parameter(torch.zeros(variable_count))

    def forward(self, values: Tensor) -> tuple[Tensor, Tensor]:
        moving_average = causal_moving_average(
            values,
            self.moving_average_window,
        )
        if self.aggregate_mode == "moving_average":
            trend = moving_average
        else:
            learned = self.patch_aggregate(values)
            if self.aggregate_mode == "learned_patch":
                trend = learned
            else:
                patch_mix = torch.sigmoid(
                    self.patch_mix_logits
                ).to(dtype=values.dtype).view(1, 1, -1)
                trend = (
                    moving_average * (1.0 - patch_mix)
                    + learned * patch_mix
                )
        return trend, values - trend


class ReversibleInstanceStandardizer(nn.Module):
    """RLinear-style per-sample, per-variable reversible normalization."""

    def __init__(self, epsilon: float = 1e-12) -> None:
        super().__init__()
        if epsilon <= 0:
            raise ValueError("normalization epsilon must be positive")
        self.epsilon = float(epsilon)

    def normalize(self, values: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if values.ndim != 3 or values.shape[1] < 1:
            raise ValueError(
                "reversible normalization expects "
                "[example, time, variable] values"
            )
        mean = values.mean(dim=1, keepdim=True).detach()
        variance = (
            values.float() - mean.float()
        ).square().mean(dim=1, keepdim=True)
        standard_deviation = (
            variance + self.epsilon
        ).sqrt().to(dtype=values.dtype).detach()
        return (
            (values - mean) / standard_deviation,
            mean,
            standard_deviation,
        )

    @staticmethod
    def denormalize(
        normalized: Tensor,
        mean: Tensor,
        standard_deviation: Tensor,
    ) -> Tensor:
        if normalized.ndim != 3 \
                or mean.ndim != 3 \
                or standard_deviation.ndim != 3 \
                or mean.shape[0] != normalized.shape[0] \
                or mean.shape[1] != 1 \
                or mean.shape[2] != normalized.shape[2] \
                or standard_deviation.shape != mean.shape:
            raise ValueError("reversible normalization state is incompatible")
        return normalized * standard_deviation + mean


class LowRankTemporalLinear(nn.Module):
    """Factorized DLinear/RLinear projection with independent variables."""

    def __init__(
        self,
        context_length: int,
        forecast_horizon: int,
        variable_count: int,
        rank: int,
    ) -> None:
        super().__init__()
        if min(
            context_length,
            forecast_horizon,
            variable_count,
            rank,
        ) < 1:
            raise ValueError("temporal linear dimensions must be positive")
        if rank > min(context_length, forecast_horizon):
            raise ValueError(
                "temporal linear rank cannot exceed its time dimensions"
            )
        self.context_length = int(context_length)
        self.forecast_horizon = int(forecast_horizon)
        self.variable_count = int(variable_count)
        self.rank = int(rank)
        self.input_weight = nn.Parameter(torch.empty(
            variable_count,
            rank,
            context_length,
        ))
        self.output_weight = nn.Parameter(torch.empty(
            variable_count,
            forecast_horizon,
            rank,
        ))
        self.bias = nn.Parameter(torch.zeros(
            variable_count,
            forecast_horizon,
        ))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            # Rank zero is an exact no-change forecast in normalized space.
            # Remaining input factors start diverse while zero output factors
            # make them non-disruptive and learnable from the first update.
            nn.init.kaiming_uniform_(
                self.input_weight,
                a=math.sqrt(5),
            )
            self.input_weight[:, 0, :].zero_()
            self.input_weight[:, 0, -1] = 1.0
            self.output_weight.zero_()
            self.output_weight[:, :, 0] = 1.0
            self.bias.zero_()

    def forward(self, values: Tensor) -> Tensor:
        if values.ndim != 3 \
                or values.shape[1] != self.context_length \
                or values.shape[2] != self.variable_count:
            raise ValueError(
                "temporal linear input does not match its context contract"
            )
        latent = torch.einsum(
            "blv,vrl->brv",
            values,
            self.input_weight,
        )
        return torch.einsum(
            "brv,vhr->bhv",
            latent,
            self.output_weight,
        ) + self.bias.transpose(0, 1).unsqueeze(0)


class OracleStyleGluLayer(nn.Module):
    """Current return-oracle GLU normalization with a global residual path."""

    def __init__(
        self,
        input_width: int,
        output_width: int,
        global_input_width: int,
        *,
        normalization_family: str,
        normalization_initial_scale: float,
        normalization_minimum_scale: float,
    ) -> None:
        super().__init__()
        if min(input_width, output_width, global_input_width) < 1:
            raise ValueError("GLU layer widths must be positive")
        self.input_width = int(input_width)
        self.output_width = int(output_width)
        self.global_input_width = int(global_input_width)
        self.projection = nn.Linear(input_width, output_width * 2)
        self.global_residual_projection = nn.Linear(
            global_input_width,
            output_width * 2,
        )

        def normalizer() -> LearnableCenteringNorm:
            return LearnableCenteringNorm(
                output_width,
                denominator_family=normalization_family,
                initial_scale=normalization_initial_scale,
                minimum_scale=normalization_minimum_scale,
            )

        self.value_normalizer = normalizer()
        self.gate_normalizer = normalizer()
        self.residual_value_normalizer = normalizer()
        self.residual_gate_normalizer = normalizer()
        # Match the active return-oracle design: all paths into a target layer
        # share C, while their radii and post-transform biases remain separate.
        self.gate_normalizer.weight = self.value_normalizer.weight
        self.residual_value_normalizer.weight = self.value_normalizer.weight
        self.residual_gate_normalizer.weight = self.value_normalizer.weight
        self.value_normalizer.weight.requires_grad_(False)
        self.value_transform = nn.Linear(
            output_width,
            output_width,
            bias=False,
        )
        self.gate_transform = nn.Linear(
            output_width,
            output_width,
            bias=False,
        )
        self.value_bias = nn.Parameter(torch.zeros(output_width))
        self.gate_bias = nn.Parameter(torch.zeros(output_width))
        self.residual_value_bias = nn.Parameter(torch.zeros(output_width))
        self.residual_gate_bias = nn.Parameter(torch.zeros(output_width))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_normal_(
            self.projection.weight,
            nonlinearity="linear",
        )
        nn.init.zeros_(self.projection.bias)
        residual_value_weight, residual_gate_weight = (
            self.global_residual_projection.weight.chunk(2, dim=0)
        )
        nn.init.zeros_(residual_value_weight)
        nn.init.kaiming_normal_(
            residual_gate_weight,
            nonlinearity="linear",
        )
        nn.init.zeros_(self.global_residual_projection.bias)
        self.value_normalizer.reset_parameters()
        self.gate_normalizer.reset_scale()
        self.residual_value_normalizer.reset_scale()
        self.residual_gate_normalizer.reset_scale()
        nn.init.eye_(self.value_transform.weight)
        nn.init.eye_(self.gate_transform.weight)
        nn.init.zeros_(self.value_bias)
        nn.init.zeros_(self.gate_bias)
        nn.init.zeros_(self.residual_value_bias)
        nn.init.zeros_(self.residual_gate_bias)

    def forward_with_regularizers(
        self,
        values: Tensor,
        global_input: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if values.ndim != 2 or values.shape[1] != self.input_width \
                or global_input.ndim != 2 \
                or global_input.shape[0] != values.shape[0] \
                or global_input.shape[1] != self.global_input_width:
            raise ValueError("GLU layer inputs do not match its width contract")
        hidden, raw_value, raw_gate = fused_glu(
            self.projection(values),
            self.value_normalizer,
            self.gate_normalizer,
            self.value_bias,
            self.gate_bias,
            self.value_transform,
            self.gate_transform,
        )
        residual, raw_residual_value, raw_residual_gate = fused_glu(
            self.global_residual_projection(global_input),
            self.residual_value_normalizer,
            self.residual_gate_normalizer,
            self.residual_value_bias,
            self.residual_gate_bias,
            self.value_transform,
            self.gate_transform,
        )
        components = [
            soft_layer_norm_components(branch)
            for branch in (
                raw_value,
                raw_gate,
                raw_residual_value,
                raw_residual_gate,
            )
        ]
        mean_penalty = torch.stack([
            mean for mean, _variance in components
        ]).mean()
        variance_penalty = torch.stack([
            variance for _mean, variance in components
        ]).mean()
        return hidden + residual, mean_penalty, variance_penalty


class TiDEStream(nn.Module):
    """Dense TiDE encoder/decoder for one normalized decomposition stream."""

    def __init__(
        self,
        context_length: int,
        forecast_horizon: int,
        variable_count: int,
        hidden_width: int,
        layer_count: int,
        dropout: float,
        *,
        normalization_family: str,
        normalization_initial_scale: float,
        normalization_minimum_scale: float,
    ) -> None:
        super().__init__()
        if min(
            context_length,
            forecast_horizon,
            variable_count,
            hidden_width,
            layer_count,
        ) < 1:
            raise ValueError("TiDE dimensions must be positive")
        if not 0 <= dropout < 1:
            raise ValueError("TiDE dropout must be in [0, 1)")
        self.context_length = int(context_length)
        self.forecast_horizon = int(forecast_horizon)
        self.variable_count = int(variable_count)
        input_width = context_length * variable_count
        self.layers = nn.ModuleList([
            OracleStyleGluLayer(
                input_width if index == 0 else hidden_width,
                hidden_width,
                input_width,
                normalization_family=normalization_family,
                normalization_initial_scale=normalization_initial_scale,
                normalization_minimum_scale=normalization_minimum_scale,
            )
            for index in range(layer_count)
        ])
        self.dropout = nn.Dropout(dropout)
        self.temporal_decoder = nn.Linear(
            hidden_width,
            forecast_horizon * variable_count,
        )
        # The DLinear path owns the initial persistence forecast.  A zero TiDE
        # decoder starts as an exact non-disruptive residual and still receives
        # a gradient on the first optimizer update.
        nn.init.zeros_(self.temporal_decoder.weight)
        nn.init.zeros_(self.temporal_decoder.bias)

    def forward_with_regularizers(
        self,
        normalized: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if normalized.ndim != 3 \
                or normalized.shape[1] != self.context_length \
                or normalized.shape[2] != self.variable_count:
            raise ValueError("TiDE stream input does not match its contract")
        global_input = normalized.flatten(start_dim=1)
        hidden = global_input
        mean_penalties: list[Tensor] = []
        variance_penalties: list[Tensor] = []
        for layer in self.layers:
            hidden, mean_penalty, variance_penalty = (
                layer.forward_with_regularizers(hidden, global_input)
            )
            hidden = self.dropout(hidden)
            mean_penalties.append(mean_penalty)
            variance_penalties.append(variance_penalty)
        forecast = self.temporal_decoder(hidden).reshape(
            normalized.shape[0],
            self.forecast_horizon,
            self.variable_count,
        )
        return (
            forecast,
            torch.stack(mean_penalties).mean(),
            torch.stack(variance_penalties).mean(),
        )


class ForecastBackbone(nn.Module):
    """DLinear decomposition plus dual TiDE residual forecasts under RevIN."""

    def __init__(
        self,
        context_length: int,
        forecast_horizon: int,
        variable_count: int = 1,
        moving_average_window: int = 60,
        patch_length: int = 60,
        linear_rank: int = 64,
        aggregate_mode: str = "hybrid",
        tide_hidden_width: int = 256,
        tide_layer_count: int = 4,
        dropout: float = 0.05,
        *,
        normalization_epsilon: float = 1e-12,
        normalization_family: str = "tanh",
        normalization_initial_scale: float = (
            BRANCH_NORMALIZATION_INITIAL_RADIUS
        ),
        normalization_minimum_scale: float = 1e-4,
    ) -> None:
        super().__init__()
        if context_length < max(moving_average_window, patch_length):
            raise ValueError(
                "forecast context must contain the configured aggregates"
            )
        self.context_length = int(context_length)
        self.forecast_horizon = int(forecast_horizon)
        self.variable_count = int(variable_count)
        self.decomposition = HybridTrendDecomposition(
            variable_count,
            moving_average_window,
            patch_length,
            aggregate_mode,
        )
        self.standardizer = ReversibleInstanceStandardizer(
            normalization_epsilon
        )
        stream_arguments = dict(
            context_length=context_length,
            forecast_horizon=forecast_horizon,
            variable_count=variable_count,
            hidden_width=tide_hidden_width,
            layer_count=tide_layer_count,
            dropout=dropout,
            normalization_family=normalization_family,
            normalization_initial_scale=normalization_initial_scale,
            normalization_minimum_scale=normalization_minimum_scale,
        )
        self.trend_tide = TiDEStream(**stream_arguments)
        self.residual_tide = TiDEStream(**stream_arguments)
        self.trend_linear = LowRankTemporalLinear(
            context_length,
            forecast_horizon,
            variable_count,
            linear_rank,
        )
        self.residual_linear = LowRankTemporalLinear(
            context_length,
            forecast_horizon,
            variable_count,
            linear_rank,
        )
        self.trend_tide_gain_logits = nn.Parameter(
            torch.full((variable_count,), -2.0)
        )
        self.residual_tide_gain_logits = nn.Parameter(
            torch.full((variable_count,), -2.0)
        )

    def forward_with_regularizers(
        self,
        closes: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if closes.ndim != 3 \
                or closes.shape[1] != self.context_length \
                or closes.shape[2] != self.variable_count:
            raise ValueError(
                "close input must match [example, context, variable]"
            )
        # Decomposing log closes makes the reconstruction additive and ensures
        # that every predicted close remains positive after denormalization.
        log_closes = closes.float().log()
        trend, residual = self.decomposition(log_closes)
        normalized_trend, trend_mean, trend_std = (
            self.standardizer.normalize(trend)
        )
        normalized_residual, residual_mean, residual_std = (
            self.standardizer.normalize(residual)
        )
        trend_tide, trend_mean_penalty, trend_variance_penalty = (
            self.trend_tide.forward_with_regularizers(normalized_trend)
        )
        residual_tide, residual_mean_penalty, residual_variance_penalty = (
            self.residual_tide.forward_with_regularizers(normalized_residual)
        )
        trend_gain = torch.sigmoid(
            self.trend_tide_gain_logits
        ).view(1, 1, -1)
        residual_gain = torch.sigmoid(
            self.residual_tide_gain_logits
        ).view(1, 1, -1)
        normalized_trend_forecast = (
            self.trend_linear(normalized_trend)
            + trend_tide * trend_gain
        )
        normalized_residual_forecast = (
            self.residual_linear(normalized_residual)
            + residual_tide * residual_gain
        )
        forecast_log_closes = (
            self.standardizer.denormalize(
                normalized_trend_forecast,
                trend_mean,
                trend_std,
            )
            + self.standardizer.denormalize(
                normalized_residual_forecast,
                residual_mean,
                residual_std,
            )
        )
        last_log_close = log_closes[:, -1:, :]
        log_movements = forecast_log_closes - last_log_close
        predicted_closes = torch.exp(forecast_log_closes)
        mean_penalty = torch.stack((
            trend_mean_penalty,
            residual_mean_penalty,
        )).mean()
        variance_penalty = torch.stack((
            trend_variance_penalty,
            residual_variance_penalty,
        )).mean()
        return (
            predicted_closes,
            log_movements,
            mean_penalty,
            variance_penalty,
        )


class MovementPolicyHead(nn.Module):
    """Learn the oracle policy only from the model's forecast movement path."""

    def __init__(
        self,
        forecast_horizon: int,
        variable_count: int,
        hidden_width: int,
        layer_count: int,
        action_count: int,
        dropout: float,
        *,
        policy_logit_rank: int | None = None,
        normalization_family: str,
        normalization_initial_scale: float,
        normalization_minimum_scale: float,
    ) -> None:
        super().__init__()
        if min(
            forecast_horizon,
            variable_count,
            hidden_width,
            layer_count,
            action_count,
        ) < 1:
            raise ValueError("policy head dimensions must be positive")
        if not 0 <= dropout < 1:
            raise ValueError("policy dropout must be in [0, 1)")
        self.forecast_horizon = int(forecast_horizon)
        self.variable_count = int(variable_count)
        self.action_count = int(action_count)
        input_width = forecast_horizon * variable_count
        self.layers = nn.ModuleList([
            OracleStyleGluLayer(
                input_width if index == 0 else hidden_width,
                hidden_width,
                input_width,
                normalization_family=normalization_family,
                normalization_initial_scale=normalization_initial_scale,
                normalization_minimum_scale=normalization_minimum_scale,
            )
            for index in range(layer_count)
        ])
        self.dropout = nn.Dropout(dropout)
        if policy_logit_rank is None:
            # Keep the legacy module and state-dict names byte-for-byte when
            # the structured decoder is not requested.
            self.output = nn.Linear(hidden_width, self.action_count)
            nn.init.normal_(self.output.weight, std=0.01)
            nn.init.zeros_(self.output.bias)
        else:
            self.output = FixedDctOrdinalLogitProjection(
                hidden_width,
                self.action_count,
                policy_logit_rank,
            )

    def forward_with_regularizers(
        self,
        predicted_log_movements: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if predicted_log_movements.ndim != 3 \
                or predicted_log_movements.shape[1] \
                != self.forecast_horizon \
                or predicted_log_movements.shape[2] != self.variable_count:
            raise ValueError("policy input does not match the forecast contract")
        global_input = predicted_log_movements.flatten(start_dim=1)
        hidden = global_input
        mean_penalties: list[Tensor] = []
        variance_penalties: list[Tensor] = []
        for layer in self.layers:
            hidden, mean_penalty, variance_penalty = (
                layer.forward_with_regularizers(hidden, global_input)
            )
            hidden = self.dropout(hidden)
            mean_penalties.append(mean_penalty)
            variance_penalties.append(variance_penalty)
        return (
            self.output(hidden),
            torch.stack(mean_penalties).mean(),
            torch.stack(variance_penalties).mean(),
        )


class JointPriceOracleModel(nn.Module):
    """Causal close forecast followed by a learned raw-oracle policy head."""

    architecture_contract = ARCHITECTURE_CONTRACT

    def __init__(
        self,
        context_length: int,
        forecast_horizon: int,
        variable_count: int = 1,
        moving_average_window: int = 60,
        patch_length: int = 60,
        linear_rank: int = 64,
        aggregate_mode: str = "hybrid",
        tide_hidden_width: int = 256,
        tide_layer_count: int = 4,
        policy_hidden_width: int = 256,
        policy_layer_count: int = 4,
        action_count: int = OUTPUT_ACTION_COUNT,
        dropout: float = 0.05,
        *,
        policy_logit_rank: int | None = None,
        normalization_epsilon: float = 1e-12,
        normalization_family: str = "tanh",
        normalization_initial_scale: float = (
            BRANCH_NORMALIZATION_INITIAL_RADIUS
        ),
        normalization_minimum_scale: float = 1e-4,
    ) -> None:
        super().__init__()
        self.context_length = int(context_length)
        self.forecast_horizon = int(forecast_horizon)
        self.variable_count = int(variable_count)
        self.action_count = int(action_count)
        self.architecture_contract = architecture_contract_with_policy_decoder(
            ARCHITECTURE_CONTRACT,
            policy_logit_rank,
        )
        common_normalization = dict(
            normalization_family=normalization_family,
            normalization_initial_scale=normalization_initial_scale,
            normalization_minimum_scale=normalization_minimum_scale,
        )
        self.forecast_backbone = ForecastBackbone(
            context_length=context_length,
            forecast_horizon=forecast_horizon,
            variable_count=variable_count,
            moving_average_window=moving_average_window,
            patch_length=patch_length,
            linear_rank=linear_rank,
            aggregate_mode=aggregate_mode,
            tide_hidden_width=tide_hidden_width,
            tide_layer_count=tide_layer_count,
            dropout=dropout,
            normalization_epsilon=normalization_epsilon,
            **common_normalization,
        )
        self.policy_head = MovementPolicyHead(
            forecast_horizon,
            variable_count,
            policy_hidden_width,
            policy_layer_count,
            action_count,
            dropout,
            policy_logit_rank=policy_logit_rank,
            **common_normalization,
        )

    def forward_with_forecast(self, closes: Tensor) -> JointPriceOracleOutput:
        (
            predicted_closes,
            predicted_log_movements,
            forecast_mean_penalty,
            forecast_variance_penalty,
        ) = self.forecast_backbone.forward_with_regularizers(closes)
        (
            policy_logits,
            policy_mean_penalty,
            policy_variance_penalty,
        ) = self.policy_head.forward_with_regularizers(
            predicted_log_movements
        )
        return JointPriceOracleOutput(
            policy_logits=policy_logits,
            predicted_closes=predicted_closes,
            predicted_log_movements=predicted_log_movements,
            predicted_movements=torch.expm1(predicted_log_movements),
            soft_layer_norm_mean=torch.stack((
                forecast_mean_penalty,
                policy_mean_penalty,
            )).mean(),
            soft_layer_norm_variance=torch.stack((
                forecast_variance_penalty,
                policy_variance_penalty,
            )).mean(),
        )

    def forward(self, closes: Tensor) -> Tensor:
        return self.forward_with_forecast(closes).policy_logits


def joint_price_oracle_objective(
    output: JointPriceOracleOutput,
    input_closes: Tensor,
    future_closes: Tensor,
    target_policy: Tensor,
    weights: JointLossWeights = JointLossWeights(),
    *,
    example_weights: Tensor | None = None,
    action_grid: Tensor | None = None,
    policy_friction: float = 0.0,
    policy_temperature: float = 1.0,
    forecast_huber_delta: float = 1.0,
    volatility_floor: float = 1e-5,
) -> dict[str, Tensor]:
    """Jointly score causal price forecasting and soft oracle decisions."""
    if input_closes.ndim != 3 or future_closes.ndim != 3 \
            or input_closes.shape[0] != future_closes.shape[0] \
            or input_closes.shape[2] != future_closes.shape[2] \
            or output.predicted_closes.shape != future_closes.shape:
        raise ValueError("forecast objective close tensors are incompatible")
    if target_policy.ndim != 2 \
            or target_policy.shape != output.policy_logits.shape:
        raise ValueError("target policy must match the model action logits")
    if forecast_huber_delta <= 0 or volatility_floor <= 0:
        raise ValueError("forecast loss scales must be positive")
    batch_size = input_closes.shape[0]
    (
        normalized_weights,
        target_policy_float,
        predicted_log_probabilities,
        cross_entropy_per_example,
        kl_per_example,
    ) = _raw_policy_distribution_terms(
        output.policy_logits,
        target_policy,
        example_weights,
    )

    conditioned_cross_entropy_per_example = torch.zeros_like(
        cross_entropy_per_example
    )
    conditioned_kl_per_example = torch.zeros_like(kl_per_example)
    if weights.conditioned_policy_cross_entropy > 0:
        if action_grid is None \
                or action_grid.shape != (target_policy.shape[1],) \
                or not bool(torch.isfinite(action_grid).all()):
            raise ValueError(
                "conditioned policy loss requires a finite action grid"
            )
        if policy_friction < 0 or policy_temperature <= 0 \
                or not math.isfinite(policy_friction) \
                or not math.isfinite(policy_temperature):
            raise ValueError(
                "conditioned policy friction/temperature are invalid"
            )
        grid = action_grid.float()
        anchors = torch.stack((-grid[-1], grid.new_zeros(()), grid[-1]))
        difference = grid.unsqueeze(0) - anchors.unsqueeze(1)
        buy_denominator = 1 - policy_friction + policy_friction * grid
        sell_denominator = 1 - policy_friction * grid
        buy_factor = 1 - policy_friction * difference / buy_denominator
        sell_factor = 1 - policy_friction * (-difference) / sell_denominator
        rebalance_factor = torch.where(
            difference > 0,
            buy_factor,
            torch.where(difference < 0, sell_factor, torch.ones_like(difference)),
        )
        if bool((rebalance_factor <= 0).any()):
            raise ValueError("conditioned policy grid has infeasible transitions")
        transition_logits = rebalance_factor.log() / policy_temperature
        base_target_logits = torch.where(
            target_policy_float > 0,
            target_policy_float.clamp_min(1e-12).log(),
            torch.full_like(target_policy_float, -torch.inf),
        )
        conditioned_target_logits = (
            base_target_logits.unsqueeze(1)
            + transition_logits.unsqueeze(0)
        )
        conditioned_target = torch.softmax(
            conditioned_target_logits,
            dim=-1,
        )
        conditioned_predicted_log = torch.log_softmax(
            output.policy_logits.float().unsqueeze(1)
            + transition_logits.unsqueeze(0),
            dim=-1,
        )
        conditioned_cross_entropy_per_example = -(
            conditioned_target * conditioned_predicted_log
        ).sum(dim=-1).mean(dim=-1)
        conditioned_target_log = torch.where(
            conditioned_target > 0,
            conditioned_target.clamp_min(1e-12).log(),
            torch.zeros_like(conditioned_target),
        )
        conditioned_kl_per_example = (
            conditioned_target
            * (conditioned_target_log - conditioned_predicted_log)
        ).sum(dim=-1).mean(dim=-1)

    input_log_closes = input_closes.float().log()
    future_log_closes = future_closes.float().log()
    last_log_close = input_log_closes[:, -1:, :]
    target_log_movements = future_log_closes - last_log_close
    past_log_returns = (
        input_log_closes[:, 1:, :] - input_log_closes[:, :-1, :]
    )
    if past_log_returns.shape[1] < 1:
        volatility = torch.full(
            (batch_size, 1, input_closes.shape[2]),
            volatility_floor,
            device=input_closes.device,
            dtype=torch.float32,
        )
    else:
        volatility = past_log_returns.std(
            dim=1,
            correction=0,
            keepdim=True,
        ).clamp_min(volatility_floor)
    horizon_scale = torch.arange(
        1,
        future_closes.shape[1] + 1,
        device=future_closes.device,
        dtype=torch.float32,
    ).sqrt().view(1, -1, 1)
    normalized_error = (
        output.predicted_log_movements.float() - target_log_movements
    ) / (volatility * horizon_scale)
    absolute_error = normalized_error.abs()
    forecast_per_cell = torch.where(
        absolute_error <= forecast_huber_delta,
        0.5 * normalized_error.square(),
        forecast_huber_delta * (
            absolute_error - 0.5 * forecast_huber_delta
        ),
    )
    forecast_per_example = forecast_per_cell.mean(dim=(1, 2))

    def weighted_mean(values: Tensor) -> Tensor:
        return (values.float() * normalized_weights).mean()

    cross_entropy = weighted_mean(cross_entropy_per_example)
    kl_divergence = weighted_mean(kl_per_example)
    conditioned_cross_entropy = weighted_mean(
        conditioned_cross_entropy_per_example
    )
    conditioned_kl_divergence = weighted_mean(
        conditioned_kl_per_example
    )
    forecast_loss = weighted_mean(forecast_per_example)
    soft_layer_norm = (
        output.soft_layer_norm_mean
        + output.soft_layer_norm_variance
    )
    loss = (
        cross_entropy * weights.policy_cross_entropy
        + conditioned_cross_entropy
        * weights.conditioned_policy_cross_entropy
        + forecast_loss * weights.forecast
        + soft_layer_norm * weights.soft_layer_norm
    )
    predicted_probabilities = predicted_log_probabilities.exp()
    probability_mse = weighted_mean(
        (
            predicted_probabilities - target_policy_float
        ).square().mean(dim=-1)
    )
    next_direction_correct = (
        output.predicted_log_movements[:, 0, :].sign()
        == target_log_movements[:, 0, :].sign()
    ).float().mean(dim=-1)
    direction_accuracy = weighted_mean(next_direction_correct)
    next_movement_rmse = (
        weighted_mean(
            (
                output.predicted_log_movements[:, 0, :]
                - target_log_movements[:, 0, :]
            ).square().mean(dim=-1)
        ).sqrt()
    )
    return {
        "loss": loss,
        "crossEntropy": cross_entropy,
        "klDivergence": kl_divergence,
        "conditionedCrossEntropy": conditioned_cross_entropy,
        "conditionedKlDivergence": conditioned_kl_divergence,
        "probabilityMse": probability_mse,
        "forecastLoss": forecast_loss,
        "nextMovementRmse": next_movement_rmse,
        "directionAccuracy": direction_accuracy,
        "softLayerNorm": soft_layer_norm,
        "softLayerNormMeanPenalty": output.soft_layer_norm_mean,
        "softLayerNormVariancePenalty": (
            output.soft_layer_norm_variance
        ),
    }


def joint_price_oracle_policy_only_objective(
    policy_logits: Tensor,
    target_policy: Tensor,
    weights: JointLossWeights = JointLossWeights(
        forecast=0.0,
        soft_layer_norm=0.0,
    ),
    *,
    example_weights: Tensor | None = None,
) -> dict[str, Tensor]:
    """Score raw oracle probabilities without constructing forecast tensors.

    The raw cross-entropy, forward KL, and probability MSE use the exact same
    implementation as :func:`joint_price_oracle_objective`.  Metrics that do
    not exist in an explicitly policy-only run remain zero-valued so training
    logs and checkpoint validation metadata retain one stable schema.
    """
    incompatible_weights = (
        weights.conditioned_policy_cross_entropy,
        weights.forecast,
        weights.soft_layer_norm,
    )
    if any(value != 0 for value in incompatible_weights):
        raise ValueError(
            "policy-only objective requires conditioned, forecast, and "
            "soft-layernorm weights to be zero"
        )
    if not math.isfinite(weights.policy_cross_entropy) \
            or weights.policy_cross_entropy <= 0:
        raise ValueError(
            "policy-only cross-entropy weight must be finite and positive"
        )
    (
        normalized_weights,
        target_policy_float,
        predicted_log_probabilities,
        cross_entropy_per_example,
        kl_per_example,
    ) = _raw_policy_distribution_terms(
        policy_logits,
        target_policy,
        example_weights,
    )

    def weighted_mean(values: Tensor) -> Tensor:
        return (values.float() * normalized_weights).mean()

    cross_entropy = weighted_mean(cross_entropy_per_example)
    kl_divergence = weighted_mean(kl_per_example)
    probability_mse = weighted_mean(
        (
            predicted_log_probabilities.exp() - target_policy_float
        ).square().mean(dim=-1)
    )
    zero = cross_entropy.new_zeros(())
    return {
        "loss": cross_entropy * weights.policy_cross_entropy,
        "crossEntropy": cross_entropy,
        "klDivergence": kl_divergence,
        "conditionedCrossEntropy": zero,
        "conditionedKlDivergence": zero,
        "probabilityMse": probability_mse,
        "forecastLoss": zero,
        "nextMovementRmse": zero,
        "directionAccuracy": zero,
        "softLayerNorm": zero,
        "softLayerNormMeanPenalty": zero,
        "softLayerNormVariancePenalty": zero,
    }


def _raw_policy_distribution_terms(
    policy_logits: Tensor,
    target_policy: Tensor,
    example_weights: Tensor | None,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Return shared raw-policy terms for joint and policy-only objectives."""
    if policy_logits.ndim != 2 \
            or target_policy.ndim != 2 \
            or target_policy.shape != policy_logits.shape:
        raise ValueError("target policy must match the model action logits")
    batch_size = policy_logits.shape[0]
    if batch_size < 1:
        raise ValueError("policy objective requires a non-empty batch")
    if example_weights is None:
        normalized_weights = torch.ones(
            batch_size,
            device=policy_logits.device,
            dtype=torch.float32,
        )
    else:
        if example_weights.shape != (batch_size,) \
                or not bool(torch.isfinite(example_weights).all()) \
                or bool((example_weights <= 0).any()):
            raise ValueError(
                "example weights must be finite positive batch weights"
            )
        normalized_weights = example_weights.float()
    normalized_weights = (
        normalized_weights / normalized_weights.mean().clamp_min(1e-12)
    )

    target_policy_float = target_policy.float()
    target_policy_float = target_policy_float / (
        target_policy_float.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    )
    predicted_log_probabilities = functional.log_softmax(
        policy_logits.float(),
        dim=-1,
    )
    cross_entropy_per_example = -(
        target_policy_float * predicted_log_probabilities
    ).sum(dim=-1)
    target_log_probabilities = torch.where(
        target_policy_float > 0,
        target_policy_float.clamp_min(1e-12).log(),
        torch.zeros_like(target_policy_float),
    )
    kl_per_example = (
        target_policy_float
        * (target_log_probabilities - predicted_log_probabilities)
    ).sum(dim=-1)
    return (
        normalized_weights,
        target_policy_float,
        predicted_log_probabilities,
        cross_entropy_per_example,
        kl_per_example,
    )


def parameter_count(model: nn.Module) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
