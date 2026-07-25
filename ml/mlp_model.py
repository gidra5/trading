from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as functional


FEATURE_SCHEMA_VERSION = 5
INPUT_FEATURE_COUNT = 901
OUTPUT_ACTION_COUNT = 255
TEACHER_PARAMETER_COUNT = 8
DEPLOYMENT_PROBABILITY_FLOOR = 1e-8
HIDDEN_LAYER_COUNT = 16
HIDDEN_WIDTH = 1024


@dataclass(frozen=True)
class PolicySupport:
    latent_lower: float | Tensor
    latent_upper: float | Tensor
    visible_lower: float | Tensor
    visible_upper: float | Tensor
    friction: float | Tensor
    temperature: float | Tensor
    hinge_span: float | Tensor | None = None


@dataclass(frozen=True)
class LossWeights:
    cross_entropy: float | Tensor = 1.0
    probability_mse: float | Tensor = 1.0
    parameter_mse: float | Tensor = 1.0
    excess_entropy: float | Tensor = 1.0
    temporal_mutual_information: float | Tensor = 1.0
    oracle_mutual_information: float | Tensor = 1.0


@dataclass(frozen=True)
class DirectLossWeights:
    """Loss weights for the deployable direct-distribution model."""

    cross_entropy: float | Tensor = 1.0
    probability_mse: float | Tensor = 0.1
    excess_entropy: float | Tensor = 0.0
    temporal_mutual_information: float | Tensor = 1.0
    oracle_mutual_information: float | Tensor = 1.0


@dataclass(frozen=True)
class TimeWeighting:
    """Distance-imbalance weighting applied across training timestamps."""

    distance_epsilon: float = 1e-6
    minimum_weight: float = 1e-6
    minimum_advice_magnitude: float = 0.25
    memory_half_life_steps: float = 15.0
    growth_per_prior_advice: float = 0.25
    maximum_multiplier: float = 4.0
    reset_after_gap_steps: float = 60.0
    resolution_divergence_multiplier: float = 0.0


def policy_support_value(raw: Tensor, value: float | Tensor) -> Tensor:
    """Materialize support constants without promoting compiled CUDA math."""
    if isinstance(value, Tensor):
        return value.to(device=raw.device, dtype=raw.dtype)
    return raw.new_tensor(value)


class ExposureMlp(nn.Module):
    """A 16x1024 residual MLP producing the stored oracle's base-action logits."""

    def __init__(
        self,
        feature_mean: Tensor,
        feature_std: Tensor,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        if feature_mean.shape != (INPUT_FEATURE_COUNT,) or feature_std.shape != (INPUT_FEATURE_COUNT,):
            raise ValueError(
                f"feature normalization must match the {INPUT_FEATURE_COUNT}-value input contract"
            )
        self.register_buffer("feature_mean", feature_mean.float().clone())
        self.register_buffer("feature_std", feature_std.float().clamp_min(1e-6).clone())
        self.layers = nn.ModuleList([
            nn.Linear(INPUT_FEATURE_COUNT if index == 0 else HIDDEN_WIDTH, HIDDEN_WIDTH)
            for index in range(HIDDEN_LAYER_COUNT)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(HIDDEN_WIDTH) for _ in range(HIDDEN_LAYER_COUNT)])
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(HIDDEN_WIDTH, OUTPUT_ACTION_COUNT)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for layer in self.layers:
            nn.init.kaiming_normal_(layer.weight, nonlinearity="linear")
            nn.init.zeros_(layer.bias)
        nn.init.zeros_(self.output.bias)
        nn.init.normal_(self.output.weight, std=0.01)

    def encode(self, features: Tensor) -> Tensor:
        hidden = (features - self.feature_mean) / self.feature_std
        for index, (layer, norm) in enumerate(zip(self.layers, self.norms, strict=True)):
            update = self.dropout(functional.silu(norm(layer(hidden))))
            hidden = update if index == 0 else (hidden + update) * (2.0 ** -0.5)
        return hidden

    def forward(self, features: Tensor) -> Tensor:
        return self.output(self.encode(features))


class ParameterExposureMlp(ExposureMlp):
    """Eight-parameter projection head retained only for fit diagnostics."""

    def __init__(
        self,
        feature_mean: Tensor,
        feature_std: Tensor,
        dropout: float = 0.05,
    ) -> None:
        super().__init__(feature_mean, feature_std, dropout)
        self.output = nn.Linear(HIDDEN_WIDTH, TEACHER_PARAMETER_COUNT)
        self.reset_parameters()
        with torch.no_grad():
            self.output.bias[6] = -3.0
            self.output.bias[7] = 3.0

    def forward(self, features: Tensor) -> Tensor:
        raw = self.output(self.encode(features))
        bounded = torch.tanh(torch.cat((raw[:, :2], raw[:, 6:8]), dim=-1)) * 14.0
        return torch.cat((bounded[:, :2], raw[:, 2:6], bounded[:, 2:]), dim=-1)


def conditional_policy_logits(
    raw: Tensor,
    actions: Tensor,
    current: Tensor,
    support: PolicySupport,
    cutoff_raw: Tensor | None = None,
) -> Tensor:
    """Differentiable equivalent of conditionalFourSegmentParametersFromRaw/logKernel."""
    # Keep support scalars in the policy tensor's arithmetic dtype.  When the
    # JSON plan spells a bound as an integer, torch.compile otherwise carries
    # it as a SymInt and lowers divisions/powers through FP64.  On consumer
    # Ampere GPUs that turns the fused objective into an FP64-bound kernel even
    # though every input and output tensor is float32.
    latent_lower = policy_support_value(raw, support.latent_lower)
    latent_upper = policy_support_value(raw, support.latent_upper)
    visible_lower = policy_support_value(raw, support.visible_lower)
    visible_upper = policy_support_value(raw, support.visible_upper)
    friction = policy_support_value(raw, support.friction)
    temperature = policy_support_value(raw, support.temperature)
    basis_span = latent_upper - latent_lower
    hinge_span = basis_span if support.hinge_span is None else policy_support_value(
        raw, support.hinge_span
    )
    inverse_basis_span = torch.reciprocal(basis_span)
    half_basis_span = basis_span * 0.5
    basis_center = (latent_lower + latent_upper) * 0.5
    first_fraction = torch.sigmoid(raw[..., 0])
    latent_span = latent_upper - latent_lower
    c1 = latent_lower + latent_span * first_fraction
    second_fraction = torch.sigmoid(raw[..., 1])
    c2 = c1 + (latent_upper - c1) * second_fraction
    slope_scale = inverse_basis_span
    precision_scale = torch.reciprocal(half_basis_span.square())
    base = raw[..., 2] * slope_scale
    precision = raw[..., 3] * precision_scale
    beta_c1 = raw[..., 4] * slope_scale
    beta_c2 = raw[..., 5] * slope_scale
    buy_slope_at_zero = friction * torch.reciprocal(1.0 - friction)
    sell_slope_at_zero = friction
    beta_x_value = -(buy_slope_at_zero + sell_slope_at_zero) \
        * torch.reciprocal(temperature)
    beta_x = beta_x_value.expand_as(base)
    kappa_c1 = (82.0 * torch.reciprocal(hinge_span)).expand_as(base)
    # ``hinge_span`` may deliberately differ from the latent support during a
    # compact-support initialization.  Keeping every fixed transition width on
    # that shared span makes the compact score exactly remappable to the full
    # effective support instead of subtly changing the moving fee hinge.
    kappa_x = (678.0 * torch.reciprocal(hinge_span)).expand_as(base)
    kappa_c2 = kappa_c1
    cutoff_source = raw if cutoff_raw is None else cutoff_raw
    if cutoff_source.shape != raw.shape:
        raise ValueError("cutoff source must match the raw parameter tensor")
    lower_fraction = torch.sigmoid(cutoff_source[..., 6])
    upper_fraction = torch.sigmoid(cutoff_source[..., 7])
    cutoff_lower = latent_lower + (-latent_lower) * lower_fraction
    cutoff_upper = latent_upper * upper_fraction
    cutoff_lower = torch.where(
        cutoff_source[..., 6] <= -13.999999, latent_lower, cutoff_lower
    )
    cutoff_lower = torch.where(cutoff_source[..., 6] >= 13.999999, 0.0, cutoff_lower)
    cutoff_upper = torch.where(cutoff_source[..., 7] <= -13.999999, 0.0, cutoff_upper)
    cutoff_upper = torch.where(
        cutoff_source[..., 7] >= 13.999999, latent_upper, cutoff_upper
    )

    action = actions.view(1, 1, -1)
    c1 = c1[:, :, None]
    c2 = c2[:, :, None]
    current = current[:, :, None]
    base = base[:, :, None]
    precision = precision[:, :, None]
    beta_c1 = beta_c1[:, :, None]
    beta_x = beta_x[:, :, None]
    beta_c2 = beta_c2[:, :, None]
    kappa_c1 = kappa_c1[:, :, None]
    kappa_x = kappa_x[:, :, None]
    kappa_c2 = kappa_c2[:, :, None]
    cutoff_lower = cutoff_lower[:, :, None]
    cutoff_upper = cutoff_upper[:, :, None]

    logits = (
        base * (action - latent_lower)
        - 0.5 * precision * (action - basis_center).square()
        + beta_c1 * scaled_softplus(action - c1, kappa_c1)
        + beta_x * scaled_softplus(action - current, kappa_x)
        + beta_c2 * scaled_softplus(action - c2, kappa_c2)
    )
    feasible = (
        (action >= visible_lower) & (action <= visible_upper)
        & (action >= cutoff_lower) & (action <= cutoff_upper)
    )
    return logits.masked_fill(~feasible, torch.finfo(logits.dtype).min)


def oracle_policy_logits(
    action_values: Tensor,
    actions: Tensor,
    current: Tensor,
    support: PolicySupport,
) -> Tensor:
    action = actions.view(1, 1, -1)
    current = current[:, :, None]
    difference = action - current
    buy_denominator = 1.0 - support.friction + support.friction * action
    sell_denominator = 1.0 - support.friction * action
    buy_factor = 1.0 - support.friction * difference / buy_denominator
    sell_factor = 1.0 - support.friction * (-difference) / sell_denominator
    factor = torch.where(difference > 0, buy_factor, torch.where(difference < 0, sell_factor, 1.0))
    transition = torch.log(factor.clamp_min(torch.finfo(action_values.dtype).tiny))
    return (action_values[:, None, :] + transition) / support.temperature


def materialize_raw_oracle_policy_map(
    base_probabilities: Tensor,
    actions: Tensor,
    current: Tensor,
    support: PolicySupport,
    cutoff_raw: Tensor | None = None,
) -> Tensor:
    """Expand the lossless 255-value oracle factor into [batch, current, action]."""
    if base_probabilities.ndim != 2 or actions.ndim != 1:
        raise ValueError("raw oracle factors must have [batch, action] layout")
    current_values = current.reshape(-1)
    if base_probabilities.shape[1] != actions.numel() or current_values.numel() < 1:
        raise ValueError("raw oracle factors do not match the requested map axes")
    if not 0 <= support.friction < 1 or support.temperature <= 0:
        raise ValueError("raw oracle map support has invalid friction or temperature")
    action = actions.float().view(1, 1, -1)
    current_value = current_values.float().view(1, -1, 1)
    difference = action - current_value
    buy_denominator = 1.0 - support.friction + support.friction * action
    sell_denominator = 1.0 - support.friction * action
    buy_factor = 1.0 - support.friction * difference / buy_denominator
    sell_factor = 1.0 - support.friction * (-difference) / sell_denominator
    factor = torch.where(
        difference > 0,
        buy_factor,
        torch.where(difference < 0, sell_factor, torch.ones_like(difference)),
    )
    base = base_probabilities.float()
    base_log = torch.where(
        base > 0,
        base.clamp_min(torch.finfo(base.dtype).tiny).log(),
        torch.full_like(base, torch.finfo(base.dtype).min),
    )
    logits = base_log[:, None, :] \
        + factor.clamp_min(torch.finfo(base.dtype).tiny).log() / support.temperature
    if cutoff_raw is not None:
        if cutoff_raw.ndim != 2 or cutoff_raw.shape[0] != base.shape[0] \
                or cutoff_raw.shape[1] not in (2, TEACHER_PARAMETER_COUNT):
            raise ValueError("raw oracle cutoff coordinates must match the map batch")
        cutoff = cutoff_raw[:, -2:].float()
        lower_fraction = torch.sigmoid(cutoff[:, 0])
        upper_fraction = torch.sigmoid(cutoff[:, 1])
        cutoff_lower = support.latent_lower + (-support.latent_lower) * lower_fraction
        cutoff_upper = support.latent_upper * upper_fraction
        cutoff_lower = torch.where(
            cutoff[:, 0] <= -13.999999, support.latent_lower, cutoff_lower
        )
        cutoff_lower = torch.where(cutoff[:, 0] >= 13.999999, 0.0, cutoff_lower)
        cutoff_upper = torch.where(cutoff[:, 1] <= -13.999999, 0.0, cutoff_upper)
        cutoff_upper = torch.where(
            cutoff[:, 1] >= 13.999999, support.latent_upper, cutoff_upper
        )
        feasible = (
            (action >= cutoff_lower[:, None, None])
            & (action <= cutoff_upper[:, None, None])
        )
        logits = logits.masked_fill(~feasible, torch.finfo(logits.dtype).min)
    return torch.softmax(logits, dim=-1)


def direct_oracle_loss(
    predicted_base_logits: Tensor,
    target_base_probabilities: Tensor,
    actions: Tensor,
    current: Tensor,
    support: PolicySupport,
    weights: DirectLossWeights,
    example_time_weights: Tensor,
    example_times_ms: Tensor,
    sampling_interval_ms: int,
) -> dict[str, Tensor]:
    """Score direct base-action logits through the deterministic fee transform."""
    if predicted_base_logits.shape != target_base_probabilities.shape \
            or predicted_base_logits.ndim != 2 \
            or predicted_base_logits.shape[1] != actions.numel():
        raise ValueError(
            "direct oracle logits and stored probabilities must match the action grid"
        )
    if current.ndim != 2 or current.shape[0] != predicted_base_logits.shape[0]:
        raise ValueError("current-exposure states must match the direct oracle batch")
    if example_time_weights.ndim != 1 \
            or example_time_weights.shape[0] != predicted_base_logits.shape[0]:
        raise ValueError("persisted example weights must match the training batch")
    if example_times_ms.ndim != 1 \
            or example_times_ms.shape != example_time_weights.shape:
        raise ValueError("example timestamps must match the training batch")
    if sampling_interval_ms <= 0:
        raise ValueError("sampling interval must be positive")

    action = actions.float().view(1, 1, -1)
    visible = (
        (actions.float() >= support.visible_lower)
        & (actions.float() <= support.visible_upper)
    ).view(1, -1)
    action_count = actions.numel()

    minimum = torch.finfo(torch.float32).min
    predicted_base_log = torch.log_softmax(
        predicted_base_logits.float().masked_fill(~visible, minimum),
        dim=-1,
    )
    target_base = target_base_probabilities.float().masked_fill(~visible, 0.0)
    target_base = target_base / target_base.sum(
        dim=-1,
        keepdim=True,
    ).clamp_min(torch.finfo(torch.float32).tiny)
    target_base_log = torch.where(
        target_base > 0,
        target_base.clamp_min(torch.finfo(torch.float32).tiny).log(),
        torch.full_like(target_base, minimum),
    )

    current_value = current.float()[:, :, None]
    difference = action - current_value
    friction = policy_support_value(predicted_base_logits, support.friction).float()
    temperature = policy_support_value(
        predicted_base_logits,
        support.temperature,
    ).float()
    buy_denominator = 1.0 - friction + friction * action
    sell_denominator = 1.0 - friction * action
    buy_factor = 1.0 - friction * difference / buy_denominator
    sell_factor = 1.0 - friction * (-difference) / sell_denominator
    factor = torch.where(
        difference > 0,
        buy_factor,
        torch.where(difference < 0, sell_factor, torch.ones_like(difference)),
    )
    transition = factor.clamp_min(torch.finfo(torch.float32).tiny).log() \
        / temperature
    predicted_log = torch.log_softmax(
        predicted_base_log[:, None, :] + transition,
        dim=-1,
    )
    target_log = torch.log_softmax(
        target_base_log[:, None, :] + transition,
        dim=-1,
    )
    predicted = predicted_log.exp()
    target = target_log.exp()

    time_weight = example_time_weights.float()
    effective_sample_ratio = (
        time_weight.sum().square()
        / (time_weight.numel() * time_weight.square().sum()).clamp_min(1e-8)
    )
    normalized_weight = time_weight / time_weight.mean().clamp_min(1e-8)
    denominator = normalized_weight.sum().clamp_min(1e-8)

    def weighted_mean(value: Tensor) -> Tensor:
        return (value * normalized_weight).sum() / denominator

    cross_entropy_per_example = -(target * predicted_log).sum(
        dim=-1,
    ).mean(dim=-1)
    kl_divergence_per_example = (
        target * (target_log - predicted_log)
    ).sum(dim=-1).mean(dim=-1).clamp_min(0)
    base_kl_divergence_per_example = (
        target_base
        * (target_base_log - predicted_base_log)
    ).sum(dim=-1).clamp_min(0)
    probability_mse_per_example = surface_probability_mse_per_example(
        predicted,
        target,
    )
    predicted_entropy = -(predicted * predicted_log).sum(dim=-1).mean(dim=-1)
    target_entropy = -(target * target_log).sum(dim=-1).mean(dim=-1)
    excess_entropy_per_example = (
        (predicted_entropy - target_entropy).clamp_min(0)
        / math.log(action_count)
    ).square()

    predicted_state_mean = (predicted * action).sum(dim=-1)
    predicted_state_second = (predicted * action.square()).sum(dim=-1)
    target_state_mean = (target * action).sum(dim=-1)
    target_state_second = (target * action.square()).sum(dim=-1)
    temporal_mutual_information_reward, temporal_mutual_information, \
        target_temporal_mutual_information, temporal_example_count = (
            gaussian_temporal_mutual_information(
                predicted_state_mean,
                predicted_state_second,
                target_state_mean,
                target_state_second,
                example_times_ms,
                normalized_weight,
                sampling_interval_ms,
                action_count,
            )
        )
    oracle_mutual_information = gaussian_oracle_mutual_information(
        predicted_state_mean,
        predicted_state_second,
        target_state_mean,
        target_state_second,
        example_times_ms,
        normalized_weight,
        sampling_interval_ms,
        action_count,
    )

    cross_entropy = weighted_mean(cross_entropy_per_example)
    kl_divergence = weighted_mean(kl_divergence_per_example)
    kl_weight_sum = time_weight.sum()
    kl_centered_square_sum = (
        time_weight
        * (kl_divergence_per_example - kl_divergence).square()
    ).sum()
    probability_mse = weighted_mean(probability_mse_per_example)
    probability_mse_weight_sum = time_weight.sum()
    probability_mse_centered_square_sum = (
        time_weight
        * (probability_mse_per_example - probability_mse).square()
    ).sum()
    excess_entropy = weighted_mean(excess_entropy_per_example)
    loss = (
        weights.cross_entropy * cross_entropy
        + weights.probability_mse * probability_mse
        + weights.excess_entropy * excess_entropy
        - weights.temporal_mutual_information
        * temporal_mutual_information_reward
        - weights.oracle_mutual_information * oracle_mutual_information
    )
    return {
        "loss": loss,
        "klDivergence": kl_divergence,
        "klDivergenceVariance": (
            kl_centered_square_sum / kl_weight_sum.clamp_min(1e-8)
        ).clamp_min(0),
        "klDivergenceStdDev": (
            kl_centered_square_sum / kl_weight_sum.clamp_min(1e-8)
        ).clamp_min(0).sqrt(),
        "klWeightSum": kl_weight_sum,
        "klCenteredSquareSum": kl_centered_square_sum,
        "baseKlDivergence": weighted_mean(
            base_kl_divergence_per_example
        ),
        "probabilityMse": probability_mse,
        "probabilityMseVariance": (
            probability_mse_centered_square_sum
            / probability_mse_weight_sum.clamp_min(1e-8)
        ).clamp_min(0),
        "probabilityMseStdDev": (
            probability_mse_centered_square_sum
            / probability_mse_weight_sum.clamp_min(1e-8)
        ).clamp_min(0).sqrt(),
        "probabilityMseWeightSum": probability_mse_weight_sum,
        "probabilityMseCenteredSquareSum":
            probability_mse_centered_square_sum,
        "excessEntropy": excess_entropy,
        "temporalMutualInformation": temporal_mutual_information,
        "targetTemporalMutualInformation":
            target_temporal_mutual_information,
        "temporalMutualInformationReward":
            temporal_mutual_information_reward,
        "temporalExampleCount": temporal_example_count,
        "oracleMutualInformation": oracle_mutual_information,
        "targetEntropy": weighted_mean(target_entropy),
        "predictedEntropy": weighted_mean(predicted_entropy),
        "distanceImbalanceWeight": time_weight.mean(),
        "timeWeightEffectiveSampleRatio": effective_sample_ratio,
        "timeWeightSum": time_weight.sum(),
        "timeWeightSquareSum": time_weight.square().sum(),
    }


def fitted_teacher_loss(
    predicted_raw: Tensor,
    target_raw: Tensor,
    actions: Tensor,
    current: Tensor,
    support: PolicySupport,
    parameter_scale: Tensor,
    weights: LossWeights,
    example_time_weights: Tensor,
    example_times_ms: Tensor,
    sampling_interval_ms: int,
    include_deployment_metrics: bool = True,
) -> dict[str, Tensor]:
    """Apply persisted whole-example weights, normalized only by this batch's mean."""
    state_count = current.shape[-1]
    predicted_rows = predicted_raw.float()[:, None, :].expand(-1, state_count, -1)
    target_rows = target_raw.float()[:, None, :].expand(-1, state_count, -1)
    # The differentiable training objectives use the teacher's feasible mask
    # for both policies so a temporarily too-narrow predicted cutoff cannot
    # create infinite CE. Deployment KL below separately scores the predicted
    # cutoff and is deliberately excluded from the optimization objective.
    predicted_logits = conditional_policy_logits(
        predicted_rows,
        actions.float(),
        current.float(),
        support,
        cutoff_raw=target_rows,
    )
    target_logits = conditional_policy_logits(target_rows, actions.float(), current.float(), support)
    predicted_log = torch.log_softmax(predicted_logits, dim=-1)
    target_log = torch.log_softmax(target_logits, dim=-1)
    predicted = predicted_log.exp()
    target = target_log.exp()
    if include_deployment_metrics:
        deployment_logits = conditional_policy_logits(
            predicted_rows.detach(),
            actions.float(),
            current.float(),
            support,
        )
        deployment_predicted = torch.softmax(
            deployment_logits.float(),
            dim=-1,
        )
        # Hard cutoff disagreement would make KL(target || prediction)
        # infinite. Clamp and renormalize only the predicted deployment policy
        # to retain a finite, strongly penalized metric with a clear per-cell
        # probability floor.
        deployment_predicted = deployment_predicted.clamp_min(
            DEPLOYMENT_PROBABILITY_FLOOR
        )
        deployment_predicted = (
            deployment_predicted
            / deployment_predicted.sum(dim=-1, keepdim=True)
        )
        deployment_predicted_log = deployment_predicted.log()

    if example_time_weights.ndim != 1 \
            or example_time_weights.shape[0] != predicted_raw.shape[0]:
        raise ValueError("persisted example weights must match the training batch")
    if example_times_ms.ndim != 1 or example_times_ms.shape != example_time_weights.shape:
        raise ValueError("example timestamps must match the training batch")
    if sampling_interval_ms <= 0:
        raise ValueError("sampling interval must be positive")
    time_weight = example_time_weights.float()
    effective_sample_ratio = (
        time_weight.sum().square()
        / (time_weight.numel() * time_weight.square().sum()).clamp_min(1e-8)
    )
    normalized_weight = time_weight / time_weight.mean().clamp_min(1e-8)
    denominator = normalized_weight.sum().clamp_min(1e-8)

    def weighted_mean(value: Tensor) -> Tensor:
        return (value * normalized_weight).sum() / denominator

    cross_entropy_per_example = -(target * predicted_log).sum(dim=-1).mean(dim=-1)
    kl_divergence_per_example = (
        target * (target_log - predicted_log)
    ).sum(dim=-1).mean(dim=-1).clamp_min(0)
    if include_deployment_metrics:
        deployment_kl_divergence_per_example = (
            target.detach()
            * (target_log.detach() - deployment_predicted_log)
        ).sum(dim=-1).mean(dim=-1).clamp_min(0)
    probability_mse_per_example = surface_probability_mse_per_example(
        predicted,
        target,
    )
    parameter_mse_per_example = (
        (predicted_raw.float() - target_raw.float()) / parameter_scale.float()
    ).square().mean(dim=-1)
    predicted_entropy = -(predicted * predicted_log).sum(dim=-1).mean(dim=-1)
    target_entropy = -(target * target_log).sum(dim=-1).mean(dim=-1)
    excess_entropy_per_example = (
        (predicted_entropy - target_entropy).clamp_min(0) / math.log(actions.numel())
    ).square()

    action = actions.float().view(1, 1, -1)
    predicted_state_mean = (predicted * action).sum(dim=-1)
    predicted_state_second = (predicted * action.square()).sum(dim=-1)
    target_state_mean = (target * action).sum(dim=-1)
    target_state_second = (target * action.square()).sum(dim=-1)
    temporal_mutual_information_reward, temporal_mutual_information, \
        target_temporal_mutual_information, temporal_example_count = (
        gaussian_temporal_mutual_information(
            predicted_state_mean,
            predicted_state_second,
            target_state_mean,
            target_state_second,
            example_times_ms,
            normalized_weight,
            sampling_interval_ms,
            actions.numel(),
        )
    )
    oracle_mutual_information = gaussian_oracle_mutual_information(
        predicted_state_mean,
        predicted_state_second,
        target_state_mean,
        target_state_second,
        example_times_ms,
        normalized_weight,
        sampling_interval_ms,
        actions.numel(),
    )

    cross_entropy = weighted_mean(cross_entropy_per_example)
    kl_divergence = weighted_mean(kl_divergence_per_example)
    kl_weight_sum = time_weight.sum()
    kl_centered_square_sum = (
        time_weight * (kl_divergence_per_example - kl_divergence).square()
    ).sum()
    kl_divergence_std_dev = (
        kl_centered_square_sum / kl_weight_sum.clamp_min(1e-8)
    ).clamp_min(0).sqrt()
    if include_deployment_metrics:
        deployment_kl_divergence = weighted_mean(
            deployment_kl_divergence_per_example
        )
        deployment_kl_weight_sum = time_weight.sum()
        deployment_kl_centered_square_sum = (
            time_weight
            * (
                deployment_kl_divergence_per_example
                - deployment_kl_divergence
            ).square()
        ).sum()
        deployment_kl_divergence_std_dev = (
            deployment_kl_centered_square_sum
            / deployment_kl_weight_sum.clamp_min(1e-8)
        ).clamp_min(0).sqrt()
    probability_mse = weighted_mean(probability_mse_per_example)
    probability_mse_weight_sum = time_weight.sum()
    probability_mse_centered_square_sum = (
        time_weight
        * (probability_mse_per_example - probability_mse).square()
    ).sum()
    parameter_mse = weighted_mean(parameter_mse_per_example)
    excess_entropy = weighted_mean(excess_entropy_per_example)
    loss = (
        weights.cross_entropy * cross_entropy
        + weights.probability_mse * probability_mse
        + weights.parameter_mse * parameter_mse
        + weights.excess_entropy * excess_entropy
        - weights.temporal_mutual_information * temporal_mutual_information_reward
        - weights.oracle_mutual_information * oracle_mutual_information
    )
    metrics = {
        "loss": loss,
        "klDivergence": kl_divergence,
        "klDivergenceVariance": (
            kl_centered_square_sum / kl_weight_sum.clamp_min(1e-8)
        ).clamp_min(0),
        "klDivergenceStdDev": kl_divergence_std_dev,
        "klWeightSum": kl_weight_sum,
        "klCenteredSquareSum": kl_centered_square_sum,
        "probabilityMse": probability_mse,
        "probabilityMseVariance": (
            probability_mse_centered_square_sum
            / probability_mse_weight_sum.clamp_min(1e-8)
        ).clamp_min(0),
        "probabilityMseStdDev": (
            probability_mse_centered_square_sum
            / probability_mse_weight_sum.clamp_min(1e-8)
        ).clamp_min(0).sqrt(),
        "probabilityMseWeightSum": probability_mse_weight_sum,
        "probabilityMseCenteredSquareSum":
            probability_mse_centered_square_sum,
        "parameterMse": parameter_mse,
        "excessEntropy": excess_entropy,
        "temporalMutualInformation": temporal_mutual_information,
        "targetTemporalMutualInformation": target_temporal_mutual_information,
        "temporalMutualInformationReward": temporal_mutual_information_reward,
        "temporalExampleCount": temporal_example_count,
        "oracleMutualInformation": oracle_mutual_information,
        "targetEntropy": weighted_mean(target_entropy),
        "predictedEntropy": weighted_mean(predicted_entropy),
        "rawParameterMae": weighted_mean((predicted_raw.float() - target_raw.float()).abs().mean(dim=-1)),
        "distanceImbalanceWeight": time_weight.mean(),
        "timeWeightEffectiveSampleRatio": effective_sample_ratio,
        "timeWeightSum": time_weight.sum(),
        "timeWeightSquareSum": time_weight.square().sum(),
    }
    if include_deployment_metrics:
        metrics.update({
            "deploymentKlDivergence": deployment_kl_divergence,
            "deploymentKlDivergenceStdDev":
                deployment_kl_divergence_std_dev,
            "deploymentKlWeightSum": deployment_kl_weight_sum,
            "deploymentKlCenteredSquareSum":
                deployment_kl_centered_square_sum,
        })
    return metrics


def gaussian_temporal_mutual_information(
    predicted_mean: Tensor,
    predicted_second: Tensor,
    target_mean: Tensor,
    target_second: Tensor,
    times_ms: Tensor,
    example_weights: Tensor,
    sampling_interval_ms: int,
    action_count: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Gaussian I(time; action | current exposure), capped by the teacher."""
    if predicted_mean.shape != predicted_second.shape \
            or predicted_mean.shape != target_mean.shape \
            or predicted_mean.shape != target_second.shape \
            or predicted_mean.ndim != 2:
        raise ValueError("temporal moments must have matching [time, current exposure] shapes")
    if times_ms.ndim != 1 or example_weights.ndim != 1 \
            or times_ms.shape != example_weights.shape \
            or times_ms.shape[0] != predicted_mean.shape[0]:
        raise ValueError("temporal policy metadata must match the timestamp axis")
    if sampling_interval_ms <= 0 or action_count < 2:
        raise ValueError("temporal MI requires a positive cadence and at least two actions")
    if predicted_mean.shape[0] < 2:
        zero = predicted_mean.sum() * 0.0
        return zero, zero, zero, torch.zeros((), device=predicted_mean.device)

    weight = example_weights.float().view(-1, 1)
    denominator = weight.sum().clamp_min(1e-8)

    def gaussian_information(mean: Tensor, second: Tensor) -> Tensor:
        mean = mean.float()
        second = second.float()
        time_mean = (weight * mean).sum(dim=0) / denominator
        total_variance = (
            (weight * second).sum(dim=0) / denominator - time_mean.square()
        ).clamp_min(0)
        within_variance = (
            weight * (second - mean.square()).clamp_min(0)
        ).sum(dim=0) / denominator
        return (
            0.5 * torch.log(
                ((total_variance + 1e-8) / (within_variance + 1e-8)).clamp_min(1)
            ) / math.log(action_count)
        ).clamp(0, 1)

    predicted_information = gaussian_information(predicted_mean, predicted_second)
    target_information_by_state = gaussian_information(target_mean, target_second).detach()
    contiguous = (
        (times_ms[1:] - times_ms[:-1]) == sampling_interval_ms
    ).all().to(predicted_information.dtype)
    # Cap each conditioning state's reward separately so temporal variation at
    # one exposure cannot compensate for a missing teacher change at another.
    temporal_information = torch.minimum(
        predicted_information,
        target_information_by_state,
    ).mean() * contiguous
    predicted_information_mean = predicted_information.mean() * contiguous
    target_information = target_information_by_state.mean() * contiguous
    example_count = torch.as_tensor(
        predicted_mean.shape[0],
        device=predicted_mean.device,
        dtype=predicted_information.dtype,
    ) * contiguous
    return temporal_information, predicted_information_mean, target_information, example_count


def gaussian_oracle_mutual_information(
    predicted_mean: Tensor,
    predicted_second: Tensor,
    target_mean: Tensor,
    target_second: Tensor,
    times_ms: Tensor,
    example_weights: Tensor,
    sampling_interval_ms: int,
    action_count: int,
) -> Tensor:
    """Gaussian I(predicted action; oracle action | current exposure) over time."""
    if predicted_mean.shape != predicted_second.shape \
            or predicted_mean.shape != target_mean.shape \
            or predicted_mean.shape != target_second.shape \
            or predicted_mean.ndim != 2:
        raise ValueError("oracle MI moments must have matching [time, current exposure] shapes")
    if times_ms.ndim != 1 or example_weights.ndim != 1 \
            or times_ms.shape != example_weights.shape \
            or times_ms.shape[0] != predicted_mean.shape[0]:
        raise ValueError("oracle MI metadata must match the timestamp axis")
    if sampling_interval_ms <= 0 or action_count < 2:
        raise ValueError("oracle MI requires a positive cadence and at least two actions")
    if predicted_mean.shape[0] < 2:
        return predicted_mean.sum() * 0.0

    weight = example_weights.float().view(-1, 1)
    denominator = weight.sum().clamp_min(1e-8)
    predicted_mean = predicted_mean.float()
    predicted_second = predicted_second.float()
    target_mean = target_mean.float()
    target_second = target_second.float()
    predicted_time_mean = (weight * predicted_mean).sum(dim=0) / denominator
    target_time_mean = (weight * target_mean).sum(dim=0) / denominator
    predicted_variance = (
        (weight * predicted_second).sum(dim=0) / denominator
        - predicted_time_mean.square()
    ).clamp_min(0)
    target_variance = (
        (weight * target_second).sum(dim=0) / denominator
        - target_time_mean.square()
    ).clamp_min(0)
    covariance = (
        (weight * predicted_mean * target_mean).sum(dim=0) / denominator
        - predicted_time_mean * target_time_mean
    )
    correlation_squared = (
        covariance.square() / (predicted_variance * target_variance).clamp_min(1e-8)
    ).clamp(0, 1 - 1e-6)
    information_by_state = (
        -0.5 * torch.log1p(-correlation_squared) / math.log(action_count)
    ).clamp(0, 1)
    contiguous = (
        (times_ms[1:] - times_ms[:-1]) == sampling_interval_ms
    ).all().to(information_by_state.dtype)
    return information_by_state.mean() * contiguous


def distance_imbalance_time_weights(
    target_probability: Tensor,
    actions: Tensor,
    current: Tensor,
    weighting: TimeWeighting,
) -> Tensor:
    """Return epsilon plus the absolute global distance-sensitive imbalance."""
    validate_time_weighting(weighting)
    advice = distance_imbalance_advice(
        target_probability,
        actions,
        current,
        weighting.distance_epsilon,
    )
    return advice.abs() + weighting.minimum_weight


def distance_imbalance_advice(
    target_probability: Tensor,
    actions: Tensor,
    current: Tensor,
    distance_epsilon: float,
) -> Tensor:
    """Signed timestamp advice: sum_x E[a-x|x] / (sum_x E[|a-x||x] + epsilon)."""
    if distance_epsilon < 0:
        raise ValueError("distance imbalance epsilon must be non-negative")
    if target_probability.ndim != 3:
        raise ValueError("target probability must have batch, state, and action dimensions")
    if target_probability.shape[-1] != actions.numel() \
            or target_probability.shape[:2] != current.shape:
        raise ValueError("target probability, action, and current-state dimensions do not match")
    displacement = actions.float().view(1, 1, -1) - current.float()[:, :, None]
    expected_displacement = (target_probability.float() * displacement).sum(dim=-1)
    expected_distance = (target_probability.float() * displacement.abs()).sum(dim=-1)
    return expected_displacement.sum(dim=-1) / (
        expected_distance.sum(dim=-1) + distance_epsilon
    )


def persistent_distance_imbalance_time_weights(
    advice: Tensor,
    times_ms: Tensor,
    sampling_interval_ms: int,
    weighting: TimeWeighting,
) -> Tensor:
    """Causally boost repeated important same-side advice in chronological order."""
    validate_time_weighting(weighting)
    if advice.ndim != 1 or times_ms.ndim != 1 or advice.shape != times_ms.shape:
        raise ValueError("advice and timestamps must be matching one-dimensional tensors")
    if sampling_interval_ms <= 0:
        raise ValueError("sampling interval must be positive")
    advice_values = advice.detach().float().cpu().tolist()
    time_values = times_ms.detach().long().cpu().tolist()
    result = torch.empty_like(advice, dtype=torch.float32, device="cpu")
    evidence = 0.0
    side = 0
    previous_time: int | None = None
    decay_per_step = 0.5 ** (1.0 / weighting.memory_half_life_steps)
    for index, (value, time_ms) in enumerate(zip(advice_values, time_values, strict=True)):
        if previous_time is not None:
            elapsed = time_ms - previous_time
            if elapsed <= 0:
                raise ValueError("persistence timestamps must be strictly increasing")
            elapsed_steps = elapsed / sampling_interval_ms
            if elapsed_steps > weighting.reset_after_gap_steps:
                evidence = 0.0
                side = 0
            else:
                missing_steps = max(0.0, elapsed_steps - 1.0)
                evidence *= decay_per_step ** missing_steps
        magnitude = abs(value)
        multiplier = 1.0
        if magnitude >= weighting.minimum_advice_magnitude:
            current_side = 1 if value > 0 else -1
            if current_side != side:
                evidence = 0.0
                side = current_side
            multiplier = min(
                weighting.maximum_multiplier,
                1.0 + weighting.growth_per_prior_advice * evidence,
            )
            evidence += 1.0
        else:
            evidence *= decay_per_step
        result[index] = weighting.minimum_weight + magnitude * multiplier
        previous_time = time_ms
    return result.to(device=advice.device)


def validate_time_weighting(weighting: TimeWeighting) -> None:
    if weighting.distance_epsilon < 0 or weighting.minimum_weight <= 0 \
            or not 0 <= weighting.minimum_advice_magnitude <= 1 \
            or weighting.memory_half_life_steps <= 0 \
            or weighting.growth_per_prior_advice < 0 \
            or weighting.maximum_multiplier < 1 \
            or weighting.reset_after_gap_steps < 1 \
            or weighting.resolution_divergence_multiplier < 0:
        raise ValueError("invalid distance-imbalance time-weighting configuration")


def surface_probability_mse_per_example(
    predicted_probability: Tensor,
    target_probability: Tensor,
) -> Tensor:
    """Average pMSE across the supplied state/action surface."""
    if predicted_probability.shape != target_probability.shape \
            or predicted_probability.ndim != 3:
        raise ValueError("visible probability MSE inputs do not match")
    return (target_probability - predicted_probability).square().mean(dim=(-1, -2))


def scaled_softplus(offset: Tensor, kappa: Tensor) -> Tensor:
    return functional.softplus(kappa * offset) / kappa


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
