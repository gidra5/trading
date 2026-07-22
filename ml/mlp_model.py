from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as functional


INPUT_FEATURE_COUNT = 909
OUTPUT_PARAMETER_COUNT = 8
HIDDEN_LAYER_COUNT = 16
HIDDEN_WIDTH = 1024


@dataclass(frozen=True)
class PolicySupport:
    latent_lower: float
    latent_upper: float
    visible_lower: float
    visible_upper: float
    friction: float
    temperature: float


@dataclass(frozen=True)
class LossWeights:
    cross_entropy: float = 1.0
    probability_mse: float = 1.0
    parameter_mse: float = 1.0
    excess_entropy: float = 1.0
    state_mutual_information: float = 1.0
    oracle_mutual_information: float = 1.0


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


class ExposureMlp(nn.Module):
    """A 16x1024 residual MLP producing score shape plus hard survival cutoffs."""

    def __init__(
        self,
        feature_mean: Tensor,
        feature_std: Tensor,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        if feature_mean.shape != (INPUT_FEATURE_COUNT,) or feature_std.shape != (INPUT_FEATURE_COUNT,):
            raise ValueError("feature normalization must match the 909-value input contract")
        self.register_buffer("feature_mean", feature_mean.float().clone())
        self.register_buffer("feature_std", feature_std.float().clamp_min(1e-6).clone())
        self.layers = nn.ModuleList([
            nn.Linear(INPUT_FEATURE_COUNT if index == 0 else HIDDEN_WIDTH, HIDDEN_WIDTH)
            for index in range(HIDDEN_LAYER_COUNT)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(HIDDEN_WIDTH) for _ in range(HIDDEN_LAYER_COUNT)])
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(HIDDEN_WIDTH, OUTPUT_PARAMETER_COUNT)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for layer in self.layers:
            nn.init.kaiming_normal_(layer.weight, nonlinearity="linear")
            nn.init.zeros_(layer.bias)
        nn.init.zeros_(self.output.bias)
        nn.init.normal_(self.output.weight, std=0.01)
        # Start with the complete effective range feasible. The two cutoff
        # outputs are bounded below, so moderately saturated logits are enough.
        with torch.no_grad():
            self.output.bias[6] = -3.0
            self.output.bias[7] = 3.0

    def forward(self, features: Tensor) -> Tensor:
        hidden = (features - self.feature_mean) / self.feature_std
        for index, (layer, norm) in enumerate(zip(self.layers, self.norms, strict=True)):
            update = self.dropout(functional.silu(norm(layer(hidden))))
            hidden = update if index == 0 else (hidden + update) * (2.0 ** -0.5)
        raw = self.output(hidden)
        # Match the exact bounds applied by the shared TypeScript decoder.
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
    basis_span = support.latent_upper - support.latent_lower
    half_basis_span = basis_span / 2.0
    basis_center = (support.latent_lower + support.latent_upper) / 2.0
    first_fraction = torch.sigmoid(raw[..., 0])
    latent_span = support.latent_upper - support.latent_lower
    c1 = support.latent_lower + latent_span * first_fraction
    second_fraction = torch.sigmoid(raw[..., 1])
    c2 = c1 + (support.latent_upper - c1) * second_fraction
    slope_scale = 1.0 / basis_span
    precision_scale = 1.0 / (half_basis_span * half_basis_span)
    base = raw[..., 2] * slope_scale
    precision = raw[..., 3] * precision_scale
    beta_c1 = raw[..., 4] * slope_scale
    beta_c2 = raw[..., 5] * slope_scale
    buy_slope_at_zero = support.friction / (1.0 - support.friction)
    sell_slope_at_zero = support.friction
    beta_x_value = -(buy_slope_at_zero + sell_slope_at_zero) / support.temperature
    beta_x = torch.full_like(base, beta_x_value)
    kappa_c1 = torch.full_like(base, 82.0 / basis_span)
    kappa_x = torch.full_like(base, 678.0 / basis_span)
    kappa_c2 = torch.full_like(base, 82.0 / basis_span)
    cutoff_source = raw if cutoff_raw is None else cutoff_raw
    if cutoff_source.shape != raw.shape:
        raise ValueError("cutoff source must match the raw parameter tensor")
    lower_fraction = torch.sigmoid(cutoff_source[..., 6])
    upper_fraction = torch.sigmoid(cutoff_source[..., 7])
    cutoff_lower = support.latent_lower + (-support.latent_lower) * lower_fraction
    cutoff_upper = support.latent_upper * upper_fraction
    cutoff_lower = torch.where(
        cutoff_source[..., 6] <= -13.999999, support.latent_lower, cutoff_lower
    )
    cutoff_lower = torch.where(cutoff_source[..., 6] >= 13.999999, 0.0, cutoff_lower)
    cutoff_upper = torch.where(cutoff_source[..., 7] <= -13.999999, 0.0, cutoff_upper)
    cutoff_upper = torch.where(
        cutoff_source[..., 7] >= 13.999999, support.latent_upper, cutoff_upper
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
        base * (action - support.latent_lower)
        - 0.5 * precision * (action - basis_center).square()
        + beta_c1 * scaled_softplus(action - c1, kappa_c1)
        + beta_x * scaled_softplus(action - current, kappa_x)
        + beta_c2 * scaled_softplus(action - c2, kappa_c2)
    )
    feasible = (
        (action >= support.visible_lower) & (action <= support.visible_upper)
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


def fitted_teacher_loss(
    predicted_raw: Tensor,
    target_raw: Tensor,
    actions: Tensor,
    current: Tensor,
    support: PolicySupport,
    parameter_scale: Tensor,
    weights: LossWeights,
    time_weighting: TimeWeighting,
    example_time_weights: Tensor | None = None,
) -> dict[str, Tensor]:
    """Distance-imbalance-weighted objective against the revised fitted policy."""
    state_count = current.shape[-1]
    predicted_rows = predicted_raw.float()[:, None, :].expand(-1, state_count, -1)
    target_rows = target_raw.float()[:, None, :].expand(-1, state_count, -1)
    # The hard cutoff coordinates are supervised by parameter MSE. Score
    # objectives use the teacher's feasible mask for both policies so a
    # temporarily too-narrow predicted cutoff cannot create infinite CE.
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

    if example_time_weights is None:
        time_weight = distance_imbalance_time_weights(
            target,
            actions,
            current,
            time_weighting,
        )
    else:
        if example_time_weights.ndim != 1 \
                or example_time_weights.shape[0] != predicted_raw.shape[0]:
            raise ValueError("precomputed time weights must match the training batch")
        time_weight = example_time_weights.float().clamp_min(time_weighting.minimum_weight)
    effective_sample_ratio = (
        time_weight.sum().square()
        / (time_weight.numel() * time_weight.square().sum()).clamp_min(1e-8)
    )
    normalized_weight = time_weight
    normalized_weight = normalized_weight / normalized_weight.mean().clamp_min(1e-8)
    denominator = normalized_weight.sum().clamp_min(1e-8)

    def weighted_mean(value: Tensor) -> Tensor:
        return (value * normalized_weight).sum() / denominator

    cross_entropy_per_example = -(target * predicted_log).sum(dim=-1).mean(dim=-1)
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
    predicted_mean = predicted_state_mean.mean(dim=-1)
    predicted_second = predicted_state_second.mean(dim=-1)
    target_mean = target_state_mean.mean(dim=-1)
    target_second = target_state_second.mean(dim=-1)

    global_predicted_mean = weighted_mean(predicted_mean)
    total_variance = (weighted_mean(predicted_second) - global_predicted_mean.square()).clamp_min(0)
    conditional_variance = weighted_mean(
        (predicted_state_second - predicted_state_mean.square()).clamp_min(0).mean(dim=-1)
    )
    state_mutual_information = (
        0.5 * torch.log((total_variance + 1e-8) / (conditional_variance + 1e-8))
        / math.log(actions.numel())
    ).clamp(0, 1)

    global_target_mean = weighted_mean(target_mean)
    oracle_variance = (weighted_mean(target_second) - global_target_mean.square()).clamp_min(0)
    strategy_variance = total_variance
    covariance = weighted_mean(target_mean * predicted_mean) - global_target_mean * global_predicted_mean
    correlation_squared = (
        covariance.square() / (oracle_variance * strategy_variance).clamp_min(1e-8)
    ).clamp(0, 1 - 1e-6)
    oracle_mutual_information = (
        -0.5 * torch.log1p(-correlation_squared) / math.log(actions.numel())
    ).clamp(0, 1)

    cross_entropy = weighted_mean(cross_entropy_per_example)
    probability_mse = weighted_mean(probability_mse_per_example)
    parameter_mse = weighted_mean(parameter_mse_per_example)
    excess_entropy = weighted_mean(excess_entropy_per_example)
    loss = (
        weights.cross_entropy * cross_entropy
        + weights.probability_mse * probability_mse
        + weights.parameter_mse * parameter_mse
        + weights.excess_entropy * excess_entropy
        - weights.state_mutual_information * state_mutual_information
        - weights.oracle_mutual_information * oracle_mutual_information
    )
    return {
        "loss": loss,
        "crossEntropy": cross_entropy,
        "probabilityMse": probability_mse,
        "parameterMse": parameter_mse,
        "excessEntropy": excess_entropy,
        "stateMutualInformation": state_mutual_information,
        "oracleMutualInformation": oracle_mutual_information,
        "targetEntropy": weighted_mean(target_entropy),
        "predictedEntropy": weighted_mean(predicted_entropy),
        "rawParameterMae": weighted_mean((predicted_raw.float() - target_raw.float()).abs().mean(dim=-1)),
        "distanceImbalanceWeight": time_weight.mean(),
        "timeWeightEffectiveSampleRatio": effective_sample_ratio,
        "timeWeightSum": time_weight.sum(),
        "timeWeightSquareSum": time_weight.square().sum(),
    }


def distance_imbalance_time_weights(
    target_probability: Tensor,
    actions: Tensor,
    current: Tensor,
    weighting: TimeWeighting,
) -> Tensor:
    """Return epsilon + |mean_x E[a-x|x] / (E[|a-x||x] + epsilon)|."""
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
    """Signed timestamp advice: mean_x E[a-x|x] / (E[|a-x||x] + epsilon)."""
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
    state_imbalance = expected_displacement / (
        expected_distance + distance_epsilon
    )
    return state_imbalance.mean(dim=-1)


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
            or weighting.reset_after_gap_steps < 1:
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
