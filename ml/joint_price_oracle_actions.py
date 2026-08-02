"""Execution-aligned losses and metrics for the joint price/oracle model.

The network predicts a *base* action-value distribution which is independent
of the bot's current exposure.  This module deliberately preserves that
contract.  Current exposure and transaction friction are applied analytically
after the network output, using the same rebalance factor as the TypeScript
runtime.

The helpers are standalone so a trainer can adopt execution-aligned labels,
losses, and checkpoint metrics without changing the model or ONNX output.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as functional


DEFAULT_ORACLE_FRICTION = 0.00175
DEFAULT_ORACLE_TEMPERATURE = 0.01
EXECUTION_POLICY_VERSION = 2
DEFAULT_EXECUTION_MAXIMUM_LEVERAGE = 100.0
DEFAULT_EXECUTION_MINIMUM_CONFIDENCE = 0.05
DEFAULT_EXECUTION_CONFIDENCE_EXPOSURE_POWER = 0.0
DEFAULT_EXECUTION_CONFIDENCE_LEVERAGE_FLOOR = 0.75


def resolve_execution_policy_config(
    configuration: dict[str, Any],
) -> dict[str, int | float]:
    """Validate the opt-in bot-execution policy while preserving native grid state."""
    if not isinstance(configuration, dict):
        raise ValueError("executionPolicy must be an object")
    allowed = {
        "version",
        "maximumLeverage",
        "minimumConfidence",
        "confidenceExposurePower",
        "confidenceLeverageFloor",
    }
    unknown = set(configuration) - allowed
    if unknown:
        raise ValueError(f"unknown executionPolicy settings: {sorted(unknown)}")
    version = configuration.get("version")
    if isinstance(version, bool) or version != EXECUTION_POLICY_VERSION:
        raise ValueError("executionPolicy.version must be 2")
    for key in (
        "maximumLeverage",
        "minimumConfidence",
        "confidenceExposurePower",
        "confidenceLeverageFloor",
    ):
        if isinstance(configuration.get(key), bool):
            raise ValueError(f"executionPolicy.{key} must be numeric")
    resolved: dict[str, int | float] = {
        "version": EXECUTION_POLICY_VERSION,
        "maximumLeverage": float(configuration.get(
            "maximumLeverage",
            DEFAULT_EXECUTION_MAXIMUM_LEVERAGE,
        )),
        "minimumConfidence": float(configuration.get(
            "minimumConfidence",
            DEFAULT_EXECUTION_MINIMUM_CONFIDENCE,
        )),
        "confidenceExposurePower": float(configuration.get(
            "confidenceExposurePower",
            DEFAULT_EXECUTION_CONFIDENCE_EXPOSURE_POWER,
        )),
        "confidenceLeverageFloor": float(configuration.get(
            "confidenceLeverageFloor",
            DEFAULT_EXECUTION_CONFIDENCE_LEVERAGE_FLOOR,
        )),
    }
    maximum_leverage = float(resolved["maximumLeverage"])
    minimum_confidence = float(resolved["minimumConfidence"])
    exposure_power = float(resolved["confidenceExposurePower"])
    leverage_floor = float(resolved["confidenceLeverageFloor"])
    if not math.isfinite(maximum_leverage) or maximum_leverage <= 0:
        raise ValueError(
            "executionPolicy.maximumLeverage must be finite and positive"
        )
    if not math.isfinite(minimum_confidence) \
            or not 0 <= minimum_confidence <= 1:
        raise ValueError(
            "executionPolicy.minimumConfidence must be in [0, 1]"
        )
    if not math.isfinite(exposure_power) or exposure_power < 0:
        raise ValueError(
            "executionPolicy.confidenceExposurePower must be finite and "
            "non-negative"
        )
    if not math.isfinite(leverage_floor) or not 0 <= leverage_floor <= 1:
        raise ValueError(
            "executionPolicy.confidenceLeverageFloor must be in [0, 1]"
        )
    return resolved


def execution_policy_native_scale(
    action_grid: np.ndarray,
    execution_policy: dict[str, Any],
) -> float:
    """Return execution/native scale used by the bot for one leverage ceiling."""
    grid = _numpy_action_grid(action_grid)
    policy = resolve_execution_policy_config(execution_policy)
    native_maximum = float(np.abs(grid).max())
    return min(1.0, float(policy["maximumLeverage"]) / native_maximum)


@dataclass(frozen=True)
class GreedyActionRollout:
    """One chronological greedy policy path on a NumPy action grid."""

    current_exposures: np.ndarray
    target_indices: np.ndarray
    target_exposures: np.ndarray
    switch_labels: np.ndarray
    signed_transition_labels: np.ndarray
    conditional_margins: np.ndarray
    conditional_entropies: np.ndarray
    confidences: np.ndarray

    @property
    def hold_labels(self) -> np.ndarray:
        return ~self.switch_labels


@dataclass(frozen=True)
class TensorGreedyActionRollout:
    """Tensor counterpart of :class:`GreedyActionRollout`."""

    current_exposures: Tensor
    target_indices: Tensor
    target_exposures: Tensor
    switch_labels: Tensor
    signed_transition_labels: Tensor
    conditional_margins: Tensor
    conditional_entropies: Tensor
    confidences: Tensor

    @property
    def hold_labels(self) -> Tensor:
        return ~self.switch_labels


@dataclass(frozen=True)
class SwitchBalancedActionLossWeights:
    """Weights for the execution-aligned composite objective."""

    hard_action: float = 1.0
    ranking: float = 1.0
    direction: float = 1.0
    conditional_kl: float = 0.1


def rebalance_equity_factor_numpy(
    current_exposure: np.ndarray | float,
    target_exposure: np.ndarray | float,
    friction: float = DEFAULT_ORACLE_FRICTION,
) -> np.ndarray:
    """Return the exact fee-aware equity factor used by the bot runtime."""
    _validate_friction(friction)
    current = np.asarray(current_exposure, dtype=np.float64)
    target = np.asarray(target_exposure, dtype=np.float64)
    difference = target - current
    buy_denominator = 1 - friction + friction * target
    sell_denominator = 1 - friction * target
    buy_factor = 1 - friction * difference / buy_denominator
    sell_factor = 1 - friction * (-difference) / sell_denominator
    return np.where(
        difference > 0,
        buy_factor,
        np.where(difference < 0, sell_factor, 1.0),
    )


def rebalance_equity_factor_tensor(
    current_exposure: Tensor | float,
    target_exposure: Tensor,
    friction: float = DEFAULT_ORACLE_FRICTION,
) -> Tensor:
    """Tensor implementation of :func:`rebalance_equity_factor_numpy`."""
    _validate_friction(friction)
    if not target_exposure.is_floating_point():
        raise ValueError("target exposure tensor must be floating point")
    current = torch.as_tensor(
        current_exposure,
        device=target_exposure.device,
        dtype=target_exposure.dtype,
    )
    difference = target_exposure - current
    buy_denominator = 1 - friction + friction * target_exposure
    sell_denominator = 1 - friction * target_exposure
    buy_factor = 1 - friction * difference / buy_denominator
    sell_factor = 1 - friction * (-difference) / sell_denominator
    return torch.where(
        difference > 0,
        buy_factor,
        torch.where(
            difference < 0,
            sell_factor,
            torch.ones_like(difference),
        ),
    )


def transition_logits_numpy(
    action_grid: np.ndarray,
    current_exposures: np.ndarray | float,
    *,
    friction: float = DEFAULT_ORACLE_FRICTION,
    temperature: float = DEFAULT_ORACLE_TEMPERATURE,
) -> np.ndarray:
    """Return ``log(R(current -> action)) / temperature``.

    The result has ``current_exposures.shape + (actions,)``.  A scalar current
    exposure therefore returns one action row.
    """
    grid = _numpy_action_grid(action_grid)
    _validate_temperature(temperature)
    current = np.asarray(current_exposures, dtype=np.float64)
    factors = rebalance_equity_factor_numpy(
        current[..., None],
        grid,
        friction,
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(factors > 0, np.log(factors) / temperature, -np.inf)
    return result


def transition_logits_tensor(
    action_grid: Tensor,
    current_exposures: Tensor | float,
    *,
    friction: float = DEFAULT_ORACLE_FRICTION,
    temperature: float = DEFAULT_ORACLE_TEMPERATURE,
) -> Tensor:
    """Tensor counterpart of :func:`transition_logits_numpy`."""
    grid = _tensor_action_grid(action_grid)
    _validate_temperature(temperature)
    current = torch.as_tensor(
        current_exposures,
        device=grid.device,
        dtype=grid.dtype,
    )
    factors = rebalance_equity_factor_tensor(
        current.unsqueeze(-1),
        grid,
        friction,
    )
    return torch.where(
        factors > 0,
        factors.clamp_min(torch.finfo(grid.dtype).tiny).log() / temperature,
        torch.full_like(factors, -torch.inf),
    )


def transition_conditioned_logits_numpy(
    base_logits: np.ndarray,
    action_grid: np.ndarray,
    current_exposures: np.ndarray | float,
    *,
    friction: float = DEFAULT_ORACLE_FRICTION,
    temperature: float = DEFAULT_ORACLE_TEMPERATURE,
) -> np.ndarray:
    """Apply exact transition costs to state-independent base logits."""
    logits = _numpy_action_values(base_logits, action_grid, "base logits")
    transition = transition_logits_numpy(
        action_grid,
        current_exposures,
        friction=friction,
        temperature=temperature,
    )
    try:
        return logits + transition
    except ValueError as error:
        raise ValueError(
            "current exposures do not broadcast over base-logit rows"
        ) from error


def transition_conditioned_logits_tensor(
    base_logits: Tensor,
    action_grid: Tensor,
    current_exposures: Tensor | float,
    *,
    friction: float = DEFAULT_ORACLE_FRICTION,
    temperature: float = DEFAULT_ORACLE_TEMPERATURE,
) -> Tensor:
    """Tensor counterpart of :func:`transition_conditioned_logits_numpy`."""
    _tensor_action_values(base_logits, action_grid, "base logits")
    transition = transition_logits_tensor(
        action_grid.to(device=base_logits.device, dtype=base_logits.dtype),
        torch.as_tensor(
            current_exposures,
            device=base_logits.device,
            dtype=base_logits.dtype,
        ),
        friction=friction,
        temperature=temperature,
    )
    try:
        return base_logits + transition
    except RuntimeError as error:
        raise ValueError(
            "current exposures do not broadcast over base-logit rows"
        ) from error


def transition_conditioned_probabilities_numpy(
    base_probabilities: np.ndarray,
    action_grid: np.ndarray,
    current_exposures: np.ndarray | float,
    *,
    friction: float = DEFAULT_ORACLE_FRICTION,
    temperature: float = DEFAULT_ORACLE_TEMPERATURE,
) -> np.ndarray:
    """Condition normalized or unnormalized non-negative base probabilities."""
    probabilities = _normalized_numpy_probabilities(
        base_probabilities,
        action_grid,
    )
    with np.errstate(divide="ignore"):
        base_logits = np.where(probabilities > 0, np.log(probabilities), -np.inf)
    return _numpy_softmax(transition_conditioned_logits_numpy(
        base_logits,
        action_grid,
        current_exposures,
        friction=friction,
        temperature=temperature,
    ))


def transition_conditioned_probabilities_tensor(
    base_probabilities: Tensor,
    action_grid: Tensor,
    current_exposures: Tensor | float,
    *,
    friction: float = DEFAULT_ORACLE_FRICTION,
    temperature: float = DEFAULT_ORACLE_TEMPERATURE,
) -> Tensor:
    """Tensor counterpart of :func:`transition_conditioned_probabilities_numpy`."""
    probabilities = _normalized_tensor_probabilities(
        base_probabilities,
        action_grid,
    )
    base_logits = torch.where(
        probabilities > 0,
        probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny).log(),
        torch.full_like(probabilities, -torch.inf),
    )
    return torch.softmax(transition_conditioned_logits_tensor(
        base_logits,
        action_grid,
        current_exposures,
        friction=friction,
        temperature=temperature,
    ), dim=-1)


def greedy_teacher_rollout_numpy(
    target_probabilities: np.ndarray,
    action_grid: np.ndarray,
    *,
    initial_exposure: float = 0.0,
    reset_mask: np.ndarray | None = None,
    friction: float = DEFAULT_ORACLE_FRICTION,
    temperature: float = DEFAULT_ORACLE_TEMPERATURE,
    switch_threshold: float | None = None,
    execution_policy: dict[str, Any] | None = None,
) -> GreedyActionRollout:
    """Roll target distributions forward using each prior greedy target.

    ``reset_mask[t]`` resets the current exposure immediately before row ``t``.
    This prevents separate market-history segments from leaking policy state.
    """
    probabilities = _normalized_numpy_probabilities(
        target_probabilities,
        action_grid,
    )
    with np.errstate(divide="ignore"):
        logits = np.where(probabilities > 0, np.log(probabilities), -np.inf)
    feasible_counts = (probabilities > 0).sum(axis=-1)
    return _greedy_logits_rollout_numpy(
        logits,
        action_grid,
        initial_exposure=initial_exposure,
        reset_mask=reset_mask,
        friction=friction,
        temperature=temperature,
        switch_threshold=switch_threshold,
        feasible_counts=feasible_counts,
        execution_policy=execution_policy,
    )


def greedy_base_logit_rollout_numpy(
    base_logits: np.ndarray,
    action_grid: np.ndarray,
    *,
    initial_exposure: float = 0.0,
    reset_mask: np.ndarray | None = None,
    friction: float = DEFAULT_ORACLE_FRICTION,
    temperature: float = DEFAULT_ORACLE_TEMPERATURE,
    switch_threshold: float | None = None,
    execution_policy: dict[str, Any] | None = None,
) -> GreedyActionRollout:
    """Roll model base logits forward without changing their base semantics."""
    logits = _numpy_action_values(base_logits, action_grid, "base logits")
    return _greedy_logits_rollout_numpy(
        logits,
        action_grid,
        initial_exposure=initial_exposure,
        reset_mask=reset_mask,
        friction=friction,
        temperature=temperature,
        switch_threshold=switch_threshold,
        feasible_counts=np.isfinite(logits).sum(axis=-1),
        execution_policy=execution_policy,
    )


def teacher_actions_at_current_exposures_numpy(
    target_probabilities: np.ndarray,
    action_grid: np.ndarray,
    current_exposures: np.ndarray,
    *,
    friction: float = DEFAULT_ORACLE_FRICTION,
    temperature: float = DEFAULT_ORACLE_TEMPERATURE,
    switch_threshold: float | None = None,
    execution_policy: dict[str, Any] | None = None,
) -> GreedyActionRollout:
    """Evaluate teacher decisions at externally supplied native-grid states.

    This is the hook used with simulator traces: every timestamp is conditioned
    on its exact marked exposure instead of pretending that the previous capped
    target remains the next current exposure.
    """
    probabilities = _normalized_numpy_probabilities(
        target_probabilities,
        action_grid,
    )
    if probabilities.ndim != 2:
        raise ValueError("teacher action rows must have [time, action] shape")
    grid = _numpy_action_grid(action_grid)
    current = np.asarray(current_exposures, dtype=np.float64)
    if current.shape != (probabilities.shape[0],) \
            or not np.isfinite(current).all():
        raise ValueError(
            "exact teacher current exposures must contain one finite row per "
            "target"
        )
    threshold = _switch_threshold_numpy(grid, switch_threshold)
    policy = (
        resolve_execution_policy_config(execution_policy)
        if execution_policy is not None
        else None
    )
    with np.errstate(divide="ignore"):
        base_logits = np.where(
            probabilities > 0,
            np.log(probabilities),
            -np.inf,
        )
    conditioned = transition_conditioned_logits_numpy(
        base_logits,
        grid,
        current,
        friction=friction,
        temperature=temperature,
    )
    target_indices = conditioned.argmax(axis=-1)
    target = grid[target_indices].astype(np.float64, copy=True)
    conditioned_probabilities = _numpy_softmax(conditioned)
    positive = conditioned_probabilities > 0
    entropy = -np.where(
        positive,
        conditioned_probabilities * np.log(
            np.where(positive, conditioned_probabilities, 1.0)
        ),
        0.0,
    ).sum(axis=-1)
    feasible_counts = (probabilities > 0).sum(axis=-1)
    maximum_entropy = np.log(np.maximum(1, feasible_counts))
    confidence = np.where(
        maximum_entropy > 0,
        np.clip(1 - entropy / np.maximum(maximum_entropy, 1e-300), 0, 1),
        1.0,
    )
    if policy is not None:
        scale = execution_policy_native_scale(grid, policy)
        native_leverage_ceiling = float(policy["maximumLeverage"]) / scale
        target *= np.power(
            confidence,
            float(policy["confidenceExposurePower"]),
        )
        leverage_fraction = (
            float(policy["confidenceLeverageFloor"])
            + (1 - float(policy["confidenceLeverageFloor"])) * confidence
        )
        cap = native_leverage_ceiling * leverage_fraction
        target = np.clip(target, -cap, cap)
    delta = target - current
    emitted = np.abs(delta) >= threshold
    if policy is not None:
        emitted &= confidence >= float(policy["minimumConfidence"])
        target = np.where(emitted, target, current)
        delta = target - current
    switches = np.abs(delta) >= threshold
    directions = np.where(
        delta <= -threshold,
        -1,
        np.where(delta >= threshold, 1, 0),
    ).astype(np.int8)
    margins = np.empty(probabilities.shape[0], dtype=np.float64)
    for row_index, row in enumerate(conditioned):
        finite = row[np.isfinite(row)]
        margins[row_index] = (
            float(np.partition(finite, -2)[-1] - np.partition(finite, -2)[-2])
            if finite.size > 1 else math.inf
        )
    return GreedyActionRollout(
        current_exposures=current.copy(),
        target_indices=target_indices.astype(np.int64, copy=False),
        target_exposures=target,
        switch_labels=switches,
        signed_transition_labels=directions,
        conditional_margins=margins,
        conditional_entropies=entropy,
        confidences=confidence,
    )


def greedy_teacher_rollout_tensor(
    target_probabilities: Tensor,
    action_grid: Tensor,
    *,
    initial_exposure: float = 0.0,
    reset_mask: Tensor | None = None,
    friction: float = DEFAULT_ORACLE_FRICTION,
    temperature: float = DEFAULT_ORACLE_TEMPERATURE,
    switch_threshold: float | None = None,
    execution_policy: dict[str, Any] | None = None,
) -> TensorGreedyActionRollout:
    """Tensor teacher rollout intended for offline label construction."""
    probabilities = _normalized_tensor_probabilities(
        target_probabilities,
        action_grid,
    ).detach()
    base_logits = torch.where(
        probabilities > 0,
        probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny).log(),
        torch.full_like(probabilities, -torch.inf),
    )
    feasible_counts = (probabilities > 0).sum(dim=-1)
    return _greedy_logits_rollout_tensor(
        base_logits,
        action_grid,
        initial_exposure=initial_exposure,
        reset_mask=reset_mask,
        friction=friction,
        temperature=temperature,
        switch_threshold=switch_threshold,
        feasible_counts=feasible_counts,
        execution_policy=execution_policy,
    )


def greedy_base_logit_rollout_tensor(
    base_logits: Tensor,
    action_grid: Tensor,
    *,
    initial_exposure: float = 0.0,
    reset_mask: Tensor | None = None,
    friction: float = DEFAULT_ORACLE_FRICTION,
    temperature: float = DEFAULT_ORACLE_TEMPERATURE,
    switch_threshold: float | None = None,
    execution_policy: dict[str, Any] | None = None,
) -> TensorGreedyActionRollout:
    """Roll model logits forward without differentiating through state.

    This is intended for scheduled/self-conditioned action losses.  The
    network remains state-independent: its logits are detached only for the
    discrete rollout, while callers apply their differentiable loss to the
    original logits at the resulting visited exposures.
    """
    logits = _tensor_action_values(
        base_logits,
        action_grid,
        "base logits",
    ).detach()
    return _greedy_logits_rollout_tensor(
        logits,
        action_grid,
        initial_exposure=initial_exposure,
        reset_mask=reset_mask,
        friction=friction,
        temperature=temperature,
        switch_threshold=switch_threshold,
        feasible_counts=torch.isfinite(logits).sum(dim=-1),
        execution_policy=execution_policy,
    )


def self_conditioned_current_exposures_tensor(
    base_logits: Tensor,
    action_grid: Tensor,
    *,
    initial_exposure: float = 0.0,
    friction: float = DEFAULT_ORACLE_FRICTION,
    temperature: float = DEFAULT_ORACLE_TEMPERATURE,
    execution_policy: dict[str, Any] | None = None,
) -> Tensor:
    """Return only visited model states for an efficient training rollout."""
    logits = _tensor_action_values(
        base_logits,
        action_grid,
        "base logits",
    ).detach()
    if logits.ndim != 2:
        raise ValueError("self-conditioned rollout expects [time, action] rows")
    if not math.isfinite(initial_exposure):
        raise ValueError("initial exposure must be finite")
    _validate_friction(friction)
    _validate_temperature(temperature)
    grid = action_grid.to(device=logits.device, dtype=logits.dtype)
    policy = (
        resolve_execution_policy_config(execution_policy)
        if execution_policy is not None
        else None
    )
    threshold = _switch_threshold_tensor(grid, None)
    execution_scale = (
        torch.minimum(
            grid.new_tensor(1.0),
            grid.new_tensor(float(policy["maximumLeverage"]))
            / grid.abs().max(),
        )
        if policy is not None
        else grid.new_tensor(1.0)
    )
    native_leverage_ceiling = (
        grid.new_tensor(float(policy["maximumLeverage"])) / execution_scale
        if policy is not None
        else grid.new_tensor(math.inf)
    )
    current = grid.new_tensor(initial_exposure)
    visited: list[Tensor] = []
    with torch.no_grad():
        for row in logits.unbind(0):
            visited.append(current)
            factors = rebalance_equity_factor_tensor(
                current,
                grid,
                friction,
            )
            transition = torch.where(
                factors > 0,
                factors.clamp_min(torch.finfo(grid.dtype).tiny).log()
                / temperature,
                torch.full_like(factors, -torch.inf),
            )
            conditioned = row + transition
            target = grid[conditioned.argmax()]
            if policy is not None:
                probability = torch.softmax(conditioned, dim=-1)
                entropy = -(probability * torch.where(
                    probability > 0,
                    probability.clamp_min(
                        torch.finfo(grid.dtype).tiny
                    ).log(),
                    torch.zeros_like(probability),
                )).sum()
                feasible_count = torch.isfinite(row).sum().clamp_min(1)
                maximum_entropy = feasible_count.float().log()
                confidence = torch.where(
                    maximum_entropy > 0,
                    (1 - entropy.float() / maximum_entropy).clamp(0, 1),
                    torch.ones_like(maximum_entropy),
                ).to(dtype=grid.dtype)
                target = target * confidence.pow(
                    float(policy["confidenceExposurePower"])
                )
                leverage_fraction = (
                    float(policy["confidenceLeverageFloor"])
                    + (
                        1 - float(policy["confidenceLeverageFloor"])
                    ) * confidence
                )
                cap = native_leverage_ceiling * leverage_fraction
                target = target.clamp(min=-cap, max=cap)
                emitted = (
                    confidence >= float(policy["minimumConfidence"])
                ) & ((target - current).abs() >= threshold)
                target = torch.where(emitted, target, current)
            current = target
    return torch.stack(visited)


def switch_balanced_weights_tensor(
    switch_labels: Tensor,
    *,
    target_switch_fraction: float = 0.5,
    source_switch_fraction: float | None = None,
) -> Tensor:
    """Return switch/hold weights for dynamic or static class balancing.

    When ``source_switch_fraction`` is omitted, weights are estimated from the
    current batch for compatibility with the original objective.  Supplying a
    train-split fraction produces fixed weights whose expected dataset-wide
    mean is one, avoiding class-weight changes in small or single-class
    batches.
    """
    if switch_labels.ndim != 1:
        raise ValueError("switch labels must be one-dimensional")
    if not (0 <= target_switch_fraction <= 1) \
            or not math.isfinite(target_switch_fraction):
        raise ValueError("target switch fraction must be in [0, 1]")
    labels = switch_labels.bool()
    count = labels.numel()
    if count < 1:
        raise ValueError("switch labels cannot be empty")
    if source_switch_fraction is not None:
        if not math.isfinite(source_switch_fraction) \
                or not 0 < source_switch_fraction < 1:
            raise ValueError("source switch fraction must be in (0, 1)")
        switch_weight = target_switch_fraction / source_switch_fraction
        hold_weight = (
            (1 - target_switch_fraction) / (1 - source_switch_fraction)
        )
        return torch.where(
            labels,
            torch.full(
                (count,),
                switch_weight,
                device=labels.device,
                dtype=torch.float32,
            ),
            torch.full(
                (count,),
                hold_weight,
                device=labels.device,
                dtype=torch.float32,
            ),
        )
    switches = int(labels.sum().detach())
    holds = count - switches
    if switches == 0 or holds == 0:
        return torch.ones(
            count,
            device=labels.device,
            dtype=torch.float32,
        )
    switch_weight = target_switch_fraction * count / switches
    hold_weight = (1 - target_switch_fraction) * count / holds
    return torch.where(
        labels,
        torch.full(
            (count,),
            switch_weight,
            device=labels.device,
            dtype=torch.float32,
        ),
        torch.full(
            (count,),
            hold_weight,
            device=labels.device,
            dtype=torch.float32,
        ),
    )


def switch_balanced_action_objective(
    predicted_base_logits: Tensor,
    target_base_probabilities: Tensor,
    action_grid: Tensor,
    current_exposures: Tensor,
    *,
    weights: SwitchBalancedActionLossWeights = SwitchBalancedActionLossWeights(),
    target_switch_fraction: float = 0.5,
    source_switch_fraction: float | None = None,
    ranking_margin: float = 0.1,
    switch_threshold: float | None = None,
    friction: float = DEFAULT_ORACLE_FRICTION,
    temperature: float = DEFAULT_ORACLE_TEMPERATURE,
    execution_policy: dict[str, Any] | None = None,
) -> dict[str, Tensor]:
    """Execution-aligned hard, ranking, direction, and distillation losses.

    Predicted logits remain state-independent.  The same deterministic
    transition row is added to predictions and targets for the supplied
    teacher-forced current exposure.
    """
    _tensor_action_values(predicted_base_logits, action_grid, "predicted logits")
    target = _normalized_tensor_probabilities(
        target_base_probabilities,
        action_grid,
    ).to(
        device=predicted_base_logits.device,
        dtype=predicted_base_logits.dtype,
    )
    if predicted_base_logits.ndim != 2 or target.ndim != 2:
        raise ValueError("action objective expects [example, action] tensors")
    if predicted_base_logits.shape != target.shape:
        raise ValueError("prediction and target action shapes must match")
    if current_exposures.shape != (predicted_base_logits.shape[0],):
        raise ValueError("current exposures must contain one value per example")
    if not math.isfinite(ranking_margin) or ranking_margin < 0:
        raise ValueError("ranking margin must be finite and non-negative")
    for value in (
        weights.hard_action,
        weights.ranking,
        weights.direction,
        weights.conditional_kl,
    ):
        if not math.isfinite(value) or value < 0:
            raise ValueError("action loss weights must be finite and non-negative")

    grid = action_grid.to(
        device=predicted_base_logits.device,
        dtype=predicted_base_logits.dtype,
    )
    current = current_exposures.to(
        device=predicted_base_logits.device,
        dtype=predicted_base_logits.dtype,
    )
    threshold = _switch_threshold_tensor(grid, switch_threshold)
    transition = transition_logits_tensor(
        grid,
        current,
        friction=friction,
        temperature=temperature,
    )
    target_base_logits = torch.where(
        target > 0,
        target.clamp_min(torch.finfo(target.dtype).tiny).log(),
        torch.full_like(target, -torch.inf),
    )
    target_conditioned_logits = target_base_logits + transition
    predicted_conditioned_logits = predicted_base_logits + transition
    target_indices = target_conditioned_logits.argmax(dim=-1)
    target_exposures = grid[target_indices]
    target_conditioned = torch.softmax(target_conditioned_logits, dim=-1)
    policy = (
        resolve_execution_policy_config(execution_policy)
        if execution_policy is not None
        else None
    )
    if policy is not None:
        target_entropy = -(target_conditioned * torch.where(
            target_conditioned > 0,
            target_conditioned.clamp_min(
                torch.finfo(target.dtype).tiny
            ).log(),
            torch.zeros_like(target_conditioned),
        )).sum(dim=-1)
        feasible_counts = (target > 0).sum(dim=-1).clamp_min(1)
        maximum_entropy = feasible_counts.float().log()
        confidence = torch.where(
            maximum_entropy > 0,
            (1 - target_entropy.float() / maximum_entropy).clamp(0, 1),
            torch.ones_like(maximum_entropy),
        ).to(dtype=grid.dtype)
        execution_scale = torch.minimum(
            grid.new_tensor(1.0),
            grid.new_tensor(float(policy["maximumLeverage"]))
            / grid.abs().max(),
        )
        native_leverage_ceiling = (
            grid.new_tensor(float(policy["maximumLeverage"]))
            / execution_scale
        )
        target_exposures = target_exposures * confidence.pow(
            float(policy["confidenceExposurePower"])
        )
        leverage_fraction = (
            float(policy["confidenceLeverageFloor"])
            + (1 - float(policy["confidenceLeverageFloor"])) * confidence
        )
        cap = native_leverage_ceiling * leverage_fraction
        target_exposures = target_exposures.clamp(min=-cap, max=cap)
        emitted = (
            confidence >= float(policy["minimumConfidence"])
        ) & ((target_exposures - current).abs() >= threshold)
        target_exposures = torch.where(
            emitted,
            target_exposures,
            current,
        )
    differences = target_exposures - current
    switch_labels = differences.abs() >= threshold
    direction_labels = torch.where(
        differences <= -threshold,
        torch.zeros_like(target_indices),
        torch.where(
            differences >= threshold,
            torch.full_like(target_indices, 2),
            torch.ones_like(target_indices),
        ),
    )
    example_weights = switch_balanced_weights_tensor(
        switch_labels,
        target_switch_fraction=target_switch_fraction,
        source_switch_fraction=source_switch_fraction,
    ).to(dtype=predicted_base_logits.dtype)

    hard_per_example = functional.cross_entropy(
        predicted_conditioned_logits,
        target_indices,
        reduction="none",
    )
    chosen_logits = predicted_conditioned_logits.gather(
        1,
        target_indices[:, None],
    ).squeeze(1)
    wrong_logits = predicted_conditioned_logits.masked_fill(
        functional.one_hot(
            target_indices,
            num_classes=predicted_base_logits.shape[1],
        ).bool(),
        -torch.inf,
    ).max(dim=-1).values
    ranking_per_example = functional.softplus(
        ranking_margin - (chosen_logits - wrong_logits)
    )

    action_differences = grid.unsqueeze(0) - current.unsqueeze(1)
    direction_masks = (
        action_differences <= -threshold,
        action_differences.abs() < threshold,
        action_differences >= threshold,
    )
    direction_logits = torch.stack(tuple(
        torch.logsumexp(
            predicted_conditioned_logits.masked_fill(~mask, -torch.inf),
            dim=-1,
        )
        for mask in direction_masks
    ), dim=-1)
    direction_per_example = functional.cross_entropy(
        direction_logits,
        direction_labels,
        reduction="none",
    )

    target_conditioned_log = torch.where(
        target_conditioned > 0,
        target_conditioned.clamp_min(torch.finfo(target.dtype).tiny).log(),
        torch.zeros_like(target_conditioned),
    )
    predicted_conditioned_log = torch.log_softmax(
        predicted_conditioned_logits,
        dim=-1,
    )
    conditional_kl_per_example = (
        target_conditioned
        * (target_conditioned_log - predicted_conditioned_log)
    ).sum(dim=-1)

    normalize_actual_weights = source_switch_fraction is None
    hard_action = _weighted_mean(
        hard_per_example,
        example_weights,
        normalize_actual_weights=normalize_actual_weights,
    )
    ranking = _weighted_mean(
        ranking_per_example,
        example_weights,
        normalize_actual_weights=normalize_actual_weights,
    )
    direction = _weighted_mean(
        direction_per_example,
        example_weights,
        normalize_actual_weights=normalize_actual_weights,
    )
    conditional_kl = _weighted_mean(
        conditional_kl_per_example,
        example_weights,
        normalize_actual_weights=normalize_actual_weights,
    )
    total = (
        weights.hard_action * hard_action
        + weights.ranking * ranking
        + weights.direction * direction
        + weights.conditional_kl * conditional_kl
    )
    predicted_indices = predicted_conditioned_logits.argmax(dim=-1)
    return {
        "loss": total,
        "hardActionCrossEntropy": hard_action,
        "rankingLoss": ranking,
        "directionCrossEntropy": direction,
        "conditionalKlDivergence": conditional_kl,
        "hardActionAccuracy": (predicted_indices == target_indices).float().mean(),
        "switchRate": switch_labels.float().mean(),
        "meanExampleWeight": example_weights.mean(),
    }


def actionable_policy_metrics_numpy(
    predicted_base_logits: np.ndarray,
    target_base_probabilities: np.ndarray,
    action_grid: np.ndarray,
    *,
    initial_exposure: float = 0.0,
    reset_mask: np.ndarray | None = None,
    friction: float = DEFAULT_ORACLE_FRICTION,
    temperature: float = DEFAULT_ORACLE_TEMPERATURE,
    switch_threshold: float | None = None,
    execution_policy: dict[str, Any] | None = None,
) -> dict[str, int | float]:
    """Return transition, path, and teacher-value metrics.

    Transition counts compare independent chronological teacher and predicted
    rollouts.  ``mode*`` fields condition both policies on the teacher current
    exposure and therefore isolate one-step ranking quality.  ``path*`` fields
    expose compounding state errors.
    """
    predicted = _numpy_action_values(
        predicted_base_logits,
        action_grid,
        "predicted logits",
    )
    target = _normalized_numpy_probabilities(
        target_base_probabilities,
        action_grid,
    )
    if predicted.ndim != 2 or target.ndim != 2 \
            or predicted.shape != target.shape:
        raise ValueError("policy metrics expect matching [time, action] rows")
    grid = _numpy_action_grid(action_grid)
    threshold = _switch_threshold_numpy(grid, switch_threshold)
    teacher_rollout = greedy_teacher_rollout_numpy(
        target,
        grid,
        initial_exposure=initial_exposure,
        reset_mask=reset_mask,
        friction=friction,
        temperature=temperature,
        switch_threshold=threshold,
        execution_policy=execution_policy,
    )
    predicted_rollout = greedy_base_logit_rollout_numpy(
        predicted,
        grid,
        initial_exposure=initial_exposure,
        reset_mask=reset_mask,
        friction=friction,
        temperature=temperature,
        switch_threshold=threshold,
        execution_policy=execution_policy,
    )

    with np.errstate(divide="ignore"):
        target_base_logits = np.where(target > 0, np.log(target), -np.inf)
    teacher_conditioned = transition_conditioned_logits_numpy(
        target_base_logits,
        grid,
        teacher_rollout.current_exposures,
        friction=friction,
        temperature=temperature,
    )
    predicted_common_state = transition_conditioned_logits_numpy(
        predicted,
        grid,
        teacher_rollout.current_exposures,
        friction=friction,
        temperature=temperature,
    )
    teacher_indices = teacher_conditioned.argmax(axis=-1)
    predicted_common_indices = predicted_common_state.argmax(axis=-1)
    teacher_modes = grid[teacher_indices]
    predicted_common_modes = grid[predicted_common_indices]
    row_indices = np.arange(predicted.shape[0])
    regret = (
        teacher_conditioned[row_indices, teacher_indices]
        - teacher_conditioned[row_indices, predicted_common_indices]
    ).clip(min=0)

    target_switch = teacher_rollout.switch_labels
    predicted_switch = predicted_rollout.switch_labels
    transition_tp = int(np.logical_and(target_switch, predicted_switch).sum())
    signed_tp = int(np.logical_and.reduce((
        target_switch,
        predicted_switch,
        teacher_rollout.signed_transition_labels
        == predicted_rollout.signed_transition_labels,
    )).sum())
    exact_tp = int(np.logical_and.reduce((
        target_switch,
        predicted_switch,
        teacher_rollout.target_indices == predicted_rollout.target_indices,
    )).sum())
    target_switches = int(target_switch.sum())
    predicted_switches = int(predicted_switch.sum())
    transition = _precision_recall_f1(
        transition_tp,
        predicted_switches,
        target_switches,
    )
    signed = _precision_recall_f1(
        signed_tp,
        predicted_switches,
        target_switches,
    )
    exact = _precision_recall_f1(
        exact_tp,
        predicted_switches,
        target_switches,
    )
    mode_error = np.abs(predicted_common_modes - teacher_modes)
    path_error = np.abs(
        predicted_rollout.target_exposures
        - teacher_rollout.target_exposures
    )
    target_turnover = float(np.abs(
        teacher_rollout.target_exposures
        - teacher_rollout.current_exposures
    ).sum())
    predicted_turnover = float(np.abs(
        predicted_rollout.target_exposures
        - predicted_rollout.current_exposures
    ).sum())
    switch_regret = regret[target_switch]
    return {
        "decisions": int(predicted.shape[0]),
        "targetSwitches": target_switches,
        "predictedSwitches": predicted_switches,
        "transitionTruePositive": transition_tp,
        "transitionFalsePositive": predicted_switches - transition_tp,
        "transitionFalseNegative": target_switches - transition_tp,
        "transitionPrecision": transition[0],
        "transitionRecall": transition[1],
        "transitionF1": transition[2],
        "signedTransitionTruePositive": signed_tp,
        "signedTransitionPrecision": signed[0],
        "signedTransitionRecall": signed[1],
        "signedTransitionF1": signed[2],
        "exactTransitionTruePositive": exact_tp,
        "exactTransitionPrecision": exact[0],
        "exactTransitionRecall": exact[1],
        "exactTransitionF1": exact[2],
        "modeAccuracy": float((predicted_common_indices == teacher_indices).mean()),
        "modeMeanAbsoluteError": float(mode_error.mean()),
        "switchModeMeanAbsoluteError": (
            float(mode_error[target_switch].mean()) if target_switches else 0.0
        ),
        "pathMeanAbsoluteError": float(path_error.mean()),
        "pathDirectionalAgreement": float((
            np.sign(predicted_rollout.target_exposures)
            == np.sign(teacher_rollout.target_exposures)
        ).mean()),
        "meanConditionalRegret": float(regret.mean()),
        "switchConditionalRegret": (
            float(switch_regret.mean()) if target_switches else 0.0
        ),
        "targetTurnover": target_turnover,
        "predictedTurnover": predicted_turnover,
        "turnoverRatio": (
            predicted_turnover / target_turnover
            if target_turnover > 0 else 0.0 if predicted_turnover == 0 else math.inf
        ),
    }


def exact_state_actionable_policy_metrics_numpy(
    predicted_base_logits: np.ndarray,
    target_base_probabilities: np.ndarray,
    action_grid: np.ndarray,
    exact_current_exposures: np.ndarray,
    *,
    friction: float = DEFAULT_ORACLE_FRICTION,
    temperature: float = DEFAULT_ORACLE_TEMPERATURE,
    switch_threshold: float | None = None,
    execution_policy: dict[str, Any] | None = None,
) -> dict[str, int | float]:
    """Compare model and teacher executable decisions at identical state.

    Unlike a chronological surrogate rollout, neither side is allowed to
    carry its own predicted target into the next row.  Both are independently
    conditioned on the simulator's recorded marked exposure for that exact
    timestamp.  This isolates the causal model's one-step policy quality from
    unavoidable marked-position drift.
    """
    predicted = _numpy_action_values(
        predicted_base_logits,
        action_grid,
        "predicted logits",
    )
    target = _normalized_numpy_probabilities(
        target_base_probabilities,
        action_grid,
    )
    current = np.asarray(exact_current_exposures, dtype=np.float64)
    if predicted.ndim != 2 or target.shape != predicted.shape \
            or current.shape != (predicted.shape[0],) \
            or not np.isfinite(current).all():
        raise ValueError(
            "exact-state metrics require matching finite policy rows and "
            "one current exposure per row"
        )
    grid = _numpy_action_grid(action_grid)
    if execution_policy is None:
        raise ValueError("exact-state metrics require an execution policy")
    policy = resolve_execution_policy_config(execution_policy)
    threshold = _switch_threshold_numpy(grid, switch_threshold)
    teacher = teacher_actions_at_current_exposures_numpy(
        target,
        grid,
        current,
        friction=friction,
        temperature=temperature,
        switch_threshold=threshold,
        execution_policy=policy,
    )
    predicted_actions = teacher_actions_at_current_exposures_numpy(
        _numpy_softmax(predicted),
        grid,
        current,
        friction=friction,
        temperature=temperature,
        switch_threshold=threshold,
        execution_policy=policy,
    )

    target_switch = teacher.switch_labels
    predicted_switch = predicted_actions.switch_labels
    target_switches = int(target_switch.sum())
    predicted_switches = int(predicted_switch.sum())
    transition_tp = int(np.logical_and(target_switch, predicted_switch).sum())
    signed_tp = int(np.logical_and.reduce((
        target_switch,
        predicted_switch,
        teacher.signed_transition_labels
        == predicted_actions.signed_transition_labels,
    )).sum())
    exact_tp = int(np.logical_and.reduce((
        target_switch,
        predicted_switch,
        teacher.target_indices == predicted_actions.target_indices,
    )).sum())
    transition = _precision_recall_f1(
        transition_tp,
        predicted_switches,
        target_switches,
    )
    signed = _precision_recall_f1(
        signed_tp,
        predicted_switches,
        target_switches,
    )
    exact = _precision_recall_f1(
        exact_tp,
        predicted_switches,
        target_switches,
    )

    target_delta = teacher.target_exposures - current
    predicted_delta = predicted_actions.target_exposures - current
    target_error = np.abs(
        predicted_actions.target_exposures - teacher.target_exposures
    )
    target_turnover = float(np.abs(target_delta).sum())
    predicted_turnover = float(np.abs(predicted_delta).sum())
    grid_span = float(grid[-1] - grid[0])
    normalized_target_error = min(1.0, float(target_error.mean()) / grid_span)
    turnover_relative_error = min(1.0, abs(
        predicted_turnover - target_turnover
    ) / max(target_turnover, grid_span))
    directional_agreement = float((
        np.sign(predicted_delta) == np.sign(target_delta)
    ).mean())
    exact_state_score = (
        0.60 * (1 - float(signed[2]))
        + 0.15 * (1 - directional_agreement)
        + 0.15 * normalized_target_error
        + 0.10 * turnover_relative_error
    )
    if not math.isfinite(exact_state_score):
        raise ValueError("exact-state score is non-finite")
    return {
        "decisions": int(predicted.shape[0]),
        "targetSwitches": target_switches,
        "predictedSwitches": predicted_switches,
        "transitionTruePositive": transition_tp,
        "transitionPrecision": transition[0],
        "transitionRecall": transition[1],
        "transitionF1": transition[2],
        "signedTransitionTruePositive": signed_tp,
        "signedTransitionPrecision": signed[0],
        "signedTransitionRecall": signed[1],
        "signedTransitionF1": signed[2],
        "exactTransitionTruePositive": exact_tp,
        "exactTransitionPrecision": exact[0],
        "exactTransitionRecall": exact[1],
        "exactTransitionF1": exact[2],
        "executableTargetMeanAbsoluteError": float(target_error.mean()),
        "executableTargetDirectionalAgreement": directional_agreement,
        "targetTurnover": target_turnover,
        "predictedTurnover": predicted_turnover,
        "turnoverRelativeError": turnover_relative_error,
        "exactStateScore": exact_state_score,
    }


def _greedy_logits_rollout_numpy(
    base_logits: np.ndarray,
    action_grid: np.ndarray,
    *,
    initial_exposure: float,
    reset_mask: np.ndarray | None,
    friction: float,
    temperature: float,
    switch_threshold: float | None,
    feasible_counts: np.ndarray,
    execution_policy: dict[str, Any] | None,
) -> GreedyActionRollout:
    logits = _numpy_action_values(base_logits, action_grid, "base logits")
    if logits.ndim != 2:
        raise ValueError("greedy rollout expects [time, action] rows")
    if not math.isfinite(initial_exposure):
        raise ValueError("initial exposure must be finite")
    grid = _numpy_action_grid(action_grid)
    resets = _numpy_reset_mask(reset_mask, logits.shape[0])
    threshold = _switch_threshold_numpy(grid, switch_threshold)
    policy = (
        resolve_execution_policy_config(execution_policy)
        if execution_policy is not None
        else None
    )
    execution_scale = (
        execution_policy_native_scale(grid, policy)
        if policy is not None
        else 1.0
    )
    native_leverage_ceiling = (
        float(policy["maximumLeverage"]) / execution_scale
        if policy is not None
        else math.inf
    )
    count = logits.shape[0]
    current_values = np.empty(count, dtype=np.float64)
    target_indices = np.empty(count, dtype=np.int64)
    target_values = np.empty(count, dtype=np.float64)
    switches = np.empty(count, dtype=np.bool_)
    directions = np.empty(count, dtype=np.int8)
    margins = np.empty(count, dtype=np.float64)
    entropies = np.empty(count, dtype=np.float64)
    confidences = np.empty(count, dtype=np.float64)
    current = float(initial_exposure)
    for time in range(count):
        if resets[time]:
            current = float(initial_exposure)
        conditioned = logits[time] + transition_logits_numpy(
            grid,
            current,
            friction=friction,
            temperature=temperature,
        )
        if not np.isfinite(conditioned).any():
            raise ValueError(f"conditioned policy row {time} has no feasible action")
        index = int(np.argmax(conditioned))
        modal_target = float(grid[index])
        probability = _numpy_softmax(conditioned)
        positive = probability > 0
        entropy = float(-(
            probability[positive] * np.log(probability[positive])
        ).sum())
        feasible = max(1, int(feasible_counts[time]))
        maximum_entropy = math.log(feasible)
        confidence = (
            min(1.0, max(0.0, 1 - entropy / maximum_entropy))
            if maximum_entropy > 0 else 1.0
        )
        target = modal_target
        if policy is not None:
            target *= confidence ** float(
                policy["confidenceExposurePower"]
            )
            leverage_fraction = (
                float(policy["confidenceLeverageFloor"])
                + (1 - float(policy["confidenceLeverageFloor"]))
                * confidence
            )
            cap = native_leverage_ceiling * leverage_fraction
            target = min(cap, max(-cap, target))
        delta = target - current
        emitted = (
            policy is None
            or (
                confidence >= float(policy["minimumConfidence"])
                and abs(delta) >= threshold
            )
        )
        if policy is not None and not emitted:
            target = current
            delta = 0.0
        finite = conditioned[np.isfinite(conditioned)]
        margin = (
            float(np.partition(finite, -2)[-1] - np.partition(finite, -2)[-2])
            if finite.size > 1 else math.inf
        )
        current_values[time] = current
        target_indices[time] = index
        target_values[time] = target
        switches[time] = abs(delta) >= threshold
        directions[time] = -1 if delta <= -threshold else 1 if delta >= threshold else 0
        margins[time] = margin
        entropies[time] = entropy
        confidences[time] = confidence
        current = target
    return GreedyActionRollout(
        current_exposures=current_values,
        target_indices=target_indices,
        target_exposures=target_values,
        switch_labels=switches,
        signed_transition_labels=directions,
        conditional_margins=margins,
        conditional_entropies=entropies,
        confidences=confidences,
    )


def _greedy_logits_rollout_tensor(
    base_logits: Tensor,
    action_grid: Tensor,
    *,
    initial_exposure: float,
    reset_mask: Tensor | None,
    friction: float,
    temperature: float,
    switch_threshold: float | None,
    feasible_counts: Tensor,
    execution_policy: dict[str, Any] | None,
) -> TensorGreedyActionRollout:
    _tensor_action_values(base_logits, action_grid, "base logits")
    if base_logits.ndim != 2:
        raise ValueError("greedy rollout expects [time, action] rows")
    if not math.isfinite(initial_exposure):
        raise ValueError("initial exposure must be finite")
    grid = action_grid.to(device=base_logits.device, dtype=base_logits.dtype)
    threshold = _switch_threshold_tensor(grid, switch_threshold)
    policy = (
        resolve_execution_policy_config(execution_policy)
        if execution_policy is not None
        else None
    )
    execution_scale = (
        torch.minimum(
            grid.new_tensor(1.0),
            grid.new_tensor(float(policy["maximumLeverage"]))
            / grid.abs().max(),
        )
        if policy is not None
        else grid.new_tensor(1.0)
    )
    native_leverage_ceiling = (
        grid.new_tensor(float(policy["maximumLeverage"])) / execution_scale
        if policy is not None
        else grid.new_tensor(math.inf)
    )
    count = base_logits.shape[0]
    resets = _tensor_reset_mask(reset_mask, count, base_logits.device)
    current_values = torch.empty(count, device=grid.device, dtype=grid.dtype)
    target_indices = torch.empty(count, device=grid.device, dtype=torch.long)
    target_values = torch.empty(count, device=grid.device, dtype=grid.dtype)
    switches = torch.empty(count, device=grid.device, dtype=torch.bool)
    directions = torch.empty(count, device=grid.device, dtype=torch.int8)
    margins = torch.empty(count, device=grid.device, dtype=grid.dtype)
    entropies = torch.empty(count, device=grid.device, dtype=grid.dtype)
    confidences = torch.empty(count, device=grid.device, dtype=grid.dtype)
    current = grid.new_tensor(initial_exposure)
    with torch.no_grad():
        for time in range(count):
            if bool(resets[time]):
                current = grid.new_tensor(initial_exposure)
            conditioned = base_logits[time] + transition_logits_tensor(
                grid,
                current,
                friction=friction,
                temperature=temperature,
            )
            if not bool(torch.isfinite(conditioned).any()):
                raise ValueError(
                    f"conditioned policy row {time} has no feasible action"
                )
            index = conditioned.argmax()
            target = grid[index]
            probability = torch.softmax(conditioned, dim=-1)
            entropy = -(probability * torch.where(
                probability > 0,
                probability.clamp_min(torch.finfo(grid.dtype).tiny).log(),
                torch.zeros_like(probability),
            )).sum()
            maximum_entropy = feasible_counts[time].clamp_min(1).float().log()
            confidence = torch.where(
                maximum_entropy > 0,
                (1 - entropy.float() / maximum_entropy).clamp(0, 1),
                torch.ones_like(maximum_entropy),
            ).to(dtype=grid.dtype)
            if policy is not None:
                target = target * confidence.pow(
                    float(policy["confidenceExposurePower"])
                )
                leverage_fraction = (
                    float(policy["confidenceLeverageFloor"])
                    + (
                        1 - float(policy["confidenceLeverageFloor"])
                    ) * confidence
                )
                cap = native_leverage_ceiling * leverage_fraction
                target = target.clamp(min=-cap, max=cap)
            delta = target - current
            if policy is not None:
                emitted = (
                    confidence >= float(policy["minimumConfidence"])
                ) & (delta.abs() >= threshold)
                target = torch.where(emitted, target, current)
                delta = target - current
            finite = conditioned[torch.isfinite(conditioned)]
            margin = (
                torch.topk(finite, 2).values.diff().abs().squeeze(0)
                if finite.numel() > 1 else grid.new_tensor(math.inf)
            )
            current_values[time] = current
            target_indices[time] = index
            target_values[time] = target
            switches[time] = delta.abs() >= threshold
            directions[time] = torch.where(
                delta <= -threshold,
                delta.new_tensor(-1, dtype=torch.int8),
                torch.where(
                    delta >= threshold,
                    delta.new_tensor(1, dtype=torch.int8),
                    delta.new_tensor(0, dtype=torch.int8),
                ),
            )
            margins[time] = margin
            entropies[time] = entropy
            confidences[time] = confidence
            current = target
    return TensorGreedyActionRollout(
        current_exposures=current_values,
        target_indices=target_indices,
        target_exposures=target_values,
        switch_labels=switches,
        signed_transition_labels=directions,
        conditional_margins=margins,
        conditional_entropies=entropies,
        confidences=confidences,
    )


def _numpy_action_grid(action_grid: np.ndarray) -> np.ndarray:
    grid = np.asarray(action_grid, dtype=np.float64)
    if grid.ndim != 1 or grid.size < 2 \
            or not np.isfinite(grid).all() \
            or not bool((np.diff(grid) > 0).all()):
        raise ValueError("action grid must be finite, increasing, and non-trivial")
    return grid


def _tensor_action_grid(action_grid: Tensor) -> Tensor:
    if not isinstance(action_grid, Tensor) or not action_grid.is_floating_point() \
            or action_grid.ndim != 1 or action_grid.numel() < 2 \
            or not bool(torch.isfinite(action_grid).all()) \
            or not bool((action_grid.diff() > 0).all()):
        raise ValueError("action grid must be finite, increasing, and non-trivial")
    return action_grid


def _numpy_action_values(
    values: np.ndarray,
    action_grid: np.ndarray,
    label: str,
) -> np.ndarray:
    grid = _numpy_action_grid(action_grid)
    result = np.asarray(values, dtype=np.float64)
    if result.ndim < 1 or result.shape[-1] != grid.size \
            or np.isnan(result).any() or np.isposinf(result).any():
        raise ValueError(f"{label} must end in a valid action axis")
    return result


def _tensor_action_values(
    values: Tensor,
    action_grid: Tensor,
    label: str,
) -> Tensor:
    grid = _tensor_action_grid(action_grid)
    if not isinstance(values, Tensor) or not values.is_floating_point() \
            or values.ndim < 1 or values.shape[-1] != grid.numel() \
            or bool(torch.isnan(values).any()) or bool(torch.isposinf(values).any()):
        raise ValueError(f"{label} must end in a valid action axis")
    return values


def _normalized_numpy_probabilities(
    probabilities: np.ndarray,
    action_grid: np.ndarray,
) -> np.ndarray:
    result = _numpy_action_values(probabilities, action_grid, "probabilities")
    if not np.isfinite(result).all() or bool((result < 0).any()):
        raise ValueError("probabilities must be finite and non-negative")
    totals = result.sum(axis=-1, keepdims=True)
    if bool((totals <= 0).any()):
        raise ValueError("every probability row must have positive mass")
    return result / totals


def _normalized_tensor_probabilities(
    probabilities: Tensor,
    action_grid: Tensor,
) -> Tensor:
    result = _tensor_action_values(probabilities, action_grid, "probabilities")
    if not bool(torch.isfinite(result).all()) or bool((result < 0).any()):
        raise ValueError("probabilities must be finite and non-negative")
    totals = result.sum(dim=-1, keepdim=True)
    if bool((totals <= 0).any()):
        raise ValueError("every probability row must have positive mass")
    return result / totals


def _numpy_softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    maximum = np.max(values, axis=-1, keepdims=True)
    if not np.isfinite(maximum).all():
        raise ValueError("conditioned policy has no feasible action")
    exponential = np.exp(values - maximum)
    return exponential / exponential.sum(axis=-1, keepdims=True)


def _numpy_reset_mask(reset_mask: np.ndarray | None, count: int) -> np.ndarray:
    if reset_mask is None:
        return np.zeros(count, dtype=np.bool_)
    result = np.asarray(reset_mask, dtype=np.bool_)
    if result.shape != (count,):
        raise ValueError("reset mask must contain one value per time row")
    return result


def _tensor_reset_mask(
    reset_mask: Tensor | None,
    count: int,
    device: torch.device,
) -> Tensor:
    if reset_mask is None:
        return torch.zeros(count, device=device, dtype=torch.bool)
    result = reset_mask.to(device=device, dtype=torch.bool)
    if result.shape != (count,):
        raise ValueError("reset mask must contain one value per time row")
    return result


def _switch_threshold_numpy(
    action_grid: np.ndarray,
    threshold: float | None,
) -> float:
    result = abs(float(action_grid[1] - action_grid[0])) / 2 \
        if threshold is None else float(threshold)
    if not math.isfinite(result) or result <= 0:
        raise ValueError("switch threshold must be finite and positive")
    return result


def _switch_threshold_tensor(
    action_grid: Tensor,
    threshold: float | None,
) -> Tensor:
    value = abs(float((action_grid[1] - action_grid[0]).detach())) / 2 \
        if threshold is None else float(threshold)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("switch threshold must be finite and positive")
    return action_grid.new_tensor(value)


def _weighted_mean(
    values: Tensor,
    weights: Tensor,
    *,
    normalize_actual_weights: bool = True,
) -> Tensor:
    weighted = values * weights
    if not normalize_actual_weights:
        return weighted.mean()
    return weighted.sum() / weights.sum().clamp_min(1e-12)


def _precision_recall_f1(
    true_positive: int,
    predicted_positive: int,
    target_positive: int,
) -> tuple[float, float, float]:
    precision = true_positive / predicted_positive \
        if predicted_positive > 0 else 0.0
    recall = true_positive / target_positive if target_positive > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0 else 0.0
    )
    return precision, recall, f1


def _validate_friction(friction: float) -> None:
    if not math.isfinite(friction) or friction < 0:
        raise ValueError("friction must be finite and non-negative")


def _validate_temperature(temperature: float) -> None:
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
