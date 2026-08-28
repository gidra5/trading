from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Callable, Iterable, Protocol

import numpy as np


QUALITY_EQUIVALENCE_BITS = 0.001


@dataclass(frozen=True)
class BasisCandidate:
    id: str
    validation_bits: float
    fold_bits: tuple[float, ...]
    availability_minimum: float
    availability_mean: float
    acquisition_cost: int
    feature_count: int
    is_incumbent: bool = False


def equivalent_to_best(
    candidate: BasisCandidate,
    best: BasisCandidate,
    tolerance_bits: float = QUALITY_EQUIVALENCE_BITS,
) -> bool:
    if candidate.validation_bits < best.validation_bits - tolerance_bits:
        return False
    if len(candidate.fold_bits) != len(best.fold_bits):
        raise ValueError("Candidates must use the same chronological folds.")
    return all(
        candidate_fold >= best_fold - tolerance_bits
        for candidate_fold, best_fold in zip(candidate.fold_bits, best.fold_bits)
    )


def choose_lexicographic_basis(
    candidates: Iterable[BasisCandidate],
    tolerance_bits: float = QUALITY_EQUIVALENCE_BITS,
) -> BasisCandidate:
    rows = list(candidates)
    if not rows:
        raise ValueError("At least one basis candidate is required.")
    # Robust quality is the weakest chronological fold first, then the pooled
    # held-out score. This prevents a single favorable regime from defining the
    # reference that all availability/size trade-offs are compared against.
    best = max(rows, key=lambda row: (min(row.fold_bits, default=-np.inf), row.validation_bits))
    equivalent = [row for row in rows if equivalent_to_best(row, best, tolerance_bits)]
    return min(
        equivalent,
        key=lambda row: (
            -row.availability_minimum,
            -row.availability_mean,
            row.acquisition_cost,
            row.feature_count,
            -row.validation_bits,
            row.id,
        ),
    )


@dataclass(frozen=True)
class FeatureGroup:
    id: str
    arity: int
    penalty_weight: float
    output_indices: tuple[int, ...] | None = None


@dataclass
class AdditiveFit:
    intercept: np.ndarray
    coefficients: list[np.ndarray]
    objective: float
    iterations: int
    converged: bool


@dataclass
class MultiTaskAdditiveFit:
    """One additive categorical model with a shared group support.

    Coefficients are stored as ``[non-reference levels, sum(classes)]`` for
    each feature group.  Contiguous output slices belong to distinct softmax
    tasks, so a group is either retained or removed jointly across horizons.
    """

    intercept: np.ndarray
    coefficients: list[np.ndarray]
    classes: tuple[int, ...]
    objective: float
    iterations: int
    converged: bool
    stationarity_maximum: float | None = None


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / np.sum(exponent, axis=1, keepdims=True)


def task_slices(classes: tuple[int, ...]) -> tuple[slice, ...]:
    offsets = np.cumsum((0, *classes))
    return tuple(slice(int(offsets[index]), int(offsets[index + 1])) for index in range(len(classes)))


def multitask_probabilities(logits: np.ndarray, classes: tuple[int, ...]) -> np.ndarray:
    probability = np.empty_like(logits, dtype=np.float64)
    for output_slice in task_slices(classes):
        probability[:, output_slice] = softmax(logits[:, output_slice])
    return probability


def one_hot_target(labels: np.ndarray, classes: int) -> np.ndarray:
    output = np.zeros((labels.size, classes), dtype=np.float64)
    output[np.arange(labels.size), labels] = 1.0
    return output


def additive_logits(
    states: np.ndarray,
    intercept: np.ndarray,
    coefficients: list[np.ndarray],
) -> np.ndarray:
    logits = np.broadcast_to(intercept, (states.shape[0], intercept.size)).copy()
    for index, beta in enumerate(coefficients):
        state = states[:, index]
        active = state > 0
        if np.any(active):
            logits[active] += beta[state[active] - 1]
    return logits


def group_penalty(coefficients: list[np.ndarray], weights: np.ndarray) -> float:
    return float(sum(weight * np.linalg.norm(beta) for weight, beta in zip(weights, coefficients)))


def objective_value(
    states: np.ndarray,
    labels: np.ndarray,
    intercept: np.ndarray,
    coefficients: list[np.ndarray],
    regularization: float,
    weights: np.ndarray,
) -> float:
    probability = softmax(additive_logits(states, intercept, coefficients))
    loss = -np.mean(np.log(np.maximum(probability[np.arange(labels.size), labels], 1e-300)))
    return float(loss + regularization * group_penalty(coefficients, weights))


def gradients(
    states: np.ndarray,
    labels: np.ndarray,
    intercept: np.ndarray,
    coefficients: list[np.ndarray],
) -> tuple[np.ndarray, list[np.ndarray]]:
    probability = softmax(additive_logits(states, intercept, coefficients))
    residual = (probability - one_hot_target(labels, intercept.size)) / labels.size
    intercept_gradient = residual.sum(axis=0)
    coefficient_gradients = []
    for index, beta in enumerate(coefficients):
        gradient = np.zeros_like(beta)
        state = states[:, index]
        for level in range(1, beta.shape[0] + 1):
            gradient[level - 1] = residual[state == level].sum(axis=0)
        coefficient_gradients.append(gradient)
    return intercept_gradient, coefficient_gradients


def proximal_groups(
    coefficients: list[np.ndarray],
    thresholds: np.ndarray,
) -> list[np.ndarray]:
    output = []
    for beta, threshold in zip(coefficients, thresholds):
        norm = float(np.linalg.norm(beta))
        scale = max(0.0, 1.0 - float(threshold) / max(norm, 1e-300))
        output.append(beta * scale)
    return output


def fit_group_lasso_additive(
    states: np.ndarray,
    labels: np.ndarray,
    groups: list[FeatureGroup],
    classes: int,
    regularization: float,
    *,
    max_iterations: int = 2_000,
    tolerance: float = 1e-8,
    initial_step: float = 1.0,
) -> AdditiveFit:
    states = np.asarray(states, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64)
    if states.ndim != 2 or states.shape[1] != len(groups) or states.shape[0] != labels.size:
        raise ValueError("States, labels, and feature groups have incompatible shapes.")
    if np.any(labels < 0) or np.any(labels >= classes):
        raise ValueError("Target labels are outside the declared class range.")
    for index, group in enumerate(groups):
        if group.arity < 2 or np.any(states[:, index] < 0) or np.any(states[:, index] >= group.arity):
            raise ValueError(f"Invalid states for {group.id}.")

    frequencies = np.bincount(labels, minlength=classes).astype(np.float64) + 1.0
    intercept = np.log(frequencies / frequencies.sum())
    intercept -= intercept.mean()
    coefficients = [np.zeros((group.arity - 1, classes), dtype=np.float64) for group in groups]
    weights = np.asarray([group.penalty_weight for group in groups], dtype=np.float64)
    step = initial_step
    previous = objective_value(states, labels, intercept, coefficients, regularization, weights)

    for iteration in range(1, max_iterations + 1):
        intercept_gradient, coefficient_gradients = gradients(states, labels, intercept, coefficients)
        accepted = False
        trial_step = step
        for _ in range(40):
            candidate_intercept = intercept - trial_step * intercept_gradient
            candidate_intercept -= candidate_intercept.mean()
            raw = [beta - trial_step * gradient for beta, gradient in zip(coefficients, coefficient_gradients)]
            candidate_coefficients = proximal_groups(raw, trial_step * regularization * weights)
            candidate = objective_value(
                states, labels, candidate_intercept, candidate_coefficients, regularization, weights
            )
            if candidate <= previous + 1e-14:
                accepted = True
                break
            trial_step *= 0.5
        if not accepted:
            return AdditiveFit(intercept, coefficients, previous, iteration, False)
        change = previous - candidate
        intercept, coefficients, previous = candidate_intercept, candidate_coefficients, candidate
        step = min(trial_step * 1.05, 10.0)
        if change <= tolerance * max(1.0, abs(previous)):
            return AdditiveFit(intercept, coefficients, previous, iteration, True)
    return AdditiveFit(intercept, coefficients, previous, max_iterations, False)


def multitask_objective_value(
    states: np.ndarray,
    labels: np.ndarray,
    intercept: np.ndarray,
    coefficients: list[np.ndarray],
    classes: tuple[int, ...],
    task_weights: np.ndarray,
    regularization: float,
    weights: np.ndarray,
) -> float:
    probability = multitask_probabilities(additive_logits(states, intercept, coefficients), classes)
    loss = 0.0
    rows = np.arange(labels.shape[0])
    for task, output_slice in enumerate(task_slices(classes)):
        selected = probability[rows, output_slice.start + labels[:, task]]
        loss -= float(task_weights[task]) * float(np.mean(np.log(np.maximum(selected, 1e-300))))
    return float(loss + regularization * group_penalty(coefficients, weights))


def multitask_gradients(
    states: np.ndarray,
    labels: np.ndarray,
    intercept: np.ndarray,
    coefficients: list[np.ndarray],
    classes: tuple[int, ...],
    task_weights: np.ndarray,
) -> tuple[np.ndarray, list[np.ndarray]]:
    probability = multitask_probabilities(additive_logits(states, intercept, coefficients), classes)
    residual = probability
    rows = np.arange(labels.shape[0])
    for task, output_slice in enumerate(task_slices(classes)):
        residual[rows, output_slice.start + labels[:, task]] -= 1.0
        residual[:, output_slice] *= float(task_weights[task]) / labels.shape[0]
    intercept_gradient = residual.sum(axis=0)
    coefficient_gradients = []
    for index, beta in enumerate(coefficients):
        gradient = np.zeros_like(beta)
        state = states[:, index]
        for level in range(1, beta.shape[0] + 1):
            gradient[level - 1] = residual[state == level].sum(axis=0)
        coefficient_gradients.append(gradient)
    return intercept_gradient, coefficient_gradients


def fit_multitask_group_lasso_additive(
    states: np.ndarray,
    labels: np.ndarray,
    groups: list[FeatureGroup],
    classes: tuple[int, ...],
    regularization: float,
    *,
    task_weights: np.ndarray | None = None,
    max_iterations: int = 2_000,
    tolerance: float = 1e-8,
    initial_step: float = 1.0,
) -> MultiTaskAdditiveFit:
    """Fit a convex multi-task additive softmax model with shared group lasso."""
    states = np.asarray(states, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64)
    classes = tuple(int(value) for value in classes)
    if labels.ndim != 2 or labels.shape[1] != len(classes):
        raise ValueError("Labels must be observations by tasks and match classes.")
    if states.ndim != 2 or states.shape[1] != len(groups) or states.shape[0] != labels.shape[0]:
        raise ValueError("States, labels, and feature groups have incompatible shapes.")
    if not classes or any(value < 2 for value in classes):
        raise ValueError("Every task must contain at least two classes.")
    for task, count in enumerate(classes):
        if np.any(labels[:, task] < 0) or np.any(labels[:, task] >= count):
            raise ValueError(f"Target labels for task {task} are outside the declared class range.")
    for index, group in enumerate(groups):
        if group.arity < 2 or np.any(states[:, index] < 0) or np.any(states[:, index] >= group.arity):
            raise ValueError(f"Invalid states for {group.id}.")
    if task_weights is None:
        task_weights = np.full(len(classes), 1.0 / len(classes), dtype=np.float64)
    else:
        task_weights = np.asarray(task_weights, dtype=np.float64)
        if task_weights.shape != (len(classes),) or np.any(task_weights < 0) or task_weights.sum() <= 0:
            raise ValueError("Task weights must be nonnegative and match the number of tasks.")
        task_weights = task_weights / task_weights.sum()

    intercept_parts = []
    for task, count in enumerate(classes):
        frequencies = np.bincount(labels[:, task], minlength=count).astype(np.float64) + 1.0
        part = np.log(frequencies / frequencies.sum())
        intercept_parts.append(part - part.mean())
    intercept = np.concatenate(intercept_parts)
    outputs = sum(classes)
    coefficients = [np.zeros((group.arity - 1, outputs), dtype=np.float64) for group in groups]
    coefficient_masks = []
    for group in groups:
        mask = np.ones((group.arity - 1, outputs), dtype=np.float64)
        if group.output_indices is not None:
            mask[:] = 0.0
            mask[:, list(group.output_indices)] = 1.0
        coefficient_masks.append(mask)
    weights = np.asarray([group.penalty_weight for group in groups], dtype=np.float64)
    step = initial_step
    previous = multitask_objective_value(
        states, labels, intercept, coefficients, classes, task_weights, regularization, weights
    )
    slices = task_slices(classes)

    for iteration in range(1, max_iterations + 1):
        intercept_gradient, coefficient_gradients = multitask_gradients(
            states, labels, intercept, coefficients, classes, task_weights
        )
        coefficient_gradients = [
            gradient * mask for gradient, mask in zip(coefficient_gradients, coefficient_masks)
        ]
        accepted = False
        trial_step = step
        for _ in range(40):
            candidate_intercept = intercept - trial_step * intercept_gradient
            for output_slice in slices:
                candidate_intercept[output_slice] -= candidate_intercept[output_slice].mean()
            raw = [
                (beta - trial_step * gradient) * mask
                for beta, gradient, mask in zip(coefficients, coefficient_gradients, coefficient_masks)
            ]
            candidate_coefficients = proximal_groups(raw, trial_step * regularization * weights)
            candidate = multitask_objective_value(
                states,
                labels,
                candidate_intercept,
                candidate_coefficients,
                classes,
                task_weights,
                regularization,
                weights,
            )
            if candidate <= previous + 1e-14:
                accepted = True
                break
            trial_step *= 0.5
        if not accepted:
            return MultiTaskAdditiveFit(
                intercept, coefficients, classes, previous, iteration, False
            )
        change = previous - candidate
        intercept, coefficients, previous = candidate_intercept, candidate_coefficients, candidate
        step = min(trial_step * 1.05, 10.0)
        if change <= tolerance * max(1.0, abs(previous)):
            return MultiTaskAdditiveFit(
                intercept, coefficients, classes, previous, iteration, True
            )
    return MultiTaskAdditiveFit(
        intercept, coefficients, classes, previous, max_iterations, False
    )


def multitask_residual(
    states: np.ndarray,
    labels: np.ndarray,
    fit: MultiTaskAdditiveFit,
    task_weights: np.ndarray | None = None,
    offset_logits: np.ndarray | None = None,
) -> np.ndarray:
    """Return the objective-scaled residual used by a full-universe KKT scan."""
    labels = np.asarray(labels, dtype=np.int64)
    tasks = len(fit.classes)
    if task_weights is None:
        task_weights = np.full(tasks, 1.0 / tasks, dtype=np.float64)
    else:
        task_weights = np.asarray(task_weights, dtype=np.float64)
        task_weights = task_weights / task_weights.sum()
    logits = additive_logits(np.asarray(states), fit.intercept, fit.coefficients)
    if offset_logits is not None:
        offset = np.asarray(offset_logits, dtype=np.float64)
        if offset.shape != logits.shape:
            raise ValueError("Offset logits have an incompatible shape.")
        logits += offset
    probability = multitask_probabilities(logits, fit.classes)
    rows = np.arange(labels.shape[0])
    for task, output_slice in enumerate(task_slices(fit.classes)):
        probability[rows, output_slice.start + labels[:, task]] -= 1.0
        probability[:, output_slice] *= float(task_weights[task])
    return probability


def fit_multitask_group_lasso_torch(
    states: np.ndarray,
    labels: np.ndarray,
    groups: list[FeatureGroup],
    classes: tuple[int, ...],
    regularization: float,
    *,
    task_weights: np.ndarray | None = None,
    offset_logits: np.ndarray | None = None,
    fit_intercept: bool = True,
    initial: MultiTaskAdditiveFit | None = None,
    max_iterations: int = 1_000,
    tolerance: float = 1e-7,
    stationarity_tolerance: float | None = None,
    initial_step: float = 0.1,
    device: str | None = None,
) -> MultiTaskAdditiveFit:
    """GPU-capable full-batch proximal solver for a wide active working set.

    The implementation uses level-indicator GEMMs instead of constructing a
    one-hot design matrix.  This keeps the 42k-by-thousands state matrix in
    uint8 and makes exact working-set/KKT iterations feasible on a commodity
    GPU.  Float32 determines the numerical certificate tolerance.
    """
    import torch
    import torch.nn.functional as functional

    state_array = np.asarray(states, dtype=np.uint8)
    label_array = np.asarray(labels, dtype=np.int64)
    classes = tuple(int(value) for value in classes)
    if state_array.ndim != 2 or state_array.shape[1] != len(groups):
        raise ValueError("States and groups have incompatible shapes.")
    if label_array.shape != (state_array.shape[0], len(classes)):
        raise ValueError("Labels must be observations by tasks and match classes.")
    for index, group in enumerate(groups):
        if group.arity < 2 or np.any(state_array[:, index] >= group.arity):
            raise ValueError(f"Invalid states for {group.id}.")
    if task_weights is None:
        weight_array = np.full(len(classes), 1.0 / len(classes), dtype=np.float32)
    else:
        weight_array = np.asarray(task_weights, dtype=np.float32)
        if weight_array.shape != (len(classes),) or np.any(weight_array < 0) or weight_array.sum() <= 0:
            raise ValueError("Task weights must be nonnegative and match the number of tasks.")
        weight_array /= weight_array.sum()

    selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    state_tensor = torch.as_tensor(state_array, dtype=torch.uint8, device=selected_device)
    label_tensor = torch.as_tensor(label_array, dtype=torch.int64, device=selected_device)
    task_weight_tensor = torch.as_tensor(weight_array, dtype=torch.float32, device=selected_device)
    penalty_tensor = torch.as_tensor(
        [group.penalty_weight for group in groups], dtype=torch.float32, device=selected_device
    )
    if offset_logits is None:
        offset_tensor = torch.zeros(
            (state_array.shape[0], sum(classes)), dtype=torch.float32, device=selected_device
        )
    else:
        offset_array = np.asarray(offset_logits, dtype=np.float32)
        if offset_array.shape != (state_array.shape[0], sum(classes)):
            raise ValueError("Offset logits must be observations by total task outputs.")
        offset_tensor = torch.as_tensor(offset_array, dtype=torch.float32, device=selected_device)
    maximum_levels = max((group.arity - 1 for group in groups), default=0)
    outputs = sum(classes)
    valid_levels = torch.zeros(
        (len(groups), maximum_levels, outputs), dtype=torch.float32, device=selected_device
    )
    for index, group in enumerate(groups):
        if group.output_indices is None:
            valid_levels[index, : group.arity - 1, :] = 1.0
        else:
            valid_levels[index, : group.arity - 1, list(group.output_indices)] = 1.0
    eligible_by_level = [
        torch.as_tensor(
            [index for index, group in enumerate(groups) if group.arity > level],
            dtype=torch.int64,
            device=selected_device,
        )
        for level in range(1, maximum_levels + 1)
    ]
    # States and active groups are fixed for the complete proximal solve.  The
    # old hot path rebuilt these indicator matrices for every gradient and
    # every line-search objective, leaving a commodity GPU mostly idle on
    # allocation/conversion kernels.  Cache them once; the GEMMs and numerical
    # objective are otherwise unchanged.
    masks_by_level = [
        (state_tensor[:, eligible] == level).to(torch.float32)
        for level, eligible in enumerate(eligible_by_level, start=1)
    ]

    if initial is None:
        intercept_parts = []
        for task, count in enumerate(classes):
            if fit_intercept and offset_logits is None:
                frequencies = np.bincount(label_array[:, task], minlength=count).astype(np.float64) + 1.0
                part = np.log(frequencies / frequencies.sum())
                intercept_parts.append(part - part.mean())
            else:
                intercept_parts.append(np.zeros(count, dtype=np.float64))
        intercept = torch.as_tensor(np.concatenate(intercept_parts), dtype=torch.float32, device=selected_device)
        beta = torch.zeros((len(groups), maximum_levels, outputs), dtype=torch.float32, device=selected_device)
    else:
        if initial.classes != classes or len(initial.coefficients) != len(groups):
            raise ValueError("The warm start is incompatible with the active groups or tasks.")
        intercept = torch.as_tensor(initial.intercept, dtype=torch.float32, device=selected_device).clone()
        beta = torch.zeros((len(groups), maximum_levels, outputs), dtype=torch.float32, device=selected_device)
        for index, values in enumerate(initial.coefficients):
            beta[index, : values.shape[0]] = torch.as_tensor(values, dtype=torch.float32, device=selected_device)
        beta *= valid_levels

    slices = task_slices(classes)

    def logits_for(candidate_intercept, candidate_beta):
        logits = offset_tensor + candidate_intercept.unsqueeze(0)
        for level, (eligible, mask) in enumerate(
            zip(eligible_by_level, masks_by_level), start=1
        ):
            logits.add_(mask @ candidate_beta[eligible, level - 1, :])
        return logits

    def probabilities_for(logits):
        parts = [torch.softmax(logits[:, output_slice], dim=1) for output_slice in slices]
        return torch.cat(parts, dim=1)

    def objective_for(candidate_intercept, candidate_beta):
        logits = logits_for(candidate_intercept, candidate_beta)
        loss = torch.zeros((), dtype=torch.float32, device=selected_device)
        for task, output_slice in enumerate(slices):
            loss = loss + task_weight_tensor[task] * functional.cross_entropy(
                logits[:, output_slice], label_tensor[:, task], reduction="mean"
            )
        norms = torch.linalg.vector_norm(candidate_beta, dim=(1, 2))
        return loss + float(regularization) * torch.sum(penalty_tensor * norms)

    with torch.no_grad():
        previous = float(objective_for(intercept, beta).item())
        step = float(initial_step)
        stationarity_maximum = np.inf
        needs_support_refinement = False
        for iteration in range(1, max_iterations + 1):
            logits = logits_for(intercept, beta)
            residual = probabilities_for(logits)
            for task, output_slice in enumerate(slices):
                residual[:, output_slice] *= task_weight_tensor[task] / state_tensor.shape[0]
                residual[torch.arange(state_tensor.shape[0], device=selected_device), output_slice.start + label_tensor[:, task]] -= (
                    task_weight_tensor[task] / state_tensor.shape[0]
                )
            intercept_gradient = residual.sum(dim=0)
            if not fit_intercept:
                intercept_gradient.zero_()
            beta_gradient = torch.zeros_like(beta)
            for level, (eligible, mask) in enumerate(
                zip(eligible_by_level, masks_by_level), start=1
            ):
                beta_gradient[eligible, level - 1, :] = mask.T @ residual
            beta_gradient *= valid_levels

            accepted = False
            trial_step = step
            for _ in range(40):
                candidate_intercept = intercept - trial_step * intercept_gradient
                if fit_intercept:
                    for output_slice in slices:
                        candidate_intercept[output_slice] -= candidate_intercept[output_slice].mean()
                raw = (beta - trial_step * beta_gradient) * valid_levels
                norms = torch.linalg.vector_norm(raw, dim=(1, 2), keepdim=True)
                threshold = trial_step * float(regularization) * penalty_tensor[:, None, None]
                scale = torch.clamp(1.0 - threshold / torch.clamp(norms, min=1e-30), min=0.0)
                candidate_beta = raw * scale
                candidate = float(objective_for(candidate_intercept, candidate_beta).item())
                if candidate <= previous + 1e-7:
                    accepted = True
                    break
                trial_step *= 0.5
            if not accepted:
                converged = False
                break
            beta_mapping = torch.linalg.vector_norm(
                (beta - candidate_beta) / trial_step, dim=(1, 2)
            )
            mapping_maximum = float(
                torch.max(beta_mapping).item() if beta_mapping.numel() else 0.0
            )
            if fit_intercept:
                mapping_maximum = max(
                    mapping_maximum,
                    float(torch.max(torch.abs((intercept - candidate_intercept) / trial_step)).item()),
                )
            # Evaluate the actual group-lasso KKT equations at the state that
            # would be returned.  A proximal-gradient mapping can materially
            # understate this residual for tiny nonzero groups because the
            # shrinkage direction changes quickly near the zero boundary.
            # The direct residual is also what the independent streamed audit
            # computes, so convergence and certification now use one metric.
            beta_norms = torch.linalg.vector_norm(beta, dim=(1, 2))
            gradient_norms = torch.linalg.vector_norm(beta_gradient, dim=(1, 2))
            nonzero = beta_norms > 1e-10
            group_stationarity = torch.clamp(
                gradient_norms - float(regularization) * penalty_tensor,
                min=0.0,
            )
            if torch.any(nonzero):
                directions = beta[nonzero] / beta_norms[nonzero, None, None]
                active_residual = (
                    beta_gradient[nonzero]
                    + float(regularization)
                    * penalty_tensor[nonzero, None, None]
                    * directions
                )
                group_stationarity[nonzero] = torch.linalg.vector_norm(
                    active_residual, dim=(1, 2)
                )
            stationarity_maximum = float(
                torch.max(group_stationarity).item()
                if group_stationarity.numel() else 0.0
            )
            if fit_intercept:
                stationarity_maximum = max(
                    stationarity_maximum,
                    float(torch.max(torch.abs((intercept - candidate_intercept) / trial_step)).item()),
                )
            change = previous - candidate
            if stationarity_tolerance is not None:
                # The KKT residual above is evaluated at the current
                # (intercept, beta). Return that same state when it passes;
                # returning candidate_beta would attach the measurement to a
                # different point.
                if stationarity_maximum <= stationarity_tolerance:
                    converged = True
                    break
                # Once the proximal mapping is small, first-order iterations
                # can spend thousands of steps rotating tiny nonzero group
                # directions. Switch to a smooth quasi-Newton refinement on
                # the identified nonzero support, then evaluate the same exact
                # KKT residual again.
                if mapping_maximum <= stationarity_tolerance:
                    needs_support_refinement = True
                    converged = False
                    break
            else:
                reached_tolerance = change <= tolerance * max(1.0, abs(candidate))
            intercept, beta, previous = candidate_intercept, candidate_beta, candidate
            step = min(trial_step * 1.05, 10.0)
            if stationarity_tolerance is None and reached_tolerance:
                converged = True
                break
        else:
            iteration = max_iterations
            converged = False

    if needs_support_refinement:
        support_rows = (
            torch.linalg.vector_norm(beta, dim=(1, 2)) > 1e-8
        ).to(torch.float32)[:, None, None]
        beta_parameter = torch.nn.Parameter(beta.detach().clone())
        parameters = [beta_parameter]
        intercept_parameter = None
        if fit_intercept:
            intercept_parameter = torch.nn.Parameter(intercept.detach().clone())
            parameters.append(intercept_parameter)
        optimizer = torch.optim.LBFGS(
            parameters,
            lr=1.0,
            max_iter=min(250, max_iterations),
            tolerance_grad=max(1e-9, float(stationarity_tolerance) * 0.05),
            tolerance_change=max(1e-12, float(tolerance) * 0.1),
            history_size=50,
            line_search_fn="strong_wolfe",
        )

        def refinement_closure():
            optimizer.zero_grad(set_to_none=True)
            candidate_beta = beta_parameter * support_rows * valid_levels
            candidate_intercept = (
                intercept_parameter if intercept_parameter is not None else intercept
            )
            loss = objective_for(candidate_intercept, candidate_beta)
            loss.backward()
            return loss

        optimizer.step(refinement_closure)
        with torch.no_grad():
            beta = beta_parameter * support_rows * valid_levels
            if intercept_parameter is not None:
                intercept = intercept_parameter
                for output_slice in slices:
                    intercept[output_slice] -= intercept[output_slice].mean()
            logits = logits_for(intercept, beta)
            residual = probabilities_for(logits)
            for task, output_slice in enumerate(slices):
                residual[:, output_slice] *= task_weight_tensor[task] / state_tensor.shape[0]
                residual[
                    torch.arange(state_tensor.shape[0], device=selected_device),
                    output_slice.start + label_tensor[:, task],
                ] -= task_weight_tensor[task] / state_tensor.shape[0]
            intercept_gradient = residual.sum(dim=0)
            if not fit_intercept:
                intercept_gradient.zero_()
            beta_gradient = torch.zeros_like(beta)
            for level in range(1, maximum_levels + 1):
                eligible = eligible_by_level[level - 1]
                mask = (state_tensor[:, eligible] == level).to(torch.float32)
                beta_gradient[eligible, level - 1, :] = mask.T @ residual
            beta_gradient *= valid_levels
            beta_norms = torch.linalg.vector_norm(beta, dim=(1, 2))
            gradient_norms = torch.linalg.vector_norm(beta_gradient, dim=(1, 2))
            nonzero = beta_norms > 1e-10
            group_stationarity = torch.clamp(
                gradient_norms - float(regularization) * penalty_tensor,
                min=0.0,
            )
            if torch.any(nonzero):
                directions = beta[nonzero] / beta_norms[nonzero, None, None]
                active_residual = (
                    beta_gradient[nonzero]
                    + float(regularization)
                    * penalty_tensor[nonzero, None, None]
                    * directions
                )
                group_stationarity[nonzero] = torch.linalg.vector_norm(
                    active_residual, dim=(1, 2)
                )
            stationarity_maximum = float(
                torch.max(group_stationarity).item()
                if group_stationarity.numel() else 0.0
            )
            if fit_intercept:
                stationarity_maximum = max(
                    stationarity_maximum,
                    float(torch.max(torch.abs(intercept_gradient)).item()),
                )
            previous = float(objective_for(intercept, beta).item())
            converged = bool(stationarity_maximum <= stationarity_tolerance)
        refinement_state = optimizer.state.get(beta_parameter, {})
        iteration += int(refinement_state.get("n_iter", 0))

    intercept_array = intercept.cpu().numpy().astype(np.float64)
    beta_array = beta.cpu().numpy().astype(np.float64)
    coefficients = [beta_array[index, : group.arity - 1] for index, group in enumerate(groups)]
    return MultiTaskAdditiveFit(
        intercept_array, coefficients, classes, previous, iteration, converged,
        stationarity_maximum,
    )


def fit_multitask_group_lasso_active_set_torch(
    states: np.ndarray,
    labels: np.ndarray,
    groups: list[FeatureGroup],
    classes: tuple[int, ...],
    regularization: float,
    *,
    task_weights: np.ndarray | None = None,
    offset_logits: np.ndarray | None = None,
    fit_intercept: bool = True,
    initial: MultiTaskAdditiveFit | None = None,
    max_iterations: int = 1_000,
    tolerance: float = 1e-7,
    stationarity_tolerance: float | None = None,
    kkt_tolerance: float = 1e-5,
    add_limit: int = 64,
    scan_chunk_size: int = 256,
    max_active_rounds: int = 64,
    prune_converged_zeros: bool = True,
    initial_step: float = 0.1,
    device: str | None = None,
    progress_callback: Callable[[dict[str, object]], None] | None = None,
) -> MultiTaskAdditiveFit:
    """Solve the same convex problem with an exact KKT active-set loop.

    Wide feature ledgers are dominated by groups whose optimum is exactly zero.
    Fitting all of them at every proximal iteration is wasteful.  This routine
    fits only the current nonzero/violating groups, streams every omitted group,
    adds its KKT violators, and repeats until the complete working set satisfies
    the requested tolerance.  The returned coefficient list still follows the
    original full group order, so callers and saved model artifacts are
    unchanged.
    """
    state_array = np.asarray(states, dtype=np.uint8)
    label_array = np.asarray(labels, dtype=np.int64)
    classes = tuple(int(value) for value in classes)
    if state_array.ndim != 2 or state_array.shape[1] != len(groups):
        raise ValueError("States and groups have incompatible shapes.")
    if label_array.shape != (state_array.shape[0], len(classes)):
        raise ValueError("Labels must be observations by tasks and match classes.")
    if add_limit < 1 or scan_chunk_size < 1 or max_active_rounds < 1:
        raise ValueError("Active-set limits must be positive.")
    if initial is not None and (
        initial.classes != classes or len(initial.coefficients) != len(groups)
    ):
        raise ValueError("The warm start is incompatible with the full groups or tasks.")

    index_by_id = {group.id: index for index, group in enumerate(groups)}
    if len(index_by_id) != len(groups):
        raise ValueError("Feature-group IDs must be unique.")
    active = [] if initial is None else [
        index for index, coefficient in enumerate(initial.coefficients)
        if float(np.linalg.norm(coefficient)) > 1e-5
    ]
    intercept = None if initial is None else np.asarray(initial.intercept, dtype=np.float64)
    coefficient_by_index = {} if initial is None else {
        index: np.asarray(initial.coefficients[index], dtype=np.float64)
        for index in active
    }
    total_iterations = 0
    last_fit: MultiTaskAdditiveFit | None = None
    omitted_maximum = np.inf

    for active_round in range(1, max_active_rounds + 1):
        active = sorted(set(active))
        active_groups = [groups[index] for index in active]
        active_states = state_array[:, active]
        active_initial = None
        if intercept is not None:
            active_initial = MultiTaskAdditiveFit(
                intercept=np.asarray(intercept, dtype=np.float64),
                coefficients=[coefficient_by_index[index] for index in active],
                classes=classes,
                objective=np.inf,
                iterations=0,
                converged=False,
            )
        fit = fit_multitask_group_lasso_torch(
            active_states,
            label_array,
            active_groups,
            classes,
            regularization,
            task_weights=task_weights,
            offset_logits=offset_logits,
            fit_intercept=fit_intercept,
            initial=active_initial,
            max_iterations=max_iterations,
            tolerance=tolerance,
            stationarity_tolerance=stationarity_tolerance,
            initial_step=initial_step,
            device=device,
        )
        total_iterations += fit.iterations
        last_fit = fit
        intercept = fit.intercept
        # A converged proximal solve certifies its thresholded zeros, so prune
        # them before the next full omitted-group scan.  If the bounded inner
        # solve has not converged, retain every active group and continue from
        # its warm state rather than repeatedly re-adding the same coordinate.
        coefficient_by_index = {
            index: np.asarray(coefficient, dtype=np.float64)
            for index, coefficient in zip(active, fit.coefficients)
            if (
                not prune_converged_zeros
                or not fit.converged
                or float(np.linalg.norm(coefficient)) > 1e-10
            )
        }
        active = sorted(coefficient_by_index)
        active_ids = {groups[index].id for index in active}
        support_groups = [groups[index] for index in active]
        support_coefficients = [coefficient_by_index[index] for index in active]
        support_fit = MultiTaskAdditiveFit(
            intercept=np.asarray(intercept, dtype=np.float64),
            coefficients=support_coefficients,
            classes=classes,
            objective=fit.objective,
            iterations=fit.iterations,
            converged=fit.converged,
            stationarity_maximum=fit.stationarity_maximum,
        )
        residual = multitask_residual(
            state_array[:, active],
            label_array,
            support_fit,
            task_weights=task_weights,
            offset_logits=offset_logits,
        )

        def batches():
            for start in range(0, len(groups), scan_chunk_size):
                end = min(start + scan_chunk_size, len(groups))
                yield SimpleNamespace(
                    groups=groups[start:end], states=state_array[:, start:end]
                )

        scan = scan_quantized_batches_residual(
            residual,
            batches,
            regularization,
            active_ids,
            add_limit=add_limit,
            tolerance=kkt_tolerance,
            device=device,
        )
        omitted_maximum = scan.maximum_violation
        additions = [
            index_by_id[feature_id]
            for feature_id, _ in scan.violating_groups
            if feature_id not in active_ids
        ]
        if progress_callback is not None:
            progress_callback({
                "round": active_round,
                "activeGroups": len(active),
                "additions": len(additions),
                "innerIterations": int(fit.iterations),
                "totalIterations": int(total_iterations),
                "innerConverged": bool(fit.converged),
                "activeStationarityMaximum": float(fit.stationarity_maximum),
                "omittedMaximumViolation": float(omitted_maximum),
            })
        if not additions:
            # The active groups can need more proximal iterations even after
            # every omitted group is feasible.  Continue from the warm state
            # instead of returning an under-converged path point merely because
            # one bounded inner call exhausted its iteration budget.
            if not fit.converged:
                continue
            full_coefficients = [
                coefficient_by_index.get(
                    index,
                    np.zeros((group.arity - 1, sum(classes)), dtype=np.float64),
                )
                for index, group in enumerate(groups)
            ]
            return MultiTaskAdditiveFit(
                intercept=np.asarray(intercept, dtype=np.float64),
                coefficients=full_coefficients,
                classes=classes,
                objective=fit.objective,
                iterations=total_iterations,
                converged=bool(fit.converged and omitted_maximum <= kkt_tolerance),
                stationarity_maximum=max(
                    float(fit.stationarity_maximum), max(0.0, float(omitted_maximum))
                ),
            )
        active.extend(additions)
        for index in additions:
            coefficient_by_index[index] = np.zeros(
                (groups[index].arity - 1, sum(classes)), dtype=np.float64
            )

    if last_fit is None:
        raise RuntimeError("Active-set solver did not execute.")
    full_coefficients = [
        coefficient_by_index.get(
            index,
            np.zeros((group.arity - 1, sum(classes)), dtype=np.float64),
        )
        for index, group in enumerate(groups)
    ]
    return MultiTaskAdditiveFit(
        intercept=np.asarray(intercept, dtype=np.float64),
        coefficients=full_coefficients,
        classes=classes,
        objective=last_fit.objective,
        iterations=total_iterations,
        converged=False,
        stationarity_maximum=max(
            float(last_fit.stationarity_maximum), max(0.0, float(omitted_maximum))
        ),
    )


def excluded_group_gradient(
    states: np.ndarray,
    residual: np.ndarray,
    arity: int,
) -> np.ndarray:
    gradient = np.zeros((arity - 1, residual.shape[1]), dtype=np.float64)
    for level in range(1, arity):
        gradient[level - 1] = residual[states == level].sum(axis=0) / states.size
    return gradient


def kkt_violation(gradient: np.ndarray, regularization: float, penalty_weight: float) -> float:
    return float(np.linalg.norm(gradient) - regularization * penalty_weight)


def active_group_kkt_residuals(
    states: np.ndarray,
    residual: np.ndarray,
    groups: list[FeatureGroup],
    coefficients: list[np.ndarray],
    regularization: float,
    *,
    device: str | None = None,
) -> np.ndarray:
    """Return stationarity residual norms for nonzero active groups.

    ``residual`` is the unnormalized probability-minus-target matrix returned
    by :func:`multitask_residual`; gradients are divided by the observation
    count here, matching the fitting objective and streamed omitted-group scan.
    """
    import torch

    state_array = np.asarray(states, dtype=np.uint8)
    residual_array = np.asarray(residual, dtype=np.float32)
    if state_array.ndim != 2 or state_array.shape[1] != len(groups):
        raise ValueError("Active states and groups are incompatible.")
    if residual_array.ndim != 2 or residual_array.shape[0] != state_array.shape[0]:
        raise ValueError("Active residual has an incompatible shape.")
    if len(coefficients) != len(groups):
        raise ValueError("Active coefficients and groups are incompatible.")
    if not groups:
        return np.empty(0, dtype=np.float64)

    selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    states_tensor = torch.as_tensor(state_array, dtype=torch.uint8, device=selected_device)
    residual_tensor = torch.as_tensor(residual_array, dtype=torch.float32, device=selected_device)
    maximum_levels = max(group.arity - 1 for group in groups)
    outputs = residual_array.shape[1]
    gradients = torch.zeros(
        (len(groups), maximum_levels, outputs), dtype=torch.float32, device=selected_device
    )
    beta = torch.zeros_like(gradients)
    valid = torch.zeros_like(gradients)
    for index, (group, values) in enumerate(zip(groups, coefficients)):
        coefficient = np.asarray(values, dtype=np.float32)
        expected = (group.arity - 1, outputs)
        if coefficient.shape != expected:
            raise ValueError(f"Active coefficient for {group.id} has shape {coefficient.shape}, expected {expected}.")
        beta[index, : group.arity - 1] = torch.as_tensor(
            coefficient, dtype=torch.float32, device=selected_device
        )
        valid[index, : group.arity - 1] = 1.0
    with torch.inference_mode():
        for level in range(1, maximum_levels + 1):
            eligible = torch.as_tensor(
                [index for index, group in enumerate(groups) if group.arity > level],
                dtype=torch.int64,
                device=selected_device,
            )
            mask = (states_tensor[:, eligible] == level).to(torch.float32)
            gradients[eligible, level - 1] = (
                mask.T @ residual_tensor / state_array.shape[0]
            )
        beta *= valid
        norms = torch.linalg.vector_norm(beta, dim=(1, 2))
        penalties = torch.as_tensor(
            [group.penalty_weight for group in groups],
            dtype=torch.float32,
            device=selected_device,
        )
        directions = beta / torch.clamp(norms[:, None, None], min=1e-30)
        stationarity = gradients + float(regularization) * penalties[:, None, None] * directions
        result = torch.linalg.vector_norm(stationarity, dim=(1, 2))
    return result.cpu().numpy().astype(np.float64)


class CandidateBatchProvider(Protocol):
    def __call__(self) -> Iterable[tuple[list[FeatureGroup], np.ndarray]]:
        """Yield (groups, states) with states shaped [observations, groups]."""


@dataclass(frozen=True)
class KktScan:
    maximum_violation: float
    violating_groups: tuple[tuple[str, float], ...]
    scanned_groups: int


def scan_quantized_batches_kkt(
    labels: np.ndarray,
    active_states: np.ndarray,
    fit: AdditiveFit,
    batches: Callable[[], Iterable[object]],
    regularization: float,
    active_ids: set[str],
    *,
    add_limit: int = 64,
    tolerance: float = 1e-7,
    device: str | None = None,
) -> KktScan:
    """Scan every streamed candidate with batched level-indicator GEMMs.

    A batch object must expose ``groups`` and a uint8 ``states`` matrix. The
    implementation imports torch lazily so registry/count-only commands do not
    pay CUDA startup cost.
    """
    import torch

    labels = np.asarray(labels, dtype=np.int64)
    probability = softmax(additive_logits(active_states, fit.intercept, fit.coefficients))
    residual = probability - one_hot_target(labels, fit.intercept.size)
    return scan_quantized_batches_residual(
        residual,
        batches,
        regularization,
        active_ids,
        add_limit=add_limit,
        tolerance=tolerance,
        device=device,
    )


def scan_quantized_batches_residual(
    residual: np.ndarray,
    batches: Callable[[], Iterable[object]],
    regularization: float,
    active_ids: set[str],
    *,
    add_limit: int = 64,
    tolerance: float = 1e-7,
    device: str | None = None,
) -> KktScan:
    """KKT scan for a precomputed single- or multi-task residual matrix."""
    import torch

    residual = np.asarray(residual, dtype=np.float64)
    if residual.ndim != 2:
        raise ValueError("Residuals must be an observations by output matrix.")
    observations = residual.shape[0]
    selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    residual_tensor = torch.as_tensor(residual, dtype=torch.float32, device=selected_device)
    violations: list[tuple[str, float]] = []
    scanned = 0
    maximum = -np.inf
    with torch.inference_mode():
        for batch in batches():
            groups: list[FeatureGroup] = batch.groups
            state_array = np.asarray(batch.states, dtype=np.uint8)
            if state_array.shape != (observations, len(groups)):
                raise ValueError("A candidate batch has an invalid state matrix shape.")
            keep = [index for index, group in enumerate(groups) if group.id not in active_ids]
            if not keep:
                continue
            kept_groups = [groups[index] for index in keep]
            states = torch.as_tensor(state_array[:, keep], dtype=torch.uint8, device=selected_device)
            maximum_arity = max(group.arity for group in kept_groups)
            gradients = torch.zeros(
                (len(kept_groups), maximum_arity - 1, residual.shape[1]),
                dtype=torch.float32,
                device=selected_device,
            )
            for level in range(1, maximum_arity):
                mask = (states == level).to(torch.float32)
                gradients[:, level - 1] = mask.T @ residual_tensor / observations
            norms = torch.linalg.vector_norm(gradients, dim=(1, 2))
            penalties = torch.as_tensor(
                [group.penalty_weight for group in kept_groups],
                dtype=torch.float32,
                device=selected_device,
            )
            batch_violations = (norms - regularization * penalties).cpu().numpy()
            scanned += len(kept_groups)
            if batch_violations.size:
                maximum = max(maximum, float(np.max(batch_violations)))
            for group, violation in zip(kept_groups, batch_violations):
                if float(violation) > tolerance:
                    violations.append((group.id, float(violation)))
            if len(violations) > add_limit * 8:
                violations = sorted(violations, key=lambda row: (-row[1], row[0]))[: add_limit * 2]
    violations.sort(key=lambda row: (-row[1], row[0]))
    return KktScan(float(maximum), tuple(violations[:add_limit]), scanned)


def scan_quantized_batches_multitask_kkt(
    residual: np.ndarray,
    batches: Callable[[], Iterable[object]],
    regularizations: tuple[float, ...],
    active_ids_by_task: tuple[set[str], ...],
    output_slices: tuple[slice, ...],
    *,
    add_limit: int = 64,
    tolerance: float = 1e-7,
    device: str | None = None,
) -> tuple[KktScan, ...]:
    """Scan one streamed coordinate universe for several independent heads."""
    import torch

    residual = np.asarray(residual, dtype=np.float64)
    tasks = len(regularizations)
    if residual.ndim != 2 or len(active_ids_by_task) != tasks or len(output_slices) != tasks:
        raise ValueError("Residuals and task KKT specifications are incompatible.")
    observations = residual.shape[0]
    selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    residual_tensor = torch.as_tensor(residual, dtype=torch.float32, device=selected_device)
    # Keep one operationally preferable representative for candidates whose
    # state partitions induce exactly the same gradient.  We still count and
    # certify every coordinate; this only prevents synchronized calendars or
    # duplicate transforms from crowding all useful violations out of the
    # bounded expansion list.
    violations: list[dict[bytes, tuple[str, float]]] = [dict() for _ in range(tasks)]
    scanned = [0] * tasks
    maximum = [-np.inf] * tasks
    with torch.inference_mode():
        for batch in batches():
            groups: list[FeatureGroup] = batch.groups
            state_array = np.asarray(batch.states, dtype=np.uint8)
            if state_array.shape != (observations, len(groups)):
                raise ValueError("A candidate batch has an invalid state matrix shape.")
            states = torch.as_tensor(state_array, dtype=torch.uint8, device=selected_device)
            maximum_arity = max(group.arity for group in groups)
            gradients = torch.zeros(
                (len(groups), maximum_arity - 1, residual.shape[1]),
                dtype=torch.float32,
                device=selected_device,
            )
            for level in range(1, maximum_arity):
                mask = (states == level).to(torch.float32)
                gradients[:, level - 1] = mask.T @ residual_tensor / observations
            penalties = torch.as_tensor(
                [group.penalty_weight for group in groups],
                dtype=torch.float32,
                device=selected_device,
            )
            for task, output_slice in enumerate(output_slices):
                norms = torch.linalg.vector_norm(gradients[:, :, output_slice], dim=(1, 2))
                task_gradients = gradients[:, :, output_slice].cpu().numpy()
                batch_violations = (
                    norms - float(regularizations[task]) * penalties
                ).cpu().numpy()
                for group_index, (group, violation) in enumerate(zip(groups, batch_violations)):
                    if group.id in active_ids_by_task[task]:
                        continue
                    value = float(violation)
                    scanned[task] += 1
                    maximum[task] = max(maximum[task], value)
                    if value > tolerance:
                        signature = task_gradients[group_index].tobytes()
                        existing = violations[task].get(signature)
                        candidate = (group.id, value)
                        if existing is None or operational_id_rank(candidate[0]) < operational_id_rank(existing[0]):
                            violations[task][signature] = candidate
                if len(violations[task]) > add_limit * 8:
                    retained = sorted(
                        violations[task].items(),
                        key=lambda item: (-item[1][1], operational_id_rank(item[1][0])),
                    )[: add_limit * 2]
                    violations[task] = dict(retained)
    output = []
    for task in range(tasks):
        retained = sorted(
            violations[task].values(), key=lambda row: (-row[1], operational_id_rank(row[0]))
        )
        output.append(KktScan(
            float(maximum[task]), tuple(retained[:add_limit]), scanned[task]
        ))
    return tuple(output)


def operational_id_rank(feature_id: str) -> tuple[int, str]:
    """Deterministic availability proxy for exact in-sample equivalents."""
    subject = feature_id.split("/", 4)[1] if "/" in feature_id else ""
    preferred = {"btc": 0, "eth": 1, "sol": 2, "xrp": 3}.get(subject, 4)
    return preferred, feature_id


def scan_excluded_kkt(
    labels: np.ndarray,
    active_states: np.ndarray,
    fit: AdditiveFit,
    provider: CandidateBatchProvider,
    regularization: float,
    active_ids: set[str],
    *,
    add_limit: int = 64,
    tolerance: float = 1e-7,
) -> KktScan:
    probability = softmax(additive_logits(active_states, fit.intercept, fit.coefficients))
    residual = probability - one_hot_target(np.asarray(labels), fit.intercept.size)
    violations: list[tuple[str, float]] = []
    scanned = 0
    maximum = -np.inf
    for groups, states in provider():
        if states.shape != (labels.size, len(groups)):
            raise ValueError("A candidate batch has an invalid state matrix shape.")
        for index, group in enumerate(groups):
            if group.id in active_ids:
                continue
            gradient = excluded_group_gradient(states[:, index], residual, group.arity)
            violation = kkt_violation(gradient, regularization, group.penalty_weight)
            maximum = max(maximum, violation)
            scanned += 1
            if violation > tolerance:
                violations.append((group.id, violation))
    violations.sort(key=lambda row: (-row[1], row[0]))
    return KktScan(float(maximum), tuple(violations[:add_limit]), scanned)


def held_out_bits(labels: np.ndarray, probabilities: np.ndarray, baseline: np.ndarray) -> float:
    rows = np.arange(labels.size)
    ratio = np.log2(np.maximum(probabilities[rows, labels], 1e-300)) - np.log2(
        np.maximum(baseline[rows, labels], 1e-300)
    )
    return float(np.mean(ratio))
