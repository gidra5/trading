from __future__ import annotations

import hashlib
import itertools
import math
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np


LOSS_TERMS = (
    "crossEntropy",
    "probabilityMse",
    "parameterMse",
    "excessEntropy",
    "temporalMutualInformation",
    "oracleMutualInformation",
)
DISTRIBUTION_ANCHORS = (
    "crossEntropy",
    "probabilityMse",
    "parameterMse",
)


@dataclass(frozen=True)
class WeightCandidate:
    key: str
    values: tuple[float, ...]

    def as_dict(self) -> dict[str, float]:
        return dict(zip(LOSS_TERMS, self.values, strict=True))


@dataclass(frozen=True)
class ProjectedQuadratic:
    """Local validation model for a clipped weighted training direction."""

    base_validation: float
    linear: np.ndarray
    quadratic: np.ndarray
    gradient_gram: np.ndarray
    learning_rate: float
    maximum_gradient_norm: float = 1.0

    def scores(self, candidates: np.ndarray) -> np.ndarray:
        weights = np.atleast_2d(np.asarray(candidates, dtype=np.float64))
        if weights.shape[1] != len(LOSS_TERMS):
            raise ValueError("candidate matrix must contain all loss terms")
        linear = np.asarray(self.linear, dtype=np.float64)
        quadratic = np.asarray(self.quadratic, dtype=np.float64)
        gram = np.asarray(self.gradient_gram, dtype=np.float64)
        if linear.shape != (len(LOSS_TERMS),):
            raise ValueError("projected linear coefficients have the wrong shape")
        expected = (len(LOSS_TERMS), len(LOSS_TERMS))
        if quadratic.shape != expected or gram.shape != expected:
            raise ValueError("projected quadratic matrices have the wrong shape")
        direction_norm = np.sqrt(np.maximum(
            np.einsum("bi,ij,bj->b", weights, gram, weights),
            0,
        ))
        scale = np.minimum(
            1.0,
            self.maximum_gradient_norm / np.maximum(direction_norm, 1e-12),
        )
        first = -self.learning_rate * scale * (weights @ linear)
        second = 0.5 * self.learning_rate**2 * scale**2 * np.einsum(
            "bi,ij,bj->b",
            weights,
            quadratic,
            weights,
        )
        return self.base_validation + first + second


@dataclass(frozen=True)
class DelayContinuation:
    anchor_delay_seconds: int
    trial_delay_seconds: int
    step_seconds: int
    dwell_epochs: int
    stale_epochs: int
    best_validation: float


@dataclass(frozen=True)
class DelayDecision:
    action: str
    state: DelayContinuation
    reason: str


class GaussianProcessResidual:
    """Small exact GP used to correct projected scores after real probes."""

    def __init__(
        self,
        length_scale: float = 0.7,
        zero_mismatch_penalty: float = 1.0,
        noise: float = 1e-5,
    ) -> None:
        if length_scale <= 0 or zero_mismatch_penalty < 0 or noise <= 0:
            raise ValueError("invalid Gaussian-process configuration")
        self.length_scale = float(length_scale)
        self.zero_mismatch_penalty = float(zero_mismatch_penalty)
        self.noise = float(noise)
        self._inputs: np.ndarray | None = None
        self._cholesky: np.ndarray | None = None
        self._alpha: np.ndarray | None = None

    def fit(self, inputs: np.ndarray, residuals: np.ndarray) -> None:
        points = np.atleast_2d(np.asarray(inputs, dtype=np.float64))
        values = np.asarray(residuals, dtype=np.float64)
        if points.ndim != 2 or points.shape[0] != values.shape[0] or not len(values):
            raise ValueError("GP inputs and residuals must contain matching observations")
        covariance = self._kernel(points, points)
        jitter = self.noise
        for _ in range(8):
            try:
                cholesky = np.linalg.cholesky(
                    covariance + np.eye(len(points), dtype=np.float64) * jitter
                )
                break
            except np.linalg.LinAlgError:
                jitter *= 10
        else:
            raise np.linalg.LinAlgError("unable to stabilize residual GP")
        self._inputs = points
        self._cholesky = cholesky
        self._alpha = np.linalg.solve(
            cholesky.T,
            np.linalg.solve(cholesky, values),
        )

    def predict(self, inputs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self._inputs is None or self._cholesky is None or self._alpha is None:
            raise RuntimeError("residual GP has not been fitted")
        points = np.atleast_2d(np.asarray(inputs, dtype=np.float64))
        cross = self._kernel(points, self._inputs)
        mean = cross @ self._alpha
        solved = np.linalg.solve(self._cholesky, cross.T)
        variance = np.maximum(1.0 - np.square(solved).sum(axis=0), 1e-12)
        return mean, np.sqrt(variance)

    def _kernel(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        left_positive = left > 0
        right_positive = right > 0
        left_log = np.where(left_positive, np.log(np.maximum(left, 1e-12)), 0)
        right_log = np.where(right_positive, np.log(np.maximum(right, 1e-12)), 0)
        difference = left_log[:, None, :] - right_log[None, :, :]
        shared_positive = left_positive[:, None, :] & right_positive[None, :, :]
        squared_distance = np.square(difference * shared_positive).sum(axis=-1)
        zero_mismatch = np.logical_xor(
            left_positive[:, None, :],
            right_positive[None, :, :],
        ).sum(axis=-1)
        return np.exp(
            -0.5 * squared_distance / self.length_scale**2
            - self.zero_mismatch_penalty * zero_mismatch
        )


def enumerate_weight_candidates(
    levels: Sequence[float],
    *,
    canonicalize_global_scale: bool = True,
    fixed_loss_weights: dict[str, float] | None = None,
    term_weight_levels: dict[str, Sequence[float]] | None = None,
) -> list[WeightCandidate]:
    """Enumerate active loss tuples, retaining every valid weight direction."""
    values = tuple(sorted({float(level) for level in levels}))
    if len(values) < 2 or values[0] != 0 or any(
        not math.isfinite(value) or value < 0 for value in values
    ):
        raise ValueError("absolute weight levels require zero and positive finite values")
    fixed = {
        str(term): float(value)
        for term, value in (fixed_loss_weights or {}).items()
    }
    unknown = set(fixed) - set(LOSS_TERMS)
    if unknown:
        raise ValueError(f"unknown fixed loss terms: {sorted(unknown)}")
    if any(
        not math.isfinite(value) or value < 0 or value not in values
        for value in fixed.values()
    ):
        raise ValueError("fixed loss weights must belong to the absolute weight levels")
    overrides = {
        str(term): tuple(sorted({float(value) for value in term_values}))
        for term, term_values in (term_weight_levels or {}).items()
    }
    unknown = set(overrides) - set(LOSS_TERMS)
    if unknown:
        raise ValueError(f"unknown term-specific weight levels: {sorted(unknown)}")
    overlap = set(overrides) & set(fixed)
    if overlap:
        raise ValueError(
            f"fixed terms cannot also define weight levels: {sorted(overlap)}"
        )
    if any(
        not term_values
        or any(value not in values for value in term_values)
        for term_values in overrides.values()
    ):
        raise ValueError(
            "term-specific weight levels must be a non-empty subset "
            "of the absolute levels"
        )
    active_indexes = tuple(
        index
        for index, term in enumerate(LOSS_TERMS)
        if term not in fixed
    )
    active_levels = tuple(
        overrides.get(LOSS_TERMS[index], values)
        for index in active_indexes
    )
    anchor_indexes = tuple(LOSS_TERMS.index(term) for term in DISTRIBUTION_ANCHORS)
    unique: dict[tuple[float, ...], WeightCandidate] = {}
    for active in itertools.product(*active_levels):
        raw_values = [
            fixed.get(term, 0.0)
            for term in LOSS_TERMS
        ]
        for index, value in zip(active_indexes, active, strict=True):
            raw_values[index] = value
        raw = tuple(raw_values)
        if not any(raw[index] > 0 for index in anchor_indexes):
            continue
        scale = max(raw) if canonicalize_global_scale else 1.0
        canonical = tuple(round(value / scale, 12) for value in raw)
        if (
            canonical in unique
            and max(unique[canonical].values) >= max(raw)
        ):
            continue
        digest = hashlib.sha256(
            ",".join(f"{value:.12g}" for value in canonical).encode()
        ).hexdigest()[:12]
        unique[canonical] = WeightCandidate(
            key=f"absolute-{digest}",
            # The normalized tuple is only the equivalence key. Train with a
            # member of the configured absolute grid, choosing its largest
            # available common scale so no synthetic weight level is created.
            values=tuple(float(value) for value in raw),
        )
    return sorted(unique.values(), key=lambda candidate: candidate.values)


def candidate_matrix(candidates: Sequence[WeightCandidate]) -> np.ndarray:
    return np.asarray([candidate.values for candidate in candidates], dtype=np.float64)


def one_coordinate_neighbor_keys(
    candidate: WeightCandidate,
    candidates: Sequence[WeightCandidate],
    levels: Sequence[float],
    *,
    canonicalize_global_scale: bool = True,
    fixed_loss_weights: dict[str, float] | None = None,
    term_weight_levels: dict[str, Sequence[float]] | None = None,
) -> list[str]:
    """Return canonical candidates one configured grid step from a candidate."""
    configured_levels = tuple(sorted({float(level) for level in levels}))
    fixed = {
        str(term): float(value)
        for term, value in (fixed_loss_weights or {}).items()
    }
    overrides = {
        str(term): tuple(sorted({float(value) for value in values}))
        for term, values in (term_weight_levels or {}).items()
    }

    def canonical(values: Sequence[float]) -> tuple[float, ...]:
        scale = max(values) if canonicalize_global_scale else 1.0
        return tuple(round(float(value) / scale, 12) for value in values)

    by_canonical = {
        canonical(value.values): value
        for value in candidates
    }
    origin = list(candidate.values)
    origin_key = canonical(origin)
    neighbors: set[str] = set()
    anchor_indexes = {
        LOSS_TERMS.index(term)
        for term in DISTRIBUTION_ANCHORS
    }
    for index, term in enumerate(LOSS_TERMS):
        if term in fixed:
            continue
        allowed = overrides.get(term, configured_levels)
        try:
            position = allowed.index(float(origin[index]))
        except ValueError as error:
            raise ValueError(
                f"candidate weight for {term} is outside its configured levels"
            ) from error
        for neighbor_position in (position - 1, position + 1):
            if not 0 <= neighbor_position < len(allowed):
                continue
            values = list(origin)
            values[index] = allowed[neighbor_position]
            if not any(values[anchor] > 0 for anchor in anchor_indexes):
                continue
            normalized = canonical(values)
            if normalized == origin_key:
                continue
            neighbor = by_canonical.get(normalized)
            if neighbor is not None:
                neighbors.add(neighbor.key)
    return sorted(neighbors)


def select_probe_indices(
    candidates: np.ndarray,
    projected_scores: np.ndarray,
    count: int,
    *,
    exploitation_fraction: float = 0.5,
    random_fraction: float = 0.2,
    seed: int = 1337,
    excluded: Iterable[int] = (),
) -> list[int]:
    """Mix predicted winners, covering points, and deterministic exploration."""
    points = np.atleast_2d(np.asarray(candidates, dtype=np.float64))
    scores = np.asarray(projected_scores, dtype=np.float64)
    if points.shape[0] != scores.shape[0] or count < 1:
        raise ValueError("probe candidates and scores are inconsistent")
    blocked = {int(index) for index in excluded}
    available = [index for index in range(len(points)) if index not in blocked]
    count = min(count, len(available))
    exploit_count = min(
        count,
        max(1, round(count * exploitation_fraction)),
    )
    selected = sorted(available, key=lambda index: (scores[index], index))[:exploit_count]
    remaining_target = count - len(selected)
    random_count = min(remaining_target, round(count * random_fraction))
    diverse_count = remaining_target - random_count
    normalized = weight_embeddings(points)
    while diverse_count > 0:
        remaining = [
            index for index in available
            if index not in selected
        ]
        if not remaining:
            break
        if not selected:
            chosen = remaining[0]
        else:
            chosen = max(
                remaining,
                key=lambda index: (
                    min(
                        float(np.linalg.norm(
                            normalized[index] - normalized[other]
                        ))
                        for other in selected
                    ),
                    -scores[index],
                    -index,
                ),
            )
        selected.append(chosen)
        diverse_count -= 1
    if random_count:
        remaining = np.asarray([
            index for index in available if index not in selected
        ], dtype=np.int64)
        rng = np.random.default_rng(seed)
        if len(remaining):
            chosen = rng.choice(
                remaining,
                size=min(random_count, len(remaining)),
                replace=False,
            )
            selected.extend(int(index) for index in chosen)
    if len(selected) < count:
        selected.extend(
            index for index in sorted(available, key=lambda item: (scores[item], item))
            if index not in selected
        )
    return selected[:count]


def select_gp_acquisition_indices(
    candidates: np.ndarray,
    prior_scores: np.ndarray,
    observed_indices: Sequence[int],
    observed_values: Sequence[float],
    count: int,
    *,
    exploration: float = 1.5,
) -> tuple[list[int], np.ndarray, np.ndarray]:
    """Select unobserved full tuples by lower confidence bound."""
    points = np.atleast_2d(np.asarray(candidates, dtype=np.float64))
    prior = np.asarray(prior_scores, dtype=np.float64)
    indexes = np.asarray(observed_indices, dtype=np.int64)
    values = np.asarray(observed_values, dtype=np.float64)
    if not len(indexes):
        raise ValueError("GP acquisition requires actual probes")
    design = weight_embeddings(points)
    calibration = np.column_stack([
        np.ones(len(indexes), dtype=np.float64),
        prior[indexes],
    ])
    coefficients = np.linalg.lstsq(calibration, values, rcond=None)[0]
    calibrated = coefficients[0] + coefficients[1] * prior
    residual = values - calibrated[indexes]
    gp = GaussianProcessResidual()
    gp.fit(design[indexes], residual)
    correction, uncertainty = gp.predict(design)
    prediction = calibrated + correction
    acquisition = prediction - float(exploration) * uncertainty
    observed = set(int(index) for index in indexes)
    selected = [
        index
        for index in np.argsort(acquisition)
        if int(index) not in observed
    ][:count]
    return [int(index) for index in selected], prediction, uncertainty


def weight_embeddings(weights: np.ndarray) -> np.ndarray:
    points = np.atleast_2d(np.asarray(weights, dtype=np.float64))
    positive = points > 0
    log_value = np.where(
        positive,
        np.log(np.maximum(points, 1e-12)) / math.log(4),
        -3.0,
    )
    return np.concatenate([log_value, positive.astype(np.float64)], axis=1)


def propose_delay_seconds(
    anchor_delay_seconds: int,
    step_seconds: int,
    minimum_delay_seconds: int,
    count: int = 3,
) -> list[int]:
    """Return aggressive-to-conservative integer-second continuation proposals."""
    if minimum_delay_seconds < 1 or anchor_delay_seconds < minimum_delay_seconds:
        raise ValueError("delay bounds are invalid")
    if step_seconds < 1 or count < 1:
        raise ValueError("delay step and proposal count must be positive")
    if anchor_delay_seconds == minimum_delay_seconds:
        return [minimum_delay_seconds]
    maximum_step = anchor_delay_seconds - minimum_delay_seconds
    step = min(step_seconds, maximum_step)
    proposals = {
        max(minimum_delay_seconds, anchor_delay_seconds - max(1, round(
            step / (2**index)
        )))
        for index in range(count)
    }
    proposals.add(max(minimum_delay_seconds, anchor_delay_seconds - step))
    return sorted(proposals)


def initial_delay_continuation(
    maximum_delay_seconds: int,
    minimum_delay_seconds: int,
    initial_step_seconds: int,
) -> DelayContinuation:
    proposal = propose_delay_seconds(
        maximum_delay_seconds,
        initial_step_seconds,
        minimum_delay_seconds,
        1,
    )[0]
    return DelayContinuation(
        anchor_delay_seconds=maximum_delay_seconds,
        trial_delay_seconds=proposal,
        step_seconds=initial_step_seconds,
        dwell_epochs=0,
        stale_epochs=0,
        best_validation=math.inf,
    )


def update_delay_continuation(
    state: DelayContinuation,
    validation_kl: float,
    reference_kl: float,
    *,
    minimum_delay_seconds: int,
    absolute_tolerance: float,
    relative_tolerance: float,
    minimum_improvement: float,
    patience: int,
    maximum_dwell_epochs: int,
    step_growth_factor: float = 1.5,
) -> DelayDecision:
    if not math.isfinite(validation_kl) or not math.isfinite(reference_kl):
        raise ValueError("delay decisions require finite validation scores")
    if not math.isfinite(step_growth_factor) or step_growth_factor <= 1:
        raise ValueError("delay step growth factor must be greater than one")
    threshold = reference_kl + max(
        absolute_tolerance,
        relative_tolerance * reference_kl,
    )
    improvement = state.best_validation - validation_kl
    best = min(state.best_validation, validation_kl)
    stale = 0 if improvement >= minimum_improvement else state.stale_epochs + 1
    dwell = state.dwell_epochs + 1
    updated = DelayContinuation(
        anchor_delay_seconds=state.anchor_delay_seconds,
        trial_delay_seconds=state.trial_delay_seconds,
        step_seconds=state.step_seconds,
        dwell_epochs=dwell,
        stale_epochs=stale,
        best_validation=best,
    )
    if validation_kl <= threshold:
        if state.trial_delay_seconds <= minimum_delay_seconds:
            return DelayDecision("complete", updated, "minimum delay recovered")
        next_step = min(
            state.trial_delay_seconds - minimum_delay_seconds,
            max(
                state.step_seconds + 1,
                round(
                    state.step_seconds
                    * (
                        1
                        + (step_growth_factor - 1)
                        / max(1, dwell)
                    )
                ),
            ),
        )
        next_delay = propose_delay_seconds(
            state.trial_delay_seconds,
            next_step,
            minimum_delay_seconds,
            1,
        )[0]
        return DelayDecision(
            "advance",
            DelayContinuation(
                anchor_delay_seconds=state.trial_delay_seconds,
                trial_delay_seconds=next_delay,
                step_seconds=next_step,
                dwell_epochs=0,
                stale_epochs=0,
                best_validation=math.inf,
            ),
            "delay-specific reference accuracy recovered",
        )
    if dwell < maximum_dwell_epochs and stale < patience:
        return DelayDecision("dwell", updated, "accuracy is still recovering")
    reduced_step = max(1, state.step_seconds // 2)
    if reduced_step == state.step_seconds:
        return DelayDecision(
            "accept-plateau",
            updated,
            "minimum continuation step reached before reference recovery",
        )
    next_delay = propose_delay_seconds(
        state.anchor_delay_seconds,
        reduced_step,
        minimum_delay_seconds,
        1,
    )[0]
    return DelayDecision(
        "backtrack",
        DelayContinuation(
            anchor_delay_seconds=state.anchor_delay_seconds,
            trial_delay_seconds=next_delay,
            step_seconds=reduced_step,
            dwell_epochs=0,
            stale_epochs=0,
            best_validation=math.inf,
        ),
        "transition plateaued; retrying a smaller delay step",
    )


def candidate_training_batches(candidate: dict[str, Any]) -> int:
    """Return the training batches on one candidate's realized lineage."""
    return sum(
        int(step.get(
            "trainingBatches",
            step.get("cumulativeTrainingBatches", 0),
        ))
        for step in candidate.get("lineage", [])
    )


def schedule_point(
    candidate: dict[str, Any],
    *,
    initial_delay_seconds: int,
    reference_kl: float,
    absolute_quality_tolerance: float,
    relative_quality_tolerance: float,
    batch_size: int,
    gradient_accumulation: int,
) -> dict[str, Any]:
    """Summarize one realized schedule without scalarizing its objectives."""
    delay_seconds = int(candidate["delayMs"]) // 1_000
    progress_seconds = max(0, initial_delay_seconds - delay_seconds)
    metrics = candidate["validation"]
    validation_kl = float(metrics["klDivergence"])
    validation_std = float(metrics.get("klDivergenceStdDev", math.inf))
    quality_gap = validation_kl - float(reference_kl)
    tolerance = max(
        float(absolute_quality_tolerance),
        float(relative_quality_tolerance) * float(reference_kl),
    )
    lineage = candidate.get("lineage", [])
    batches = candidate_training_batches(candidate)
    optimizer_updates = sum(
        math.ceil(
            int(step.get(
                "trainingBatches",
                step.get("cumulativeTrainingBatches", 0),
            ))
            / gradient_accumulation
        )
        for step in lineage
    )
    recorded_gaps = [
        float(step["qualityGap"])
        for step in lineage
        if isinstance(step.get("qualityGap"), (int, float))
        and math.isfinite(float(step["qualityGap"]))
    ]
    maximum_positive_gap = max(
        [0.0, quality_gap, *recorded_gaps],
    )
    return {
        "key": candidate["key"],
        "delaySeconds": delay_seconds,
        "progressSeconds": progress_seconds,
        "trainingBatches": batches,
        "trainingExamples": batches * batch_size,
        "optimizerUpdates": optimizer_updates,
        "validationKl": validation_kl,
        "validationKlStdDev": validation_std,
        "referenceKl": float(reference_kl),
        "qualityGap": quality_gap,
        "maximumPositiveQualityGap": maximum_positive_gap,
        "qualityTolerance": tolerance,
        "withinQualityTolerance": quality_gap <= tolerance,
        "progressSecondsPer1024Batches": (
            progress_seconds * 1_024 / batches
            if batches > 0
            else 0.0
        ),
    }


def schedule_dominates(
    left: dict[str, Any],
    right: dict[str, Any],
) -> bool:
    """Compare progress, final-model effort, KL gap, and KL dispersion."""
    comparisons = (
        left["delaySeconds"] <= right["delaySeconds"],
        left["trainingBatches"] <= right["trainingBatches"],
        left["maximumPositiveQualityGap"]
        <= right["maximumPositiveQualityGap"],
        left["validationKlStdDev"] <= right["validationKlStdDev"],
    )
    strict = (
        left["delaySeconds"] < right["delaySeconds"]
        or left["trainingBatches"] < right["trainingBatches"]
        or left["maximumPositiveQualityGap"]
        < right["maximumPositiveQualityGap"]
        or left["validationKlStdDev"] < right["validationKlStdDev"]
    )
    return all(comparisons) and strict


def schedule_pareto_front(
    points: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    front = [
        point
        for point in points
        if not any(
            schedule_dominates(other, point)
            for other in points
            if other is not point
        )
    ]
    return sorted(
        front,
        key=lambda point: (
            point["delaySeconds"],
            point["trainingBatches"],
            point["maximumPositiveQualityGap"],
            point["validationKlStdDev"],
            point["key"],
        ),
    )


def recommended_schedule(
    points: Sequence[dict[str, Any]],
) -> dict[str, Any] | None:
    """Choose the furthest quality-feasible path, then effort and quality."""
    feasible = [
        point for point in points
        if point["withinQualityTolerance"]
    ]
    if not feasible:
        return None
    return min(
        feasible,
        key=lambda point: (
            point["delaySeconds"],
            point["trainingBatches"],
            point["maximumPositiveQualityGap"],
            point["validationKlStdDev"],
            point["key"],
        ),
    )


def interpolate_delay_reference(
    delay_seconds: int,
    references: dict[int, float],
) -> float:
    """Interpolate the best-known direct KL on log(1 + delay)."""
    if delay_seconds < 0 or not references:
        raise ValueError("delay reference inputs are invalid")
    ordered = sorted(
        (int(delay), float(value))
        for delay, value in references.items()
        if delay >= 0 and math.isfinite(value)
    )
    if not ordered:
        raise ValueError("delay references contain no finite observations")
    for delay, value in ordered:
        if delay == delay_seconds:
            return value
    if delay_seconds <= ordered[0][0]:
        return ordered[0][1]
    if delay_seconds >= ordered[-1][0]:
        return ordered[-1][1]
    right_index = next(
        index for index, item in enumerate(ordered)
        if item[0] > delay_seconds
    )
    left_delay, left_value = ordered[right_index - 1]
    right_delay, right_value = ordered[right_index]
    coordinate = math.log1p(delay_seconds)
    left_coordinate = math.log1p(left_delay)
    right_coordinate = math.log1p(right_delay)
    fraction = (coordinate - left_coordinate) / (
        right_coordinate - left_coordinate
    )
    return left_value + fraction * (right_value - left_value)
