from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np


LOSS_TERMS = (
    "crossEntropy",
    "probabilityMse",
    "parameterMse",
    "excessEntropy",
    "oracleMutualInformation",
)
PYTHON_LOSS_TERMS = {
    "cross_entropy": "crossEntropy",
    "probability_mse": "probabilityMse",
    "parameter_mse": "parameterMse",
    "excess_entropy": "excessEntropy",
    "oracle_mutual_information": "oracleMutualInformation",
}
SURROGATE_METRIC = "klDivergence"
DEFAULT_PROBE_BUDGETS = (24, 28, 32)


@dataclass(frozen=True)
class TrainingTrajectory:
    delay_ms: int
    weights: dict[str, float]
    epochs: dict[int, dict[str, float]]


def parse_training_trajectories(lines: Iterable[str]) -> list[TrainingTrajectory]:
    active_key: tuple[int, tuple[float, ...]] | None = None
    active_weights: dict[str, float] | None = None
    epoch_buffers: dict[
        tuple[int, tuple[float, ...]], dict[int, dict[str, float]]
    ] = {}
    completed: list[TrainingTrajectory] = []

    for line in lines:
        try:
            event = json.loads(line)
        except (json.JSONDecodeError, TypeError):
            continue
        event_name = event.get("event")
        if event_name == "training-start":
            delay_ms = int(event.get("predictionDelayMs", -1))
            weights = normalize_weights(event.get("lossWeights"))
            if delay_ms < 0 or weights is None:
                active_key = None
                active_weights = None
                continue
            active_key = trajectory_key(delay_ms, weights)
            active_weights = weights
            start_epoch = int(event.get("startEpoch", 0))
            epochs = epoch_buffers.setdefault(active_key, {})
            for epoch in tuple(epochs):
                if epoch >= start_epoch:
                    del epochs[epoch]
            continue
        if event_name == "epoch" and active_key is not None:
            validation = finite_metrics(event.get("validation"))
            epoch = event.get("epoch")
            if isinstance(epoch, int) and validation:
                epoch_buffers.setdefault(active_key, {})[epoch] = validation
            continue
        if event_name != "training-study-complete":
            continue
        delay_ms = int(event.get("predictionDelayMs", -1))
        weights = normalize_weights(event.get("lossWeights"))
        if delay_ms < 0 or weights is None:
            continue
        key = trajectory_key(delay_ms, weights)
        epochs = epoch_buffers.get(key, {})
        if epochs:
            completed.append(TrainingTrajectory(
                delay_ms=delay_ms,
                weights=weights,
                epochs={epoch: dict(metrics) for epoch, metrics in epochs.items()},
            ))
        if active_key == key:
            active_key = None
            active_weights = None

    latest: dict[tuple[int, tuple[float, ...]], TrainingTrajectory] = {}
    for trajectory in completed:
        latest[trajectory_key(trajectory.delay_ms, trajectory.weights)] = trajectory
    return list(latest.values())


def normalize_weights(value: Any) -> dict[str, float] | None:
    if not isinstance(value, dict):
        return None
    normalized: dict[str, float] = {}
    for key, item in value.items():
        term = PYTHON_LOSS_TERMS.get(key, key)
        if term not in LOSS_TERMS:
            continue
        number = float(item)
        if not math.isfinite(number) or number <= 0:
            return None
        normalized[term] = number
    if set(normalized) != set(LOSS_TERMS):
        return None
    return normalized


def finite_metrics(value: Any) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    metrics: dict[str, float] = {}
    for key, item in value.items():
        if isinstance(item, (int, float)) and math.isfinite(float(item)):
            metrics[key] = float(item)
    return metrics


def trajectory_key(
    delay_ms: int,
    weights: dict[str, float],
) -> tuple[int, tuple[float, ...]]:
    return delay_ms, tuple(round(float(weights[term]), 12) for term in LOSS_TERMS)


def normalized_log_weights(
    weights: dict[str, float],
    base_weights: dict[str, float],
    scale_base: float = 4.0,
) -> np.ndarray:
    values = []
    denominator = math.log(scale_base)
    for term in LOSS_TERMS:
        base = float(base_weights[term])
        value = float(weights[term])
        if base <= 0 or value <= 0:
            raise ValueError("response analysis requires positive base and candidate weights")
        values.append(math.log(value / base) / denominator)
    return np.asarray(values, dtype=np.float64)


def response_features(weight_coordinates: np.ndarray) -> np.ndarray:
    coordinates = np.atleast_2d(np.asarray(weight_coordinates, dtype=np.float64))
    if coordinates.shape[1] != len(LOSS_TERMS):
        raise ValueError("weight coordinates must contain all six loss terms")
    columns = [np.ones(coordinates.shape[0], dtype=np.float64)]
    columns.extend(coordinates[:, index] for index in range(coordinates.shape[1]))
    for left in range(coordinates.shape[1]):
        for right in range(left + 1, coordinates.shape[1]):
            columns.append(coordinates[:, left] * coordinates[:, right])
    columns.append(np.mean(np.square(coordinates), axis=1))
    return np.column_stack(columns)


def response_feature_names() -> list[str]:
    names = ["intercept", *(f"linear:{term}" for term in LOSS_TERMS)]
    names.extend(
        f"interaction:{LOSS_TERMS[left]}:{LOSS_TERMS[right]}"
        for left in range(len(LOSS_TERMS))
        for right in range(left + 1, len(LOSS_TERMS))
    )
    names.append("radialCurvature")
    return names


def fit_ridge(
    features: np.ndarray,
    response: np.ndarray,
    ridge: float = 1e-6,
) -> np.ndarray:
    matrix = np.asarray(features, dtype=np.float64)
    values = np.asarray(response, dtype=np.float64)
    penalty = np.eye(matrix.shape[1], dtype=np.float64) * ridge
    penalty[0, 0] = 0
    system = matrix.T @ matrix + penalty
    target = matrix.T @ values
    try:
        return np.linalg.solve(system, target)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(system, target, rcond=None)[0]


def leave_one_out_predictions(
    features: np.ndarray,
    response: np.ndarray,
    ridge: float = 1e-6,
    protected: Iterable[int] = (),
) -> np.ndarray:
    predictions = np.empty(len(response), dtype=np.float64)
    protected_indices = set(int(index) for index in protected)
    for held_out in range(len(response)):
        if held_out in protected_indices:
            predictions[held_out] = response[held_out]
            continue
        mask = np.arange(len(response)) != held_out
        coefficients = fit_ridge(features[mask], response[mask], ridge)
        predictions[held_out] = float(features[held_out] @ coefficients)
    return predictions


def greedy_d_optimal_indices(
    features: np.ndarray,
    budget: int,
    required: Iterable[int] = (),
    ridge: float = 1e-6,
) -> list[int]:
    matrix = np.asarray(features, dtype=np.float64)
    if budget < 1 or budget > matrix.shape[0]:
        raise ValueError("probe budget must be between one and the candidate count")
    selected = list(dict.fromkeys(int(index) for index in required))
    if any(index < 0 or index >= matrix.shape[0] for index in selected):
        raise IndexError("required D-optimal index is out of range")
    if len(selected) > budget:
        raise ValueError("required D-optimal indices exceed the probe budget")
    information = np.eye(matrix.shape[1], dtype=np.float64) * ridge
    for index in selected:
        row = matrix[index]
        information += np.outer(row, row)
    remaining = set(range(matrix.shape[0])) - set(selected)
    while len(selected) < budget:
        inverse = np.linalg.pinv(information, hermitian=True)
        candidate = max(
            remaining,
            key=lambda index: (
                float(matrix[index] @ inverse @ matrix[index]),
                -index,
            ),
        )
        selected.append(candidate)
        remaining.remove(candidate)
        row = matrix[candidate]
        information += np.outer(row, row)
    return selected


def analyze_epoch_response(
    observations: list[dict[str, Any]],
    base_weights: dict[str, float],
    probe_budgets: tuple[int, ...] = DEFAULT_PROBE_BUDGETS,
    equivalence_absolute_kl: float = 0.02,
    equivalence_relative_kl: float = 0.01,
) -> dict[str, Any]:
    coordinates = np.vstack([
        normalized_log_weights(observation["weights"], base_weights)
        for observation in observations
    ])
    features = response_features(coordinates)
    response = np.asarray([
        observation["metrics"][SURROGATE_METRIC] for observation in observations
    ], dtype=np.float64)
    center_indices = [
        index for index, coordinate in enumerate(coordinates)
        if np.allclose(coordinate, 0)
    ]
    coefficients = fit_ridge(features, response)
    fitted = features @ coefficients
    cross_validated = leave_one_out_predictions(
        features,
        response,
        protected=center_indices,
    )
    cross_validation_indices = [
        index for index in range(len(observations)) if index not in center_indices
    ]
    actual_order = np.argsort(response)
    predicted_order = np.argsort(cross_validated)
    actual_best = int(actual_order[0])
    predicted_best = int(predicted_order[0])
    best_rank = int(np.where(predicted_order == actual_best)[0][0]) + 1
    threshold = max(
        equivalence_absolute_kl,
        equivalence_relative_kl * float(response[actual_best]),
    )
    equivalent = np.flatnonzero(response <= response[actual_best] + threshold)
    budget_results = []
    for budget in probe_budgets:
        if budget >= len(observations) or budget < features.shape[1]:
            continue
        selected = greedy_d_optimal_indices(features, budget, center_indices[:1])
        selected_set = set(selected)
        held_out = [index for index in range(len(observations)) if index not in selected_set]
        probe_coefficients = fit_ridge(features[selected], response[selected])
        predictions = features @ probe_coefficients
        winner = int(np.argmin(predictions))
        budget_results.append({
            "budget": budget,
            "selectedActualBest": actual_best in selected_set,
            "predictedWinner": observations[winner]["variant"],
            "predictedWinnerActualKl": float(response[winner]),
            "actualBestRank": int(np.where(np.argsort(predictions) == actual_best)[0][0]) + 1,
            "actualRegret": float(response[winner] - response[actual_best]),
            "heldOutRmse": root_mean_square(
                predictions[held_out] - response[held_out],
            ) if held_out else 0,
            "heldOutCount": len(held_out),
        })
    return {
        "observations": len(observations),
        "featureCount": features.shape[1],
        "actualWinner": observations[actual_best]["variant"],
        "actualBestKl": float(response[actual_best]),
        "crossValidatedWinner": observations[predicted_best]["variant"],
        "crossValidatedWinnerActualKl": float(response[predicted_best]),
        "crossValidatedActualBestRank": best_rank,
        "crossValidatedTop4Recall": best_rank <= 4,
        "crossValidatedRmse": root_mean_square(
            cross_validated[cross_validation_indices]
            - response[cross_validation_indices],
        ),
        "crossValidatedMae": float(np.mean(np.abs(
            cross_validated[cross_validation_indices]
            - response[cross_validation_indices],
        ))),
        "crossValidatedSpearman": spearman_correlation(response, cross_validated),
        "fitRmse": root_mean_square(fitted - response),
        "fitR2": coefficient_of_determination(response, fitted),
        "equivalenceKlThreshold": threshold,
        "equivalentBestVariants": [
            observations[index]["variant"] for index in equivalent
        ],
        "coefficients": dict(zip(response_feature_names(), coefficients.tolist())),
        "probeBudgets": budget_results,
    }


def build_analysis(
    summary: dict[str, Any],
    trajectories: list[TrainingTrajectory],
) -> dict[str, Any]:
    base_weights = summary["weightDesign"]["baseLossWeights"]
    row_lookup = {
        trajectory_key(int(row["delayMs"]), row["weights"]): row
        for row in summary.get("results", [])
    }
    observations: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for trajectory in trajectories:
        row = row_lookup.get(trajectory_key(trajectory.delay_ms, trajectory.weights))
        if row is None:
            continue
        for epoch, metrics in trajectory.epochs.items():
            if SURROGATE_METRIC not in metrics:
                continue
            observations.setdefault((trajectory.delay_ms, epoch), []).append({
                "variant": row["weightVariant"],
                "weights": trajectory.weights,
                "metrics": metrics,
            })

    minimum_observations = len(response_feature_names())
    epoch_analyses = []
    for (delay_ms, epoch), epoch_observations in sorted(observations.items()):
        if len(epoch_observations) < minimum_observations:
            continue
        analysis = analyze_epoch_response(epoch_observations, base_weights)
        epoch_analyses.append({
            "delayMs": delay_ms,
            "delayMinutes": delay_ms / 60_000,
            "epoch": epoch,
            **analysis,
        })

    transition_probes = []
    by_delay: dict[int, list[dict[str, Any]]] = {}
    for analysis in epoch_analyses:
        by_delay.setdefault(int(analysis["delayMs"]), []).append(analysis)
    for delay_ms, analyses in by_delay.items():
        ordered = sorted(analyses, key=lambda item: int(item["epoch"]))
        for previous, current in zip(ordered, ordered[1:]):
            if previous["actualWinner"] == current["actualWinner"]:
                continue
            transition_probes.append({
                "delayMs": delay_ms,
                "delayMinutes": delay_ms / 60_000,
                "fromEpoch": previous["epoch"],
                "toEpoch": current["epoch"],
                "profileA": previous["actualWinner"],
                "profileB": current["actualWinner"],
                "experiments": [
                    f"{previous['actualWinner']}→{current['actualWinner']}",
                    f"{current['actualWinner']}→{previous['actualWinner']}",
                ],
            })

    top4_recall = [
        bool(analysis["crossValidatedTop4Recall"]) for analysis in epoch_analyses
    ]
    probe_candidates = [
        result
        for analysis in epoch_analyses
        for result in analysis["probeBudgets"]
    ]
    return {
        "version": 1,
        "generatedAt": datetime.now(UTC).isoformat(),
        "sourceExperimentFingerprint": summary.get("experimentFingerprint"),
        "sourceCompletedRuns": summary.get("completedRuns"),
        "terms": list(LOSS_TERMS),
        "surrogate": {
            "response": SURROGATE_METRIC,
            "coordinates": "log4(candidateWeight/baseWeight)",
            "model": "intercept + six main effects + 15 pairwise interactions + shared radial curvature",
            "featureCount": len(response_feature_names()),
            "ridge": 1e-6,
        },
        "trajectoryCount": len(trajectories),
        "epochAnalyses": epoch_analyses,
        "transitionOrderProbes": transition_probes,
        "aggregate": {
            "analyzedDelayEpochs": len(epoch_analyses),
            "crossValidatedTop4Recall": (
                sum(top4_recall) / len(top4_recall) if top4_recall else None
            ),
            "meanCrossValidatedSpearman": mean_or_none(
                analysis["crossValidatedSpearman"] for analysis in epoch_analyses
            ),
            "meanCrossValidatedRmse": mean_or_none(
                analysis["crossValidatedRmse"] for analysis in epoch_analyses
            ),
            "probeBudgetResults": summarize_probe_budgets(probe_candidates),
        },
    }


def summarize_probe_budgets(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    budgets = sorted({int(result["budget"]) for result in results})
    summary = []
    for budget in budgets:
        matching = [result for result in results if int(result["budget"]) == budget]
        summary.append({
            "budget": budget,
            "analyses": len(matching),
            "top4Recall": sum(
                int(result["actualBestRank"]) <= 4 for result in matching
            ) / len(matching),
            "meanActualRegret": mean_or_none(
                float(result["actualRegret"]) for result in matching
            ),
            "meanHeldOutRmse": mean_or_none(
                float(result["heldOutRmse"]) for result in matching
            ),
        })
    return summary


def root_mean_square(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(array))))


def coefficient_of_determination(actual: np.ndarray, predicted: np.ndarray) -> float:
    centered = actual - np.mean(actual)
    denominator = float(centered @ centered)
    if denominator == 0:
        return 1.0 if np.allclose(actual, predicted) else 0.0
    residual = actual - predicted
    return 1 - float(residual @ residual) / denominator


def spearman_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left_rank = rank_values(left)
    right_rank = rank_values(right)
    left_centered = left_rank - np.mean(left_rank)
    right_centered = right_rank - np.mean(right_rank)
    denominator = math.sqrt(
        float(left_centered @ left_centered) * float(right_centered @ right_centered)
    )
    if denominator == 0:
        return 0
    return float(left_centered @ right_centered) / denominator


def rank_values(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        rank = (start + stop - 1) / 2
        ranks[order[start:stop]] = rank
        start = stop
    return ranks


def mean_or_none(values: Iterable[float]) -> float | None:
    collected = list(values)
    return float(np.mean(collected)) if collected else None


def markdown_report(analysis: dict[str, Any]) -> str:
    aggregate = analysis["aggregate"]
    lines = [
        "# Dynamic MLP study offline analysis",
        "",
        f"Generated: {analysis['generatedAt']}",
        "",
        f"Source fingerprint: `{analysis['sourceExperimentFingerprint']}`",
        "",
        f"Recovered completed trajectories: {analysis['trajectoryCount']}",
        "",
        "## Surrogate validation",
        "",
        "| Delay | Epoch | Runs | Actual winner | LOO best rank | LOO Spearman | LOO RMSE | Equivalent best set |",
        "|---:|---:|---:|---|---:|---:|---:|---:|",
    ]
    for item in analysis["epochAnalyses"]:
        lines.append(
            f"| {item['delayMinutes']:g}m | {item['epoch'] + 1} "
            f"| {item['observations']} | {item['actualWinner']} "
            f"| {item['crossValidatedActualBestRank']} "
            f"| {item['crossValidatedSpearman']:.3f} "
            f"| {item['crossValidatedRmse']:.6f} "
            f"| {len(item['equivalentBestVariants'])} |"
        )
    lines.extend([
        "",
        f"Overall leave-one-out top-4 recall: "
        f"{format_optional_percent(aggregate['crossValidatedTop4Recall'])}.",
        "",
        "## D-optimal probe budgets",
        "",
        "| Probes | Analyses | Top-4 recall | Mean regret | Held-out RMSE |",
        "|---:|---:|---:|---:|---:|",
    ])
    for item in aggregate["probeBudgetResults"]:
        lines.append(
            f"| {item['budget']} | {item['analyses']} "
            f"| {format_optional_percent(item['top4Recall'])} "
            f"| {item['meanActualRegret']:.6f} "
            f"| {item['meanHeldOutRmse']:.6f} |"
        )
    lines.extend([
        "",
        "## Order-sensitivity pilot transitions",
        "",
    ])
    if not analysis["transitionOrderProbes"]:
        lines.append("No sufficiently covered epoch changed its observed winner.")
    else:
        for item in analysis["transitionOrderProbes"]:
            lines.append(
                f"- {item['delayMinutes']:g}m, epochs "
                f"{item['fromEpoch'] + 1}→{item['toEpoch'] + 1}: "
                f"`{item['experiments'][0]}` versus `{item['experiments'][1]}`."
            )
    lines.extend([
        "",
        "These order probes are required before two schedules can be merged as empirically equivalent.",
        "",
    ])
    return "\n".join(lines)


def format_optional_percent(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:.1f}%"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate dynamic loss-weight response models from study epoch logs.",
    )
    parser.add_argument("--plan", type=Path, default=Path("ml/training-plan.json"))
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--log", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    plan = json.loads(arguments.plan.read_text())
    study = plan["lossWeightStudy"]
    summary_file = arguments.summary or Path(study["outputDir"]) / "summary.json"
    log_file = arguments.log or Path(study["runDir"]) / "study.log"
    output = arguments.output or (
        Path("data/ml-dynamic-studies") / plan["id"] / "offline-analysis.json"
    )
    summary = json.loads(summary_file.read_text())
    trajectories = parse_training_trajectories(log_file.open())
    analysis = build_analysis(summary, trajectories)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(analysis, indent=2) + "\n")
    markdown = output.with_suffix(".md")
    markdown.write_text(markdown_report(analysis))
    print(json.dumps({
        "event": "dynamic-study-offline-analysis",
        "output": str(output.resolve()),
        "markdown": str(markdown.resolve()),
        **analysis["aggregate"],
    }))


if __name__ == "__main__":
    main()
