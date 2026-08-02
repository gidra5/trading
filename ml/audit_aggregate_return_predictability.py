"""Read-only CPU baselines for aggregate future close-log-return prediction.

This uses the feature-screen harness's exact train/validation timestamp corpus,
compact-minute multiplicities, and seven-hour cross-split embargo.  Held-out
test examples and payloads are never selected or read.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys

import numpy as np

from future_price_feature_screen import FeatureSpec
from train_future_price_feature_screen import (
    EXAMPLE_SPAN_MS,
    FeatureScreenDataset,
)
from train_future_price_predictor import (
    compact_pair_rows,
    select_pair_segments,
    validate_predictor_split_disjointness,
    validate_source_manifest,
)


AGGREGATIONS = (1, 5, 15)
HORIZONS = (5, 15, 60)
HISTORY_MINUTES = 360
RECENT_MINUTES = 60
LONG_SUMMARY_MINUTES = (120, 180, 360)
RIDGE_CANDIDATES = (1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1.0)
TRAIN_FIT_FRACTION = 0.8
BATCH_SIZE = 8_192


@dataclass
class RegressionStatistics:
    dimension: int

    def __post_init__(self) -> None:
        self.weight = 0.0
        self.xtx = np.zeros((self.dimension, self.dimension), dtype=np.float64)
        self.xty = np.zeros((self.dimension, len(HORIZONS)), dtype=np.float64)
        self.y_sum = np.zeros(len(HORIZONS), dtype=np.float64)
        self.y_square_sum = np.zeros(len(HORIZONS), dtype=np.float64)

    def add(self, features: np.ndarray, targets: np.ndarray, weights: np.ndarray) -> None:
        if features.shape[1] != self.dimension \
                or targets.shape != (features.shape[0], len(HORIZONS)) \
                or weights.shape != (features.shape[0],):
            raise ValueError("regression sufficient statistics are misaligned")
        x = np.asarray(features, dtype=np.float64)
        y = np.asarray(targets, dtype=np.float64)
        w = np.asarray(weights, dtype=np.float64)
        weighted_x = x * np.sqrt(w)[:, None]
        self.weight += float(w.sum())
        self.xtx += weighted_x.T @ weighted_x
        self.xty += x.T @ (w[:, None] * y)
        self.y_sum += (w[:, None] * y).sum(axis=0)
        self.y_square_sum += (w[:, None] * np.square(y)).sum(axis=0)


@dataclass
class PredictionStatistics:
    def __post_init__(self) -> None:
        self.weight = 0.0
        self.target_nonzero_weight = 0.0
        self.error_square_sum = 0.0
        self.prediction_sum = 0.0
        self.target_sum = 0.0
        self.prediction_square_sum = 0.0
        self.target_square_sum = 0.0
        self.cross_sum = 0.0
        self.direction_correct_weight = 0.0

    def add(
        self,
        prediction: np.ndarray,
        target: np.ndarray,
        weights: np.ndarray,
    ) -> None:
        prediction = np.asarray(prediction, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)
        weights = np.asarray(weights, dtype=np.float64)
        if prediction.shape != target.shape or target.shape != weights.shape:
            raise ValueError("prediction statistics are misaligned")
        error = prediction - target
        self.weight += float(weights.sum())
        self.error_square_sum += float((weights * np.square(error)).sum())
        self.prediction_sum += float((weights * prediction).sum())
        self.target_sum += float((weights * target).sum())
        self.prediction_square_sum += float((weights * np.square(prediction)).sum())
        self.target_square_sum += float((weights * np.square(target)).sum())
        self.cross_sum += float((weights * prediction * target).sum())
        nonzero = target != 0
        self.target_nonzero_weight += float(weights[nonzero].sum())
        correct = nonzero & (np.signbit(prediction) == np.signbit(target)) \
            & (prediction != 0)
        self.direction_correct_weight += float(weights[correct].sum())

    def metrics(self) -> dict[str, float]:
        prediction_mean = self.prediction_sum / self.weight
        target_mean = self.target_sum / self.weight
        prediction_variance = max(
            0.0,
            self.prediction_square_sum / self.weight - prediction_mean ** 2,
        )
        target_variance = max(
            0.0,
            self.target_square_sum / self.weight - target_mean ** 2,
        )
        covariance = self.cross_sum / self.weight - prediction_mean * target_mean
        correlation = (
            covariance / math.sqrt(prediction_variance * target_variance)
            if prediction_variance > 0 and target_variance > 0
            else 0.0
        )
        return {
            "mse": self.error_square_sum / self.weight,
            "correlation": correlation,
            "directionAccuracy": (
                self.direction_correct_weight / self.target_nonzero_weight
                if self.target_nonzero_weight > 0
                else 0.0
            ),
        }


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    source_manifest = json.loads((
        repo_root
        / "data/training/datasets"
        / "mlp-direct-oracle-temporal-v12-delay-3600s-full-minute-oracle"
        / "dataset.json"
    ).read_text(encoding="utf-8"))
    validate_source_manifest(source_manifest)
    segments = select_pair_segments(
        source_manifest,
        cross_split_purge_ms=EXAMPLE_SPAN_MS,
    )
    validate_predictor_split_disjointness(
        segments,
        example_span_ms=EXAMPLE_SPAN_MS,
    )
    component_root = (
        repo_root
        / "data/training/immutable/refs/features"
        / "future-price-predictor-minute-base-v1"
        / "future-price-predictor-6h-feature-screen-v1"
    )
    component_files = {
        file.stem: file
        for file in component_root.glob("*.json")
    }
    dataset = FeatureScreenDataset(
        segments,
        component_files,
        FeatureSpec(format="close"),
    )
    compact_counts = {
        split: sum(
            compact_pair_rows(
                shard.history_row_offset,
                shard.future_row_offset,
                shard.count,
            )[2].shape[0]
            for shard in segments[split]
        )
        for split in ("train", "validation")
    }
    train_fit_rows = int(compact_counts["train"] * TRAIN_FIT_FRACTION)
    dimensions = {
        aggregation: feature_dimension(aggregation)
        for aggregation in AGGREGATIONS
    }
    full_statistics = {
        aggregation: RegressionStatistics(dimensions[aggregation])
        for aggregation in AGGREGATIONS
    }
    fit_statistics = {
        aggregation: RegressionStatistics(dimensions[aggregation])
        for aggregation in AGGREGATIONS
    }
    calibration_statistics = {
        aggregation: RegressionStatistics(dimensions[aggregation])
        for aggregation in AGGREGATIONS
    }

    seen_rows = 0
    for batch_index, (history, future, weights) in enumerate(dataset.iter_batches(
        "train",
        BATCH_SIZE,
        shuffle=False,
        seed=0,
    ), start=1):
        history_array = history.numpy()[:, :, 0]
        future_array = future.numpy()
        weight_array = weights.numpy()
        targets = horizon_targets(future_array)
        batch_count = history_array.shape[0]
        split_at = min(batch_count, max(0, train_fit_rows - seen_rows))
        for aggregation in AGGREGATIONS:
            features = regression_features(history_array, aggregation)
            full_statistics[aggregation].add(features, targets, weight_array)
            if split_at:
                fit_statistics[aggregation].add(
                    features[:split_at],
                    targets[:split_at],
                    weight_array[:split_at],
                )
            if split_at < batch_count:
                calibration_statistics[aggregation].add(
                    features[split_at:],
                    targets[split_at:],
                    weight_array[split_at:],
                )
        seen_rows += batch_count
        if batch_index % 20 == 0:
            print(
                f"train sufficient statistics: {seen_rows:,}/"
                f"{compact_counts['train']:,}",
                file=sys.stderr,
                flush=True,
            )
    if seen_rows != compact_counts["train"]:
        raise RuntimeError("train compact count changed during the audit")

    target_means = full_statistics[AGGREGATIONS[0]].y_sum \
        / full_statistics[AGGREGATIONS[0]].weight
    ols_coefficients: dict[int, np.ndarray] = {}
    ridge_coefficients: dict[int, np.ndarray] = {}
    selected_ridges: dict[int, list[float]] = {}
    for aggregation in AGGREGATIONS:
        full = full_statistics[aggregation]
        ols_coefficients[aggregation] = solve(full, 0.0)
        selected = []
        coefficients = np.empty_like(ols_coefficients[aggregation])
        for horizon_index, _horizon in enumerate(HORIZONS):
            best: tuple[float, float, np.ndarray] | None = None
            for ridge in RIDGE_CANDIDATES:
                candidate = solve(fit_statistics[aggregation], ridge)
                mse = sufficient_mse(
                    calibration_statistics[aggregation],
                    candidate[:, horizon_index],
                    horizon_index,
                )
                if best is None or mse < best[0]:
                    best = (mse, ridge, candidate[:, horizon_index])
            assert best is not None
            selected.append(best[1])
            coefficients[:, horizon_index] = solve(
                full,
                best[1],
            )[:, horizon_index]
        selected_ridges[aggregation] = selected
        ridge_coefficients[aggregation] = coefficients

    metrics = {
        (aggregation, horizon, model): PredictionStatistics()
        for aggregation in AGGREGATIONS
        for horizon in HORIZONS
        for model in ("persistence", "linear", "ridge")
    }
    zero_metrics = {horizon: PredictionStatistics() for horizon in HORIZONS}
    mean_metrics = {horizon: PredictionStatistics() for horizon in HORIZONS}
    validation_rows = 0
    for batch_index, (history, future, weights) in enumerate(dataset.iter_batches(
        "validation",
        BATCH_SIZE,
        shuffle=False,
        seed=0,
    ), start=1):
        history_array = history.numpy()[:, :, 0]
        future_array = future.numpy()
        weight_array = weights.numpy()
        targets = horizon_targets(future_array)
        validation_rows += history_array.shape[0]
        for horizon_index, horizon in enumerate(HORIZONS):
            target = targets[:, horizon_index]
            zero_metrics[horizon].add(
                np.zeros_like(target),
                target,
                weight_array,
            )
            mean_metrics[horizon].add(
                np.full_like(target, target_means[horizon_index]),
                target,
                weight_array,
            )
        for aggregation in AGGREGATIONS:
            features = regression_features(history_array, aggregation)
            last_block = history_array[:, -aggregation:].sum(axis=1)
            linear_prediction = features @ ols_coefficients[aggregation]
            ridge_prediction = features @ ridge_coefficients[aggregation]
            for horizon_index, horizon in enumerate(HORIZONS):
                target = targets[:, horizon_index]
                predictions = {
                    "persistence": last_block * (horizon / aggregation),
                    "linear": linear_prediction[:, horizon_index],
                    "ridge": ridge_prediction[:, horizon_index],
                }
                for model, prediction in predictions.items():
                    metrics[(aggregation, horizon, model)].add(
                        prediction,
                        target,
                        weight_array,
                    )
        if batch_index % 20 == 0:
            print(
                f"validation metrics: {validation_rows:,}/"
                f"{compact_counts['validation']:,}",
                file=sys.stderr,
                flush=True,
            )
    if validation_rows != compact_counts["validation"]:
        raise RuntimeError("validation compact count changed during the audit")

    cells = []
    for aggregation in AGGREGATIONS:
        for horizon_index, horizon in enumerate(HORIZONS):
            zero = zero_metrics[horizon].metrics()
            mean = mean_metrics[horizon].metrics()
            model_metrics = {}
            for model in ("persistence", "linear", "ridge"):
                values = metrics[(aggregation, horizon, model)].metrics()
                values.update({
                    "normalizedMseVsZero": values["mse"] / zero["mse"],
                    "mseImprovementVsZero": 1 - values["mse"] / zero["mse"],
                    "mseImprovementVsTrainMean": 1 - values["mse"] / mean["mse"],
                    "directionLiftVsTrainMean": (
                        values["directionAccuracy"] - mean["directionAccuracy"]
                    ),
                })
                model_metrics[model] = values
            cells.append({
                "aggregationMinutes": aggregation,
                "horizonMinutes": horizon,
                "trainSelectedRidge": selected_ridges[aggregation][horizon_index],
                "targetTrainMean": float(target_means[horizon_index]),
                "zeroMse": zero["mse"],
                "trainMeanMse": mean["mse"],
                "trainMeanDirectionAccuracy": mean["directionAccuracy"],
                "models": model_metrics,
            })
    result = {
        "schemaVersion": 1,
        "accessContract": {
            "testExamplesSelected": 0,
            "testPayloadsRead": 0,
            "gpuUsed": False,
        },
        "corpus": {
            "historyMinutes": HISTORY_MINUTES,
            "maximumTargetMinutes": max(HORIZONS),
            "crossSplitPurgeMs": EXAMPLE_SPAN_MS,
            "trainLogicalRows": sum(shard.count for shard in segments["train"]),
            "validationLogicalRows": sum(
                shard.count for shard in segments["validation"]
            ),
            "trainCompactRows": compact_counts["train"],
            "validationCompactRows": compact_counts["validation"],
            "ridgeFitCompactRows": train_fit_rows,
            "ridgeCalibrationCompactRows": (
                compact_counts["train"] - train_fit_rows
            ),
            "metricWeighting": "original one-second multiplicity",
        },
        "featureContract": {
            "aggregationMinutes": list(AGGREGATIONS),
            "recentBlockHistoryMinutes": RECENT_MINUTES,
            "longTrailingSummaryMinutes": list(LONG_SUMMARY_MINUTES),
            "target": "cumulative next-H-minute close log return",
            "persistence": "repeat latest aggregate return at a constant rate",
            "linear": "multivariate ordinary least squares with intercept",
            "ridge": (
                "same causal regressors; diagonal-scale ridge selected on final "
                "20% of training chronology"
            ),
        },
        "cells": cells,
    }
    print(json.dumps(result, indent=2, allow_nan=False))


def feature_dimension(aggregation: int) -> int:
    return 1 + RECENT_MINUTES // aggregation + len(LONG_SUMMARY_MINUTES)


def regression_features(history: np.ndarray, aggregation: int) -> np.ndarray:
    if history.shape[1] != HISTORY_MINUTES \
            or HISTORY_MINUTES % aggregation != 0 \
            or RECENT_MINUTES % aggregation != 0:
        raise ValueError("aggregate regression history is invalid")
    blocks = history.reshape(
        history.shape[0],
        HISTORY_MINUTES // aggregation,
        aggregation,
    ).sum(axis=2, dtype=np.float64)
    recent = blocks[:, -(RECENT_MINUTES // aggregation):]
    summaries = np.column_stack(tuple(
        history[:, -minutes:].sum(axis=1, dtype=np.float64)
        for minutes in LONG_SUMMARY_MINUTES
    ))
    return np.column_stack((
        np.ones(history.shape[0], dtype=np.float64),
        recent,
        summaries,
    ))


def horizon_targets(future: np.ndarray) -> np.ndarray:
    return np.column_stack(tuple(
        future[:, :horizon].sum(axis=1, dtype=np.float64)
        for horizon in HORIZONS
    ))


def solve(statistics: RegressionStatistics, ridge: float) -> np.ndarray:
    matrix = statistics.xtx.copy()
    if ridge > 0:
        penalty = np.diag(matrix).copy()
        penalty[0] = 0
        matrix += np.diag(penalty * ridge)
    return np.linalg.lstsq(matrix, statistics.xty, rcond=1e-12)[0]


def sufficient_mse(
    statistics: RegressionStatistics,
    coefficients: np.ndarray,
    horizon_index: int,
) -> float:
    cross = statistics.xty[:, horizon_index]
    square_error = (
        statistics.y_square_sum[horizon_index]
        - 2 * coefficients @ cross
        + coefficients @ statistics.xtx @ coefficients
    )
    return max(0.0, float(square_error / statistics.weight))


if __name__ == "__main__":
    main()
