"""Evaluate deployable point forecasts from the hierarchical one-second ensemble.

The fitted process is probabilistic: it produces coherent future paths, not a
single privileged path.  This audit converts the paths to common point
summaries (mean, median, a local-density mode, and a forecast-only medoid) and
scores them on the untouched holdout.  All choices are frozen from the prior
hierarchical-process report; no realized holdout value is used to choose a
member, estimator, threshold, or model parameter.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path

import numpy as np

from analyze_one_second_dependence_model import selected_files
from analyze_parametric_one_second_process import (
    DAY_SECONDS,
    gaussianize_quantile_spline,
    generate_projected_second_batch,
)
from evaluate_full_hierarchical_one_second_process import (
    BOTTOM_UP_LEVELS,
    FEATURES,
    LEVEL_MINUTES,
    MINUTES_PER_DAY,
    RNG_SEED,
    aggregate_leaves,
    deserialize_fit,
    feasible_minute_targets,
    gaussian_member_ranks,
    joint_couple_leaf_features,
    match_leaf_marginals,
    rank_couple_parent,
    reconcile_immediate_children,
    round_activity_counts,
)
from trading_storage import read_candle_column


REPORT_PATH = "data/benchmarks/full-hierarchical-one-second-process.json"
FORECAST_CACHE_PATH = "data/benchmarks/full-hierarchy-calibrated-forecasts.npz"
FIT_CACHE_PATH = "data/benchmarks/full-hierarchy-one-second-fit.json"
LEAF_CACHE_PATH = "data/benchmarks/full-hierarchy-point-forecast-leaves.npz"
OUTPUT_PATH = "data/benchmarks/hierarchical-point-forecasts.json"
DOC_PATH = "docs/experiments/hierarchical-point-forecasts-2026-08-13.md"
MODE_NEIGHBORS = 4

SCALE_SECONDS = {
    "1s": 1,
    "1m": 60,
    "15m": 15 * 60,
    "30m": 30 * 60,
    "1h": 60 * 60,
    "2h": 2 * 60 * 60,
    "4h": 4 * 60 * 60,
    "8h": 8 * 60 * 60,
    "1d": DAY_SECONDS,
}
HORIZON_SECONDS = {
    "15m": 15 * 60,
    "30m": 30 * 60,
    "1h": 60 * 60,
    "2h": 2 * 60 * 60,
    "4h": 4 * 60 * 60,
    "8h": 8 * 60 * 60,
    "1d": DAY_SECONDS,
}
LEAD_BUCKETS = (
    ("0-15m", 0, 15 * 60),
    ("15-30m", 15 * 60, 30 * 60),
    ("30m-1h", 30 * 60, 60 * 60),
    ("1-2h", 60 * 60, 2 * 60 * 60),
    ("2-4h", 2 * 60 * 60, 4 * 60 * 60),
    ("4-8h", 4 * 60 * 60, 8 * 60 * 60),
    ("8-24h", 8 * 60 * 60, DAY_SECONDS),
)
RAW_ESTIMATORS = (
    "ensembleMean",
    "ensembleMedian",
    "localDensityMode",
    "pathMedoid",
    "zeroReturnBaseline",
)
SELECTED_BLEND = "validationSelectedShrinkageBlend"
SCORED_ESTIMATORS = (
    "ensembleMean",
    "ensembleMedian",
    "localDensityMode",
    "pathMedoid",
    SELECTED_BLEND,
    "zeroReturnBaseline",
)
BLEND_FEATURES = (
    "ensembleMean",
    "ensembleMedian",
    "localDensityMode",
)
BLEND_WEIGHTS = {
    "mean": (1.0, 0.0, 0.0),
    "median": (0.0, 1.0, 0.0),
    "mode": (0.0, 0.0, 1.0),
    "meanMedian": (0.5, 0.5, 0.0),
    "meanMode": (0.5, 0.0, 0.5),
    "medianMode": (0.0, 0.5, 0.5),
    "equal": (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
    "meanHeavy": (0.5, 0.25, 0.25),
    "medianHeavy": (0.25, 0.5, 0.25),
    "modeHeavy": (0.25, 0.25, 0.5),
}
MINIMUM_BLEND_VALIDATION_GAIN = 0.001


@dataclass
class RegressionAccumulator:
    count: int = 0
    sum_actual: float = 0.0
    sum_prediction: float = 0.0
    sum_actual_squared: float = 0.0
    sum_prediction_squared: float = 0.0
    sum_cross: float = 0.0
    sum_absolute_actual: float = 0.0
    sum_absolute_error: float = 0.0
    sum_squared_error: float = 0.0
    sign_correct: int = 0
    active_actual: int = 0
    sign_correct_active_actual: int = 0
    predicted_nonzero: int = 0

    def finish(self) -> dict:
        n = self.count
        if n == 0:
            raise ValueError("cannot finish an empty regression accumulator")
        covariance = self.sum_cross - self.sum_actual * self.sum_prediction / n
        actual_variation = self.sum_actual_squared - self.sum_actual ** 2 / n
        prediction_variation = (
            self.sum_prediction_squared - self.sum_prediction ** 2 / n
        )
        denominator = math.sqrt(max(actual_variation * prediction_variation, 0.0))
        correlation = covariance / denominator if denominator > 0.0 else None
        zero_mae = self.sum_absolute_actual / n
        zero_rmse_squared = self.sum_actual_squared / n
        mae = self.sum_absolute_error / n
        rmse = math.sqrt(self.sum_squared_error / n)
        return {
            "observations": n,
            "pearsonCorrelation": correlation,
            "meanActualBps": self.sum_actual / n,
            "meanPredictionBps": self.sum_prediction / n,
            "maeBps": mae,
            "rmseBps": rmse,
            "maeSkillVsZeroReturn": 1.0 - mae / zero_mae if zero_mae > 0.0 else None,
            "mseSkillVsZeroReturn": (
                1.0 - (self.sum_squared_error / n) / zero_rmse_squared
                if zero_rmse_squared > 0.0 else None
            ),
            "signAccuracyAll": self.sign_correct / n,
            "signAccuracyWhenActualNonzero": (
                self.sign_correct_active_actual / self.active_actual
                if self.active_actual else None
            ),
            "actualNonzeroRate": self.active_actual / n,
            "forecastNonzeroRate": self.predicted_nonzero / n,
        }

    def add(self, actual: np.ndarray, prediction: np.ndarray) -> None:
        actual = np.asarray(actual, dtype=np.float64).reshape(-1)
        prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
        if actual.shape != prediction.shape:
            raise ValueError("actual and prediction shapes differ")
        error = prediction - actual
        actual_sign = np.sign(actual)
        prediction_sign = np.sign(prediction)
        active = actual_sign != 0.0
        self.count += actual.size
        self.sum_actual += float(np.sum(actual))
        self.sum_prediction += float(np.sum(prediction))
        self.sum_actual_squared += float(np.dot(actual, actual))
        self.sum_prediction_squared += float(np.dot(prediction, prediction))
        self.sum_cross += float(np.dot(actual, prediction))
        self.sum_absolute_actual += float(np.sum(np.abs(actual)))
        self.sum_absolute_error += float(np.sum(np.abs(error)))
        self.sum_squared_error += float(np.dot(error, error))
        self.sign_correct += int(np.sum(actual_sign == prediction_sign))
        self.active_actual += int(np.sum(active))
        self.sign_correct_active_actual += int(np.sum(
            (actual_sign == prediction_sign) & active
        ))
        self.predicted_nonzero += int(np.sum(prediction_sign != 0.0))


@dataclass
class ProbabilisticAccumulator:
    count: int = 0
    crps_sum: float = 0.0
    coverage_counts: dict[str, int] = field(default_factory=lambda: {
        "50": 0,
        "80": 0,
        "90": 0,
    })
    width_sums: dict[str, float] = field(default_factory=lambda: {
        "50": 0.0,
        "80": 0.0,
        "90": 0.0,
    })

    def add(self, actual: np.ndarray, ensemble: np.ndarray) -> np.ndarray:
        actual = np.asarray(actual, dtype=np.float64).reshape(-1)
        ensemble = np.asarray(ensemble, dtype=np.float64)
        if ensemble.ndim != 2 or ensemble.shape[0] != actual.size:
            raise ValueError("ensemble needs observation/member axes")
        ordered = np.sort(ensemble, axis=1)
        members = ensemble.shape[1]
        coefficients = 2.0 * np.arange(1, members + 1) - members - 1.0
        absolute = np.mean(np.abs(ensemble - actual[:, None]), axis=1)
        pairwise_half = ordered @ coefficients / (members * members)
        self.crps_sum += float(np.sum(absolute - pairwise_half))
        self.count += actual.size
        for central in (50, 80, 90):
            tail = (100.0 - central) / 200.0
            lower = sorted_quantile(ordered, tail)
            upper = sorted_quantile(ordered, 1.0 - tail)
            key = str(central)
            self.coverage_counts[key] += int(np.sum(
                (actual >= lower) & (actual <= upper)
            ))
            self.width_sums[key] += float(np.sum(upper - lower))
        return ordered

    def finish(self) -> dict:
        return {
            "observations": self.count,
            "meanCrpsBps": self.crps_sum / self.count,
            "centralIntervals": {
                key: {
                    "nominalCoverage": int(key) / 100.0,
                    "empiricalCoverage": self.coverage_counts[key] / self.count,
                    "meanWidthBps": self.width_sums[key] / self.count,
                }
                for key in ("50", "80", "90")
            },
        }


@dataclass
class BinaryAccumulator:
    bins: int = 17
    count: int = 0
    positives: int = 0
    brier_sum: float = 0.0
    log_loss_sum: float = 0.0
    true_positive: int = 0
    false_positive: int = 0
    false_negative: int = 0
    positive_histogram: np.ndarray = field(default_factory=lambda: np.zeros(17, dtype=np.int64))
    negative_histogram: np.ndarray = field(default_factory=lambda: np.zeros(17, dtype=np.int64))

    def add(self, actual: np.ndarray, probability: np.ndarray) -> None:
        actual = np.asarray(actual, dtype=bool).reshape(-1)
        probability = np.clip(np.asarray(probability, dtype=np.float64).reshape(-1), 0.0, 1.0)
        if actual.shape != probability.shape:
            raise ValueError("binary actual and probability shapes differ")
        epsilon = 1e-12
        prediction = probability >= 0.5
        self.count += actual.size
        self.positives += int(np.sum(actual))
        self.brier_sum += float(np.sum((probability - actual.astype(float)) ** 2))
        self.log_loss_sum += float(-np.sum(
            actual * np.log(np.maximum(probability, epsilon))
            + (~actual) * np.log(np.maximum(1.0 - probability, epsilon))
        ))
        self.true_positive += int(np.sum(prediction & actual))
        self.false_positive += int(np.sum(prediction & ~actual))
        self.false_negative += int(np.sum(~prediction & actual))
        indexes = np.minimum((probability * (self.bins - 1)).round().astype(int), self.bins - 1)
        self.positive_histogram += np.bincount(indexes[actual], minlength=self.bins)
        self.negative_histogram += np.bincount(indexes[~actual], minlength=self.bins)

    def finish(self) -> dict:
        negatives_below = 0
        favorable = 0.0
        for positive, negative in zip(self.positive_histogram, self.negative_histogram):
            favorable += positive * negatives_below + 0.5 * positive * negative
            negatives_below += int(negative)
        negatives = self.count - self.positives
        auc = favorable / (self.positives * negatives) if self.positives and negatives else None
        precision_denominator = self.true_positive + self.false_positive
        recall_denominator = self.true_positive + self.false_negative
        return {
            "observations": self.count,
            "observedActiveRate": self.positives / self.count,
            "brierScore": self.brier_sum / self.count,
            "logLoss": self.log_loss_sum / self.count,
            "approximateRocAuc": auc,
            "threshold": 0.5,
            "precision": self.true_positive / precision_denominator if precision_denominator else None,
            "recall": self.true_positive / recall_denominator if recall_denominator else None,
        }


@dataclass
class QuadraticSufficientStatistics:
    count: int = 0
    gram: np.ndarray = field(default_factory=lambda: np.zeros((3, 3), dtype=np.float64))
    cross: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    actual_squared: float = 0.0

    def add(self, actual: np.ndarray, features: np.ndarray) -> None:
        actual = np.asarray(actual, dtype=np.float64).reshape(-1)
        features = np.asarray(features, dtype=np.float64)
        if features.shape != (actual.size, 3):
            raise ValueError("blend features need observation/three-feature axes")
        self.count += actual.size
        self.gram += features.T @ features
        self.cross += features.T @ actual
        self.actual_squared += float(np.dot(actual, actual))

    def mse(self, coefficients: np.ndarray) -> float:
        if self.count == 0:
            raise ValueError("cannot score empty blend statistics")
        coefficients = np.asarray(coefficients, dtype=np.float64)
        squared_error = (
            self.actual_squared
            - 2.0 * float(coefficients @ self.cross)
            + float(coefficients @ self.gram @ coefficients)
        )
        return max(squared_error, 0.0) / self.count


@dataclass
class BlendCalibrationAccumulator:
    training: QuadraticSufficientStatistics = field(default_factory=QuadraticSufficientStatistics)
    validation: QuadraticSufficientStatistics = field(default_factory=QuadraticSufficientStatistics)
    validation_folds: tuple[QuadraticSufficientStatistics, ...] = field(
        default_factory=lambda: tuple(QuadraticSufficientStatistics() for _ in range(3))
    )

    def add(
        self,
        actual: np.ndarray,
        summaries: dict[str, np.ndarray],
        *,
        phase: str,
        fold: int | None = None,
    ) -> None:
        features = np.column_stack([summaries[name] for name in BLEND_FEATURES])
        if phase == "training":
            self.training.add(actual, features)
            return
        if phase != "validation" or fold is None:
            raise ValueError("validation blend observations require a fold")
        self.validation.add(actual, features)
        self.validation_folds[fold].add(actual, features)


def fit_blend_calibration(values: BlendCalibrationAccumulator) -> dict:
    zero = np.zeros(3, dtype=np.float64)
    baseline_mse = values.validation.mse(zero)
    candidates = {
        "zero": {
            "coefficients": zero.tolist(),
            "trainingShrinkage": 0.0,
            "validationMse": baseline_mse,
            "validationMseGainVsZero": 0.0,
            "foldMseGainsVsZero": [0.0] * len(values.validation_folds),
            "stable": True,
        }
    }
    stable = ["zero"]
    for name, raw_weights in BLEND_WEIGHTS.items():
        weights = np.asarray(raw_weights, dtype=np.float64)
        denominator = float(weights @ values.training.gram @ weights)
        numerator = float(weights @ values.training.cross)
        shrinkage = float(np.clip(numerator / denominator, 0.0, 1.0)) if denominator > 0 else 0.0
        coefficients = shrinkage * weights
        validation_mse = values.validation.mse(coefficients)
        gain = 1.0 - validation_mse / baseline_mse if baseline_mse > 0 else 0.0
        fold_gains = []
        for fold in values.validation_folds:
            fold_baseline = fold.mse(zero)
            fold_mse = fold.mse(coefficients)
            fold_gains.append(
                1.0 - fold_mse / fold_baseline if fold_baseline > 0 else 0.0
            )
        is_stable = gain >= MINIMUM_BLEND_VALIDATION_GAIN and min(fold_gains) >= 0.0
        candidates[name] = {
            "coefficients": coefficients.tolist(),
            "trainingShrinkage": shrinkage,
            "validationMse": validation_mse,
            "validationMseGainVsZero": gain,
            "foldMseGainsVsZero": fold_gains,
            "stable": is_stable,
        }
        if is_stable:
            stable.append(name)
    selected = min(stable, key=lambda name: candidates[name]["validationMse"])
    return {
        "selectedCandidate": selected,
        "selectedCoefficients": candidates[selected]["coefficients"],
        "featureOrder": list(BLEND_FEATURES),
        "minimumValidationMseGain": MINIMUM_BLEND_VALIDATION_GAIN,
        "requiredNonnegativeGainInEveryFold": True,
        "trainingObservations": values.training.count,
        "validationObservations": values.validation.count,
        "candidateScores": candidates,
    }


def sorted_quantile(ordered: np.ndarray, probability: float) -> np.ndarray:
    position = probability * (ordered.shape[1] - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    fraction = position - lower
    return ordered[:, lower] * (1.0 - fraction) + ordered[:, upper] * fraction


def local_density_mode(
    ensemble: np.ndarray,
    *,
    neighbors: int = MODE_NEIGHBORS,
    ordered: np.ndarray | None = None,
) -> np.ndarray:
    """Estimate the row mode by the center of the densest k-sample interval.

    Exact zero mass is handled explicitly.  This is more suitable than a KDE
    at one second, where the distribution has a true atom at zero.
    """
    ensemble = np.asarray(ensemble, dtype=np.float64)
    if ensemble.ndim != 2:
        raise ValueError("mode estimator needs observation/member axes")
    ordered = np.sort(ensemble, axis=1) if ordered is None else ordered
    k = min(max(2, neighbors), ensemble.shape[1])
    widths = ordered[:, k - 1:] - ordered[:, :ordered.shape[1] - k + 1]
    starts = np.argmin(widths, axis=1)
    offsets = np.arange(k)
    windows = ordered[np.arange(ordered.shape[0])[:, None], starts[:, None] + offsets]
    result = np.mean(windows, axis=1)
    zero_count = np.sum(ensemble == 0.0, axis=1)
    result[zero_count >= k] = 0.0
    return result


def aggregate_seconds(values: np.ndarray, seconds: int) -> np.ndarray:
    values = np.asarray(values)
    if values.shape[-1] != DAY_SECONDS or DAY_SECONDS % seconds:
        raise ValueError("values must contain a complete day divisible by scale")
    return values.reshape(*values.shape[:-1], DAY_SECONDS // seconds, seconds).sum(axis=-1)


def forecast_only_medoid_member(second_paths: np.ndarray) -> int:
    """Choose the sampled path closest to the ensemble center without outcomes."""
    minute = aggregate_seconds(second_paths, 60)
    cumulative = np.cumsum(minute, axis=1)
    center = np.mean(cumulative, axis=0)
    scale = np.std(cumulative, axis=0)
    floor = max(float(np.median(scale[scale > 0.0])) * 0.1, 1e-6) if np.any(scale > 0.0) else 1.0
    distance = np.mean(((cumulative - center) / np.maximum(scale, floor)) ** 2, axis=1)
    return int(np.argmin(distance))


def point_summaries(
    ensemble: np.ndarray,
    medoid_member: int,
    *,
    ordered: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    ensemble = np.asarray(ensemble, dtype=np.float64)
    ordered = np.sort(ensemble, axis=1) if ordered is None else ordered
    return {
        "ensembleMean": np.mean(ensemble, axis=1),
        "ensembleMedian": sorted_quantile(ordered, 0.5),
        "localDensityMode": local_density_mode(ensemble, ordered=ordered),
        "pathMedoid": ensemble[:, medoid_member],
        "zeroReturnBaseline": np.zeros(ensemble.shape[0], dtype=np.float64),
    }


def blend_prediction(
    summaries: dict[str, np.ndarray],
    calibration: dict,
) -> np.ndarray:
    coefficients = np.asarray(calibration["selectedCoefficients"], dtype=np.float64)
    features = np.column_stack([summaries[name] for name in BLEND_FEATURES])
    return features @ coefficients


def factor_copula_scores_with_day_offset(
    fit,
    *,
    days: int,
    steps: int,
    members: int,
    seed: int,
    day_offset: int,
) -> np.ndarray:
    result = np.empty((days, steps, members), dtype=np.float64)
    phi = np.exp(-1.0 / fit.timescales)
    innovation_scale = np.sqrt(1.0 - phi * phi)
    observation_weights = np.sqrt(np.maximum(fit.weights, 0.0))
    for local_day in range(days):
        rng = np.random.default_rng(seed + day_offset + local_day)
        state = rng.standard_normal((members, fit.timescales.size))
        for step in range(steps):
            state = (
                phi[None, :] * state
                + innovation_scale[None, :] * rng.standard_normal((members, fit.timescales.size))
            )
            result[local_day, step] = state @ observation_weights
            if fit.white_variance > 0:
                result[local_day, step] += math.sqrt(fit.white_variance) * rng.standard_normal(members)
    return result


def apply_frozen_hierarchy(
    calibrated: dict,
    fitted,
    process_report: dict,
    *,
    day_offset: int,
) -> dict[str, np.ndarray]:
    """Apply only architecture decisions selected before the holdout."""
    reconciliation = process_report["hierarchicalReconciliation"]
    output = {}
    for feature in FEATURES:
        details = reconciliation[feature]
        fallback = calibrated["1m"][feature].astype(np.float64)
        leaves = fallback
        mode = details["featureGate"]["selectedMode"]
        if mode != "bottomUpMinuteFallback":
            for level in BOTTOM_UP_LEVELS:
                weight = float(details[level]["effectiveParentWeight"])
                child_sum = aggregate_leaves(leaves, level)
                parent = calibrated[level][feature].astype(np.float64)
                coupled = rank_couple_parent(parent, child_sum)
                target = weight * coupled + (1.0 - weight) * child_sum
                leaves = reconcile_immediate_children(
                    leaves,
                    target,
                    parent_level=level,
                    feature=feature,
                )
            if mode == "copulaOnlyParentConditioning":
                leaves = match_leaf_marginals(leaves, fallback)
        else:
            leaves = fallback
        output[feature] = leaves.astype(np.float32)

    volatility = reconciliation["oneSecondRealizedVarianceBpsSquared"].get(
        "volatilityCopula", {}
    )
    weight = float(volatility.get("selectedWeight", 0.0))
    if weight > 0.0:
        leaves = output["oneSecondRealizedVarianceBpsSquared"]
        latent = factor_copula_scores_with_day_offset(
            fitted.volatility_factors,
            days=leaves.shape[0],
            steps=leaves.shape[1],
            members=leaves.shape[2],
            seed=RNG_SEED + 1_100_000,
            day_offset=day_offset,
        )
        existing = gaussian_member_ranks(leaves)
        dependence = math.sqrt(max(0.0, 1.0 - weight * weight)) * existing + weight * latent
        output["oneSecondRealizedVarianceBpsSquared"] = match_leaf_marginals(
            dependence, leaves
        )
    output, _ = joint_couple_leaf_features(output)
    return output


def load_calibrated_holdout(
    cache_path: Path,
    *,
    start: int,
    end: int | None = None,
) -> tuple[dict, dict]:
    with np.load(cache_path, allow_pickle=False) as cache:
        metadata = json.loads(str(cache["metadataJson"].item()))
        calibrated = {
            level: {
                feature: cache[f"calibrated__{level}__{feature}"][start:end].astype(np.float32)
                for feature in FEATURES
            }
            for level in LEVEL_MINUTES
        }
    return calibrated, metadata


def load_or_build_holdout_leaves(
    cache_path: Path,
    *,
    calibrated: dict,
    fitted,
    process_report: dict,
    holdout_start_index: int,
) -> dict[str, np.ndarray]:
    expected = {
        "version": 1,
        "sourceReportGeneratedAt": process_report["generatedAt"],
        "holdoutStartIndex": holdout_start_index,
        "days": calibrated["1m"]["periodReturnBps"].shape[0],
        "members": calibrated["1m"]["periodReturnBps"].shape[2],
    }
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as cache:
            metadata = json.loads(str(cache["metadataJson"].item()))
            if all(metadata.get(key) == value for key, value in expected.items()):
                print("Loading cached frozen holdout leaves...", flush=True)
                return {feature: cache[feature].astype(np.float32) for feature in FEATURES}
    print("Applying frozen hierarchical architecture to holdout leaves...", flush=True)
    leaves = apply_frozen_hierarchy(
        calibrated,
        fitted,
        process_report,
        day_offset=holdout_start_index,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, metadataJson=json.dumps(expected), **leaves)
    return leaves


def generate_day_paths(
    *,
    fitted,
    calibration: dict,
    leaves: dict[str, np.ndarray],
    day: int,
    rng_seed: int,
) -> tuple[np.ndarray, dict]:
    members = leaves["periodReturnBps"].shape[2]
    paths = np.empty((members, DAY_SECONDS), dtype=np.float32)
    feasibility_added = 0.0
    feasibility_input = 0.0
    for member in range(members):
        counts = round_activity_counts(leaves["activeSeconds"][day, :, member])
        target_return, variance, feasibility = feasible_minute_targets(
            leaves["periodReturnBps"][day, :, member],
            leaves["oneSecondRealizedVarianceBpsSquared"][day, :, member],
            counts,
        )
        volatility_score = gaussianize_quantile_spline(
            np.log(variance + fitted.variance_floor),
            fitted.variance_quantile_knots,
            fitted.variance_log_quantiles,
        )
        _, generated = generate_projected_second_batch(
            fitted,
            counts=counts,
            volatility_score=volatility_score,
            target_minute_returns=target_return,
            realized_variance=variance,
            activity_timing_rho=float(calibration["activityTimingGaussianRho"]),
            magnitude_dispersion_scale=float(calibration["magnitudeDispersionScale"]),
            micro_probability_scale=float(calibration["microProbabilityScale"]),
            rng=np.random.default_rng(rng_seed + day * 10_007 + member),
            sign_timing_rho=float(calibration.get("signTimingGaussianRho", 0.0)),
            magnitude_share_score_rho=float(calibration.get(
                "magnitudeShareScoreRho",
                fitted.magnitude_mixture.share_score_rho,
            )),
        )
        paths[member] = generated.reshape(-1).astype(np.float32)
        feasibility_added += float(feasibility["addedVarianceBpsSquared"])
        feasibility_input += float(feasibility["inputVarianceTotalBpsSquared"])
    return paths, {
        "addedVarianceBpsSquared": feasibility_added,
        "inputVarianceBpsSquared": feasibility_input,
    }


def read_actual_returns(reference: Path, previous_close: float) -> tuple[np.ndarray, float]:
    closes = read_candle_column(reference, "close").astype(np.float64)
    actual = np.diff(np.log(np.concatenate((
        np.asarray([previous_close], dtype=np.float64), closes,
    )))) * 10_000.0
    if actual.size != DAY_SECONDS:
        raise ValueError(f"{reference} does not contain a complete UTC day")
    return actual, float(closes[-1])


def safe_correlation(actual: np.ndarray, prediction: np.ndarray) -> float | None:
    actual = np.asarray(actual, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if np.std(actual) == 0.0 or np.std(prediction) == 0.0:
        return None
    return float(np.corrcoef(actual, prediction)[0, 1])


def calibrate_point_blends(
    *,
    repo: Path,
    process_report: dict,
    forecast_cache_path: Path,
    leaf_cache_path: Path,
    source: Path,
    fitted,
    calibration: dict,
) -> dict:
    """Fit summary weights before the untouched holdout and freeze them."""
    calibration_origins = int(process_report["design"]["calibrationOrigins"])
    validation_end = (
        calibration_origins
        + int(process_report["design"]["hierarchyArchitectureSelectionOrigins"])
    )
    training_start = calibration_origins // 2
    training_end = calibration_origins
    calibrated, _ = load_calibrated_holdout(
        forecast_cache_path,
        start=training_start,
        end=validation_end,
    )
    leaves = load_or_build_holdout_leaves(
        leaf_cache_path,
        calibrated=calibrated,
        fitted=fitted,
        process_report=process_report,
        holdout_start_index=training_start,
    )
    forecast_start = datetime.fromisoformat(
        process_report["design"]["forecastOriginStart"].replace("Z", "+00:00")
    )
    period_start = forecast_start + timedelta(days=training_start)
    period_end = forecast_start + timedelta(days=validation_end)
    files = selected_files(source, period_start, period_end)
    if len(files) != validation_end - training_start:
        raise RuntimeError("point-calibration second files are incomplete")
    prior = selected_files(source, period_start - timedelta(days=1), period_start)
    previous_close = float(read_candle_column(prior[-1], "close")[-1])
    scale_values = {
        scale: BlendCalibrationAccumulator() for scale in SCALE_SECONDS
    }
    horizon_values = {
        horizon: BlendCalibrationAccumulator() for horizon in HORIZON_SECONDS
    }
    validation_days = validation_end - training_end
    for local_day, reference in enumerate(files):
        if local_day % 20 == 0:
            print(f"Point blend calibration {local_day}/{len(files)}...", flush=True)
        absolute_day = training_start + local_day
        phase = "training" if absolute_day < training_end else "validation"
        fold = None
        if phase == "validation":
            fold = min(
                2,
                3 * (absolute_day - training_end) // validation_days,
            )
        actual_second, previous_close = read_actual_returns(reference, previous_close)
        paths, _ = generate_day_paths(
            fitted=fitted,
            calibration=calibration,
            leaves=leaves,
            day=local_day,
            rng_seed=RNG_SEED + 2_500_000 + training_start * 10_007,
        )
        medoid_member = forecast_only_medoid_member(paths)
        for scale, seconds in SCALE_SECONDS.items():
            actual_values = (
                actual_second
                if seconds == 1
                else aggregate_seconds(actual_second[None, :], seconds)[0]
            )
            ensemble_values = (
                paths.T.astype(np.float64)
                if seconds == 1
                else aggregate_seconds(paths, seconds).T.astype(np.float64)
            )
            summaries = point_summaries(ensemble_values, medoid_member)
            scale_values[scale].add(
                actual_values,
                summaries,
                phase=phase,
                fold=fold,
            )
        for horizon, seconds in HORIZON_SECONDS.items():
            actual_value = np.asarray([np.sum(actual_second[:seconds])])
            ensemble_values = np.sum(
                paths[:, :seconds], axis=1, dtype=np.float64
            )[None, :]
            summaries = point_summaries(ensemble_values, medoid_member)
            horizon_values[horizon].add(
                actual_value,
                summaries,
                phase=phase,
                fold=fold,
            )
    return {
        "design": {
            "trainingStart": (forecast_start + timedelta(days=training_start)).isoformat().replace("+00:00", "Z"),
            "trainingEndExclusive": (forecast_start + timedelta(days=training_end)).isoformat().replace("+00:00", "Z"),
            "validationStart": (forecast_start + timedelta(days=training_end)).isoformat().replace("+00:00", "Z"),
            "validationEndExclusive": (forecast_start + timedelta(days=validation_end)).isoformat().replace("+00:00", "Z"),
            "trainingDays": training_end - training_start,
            "validationDays": validation_days,
            "validationFolds": 3,
            "historicalOutcomesUsedForCoefficientFittingAndValidation": True,
            "outcomesAfterValidationEndUsed": False,
            "untouchedHoldoutOutcomesUsed": False,
            "constraint": "nonnegative convex summary weights followed by shrinkage in [0, 1]",
        },
        "byScale": {
            scale: fit_blend_calibration(values)
            for scale, values in scale_values.items()
        },
        "byCumulativeHorizon": {
            horizon: fit_blend_calibration(values)
            for horizon, values in horizon_values.items()
        },
    }


def evaluate(args: argparse.Namespace) -> dict:
    repo = Path(__file__).resolve().parents[1]
    process_report = json.loads((repo / args.process_report).read_text(encoding="utf-8"))
    fit_cache = json.loads((repo / args.fit_cache).read_text(encoding="utf-8"))
    fitted = deserialize_fit(fit_cache["fittedParameters"])
    calibration = process_report["oneSecondKernel"]["intradayCalibration"]
    holdout_start_index = (
        process_report["design"]["calibrationOrigins"]
        + process_report["design"]["hierarchyArchitectureSelectionOrigins"]
    )
    calibrated, forecast_metadata = load_calibrated_holdout(
        repo / args.forecast_cache,
        start=holdout_start_index,
    )
    leaves = load_or_build_holdout_leaves(
        repo / args.leaf_cache,
        calibrated=calibrated,
        fitted=fitted,
        process_report=process_report,
        holdout_start_index=holdout_start_index,
    )
    holdout_start = datetime.fromisoformat(
        process_report["design"]["untouchedTestStart"].replace("Z", "+00:00")
    )
    holdout_end = datetime.fromisoformat(
        process_report["design"]["forecastOriginEndExclusive"].replace("Z", "+00:00")
    )
    analysis = json.loads((repo / args.analysis).read_text(encoding="utf-8"))
    source = repo / analysis["source"]["oneSecond"]["referenceDirectory"]
    blend_calibration = calibrate_point_blends(
        repo=repo,
        process_report=process_report,
        forecast_cache_path=repo / args.forecast_cache,
        leaf_cache_path=repo / args.blend_leaf_cache,
        source=source,
        fitted=fitted,
        calibration=calibration,
    )
    files = selected_files(source, holdout_start, holdout_end)
    if len(files) != leaves["periodReturnBps"].shape[0]:
        raise RuntimeError("holdout files and frozen forecasts have different day counts")
    previous_file = selected_files(source, holdout_start - timedelta(days=1), holdout_start)
    previous_close = float(read_candle_column(previous_file[-1], "close")[-1])

    return_metrics = {
        scale: {estimator: RegressionAccumulator() for estimator in SCORED_ESTIMATORS}
        for scale in SCALE_SECONDS
    }
    probabilistic_metrics = {scale: ProbabilisticAccumulator() for scale in SCALE_SECONDS}
    absolute_metrics = {scale: RegressionAccumulator() for scale in SCALE_SECONDS}
    squared_metrics = {scale: RegressionAccumulator() for scale in SCALE_SECONDS}
    horizon_return = {
        horizon: {estimator: RegressionAccumulator() for estimator in SCORED_ESTIMATORS}
        for horizon in HORIZON_SECONDS
    }
    horizon_probability = {
        horizon: ProbabilisticAccumulator() for horizon in HORIZON_SECONDS
    }
    lead_return = {
        name: {
            estimator: RegressionAccumulator()
            for estimator in (*BLEND_FEATURES, SELECTED_BLEND)
        }
        for name, _, _ in LEAD_BUCKETS
    }
    lead_absolute = {name: RegressionAccumulator() for name, _, _ in LEAD_BUCKETS}
    activity = BinaryAccumulator()
    cumulative_path = {
        estimator: RegressionAccumulator() for estimator in RAW_ESTIMATORS[:4]
    }
    daily_path_stats = {estimator: [] for estimator in RAW_ESTIMATORS[:4]}
    medoid_indexes = []
    total_added_variance = 0.0
    total_input_variance = 0.0

    for day, reference in enumerate(files):
        if day % 10 == 0:
            print(f"Point forecast evaluation {day}/{len(files)}...", flush=True)
        actual_second, previous_close = read_actual_returns(reference, previous_close)
        paths, feasibility = generate_day_paths(
            fitted=fitted,
            calibration=calibration,
            leaves=leaves,
            day=day,
            rng_seed=RNG_SEED + 2_500_000,
        )
        total_added_variance += feasibility["addedVarianceBpsSquared"]
        total_input_variance += feasibility["inputVarianceBpsSquared"]
        medoid_member = forecast_only_medoid_member(paths)
        medoid_indexes.append(medoid_member)

        second_rows = paths.T.astype(np.float64)
        second_ordered = probabilistic_metrics["1s"].add(actual_second, second_rows)
        second_points = point_summaries(
            second_rows,
            medoid_member,
            ordered=second_ordered,
        )
        for estimator, prediction in second_points.items():
            return_metrics["1s"][estimator].add(actual_second, prediction)
        second_blend = blend_prediction(
            second_points,
            blend_calibration["byScale"]["1s"],
        )
        return_metrics["1s"][SELECTED_BLEND].add(actual_second, second_blend)
        expected_absolute = np.mean(np.abs(second_rows), axis=1)
        expected_squared = np.mean(second_rows * second_rows, axis=1)
        absolute_metrics["1s"].add(np.abs(actual_second), expected_absolute)
        squared_metrics["1s"].add(actual_second * actual_second, expected_squared)
        activity.add(actual_second != 0.0, np.mean(second_rows != 0.0, axis=1))
        for name, start, stop in LEAD_BUCKETS:
            for estimator in BLEND_FEATURES:
                lead_return[name][estimator].add(
                    actual_second[start:stop], second_points[estimator][start:stop]
                )
            lead_return[name][SELECTED_BLEND].add(
                actual_second[start:stop], second_blend[start:stop]
            )
            lead_absolute[name].add(
                np.abs(actual_second[start:stop]), expected_absolute[start:stop]
            )

        for scale, seconds in SCALE_SECONDS.items():
            if scale == "1s":
                continue
            actual_values = aggregate_seconds(actual_second[None, :], seconds)[0]
            ensemble_values = aggregate_seconds(paths, seconds).T.astype(np.float64)
            ordered = probabilistic_metrics[scale].add(actual_values, ensemble_values)
            summaries = point_summaries(ensemble_values, medoid_member, ordered=ordered)
            for estimator, prediction in summaries.items():
                return_metrics[scale][estimator].add(actual_values, prediction)
            return_metrics[scale][SELECTED_BLEND].add(
                actual_values,
                blend_prediction(summaries, blend_calibration["byScale"][scale]),
            )
            absolute_metrics[scale].add(
                np.abs(actual_values), np.mean(np.abs(ensemble_values), axis=1)
            )
            squared_metrics[scale].add(
                actual_values * actual_values,
                np.mean(ensemble_values * ensemble_values, axis=1),
            )

        for horizon, seconds in HORIZON_SECONDS.items():
            actual_value = np.asarray([np.sum(actual_second[:seconds])])
            ensemble_values = np.sum(paths[:, :seconds], axis=1, dtype=np.float64)[None, :]
            ordered = horizon_probability[horizon].add(actual_value, ensemble_values)
            summaries = point_summaries(ensemble_values, medoid_member, ordered=ordered)
            for estimator, prediction in summaries.items():
                horizon_return[horizon][estimator].add(actual_value, prediction)
            horizon_return[horizon][SELECTED_BLEND].add(
                actual_value,
                blend_prediction(
                    summaries,
                    blend_calibration["byCumulativeHorizon"][horizon],
                ),
            )

        actual_minute = aggregate_seconds(actual_second[None, :], 60)[0]
        ensemble_minute = aggregate_seconds(paths, 60).T.astype(np.float64)
        actual_cumulative = np.cumsum(actual_minute)
        ensemble_cumulative = np.cumsum(ensemble_minute, axis=0)
        cumulative_summaries = point_summaries(ensemble_cumulative, medoid_member)
        for estimator, prediction in cumulative_summaries.items():
            if estimator not in cumulative_path:
                continue
            cumulative_path[estimator].add(actual_cumulative, prediction)
            error = prediction - actual_cumulative
            daily_path_stats[estimator].append({
                "correlation": safe_correlation(actual_cumulative, prediction),
                "rmseBps": float(np.sqrt(np.mean(error * error))),
                "terminalAbsoluteErrorBps": float(abs(error[-1])),
            })

    def finish_regression(group: dict[str, RegressionAccumulator]) -> dict:
        return {key: accumulator.finish() for key, accumulator in group.items()}

    path_report = {}
    for estimator, accumulator in cumulative_path.items():
        daily = daily_path_stats[estimator]
        correlations = [row["correlation"] for row in daily if row["correlation"] is not None]
        path_report[estimator] = {
            "pooledMinuteEndpointMetrics": accumulator.finish(),
            "meanDailyPathCorrelation": float(np.mean(correlations)) if correlations else None,
            "medianDailyPathCorrelation": float(np.median(correlations)) if correlations else None,
            "meanDailyPathRmseBps": float(np.mean([row["rmseBps"] for row in daily])),
            "medianDailyPathRmseBps": float(np.median([row["rmseBps"] for row in daily])),
            "meanTerminalAbsoluteErrorBps": float(np.mean([
                row["terminalAbsoluteErrorBps"] for row in daily
            ])),
        }

    report = {
        "version": 1,
        "generatedAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "symbol": process_report["symbol"],
        "design": {
            "sourceProcessReport": args.process_report,
            "holdoutStart": process_report["design"]["untouchedTestStart"],
            "holdoutEndExclusive": process_report["design"]["forecastOriginEndExclusive"],
            "holdoutDays": len(files),
            "ensembleMembers": int(leaves["periodReturnBps"].shape[2]),
            "forecastOriginCadence": "one forecast at each UTC day boundary",
            "futureCandlesUsedAtForecastTime": False,
            "holdoutUsedForEstimatorOrMemberSelection": False,
            "estimatorDefinitions": {
                "ensembleMean": "Arithmetic mean across generated paths at the scored candle or horizon; MSE-optimal for a calibrated ensemble.",
                "ensembleMedian": "Cross-path median at the scored candle or horizon; MAE-optimal for a calibrated ensemble.",
                "localDensityMode": f"Center of the narrowest interval containing {MODE_NEIGHBORS} of the 16 paths, with an explicit exact-zero atom.",
                "pathMedoid": "One generated member chosen before observing outcomes by closeness to the ensemble mean cumulative path.",
                SELECTED_BLEND: "A pre-holdout validation-gated nonnegative blend of mean, median, and mode, shrunk toward zero using only earlier outcomes.",
                "zeroReturnBaseline": "Always predict zero log return.",
            },
            "modeIsAValidWholePath": False,
            "medianIsAValidWholePath": False,
            "medoidIsAValidWholePath": True,
            "forecastCacheMetadata": {
                key: forecast_metadata[key]
                for key in ("forecastStart", "forecastEndExclusive", "ensembleSize")
                if key in forecast_metadata
            },
        },
        "pointBlendCalibration": blend_calibration,
        "candleReturnForecastsByScale": {
            scale: {
                "pointEstimators": finish_regression(return_metrics[scale]),
                "probabilisticEnsemble": probabilistic_metrics[scale].finish(),
                "conditionalMagnitude": {
                    "expectedAbsoluteReturn": absolute_metrics[scale].finish(),
                    "expectedSquaredReturn": squared_metrics[scale].finish(),
                },
            }
            for scale in SCALE_SECONDS
        },
        "cumulativeReturnForecastsFromDailyOrigin": {
            horizon: {
                "pointEstimators": finish_regression(horizon_return[horizon]),
                "probabilisticEnsemble": horizon_probability[horizon].finish(),
            }
            for horizon in HORIZON_SECONDS
        },
        "oneSecondForecastByLead": {
            name: {
                "returnPointEstimators": finish_regression(lead_return[name]),
                "expectedAbsoluteReturn": lead_absolute[name].finish(),
            }
            for name, _, _ in LEAD_BUCKETS
        },
        "oneSecondActivityProbability": activity.finish(),
        "cumulativePricePathAtMinuteEndpoints": path_report,
        "generationDiagnostics": {
            "medoidMemberHistogram": np.bincount(
                medoid_indexes,
                minlength=leaves["periodReturnBps"].shape[2],
            ).tolist(),
            "varianceAddedForFeasibilityBpsSquared": total_added_variance,
            "inputVarianceBpsSquared": total_input_variance,
            "relativeVarianceAddedForFeasibility": (
                total_added_variance / total_input_variance
                if total_input_variance > 0.0 else 0.0
            ),
        },
    }
    accepted_scales = [
        scale
        for scale, details in blend_calibration["byScale"].items()
        if details["selectedCandidate"] != "zero"
    ]
    accepted_horizons = [
        horizon
        for horizon, details in blend_calibration["byCumulativeHorizon"].items()
        if details["selectedCandidate"] != "zero"
    ]
    report["conclusions"] = {
        "validatedSignedReturnPointForecastFound": bool(
            accepted_scales or accepted_horizons
        ),
        "acceptedCandleScales": accepted_scales,
        "acceptedCumulativeHorizons": accepted_horizons,
        "recommendedSignedReturnPointForecast": (
            SELECTED_BLEND if accepted_scales or accepted_horizons
            else "zeroReturnBaseline"
        ),
        "interpretation": (
            "No mean/median/mode blend produced a stable pre-holdout MSE gain over "
            "zero in every chronological fold. The fitted process remains useful as "
            "a scenario distribution, but its historical-distribution state does not "
            "identify the realized future return path."
            if not accepted_scales and not accepted_horizons else
            "At least one shrinkage-only ensemble summary passed every pre-holdout fold; "
            "its untouched performance is reported without reselection."
        ),
        "oneSecondRawMeanCorrelation": return_metrics["1s"]["ensembleMean"].finish()[
            "pearsonCorrelation"
        ],
        "oneMinuteExpectedAbsoluteReturnCorrelation": absolute_metrics["1m"].finish()[
            "pearsonCorrelation"
        ],
        "oneSecondActivityProbabilityAuc": activity.finish()["approximateRocAuc"],
        "meanDailyCumulativePathCorrelation": path_report["ensembleMean"][
            "meanDailyPathCorrelation"
        ],
    }
    return report


def render_document(report: dict) -> str:
    conclusions = report["conclusions"]
    one_second = report["candleReturnForecastsByScale"]["1s"]
    one_minute = report["candleReturnForecastsByScale"]["1m"]
    activity = report["oneSecondActivityProbability"]
    gate_detail = (
        "Accepted scales: " + ", ".join(conclusions["acceptedCandleScales"])
        + "; accepted horizons: " + ", ".join(conclusions["acceptedCumulativeHorizons"])
        if conclusions["validatedSignedReturnPointForecastFound"]
        else "Every scale and cumulative horizon selected the zero-return fallback after the pre-holdout fold gate."
    )
    lines = [
        "# Hierarchical ensemble point forecasts",
        "",
        f"Generated: {report['generatedAt']}",
        "",
        "This audit asks whether the fitted probabilistic process can forecast the realized future path, not merely reproduce its unconditional statistics. All forecasts originate at 00:00 UTC; the final 91 days remain untouched by model, estimator, or member selection.",
        "",
        "## Key findings",
        "",
        f"- Validated signed-return point edge found: **{'yes' if conclusions['validatedSignedReturnPointForecastFound'] else 'no'}**. {gate_detail}",
        f"- At 1s, the raw ensemble mean correlation was {one_second['pointEstimators']['ensembleMean']['pearsonCorrelation']:.6f}; its MSE skill versus zero was {one_second['pointEstimators']['ensembleMean']['mseSkillVsZeroReturn']:.6f}.",
        f"- The process contains more scale information than direction information: expected absolute 1m return correlated {one_minute['conditionalMagnitude']['expectedAbsoluteReturn']['pearsonCorrelation']:.6f} with realized absolute return, while 1s activity probability AUC was only {activity['approximateRocAuc']:.6f}.",
        f"- The ensemble-mean cumulative price path had mean within-day correlation {report['cumulativePricePathAtMinuteEndpoints']['ensembleMean']['meanDailyPathCorrelation']:.6f} with the realized path.",
        "- Mean is still the correct ensemble summary for squared-error decisions and median for absolute-error decisions, but here their forecast-only validation says to shrink them completely to zero for signed returns.",
        "",
        "## Results",
        "",
        "| scale | estimator | correlation | MAE (bps) | RMSE (bps) | MSE skill vs zero |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for scale in SCALE_SECONDS:
        estimators = report["candleReturnForecastsByScale"][scale]["pointEstimators"]
        for estimator in (
            "ensembleMean",
            "ensembleMedian",
            "localDensityMode",
            SELECTED_BLEND,
            "pathMedoid",
        ):
            row = estimators[estimator]
            correlation = row["pearsonCorrelation"]
            lines.append(
                f"| {scale} | {estimator} | "
                f"{correlation:.6f} | {row['maeBps']:.6f} | {row['rmseBps']:.6f} | "
                f"{row['mseSkillVsZeroReturn']:.6f} |"
                if correlation is not None else
                f"| {scale} | {estimator} | n/a | {row['maeBps']:.6f} | {row['rmseBps']:.6f} | {row['mseSkillVsZeroReturn']:.6f} |"
            )
    lines += [
        "",
        "## Interpretation",
        "",
        "The ensemble mean is the correct point summary when optimizing squared return error, the median when optimizing absolute error, and the mode only when the most likely local value is the desired decision. At one second the exact-zero atom makes the mode frequently zero. A pointwise median or mode is not necessarily a dynamically valid sampled path; the medoid is included when one coherent representative scenario is required.",
        "",
        "The validation-selected blend is admitted only when its shrinkage-only coefficients beat a zero-return forecast overall and in every pre-holdout chronological validation fold. If no candidate passes, it is exactly the zero-return baseline; this prevents an apparently useful final-holdout correlation from being selected after it is observed.",
        "",
        "Signed-return accuracy and distributional scenario quality answer different questions. Low signed correlation does not invalidate calibrated uncertainty, volatility, or activity forecasts. Conversely, a plausible generated distribution does not establish that a particular realized candle path is predictable.",
        "",
        "Machine-readable results: `data/benchmarks/hierarchical-point-forecasts.json`.",
        "",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--process-report", default=REPORT_PATH)
    parser.add_argument("--analysis", default="data/benchmarks/log-return-distributions.json")
    parser.add_argument("--forecast-cache", default=FORECAST_CACHE_PATH)
    parser.add_argument("--fit-cache", default=FIT_CACHE_PATH)
    parser.add_argument("--leaf-cache", default=LEAF_CACHE_PATH)
    parser.add_argument(
        "--blend-leaf-cache",
        default="data/benchmarks/full-hierarchy-point-blend-leaves.npz",
    )
    parser.add_argument("--output", default=OUTPUT_PATH)
    parser.add_argument("--document", default=DOC_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = Path(__file__).resolve().parents[1]
    report = evaluate(args)
    output = repo / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    document = repo / args.document
    document.parent.mkdir(parents=True, exist_ok=True)
    document.write_text(render_document(report), encoding="utf-8")
    print(output)
    print(document)


if __name__ == "__main__":
    main()
