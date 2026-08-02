from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch
from torch import Tensor

from future_price_resolution_screen import ResolutionSpec


RIDGE_FEATURE_CONTRACT = (
    "causal-6h-5m-audited-linear-features-v1:"
    "recent-12x5m-plus-trailing-2h-3h-6h"
)
RIDGE_FIT_CONTRACT = (
    "weighted-closed-form-standardized-ridge-v2:"
    "chronological-train-internal-calibration-with-history-target-embargo-"
    "and-explicit-training-mean-limit"
)
RIDGE_HISTORY_MINUTES = 360
RIDGE_CANDLE_MINUTES = 5
RIDGE_TARGET_STEPS = 1
RIDGE_HISTORY_STEPS = RIDGE_HISTORY_MINUTES // RIDGE_CANDLE_MINUTES
RIDGE_RECENT_STEPS = 12
RIDGE_SUMMARY_STEPS = (24, 36, 72)
RIDGE_FEATURE_NAMES = tuple(
    f"recent5mLag{lag}"
    for lag in range(RIDGE_RECENT_STEPS, 0, -1)
) + (
    "trailing2hReturn",
    "trailing3hReturn",
    "trailing6hReturn",
)


def validate_ridge_spec(spec: ResolutionSpec) -> None:
    expected = (
        RIDGE_HISTORY_MINUTES,
        RIDGE_CANDLE_MINUTES,
        RIDGE_TARGET_STEPS,
    )
    actual = (
        spec.history_minutes,
        spec.candle_minutes,
        spec.target_steps,
    )
    if actual != expected:
        raise ValueError(
            "audited ridge candidate requires 6h 5m history -> next 5m: "
            f"{actual} != {expected}"
        )


def audited_ridge_features(history: np.ndarray) -> np.ndarray:
    history = np.asarray(history)
    if history.ndim != 2 or history.shape[1] != RIDGE_HISTORY_STEPS:
        raise ValueError("ridge history must contain 72 completed 5m returns")
    if not np.isfinite(history).all():
        raise ValueError("ridge history must be finite")
    recent = history[:, -RIDGE_RECENT_STEPS:]
    summaries = np.column_stack(tuple(
        history[:, -steps:].sum(axis=1, dtype=np.float64)
        for steps in RIDGE_SUMMARY_STEPS
    ))
    return np.column_stack((recent, summaries)).astype(np.float64, copy=False)


def audited_ridge_features_torch(history: Tensor) -> Tensor:
    if history.ndim != 2 or history.shape[1] != RIDGE_HISTORY_STEPS:
        raise ValueError("ridge history must contain 72 completed 5m returns")
    recent = history[:, -RIDGE_RECENT_STEPS:]
    summaries = torch.stack(tuple(
        history[:, -steps:].sum(dim=1)
        for steps in RIDGE_SUMMARY_STEPS
    ), dim=1)
    return torch.cat((recent, summaries), dim=1)


@dataclass
class WeightedRegressionStatistics:
    dimension: int = len(RIDGE_FEATURE_NAMES)

    def __post_init__(self) -> None:
        if self.dimension < 1:
            raise ValueError("regression dimension must be positive")
        self.weight = 0.0
        self.x_sum = np.zeros(self.dimension, dtype=np.float64)
        self.x_square_sum = np.zeros(self.dimension, dtype=np.float64)
        self.xtx = np.zeros((self.dimension, self.dimension), dtype=np.float64)
        self.xty = np.zeros(self.dimension, dtype=np.float64)
        self.y_sum = 0.0
        self.y_square_sum = 0.0

    def add(
        self,
        features: np.ndarray,
        target: np.ndarray,
        weights: np.ndarray,
    ) -> None:
        features = np.asarray(features, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64).reshape(-1)
        weights = np.asarray(weights, dtype=np.float64).reshape(-1)
        if features.shape != (target.shape[0], self.dimension) \
                or weights.shape != target.shape:
            raise ValueError("ridge sufficient statistics are misaligned")
        if not np.isfinite(features).all() \
                or not np.isfinite(target).all() \
                or not np.isfinite(weights).all() \
                or np.any(weights <= 0):
            raise ValueError("ridge sufficient statistics must be finite and positive")
        weighted = features * weights[:, None]
        self.weight += float(weights.sum(dtype=np.float64))
        self.x_sum += weighted.sum(axis=0, dtype=np.float64)
        self.x_square_sum += (
            np.square(features) * weights[:, None]
        ).sum(axis=0, dtype=np.float64)
        self.xtx += features.T @ weighted
        self.xty += features.T @ (weights * target)
        self.y_sum += float((weights * target).sum(dtype=np.float64))
        self.y_square_sum += float(
            (weights * np.square(target)).sum(dtype=np.float64)
        )

    def to_arrays(self, prefix: str) -> dict[str, np.ndarray]:
        return {
            f"{prefix}Weight": np.asarray(self.weight, dtype=np.float64),
            f"{prefix}XSum": self.x_sum,
            f"{prefix}XSquareSum": self.x_square_sum,
            f"{prefix}Xtx": self.xtx,
            f"{prefix}Xty": self.xty,
            f"{prefix}YSum": np.asarray(self.y_sum, dtype=np.float64),
            f"{prefix}YSquareSum": np.asarray(
                self.y_square_sum,
                dtype=np.float64,
            ),
        }

    @classmethod
    def from_arrays(
        cls,
        values: dict[str, np.ndarray],
        prefix: str,
    ) -> WeightedRegressionStatistics:
        result = cls(int(np.asarray(values[f"{prefix}XSum"]).shape[0]))
        result.weight = float(values[f"{prefix}Weight"])
        result.x_sum = np.asarray(values[f"{prefix}XSum"], dtype=np.float64)
        result.x_square_sum = np.asarray(
            values[f"{prefix}XSquareSum"],
            dtype=np.float64,
        )
        result.xtx = np.asarray(values[f"{prefix}Xtx"], dtype=np.float64)
        result.xty = np.asarray(values[f"{prefix}Xty"], dtype=np.float64)
        result.y_sum = float(values[f"{prefix}YSum"])
        result.y_square_sum = float(values[f"{prefix}YSquareSum"])
        return result


@dataclass(frozen=True)
class FeatureScaler:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def from_statistics(
        cls,
        statistics: WeightedRegressionStatistics,
    ) -> FeatureScaler:
        if statistics.weight <= 0:
            raise ValueError("cannot scale empty regression statistics")
        mean = statistics.x_sum / statistics.weight
        variance = np.maximum(
            1e-24,
            statistics.x_square_sum / statistics.weight - np.square(mean),
        )
        return cls(mean=mean, std=np.sqrt(variance))

    def validate(self, dimension: int = len(RIDGE_FEATURE_NAMES)) -> None:
        if self.mean.shape != (dimension,) \
                or self.std.shape != (dimension,) \
                or not np.isfinite(self.mean).all() \
                or not np.isfinite(self.std).all() \
                or np.any(self.std <= 0):
            raise ValueError("ridge feature scaler is invalid")


@dataclass(frozen=True)
class StandardizedRegressionStatistics:
    weight: float
    xtx: np.ndarray
    xty: np.ndarray
    y_sum: float
    y_square_sum: float


def standardize_statistics(
    statistics: WeightedRegressionStatistics,
    scaler: FeatureScaler,
) -> StandardizedRegressionStatistics:
    scaler.validate(statistics.dimension)
    weight = statistics.weight
    z_sum = (statistics.x_sum - weight * scaler.mean) / scaler.std
    centered_xtx = (
        statistics.xtx
        - np.outer(scaler.mean, statistics.x_sum)
        - np.outer(statistics.x_sum, scaler.mean)
        + weight * np.outer(scaler.mean, scaler.mean)
    )
    ztz = centered_xtx / np.outer(scaler.std, scaler.std)
    zty = (statistics.xty - scaler.mean * statistics.y_sum) / scaler.std
    dimension = statistics.dimension + 1
    xtx = np.empty((dimension, dimension), dtype=np.float64)
    xtx[0, 0] = weight
    xtx[0, 1:] = z_sum
    xtx[1:, 0] = z_sum
    xtx[1:, 1:] = ztz
    xty = np.concatenate((np.asarray([statistics.y_sum]), zty))
    xtx = (xtx + xtx.T) * 0.5
    return StandardizedRegressionStatistics(
        weight=weight,
        xtx=xtx,
        xty=xty,
        y_sum=statistics.y_sum,
        y_square_sum=statistics.y_square_sum,
    )


def solve_standardized_ridge(
    statistics: StandardizedRegressionStatistics,
    ridge_lambda: float,
) -> np.ndarray:
    if not math.isfinite(ridge_lambda) or ridge_lambda < 0:
        raise ValueError("ridge lambda must be finite and non-negative")
    matrix = statistics.xtx.copy()
    if ridge_lambda:
        penalty = np.eye(matrix.shape[0], dtype=np.float64)
        penalty[0, 0] = 0.0
        matrix += statistics.weight * ridge_lambda * penalty
    return np.linalg.lstsq(matrix, statistics.xty, rcond=1e-12)[0]


def sufficient_mse(
    statistics: StandardizedRegressionStatistics,
    coefficients: np.ndarray,
) -> float:
    coefficients = np.asarray(coefficients, dtype=np.float64)
    if coefficients.shape != statistics.xty.shape:
        raise ValueError("ridge coefficients are misaligned")
    squared_error = (
        statistics.y_square_sum
        - 2.0 * coefficients @ statistics.xty
        + coefficients @ statistics.xtx @ coefficients
    )
    return max(0.0, float(squared_error / statistics.weight))


def predict_standardized_ridge(
    history: np.ndarray,
    scaler: FeatureScaler,
    coefficients: np.ndarray,
) -> np.ndarray:
    scaler.validate()
    coefficients = np.asarray(coefficients, dtype=np.float64)
    if coefficients.shape != (len(RIDGE_FEATURE_NAMES) + 1,):
        raise ValueError("ridge coefficients are misaligned")
    features = audited_ridge_features(history)
    return coefficients[0] + (
        (features - scaler.mean) / scaler.std
    ) @ coefficients[1:]


def raw_feature_coefficients(
    scaler: FeatureScaler,
    coefficients: np.ndarray,
) -> tuple[float, np.ndarray]:
    scaler.validate()
    coefficients = np.asarray(coefficients, dtype=np.float64)
    slopes = coefficients[1:] / scaler.std
    intercept = float(coefficients[0] - scaler.mean @ slopes)
    return intercept, slopes


def raw_history_coefficients(feature_slopes: np.ndarray) -> np.ndarray:
    feature_slopes = np.asarray(feature_slopes, dtype=np.float64)
    if feature_slopes.shape != (len(RIDGE_FEATURE_NAMES),):
        raise ValueError("ridge feature slopes are misaligned")
    result = np.zeros(RIDGE_HISTORY_STEPS, dtype=np.float64)
    result[-RIDGE_RECENT_STEPS:] += feature_slopes[:RIDGE_RECENT_STEPS]
    for coefficient, steps in zip(
        feature_slopes[RIDGE_RECENT_STEPS:],
        RIDGE_SUMMARY_STEPS,
        strict=True,
    ):
        result[-steps:] += coefficient
    return result
