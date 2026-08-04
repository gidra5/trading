from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch
from torch import Tensor

from next_return_dataset import HISTORY_RETURN_COUNT
from train_normalized_glu_next_return import Normalization


MODEL_CONTRACT = (
    "train-standardized-120-second-log-returns-to-next-standardized-second-"
    "return-linear-ridge-v1"
)


@dataclass
class StandardizedRegressionStatistics:
    device: torch.device

    def __post_init__(self) -> None:
        dimension = HISTORY_RETURN_COUNT + 1
        self.weight = 0.0
        self.xtx = torch.zeros(
            (dimension, dimension), dtype=torch.float64, device=self.device
        )
        self.xty = torch.zeros(dimension, dtype=torch.float64, device=self.device)
        self.y_sum = 0.0
        self.y_square_sum = 0.0

    @torch.no_grad()
    def add(
        self,
        features: Tensor,
        targets: Tensor,
        weights: Tensor,
        normalization: Normalization,
    ) -> None:
        if features.ndim != 2 \
                or features.shape[1] != HISTORY_RETURN_COUNT \
                or targets.shape != (features.shape[0],) \
                or weights.shape != targets.shape:
            raise ValueError("linear sufficient-statistic batch is misaligned")
        feature_mean = torch.as_tensor(
            normalization.feature_mean,
            device=features.device,
            dtype=torch.float32,
        )
        feature_std = torch.as_tensor(
            normalization.feature_std,
            device=features.device,
            dtype=torch.float32,
        )
        standardized = (features.float() - feature_mean) / feature_std
        design = torch.cat((
            torch.ones(
                (features.shape[0], 1),
                device=features.device,
                dtype=torch.float32,
            ),
            standardized,
        ), dim=1)
        normalized_target = (
            targets.float() - normalization.target_mean
        ) / normalization.target_std
        float_weights = weights.float()
        weighted_design = design * float_weights.unsqueeze(1)
        # Products use full FP32 matmul; only the small reduced matrices are
        # promoted to FP64 for stable accumulation across the entire corpus.
        self.xtx += (design.T @ weighted_design).to(dtype=torch.float64)
        self.xty += (
            design.T @ (float_weights * normalized_target)
        ).to(dtype=torch.float64)
        self.weight += float(float_weights.sum())
        self.y_sum += float((float_weights * normalized_target).sum())
        self.y_square_sum += float(
            (float_weights * normalized_target.square()).sum()
        )

    def cpu(self) -> StandardizedRegressionStatistics:
        result = StandardizedRegressionStatistics(torch.device("cpu"))
        result.weight = self.weight
        result.xtx = self.xtx.cpu()
        result.xty = self.xty.cpu()
        result.y_sum = self.y_sum
        result.y_square_sum = self.y_square_sum
        return result


def solve_ridge(
    statistics: StandardizedRegressionStatistics,
    ridge_lambda: float,
) -> np.ndarray:
    if not math.isfinite(ridge_lambda) or ridge_lambda < 0:
        raise ValueError("ridge lambda must be finite and non-negative")
    matrix = statistics.xtx.cpu().numpy().copy()
    target = statistics.xty.cpu().numpy()
    if ridge_lambda:
        penalty = np.eye(matrix.shape[0], dtype=np.float64)
        penalty[0, 0] = 0.0
        matrix += statistics.weight * ridge_lambda * penalty
    return np.linalg.lstsq(matrix, target, rcond=1e-12)[0]


def sufficient_normalized_mse(
    statistics: StandardizedRegressionStatistics,
    coefficients: np.ndarray,
) -> float:
    coefficients = np.asarray(coefficients, dtype=np.float64)
    xtx = statistics.xtx.cpu().numpy()
    xty = statistics.xty.cpu().numpy()
    if coefficients.shape != xty.shape:
        raise ValueError("linear coefficients are misaligned")
    squared = (
        statistics.y_square_sum
        - 2.0 * coefficients @ xty
        + coefficients @ xtx @ coefficients
    )
    return max(0.0, float(squared / statistics.weight))


def raw_coefficients(
    coefficients: np.ndarray,
    normalization: Normalization,
) -> tuple[float, np.ndarray]:
    coefficients = np.asarray(coefficients, dtype=np.float64)
    if coefficients.shape != (HISTORY_RETURN_COUNT + 1,):
        raise ValueError("linear coefficients must contain intercept plus 120 lags")
    slopes = (
        normalization.target_std
        * coefficients[1:]
        / normalization.feature_std.astype(np.float64)
    )
    intercept = float(
        normalization.target_mean
        + normalization.target_std * coefficients[0]
        - normalization.feature_mean.astype(np.float64) @ slopes
    )
    return intercept, slopes


def predict_raw(
    features: Tensor,
    intercept: float,
    slopes: Tensor,
) -> Tensor:
    if features.ndim != 2 or features.shape[1] != HISTORY_RETURN_COUNT \
            or slopes.shape != (HISTORY_RETURN_COUNT,):
        raise ValueError("linear prediction input is misaligned")
    return features.float() @ slopes + float(intercept)
