from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import Tensor

from next_return_dataset import HISTORY_RETURN_COUNT

if TYPE_CHECKING:
    from train_normalized_glu_next_return import NextReturnDataset


SUMMARY_NAMES = (
    "mean",
    "variance",
    "minimum",
    "maximum",
    "cumulativeReturn",
)
OBJECTIVE_CONTRACT = (
    "train-normalized-per-candle-mse-plus-mean-variance-min-max-compounded-"
    "cumulative-return-mse-v1"
)


def numpy_path_summaries(returns: np.ndarray) -> np.ndarray:
    if returns.ndim != 2 or returns.shape[1] < 1:
        raise ValueError("return paths must have shape [example, horizon]")
    values = returns.astype(np.float64, copy=False)
    return np.stack((
        values.mean(axis=1),
        values.var(axis=1),
        values.min(axis=1),
        values.max(axis=1),
        np.expm1(values.sum(axis=1)),
    ), axis=1)


def torch_path_summaries(returns: Tensor) -> Tensor:
    if returns.ndim != 2 or returns.shape[1] < 1:
        raise ValueError("return paths must have shape [example, horizon]")
    values = returns.float()
    return torch.stack((
        values.mean(dim=1),
        values.var(dim=1, correction=0),
        values.amin(dim=1),
        values.amax(dim=1),
        torch.expm1(values.sum(dim=1)),
    ), dim=1)


@dataclass(frozen=True)
class SequenceNormalization:
    feature_mean: np.ndarray
    feature_std: np.ndarray
    target_mean: np.ndarray
    target_std: np.ndarray
    summary_mean: np.ndarray
    summary_std: np.ndarray

    @property
    def horizon_return_count(self) -> int:
        return int(self.target_mean.shape[0])

    def validate(self) -> None:
        horizon = self.horizon_return_count
        if self.feature_mean.shape != (HISTORY_RETURN_COUNT,) \
                or self.feature_std.shape != self.feature_mean.shape \
                or self.target_std.shape != (horizon,) \
                or self.summary_mean.shape != (len(SUMMARY_NAMES),) \
                or self.summary_std.shape != self.summary_mean.shape \
                or horizon < 2:
            raise ValueError("sequence normalization shapes are invalid")
        for values in (
            self.feature_mean,
            self.feature_std,
            self.target_mean,
            self.target_std,
            self.summary_mean,
            self.summary_std,
        ):
            if not np.isfinite(values).all():
                raise ValueError("sequence normalization must be finite")
        if bool((self.feature_std <= 0).any()) \
                or bool((self.target_std <= 0).any()) \
                or bool((self.summary_std <= 0).any()):
            raise ValueError("sequence normalization scales must be positive")


def training_sequence_normalization(
    dataset: NextReturnDataset,
    *,
    batch_size: int,
) -> SequenceNormalization:
    horizon = dataset.horizon_return_count
    if horizon < 2:
        raise ValueError("sequence normalization requires horizon greater than one")
    feature_sum = np.zeros(HISTORY_RETURN_COUNT, dtype=np.float64)
    feature_square_sum = np.zeros(HISTORY_RETURN_COUNT, dtype=np.float64)
    target_sum = np.zeros(horizon, dtype=np.float64)
    target_square_sum = np.zeros(horizon, dtype=np.float64)
    summary_sum = np.zeros(len(SUMMARY_NAMES), dtype=np.float64)
    summary_square_sum = np.zeros(len(SUMMARY_NAMES), dtype=np.float64)
    total = 0.0
    for feature_tensor, target_tensor, weight_tensor in dataset.iter_batches(
        "train", batch_size, shuffle=False, seed=0
    ):
        features = feature_tensor.numpy().astype(np.float64, copy=False)
        targets = target_tensor.numpy().astype(np.float64, copy=False)
        weights = weight_tensor.numpy().astype(np.float64, copy=False)
        summaries = numpy_path_summaries(targets)
        feature_sum += np.einsum("i,ij->j", weights, features)
        feature_square_sum += np.einsum(
            "i,ij->j", weights, np.square(features)
        )
        target_sum += np.einsum("i,ij->j", weights, targets)
        target_square_sum += np.einsum(
            "i,ij->j", weights, np.square(targets)
        )
        summary_sum += np.einsum("i,ij->j", weights, summaries)
        summary_square_sum += np.einsum(
            "i,ij->j", weights, np.square(summaries)
        )
        total += float(weights.sum(dtype=np.float64))
    if int(round(total)) != dataset.logical_count("train"):
        raise RuntimeError("sequence normalization did not cover the corpus")

    def mean_and_std(
        values_sum: np.ndarray,
        values_square_sum: np.ndarray,
        *,
        minimum_variance: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        mean = values_sum / total
        variance = values_square_sum / total - np.square(mean)
        return (
            mean.astype(np.float32),
            np.sqrt(np.maximum(variance, minimum_variance)).astype(np.float32),
        )

    feature_mean, feature_std = mean_and_std(
        feature_sum, feature_square_sum, minimum_variance=1e-20
    )
    target_mean, target_std = mean_and_std(
        target_sum, target_square_sum, minimum_variance=1e-20
    )
    summary_mean, summary_std = mean_and_std(
        summary_sum, summary_square_sum, minimum_variance=1e-30
    )
    result = SequenceNormalization(
        feature_mean=feature_mean,
        feature_std=feature_std,
        target_mean=target_mean,
        target_std=target_std,
        summary_mean=summary_mean,
        summary_std=summary_std,
    )
    result.validate()
    return result


def sequence_objective_components(
    prediction: Tensor,
    target: Tensor,
    *,
    target_std: Tensor,
    summary_std: Tensor,
) -> tuple[Tensor, Tensor]:
    if prediction.ndim != 2 or target.shape != prediction.shape \
            or target_std.shape != (prediction.shape[1],) \
            or summary_std.shape != (len(SUMMARY_NAMES),):
        raise ValueError("sequence objective inputs are misaligned")
    prediction = prediction.float()
    target = target.float()
    candle_loss = ((prediction - target) / target_std.float()).square().mean(dim=1)
    summary_error = (
        torch_path_summaries(prediction) - torch_path_summaries(target)
    ) / summary_std.float()
    return candle_loss, summary_error.square()


def sequence_objective_loss(
    prediction: Tensor,
    target: Tensor,
    weights: Tensor,
    *,
    target_std: Tensor,
    summary_std: Tensor,
    summary_metric_weights: Tensor,
    candle_weight: float,
    summary_weight: float,
) -> Tensor:
    if float(summary_weight) == 0:
        if prediction.ndim != 2 or target.shape != prediction.shape \
                or target_std.shape != (prediction.shape[1],):
            raise ValueError("sequence objective inputs are misaligned")
        candle = (
            (prediction.float() - target.float()) / target_std.float()
        ).square().mean(dim=1)
        return (
            float(candle_weight) * candle * weights.float()
        ).sum() / weights.sum()
    candle, summaries = sequence_objective_components(
        prediction,
        target,
        target_std=target_std,
        summary_std=summary_std,
    )
    if summary_metric_weights.shape != (len(SUMMARY_NAMES),):
        raise ValueError("summary metric weights must contain five values")
    weighted_summary = (
        summaries * summary_metric_weights.float()
    ).sum(dim=1) / summary_metric_weights.sum()
    per_example = (
        float(candle_weight) * candle
        + float(summary_weight) * weighted_summary
    )
    return (per_example * weights.float()).sum() / weights.sum()


class SequenceMetricAccumulator:
    def __init__(
        self,
        normalization: SequenceNormalization,
        *,
        candle_weight: float,
        summary_weight: float,
        summary_metric_weights: tuple[float, ...] | list[float],
        device: torch.device,
        track_per_lead: bool = True,
    ) -> None:
        normalization.validate()
        self.horizon = normalization.horizon_return_count
        self.candle_weight = float(candle_weight)
        self.summary_weight = float(summary_weight)
        if len(summary_metric_weights) != len(SUMMARY_NAMES) \
                or any(
                    not math.isfinite(float(value)) or float(value) < 0
                    for value in summary_metric_weights
                ) \
                or sum(float(value) for value in summary_metric_weights) <= 0:
            raise ValueError(
                "summary metric weights must be non-negative with positive sum"
            )
        self.summary_metric_weights = torch.as_tensor(
            summary_metric_weights, dtype=torch.float64, device=device
        )
        self.track_per_lead = bool(track_per_lead)
        self.target_std = torch.as_tensor(
            normalization.target_std, dtype=torch.float64, device=device
        )
        self.summary_std = torch.as_tensor(
            normalization.summary_std, dtype=torch.float64, device=device
        )
        self.weight = torch.zeros((), dtype=torch.float64, device=device)
        self.objective_sum = torch.zeros((), dtype=torch.float64, device=device)
        self.candle_normalized_sum = torch.zeros(
            (), dtype=torch.float64, device=device
        )
        self.summary_normalized_sum = torch.zeros(
            len(SUMMARY_NAMES), dtype=torch.float64, device=device
        )
        # weight, SSE, SAE, direction, prediction sum, target sum,
        # prediction square, target square, product
        self.path_values = torch.zeros(9, dtype=torch.float64, device=device)
        self.lead_values = torch.zeros(
            (self.horizon, 9), dtype=torch.float64, device=device
        ) if self.track_per_lead else None
        # SSE, SAE, prediction sum, target sum
        self.summary_values = torch.zeros(
            (len(SUMMARY_NAMES), 4), dtype=torch.float64, device=device
        )

    @torch.no_grad()
    def add(self, prediction: Tensor, target: Tensor, weights: Tensor) -> None:
        prediction = prediction.detach().to(dtype=torch.float64)
        target = target.detach().to(dtype=torch.float64)
        weights = weights.detach().to(dtype=torch.float64)
        if prediction.shape != target.shape \
                or prediction.ndim != 2 \
                or prediction.shape[1] != self.horizon \
                or weights.shape != (prediction.shape[0],):
            raise ValueError("sequence metric batch is misaligned")
        weight_matrix = weights.unsqueeze(1)
        error = prediction - target
        normalized_candle = (error / self.target_std).square().mean(dim=1)
        prediction_summary = torch_path_summaries(prediction).double()
        target_summary = torch_path_summaries(target).double()
        summary_error = prediction_summary - target_summary
        normalized_summary = (summary_error / self.summary_std).square()
        self.weight += weights.sum()
        self.candle_normalized_sum += (weights * normalized_candle).sum()
        self.summary_normalized_sum += (
            weight_matrix * normalized_summary
        ).sum(dim=0)
        weighted_summary = (
            normalized_summary * self.summary_metric_weights
        ).sum(dim=1) / self.summary_metric_weights.sum()
        self.objective_sum += (
            weights
            * (
                self.candle_weight * normalized_candle
                + self.summary_weight * weighted_summary
            )
        ).sum()

        weight_sum = weights.sum()
        batch_leads = torch.stack((
            weight_sum.expand(self.horizon),
            (weight_matrix * error.square()).sum(dim=0),
            (weight_matrix * error.abs()).sum(dim=0),
            (
                weight_matrix
                * ((prediction >= 0) == (target >= 0)).double()
            ).sum(dim=0),
            (weight_matrix * prediction).sum(dim=0),
            (weight_matrix * target).sum(dim=0),
            (weight_matrix * prediction.square()).sum(dim=0),
            (weight_matrix * target.square()).sum(dim=0),
            (weight_matrix * prediction * target).sum(dim=0),
        ), dim=1)
        if self.lead_values is not None:
            self.lead_values += batch_leads
        self.path_values += batch_leads.sum(dim=0)
        self.summary_values += torch.stack((
            (weight_matrix * summary_error.square()).sum(dim=0),
            (weight_matrix * summary_error.abs()).sum(dim=0),
            (weight_matrix * prediction_summary).sum(dim=0),
            (weight_matrix * target_summary).sum(dim=0),
        ), dim=1)

    @staticmethod
    def _path_result(values: Tensor) -> dict[str, float | int | None]:
        (
            weight,
            squared,
            absolute,
            direction,
            prediction_sum,
            target_sum,
            prediction_square,
            target_square,
            product,
        ) = (float(value) for value in values)
        prediction_mean = prediction_sum / weight
        target_mean = target_sum / weight
        prediction_variance = max(
            0.0, prediction_square / weight - prediction_mean**2
        )
        target_variance = max(0.0, target_square / weight - target_mean**2)
        covariance = product / weight - prediction_mean * target_mean
        denominator = math.sqrt(prediction_variance * target_variance)
        mse = squared / weight
        zero_mse = target_square / weight
        return {
            "values": int(round(weight)),
            "mse": mse,
            "rmse": math.sqrt(mse),
            "mae": absolute / weight,
            "directionAccuracy": direction / weight,
            "correlation": covariance / denominator if denominator > 0 else None,
            "predictionMean": prediction_mean,
            "predictionStd": math.sqrt(prediction_variance),
            "targetMean": target_mean,
            "targetStd": math.sqrt(target_variance),
            "zeroBaselineMse": zero_mse,
            "mseSkillVsZero": 1.0 - mse / zero_mse if zero_mse > 0 else 0.0,
        }

    def result(self, *, include_per_lead: bool = True) -> dict:
        weight = float(self.weight)
        if weight <= 0:
            raise RuntimeError("cannot finalize empty sequence metrics")
        summary_normalized = self.summary_normalized_sum / weight
        summary_weighted = float(
            (summary_normalized * self.summary_metric_weights).sum()
            / self.summary_metric_weights.sum()
        )
        summary_values = self.summary_values.cpu().numpy()
        result = {
            "examples": int(round(weight)),
            "horizonSeconds": self.horizon,
            "objective": float(self.objective_sum / weight),
            "candleNormalizedMse": float(self.candle_normalized_sum / weight),
            "summaryAverageNormalizedMse": float(summary_normalized.mean()),
            "summaryWeightedNormalizedMse": summary_weighted,
            "summaryMetricWeights": {
                name: float(self.summary_metric_weights[index])
                for index, name in enumerate(SUMMARY_NAMES)
            },
            "summaries": {
                name: {
                    "normalizedMse": float(summary_normalized[index]),
                    "mse": float(summary_values[index, 0] / weight),
                    "mae": float(summary_values[index, 1] / weight),
                    "predictionMean": float(summary_values[index, 2] / weight),
                    "targetMean": float(summary_values[index, 3] / weight),
                }
                for index, name in enumerate(SUMMARY_NAMES)
            },
            "allCandles": self._path_result(self.path_values),
        }
        if include_per_lead:
            if self.lead_values is None:
                raise RuntimeError("per-lead metrics were not tracked")
            result["perLead"] = [
                {"second": index + 1, **self._path_result(values)}
                for index, values in enumerate(self.lead_values)
            ]
        return result
