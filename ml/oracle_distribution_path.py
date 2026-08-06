from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor


OBJECTIVE_CONTRACT = (
    "predicted-fifteen-minute-return-path-differentiable-oracle-forward-kl-v1"
)
MEAN_PLUS_P50_OBJECTIVE_CONTRACT = (
    "predicted-fifteen-minute-return-path-differentiable-oracle-"
    "mean-plus-batch-p50-forward-kl-v1"
)


def normalized_probabilities(probabilities: Tensor) -> Tensor:
    if probabilities.ndim != 2:
        raise ValueError("oracle probabilities must have shape [example, action]")
    values = probabilities.float()
    if not bool(torch.isfinite(values).all().item()) \
            or bool((values < 0).any().item()):
        raise ValueError("oracle probabilities must be finite and non-negative")
    return values / values.sum(dim=-1, keepdim=True).clamp_min(
        torch.finfo(torch.float32).tiny
    )


def oracle_forward_kl_per_example(
    predicted_probabilities: Tensor,
    target_probabilities: Tensor,
    *,
    probability_floor: float,
) -> Tensor:
    if predicted_probabilities.shape != target_probabilities.shape:
        raise ValueError("predicted and target oracle probabilities must match")
    if not math.isfinite(probability_floor) or probability_floor <= 0:
        raise ValueError("probability floor must be positive and finite")
    target = normalized_probabilities(target_probabilities)
    predicted = normalized_probabilities(predicted_probabilities)
    action_count = predicted.shape[-1]
    predicted = (
        predicted + float(probability_floor)
    ) / (1.0 + action_count * float(probability_floor))
    target_log = torch.where(
        target > 0,
        target.clamp_min(torch.finfo(torch.float32).tiny).log(),
        torch.zeros_like(target),
    )
    return torch.where(
        target > 0,
        target * (target_log - predicted.log()),
        torch.zeros_like(target),
    ).sum(dim=-1)


def oracle_forward_kl_loss(
    predicted_probabilities: Tensor,
    target_probabilities: Tensor,
    weights: Tensor,
    *,
    probability_floor: float,
) -> Tensor:
    if weights.shape != (predicted_probabilities.shape[0],):
        raise ValueError("oracle KL weights must contain one value per example")
    normalized_weights = weights.float().clamp_min(0)
    return (
        oracle_forward_kl_per_example(
            predicted_probabilities,
            target_probabilities,
            probability_floor=probability_floor,
        ) * normalized_weights
    ).sum() / normalized_weights.sum().clamp_min(
        torch.finfo(torch.float32).tiny
    )


def oracle_mean_plus_p50_kl_loss(
    predicted_probabilities: Tensor,
    target_probabilities: Tensor,
    weights: Tensor,
    *,
    probability_floor: float,
    mean_weight: float,
    p50_weight: float,
) -> Tensor:
    if weights.shape != (predicted_probabilities.shape[0],):
        raise ValueError("oracle KL weights must contain one value per example")
    if not math.isfinite(mean_weight) or mean_weight < 0 \
            or not math.isfinite(p50_weight) or p50_weight < 0 \
            or mean_weight + p50_weight <= 0:
        raise ValueError("mean and P50 KL weights must be non-negative")
    per_example = oracle_forward_kl_per_example(
        predicted_probabilities,
        target_probabilities,
        probability_floor=probability_floor,
    )
    normalized_weights = weights.float().clamp_min(0)
    mean_kl = (
        per_example * normalized_weights
    ).sum() / normalized_weights.sum().clamp_min(
        torch.finfo(torch.float32).tiny
    )
    if p50_weight == 0:
        return float(mean_weight) * mean_kl
    if not torch.allclose(
        normalized_weights,
        torch.ones_like(normalized_weights),
        rtol=0,
        atol=0,
    ):
        raise ValueError("differentiable P50 currently requires unit weights")
    p50_kl = torch.quantile(per_example, 0.5)
    return float(mean_weight) * mean_kl + float(p50_weight) * p50_kl


@dataclass
class OracleDistributionMetricAccumulator:
    action_grid: Tensor
    probability_floor: float
    track_kl_percentiles: bool = False

    def __post_init__(self) -> None:
        if self.action_grid.ndim != 1:
            raise ValueError("oracle action grid must be one-dimensional")
        self.action_grid = self.action_grid.detach().double()
        self.values = torch.zeros(
            12, dtype=torch.float64, device=self.action_grid.device
        )
        self.kl_samples: list[Tensor] = []
        self.kl_sample_weights: list[Tensor] = []

    @torch.no_grad()
    def add(
        self,
        predicted_returns: Tensor,
        target_returns: Tensor,
        predicted_probabilities: Tensor,
        target_probabilities: Tensor,
        weights: Tensor,
    ) -> None:
        if predicted_returns.ndim != 2 \
                or target_returns.shape != predicted_returns.shape \
                or predicted_probabilities.shape != target_probabilities.shape \
                or predicted_probabilities.shape[0] != predicted_returns.shape[0] \
                or predicted_probabilities.shape[1] != self.action_grid.numel() \
                or weights.shape != (predicted_returns.shape[0],):
            raise ValueError("oracle distribution metric batch is misaligned")
        predicted = normalized_probabilities(predicted_probabilities).double()
        target = normalized_probabilities(target_probabilities).double()
        sample_weights = weights.detach().double().clamp_min(0)
        kl = oracle_forward_kl_per_example(
            predicted.float(), target.float(),
            probability_floor=self.probability_floor,
        ).double()
        if self.track_kl_percentiles:
            self.kl_samples.append(kl.detach().cpu())
            self.kl_sample_weights.append(sample_weights.detach().cpu())
        target_log = torch.where(
            target > 0, target.clamp_min(torch.finfo(torch.float64).tiny).log(),
            torch.zeros_like(target),
        )
        target_entropy = -(target * target_log).sum(dim=-1)
        predicted_entropy = -(predicted * predicted.clamp_min(
            torch.finfo(torch.float64).tiny
        ).log()).sum(dim=-1)
        probability_mse = (predicted - target).square().mean(dim=-1)
        total_variation = 0.5 * (predicted - target).abs().sum(dim=-1)
        mode_agreement = (
            predicted.argmax(dim=-1) == target.argmax(dim=-1)
        ).double()
        predicted_exposure = (predicted * self.action_grid).sum(dim=-1)
        target_exposure = (target * self.action_grid).sum(dim=-1)
        exposure_mae = (predicted_exposure - target_exposure).abs()
        return_error = predicted_returns.double() - target_returns.double()
        path_mse = return_error.square().mean(dim=-1)
        zero_path_mse = target_returns.double().square().mean(dim=-1)
        cumulative_error = (
            predicted_returns.double().sum(dim=-1)
            - target_returns.double().sum(dim=-1)
        ).square()
        zero_cumulative_mse = target_returns.double().sum(dim=-1).square()
        rows = torch.stack((
            sample_weights,
            sample_weights * kl,
            sample_weights * target_entropy,
            sample_weights * predicted_entropy,
            sample_weights * probability_mse,
            sample_weights * total_variation,
            sample_weights * mode_agreement,
            sample_weights * exposure_mae,
            sample_weights * path_mse,
            sample_weights * zero_path_mse,
            sample_weights * cumulative_error,
            sample_weights * zero_cumulative_mse,
        ), dim=1).sum(dim=0)
        self.values += rows

    def result(self) -> dict[str, float | int]:
        weight = float(self.values[0])
        if weight <= 0:
            raise RuntimeError("cannot finalize empty oracle metrics")
        values = [float(value) / weight for value in self.values[1:]]
        path_mse = values[7]
        zero_path_mse = values[8]
        cumulative_mse = values[9]
        zero_cumulative_mse = values[10]
        result: dict[str, float | int | dict[str, float]] = {
            "examples": int(round(weight)),
            "klDivergence": values[0],
            "targetEntropy": values[1],
            "predictedEntropy": values[2],
            "probabilityMse": values[3],
            "totalVariation": values[4],
            "modalActionAgreement": values[5],
            "expectedExposureMae": values[6],
            "pathMse": path_mse,
            "pathMseSkillVsZero": (
                1.0 - path_mse / zero_path_mse if zero_path_mse > 0 else 0.0
            ),
            "cumulativeLogReturnMse": cumulative_mse,
            "cumulativeMseSkillVsZero": (
                1.0 - cumulative_mse / zero_cumulative_mse
                if zero_cumulative_mse > 0 else 0.0
            ),
        }
        if self.track_kl_percentiles:
            samples = torch.cat(self.kl_samples)
            sample_weights = torch.cat(self.kl_sample_weights)
            samples, order = torch.sort(samples)
            cumulative_weight = torch.cumsum(sample_weights[order], dim=0)
            total_weight = cumulative_weight[-1]
            percentiles: dict[str, float] = {}
            for label, probability in (
                ("p50", 0.50), ("p90", 0.90), ("p95", 0.95)
            ):
                index = int(torch.searchsorted(
                    cumulative_weight,
                    total_weight * probability,
                    right=False,
                ).clamp_max(samples.numel() - 1))
                percentiles[label] = float(samples[index])
            result["klPercentiles"] = percentiles
        return result
