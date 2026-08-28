from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Iterator

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

from active_return_path_dataset import CandleCloseCache
from adversarial_union530_base import BaseHistoryAdversary
from causal_return_normalization import (
    append_return_statistics,
    trailing_log_price_statistics,
    trailing_log_return_statistics,
)
from actor_market_process_density import (
    ARCHITECTURE_CONTRACT as ACTOR_MARKET_PROCESS_ARCHITECTURE_CONTRACT,
    ActorMarketProcessDensity,
)
from actor_market_path_matrix_density import (
    ARCHITECTURE_CONTRACT as ACTOR_MARKET_PATH_MATRIX_ARCHITECTURE_CONTRACT,
    ActorMarketPathMatrixDensity,
)
from compressed_path_return_density import (
    ARCHITECTURE_CONTRACT,
    CompressedPathOutput,
    CompressedPathReturnDensity,
    path_log_density_terms,
)
from recurrent_market_path_density import (
    ARCHITECTURE_CONTRACT as RECURRENT_MARKET_ARCHITECTURE_CONTRACT,
    RESIDUAL_ARCHITECTURE_CONTRACT as RESIDUAL_RECURRENT_MARKET_ARCHITECTURE_CONTRACT,
    RecurrentMarketPathDensity,
    ResidualRecurrentMarketPathDensity,
)
from low_rank_path_matrix_density import (
    ARCHITECTURE_CONTRACT as LOW_RANK_PATH_MATRIX_ARCHITECTURE_CONTRACT,
    CYCLIC_DENSE_COMPRESSED_ARCHITECTURE_CONTRACT,
    DIRECT_FACTORIZED_ARCHITECTURE_CONTRACT,
    JOINT_PREFIX_CONTRACTED_CYCLIC_ARCHITECTURE_CONTRACT,
    CyclicDenseCompressedPathMatrixDensity,
    DirectFactorizedPathMatrixDensity,
    DynamicLowRankPathMatrixDensity,
    JointPrefixContractedCyclicPathMatrixDensity,
)
from normalized_glu_next_return import optimizer_parameter_groups
from return_knot_density import KnotDensityContract
from trading_storage import (
    checkpoint_exists,
    load_torch_checkpoint,
    save_torch_checkpoint,
)
from train_autoregressive_minute_return import build_optimizers
from train_feature_augmented_next_return import FeatureMatrixDataset
from union530_base_dataset import DifferentiableUnion530Dataset
from train_next_return_knot_density import PAUSE_EXIT_CODE, canonical_hash
from train_next_return_memorization import (
    mean_teacher_ema_decay,
    sam_perturb_parameters,
    sam_restore_parameters,
)
from train_normalized_glu_next_return import (
    MetricAccumulator,
    Reporter,
    atomic_json,
)


RUNNER_CONTRACT = "feature-immediate-compressed-active-return-path-density-v3"
RECURRENT_MARKET_RUNNER_CONTRACT = (
    "feature-lag3-recurrent-market-compressed-path-density-v1"
)
LOW_RANK_PATH_MATRIX_RUNNER_CONTRACT = (
    "feature-lag3-dynamic-low-rank-path-matrix-density-v1"
)
DIRECT_FACTORIZED_PATH_MATRIX_RUNNER_CONTRACT = (
    "feature-lag3-direct-unpacked-factorized-path-matrix-density-v1"
)
CYCLIC_DENSE_COMPRESSED_PATH_MATRIX_RUNNER_CONTRACT = (
    "feature-lag3-cyclic-dense-compressed-path-matrix-density-v1"
)
UNION_IMMEDIATE_CYCLIC_DENSE_PATH_MATRIX_RUNNER_CONTRACT = (
    "feature-union530-immediate-cyclic-dense-compressed-path-matrix-density-v1"
)
UNION_JOINT_PREFIX_CYCLIC_PATH_MATRIX_RUNNER_CONTRACT = (
    "feature-union530-joint-prefix-contracted-cyclic-path-matrix-density-v1"
)
ACTOR_MARKET_PROCESS_RUNNER_CONTRACT = (
    "feature-lag3-recurrent-actor-market-process-density-packed-linear-heads-v2"
)
ACTOR_MARKET_PATH_MATRIX_RUNNER_CONTRACT = (
    "feature-lag3-recurrent-actor-market-cyclic-dense-path-matrix-density-v1"
)
GEOMETRIC_KEEP_EMBEDDING_DROPOUT_SCHEDULE = (
    "geometric-keep-probability-v1"
)
LINEAR_EXPECTED_RETURN_LOSS_WEIGHT_SCHEDULE = "linear-v1"
GEOMETRIC_EXPECTED_RETURN_LOSS_WEIGHT_SCHEDULE = "geometric-v1"


def embedding_dropout_probability_at_epoch(training: dict, epoch: int) -> float:
    """Resolve static or endpoint-exact geometric keep-probability dropout."""
    if epoch < 0:
        raise ValueError("embedding dropout epoch must be non-negative")
    static_probability = float(training.get("embeddingDropoutProbability", 0.0))
    if not 0.0 <= static_probability < 1.0:
        raise ValueError("embedding dropout probability must be in [0, 1)")
    schedule = training.get("embeddingDropoutSchedule")
    if schedule is None:
        return static_probability
    if schedule.get("type") != GEOMETRIC_KEEP_EMBEDDING_DROPOUT_SCHEDULE:
        raise ValueError("unsupported embedding dropout schedule")
    start_probability = float(schedule["startProbability"])
    end_probability = float(schedule["endProbability"])
    end_epoch = int(schedule["endEpoch"])
    if not 0.0 <= start_probability < 1.0 \
            or not 0.0 <= end_probability < 1.0:
        raise ValueError("scheduled embedding dropout must remain in [0, 1)")
    if end_epoch <= 0:
        raise ValueError("embedding dropout schedule end epoch must be positive")
    if not math.isclose(static_probability, start_probability, abs_tol=1e-12):
        raise ValueError(
            "embedding dropout probability must equal the schedule start probability"
        )
    if epoch >= end_epoch:
        return end_probability
    progress = float(epoch) / float(end_epoch)
    start_keep_probability = 1.0 - start_probability
    end_keep_probability = 1.0 - end_probability
    keep_probability = math.exp(
        math.log(start_keep_probability)
        + progress * math.log(end_keep_probability / start_keep_probability)
    )
    return 1.0 - keep_probability


def weighted_expected_return_correlation(
    prediction: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
    *,
    target_std: float,
    variance_epsilon: float = 1e-8,
) -> torch.Tensor:
    """Differentiable weighted Pearson correlation over expected returns."""
    if prediction.shape != target.shape:
        raise ValueError("expected-return prediction and target shapes differ")
    if prediction.ndim != 2 or weights.shape != prediction.shape[:1]:
        raise ValueError("expected-return correlation requires [batch, step] values")
    if not math.isfinite(target_std) or target_std <= 0:
        raise ValueError("expected-return correlation target std must be positive")
    if not math.isfinite(variance_epsilon) or variance_epsilon <= 0:
        raise ValueError("expected-return correlation epsilon must be positive")
    expanded_weights = weights[:, None].expand_as(target)
    weight_sum = expanded_weights.sum()
    scale = prediction.new_tensor(target_std)
    normalized_prediction = prediction / scale
    normalized_target = target / scale
    prediction_mean = (
        expanded_weights * normalized_prediction
    ).sum() / weight_sum
    target_mean = (expanded_weights * normalized_target).sum() / weight_sum
    prediction_centered = normalized_prediction - prediction_mean
    target_centered = normalized_target - target_mean
    covariance = (
        expanded_weights * prediction_centered * target_centered
    ).sum() / weight_sum
    prediction_variance = (
        expanded_weights * prediction_centered.square()
    ).sum() / weight_sum
    target_variance = (
        expanded_weights * target_centered.square()
    ).sum() / weight_sum
    epsilon = prediction.new_tensor(variance_epsilon)
    denominator = torch.sqrt(
        (prediction_variance + epsilon) * (target_variance + epsilon)
    )
    return covariance / denominator


def _expected_return_loss_weight_at_epoch(
    training: dict,
    epoch: int,
    *,
    weight_key: str,
    schedule_key: str,
    label: str,
) -> float:
    """Resolve a static or endpoint-exact scheduled auxiliary-loss weight."""
    if epoch < 0:
        raise ValueError(f"{label} weight epoch must be non-negative")
    static_weight = float(training.get(weight_key, 0.0))
    if static_weight < 0 or not math.isfinite(static_weight):
        raise ValueError(f"{label} loss weight must be finite and non-negative")
    schedule = training.get(schedule_key)
    if schedule is None:
        return static_weight
    schedule_type = schedule.get("type")
    if schedule_type not in (
        LINEAR_EXPECTED_RETURN_LOSS_WEIGHT_SCHEDULE,
        GEOMETRIC_EXPECTED_RETURN_LOSS_WEIGHT_SCHEDULE,
    ):
        raise ValueError(f"unsupported {label} loss weight schedule")
    start_weight = float(schedule["startWeight"])
    end_weight = float(schedule["endWeight"])
    start_epoch = int(schedule.get("startEpoch", 0))
    end_epoch = int(schedule["endEpoch"])
    if start_weight < 0 or end_weight < 0 \
            or not math.isfinite(start_weight) or not math.isfinite(end_weight):
        raise ValueError(f"scheduled {label} weights must be non-negative")
    if start_epoch < 0 or end_epoch <= start_epoch:
        raise ValueError(
            f"{label} weight schedule must have 0 <= start epoch < end epoch"
        )
    if not math.isclose(static_weight, start_weight, abs_tol=1e-12):
        raise ValueError(
            f"{label} loss weight must equal the schedule start weight"
        )
    if schedule_type == GEOMETRIC_EXPECTED_RETURN_LOSS_WEIGHT_SCHEDULE \
            and (start_weight <= 0 or end_weight <= 0):
        raise ValueError(
            f"geometrically scheduled {label} weights must be positive"
        )
    if epoch <= start_epoch:
        return start_weight
    if epoch >= end_epoch:
        return end_weight
    progress = float(epoch - start_epoch) / float(end_epoch - start_epoch)
    if schedule_type == GEOMETRIC_EXPECTED_RETURN_LOSS_WEIGHT_SCHEDULE:
        return start_weight * math.pow(end_weight / start_weight, progress)
    return start_weight + progress * (end_weight - start_weight)


def expected_return_correlation_loss_weight_at_epoch(
    training: dict, epoch: int
) -> float:
    return _expected_return_loss_weight_at_epoch(
        training,
        epoch,
        weight_key="expectedReturnCorrelationLossWeight",
        schedule_key="expectedReturnCorrelationLossWeightSchedule",
        label="expected-return correlation",
    )


def expected_return_mse_loss_weight_at_epoch(training: dict, epoch: int) -> float:
    return _expected_return_loss_weight_at_epoch(
        training,
        epoch,
        weight_key="expectedReturnMseLossWeight",
        schedule_key="expectedReturnMseLossWeightSchedule",
        label="expected-return MSE",
    )


def weighted_expected_return_normalized_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
    *,
    target_std: float,
) -> torch.Tensor:
    """Weighted expected-return MSE in target-standard-deviation units."""
    if prediction.shape != target.shape:
        raise ValueError("expected-return prediction and target shapes differ")
    if prediction.ndim != 2 or weights.shape != prediction.shape[:1]:
        raise ValueError("expected-return MSE requires [batch, step] values")
    if not math.isfinite(target_std) or target_std <= 0:
        raise ValueError("expected-return MSE target std must be positive")
    expanded_weights = weights[:, None].expand_as(target)
    normalized_error = (prediction - target) / prediction.new_tensor(target_std)
    return (expanded_weights * normalized_error.square()).sum() / (
        expanded_weights.sum()
    )


NLL_DISTRIBUTION_LOSS = "interpolated-marginal-nll-v1"
JOINT_PREFIX_NLL_DISTRIBUTION_LOSS = "forward-contracted-joint-path-nll-v1"
DISCRETE_CRPS_DISTRIBUTION_LOSS = "discrete-component-crps-v1"
COMPONENT_LOGIT_OUTPUT_ADVERSARY = "component-logit-rms-fgsm-v1"


def distribution_loss_type(training: dict) -> str:
    specification = training.get("distributionLoss")
    if specification is None:
        return NLL_DISTRIBUTION_LOSS
    loss_type = specification.get("type")
    if loss_type not in {
        NLL_DISTRIBUTION_LOSS,
        JOINT_PREFIX_NLL_DISTRIBUTION_LOSS,
        DISCRETE_CRPS_DISTRIBUTION_LOSS,
    }:
        raise ValueError("unsupported distribution training loss")
    if loss_type == DISCRETE_CRPS_DISTRIBUTION_LOSS \
            and specification.get("scale") != "training-target-standard-deviation":
        raise ValueError("discrete CRPS must use the fixed training target scale")
    return str(loss_type)


def discrete_component_crps_terms(
    output: CompressedPathOutput,
    targets: torch.Tensor,
    model: torch.nn.Module,
) -> torch.Tensor:
    """CRPS for the emitted component masses treated at their return means.

    This is the exact discrete-distribution identity. It deliberately avoids
    target transformation, interval lookup, basis heights, and interpolated
    density evaluation. The dynamic component return means remain live, so the
    score trains both probabilities and output values.
    """
    if targets.ndim != 2 or targets.shape[1] != model.return_count:
        raise ValueError("CRPS targets have the wrong shape")
    if len(output.log_masses) != model.return_count:
        raise ValueError("CRPS output horizon changed")
    terms: list[torch.Tensor] = []
    for step, log_masses in enumerate(output.log_masses):
        probabilities = torch.exp(log_masses.float())
        probabilities = probabilities / probabilities.sum(
            dim=1, keepdim=True
        ).clamp_min(1e-12)
        if output.component_means is not None:
            values = output.component_means[step].float()
        elif hasattr(model, "means"):
            values = model.means(step).float()
        elif hasattr(model, "density_means"):
            values = model.density_means.float()
        else:
            raise ValueError("model does not expose component return means")
        if values.ndim == 1:
            values = values[None, :].expand_as(probabilities)
        if values.shape != probabilities.shape:
            raise ValueError("CRPS component values and probabilities differ")
        target = targets[:, step:step + 1].float()
        observation_distance = (
            probabilities * torch.abs(values - target)
        ).sum(dim=1)
        pairwise_distance = torch.abs(
            values[:, :, None] - values[:, None, :]
        )
        distribution_distance = 0.5 * (
            probabilities[:, :, None]
            * probabilities[:, None, :]
            * pairwise_distance
        ).sum(dim=(1, 2))
        terms.append(torch.clamp_min(
            observation_distance - distribution_distance, 0
        ))
    return torch.stack(terms, dim=1)


def weighted_normalized_crps(
    output: CompressedPathOutput,
    targets: torch.Tensor,
    weights: torch.Tensor,
    model: torch.nn.Module,
    *,
    target_std: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not math.isfinite(target_std) or target_std <= 0:
        raise ValueError("CRPS target standard deviation must be positive")
    terms = discrete_component_crps_terms(output, targets, model)
    expanded_weights = weights[:, None].expand_as(terms)
    normalized = terms / terms.new_tensor(target_std)
    return (
        (expanded_weights * normalized).sum() / expanded_weights.sum(),
        terms,
    )


def output_adversarial_specification(training: dict) -> dict | None:
    specification = training.get("adversarialOutput")
    if specification is None:
        return None
    if specification.get("type") != COMPONENT_LOGIT_OUTPUT_ADVERSARY:
        raise ValueError("unsupported output adversarial training type")
    if int(specification.get("steps", 0)) != 1:
        raise ValueError("output adversarial training currently uses one step")
    epsilon = float(specification.get("epsilonRms", 0.0))
    step_size = float(specification.get("stepSizeRms", 0.0))
    weight = float(specification.get("adversarialWeight", 0.5))
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("output adversarial RMS must be finite and positive")
    if not math.isfinite(step_size) or step_size != epsilon:
        raise ValueError("one-step output adversarial step size must equal epsilon")
    if not math.isfinite(weight) or not 0 < weight <= 1:
        raise ValueError("invalid output adversarial loss weight")
    if specification.get("applyTo") != "clean-and-input-adversarial":
        raise ValueError(
            "output adversarial training must cover clean and input-adversarial paths"
        )
    return specification


def component_return_values(
    output: CompressedPathOutput,
    model: torch.nn.Module,
    step: int,
) -> torch.Tensor:
    if output.component_means is not None:
        values = output.component_means[step].float()
    elif hasattr(model, "means"):
        values = model.means(step).float()
    elif hasattr(model, "density_means"):
        values = model.density_means.float()
    else:
        raise ValueError("model does not expose component return means")
    return values


def perturb_component_logits(
    output: CompressedPathOutput,
    deltas: tuple[torch.Tensor, ...],
    model: torch.nn.Module,
) -> CompressedPathOutput:
    """Apply coherent logit perturbations and recompute distribution moments."""
    if len(deltas) != len(output.log_masses):
        raise ValueError("one output perturbation is required per path step")
    log_masses: list[torch.Tensor] = []
    expectations: list[torch.Tensor] = []
    for step, (base, delta) in enumerate(zip(
        output.log_masses, deltas, strict=True
    )):
        if delta.shape != base.shape:
            raise ValueError("output logit perturbation shape changed")
        perturbed = torch.log_softmax(base.float() + delta.float(), dim=1)
        probabilities = torch.exp(perturbed)
        values = component_return_values(output, model, step)
        if values.ndim == 1:
            values = values[None, :].expand_as(probabilities)
        if values.shape != probabilities.shape:
            raise ValueError("component values and perturbed masses differ")
        log_masses.append(perturbed)
        expectations.append((probabilities * values).sum(dim=1))
    return CompressedPathOutput(
        log_masses=tuple(log_masses),
        expectations=torch.stack(expectations, dim=1),
        knots_unit=output.knots_unit,
        areas_unit=output.areas_unit,
        component_means=output.component_means,
        arithmetic_component_means=output.arithmetic_component_means,
        normalization_location=output.normalization_location,
        normalization_scale=output.normalization_scale,
    )


def generate_output_adversarial_variant(
    output: CompressedPathOutput,
    model: torch.nn.Module,
    objective,
    *,
    epsilon_rms: float,
    collect_metrics: bool = False,
) -> tuple[CompressedPathOutput, dict | None]:
    """One-step worst-case perturbation of every emitted component-logit vector."""
    probes = tuple(
        torch.zeros_like(value, requires_grad=True)
        for value in output.log_masses
    )
    probe_output = perturb_component_logits(output, probes, model)
    probe_loss = objective(probe_output)
    gradients = torch.autograd.grad(
        probe_loss, probes, retain_graph=True, create_graph=False
    )
    deltas: list[torch.Tensor] = []
    for gradient in gradients:
        centered = gradient.detach().float() - gradient.detach().float().mean(
            dim=1, keepdim=True
        )
        rms = centered.square().mean(dim=1, keepdim=True).sqrt()
        direction = centered / rms.clamp_min(torch.finfo(centered.dtype).tiny)
        deltas.append(direction * float(epsilon_rms))
    detached = tuple(deltas)
    adversarial = perturb_component_logits(output, detached, model)
    metrics = None
    if collect_metrics:
        flattened = torch.cat(tuple(value.flatten() for value in detached))
        gradient_flattened = torch.cat(tuple(
            value.detach().flatten() for value in gradients
        ))
        metrics = {
            "type": COMPONENT_LOGIT_OUTPUT_ADVERSARY,
            "normalizedLogitDeltaRms": float(
                flattened.square().mean().sqrt()
            ),
            "normalizedLogitDeltaMaximum": float(flattened.abs().max()),
            "finiteGradientCoordinates": int(
                torch.isfinite(gradient_flattened).sum().item()
            ),
            "gradientCoordinates": int(gradient_flattened.numel()),
        }
    return adversarial, metrics


def weighted_path_training_objective(
    output: CompressedPathOutput,
    targets: torch.Tensor,
    weights: torch.Tensor,
    model: torch.nn.Module,
    *,
    crps_training: bool,
    target_std: float,
    correlation_weight: float,
    mse_weight: float,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor,
    torch.Tensor,
]:
    nll_loss = None
    crps_loss = None
    if crps_training:
        crps_loss, _terms = weighted_normalized_crps(
            output, targets, weights, model, target_std=target_std
        )
        distribution_loss = crps_loss
    else:
        terms = path_log_density_terms(output, targets, model)
        nll_loss = -(terms * weights[:, None]).sum() / (
            weights.sum() * model.return_count
        )
        distribution_loss = nll_loss
    correlation = weighted_expected_return_correlation(
        output.expectations,
        targets,
        weights,
        target_std=target_std,
    )
    mse = weighted_expected_return_normalized_mse(
        output.expectations,
        targets,
        weights,
        target_std=target_std,
    )
    objective = distribution_loss + correlation_weight * (1.0 - correlation) \
        + mse_weight * mse
    return objective, distribution_loss, nll_loss, crps_loss, correlation, mse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the compressed multi-step return-density model."
    )
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--pause-file", type=Path)
    parser.add_argument("--smoke-batches", type=int)
    parser.add_argument("--replace-smoke", action="store_true")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--evaluation-batch-size", type=int)
    parser.add_argument("--evaluation-interval-epochs", type=int, default=1)
    parser.add_argument(
        "--matmul-precision", choices=("highest", "high", "medium"),
        default="high",
    )
    parser.add_argument(
        "--compile-mode",
        choices=("none", "default", "max-autotune-no-cudagraphs"),
        default="none",
        help=(
            "Compile fixed-shape training only. CUDA Graphs are always "
            "disabled and evaluation remains eager to prevent retained VRAM "
            "pools for multiple graph variants."
        ),
    )
    parser.add_argument(
        "--dynamic-batch-compile",
        action="store_true",
        help=(
            "Compile one graph with a dynamic batch dimension so final partial "
            "and evaluation batches reuse the training graph."
        ),
    )
    return parser.parse_args()


class ImmediateFeatureActivePathDataset:
    """Recent channel rows with the next H chronological active returns."""

    def __init__(
        self,
        root: Path,
        history_root: Path,
        return_count: int,
        feature_history: int = 1,
        train_examples: int | None = None,
        normalization_window_seconds: int | None = None,
        normalization_variance_floor: float = 1e-16,
        normalization_statistic: str = "log-return",
    ) -> None:
        self.root = root
        self.history_root = history_root
        self.return_count = int(return_count)
        self.manifest = json.loads((root / "manifest.json").read_text(
            encoding="utf-8"
        ))
        if self.manifest.get("storageLayout") != "temporal-channel-timeline-v1":
            raise ValueError("compressed path run requires a compact timeline dataset")
        self.channel_count = int(self.manifest["temporalChannelCount"])
        self.feature_history = int(feature_history)
        if self.feature_history <= 0:
            raise ValueError("feature history must be positive")
        self.base_feature_count = self.channel_count * self.feature_history
        self.normalization_window_seconds = normalization_window_seconds
        self.normalization_variance_floor = float(normalization_variance_floor)
        if normalization_statistic not in {"log-return", "log-price"}:
            raise ValueError("unsupported normalization statistic")
        self.normalization_statistic = normalization_statistic
        self.feature_count = self.base_feature_count + (
            2 if normalization_window_seconds is not None else 0
        )
        self.splits: dict[str, tuple[
            np.memmap, np.memmap, np.ndarray, np.ndarray | None, np.ndarray | None
        ]] = {}
        for split in ("train", "validation", "test"):
            stored_count = int(self.manifest["examplesBySplit"][split])
            count = (
                min(stored_count, int(train_examples))
                if split == "train" and train_examples is not None
                else stored_count
            )
            timeline_rows = int(self.manifest["timelineRowsBySplit"][split])
            timeline = np.memmap(
                root / f"{split}.timeline-features.f32", dtype="<f4", mode="r",
                shape=(timeline_rows, self.channel_count),
            )
            origins = np.memmap(
                root / f"{split}.origins.i32", dtype="<i4", mode="r",
                shape=(stored_count,),
            )[:count]
            targets = np.asarray(np.memmap(
                root / f"{split}.targets.f32", dtype="<f4", mode="r",
                shape=(stored_count,),
            )[:count], dtype=np.float32)
            times = np.asarray(np.memmap(
                root / f"{split}.times.f64", dtype="<f8", mode="r",
                shape=(stored_count,),
            )[:count], dtype=np.float64)
            if np.any(targets == 0) or np.any(np.diff(times) <= 0):
                raise ValueError(f"{split} is not a chronological clean corpus")
            if int(origins.min()) < self.feature_history - 1:
                raise ValueError(f"{split} lacks the requested feature history")
            tail = self._active_tail(int(times[-1]), self.return_count - 1)
            extended = np.concatenate((targets, tail))
            paths = np.lib.stride_tricks.sliding_window_view(
                extended, self.return_count
            )[:count]
            if normalization_window_seconds is None:
                return_means, return_variances = None, None
            else:
                statistics_function = (
                    trailing_log_price_statistics
                    if normalization_statistic == "log-price"
                    else trailing_log_return_statistics
                )
                return_means, return_variances = statistics_function(
                    history_root,
                    times,
                    window_seconds=int(normalization_window_seconds),
                    variance_floor=self.normalization_variance_floor,
                )
            self.splits[split] = (
                timeline, origins, paths, return_means, return_variances
            )

    def _active_tail(self, final_target_time_ms: int, count: int) -> np.ndarray:
        cache = CandleCloseCache(self.history_root, rows_per_day=86_400)
        values: list[np.float32] = []
        timestamp = int(final_target_time_ms) + 1_000
        while len(values) < count:
            point = datetime.fromtimestamp(timestamp / 1_000, timezone.utc)
            day = point.date()
            second = point.hour * 3600 + point.minute * 60 + point.second
            current = cache.load(day.isoformat())
            if second == 0:
                previous = cache.load((day - timedelta(days=1)).isoformat())[-1]
            else:
                previous = current[second - 1]
            value = np.float32(np.log(current[second] / previous))
            if value != 0:
                values.append(value)
            timestamp += 1_000
        return np.asarray(values, dtype=np.float32)

    def logical_count(self, split: str) -> int:
        return int(self.splits[split][1].size)

    def iter_batches(
        self,
        split: str,
        batch_size: int,
        *,
        shuffle: bool,
        seed: int,
        limit: int | None = None,
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        timeline, origins, targets, return_means, return_variances = (
            self.splits[split]
        )
        count = origins.size
        if limit is not None:
            count = min(count, int(limit))
        order = np.arange(origins.size, dtype=np.int64)
        if limit is not None and count < origins.size:
            order = np.linspace(0, origins.size - 1, count, dtype=np.int64)
        if shuffle:
            np.random.default_rng(seed).shuffle(order)
        for start in range(0, count, batch_size):
            selected = order[start:start + batch_size]
            selected_origins = np.asarray(origins[selected], dtype=np.int64)
            offsets = np.arange(
                self.feature_history - 1, -1, -1, dtype=np.int64
            )
            feature_rows = np.asarray(
                timeline[selected_origins[:, None] - offsets[None, :]],
                dtype=np.float32,
            ).reshape(selected.size, self.base_feature_count)
            if return_means is not None and return_variances is not None:
                feature_rows = append_return_statistics(
                    feature_rows,
                    return_means[selected],
                    return_variances[selected],
                )
            target_rows = np.asarray(targets[selected], dtype=np.float32)
            yield (
                torch.from_numpy(feature_rows.copy()),
                torch.from_numpy(target_rows.copy()),
                torch.ones(selected.size, dtype=torch.float32),
            )


class CalibrationPathDataset:
    """Immediate features and clean paths immediately before validation."""

    def __init__(
        self,
        root: Path,
        return_count: int,
        feature_history: int = 1,
        *,
        history_root: Path | None = None,
        normalization_window_seconds: int | None = None,
        normalization_variance_floor: float = 1e-16,
        normalization_statistic: str = "log-return",
    ) -> None:
        self.root = root
        self.return_count = int(return_count)
        self.manifest = json.loads((root / "manifest.json").read_text("utf-8"))
        count = int(self.manifest["examples"])
        feature_count = int(self.manifest["featureCount"])
        matrix = np.memmap(
            root / "calibration.features.f32", dtype="<f4", mode="r",
            shape=(count, feature_count),
        )
        channels_value = self.manifest.get("temporalChannelCount")
        history = int(self.manifest.get("featureHistorySeconds", 1))
        requested_history = int(feature_history)
        if requested_history <= 0:
            raise ValueError("feature history must be positive")
        if channels_value is None:
            if requested_history != 1:
                raise ValueError("flat calibration data has no temporal history")
            self.features = matrix
            self.feature_count = feature_count
        else:
            channels = int(channels_value)
            if feature_count != channels * history:
                raise ValueError("calibration temporal dimensions are inconsistent")
            if requested_history > history:
                raise ValueError("calibration history is shorter than requested")
            if self.manifest.get("temporalLayout") == "time-major":
                self.features = np.ascontiguousarray(
                    matrix.reshape(count, history, channels)[:, -requested_history:, :]
                    .reshape(count, channels * requested_history)
                )
            else:
                self.features = np.ascontiguousarray(
                    matrix.reshape(count, channels, history)[:, :, -requested_history:]
                    .transpose(0, 2, 1)
                    .reshape(count, channels * requested_history)
                )
            self.feature_count = channels * requested_history
        targets = np.asarray(np.memmap(
            root / "calibration.targets.f32", dtype="<f4", mode="r",
            shape=(count,),
        ), dtype=np.float32)
        self.times = np.asarray(np.memmap(
            root / "calibration.times.f64", dtype="<f8", mode="r",
            shape=(count,),
        ), dtype=np.float64)
        if np.any(targets == 0) or np.any(np.diff(self.times) <= 0):
            raise ValueError("calibration examples are not clean and chronological")
        if count < self.return_count:
            raise ValueError("calibration split is shorter than one path")
        self.targets = np.lib.stride_tricks.sliding_window_view(
            targets, self.return_count
        )
        if normalization_window_seconds is None:
            self.return_means = None
            self.return_variances = None
        else:
            if history_root is None:
                raise ValueError("normalized calibration requires candle history")
            if normalization_statistic not in {"log-return", "log-price"}:
                raise ValueError("unsupported normalization statistic")
            statistics_function = (
                trailing_log_price_statistics
                if normalization_statistic == "log-price"
                else trailing_log_return_statistics
            )
            self.return_means, self.return_variances = (
                statistics_function(
                    history_root,
                    self.times,
                    window_seconds=int(normalization_window_seconds),
                    variance_floor=float(normalization_variance_floor),
                )
            )
            self.feature_count += 2

    def logical_count(self, split: str) -> int:
        if split != "calibration":
            raise KeyError(split)
        return int(self.targets.shape[0])

    def iter_batches(
        self, split: str, batch_size: int, *, shuffle: bool, seed: int,
        limit: int | None = None,
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        del shuffle, seed
        if split != "calibration":
            raise KeyError(split)
        count = self.logical_count(split)
        if limit is not None:
            count = min(count, int(limit))
        for start in range(0, count, batch_size):
            stop = min(count, start + batch_size)
            feature_rows = np.asarray(
                self.features[start:stop], dtype=np.float32
            )
            if self.return_means is not None \
                    and self.return_variances is not None:
                feature_rows = append_return_statistics(
                    feature_rows,
                    self.return_means[start:stop],
                    self.return_variances[start:stop],
                )
            yield (
                torch.from_numpy(feature_rows.copy()),
                torch.from_numpy(np.asarray(
                    self.targets[start:stop], dtype=np.float32
                ).copy()),
                torch.ones(stop - start, dtype=torch.float32),
            )


class UnionImmediateFeatureActivePathDataset:
    """One current 530-channel union state and one active-return target."""

    def __init__(
        self,
        global_root: Path,
        history_root: Path,
        *,
        return_count: int,
        examples_by_split: dict[str, int],
    ) -> None:
        if int(return_count) != 1:
            raise ValueError(
                "the immediate union path dataset currently supports one return"
            )
        self.return_count = 1
        self.source = FeatureMatrixDataset(
            global_root,
            union_history_root=history_root,
            feature_history_seconds=1,
        )
        self.feature_count = int(self.source.feature_count)
        self.manifest = self.source.manifest
        if int(self.manifest.get("baseFeatureCount", 0)) != 530:
            raise ValueError("immediate path dataset requires the 530-channel union")
        unknown = set(examples_by_split) - {"train", "validation", "test"}
        if unknown:
            raise ValueError(f"unknown union split limits: {sorted(unknown)}")
        self.splits: dict[str, tuple[object, int]] = {}
        for split in ("train", "validation", "test"):
            source_split = self.source.splits[split]
            count = int(examples_by_split[split])
            if count < 1 or count > source_split.count:
                raise ValueError(
                    f"union {split} requested {count:,} of "
                    f"{source_split.count:,} aligned examples"
                )
            self.splits[split] = (source_split, count)

    def logical_count(self, split: str) -> int:
        return self.splits[split][1]

    def iter_batches(
        self,
        split: str,
        batch_size: int,
        *,
        shuffle: bool,
        seed: int,
        limit: int | None = None,
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        values, available = self.splits[split]
        count = available if limit is None else min(available, int(limit))
        order = np.arange(available, dtype=np.int64)
        if count < available:
            order = np.linspace(0, available - 1, count, dtype=np.int64)
        if shuffle:
            np.random.default_rng(seed).shuffle(order)
        for start in range(0, count, batch_size):
            selected = order[start:start + batch_size]
            if shuffle:
                selected = np.sort(selected)
            features = np.asarray(
                values.features[selected], dtype=np.float32
            )
            targets = np.asarray(
                values.targets[selected], dtype=np.float32
            )[:, None]
            yield (
                torch.from_numpy(features.copy()),
                torch.from_numpy(targets.copy()),
                torch.ones(selected.size, dtype=torch.float32),
            )


class UnionImmediateCalibrationDataset:
    """Trailing aligned training states immediately before validation."""

    def __init__(
        self,
        source: UnionImmediateFeatureActivePathDataset | DifferentiableUnion530Dataset,
        examples: int,
    ) -> None:
        if isinstance(source, DifferentiableUnion530Dataset):
            self.source = source.all_splits["train"]
            self.return_count = source.return_count
        else:
            self.source = source.source.splits["train"]
            self.return_count = source.return_count
        self.count = int(examples)
        available = self.source.count - self.return_count + 1
        if self.count < 1 or self.count > available:
            raise ValueError("union calibration count exceeds aligned training data")
        self.offset = available - self.count
        self.feature_count = source.feature_count

    def logical_count(self, split: str) -> int:
        if split != "calibration":
            raise KeyError(split)
        return self.count

    def iter_batches(
        self,
        split: str,
        batch_size: int,
        *,
        shuffle: bool,
        seed: int,
        limit: int | None = None,
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        del shuffle, seed
        if split != "calibration":
            raise KeyError(split)
        count = self.count if limit is None else min(self.count, int(limit))
        for start in range(0, count, batch_size):
            stop = min(count, start + batch_size)
            selected = slice(self.offset + start, self.offset + stop)
            features = np.asarray(
                self.source.features[selected], dtype=np.float32
            )
            targets = np.asarray(
                np.lib.stride_tricks.sliding_window_view(
                    self.source.targets, self.return_count
                )[selected],
                dtype=np.float32,
            )
            yield (
                torch.from_numpy(features.copy()),
                torch.from_numpy(targets.copy()),
                torch.ones(stop - start, dtype=torch.float32),
            )


def pad_weighted_batch(
    features: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Keep compiled calls shape-stable without counting padded examples."""
    count = int(features.shape[0])
    if count == batch_size:
        return features, targets, weights
    if count < 1 or count > batch_size:
        raise ValueError("batch size is outside its requested physical shape")
    padding = batch_size - count
    repeated = torch.arange(padding) % count
    return (
        torch.cat((features, features[repeated]), dim=0),
        torch.cat((targets, targets[repeated]), dim=0),
        torch.cat((weights, torch.zeros(padding, dtype=weights.dtype)), dim=0),
    )


@torch.no_grad()
def collect_expectation_arrays(
    model: CompressedPathReturnDensity, dataset, split: str, *,
    batch_size: int, device: torch.device, limit: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    model.eval()
    for features, target, weights in dataset.iter_batches(
        split, batch_size, shuffle=False, seed=0, limit=limit
    ):
        features, target, weights = pad_weighted_batch(
            features, target, weights, batch_size
        )
        active = weights.bool()
        predictions.append(model(features.to(
            device, non_blocking=True
        )).expectations.cpu()[active].numpy())
        targets.append(target[active].numpy())
    return (
        np.concatenate(predictions).astype(np.float64, copy=False),
        np.concatenate(targets).astype(np.float64, copy=False),
    )


def rolling_online_per_step_affine(
    history_prediction: np.ndarray,
    history_target: np.ndarray,
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    window: int,
    ridge: float,
    input_scales: np.ndarray | None = None,
    output_scales: np.ndarray | None = None,
) -> np.ndarray:
    """Causal affine fits using only targets resolved before each origin."""
    if history_prediction.shape != history_target.shape \
            or prediction.shape != target.shape:
        raise ValueError("online calibration prediction/target shapes differ")
    steps = prediction.shape[1]
    input_scales = np.maximum(
        history_prediction.std(axis=0), 1e-12
    ) if input_scales is None else input_scales
    output_scales = np.maximum(
        history_target.std(axis=0), 1e-12
    ) if output_scales is None else output_scales
    output = np.empty_like(prediction, dtype=np.float64)
    history_events = history_prediction.shape[0]
    current_events = prediction.shape[0]
    ends = history_events + np.arange(current_events)
    starts = np.maximum(0, ends - int(window))
    penalty = np.diag((0.0, float(ridge)))
    for lead in range(steps):
        gram_parts: list[np.ndarray] = []
        right_parts: list[np.ndarray] = []
        for values, targets in (
            (history_prediction, history_target[:, 0]),
            (prediction, target[:, 0]),
        ):
            events = values.shape[0]
            gram = np.zeros((events, 2, 2), dtype=np.float64)
            right = np.zeros((events, 2), dtype=np.float64)
            origins = events - lead
            if origins > 0:
                normalized = values[:origins, lead] / input_scales[lead]
                design = np.stack((np.ones_like(normalized), normalized), axis=1)
                resolved = targets[lead:lead + origins] / output_scales[lead]
                gram[lead:lead + origins] = np.einsum(
                    "ni,nj->nij", design, design
                )
                right[lead:lead + origins] = design * resolved[:, None]
            gram_parts.append(gram)
            right_parts.append(right)
        gram = np.concatenate(gram_parts)
        right = np.concatenate(right_parts)
        cumulative_gram = np.concatenate((
            np.zeros((1, 2, 2), dtype=np.float64), np.cumsum(gram, axis=0)
        ))
        cumulative_right = np.concatenate((
            np.zeros((1, 2), dtype=np.float64), np.cumsum(right, axis=0)
        ))
        rolling_gram = cumulative_gram[ends] - cumulative_gram[starts]
        rolling_right = cumulative_right[ends] - cumulative_right[starts]
        coefficients = np.linalg.solve(
            rolling_gram + penalty[None, :, :], rolling_right[..., None]
        )[..., 0]
        normalized = prediction[:, lead] / input_scales[lead]
        output[:, lead] = output_scales[lead] * (
            coefficients[:, 0] + coefficients[:, 1] * normalized
        )
    return output


def calibration_scales(
    prediction: np.ndarray, target: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.maximum(prediction.std(axis=0), 1e-12),
        np.maximum(target.std(axis=0), 1e-12),
    )


def expectation_metrics_from_arrays(
    prediction: np.ndarray, target: np.ndarray, *, target_std: float,
    cumulative_std: float, device: torch.device,
) -> dict:
    prediction_tensor = torch.from_numpy(prediction).to(device=device)
    target_tensor = torch.from_numpy(target).to(device=device)
    weights = torch.ones(prediction.shape[0], dtype=torch.float64, device=device)
    pooled = MetricAccumulator(target_std, device)
    pooled.add(prediction_tensor.flatten(), target_tensor.flatten(),
               weights[:, None].expand_as(target_tensor).flatten())
    per_lead = []
    for lead in range(prediction.shape[1]):
        metric = MetricAccumulator(target_std, device)
        metric.add(prediction_tensor[:, lead], target_tensor[:, lead], weights)
        per_lead.append(metric.result())
    cumulative = MetricAccumulator(cumulative_std, device)
    cumulative.add(prediction_tensor.sum(dim=1), target_tensor.sum(dim=1), weights)
    return {
        "expectation": pooled.result(),
        "perLeadExpectation": per_lead,
        "cumulativeExpectation": cumulative.result(),
    }


def training_statistics(
    dataset: ImmediateFeatureActivePathDataset, batch_size: int
) -> dict[str, np.ndarray | float]:
    feature_sum = np.zeros(dataset.feature_count, dtype=np.float64)
    feature_square = np.zeros_like(feature_sum)
    target_sum = 0.0
    target_square = 0.0
    cumulative_sum = 0.0
    cumulative_square = 0.0
    examples = 0
    returns = 0
    for features, targets, _weights in dataset.iter_batches(
        "train", batch_size, shuffle=False, seed=0
    ):
        x = features.numpy().astype(np.float64, copy=False)
        y = targets.numpy().astype(np.float64, copy=False)
        feature_sum += x.sum(axis=0)
        feature_square += np.square(x).sum(axis=0)
        target_sum += float(y.sum())
        target_square += float(np.square(y).sum())
        cumulative = y.sum(axis=1)
        cumulative_sum += float(cumulative.sum())
        cumulative_square += float(np.square(cumulative).sum())
        examples += x.shape[0]
        returns += y.size
    feature_mean = feature_sum / examples
    feature_variance = np.maximum(
        feature_square / examples - np.square(feature_mean), 1e-20
    )
    target_mean = target_sum / returns
    target_std = math.sqrt(max(
        target_square / returns - target_mean * target_mean, 1e-20
    ))
    cumulative_mean = cumulative_sum / examples
    cumulative_std = math.sqrt(max(
        cumulative_square / examples - cumulative_mean * cumulative_mean, 1e-20
    ))
    return {
        "featureMean": feature_mean.astype(np.float32),
        "featureStd": np.sqrt(feature_variance).astype(np.float32),
        "targetStd": target_std,
        "cumulativeStd": cumulative_std,
    }


class PathMetrics:
    def __init__(self, target_std: float, cumulative_std: float, steps: int,
                 device: torch.device) -> None:
        self.pooled = MetricAccumulator(target_std, device)
        self.per_step = tuple(MetricAccumulator(target_std, device) for _ in range(steps))
        self.cumulative = MetricAccumulator(cumulative_std, device)
        self.nll_sum = torch.zeros((), dtype=torch.float64, device=device)
        self.per_step_nll_sum = torch.zeros(
            steps, dtype=torch.float64, device=device
        )
        self.crps_sum = torch.zeros((), dtype=torch.float64, device=device)
        self.per_step_crps_sum = torch.zeros(
            steps, dtype=torch.float64, device=device
        )
        self.crps_count = torch.zeros((), dtype=torch.float64, device=device)
        self.example_count = torch.zeros((), dtype=torch.float64, device=device)
        self.target_std = float(target_std)
        self.count = torch.zeros((), dtype=torch.float64, device=device)

    def add(self, prediction: torch.Tensor, target: torch.Tensor,
            weights: torch.Tensor, log_density: torch.Tensor,
            crps: torch.Tensor | None = None) -> None:
        expanded = weights[:, None].expand_as(target)
        self.pooled.add(prediction.flatten(), target.flatten(), expanded.flatten())
        for index, metric in enumerate(self.per_step):
            metric.add(prediction[:, index], target[:, index], weights)
        self.cumulative.add(prediction.sum(dim=1), target.sum(dim=1), weights)
        self.nll_sum += (expanded.double() * -log_density.double()).sum()
        self.per_step_nll_sum += (
            weights[:, None].double() * -log_density.double()
        ).sum(dim=0)
        self.example_count += weights.double().sum()
        if crps is not None:
            if crps.shape != target.shape:
                raise ValueError("CRPS metric shape differs from targets")
            self.crps_sum += (expanded.double() * crps.double()).sum()
            self.per_step_crps_sum += (
                weights[:, None].double() * crps.double()
            ).sum(dim=0)
            self.crps_count += expanded.double().sum()
        self.count += expanded.double().sum()

    def result(self) -> dict:
        return {
            "negativeLogLikelihood": float(self.nll_sum / self.count),
            "perLeadNegativeLogLikelihood": [
                float(value / self.example_count)
                for value in self.per_step_nll_sum
            ],
            "meanCrps": (
                float(self.crps_sum / self.crps_count)
                if float(self.crps_count) > 0 else None
            ),
            "normalizedCrps": (
                float(self.crps_sum / self.crps_count / self.target_std)
                if float(self.crps_count) > 0 else None
            ),
            "perLeadMeanCrps": (
                [
                    float(value / self.example_count)
                    for value in self.per_step_crps_sum
                ]
                if float(self.crps_count) > 0 else None
            ),
            "perLeadNormalizedCrps": (
                [
                    float(value / self.example_count / self.target_std)
                    for value in self.per_step_crps_sum
                ]
                if float(self.crps_count) > 0 else None
            ),
            "expectation": self.pooled.result(),
            "perLeadExpectation": [value.result() for value in self.per_step],
            "cumulativeExpectation": self.cumulative.result(),
        }


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataset: ImmediateFeatureActivePathDataset,
    split: str,
    *,
    batch_size: int,
    target_std: float,
    cumulative_std: float,
    device: torch.device,
    limit: int | None = None,
    collect_arrays: bool = False,
) -> dict:
    model.eval()
    result = PathMetrics(
        target_std, cumulative_std, model.return_count, device
    )
    predictions: list[np.ndarray] = []
    targets_collected: list[np.ndarray] = []
    for features, targets, weights in dataset.iter_batches(
        split, batch_size, shuffle=False, seed=0, limit=limit
    ):
        features, targets, weights = pad_weighted_batch(
            features, targets, weights, batch_size
        )
        features = features.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        weights = weights.to(device, non_blocking=True)
        output = (
            model(features, targets)
            if getattr(model, "requires_joint_targets", False)
            else model(features)
        )
        result.add(
            output.expectations, targets, weights,
            path_log_density_terms(output, targets, model),
            discrete_component_crps_terms(output, targets, model),
        )
        if collect_arrays:
            active = weights.bool()
            predictions.append(output.expectations[active].cpu().numpy())
            targets_collected.append(targets[active].cpu().numpy())
    values = result.result()
    if collect_arrays:
        values["arrays"] = {
            "prediction": np.concatenate(predictions).astype(
                np.float64, copy=False
            ),
            "target": np.concatenate(targets_collected).astype(
                np.float64, copy=False
            ),
        }
    return values


@torch.no_grad()
def evaluate_joint_first_step(
    model: JointPrefixContractedCyclicPathMatrixDensity,
    dataset: ImmediateFeatureActivePathDataset,
    split: str,
    *,
    batch_size: int,
    target_std: float,
    device: torch.device,
    limit: int | None = None,
    collect_arrays: bool = False,
) -> dict:
    """Evaluate the exact first conditional without unrolling later leads."""
    model.eval()
    result = PathMetrics(target_std, target_std, 1, device)
    predictions: list[np.ndarray] = []
    targets_collected: list[np.ndarray] = []
    for features, targets, weights in dataset.iter_batches(
        split, batch_size, shuffle=False, seed=0, limit=limit
    ):
        features, targets, weights = pad_weighted_batch(
            features, targets, weights, batch_size
        )
        features = features.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        weights = weights.to(device, non_blocking=True)
        terms, expectation = model.contracted_joint_training_terms(
            features,
            targets,
            step_count=1,
        )
        result.add(
            expectation[:, None],
            targets[:, :1],
            weights,
            terms,
        )
        if collect_arrays:
            active = weights.bool()
            predictions.append(expectation[active, None].cpu().numpy())
            targets_collected.append(targets[active, :1].cpu().numpy())
    values = result.result()
    if collect_arrays:
        values["arrays"] = {
            "prediction": np.concatenate(predictions).astype(
                np.float64, copy=False
            ),
            "target": np.concatenate(targets_collected).astype(
                np.float64, copy=False
            ),
        }
    return values


@torch.no_grad()
def collect_joint_first_step_expectation_arrays(
    model: JointPrefixContractedCyclicPathMatrixDensity,
    dataset,
    split: str,
    *,
    batch_size: int,
    device: torch.device,
    limit: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    values = evaluate_joint_first_step(
        model,
        dataset,
        split,
        batch_size=batch_size,
        target_std=1.0,
        device=device,
        limit=limit,
        collect_arrays=True,
    )
    arrays = values["arrays"]
    return arrays["prediction"], arrays["target"]


@torch.no_grad()
def update_weight_ema(
    ema: dict[str, torch.Tensor], model: torch.nn.Module, decay: float
) -> None:
    for name, value in model.state_dict().items():
        if value.is_floating_point():
            ema[name].lerp_(value.detach(), 1.0 - decay)
        else:
            ema[name].copy_(value)


@contextmanager
def use_state(model: torch.nn.Module, state: dict[str, torch.Tensor]):
    original = {name: value.detach().clone() for name, value in model.state_dict().items()}
    model.load_state_dict(state)
    try:
        yield
    finally:
        model.load_state_dict(original)


def checkpoint_payload(
    model: torch.nn.Module,
    ema: dict[str, torch.Tensor],
    optimizers: tuple[torch.optim.Optimizer, ...],
    *, epoch: int, global_step: int, plan_hash: str,
    runner_contract: str, best: dict,
) -> dict:
    return {
        "model": model.state_dict(), "emaModel": ema,
        "optimizers": [value.state_dict() for value in optimizers],
        "epoch": epoch, "globalStep": global_step, "best": best,
        "planSha256": plan_hash, "runnerContract": runner_contract,
    }


def main() -> None:
    args = parse_args()
    if args.batch_size is not None and args.batch_size < 1:
        raise ValueError("batch size must be positive")
    if args.evaluation_batch_size is not None \
            and args.evaluation_batch_size < 1:
        raise ValueError("evaluation batch size must be positive")
    if args.evaluation_interval_epochs < 1:
        raise ValueError("evaluation interval must be positive")
    torch.set_float32_matmul_precision(args.matmul_precision)
    repo = Path(__file__).resolve().parents[1]
    plan_file = args.plan if args.plan.is_absolute() else repo / args.plan
    plan = json.loads(plan_file.read_text(encoding="utf-8"))
    plan_hash = canonical_hash(plan)
    run_root = (repo / plan["runDir"]).resolve()
    if args.replace_smoke:
        status_file = run_root / "state/status.json"
        status = json.loads(status_file.read_text(encoding="utf-8")) \
            if status_file.is_file() else {}
        if status.get("stage") != "smoke-complete":
            raise ValueError("--replace-smoke may only replace a completed smoke run")
        smoke_log = run_root / "logs/training.jsonl"
        if smoke_log.is_file():
            archive = run_root / "logs/smoke-training.jsonl"
            if archive.exists():
                timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                archive = run_root / f"logs/smoke-training-{timestamp}.jsonl"
            smoke_log.replace(archive)
        for file in (
            run_root / "checkpoints/last.json",
            run_root / "checkpoints/selections/validation-nll.json",
            run_root / "checkpoints/selections/validation-crps.json",
            run_root / "checkpoints/selections/validation-mse.json",
            run_root / "checkpoints/selections/validation-correlation.json",
            run_root / "state/plan.json",
            status_file,
        ):
            file.unlink(missing_ok=True)
    reporter = Reporter(run_root)
    pause_file = None if args.pause_file is None else (
        args.pause_file if args.pause_file.is_absolute() else repo / args.pause_file
    )
    try:
        training = plan["training"]
        architecture = plan["architecture"]
        architecture_contract = architecture.get("contract")
        if architecture_contract not in {
            ARCHITECTURE_CONTRACT,
            RECURRENT_MARKET_ARCHITECTURE_CONTRACT,
            RESIDUAL_RECURRENT_MARKET_ARCHITECTURE_CONTRACT,
            LOW_RANK_PATH_MATRIX_ARCHITECTURE_CONTRACT,
            DIRECT_FACTORIZED_ARCHITECTURE_CONTRACT,
            CYCLIC_DENSE_COMPRESSED_ARCHITECTURE_CONTRACT,
            JOINT_PREFIX_CONTRACTED_CYCLIC_ARCHITECTURE_CONTRACT,
            ACTOR_MARKET_PROCESS_ARCHITECTURE_CONTRACT,
            ACTOR_MARKET_PATH_MATRIX_ARCHITECTURE_CONTRACT,
        }:
            raise ValueError("compressed path architecture contract changed")
        recurrent_market = architecture_contract in {
            RECURRENT_MARKET_ARCHITECTURE_CONTRACT,
            RESIDUAL_RECURRENT_MARKET_ARCHITECTURE_CONTRACT,
        }
        residual_recurrent_market = (
            architecture_contract
            == RESIDUAL_RECURRENT_MARKET_ARCHITECTURE_CONTRACT
        )
        low_rank_path_matrix = (
            architecture_contract == LOW_RANK_PATH_MATRIX_ARCHITECTURE_CONTRACT
        )
        direct_factorized_path_matrix = (
            architecture_contract == DIRECT_FACTORIZED_ARCHITECTURE_CONTRACT
        )
        joint_prefix_contracted_path_matrix = (
            architecture_contract
            == JOINT_PREFIX_CONTRACTED_CYCLIC_ARCHITECTURE_CONTRACT
        )
        recurrent_activation_checkpointing = bool(
            architecture.get("recurrentActivationCheckpointing", False)
        )
        if recurrent_activation_checkpointing \
                and not joint_prefix_contracted_path_matrix:
            raise ValueError(
                "recurrent activation checkpointing currently requires the "
                "joint-prefix architecture"
            )
        cyclic_dense_compressed_path_matrix = architecture_contract in {
            CYCLIC_DENSE_COMPRESSED_ARCHITECTURE_CONTRACT,
            JOINT_PREFIX_CONTRACTED_CYCLIC_ARCHITECTURE_CONTRACT,
        }
        actor_market_process = (
            architecture_contract
            == ACTOR_MARKET_PROCESS_ARCHITECTURE_CONTRACT
        )
        actor_market_path_matrix = (
            architecture_contract
            == ACTOR_MARKET_PATH_MATRIX_ARCHITECTURE_CONTRACT
        )
        base_history_union = plan.get("baseHistoryDatasetDir") is not None
        union_immediate = (
            plan.get("unionHistoryDatasetDir") is not None or base_history_union
        )
        if actor_market_path_matrix:
            runner_contract = ACTOR_MARKET_PATH_MATRIX_RUNNER_CONTRACT
        elif actor_market_process:
            runner_contract = ACTOR_MARKET_PROCESS_RUNNER_CONTRACT
        elif joint_prefix_contracted_path_matrix:
            runner_contract = UNION_JOINT_PREFIX_CYCLIC_PATH_MATRIX_RUNNER_CONTRACT
        elif cyclic_dense_compressed_path_matrix:
            runner_contract = (
                UNION_IMMEDIATE_CYCLIC_DENSE_PATH_MATRIX_RUNNER_CONTRACT
                if union_immediate
                else CYCLIC_DENSE_COMPRESSED_PATH_MATRIX_RUNNER_CONTRACT
            )
        elif direct_factorized_path_matrix:
            runner_contract = DIRECT_FACTORIZED_PATH_MATRIX_RUNNER_CONTRACT
        elif low_rank_path_matrix:
            runner_contract = LOW_RANK_PATH_MATRIX_RUNNER_CONTRACT
        elif recurrent_market:
            runner_contract = RECURRENT_MARKET_RUNNER_CONTRACT
        else:
            runner_contract = RUNNER_CONTRACT
        active_distribution_loss = distribution_loss_type(training)
        crps_training = (
            active_distribution_loss == DISCRETE_CRPS_DISTRIBUTION_LOSS
        )
        if joint_prefix_contracted_path_matrix and active_distribution_loss \
                != JOINT_PREFIX_NLL_DISTRIBUTION_LOSS:
            raise ValueError(
                "joint-prefix architecture requires contracted joint-path NLL"
            )
        if not joint_prefix_contracted_path_matrix and active_distribution_loss \
                == JOINT_PREFIX_NLL_DISTRIBUTION_LOSS:
            raise ValueError(
                "contracted joint-path NLL requires the joint-prefix architecture"
            )
        if joint_prefix_contracted_path_matrix and crps_training:
            raise ValueError("joint-prefix contraction currently trains with NLL")
        adversarial_input = training.get("adversarialInput")
        if adversarial_input is not None:
            if not base_history_union or adversarial_input.get("type") \
                    != "base-history-projected-gradient-ascent-v1":
                raise ValueError(
                    "adversarial inputs require the differentiable base-history union"
                )
            if int(adversarial_input.get("steps", 0)) != 1:
                raise ValueError("base-history adversarial generation currently uses one step")
            adversarial_epsilon_rms = float(adversarial_input["epsilonRms"])
            adversarial_weight = float(adversarial_input.get("adversarialWeight", 0.5))
            if not 0 < adversarial_epsilon_rms or not 0 < adversarial_weight <= 1:
                raise ValueError("invalid base-history adversarial settings")
        else:
            adversarial_epsilon_rms = 0.0
            adversarial_weight = 0.0
        adversarial_output = output_adversarial_specification(training)
        if joint_prefix_contracted_path_matrix and adversarial_output is not None:
            raise ValueError(
                "joint-prefix output adversarial training needs branch-level perturbations"
            )
        if adversarial_output is None:
            output_adversarial_epsilon_rms = 0.0
            output_adversarial_weight = 0.0
        else:
            output_adversarial_epsilon_rms = float(
                adversarial_output["epsilonRms"]
            )
            output_adversarial_weight = float(
                adversarial_output.get("adversarialWeight", 0.5)
            )
        sam = training.get("sam")
        sam_rho = 0.0 if sam is None else float(sam["rho"])
        if sam_rho < 0 or not math.isfinite(sam_rho):
            raise ValueError("SAM rho must be finite and non-negative")
        if adversarial_input is not None and sam_rho > 0:
            raise ValueError("base-history adversarial training does not combine with SAM")
        if adversarial_output is not None and sam_rho > 0:
            raise ValueError("output adversarial training does not combine with SAM")
        if crps_training and sam_rho > 0:
            raise ValueError("discrete CRPS training does not combine with SAM")
        expected_return_correlation_loss_weight_schedule = training.get(
            "expectedReturnCorrelationLossWeightSchedule"
        )
        expected_return_correlation_loss_weight = (
            expected_return_correlation_loss_weight_at_epoch(training, 0)
        )
        expected_return_mse_loss_weight_schedule = training.get(
            "expectedReturnMseLossWeightSchedule"
        )
        expected_return_mse_loss_weight = expected_return_mse_loss_weight_at_epoch(
            training, 0
        )
        contracted_only_joint_training = bool(
            joint_prefix_contracted_path_matrix
            and not crps_training
            and adversarial_input is None
            and adversarial_output is None
            and sam_rho == 0
            and expected_return_correlation_loss_weight == 0
            and expected_return_mse_loss_weight == 0
            and expected_return_correlation_loss_weight_schedule is None
            and expected_return_mse_loss_weight_schedule is None
        )
        input_dropout_probability = float(
            training.get("inputDropoutProbability", 0.0)
        )
        if not 0.0 <= input_dropout_probability < 1.0:
            raise ValueError("input dropout probability must be in [0, 1)")
        if input_dropout_probability > 0 \
                and not cyclic_dense_compressed_path_matrix:
            raise ValueError(
                "input dropout is currently implemented for the cyclic dense model"
            )
        embedding_dropout_schedule = training.get("embeddingDropoutSchedule")
        embedding_dropout_probability = embedding_dropout_probability_at_epoch(
            training, 0
        )
        if embedding_dropout_probability > 0 \
                and not cyclic_dense_compressed_path_matrix:
            raise ValueError(
                "embedding dropout is currently implemented for the cyclic dense model"
            )
        ema_half_life = float(training["weightEma"]["halfLifeEpochs"])
        seed = int(training["seed"])
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        device = torch.device(training["device"])
        feature_history = int(architecture.get("inputFeatureLags", 1))
        target_normalization = plan.get("targetNormalization")
        if target_normalization is None:
            normalization_window_seconds = None
            normalization_variance_floor = 1e-16
        else:
            normalization_type = target_normalization.get("type")
            if normalization_type not in {
                "causal-trailing-log-return-zscore",
                "causal-trailing-log-price-zscore-difference",
            }:
                raise ValueError("unsupported target normalization")
            if not cyclic_dense_compressed_path_matrix:
                raise ValueError(
                    "trailing target normalization currently requires the "
                    "cyclic dense path model"
                )
            normalization_window_seconds = int(
                target_normalization["windowSeconds"]
            )
            normalization_variance_floor = float(
                target_normalization.get("varianceFloor", 1e-16)
            )
            normalization_statistic = (
                "log-price"
                if normalization_type
                == "causal-trailing-log-price-zscore-difference"
                else "log-return"
            )
        if target_normalization is None:
            normalization_statistic = "log-return"
            target_normalization_center = True
        else:
            target_normalization_center = (
                normalization_type == "causal-trailing-log-return-zscore"
            )
        if union_immediate:
            if feature_history != 1 or int(plan.get("featureHistorySeconds", 0)) != 1:
                raise ValueError("immediate union density requires one feature second")
            if target_normalization is not None:
                raise ValueError("immediate union density uses raw return targets")
            if base_history_union:
                dataset = DifferentiableUnion530Dataset(
                    (repo / plan["baseHistoryDatasetDir"]).resolve(),
                    examples_by_split={
                        name: int(value)
                        for name, value in plan["examplesBySplit"].items()
                    },
                    return_count=int(architecture["returnCount"]),
                )
            else:
                dataset = UnionImmediateFeatureActivePathDataset(
                    (repo / plan["datasetDir"]).resolve(),
                    (repo / plan["unionHistoryDatasetDir"]).resolve(),
                    return_count=int(architecture["returnCount"]),
                    examples_by_split={
                        name: int(value)
                        for name, value in plan["examplesBySplit"].items()
                    },
                )
        else:
            dataset = ImmediateFeatureActivePathDataset(
                (repo / plan["datasetDir"]).resolve(),
                (repo / plan["historyDir"]).resolve(),
                int(architecture["returnCount"]),
                feature_history,
                int(plan["subset"]["examples"]),
                normalization_window_seconds,
                normalization_variance_floor,
                normalization_statistic,
            )
        calibration_spec = plan.get("validationCalibration")
        if calibration_spec is None \
                or calibration_spec.get("type") != "online-affine-log-per-step":
            raise ValueError(
                "compressed path training requires online per-step calibration"
            )
        calibration_window = int(calibration_spec["windowActiveReturns"])
        calibration_ridge = float(calibration_spec.get("ridge", 0.0))
        if union_immediate:
            if calibration_spec.get("source") != "trailing-union-train":
                raise ValueError("union calibration must use trailing training states")
            calibration_dataset = UnionImmediateCalibrationDataset(
                dataset,
                int(calibration_spec["examples"]),
            )
        else:
            calibration_dataset = CalibrationPathDataset(
                (repo / calibration_spec["datasetDir"]).resolve(),
                int(architecture["returnCount"]),
                feature_history,
                history_root=(repo / plan["historyDir"]).resolve(),
                normalization_window_seconds=normalization_window_seconds,
                normalization_variance_floor=normalization_variance_floor,
                normalization_statistic=normalization_statistic,
            )
        if calibration_dataset.feature_count != dataset.feature_count:
            raise ValueError("calibration and training feature counts differ")
        if calibration_dataset.logical_count("calibration") < calibration_window:
            raise ValueError("calibration corpus is shorter than its online window")
        counts = {name: dataset.logical_count(name) for name in dataset.splits}
        if counts["train"] != int(plan["subset"]["examples"]):
            raise RuntimeError(f"training clean count changed: {counts}")
        snapshot = {"planSha256": plan_hash, "plan": plan}
        snapshot_file = run_root / "state/plan.json"
        if snapshot_file.is_file() and json.loads(snapshot_file.read_text(
            encoding="utf-8"
        )) != snapshot:
            raise ValueError("run directory belongs to a different plan")
        atomic_json(snapshot, snapshot_file)
        reporter.emit({
            "event": "minute-return-dataset-selected", "planId": plan["id"],
            "counts": counts, "featureCount": dataset.feature_count,
            "returnCount": int(architecture["returnCount"]),
        })
        reporter.status("computing-training-statistics", planId=plan["id"])
        stats = training_statistics(dataset, int(training["evaluationBatchSize"]))
        density_file = (repo / plan["density"]["source"]).resolve()
        if actor_market_path_matrix:
            density = KnotDensityContract.load(
                density_file, fit=str(int(architecture["outputKnots"]))
            )
            model = ActorMarketPathMatrixDensity(
                torch.from_numpy(stats["featureMean"]),
                torch.from_numpy(stats["featureStd"]),
                density,
                embedding_width=int(architecture["embeddingWidth"]),
                actor_width=int(architecture["actorWidth"]),
                actor_decision_width=int(architecture["actorDecisionWidth"]),
                market_width=int(architecture["marketWidth"]),
                actor_count=int(architecture["actorCount"]),
                market_count=int(architecture["marketCount"]),
                action_count=int(architecture["actionCount"]),
                action_basis_width=int(architecture["actionBasisWidth"]),
                reward_width=int(architecture["rewardWidth"]),
                path_embedding_width=int(architecture["pathEmbeddingWidth"]),
                path_count=int(architecture["pathCount"]),
                return_count=int(architecture["returnCount"]),
                stage_block_count=int(architecture["stageBlockCount"]),
                path_compression_width=int(
                    architecture["pathCompressionWidth"]
                ),
                joint_compression_width=int(
                    architecture["jointCompressionWidth"]
                ),
                certainty_maximum=float(architecture["certaintyMaximum"]),
                quadrature_order=int(architecture.get("quadratureOrder", 16)),
                initial_radius=float(architecture["initialRadius"]),
                minimum_radius=float(architecture["minimumRadius"]),
                learnable_centering=bool(architecture["learnableCentering"]),
            ).to(device)
        elif actor_market_process:
            density = KnotDensityContract.load(
                density_file, fit=str(int(architecture["outputKnots"]))
            )
            model = ActorMarketProcessDensity(
                torch.from_numpy(stats["featureMean"]),
                torch.from_numpy(stats["featureStd"]),
                density,
                embedding_width=int(architecture["embeddingWidth"]),
                actor_width=int(architecture["actorWidth"]),
                actor_decision_width=int(architecture["actorDecisionWidth"]),
                market_width=int(architecture["marketWidth"]),
                actor_count=int(architecture["actorCount"]),
                market_count=int(architecture["marketCount"]),
                action_count=int(architecture["actionCount"]),
                action_basis_width=int(architecture["actionBasisWidth"]),
                reward_width=int(architecture["rewardWidth"]),
                return_count=int(architecture["returnCount"]),
                certainty_maximum=float(architecture["certaintyMaximum"]),
                quadrature_order=int(architecture.get("quadratureOrder", 16)),
                initial_radius=float(architecture["initialRadius"]),
                minimum_radius=float(architecture["minimumRadius"]),
                learnable_centering=bool(architecture["learnableCentering"]),
            ).to(device)
        elif cyclic_dense_compressed_path_matrix:
            density = KnotDensityContract.load(
                density_file, fit=str(int(architecture["outputKnots"]))
            )
            model_type = (
                JointPrefixContractedCyclicPathMatrixDensity
                if joint_prefix_contracted_path_matrix
                else CyclicDenseCompressedPathMatrixDensity
            )
            model = model_type(
                torch.from_numpy(stats["featureMean"]),
                torch.from_numpy(stats["featureStd"]),
                density,
                market_width=int(architecture["marketWidth"]),
                path_embedding_width=int(architecture["pathEmbeddingWidth"]),
                path_count=int(architecture["pathCount"]),
                return_count=int(architecture["returnCount"]),
                stage_block_count=int(architecture["stageBlockCount"]),
                path_compression_width=int(
                    architecture["pathCompressionWidth"]
                ),
                joint_compression_width=int(
                    architecture["jointCompressionWidth"]
                ),
                initial_radius=float(architecture["initialRadius"]),
                minimum_radius=float(architecture["minimumRadius"]),
                learnable_centering=bool(architecture["learnableCentering"]),
                input_dropout_probability=input_dropout_probability,
                embedding_dropout_probability=embedding_dropout_probability,
                target_normalization_variance_floor=(
                    None if target_normalization is None
                    else normalization_variance_floor
                ),
                target_normalization_center=target_normalization_center,
                **({
                    "recurrent_activation_checkpointing": (
                        recurrent_activation_checkpointing
                    ),
                } if joint_prefix_contracted_path_matrix else {}),
            ).to(device)
        elif direct_factorized_path_matrix:
            density = KnotDensityContract.load(
                density_file, fit=str(int(architecture["outputKnots"]))
            )
            model = DirectFactorizedPathMatrixDensity(
                torch.from_numpy(stats["featureMean"]),
                torch.from_numpy(stats["featureStd"]),
                density,
                market_width=int(architecture["marketWidth"]),
                path_embedding_width=int(architecture["pathEmbeddingWidth"]),
                path_count=int(architecture["pathCount"]),
                return_count=int(architecture["returnCount"]),
                factor_rank=int(architecture["factorRank"]),
                hidden_width_threshold=int(
                    architecture["hiddenWidthThreshold"]
                ),
                initial_radius=float(architecture["initialRadius"]),
                minimum_radius=float(architecture["minimumRadius"]),
                learnable_centering=bool(architecture["learnableCentering"]),
            ).to(device)
        elif low_rank_path_matrix:
            density = KnotDensityContract.load(
                density_file, fit=str(int(architecture["outputKnots"]))
            )
            model = DynamicLowRankPathMatrixDensity(
                torch.from_numpy(stats["featureMean"]),
                torch.from_numpy(stats["featureStd"]),
                density,
                market_width=int(architecture["marketWidth"]),
                path_embedding_width=int(architecture["pathEmbeddingWidth"]),
                path_count=int(architecture["pathCount"]),
                return_count=int(architecture["returnCount"]),
                matrix_rank=int(architecture["matrixRank"]),
                hidden_width_cap=int(architecture["hiddenWidthCap"]),
                initial_radius=float(architecture["initialRadius"]),
                minimum_radius=float(architecture["minimumRadius"]),
                learnable_centering=bool(architecture["learnableCentering"]),
            ).to(device)
        elif recurrent_market:
            widths = tuple(int(value) for value in architecture["stateWidths"])
            density = KnotDensityContract.load(
                density_file, fit=str(int(architecture["outputKnots"]))
            )
            model_type = (
                ResidualRecurrentMarketPathDensity
                if residual_recurrent_market else RecurrentMarketPathDensity
            )
            model = model_type(
                torch.from_numpy(stats["featureMean"]),
                torch.from_numpy(stats["featureStd"]),
                density,
                market_width=int(architecture["marketWidth"]),
                state_widths=widths,
                transition_rank=int(architecture.get("transitionRank", 1)),
                initial_radius=float(architecture["initialRadius"]),
                minimum_radius=float(architecture["minimumRadius"]),
                learnable_centering=bool(architecture["learnableCentering"]),
            ).to(device)
        else:
            widths = tuple(int(value) for value in architecture["stateWidths"])
            densities = tuple(
                KnotDensityContract.load(density_file, fit=str(width))
                for width in widths
            )
            model = CompressedPathReturnDensity(
                torch.from_numpy(stats["featureMean"]),
                torch.from_numpy(stats["featureStd"]),
                densities,
                market_width=int(architecture["marketWidth"]),
                state_widths=widths,
                initial_radius=float(architecture["initialRadius"]),
                minimum_radius=float(architecture["minimumRadius"]),
                learnable_centering=bool(architecture["learnableCentering"]),
            ).to(device)
        parameter_count = sum(value.numel() for value in model.parameters())
        trainable_count = sum(
            value.numel() for value in model.parameters() if value.requires_grad
        )
        optimizer_parameter_groups(model)
        optimizers = build_optimizers(model, training, device)
        epochs = int(training["epochs"])
        batch_size = (
            int(args.batch_size)
            if args.batch_size is not None
            else int(training["batchSize"])
        )
        evaluation_batch_size = (
            int(args.evaluation_batch_size)
            if args.evaluation_batch_size is not None
            else int(training["evaluationBatchSize"])
        )
        evaluation_interval = int(args.evaluation_interval_epochs)
        steps_per_epoch = math.ceil(counts["train"] / batch_size)
        ema_decay = mean_teacher_ema_decay(ema_half_life, steps_per_epoch)
        ema = {
            name: value.detach().clone()
            for name, value in model.state_dict().items()
        }
        best = {
            "validation-nll": {"score": math.inf, "epoch": -1},
            "validation-mse": {"score": math.inf, "epoch": -1},
            "validation-correlation": {"score": -math.inf, "epoch": -1},
        }
        if crps_training:
            best["validation-crps"] = {"score": math.inf, "epoch": -1}
        last_file = run_root / "checkpoints/last.json"
        start_epoch = 0
        global_step = 0
        if checkpoint_exists(last_file):
            saved = load_torch_checkpoint(last_file, map_location=device,
                                          weights_only=False)
            if saved.get("planSha256") != plan_hash \
                    or saved.get("runnerContract") != runner_contract:
                raise ValueError("compressed path checkpoint contract changed")
            model.load_state_dict(saved["model"])
            ema = saved["emaModel"]
            for optimizer, state in zip(
                optimizers, saved["optimizers"], strict=True
            ):
                optimizer.load_state_dict(state)
            start_epoch = int(saved["epoch"]) + 1
            global_step = int(saved["globalStep"])
            best = saved["best"]
            if crps_training:
                best.setdefault(
                    "validation-crps", {"score": math.inf, "epoch": -1}
                )
        training_model = model
        compile_scope = "disabled"
        if args.compile_mode != "none":
            # TorchInductor keys artifacts by graph, inputs, compiler options,
            # and generated source. Keep one repository-wide cache so model
            # variants with compatible blocks and batch shapes reuse compiled
            # kernels instead of paying the multi-minute warm-up per run.
            compile_cache = (
                repo / "data/training/cache/torchinductor"
            ).resolve()
            compile_cache.mkdir(parents=True, exist_ok=True)
            os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(compile_cache)
            os.environ["TRITON_CACHE_DIR"] = str(compile_cache / "triton")
            if joint_prefix_contracted_path_matrix:
                if not isinstance(
                    model, JointPrefixContractedCyclicPathMatrixDensity
                ):
                    raise TypeError("joint-prefix model type changed")
                model.compile_shared_recurrent_step(
                    mode=args.compile_mode,
                    dynamic=args.dynamic_batch_compile,
                )
                compile_scope = (
                    "one-shared-contracted-recurrent-step-training-only"
                    if contracted_only_joint_training
                    else "one-shared-recurrent-step-training-only"
                )
            else:
                compile_arguments = {
                    "fullgraph": False,
                    "dynamic": args.dynamic_batch_compile,
                }
                if args.compile_mode == "default":
                    compile_arguments["options"] = {
                        "triton.cudagraphs": False
                    }
                else:
                    # PyTorch exposes this as a first-class mode whose settings
                    # already exclude CUDA Graphs. Its API rejects combining a
                    # named mode with an explicit options mapping.
                    compile_arguments["mode"] = args.compile_mode
                training_model = torch.compile(model, **compile_arguments)
                compile_scope = "fixed-shape-training-only"
        base_adversary = None
        if adversarial_input is not None:
            if not isinstance(dataset, DifferentiableUnion530Dataset):
                raise TypeError("base-history adversarial dataset was not constructed")
            base_adversary = BaseHistoryAdversary(
                dataset.base,
                device=device,
                epsilon_rms=adversarial_epsilon_rms,
                attack_examples=int(
                    adversarial_input.get("attackExamples", batch_size)
                ),
                seed=seed,
            )
        objective_name = f'mean-{int(architecture["returnCount"])}-step-'
        objective_name += (
            "normalized-discrete-component-crps"
            if crps_training else (
                "forward-contracted-joint-path-negative-log-likelihood"
                if joint_prefix_contracted_path_matrix
                else "marginal-negative-log-likelihood"
            )
        )
        if expected_return_correlation_loss_weight > 0:
            objective_name += "-plus-expected-return-correlation"
        if expected_return_mse_loss_weight > 0:
            objective_name += "-plus-expected-return-normalized-mse"
        if adversarial_output is not None:
            objective_name += "-plus-component-logit-output-adversarial"
        reporter.emit({
            "event": "training-start", "planId": plan["id"],
            "startEpoch": start_epoch, "epochs": epochs,
            "parameters": parameter_count, "trainableParameters": trainable_count,
            "objective": objective_name,
            "distributionLoss": training.get("distributionLoss", {
                "type": (
                    JOINT_PREFIX_NLL_DISTRIBUTION_LOSS
                    if joint_prefix_contracted_path_matrix
                    else NLL_DISTRIBUTION_LOSS
                ),
            }),
            "expectedReturnCorrelationLossWeight": (
                expected_return_correlation_loss_weight
            ),
            "expectedReturnCorrelationLossWeightSchedule": (
                expected_return_correlation_loss_weight_schedule
            ),
            "expectedReturnMseLossWeight": expected_return_mse_loss_weight,
            "expectedReturnMseLossWeightSchedule": (
                expected_return_mse_loss_weight_schedule
            ),
            "samRho": sam_rho,
            "inputDropoutProbability": input_dropout_probability,
            "embeddingDropoutProbability": embedding_dropout_probability_at_epoch(
                training, start_epoch
            ),
            "embeddingDropoutSchedule": embedding_dropout_schedule,
            "weightEmaHalfLifeEpochs": ema_half_life,
            "weightEmaDecayPerStep": ema_decay,
            "adversarialInput": adversarial_input,
            "adversarialOutput": adversarial_output,
            "validationCalibration": calibration_spec,
            "runtimeOptimization": {
                "batchSize": batch_size,
                "evaluationBatchSize": evaluation_batch_size,
                "evaluationIntervalEpochs": evaluation_interval,
                "recurrentActivationCheckpointing": (
                    recurrent_activation_checkpointing
                ),
                "matmulPrecision": args.matmul_precision,
                "compileMode": args.compile_mode,
                "compileCacheScope": (
                    "disabled" if args.compile_mode == "none"
                    else "repository-shared"
                ),
                "compileScope": compile_scope,
                "cudaGraphsEnabled": False,
                "evaluationExecution": "eager",
                "jointTrainingExecution": (
                    "contracted-only-exact"
                    if contracted_only_joint_training
                    else "complete-forecast-and-contracted"
                ) if joint_prefix_contracted_path_matrix else None,
                "dynamicBatchCompile": bool(args.dynamic_batch_compile),
                "tf32MatmulEnabled": bool(
                    device.type == "cuda"
                    and torch.backends.cuda.matmul.allow_tf32
                ),
            },
        })
        reporter.status(
            "training", planId=plan["id"], epochs=epochs,
            parameters=parameter_count, examples=counts["train"],
        )
        started = time.monotonic()
        smoke_limit = None if args.smoke_batches is None else (
            int(args.smoke_batches) * batch_size
        )
        for epoch in range(start_epoch, epochs):
            if pause_file is not None and pause_file.is_file():
                reporter.status("paused", planId=plan["id"], epoch=epoch)
                raise SystemExit(PAUSE_EXIT_CODE)
            current_embedding_dropout_probability = (
                embedding_dropout_probability_at_epoch(training, epoch)
            )
            current_expected_return_mse_loss_weight = (
                expected_return_mse_loss_weight_at_epoch(training, epoch)
            )
            current_expected_return_correlation_loss_weight = (
                expected_return_correlation_loss_weight_at_epoch(training, epoch)
            )
            if cyclic_dense_compressed_path_matrix:
                model.embedding_dropout_probability = (
                    current_embedding_dropout_probability
                )
            adversarial_view = None
            adversarial_metrics = None
            if base_adversary is not None:
                for optimizer in optimizers:
                    optimizer.zero_grad(set_to_none=True)
                model.eval()

                def attack_objective(
                    attack_features: torch.Tensor,
                    attack_targets: torch.Tensor,
                    attack_weights: torch.Tensor,
                ) -> torch.Tensor:
                    attack_output = (
                        training_model(attack_features, attack_targets)
                        if joint_prefix_contracted_path_matrix
                        else training_model(attack_features)
                    )
                    attack_loss, *_components = weighted_path_training_objective(
                        attack_output,
                        attack_targets,
                        attack_weights,
                        model,
                        crps_training=crps_training,
                        target_std=float(stats["targetStd"]),
                        correlation_weight=(
                            current_expected_return_correlation_loss_weight
                        ),
                        mse_weight=current_expected_return_mse_loss_weight,
                    )
                    return attack_loss

                adversarial_view, adversarial_metrics = base_adversary.generate(
                    training_model,
                    dataset.splits["train"].physical_rows,
                    dataset.splits["train"].targets,
                    epoch=epoch,
                    objective=attack_objective,
                )
                reporter.emit({
                    "event": "base-history-adversarial-input-generated",
                    "planId": plan["id"],
                    "epoch": epoch,
                    **adversarial_metrics,
                })
            model.train()
            objective_sum = torch.zeros((), dtype=torch.float64, device=device)
            nll_sum = torch.zeros((), dtype=torch.float64, device=device)
            first_step_nll_sum = torch.zeros(
                (), dtype=torch.float64, device=device
            )
            adversarial_nll_sum = torch.zeros(
                (), dtype=torch.float64, device=device
            )
            output_adversarial_nll_sum = torch.zeros(
                (), dtype=torch.float64, device=device
            )
            input_output_adversarial_nll_sum = torch.zeros(
                (), dtype=torch.float64, device=device
            )
            crps_sum = torch.zeros((), dtype=torch.float64, device=device)
            adversarial_crps_sum = torch.zeros(
                (), dtype=torch.float64, device=device
            )
            output_adversarial_crps_sum = torch.zeros(
                (), dtype=torch.float64, device=device
            )
            input_output_adversarial_crps_sum = torch.zeros(
                (), dtype=torch.float64, device=device
            )
            correlation_sum = torch.zeros((), dtype=torch.float64, device=device)
            mse_sum = torch.zeros((), dtype=torch.float64, device=device)
            example_count = 0
            batch_counter = 0
            output_adversarial_metrics = None
            online_expectation = MetricAccumulator(float(stats["targetStd"]), device)
            online_first_step_expectation = MetricAccumulator(
                float(stats["targetStd"]), device
            )
            batch_iterator = (
                dataset.iter_adversarial_batches(
                    adversarial_view,
                    batch_size,
                    seed=seed + epoch,
                    limit=smoke_limit,
                )
                if adversarial_view is not None
                else dataset.iter_batches(
                    "train", batch_size, shuffle=True, seed=seed + epoch,
                    limit=smoke_limit,
                )
            )
            for batch in batch_iterator:
                if adversarial_view is None:
                    features, targets, weights = batch
                    adversarial_features = None
                else:
                    features, adversarial_features, targets, weights = batch
                valid_count = int(weights.sum().item())
                if adversarial_features is not None:
                    adversarial_features, _, _ = pad_weighted_batch(
                        adversarial_features, targets, weights, batch_size
                    )
                features, targets, weights = pad_weighted_batch(
                    features, targets, weights, batch_size
                )
                features = features.to(device, non_blocking=True)
                if adversarial_features is not None:
                    adversarial_features = adversarial_features.to(
                        device, non_blocking=True
                    )
                targets = targets.to(device, non_blocking=True)
                weights = weights.to(device, non_blocking=True)
                for optimizer in optimizers:
                    optimizer.zero_grad(set_to_none=True)
                contracted_terms = None
                first_step_expectation = None
                if contracted_only_joint_training:
                    if not isinstance(
                        training_model,
                        JointPrefixContractedCyclicPathMatrixDensity,
                    ):
                        raise TypeError("contracted-only training model changed")
                    contracted_terms, first_step_expectation = (
                        training_model.contracted_joint_training_terms(
                            features,
                            targets,
                        )
                    )
                    nll_loss = -(
                        contracted_terms * weights[:, None]
                    ).sum() / (weights.sum() * model.return_count)
                    clean_loss = nll_loss
                    crps_loss = None
                    with torch.no_grad():
                        expected_return_correlation = (
                            weighted_expected_return_correlation(
                                first_step_expectation[:, None],
                                targets[:, :1],
                                weights,
                                target_std=float(stats["targetStd"]),
                            )
                        )
                        expected_return_mse = (
                            weighted_expected_return_normalized_mse(
                                first_step_expectation[:, None],
                                targets[:, :1],
                                weights,
                                target_std=float(stats["targetStd"]),
                            )
                        )
                    output = None
                else:
                    output = (
                        training_model(features, targets)
                        if joint_prefix_contracted_path_matrix
                        else training_model(features)
                    )
                    (
                        clean_loss,
                        _distribution_loss,
                        nll_loss,
                        crps_loss,
                        expected_return_correlation,
                        expected_return_mse,
                    ) = weighted_path_training_objective(
                        output, targets, weights, model,
                        crps_training=crps_training,
                        target_std=float(stats["targetStd"]),
                        correlation_weight=(
                            current_expected_return_correlation_loss_weight
                        ),
                        mse_weight=current_expected_return_mse_loss_weight,
                    )

                output_adversarial_nll_loss = None
                output_adversarial_crps_loss = None
                if adversarial_output is None:
                    clean_branch_loss = clean_loss
                else:
                    assert output is not None
                    def clean_output_attack_objective(candidate):
                        return weighted_path_training_objective(
                            candidate, targets, weights, model,
                            crps_training=crps_training,
                            target_std=float(stats["targetStd"]),
                            correlation_weight=(
                                current_expected_return_correlation_loss_weight
                            ),
                            mse_weight=current_expected_return_mse_loss_weight,
                        )[0]

                    output_adversarial_variant, collected_output_metrics = (
                        generate_output_adversarial_variant(
                            output,
                            model,
                            clean_output_attack_objective,
                            epsilon_rms=output_adversarial_epsilon_rms,
                            collect_metrics=output_adversarial_metrics is None,
                        )
                    )
                    if collected_output_metrics is not None:
                        output_adversarial_metrics = collected_output_metrics
                    (
                        output_adversarial_loss,
                        _output_adversarial_distribution_loss,
                        output_adversarial_nll_loss,
                        output_adversarial_crps_loss,
                        _output_adversarial_correlation,
                        _output_adversarial_mse,
                    ) = weighted_path_training_objective(
                        output_adversarial_variant,
                        targets,
                        weights,
                        model,
                        crps_training=crps_training,
                        target_std=float(stats["targetStd"]),
                        correlation_weight=(
                            current_expected_return_correlation_loss_weight
                        ),
                        mse_weight=current_expected_return_mse_loss_weight,
                    )
                    clean_branch_loss = (
                        (1.0 - output_adversarial_weight) * clean_loss
                        + output_adversarial_weight * output_adversarial_loss
                    )

                adversarial_nll_loss = None
                adversarial_crps_loss = None
                input_output_adversarial_nll_loss = None
                input_output_adversarial_crps_loss = None
                if adversarial_features is None:
                    loss = clean_branch_loss
                else:
                    input_adversarial_output = (
                        training_model(adversarial_features, targets)
                        if joint_prefix_contracted_path_matrix
                        else training_model(adversarial_features)
                    )
                    (
                        input_adversarial_loss,
                        _input_adversarial_distribution_loss,
                        adversarial_nll_loss,
                        adversarial_crps_loss,
                        _input_adversarial_correlation,
                        _input_adversarial_mse,
                    ) = weighted_path_training_objective(
                        input_adversarial_output,
                        targets,
                        weights,
                        model,
                        crps_training=crps_training,
                        target_std=float(stats["targetStd"]),
                        correlation_weight=(
                            current_expected_return_correlation_loss_weight
                        ),
                        mse_weight=current_expected_return_mse_loss_weight,
                    )
                    if adversarial_output is None:
                        input_adversarial_branch_loss = input_adversarial_loss
                    else:
                        def joint_output_attack_objective(candidate):
                            return weighted_path_training_objective(
                                candidate, targets, weights, model,
                                crps_training=crps_training,
                                target_std=float(stats["targetStd"]),
                                correlation_weight=(
                                    current_expected_return_correlation_loss_weight
                                ),
                                mse_weight=(
                                    current_expected_return_mse_loss_weight
                                ),
                            )[0]

                        input_output_adversarial_variant, _joint_metrics = (
                            generate_output_adversarial_variant(
                                input_adversarial_output,
                                model,
                                joint_output_attack_objective,
                                epsilon_rms=output_adversarial_epsilon_rms,
                            )
                        )
                        (
                            input_output_adversarial_loss,
                            _joint_distribution_loss,
                            input_output_adversarial_nll_loss,
                            input_output_adversarial_crps_loss,
                            _joint_correlation,
                            _joint_mse,
                        ) = weighted_path_training_objective(
                            input_output_adversarial_variant,
                            targets,
                            weights,
                            model,
                            crps_training=crps_training,
                            target_std=float(stats["targetStd"]),
                            correlation_weight=(
                                current_expected_return_correlation_loss_weight
                            ),
                            mse_weight=current_expected_return_mse_loss_weight,
                        )
                        input_adversarial_branch_loss = (
                            (1.0 - output_adversarial_weight)
                            * input_adversarial_loss
                            + output_adversarial_weight
                            * input_output_adversarial_loss
                        )
                    loss = (
                        (1.0 - adversarial_weight) * clean_branch_loss
                        + adversarial_weight * input_adversarial_branch_loss
                    )
                loss.backward()
                if sam_rho > 0:
                    assert output is not None
                    perturbations = sam_perturb_parameters(model, sam_rho)
                    try:
                        for optimizer in optimizers:
                            optimizer.zero_grad(set_to_none=True)
                        perturbed = (
                            training_model(features, targets)
                            if joint_prefix_contracted_path_matrix
                            else training_model(features)
                        )
                        perturbed_nll_loss = -(
                            path_log_density_terms(perturbed, targets, model)
                            * weights[:, None]
                        ).sum() / (weights.sum() * model.return_count)
                        perturbed_correlation = weighted_expected_return_correlation(
                            perturbed.expectations,
                            targets,
                            weights,
                            target_std=float(stats["targetStd"]),
                        )
                        perturbed_mse = weighted_expected_return_normalized_mse(
                            perturbed.expectations,
                            targets,
                            weights,
                            target_std=float(stats["targetStd"]),
                        )
                        perturbed_loss = perturbed_nll_loss + (
                            current_expected_return_correlation_loss_weight
                            * (1.0 - perturbed_correlation)
                        ) + (
                            current_expected_return_mse_loss_weight * perturbed_mse
                        )
                        perturbed_loss.backward()
                    finally:
                        sam_restore_parameters(perturbations)
                clip_grad_norm_(model.parameters(), float(training["gradientClip"]),
                                foreach=device.type == "cuda")
                for optimizer in optimizers:
                    optimizer.step()
                update_weight_ema(ema, model, ema_decay)
                objective_sum += loss.detach().double() * valid_count
                if nll_loss is not None:
                    nll_sum += nll_loss.detach().double() * valid_count
                    density_terms = (
                        contracted_terms
                        if contracted_terms is not None
                        else path_log_density_terms(output, targets, model)
                    )
                    first_step_nll_sum += (
                        -density_terms[:, 0].detach().double()
                        * weights.double()
                    ).sum()
                if adversarial_nll_loss is not None:
                    adversarial_nll_sum += (
                        adversarial_nll_loss.detach().double() * valid_count
                    )
                if output_adversarial_nll_loss is not None:
                    output_adversarial_nll_sum += (
                        output_adversarial_nll_loss.detach().double() * valid_count
                    )
                if input_output_adversarial_nll_loss is not None:
                    input_output_adversarial_nll_sum += (
                        input_output_adversarial_nll_loss.detach().double()
                        * valid_count
                    )
                if crps_loss is not None:
                    crps_sum += crps_loss.detach().double() * valid_count
                if adversarial_crps_loss is not None:
                    adversarial_crps_sum += (
                        adversarial_crps_loss.detach().double() * valid_count
                    )
                if output_adversarial_crps_loss is not None:
                    output_adversarial_crps_sum += (
                        output_adversarial_crps_loss.detach().double() * valid_count
                    )
                if input_output_adversarial_crps_loss is not None:
                    input_output_adversarial_crps_sum += (
                        input_output_adversarial_crps_loss.detach().double()
                        * valid_count
                    )
                correlation_sum += (
                    expected_return_correlation.detach().double() * valid_count
                )
                mse_sum += expected_return_mse.detach().double() * valid_count
                example_count += valid_count
                displayed_expectation = (
                    first_step_expectation
                    if first_step_expectation is not None
                    else output.expectations[:, 0]
                )
                if output is None:
                    online_expectation.add(
                        displayed_expectation.detach(),
                        targets[:, 0].detach(),
                        weights,
                    )
                else:
                    online_expectation.add(
                        output.expectations.detach().flatten(),
                        targets.detach().flatten(),
                        weights[:, None].expand_as(targets).flatten(),
                    )
                online_first_step_expectation.add(
                    displayed_expectation.detach(),
                    targets[:, 0].detach(),
                    weights,
                )
                global_step += 1
                batch_counter += 1
                if batch_counter % 25 == 0:
                    reporter.status("training", planId=plan["id"], latest={
                        "epoch": epoch, "epochs": epochs,
                        "batch": batch_counter, "batches": steps_per_epoch,
                        "globalStep": global_step,
                        "onlineNegativeLogLikelihood": (
                            float(nll_sum) / example_count
                            if not crps_training else None
                        ),
                        "onlineFirstStepNegativeLogLikelihood": (
                            float(first_step_nll_sum) / example_count
                            if not crps_training else None
                        ),
                        "onlineAdversarialNegativeLogLikelihood": (
                            float(adversarial_nll_sum) / example_count
                            if adversarial_view is not None and not crps_training
                            else None
                        ),
                        "onlineOutputAdversarialNegativeLogLikelihood": (
                            float(output_adversarial_nll_sum) / example_count
                            if adversarial_output is not None and not crps_training
                            else None
                        ),
                        "onlineInputOutputAdversarialNegativeLogLikelihood": (
                            float(input_output_adversarial_nll_sum) / example_count
                            if adversarial_view is not None
                            and adversarial_output is not None
                            and not crps_training else None
                        ),
                        "onlineNormalizedCrps": (
                            float(crps_sum) / example_count
                            if crps_training else None
                        ),
                        "onlineAdversarialNormalizedCrps": (
                            float(adversarial_crps_sum) / example_count
                            if adversarial_view is not None and crps_training
                            else None
                        ),
                        "onlineOutputAdversarialNormalizedCrps": (
                            float(output_adversarial_crps_sum) / example_count
                            if adversarial_output is not None and crps_training
                            else None
                        ),
                        "onlineInputOutputAdversarialNormalizedCrps": (
                            float(input_output_adversarial_crps_sum) / example_count
                            if adversarial_view is not None
                            and adversarial_output is not None
                            and crps_training else None
                        ),
                        "onlineExpectedReturnCorrelation": (
                            float(correlation_sum) / example_count
                        ),
                        "onlineExpectedReturnNormalizedMse": (
                            float(mse_sum) / example_count
                        ),
                        "expectedReturnMseLossWeight": (
                            current_expected_return_mse_loss_weight
                        ),
                        "onlineObjective": float(objective_sum) / example_count,
                        "onlineTrain": online_expectation.result(),
                        "onlineFirstStepTrain": (
                            online_first_step_expectation.result()
                        ),
                        "seconds": time.monotonic() - started,
                        "adversarialInput": adversarial_metrics,
                        "adversarialOutput": output_adversarial_metrics,
                    })
            if adversarial_view is not None:
                del adversarial_view
            if device.type == "cuda":
                # Training compilation is intentionally isolated from eager
                # evaluation. Release unused training workspaces before the
                # evaluation allocations are created so WDDM does not spill
                # their combined allocator reserve into shared GPU memory.
                torch.cuda.empty_cache()
            should_full_evaluate = (
                args.smoke_batches is not None
                or epoch % evaluation_interval == 0
                or epoch == epochs - 1
            )
            should_first_step_evaluate = bool(
                joint_prefix_contracted_path_matrix
                and contracted_only_joint_training
            )
            if not should_full_evaluate and not should_first_step_evaluate:
                save_torch_checkpoint(checkpoint_payload(
                    model, ema, optimizers, epoch=epoch,
                    global_step=global_step, plan_hash=plan_hash,
                    runner_contract=runner_contract, best=best,
                ), last_file)
                optimization_event = {
                    "event": "minute-return-optimization-epoch",
                    "epoch": epoch,
                    "epochs": epochs,
                    "globalStep": global_step,
                    "embeddingDropoutProbability": (
                        current_embedding_dropout_probability
                    ),
                    "seconds": time.monotonic() - started,
                    "onlineNegativeLogLikelihood": (
                        float(nll_sum) / example_count
                        if not crps_training else None
                    ),
                    "onlineFirstStepNegativeLogLikelihood": (
                        float(first_step_nll_sum) / example_count
                        if not crps_training else None
                    ),
                    "onlineAdversarialNegativeLogLikelihood": (
                        float(adversarial_nll_sum) / example_count
                        if adversarial_metrics is not None and not crps_training
                        else None
                    ),
                    "onlineOutputAdversarialNegativeLogLikelihood": (
                        float(output_adversarial_nll_sum) / example_count
                        if adversarial_output is not None and not crps_training
                        else None
                    ),
                    "onlineInputOutputAdversarialNegativeLogLikelihood": (
                        float(input_output_adversarial_nll_sum) / example_count
                        if adversarial_metrics is not None
                        and adversarial_output is not None
                        and not crps_training else None
                    ),
                    "onlineNormalizedCrps": (
                        float(crps_sum) / example_count
                        if crps_training else None
                    ),
                    "onlineAdversarialNormalizedCrps": (
                        float(adversarial_crps_sum) / example_count
                        if adversarial_metrics is not None and crps_training
                        else None
                    ),
                    "onlineOutputAdversarialNormalizedCrps": (
                        float(output_adversarial_crps_sum) / example_count
                        if adversarial_output is not None and crps_training
                        else None
                    ),
                    "onlineInputOutputAdversarialNormalizedCrps": (
                        float(input_output_adversarial_crps_sum) / example_count
                        if adversarial_metrics is not None
                        and adversarial_output is not None
                        and crps_training else None
                    ),
                    "onlineExpectedReturnCorrelation": (
                        float(correlation_sum) / example_count
                    ),
                    "onlineExpectedReturnNormalizedMse": (
                        float(mse_sum) / example_count
                    ),
                    "expectedReturnMseLossWeight": (
                        current_expected_return_mse_loss_weight
                    ),
                    "onlineObjective": float(objective_sum) / example_count,
                    "onlineTrain": online_expectation.result(),
                    "onlineFirstStepTrain": (
                        online_first_step_expectation.result()
                    ),
                    "evaluationDeferredToEpoch": (
                        epoch + (evaluation_interval - epoch % evaluation_interval)
                    ),
                    "parameterCount": parameter_count,
                    "adversarialInput": adversarial_metrics,
                    "adversarialOutput": output_adversarial_metrics,
                }
                reporter.emit(optimization_event)
                reporter.status(
                    "training", planId=plan["id"], latest=optimization_event,
                )
                continue
            with use_state(model, ema):
                train_evaluation_limit = (
                    smoke_limit if args.smoke_batches is not None
                    else int(training["epochTrainEvaluationExamples"])
                )
                if should_full_evaluate:
                    train_metrics = evaluate(
                        model, dataset, "train",
                        batch_size=evaluation_batch_size,
                        target_std=float(stats["targetStd"]),
                        cumulative_std=float(stats["cumulativeStd"]),
                        device=device,
                        limit=train_evaluation_limit,
                    )
                    validation_metrics = evaluate(
                        model, dataset, "validation",
                        batch_size=evaluation_batch_size,
                        target_std=float(stats["targetStd"]),
                        cumulative_std=float(stats["cumulativeStd"]),
                        device=device,
                        collect_arrays=True,
                        limit=smoke_limit,
                    )
                    calibration_prediction, calibration_target = (
                        collect_expectation_arrays(
                            model, calibration_dataset, "calibration",
                            batch_size=evaluation_batch_size, device=device,
                            limit=smoke_limit,
                        )
                    )
                else:
                    if not isinstance(
                        model,
                        JointPrefixContractedCyclicPathMatrixDensity,
                    ):
                        raise TypeError(
                            "first-step joint evaluation model changed"
                        )
                    train_metrics = evaluate_joint_first_step(
                        model, dataset, "train",
                        batch_size=evaluation_batch_size,
                        target_std=float(stats["targetStd"]),
                        device=device,
                        limit=train_evaluation_limit,
                    )
                    validation_metrics = evaluate_joint_first_step(
                        model, dataset, "validation",
                        batch_size=evaluation_batch_size,
                        target_std=float(stats["targetStd"]),
                        device=device,
                        collect_arrays=True,
                        limit=smoke_limit,
                    )
                    calibration_prediction, calibration_target = (
                        collect_joint_first_step_expectation_arrays(
                            model, calibration_dataset, "calibration",
                            batch_size=evaluation_batch_size, device=device,
                            limit=smoke_limit,
                        )
                    )
            if device.type == "cuda":
                torch.cuda.empty_cache()
            effective_calibration_window = min(
                calibration_window, calibration_prediction.shape[0]
            )
            calibration_prediction = calibration_prediction[
                -effective_calibration_window:
            ]
            calibration_target = calibration_target[-effective_calibration_window:]
            input_scales, output_scales = calibration_scales(
                calibration_prediction, calibration_target
            )
            validation_arrays = validation_metrics.pop("arrays")
            calibrated_validation_prediction = rolling_online_per_step_affine(
                calibration_prediction,
                calibration_target,
                validation_arrays["prediction"],
                validation_arrays["target"],
                window=effective_calibration_window,
                ridge=calibration_ridge,
                input_scales=input_scales,
                output_scales=output_scales,
            )
            calibrated_validation = expectation_metrics_from_arrays(
                calibrated_validation_prediction,
                validation_arrays["target"],
                target_std=float(stats["targetStd"]),
                cumulative_std=float(stats["cumulativeStd"]),
                device=device,
            )
            candidates = {
                "validation-mse": float(
                    calibrated_validation["perLeadExpectation"][0][
                        "normalizedMse"
                    ]
                ),
                "validation-correlation": float(
                    calibrated_validation["perLeadExpectation"][0][
                        "correlation"
                    ]
                ),
            }
            if should_full_evaluate:
                # Only complete-horizon evaluations may select the joint-NLL
                # policy. First-step-only epochs keep MSE/correlation current
                # without silently changing the definition of validation NLL.
                candidates["validation-nll"] = float(
                    validation_metrics["negativeLogLikelihood"]
                )
            if crps_training:
                candidates["validation-crps"] = float(
                    validation_metrics["normalizedCrps"]
                )
            for policy, score in candidates.items():
                improved = score > best[policy]["score"] \
                    if policy.endswith("correlation") \
                    else score < best[policy]["score"]
                if improved:
                    best[policy] = {"score": score, "epoch": epoch}
                    save_torch_checkpoint({
                        "model": ema, "epoch": epoch, "score": score,
                        "policy": policy, "planSha256": plan_hash,
                        "runnerContract": runner_contract,
                    }, run_root / f"checkpoints/selections/{policy}.json")
            save_torch_checkpoint(checkpoint_payload(
                model, ema, optimizers, epoch=epoch, global_step=global_step,
                plan_hash=plan_hash, runner_contract=runner_contract, best=best,
            ), last_file)
            event = {
                "event": "minute-return-epoch", "epoch": epoch,
                "epochs": epochs, "globalStep": global_step,
                "embeddingDropoutProbability": (
                    current_embedding_dropout_probability
                ),
                "seconds": time.monotonic() - started,
                # Headline point metrics deliberately report only the next 1s
                # return so H15 runs remain directly comparable with H1 runs.
                # The complete pooled/per-lead/cumulative evaluation remains
                # available in the distribution and calibrated payloads.
                "evaluationHorizon": {"steps": 1, "seconds": 1, "lead": 0},
                "evaluationScope": (
                    "full-joint-horizon"
                    if should_full_evaluate else "first-step-only"
                ),
                "fullJointEvaluation": bool(should_full_evaluate),
                "train": train_metrics["perLeadExpectation"][0],
                "validation": calibrated_validation["perLeadExpectation"][0],
                "rawValidation": validation_metrics["perLeadExpectation"][0],
                "trainDistribution": train_metrics,
                "validationDistribution": validation_metrics,
                "calibratedValidation": calibrated_validation,
                "validationCalibration": calibration_spec,
                "onlineNegativeLogLikelihood": (
                    float(nll_sum) / example_count
                    if not crps_training else None
                ),
                "onlineFirstStepNegativeLogLikelihood": (
                    float(first_step_nll_sum) / example_count
                    if not crps_training else None
                ),
                "onlineAdversarialNegativeLogLikelihood": (
                    float(adversarial_nll_sum) / example_count
                    if adversarial_metrics is not None and not crps_training
                    else None
                ),
                "onlineOutputAdversarialNegativeLogLikelihood": (
                    float(output_adversarial_nll_sum) / example_count
                    if adversarial_output is not None and not crps_training
                    else None
                ),
                "onlineInputOutputAdversarialNegativeLogLikelihood": (
                    float(input_output_adversarial_nll_sum) / example_count
                    if adversarial_metrics is not None
                    and adversarial_output is not None
                    and not crps_training else None
                ),
                "onlineNormalizedCrps": (
                    float(crps_sum) / example_count if crps_training else None
                ),
                "onlineAdversarialNormalizedCrps": (
                    float(adversarial_crps_sum) / example_count
                    if adversarial_metrics is not None and crps_training
                    else None
                ),
                "onlineOutputAdversarialNormalizedCrps": (
                    float(output_adversarial_crps_sum) / example_count
                    if adversarial_output is not None and crps_training
                    else None
                ),
                "onlineInputOutputAdversarialNormalizedCrps": (
                    float(input_output_adversarial_crps_sum) / example_count
                    if adversarial_metrics is not None
                    and adversarial_output is not None
                    and crps_training else None
                ),
                "onlineExpectedReturnCorrelation": (
                    float(correlation_sum) / example_count
                ),
                "onlineExpectedReturnNormalizedMse": (
                    float(mse_sum) / example_count
                ),
                "expectedReturnMseLossWeight": (
                    current_expected_return_mse_loss_weight
                ),
                "onlineObjective": float(objective_sum) / example_count,
                "onlineFirstStepTrain": online_first_step_expectation.result(),
                "objective": objective_name,
                "expectedReturnCorrelationLossWeight": (
                    current_expected_return_correlation_loss_weight
                ),
                "expectedReturnCorrelationLossWeightSchedule": (
                    expected_return_correlation_loss_weight_schedule
                ),
                "expectedReturnMseLossWeightSchedule": (
                    expected_return_mse_loss_weight_schedule
                ),
                "bestValidationMse": best["validation-mse"]["score"],
                "bestValidationNll": best["validation-nll"]["score"],
                "bestValidationCrps": (
                    best["validation-crps"]["score"]
                    if crps_training else None
                ),
                "bestEpoch": best[
                    "validation-crps" if crps_training else "validation-nll"
                ]["epoch"],
                "parameterCount": parameter_count,
                "adversarialInput": adversarial_metrics,
                "adversarialOutput": output_adversarial_metrics,
            }
            reporter.emit(event)
            reporter.status("training", planId=plan["id"], latest=event)
            if args.smoke_batches is not None:
                reporter.status("smoke-complete", planId=plan["id"], latest=event)
                return

        policies: dict[str, dict] = {}
        for policy in best:
            file = run_root / f"checkpoints/selections/{policy}.json"
            saved = load_torch_checkpoint(file, map_location=device, weights_only=False)
            model.load_state_dict(saved["model"])
            values = {
                "train": evaluate(
                    model, dataset, "train", batch_size=evaluation_batch_size,
                    target_std=float(stats["targetStd"]),
                    cumulative_std=float(stats["cumulativeStd"]), device=device,
                ),
                "validation": evaluate(
                    model, dataset, "validation", batch_size=evaluation_batch_size,
                    target_std=float(stats["targetStd"]),
                    cumulative_std=float(stats["cumulativeStd"]), device=device,
                    collect_arrays=True,
                ),
                "test": evaluate(
                    model, dataset, "test", batch_size=evaluation_batch_size,
                    target_std=float(stats["targetStd"]),
                    cumulative_std=float(stats["cumulativeStd"]), device=device,
                    collect_arrays=True,
                ),
            }
            calibration_prediction, calibration_target = collect_expectation_arrays(
                model, calibration_dataset, "calibration",
                batch_size=evaluation_batch_size, device=device,
            )
            calibration_prediction = calibration_prediction[-calibration_window:]
            calibration_target = calibration_target[-calibration_window:]
            input_scales, output_scales = calibration_scales(
                calibration_prediction, calibration_target
            )
            validation_arrays = values["validation"].pop("arrays")
            test_arrays = values["test"].pop("arrays")
            calibrated_validation_prediction = rolling_online_per_step_affine(
                calibration_prediction, calibration_target,
                validation_arrays["prediction"], validation_arrays["target"],
                window=calibration_window, ridge=calibration_ridge,
                input_scales=input_scales, output_scales=output_scales,
            )
            calibrated_test_prediction = rolling_online_per_step_affine(
                validation_arrays["prediction"], validation_arrays["target"],
                test_arrays["prediction"], test_arrays["target"],
                window=calibration_window, ridge=calibration_ridge,
                input_scales=input_scales, output_scales=output_scales,
            )
            calibrated_validation = expectation_metrics_from_arrays(
                calibrated_validation_prediction, validation_arrays["target"],
                target_std=float(stats["targetStd"]),
                cumulative_std=float(stats["cumulativeStd"]), device=device,
            )
            calibrated_test = expectation_metrics_from_arrays(
                calibrated_test_prediction, test_arrays["target"],
                target_std=float(stats["targetStd"]),
                cumulative_std=float(stats["cumulativeStd"]), device=device,
            )
            policies[policy] = {
                "epoch": int(saved["epoch"]),
                "selectionScore": float(saved["score"]),
                "evaluationHorizon": {"steps": 1, "seconds": 1, "lead": 0},
                "train": values["train"]["perLeadExpectation"][0],
                "validation": calibrated_validation["perLeadExpectation"][0],
                "test": calibrated_test["perLeadExpectation"][0],
                "rawValidation": values["validation"]["perLeadExpectation"][0],
                "rawTest": values["test"]["perLeadExpectation"][0],
                "calibratedValidation": calibrated_validation,
                "calibratedTest": calibrated_test,
                "distribution": values,
                "validationCalibration": calibration_spec,
                "checkpoint": str(file.relative_to(repo)),
            }
        atomic_json({
            "contract": "compressed-path-checkpoint-selection-comparison-v1",
            "policies": policies,
        }, run_root / "state/checkpoint-selection-comparison.json")
        selected_policy = (
            "validation-crps" if crps_training else "validation-nll"
        )
        selected = policies[selected_policy]
        result = {
            "version": 1, "planId": plan["id"], "planSha256": plan_hash,
            "runnerContract": runner_contract, "examples": counts["train"],
            "featureCount": dataset.feature_count,
            "parameterCount": parameter_count,
            "bestEpoch": selected["epoch"],
            "selectionPolicy": selected_policy,
            "bestValidationScore": selected["selectionScore"],
            "train": selected["train"], "validation": selected["validation"],
            "test": selected["test"],
            "distribution": selected["distribution"],
            "checkpoint": selected["checkpoint"],
            "robustTraining": {
                "samRho": sam_rho,
                "inputDropoutProbability": input_dropout_probability,
                "embeddingDropoutProbability": embedding_dropout_probability,
                "embeddingDropoutSchedule": embedding_dropout_schedule,
                "expectedReturnCorrelationLossWeight": (
                    expected_return_correlation_loss_weight
                ),
                "expectedReturnCorrelationLossWeightSchedule": (
                    expected_return_correlation_loss_weight_schedule
                ),
                "expectedReturnMseLossWeight": expected_return_mse_loss_weight,
                "expectedReturnMseLossWeightSchedule": (
                    expected_return_mse_loss_weight_schedule
                ),
                "weightEmaHalfLifeEpochs": ema_half_life,
                "weightEmaDecayPerOptimizerStep": ema_decay,
                "adversarialInput": adversarial_input,
                "adversarialOutput": adversarial_output,
                "distributionLoss": training.get("distributionLoss", {
                    "type": NLL_DISTRIBUTION_LOSS,
                }),
            },
            "validationCalibration": calibration_spec,
        }
        atomic_json(result, run_root / "state/result.json")
        reporter.emit({"event": "minute-return-complete", **result})
        reporter.status("complete", planId=plan["id"], latest=result)
    except SystemExit:
        raise
    except KeyboardInterrupt:
        reporter.status("paused", planId=plan.get("id", "unknown"))
        raise
    except BaseException as error:
        reporter.status("failed", planId=plan.get("id", "unknown"),
                        error=f"{type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    main()
