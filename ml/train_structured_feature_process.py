from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Iterator, Mapping

import numpy as np
import torch
from torch import Tensor
from torch.nn.utils import clip_grad_norm_
from hindsight_noise import (
    TARGET_FREE_SCHEDULE_TYPE,
    advance_hindsight_curriculum,
    corrupt_standardized_hindsight,
    evaluation_noise_rng,
    expand_hindsight_samples,
    gaussian_hindsight,
    hindsight_variance_for_epoch,
    initial_hindsight_curriculum,
    validate_hindsight_plan,
)
from teacher_embedding_hindsight import TeacherEmbeddingDataset

from normalized_glu_next_return import optimizer_parameter_groups
from return_knot_density import (
    KnotDensityContract,
    component_log_masses,
    component_return_means,
    return_negative_log_likelihood,
    triangular_basis_areas,
)
from structured_feature_process import (
    ARCHITECTURE_CONTRACT,
    StructuredSharedIoFeatureProcess,
)
from trading_storage import (
    checkpoint_exists,
    load_torch_checkpoint,
    require_under,
    save_torch_checkpoint,
    training_storage_layout,
)
from train_autoregressive_minute_return import build_optimizers
from train_next_return_memorization import mean_teacher_ema_decay
from train_normalized_glu_next_return import Reporter, atomic_json
from structured_union530_base import (
    BASE_COORDINATE_COUNT,
    DATASET_CONTRACT as UNION530_BASE_DATASET_CONTRACT,
    OBJECTIVE_CONTRACT as UNION530_BASE_OBJECTIVE_CONTRACT,
    PRODUCTION59_DATASET_CONTRACT,
    PRODUCTION59_BALANCED_OBJECTIVE_CONTRACT,
    PRODUCTION59_OBJECTIVE_CONTRACT,
    PRODUCTION_BOUNDARY_BASE_COORDINATE_PERIODS,
    PRODUCTION_BASE_COORDINATE_IDS,
    PRODUCTION_BASE_COORDINATE_COUNT,
    PRODUCTION_FAST_BASE_COORDINATE_COUNT,
    StructuredProduction59BaseDataset,
    StructuredUnion530Batch,
    StructuredUnion530BaseDataset,
)


RUNNER_CONTRACT = "structured-shared-io-feature-process-training-v1"
OBJECTIVE_CONTRACT = "mean-standardized-next-feature-mse-v1"
FEATURE_EMBEDDING_DENSITY_OBJECTIVE_CONTRACT = (
    "conditional-feature-embedding-gaussian-mixture-nll-v1"
)
BASE_SUPPORT_EMBEDDING_DENSITY_OBJECTIVE_CONTRACT = (
    "conditional-feature-embedding-base-support-gaussian-mixture-nll-v2"
)
STRUCTURED_FEATURE_RETURN_DENSITY_OBJECTIVE_CONTRACT = (
    "equal-structured-feature-mse-return-density-nll-v1"
)
STRUCTURED_FEATURE_JOINT_RETURN_DENSITY_OBJECTIVE_CONTRACT = (
    "equal-structured-feature-mse-joint-path-nll-v2"
)
COMPARABLE_RETURN_CHANNEL = 0
COMPARABLE_EVALUATION_SCOPE = "next-completed-1s-signed-log-return"
FEATURE_STATE_EVALUATION_SCOPE = "complete-next-feature-state"


def objective_contract(plan: dict) -> str:
    return str(plan["training"]["loss"]["type"])


def uses_embedding_density_objective(plan: dict) -> bool:
    return objective_contract(plan) in {
        FEATURE_EMBEDDING_DENSITY_OBJECTIVE_CONTRACT,
        BASE_SUPPORT_EMBEDDING_DENSITY_OBJECTIVE_CONTRACT,
    }


def uses_base_support_embedding_density(plan: dict) -> bool:
    return objective_contract(plan) \
        == BASE_SUPPORT_EMBEDDING_DENSITY_OBJECTIVE_CONTRACT


def uses_structured_return_density_objective(plan: dict) -> bool:
    return objective_contract(plan) in {
        STRUCTURED_FEATURE_RETURN_DENSITY_OBJECTIVE_CONTRACT,
        STRUCTURED_FEATURE_JOINT_RETURN_DENSITY_OBJECTIVE_CONTRACT,
    }


def uses_joint_structured_return_density_objective(plan: dict) -> bool:
    return objective_contract(plan) \
        == STRUCTURED_FEATURE_JOINT_RETURN_DENSITY_OBJECTIVE_CONTRACT


def is_union530_base_plan(plan: dict) -> bool:
    return plan.get("datasetContract") == UNION530_BASE_DATASET_CONTRACT


def is_production59_base_plan(plan: dict) -> bool:
    return plan.get("datasetContract") == PRODUCTION59_DATASET_CONTRACT


def is_derived_base_plan(plan: dict) -> bool:
    return is_union530_base_plan(plan) or is_production59_base_plan(plan)


def build_dataset(plan: dict, repo: Path):
    architecture = plan["architecture"]
    if is_union530_base_plan(plan):
        if int(architecture["inputSteps"]) != 2 \
                or int(architecture["outputSteps"]) != 2:
            raise ValueError("the exact union530 base rollout currently requires K1=K2=2")
        subset = plan["subset"]
        return StructuredUnion530BaseDataset(
            (repo / plan["datasetDir"]).resolve(),
            train_examples=int(subset["examples"]),
            validation_examples=int(subset.get("validationExamples", 65_536)),
            test_examples=int(subset.get("testExamples", 65_536)),
        )
    if is_production59_base_plan(plan):
        return StructuredProduction59BaseDataset(
            (repo / plan["datasetDir"]).resolve(),
            (repo / plan["baseHistoryDir"]).resolve(),
            train_examples=int(plan["subset"]["examples"]),
            input_steps=int(architecture["inputSteps"]),
            output_steps=int(architecture["outputSteps"]),
        )
    source = StructuredFeatureSequenceDataset(
        (repo / plan["datasetDir"]).resolve(),
        input_steps=int(architecture["inputSteps"]),
        output_steps=int(architecture["outputSteps"]),
        train_examples=int(plan["subset"]["examples"]),
    )
    return TeacherEmbeddingDataset(source, plan["hindsightTeacher"], repo) \
        if plan.get("hindsightTeacher") is not None else source


def canonical_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("utf-8")).hexdigest()


def run_epoch_limit(run_root: Path, plan: dict) -> int:
    """Extend a run's total epoch budget without changing checkpoint identity."""
    planned = plan["training"]["epochs"]
    if type(planned) is not int or planned < 1:
        raise ValueError("training epochs must be a positive integer")
    extension_file = run_root / "state/epoch-limit.json"
    if not extension_file.is_file():
        return planned
    extension = json.loads(extension_file.read_text(encoding="utf-8"))
    if extension.get("planSha256") != canonical_hash(plan):
        raise ValueError("epoch limit belongs to a different training plan")
    epochs = extension.get("epochs")
    if type(epochs) is not int or epochs < planned:
        raise ValueError("epoch limit must be an integer at least the planned total")
    return epochs


def target_free_continuation(run_root: Path, plan: dict, epochs: int) -> dict | None:
    """Validate a pinned, explicitly requested teacher-only continuation phase."""
    phase_file = run_root / "state/target-free-phase.json"
    if not phase_file.is_file():
        return None
    phase = json.loads(phase_file.read_text(encoding="utf-8"))
    if phase.get("type") != "fixed-target-free-continuation-v1" \
            or phase.get("planSha256") != canonical_hash(plan) \
            or plan.get("hindsightTeacher") is None \
            or phase.get("replacementFraction") != 1.0:
        raise ValueError("invalid target-free continuation configuration")
    start, count = phase.get("sourceCompletedEpochs"), phase.get("additionalEpochs")
    if type(start) is not int or start < 1 or type(count) is not int or count < 1 \
            or phase.get("totalEpochs") != start + count or epochs != start + count:
        raise ValueError("target-free continuation epoch budget does not match")
    pointer = require_under(run_root / phase["sourceCheckpoint"],
                            run_root / "checkpoints/milestones", "pinned source checkpoint")
    if not checkpoint_exists(pointer):
        raise ValueError("target-free continuation source checkpoint is missing")
    metadata = json.loads(pointer.read_text(encoding="utf-8"))
    if metadata["object"]["contentHash"] != phase.get("sourceObjectSha256"):
        raise ValueError("target-free continuation source checkpoint changed")
    return phase


def validate_target_free_resume(phase: dict, saved: dict, checkpoint_hash: str) -> None:
    start = int(saved["epoch"]) + 1
    if not phase["sourceCompletedEpochs"] <= start <= phase["totalEpochs"]:
        raise ValueError("checkpoint lies outside the target-free continuation")
    if start == phase["sourceCompletedEpochs"]:
        if checkpoint_hash != phase["sourceObjectSha256"]:
            raise ValueError("target-free continuation must begin at the pinned checkpoint")
    elif saved.get("targetFreePhase") != phase:
        raise ValueError("checkpoint belongs to a different target-free phase")


class StructuredFeatureSequenceDataset:
    """Chronological K1 input and K2 target states over the 59-channel timeline."""

    def __init__(
        self,
        root: Path,
        *,
        input_steps: int,
        output_steps: int,
        train_examples: int,
    ) -> None:
        self.root = root.resolve()
        self.manifest = json.loads(
            (self.root / "manifest.json").read_text(encoding="utf-8")
        )
        if self.manifest.get("storageLayout") != "temporal-channel-timeline-v1":
            raise ValueError("structured feature training requires a timeline dataset")
        self.feature_count = int(self.manifest["temporalChannelCount"])
        self.input_steps = int(input_steps)
        self.output_steps = int(output_steps)
        if min(self.input_steps, self.output_steps, int(train_examples)) <= 0:
            raise ValueError("sequence lengths and training count must be positive")

        self.timeline: dict[str, np.memmap] = {}
        self.origins: dict[str, np.ndarray] = {}
        self.counts: dict[str, int] = {}
        files = self.manifest["files"]
        examples = self.manifest["examplesBySplit"]
        rows = self.manifest["timelineRowsBySplit"]
        for split in ("train", "validation", "test"):
            physical_count = int(examples[split])
            timeline_rows = int(rows[split])
            timeline = np.memmap(
                self.root / files[split]["timelineFeatures"],
                dtype="<f4",
                mode="r",
                shape=(timeline_rows, self.feature_count),
            )
            origins = np.memmap(
                self.root / files[split]["origins"],
                dtype="<i4",
                mode="r",
                shape=(physical_count,),
            )
            all_origins = np.asarray(origins, dtype=np.int64)
            constructible = all_origins[
                (all_origins >= self.input_steps - 1)
                & (all_origins + self.output_steps < timeline_rows)
            ]
            logical_count = min(constructible.size, int(train_examples)) \
                if split == "train" else int(constructible.size)
            selected = constructible[:logical_count]
            if selected.size != logical_count or (
                split == "train" and logical_count != int(train_examples)
            ):
                raise ValueError(f"{split} cannot supply requested feature sequences")
            if not np.all(np.diff(selected) > 0):
                raise ValueError(f"{split} origins are not chronological and unique")
            self.timeline[split] = timeline
            self.origins[split] = selected
            self.counts[split] = logical_count

    def close(self) -> None:
        """Release Windows memory-map handles deterministically."""
        for values in self.timeline.values():
            mapping = getattr(values, "_mmap", None)
            if mapping is not None:
                mapping.close()
        self.timeline.clear()

    def _examples(
        self, split: str, logical_indices: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        origins = self.origins[split][logical_indices]
        input_offsets = np.arange(
            -self.input_steps + 1, 1, dtype=np.int64
        )
        output_offsets = np.arange(1, self.output_steps + 1, dtype=np.int64)
        inputs = np.asarray(
            self.timeline[split][origins[:, None] + input_offsets[None, :]],
            dtype=np.float32,
        )
        targets = np.asarray(
            self.timeline[split][origins[:, None] + output_offsets[None, :]],
            dtype=np.float32,
        )
        if not np.isfinite(inputs).all() or not np.isfinite(targets).all():
            raise ValueError(f"{split} structured feature sequence is non-finite")
        return inputs, targets

    def iter_batches(
        self,
        split: str,
        batch_size: int,
        *,
        shuffle: bool,
        seed: int,
        limit: int | None = None,
        pad: bool = True,
    ) -> Iterator[tuple[Tensor, Tensor, Tensor]]:
        count = self.counts[split]
        if limit is not None:
            count = min(count, int(limit))
        indices = np.arange(count, dtype=np.int64)
        if shuffle:
            np.random.default_rng(seed).shuffle(indices)
        for start in range(0, count, int(batch_size)):
            selected = indices[start:start + int(batch_size)]
            inputs, targets = self._examples(split, selected)
            valid = selected.size
            weights = np.ones(valid, dtype=np.float32)
            if pad and valid < batch_size:
                missing = int(batch_size) - valid
                inputs = np.concatenate((inputs, np.zeros(
                    (missing, self.input_steps, self.feature_count),
                    dtype=np.float32,
                )))
                targets = np.concatenate((targets, np.zeros(
                    (missing, self.output_steps, self.feature_count),
                    dtype=np.float32,
                )))
                weights = np.concatenate((weights, np.zeros(missing, dtype=np.float32)))
            yield (
                torch.from_numpy(inputs),
                torch.from_numpy(targets),
                torch.from_numpy(weights),
            )

    def statistics(self, batch_size: int) -> dict[str, np.ndarray]:
        input_sum = np.zeros(self.feature_count, dtype=np.float64)
        input_square = np.zeros(self.feature_count, dtype=np.float64)
        output_sum = np.zeros(self.feature_count, dtype=np.float64)
        output_square = np.zeros(self.feature_count, dtype=np.float64)
        input_count = 0
        output_count = 0
        for inputs, targets, weights in self.iter_batches(
            "train", batch_size, shuffle=False, seed=0, pad=False,
        ):
            del weights
            x = inputs.numpy().astype(np.float64, copy=False)
            y = targets.numpy().astype(np.float64, copy=False)
            input_sum += x.sum(axis=(0, 1))
            input_square += np.square(x).sum(axis=(0, 1))
            output_sum += y.sum(axis=(0, 1))
            output_square += np.square(y).sum(axis=(0, 1))
            input_count += x.shape[0] * x.shape[1]
            output_count += y.shape[0] * y.shape[1]

        def finish(total: np.ndarray, square: np.ndarray, count: int):
            mean = total / count
            variance = np.maximum(square / count - np.square(mean), 0.0)
            std = np.sqrt(variance)
            # Truly constant channels cannot carry predictive loss. A unit
            # scale leaves them numerically valid without amplifying noise.
            std = np.where(std > 1e-8, std, 1.0)
            return mean.astype(np.float32), std.astype(np.float32)

        input_mean, input_std = finish(input_sum, input_square, input_count)
        output_mean, output_std = finish(output_sum, output_square, output_count)
        return {
            "inputMean": input_mean,
            "inputStd": input_std,
            "outputMean": output_mean,
            "outputStd": output_std,
        }


class FeatureMetricAccumulator:
    def __init__(self, output_mean: Tensor, output_std: Tensor) -> None:
        width = int(output_mean.numel())
        self.output_mean = output_mean.detach().cpu().double()
        self.output_std = output_std.detach().cpu().double()
        self.count = torch.zeros(width, dtype=torch.float64)
        self.sum_prediction = torch.zeros(width, dtype=torch.float64)
        self.sum_target = torch.zeros(width, dtype=torch.float64)
        self.sum_prediction_square = torch.zeros(width, dtype=torch.float64)
        self.sum_target_square = torch.zeros(width, dtype=torch.float64)
        self.sum_product = torch.zeros(width, dtype=torch.float64)
        self.sum_square_error = torch.zeros(width, dtype=torch.float64)
        self.sum_absolute_error = torch.zeros(width, dtype=torch.float64)
        self.direction_correct = torch.zeros(width, dtype=torch.float64)
        self.return_predictions: list[Tensor] = []
        self.return_targets: list[Tensor] = []

    def add(self, prediction: Tensor, target: Tensor, weights: Tensor) -> None:
        active = weights.detach().cpu().bool()
        if not bool(active.any()):
            return
        prediction = prediction.detach().cpu().double()[active].reshape(
            -1, prediction.shape[-1]
        )
        target = target.detach().cpu().double()[active].reshape(-1, target.shape[-1])
        error = prediction - target
        rows = prediction.shape[0]
        self.count += rows
        self.sum_prediction += prediction.sum(0)
        self.sum_target += target.sum(0)
        self.sum_prediction_square += prediction.square().sum(0)
        self.sum_target_square += target.square().sum(0)
        self.sum_product += (prediction * target).sum(0)
        self.sum_square_error += error.square().sum(0)
        self.sum_absolute_error += error.abs().sum(0)
        self.direction_correct += (
            (prediction >= 0) == (target >= 0)
        ).double().sum(0)
        self.return_predictions.append(
            prediction[:, COMPARABLE_RETURN_CHANNEL].clone()
        )
        self.return_targets.append(target[:, COMPARABLE_RETURN_CHANNEL].clone())

    def _absolute_return_magnitude_deciles(self) -> list[dict]:
        if not self.return_predictions:
            return []
        prediction = torch.cat(self.return_predictions)
        target = torch.cat(self.return_targets)
        absolute_target = target.abs()
        order = torch.argsort(absolute_target, stable=True)
        count = int(order.numel())
        bucket_count = min(10, count)
        deciles: list[dict] = []
        for index in range(bucket_count):
            start = count * index // bucket_count
            stop = count * (index + 1) // bucket_count
            selected = order[start:stop]
            bucket_prediction = prediction[selected]
            bucket_target = target[selected]
            bucket_absolute_target = absolute_target[selected]
            correct = (
                (bucket_prediction >= 0) == (bucket_target >= 0)
            ).double()
            absolute_weight = float(bucket_absolute_target.sum())
            signed_capture = float(
                (torch.sign(bucket_prediction) * bucket_target).sum()
            )
            error = bucket_prediction - bucket_target
            zero_mse = float(bucket_target.square().mean())
            mse = float(error.square().mean())
            deciles.append({
                "index": index,
                "quantileLow": index / bucket_count,
                "quantileHigh": (index + 1) / bucket_count,
                "examples": int(selected.numel()),
                "minimumAbsoluteTarget": float(bucket_absolute_target.min()),
                "maximumAbsoluteTarget": float(bucket_absolute_target.max()),
                "meanAbsoluteTarget": float(bucket_absolute_target.mean()),
                "directionAccuracy": float(correct.mean()),
                "magnitudeWeightedDirectionAccuracy": float(
                    (correct * bucket_absolute_target).sum() / absolute_weight
                ) if absolute_weight > 0 else 0.0,
                "signedReturnCapture": signed_capture / absolute_weight
                if absolute_weight > 0 else 0.0,
                "mse": mse,
                "zeroBaselineMse": zero_mse,
                "mseSkillVsZero": 1.0 - mse / zero_mse
                if zero_mse > 0 else 0.0,
            })
        return deciles

    def result(self) -> dict:
        count = self.count.clamp_min(1.0)
        mse = self.sum_square_error / count
        mae = self.sum_absolute_error / count
        normalized_mse = mse / self.output_std.square()
        centered_product = self.sum_product - (
            self.sum_prediction * self.sum_target / count
        )
        prediction_ss = self.sum_prediction_square - self.sum_prediction.square() / count
        target_ss = self.sum_target_square - self.sum_target.square() / count
        correlation = centered_product / torch.sqrt(
            prediction_ss.clamp_min(0) * target_ss.clamp_min(0)
        ).clamp_min(1e-30)
        valid_correlation = (prediction_ss > 0) & (target_ss > 0)
        correlation = torch.where(valid_correlation, correlation, torch.zeros_like(correlation))
        mean_baseline_mse = (
            self.sum_target_square
            - 2 * self.output_mean * self.sum_target
            + count * self.output_mean.square()
        ) / count
        pooled_mse = float(self.sum_square_error.sum() / self.count.sum())
        pooled_mean_baseline_mse = float(mean_baseline_mse.mean())
        feature_state = {
            "normalizedMse": float(normalized_mse.mean()),
            "mse": pooled_mse,
            "rmse": math.sqrt(max(0.0, pooled_mse)),
            "mae": float(self.sum_absolute_error.sum() / self.count.sum()),
            "correlation": float(correlation[valid_correlation].mean())
            if bool(valid_correlation.any()) else 0.0,
            "trainingMeanBaselineMse": pooled_mean_baseline_mse,
            "examples": int(self.count[0].item() / max(1, 1)),
            "featureCount": int(self.count.numel()),
            "perFeatureNormalizedMse": normalized_mse.tolist(),
            "perFeatureCorrelation": correlation.tolist(),
        }
        channel = COMPARABLE_RETURN_CHANNEL
        channel_count = float(count[channel])
        prediction_mean = float(self.sum_prediction[channel] / channel_count)
        target_mean = float(self.sum_target[channel] / channel_count)
        prediction_variance = max(
            0.0,
            float(self.sum_prediction_square[channel] / channel_count)
            - prediction_mean**2,
        )
        target_variance = max(
            0.0,
            float(self.sum_target_square[channel] / channel_count) - target_mean**2,
        )
        channel_mse = float(mse[channel])
        zero_mse = float(self.sum_target_square[channel] / channel_count)
        next_return = {
            "examples": int(round(channel_count)),
            "normalizedMse": float(normalized_mse[channel]),
            "mse": channel_mse,
            "rmse": math.sqrt(max(0.0, channel_mse)),
            "mae": float(mae[channel]),
            "directionAccuracy": float(
                self.direction_correct[channel] / channel_count
            ),
            "correlation": float(correlation[channel])
            if bool(valid_correlation[channel]) else None,
            "predictionMean": prediction_mean,
            "predictionStd": math.sqrt(prediction_variance),
            "targetMean": target_mean,
            "targetStd": math.sqrt(target_variance),
            "zeroBaselineMse": zero_mse,
            "mseSkillVsZero": 1.0 - channel_mse / zero_mse
            if zero_mse > 0 else 0.0,
            "channelIndex": channel,
            "channelId": "return-lag-0s",
            "evaluationScope": COMPARABLE_EVALUATION_SCOPE,
            "directionByAbsoluteTargetDecile": (
                self._absolute_return_magnitude_deciles()
            ),
        }
        if not all(math.isfinite(value) for value in (
            feature_state["normalizedMse"], feature_state["mse"],
            feature_state["rmse"], feature_state["mae"],
            feature_state["correlation"], next_return["normalizedMse"],
            next_return["mse"], next_return["rmse"], next_return["mae"],
            next_return["directionAccuracy"], next_return["mseSkillVsZero"],
        )):
            raise FloatingPointError("non-finite structured feature metrics")
        if next_return["correlation"] is not None and not math.isfinite(
            next_return["correlation"]
        ):
            raise FloatingPointError("non-finite next-return correlation")
        return {
            "nextReturn": next_return,
            "featureState": feature_state,
        }


class FeatureSequenceMetricAccumulator:
    def __init__(
        self, output_mean: Tensor, output_std: Tensor, *, output_steps: int
    ) -> None:
        if output_steps <= 0:
            raise ValueError("output steps must be positive")
        self.output_steps = int(output_steps)
        self.pooled = FeatureMetricAccumulator(output_mean, output_std)
        self.per_step = [
            FeatureMetricAccumulator(output_mean, output_std)
            for _ in range(self.output_steps)
        ]

    def add(self, prediction: Tensor, target: Tensor, weights: Tensor) -> None:
        if prediction.ndim != 3 or target.shape != prediction.shape:
            raise ValueError("sequence metrics require matching [batch, step, feature]")
        if prediction.shape[1] != self.output_steps:
            raise ValueError("sequence metric output-step count changed")
        self.pooled.add(prediction, target, weights)
        for step, metrics in enumerate(self.per_step):
            metrics.add(
                prediction[:, step:step + 1],
                target[:, step:step + 1],
                weights,
            )

    def result(self) -> dict:
        pooled = self.pooled.result()
        per_step = [metrics.result() for metrics in self.per_step]
        return {
            # The dashboard headline is always lead 1 so it remains directly
            # comparable with historical next-1s runs.
            "nextReturn": per_step[0]["nextReturn"],
            "returnPath": pooled["nextReturn"],
            "featureState": pooled["featureState"],
            "perStepNextReturn": [value["nextReturn"] for value in per_step],
            "perStepFeatureState": [value["featureState"] for value in per_step],
        }


def weighted_standardized_mse(
    prediction: Tensor, target: Tensor, weights: Tensor
) -> Tensor:
    per_example = (prediction - target).square().flatten(1).mean(dim=1)
    return (per_example * weights).sum() / weights.sum().clamp_min(1)


def structured_feature_return_density_objective(
    feature_mse: Tensor,
    return_density_nll: Tensor,
    config: Mapping[str, object],
) -> dict[str, Tensor]:
    """Combine causal 59-feature reconstruction and scalar return NLL 50/50."""
    expected_weights = {
        "derivedFeatureMse": 0.5,
        "returnDensityNll": 0.5,
    }
    if config.get("weights") != expected_weights \
            or config.get("featureObjective") \
            != PRODUCTION59_OBJECTIVE_CONTRACT:
        raise ValueError("structured return-density objective changed")
    objective = (
        expected_weights["derivedFeatureMse"] * feature_mse
        + expected_weights["returnDensityNll"] * return_density_nll
    )
    return {
        "objective": objective,
        "derivedFeatureMse": feature_mse,
        "returnDensityNegativeLogLikelihood": return_density_nll,
    }


def _weighted_masked_mean(
    values: Tensor, mask: Tensor, weights: Tensor
) -> Tensor:
    if values.shape != mask.shape or values.shape[0] != weights.shape[0]:
        raise ValueError("masked structured loss shapes changed")
    weighted_mask = mask.to(values.dtype) * weights[:, None, None]
    return (values * weighted_mask).sum() / weighted_mask.sum().clamp_min(1)


def production59_balanced_objective(
    standardized_derived_prediction: Tensor,
    standardized_derived_target: Tensor,
    standardized_base_prediction: Tensor,
    standardized_base_target: Tensor,
    weights: Tensor,
    context: Mapping[str, Tensor],
    config: Mapping[str, object],
) -> dict[str, Tensor]:
    """Balance return, remaining state, and causally valid primitive losses."""
    if standardized_derived_prediction.shape \
            != standardized_derived_target.shape \
            or standardized_derived_prediction.ndim != 3 \
            or standardized_derived_prediction.shape[-1] != 59:
        raise ValueError("balanced production59 derived shapes changed")
    if standardized_base_prediction.shape != standardized_base_target.shape \
            or standardized_base_prediction.ndim != 3 \
            or standardized_base_prediction.shape[-1] \
            != PRODUCTION_BASE_COORDINATE_COUNT:
        raise ValueError("balanced production59 primitive shapes changed")
    if standardized_base_prediction.shape[:2] \
            != standardized_derived_prediction.shape[:2]:
        raise ValueError("balanced production59 step shapes changed")

    configured_weights = config.get("weights")
    expected_weights = {
        "return": 0.5,
        "otherDerivedFeatures": 0.25,
        "primitiveCoordinates": 0.25,
    }
    if configured_weights != expected_weights:
        raise ValueError("balanced production59 objective weights changed")

    derived_error = (
        standardized_derived_prediction - standardized_derived_target
    ).square()
    return_loss = _weighted_masked_mean(
        derived_error[:, :, :1],
        torch.ones_like(derived_error[:, :, :1], dtype=torch.bool),
        weights,
    )
    other_derived_loss = _weighted_masked_mean(
        derived_error[:, :, 1:],
        torch.ones_like(derived_error[:, :, 1:], dtype=torch.bool),
        weights,
    )

    primitive_error = (
        standardized_base_prediction - standardized_base_target
    ).square()
    primitive_mask = torch.ones_like(primitive_error, dtype=torch.bool)
    second_index = context.get("secondIndex")
    if second_index is None or second_index.ndim != 1 \
            or second_index.shape[0] != primitive_error.shape[0]:
        raise ValueError("balanced production59 second indices changed")
    steps = torch.arange(
        primitive_error.shape[1], device=primitive_error.device,
        dtype=second_index.dtype,
    )
    # This is the same completion index used by Production59BaseRollout:
    # future second = current + step + 1, completed boundary = future + 1.
    completion_index = second_index[:, None] + steps[None, :] + 2
    coordinate_index = {
        coordinate_id: index
        for index, coordinate_id in enumerate(PRODUCTION_BASE_COORDINATE_IDS)
    }
    for coordinate_id, period in \
            PRODUCTION_BOUNDARY_BASE_COORDINATE_PERIODS.items():
        primitive_mask[:, :, coordinate_index[coordinate_id]] = (
            torch.remainder(completion_index, int(period)) == 0
        )
    if not bool(primitive_mask[:, :, :PRODUCTION_FAST_BASE_COORDINATE_COUNT].all()):
        raise AssertionError("fast primitive supervision must remain dense")
    primitive_loss = _weighted_masked_mean(
        primitive_error, primitive_mask, weights
    )
    objective = (
        expected_weights["return"] * return_loss
        + expected_weights["otherDerivedFeatures"] * other_derived_loss
        + expected_weights["primitiveCoordinates"] * primitive_loss
    )
    return {
        "objective": objective,
        "return": return_loss,
        "otherDerivedFeatures": other_derived_loss,
        "primitiveCoordinates": primitive_loss,
        "activePrimitiveFraction": primitive_mask.to(
            primitive_error.dtype
        ).mean(),
    }


@torch.no_grad()
def evaluate(
    model: StructuredSharedIoFeatureProcess,
    dataset: StructuredFeatureSequenceDataset | StructuredUnion530BaseDataset,
    split: str,
    *,
    batch_size: int,
    device: torch.device,
    limit: int | None = None,
) -> dict:
    model.eval()
    hindsight_rng = (
        evaluation_noise_rng(model.hindsight_conditioning, split)
        if model.hindsight_conditioning is not None else None
    )
    derived_base = isinstance(dataset, StructuredUnion530BaseDataset)
    output_mean = (
        dataset.derived_output_mean if derived_base else model.output_mean
    )
    output_std = (
        dataset.derived_output_std if derived_base else model.output_std
    )
    metrics = FeatureSequenceMetricAccumulator(
        output_mean,
        output_std,
        output_steps=dataset.output_steps,
    )
    for batch in dataset.iter_batches(
        split, batch_size, shuffle=False, seed=0, limit=limit,
        # Synthetic zero rows can enter undefined derived-feature geometry;
        # evaluate the final short batch directly instead of relying on a
        # zero weight to mask NaNs after the fact.
        pad=False,
    ):
        if derived_base:
            assert isinstance(batch, StructuredUnion530Batch)
            inputs, targets, weights = batch.inputs, batch.targets, batch.weights
            context = {
                name: value.to(device, non_blocking=True)
                for name, value in batch.context.items()
            }
        else:
            inputs, targets, weights = batch
            context = None
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        weights = weights.to(device, non_blocking=True)
        if hindsight_rng is None:
            prediction_standardized = model(inputs)
        elif isinstance(dataset, TeacherEmbeddingDataset):
            noise = dataset.hindsight(inputs, batch.teacher_context, targets=None,
                fraction=1.0, samples=model.hindsight_sample_count, rng=hindsight_rng)
            variance = torch.ones((inputs.shape[0], 1), device=device)
            prediction_standardized = model(inputs, noise, variance)
        else:
            # Held-out targets are used only below, to measure the prediction.
            noise, variance = gaussian_hindsight(
                hindsight_rng, inputs.shape[0], model.output_steps,
                model.output_width, device,
                samples=model.hindsight_sample_count,
            )
            prediction_standardized = model(inputs, noise, variance)
        direct = model.raw_outputs(prediction_standardized)
        prediction = (
            dataset.derive(direct, inputs, context)
            if derived_base else direct
        )
        metrics.add(prediction, targets, weights)
    result = metrics.result()
    if hindsight_rng is not None:
        result["hindsightEvaluation"] = {
            "noiseVariance": 1.0,
            "targetContribution": 0.0,
            "noiseSeedBase": int(model.hindsight_conditioning["evaluationSeed"]),
            "noiseStream": split,
            "mode": "one-pass-frozen-teacher-component-embeddings"
            if isinstance(dataset, TeacherEmbeddingDataset) else "one-pass-pure-gaussian-noise",
            "parameterMeaning": "teacher-replacement-fraction"
            if isinstance(dataset, TeacherEmbeddingDataset) else "gaussian-noise-variance",
        }
    return result


@torch.no_grad()
def evaluate_hindsight_reconstruction(
    model: StructuredSharedIoFeatureProcess,
    dataset: StructuredFeatureSequenceDataset,
    schedule: dict,
    *,
    split: str,
    variance: float,
    batch_size: int,
    device: torch.device,
    limit: int | None = None,
) -> dict:
    """Target-assisted diagnostic, separate from pure-noise forecast metrics."""
    if split not in ("train", "validation"):
        raise ValueError("hindsight reconstruction split must be train or validation")
    model.eval()
    noise_seed = int(schedule["probeSeed"]) + int(split == "validation")
    rng = np.random.default_rng(noise_seed)
    metrics = FeatureSequenceMetricAccumulator(
        model.output_mean, model.output_std, output_steps=model.output_steps,
    )
    probe_limit = int(schedule["probeExamples"])
    if limit is not None:
        probe_limit = min(probe_limit, limit)
    for batch in dataset.iter_batches(
        split, batch_size, shuffle=False, seed=0, limit=probe_limit, pad=False,
    ):
        inputs, targets, weights = batch
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        weights = weights.to(device, non_blocking=True)
        if isinstance(dataset, TeacherEmbeddingDataset):
            noisy_hindsight = dataset.hindsight(inputs, batch.teacher_context,
                targets=targets if variance < 1 else None, fraction=variance,
                samples=model.hindsight_sample_count, rng=rng)
        else:
            noise, _ = gaussian_hindsight(
                rng, inputs.shape[0], model.output_steps, model.output_width, device,
                samples=model.hindsight_sample_count,
            )
            noisy_hindsight = corrupt_standardized_hindsight(
                expand_hindsight_samples(model.standardized_targets(targets), model.hindsight_sample_count),
                noise, variance,
            )
        variance_input = torch.full(
            (inputs.shape[0], 1), variance, dtype=inputs.dtype, device=device,
        )
        prediction = model.raw_outputs(model(inputs, noisy_hindsight, variance_input))
        metrics.add(prediction, targets, weights)
    result = metrics.result()
    return {
        "evaluationScope": "target-assisted-hindsight-reconstruction",
        "split": split,
        "weightSource": "raw-training-weights",
        "noiseVariance": variance,
        "noiseSeed": noise_seed,
        "targetContribution": 1.0 - variance if isinstance(dataset, TeacherEmbeddingDataset)
        else math.sqrt(1.0 - variance),
        "parameterMeaning": "teacher-replacement-fraction"
        if isinstance(dataset, TeacherEmbeddingDataset) else "gaussian-noise-variance",
        "usedForCurriculum": False,
        "usedForCheckpointSelection": False,
        "metric": "nextReturn.correlation",
        "examples": result["nextReturn"]["examples"],
        "correlation": result["nextReturn"]["correlation"],
        "normalizedMse": result["nextReturn"]["normalizedMse"],
        "mseSkillVsZero": result["nextReturn"]["mseSkillVsZero"],
        "nextReturn": {
            key: value for key, value in result["nextReturn"].items()
            if key != "directionByAbsoluteTargetDecile"
        },
        "perStepNextReturn": [
            {key: value for key, value in step.items()
             if key != "directionByAbsoluteTargetDecile"}
            for step in result["perStepNextReturn"]
        ],
        "perStepReturnCorrelation": [
            step["correlation"] for step in result["perStepNextReturn"]
        ],
    }


def evaluate_hindsight_curriculum_probe(
    model: StructuredSharedIoFeatureProcess,
    dataset: StructuredFeatureSequenceDataset,
    schedule: dict,
    *,
    variance: float,
    batch_size: int,
    device: torch.device,
    limit: int | None = None,
) -> dict:
    """Only this training-split probe may control the noise curriculum."""
    result = evaluate_hindsight_reconstruction(
        model, dataset, schedule, split="train", variance=variance,
        batch_size=batch_size, device=device, limit=limit,
    )
    result.update({
        "evaluationScope": "training-hindsight-reconstruction-probe",
        "usedForCurriculum": True,
    })
    return result


@torch.no_grad()
def evaluate_structured_return_density(
    model: StructuredSharedIoFeatureProcess,
    dataset: StructuredProduction59BaseDataset,
    split: str,
    *,
    batch_size: int,
    device: torch.device,
    density: KnotDensityContract,
    limit: int | None = None,
) -> dict:
    """Evaluate return-density expectations and causal feature reconstruction."""
    model.eval()
    if dataset.derived_output_mean is None \
            or dataset.derived_output_std is None:
        raise ValueError("structured density evaluation requires statistics")
    feature_metrics = FeatureSequenceMetricAccumulator(
        dataset.derived_output_mean,
        dataset.derived_output_std,
        output_steps=dataset.output_steps,
    )
    return_metrics = FeatureSequenceMetricAccumulator(
        dataset.derived_output_mean[:1],
        dataset.derived_output_std[:1],
        output_steps=dataset.output_steps,
    )
    knots = torch.from_numpy(density.knots_unit).to(device=device).float()
    areas = triangular_basis_areas(knots)
    means = torch.from_numpy(component_return_means(
        density.knots_unit, density.transform,
    )).to(device=device).float()
    total_nll = 0.0
    total_examples = 0
    per_step_nll = torch.zeros(dataset.output_steps, dtype=torch.float64)
    per_step_examples = torch.zeros(dataset.output_steps, dtype=torch.float64)
    for batch in dataset.iter_batches(
        split, batch_size, shuffle=False, seed=0, limit=limit, pad=False,
    ):
        inputs, targets, weights = (
            batch.inputs.to(device, non_blocking=True),
            batch.targets.to(device, non_blocking=True),
            batch.weights.to(device, non_blocking=True),
        )
        context = {
            name: value.to(device, non_blocking=True)
            for name, value in batch.context.items()
        }
        joint_path_density = (
            isinstance(model.return_density, dict)
            and model.return_density.get("type")
            == "joint-prefix-contracted-path-matrix-return-density-v1"
        )
        if joint_path_density:
            standardized_base, density_output = \
                model.forward_with_joint_return_density(
                    inputs,
                    targets[:, :, COMPARABLE_RETURN_CHANNEL],
                )
            if density_output.joint_log_density_terms is None:
                raise RuntimeError(
                    "joint path density did not emit contracted terms"
                )
            nll = -density_output.joint_log_density_terms
            expected_return = density_output.expectations
        else:
            standardized_base, density_logits = \
                model.forward_with_return_density(inputs)
            nll, log_masses, _ = return_negative_log_likelihood(
                density_logits,
                targets[:, :, COMPARABLE_RETURN_CHANNEL],
                knots,
                areas,
                density.transform,
            )
            expected_return = log_masses.exp() @ means
        raw_base = model.raw_outputs(standardized_base)
        feature_prediction = dataset.derive(raw_base, inputs, context)
        feature_metrics.add(feature_prediction, targets, weights)
        return_metrics.add(
            expected_return.unsqueeze(-1),
            targets[:, :, COMPARABLE_RETURN_CHANNEL:COMPARABLE_RETURN_CHANNEL + 1],
            weights,
        )
        active = weights[:, None]
        total_nll += float((nll * active).sum().detach())
        valid = int(weights.sum().item())
        total_examples += valid * dataset.output_steps
        per_step_nll += (nll * active).sum(dim=0).detach().cpu().double()
        per_step_examples += active.sum(dim=0).detach().cpu().double()
    mean_nll = total_nll / max(1, total_examples)
    density_expectation = return_metrics.result()
    feature_prediction = feature_metrics.result()
    return {
        "negativeLogLikelihood": mean_nll,
        "perLeadNegativeLogLikelihood": (
            per_step_nll / per_step_examples.clamp_min(1.0)
        ).tolist(),
        "bitsPerExample": mean_nll / math.log(2.0),
        "examples": int(total_examples),
        "steps": int(dataset.output_steps),
        "components": int(knots.numel()),
        "densitySpace": "continuous-next-1s-signed-log-return",
        "expectationType": (
            "joint-prefix-path-matrix-marginal-mean"
            if isinstance(model.return_density, dict)
            and model.return_density.get("type")
            == "joint-prefix-contracted-path-matrix-return-density-v1"
            else "fixed-knot-component-return-mean"
        ),
        "nextReturn": density_expectation["nextReturn"],
        "returnPath": density_expectation["returnPath"],
        "perStepNextReturn": density_expectation["perStepNextReturn"],
        "featureState": feature_prediction["featureState"],
        "perStepFeatureState": feature_prediction["perStepFeatureState"],
        "featurePredictionNextReturn": feature_prediction["nextReturn"],
        "featurePredictionReturnPath": feature_prediction["returnPath"],
        "perStepFeaturePredictionNextReturn": feature_prediction[
            "perStepNextReturn"
        ],
    }


def derive_base_support_feature_states(
    dataset: StructuredUnion530BaseDataset,
    raw_base_components: Tensor,
    inputs: Tensor,
    context: Mapping[str, Tensor],
) -> Tensor:
    """Causally rederive each base-support path into complete feature states."""
    if raw_base_components.ndim != 4:
        raise ValueError("base support must have [batch,step,component,feature]")
    batch, steps, components, base_width = raw_base_components.shape
    if steps != dataset.output_steps \
            or base_width != dataset.base_coordinate_count \
            or inputs.shape[0] != batch:
        raise ValueError("base-support rollout dimensions changed")
    expanded_base = raw_base_components.permute(0, 2, 1, 3).reshape(
        batch * components, steps, base_width,
    )
    expanded_inputs = inputs[:, None].expand(
        batch, components, *inputs.shape[1:],
    ).reshape(batch * components, *inputs.shape[1:])
    expanded_context: dict[str, Tensor] = {}
    for name, value in context.items():
        if value.ndim < 1 or value.shape[0] != batch:
            raise ValueError(f"base-support context {name} has the wrong batch")
        expanded_context[name] = value[:, None].expand(
            batch, components, *value.shape[1:],
        ).reshape(batch * components, *value.shape[1:])
    derived = dataset.derive(
        expanded_base, expanded_inputs, expanded_context,
    )
    return derived.reshape(
        batch, components, steps, dataset.feature_count,
    ).permute(0, 2, 1, 3)


def base_support_density_outputs(
    model: StructuredSharedIoFeatureProcess,
    dataset: StructuredUnion530BaseDataset,
    inputs: Tensor,
    targets: Tensor,
    context: Mapping[str, Tensor],
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return log PDF, exact expected base, expected state, and components."""
    raw_components, log_weights = model.feature_embedding_base_distribution(
        inputs
    )
    component_feature_states = derive_base_support_feature_states(
        dataset, raw_components, inputs, context,
    )
    log_density = model.feature_embedding_log_density_from_base_support(
        targets, component_feature_states, log_weights,
    )
    expected_base = model.expected_raw_base_features(
        raw_components, log_weights,
    )
    expected_features = (
        log_weights.exp().unsqueeze(-1) * component_feature_states
    ).sum(dim=-2)
    return log_density, expected_base, expected_features, component_feature_states


@torch.no_grad()
def evaluate_feature_embedding_density(
    model: StructuredSharedIoFeatureProcess,
    dataset: StructuredFeatureSequenceDataset | StructuredUnion530BaseDataset,
    split: str,
    *,
    batch_size: int,
    device: torch.device,
    limit: int | None = None,
) -> dict:
    """Evaluate embedding NLL and any exact base-support expectation."""
    model.eval()
    base_support = model.feature_embedding_density is not None \
        and model.feature_embedding_density.get("type") \
        == "causal-base-support-gaussian-mixture-density-v2"
    expectation_metrics: FeatureSequenceMetricAccumulator | None = None
    base_metrics: FeatureSequenceMetricAccumulator | None = None
    if base_support:
        if not isinstance(dataset, StructuredUnion530BaseDataset):
            raise ValueError("base-support density requires a causal base dataset")
        if dataset.derived_output_mean is None \
                or dataset.derived_output_std is None:
            raise ValueError("base-support evaluation requires dataset statistics")
        expectation_metrics = FeatureSequenceMetricAccumulator(
            dataset.derived_output_mean,
            dataset.derived_output_std,
            output_steps=dataset.output_steps,
        )
        base_metrics = FeatureSequenceMetricAccumulator(
            model.output_mean,
            model.output_std,
            output_steps=dataset.output_steps,
        )
    total_nll = 0.0
    total_examples = 0
    per_step_nll = torch.zeros(dataset.output_steps, dtype=torch.float64)
    per_step_examples = torch.zeros(dataset.output_steps, dtype=torch.float64)
    for batch in dataset.iter_batches(
        split, batch_size, shuffle=False, seed=0, limit=limit, pad=False,
    ):
        if isinstance(batch, StructuredUnion530Batch):
            inputs, targets, weights = batch.inputs, batch.targets, batch.weights
            context = {
                name: value.to(device, non_blocking=True)
                for name, value in batch.context.items()
            }
        else:
            inputs, targets, weights = batch
            context = None
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        weights = weights.to(device, non_blocking=True)
        if base_support:
            assert isinstance(dataset, StructuredUnion530BaseDataset) \
                and context is not None \
                and expectation_metrics is not None \
                and base_metrics is not None
            log_density, expected_base, expected_features, _ = \
                base_support_density_outputs(
                    model, dataset, inputs, targets, context,
                )
            expectation_metrics.add(expected_features, targets, weights)
            base_metrics.add(
                expected_base, context["baseTargets"], weights,
            )
        else:
            log_density = model.feature_embedding_log_density(inputs, targets)
        nll = -log_density
        active = weights[:, None]
        total_nll += float((nll * active).sum().detach())
        valid = int(weights.sum().item())
        total_examples += valid * dataset.output_steps
        per_step_nll += (nll * active).sum(dim=0).detach().cpu().double()
        per_step_examples += active.sum(dim=0).detach().cpu().double()
    mean_nll = total_nll / max(1, total_examples)
    per_step = (
        per_step_nll / per_step_examples.clamp_min(1.0)
    ).tolist()
    density = model.embedding_density_attention
    if density is None:
        raise AssertionError("embedding density evaluator disappeared")
    result = {
        "negativeLogLikelihood": mean_nll,
        "perLeadNegativeLogLikelihood": per_step,
        "bitsPerExample": mean_nll / math.log(2.0),
        "unitNegativeLogLikelihood": mean_nll / density.embedding_width,
        "examples": int(total_examples),
        "steps": int(dataset.output_steps),
        "embeddingWidth": int(density.embedding_width),
        "components": int(density.component_count),
        "densitySpace": "layer-1-unit-rms-feature-embedding",
    }
    if base_support:
        assert expectation_metrics is not None and base_metrics is not None
        expectation = expectation_metrics.result()
        base_expectation = base_metrics.result()
        result.update({
            "expectationType": "mixture-weighted-causal-base-support-v2",
            "nextReturn": expectation["nextReturn"],
            "returnPath": expectation["returnPath"],
            "featureState": expectation["featureState"],
            "perStepNextReturn": expectation["perStepNextReturn"],
            "perStepFeatureState": expectation["perStepFeatureState"],
            "baseFeatureState": base_expectation["featureState"],
            "perStepBaseFeatureState": base_expectation["perStepFeatureState"],
        })
    return result


def comparable_policy_metrics(evaluations: dict[str, dict]) -> dict:
    expected = {"train", "validation", "test"}
    if set(evaluations) != expected:
        raise ValueError(f"structured evaluation splits changed: {sorted(evaluations)}")
    result = {
        "evaluationScope": COMPARABLE_EVALUATION_SCOPE,
        "featureStateEvaluationScope": FEATURE_STATE_EVALUATION_SCOPE,
        "train": evaluations["train"]["nextReturn"],
        "validation": evaluations["validation"]["nextReturn"],
        "test": evaluations["test"]["nextReturn"],
        "featureState": {
            split: evaluations[split]["featureState"] for split in (
                "train", "validation", "test"
            )
        },
        "distribution": {
            split: {
                "expectation": evaluations[split]["returnPath"],
                "perLeadExpectation": evaluations[split]["perStepNextReturn"],
            }
            for split in ("train", "validation", "test")
        },
        "perStepFeatureState": {
            split: evaluations[split]["perStepFeatureState"]
            for split in ("train", "validation", "test")
        },
    }
    if "hindsightEvaluation" in evaluations["validation"]:
        result["hindsightEvaluation"] = {
            split: evaluations[split]["hindsightEvaluation"] for split in expected
        }
    return result


def compact_feature_state_metrics(metrics: dict) -> dict:
    return {
        key: value for key, value in metrics.items()
        if key not in {"perFeatureNormalizedMse", "perFeatureCorrelation"}
    }


@torch.no_grad()
def update_weight_ema(
    ema: dict[str, Tensor], model: torch.nn.Module, decay: float
) -> None:
    for name, value in model.state_dict().items():
        if value.is_floating_point():
            ema[name].lerp_(value.detach(), 1.0 - decay)
        else:
            ema[name].copy_(value)


@contextmanager
def use_state(model: torch.nn.Module, state: dict[str, Tensor]):
    original = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    model.load_state_dict(state)
    try:
        yield
    finally:
        model.load_state_dict(original)


def checkpoint_payload(
    model: torch.nn.Module,
    ema: dict[str, Tensor],
    optimizers: tuple[torch.optim.Optimizer, ...],
    *,
    epoch: int,
    global_step: int,
    plan_hash: str,
    best: dict,
    hindsight_curriculum: dict | None = None,
    target_free_phase: dict | None = None,
) -> dict:
    return {
        "model": model.state_dict(),
        "emaModel": ema,
        "optimizers": [optimizer.state_dict() for optimizer in optimizers],
        "epoch": epoch,
        "globalStep": global_step,
        "best": best,
        "planSha256": plan_hash,
        "runnerContract": RUNNER_CONTRACT,
        **({"targetFreePhase": dict(target_free_phase)} if target_free_phase is not None else {}),
        **(
            {"hindsightCurriculum": dict(hindsight_curriculum)}
            if hindsight_curriculum is not None else {}
        ),
    }


def persist_structured_feature_evaluation(
    *,
    repo: Path,
    run_root: Path,
    plan: dict,
    dataset: StructuredFeatureSequenceDataset,
    parameter_count: int,
    policies: dict[str, dict],
    generated_at: str,
) -> tuple[dict, dict]:
    """Persist both the detailed artifact and the dashboard result contract."""
    required = {"best-validation-mse", "best-validation-correlation", "last"}
    if set(policies) != required:
        raise ValueError(f"structured evaluation policies changed: {sorted(policies)}")
    selects_next_return = plan["architecture"].get("hindsightConditioning") is not None
    artifact_file = (
        repo / "data/benchmarks" / f'{plan["id"]}-completed-eval.json'
    ).resolve()
    artifact = {
        "contract": "structured-shared-io-feature-process-evaluation-v2",
        "generatedAt": generated_at,
        "planId": plan["id"],
        "selectionPolicy": "best-validation-mse",
        "selectionDoesNotUseTest": True,
        "completedAfterEpoch": int(policies["last"]["epoch"]),
        "examplesBySplit": dataset.counts,
        "featureCount": dataset.feature_count,
        "parameterCount": int(parameter_count),
        "headlineEvaluationScope": COMPARABLE_EVALUATION_SCOPE,
        "checkpointSelectionMetric": (
            "nextReturn.validation" if selects_next_return
            else "featureState.validation"
        ),
        "policies": policies,
    }
    atomic_json(artifact, artifact_file)
    atomic_json(artifact, run_root / "state/stopped-evaluation.json")
    comparison_names = {
        "best-validation-mse": "validation-mse",
        "best-validation-correlation": "validation-correlation",
        "last": "last",
    }
    atomic_json({
        "contract": "next-return-checkpoint-selection-comparison-v1",
        "policies": {
            comparison_names[name]: value for name, value in policies.items()
        },
    }, run_root / "state/checkpoint-selection-comparison.json")
    best = policies["best-validation-mse"]
    result = {
        "contract": (
            "structured-shared-io-feature-process-dashboard-result-v2"
        ),
        "planId": plan["id"],
        "selectionPolicy": "best-validation-mse",
        "selectionMetric": (
            "nextReturn.normalizedMse" if selects_next_return
            else "featureState.normalizedMse"
        ),
        "selectionDoesNotUseTest": True,
        "evaluationScope": COMPARABLE_EVALUATION_SCOPE,
        "examples": dataset.counts["train"],
        "examplesBySplit": dataset.counts,
        "featureCount": dataset.feature_count,
        "parameterCount": int(parameter_count),
        "trainableParameterCount": int(parameter_count),
        "bestEpoch": int(best["epoch"]),
        "completedAfterEpoch": int(policies["last"]["epoch"]),
        "bestValidationScore": float(best["selectionScore"]),
        "headlineValidationScore": float(best["validation"]["normalizedMse"]),
        "train": best["train"],
        "validation": best["validation"],
        "test": best["test"],
        "featureState": best["featureState"],
        "distribution": best["distribution"],
        "perStepFeatureState": best["perStepFeatureState"],
        "evaluationArtifact": str(artifact_file.relative_to(repo)).replace(
            "\\", "/"
        ),
    }
    atomic_json(result, run_root / "state/result.json")
    return artifact, result


def persist_embedding_density_evaluation(
    *,
    repo: Path,
    run_root: Path,
    plan: dict,
    dataset: StructuredFeatureSequenceDataset | StructuredUnion530BaseDataset,
    parameter_count: int,
    policies: dict[str, dict],
    generated_at: str,
) -> tuple[dict, dict]:
    base_support = uses_base_support_embedding_density(plan)
    required = (
        {
            "best-validation-nll",
            "best-validation-mse",
            "best-validation-correlation",
            "last",
        }
        if base_support else {"best-validation-nll", "last"}
    )
    if set(policies) != required:
        raise ValueError(
            f"embedding-density evaluation policies changed: {sorted(policies)}"
        )
    artifact_file = (
        repo / "data/benchmarks" / f'{plan["id"]}-completed-eval.json'
    ).resolve()
    artifact = {
        "contract": (
            "structured-feature-embedding-base-support-density-evaluation-v2"
            if base_support
            else "structured-feature-embedding-density-evaluation-v1"
        ),
        "generatedAt": generated_at,
        "planId": plan["id"],
        "selectionPolicy": "best-validation-nll",
        "selectionDoesNotUseTest": True,
        "completedAfterEpoch": int(policies["last"]["epoch"]),
        "examplesBySplit": dataset.counts,
        "featureCount": dataset.feature_count,
        "parameterCount": int(parameter_count),
        "evaluationScope": "realized-layer-1-feature-embedding-density",
        "policies": policies,
    }
    atomic_json(artifact, artifact_file)
    atomic_json(artifact, run_root / "state/stopped-evaluation.json")
    comparison_policies = {
        "validation-nll": policies["best-validation-nll"],
        "last": policies["last"],
    }
    if base_support:
        comparison_policies.update({
            "validation-mse": policies["best-validation-mse"],
            "validation-correlation": policies[
                "best-validation-correlation"
            ],
        })
    atomic_json({
        "contract": "embedding-density-checkpoint-selection-comparison-v1",
        "policies": comparison_policies,
    }, run_root / "state/checkpoint-selection-comparison.json")
    best = policies["best-validation-nll"]
    result = {
        "contract": (
            "structured-feature-embedding-base-support-density-dashboard-result-v2"
            if base_support
            else "structured-feature-embedding-density-dashboard-result-v1"
        ),
        "planId": plan["id"],
        "selectionPolicy": "best-validation-nll",
        "selectionMetric": "distribution.validation.negativeLogLikelihood",
        "selectionDoesNotUseTest": True,
        "evaluationScope": "realized-layer-1-feature-embedding-density",
        "examples": dataset.counts["train"],
        "examplesBySplit": dataset.counts,
        "featureCount": dataset.feature_count,
        "parameterCount": int(parameter_count),
        "trainableParameterCount": int(parameter_count),
        "bestEpoch": int(best["epoch"]),
        "completedAfterEpoch": int(policies["last"]["epoch"]),
        "bestValidationScore": float(best["selectionScore"]),
        **(
            {
                "density": best["density"],
                "train": best["train"],
                "validation": best["validation"],
                "test": best["test"],
                "featureState": best["featureState"],
                "distribution": best["distribution"],
                "perStepFeatureState": best["perStepFeatureState"],
            }
            if base_support else {"distribution": best["distribution"]}
        ),
        "evaluationArtifact": str(artifact_file.relative_to(repo)).replace(
            "\\", "/"
        ),
    }
    atomic_json(result, run_root / "state/result.json")
    return artifact, result


def persist_structured_return_density_evaluation(
    *,
    repo: Path,
    run_root: Path,
    plan: dict,
    dataset: StructuredProduction59BaseDataset,
    parameter_count: int,
    policies: dict[str, dict],
    generated_at: str,
) -> tuple[dict, dict]:
    """Persist all return-PDF and feature-reconstruction checkpoint policies."""
    required = {
        "best-validation-nll",
        "best-validation-mse",
        "best-validation-correlation",
        "last",
    }
    if set(policies) != required:
        raise ValueError(
            f"structured return-density policies changed: {sorted(policies)}"
        )
    artifact_file = (
        repo / "data/benchmarks" / f'{plan["id"]}-completed-eval.json'
    ).resolve()
    artifact = {
        "contract": "structured-feature-return-density-evaluation-v1",
        "generatedAt": generated_at,
        "planId": plan["id"],
        "selectionPolicy": "best-validation-mse",
        "selectionDoesNotUseTest": True,
        "completedAfterEpoch": int(policies["last"]["epoch"]),
        "examplesBySplit": dataset.counts,
        "featureCount": dataset.feature_count,
        "parameterCount": int(parameter_count),
        "headlineEvaluationScope": COMPARABLE_EVALUATION_SCOPE,
        "densityEvaluationScope": "continuous-next-1s-signed-log-return",
        "checkpointSelectionMetric": "return-density expectation validation",
        "policies": policies,
    }
    atomic_json(artifact, artifact_file)
    atomic_json(artifact, run_root / "state/stopped-evaluation.json")
    atomic_json({
        "contract": "structured-return-density-checkpoint-comparison-v1",
        "policies": {
            "validation-nll": policies["best-validation-nll"],
            "validation-mse": policies["best-validation-mse"],
            "validation-correlation": policies[
                "best-validation-correlation"
            ],
            "last": policies["last"],
        },
    }, run_root / "state/checkpoint-selection-comparison.json")
    best = policies["best-validation-mse"]
    result = {
        "contract": "structured-feature-return-density-dashboard-result-v1",
        "planId": plan["id"],
        "selectionPolicy": "best-validation-mse",
        "selectionMetric": "returnDensityExpectation.normalizedMse",
        "selectionDoesNotUseTest": True,
        "evaluationScope": COMPARABLE_EVALUATION_SCOPE,
        "examples": dataset.counts["train"],
        "examplesBySplit": dataset.counts,
        "featureCount": dataset.feature_count,
        "parameterCount": int(parameter_count),
        "trainableParameterCount": int(parameter_count),
        "bestEpoch": int(best["epoch"]),
        "completedAfterEpoch": int(policies["last"]["epoch"]),
        "bestValidationScore": float(best["selectionScore"]),
        "headlineValidationScore": float(best["validation"]["normalizedMse"]),
        "density": best["density"],
        "train": best["train"],
        "validation": best["validation"],
        "test": best["test"],
        "featureState": best["featureState"],
        "distribution": best["distribution"],
        "perStepFeatureState": best["perStepFeatureState"],
        "evaluationArtifact": str(artifact_file.relative_to(repo)).replace(
            "\\", "/"
        ),
    }
    atomic_json(result, run_root / "state/result.json")
    return artifact, result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--evaluation-batch-size", type=int)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--smoke-batches", type=int)
    parser.add_argument("--replace-smoke", action="store_true")
    parser.add_argument(
        "--compile-mode", choices=("none", "default", "reduce-overhead"),
        default="default",
    )
    parser.add_argument(
        "--matmul-precision", choices=("highest", "high", "medium"), default="high"
    )
    return parser.parse_args()


def validate_plan(plan: dict) -> None:
    for name in ("id", "datasetDir", "runDir", "subset", "architecture", "training"):
        if name not in plan:
            raise ValueError(f"structured feature plan is missing {name}")
    architecture = plan["architecture"]
    if architecture.get("contract") != ARCHITECTURE_CONTRACT:
        raise ValueError("structured feature architecture contract changed")
    validate_hindsight_plan(plan)
    expected_positive = (
        "inputSteps", "outputSteps", "inputFeatures", "outputFeatures",
        "featureWidth", "marketWidth", "prefixWidth",
        "featureDistributionWidth", "extendedPrefixWidth",
        "nextFeatureDistributionWidth",
    )
    if any(int(architecture.get(name, 0)) <= 0 for name in expected_positive):
        raise ValueError("structured feature architecture dimensions must be positive")
    factorization = architecture.get("linearFactorization")
    if factorization is not None:
        if not isinstance(factorization, dict) \
                or factorization.get("type") != "trainable-low-rank-v1" \
                or int(factorization.get("rank", 0)) <= 0:
            raise ValueError("invalid structured low-rank linear factorization")
    layer8_attention = architecture.get("layer8Attention")
    layer8_function = architecture.get("layer8FunctionApproximator")
    if layer8_attention is not None and layer8_function is not None:
        raise ValueError("layer 8 cannot configure two sequence operators")
    if layer8_attention is not None:
        if not isinstance(layer8_attention, dict) \
                or layer8_attention.get("type") \
                != "causal-self-attention-v1":
            raise ValueError("invalid layer-8 self-attention configuration")
        attention_dimensions = (
            "queryWidth", "keyWidth", "valueWidth", "outputWidth",
        )
        if any(int(layer8_attention.get(name, 0)) <= 0
               for name in attention_dimensions):
            raise ValueError("layer-8 self-attention dimensions must be positive")
        if int(layer8_attention["queryWidth"]) \
                != int(layer8_attention["keyWidth"]):
            raise ValueError("layer-8 attention query/key widths must match")
        if int(layer8_attention["outputWidth"]) \
                != int(architecture["featureWidth"]):
            raise ValueError(
                "layer-8 attention output width must match feature width"
            )
        if int(layer8_attention.get("heads", 0)) != 1:
            raise ValueError("layer-8 attention currently requires one head")
    if layer8_function is not None:
        if not isinstance(layer8_function, dict) \
                or layer8_function.get("type") \
                != "causal-dog-knot-basis-gram-loss-v1":
            raise ValueError("invalid layer-8 function approximator")
        function_dimensions = (
            "pointWidth", "knotWidth", "valueWidth", "outputWidth",
        )
        if any(int(layer8_function.get(name, 0)) <= 0
               for name in function_dimensions):
            raise ValueError("layer-8 function dimensions must be positive")
        if int(layer8_function["pointWidth"]) \
                != int(layer8_function["knotWidth"]):
            raise ValueError("layer-8 function point/knot widths must match")
        if int(layer8_function["outputWidth"]) \
                != int(architecture["featureWidth"]):
            raise ValueError("layer-8 function output must match feature width")
        kernel = layer8_function.get("kernel")
        orthogonalization_loss = layer8_function.get(
            "orthogonalizationLoss"
        )
        if not isinstance(kernel, dict) \
                or kernel.get("type") != "normalized-distance-dog-v1" \
                or float(kernel.get("bandwidth", 0)) <= 0:
            raise ValueError("invalid layer-8 function kernel")
        if not isinstance(orthogonalization_loss, dict) \
                or orthogonalization_loss.get("type") \
                != "gram-identity-mean-square-v1" \
                or orthogonalization_loss.get("gramEstimator") \
                != "current-batch-causal-prefix-v1" \
                or float(orthogonalization_loss.get("weight", 0)) <= 0:
            raise ValueError(
                "invalid layer-8 function orthogonalization loss"
            )
        if float(layer8_function.get("normalizationEpsilon", 0)) <= 0:
            raise ValueError("invalid layer-8 function normalization epsilon")
    recurrent_memory = architecture.get("recurrentMemory")
    if recurrent_memory is not None:
        if not isinstance(recurrent_memory, dict) \
                or recurrent_memory.get("type") \
                != "dual-state-gated-exchange-v1" \
                or int(recurrent_memory.get("hiddenWidth", 0)) <= 0 \
                or tuple(recurrent_memory.get("layers", ())) \
                != (1, 2, 3, 4, 5, 6, 7, 9, 10, 11) \
                or recurrent_memory.get("activation") != "sigmoid" \
                or recurrent_memory.get("mix") != "a*t+b*(t-1)":
            raise ValueError("invalid recurrent-memory configuration")
        if factorization is not None:
            raise ValueError(
                "recurrent-memory cells do not use linear factorization"
            )
    embedding_density = architecture.get("featureEmbeddingDensity")
    if embedding_density is not None:
        density_type = embedding_density.get("type") \
            if isinstance(embedding_density, dict) else None
        density_value_width = int(embedding_density.get("valueWidth", 0)) \
            if isinstance(embedding_density, dict) else 0
        density_type_valid = (
            density_type == "conditional-gaussian-mixture-attention-v1"
            and density_value_width == 1
        ) or (
            density_type == "causal-base-support-gaussian-mixture-density-v2"
            and density_value_width == int(architecture["outputFeatures"])
            and embedding_density.get("keyConstruction")
            == "causal-derived-feature-state-through-shared-layer1"
            and is_derived_base_plan(plan)
        )
        if not isinstance(embedding_density, dict) \
                or not density_type_valid \
                or int(embedding_density.get("components", 0)) <= 1 \
                or int(embedding_density.get("queryWidth", 0)) \
                != int(architecture["featureWidth"]) \
                or int(embedding_density.get("keyWidth", 0)) \
                != int(architecture["featureWidth"]) \
                or float(embedding_density.get(
                    "fixedStandardDeviation", 0,
                )) <= 0 \
                or float(embedding_density.get(
                    "normalizationEpsilon", 0,
                )) <= 0 \
                or embedding_density.get("queryNormalization") \
                != "unit-rms" \
                or embedding_density.get("targetEncoderGradient") \
                != "stop-gradient":
            raise ValueError("invalid feature-embedding density configuration")
        if recurrent_memory is not None:
            raise ValueError(
                "feature-embedding density run must not carry recurrent layer state"
            )
    return_density = architecture.get("returnDensity")
    if return_density is not None:
        density_type = return_density.get("type") \
            if isinstance(return_density, dict) else None
        common_valid = (
            isinstance(return_density, dict)
            and return_density.get("latentSource")
            == "expected-feature-embedding"
            and int(return_density.get("knotCount", 0)) >= 3
            and embedding_density is None
            and recurrent_memory is None
        )
        scalar_valid = density_type \
            == "fixed-knot-piecewise-linear-return-density-v1"
        joint_valid = (
            density_type
            == "joint-prefix-contracted-path-matrix-return-density-v1"
            and int(return_density.get("returnCount", 0))
            == int(architecture["outputSteps"])
            and int(return_density.get("stageBlockCount", 0)) == 1
            and bool(return_density.get(
                "recurrentActivationCheckpointing", False
            ))
            and all(int(return_density.get(name, 0)) > 0 for name in (
                "marketWidth", "pathEmbeddingWidth", "pathCount",
                "pathCompressionWidth", "jointCompressionWidth",
            ))
        )
        if not common_valid or not (scalar_valid or joint_valid):
            raise ValueError("invalid structured return-density head")
    training = plan["training"]
    loss_type = training.get("loss", {}).get("type")
    expected_losses = (
        {UNION530_BASE_OBJECTIVE_CONTRACT}
        if is_union530_base_plan(plan)
        else {
            PRODUCTION59_OBJECTIVE_CONTRACT,
            PRODUCTION59_BALANCED_OBJECTIVE_CONTRACT,
            FEATURE_EMBEDDING_DENSITY_OBJECTIVE_CONTRACT,
            BASE_SUPPORT_EMBEDDING_DENSITY_OBJECTIVE_CONTRACT,
            STRUCTURED_FEATURE_RETURN_DENSITY_OBJECTIVE_CONTRACT,
            STRUCTURED_FEATURE_JOINT_RETURN_DENSITY_OBJECTIVE_CONTRACT,
        }
        if is_production59_base_plan(plan)
        else {OBJECTIVE_CONTRACT}
    )
    if loss_type not in expected_losses:
        raise ValueError("structured feature objective changed")
    if (loss_type in {
        FEATURE_EMBEDDING_DENSITY_OBJECTIVE_CONTRACT,
        BASE_SUPPORT_EMBEDDING_DENSITY_OBJECTIVE_CONTRACT,
    }) \
            != (embedding_density is not None):
        raise ValueError(
            "feature-embedding density architecture/objective must be paired"
        )
    if (loss_type in {
        STRUCTURED_FEATURE_RETURN_DENSITY_OBJECTIVE_CONTRACT,
        STRUCTURED_FEATURE_JOINT_RETURN_DENSITY_OBJECTIVE_CONTRACT,
    }) \
            != (return_density is not None):
        raise ValueError(
            "scalar return-density architecture/objective must be paired"
        )
    if is_union530_base_plan(plan):
        if int(architecture["inputFeatures"]) != 530 \
                or int(architecture["outputFeatures"]) != BASE_COORDINATE_COUNT \
                or int(architecture.get("derivedOutputFeatures", 0)) != 530:
            raise ValueError("union530 base-output dimensions changed")
    if is_production59_base_plan(plan):
        if not isinstance(plan.get("baseHistoryDir"), str) \
                or int(architecture["inputFeatures"]) != 59 \
                or int(architecture["outputFeatures"]) \
                != PRODUCTION_BASE_COORDINATE_COUNT \
                or int(architecture.get("derivedOutputFeatures", 0)) != 59 \
                or tuple(architecture.get("baseOutputCoordinateIds", ())) \
                != PRODUCTION_BASE_COORDINATE_IDS:
            raise ValueError("production59 base-output dimensions changed")
        if loss_type == PRODUCTION59_BALANCED_OBJECTIVE_CONTRACT:
            loss = training["loss"]
            if loss.get("weights") != {
                "return": 0.5,
                "otherDerivedFeatures": 0.25,
                "primitiveCoordinates": 0.25,
            } or loss.get("primitiveSupervision") != {
                "fastCoordinateIds": list(
                    PRODUCTION_BASE_COORDINATE_IDS[
                        :PRODUCTION_FAST_BASE_COORDINATE_COUNT
                    ]
                ),
                "boundaryCoordinatePeriodsSeconds": dict(
                    PRODUCTION_BOUNDARY_BASE_COORDINATE_PERIODS
                ),
            }:
                raise ValueError(
                    "balanced production59 objective configuration changed"
                )
        if loss_type in {
            STRUCTURED_FEATURE_RETURN_DENSITY_OBJECTIVE_CONTRACT,
            STRUCTURED_FEATURE_JOINT_RETURN_DENSITY_OBJECTIVE_CONTRACT,
        }:
            loss = training["loss"]
            density = plan.get("density")
            scalar_density = loss_type \
                == STRUCTURED_FEATURE_RETURN_DENSITY_OBJECTIVE_CONTRACT
            expected_output_steps = 1 if scalar_density else 15
            if int(architecture["inputSteps"]) != 1 \
                    or int(architecture["outputSteps"]) \
                    != expected_output_steps \
                    or loss.get("weights") != {
                        "derivedFeatureMse": 0.5,
                        "returnDensityNll": 0.5,
                    } \
                    or loss.get("featureObjective") \
                    != PRODUCTION59_OBJECTIVE_CONTRACT \
                    or not isinstance(density, dict) \
                    or not isinstance(density.get("source"), str) \
                    or int(density.get("fit", 0)) \
                    != int(return_density["knotCount"]):
                raise ValueError(
                    "structured return-density experiment configuration changed"
                )
            if not scalar_density and (
                not bool(architecture.get(
                    "recurrentActivationCheckpointing", False
                ))
                or int(return_density.get("returnCount", 0)) != 15
            ):
                raise ValueError(
                    "structured joint return-density recurrence changed"
                )
    if int(training["batchSize"]) != int(training["evaluationBatchSize"]):
        raise ValueError("training and evaluation batch sizes must be equal")
    if any(training.get(name) not in (None, 0, 0.0) for name in (
        "sam", "inputDropoutProbability", "embeddingDropoutProbability",
        "adversarialInput", "adversarialOutput",
    )):
        raise ValueError("the initial structured feature run has no regularizers")


def replace_completed_smoke(run_root: Path) -> None:
    status_file = run_root / "state/status.json"
    status = json.loads(status_file.read_text(encoding="utf-8")) \
        if status_file.is_file() else {}
    if status.get("stage") != "smoke-complete":
        raise ValueError("--replace-smoke requires a completed smoke run")
    log_file = run_root / "logs/training.jsonl"
    if log_file.is_file():
        archive = run_root / "logs/smoke-training.jsonl"
        if archive.exists():
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            archive = run_root / f"logs/smoke-training-{stamp}.jsonl"
        log_file.replace(archive)
    for file in (
        run_root / "checkpoints/last.json",
        run_root / "checkpoints/selections/validation-mse.json",
        run_root / "checkpoints/selections/validation-correlation.json",
        run_root / "checkpoints/selections/validation-nll.json",
        run_root / "state/plan.json",
        run_root / "state/result.json",
        status_file,
    ):
        file.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    if args.batch_size is not None and args.batch_size < 1:
        raise ValueError("batch size must be positive")
    if args.evaluation_batch_size is not None and args.evaluation_batch_size < 1:
        raise ValueError("evaluation batch size must be positive")
    if args.smoke_batches is not None and args.smoke_batches < 1:
        raise ValueError("smoke batch count must be positive")
    torch.set_float32_matmul_precision(args.matmul_precision)
    repo = Path(__file__).resolve().parents[1]
    plan_file = args.plan if args.plan.is_absolute() else repo / args.plan
    plan = json.loads(plan_file.read_text(encoding="utf-8"))
    validate_plan(plan)
    plan_hash = canonical_hash(plan)
    layout = training_storage_layout(repo)
    run_root = require_under(repo / plan["runDir"], layout.runs, "run directory")
    if args.replace_smoke:
        replace_completed_smoke(run_root)
    reporter = Reporter(run_root)
    reporter.status("initializing", planId=plan["id"])
    try:
        architecture = plan["architecture"]
        training = plan["training"]
        device = torch.device(args.device or training["device"])
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        seed = int(training["seed"])
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
            torch.backends.cuda.matmul.allow_tf32 = True

        batch_size = int(args.batch_size or training["batchSize"])
        evaluation_batch_size = int(
            args.evaluation_batch_size or training["evaluationBatchSize"]
        )
        if batch_size != evaluation_batch_size:
            raise ValueError("training and evaluation batch sizes must remain equal")
        dataset = build_dataset(plan, repo)
        expected_output_width = (
            dataset.base_coordinate_count
            if isinstance(dataset, StructuredUnion530BaseDataset)
            else dataset.feature_count
        )
        if dataset.feature_count != int(architecture["inputFeatures"]) \
                or expected_output_width != int(architecture["outputFeatures"]):
            raise ValueError("dataset and structured IO feature widths differ")
        snapshot = {"planSha256": plan_hash, "plan": plan}
        snapshot_file = run_root / "state/plan.json"
        if snapshot_file.is_file() and json.loads(
            snapshot_file.read_text(encoding="utf-8")
        ) != snapshot:
            raise ValueError("run directory belongs to a different plan")
        atomic_json(snapshot, snapshot_file)
        reporter.emit({
            "event": "minute-return-dataset-selected",
            "planId": plan["id"],
            "counts": dataset.counts,
            "featureCount": dataset.feature_count,
            "inputSteps": dataset.input_steps,
            "outputSteps": dataset.output_steps,
            "inputSeconds": dataset.input_steps,
            "outputSeconds": dataset.output_steps,
            "target": (
                "primitive future coordinates rederived to complete production59 state"
                if isinstance(dataset, StructuredProduction59BaseDataset)
                else "primitive future coordinates rederived to complete union530 state"
                if isinstance(dataset, StructuredUnion530BaseDataset)
                else "complete next-second production-basis feature state"
            ),
        })
        reporter.status("computing-training-statistics", planId=plan["id"])
        stats = dataset.statistics(evaluation_batch_size)
        derived_base = isinstance(dataset, StructuredUnion530BaseDataset)
        derived_output_mean = (
            dataset.derived_output_mean.to(device)
            if derived_base else None
        )
        derived_output_std = (
            dataset.derived_output_std.to(device)
            if derived_base else None
        )
        uses_return_density = uses_structured_return_density_objective(plan)
        uses_joint_return_density = \
            uses_joint_structured_return_density_objective(plan)
        return_density = (
            KnotDensityContract.load(
                (repo / plan["density"]["source"]).resolve(),
                fit=str(plan["density"]["fit"]),
            )
            if uses_return_density else None
        )
        model = StructuredSharedIoFeatureProcess(
            torch.from_numpy(stats["inputMean"]),
            torch.from_numpy(stats["inputStd"]),
            torch.from_numpy(stats["outputMean"]),
            torch.from_numpy(stats["outputStd"]),
            input_steps=int(architecture["inputSteps"]),
            output_steps=int(architecture["outputSteps"]),
            feature_width=int(architecture["featureWidth"]),
            market_width=int(architecture["marketWidth"]),
            prefix_width=int(architecture["prefixWidth"]),
            feature_distribution_width=int(architecture["featureDistributionWidth"]),
            extended_prefix_width=int(architecture["extendedPrefixWidth"]),
            next_feature_distribution_width=int(
                architecture["nextFeatureDistributionWidth"]
            ),
            initial_radius=float(architecture["initialRadius"]),
            minimum_radius=float(architecture["minimumRadius"]),
            learnable_centering=bool(architecture["learnableCentering"]),
            linear_rank=(
                None
                if architecture.get("linearFactorization") is None
                else int(architecture["linearFactorization"]["rank"])
            ),
            layer8_attention=architecture.get("layer8Attention"),
            layer8_function_approximator=architecture.get(
                "layer8FunctionApproximator"
            ),
            recurrent_memory=architecture.get("recurrentMemory"),
            feature_embedding_density=architecture.get(
                "featureEmbeddingDensity"
            ),
            return_density=architecture.get("returnDensity"),
            return_density_contract=return_density,
            hindsight_conditioning=architecture.get("hindsightConditioning"),
            recurrent_activation_checkpointing=bool(
                architecture.get("recurrentActivationCheckpointing", False)
            ),
        ).to(device)
        if return_density is not None and not uses_joint_return_density:
            return_knots = torch.from_numpy(
                return_density.knots_unit
            ).to(device=device).float()
            return_areas = triangular_basis_areas(return_knots)
            return_prior_masses = torch.from_numpy(
                return_density.prior_component_masses
            ).to(device=device).float()
            model.initialize_return_density_prior(
                return_areas, return_prior_masses,
            )
            return_component_means = torch.from_numpy(component_return_means(
                return_density.knots_unit, return_density.transform,
            )).to(device=device).float()
        else:
            return_knots = None
            return_areas = None
            return_component_means = None
        parameter_count = sum(value.numel() for value in model.parameters())
        trainable_count = sum(
            value.numel() for value in model.parameters() if value.requires_grad
        )
        optimizer_parameter_groups(model)
        optimizers = build_optimizers(model, training, device)
        epochs = run_epoch_limit(run_root, plan)
        target_free_phase = target_free_continuation(run_root, plan, epochs)
        steps_per_epoch = math.ceil(dataset.counts["train"] / batch_size)
        ema_half_life = float(training["weightEma"]["halfLifeEpochs"])
        ema_decay = mean_teacher_ema_decay(ema_half_life, steps_per_epoch)
        ema = {name: value.detach().clone() for name, value in model.state_dict().items()}
        uses_embedding_density = uses_embedding_density_objective(plan)
        uses_base_support_density = uses_base_support_embedding_density(plan)
        best = (
            {
                "validation-nll": {"score": math.inf, "epoch": -1},
                "validation-mse": {"score": math.inf, "epoch": -1},
                "validation-correlation": {
                    "score": -math.inf, "epoch": -1,
                },
            }
            if uses_return_density else
            {
                "validation-nll": {"score": math.inf, "epoch": -1},
                **(
                    {
                        "validation-mse": {"score": math.inf, "epoch": -1},
                        "validation-correlation": {
                            "score": -math.inf, "epoch": -1,
                        },
                    }
                    if uses_base_support_density else {}
                ),
            }
            if uses_embedding_density
            else {
                "validation-mse": {"score": math.inf, "epoch": -1},
                "validation-correlation": {"score": -math.inf, "epoch": -1},
            }
        )
        last_file = run_root / "checkpoints/last.json"
        start_epoch = 0
        global_step = 0
        hindsight_schedule = (
            {"type": TARGET_FREE_SCHEDULE_TYPE}
            if target_free_phase is not None else training.get("hindsightNoiseSchedule")
        )
        hindsight_curriculum = initial_hindsight_curriculum(hindsight_schedule)
        if checkpoint_exists(last_file):
            saved = load_torch_checkpoint(last_file, map_location=device, weights_only=False)
            if saved.get("planSha256") != plan_hash \
                    or saved.get("runnerContract") != RUNNER_CONTRACT:
                raise ValueError("structured feature checkpoint contract changed")
            if target_free_phase is not None:
                pointer = json.loads(last_file.read_text(encoding="utf-8"))
                validate_target_free_resume(target_free_phase, saved, pointer["object"]["contentHash"])
            model.load_state_dict(saved["model"])
            ema = saved["emaModel"]
            for optimizer, state in zip(optimizers, saved["optimizers"], strict=True):
                optimizer.load_state_dict(state)
            start_epoch = int(saved["epoch"]) + 1
            global_step = int(saved["globalStep"])
            best = saved["best"]
            if hindsight_curriculum is not None:
                hindsight_curriculum = saved.get("hindsightCurriculum")
                if not isinstance(hindsight_curriculum, dict):
                    raise ValueError("checkpoint is missing its hindsight curriculum")
                hindsight_variance_for_epoch(
                    hindsight_schedule, start_epoch, hindsight_curriculum,
                )
        elif target_free_phase is not None:
            raise ValueError("target-free continuation requires its saved checkpoint")

        layer8_function = architecture.get("layer8FunctionApproximator")
        gram_identity_loss_weight = 0.0
        if isinstance(layer8_function, dict):
            loss_configuration = layer8_function.get(
                "orthogonalizationLoss"
            )
            if isinstance(loss_configuration, dict):
                gram_identity_loss_weight = float(
                    loss_configuration.get("weight", 0.0)
                )
        uses_layer8_auxiliary_loss = gram_identity_loss_weight > 0
        training_forward = (
            model.forward_with_joint_return_density
            if uses_joint_return_density
            else model.forward_with_return_density
            if uses_return_density
            else model.feature_embedding_log_density
            if uses_embedding_density and not uses_base_support_density
            else model.forward_with_auxiliary_loss
            if uses_layer8_auxiliary_loss
            else model.forward_samples
            if model.hindsight_sample_count > 1
            else model
        )
        training_model = training_forward
        compile_scope = "disabled"
        if uses_joint_return_density and args.compile_mode != "none":
            compile_cache = (repo / "data/training/cache/torchinductor").resolve()
            compile_cache.mkdir(parents=True, exist_ok=True)
            os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(compile_cache)
            os.environ["TRITON_CACHE_DIR"] = str(compile_cache / "triton")
            model.return_density_head.compile_shared_recurrent_step(
                mode="default",
                dynamic=False,
            )
            compile_scope = "joint-head-shared-recurrent-step"
        elif args.compile_mode != "none" and not uses_base_support_density:
            compile_cache = (repo / "data/training/cache/torchinductor").resolve()
            compile_cache.mkdir(parents=True, exist_ok=True)
            os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(compile_cache)
            os.environ["TRITON_CACHE_DIR"] = str(compile_cache / "triton")
            compile_arguments: dict = {"fullgraph": False, "dynamic": False}
            if args.compile_mode == "default":
                compile_arguments["options"] = {"triton.cudagraphs": False}
            else:
                compile_arguments["mode"] = args.compile_mode
            training_model = torch.compile(
                training_forward, **compile_arguments
            )
            compile_scope = "fixed-shape-training-only"
        elif uses_base_support_density:
            compile_scope = "eager-causal-base-support-density"

        reporter.emit({
            "event": "training-start",
            "planId": plan["id"],
            "startEpoch": start_epoch,
            "epochs": epochs,
            "originalPlanEpochs": training["epochs"],
            "epochLimitSource": (
                "state/epoch-limit.json" if epochs != training["epochs"] else "plan"
            ),
            "parameters": parameter_count,
            "trainableParameters": trainable_count,
            "architecture": architecture,
            "objective": objective_contract(plan),
            "distributionLoss": (
                architecture.get("featureEmbeddingDensity")
                if uses_embedding_density else None
            ),
            "returnDensityLoss": (
                {
                    **architecture["returnDensity"],
                    "source": plan["density"]["source"],
                    "fit": plan["density"]["fit"],
                    "objectiveWeight": training["loss"]["weights"][
                        "returnDensityNll"
                    ],
                }
                if uses_return_density else None
            ),
            "samRho": 0,
            "inputDropoutProbability": 0,
            "embeddingDropoutProbability": 0,
            "adversarialInput": None,
            "adversarialOutput": None,
            "hindsightConditioning": architecture.get("hindsightConditioning"),
            "hindsightTeacher": plan.get("hindsightTeacher"),
            "hindsightNoiseSchedule": hindsight_schedule,
            "targetFreePhase": target_free_phase,
            "hindsightCurriculum": hindsight_curriculum,
            "layer8AuxiliaryLoss": (
                {
                    "type": "gram-identity-mean-square-v1",
                    "weight": gram_identity_loss_weight,
                }
                if uses_layer8_auxiliary_loss
                else None
            ),
            "weightEmaHalfLifeEpochs": ema_half_life,
            "weightEmaDecayPerStep": ema_decay,
            "runtimeOptimization": {
                "device": str(device),
                "batchSize": batch_size,
                "evaluationBatchSize": evaluation_batch_size,
                "matmulPrecision": args.matmul_precision,
                "compileMode": args.compile_mode,
                "compileScope": compile_scope,
                "cudaGraphsEnabled": False,
                "evaluationExecution": "eager",
            },
        })
        reporter.status(
            "training", planId=plan["id"], epochs=epochs,
            parameters=parameter_count, examples=dataset.counts["train"],
        )
        started = time.monotonic()
        smoke_limit = None if args.smoke_batches is None \
            else args.smoke_batches * batch_size
        for epoch in range(start_epoch, epochs):
            model.train()
            hindsight_variance = (
                hindsight_variance_for_epoch(
                    hindsight_schedule, epoch, hindsight_curriculum,
                )
                if model.hindsight_conditioning is not None else None
            )
            hindsight_generator = (
                torch.Generator(device=device).manual_seed(seed + 1_000_003 + epoch)
                if hindsight_variance is not None else None
            )
            teacher_rng = np.random.default_rng(seed + 1_000_003 + epoch) \
                if isinstance(dataset, TeacherEmbeddingDataset) else None
            objective_sum = 0.0
            objective_component_sums: dict[str, float] = {}
            example_count = 0
            optimization_started = time.monotonic()
            for batch in dataset.iter_batches(
                "train", batch_size, shuffle=True, seed=seed + epoch,
                limit=smoke_limit,
                # Keep every optimized row real. The final short batch is
                # exact and avoids NaN-producing synthetic feature geometry.
                pad=False,
            ):
                if derived_base:
                    assert isinstance(batch, StructuredUnion530Batch)
                    inputs, targets, weights = (
                        batch.inputs, batch.targets, batch.weights
                    )
                    context = {
                        name: value.to(device, non_blocking=True)
                        for name, value in batch.context.items()
                    }
                else:
                    inputs, targets, weights = batch
                    context = None
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                weights = weights.to(device, non_blocking=True)
                for optimizer in optimizers:
                    optimizer.zero_grad(set_to_none=True)
                if uses_return_density:
                    if not isinstance(dataset, StructuredProduction59BaseDataset) \
                            or context is None \
                            or return_density is None:
                        raise AssertionError(
                            "structured return density requires production59 context"
                        )
                    if uses_joint_return_density:
                        standardized_direct, density_output = training_model(
                            inputs,
                            targets[:, :, COMPARABLE_RETURN_CHANNEL],
                        )
                    else:
                        if return_knots is None or return_areas is None:
                            raise AssertionError(
                                "scalar return-density grid is missing"
                            )
                        standardized_direct, density_logits = training_model(
                            inputs
                        )
                    direct = model.raw_outputs(standardized_direct)
                    derived_prediction = dataset.derive(
                        direct, inputs, context,
                    )
                    assert derived_output_mean is not None \
                        and derived_output_std is not None
                    standardized_prediction = (
                        derived_prediction - derived_output_mean
                    ) / derived_output_std
                    standardized_target = (
                        targets - derived_output_mean
                    ) / derived_output_std
                    feature_mse = weighted_standardized_mse(
                        standardized_prediction, standardized_target, weights,
                    )
                    if uses_joint_return_density:
                        if density_output.joint_log_density_terms is None:
                            raise RuntimeError(
                                "joint path density did not emit contracted terms"
                            )
                        per_step_return_nll = \
                            -density_output.joint_log_density_terms
                    else:
                        per_step_return_nll, _log_masses, \
                            _unit_log_density = \
                            return_negative_log_likelihood(
                                density_logits,
                                targets[:, :, COMPARABLE_RETURN_CHANNEL],
                                return_knots,
                                return_areas,
                                return_density.transform,
                            )
                    active = weights[:, None]
                    return_nll = (per_step_return_nll * active).sum() / (
                        active.sum().clamp_min(1.0) * dataset.output_steps
                    )
                    first_step_return_nll = (
                        per_step_return_nll[:, 0] * weights
                    ).sum() / weights.sum().clamp_min(1.0)
                    components = structured_feature_return_density_objective(
                        feature_mse, return_nll, training["loss"],
                    )
                    components[
                        "returnDensityFirstStepNegativeLogLikelihood"
                    ] = first_step_return_nll
                    loss = components["objective"]
                    prediction = None
                    layer8_gram_identity_loss = None
                elif uses_embedding_density:
                    if uses_base_support_density:
                        if not isinstance(dataset, StructuredUnion530BaseDataset) \
                                or context is None:
                            raise AssertionError(
                                "base-support density requires causal context"
                            )
                        log_density, expected_base, expected_features, _ = \
                            base_support_density_outputs(
                                model, dataset, inputs, targets, context,
                            )
                    else:
                        log_density = training_model(inputs, targets)
                    active = weights[:, None]
                    loss = -(log_density * active).sum() / (
                        active.sum().clamp_min(1.0) * dataset.output_steps
                    )
                    prediction = None
                    layer8_gram_identity_loss = None
                    components = {
                        "objective": loss,
                        "embeddingNegativeLogLikelihood": loss,
                    }
                    if uses_base_support_density:
                        standardized_expected_base = (
                            expected_base - model.output_mean
                        ) / model.output_std
                        standardized_base_target = model.standardized_targets(
                            context["baseTargets"]
                        )
                        components["expectedBaseNormalizedMse"] = \
                            weighted_standardized_mse(
                                standardized_expected_base,
                                standardized_base_target,
                                weights,
                            )
                elif uses_layer8_auxiliary_loss:
                    prediction, layer8_gram_identity_loss = \
                        training_model(inputs)
                elif hindsight_variance is not None:
                    if isinstance(dataset, TeacherEmbeddingDataset):
                        noisy_hindsight = dataset.hindsight(inputs, batch.teacher_context,
                            targets=targets if hindsight_variance < 1 else None,
                            fraction=hindsight_variance, samples=model.hindsight_sample_count,
                            rng=teacher_rng)
                    else:
                        clean_hindsight = expand_hindsight_samples(
                            model.standardized_targets(targets), model.hindsight_sample_count,
                        )
                        hindsight_noise = torch.randn(
                            clean_hindsight.shape, device=device,
                            dtype=clean_hindsight.dtype, generator=hindsight_generator,
                        )
                        noisy_hindsight = corrupt_standardized_hindsight(
                            clean_hindsight, hindsight_noise, hindsight_variance,
                        )
                    variance_input = torch.full(
                        (inputs.shape[0], 1), hindsight_variance,
                        dtype=inputs.dtype, device=device,
                    )
                    prediction = training_model(inputs, noisy_hindsight, variance_input)
                    layer8_gram_identity_loss = None
                else:
                    prediction = training_model(inputs)
                    layer8_gram_identity_loss = None
                if not uses_embedding_density and not uses_return_density:
                    assert prediction is not None
                    standardized_direct = prediction
                    if derived_base:
                        direct = model.raw_outputs(prediction)
                        prediction = dataset.derive(direct, inputs, context)
                        assert derived_output_mean is not None \
                            and derived_output_std is not None
                        prediction = (
                            prediction - derived_output_mean
                        ) / derived_output_std
                        standardized_target = (
                            targets - derived_output_mean
                        ) / derived_output_std
                    else:
                        standardized_target = model.standardized_targets(targets)
                        if model.hindsight_sample_count > 1:
                            standardized_target = expand_hindsight_samples(
                                standardized_target, model.hindsight_sample_count,
                            )
                    if objective_contract(plan) \
                            == PRODUCTION59_BALANCED_OBJECTIVE_CONTRACT:
                        if context is None:
                            raise AssertionError(
                                "balanced production59 objective requires context"
                            )
                        components = production59_balanced_objective(
                            prediction,
                            standardized_target,
                            standardized_direct,
                            model.standardized_targets(context["baseTargets"]),
                            weights,
                            context,
                            training["loss"],
                        )
                        loss = components["objective"]
                    else:
                        loss = weighted_standardized_mse(
                            prediction, standardized_target, weights
                        )
                        components = {"objective": loss}
                if layer8_gram_identity_loss is not None:
                    data_objective = loss
                    loss = data_objective + (
                        gram_identity_loss_weight
                        * layer8_gram_identity_loss
                    )
                    components = {
                        **components,
                        "dataObjective": data_objective,
                        "basisGramIdentity": layer8_gram_identity_loss,
                        "objective": loss,
                    }
                loss.backward()
                gradient_norm = clip_grad_norm_(
                    model.parameters(), float(training["gradientClip"]),
                    foreach=device.type == "cuda",
                )
                if not torch.isfinite(gradient_norm):
                    raise FloatingPointError("structured feature gradient is non-finite")
                for optimizer in optimizers:
                    optimizer.step()
                update_weight_ema(ema, model, ema_decay)
                valid = int(weights.sum().item())
                objective_sum += float(loss.detach()) * valid
                for name, value in components.items():
                    objective_component_sums[name] = (
                        objective_component_sums.get(name, 0.0)
                        + float(value.detach()) * valid
                    )
                example_count += valid
                global_step += 1

            if device.type == "cuda":
                torch.cuda.synchronize()
            optimization_seconds = time.monotonic() - optimization_started
            if device.type == "cuda":
                torch.cuda.empty_cache()
            hindsight_gate_probe = None
            hindsight_validation_reconstruction = None
            hindsight_validation_seconds = 0.0
            next_hindsight_variance = hindsight_variance
            if hindsight_curriculum is not None:
                hindsight_gate_probe = evaluate_hindsight_curriculum_probe(
                    model, dataset, hindsight_schedule,
                    variance=hindsight_variance,
                    batch_size=evaluation_batch_size, device=device,
                    limit=smoke_limit,
                )
                hindsight_curriculum = advance_hindsight_curriculum(
                    hindsight_schedule, hindsight_curriculum,
                    epoch=epoch, correlation=hindsight_gate_probe["correlation"],
                )
                next_hindsight_variance = hindsight_variance_for_epoch(
                    hindsight_schedule, epoch + 1, hindsight_curriculum,
                )
                reconstruction_started = time.monotonic()
                hindsight_validation_reconstruction = evaluate_hindsight_reconstruction(
                    model, dataset, hindsight_schedule,
                    split="validation", variance=hindsight_variance,
                    batch_size=evaluation_batch_size, device=device,
                    limit=smoke_limit,
                )
                hindsight_validation_seconds = time.monotonic() - reconstruction_started
            with use_state(model, ema):
                if uses_return_density:
                    assert isinstance(dataset, StructuredProduction59BaseDataset) \
                        and return_density is not None
                    train_evaluation = evaluate_structured_return_density(
                        model, dataset, "train",
                        batch_size=evaluation_batch_size,
                        device=device,
                        density=return_density,
                        limit=(
                            smoke_limit if smoke_limit is not None else int(
                                training["epochTrainEvaluationExamples"]
                            )
                        ),
                    )
                    validation_evaluation = \
                        evaluate_structured_return_density(
                            model, dataset, "validation",
                            batch_size=evaluation_batch_size,
                            device=device,
                            density=return_density,
                            limit=smoke_limit,
                        )
                else:
                    evaluation_function = (
                        evaluate_feature_embedding_density
                        if uses_embedding_density else evaluate
                    )
                    train_evaluation = evaluation_function(
                        model, dataset, "train", batch_size=evaluation_batch_size,
                        device=device,
                        limit=smoke_limit if smoke_limit is not None else int(
                            training["epochTrainEvaluationExamples"]
                        ),
                    )
                    validation_evaluation = evaluation_function(
                        model, dataset, "validation",
                        batch_size=evaluation_batch_size, device=device,
                        limit=smoke_limit,
                    )
            if uses_return_density:
                train_metrics = train_evaluation["nextReturn"]
                validation_metrics = validation_evaluation["nextReturn"]
                train_feature_state = train_evaluation["featureState"]
                validation_feature_state = validation_evaluation["featureState"]
                validation_correlation = validation_metrics["correlation"]
                candidates = {
                    "validation-nll": float(
                        validation_evaluation["negativeLogLikelihood"]
                    ),
                    "validation-mse": float(
                        validation_metrics["normalizedMse"]
                    ),
                    "validation-correlation": (
                        -math.inf
                        if validation_correlation is None
                        else float(validation_correlation)
                    ),
                }
            elif uses_embedding_density:
                candidates = {
                    "validation-nll": float(
                        validation_evaluation["negativeLogLikelihood"]
                    ),
                }
                if uses_base_support_density:
                    candidates.update({
                        "validation-mse": float(
                            validation_evaluation["featureState"][
                                "normalizedMse"
                            ]
                        ),
                        "validation-correlation": float(
                            validation_evaluation["featureState"][
                                "correlation"
                            ]
                        ),
                    })
            else:
                train_metrics = train_evaluation["nextReturn"]
                validation_metrics = validation_evaluation["nextReturn"]
                train_feature_state = train_evaluation["featureState"]
                validation_feature_state = validation_evaluation["featureState"]
                selection_metrics = (
                    validation_metrics if model.hindsight_conditioning is not None
                    else validation_feature_state
                )
                selection_correlation = selection_metrics["correlation"]
                candidates = {
                    "validation-mse": float(
                        selection_metrics["normalizedMse"]
                    ),
                    "validation-correlation": (
                        -math.inf if selection_correlation is None
                        else float(selection_correlation)
                    ),
                }
            for policy, score in candidates.items():
                improved = score > best[policy]["score"] \
                    if policy.endswith("correlation") else score < best[policy]["score"]
                if improved:
                    best[policy] = {"score": score, "epoch": epoch}
                    save_torch_checkpoint({
                        "model": ema,
                        "epoch": epoch,
                        "score": score,
                        "policy": policy,
                        "planSha256": plan_hash,
                        "runnerContract": RUNNER_CONTRACT,
                    }, run_root / f"checkpoints/selections/{policy}.json")
            save_torch_checkpoint(checkpoint_payload(
                model, ema, optimizers, epoch=epoch, global_step=global_step,
                plan_hash=plan_hash, best=best,
                hindsight_curriculum=hindsight_curriculum,
                target_free_phase=target_free_phase,
            ), last_file)
            event = {
                "event": "minute-return-epoch",
                "planId": plan["id"],
                "epoch": epoch,
                "epochs": epochs,
                "globalStep": global_step,
                "seconds": time.monotonic() - started,
                "evaluationHorizon": {"steps": 1, "seconds": 1, "lead": 0},
                "modelInputSteps": dataset.input_steps,
                "modelOutputSteps": dataset.output_steps,
                "onlineObjective": objective_sum / example_count,
                "onlineObjectiveComponents": {
                    name: value / example_count
                    for name, value in objective_component_sums.items()
                },
                "objective": objective_contract(plan),
                "parameterCount": parameter_count,
            }
            if uses_return_density:
                event.update({
                    "evaluationScope": COMPARABLE_EVALUATION_SCOPE,
                    "train": train_metrics,
                    "validation": validation_metrics,
                    "trainDistribution": {
                        "negativeLogLikelihood": train_evaluation[
                            "negativeLogLikelihood"
                        ],
                        "perLeadNegativeLogLikelihood": train_evaluation[
                            "perLeadNegativeLogLikelihood"
                        ],
                        "expectation": train_evaluation["returnPath"],
                        "perLeadExpectation": train_evaluation[
                            "perStepNextReturn"
                        ],
                    },
                    "validationDistribution": {
                        "negativeLogLikelihood": validation_evaluation[
                            "negativeLogLikelihood"
                        ],
                        "perLeadNegativeLogLikelihood": validation_evaluation[
                            "perLeadNegativeLogLikelihood"
                        ],
                        "expectation": validation_evaluation["returnPath"],
                        "perLeadExpectation": validation_evaluation[
                            "perStepNextReturn"
                        ],
                    },
                    "featurePredictionNextReturn": {
                        "train": train_evaluation[
                            "featurePredictionNextReturn"
                        ],
                        "validation": validation_evaluation[
                            "featurePredictionNextReturn"
                        ],
                    },
                    "featureState": {
                        "evaluationScope": FEATURE_STATE_EVALUATION_SCOPE,
                        "train": compact_feature_state_metrics(train_feature_state),
                        "validation": compact_feature_state_metrics(
                            validation_feature_state
                        ),
                    },
                    "perStepFeatureState": {
                        "train": [
                            compact_feature_state_metrics(value)
                            for value in train_evaluation["perStepFeatureState"]
                        ],
                        "validation": [
                            compact_feature_state_metrics(value)
                            for value in validation_evaluation[
                                "perStepFeatureState"
                            ]
                        ],
                    },
                    "onlineNegativeLogLikelihood": (
                        objective_component_sums[
                            "returnDensityNegativeLogLikelihood"
                        ] / example_count
                    ),
                    "onlineFirstStepNegativeLogLikelihood": (
                        objective_component_sums[
                            "returnDensityFirstStepNegativeLogLikelihood"
                        ] / example_count
                    ),
                    "checkpointSelectionScope": (
                        "validation return-density expectation and NLL"
                    ),
                    "bestValidationNegativeLogLikelihood": best[
                        "validation-nll"
                    ]["score"],
                    "bestValidationMse": best["validation-mse"]["score"],
                    "bestValidationCorrelation": best[
                        "validation-correlation"
                    ]["score"],
                })
            elif uses_embedding_density:
                event.update({
                    "evaluationScope": "realized-layer-1-feature-embedding-density",
                    "trainDistribution": train_evaluation,
                    "validationDistribution": validation_evaluation,
                    "onlineNegativeLogLikelihood": (
                        objective_component_sums[
                            "embeddingNegativeLogLikelihood"
                        ] / example_count
                    ),
                    "onlineFirstStepNegativeLogLikelihood": (
                        objective_component_sums[
                            "embeddingNegativeLogLikelihood"
                        ] / example_count
                    ),
                    "checkpointSelectionScope": "validation-embedding-nll",
                    "bestValidationNegativeLogLikelihood": best[
                        "validation-nll"
                    ]["score"],
                })
                if uses_base_support_density:
                    event.update({
                        "train": train_evaluation["nextReturn"],
                        "validation": validation_evaluation["nextReturn"],
                        "featureState": {
                            "evaluationScope": FEATURE_STATE_EVALUATION_SCOPE,
                            "train": compact_feature_state_metrics(
                                train_evaluation["featureState"]
                            ),
                            "validation": compact_feature_state_metrics(
                                validation_evaluation["featureState"]
                            ),
                        },
                        "perStepFeatureState": {
                            "train": [
                                compact_feature_state_metrics(value)
                                for value in train_evaluation[
                                    "perStepFeatureState"
                                ]
                            ],
                            "validation": [
                                compact_feature_state_metrics(value)
                                for value in validation_evaluation[
                                    "perStepFeatureState"
                                ]
                            ],
                        },
                        "bestFeatureStateValidationMse": best[
                            "validation-mse"
                        ]["score"],
                        "bestFeatureStateValidationCorrelation": best[
                            "validation-correlation"
                        ]["score"],
                    })
            else:
                event.update({
                    "evaluationScope": COMPARABLE_EVALUATION_SCOPE,
                    "train": train_metrics,
                    "validation": validation_metrics,
                    "trainDistribution": {
                        "expectation": train_evaluation["returnPath"],
                        "perLeadExpectation": train_evaluation[
                            "perStepNextReturn"
                        ],
                    },
                    "validationDistribution": {
                        "expectation": validation_evaluation["returnPath"],
                        "perLeadExpectation": validation_evaluation[
                            "perStepNextReturn"
                        ],
                    },
                    "featureState": {
                        "evaluationScope": FEATURE_STATE_EVALUATION_SCOPE,
                        "train": compact_feature_state_metrics(train_feature_state),
                        "validation": compact_feature_state_metrics(
                            validation_feature_state
                        ),
                    },
                    "perStepFeatureState": {
                        "train": [
                            compact_feature_state_metrics(value)
                            for value in train_evaluation["perStepFeatureState"]
                        ],
                        "validation": [
                            compact_feature_state_metrics(value)
                            for value in validation_evaluation[
                                "perStepFeatureState"
                            ]
                        ],
                    },
                    "checkpointSelectionScope": FEATURE_STATE_EVALUATION_SCOPE,
                    "bestFeatureStateValidationMse": best[
                        "validation-mse"
                    ]["score"],
                    "bestFeatureStateValidationCorrelation": best[
                        "validation-correlation"
                    ]["score"],
                })
            if hindsight_variance is not None:
                event["hindsight"] = {
                    "trainingNoiseVariance": hindsight_variance,
                    "signalScale": 1.0 - hindsight_variance if isinstance(dataset, TeacherEmbeddingDataset)
                    else math.sqrt(1.0 - hindsight_variance),
                    "noiseScale": hindsight_variance if isinstance(dataset, TeacherEmbeddingDataset)
                    else math.sqrt(hindsight_variance),
                    "parameterMeaning": "teacher-replacement-fraction"
                    if isinstance(dataset, TeacherEmbeddingDataset) else "gaussian-noise-variance",
                    "conditioningSource": "frozen-teacher-component-centers"
                    if isinstance(dataset, TeacherEmbeddingDataset) else "independent-gaussian",
                    "sampleInputWidth": model.hindsight_input_width,
                    "evaluationNoiseVariance": 1.0,
                    "evaluationTargetContribution": 0.0,
                    "scheduleType": hindsight_schedule["type"],
                    "denoisingPasses": 1,
                    "layer2ResidualPasses": model.readout_residual_passes,
                    "layer2ParameterSharing": "across-forecast-steps-only",
                    "layer13Enabled": model.layer13 is not None,
                    "layer13Count": (1 + len(model.layer13_refinements)) if model.layer13 is not None else 0,
                    "sampleCount": model.hindsight_sample_count,
                    "sampleEmbeddingWidth": model.readout_sample_width,
                    "samplePrediction": "arithmetic-mean",
                    "sampleObjective": "mean-per-sample-mse",
                }
                if hindsight_gate_probe is not None:
                    event["hindsight"].update({
                        "gateProbe": hindsight_gate_probe,
                        "validationReconstruction": hindsight_validation_reconstruction,
                        "validationReconstructionSeconds": hindsight_validation_seconds,
                        "optimizationSeconds": optimization_seconds,
                        "requiredCorrelation": hindsight_schedule["requiredCorrelation"],
                        "varianceIncrement": hindsight_schedule["varianceIncrement"],
                        "nextTrainingNoiseVariance": next_hindsight_variance,
                        "noiseIncreased": next_hindsight_variance > hindsight_variance,
                    })
                elif target_free_phase is not None:
                    event["hindsight"].update({
                        "gateEnabled": False,
                        "trainingTargetContribution": 0.0,
                        "nextTrainingNoiseVariance": 1.0,
                        "phaseEpoch": epoch - target_free_phase["sourceCompletedEpochs"] + 1,
                        "phaseEpochs": target_free_phase["additionalEpochs"],
                        "optimizationSeconds": optimization_seconds,
                    })
                else:
                    event["hindsight"]["scheduleEndEpoch"] = hindsight_schedule["endEpoch"]
                event["checkpointSelectionScope"] = COMPARABLE_EVALUATION_SCOPE
                event["bestValidationMse"] = best["validation-mse"]["score"]
                event["bestValidationCorrelation"] = best["validation-correlation"]["score"]
                event.pop("bestFeatureStateValidationMse", None)
                event.pop("bestFeatureStateValidationCorrelation", None)
            reporter.emit(event)
            reporter.status("training", planId=plan["id"], latest=event)
            if args.smoke_batches is not None:
                reporter.status("smoke-complete", planId=plan["id"], latest=event)
                return

        policies: dict[str, dict] = {}
        for policy in best:
            selected = load_torch_checkpoint(
                run_root / f"checkpoints/selections/{policy}.json",
                map_location=device,
                weights_only=False,
            )
            model.load_state_dict(selected["model"])
            if uses_return_density:
                assert isinstance(dataset, StructuredProduction59BaseDataset) \
                    and return_density is not None
                evaluations = {
                    split: evaluate_structured_return_density(
                        model, dataset, split,
                        batch_size=evaluation_batch_size,
                        device=device,
                        density=return_density,
                    )
                    for split in ("train", "validation", "test")
                }
            else:
                completion_evaluator = (
                    evaluate_feature_embedding_density
                    if uses_embedding_density else evaluate
                )
                evaluations = {
                    split: completion_evaluator(
                        model, dataset, split, batch_size=evaluation_batch_size,
                        device=device,
                    )
                    for split in ("train", "validation", "test")
                }
            if uses_return_density:
                policies[f"best-{policy}"] = {
                    "epoch": int(selected["epoch"]),
                    "selectionScore": float(selected["score"]),
                    "selectionMetric": (
                        "returnDensityExpectation.correlation"
                        if policy.endswith("correlation")
                        else "returnDensityExpectation.normalizedMse"
                        if policy.endswith("mse")
                        else "validation.negativeLogLikelihood"
                    ),
                    "checkpointPolicy": f"best-{policy}",
                    "density": evaluations,
                    **comparable_policy_metrics(evaluations),
                }
            elif uses_embedding_density:
                policies[f"best-{policy}"] = {
                    "epoch": int(selected["epoch"]),
                    "selectionScore": float(selected["score"]),
                    "selectionMetric": (
                        "featureState.correlation"
                        if policy.endswith("correlation")
                        else "featureState.normalizedMse"
                        if policy.endswith("mse")
                        else "validation.negativeLogLikelihood"
                    ),
                    "checkpointPolicy": f"best-{policy}",
                    **(
                        {
                            "density": evaluations,
                            **comparable_policy_metrics(evaluations),
                        }
                        if uses_base_support_density
                        else {"distribution": evaluations}
                    ),
                }
            else:
                policies[f"best-{policy}"] = {
                    "epoch": int(selected["epoch"]),
                    "selectionScore": float(selected["score"]),
                    "selectionMetric": (
                        "featureState.correlation"
                        if policy.endswith("correlation")
                        else "featureState.normalizedMse"
                    ),
                    "checkpointPolicy": f"best-{policy}",
                    **comparable_policy_metrics(evaluations),
                }
        last = load_torch_checkpoint(last_file, map_location=device, weights_only=False)
        model.load_state_dict(last["emaModel"])
        if uses_return_density:
            assert isinstance(dataset, StructuredProduction59BaseDataset) \
                and return_density is not None
            last_evaluations = {
                split: evaluate_structured_return_density(
                    model, dataset, split,
                    batch_size=evaluation_batch_size,
                    device=device,
                    density=return_density,
                )
                for split in ("train", "validation", "test")
            }
        else:
            last_evaluations = {
                split: completion_evaluator(
                    model, dataset, split, batch_size=evaluation_batch_size,
                    device=device,
                )
                for split in ("train", "validation", "test")
            }
        policies["last"] = {
            "epoch": int(last["epoch"]),
            "selectionScore": None,
            "selectionMetric": None,
            "checkpointPolicy": "last",
            **(
                {
                    "density": last_evaluations,
                    **comparable_policy_metrics(last_evaluations),
                }
                if uses_return_density
                else
                {
                    "density": last_evaluations,
                    **comparable_policy_metrics(last_evaluations),
                }
                if uses_base_support_density
                else {"distribution": last_evaluations}
                if uses_embedding_density
                else comparable_policy_metrics(last_evaluations)
            ),
        }
        completed_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        persistence = (
            persist_structured_return_density_evaluation
            if uses_return_density
            else persist_embedding_density_evaluation
            if uses_embedding_density
            else persist_structured_feature_evaluation
        )
        artifact, result = persistence(
            repo=repo,
            run_root=run_root,
            plan=plan,
            dataset=dataset,
            parameter_count=parameter_count,
            policies=policies,
            generated_at=completed_at,
        )
        reporter.emit({
            "event": "training-complete",
            "planId": plan["id"],
            "completedAt": completed_at,
            "selectionPolicy": result["selectionPolicy"],
            "bestEpoch": result["bestEpoch"],
            "bestValidationScore": result["bestValidationScore"],
            "evaluationArtifact": result["evaluationArtifact"],
        })
        reporter.status(
            "complete", planId=plan["id"], evaluatedAt=completed_at,
            evaluation={
                "artifact": result["evaluationArtifact"],
                "bestEpoch": result["bestEpoch"],
                **(
                    {
                        "bestTest": result["density"]["test"],
                        "lastTest": artifact["policies"]["last"][
                            "density"
                        ]["test"],
                    }
                    if uses_return_density
                    else
                    {
                        "bestTest": (
                            result["density"]["test"]
                            if uses_base_support_density
                            else result["distribution"]["test"]
                        ),
                        "lastTest": (
                            artifact["policies"]["last"]["density"]["test"]
                            if uses_base_support_density
                            else artifact["policies"]["last"][
                                "distribution"
                            ]["test"]
                        ),
                    }
                    if uses_embedding_density else {
                        "bestTest": result["test"],
                        "lastTest": artifact["policies"]["last"]["test"],
                    }
                ),
            },
        )
    except BaseException as error:
        if not isinstance(error, SystemExit):
            reporter.status(
                "failed", planId=plan.get("id"), error=repr(error),
            )
        raise


if __name__ == "__main__":
    main()
