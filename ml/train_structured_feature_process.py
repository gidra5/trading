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

from normalized_glu_next_return import optimizer_parameter_groups
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
COMPARABLE_RETURN_CHANNEL = 0
COMPARABLE_EVALUATION_SCOPE = "next-completed-1s-signed-log-return"
FEATURE_STATE_EVALUATION_SCOPE = "complete-next-feature-state"


def objective_contract(plan: dict) -> str:
    return str(plan["training"]["loss"]["type"])


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
    return StructuredFeatureSequenceDataset(
        (repo / plan["datasetDir"]).resolve(),
        input_steps=int(architecture["inputSteps"]),
        output_steps=int(architecture["outputSteps"]),
        train_examples=int(plan["subset"]["examples"]),
    )


def canonical_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("utf-8")).hexdigest()


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
    per_example = (prediction - target).square().mean(dim=(1, 2))
    return (per_example * weights).sum() / weights.sum().clamp_min(1)


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
        direct = model.raw_outputs(model(inputs))
        prediction = (
            dataset.derive(direct, inputs, context)
            if derived_base else direct
        )
        metrics.add(prediction, targets, weights)
    return metrics.result()


def comparable_policy_metrics(evaluations: dict[str, dict]) -> dict:
    expected = {"train", "validation", "test"}
    if set(evaluations) != expected:
        raise ValueError(f"structured evaluation splits changed: {sorted(evaluations)}")
    return {
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
        "checkpointSelectionMetric": "featureState.validation",
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
        "selectionMetric": "featureState.normalizedMse",
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
    training = plan["training"]
    loss_type = training.get("loss", {}).get("type")
    expected_losses = (
        {UNION530_BASE_OBJECTIVE_CONTRACT}
        if is_union530_base_plan(plan)
        else {
            PRODUCTION59_OBJECTIVE_CONTRACT,
            PRODUCTION59_BALANCED_OBJECTIVE_CONTRACT,
        }
        if is_production59_base_plan(plan)
        else {OBJECTIVE_CONTRACT}
    )
    if loss_type not in expected_losses:
        raise ValueError("structured feature objective changed")
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
        ).to(device)
        parameter_count = sum(value.numel() for value in model.parameters())
        trainable_count = sum(
            value.numel() for value in model.parameters() if value.requires_grad
        )
        optimizer_parameter_groups(model)
        optimizers = build_optimizers(model, training, device)
        epochs = int(training["epochs"])
        steps_per_epoch = math.ceil(dataset.counts["train"] / batch_size)
        ema_half_life = float(training["weightEma"]["halfLifeEpochs"])
        ema_decay = mean_teacher_ema_decay(ema_half_life, steps_per_epoch)
        ema = {name: value.detach().clone() for name, value in model.state_dict().items()}
        best = {
            "validation-mse": {"score": math.inf, "epoch": -1},
            "validation-correlation": {"score": -math.inf, "epoch": -1},
        }
        last_file = run_root / "checkpoints/last.json"
        start_epoch = 0
        global_step = 0
        if checkpoint_exists(last_file):
            saved = load_torch_checkpoint(last_file, map_location=device, weights_only=False)
            if saved.get("planSha256") != plan_hash \
                    or saved.get("runnerContract") != RUNNER_CONTRACT:
                raise ValueError("structured feature checkpoint contract changed")
            model.load_state_dict(saved["model"])
            ema = saved["emaModel"]
            for optimizer, state in zip(optimizers, saved["optimizers"], strict=True):
                optimizer.load_state_dict(state)
            start_epoch = int(saved["epoch"]) + 1
            global_step = int(saved["globalStep"])
            best = saved["best"]

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
            model.forward_with_auxiliary_loss
            if uses_layer8_auxiliary_loss
            else model
        )
        training_model = training_forward
        compile_scope = "disabled"
        if args.compile_mode != "none":
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

        reporter.emit({
            "event": "training-start",
            "planId": plan["id"],
            "startEpoch": start_epoch,
            "epochs": epochs,
            "parameters": parameter_count,
            "trainableParameters": trainable_count,
            "architecture": architecture,
            "objective": objective_contract(plan),
            "distributionLoss": None,
            "samRho": 0,
            "inputDropoutProbability": 0,
            "embeddingDropoutProbability": 0,
            "adversarialInput": None,
            "adversarialOutput": None,
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
            objective_sum = 0.0
            objective_component_sums: dict[str, float] = {}
            example_count = 0
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
                if uses_layer8_auxiliary_loss:
                    prediction, layer8_gram_identity_loss = \
                        training_model(inputs)
                else:
                    prediction = training_model(inputs)
                    layer8_gram_identity_loss = None
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
                torch.cuda.empty_cache()
            with use_state(model, ema):
                train_evaluation = evaluate(
                    model, dataset, "train", batch_size=evaluation_batch_size,
                    device=device,
                    limit=smoke_limit if smoke_limit is not None else int(
                        training["epochTrainEvaluationExamples"]
                    ),
                )
                validation_evaluation = evaluate(
                    model, dataset, "validation",
                    batch_size=evaluation_batch_size, device=device,
                    limit=smoke_limit,
                )
            train_metrics = train_evaluation["nextReturn"]
            validation_metrics = validation_evaluation["nextReturn"]
            train_feature_state = train_evaluation["featureState"]
            validation_feature_state = validation_evaluation["featureState"]
            candidates = {
                "validation-mse": float(
                    validation_feature_state["normalizedMse"]
                ),
                "validation-correlation": float(
                    validation_feature_state["correlation"]
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
                "onlineObjective": objective_sum / example_count,
                "onlineObjectiveComponents": {
                    name: value / example_count
                    for name, value in objective_component_sums.items()
                },
                "objective": objective_contract(plan),
                "checkpointSelectionScope": FEATURE_STATE_EVALUATION_SCOPE,
                "bestFeatureStateValidationMse": best["validation-mse"]["score"],
                "bestFeatureStateValidationCorrelation": (
                    best["validation-correlation"]["score"]
                ),
                "parameterCount": parameter_count,
            }
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
            evaluations = {
                split: evaluate(
                    model, dataset, split, batch_size=evaluation_batch_size,
                    device=device,
                )
                for split in ("train", "validation", "test")
            }
            policies[f"best-{policy}"] = {
                "epoch": int(selected["epoch"]),
                "selectionScore": float(selected["score"]),
                "selectionMetric": (
                    "featureState.correlation" if policy.endswith("correlation")
                    else "featureState.normalizedMse"
                ),
                "checkpointPolicy": f"best-{policy}",
                **comparable_policy_metrics(evaluations),
            }
        last = load_torch_checkpoint(last_file, map_location=device, weights_only=False)
        model.load_state_dict(last["emaModel"])
        last_evaluations = {
            split: evaluate(
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
            **comparable_policy_metrics(last_evaluations),
        }
        completed_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        artifact, result = persist_structured_feature_evaluation(
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
                "bestTest": result["test"],
                "lastTest": artifact["policies"]["last"]["test"],
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
