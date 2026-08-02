from __future__ import annotations

import argparse
from datetime import date, timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Iterator

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn.utils import clip_grad_norm_

from future_price_feature_screen import (
    BASE_CHANNEL_NAMES,
    BASE_COMPONENT_CONTRACT,
)
from future_price_resolution_screen import (
    SOURCE_FUTURE_MINUTES,
    ResolutionNormalization,
    ResolutionSpec,
    architecture_contract,
    build_resolution_examples,
    build_resolution_predictor,
    parameter_count,
)
from train_future_price_feature_screen import (
    BASE_ROWS_PER_DAY,
    BaseComponentCache,
    prepare_base_components,
)
from train_future_price_predictor import (
    HOUR_MS,
    MINUTE_MS,
    SECOND_MS,
    JsonReporter,
    PairShard,
    atomic_json,
    compact_pair_rows,
    corpus_fingerprint,
    count_examples,
    iso_now,
    restore_random_states,
    source_pair_shards,
    validate_predictor_split_disjointness,
    validate_source_manifest,
)
from trading_storage import (
    checkpoint_exists,
    load_torch_checkpoint,
    require_under,
    save_torch_checkpoint,
    training_storage_layout,
)


RESOLUTION_CORPUS_CONTRACT = (
    "immutable-decoder-assignments-shifted-to-resolution-target-end-with-"
    "history-plus-horizon-cross-split-purge-v1"
)


def _previous_date(value: str) -> str:
    return (date.fromisoformat(value) - timedelta(days=1)).isoformat()


def resolution_pair_shards(
    source_manifest: dict,
    spec: ResolutionSpec,
) -> list[PairShard]:
    """Express each immutable assignment at this target's actual end time."""
    target_end_shift = (
        SOURCE_FUTURE_MINUTES - spec.target_minutes
    ) * MINUTE_MS
    return [
        PairShard(
            split=shard.split,
            prediction_time_start=shard.prediction_time_start - target_end_shift,
            count=shard.count,
            future_date=shard.future_date,
            history_date=shard.history_date,
            future_row_offset=shard.future_row_offset,
            history_row_offset=shard.history_row_offset,
        )
        for shard in source_pair_shards(source_manifest)
    ]


def select_resolution_segments(
    source_manifest: dict,
    spec: ResolutionSpec,
) -> dict[str, list[PairShard]]:
    example_span_ms = spec.example_span_minutes * MINUTE_MS
    selected: dict[str, list[PairShard]] = {"train": [], "validation": []}
    selected_end: dict[str, int | None] = {
        "train": None,
        "validation": None,
        "test": None,
    }
    previous_end: int | None = None
    for shard in resolution_pair_shards(source_manifest, spec):
        if previous_end is not None and shard.prediction_time_start <= previous_end:
            raise ValueError("resolution assignment timestamps overlap")
        local_offset = 0
        other_ends = tuple(
            end
            for split, end in selected_end.items()
            if split != shard.split and end is not None
        )
        if other_ends:
            earliest = max(other_ends) + example_span_ms + SECOND_MS
            if shard.prediction_time_start < earliest:
                local_offset = math.ceil(
                    (earliest - shard.prediction_time_start) / SECOND_MS
                )
        if local_offset < shard.count:
            chosen = shard.shifted(local_offset) if local_offset else shard
            selected_end[shard.split] = chosen.prediction_time_end
            if shard.split in selected:
                selected[shard.split].append(chosen)
        previous_end = shard.prediction_time_end
    if not selected["train"] or not selected["validation"]:
        raise ValueError("resolution train and validation splits must be non-empty")
    validate_predictor_split_disjointness(
        selected,
        example_span_ms=example_span_ms,
    )
    return selected


ResolutionBatch = tuple[Tensor, Tensor, Tensor]


class ResolutionDataset:
    def __init__(
        self,
        segments: dict[str, list[PairShard]],
        component_files: dict[str, Path],
        spec: ResolutionSpec,
    ) -> None:
        self.segments = segments
        self.component_files = component_files
        self.spec = spec

    def logical_count(self, split: str) -> int:
        return sum(shard.count for shard in self.segments[split])

    def compact_count(self, split: str) -> int:
        return sum(
            compact_pair_rows(
                shard.history_row_offset,
                shard.future_row_offset,
                shard.count,
            )[2].shape[0]
            for shard in self.segments[split]
        )

    def _backward_windows(
        self,
        cache: BaseComponentCache,
        component_date: str,
        completed_rows: np.ndarray,
        length: int,
    ) -> np.ndarray:
        current = cache.load(self.component_files[component_date])
        previous = cache.load(self.component_files[_previous_date(component_date)])
        series = np.concatenate((previous, current), axis=0)
        end = BASE_ROWS_PER_DAY + completed_rows - 1
        indices = end[:, None] - np.arange(length - 1, -1, -1, dtype=np.int64)
        if int(indices.min()) < 0 or int(indices.max()) >= series.shape[0]:
            raise IndexError("resolution window escapes adjacent UTC components")
        return np.asarray(series[indices], dtype=np.float32)

    def iter_batches(
        self,
        split: str,
        batch_size: int,
        *,
        shuffle: bool,
        seed: int,
    ) -> Iterator[ResolutionBatch]:
        if batch_size < 1:
            raise ValueError("resolution batch size must be positive")
        generator = np.random.default_rng(seed)
        shards = list(self.segments[split])
        if shuffle:
            generator.shuffle(shards)
        cache = BaseComponentCache()
        pending_history: list[np.ndarray] = []
        pending_target: list[np.ndarray] = []
        pending_weights: list[np.ndarray] = []
        pending_count = 0

        def flush() -> ResolutionBatch:
            nonlocal pending_count
            history = np.concatenate(pending_history, axis=0).astype(
                np.float32,
                copy=False,
            )
            target = np.concatenate(pending_target, axis=0).astype(
                np.float32,
                copy=False,
            )
            weights = np.concatenate(pending_weights).astype(np.float32, copy=False)
            pending_history.clear()
            pending_target.clear()
            pending_weights.clear()
            pending_count = 0
            return (
                torch.from_numpy(history),
                torch.from_numpy(target),
                torch.from_numpy(weights),
            )

        for shard in shards:
            history_rows, future_rows, weights = compact_pair_rows(
                shard.history_row_offset,
                shard.future_row_offset,
                shard.count,
            )
            history_minute = self._backward_windows(
                cache,
                shard.history_date,
                history_rows,
                self.spec.history_minutes,
            )[:, :, 0]
            assigned_future_hour = self._backward_windows(
                cache,
                shard.future_date,
                future_rows,
                SOURCE_FUTURE_MINUTES,
            )[:, :, 0]
            history, target = build_resolution_examples(
                history_minute,
                assigned_future_hour,
                self.spec,
            )
            if shuffle:
                order = generator.permutation(weights.shape[0])
                history = history[order]
                target = target[order]
                weights = weights[order]
            offset = 0
            while offset < weights.shape[0]:
                take = min(batch_size - pending_count, weights.shape[0] - offset)
                end = offset + take
                pending_history.append(history[offset:end])
                pending_target.append(target[offset:end])
                pending_weights.append(weights[offset:end])
                pending_count += take
                offset = end
                if pending_count == batch_size:
                    yield flush()
        if pending_count:
            yield flush()


def resolution_data_fingerprint(
    timestamp_fingerprint: str,
    spec: ResolutionSpec,
) -> str:
    payload = {
        "timestampCorpusFingerprint": timestamp_fingerprint,
        "baseComponentContract": BASE_COMPONENT_CONTRACT,
        "resolutionContract": spec.contract,
        "target": "immediate-next-aggregated-completed-close-log-returns-v1",
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def compute_training_normalization(
    dataset: ResolutionDataset,
    cache_file: Path,
    *,
    fingerprint: str,
    batch_size: int,
    reporter: JsonReporter,
) -> ResolutionNormalization:
    expected_count = dataset.logical_count("train")
    if cache_file.is_file():
        with np.load(cache_file, allow_pickle=False) as cached:
            if str(cached["datasetFingerprint"]) != fingerprint \
                    or int(cached["count"]) != expected_count:
                raise ValueError("resolution normalization cache is stale")
            normalization = ResolutionNormalization(
                torch.from_numpy(cached["inputMean"].astype(np.float32)),
                torch.from_numpy(cached["inputStd"].astype(np.float32)),
                torch.from_numpy(cached["targetMean"].astype(np.float32)),
                torch.from_numpy(cached["targetStd"].astype(np.float32)),
            )
        normalization.validate(dataset.spec)
        reporter.emit({
            "event": "training-statistics-cache",
            "hit": True,
            "examples": expected_count,
            "file": str(cache_file),
        })
        return normalization
    input_sum = np.zeros(dataset.spec.history_steps, dtype=np.float64)
    input_square_sum = np.zeros_like(input_sum)
    target_sum = np.zeros(dataset.spec.target_steps, dtype=np.float64)
    target_square_sum = np.zeros_like(target_sum)
    total = 0.0
    for history, target, weights in dataset.iter_batches(
        "train",
        batch_size,
        shuffle=False,
        seed=0,
    ):
        x = history.numpy().astype(np.float64, copy=False)
        y = target.numpy().astype(np.float64, copy=False)
        w = weights.numpy().astype(np.float64, copy=False)
        input_sum += np.einsum("i,ij->j", w, x)
        input_square_sum += np.einsum("i,ij->j", w, np.square(x))
        target_sum += np.einsum("i,ij->j", w, y)
        target_square_sum += np.einsum("i,ij->j", w, np.square(y))
        total += float(w.sum())
    if int(total) != expected_count:
        raise RuntimeError("resolution normalization missed training examples")

    def moments(total_sum: np.ndarray, square_sum: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mean = total_sum / total
        variance = np.maximum(1e-12, square_sum / total - mean * mean)
        return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)

    input_mean, input_std = moments(input_sum, input_square_sum)
    target_mean, target_std = moments(target_sum, target_square_sum)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_file.with_suffix(cache_file.suffix + ".tmp")
    with temporary.open("wb") as output:
        np.savez(
            output,
            datasetFingerprint=np.asarray(fingerprint),
            count=np.asarray(expected_count, dtype=np.int64),
            inputMean=input_mean,
            inputStd=input_std,
            targetMean=target_mean,
            targetStd=target_std,
        )
    os.replace(temporary, cache_file)
    normalization = ResolutionNormalization(
        torch.from_numpy(input_mean),
        torch.from_numpy(input_std),
        torch.from_numpy(target_mean),
        torch.from_numpy(target_std),
    )
    normalization.validate(dataset.spec)
    reporter.emit({
        "event": "training-statistics-cache",
        "hit": False,
        "examples": expected_count,
        "file": str(cache_file),
    })
    return normalization


def _normalization_json(normalization: ResolutionNormalization) -> dict:
    return {
        "inputMean": normalization.input_mean.tolist(),
        "inputStd": normalization.input_std.tolist(),
        "targetMean": normalization.target_mean.tolist(),
        "targetStd": normalization.target_std.tolist(),
    }


def normalization_fingerprint(normalization: ResolutionNormalization) -> str:
    digest = hashlib.sha256()
    for value in (
        normalization.input_mean,
        normalization.input_std,
        normalization.target_mean,
        normalization.target_std,
    ):
        array = value.detach().cpu().numpy().astype("<f4", copy=False)
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def dummy_normalization(spec: ResolutionSpec) -> ResolutionNormalization:
    return ResolutionNormalization(
        torch.zeros(spec.history_steps),
        torch.ones(spec.history_steps),
        torch.zeros(spec.target_steps),
        torch.ones(spec.target_steps),
    )


def normalized_resolution_loss(
    prediction: Tensor,
    target: Tensor,
    weights: Tensor,
    target_std: Tensor,
    *,
    objective: str,
    huber_delta: float,
) -> Tensor:
    if prediction.shape != target.shape \
            or prediction.ndim != 2 \
            or target_std.shape != (prediction.shape[-1],):
        raise ValueError("resolution loss tensors are misaligned")
    error = (prediction - target) / target_std
    if objective == "mse":
        elements = error.square()
    elif objective == "huber":
        if huber_delta <= 0:
            raise ValueError("resolution Huber delta must be positive")
        elements = torch.nn.functional.huber_loss(
            error,
            torch.zeros_like(error),
            reduction="none",
            delta=huber_delta,
        )
    else:
        raise ValueError(f"unsupported resolution objective: {objective}")
    expanded_weights = weights.to(dtype=elements.dtype).unsqueeze(-1)
    return (elements * expanded_weights).sum() / (
        expanded_weights.sum() * prediction.shape[-1]
    )


class ResolutionMetricAccumulator:
    def __init__(
        self,
        normalization: ResolutionNormalization,
        *,
        huber_delta: float,
        target_steps: int,
        device: torch.device,
    ) -> None:
        self.huber_delta = float(huber_delta)
        self.target_steps = int(target_steps)
        self.target_mean = normalization.target_mean.to(
            device=device,
            dtype=torch.float64,
        )
        self.target_std = normalization.target_std.to(
            device=device,
            dtype=torch.float64,
        )
        self.weight = torch.zeros((), dtype=torch.float64, device=device)
        self.squared = torch.zeros(target_steps, dtype=torch.float64, device=device)
        self.raw_huber = torch.zeros_like(self.squared)
        self.absolute = torch.zeros_like(self.squared)
        self.normalized_squared = torch.zeros_like(self.squared)
        self.normalized_huber = torch.zeros_like(self.squared)
        self.direction = torch.zeros_like(self.squared)
        self.cumulative_squared = torch.zeros_like(self.squared)
        self.prediction_sum = torch.zeros_like(self.squared)
        self.target_sum = torch.zeros_like(self.squared)
        self.prediction_square_sum = torch.zeros_like(self.squared)
        self.target_square_sum = torch.zeros_like(self.squared)
        self.product_sum = torch.zeros_like(self.squared)
        self.endpoint_squared = torch.zeros((), dtype=torch.float64, device=device)
        self.endpoint_direction = torch.zeros_like(self.endpoint_squared)
        self.endpoint_prediction_sum = torch.zeros_like(self.endpoint_squared)
        self.endpoint_target_sum = torch.zeros_like(self.endpoint_squared)
        self.endpoint_prediction_square_sum = torch.zeros_like(self.endpoint_squared)
        self.endpoint_target_square_sum = torch.zeros_like(self.endpoint_squared)
        self.endpoint_product_sum = torch.zeros_like(self.endpoint_squared)

    def add(self, prediction: Tensor, target: Tensor, weights: Tensor) -> None:
        if prediction.shape != target.shape \
                or prediction.shape[-1] != self.target_steps:
            raise ValueError("resolution metrics are misaligned")
        prediction = prediction.detach().to(dtype=torch.float64)
        target = target.detach().to(dtype=torch.float64)
        weights = weights.detach().to(dtype=torch.float64).unsqueeze(-1)
        error = prediction - target
        absolute = error.abs()
        raw_huber = torch.where(
            absolute <= self.huber_delta,
            0.5 * error.square(),
            self.huber_delta * (absolute - 0.5 * self.huber_delta),
        )
        normalized = error / self.target_std
        normalized_absolute = normalized.abs()
        normalized_huber = torch.where(
            normalized_absolute <= self.huber_delta,
            0.5 * normalized.square(),
            self.huber_delta * (
                normalized_absolute - 0.5 * self.huber_delta
            ),
        )
        self.weight += weights.sum()
        self.squared += (error.square() * weights).sum(dim=0)
        self.raw_huber += (raw_huber * weights).sum(dim=0)
        self.absolute += (absolute * weights).sum(dim=0)
        self.normalized_squared += (normalized.square() * weights).sum(dim=0)
        self.normalized_huber += (normalized_huber * weights).sum(dim=0)
        self.direction += (
            (torch.sign(prediction) == torch.sign(target)).to(torch.float64)
            * weights
        ).sum(dim=0)
        cumulative_error = error.cumsum(dim=-1)
        self.cumulative_squared += (
            cumulative_error.square() * weights
        ).sum(dim=0)
        self.prediction_sum += (prediction * weights).sum(dim=0)
        self.target_sum += (target * weights).sum(dim=0)
        self.prediction_square_sum += (prediction.square() * weights).sum(dim=0)
        self.target_square_sum += (target.square() * weights).sum(dim=0)
        self.product_sum += (prediction * target * weights).sum(dim=0)
        endpoint_prediction = prediction.sum(dim=-1)
        endpoint_target = target.sum(dim=-1)
        endpoint_error = endpoint_prediction - endpoint_target
        flat_weights = weights.squeeze(-1)
        self.endpoint_squared += (endpoint_error.square() * flat_weights).sum()
        self.endpoint_direction += (
            (torch.sign(endpoint_prediction) == torch.sign(endpoint_target)).to(
                torch.float64
            ) * flat_weights
        ).sum()
        self.endpoint_prediction_sum += (endpoint_prediction * flat_weights).sum()
        self.endpoint_target_sum += (endpoint_target * flat_weights).sum()
        self.endpoint_prediction_square_sum += (
            endpoint_prediction.square() * flat_weights
        ).sum()
        self.endpoint_target_square_sum += (
            endpoint_target.square() * flat_weights
        ).sum()
        self.endpoint_product_sum += (
            endpoint_prediction * endpoint_target * flat_weights
        ).sum()

    def result(self) -> dict:
        if float(self.weight) <= 0:
            raise RuntimeError("cannot finalize empty resolution metrics")
        weight = self.weight
        raw_mse = self.squared / weight
        raw_huber = self.raw_huber / weight
        raw_mae = self.absolute / weight
        normalized_mse = self.normalized_squared / weight
        normalized_huber = self.normalized_huber / weight
        cumulative_mse = self.cumulative_squared / weight
        prediction_mean = self.prediction_sum / weight
        target_mean = self.target_sum / weight
        prediction_variance = (
            self.prediction_square_sum / weight - prediction_mean.square()
        ).clamp_min(0)
        target_variance = (
            self.target_square_sum / weight - target_mean.square()
        ).clamp_min(0)
        covariance = self.product_sum / weight - prediction_mean * target_mean
        valid_correlation = (
            (prediction_variance > 1e-30) & (target_variance > 1e-30)
        )
        correlation = torch.where(
            valid_correlation,
            covariance / (
                prediction_variance.sqrt() * target_variance.sqrt()
            ).clamp_min(1e-30),
            torch.zeros_like(covariance),
        )
        scale_ratio = prediction_variance.sqrt() / target_variance.sqrt().clamp_min(
            1e-30
        )
        endpoint_prediction_mean = self.endpoint_prediction_sum / weight
        endpoint_target_mean = self.endpoint_target_sum / weight
        endpoint_prediction_variance = (
            self.endpoint_prediction_square_sum / weight
            - endpoint_prediction_mean.square()
        ).clamp_min(0)
        endpoint_target_variance = (
            self.endpoint_target_square_sum / weight
            - endpoint_target_mean.square()
        ).clamp_min(0)
        endpoint_covariance = (
            self.endpoint_product_sum / weight
            - endpoint_prediction_mean * endpoint_target_mean
        )
        endpoint_correlation = torch.where(
            (endpoint_prediction_variance > 1e-30)
            & (endpoint_target_variance > 1e-30),
            endpoint_covariance / (
                endpoint_prediction_variance.sqrt()
                * endpoint_target_variance.sqrt()
            ).clamp_min(1e-30),
            torch.zeros_like(endpoint_covariance),
        )
        endpoint_scale_ratio = (
            endpoint_prediction_variance.sqrt()
            / endpoint_target_variance.sqrt().clamp_min(1e-30)
        )

        def values(value: Tensor) -> list[float]:
            return value.detach().cpu().tolist()

        return {
            "examples": int(round(float(weight))),
            "rawMse": float(raw_mse.mean()),
            "rawRmse": math.sqrt(float(raw_mse.mean())),
            "rawHuber": float(raw_huber.mean()),
            "rawMae": float(raw_mae.mean()),
            "normalizedMse": float(normalized_mse.mean()),
            "normalizedHuber": float(normalized_huber.mean()),
            "directionAccuracy": float((self.direction / weight).mean()),
            "cumulativePathRmse": math.sqrt(float(cumulative_mse.mean())),
            "endpointRmse": math.sqrt(float(cumulative_mse[-1])),
            "endpointRawMse": float(self.endpoint_squared / weight),
            "endpointDirectionAccuracy": float(self.endpoint_direction / weight),
            "endpointCorrelation": float(endpoint_correlation),
            "endpointPredictionScaleRatio": float(endpoint_scale_ratio),
            "meanHorizonCorrelation": float(correlation.mean()),
            "meanPredictionScaleRatio": float(scale_ratio.mean()),
            "rawMseByHorizon": values(raw_mse),
            "rawHuberByHorizon": values(raw_huber),
            "rawMaeByHorizon": values(raw_mae),
            "normalizedMseByHorizon": values(normalized_mse),
            "normalizedHuberByHorizon": values(normalized_huber),
            "directionAccuracyByHorizon": values(self.direction / weight),
            "correlationByHorizon": values(correlation),
            "predictionScaleRatioByHorizon": values(scale_ratio),
            "cumulativeMseByHorizon": values(cumulative_mse),
        }


RESOLUTION_BASELINE_CONTRACT = (
    "full-purged-validation-zero-return-and-training-horizon-mean-"
    "resolution-forecast-metrics-v2"
)


def compute_validation_baselines(
    dataset: ResolutionDataset,
    normalization: ResolutionNormalization,
    cache_file: Path,
    *,
    data_fingerprint: str,
    batch_size: int,
    objective: str,
    huber_delta: float,
    reporter: JsonReporter,
) -> dict:
    expected_count = dataset.logical_count("validation")
    cache_key = hashlib.sha256(json.dumps({
        "contract": RESOLUTION_BASELINE_CONTRACT,
        "datasetFingerprint": data_fingerprint,
        "normalizationFingerprint": normalization_fingerprint(normalization),
        "objective": objective,
        "huberDelta": float(huber_delta),
    }, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if cache_file.is_file():
        cached = json.loads(cache_file.read_text(encoding="utf-8"))
        if cached.get("cacheKey") != cache_key \
                or cached.get("contract") != RESOLUTION_BASELINE_CONTRACT:
            raise ValueError("resolution validation baseline cache is stale")
        baselines = cached.get("baselines", {})
        if set(baselines) != {"zeroReturn", "trainingMean"} \
                or any(
                    int(value.get("examples", -1)) != expected_count
                    for value in baselines.values()
                ):
            raise ValueError("resolution validation baselines are invalid")
        reporter.emit({
            "event": "validation-baselines-cache",
            "hit": True,
            "examples": expected_count,
            "file": str(cache_file),
            "fingerprint": cache_key,
        })
        return {
            "contract": RESOLUTION_BASELINE_CONTRACT,
            "fingerprint": cache_key,
            **baselines,
        }
    zero_metrics = ResolutionMetricAccumulator(
        normalization,
        huber_delta=huber_delta,
        target_steps=dataset.spec.target_steps,
        device=torch.device("cpu"),
    )
    mean_metrics = ResolutionMetricAccumulator(
        normalization,
        huber_delta=huber_delta,
        target_steps=dataset.spec.target_steps,
        device=torch.device("cpu"),
    )
    for _, target, weights in dataset.iter_batches(
        "validation",
        batch_size,
        shuffle=False,
        seed=0,
    ):
        zero_metrics.add(torch.zeros_like(target), target, weights)
        mean_metrics.add(
            normalization.target_mean.unsqueeze(0).expand_as(target),
            target,
            weights,
        )
    baselines = {
        "zeroReturn": zero_metrics.result(),
        "trainingMean": mean_metrics.result(),
    }
    selection_key = "normalizedHuber" if objective == "huber" else "normalizedMse"
    for value in baselines.values():
        value["loss"] = value[selection_key]
    payload = {
        "version": 1,
        "contract": RESOLUTION_BASELINE_CONTRACT,
        "cacheKey": cache_key,
        "datasetFingerprint": data_fingerprint,
        "normalizationFingerprint": normalization_fingerprint(normalization),
        "objective": objective,
        "huberDelta": float(huber_delta),
        "baselines": baselines,
    }
    atomic_json(payload, cache_file)
    reporter.emit({
        "event": "validation-baselines-cache",
        "hit": False,
        "examples": expected_count,
        "file": str(cache_file),
        "fingerprint": cache_key,
    })
    return {
        "contract": RESOLUTION_BASELINE_CONTRACT,
        "fingerprint": cache_key,
        **baselines,
    }


def evaluate(
    model: nn.Module,
    dataset: ResolutionDataset,
    normalization: ResolutionNormalization,
    *,
    batch_size: int,
    device: torch.device,
    amp_dtype: torch.dtype,
    objective: str,
    huber_delta: float,
    baselines: dict,
) -> dict:
    model.eval()
    metrics = ResolutionMetricAccumulator(
        normalization,
        huber_delta=huber_delta,
        target_steps=dataset.spec.target_steps,
        device=device,
    )
    with torch.inference_mode():
        for history, target, weights in dataset.iter_batches(
            "validation",
            batch_size,
            shuffle=False,
            seed=0,
        ):
            history = history.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            weights = weights.to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=device.type == "cuda",
            ):
                prediction = model(history)
            metrics.add(prediction, target, weights)
    result = metrics.result()
    result["loss"] = result[
        "normalizedHuber" if objective == "huber" else "normalizedMse"
    ]
    result["baselines"] = baselines
    return result


def resume_contract(
    plan: dict,
    *,
    data_fingerprint: str,
    model_contract: str,
    parameters: int,
    normalization_id: str,
) -> str:
    training = plan["training"]
    payload = {
        "datasetFingerprint": data_fingerprint,
        "modelContract": model_contract,
        "parameterCount": parameters,
        "normalizationFingerprint": normalization_id,
        "batchSize": training["batchSize"],
        "evaluationBatchSize": training["evaluationBatchSize"],
        "objective": training["objective"],
        "huberDelta": training["huberDelta"],
        "selectionMetric": training["selectionMetric"],
        "learningRate": training["learningRate"],
        "optimizer": training["optimizer"],
        "learningRateSchedule": training["learningRateSchedule"],
        "gradientClip": training["gradientClip"],
        "seed": training["seed"],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def train(
    plan: dict,
    dataset: ResolutionDataset,
    normalization: ResolutionNormalization,
    baselines: dict,
    *,
    data_fingerprint: str,
    run_dir: Path,
    reporter: JsonReporter,
    stop_after_epoch: int | None,
) -> None:
    training = plan["training"]
    device = torch.device(training["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA resolution-screen training is unavailable")
    seed = int(training["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    torch.set_float32_matmul_precision("high")
    model_contract = architecture_contract(
        plan["architecture"],
        plan["architectureConfig"],
        dataset.spec,
    )
    model = build_resolution_predictor(
        plan["architecture"],
        plan["architectureConfig"],
        dataset.spec,
        normalization,
    ).to(device)
    parameters = parameter_count(model)
    optimizer_config = training["optimizer"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learningRate"]),
        betas=tuple(float(value) for value in optimizer_config["betas"]),
        eps=float(optimizer_config["epsilon"]),
        weight_decay=float(optimizer_config["weightDecay"]),
        fused=device.type == "cuda",
    )
    schedule = training["learningRateSchedule"]
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(schedule["factor"]),
        patience=int(schedule["patience"]),
        threshold=float(schedule["threshold"]),
        threshold_mode="abs",
        min_lr=float(schedule["minimumLearningRate"]),
    )
    amp_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    active_model = torch.compile(model) if bool(training.get("compile", False)) else model
    contract = resume_contract(
        plan,
        data_fingerprint=data_fingerprint,
        model_contract=model_contract,
        parameters=parameters,
        normalization_id=normalization_fingerprint(normalization),
    )
    last_file = run_dir / "checkpoints" / "last.json"
    best_file = run_dir / "checkpoints" / "best.json"
    start_epoch = 1
    global_step = 0
    stale_epochs = 0
    best_validation = math.inf
    best_epoch = -1
    if checkpoint_exists(last_file):
        checkpoint = load_torch_checkpoint(
            last_file,
            map_location=device,
            weights_only=False,
        )
        if checkpoint.get("resumeContract") != contract:
            raise ValueError("resolution-screen resume checkpoint contract changed")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint["globalStep"])
        stale_epochs = int(checkpoint["staleEpochs"])
        best_validation = float(checkpoint["bestValidation"])
        best_epoch = int(checkpoint["bestEpoch"])
        restore_random_states(checkpoint, device)
        reporter.emit({
            "event": "resume",
            "epoch": start_epoch,
            "bestEpoch": best_epoch,
            "bestValidation": best_validation,
        })
    objective = str(training["objective"])
    huber_delta = float(training["huberDelta"])
    selection_metric = str(training["selectionMetric"])
    reporter.status(
        "training",
        planId=plan["id"],
        resumedFromEpoch=start_epoch - 1,
        bestEpoch=best_epoch,
        bestValidation=(best_validation if math.isfinite(best_validation) else None),
    )
    if stop_after_epoch is not None and start_epoch > stop_after_epoch:
        reporter.status(
            "paused",
            planId=plan["id"],
            epoch=start_epoch - 1,
            bestEpoch=best_epoch,
            bestValidation=best_validation,
            message="Requested epoch boundary was already checkpointed.",
        )
        return
    for epoch in range(start_epoch, int(training["epochs"]) + 1):
        started = time.monotonic()
        active_model.train()
        train_metrics = ResolutionMetricAccumulator(
            normalization,
            huber_delta=huber_delta,
            target_steps=dataset.spec.target_steps,
            device=device,
        )
        for history, target, weights in dataset.iter_batches(
            "train",
            int(training["batchSize"]),
            shuffle=True,
            seed=seed + epoch,
        ):
            history = history.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            weights = weights.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=device.type == "cuda",
            ):
                prediction = active_model(history)
                loss = normalized_resolution_loss(
                    prediction,
                    target,
                    weights,
                    model.target_std,
                    objective=objective,
                    huber_delta=huber_delta,
                )
            loss.backward()
            gradient_norm = clip_grad_norm_(
                model.parameters(),
                float(training["gradientClip"]),
            )
            if not bool(torch.isfinite(gradient_norm)):
                raise FloatingPointError("resolution-screen gradient is non-finite")
            optimizer.step()
            global_step += 1
            train_metrics.add(prediction, target, weights)
        train_result = train_metrics.result()
        train_result["loss"] = train_result[
            "normalizedHuber" if objective == "huber" else "normalizedMse"
        ]
        validation_result = evaluate(
            active_model,
            dataset,
            normalization,
            batch_size=int(training["evaluationBatchSize"]),
            device=device,
            amp_dtype=amp_dtype,
            objective=objective,
            huber_delta=huber_delta,
            baselines=baselines,
        )
        validation_value = float(validation_result[selection_metric])
        if not math.isfinite(validation_value):
            raise FloatingPointError("resolution validation metric is non-finite")
        improved = validation_value < best_validation - float(schedule["threshold"])
        if improved:
            best_validation = validation_value
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
        scheduler.step(validation_value)
        checkpoint = {
            "version": 1,
            "resumeContract": contract,
            "datasetFingerprint": data_fingerprint,
            "resolutionContract": dataset.spec.contract,
            "architectureContract": model_contract,
            "architecture": plan["architecture"],
            "architectureConfig": plan["architectureConfig"],
            "parameterCount": parameters,
            "normalization": _normalization_json(normalization),
            "normalizationFingerprint": normalization_fingerprint(normalization),
            "epoch": epoch,
            "globalStep": global_step,
            "staleEpochs": stale_epochs,
            "bestValidation": best_validation,
            "bestEpoch": best_epoch,
            "selectionMetric": selection_metric,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "training": train_result,
            "validation": validation_result,
            "pythonRandomState": random.getstate(),
            "numpyRandomState": np.random.get_state(),
            "torchRandomState": torch.get_rng_state(),
            "cudaRandomState": (
                torch.cuda.get_rng_state_all() if device.type == "cuda" else None
            ),
        }
        if improved:
            save_torch_checkpoint(
                {
                    key: checkpoint[key]
                    for key in (
                        "version",
                        "resumeContract",
                        "datasetFingerprint",
                        "resolutionContract",
                        "architectureContract",
                        "architecture",
                        "architectureConfig",
                        "parameterCount",
                        "normalization",
                        "normalizationFingerprint",
                        "epoch",
                        "globalStep",
                        "selectionMetric",
                        "model",
                        "validation",
                    )
                },
                best_file,
                metadata={
                    "planId": plan["id"],
                    "epoch": epoch,
                    "selectionValue": validation_value,
                },
            )
        save_torch_checkpoint(
            checkpoint,
            last_file,
            metadata={
                "planId": plan["id"],
                "epoch": epoch,
                "bestEpoch": best_epoch,
                "bestValidation": best_validation,
            },
        )
        event = {
            "event": "epoch",
            "epoch": epoch,
            "globalStep": global_step,
            "seconds": time.monotonic() - started,
            "learningRate": float(optimizer.param_groups[0]["lr"]),
            "improved": improved,
            "bestEpoch": best_epoch,
            "bestValidation": best_validation,
            "training": train_result,
            "validation": validation_result,
        }
        reporter.emit(event)
        reporter.status(
            "training",
            planId=plan["id"],
            epoch=epoch,
            bestEpoch=best_epoch,
            bestValidation=best_validation,
            latest=event,
        )
        if stop_after_epoch is not None and epoch >= stop_after_epoch:
            reporter.status(
                "paused",
                planId=plan["id"],
                epoch=epoch,
                bestEpoch=best_epoch,
                bestValidation=best_validation,
                message="Requested durable epoch boundary reached.",
            )
            return
        if stale_epochs >= int(training["patience"]):
            reporter.status(
                "complete",
                planId=plan["id"],
                epoch=epoch,
                bestEpoch=best_epoch,
                bestValidation=best_validation,
                message="Resolution screen reached validation patience.",
            )
            return
    reporter.status(
        "complete",
        planId=plan["id"],
        epoch=int(training["epochs"]),
        bestEpoch=best_epoch,
        bestValidation=best_validation,
        message="Resolution screen exhausted its configured epochs.",
    )


def validate_plan(plan: dict) -> ResolutionSpec:
    required = (
        "id",
        "corpusId",
        "baseComponentCorpusId",
        "sourceDatasetDir",
        "decoderDatasetDir",
        "datasetDir",
        "runDir",
        "historyDir",
        "resolution",
        "architecture",
        "architectureConfig",
        "training",
    )
    if any(name not in plan or plan[name] in (None, "") for name in required):
        raise ValueError("resolution-screen plan is missing required fields")
    spec = ResolutionSpec.from_config(plan["resolution"])
    if plan["architecture"] not in {"causal_patch_tcn", "patch_tide"}:
        raise ValueError("resolution-screen architecture is unsupported")
    training = plan["training"]
    if training.get("objective") not in {"huber", "mse"} \
            or training.get("selectionMetric") not in {
                "normalizedHuber",
                "normalizedMse",
                "rawMse",
            }:
        raise ValueError("resolution objective or selection metric is invalid")
    for key in ("epochs", "batchSize", "evaluationBatchSize", "patience", "seed"):
        if int(training.get(key, 0)) < 1:
            raise ValueError(f"resolution training {key} must be positive")
    optimizer = training.get("optimizer", {})
    schedule = training.get("learningRateSchedule", {})
    if optimizer.get("type") != "adamw" \
            or len(optimizer.get("betas", ())) != 2 \
            or schedule.get("type") != "reduce-on-validation-plateau" \
            or not 0 < float(schedule.get("factor", 0)) < 1:
        raise ValueError("resolution optimizer configuration is invalid")
    if float(training.get("huberDelta", 0)) <= 0 \
            or float(training.get("learningRate", 0)) <= 0 \
            or float(training.get("gradientClip", 0)) <= 0 \
            or float(optimizer.get("epsilon", 0)) <= 0 \
            or float(schedule.get("threshold", -1)) < 0 \
            or float(schedule.get("minimumLearningRate", 0)) <= 0 \
            or int(schedule.get("patience", -1)) < 0:
        raise ValueError("resolution optimization scales are invalid")
    if training.get("device") not in {"cpu", "cuda"} \
            or training.get("mixedPrecision") != "bfloat16":
        raise ValueError("resolution device/precision contract is invalid")
    build_resolution_predictor(
        plan["architecture"],
        plan["architectureConfig"],
        spec,
        dummy_normalization(spec),
    )
    return spec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Screen easier causal aggregated-close forecast resolutions."
    )
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--stop-after-epoch", type=int)
    return parser.parse_args()


def resolve(repo_root: Path, value: Path) -> Path:
    return value.resolve() if value.is_absolute() else (repo_root / value).resolve()


def main() -> None:
    args = parse_args()
    if args.validate_only and args.prepare_only:
        raise ValueError("choose only one of --validate-only and --prepare-only")
    if args.stop_after_epoch is not None and args.stop_after_epoch < 1:
        raise ValueError("--stop-after-epoch must be positive")
    repo_root = Path(__file__).resolve().parent.parent
    plan_file = resolve(repo_root, args.plan)
    plan = json.loads(plan_file.read_text(encoding="utf-8"))
    spec = validate_plan(plan)
    layout = training_storage_layout(repo_root)
    source_root = require_under(
        resolve(repo_root, Path(plan["sourceDatasetDir"])),
        layout.datasets,
        "sourceDatasetDir",
    )
    decoder_root = require_under(
        resolve(repo_root, Path(plan["decoderDatasetDir"])),
        layout.datasets,
        "decoderDatasetDir",
    )
    dataset_root = require_under(
        resolve(repo_root, Path(plan["datasetDir"])),
        layout.datasets,
        "datasetDir",
    )
    run_dir = require_under(
        resolve(repo_root, Path(plan["runDir"])),
        layout.runs,
        "runDir",
    )
    history_root = require_under(
        resolve(repo_root, Path(plan["historyDir"])),
        repo_root / "data" / "market" / "immutable" / "refs" / "candles",
        "historyDir",
    )
    reporter = JsonReporter(run_dir)
    try:
        source_manifest = json.loads(
            (source_root / "dataset.json").read_text(encoding="utf-8")
        )
        validate_source_manifest(source_manifest)
        decoder_manifest = json.loads(
            (decoder_root / "dataset.json").read_text(encoding="utf-8")
        )
        if int(decoder_manifest.get("crossSplitPurgeMs", 0)) != HOUR_MS:
            raise ValueError("reference decoder corpus contract changed")
        segments = select_resolution_segments(source_manifest, spec)
        counts = count_examples(segments)
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
        timestamp_contract = f"{RESOLUTION_CORPUS_CONTRACT}:{spec.contract}"
        timestamp_fingerprint = corpus_fingerprint(
            segments,
            contract=timestamp_contract,
        )
        data_fingerprint = resolution_data_fingerprint(
            timestamp_fingerprint,
            spec,
        )
        definition = build_resolution_predictor(
            plan["architecture"],
            plan["architectureConfig"],
            spec,
            dummy_normalization(spec),
        )
        model_contract = architecture_contract(
            plan["architecture"],
            plan["architectureConfig"],
            spec,
        )
        validation_event = {
            "event": "validation-complete",
            "planId": plan["id"],
            "resolution": {
                "historyMinutes": spec.history_minutes,
                "historySteps": spec.history_steps,
                "candleMinutes": spec.candle_minutes,
                "targetSteps": spec.target_steps,
                "targetMinutes": spec.target_minutes,
            },
            "counts": counts,
            "compactCounts": compact_counts,
            "timestampCorpusFingerprint": timestamp_fingerprint,
            "datasetFingerprint": data_fingerprint,
            "crossSplitPurgeMs": spec.example_span_minutes * MINUTE_MS,
            "parameterCount": parameter_count(definition),
            "architectureContract": model_contract,
            "computeMatchGroup": plan.get("computeMatchGroup"),
            "heldoutTest": "not selected or read",
        }
        reporter.emit(validation_event)
        if args.validate_only:
            reporter.status(
                "paused",
                planId=plan["id"],
                latest=validation_event,
                message="Resolution plan validated without reading payloads.",
            )
            return
        component_dates = {
            component_date
            for values in segments.values()
            for shard in values
            for component_date in (
                shard.history_date,
                _previous_date(shard.history_date),
                shard.future_date,
                _previous_date(shard.future_date),
            )
        }
        component_root = (
            layout.immutable
            / "refs"
            / "features"
            / "future-price-predictor-minute-base-v1"
            / plan["baseComponentCorpusId"]
        )
        reporter.status(
            "dataset-preparation",
            planId=plan["id"],
            message="Reusing canonical causal completed-minute close returns.",
        )
        component_files = prepare_base_components(
            component_dates,
            component_root=component_root,
            history_root=history_root,
            immutable_root=layout.immutable,
            corpus_id=plan["baseComponentCorpusId"],
            reporter=reporter,
        )
        dataset = ResolutionDataset(segments, component_files, spec)
        normalization = compute_training_normalization(
            dataset,
            dataset_root / "training-resolution-statistics-v1.npz",
            fingerprint=data_fingerprint,
            batch_size=int(plan["training"]["evaluationBatchSize"]),
            reporter=reporter,
        )
        baselines = compute_validation_baselines(
            dataset,
            normalization,
            dataset_root / "validation-resolution-baselines-v2.json",
            data_fingerprint=data_fingerprint,
            batch_size=int(plan["training"]["evaluationBatchSize"]),
            objective=str(plan["training"]["objective"]),
            huber_delta=float(plan["training"]["huberDelta"]),
            reporter=reporter,
        )
        reporter.emit({
            "event": "validation-baselines",
            "planId": plan["id"],
            "baselines": baselines,
        })
        manifest = {
            "version": 1,
            "createdAt": iso_now(),
            "corpusId": plan["corpusId"],
            "sourceDataset": str((source_root / "dataset.json").relative_to(repo_root)),
            "decoderDataset": str((decoder_root / "dataset.json").relative_to(repo_root)),
            "baseComponentCorpusId": plan["baseComponentCorpusId"],
            "baseComponentContract": BASE_COMPONENT_CONTRACT,
            "baseChannels": list(BASE_CHANNEL_NAMES),
            "resolutionContract": spec.contract,
            "historyMinutes": spec.history_minutes,
            "historySteps": spec.history_steps,
            "candleMinutes": spec.candle_minutes,
            "target": "immediate next completed aggregate close log returns",
            "targetMinutes": spec.target_minutes,
            "targetSteps": spec.target_steps,
            "crossSplitPurgeMs": spec.example_span_minutes * MINUTE_MS,
            "counts": counts,
            "compactCounts": compact_counts,
            "timestampCorpusFingerprint": timestamp_fingerprint,
            "datasetFingerprint": data_fingerprint,
            "normalization": {
                "source": "selected training split only",
                "axis": "each aggregate history position and target horizon separately",
                "varianceCorrection": 0,
                "fingerprint": normalization_fingerprint(normalization),
                **_normalization_json(normalization),
            },
            "validationBaselines": baselines,
            "heldoutTest": "not selected, loaded, normalized, evaluated, or exposed",
        }
        atomic_json(manifest, dataset_root / "dataset.json")
        reporter.emit({
            "event": "dataset-complete",
            "datasetFingerprint": data_fingerprint,
            "counts": counts,
            "compactCounts": compact_counts,
            "components": len(component_files),
            "parameters": parameter_count(definition),
        })
        if args.prepare_only:
            reporter.status(
                "paused",
                planId=plan["id"],
                message="Resolution dataset and baselines prepared.",
            )
            return
        train(
            plan,
            dataset,
            normalization,
            baselines,
            data_fingerprint=data_fingerprint,
            run_dir=run_dir,
            reporter=reporter,
            stop_after_epoch=args.stop_after_epoch,
        )
    except KeyboardInterrupt:
        reporter.status(
            "paused",
            planId=plan["id"],
            message="Interrupted; last completed epoch remains resumable.",
        )
        raise
    except Exception as error:
        reporter.status(
            "failed",
            planId=plan["id"],
            error=f"{type(error).__name__}: {error}",
        )
        raise


if __name__ == "__main__":
    main()
