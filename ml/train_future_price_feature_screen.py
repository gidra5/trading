from __future__ import annotations

import argparse
from collections import OrderedDict
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
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
    BASE_CHANNEL_COUNT,
    BASE_CHANNEL_NAMES,
    BASE_COMPONENT_CONTRACT,
    FORECAST_MINUTES,
    HISTORY_MINUTES,
    FeatureNormalization,
    FeatureSpec,
    architecture_contract,
    build_causal_features,
    build_feature_predictor,
    parameter_count,
)
from future_price_forecast_objectives import (
    OBJECTIVE_COMPONENTS,
    ForecastObjectiveStatistics,
    forecast_objective_loss,
    forecast_objective_value_from_metrics,
    normalized_component_errors,
    objective_contract,
    objective_for_epoch,
    target_structure,
    validate_forecast_objective,
)
from future_price_predictor import normalized_forecast_loss
from train_future_price_predictor import (
    HOUR_MS,
    MINUTE_MS,
    MINUTE_SECONDS,
    SECOND_MS,
    DAY_SECONDS,
    ForecastMetricAccumulator,
    JsonReporter,
    PairShard,
    atomic_json,
    compact_pair_rows,
    count_examples,
    corpus_fingerprint,
    iso_now,
    restore_random_states,
    select_pair_segments,
    validate_predictor_split_disjointness,
    validate_source_manifest,
)
from trading_storage import (
    checkpoint_exists,
    is_storage_reference,
    load_torch_checkpoint,
    read_shard_payload,
    read_shard_array,
    require_under,
    resolve_shard,
    save_torch_checkpoint,
    training_storage_layout,
    write_shard_payload,
)


BASE_ROWS_PER_DAY = 1_440
EXAMPLE_SPAN_MS = 7 * HOUR_MS
TIMESTAMP_CORPUS_CONTRACT = (
    "decoder-delay-3600s-train-validation-assignments-with-7h-past-future-"
    "cross-split-purge-compact-minute-multiplicity-v1"
)


def _previous_date(value: str) -> str:
    return (date.fromisoformat(value) - timedelta(days=1)).isoformat()


def aggregate_minute_base(
    previous_close: np.ndarray,
    current_open: np.ndarray,
    current_high: np.ndarray,
    current_low: np.ndarray,
    current_close: np.ndarray,
    current_volume: np.ndarray,
) -> np.ndarray:
    arrays = (
        previous_close,
        current_open,
        current_high,
        current_low,
        current_close,
        current_volume,
    )
    if any(value.shape != (DAY_SECONDS,) for value in arrays):
        raise ValueError("minute aggregation requires complete one-second days")
    if any(not np.isfinite(value).all() for value in arrays) \
            or any(bool((value <= 0).any()) for value in (
                previous_close,
                current_open,
                current_high,
                current_low,
                current_close,
            )) \
            or bool((current_volume < 0).any()):
        raise ValueError("one-second OHLCV inputs are invalid")
    minute_open = current_open.reshape(BASE_ROWS_PER_DAY, MINUTE_SECONDS)[:, 0]
    minute_high = current_high.reshape(
        BASE_ROWS_PER_DAY,
        MINUTE_SECONDS,
    ).max(axis=1)
    minute_low = current_low.reshape(
        BASE_ROWS_PER_DAY,
        MINUTE_SECONDS,
    ).min(axis=1)
    minute_close = current_close.reshape(
        BASE_ROWS_PER_DAY,
        MINUTE_SECONDS,
    )[:, -1]
    minute_volume = current_volume.reshape(
        BASE_ROWS_PER_DAY,
        MINUTE_SECONDS,
    ).sum(axis=1, dtype=np.float64)
    prior_close = np.concatenate((
        previous_close[-1:],
        minute_close[:-1],
    ))
    body_high = np.maximum(minute_open, minute_close)
    body_low = np.minimum(minute_open, minute_close)
    if bool((minute_high + 1e-12 < body_high).any()) \
            or bool((minute_low - 1e-12 > body_low).any()):
        raise ValueError("minute OHLC extrema do not contain their bodies")
    result = np.stack((
        np.log(minute_close / prior_close),
        np.log(minute_open / prior_close),
        np.log(np.maximum(minute_high, body_high) / body_high),
        np.log(body_low / np.minimum(minute_low, body_low)),
        np.log1p(minute_volume),
        (minute_volume == 0).astype(np.float64),
    ), axis=-1).astype(np.float32)
    if result.shape != (BASE_ROWS_PER_DAY, BASE_CHANNEL_COUNT) \
            or not np.isfinite(result).all():
        raise RuntimeError("completed-minute base component is invalid")
    return result


def valid_base_component(file: Path) -> bool:
    if not file.is_file() or not is_storage_reference(file):
        return False
    try:
        shard = resolve_shard(file)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    layout = shard.reference.get("layout", {})
    return shard.axis.count == BASE_ROWS_PER_DAY \
        and int(shard.reference["object"]["uncompressedBytes"]) \
        == BASE_ROWS_PER_DAY * BASE_CHANNEL_COUNT * np.dtype("<f4").itemsize \
        and layout.get("dtype") == "float32-le" \
        and tuple(layout.get("columns", ())) == BASE_CHANNEL_NAMES \
        and shard.reference.get("metadata", {}).get("contract") \
        == BASE_COMPONENT_CONTRACT


def read_dense_candle_columns(
    reference_file: Path,
    component_date: str,
    names: tuple[str, ...],
) -> dict[str, np.ndarray]:
    shard, payload = read_shard_payload(reference_file)
    layout = shard.reference.get("layout", {})
    expected_start = int(datetime.fromisoformat(component_date).replace(
        tzinfo=timezone.utc,
    ).timestamp() * 1_000)
    constants = layout.get("constants", {})
    if shard.axis.start != expected_start \
            or shard.axis.step != SECOND_MS \
            or shard.axis.count != DAY_SECONDS \
            or shard.axis.unit != "unix-ms" \
            or layout.get("encoding") != "candle-columnar-delta-v1" \
            or tuple(column.get("name") for column in layout.get("columns", ())) \
            != ("open", "high", "low", "close", "volume") \
            or constants.get("symbol") != "BTCUSDT" \
            or constants.get("interval") != "1s" \
            or constants.get("closed") is not True \
            or constants.get("closeTimeOffsetMs") != 999 \
            or layout.get("timeJumps", ()) \
            or layout.get("closedOverrides", ()):
        raise ValueError(f"one-second candle axis is not dense: {reference_file}")
    for override in layout.get("closeTimeOffsetOverrides", ()):
        if not 0 <= int(override.get("index", -1)) < DAY_SECONDS \
                or not 0 <= int(override.get("offsetMs", -1)) <= 999:
            raise ValueError(f"invalid candle close-time override: {reference_file}")
    columns = {
        str(column["name"]): column
        for column in layout["columns"]
    }
    result: dict[str, np.ndarray] = {}
    for name in names:
        column = columns[name]
        start = int(column["offset"])
        end = start + int(column["bytes"])
        encoded = payload[start:end]
        if column["encoding"] == "float64-le":
            values = np.frombuffer(encoded, dtype="<f8")
            if values.shape != (DAY_SECONDS,):
                raise ValueError(f"truncated candle column: {reference_file}")
            result[name] = values
            continue
        if column["encoding"] != "scaled-delta-zigzag-varint":
            raise ValueError(f"unsupported candle column: {reference_file}")
        scale = int(column["scale"])
        values = np.empty(DAY_SECONDS, dtype=np.float64)
        offset = 0
        previous = 0
        for index in range(DAY_SECONDS):
            encoded_value = 0
            shift = 0
            while True:
                if offset >= len(encoded) or shift > 70:
                    raise ValueError(f"truncated candle varint: {reference_file}")
                byte = encoded[offset]
                offset += 1
                encoded_value |= (byte & 0x7f) << shift
                if byte & 0x80 == 0:
                    break
                shift += 7
            delta = (
                encoded_value // 2
                if encoded_value % 2 == 0
                else -(encoded_value + 1) // 2
            )
            previous += delta
            values[index] = previous / scale
        if offset != len(encoded):
            raise ValueError(f"candle column has trailing bytes: {reference_file}")
        result[name] = values
    return result


class CandleColumnCache:
    def __init__(self, history_root: Path) -> None:
        self.history_root = history_root
        self.close: OrderedDict[str, np.ndarray] = OrderedDict()

    def close_column(self, component_date: str) -> np.ndarray:
        cached = self.close.pop(component_date, None)
        if cached is not None:
            self.close[component_date] = cached
            return cached
        file = self.history_root / f"{component_date}.json"
        if not file.is_file():
            raise FileNotFoundError(f"missing one-second candle history: {file}")
        value = read_dense_candle_columns(
            file,
            component_date,
            ("close",),
        )["close"]
        self.close[component_date] = value
        while len(self.close) > 3:
            self.close.popitem(last=False)
        return value

    def day(self, component_date: str) -> dict[str, np.ndarray]:
        file = self.history_root / f"{component_date}.json"
        if not file.is_file():
            raise FileNotFoundError(f"missing one-second candle history: {file}")
        values = read_dense_candle_columns(
            file,
            component_date,
            ("open", "high", "low", "close", "volume"),
        )
        self.close[component_date] = values["close"]
        while len(self.close) > 3:
            self.close.popitem(last=False)
        return values


def prepare_base_components(
    required_dates: set[str],
    *,
    component_root: Path,
    history_root: Path,
    immutable_root: Path,
    corpus_id: str,
    reporter: JsonReporter,
) -> dict[str, Path]:
    component_root.mkdir(parents=True, exist_ok=True)
    result: dict[str, Path] = {}
    missing: list[str] = []
    for component_date in sorted(required_dates):
        file = component_root / f"{component_date}.json"
        if valid_base_component(file):
            result[component_date] = file
        else:
            missing.append(component_date)
    cache = CandleColumnCache(history_root)
    for index, component_date in enumerate(missing, start=1):
        current = cache.day(component_date)
        base = aggregate_minute_base(
            cache.close_column(_previous_date(component_date)),
            current["open"],
            current["high"],
            current["low"],
            current["close"],
            current["volume"],
        )
        reference = write_shard_payload(
            immutable_root,
            "features/future-price-predictor-minute-base-v1",
            f"{corpus_id}/{component_date}",
            base.astype("<f4", copy=False).tobytes(),
            sequence={
                "start": int(datetime.fromisoformat(component_date).replace(
                    tzinfo=timezone.utc,
                ).timestamp() * 1_000) + MINUTE_MS - 1,
                "step": MINUTE_MS,
                "count": BASE_ROWS_PER_DAY,
                "unit": "unix-ms",
            },
            layout={
                "encoding": "row-major",
                "dtype": "float32-le",
                "rows": BASE_ROWS_PER_DAY,
                "columns": list(BASE_CHANNEL_NAMES),
            },
            metadata={
                "corpusId": corpus_id,
                "contract": BASE_COMPONENT_CONTRACT,
                "source": "canonical closed one-second spot BTCUSDT candles",
            },
        )
        expected = component_root / f"{component_date}.json"
        if reference.resolve() != expected.resolve():
            raise RuntimeError("minute-base reference path is inconsistent")
        result[component_date] = reference
        if index == 1 or index % 10 == 0 or index == len(missing):
            reporter.emit({
                "event": "base-component",
                "date": component_date,
                "completed": index,
                "total": len(missing),
            })
    if set(result) != required_dates:
        raise RuntimeError("minute-base component preparation is incomplete")
    return result


class BaseComponentCache:
    def __init__(self, max_entries: int = 8) -> None:
        self.max_entries = max_entries
        self.values: OrderedDict[Path, np.ndarray] = OrderedDict()

    def load(self, file: Path) -> np.ndarray:
        cached = self.values.pop(file, None)
        if cached is not None:
            self.values[file] = cached
            return cached
        _shard, value = read_shard_array(
            file,
            "<f4",
            (BASE_ROWS_PER_DAY, BASE_CHANNEL_COUNT),
        )
        self.values[file] = value
        while len(self.values) > self.max_entries:
            self.values.popitem(last=False)
        return value


FeatureBatch = tuple[Tensor, Tensor, Tensor]


class FeatureScreenDataset:
    def __init__(
        self,
        segments: dict[str, list[PairShard]],
        component_files: dict[str, Path],
        spec: FeatureSpec,
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

    def _windows(
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
            raise IndexError("multifeature history window escapes daily components")
        return np.asarray(series[indices], dtype=np.float32)

    def iter_batches(
        self,
        split: str,
        batch_size: int,
        *,
        shuffle: bool,
        seed: int,
    ) -> Iterator[FeatureBatch]:
        if batch_size < 1:
            raise ValueError("feature-screen batch size must be positive")
        generator = np.random.default_rng(seed)
        shards = list(self.segments[split])
        if shuffle:
            generator.shuffle(shards)
        cache = BaseComponentCache()
        pending_features: list[np.ndarray] = []
        pending_target: list[np.ndarray] = []
        pending_weights: list[np.ndarray] = []
        pending_count = 0

        def flush() -> FeatureBatch:
            nonlocal pending_count
            features = np.concatenate(pending_features, axis=0).astype(
                np.float32,
                copy=False,
            )
            target = np.concatenate(pending_target, axis=0).astype(
                np.float32,
                copy=False,
            )
            weights = np.concatenate(pending_weights).astype(np.float32, copy=False)
            pending_features.clear()
            pending_target.clear()
            pending_weights.clear()
            pending_count = 0
            return (
                torch.from_numpy(features),
                torch.from_numpy(target),
                torch.from_numpy(weights),
            )

        for shard in shards:
            history_rows, future_rows, weights = compact_pair_rows(
                shard.history_row_offset,
                shard.future_row_offset,
                shard.count,
            )
            history_base = self._windows(
                cache,
                shard.history_date,
                history_rows,
                self.spec.required_base_minutes,
            )
            target_base = self._windows(
                cache,
                shard.future_date,
                future_rows,
                FORECAST_MINUTES,
            )
            features = build_causal_features(history_base, self.spec)
            target = target_base[:, :, 0]
            if shuffle:
                order = generator.permutation(weights.shape[0])
                features = features[order]
                target = target[order]
                weights = weights[order]
            offset = 0
            while offset < weights.shape[0]:
                take = min(batch_size - pending_count, weights.shape[0] - offset)
                end = offset + take
                pending_features.append(features[offset:end])
                pending_target.append(target[offset:end])
                pending_weights.append(weights[offset:end])
                pending_count += take
                offset = end
                if pending_count == batch_size:
                    yield flush()
        if pending_count:
            yield flush()


def dataset_fingerprint(timestamp_fingerprint: str, spec: FeatureSpec) -> str:
    payload = {
        "timestampCorpusFingerprint": timestamp_fingerprint,
        "baseComponentContract": BASE_COMPONENT_CONTRACT,
        "featureContract": spec.contract,
        "target": "next-60-completed-minute-close-log-returns-v1",
        **(
            {"identityNormalizedChannels": list(spec.identity_normalized_channels)}
            if spec.identity_normalized_channels
            else {}
        ),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def compute_training_normalization(
    dataset: FeatureScreenDataset,
    cache_file: Path,
    *,
    fingerprint: str,
    batch_size: int,
    reporter: JsonReporter,
) -> FeatureNormalization:
    expected_count = dataset.logical_count("train")
    if cache_file.is_file():
        with np.load(cache_file, allow_pickle=False) as cached:
            if str(cached["datasetFingerprint"]) != fingerprint \
                    or int(cached["count"]) != expected_count:
                raise ValueError("feature normalization cache belongs to another corpus")
            normalization = FeatureNormalization(
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
    input_shape = (HISTORY_MINUTES, dataset.spec.channel_count)
    input_sum = np.zeros(input_shape, dtype=np.float64)
    input_square_sum = np.zeros(input_shape, dtype=np.float64)
    target_sum = np.zeros(FORECAST_MINUTES, dtype=np.float64)
    target_square_sum = np.zeros(FORECAST_MINUTES, dtype=np.float64)
    total = 0.0
    for features, target, weights in dataset.iter_batches(
        "train",
        batch_size,
        shuffle=False,
        seed=0,
    ):
        x = features.numpy().astype(np.float64, copy=False)
        y = target.numpy().astype(np.float64, copy=False)
        w = weights.numpy().astype(np.float64, copy=False)
        input_sum += np.einsum("i,itc->tc", w, x)
        input_square_sum += np.einsum("i,itc->tc", w, np.square(x))
        target_sum += np.einsum("i,ij->j", w, y)
        target_square_sum += np.einsum("i,ij->j", w, np.square(y))
        total += float(w.sum())
    if int(total) != expected_count:
        raise RuntimeError("feature normalization did not cover the training corpus")

    def moments(total_sum: np.ndarray, square_sum: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mean = total_sum / total
        variance = np.maximum(1e-12, square_sum / total - mean * mean)
        return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)

    input_mean, input_std = moments(input_sum, input_square_sum)
    target_mean, target_std = moments(target_sum, target_square_sum)
    for channel_name in dataset.spec.identity_normalized_channels:
        channel = dataset.spec.channel_names.index(channel_name)
        input_mean[:, channel] = 0
        input_std[:, channel] = 1
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
    normalization = FeatureNormalization(
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


def _normalization_json(normalization: FeatureNormalization) -> dict:
    return {
        "inputMean": normalization.input_mean.tolist(),
        "inputStd": normalization.input_std.tolist(),
        "targetMean": normalization.target_mean.tolist(),
        "targetStd": normalization.target_std.tolist(),
    }


def normalization_fingerprint(normalization: FeatureNormalization) -> str:
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


OBJECTIVE_STATISTICS_CONTRACT = (
    "weighted-training-only-variance-of-future-path-and-lossless-"
    "60-15-5-1-additive-components-v1"
)


def _objective_statistics_json(
    statistics: ForecastObjectiveStatistics,
) -> dict:
    return statistics.as_json()


def objective_statistics_fingerprint(
    statistics: ForecastObjectiveStatistics,
    *,
    data_fingerprint: str,
) -> str:
    statistics.validate()
    digest = hashlib.sha256()
    digest.update(OBJECTIVE_STATISTICS_CONTRACT.encode())
    digest.update(data_fingerprint.encode())
    for value in statistics.scales().values():
        array = value.detach().cpu().numpy().astype("<f4", copy=False)
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def compute_training_objective_statistics(
    dataset: FeatureScreenDataset,
    cache_file: Path,
    *,
    fingerprint: str,
    batch_size: int,
    reporter: JsonReporter,
) -> ForecastObjectiveStatistics:
    expected_count = dataset.logical_count("train")
    if cache_file.is_file():
        with np.load(cache_file, allow_pickle=False) as cached:
            if str(cached["datasetFingerprint"]) != fingerprint \
                    or str(cached["contract"]) != OBJECTIVE_STATISTICS_CONTRACT \
                    or int(cached["count"]) != expected_count:
                raise ValueError("forecast objective statistics belong to another corpus")
            statistics = ForecastObjectiveStatistics(
                path_std=torch.from_numpy(cached["pathStd"].astype(np.float32)),
                hour_std=torch.from_numpy(cached["hourStd"].astype(np.float32)),
                quarter_hour_contrast_std=torch.from_numpy(
                    cached["quarterHourContrastStd"].astype(np.float32)
                ),
                five_minute_contrast_std=torch.from_numpy(
                    cached["fiveMinuteContrastStd"].astype(np.float32)
                ),
                minute_residual_std=torch.from_numpy(
                    cached["minuteResidualStd"].astype(np.float32)
                ),
            )
        statistics.validate()
        reporter.emit({
            "event": "training-objective-statistics-cache",
            "hit": True,
            "examples": expected_count,
            "file": str(cache_file),
            "fingerprint": objective_statistics_fingerprint(
                statistics,
                data_fingerprint=fingerprint,
            ),
        })
        return statistics

    shapes = {
        "path": (FORECAST_MINUTES,),
        "hour": (1,),
        "quarterHourContrast": (4,),
        "fiveMinuteContrast": (12,),
        "minuteResidual": (FORECAST_MINUTES,),
    }
    sums = {name: np.zeros(shape, dtype=np.float64) for name, shape in shapes.items()}
    square_sums = {
        name: np.zeros(shape, dtype=np.float64)
        for name, shape in shapes.items()
    }
    total = 0.0
    for _, target, weights in dataset.iter_batches(
        "train",
        batch_size,
        shuffle=False,
        seed=0,
    ):
        structure = target_structure(target.to(dtype=torch.float64))
        weight_array = weights.numpy().astype(np.float64, copy=False)
        for name, values in structure.items():
            array = values.numpy()
            sums[name] += np.einsum("i,ij->j", weight_array, array)
            square_sums[name] += np.einsum(
                "i,ij->j",
                weight_array,
                np.square(array),
            )
        total += float(weight_array.sum())
    if int(total) != expected_count:
        raise RuntimeError("objective statistics did not cover the training corpus")

    def standard_deviation(name: str) -> Tensor:
        mean = sums[name] / total
        variance = np.maximum(
            1e-12,
            square_sums[name] / total - mean * mean,
        )
        return torch.from_numpy(np.sqrt(variance).astype(np.float32))

    statistics = ForecastObjectiveStatistics(
        path_std=standard_deviation("path"),
        hour_std=standard_deviation("hour"),
        quarter_hour_contrast_std=standard_deviation("quarterHourContrast"),
        five_minute_contrast_std=standard_deviation("fiveMinuteContrast"),
        minute_residual_std=standard_deviation("minuteResidual"),
    )
    statistics.validate()
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_file.with_suffix(cache_file.suffix + ".tmp")
    with temporary.open("wb") as output:
        np.savez(
            output,
            datasetFingerprint=np.asarray(fingerprint),
            contract=np.asarray(OBJECTIVE_STATISTICS_CONTRACT),
            count=np.asarray(expected_count, dtype=np.int64),
            pathStd=statistics.path_std.numpy(),
            hourStd=statistics.hour_std.numpy(),
            quarterHourContrastStd=(
                statistics.quarter_hour_contrast_std.numpy()
            ),
            fiveMinuteContrastStd=(
                statistics.five_minute_contrast_std.numpy()
            ),
            minuteResidualStd=statistics.minute_residual_std.numpy(),
        )
    os.replace(temporary, cache_file)
    reporter.emit({
        "event": "training-objective-statistics-cache",
        "hit": False,
        "examples": expected_count,
        "file": str(cache_file),
        "fingerprint": objective_statistics_fingerprint(
            statistics,
            data_fingerprint=fingerprint,
        ),
    })
    return statistics


class ObjectiveMetricAccumulator:
    def __init__(self) -> None:
        self.mass: Tensor | None = None
        self.sums: dict[str, Tensor] = {}
        self.stage: str | None = None

    def add(
        self,
        metrics: dict[str, Tensor],
        weights: Tensor,
        stage: str,
    ) -> None:
        mass = weights.detach().to(dtype=torch.float64).sum()
        if self.stage is None:
            self.stage = stage
        elif self.stage != stage:
            raise ValueError("objective stage changed inside an epoch")
        self.mass = mass if self.mass is None else self.mass + mass
        for name, value in metrics.items():
            weighted = value.detach().to(dtype=torch.float64) * mass
            self.sums[name] = self.sums.get(name, 0) + weighted

    def result(self) -> dict:
        if self.mass is None or float(self.mass) <= 0 or self.stage is None:
            raise RuntimeError("cannot finalize empty objective metrics")
        values = {
            name: float(value / self.mass)
            for name, value in self.sums.items()
        }
        if not all(math.isfinite(value) for value in values.values()):
            raise FloatingPointError("forecast objective metrics are non-finite")
        return {
            **values,
            "forecastObjectiveStage": self.stage,
        }


VALIDATION_BASELINE_CONTRACT = (
    "full-purged-validation-zero-return-and-training-horizon-mean-"
    "normalized-structural-losses-v1"
)


class StructuralBaselineAccumulator:
    def __init__(
        self,
        normalization: FeatureNormalization,
        statistics: ForecastObjectiveStatistics,
        *,
        huber_delta: float,
    ) -> None:
        statistics.validate()
        self.target_mean = normalization.target_mean.to(dtype=torch.float64)
        self.target_std = normalization.target_std.to(dtype=torch.float64)
        self.statistics = ForecastObjectiveStatistics(*(
            value.to(dtype=torch.float64)
            for value in (
                statistics.path_std,
                statistics.hour_std,
                statistics.quarter_hour_contrast_std,
                statistics.five_minute_contrast_std,
                statistics.minute_residual_std,
            )
        ))
        self.huber_delta = float(huber_delta)
        self.mass = 0.0
        self.dimensions: dict[str, int] = {}
        self.sums: dict[str, dict[str, float]] = {
            "zeroReturn": {},
            "trainingMean": {},
        }

    def add(self, target: Tensor, weights: Tensor) -> None:
        target = target.to(dtype=torch.float64)
        weights = weights.to(dtype=torch.float64)
        forecasts = {
            "zeroReturn": torch.zeros_like(target),
            "trainingMean": self.target_mean.unsqueeze(0).expand_as(target),
        }
        weight_column = weights.unsqueeze(-1)
        self.mass += float(weights.sum())
        for forecast_name, prediction in forecasts.items():
            errors = normalized_component_errors(
                prediction,
                target,
                self.target_std,
                self.statistics,
            )
            for component, error in errors.items():
                stem = "normalized" + component[0].upper() + component[1:]
                absolute = error.abs()
                huber = torch.where(
                    absolute <= self.huber_delta,
                    0.5 * error.square(),
                    self.huber_delta * (
                        absolute - 0.5 * self.huber_delta
                    ),
                )
                values = {
                    f"{stem}Mse": error.square(),
                    f"{stem}Huber": huber,
                }
                for metric, elements in values.items():
                    self.dimensions[metric] = elements.shape[-1]
                    contribution = float((elements * weight_column).sum())
                    totals = self.sums[forecast_name]
                    totals[metric] = totals.get(metric, 0.0) + contribution

    def result(self, expected_examples: int) -> dict:
        if int(round(self.mass)) != expected_examples:
            raise RuntimeError(
                "validation baselines did not cover the expected logical corpus"
            )
        result: dict[str, dict[str, float | int]] = {}
        for forecast_name, totals in self.sums.items():
            metrics = {
                name: total / (self.mass * self.dimensions[name])
                for name, total in totals.items()
            }
            if not metrics or not all(math.isfinite(value) for value in metrics.values()):
                raise FloatingPointError("validation baseline metrics are non-finite")
            result[forecast_name] = {
                "examples": expected_examples,
                **metrics,
            }
        return result


def _validation_baseline_cache_key(
    *,
    data_fingerprint: str,
    normalization_id: str,
    objective_statistics_id: str,
    huber_delta: float,
) -> str:
    payload = {
        "contract": VALIDATION_BASELINE_CONTRACT,
        "datasetFingerprint": data_fingerprint,
        "normalizationFingerprint": normalization_id,
        "objectiveStatisticsFingerprint": objective_statistics_id,
        "huberDelta": float(huber_delta),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _validate_structural_baselines(
    baselines: dict,
    *,
    expected_examples: int,
) -> None:
    expected_metrics = {
        f"normalized{name[0].upper() + name[1:]}{suffix}"
        for name in OBJECTIVE_COMPONENTS
        for suffix in ("Mse", "Huber")
    }
    if set(baselines) != {"zeroReturn", "trainingMean"}:
        raise ValueError("validation baseline forecast set changed")
    for name, metrics in baselines.items():
        if int(metrics.get("examples", -1)) != expected_examples \
                or not expected_metrics.issubset(metrics) \
                or not all(
                    math.isfinite(float(metrics[metric]))
                    for metric in expected_metrics
                ):
            raise ValueError(f"validation baseline {name} is invalid")


def compute_validation_structural_baselines(
    dataset: FeatureScreenDataset,
    normalization: FeatureNormalization,
    statistics: ForecastObjectiveStatistics,
    cache_dir: Path,
    *,
    data_fingerprint: str,
    batch_size: int,
    huber_delta: float,
    reporter: JsonReporter,
) -> dict:
    expected_examples = dataset.logical_count("validation")
    normalization_id = normalization_fingerprint(normalization)
    statistics_id = objective_statistics_fingerprint(
        statistics,
        data_fingerprint=data_fingerprint,
    )
    cache_key = _validation_baseline_cache_key(
        data_fingerprint=data_fingerprint,
        normalization_id=normalization_id,
        objective_statistics_id=statistics_id,
        huber_delta=huber_delta,
    )
    cache_file = cache_dir / (
        f"validation-forecast-structural-baselines-v1-{cache_key[:16]}.json"
    )
    if cache_file.is_file():
        cached = json.loads(cache_file.read_text(encoding="utf-8"))
        if cached.get("cacheKey") != cache_key \
                or cached.get("contract") != VALIDATION_BASELINE_CONTRACT:
            raise ValueError("validation baseline cache contract changed")
        baselines = cached.get("baselines", {})
        _validate_structural_baselines(
            baselines,
            expected_examples=expected_examples,
        )
        reporter.emit({
            "event": "validation-structural-baselines-cache",
            "hit": True,
            "examples": expected_examples,
            "file": str(cache_file),
            "fingerprint": cache_key,
        })
        return {
            "contract": VALIDATION_BASELINE_CONTRACT,
            "fingerprint": cache_key,
            "huberDelta": float(huber_delta),
            **baselines,
        }

    accumulator = StructuralBaselineAccumulator(
        normalization,
        statistics,
        huber_delta=huber_delta,
    )
    for _, target, weights in dataset.iter_batches(
        "validation",
        batch_size,
        shuffle=False,
        seed=0,
    ):
        accumulator.add(target, weights)
    baselines = accumulator.result(expected_examples)
    payload = {
        "version": 1,
        "contract": VALIDATION_BASELINE_CONTRACT,
        "cacheKey": cache_key,
        "datasetFingerprint": data_fingerprint,
        "normalizationFingerprint": normalization_id,
        "objectiveStatisticsFingerprint": statistics_id,
        "huberDelta": float(huber_delta),
        "baselines": baselines,
    }
    atomic_json(payload, cache_file)
    reporter.emit({
        "event": "validation-structural-baselines-cache",
        "hit": False,
        "examples": expected_examples,
        "file": str(cache_file),
        "fingerprint": cache_key,
    })
    return {
        "contract": VALIDATION_BASELINE_CONTRACT,
        "fingerprint": cache_key,
        "huberDelta": float(huber_delta),
        **baselines,
    }


def validation_baselines_for_objective(
    structural_baselines: dict,
    forecast_objective: dict,
) -> dict:
    fixed_objective, _ = objective_for_epoch(
        forecast_objective,
        1,
        validation=True,
    )
    if float(structural_baselines["huberDelta"]) \
            != float(fixed_objective["huberDelta"]):
        raise ValueError("validation baseline Huber delta does not match objective")
    result = {
        "contract": structural_baselines["contract"],
        "fingerprint": structural_baselines["fingerprint"],
        "huberDelta": structural_baselines["huberDelta"],
        "forecastObjectiveContract": objective_contract(forecast_objective),
    }
    for forecast_name in ("zeroReturn", "trainingMean"):
        metrics = dict(structural_baselines[forecast_name])
        metrics["forecastObjectiveLoss"] = (
            forecast_objective_value_from_metrics(
                metrics,
                forecast_objective,
            )
        )
        result[forecast_name] = metrics
    return result


def dummy_normalization(spec: FeatureSpec) -> FeatureNormalization:
    return FeatureNormalization(
        torch.zeros(HISTORY_MINUTES, spec.channel_count),
        torch.ones(HISTORY_MINUTES, spec.channel_count),
        torch.zeros(FORECAST_MINUTES),
        torch.ones(FORECAST_MINUTES),
    )


def resume_contract(
    plan: dict,
    *,
    data_fingerprint: str,
    model_contract: str,
    parameters: int,
    normalization_id: str,
    objective_statistics_id: str | None = None,
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
    if "forecastObjective" in training:
        if objective_statistics_id is None:
            raise ValueError("forecast objective resume contract needs statistics")
        payload.update({
            "forecastObjective": training["forecastObjective"],
            "forecastObjectiveContract": objective_contract(
                training["forecastObjective"]
            ),
            "forecastObjectiveStatisticsFingerprint": objective_statistics_id,
        })
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def evaluate(
    model: nn.Module,
    dataset: FeatureScreenDataset,
    normalization: FeatureNormalization,
    *,
    batch_size: int,
    device: torch.device,
    amp_dtype: torch.dtype,
    objective: str,
    huber_delta: float,
    forecast_objective: dict | None = None,
    objective_statistics: ForecastObjectiveStatistics | None = None,
    epoch: int = 1,
    validation_baselines: dict | None = None,
) -> dict:
    model.eval()
    metrics = ForecastMetricAccumulator(
        normalization,
        huber_delta=huber_delta,
        device=device,
    )
    if (forecast_objective is None) != (objective_statistics is None):
        raise ValueError("forecast objective evaluation settings are incomplete")
    if objective_statistics is not None:
        objective_statistics.validate()
    objective_metrics = (
        ObjectiveMetricAccumulator()
        if forecast_objective is not None
        else None
    )
    objective_target_std = normalization.target_std.to(
        device=device,
        dtype=torch.float32,
    )
    with torch.inference_mode():
        for features, target, weights in dataset.iter_batches(
            "validation",
            batch_size,
            shuffle=False,
            seed=0,
        ):
            features = features.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            weights = weights.to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=device.type == "cuda",
            ):
                prediction = model(features)
            metrics.add(prediction, target, weights)
            if forecast_objective is not None:
                _, batch_metrics, stage = forecast_objective_loss(
                    prediction.float(),
                    target.float(),
                    weights,
                    objective_target_std,
                    objective_statistics,
                    forecast_objective,
                    epoch=epoch,
                    validation=True,
                )
                objective_metrics.add(batch_metrics, weights, stage)
    result = metrics.result()
    if objective_metrics is None:
        result["loss"] = result[
            "normalizedHuber" if objective == "huber" else "normalizedMse"
        ]
    else:
        result.update(objective_metrics.result())
        result["loss"] = result["forecastObjectiveLoss"]
    if validation_baselines is not None:
        result["structuralBaselines"] = validation_baselines
    return result


def train(
    plan: dict,
    dataset: FeatureScreenDataset,
    normalization: FeatureNormalization,
    *,
    data_fingerprint: str,
    run_dir: Path,
    reporter: JsonReporter,
    stop_after_epoch: int | None,
    objective_statistics: ForecastObjectiveStatistics | None = None,
    validation_baselines: dict | None = None,
) -> None:
    training = plan["training"]
    device = torch.device(training["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA feature-screen training is unavailable")
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
    model = build_feature_predictor(
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
    forecast_objective = training.get("forecastObjective")
    if (forecast_objective is None) != (objective_statistics is None):
        raise ValueError("forecast objective training settings are incomplete")
    if objective_statistics is not None:
        objective_statistics.validate()
    objective_statistics_device = (
        objective_statistics.to(device)
        if objective_statistics is not None
        else None
    )
    objective_statistics_id = (
        objective_statistics_fingerprint(
            objective_statistics,
            data_fingerprint=data_fingerprint,
        )
        if objective_statistics is not None
        else None
    )
    contract = resume_contract(
        plan,
        data_fingerprint=data_fingerprint,
        model_contract=model_contract,
        parameters=parameters,
        normalization_id=normalization_fingerprint(normalization),
        objective_statistics_id=objective_statistics_id,
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
            raise ValueError("feature-screen resume checkpoint contract changed")
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
        train_metrics = ForecastMetricAccumulator(
            normalization,
            huber_delta=huber_delta,
            device=device,
        )
        train_objective_metrics = (
            ObjectiveMetricAccumulator()
            if forecast_objective is not None
            else None
        )
        for features, target, weights in dataset.iter_batches(
            "train",
            int(training["batchSize"]),
            shuffle=True,
            seed=seed + epoch,
        ):
            features = features.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            weights = weights.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=device.type == "cuda",
            ):
                prediction = active_model(features)
                if forecast_objective is None:
                    loss = normalized_forecast_loss(
                        prediction,
                        target,
                        weights,
                        model.target_std,
                        objective=objective,
                        huber_delta=huber_delta,
                    )
                    batch_objective_metrics = None
                    objective_stage = None
                else:
                    loss, batch_objective_metrics, objective_stage = (
                        forecast_objective_loss(
                            prediction.float(),
                            target.float(),
                            weights,
                            model.target_std.float(),
                            objective_statistics_device,
                            forecast_objective,
                            epoch=epoch,
                            validation=False,
                        )
                    )
            loss.backward()
            gradient_norm = clip_grad_norm_(
                model.parameters(),
                float(training["gradientClip"]),
            )
            if not bool(torch.isfinite(gradient_norm)):
                raise FloatingPointError("feature-screen gradient is non-finite")
            optimizer.step()
            global_step += 1
            train_metrics.add(prediction, target, weights)
            if batch_objective_metrics is not None:
                train_objective_metrics.add(
                    batch_objective_metrics,
                    weights,
                    objective_stage,
                )
        train_result = train_metrics.result()
        if train_objective_metrics is None:
            train_result["loss"] = train_result[
                "normalizedHuber" if objective == "huber" else "normalizedMse"
            ]
        else:
            train_result.update(train_objective_metrics.result())
            train_result["loss"] = train_result["forecastObjectiveLoss"]
        validation_result = evaluate(
            active_model,
            dataset,
            normalization,
            batch_size=int(training["evaluationBatchSize"]),
            device=device,
            amp_dtype=amp_dtype,
            objective=objective,
            huber_delta=huber_delta,
            forecast_objective=forecast_objective,
            objective_statistics=objective_statistics_device,
            epoch=epoch,
            validation_baselines=validation_baselines,
        )
        validation_value = float(validation_result[selection_metric])
        if not math.isfinite(validation_value):
            raise FloatingPointError("feature-screen validation metric is non-finite")
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
            "featureContract": dataset.spec.contract,
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
        if forecast_objective is not None:
            checkpoint.update({
                "forecastObjective": forecast_objective,
                "forecastObjectiveContract": objective_contract(forecast_objective),
                "forecastObjectiveStatistics": _objective_statistics_json(
                    objective_statistics
                ),
                "forecastObjectiveStatisticsFingerprint": objective_statistics_id,
            })
        if improved:
            best_checkpoint = {
                key: checkpoint[key]
                for key in (
                    "version",
                    "resumeContract",
                    "datasetFingerprint",
                    "featureContract",
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
            }
            if forecast_objective is not None:
                best_checkpoint.update({
                    key: checkpoint[key]
                    for key in (
                        "forecastObjective",
                        "forecastObjectiveContract",
                        "forecastObjectiveStatistics",
                        "forecastObjectiveStatisticsFingerprint",
                    )
                })
            save_torch_checkpoint(
                best_checkpoint,
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
                message="Feature screen reached validation patience.",
            )
            return
    reporter.status(
        "complete",
        planId=plan["id"],
        epoch=int(training["epochs"]),
        bestEpoch=best_epoch,
        bestValidation=best_validation,
        message="Feature screen exhausted its configured epochs.",
    )


def validate_plan(plan: dict) -> FeatureSpec:
    required = (
        "id",
        "corpusId",
        "sourceDatasetDir",
        "decoderDatasetDir",
        "datasetDir",
        "runDir",
        "historyDir",
        "featureFormat",
        "architecture",
        "architectureConfig",
        "training",
    )
    if any(name not in plan or plan[name] in (None, "") for name in required):
        raise ValueError("feature-screen plan is missing required fields")
    spec = FeatureSpec.from_config(plan["featureFormat"])
    if plan["architecture"] not in {
        "rlinear_dlinear",
        "causal_patch_tcn",
        "patch_tide",
    }:
        raise ValueError("feature-screen architecture is unsupported")
    training = plan["training"]
    forecast_objective = training.get("forecastObjective")
    allowed_selection_metrics = {
        "normalizedHuber",
        "normalizedMse",
        "rawMse",
    }
    if forecast_objective is not None:
        validate_forecast_objective(forecast_objective)
        allowed_selection_metrics.add("forecastObjectiveLoss")
    if training.get("objective") not in {"huber", "mse"} \
            or training.get("selectionMetric") not in allowed_selection_metrics:
        raise ValueError("feature-screen objective or selection metric is invalid")
    if forecast_objective is not None \
            and training.get("selectionMetric") != "forecastObjectiveLoss":
        raise ValueError("structured forecast runs must select their fixed objective")
    for key in ("epochs", "batchSize", "evaluationBatchSize", "patience", "seed"):
        if int(training.get(key, 0)) < 1:
            raise ValueError(f"feature-screen training {key} must be positive")
    optimizer = training.get("optimizer", {})
    schedule = training.get("learningRateSchedule", {})
    if optimizer.get("type") != "adamw" \
            or len(optimizer.get("betas", ())) != 2 \
            or schedule.get("type") != "reduce-on-validation-plateau" \
            or not 0 < float(schedule.get("factor", 0)) < 1:
        raise ValueError("feature-screen optimizer configuration is invalid")
    if float(training.get("huberDelta", 0)) <= 0 \
            or float(training.get("learningRate", 0)) <= 0 \
            or float(training.get("gradientClip", 0)) <= 0 \
            or float(optimizer.get("epsilon", 0)) <= 0 \
            or float(schedule.get("threshold", -1)) < 0 \
            or float(schedule.get("minimumLearningRate", 0)) <= 0 \
            or int(schedule.get("patience", -1)) < 0:
        raise ValueError("feature-screen optimization scales are invalid")
    if training.get("device") not in {"cpu", "cuda"} \
            or training.get("mixedPrecision") != "bfloat16":
        raise ValueError("feature-screen device/precision contract is invalid")
    # Factory validation also rejects stray architecture settings.
    build_feature_predictor(
        plan["architecture"],
        plan["architectureConfig"],
        spec,
        dummy_normalization(spec),
    )
    return spec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Screen causal six-hour feature formats for one-hour returns."
    )
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate plan, alignment, parameter count, and fingerprints only.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Prepare immutable minute features and training-only statistics.",
    )
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
        segments = select_pair_segments(
            source_manifest,
            cross_split_purge_ms=EXAMPLE_SPAN_MS,
        )
        validate_predictor_split_disjointness(
            segments,
            example_span_ms=EXAMPLE_SPAN_MS,
        )
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
        timestamp_fingerprint = corpus_fingerprint(
            segments,
            contract=TIMESTAMP_CORPUS_CONTRACT,
        )
        data_fingerprint = dataset_fingerprint(timestamp_fingerprint, spec)
        definition = build_feature_predictor(
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
            "featureFormat": spec.format,
            "channels": list(spec.channel_names),
            "counts": counts,
            "compactMinuteCounts": compact_counts,
            "timestampCorpusFingerprint": timestamp_fingerprint,
            "datasetFingerprint": data_fingerprint,
            "crossSplitPurgeMs": EXAMPLE_SPAN_MS,
            "parameterCount": parameter_count(definition),
            "architectureContract": model_contract,
            "heldoutTest": "not selected or read",
        }
        if "forecastObjective" in plan["training"]:
            validation_event["forecastObjectiveContract"] = objective_contract(
                plan["training"]["forecastObjective"]
            )
        reporter.emit(validation_event)
        if args.validate_only:
            reporter.status(
                "paused",
                planId=plan["id"],
                latest=validation_event,
                message="Feature-screen plan validated without reading payloads.",
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
            / plan["corpusId"]
        )
        reporter.status(
            "dataset-preparation",
            planId=plan["id"],
            message="Preparing causal completed-minute OHLCV base components.",
        )
        component_files = prepare_base_components(
            component_dates,
            component_root=component_root,
            history_root=history_root,
            immutable_root=layout.immutable,
            corpus_id=plan["corpusId"],
            reporter=reporter,
        )
        dataset = FeatureScreenDataset(segments, component_files, spec)
        normalization = compute_training_normalization(
            dataset,
            dataset_root / (
                "training-feature-statistics-v2.npz"
                if spec.identity_normalized_channels
                else "training-feature-statistics-v1.npz"
            ),
            fingerprint=data_fingerprint,
            batch_size=int(plan["training"]["evaluationBatchSize"]),
            reporter=reporter,
        )
        objective_statistics = None
        structural_baselines = None
        validation_baselines = None
        if "forecastObjective" in plan["training"]:
            objective_statistics = compute_training_objective_statistics(
                dataset,
                dataset_root / "training-forecast-objective-statistics-v1.npz",
                fingerprint=data_fingerprint,
                batch_size=int(plan["training"]["evaluationBatchSize"]),
                reporter=reporter,
            )
            fixed_validation_objective, _ = objective_for_epoch(
                plan["training"]["forecastObjective"],
                1,
                validation=True,
            )
            structural_baselines = compute_validation_structural_baselines(
                dataset,
                normalization,
                objective_statistics,
                dataset_root,
                data_fingerprint=data_fingerprint,
                batch_size=int(plan["training"]["evaluationBatchSize"]),
                huber_delta=float(fixed_validation_objective["huberDelta"]),
                reporter=reporter,
            )
            validation_baselines = validation_baselines_for_objective(
                structural_baselines,
                plan["training"]["forecastObjective"],
            )
            reporter.emit({
                "event": "validation-structural-baselines",
                "planId": plan["id"],
                "baselines": validation_baselines,
            })
        manifest = {
            "version": 1,
            "createdAt": iso_now(),
            "corpusId": plan["corpusId"],
            "sourceDataset": str((source_root / "dataset.json").relative_to(repo_root)),
            "decoderDataset": str((decoder_root / "dataset.json").relative_to(repo_root)),
            "baseComponentContract": BASE_COMPONENT_CONTRACT,
            "baseChannels": list(BASE_CHANNEL_NAMES),
            "featureFormat": spec.format,
            "featureContract": spec.contract,
            "featureChannels": list(spec.channel_names),
            "historyMinutes": HISTORY_MINUTES,
            "target": "next 60 completed one-minute close log returns",
            "targetMinutes": FORECAST_MINUTES,
            "crossSplitPurgeMs": EXAMPLE_SPAN_MS,
            "counts": counts,
            "compactMinuteCounts": compact_counts,
            "timestampCorpusFingerprint": timestamp_fingerprint,
            "datasetFingerprint": data_fingerprint,
            "normalization": {
                "source": "selected training split only",
                "axis": "each history position/channel and target horizon separately",
                "varianceCorrection": 0,
                "identityChannels": list(spec.identity_normalized_channels),
                "fingerprint": normalization_fingerprint(normalization),
                **_normalization_json(normalization),
            },
            "heldoutTest": "not selected, loaded, normalized, evaluated, or exposed",
        }
        if objective_statistics is not None:
            manifest["forecastObjectiveNormalization"] = {
                "contract": OBJECTIVE_STATISTICS_CONTRACT,
                "source": "selected training split only",
                "varianceCorrection": 0,
                "fingerprint": objective_statistics_fingerprint(
                    objective_statistics,
                    data_fingerprint=data_fingerprint,
                ),
                **_objective_statistics_json(objective_statistics),
            }
            manifest["validationStructuralBaselines"] = structural_baselines
        atomic_json(manifest, dataset_root / "dataset.json")
        reporter.emit({
            "event": "dataset-complete",
            "datasetFingerprint": data_fingerprint,
            "counts": counts,
            "compactMinuteCounts": compact_counts,
            "components": len(component_files),
            "parameters": parameter_count(definition),
        })
        if args.prepare_only:
            reporter.status(
                "paused",
                planId=plan["id"],
                message="Feature-screen preparation completed.",
            )
            return
        train(
            plan,
            dataset,
            normalization,
            data_fingerprint=data_fingerprint,
            run_dir=run_dir,
            reporter=reporter,
            stop_after_epoch=args.stop_after_epoch,
            objective_statistics=objective_statistics,
            validation_baselines=validation_baselines,
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
