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

from future_price_predictor import (
    FEATURE_CONTRACT,
    FORECAST_RETURN_COUNT,
    HISTORY_RETURN_COUNT,
    ForecastNormalization,
    architecture_contract,
    build_close_return_predictor,
    normalized_forecast_loss,
    parameter_count,
)
from trading_storage import (
    checkpoint_exists,
    is_storage_reference,
    load_torch_checkpoint,
    read_candle_column,
    read_shard_array,
    require_under,
    resolve_shard,
    save_torch_checkpoint,
    training_storage_layout,
    write_shard_payload,
)


SECOND_MS = 1_000
MINUTE_SECONDS = 60
MINUTE_MS = 60_000
HOUR_MS = 3_600_000
DAY_SECONDS = 86_400
MINUTE_ROWS_PER_DAY = 1_441
PREDICTOR_EXAMPLE_SPAN_MS = 2 * HOUR_MS
CORPUS_CONTRACT = (
    "decoder-delay-3600s-train-validation-assignments-with-2h-past-future-"
    "cross-split-purge-compact-minute-multiplicity-v1"
)
COMPONENT_SUFFIX = ".completed-minute-simple-returns-60.compact.json"


@dataclass(frozen=True)
class PairShard:
    split: str
    prediction_time_start: int
    count: int
    future_date: str
    history_date: str
    future_row_offset: int
    history_row_offset: int

    @property
    def prediction_time_end(self) -> int:
        return self.prediction_time_start + (self.count - 1) * SECOND_MS

    def shifted(self, local_offset: int) -> PairShard:
        if local_offset < 0 or local_offset >= self.count:
            raise ValueError("pair-shard offset must select at least one example")
        return PairShard(
            split=self.split,
            prediction_time_start=(
                self.prediction_time_start + local_offset * SECOND_MS
            ),
            count=self.count - local_offset,
            future_date=self.future_date,
            history_date=self.history_date,
            future_row_offset=self.future_row_offset + local_offset,
            history_row_offset=self.history_row_offset + local_offset,
        )


def completed_minute_row_indices(rows: np.ndarray) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.int64)
    return rows // MINUTE_SECONDS + (rows % MINUTE_SECONDS == 59)


def compact_pair_rows(
    history_start: int,
    future_start: int,
    count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collapse identical second examples without changing their measure."""
    if count < 1:
        raise ValueError("cannot compact an empty predictor segment")
    offsets = np.arange(count, dtype=np.int64)
    history_rows = completed_minute_row_indices(history_start + offsets)
    future_rows = completed_minute_row_indices(future_start + offsets)
    history_changes = np.flatnonzero(np.concatenate((
        np.asarray([True]),
        history_rows[1:] != history_rows[:-1],
    )))
    future_changes = np.flatnonzero(np.concatenate((
        np.asarray([True]),
        future_rows[1:] != future_rows[:-1],
    )))
    if not np.array_equal(history_changes, future_changes):
        raise ValueError("history and future rows have different minute boundaries")
    weights = np.diff(np.append(history_changes, count)).astype(
        np.float32,
        copy=False,
    )
    return (
        history_rows[history_changes],
        future_rows[future_changes],
        weights,
    )


def _timestamp_date(timestamp_ms: int) -> str:
    return datetime.fromtimestamp(
        timestamp_ms / 1_000,
        tz=timezone.utc,
    ).date().isoformat()


def source_pair_shards(source_manifest: dict) -> list[PairShard]:
    result: list[PairShard] = []
    for value in source_manifest.get("shards", ()):
        split = str(value.get("split", ""))
        if split not in {"train", "validation", "test"}:
            continue
        prediction_start = int(value["predictionTimeStart"])
        oracle_start = int(value["oracleTargetTimeStart"])
        if prediction_start - oracle_start != HOUR_MS:
            raise ValueError("source shard does not preserve the one-hour pairing")
        if int(value["featureRowStride"]) != 1 \
                or int(value["oracleRowStride"]) != 1:
            raise ValueError("predictor corpus requires contiguous source rows")
        result.append(PairShard(
            split=split,
            prediction_time_start=prediction_start,
            count=int(value["count"]),
            future_date=str(value["date"]),
            history_date=_timestamp_date(oracle_start),
            future_row_offset=int(value["featureRowOffset"]),
            history_row_offset=int(value["oracleRowOffset"]),
        ))
    result.sort(key=lambda shard: shard.prediction_time_start)
    return result


def select_pair_segments(
    source_manifest: dict,
    *,
    cross_split_purge_ms: int,
    legacy_decoder_transition_semantics: bool = False,
) -> dict[str, list[PairShard]]:
    if cross_split_purge_ms < HOUR_MS:
        raise ValueError("cross-split purge cannot be shorter than one hour")
    selected: dict[str, list[PairShard]] = {"train": [], "validation": []}
    previous_end: int | None = None
    previous_split: str | None = None
    selected_end = {"train": None, "validation": None, "test": None}
    for shard in source_pair_shards(source_manifest):
        if previous_end is not None and shard.prediction_time_start <= previous_end:
            raise ValueError("source predictor timestamps overlap")
        local_offset = 0
        if legacy_decoder_transition_semantics:
            if previous_end is not None and shard.split != previous_split:
                earliest = previous_end + cross_split_purge_ms + SECOND_MS
                if shard.prediction_time_start < earliest:
                    local_offset = math.ceil(
                        (earliest - shard.prediction_time_start) / SECOND_MS
                    )
        else:
            other_ends = tuple(
                end
                for split, end in selected_end.items()
                if split != shard.split and end is not None
            )
            if other_ends:
                earliest = max(other_ends) + cross_split_purge_ms + SECOND_MS
                if shard.prediction_time_start < earliest:
                    local_offset = math.ceil(
                        (earliest - shard.prediction_time_start) / SECOND_MS
                    )
        if shard.split in selected and local_offset < shard.count:
            chosen = shard.shifted(local_offset) if local_offset else shard
            selected[shard.split].append(chosen)
            selected_end[shard.split] = chosen.prediction_time_end
        elif shard.split not in selected and local_offset < shard.count:
            chosen = shard.shifted(local_offset) if local_offset else shard
            selected_end[shard.split] = chosen.prediction_time_end
        previous_end = shard.prediction_time_end
        previous_split = shard.split
    if not selected["train"] or not selected["validation"]:
        raise ValueError("predictor train and validation splits must be non-empty")
    return selected


def validate_predictor_split_disjointness(
    segments: dict[str, list[PairShard]],
    *,
    example_span_ms: int = PREDICTOR_EXAMPLE_SPAN_MS,
) -> None:
    if example_span_ms < HOUR_MS:
        raise ValueError("predictor example span cannot be shorter than one hour")
    def merged(split: str) -> list[tuple[int, int]]:
        ranges = sorted(
            (
                shard.prediction_time_start - example_span_ms,
                shard.prediction_time_end,
            )
            for shard in segments[split]
        )
        result: list[tuple[int, int]] = []
        for start, end in ranges:
            if result and start <= result[-1][1] + SECOND_MS:
                result[-1] = (result[-1][0], max(result[-1][1], end))
            else:
                result.append((start, end))
        return result

    train_ranges = merged("train")
    validation_ranges = merged("validation")
    train_index = 0
    validation_index = 0
    while train_index < len(train_ranges) \
            and validation_index < len(validation_ranges):
        train_start, train_end = train_ranges[train_index]
        validation_start, validation_end = validation_ranges[validation_index]
        if train_start <= validation_end and validation_start <= train_end:
            raise ValueError("train and validation predictor candle spans overlap")
        if train_end < validation_start:
            train_index += 1
        else:
            validation_index += 1


def validate_source_manifest(source_manifest: dict) -> None:
    if source_manifest.get("version") != 8:
        raise ValueError("predictor source dataset must use schema 8")
    if source_manifest.get("samplingIntervalMs") != SECOND_MS:
        raise ValueError("predictor source dataset must use one-second rows")
    if source_manifest.get("predictionDelayMs") != HOUR_MS:
        raise ValueError("predictor source alignment must use a one-hour delay")
    minute_oracle = source_manifest.get("minuteOracleMap", {})
    if minute_oracle.get("valueHorizonSteps") != FORECAST_RETURN_COUNT \
            or minute_oracle.get("holdingPeriodSteps") != 1 \
            or "close-only one-minute path" not in str(
                minute_oracle.get("sampling", "")
            ):
        raise ValueError("predictor source minute-path contract is invalid")


def count_examples(segments: dict[str, list[PairShard]]) -> dict[str, int]:
    return {
        split: sum(shard.count for shard in values)
        for split, values in segments.items()
    }


def corpus_fingerprint(
    segments: dict[str, list[PairShard]],
    *,
    contract: str = CORPUS_CONTRACT,
) -> str:
    payload = {
        "contract": contract,
        "segments": [
            {
                "split": split,
                "predictionTimeStart": shard.prediction_time_start,
                "count": shard.count,
                "futureDate": shard.future_date,
                "historyDate": shard.history_date,
                "futureRowOffset": shard.future_row_offset,
                "historyRowOffset": shard.history_row_offset,
            }
            for split in ("train", "validation")
            for shard in segments[split]
        ],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


class JsonReporter:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.log_file = run_dir / "logs" / "training.jsonl"
        self.status_file = run_dir / "state" / "status.json"
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.status_file.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, event: dict) -> None:
        line = json.dumps(event, separators=(",", ":"), allow_nan=False)
        with self.log_file.open("a", encoding="utf-8", newline="\n") as output:
            output.write(line + "\n")
        print(line, flush=True)

    def status(self, stage: str, **values) -> None:
        atomic_json({
            "pid": os.getpid(),
            "stage": stage,
            "updatedAt": iso_now(),
            **values,
        }, self.status_file)


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def atomic_json(value: dict, file: Path) -> None:
    file.parent.mkdir(parents=True, exist_ok=True)
    temporary = file.with_name(f"{file.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, file)


def completed_minute_simple_return_rows(
    previous_close: np.ndarray,
    current_close: np.ndarray,
) -> np.ndarray:
    if previous_close.shape != (DAY_SECONDS,) \
            or current_close.shape != (DAY_SECONDS,):
        raise ValueError("daily close arrays must contain 86,400 rows")
    close = np.concatenate((
        previous_close[-(HISTORY_RETURN_COUNT * MINUTE_SECONDS + 1):],
        current_close,
    ))
    offsets = np.arange(
        0,
        HISTORY_RETURN_COUNT * MINUTE_SECONDS + 1,
        MINUTE_SECONDS,
        dtype=np.int64,
    )
    minute_rows = (
        np.arange(MINUTE_ROWS_PER_DAY, dtype=np.int64)[:, None]
        * MINUTE_SECONDS
    )
    boundaries = close[minute_rows + offsets[None, :]]
    returns = boundaries[:, 1:] / boundaries[:, :-1] - 1.0
    if not np.isfinite(returns).all() or bool((returns <= -1).any()):
        raise ValueError("completed-minute returns are invalid")
    return returns


def _load_daily_close(
    history_root: Path,
    component_date: str,
    cache: OrderedDict[str, np.ndarray],
) -> np.ndarray:
    cached = cache.pop(component_date, None)
    if cached is not None:
        cache[component_date] = cached
        return cached
    file = history_root / f"{component_date}.json"
    if not file.is_file():
        raise FileNotFoundError(f"missing one-second close history: {file}")
    close = read_candle_column(file, "close")
    if close.shape != (DAY_SECONDS,) \
            or not np.isfinite(close).all() \
            or bool((close <= 0).any()):
        raise ValueError(f"invalid one-second close history: {file}")
    cache[component_date] = close
    while len(cache) > 3:
        cache.popitem(last=False)
    return close


def valid_return_component(file: Path) -> bool:
    if not file.is_file() or not is_storage_reference(file):
        return False
    try:
        shard = resolve_shard(file)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    layout = shard.reference.get("layout", {})
    return shard.axis.count == MINUTE_ROWS_PER_DAY \
        and int(shard.reference["object"]["uncompressedBytes"]) \
        == MINUTE_ROWS_PER_DAY * HISTORY_RETURN_COUNT * np.dtype("<f2").itemsize \
        and layout.get("dtype") == "float16-le"


def prepare_component_files(
    *,
    required_dates: set[str],
    decoder_component_root: Path,
    supplementary_component_root: Path,
    history_root: Path,
    immutable_root: Path,
    corpus_id: str,
    reporter: JsonReporter,
) -> dict[str, Path]:
    supplementary_component_root.mkdir(parents=True, exist_ok=True)
    result: dict[str, Path] = {}
    missing: list[str] = []
    for component_date in sorted(required_dates):
        decoder_file = decoder_component_root / f"{component_date}{COMPONENT_SUFFIX}"
        supplement_file = supplementary_component_root / f"{component_date}.json"
        if valid_return_component(decoder_file):
            result[component_date] = decoder_file
        elif valid_return_component(supplement_file):
            result[component_date] = supplement_file
        else:
            missing.append(component_date)
    close_cache: OrderedDict[str, np.ndarray] = OrderedDict()
    for index, component_date in enumerate(missing, start=1):
        current_day = date.fromisoformat(component_date)
        previous_date = (current_day - timedelta(days=1)).isoformat()
        returns = completed_minute_simple_return_rows(
            _load_daily_close(history_root, previous_date, close_cache),
            _load_daily_close(history_root, component_date, close_cache),
        )
        reference = write_shard_payload(
            immutable_root,
            "features/future-price-predictor-simple-returns-v1",
            f"{corpus_id}/{component_date}",
            returns.astype("<f2", copy=False).tobytes(),
            sequence={
                "start": int(datetime.fromisoformat(component_date).replace(
                    tzinfo=timezone.utc,
                ).timestamp() * 1_000) - 1,
                "step": MINUTE_MS,
                "count": MINUTE_ROWS_PER_DAY,
                "unit": "unix-ms",
            },
            layout={
                "encoding": "row-major",
                "dtype": "float16-le",
                "rows": MINUTE_ROWS_PER_DAY,
                "columns": HISTORY_RETURN_COUNT,
            },
            metadata={
                "corpusId": corpus_id,
                "reason": "history-boundary date absent from decoder features",
            },
        )
        expected = supplementary_component_root / f"{component_date}.json"
        if reference.resolve() != expected.resolve():
            raise RuntimeError("supplementary return reference path is inconsistent")
        result[component_date] = reference
        reporter.emit({
            "event": "dataset-component",
            "date": component_date,
            "completed": index,
            "total": len(missing),
        })
    if set(result) != required_dates:
        raise RuntimeError("predictor return component preparation is incomplete")
    return result


class ComponentCache:
    def __init__(self, max_entries: int = 8) -> None:
        self.max_entries = max_entries
        self.values: OrderedDict[Path, np.ndarray] = OrderedDict()

    def load(self, file: Path) -> np.ndarray:
        cached = self.values.pop(file, None)
        if cached is not None:
            self.values[file] = cached
            return cached
        _shard, values = read_shard_array(
            file,
            "<f2",
            (MINUTE_ROWS_PER_DAY, HISTORY_RETURN_COUNT),
        )
        self.values[file] = values
        while len(self.values) > self.max_entries:
            self.values.popitem(last=False)
        return values


Batch = tuple[Tensor, Tensor, Tensor]


class PredictorDataset:
    def __init__(
        self,
        segments: dict[str, list[PairShard]],
        component_files: dict[str, Path],
    ) -> None:
        self.segments = segments
        self.component_files = component_files

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

    def iter_batches(
        self,
        split: str,
        batch_size: int,
        *,
        shuffle: bool,
        seed: int,
    ) -> Iterator[Batch]:
        if batch_size < 1:
            raise ValueError("predictor batch size must be positive")
        generator = np.random.default_rng(seed)
        shards = list(self.segments[split])
        if shuffle:
            generator.shuffle(shards)
        cache = ComponentCache()
        pending_history: list[np.ndarray] = []
        pending_future: list[np.ndarray] = []
        pending_weights: list[np.ndarray] = []
        pending_count = 0

        def flush() -> Batch:
            nonlocal pending_count
            history = np.concatenate(pending_history, axis=0).astype(
                np.float32,
                copy=False,
            )
            future = np.concatenate(pending_future, axis=0).astype(
                np.float32,
                copy=False,
            )
            weights = np.concatenate(pending_weights).astype(np.float32, copy=False)
            pending_history.clear()
            pending_future.clear()
            pending_weights.clear()
            pending_count = 0
            return (
                torch.from_numpy(history),
                torch.from_numpy(future),
                torch.from_numpy(weights),
            )

        for shard in shards:
            history_source = cache.load(self.component_files[shard.history_date])
            future_source = cache.load(self.component_files[shard.future_date])
            history_rows, future_rows, weights = compact_pair_rows(
                shard.history_row_offset,
                shard.future_row_offset,
                shard.count,
            )
            if history_rows[0] < 0 \
                    or future_rows[0] < 0 \
                    or history_rows[-1] >= MINUTE_ROWS_PER_DAY \
                    or future_rows[-1] >= MINUTE_ROWS_PER_DAY:
                raise IndexError("predictor compact row falls outside its UTC day")
            if shuffle:
                order = generator.permutation(weights.shape[0])
                history_rows = history_rows[order]
                future_rows = future_rows[order]
                weights = weights[order]
            history = np.log1p(
                np.asarray(history_source[history_rows], dtype=np.float32)
            )
            future = np.log1p(
                np.asarray(future_source[future_rows], dtype=np.float32)
            )
            if not np.isfinite(history).all() or not np.isfinite(future).all():
                raise ValueError("predictor log-return conversion is non-finite")
            offset = 0
            while offset < weights.shape[0]:
                take = min(batch_size - pending_count, weights.shape[0] - offset)
                end = offset + take
                pending_history.append(history[offset:end])
                pending_future.append(future[offset:end])
                pending_weights.append(weights[offset:end])
                pending_count += take
                offset = end
                if pending_count == batch_size:
                    yield flush()
        if pending_count:
            yield flush()


def compute_training_normalization(
    dataset: PredictorDataset,
    cache_file: Path,
    *,
    fingerprint: str,
    batch_size: int,
    reporter: JsonReporter,
) -> ForecastNormalization:
    expected_count = dataset.logical_count("train")
    if cache_file.is_file():
        with np.load(cache_file, allow_pickle=False) as cached:
            cached_fingerprint = str(cached["corpusFingerprint"])
            count = int(cached["count"])
            input_mean = cached["inputMean"]
            input_std = cached["inputStd"]
            target_mean = cached["targetMean"]
            target_std = cached["targetStd"]
        if cached_fingerprint != fingerprint or count != expected_count:
            raise ValueError("cached predictor normalization has a stale corpus")
        normalization = ForecastNormalization(
            torch.from_numpy(input_mean.astype(np.float32)),
            torch.from_numpy(input_std.astype(np.float32)),
            torch.from_numpy(target_mean.astype(np.float32)),
            torch.from_numpy(target_std.astype(np.float32)),
        )
        normalization.validate()
        reporter.emit({
            "event": "training-statistics-cache",
            "hit": True,
            "examples": count,
            "file": str(cache_file),
        })
        return normalization

    input_sum = np.zeros(HISTORY_RETURN_COUNT, dtype=np.float64)
    input_square_sum = np.zeros(HISTORY_RETURN_COUNT, dtype=np.float64)
    target_sum = np.zeros(FORECAST_RETURN_COUNT, dtype=np.float64)
    target_square_sum = np.zeros(FORECAST_RETURN_COUNT, dtype=np.float64)
    total = 0.0
    for history, future, weights in dataset.iter_batches(
        "train",
        batch_size,
        shuffle=False,
        seed=0,
    ):
        x = history.numpy().astype(np.float64, copy=False)
        y = future.numpy().astype(np.float64, copy=False)
        w = weights.numpy().astype(np.float64, copy=False)
        input_sum += np.einsum("i,ij->j", w, x)
        input_square_sum += np.einsum("i,ij->j", w, np.square(x))
        target_sum += np.einsum("i,ij->j", w, y)
        target_square_sum += np.einsum("i,ij->j", w, np.square(y))
        total += float(w.sum())
    if int(total) != expected_count:
        raise RuntimeError("training normalization did not cover the predictor corpus")

    def moments(values: np.ndarray, squares: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mean = values / total
        variance = np.maximum(1e-14, squares / total - mean * mean)
        return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)

    input_mean, input_std = moments(input_sum, input_square_sum)
    target_mean, target_std = moments(target_sum, target_square_sum)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_file.with_suffix(cache_file.suffix + ".tmp")
    with temporary.open("wb") as output:
        np.savez(
            output,
            corpusFingerprint=np.asarray(fingerprint),
            count=np.asarray(expected_count, dtype=np.int64),
            inputMean=input_mean,
            inputStd=input_std,
            targetMean=target_mean,
            targetStd=target_std,
        )
    os.replace(temporary, cache_file)
    normalization = ForecastNormalization(
        torch.from_numpy(input_mean),
        torch.from_numpy(input_std),
        torch.from_numpy(target_mean),
        torch.from_numpy(target_std),
    )
    normalization.validate()
    reporter.emit({
        "event": "training-statistics-cache",
        "hit": False,
        "examples": expected_count,
        "file": str(cache_file),
    })
    return normalization


class ForecastMetricAccumulator:
    def __init__(
        self,
        normalization: ForecastNormalization,
        *,
        huber_delta: float,
        device: torch.device,
    ) -> None:
        self.huber_delta = float(huber_delta)
        self.target_std = normalization.target_std.to(device=device, dtype=torch.float64)
        self.target_mean = normalization.target_mean.to(device=device, dtype=torch.float64)
        self.weight = torch.zeros((), dtype=torch.float64, device=device)
        self.squared = torch.zeros(FORECAST_RETURN_COUNT, dtype=torch.float64, device=device)
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
        self.zero_baseline_squared = torch.zeros_like(self.squared)
        self.mean_baseline_normalized_squared = torch.zeros_like(self.squared)

    def add(self, prediction: Tensor, target: Tensor, weights: Tensor) -> None:
        prediction = prediction.detach().to(dtype=torch.float64)
        target = target.detach().to(dtype=torch.float64)
        weights = weights.detach().to(dtype=torch.float64).unsqueeze(-1)
        error = prediction - target
        normalized_error = error / self.target_std
        absolute_normalized = normalized_error.abs()
        huber = torch.where(
            absolute_normalized <= self.huber_delta,
            0.5 * normalized_error.square(),
            self.huber_delta * (absolute_normalized - 0.5 * self.huber_delta),
        )
        self.weight += weights.sum()
        self.squared += (error.square() * weights).sum(dim=0)
        self.absolute += (error.abs() * weights).sum(dim=0)
        self.normalized_squared += (normalized_error.square() * weights).sum(dim=0)
        self.normalized_huber += (huber * weights).sum(dim=0)
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
        self.zero_baseline_squared += (target.square() * weights).sum(dim=0)
        mean_error = (target - self.target_mean) / self.target_std
        self.mean_baseline_normalized_squared += (
            mean_error.square() * weights
        ).sum(dim=0)

    def result(self) -> dict:
        if float(self.weight) <= 0:
            raise RuntimeError("cannot finalize empty forecast metrics")
        weight = self.weight
        mse = self.squared / weight
        mae = self.absolute / weight
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
        correlation = covariance / (
            prediction_variance.sqrt() * target_variance.sqrt()
        ).clamp_min(1e-20)
        scale_ratio = prediction_variance.sqrt() / target_variance.sqrt().clamp_min(1e-20)

        def values(tensor: Tensor) -> list[float]:
            return tensor.detach().cpu().tolist()

        return {
            "examples": int(round(float(weight))),
            "rawMse": float(mse.mean()),
            "rawRmse": math.sqrt(float(mse.mean())),
            "rawMae": float(mae.mean()),
            "normalizedMse": float(normalized_mse.mean()),
            "normalizedHuber": float(normalized_huber.mean()),
            "directionAccuracy": float((self.direction / weight).mean()),
            "cumulativePathRmse": math.sqrt(float(cumulative_mse.mean())),
            "endpointRmse": math.sqrt(float(cumulative_mse[-1])),
            "meanHorizonCorrelation": float(correlation.mean()),
            "meanPredictionScaleRatio": float(scale_ratio.mean()),
            "zeroReturnBaselineRawMse": float(
                (self.zero_baseline_squared / weight).mean()
            ),
            "trainingMeanBaselineNormalizedMse": float(
                (self.mean_baseline_normalized_squared / weight).mean()
            ),
            "mseByHorizon": values(mse),
            "maeByHorizon": values(mae),
            "normalizedMseByHorizon": values(normalized_mse),
            "normalizedHuberByHorizon": values(normalized_huber),
            "directionAccuracyByHorizon": values(self.direction / weight),
            "correlationByHorizon": values(correlation),
            "predictionScaleRatioByHorizon": values(scale_ratio),
            "cumulativeMseByHorizon": values(cumulative_mse),
        }


def evaluate(
    model: nn.Module,
    dataset: PredictorDataset,
    normalization: ForecastNormalization,
    *,
    split: str,
    batch_size: int,
    device: torch.device,
    amp_dtype: torch.dtype,
    objective: str,
    huber_delta: float,
) -> dict:
    model.eval()
    metrics = ForecastMetricAccumulator(
        normalization,
        huber_delta=huber_delta,
        device=device,
    )
    with torch.inference_mode():
        for history, target, weights in dataset.iter_batches(
            split,
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
    return result


def _normalization_json(normalization: ForecastNormalization) -> dict:
    return {
        "inputMean": normalization.input_mean.tolist(),
        "inputStd": normalization.input_std.tolist(),
        "targetMean": normalization.target_mean.tolist(),
        "targetStd": normalization.target_std.tolist(),
    }


def resume_contract(
    plan: dict,
    *,
    corpus_id: str,
    architecture: str,
    parameters: int,
) -> str:
    training = plan["training"]
    payload = {
        "featureContract": FEATURE_CONTRACT,
        "corpusFingerprint": corpus_id,
        "architectureContract": architecture,
        "parameters": parameters,
        "batchSize": training["batchSize"],
        "evaluationBatchSize": training["evaluationBatchSize"],
        "objective": training["objective"],
        "huberDelta": training["huberDelta"],
        "optimizer": training["optimizer"],
        "learningRate": training["learningRate"],
        "learningRateSchedule": training["learningRateSchedule"],
        "gradientClip": training["gradientClip"],
        "seed": training["seed"],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def restore_random_states(checkpoint: dict, device: torch.device) -> None:
    """Restore checkpoint RNGs even when map_location moved byte states.

    PyTorch's CPU generator requires a CPU uint8 tensor. Loading a complete
    checkpoint with ``map_location='cuda'`` also maps that tensor to CUDA, so
    canonicalize both CPU and CUDA generator states before restoration. This
    intentionally migrates the epoch-1 v1 checkpoints without rewriting them.
    """
    random.setstate(checkpoint["pythonRandomState"])
    np.random.set_state(checkpoint["numpyRandomState"])
    torch_state = checkpoint["torchRandomState"]
    if not isinstance(torch_state, Tensor):
        raise TypeError("checkpoint torch RNG state must be a tensor")
    torch.set_rng_state(
        torch_state.detach().to(
            device="cpu",
            dtype=torch.uint8,
            copy=True,
        )
    )
    cuda_state = checkpoint.get("cudaRandomState")
    if device.type == "cuda" and cuda_state is not None:
        if not isinstance(cuda_state, (tuple, list)) \
                or not all(isinstance(value, Tensor) for value in cuda_state):
            raise TypeError("checkpoint CUDA RNG state must contain tensors")
        torch.cuda.set_rng_state_all([
            value.detach().to(
                device="cpu",
                dtype=torch.uint8,
                copy=True,
            )
            for value in cuda_state
        ])


def train(
    plan: dict,
    dataset: PredictorDataset,
    normalization: ForecastNormalization,
    *,
    corpus_id: str,
    run_dir: Path,
    reporter: JsonReporter,
    stop_after_epoch: int | None,
) -> None:
    training = plan["training"]
    device = torch.device(training["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA predictor training was requested but is unavailable")
    seed = int(training["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    torch.set_float32_matmul_precision("high")
    architecture = architecture_contract(
        plan["architecture"],
        plan["architectureConfig"],
    )
    model = build_close_return_predictor(
        plan["architecture"],
        normalization,
        plan["architectureConfig"],
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
    compiled_model = (
        torch.compile(model)
        if bool(training.get("compile", False))
        else model
    )
    contract = resume_contract(
        plan,
        corpus_id=corpus_id,
        architecture=architecture,
        parameters=parameters,
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
            raise ValueError("predictor resume checkpoint contract changed")
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
        architecture=architecture,
        parameters=parameters,
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
            bestValidation=(
                best_validation if math.isfinite(best_validation) else None
            ),
            message="Requested epoch boundary was already checkpointed.",
        )
        return
    for epoch in range(start_epoch, int(training["epochs"]) + 1):
        epoch_started = time.monotonic()
        compiled_model.train()
        train_metrics = ForecastMetricAccumulator(
            normalization,
            huber_delta=huber_delta,
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
                prediction = compiled_model(history)
                loss = normalized_forecast_loss(
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
                raise FloatingPointError("predictor gradient norm is non-finite")
            optimizer.step()
            global_step += 1
            train_metrics.add(prediction, target, weights)
        train_result = train_metrics.result()
        train_result["loss"] = train_result[
            "normalizedHuber" if objective == "huber" else "normalizedMse"
        ]
        validation_result = evaluate(
            compiled_model,
            dataset,
            normalization,
            split="validation",
            batch_size=int(training["evaluationBatchSize"]),
            device=device,
            amp_dtype=amp_dtype,
            objective=objective,
            huber_delta=huber_delta,
        )
        validation_value = float(validation_result[selection_metric])
        if not math.isfinite(validation_value):
            raise FloatingPointError("predictor validation selection metric is non-finite")
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
            "featureContract": FEATURE_CONTRACT,
            "corpusFingerprint": corpus_id,
            "architectureContract": architecture,
            "architecture": plan["architecture"],
            "architectureConfig": plan["architectureConfig"],
            "parameterCount": parameters,
            "normalization": _normalization_json(normalization),
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
                        "featureContract",
                        "corpusFingerprint",
                        "architectureContract",
                        "architecture",
                        "architectureConfig",
                        "parameterCount",
                        "normalization",
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
                    "selectionMetric": selection_metric,
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
            "seconds": time.monotonic() - epoch_started,
            "learningRate": float(optimizer.param_groups[0]["lr"]),
            "improved": improved,
            "bestEpoch": best_epoch,
            "bestValidation": best_validation,
            "selectionMetric": selection_metric,
            "training": train_result,
            "validation": validation_result,
        }
        reporter.emit(event)
        reporter.status(
            "training",
            planId=plan["id"],
            epoch=epoch,
            globalStep=global_step,
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
                message="Requested epoch boundary reached; last checkpoint is resumable.",
            )
            return
        if stale_epochs >= int(training["patience"]):
            reporter.status(
                "complete",
                planId=plan["id"],
                epoch=epoch,
                bestEpoch=best_epoch,
                bestValidation=best_validation,
                message="Predictor capability screen reached its validation patience.",
            )
            return
    reporter.status(
        "complete",
        planId=plan["id"],
        epoch=int(training["epochs"]),
        bestEpoch=best_epoch,
        bestValidation=best_validation,
    )


def validate_plan(plan: dict) -> None:
    required = (
        "id",
        "label",
        "corpusId",
        "sourceDatasetDir",
        "decoderDatasetDir",
        "datasetDir",
        "runDir",
        "historyDir",
        "architecture",
        "architectureConfig",
        "training",
    )
    if any(name not in plan or plan[name] in (None, "") for name in required):
        raise ValueError("predictor plan is missing required fields")
    if plan["architecture"] not in {"rlinear_dlinear", "causal_patch_tcn"}:
        raise ValueError("predictor plan selects an unsupported architecture")
    training = plan["training"]
    if training.get("objective") not in {"mse", "huber"}:
        raise ValueError("predictor objective must be MSE or Huber")
    if training.get("selectionMetric") not in {
        "normalizedMse",
        "normalizedHuber",
        "rawMse",
    }:
        raise ValueError("predictor validation selection metric is invalid")
    for key in (
        "epochs",
        "batchSize",
        "evaluationBatchSize",
        "patience",
        "seed",
    ):
        if int(training.get(key, 0)) < 1:
            raise ValueError(f"predictor training {key} must be positive")
    if float(training.get("huberDelta", 0)) <= 0 \
            or float(training.get("learningRate", 0)) <= 0 \
            or float(training.get("gradientClip", 0)) <= 0:
        raise ValueError("predictor loss and optimizer scales must be positive")
    optimizer = training.get("optimizer", {})
    if optimizer.get("type") != "adamw" \
            or len(optimizer.get("betas", ())) != 2:
        raise ValueError("predictor optimizer must be AdamW")
    schedule = training.get("learningRateSchedule", {})
    if schedule.get("type") != "reduce-on-validation-plateau" \
            or int(schedule.get("patience", -1)) < 0 \
            or not 0 < float(schedule.get("factor", 0)) < 1 \
            or float(schedule.get("minimumLearningRate", 0)) <= 0:
        raise ValueError("predictor learning-rate schedule is invalid")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a past-only close-log-return predictor on the compact "
            "train/validation timestamp corpus used by the decoder proof."
        )
    )
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Prepare/validate train and validation data without training.",
    )
    parser.add_argument(
        "--stop-after-epoch",
        type=int,
        help="Pause after this completed, durably checkpointed epoch.",
    )
    return parser.parse_args()


def resolve(repo_root: Path, value: Path) -> Path:
    return value.resolve() if value.is_absolute() else (repo_root / value).resolve()


def main() -> None:
    args = parse_args()
    if args.stop_after_epoch is not None and args.stop_after_epoch < 1:
        raise ValueError("--stop-after-epoch must be positive")
    repo_root = Path(__file__).resolve().parent.parent
    plan_file = resolve(repo_root, args.plan)
    plan = json.loads(plan_file.read_text(encoding="utf-8"))
    validate_plan(plan)
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
    reporter.status(
        "dataset-preparation",
        planId=plan["id"],
        message="Selecting past/future close-return pairs without test payloads.",
    )
    try:
        source_manifest = json.loads(
            (source_root / "dataset.json").read_text(encoding="utf-8")
        )
        validate_source_manifest(source_manifest)
        decoder_manifest = json.loads(
            (decoder_root / "dataset.json").read_text(encoding="utf-8")
        )
        decoder_segments = select_pair_segments(
            source_manifest,
            cross_split_purge_ms=HOUR_MS,
            legacy_decoder_transition_semantics=True,
        )
        decoder_counts = count_examples(decoder_segments)
        for split in ("train", "validation"):
            if decoder_counts[split] != int(decoder_manifest["counts"][split]):
                raise ValueError(
                    "source timestamp reconstruction no longer matches the "
                    f"decoder {split} corpus"
                )
        segments = select_pair_segments(
            source_manifest,
            cross_split_purge_ms=PREDICTOR_EXAMPLE_SPAN_MS,
        )
        validate_predictor_split_disjointness(segments)
        fingerprint = corpus_fingerprint(segments)
        required_dates = {
            component_date
            for values in segments.values()
            for shard in values
            for component_date in (shard.history_date, shard.future_date)
        }
        decoder_component_root = decoder_root / "components" / "returns"
        supplementary_root = (
            layout.immutable
            / "refs"
            / "features"
            / "future-price-predictor-simple-returns-v1"
            / plan["corpusId"]
        )
        component_files = prepare_component_files(
            required_dates=required_dates,
            decoder_component_root=decoder_component_root,
            supplementary_component_root=supplementary_root,
            history_root=history_root,
            immutable_root=layout.immutable,
            corpus_id=plan["corpusId"],
            reporter=reporter,
        )
        dataset = PredictorDataset(segments, component_files)
        counts = count_examples(segments)
        compact_counts = {
            split: dataset.compact_count(split)
            for split in ("train", "validation")
        }
        normalization = compute_training_normalization(
            dataset,
            dataset_root / "training-log-return-statistics-v1.npz",
            fingerprint=fingerprint,
            batch_size=int(plan["training"]["evaluationBatchSize"]),
            reporter=reporter,
        )
        definition = build_close_return_predictor(
            plan["architecture"],
            normalization,
            plan["architectureConfig"],
        )
        manifest = {
            "version": 1,
            "createdAt": iso_now(),
            "corpusId": plan["corpusId"],
            "sourceDataset": str((source_root / "dataset.json").relative_to(repo_root)),
            "decoderDataset": str((decoder_root / "dataset.json").relative_to(repo_root)),
            "featureContract": FEATURE_CONTRACT,
            "corpusContract": CORPUS_CONTRACT,
            "corpusFingerprint": fingerprint,
            "decisionTime": "source oracleTargetTime (t)",
            "input": (
                "60 completed one-minute close log returns ending at t; "
                "converted with log1p from the decoder's compact close-only "
                "simple-return components"
            ),
            "target": (
                "next 60 completed one-minute close log returns over (t,t+1h], "
                "the exact realized path used as decoder input"
            ),
            "normalization": {
                "source": "selected training split only",
                "axis": "separate per-position population mean/std for input and target",
                **_normalization_json(normalization),
            },
            "splitPolicy": (
                "decoder train/validation timestamp assignments with a two-hour "
                "purge at split transitions, covering both past input and future target"
            ),
            "crossSplitPurgeMs": PREDICTOR_EXAMPLE_SPAN_MS,
            "counts": counts,
            "compactMinuteCounts": compact_counts,
            "multiplicity": "exact one-second decoder-example weights",
            "heldoutTest": "not selected, loaded, normalized, evaluated, or exposed by this runner",
        }
        atomic_json(manifest, dataset_root / "dataset.json")
        reporter.emit({
            "event": "dataset-complete",
            "corpusFingerprint": fingerprint,
            "counts": counts,
            "compactMinuteCounts": compact_counts,
            "components": len(component_files),
            "parameters": parameter_count(definition),
        })
        if args.prepare_only:
            reporter.status(
                "paused",
                planId=plan["id"],
                message="Predictor train/validation preparation completed.",
            )
            return
        train(
            plan,
            dataset,
            normalization,
            corpus_id=fingerprint,
            run_dir=run_dir,
            reporter=reporter,
            stop_after_epoch=args.stop_after_epoch,
        )
    except KeyboardInterrupt:
        reporter.status(
            "paused",
            planId=plan["id"],
            message="Interrupted; the last completed epoch remains resumable.",
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
