from __future__ import annotations

from collections import OrderedDict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch import Tensor

from multiscale_next_return import MOVING_AVERAGE_WINDOWS, WINDOW_SECONDS
from multiscale_next_return_dataset import CloseRangeCache
from next_return_dataset import (
    HISTORY_RETURN_COUNT,
    SECOND_MS,
    ExampleShard,
)
from trading_storage import read_candle_column


HORIZON_CANDLE_COUNT = 5
RESOLUTION_SECONDS = {"1m": 60, "1h": 3_600, "1d": 86_400}


def allowed_max_windows(resolution: str) -> tuple[str, ...]:
    if resolution not in RESOLUTION_SECONDS:
        raise ValueError(f"unsupported candle resolution: {resolution}")
    seconds = RESOLUTION_SECONDS[resolution]
    return tuple(
        label
        for label, window_seconds in MOVING_AVERAGE_WINDOWS
        if window_seconds >= seconds and window_seconds % seconds == 0
    )


def selected_resolution_windows(
    resolution: str,
    max_window: str,
) -> tuple[tuple[str, int], ...]:
    labels = allowed_max_windows(resolution)
    if max_window not in labels:
        raise ValueError(
            f"{max_window} is invalid for {resolution} candle resolution"
        )
    index = labels.index(max_window)
    seconds = RESOLUTION_SECONDS[resolution]
    return tuple(
        (label, WINDOW_SECONDS[label] // seconds)
        for label in reversed(labels[:index + 1])
    )


def resolution_component_labels(
    resolution: str,
    max_window: str,
) -> tuple[str, ...]:
    windows = selected_resolution_windows(resolution, max_window)
    if len(windows) == 1:
        return (f"return_{resolution}",)
    labels = [f"ma_{windows[0][0]}"]
    for larger, smaller in zip(windows[:-1], windows[1:], strict=True):
        if smaller[0] == resolution:
            labels.append(f"return_{resolution}_minus_ma_{larger[0]}")
        else:
            labels.append(f"ma_{smaller[0]}_minus_ma_{larger[0]}")
    return tuple(labels)


def resolution_telescoping_components(
    raw_returns: np.ndarray,
    moving_averages: dict[str, np.ndarray],
    *,
    resolution: str,
    max_window: str,
) -> np.ndarray:
    if raw_returns.ndim != 1 or not np.isfinite(raw_returns).all():
        raise ValueError("aligned candle returns must be one finite vector")
    windows = selected_resolution_windows(resolution, max_window)
    if len(windows) == 1:
        return raw_returns.astype(np.float32)[:, None]
    values = [moving_averages[windows[0][0]]]
    for larger, smaller in zip(windows[:-1], windows[1:], strict=True):
        smaller_values = (
            raw_returns
            if smaller[0] == resolution
            else moving_averages[smaller[0]]
        )
        values.append(smaller_values - moving_averages[larger[0]])
    result = np.stack(values, axis=1).astype(np.float32)
    raw_float32 = raw_returns.astype(np.float32)
    result[:, -1] = raw_float32 - result[:, :-1].sum(
        axis=1, dtype=np.float32
    )
    error = np.abs(result.sum(axis=1, dtype=np.float32) - raw_float32)
    tolerance = (
        2 * np.finfo(np.float32).eps
        * np.abs(result).sum(axis=1, dtype=np.float32)
        + 1e-12
    )
    if bool((error > tolerance).any()):
        raise RuntimeError("resolution components do not reconstruct candles")
    return result


def aligned_candle_indices(
    shard: ExampleShard,
    resolution_seconds: int,
) -> tuple[np.ndarray, np.ndarray]:
    if resolution_seconds < 1 or 86_400 % resolution_seconds:
        raise ValueError("resolution must divide one UTC day")
    start_second = shard.decision_time_start // SECOND_MS
    local_offset = (-start_second) % resolution_seconds
    if local_offset >= shard.count:
        return np.empty(0, np.int64), np.empty(0, np.float32)
    second_rows = shard.row_offset + np.arange(
        local_offset, shard.count, resolution_seconds, dtype=np.int64
    )
    if bool((second_rows % resolution_seconds != 0).any()):
        raise RuntimeError("source shard does not align to candle boundaries")
    indices = second_rows // resolution_seconds
    return indices, np.ones(indices.size, dtype=np.float32)


def trim_shards_for_resolution_history(
    shards: dict[str, list[ExampleShard]],
    *,
    first_history_day: str,
    resolution: str,
    max_window: str,
) -> dict[str, list[ExampleShard]]:
    resolution_seconds = RESOLUTION_SECONDS[resolution]
    first_close = datetime.combine(
        date.fromisoformat(first_history_day), datetime.min.time(), timezone.utc
    )
    lookback_seconds = (
        WINDOW_SECONDS[max_window]
        + (HISTORY_RETURN_COUNT - 1) * resolution_seconds
    )
    earliest_ms = int(first_close.timestamp() * 1_000) + 999 \
        + lookback_seconds * SECOND_MS
    result: dict[str, list[ExampleShard]] = {}
    for split, values in shards.items():
        chosen: list[ExampleShard] = []
        for shard in values:
            if shard.decision_time_end < earliest_ms:
                continue
            offset = max(0, (
                earliest_ms - shard.decision_time_start + SECOND_MS - 1
            ) // SECOND_MS)
            chosen.append(shard.shifted(offset) if offset else shard)
        result[split] = chosen
    if not result.get("train") or not result.get("validation"):
        raise ValueError("resolution lookback removed a required split")
    return result


def daily_resolution_examples(
    close_cache: CloseRangeCache,
    day: str,
    *,
    resolution: str,
    max_window: str,
) -> tuple[np.ndarray, np.ndarray]:
    resolution_seconds = RESOLUTION_SECONDS[resolution]
    candles_per_day = 86_400 // resolution_seconds
    start = datetime.combine(
        date.fromisoformat(day), datetime.min.time(), timezone.utc
    ) - timedelta(seconds=HISTORY_RETURN_COUNT * resolution_seconds)
    count = HISTORY_RETURN_COUNT + candles_per_day + HORIZON_CANDLE_COUNT - 1
    candle_starts = start + timedelta(seconds=resolution_seconds)
    end_close = close_cache.load_range(
        candle_starts - timedelta(seconds=1),
        count * resolution_seconds,
    )[::resolution_seconds]
    previous_close = close_cache.load_range(
        start - timedelta(seconds=1),
        count * resolution_seconds,
    )[::resolution_seconds]
    if end_close.shape != (count,) or previous_close.shape != (count,):
        raise RuntimeError("aligned candle close construction is invalid")
    log_end = np.log(end_close)
    raw_returns = log_end - np.log(previous_close)
    moving_averages: dict[str, np.ndarray] = {}
    for label, window_candles in selected_resolution_windows(
        resolution, max_window
    ):
        if label == resolution:
            moving_averages[label] = raw_returns
            continue
        lagged_close = close_cache.load_range(
            candle_starts - timedelta(
                seconds=window_candles * resolution_seconds + 1
            ),
            count * resolution_seconds,
        )[::resolution_seconds]
        moving_averages[label] = (
            log_end - np.log(lagged_close)
        ) / window_candles
    components = resolution_telescoping_components(
        raw_returns,
        moving_averages,
        resolution=resolution,
        max_window=max_window,
    )
    windows = np.lib.stride_tricks.sliding_window_view(
        components,
        HISTORY_RETURN_COUNT + HORIZON_CANDLE_COUNT,
        axis=0,
    )
    expected = (
        candles_per_day,
        len(resolution_component_labels(resolution, max_window)),
        HISTORY_RETURN_COUNT + HORIZON_CANDLE_COUNT,
    )
    if windows.shape != expected:
        raise RuntimeError(f"resolution window shape {windows.shape} != {expected}")
    histories = windows[:, :, :HISTORY_RETURN_COUNT]
    targets = windows[:, :, HISTORY_RETURN_COUNT:].sum(axis=1)
    return histories, targets


class MultiscaleCandleResolutionDataset:
    def __init__(
        self,
        shards: dict[str, list[ExampleShard]],
        history_root: Path,
        *,
        resolution: str,
        max_window: str,
    ) -> None:
        self.shards = shards
        self.resolution = resolution
        self.resolution_seconds = RESOLUTION_SECONDS[resolution]
        self.max_window = max_window
        self.component_labels = resolution_component_labels(
            resolution, max_window
        )
        self.component_count = len(self.component_labels)
        self.close_cache = CloseRangeCache(history_root)
        self.component_cache: OrderedDict[
            str, tuple[np.ndarray, np.ndarray]
        ] = OrderedDict()
        self.component_cache_entries = 8

    def logical_count(self, split: str) -> int:
        return sum(
            aligned_candle_indices(shard, self.resolution_seconds)[0].size
            for shard in self.shards[split]
        )

    def _component(self, day: str) -> tuple[np.ndarray, np.ndarray]:
        cached = self.component_cache.pop(day, None)
        if cached is not None:
            self.component_cache[day] = cached
            return cached
        values = daily_resolution_examples(
            self.close_cache,
            day,
            resolution=self.resolution,
            max_window=self.max_window,
        )
        self.component_cache[day] = values
        while len(self.component_cache) > self.component_cache_entries:
            self.component_cache.popitem(last=False)
        return values

    def iter_batches(
        self,
        split: str,
        batch_size: int,
        *,
        shuffle: bool,
        seed: int,
        reuse_buffers: bool = False,
    ) -> Iterator[tuple[Tensor, Tensor, Tensor]]:
        if batch_size < 1:
            raise ValueError("batch size must be positive")
        generator = np.random.default_rng(seed)
        shards = list(self.shards[split])
        if shuffle:
            generator.shuffle(shards)
        feature_shape = (
            batch_size, self.component_count, HISTORY_RETURN_COUNT
        )
        target_shape = (batch_size, HORIZON_CANDLE_COUNT)
        feature_buffer = np.empty(feature_shape, dtype=np.float32)
        target_buffer = np.empty(target_shape, dtype=np.float32)
        weight_buffer = np.empty(batch_size, dtype=np.float32)
        filled = 0
        for shard in shards:
            history, target = self._component(shard.date)
            rows, weights = aligned_candle_indices(
                shard, self.resolution_seconds
            )
            if rows.size == 0:
                continue
            position = 0
            while position < rows.size:
                take = min(batch_size - filled, rows.size - position)
                selected = rows[position:position + take]
                destination = slice(filled, filled + take)
                feature_buffer[destination] = history[selected]
                target_buffer[destination] = target[selected]
                weight_buffer[destination] = weights[position:position + take]
                filled += take
                position += take
                if filled == batch_size:
                    yield (
                        torch.from_numpy(feature_buffer),
                        torch.from_numpy(target_buffer),
                        torch.from_numpy(weight_buffer),
                    )
                    if not reuse_buffers:
                        feature_buffer = np.empty(feature_shape, np.float32)
                        target_buffer = np.empty(target_shape, np.float32)
                        weight_buffer = np.empty(batch_size, np.float32)
                    filled = 0
        if filled:
            yield (
                torch.from_numpy(feature_buffer[:filled]),
                torch.from_numpy(target_buffer[:filled]),
                torch.from_numpy(weight_buffer[:filled]),
            )


def load_daily_log_closes(
    history_root: Path,
    *,
    first_day: str,
    last_day: str,
) -> tuple[tuple[str, ...], np.ndarray]:
    first = date.fromisoformat(first_day)
    last = date.fromisoformat(last_day)
    if last < first:
        raise ValueError("daily close range is reversed")
    days: list[str] = []
    closes: list[float] = []
    cursor = first
    while cursor <= last:
        label = cursor.isoformat()
        file = history_root / f"{label}.json"
        if not file.is_file():
            raise FileNotFoundError(f"missing one-minute candle day: {file}")
        values = read_candle_column(file, "close")
        if values.shape != (1_440,) \
                or not np.isfinite(values).all() \
                or bool((values <= 0).any()):
            raise ValueError(f"invalid one-minute candle day: {file}")
        days.append(label)
        closes.append(float(values[-1]))
        cursor += timedelta(days=1)
    return tuple(days), np.log(np.asarray(closes, dtype=np.float64))


def daily_prediction_indices(
    close_count: int,
    *,
    comparison_max_window: str,
) -> dict[str, np.ndarray]:
    if comparison_max_window not in allowed_max_windows("1d"):
        raise ValueError("invalid daily comparison maximum window")
    largest_window = WINDOW_SECONDS[comparison_max_window] // 86_400
    first_prediction = HISTORY_RETURN_COUNT + largest_window
    candidates = np.arange(
        first_prediction,
        close_count - HORIZON_CANDLE_COUNT + 1,
        dtype=np.int64,
    )
    if candidates.size < 20:
        raise ValueError("daily corpus is too short for chronological splits")
    train_boundary = int(close_count * 0.6)
    validation_boundary = int(close_count * 0.8)
    last_train_prediction_exclusive = (
        train_boundary - HORIZON_CANDLE_COUNT + 1
    )
    last_validation_prediction_exclusive = (
        validation_boundary - HORIZON_CANDLE_COUNT + 1
    )
    train = candidates[candidates < last_train_prediction_exclusive]
    validation = candidates[
        (candidates >= train_boundary)
        & (candidates < last_validation_prediction_exclusive)
    ]
    test = candidates[candidates >= validation_boundary]
    if min(train.size, validation.size, test.size) < 1:
        raise ValueError("daily chronological split is empty")
    if train[-1] + HORIZON_CANDLE_COUNT > validation[0] \
            or validation[-1] + HORIZON_CANDLE_COUNT > test[0]:
        raise RuntimeError("daily target purge failed")
    return {"train": train, "validation": validation, "test": test}


def daily_component_windows(
    log_closes: np.ndarray,
    prediction_indices: np.ndarray,
    *,
    max_window: str,
) -> tuple[np.ndarray, np.ndarray]:
    if log_closes.ndim != 1 or not np.isfinite(log_closes).all():
        raise ValueError("daily log closes must be one finite vector")
    windows = selected_resolution_windows("1d", max_window)
    first_component_day = max(window for _label, window in windows)
    raw_returns = np.diff(log_closes)
    endpoint_days = np.arange(first_component_day, log_closes.size)
    aligned_raw = raw_returns[endpoint_days - 1]
    moving_averages = {
        label: (
            log_closes[endpoint_days] - log_closes[endpoint_days - window]
        ) / window
        for label, window in windows
    }
    components = resolution_telescoping_components(
        aligned_raw,
        moving_averages,
        resolution="1d",
        max_window=max_window,
    )
    sequences = np.lib.stride_tricks.sliding_window_view(
        components,
        HISTORY_RETURN_COUNT + HORIZON_CANDLE_COUNT,
        axis=0,
    )
    rows = prediction_indices - HISTORY_RETURN_COUNT - first_component_day
    if rows.size and (rows.min() < 0 or rows.max() >= sequences.shape[0]):
        raise IndexError("daily prediction index escapes component windows")
    chosen = sequences[rows]
    histories = chosen[:, :, :HISTORY_RETURN_COUNT]
    targets = chosen[:, :, HISTORY_RETURN_COUNT:].sum(axis=1)
    return histories, targets


class DailyCandleResolutionDataset:
    def __init__(
        self,
        history_root: Path,
        *,
        first_day: str,
        last_day: str,
        max_window: str,
        comparison_max_window: str,
    ) -> None:
        self.resolution = "1d"
        self.resolution_seconds = RESOLUTION_SECONDS[self.resolution]
        self.max_window = max_window
        self.component_labels = resolution_component_labels(
            self.resolution, max_window
        )
        self.component_count = len(self.component_labels)
        self.days, self.log_closes = load_daily_log_closes(
            history_root, first_day=first_day, last_day=last_day
        )
        self.indices = daily_prediction_indices(
            self.log_closes.size,
            comparison_max_window=comparison_max_window,
        )
        all_indices = np.concatenate(tuple(self.indices.values()))
        histories, targets = daily_component_windows(
            self.log_closes, all_indices, max_window=max_window
        )
        self.histories = histories
        self.targets = targets
        self.positions: dict[str, np.ndarray] = {}
        offset = 0
        self.shards: dict[str, list[ExampleShard]] = {}
        for split in ("train", "validation", "test"):
            count = self.indices[split].size
            self.positions[split] = np.arange(offset, offset + count)
            offset += count
            self.shards[split] = [
                ExampleShard(
                    split,
                    int(datetime.combine(
                        date.fromisoformat(self.days[index]),
                        datetime.min.time(),
                        timezone.utc,
                    ).timestamp() * 1_000) + 999,
                    1,
                    self.days[index],
                    0,
                )
                for index in self.indices[split]
            ]

    def logical_count(self, split: str) -> int:
        return int(self.positions[split].size)

    def iter_batches(
        self,
        split: str,
        batch_size: int,
        *,
        shuffle: bool,
        seed: int,
        reuse_buffers: bool = False,
    ) -> Iterator[tuple[Tensor, Tensor, Tensor]]:
        del reuse_buffers
        if batch_size < 1:
            raise ValueError("batch size must be positive")
        positions = self.positions[split].copy()
        if shuffle:
            np.random.default_rng(seed).shuffle(positions)
        for start in range(0, positions.size, batch_size):
            selected = positions[start:start + batch_size]
            yield (
                torch.from_numpy(np.ascontiguousarray(self.histories[selected])),
                torch.from_numpy(np.ascontiguousarray(self.targets[selected])),
                torch.ones(selected.size, dtype=torch.float32),
            )
