from __future__ import annotations

from collections import OrderedDict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch import Tensor

from multiscale_next_return import (
    WINDOW_SECONDS,
    component_labels,
    selected_windows,
    telescoping_components,
)
from next_return_dataset import (
    DAY_SECONDS,
    HISTORY_RETURN_COUNT,
    ROWS_PER_DAY,
    SECOND_MS,
    ExampleShard,
    example_rows,
    validate_horizon_return_count,
)
from trading_storage import read_candle_column


def trim_shards_for_moving_average_history(
    shards: dict[str, list[ExampleShard]],
    *,
    first_history_day: str,
    max_window: str,
) -> dict[str, list[ExampleShard]]:
    first_close = datetime.combine(
        date.fromisoformat(first_history_day), datetime.min.time(), timezone.utc
    )
    earliest_decision_ms = int(first_close.timestamp() * 1_000) + 999 \
        + (WINDOW_SECONDS[max_window] + HISTORY_RETURN_COUNT - 1) * SECOND_MS
    result: dict[str, list[ExampleShard]] = {}
    for split, values in shards.items():
        chosen: list[ExampleShard] = []
        for shard in values:
            if shard.decision_time_end < earliest_decision_ms:
                continue
            offset = max(0, (
                earliest_decision_ms - shard.decision_time_start + SECOND_MS - 1
            ) // SECOND_MS)
            chosen.append(shard.shifted(offset) if offset else shard)
        result[split] = chosen
    if not result.get("train") or not result.get("validation"):
        raise ValueError("moving-average lookback removed a required split")
    return result


class CloseRangeCache:
    def __init__(self, history_root: Path, max_entries: int = 2_048) -> None:
        self.history_root = history_root
        self.max_entries = max_entries
        self.values: OrderedDict[str, np.ndarray] = OrderedDict()

    def load_day(self, day: str) -> np.ndarray:
        cached = self.values.pop(day, None)
        if cached is not None:
            self.values[day] = cached
            return cached
        file = self.history_root / f"{day}.json"
        if not file.is_file():
            raise FileNotFoundError(f"missing one-second close history: {file}")
        values = read_candle_column(file, "close")
        if values.shape != (DAY_SECONDS,) \
                or not np.isfinite(values).all() \
                or bool((values <= 0).any()):
            raise ValueError(f"invalid one-second close history: {file}")
        self.values[day] = values
        while len(self.values) > self.max_entries:
            self.values.popitem(last=False)
        return values

    def load_range(self, start: datetime, count: int) -> np.ndarray:
        if start.tzinfo is None or start.microsecond or count < 1:
            raise ValueError("close range must be aligned to UTC seconds")
        result = np.empty(count, dtype=np.float64)
        position = 0
        cursor = start.astimezone(timezone.utc)
        while position < count:
            offset = cursor.hour * 3_600 + cursor.minute * 60 + cursor.second
            take = min(count - position, DAY_SECONDS - offset)
            values = self.load_day(cursor.date().isoformat())
            result[position:position + take] = values[offset:offset + take]
            position += take
            cursor += timedelta(seconds=take)
        return result


def daily_multiscale_return_examples(
    close_cache: CloseRangeCache,
    day: str,
    *,
    max_window: str,
    horizon_return_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    horizon = validate_horizon_return_count(horizon_return_count)
    start = datetime.combine(
        date.fromisoformat(day), datetime.min.time(), timezone.utc
    ) - timedelta(seconds=HISTORY_RETURN_COUNT - 1)
    count = HISTORY_RETURN_COUNT + ROWS_PER_DAY + horizon - 1
    end_close = close_cache.load_range(start, count)
    log_end = np.log(end_close)
    raw_returns = log_end - np.log(
        close_cache.load_range(start - timedelta(seconds=1), count)
    )
    moving_averages = {
        label: (
            log_end - np.log(close_cache.load_range(
                start - timedelta(seconds=seconds), count
            ))
        ) / seconds
        for label, seconds in selected_windows(max_window)
    }
    components = telescoping_components(
        raw_returns, moving_averages, max_window
    )
    windows = np.lib.stride_tricks.sliding_window_view(
        components,
        HISTORY_RETURN_COUNT + horizon,
        axis=0,
    )
    expected = (
        ROWS_PER_DAY,
        len(component_labels(max_window)),
        HISTORY_RETURN_COUNT + horizon,
    )
    if windows.shape != expected:
        raise RuntimeError(
            f"multiscale daily window shape {windows.shape} != {expected}"
        )
    return (
        windows[:, :, :HISTORY_RETURN_COUNT],
        windows[:, :, HISTORY_RETURN_COUNT:],
    )


class MultiscaleNextReturnDataset:
    def __init__(
        self,
        shards: dict[str, list[ExampleShard]],
        history_root: Path,
        *,
        max_window: str,
        horizon_return_count: int,
    ) -> None:
        self.shards = shards
        self.max_window = max_window
        self.horizon_return_count = validate_horizon_return_count(
            horizon_return_count
        )
        self.component_labels = component_labels(max_window)
        self.component_count = len(self.component_labels)
        self.close_cache = CloseRangeCache(history_root)
        self.component_cache: OrderedDict[
            str, tuple[np.ndarray, np.ndarray]
        ] = OrderedDict()
        self.component_cache_entries = 8

    def logical_count(self, split: str) -> int:
        return sum(shard.count for shard in self.shards[split])

    def _component(self, day: str) -> tuple[np.ndarray, np.ndarray]:
        cached = self.component_cache.pop(day, None)
        if cached is not None:
            self.component_cache[day] = cached
            return cached
        values = daily_multiscale_return_examples(
            self.close_cache,
            day,
            max_window=self.max_window,
            horizon_return_count=self.horizon_return_count,
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
        target_shape = (
            batch_size, self.component_count, self.horizon_return_count
        )
        feature_buffer = np.empty(feature_shape, dtype=np.float32)
        target_buffer = np.empty(target_shape, dtype=np.float32)
        weight_buffer = np.empty(batch_size, dtype=np.float32)
        filled = 0
        for shard in shards:
            history, target = self._component(shard.date)
            rows, weights = example_rows(shard.row_offset, shard.count)
            if rows[0] < 0 or rows[-1] >= ROWS_PER_DAY:
                raise IndexError("one-second row falls outside its UTC day")
            position = 0
            while position < rows.shape[0]:
                take = min(batch_size - filled, rows.shape[0] - position)
                selection = slice(int(rows[position]), int(rows[position]) + take)
                destination = slice(filled, filled + take)
                feature_buffer[destination] = history[selection]
                target_buffer[destination] = target[selection]
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
                        feature_buffer = np.empty(feature_shape, dtype=np.float32)
                        target_buffer = np.empty(target_shape, dtype=np.float32)
                        weight_buffer = np.empty(batch_size, dtype=np.float32)
                    filled = 0
        if filled:
            yield (
                torch.from_numpy(feature_buffer[:filled]),
                torch.from_numpy(target_buffer[:filled]),
                torch.from_numpy(weight_buffer[:filled]),
            )
