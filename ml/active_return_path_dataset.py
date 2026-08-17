from __future__ import annotations

from collections import OrderedDict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch import Tensor

from next_return_dataset import (
    DAY_SECONDS,
    HISTORY_RETURN_COUNT,
    ExampleShard,
    example_rows,
)
from next_return_dataset import daily_log_return_examples
from trading_storage import read_candle_column


RESOLUTION_ROWS = {"1s": DAY_SECONDS, "1m": 1_440}
RESOLUTION_STEP_MS = {"1s": 1_000, "1m": 60_000}


class CandleCloseCache:
    def __init__(
        self,
        history_root: Path,
        *,
        rows_per_day: int,
        max_entries: int = 5,
    ) -> None:
        self.history_root = history_root
        self.rows_per_day = int(rows_per_day)
        self.max_entries = int(max_entries)
        self.values: OrderedDict[str, np.ndarray] = OrderedDict()

    def load(self, day: str) -> np.ndarray:
        cached = self.values.pop(day, None)
        if cached is not None:
            self.values[day] = cached
            return cached
        file = self.history_root / f"{day}.json"
        if not file.is_file():
            raise FileNotFoundError(f"missing candle close history: {file}")
        values = read_candle_column(file, "close")
        if values.shape != (self.rows_per_day,) \
                or not np.isfinite(values).all() \
                or bool((values <= 0).any()):
            raise ValueError(f"invalid candle close history: {file}")
        self.values[day] = values
        while len(self.values) > self.max_entries:
            self.values.popitem(last=False)
        return values


def daily_candle_log_return_examples(
    previous_close: np.ndarray,
    current_close: np.ndarray,
    following_close: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build 120-candle histories and immediate next-candle returns."""
    rows_per_day = int(current_close.size)
    if rows_per_day < HISTORY_RETURN_COUNT:
        raise ValueError("a candle day cannot cover the requested history")
    for name, values in (
        ("previous", previous_close),
        ("current", current_close),
        ("following", following_close),
    ):
        if values.shape != (rows_per_day,) \
                or not np.isfinite(values).all() \
                or bool((values <= 0).any()):
            raise ValueError(f"{name} candle closes are invalid")
    close = np.concatenate((
        previous_close[-HISTORY_RETURN_COUNT:],
        current_close,
        following_close[:1],
    )).astype(np.float64, copy=False)
    returns = np.diff(np.log(close)).astype(np.float32)
    windows = np.lib.stride_tricks.sliding_window_view(
        returns, HISTORY_RETURN_COUNT + 1
    )
    if windows.shape != (rows_per_day, HISTORY_RETURN_COUNT + 1):
        raise RuntimeError("daily candle-return construction is invalid")
    return windows[:, :HISTORY_RETURN_COUNT], windows[:, -1]


def fixed_nonzero_candle_subset_shards(
    history_root: Path,
    subset_date: date,
    examples: int,
    *,
    resolution: str,
) -> dict[str, list[ExampleShard]]:
    """Select exactly N active next-candle targets at a native resolution."""
    if resolution not in RESOLUTION_ROWS:
        raise ValueError(f"unsupported active-return resolution: {resolution}")
    if examples < 1:
        raise ValueError("active-return subset examples must be positive")
    rows_per_day = RESOLUTION_ROWS[resolution]
    step_ms = RESOLUTION_STEP_MS[resolution]
    cache = CandleCloseCache(history_root, rows_per_day=rows_per_day)
    train: list[ExampleShard] = []
    remaining = int(examples)
    current = subset_date
    while remaining > 0:
        previous = cache.load((current - timedelta(days=1)).isoformat())
        today = cache.load(current.isoformat())
        following = cache.load((current + timedelta(days=1)).isoformat())
        _history, target = daily_candle_log_return_examples(
            previous, today, following
        )
        nonzero_rows = np.flatnonzero(target != 0)
        if nonzero_rows.size == 0:
            current += timedelta(days=1)
            continue
        if remaining <= nonzero_rows.size:
            candidate_count = int(nonzero_rows[remaining - 1]) + 1
            retained = remaining
        else:
            candidate_count = rows_per_day
            retained = int(nonzero_rows.size)
        timestamp = int(datetime(
            current.year, current.month, current.day, tzinfo=timezone.utc
        ).timestamp() * 1_000)
        train.append(ExampleShard(
            split="train",
            decision_time_start=timestamp,
            count=candidate_count,
            date=current.isoformat(),
            row_offset=0,
        ))
        # ExampleShard's timestamp helpers are second-specific, but this
        # dataset consumes only date, row_offset, and count. Preserve the
        # native step explicitly in the dataset contract instead.
        if step_ms < 1:
            raise RuntimeError("native candle step is invalid")
        remaining -= retained
        current += timedelta(days=1)
    return {"train": train, "validation": [], "test": []}


def next_active_return_paths(
    future_returns: np.ndarray,
    start_count: int,
    return_count: int,
) -> np.ndarray:
    """Return the next H nonzero returns at every candidate start."""
    values = np.asarray(future_returns, dtype=np.float32)
    if values.ndim != 1 or start_count < 1 or start_count > values.size:
        raise ValueError("active-return source range is invalid")
    if isinstance(return_count, bool) or int(return_count) < 1:
        raise ValueError("active-return path length must be positive")
    return_count = int(return_count)
    active = np.flatnonzero(values != 0)
    starts = np.searchsorted(active, np.arange(start_count), side="left")
    requested = starts[:, None] + np.arange(return_count)[None, :]
    if requested.size == 0 or int(requested.max()) >= active.size:
        raise ValueError("future candles do not cover every active-return path")
    return values[active[requested]]


class ActiveReturnPathDataset:
    """Raw 120-candle histories paired with the next H active returns.

    Candidate rows whose immediate next return is zero are excluded, exactly
    matching the clean-count single-return datasets. Later zero-return seconds
    are skipped while collecting the remaining active returns in the path.
    """

    def __init__(
        self,
        shards: dict[str, list[ExampleShard]],
        history_root: Path,
        *,
        return_count: int,
        resolution: str = "1s",
    ) -> None:
        if isinstance(return_count, bool) or int(return_count) < 2:
            raise ValueError("active-return path datasets require H >= 2")
        self.shards = shards
        self.return_count = int(return_count)
        if resolution not in RESOLUTION_ROWS:
            raise ValueError(f"unsupported active-return resolution: {resolution}")
        self.resolution = resolution
        self.rows_per_day = RESOLUTION_ROWS[resolution]
        self.step_ms = RESOLUTION_STEP_MS[resolution]
        self.close_cache = CandleCloseCache(
            history_root, rows_per_day=self.rows_per_day
        )
        self.component_cache: OrderedDict[
            str, tuple[np.ndarray, np.ndarray, np.ndarray]
        ] = OrderedDict()

    @property
    def horizon_return_count(self) -> int:
        """Expose the sequence-dataset interface used by direct predictors."""
        return self.return_count

    def _component(
        self, day: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        cached = self.component_cache.pop(day, None)
        if cached is not None:
            self.component_cache[day] = cached
            return cached
        current_date = date.fromisoformat(day)
        previous = self.close_cache.load(
            (current_date - timedelta(days=1)).isoformat()
        )
        current = self.close_cache.load(day)
        following = self.close_cache.load(
            (current_date + timedelta(days=1)).isoformat()
        )
        if self.resolution == "1s":
            history, immediate = daily_log_return_examples(
                previous, current, following, horizon_return_count=1
            )
        else:
            history, immediate = daily_candle_log_return_examples(
                previous, current, following
            )
        future = np.diff(np.log(np.concatenate((current, following)))) \
            .astype(np.float32)
        paths = next_active_return_paths(
            future, self.rows_per_day, self.return_count
        )
        value = history, paths, immediate
        self.component_cache[day] = value
        while len(self.component_cache) > 3:
            self.component_cache.popitem(last=False)
        return value

    def logical_count(self, split: str) -> int:
        count = 0
        for shard in self.shards[split]:
            _history, _paths, immediate = self._component(shard.date)
            rows = shard.row_offset + np.arange(shard.count, dtype=np.int64)
            count += int(np.count_nonzero(immediate[rows] != 0))
        return count

    def iter_batches(
        self,
        split: str,
        batch_size: int,
        *,
        shuffle: bool,
        seed: int,
        shuffle_rows: bool = False,
        reuse_buffers: bool = False,
    ) -> Iterator[tuple[Tensor, Tensor, Tensor]]:
        if batch_size < 1:
            raise ValueError("batch size must be positive")
        generator = np.random.default_rng(seed)
        shards = list(self.shards[split])
        if shuffle:
            generator.shuffle(shards)
        feature_buffer = np.empty(
            (batch_size, HISTORY_RETURN_COUNT), dtype=np.float32
        )
        target_buffer = np.empty(
            (batch_size, self.return_count), dtype=np.float32
        )
        weight_buffer = np.ones(batch_size, dtype=np.float32)
        filled = 0
        for shard in shards:
            history, targets, immediate = self._component(shard.date)
            rows, _weights = example_rows(shard.row_offset, shard.count)
            rows = rows[immediate[rows] != 0]
            if rows.size == 0:
                continue
            if shuffle_rows:
                rows = rows[generator.permutation(rows.size)]
            position = 0
            while position < rows.size:
                take = min(batch_size - filled, rows.size - position)
                source = rows[position:position + take]
                destination = slice(filled, filled + take)
                feature_buffer[destination] = history[source]
                target_buffer[destination] = targets[source]
                filled += take
                position += take
                if filled == batch_size:
                    yield (
                        torch.from_numpy(feature_buffer),
                        torch.from_numpy(target_buffer),
                        torch.from_numpy(weight_buffer),
                    )
                    if not reuse_buffers:
                        feature_buffer = np.empty_like(feature_buffer)
                        target_buffer = np.empty_like(target_buffer)
                        weight_buffer = np.ones_like(weight_buffer)
                    filled = 0
        if filled:
            yield (
                torch.from_numpy(feature_buffer[:filled]),
                torch.from_numpy(target_buffer[:filled]),
                torch.from_numpy(weight_buffer[:filled]),
            )
