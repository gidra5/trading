from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


SECOND_MS = 1_000
DAY_SECONDS = 86_400
ROWS_PER_DAY = DAY_SECONDS
HISTORY_RETURN_COUNT = 120
TARGET_RETURN_COUNT = 1
EXAMPLE_SPAN_MS = (HISTORY_RETURN_COUNT + TARGET_RETURN_COUNT) * SECOND_MS
TEST_EXAMPLE_COUNT = 1_000_000


def validate_horizon_return_count(horizon_return_count: int) -> int:
    value = int(horizon_return_count)
    if value < 1 or value > DAY_SECONDS:
        raise ValueError("return horizon must contain between 1 and 86,400 seconds")
    return value


def example_span_ms(horizon_return_count: int) -> int:
    return (
        HISTORY_RETURN_COUNT
        + validate_horizon_return_count(horizon_return_count)
    ) * SECOND_MS


def feature_contract(horizon_return_count: int) -> str:
    horizon = validate_horizon_return_count(horizon_return_count)
    return (
        "past-120-completed-second-close-log-returns-to-next-"
        f"{horizon}-completed-second-close-log-returns-v2"
    )


def split_contract(horizon_return_count: int) -> str:
    horizon = validate_horizon_return_count(horizon_return_count)
    span_seconds = HISTORY_RETURN_COUNT + horizon
    return (
        "oracle-source-split-assignments-with-"
        f"{span_seconds}-second-cross-split-purge-and-chronological-million-"
        "example-test-tail-v2"
    )


# Horizon-one aliases keep the linear baseline and its frozen fingerprint
# directly comparable while every new sequence model uses the same parameterized
# construction.
FEATURE_CONTRACT = (
    "past-120-completed-second-close-log-returns-to-next-completed-second-"
    "close-log-return-v1"
)
SPLIT_CONTRACT = (
    "oracle-source-split-assignments-with-121-second-cross-split-purge-and-"
    "chronological-million-example-test-tail-v1"
)


@dataclass(frozen=True)
class ExampleShard:
    split: str
    decision_time_start: int
    count: int
    date: str
    row_offset: int

    @property
    def decision_time_end(self) -> int:
        return self.decision_time_start + (self.count - 1) * SECOND_MS

    def shifted(self, local_offset: int) -> ExampleShard:
        if local_offset < 0 or local_offset >= self.count:
            raise ValueError("example-shard offset must select at least one row")
        return ExampleShard(
            split=self.split,
            decision_time_start=self.decision_time_start + local_offset * SECOND_MS,
            count=self.count - local_offset,
            date=self.date,
            row_offset=self.row_offset + local_offset,
        )


def example_rows(row_offset: int, count: int) -> tuple[np.ndarray, np.ndarray]:
    """Return every distinct one-second row with unit weight."""
    if row_offset < 0 or count < 1:
        raise ValueError("one-second example range is invalid")
    return (
        row_offset + np.arange(count, dtype=np.int64),
        np.ones(count, dtype=np.float32),
    )


def daily_log_return_examples(
    previous_close: np.ndarray,
    current_close: np.ndarray,
    following_close: np.ndarray,
    *,
    horizon_return_count: int = TARGET_RETURN_COUNT,
) -> tuple[np.ndarray, np.ndarray]:
    """Build strided histories and the next T one-second return targets."""
    horizon = validate_horizon_return_count(horizon_return_count)
    for name, values in (
        ("previous", previous_close),
        ("current", current_close),
        ("following", following_close),
    ):
        if values.shape != (DAY_SECONDS,) \
                or not np.isfinite(values).all() \
                or bool((values <= 0).any()):
            raise ValueError(f"{name} daily closes must contain 86,400 positives")

    close = np.concatenate((
        previous_close[-HISTORY_RETURN_COUNT:],
        current_close,
        following_close[:horizon],
    )).astype(np.float64, copy=False)
    returns = np.diff(np.log(close)).astype(np.float32)
    windows = np.lib.stride_tricks.sliding_window_view(
        returns,
        HISTORY_RETURN_COUNT + horizon,
    )
    if windows.shape != (
        ROWS_PER_DAY,
        HISTORY_RETURN_COUNT + horizon,
    ) or not np.isfinite(returns).all():
        raise RuntimeError("daily close-return construction is invalid")
    targets = windows[:, HISTORY_RETURN_COUNT:]
    return (
        windows[:, :HISTORY_RETURN_COUNT],
        targets[:, 0] if horizon == 1 else targets,
    )


def daily_causal_volatility(
    previous_close: np.ndarray,
    current_close: np.ndarray,
    *,
    window: int,
) -> np.ndarray:
    """Return input-only RMS scales ending at each current-day close.

    Row ``t`` contains the RMS of the ``window`` completed log returns ending
    at ``current_close[t]``.  The immediately following return, which is the
    prediction target, is therefore never part of the scale.
    """
    if isinstance(window, bool) or not 1 <= int(window) <= DAY_SECONDS:
        raise ValueError("causal volatility window must be in [1, 86,400]")
    window = int(window)
    for name, values in (("previous", previous_close), ("current", current_close)):
        if values.shape != (DAY_SECONDS,) \
                or not np.isfinite(values).all() \
                or bool((values <= 0).any()):
            raise ValueError(f"{name} daily closes must contain 86,400 positives")

    close = np.concatenate((
        previous_close[-window:],
        current_close,
    )).astype(np.float64, copy=False)
    squared_returns = np.square(np.diff(np.log(close)), dtype=np.float64)
    cumulative = np.empty(squared_returns.size + 1, dtype=np.float64)
    cumulative[0] = 0
    np.cumsum(squared_returns, out=cumulative[1:])
    mean_square = (cumulative[window:] - cumulative[:-window]) / window
    if mean_square.shape != (ROWS_PER_DAY,) or bool((mean_square < 0).any()):
        raise RuntimeError("daily causal-volatility construction is invalid")
    return np.sqrt(np.maximum(mean_square, 0)).astype(np.float32)


def _source_shards(source_manifest: dict) -> list[ExampleShard]:
    result: list[ExampleShard] = []
    for value in source_manifest.get("shards", ()):
        split = str(value.get("split", ""))
        if split not in {"train", "validation", "test"}:
            continue
        if int(value.get("featureRowStride", 0)) != 1:
            raise ValueError("next-return model requires contiguous source rows")
        result.append(ExampleShard(
            split=split,
            decision_time_start=int(value["predictionTimeStart"]),
            count=int(value["count"]),
            date=str(value["date"]),
            row_offset=int(value["featureRowOffset"]),
        ))
    result.sort(key=lambda shard: shard.decision_time_start)
    return result


def take_tail(
    shards: list[ExampleShard],
    count: int,
    *,
    offset: int = 0,
) -> list[ExampleShard]:
    if count < 1 or offset < 0:
        raise ValueError("test-tail count must be positive and offset non-negative")
    remaining = count
    remaining_offset = offset
    result: list[ExampleShard] = []
    for shard in reversed(shards):
        available = shard.count
        skipped = min(available, remaining_offset)
        available -= skipped
        remaining_offset -= skipped
        selected = min(available, remaining)
        if selected:
            local_offset = available - selected
            result.append(ExampleShard(
                split=shard.split,
                decision_time_start=(
                    shard.decision_time_start + local_offset * SECOND_MS
                ),
                count=selected,
                date=shard.date,
                row_offset=shard.row_offset + local_offset,
            ))
            remaining -= selected
        if remaining == 0:
            break
    if remaining_offset or remaining:
        missing = remaining_offset + remaining
        raise ValueError(f"test split has {missing:,} fewer examples than requested")
    result.reverse()
    return result


def select_example_shards(
    source_manifest: dict,
    *,
    test_count: int = TEST_EXAMPLE_COUNT,
    test_tail_offset: int = 0,
    horizon_return_count: int = TARGET_RETURN_COUNT,
    cross_split_purge_ms: int | None = None,
) -> dict[str, list[ExampleShard]]:
    """Apply oracle-style split selection with a leakage-safe 120+T embargo."""
    horizon = validate_horizon_return_count(horizon_return_count)
    minimum_purge_ms = example_span_ms(horizon)
    purge_ms = minimum_purge_ms \
        if cross_split_purge_ms is None else int(cross_split_purge_ms)
    if purge_ms < minimum_purge_ms:
        raise ValueError("cross-split purge must cover input history and target")
    selected: dict[str, list[ExampleShard]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    selected_end: dict[str, int | None] = {
        "train": None,
        "validation": None,
        "test": None,
    }
    previous_source_end: int | None = None
    for shard in _source_shards(source_manifest):
        if previous_source_end is not None \
                and shard.decision_time_start <= previous_source_end:
            raise ValueError("source example timestamps overlap")
        other_ends = tuple(
            end
            for split, end in selected_end.items()
            if split != shard.split and end is not None
        )
        local_offset = 0
        if other_ends:
            earliest = max(other_ends) + purge_ms + SECOND_MS
            if shard.decision_time_start < earliest:
                local_offset = math.ceil(
                    (earliest - shard.decision_time_start) / SECOND_MS
                )
        if local_offset < shard.count:
            chosen = shard.shifted(local_offset) if local_offset else shard
            selected[shard.split].append(chosen)
            selected_end[shard.split] = chosen.decision_time_end
        previous_source_end = shard.decision_time_end

    if not selected["train"] or not selected["validation"]:
        raise ValueError("next-return train and validation splits must be non-empty")
    selected["test"] = take_tail(
        selected["test"], test_count, offset=test_tail_offset
    )
    validate_split_disjointness(
        selected, horizon_return_count=horizon
    )
    return selected


def validate_split_disjointness(
    shards: dict[str, list[ExampleShard]],
    *,
    horizon_return_count: int = TARGET_RETURN_COUNT,
) -> None:
    horizon = validate_horizon_return_count(horizon_return_count)
    ranges: list[tuple[int, int, str]] = []
    for split, values in shards.items():
        for shard in values:
            ranges.append((
                shard.decision_time_start - HISTORY_RETURN_COUNT * SECOND_MS,
                shard.decision_time_end + horizon * SECOND_MS,
                split,
            ))
    ranges.sort()
    for index, (start, end, split) in enumerate(ranges):
        for other_start, other_end, other_split in ranges[index + 1:]:
            if other_start > end:
                break
            if split != other_split and start <= other_end:
                raise ValueError("train, validation, and test candle spans overlap")


def count_examples(shards: dict[str, list[ExampleShard]]) -> dict[str, int]:
    return {
        split: sum(shard.count for shard in values)
        for split, values in shards.items()
    }
