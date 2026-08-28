from __future__ import annotations

from datetime import datetime, timedelta, timezone
import math
from pathlib import Path

import numpy as np

from active_return_path_dataset import CandleCloseCache


MILLISECONDS_PER_DAY = 86_400_000


def trailing_log_return_statistics(
    history_root: Path,
    origin_times_ms: np.ndarray,
    *,
    window_seconds: int,
    variance_floor: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Causal mean and variance of all 1s log returns ending by each origin.

    The window includes zero price returns.  An origin at ``t`` uses the
    returns whose candle-close boundaries lie in ``(t-window, t]`` and never
    reads a target return ending after ``t``.
    """
    times = np.asarray(origin_times_ms, dtype=np.float64)
    if times.ndim != 1 or times.size == 0 or not np.isfinite(times).all():
        raise ValueError("origin times must be a finite non-empty vector")
    if window_seconds <= 1:
        raise ValueError("normalization window must contain at least two returns")
    if not math.isfinite(variance_floor) or variance_floor <= 0:
        raise ValueError("normalization variance floor must be positive")
    rounded = np.rint(times).astype(np.int64)
    if not np.allclose(times, rounded, rtol=0, atol=1e-3):
        raise ValueError("origin times must resolve to integer milliseconds")

    earliest_required = int(rounded.min()) - window_seconds * 1_000
    first_day = datetime.fromtimestamp(
        earliest_required / 1_000, timezone.utc
    ).date()
    last_included = int(rounded.max()) - 1
    last_day = datetime.fromtimestamp(last_included / 1_000, timezone.utc).date()
    day_count = (last_day - first_day).days + 1
    if day_count <= 0:
        raise ValueError("normalization history interval is empty")

    cache = CandleCloseCache(history_root, rows_per_day=86_400, max_entries=3)
    previous_day = first_day - timedelta(days=1)
    previous_close = float(cache.load(previous_day.isoformat())[-1])
    returns_by_day: list[np.ndarray] = []
    for offset in range(day_count):
        day = first_day + timedelta(days=offset)
        closes = cache.load(day.isoformat()).astype(np.float64, copy=False)
        values = np.empty(closes.size, dtype=np.float64)
        values[0] = math.log(float(closes[0]) / previous_close)
        np.log(closes[1:] / closes[:-1], out=values[1:])
        returns_by_day.append(values)
        previous_close = float(closes[-1])
    returns = np.concatenate(returns_by_day)

    prefix = np.empty(returns.size + 1, dtype=np.float64)
    prefix_square = np.empty_like(prefix)
    prefix[0] = 0
    prefix_square[0] = 0
    np.cumsum(returns, out=prefix[1:])
    np.cumsum(np.square(returns), out=prefix_square[1:])

    first_boundary_ms = int(datetime(
        first_day.year, first_day.month, first_day.day, tzinfo=timezone.utc
    ).timestamp() * 1_000)
    stops = (rounded - first_boundary_ms) // 1_000
    starts = stops - int(window_seconds)
    if int(starts.min()) < 0 or int(stops.max()) > returns.size:
        raise ValueError("loaded candle interval does not cover a normalization window")
    sums = prefix[stops] - prefix[starts]
    squares = prefix_square[stops] - prefix_square[starts]
    means = sums / float(window_seconds)
    variances = np.maximum(
        squares / float(window_seconds) - np.square(means), variance_floor
    )
    return means.astype(np.float32), variances.astype(np.float32)


def trailing_log_price_statistics(
    history_root: Path,
    origin_times_ms: np.ndarray,
    *,
    window_seconds: int,
    variance_floor: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Causal mean and variance of completed one-second log-price levels."""
    times = np.asarray(origin_times_ms, dtype=np.float64)
    if times.ndim != 1 or times.size == 0 or not np.isfinite(times).all():
        raise ValueError("origin times must be a finite non-empty vector")
    if window_seconds <= 1:
        raise ValueError("normalization window must contain at least two prices")
    if not math.isfinite(variance_floor) or variance_floor <= 0:
        raise ValueError("normalization variance floor must be positive")
    rounded = np.rint(times).astype(np.int64)
    if not np.allclose(times, rounded, rtol=0, atol=1e-3):
        raise ValueError("origin times must resolve to integer milliseconds")

    earliest_required = int(rounded.min()) - window_seconds * 1_000
    first_day = datetime.fromtimestamp(
        earliest_required / 1_000, timezone.utc
    ).date()
    last_included = int(rounded.max()) - 1
    last_day = datetime.fromtimestamp(last_included / 1_000, timezone.utc).date()
    day_count = (last_day - first_day).days + 1
    if day_count <= 0:
        raise ValueError("normalization history interval is empty")

    cache = CandleCloseCache(history_root, rows_per_day=86_400, max_entries=3)
    log_prices = np.concatenate([
        np.log(cache.load((first_day + timedelta(days=offset)).isoformat()))
        for offset in range(day_count)
    ])
    prefix = np.empty(log_prices.size + 1, dtype=np.float64)
    prefix_square = np.empty_like(prefix)
    prefix[0] = 0
    prefix_square[0] = 0
    np.cumsum(log_prices, out=prefix[1:])
    np.cumsum(np.square(log_prices), out=prefix_square[1:])

    first_boundary_ms = int(datetime(
        first_day.year, first_day.month, first_day.day, tzinfo=timezone.utc
    ).timestamp() * 1_000)
    stops = (rounded - first_boundary_ms) // 1_000
    starts = stops - int(window_seconds)
    if int(starts.min()) < 0 or int(stops.max()) > log_prices.size:
        raise ValueError("loaded candle interval does not cover a normalization window")
    sums = prefix[stops] - prefix[starts]
    squares = prefix_square[stops] - prefix_square[starts]
    means = sums / float(window_seconds)
    variances = np.maximum(
        squares / float(window_seconds) - np.square(means), variance_floor
    )
    return means.astype(np.float32), variances.astype(np.float32)


def append_return_statistics(
    features: np.ndarray,
    means: np.ndarray,
    variances: np.ndarray,
) -> np.ndarray:
    values = np.asarray(features, dtype=np.float32)
    mean_values = np.asarray(means, dtype=np.float32)
    variance_values = np.asarray(variances, dtype=np.float32)
    if values.ndim != 2 or mean_values.shape != (values.shape[0],) \
            or variance_values.shape != mean_values.shape:
        raise ValueError("return statistics do not align with feature rows")
    return np.concatenate((
        values,
        mean_values[:, None],
        variance_values[:, None],
    ), axis=1)


def latest_log_return_statistics(
    history_root: Path,
    latest_complete_boundary_ms: int,
    *,
    window_seconds: int = 7_200,
    variance_floor: float = 1e-16,
) -> tuple[float, float]:
    """Return the mean/variance inputs to use for a live forecast origin."""
    means, variances = trailing_log_return_statistics(
        history_root,
        np.asarray([latest_complete_boundary_ms], dtype=np.float64),
        window_seconds=window_seconds,
        variance_floor=variance_floor,
    )
    return float(means[0]), float(variances[0])


def latest_log_price_statistics(
    history_root: Path,
    latest_complete_boundary_ms: int,
    *,
    window_seconds: int = 7_200,
    variance_floor: float = 1e-16,
) -> tuple[float, float]:
    """Return live mean/variance inputs for the latest log-price window."""
    means, variances = trailing_log_price_statistics(
        history_root,
        np.asarray([latest_complete_boundary_ms], dtype=np.float64),
        window_seconds=window_seconds,
        variance_floor=variance_floor,
    )
    return float(means[0]), float(variances[0])
