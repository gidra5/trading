"""Strictly causal features from Binance USD-M five-minute positioning metrics."""

from __future__ import annotations

import numpy as np


DAY_ROWS = 1_440
METRIC_ROWS = 288
MINUTES_PER_METRIC = 5
AVAILABILITY_LAG_ROWS = 1
HORIZONS = (1, 3, 12, 48)
HORIZON_LABELS = {1: "5m", 3: "15m", 12: "1h", 48: "4h"}
METRIC_COLUMNS = (
    "sumOpenInterest",
    "sumOpenInterestValue",
    "topTraderAccountLongShortRatio",
    "topTraderPositionLongShortRatio",
    "globalLongShortRatio",
    "takerBuySellVolumeRatio",
)
RATIO_COLUMNS = METRIC_COLUMNS[2:]
PRICE_STREAMS = (
    "openInterest",
    "openInterestValue",
    "impliedMarkPrice",
)

FUTURES_FEATURE_NAMES = tuple(
    f"{prefix}LogChange{HORIZON_LABELS[horizon]}"
    for prefix in PRICE_STREAMS
    for horizon in HORIZONS
) + tuple(
    f"abs{prefix[0].upper()}{prefix[1:]}LogChange{HORIZON_LABELS[horizon]}"
    for prefix in PRICE_STREAMS
    for horizon in HORIZONS
) + tuple(
    name
    for source in RATIO_COLUMNS
    for name in (
        f"{source}LogLevel",
        f"abs{source[0].upper()}{source[1:]}LogLevel",
        f"{source}LogChange5m",
        f"abs{source[0].upper()}{source[1:]}LogChange5m",
        f"{source}LogChange1h",
        f"abs{source[0].upper()}{source[1:]}LogChange1h",
        f"{source}LogChange4h",
        f"abs{source[0].upper()}{source[1:]}LogChange4h",
        f"{source}LogDeviation24h",
        f"abs{source[0].upper()}{source[1:]}LogDeviation24h",
    )
) + (
    "topTraderPositionMinusAccountLog",
    "absTopTraderPositionMinusAccountLog",
    "topTraderAccountMinusGlobalLog",
    "absTopTraderAccountMinusGlobalLog",
) + tuple(
    name
    for source in METRIC_COLUMNS
    for name in (
        f"{source}CurrentObserved",
        f"{source}ObservationAge24h",
    )
)


def causal_futures_metrics_features(
    previous_values: dict[str, np.ndarray],
    previous_validity: dict[str, np.ndarray],
    current_values: dict[str, np.ndarray],
    current_validity: dict[str, np.ndarray],
) -> np.ndarray:
    """Build minute rows using the latest metric observation at least 5m old."""
    _validate_day(previous_values, previous_validity)
    _validate_day(current_values, current_validity)
    sample_indexes = (
        METRIC_ROWS - AVAILABILITY_LAG_ROWS
        + np.arange(DAY_ROWS, dtype=np.int64) // MINUTES_PER_METRIC
    )
    streams: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for name in METRIC_COLUMNS:
        values = np.concatenate((previous_values[name], current_values[name])).astype(
            np.float64, copy=False,
        )
        valid = np.concatenate((
            previous_validity[name], current_validity[name],
        )).astype(bool, copy=False)
        streams[name] = causal_fill(values, valid)

    quantity = streams["sumOpenInterest"]
    value = streams["sumOpenInterestValue"]
    implied_values = np.divide(
        value[0], quantity[0],
        # Nullable storage requires invalid slots to remain exactly zero.
        # `causal_fill` will introduce its internal neutral value only after
        # consulting the validity mask.
        out=np.zeros_like(value[0]),
        where=(value[1] & quantity[1]),
    )
    implied_valid = value[1] & quantity[1]
    implied = causal_fill(implied_values, implied_valid)
    price_streams = {
        "openInterest": quantity,
        "openInterestValue": value,
        "impliedMarkPrice": implied,
    }

    columns: list[np.ndarray] = []
    signed_price_changes: list[np.ndarray] = []
    for prefix in PRICE_STREAMS:
        stream = price_streams[prefix]
        for horizon in HORIZONS:
            signed_price_changes.append(
                sampled_log_change(stream, sample_indexes, horizon)
            )
    columns.extend(signed_price_changes)
    columns.extend(np.abs(values) for values in signed_price_changes)

    ratio_logs: dict[str, np.ndarray] = {}
    for name in RATIO_COLUMNS:
        stream = streams[name]
        log_values = np.log(stream[0])
        ratio_logs[name] = log_values
        level = np.where(stream[1][sample_indexes], log_values[sample_indexes], 0)
        change5m = sampled_log_change(stream, sample_indexes, 1)
        change1h = sampled_log_change(stream, sample_indexes, 12)
        change4h = sampled_log_change(stream, sample_indexes, 48)
        deviation24h = sampled_log_deviation(
            log_values, stream[1], sample_indexes, 288,
        )
        for values in (level, change5m, change1h, change4h, deviation24h):
            columns.extend((values, np.abs(values)))

    top_position_minus_account = sampled_difference(
        ratio_logs["topTraderPositionLongShortRatio"],
        streams["topTraderPositionLongShortRatio"][1],
        ratio_logs["topTraderAccountLongShortRatio"],
        streams["topTraderAccountLongShortRatio"][1],
        sample_indexes,
    )
    top_account_minus_global = sampled_difference(
        ratio_logs["topTraderAccountLongShortRatio"],
        streams["topTraderAccountLongShortRatio"][1],
        ratio_logs["globalLongShortRatio"],
        streams["globalLongShortRatio"][1],
        sample_indexes,
    )
    columns.extend((
        top_position_minus_account,
        np.abs(top_position_minus_account),
        top_account_minus_global,
        np.abs(top_account_minus_global),
    ))

    for name in METRIC_COLUMNS:
        _filled, ever_observed, current_observed, age = streams[name]
        columns.extend((
            current_observed[sample_indexes].astype(np.float64),
            np.minimum(age[sample_indexes], 288) / 288,
        ))

    result = np.column_stack(columns).astype(np.float32, copy=False)
    if result.shape != (DAY_ROWS, len(FUTURES_FEATURE_NAMES)) \
            or not np.isfinite(result).all():
        raise ValueError("invalid causal futures-metrics feature matrix")
    return result


def causal_fill(
    values: np.ndarray,
    valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Forward-fill only from observations already seen; never back-fill."""
    values = np.asarray(values, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    if values.ndim != 1 or valid.shape != values.shape \
            or not np.isfinite(values).all() \
            or bool((values[valid] <= 0).any()) \
            or bool((values[~valid] != 0).any()):
        raise ValueError("invalid nullable futures-metrics stream")
    indexes = np.arange(values.size, dtype=np.int64)
    last = np.maximum.accumulate(np.where(valid, indexes, -1))
    ever = last >= 0
    filled = np.ones(values.size, dtype=np.float64)
    filled[ever] = values[last[ever]]
    age = np.where(ever, indexes - last, values.size).astype(np.float64)
    return filled, ever, valid, age


def sampled_log_change(
    stream: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    sample_indexes: np.ndarray,
    horizon: int,
) -> np.ndarray:
    values, ever, _current, _age = stream
    previous = sample_indexes - horizon
    available = ever[sample_indexes] & ever[previous]
    result = np.zeros(sample_indexes.size, dtype=np.float64)
    result[available] = np.log(
        values[sample_indexes[available]] / values[previous[available]]
    )
    return result


def sampled_log_deviation(
    log_values: np.ndarray,
    ever: np.ndarray,
    sample_indexes: np.ndarray,
    horizon: int,
) -> np.ndarray:
    weighted = np.where(ever, log_values, 0)
    cumulative = np.concatenate(([0.0], np.cumsum(weighted)))
    counts = np.concatenate(([0], np.cumsum(ever.astype(np.int64))))
    starts = np.maximum(sample_indexes - horizon + 1, 0)
    sums = cumulative[sample_indexes + 1] - cumulative[starts]
    count = counts[sample_indexes + 1] - counts[starts]
    means = np.divide(sums, count, out=np.zeros_like(sums), where=count > 0)
    return np.where(
        ever[sample_indexes] & (count > 0),
        log_values[sample_indexes] - means,
        0,
    )


def sampled_difference(
    left: np.ndarray,
    left_ever: np.ndarray,
    right: np.ndarray,
    right_ever: np.ndarray,
    sample_indexes: np.ndarray,
) -> np.ndarray:
    available = left_ever[sample_indexes] & right_ever[sample_indexes]
    result = np.zeros(sample_indexes.size, dtype=np.float64)
    result[available] = (
        left[sample_indexes[available]] - right[sample_indexes[available]]
    )
    return result


def _validate_day(
    values: dict[str, np.ndarray],
    validity: dict[str, np.ndarray],
) -> None:
    if set(values) != set(METRIC_COLUMNS) or set(validity) != set(METRIC_COLUMNS):
        raise ValueError("futures-metrics columns differ from the feature contract")
    for name in METRIC_COLUMNS:
        if np.asarray(values[name]).shape != (METRIC_ROWS,) \
                or np.asarray(validity[name]).shape != (METRIC_ROWS,):
            raise ValueError(f"invalid futures-metrics day column: {name}")
