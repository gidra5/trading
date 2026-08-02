"""Strictly causal completed-minute USD-M basis and flow features."""

from __future__ import annotations

import numpy as np


DAY_ROWS = 1_440
FUTURES_KLINE_COLUMNS = (
    "open",
    "high",
    "low",
    "close",
    "baseVolume",
    "quoteVolume",
    "tradeCount",
    "takerBuyBaseVolume",
    "takerBuyQuoteVolume",
)
HORIZONS = (1, 5, 15, 60, 240, 1_440)
FLOW_HORIZONS = (1, 5, 15, 60)
SURPRISE_HORIZONS = (1, 5, 15)
HORIZON_LABELS = {
    1: "1m",
    5: "5m",
    15: "15m",
    60: "1h",
    240: "4h",
    1_440: "24h",
}


def _absolute_name(name: str) -> str:
    return f"abs{name[0].upper()}{name[1:]}"


BASIS_SIGNED_FEATURE_NAMES = (
    "basisLogLevel",
) + tuple(
    f"basisLogChange{HORIZON_LABELS[horizon]}" for horizon in HORIZONS
) + tuple(
    f"basisLogDeviation{HORIZON_LABELS[horizon]}"
    for horizon in HORIZONS
    if horizon > 1
)
FUTURES_RETURN_SIGNED_FEATURE_NAMES = tuple(
    f"futuresLogReturn{HORIZON_LABELS[horizon]}" for horizon in HORIZONS
)
FUTURES_FLOW_SIGNED_FEATURE_NAMES = tuple(
    f"futuresTakerQuoteImbalance{HORIZON_LABELS[horizon]}"
    for horizon in FLOW_HORIZONS
)
SIGNED_FUTURES_BASIS_FEATURE_NAMES = (
    BASIS_SIGNED_FEATURE_NAMES
    + FUTURES_RETURN_SIGNED_FEATURE_NAMES
    + FUTURES_FLOW_SIGNED_FEATURE_NAMES
)

FUTURES_BASIS_FEATURE_NAMES = (
    BASIS_SIGNED_FEATURE_NAMES
    + tuple(_absolute_name(name) for name in BASIS_SIGNED_FEATURE_NAMES)
    + FUTURES_RETURN_SIGNED_FEATURE_NAMES
    + tuple(_absolute_name(name) for name in FUTURES_RETURN_SIGNED_FEATURE_NAMES)
    + tuple(
        f"futuresMeanLogRange{HORIZON_LABELS[horizon]}" for horizon in HORIZONS
    )
    + FUTURES_FLOW_SIGNED_FEATURE_NAMES
    + tuple(_absolute_name(name) for name in FUTURES_FLOW_SIGNED_FEATURE_NAMES)
    + tuple(
        f"futuresLogQuoteRate{HORIZON_LABELS[horizon]}Vs1h"
        for horizon in SURPRISE_HORIZONS
    )
    + tuple(
        f"futuresLogTradeRate{HORIZON_LABELS[horizon]}Vs1h"
        for horizon in SURPRISE_HORIZONS
    )
    + tuple(
        f"futuresSpotLogApproxQuoteVolumeRatio{HORIZON_LABELS[horizon]}"
        for horizon in FLOW_HORIZONS
    )
    + (
        "futuresRowCurrentObserved",
        "futuresNoTradeCurrent",
        "futuresPriceCurrentObserved",
        "futuresPriceObservationAge24h",
        "basisCurrentObserved",
        "basisObservationAge24h",
    )
)


def causal_futures_basis_features(
    previous_futures_values: dict[str, np.ndarray],
    previous_futures_validity: np.ndarray,
    current_futures_values: dict[str, np.ndarray],
    current_futures_validity: np.ndarray,
    previous_spot_ohlcv: np.ndarray,
    current_spot_ohlcv: np.ndarray,
) -> np.ndarray:
    """Build one feature row per target minute from candles closed before it.

    Target row ``k`` is timestamped at minute ``k + 999ms``.  Its latest
    eligible futures and Spot candle is therefore ``k - 1``; current-day
    candle zero first appears in target row one.  A futures close is considered
    a fresh price observation only when its source row exists and has trades.
    """
    _validate_futures_day(previous_futures_values, previous_futures_validity)
    _validate_futures_day(current_futures_values, current_futures_validity)
    _validate_spot_day(previous_spot_ohlcv)
    _validate_spot_day(current_spot_ohlcv)

    futures = {
        name: np.concatenate((
            np.asarray(previous_futures_values[name], dtype=np.float64),
            np.asarray(current_futures_values[name], dtype=np.float64),
        ))
        for name in FUTURES_KLINE_COLUMNS
    }
    row_valid = np.concatenate((
        np.asarray(previous_futures_validity, dtype=bool),
        np.asarray(current_futures_validity, dtype=bool),
    ))
    spot = np.concatenate((
        np.asarray(previous_spot_ohlcv, dtype=np.float64),
        np.asarray(current_spot_ohlcv, dtype=np.float64),
    ))
    sample_indexes = DAY_ROWS - 1 + np.arange(DAY_ROWS, dtype=np.int64)

    trade_count = futures["tradeCount"]
    live = row_valid & (trade_count > 0)
    no_trade = row_valid & ~live

    price = _causal_fill(futures["close"], live, missing_value=1.0)
    raw_basis = np.zeros(row_valid.size, dtype=np.float64)
    raw_basis[live] = np.log(
        futures["close"][live] / spot[live, 3]
    )
    basis = _causal_fill(raw_basis, live, missing_value=0.0)

    basis_signed = [
        np.where(basis[1][sample_indexes], basis[0][sample_indexes], 0.0),
    ]
    basis_signed.extend(
        _sampled_difference(basis[0], basis[1], sample_indexes, horizon)
        for horizon in HORIZONS
    )
    basis_signed.extend(
        _sampled_deviation(basis[0], basis[1], sample_indexes, horizon)
        for horizon in HORIZONS
        if horizon > 1
    )

    futures_returns = [
        _sampled_log_change(price[0], price[1], sample_indexes, horizon)
        for horizon in HORIZONS
    ]

    log_range = np.zeros(row_valid.size, dtype=np.float64)
    log_range[row_valid] = np.log(
        futures["high"][row_valid] / futures["low"][row_valid]
    )
    range_controls = [
        _rolling_observed_mean(log_range, row_valid, sample_indexes, horizon)
        for horizon in HORIZONS
    ]

    quote_volume = np.where(row_valid, futures["quoteVolume"], 0.0)
    taker_buy_quote = np.where(
        row_valid, futures["takerBuyQuoteVolume"], 0.0,
    )
    trade_activity = np.where(row_valid, trade_count, 0.0)
    flow_signed = [
        _rolling_imbalance(
            quote_volume, taker_buy_quote, sample_indexes, horizon,
        )
        for horizon in FLOW_HORIZONS
    ]

    quote_baseline = _rolling_observed_mean(
        quote_volume, row_valid, sample_indexes, 60,
    )
    trade_baseline = _rolling_observed_mean(
        trade_activity, row_valid, sample_indexes, 60,
    )
    observed_counts = {
        horizon: _rolling_sum(
            row_valid.astype(np.float64), sample_indexes, horizon,
        )
        for horizon in (*SURPRISE_HORIZONS, 60)
    }
    quote_surprises = [
        np.where(
            (observed_counts[horizon] > 0) & (observed_counts[60] > 0),
            _safe_log_ratio(_rolling_observed_mean(
                quote_volume, row_valid, sample_indexes, horizon,
            ), quote_baseline),
            0.0,
        )
        for horizon in SURPRISE_HORIZONS
    ]
    trade_surprises = [
        np.where(
            (observed_counts[horizon] > 0) & (observed_counts[60] > 0),
            _safe_log_ratio(_rolling_observed_mean(
                trade_activity, row_valid, sample_indexes, horizon,
            ), trade_baseline),
            0.0,
        )
        for horizon in SURPRISE_HORIZONS
    ]

    # Canonical Spot candles expose base volume but not quote volume.  Closing
    # price times base volume is a causal, scale-compatible approximation.
    spot_quote_approximation = spot[:, 3] * spot[:, 4]
    relative_quote_activity = [
        _safe_log_ratio(
            _rolling_observed_mean(
                quote_volume, row_valid, sample_indexes, horizon,
            ),
            _rolling_observed_mean(
                spot_quote_approximation, row_valid, sample_indexes, horizon,
            ),
        )
        for horizon in FLOW_HORIZONS
    ]

    columns = (
        basis_signed
        + [np.abs(values) for values in basis_signed]
        + futures_returns
        + [np.abs(values) for values in futures_returns]
        + range_controls
        + flow_signed
        + [np.abs(values) for values in flow_signed]
        + quote_surprises
        + trade_surprises
        + relative_quote_activity
        + [
            row_valid[sample_indexes].astype(np.float64),
            no_trade[sample_indexes].astype(np.float64),
            live[sample_indexes].astype(np.float64),
            np.minimum(price[3][sample_indexes], DAY_ROWS) / DAY_ROWS,
            live[sample_indexes].astype(np.float64),
            np.minimum(basis[3][sample_indexes], DAY_ROWS) / DAY_ROWS,
        ]
    )
    result = np.column_stack(columns).astype(np.float32, copy=False)
    if result.shape != (DAY_ROWS, len(FUTURES_BASIS_FEATURE_NAMES)) \
            or len(FUTURES_BASIS_FEATURE_NAMES) \
            != len(set(FUTURES_BASIS_FEATURE_NAMES)) \
            or not np.isfinite(result).all():
        raise ValueError("invalid causal futures-basis feature matrix")
    return result


def _causal_fill(
    values: np.ndarray,
    observed: np.ndarray,
    *,
    missing_value: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    observed = np.asarray(observed, dtype=bool)
    if values.ndim != 1 or observed.shape != values.shape \
            or not np.isfinite(values).all():
        raise ValueError("invalid futures-basis stream")
    indexes = np.arange(values.size, dtype=np.int64)
    last = np.maximum.accumulate(np.where(observed, indexes, -1))
    ever = last >= 0
    filled = np.full(values.size, missing_value, dtype=np.float64)
    filled[ever] = values[last[ever]]
    age = np.where(ever, indexes - last, values.size).astype(np.float64)
    return filled, ever, observed, age


def _sampled_difference(
    values: np.ndarray,
    ever: np.ndarray,
    sample_indexes: np.ndarray,
    horizon: int,
) -> np.ndarray:
    previous = sample_indexes - horizon
    available = previous >= 0
    available &= ever[sample_indexes]
    available[available] &= ever[previous[available]]
    result = np.zeros(sample_indexes.size, dtype=np.float64)
    result[available] = (
        values[sample_indexes[available]] - values[previous[available]]
    )
    return result


def _sampled_log_change(
    values: np.ndarray,
    ever: np.ndarray,
    sample_indexes: np.ndarray,
    horizon: int,
) -> np.ndarray:
    previous = sample_indexes - horizon
    available = previous >= 0
    available &= ever[sample_indexes]
    available[available] &= ever[previous[available]]
    result = np.zeros(sample_indexes.size, dtype=np.float64)
    result[available] = np.log(
        values[sample_indexes[available]] / values[previous[available]]
    )
    return result


def _sampled_deviation(
    values: np.ndarray,
    ever: np.ndarray,
    sample_indexes: np.ndarray,
    horizon: int,
) -> np.ndarray:
    weighted = np.where(ever, values, 0.0)
    cumulative = np.concatenate(([0.0], np.cumsum(weighted)))
    counts = np.concatenate(([0], np.cumsum(ever.astype(np.int64))))
    starts = np.maximum(sample_indexes - horizon + 1, 0)
    sums = cumulative[sample_indexes + 1] - cumulative[starts]
    count = counts[sample_indexes + 1] - counts[starts]
    means = np.divide(sums, count, out=np.zeros_like(sums), where=count > 0)
    return np.where(
        ever[sample_indexes] & (count > 0),
        values[sample_indexes] - means,
        0.0,
    )


def _rolling_sum(
    values: np.ndarray,
    sample_indexes: np.ndarray,
    horizon: int,
) -> np.ndarray:
    cumulative = np.concatenate(([0.0], np.cumsum(values, dtype=np.float64)))
    starts = np.maximum(sample_indexes - horizon + 1, 0)
    return cumulative[sample_indexes + 1] - cumulative[starts]


def _rolling_observed_mean(
    values: np.ndarray,
    observed: np.ndarray,
    sample_indexes: np.ndarray,
    horizon: int,
) -> np.ndarray:
    sums = _rolling_sum(np.where(observed, values, 0.0), sample_indexes, horizon)
    counts = _rolling_sum(
        np.asarray(observed, dtype=np.float64), sample_indexes, horizon,
    )
    return np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)


def _rolling_imbalance(
    total_quote: np.ndarray,
    taker_buy_quote: np.ndarray,
    sample_indexes: np.ndarray,
    horizon: int,
) -> np.ndarray:
    total = _rolling_sum(total_quote, sample_indexes, horizon)
    buy = _rolling_sum(taker_buy_quote, sample_indexes, horizon)
    return np.divide(
        2 * buy - total,
        total,
        out=np.zeros_like(total),
        where=total > 0,
    )


def _safe_log_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    scale = np.maximum(np.maximum(numerator, denominator), 0.0)
    floor = np.maximum(scale * 1e-9, np.finfo(np.float64).tiny)
    return np.log((numerator + floor) / (denominator + floor))


def _validate_futures_day(
    values: dict[str, np.ndarray],
    validity: np.ndarray,
) -> None:
    if set(values) != set(FUTURES_KLINE_COLUMNS):
        raise ValueError("futures-kline columns differ from the feature contract")
    valid = np.asarray(validity, dtype=bool)
    if valid.shape != (DAY_ROWS,):
        raise ValueError("invalid futures-kline validity shape")
    arrays = {
        name: np.asarray(values[name], dtype=np.float64)
        for name in FUTURES_KLINE_COLUMNS
    }
    if any(array.shape != (DAY_ROWS,) for array in arrays.values()) \
            or any(not np.isfinite(array).all() for array in arrays.values()) \
            or any(bool((array[~valid] != 0).any()) for array in arrays.values()):
        raise ValueError("invalid futures-kline day columns")
    if not valid.any():
        return
    open_values = arrays["open"][valid]
    high_values = arrays["high"][valid]
    low_values = arrays["low"][valid]
    close_values = arrays["close"][valid]
    base_volume = arrays["baseVolume"][valid]
    quote_volume = arrays["quoteVolume"][valid]
    trade_count = arrays["tradeCount"][valid]
    taker_base = arrays["takerBuyBaseVolume"][valid]
    taker_quote = arrays["takerBuyQuoteVolume"][valid]
    if bool((open_values <= 0).any()) \
            or bool((high_values < np.maximum(open_values, close_values)).any()) \
            or bool((low_values > np.minimum(open_values, close_values)).any()) \
            or bool((low_values <= 0).any()) \
            or bool((base_volume < 0).any()) \
            or bool((quote_volume < 0).any()) \
            or bool((trade_count < 0).any()) \
            or bool((trade_count != np.floor(trade_count)).any()) \
            or bool((trade_count > 9_007_199_254_740_991).any()) \
            or bool((taker_base < 0).any()) \
            or bool((taker_quote < 0).any()) \
            or bool((taker_base > base_volume).any()) \
            or bool((taker_quote > quote_volume).any()):
        raise ValueError("invalid futures-kline row values")
    no_trade = trade_count == 0
    if bool((open_values[no_trade] != high_values[no_trade]).any()) \
            or bool((open_values[no_trade] != low_values[no_trade]).any()) \
            or bool((open_values[no_trade] != close_values[no_trade]).any()) \
            or bool((base_volume[no_trade] != 0).any()) \
            or bool((quote_volume[no_trade] != 0).any()) \
            or bool((taker_base[no_trade] != 0).any()) \
            or bool((taker_quote[no_trade] != 0).any()):
        raise ValueError("invalid futures-kline no-trade row")
    live = ~no_trade
    if bool((base_volume[live] == 0).any()) \
            or bool((quote_volume[live] == 0).any()):
        raise ValueError("invalid futures-kline live row")


def _validate_spot_day(values: np.ndarray) -> None:
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (DAY_ROWS, 5) \
            or not np.isfinite(values).all() \
            or bool((values[:, :4] <= 0).any()) \
            or bool((values[:, 4] < 0).any()) \
            or bool((values[:, 1] < values[:, [0, 3]].max(axis=1)).any()) \
            or bool((values[:, 2] > values[:, [0, 3]].min(axis=1)).any()):
        raise ValueError("invalid canonical Spot one-minute OHLCV")
