"""Causal rolling aggressor-flow features for minute-spaced oracle targets."""

from __future__ import annotations

import numpy as np


DAY_ROWS = 1_440
MINUTE_SECONDS = 60
SECOND_ROWS = DAY_ROWS * MINUTE_SECONDS
HORIZONS = (1, 5, 60, 300)
HORIZON_LABELS = {1: "1s", 5: "5s", 60: "60s", 300: "5m"}
METRICS = (
    "quoteImbalance",
    "absQuoteImbalance",
    "tradeCountImbalance",
    "absTradeCountImbalance",
    "rawPerAggregate",
    "aggregateHhi",
    "maxAggregateShare",
    "quantitySquaredSkew",
    "absQuantitySquaredSkew",
    "maxAggregateSkew",
    "absMaxAggregateSkew",
    "signedVwapGap",
    "absVwapGap",
    "signedArrivalCentroidGap",
    "absArrivalCentroidGap",
    "aggressorFlipRate",
    "lastAggressorSide",
    "absLastAggressorSide",
    "twoSidedActivity",
)
FLOW_FEATURE_NAMES = tuple(
    f"{metric}{HORIZON_LABELS[horizon]}"
    for metric in METRICS
    for horizon in HORIZONS
) + tuple(
    f"{metric}{HORIZON_LABELS[horizon]}Vs60m"
    for metric in ("logQuoteRate", "logTradeRate")
    for horizon in HORIZONS
) + ("quoteShare5sOf60s",)

TRADE_FLOW_COLUMNS = (
    "aggressiveBuyBaseVolume",
    "aggressiveSellBaseVolume",
    "aggressiveBuyQuoteVolume",
    "aggressiveSellQuoteVolume",
    "aggressiveBuyAggregateQuantitySquared",
    "aggressiveSellAggregateQuantitySquared",
    "aggressiveBuyMaxAggregateQuantity",
    "aggressiveSellMaxAggregateQuantity",
    "aggressiveBuyBaseVolumeTimeMoment",
    "aggressiveSellBaseVolumeTimeMoment",
    "aggressiveBuyAggregateTradeCount",
    "aggressiveSellAggregateTradeCount",
    "aggressiveBuyTradeCount",
    "aggressiveSellTradeCount",
    "aggressorSideFlipCount",
    "firstAggressorSide",
    "lastAggressorSide",
)


def causal_trade_flow_features(
    previous_day: dict[str, np.ndarray],
    current_day: dict[str, np.ndarray],
) -> np.ndarray:
    """Build windows ending at each target's currently closed one-second bin.

    Oracle target row k is labelled `day + 999ms + 60s*k`. Binance's `.999`
    label denotes the completed half-open second, so the matching trade-flow
    row is `60*k` and the next row is never included.
    """
    _validate_day(previous_day)
    _validate_day(current_day)
    computed: dict[int, dict[str, np.ndarray]] = {}
    for horizon in HORIZONS:
        computed[horizon] = rolling_metrics(
            previous_day, current_day, horizon,
        )
    columns = [
        computed[horizon][metric]
        for metric in METRICS
        for horizon in HORIZONS
    ]
    baseline = rolling_totals(previous_day, current_day, 3_600)
    baseline_activity = {
        "quote": baseline["buy_quote"] + baseline["sell_quote"],
        "trade": baseline["buy_trade"] + baseline["sell_trade"],
    }
    tiny = np.finfo(np.float64).tiny
    for source in ("quote", "trade"):
        baseline_mean = baseline_activity[source] / 3_600
        floor = np.maximum(baseline_mean * 1e-9, tiny)
        for horizon in HORIZONS:
            recent_mean = computed[horizon][source] / horizon
            columns.append(np.log(
                (recent_mean + floor) / (baseline_mean + floor)
            ))
    columns.append(safe_divide(
        computed[5]["quote"], computed[60]["quote"],
    ))
    result = np.column_stack(columns).astype(np.float32, copy=False)
    if result.shape != (DAY_ROWS, len(FLOW_FEATURE_NAMES)) \
            or not np.isfinite(result).all():
        raise ValueError("invalid causal trade-flow feature matrix")
    return result


def rolling_metrics(
    previous_day: dict[str, np.ndarray],
    current_day: dict[str, np.ndarray],
    horizon: int,
) -> dict[str, np.ndarray]:
    totals = rolling_totals(previous_day, current_day, horizon)
    buy_base = totals["buy_base"]
    sell_base = totals["sell_base"]
    buy_quote = totals["buy_quote"]
    sell_quote = totals["sell_quote"]
    buy_aggregate = totals["buy_aggregate"]
    sell_aggregate = totals["sell_aggregate"]
    buy_trade = totals["buy_trade"]
    sell_trade = totals["sell_trade"]
    total_base = buy_base + sell_base
    total_quote = buy_quote + sell_quote
    total_aggregate = buy_aggregate + sell_aggregate
    total_trade = buy_trade + sell_trade
    quote_imbalance = signed_ratio(buy_quote, sell_quote)
    trade_imbalance = signed_ratio(buy_trade, sell_trade)
    buy_max = rolling_target_max(
        previous_day["aggressiveBuyMaxAggregateQuantity"],
        current_day["aggressiveBuyMaxAggregateQuantity"],
        horizon,
    )
    sell_max = rolling_target_max(
        previous_day["aggressiveSellMaxAggregateQuantity"],
        current_day["aggressiveSellMaxAggregateQuantity"],
        horizon,
    )
    buy_vwap = safe_divide(buy_quote, buy_base)
    sell_vwap = safe_divide(sell_quote, sell_base)
    both = (buy_base > 0) & (sell_base > 0)
    signed_vwap_gap = np.zeros(DAY_ROWS, dtype=np.float64)
    signed_vwap_gap[both] = (
        2 * (buy_vwap[both] - sell_vwap[both])
        / (buy_vwap[both] + sell_vwap[both])
    )
    buy_centroid, sell_centroid = arrival_centroids(
        previous_day, current_day, horizon, buy_base, sell_base,
    )
    signed_centroid_gap = np.where(
        both, (buy_centroid - sell_centroid) / horizon, 0,
    )
    flips, last_side = sequence_metrics(previous_day, current_day, horizon)
    quantity_squared = totals["buy_quantity_squared"] \
        + totals["sell_quantity_squared"]
    quantity_squared_skew = signed_ratio(
        totals["buy_quantity_squared"],
        totals["sell_quantity_squared"],
    )
    maximum_skew = signed_ratio(buy_max, sell_max)
    result = {
        "quote": total_quote,
        "trade": total_trade,
        "quoteImbalance": quote_imbalance,
        "absQuoteImbalance": np.abs(quote_imbalance),
        "tradeCountImbalance": trade_imbalance,
        "absTradeCountImbalance": np.abs(trade_imbalance),
        "rawPerAggregate": np.log(
            (total_trade + 1) / (total_aggregate + 1)
        ),
        "aggregateHhi": safe_divide(
            quantity_squared, np.square(total_base),
        ),
        "maxAggregateShare": safe_divide(
            np.maximum(buy_max, sell_max), total_base,
        ),
        "quantitySquaredSkew": quantity_squared_skew,
        "absQuantitySquaredSkew": np.abs(quantity_squared_skew),
        "maxAggregateSkew": maximum_skew,
        "absMaxAggregateSkew": np.abs(maximum_skew),
        "signedVwapGap": signed_vwap_gap,
        "absVwapGap": np.abs(signed_vwap_gap),
        "signedArrivalCentroidGap": signed_centroid_gap,
        "absArrivalCentroidGap": np.abs(signed_centroid_gap),
        "aggressorFlipRate": safe_divide(
            flips, np.maximum(total_aggregate - 1, 0),
        ),
        "lastAggressorSide": last_side,
        "absLastAggressorSide": np.abs(last_side),
        "twoSidedActivity": both.astype(np.float64),
    }
    if any(value.shape != (DAY_ROWS,) or not np.isfinite(value).all()
           for value in result.values()):
        raise ValueError(f"invalid rolling trade-flow metrics at {horizon}s")
    return result


def rolling_totals(
    previous_day: dict[str, np.ndarray],
    current_day: dict[str, np.ndarray],
    horizon: int,
) -> dict[str, np.ndarray]:
    names = {
        "buy_base": "aggressiveBuyBaseVolume",
        "sell_base": "aggressiveSellBaseVolume",
        "buy_quote": "aggressiveBuyQuoteVolume",
        "sell_quote": "aggressiveSellQuoteVolume",
        "buy_quantity_squared": "aggressiveBuyAggregateQuantitySquared",
        "sell_quantity_squared": "aggressiveSellAggregateQuantitySquared",
        "buy_aggregate": "aggressiveBuyAggregateTradeCount",
        "sell_aggregate": "aggressiveSellAggregateTradeCount",
        "buy_trade": "aggressiveBuyTradeCount",
        "sell_trade": "aggressiveSellTradeCount",
    }
    return {
        output: rolling_target_sum(
            previous_day[source], current_day[source], horizon,
        )
        for output, source in names.items()
    }


def arrival_centroids(
    previous_day: dict[str, np.ndarray],
    current_day: dict[str, np.ndarray],
    horizon: int,
    buy_base: np.ndarray,
    sell_base: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    positions = np.arange(horizon, dtype=np.float64)[None, :]
    buy_values = target_windows(
        previous_day["aggressiveBuyBaseVolume"],
        current_day["aggressiveBuyBaseVolume"], horizon,
    )
    sell_values = target_windows(
        previous_day["aggressiveSellBaseVolume"],
        current_day["aggressiveSellBaseVolume"], horizon,
    )
    buy_moments = target_windows(
        previous_day["aggressiveBuyBaseVolumeTimeMoment"],
        current_day["aggressiveBuyBaseVolumeTimeMoment"], horizon,
    )
    sell_moments = target_windows(
        previous_day["aggressiveSellBaseVolumeTimeMoment"],
        current_day["aggressiveSellBaseVolumeTimeMoment"], horizon,
    )
    return (
        safe_divide((buy_moments + buy_values * positions).sum(axis=1), buy_base),
        safe_divide((sell_moments + sell_values * positions).sum(axis=1), sell_base),
    )


def sequence_metrics(
    previous_day: dict[str, np.ndarray],
    current_day: dict[str, np.ndarray],
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    internal = target_windows(
        previous_day["aggressorSideFlipCount"],
        current_day["aggressorSideFlipCount"], horizon,
    ).sum(axis=1).astype(np.float64, copy=False)
    first = target_windows(
        previous_day["firstAggressorSide"],
        current_day["firstAggressorSide"], horizon,
    )
    last = target_windows(
        previous_day["lastAggressorSide"],
        current_day["lastAggressorSide"], horizon,
    )
    flips = internal.copy()
    last_side = np.zeros(DAY_ROWS, dtype=np.float64)
    for row in range(DAY_ROWS):
        active = np.flatnonzero(first[row] != 0)
        if active.size == 0:
            continue
        last_side[row] = last[row, active[-1]]
        if active.size > 1:
            flips[row] += np.count_nonzero(
                first[row, active[1:]] != last[row, active[:-1]]
            )
    return flips, last_side


def rolling_target_sum(
    previous: np.ndarray,
    current: np.ndarray,
    horizon: int,
) -> np.ndarray:
    values = _combined(previous, current, horizon)
    cumulative = np.concatenate((
        np.zeros(1, dtype=np.float64),
        np.cumsum(values, dtype=np.float64),
    ))
    rolling = cumulative[horizon:] - cumulative[:-horizon]
    return rolling[np.arange(DAY_ROWS) * MINUTE_SECONDS]


def rolling_target_max(
    previous: np.ndarray,
    current: np.ndarray,
    horizon: int,
) -> np.ndarray:
    return target_windows(previous, current, horizon).max(axis=1)


def target_windows(
    previous: np.ndarray,
    current: np.ndarray,
    horizon: int,
) -> np.ndarray:
    values = _combined(previous, current, horizon)
    windows = np.lib.stride_tricks.sliding_window_view(values, horizon)
    return windows[np.arange(DAY_ROWS) * MINUTE_SECONDS]


def _combined(
    previous: np.ndarray,
    current: np.ndarray,
    horizon: int,
) -> np.ndarray:
    if not 1 <= horizon <= 3_600:
        raise ValueError("trade-flow horizon must be between 1s and 60m")
    previous_values = np.asarray(previous, dtype=np.float64)
    current_values = np.asarray(current, dtype=np.float64)
    if previous_values.shape != (SECOND_ROWS,) \
            or current_values.shape != (SECOND_ROWS,):
        raise ValueError("trade-flow rolling inputs have invalid shape")
    return np.concatenate((previous_values[-(horizon - 1):], current_values)) \
        if horizon > 1 else current_values


def signed_ratio(positive: np.ndarray, negative: np.ndarray) -> np.ndarray:
    return safe_divide(positive - negative, positive + negative)


def safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=np.float64),
        where=denominator > 0,
    )


def _validate_day(day: dict[str, np.ndarray]) -> None:
    if set(day) != set(TRADE_FLOW_COLUMNS):
        raise ValueError("trade-flow day columns differ from the feature contract")
    for name in TRADE_FLOW_COLUMNS:
        values = np.asarray(day[name])
        if values.shape != (SECOND_ROWS,) or not np.isfinite(values).all():
            raise ValueError(f"invalid trade-flow day column: {name}")
