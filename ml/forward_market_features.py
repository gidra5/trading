"""Causal minute-close trade-flow, futures-basis, and positioning features."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np

from oracle_futures_basis_features import (
    FUTURES_BASIS_FEATURE_NAMES,
    FUTURES_KLINE_COLUMNS,
    causal_futures_basis_features,
)
from oracle_futures_metrics_features import (
    FUTURES_FEATURE_NAMES,
    METRIC_COLUMNS,
    causal_futures_metrics_features,
)
from oracle_trade_flow_features import (
    FLOW_FEATURE_NAMES,
    SECOND_ROWS,
    TRADE_FLOW_COLUMNS,
    causal_trade_flow_features,
)
from trading_storage import (
    read_candle_column,
    read_derivatives_kline_columns,
    read_derivatives_metrics_columns,
    read_trade_flow_columns,
)


DAY_ROWS = 1_440
FORWARD_FEATURE_NAMES = (
    FLOW_FEATURE_NAMES + FUTURES_BASIS_FEATURE_NAMES + FUTURES_FEATURE_NAMES
)
FORWARD_FEATURE_COUNT = len(FORWARD_FEATURE_NAMES)


def reference_roots(data_root: Path) -> dict[str, Path]:
    market = data_root / "market/immutable/refs"
    return {
        "flow": market / "trade-flow/spot-btcusdt/btcusdt/1s",
        "futures": market / "derivatives-klines/usdm-futures/btcusdt/1m",
        "metrics": market / "derivatives-metrics/usdm-futures/btcusdt/5m",
        "spot": market / "candles/spot-btcusdt/btcusdt/1m",
    }


def has_forward_feature_day(data_root: Path, day_value: str) -> bool:
    current = date.fromisoformat(day_value)
    previous = (current - timedelta(days=1)).isoformat()
    return all(
        (root / f"{value}.json").is_file()
        for root in reference_roots(data_root).values()
        for value in (previous, day_value)
    )


def build_forward_feature_day(data_root: Path, day_value: str) -> np.ndarray:
    """Return features observable at each completed one-minute candle close."""
    roots = reference_roots(data_root)
    current = date.fromisoformat(day_value)
    previous = (current - timedelta(days=1)).isoformat()
    if not has_forward_feature_day(data_root, day_value):
        raise FileNotFoundError(f"forward-market sources are incomplete for {day_value}")

    previous_flow = read_trade_flow_columns(
        roots["flow"] / f"{previous}.json", TRADE_FLOW_COLUMNS,
    )
    current_flow = read_trade_flow_columns(
        roots["flow"] / f"{day_value}.json", TRADE_FLOW_COLUMNS,
    )
    flow = causal_trade_flow_features(
        *_shift_second_days_to_minute_close(previous_flow, current_flow)
    )

    previous_futures, previous_futures_validity = read_derivatives_kline_columns(
        roots["futures"] / f"{previous}.json", FUTURES_KLINE_COLUMNS,
    )
    current_futures, current_futures_validity = read_derivatives_kline_columns(
        roots["futures"] / f"{day_value}.json", FUTURES_KLINE_COLUMNS,
    )
    previous_spot = _read_spot_day(roots["spot"] / f"{previous}.json")
    current_spot = _read_spot_day(roots["spot"] / f"{day_value}.json")
    shifted_futures, shifted_futures_validity = _shift_minute_days_to_close(
        previous_futures, current_futures,
        previous_futures_validity, current_futures_validity,
    )
    shifted_spot = _shift_dense_minute_matrices(previous_spot, current_spot)
    basis = causal_futures_basis_features(
        shifted_futures[0], shifted_futures_validity[0],
        shifted_futures[1], shifted_futures_validity[1],
        shifted_spot[0], shifted_spot[1],
    )

    previous_metrics, previous_metric_validity = read_derivatives_metrics_columns(
        roots["metrics"] / f"{previous}.json", METRIC_COLUMNS,
    )
    current_metrics, current_metric_validity = read_derivatives_metrics_columns(
        roots["metrics"] / f"{day_value}.json", METRIC_COLUMNS,
    )
    metrics = causal_futures_metrics_features(
        previous_metrics, previous_metric_validity,
        current_metrics, current_metric_validity,
    )
    result = np.column_stack((flow, basis, metrics)).astype(np.float32, copy=False)
    if result.shape != (DAY_ROWS, FORWARD_FEATURE_COUNT) \
            or not np.isfinite(result).all():
        raise ValueError(f"invalid forward-market features for {day_value}")
    return result


def _shift_second_days_to_minute_close(
    previous: dict[str, np.ndarray],
    current: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Make row k end at second 59+60k instead of second 60k."""
    shifted_previous: dict[str, np.ndarray] = {}
    shifted_current: dict[str, np.ndarray] = {}
    for name in TRADE_FLOW_COLUMNS:
        merged = np.concatenate((previous[name], current[name]))
        shifted_previous[name] = merged[59:SECOND_ROWS + 59]
        tail = merged[SECOND_ROWS + 59:]
        shifted_current[name] = np.pad(tail, (0, 59), constant_values=0)
    return shifted_previous, shifted_current


def _shift_minute_days_to_close(
    previous: dict[str, np.ndarray],
    current: dict[str, np.ndarray],
    previous_validity: np.ndarray,
    current_validity: np.ndarray,
) -> tuple[
    tuple[dict[str, np.ndarray], dict[str, np.ndarray]],
    tuple[np.ndarray, np.ndarray],
]:
    """Make the basis row include the candle closing at its prediction time."""
    shifted_previous: dict[str, np.ndarray] = {}
    shifted_current: dict[str, np.ndarray] = {}
    for name in FUTURES_KLINE_COLUMNS:
        merged = np.concatenate((previous[name], current[name]))
        shifted_previous[name] = merged[1:DAY_ROWS + 1]
        shifted_current[name] = np.concatenate((
            merged[DAY_ROWS + 1:], np.zeros(1, dtype=merged.dtype),
        ))
    validity = np.concatenate((previous_validity, current_validity))
    return (
        (shifted_previous, shifted_current),
        (validity[1:DAY_ROWS + 1], np.concatenate((validity[DAY_ROWS + 1:], [False]))),
    )


def _shift_dense_minute_matrices(
    previous: np.ndarray,
    current: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    merged = np.concatenate((previous, current), axis=0)
    return merged[1:DAY_ROWS + 1], np.concatenate((merged[DAY_ROWS + 1:], merged[-1:]))


def _read_spot_day(reference: Path) -> np.ndarray:
    result = np.column_stack(tuple(
        read_candle_column(reference, name)
        for name in ("open", "high", "low", "close", "volume")
    ))
    if result.shape != (DAY_ROWS, 5) or not np.isfinite(result).all():
        raise ValueError(f"invalid spot OHLCV day: {reference}")
    return result
