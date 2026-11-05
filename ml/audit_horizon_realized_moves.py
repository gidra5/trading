"""Read-only CPU diagnostics of realized, nonoverlapping endpoint moves.

These are hindsight outcomes, not model predictions or a model-selection score.
No checkpoint is loaded and no training data or execution state is modified.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import torch

from export_stochastic_price_forecast import load_close_prices
from horizon_price_forecast import load_horizon_split


def utc(milliseconds: int) -> str:
    return datetime.fromtimestamp(milliseconds / 1000, timezone.utc).isoformat()


def optional_mean(values: np.ndarray) -> float | None:
    return float(values.mean()) if len(values) else None


def diagnose(dataset: Path, split_name: str, horizon: int, test_seconds: int) -> dict:
    split = load_horizon_split(dataset, split_name, horizon)
    origins = split.origins[::horizon]
    if split_name == "test":
        if test_seconds % horizon:
            raise ValueError("common test duration must be divisible by each horizon")
        requested = test_seconds // horizon
        if len(origins) < requested:
            raise ValueError("test split does not contain the complete requested common window")
        origins = origins[:requested]
    if not len(origins) or np.any(np.diff(origins) != horizon):
        raise ValueError("expected nonempty contiguous nonoverlapping decision intervals")
    times = split.source.axis_start_ms + origins * 1000
    endpoint_times = np.append(times, times[-1] + horizon * 1000)
    prices = load_close_prices(endpoint_times,
        Path("data/market/immutable/refs/candles/spot-btcusdt/btcusdt/1s"))
    gross = prices[1:] / prices[:-1]
    np.testing.assert_array_equal(np.log(gross).astype(np.float32), split.targets()[::horizon][:len(origins)])
    returns = (gross - 1) * 10_000
    positive = returns > 0
    negative = returns < 0
    threshold = (1 / (1 - 0.0005)**2 - 1) * 10_000
    positive_large = returns > threshold
    negative_large = returns < -threshold
    count = len(origins)
    first, end = int(times[0]), int(endpoint_times[-1])
    return {
        "split": split_name, "horizonSeconds": horizon,
        "nonoverlapCount": count, "observedSeconds": count * horizon,
        "firstTimestamp": first, "endTimestamp": end, "firstUtc": utc(first), "endUtc": utc(end),
        "medianAbsoluteReturnBps": float(np.median(np.abs(returns))),
        "p90AbsoluteReturnBps": float(np.quantile(np.abs(returns), 0.9)),
        "meanRealizedReturnBps": float(returns.mean()),
        "positiveCount": int(positive.sum()), "positiveFraction": float(positive.mean()),
        "negativeCount": int(negative.sum()), "negativeFraction": float(negative.mean()),
        "zeroCount": int((returns == 0).sum()),
        "meanPositiveReturnBps": optional_mean(returns[positive]),
        "meanNegativeMagnitudeBps": optional_mean(-returns[negative]),
        "fullCloseLongThresholdBps": threshold,
        "positiveAboveThresholdCount": int(positive_large.sum()),
        "positiveAboveThresholdFractionOfAll": float(positive_large.mean()),
        "positiveAboveThresholdFractionOfPositive": float(positive_large.sum() / positive.sum()) if positive.any() else None,
        "negativeBeyondSameMagnitudeCount": int(negative_large.sum()),
        "negativeBeyondSameMagnitudeFractionOfAll": float(negative_large.mean()),
        "negativeBeyondSameMagnitudeFractionOfNegative": float(negative_large.sum() / negative.sum()) if negative.any() else None,
        "hindsightMeanGivenRealizedPositiveAboveThresholdBps": optional_mean(returns[positive_large]),
        "hindsightMeanNegativeMagnitudeGivenRealizedBeyondThresholdBps": optional_mean(-returns[negative_large]),
        "endpointUnderlyingReturnPercent": float((prices[-1] / prices[0] - 1) * 100),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path,
        default=Path("data/training/datasets/next-return-production-basis-history120-4l-train512k-v1"))
    parser.add_argument("--horizons", type=int, nargs="+", default=[300, 900])
    parser.add_argument("--test-seconds", type=int, default=151200)
    parser.add_argument("--output", type=Path,
        default=Path("data/backtests/stochastic-crra/h300-h900-realized-moves.json"))
    args = parser.parse_args()
    if args.test_seconds < 1 or any(h < 1 for h in args.horizons):
        raise ValueError("positive durations required")
    torch.set_num_threads(2)
    rows = []
    for split_name in ["validation", "test"]:
        for horizon in args.horizons:
            row = diagnose(args.dataset, split_name, horizon, args.test_seconds)
            rows.append(row)
            print(json.dumps(row, allow_nan=False), flush=True)
    result = {"contract": "realized-nonoverlapping-horizon-price-moves-v1", "dataset": str(args.dataset),
        "device": "cpu", "checkpointsLoaded": False, "changesModelSelection": False,
        "interpretation": [
            "Realized outcomes selected after the fact are not conditional predictions available at entry.",
            "Large realized moves do not imply that the conditional expected move clears execution costs.",
            "Positive threshold assumes an Oracle cash-starting surviving valid full close, 5 bps per side and unit funding.",
            "Negative threshold uses the same absolute magnitude for descriptive comparison, not a short profitability theorem.",
            f"Validation uses all complete nonoverlapping within-split horizons; test uses the same fixed {args.test_seconds}-second window.",
        ], "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
