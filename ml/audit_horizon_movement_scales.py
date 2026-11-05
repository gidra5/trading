"""CPU-only realized movement/range audit; no forecast or trading selection.

Use complete nonoverlapping windows inside each existing dataset split. Also
reconstruct historical candle statistics from the already saved chart artifact.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import torch

from audit_horizon_realized_moves import utc
from horizon_price_forecast import horizon_split
from stochastic_price_forecast import load_timeline_split
from trading_storage import candle_times, read_candle_column


def ohlc_at_open_times(times: np.ndarray) -> dict[str, np.ndarray]:
    source = Path("data/market/immutable/refs/candles/spot-btcusdt/btcusdt/1s")
    result = {column: np.empty(len(times), dtype=np.float64) for column in ("open", "high", "low", "close")}
    days = times // 86_400_000
    for day in np.unique(days):
        date = datetime.fromtimestamp(int(day * 86400), timezone.utc).strftime("%Y-%m-%d")
        file = source / (date + ".json")
        candle_time = candle_times(file)
        selected = np.flatnonzero(days == day)
        indices = np.searchsorted(candle_time, times[selected])
        if np.any(indices >= len(candle_time)) or np.any(candle_time[indices] != times[selected]):
            raise ValueError("missing one-second OHLC candle")
        for column in result:
            result[column][selected] = read_candle_column(file, column)[indices]
    if any(not np.isfinite(values).all() or np.any(values <= 0) for values in result.values()):
        raise ValueError("expected finite positive OHLC prices")
    return result


def summary(values: np.ndarray) -> dict:
    return {"mean": float(values.mean()), "median": float(np.median(values)),
            "p90": float(np.quantile(values, .9)), "max": float(values.max()),
            "fractionAbove10Bps": float((values > 10).mean())}


def current_split(dataset: Path, split_name: str, horizons: list[int]) -> list[dict]:
    source = load_timeline_split(dataset, split_name)
    first = int(source.axis_start_ms + source.origins[0] * 1000)
    final = int(source.axis_start_ms + (len(source.timeline) - 1) * 1000)
    # The initial row supplies the observed close. Subsequent OHLC rows cover
    # the full future window, with no approximation by close-only extrema.
    tape = ohlc_at_open_times(np.arange(first - 1000, final, 1000, dtype=np.int64))
    rows = []
    for horizon in horizons:
        split = horizon_split(source, horizon)
        origins = split.origins[::horizon]
        count = len(origins)
        if not count:
            continue
        if np.any(origins != source.origins[0] + np.arange(count) * horizon):
            raise ValueError("nonoverlap window origin alignment failed")
        stop = count * horizon
        endpoint = tape["close"][:stop + 1:horizon]
        gross = endpoint[1:] / endpoint[:-1]
        np.testing.assert_allclose(np.log(gross).astype(np.float32), split.targets()[::horizon],
                                   atol=3e-9, rtol=2e-5)
        returns = (gross - 1) * 10_000
        first_open = tape["open"][1:stop + 1:horizon]
        high = tape["high"][1:stop + 1].reshape(count, horizon).max(axis=1)
        low = tape["low"][1:stop + 1].reshape(count, horizon).min(axis=1)
        candle_range = (high - low) / first_open * 10_000
        body = (endpoint[1:] / first_open - 1) * 10_000
        close_windows = np.lib.stride_tricks.sliding_window_view(tape["close"][:stop + 1], horizon + 1)[::horizon]
        close_range = (close_windows.max(axis=1) - close_windows.min(axis=1)) / endpoint[:-1] * 10_000
        threshold = (1 / .9995**2 - 1) * 10_000
        row = {"split": split_name, "horizonSeconds": horizon, "nonoverlapCount": count,
               "firstTimestamp": first, "endTimestamp": first + stop * 1000,
               "firstUtc": utc(first), "endUtc": utc(first + stop * 1000), "observedSeconds": stop,
               "sampleWarning": "fewer than 30 nonoverlapping windows" if count < 30 else None,
               "absoluteEndpointReturnBps": summary(np.abs(returns)),
               "absoluteCandleBodyBps": summary(np.abs(body)),
               "ohlcHighLowOverFirstOpenBps": summary(candle_range),
               "closeOnlyRangeOverObservedPriceBps": summary(close_range),
               "signedEndpointMeanBps": float(returns.mean()),
               "signedEndpointStdBps": float(returns.std(ddof=1)) if count > 1 else None,
               "positiveAboveFeeThresholdFraction": float((returns > threshold).mean()),
               "negativeBeyondSameMagnitudeFraction": float((returns < -threshold).mean()),
               "feeThresholdBps": threshold}
        rows.append(row)
        print(json.dumps({"split": split_name, "horizon": horizon, "n": count,
              "meanAbsBps": row["absoluteEndpointReturnBps"]["mean"],
              "medianAbsBps": row["absoluteEndpointReturnBps"]["median"],
              "meanOhlcRangeBps": row["ohlcHighLowOverFirstOpenBps"]["mean"],
              "meanSignedBps": row["signedEndpointMeanBps"]}), flush=True)
    return rows


def historical_chart() -> dict:
    file = Path("docs/interval-price-movement.html")
    text = file.read_text(encoding="utf-8")
    marker = "const chartCases = "
    cases = json.JSONDecoder().raw_decode(text[text.index(marker) + len(marker):])[0]
    rows = []
    for case in cases:
        candles = case["candles"]
        ranges = np.array([(c["high"] - c["low"]) / c["open"] * 100 for c in candles])
        bodies = np.array([(c["close"] / c["open"] - 1) * 100 for c in candles])
        rows.append({"period": case["intervalLabel"], "label": case["label"], "candles": len(candles),
                     "meanHighLowOverOpenPct": float(ranges.mean()),
                     "medianHighLowOverOpenPct": float(np.median(ranges)),
                     "meanAbsoluteBodyPct": float(np.abs(bodies).mean()),
                     "meanSignedBodyPct": float(bodies.mean())})
    distribution = json.loads(Path("data/benchmarks/log-return-distributions.json").read_text(encoding="utf-8"))
    full = distribution["fullHistory"]
    fifteen = next(row for row in full["scales"] if row["id"] == "15m")
    return {"chartSource": str(file), "chartAsset": "BTCUSDT Binance spot", "chartCandleInterval": "15m",
            "chartCaseSelection": "old hand-selected regime intervals; cases can overlap and are not a representative pooled sample",
            "chartCaseStatistics": rows,
            "closestChartMeanRangesToRecalledPointThreePercent": sorted(rows, key=lambda r: abs(r["meanHighLowOverOpenPct"] - .3))[:3],
            "oldDistributionSource": "data/benchmarks/log-return-distributions.json",
            "oldDistributionStart": full["startTime"], "oldDistributionEnd": full["endTime"],
            "oldFifteenMinuteDistribution": fifteen,
            "recallStatus": "No exact archived statement that 15m mean movement equals 0.3% was found; reconstructed nearby range values and the old 28.319 bps standard deviation are distinct plausible references."}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path,
        default=Path("data/training/datasets/next-return-production-basis-history120-4l-train512k-v1"))
    parser.add_argument("--horizons", type=int, nargs="+", default=[900, 1800, 3600, 7200, 14400, 28800, 43200, 86400])
    parser.add_argument("--output", type=Path,
        default=Path("data/backtests/stochastic-crra/horizon-movement-range-audit.json"))
    args = parser.parse_args()
    if any(horizon < 1 for horizon in args.horizons):
        raise ValueError("positive horizons required")
    torch.set_num_threads(2)
    rows = [row for split in ["train", "validation", "test"]
            for row in current_split(args.dataset, split, args.horizons)]
    result = {"contract": "actual-ohlc-and-endpoint-horizon-movement-v1", "device": "cpu",
              "checkpointsLoaded": False, "modelSelectionChanged": False, "dataset": str(args.dataset),
              "method": ["All complete nonoverlapping windows within each existing chronological split; horizons have different final cutoffs.",
                         "Endpoint return is next endpoint close / currently observed close minus one.",
                         "OHLC range uses actual one-second candle highs and lows, divided by the first future candle open.",
                         "Candle body uses the same first future open and final close; close-only range is also reported separately.",
                         "These are realized hindsight movement magnitudes, not predictable conditional mean returns or attainable high-low profits.",
                         "1 percent equals 100 bps. Long fee threshold uses 0.0005 per side and unit funding under the valid full-close hypotheses."],
              "rows": rows, "historicalSources": historical_chart()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
