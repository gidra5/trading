"""Export nonoverlapping endpoint laws plus every realized intermediate mark.

These laws are sufficient only for a declared fixed-hold, long/cash endpoint
problem with nonnegative cash and inventory, no limits, and unit funding.
They are not forecasts of the intermediate path or of leveraged liquidation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from export_stochastic_price_forecast import load_close_prices
from horizon_price_forecast import CONTRACT, load_horizon_forecast, load_horizon_split
from train_stochastic_price_forecast import write_json


@torch.no_grad()
def export(checkpoint: Path, output: Path, split_name: str = "test", branches: int = 7,
           batch_size: int = 512, count: int | None = None) -> dict:
    device = torch.device("cuda")
    model, payload = load_horizon_forecast(checkpoint, device)
    horizon = int(payload["plan"]["horizonSeconds"])
    split = load_horizon_split(Path(payload["plan"]["dataset"]), split_name, horizon)
    origins = split.origins[::horizon]
    if count is not None:
        origins = origins[:count]
    if not len(origins):
        raise ValueError("requested endpoint export is empty")
    times = split.source.axis_start_ms + origins * 1000
    all_times = np.arange(times[0], times[-1] + horizon * 1000 + 1, 1000, dtype=np.int64)
    prices = load_close_prices(all_times, Path("data/market/immutable/refs/candles/spot-btcusdt/btcusdt/1s"))
    if np.any(np.diff(times) != horizon * 1000):
        raise ValueError("endpoint decisions must use a contiguous nonoverlapping horizon schedule")
    expected_targets = np.log(prices[horizon::horizon] / prices[:-horizon:horizon]).astype(np.float32)
    np.testing.assert_array_equal(expected_targets, split.targets()[::horizon][:len(origins)])
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    max_price_mean_error, max_log_price_mean_difference = 0.0, 0.0
    component_excess = model._component_moments("excess")
    quadrature_comparison = float(np.max(np.abs(component_excess - model._component_moments("excess", 384))))
    with output.open("w", encoding="utf-8") as sink:
        for start in range(0, len(origins), batch_size):
            selected = origins[start:start + batch_size]
            features = torch.from_numpy(split.features(selected)).to(device)
            gross, probabilities = model.endpoint_scenarios(features, branches)
            zero, masses = model.probabilities(features)
            masses /= masses.sum(axis=1, keepdims=True)
            expected_gross = 1 + (1 - zero) * (masses @ component_excess)
            represented_gross = (gross * probabilities).sum(axis=1)
            max_price_mean_error = max(max_price_mean_error, float(np.max(np.abs(represented_gross - expected_gross))))
            log_mean_gross = np.exp((np.log(gross) * probabilities).sum(axis=1))
            max_log_price_mean_difference = max(max_log_price_mean_difference,
                                                float(np.max(np.abs(represented_gross - log_mean_gross))))
            for local in range(len(selected)):
                index = start + local
                price_index = index * horizon
                observed = float(prices[price_index])
                row = {"horizonSeconds": horizon, "timestamp": int(times[index]), "observedPrice": observed,
                       "realizedNextTimestamp": int(times[index] + horizon * 1000),
                       "realizedNextPrice": float(prices[price_index + horizon]),
                       "realizedPrices": prices[price_index + 1:price_index + horizon + 1].tolist(),
                       "tree": {"children": [{"probability": float(probabilities[local, branch]),
                                                "price": observed * float(gross[local, branch])}
                                               for branch in range(branches + 1)]}}
                sink.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
            if (start // batch_size) % 40 == 0:
                print(json.dumps({"horizonSeconds": horizon, "rows": min(start + batch_size, len(origins)),
                                  "total": len(origins), "seconds": time.monotonic() - started}), flush=True)
    manifest = {"contract": CONTRACT, "checkpoint": str(checkpoint),
                "checkpointSha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                "forecastFile": str(output), "split": split_name, "rows": len(origins),
                "horizonSeconds": horizon, "decisionStrideSeconds": horizon, "horizon": 1,
                "firstTimestamp": int(times[0]), "lastTimestamp": int(times[-1]),
                "lastRealizedTimestamp": int(times[-1] + horizon * 1000),
                "continuousBranchesPerNode": branches, "exactZeroAtom": True,
                "representatives": "conditional mean gross price ratio in equal-probability nonzero-return bins",
                "quadrature": "96-point Gauss-Legendre per triangular-knot interval in Gaussian-CDF coordinates",
                "maximumPriceRatioMeanError": max_price_mean_error,
                "maximumComponentMeanDifference96Vs384Quadrature": quadrature_comparison,
                "quadratureComparisonScope": "numerical convergence check, not a certified error bound",
                "maximumExpMeanLogVersusMeanPriceDifference": max_log_price_mean_difference,
                "features": "current and past observed inputs only; direct horizon targets, no hypothetical future features",
                "realizedPrices": "all actual one-second intermediate marks, for replay only; not inputs to decision",
                "executionScope": "fixed-hold long/cash endpoint problem, nonnegative cash and asset, no limits, no shorts, funding factor 1",
                "pathLimitation": "endpoint law does not determine intra-horizon leveraged liquidation or limit fills",
                "selection": payload["plan"]["selection"], "meanLossWeight": payload["plan"]["meanLossWeight"],
                "testUsedForSelection": False,
                "sourceMarket": "historical BTCUSDT spot one-second closes, not actual USD-M futures execution data",
                "seconds": time.monotonic() - started}
    write_json(output.with_suffix(".manifest.json"), manifest)
    print(json.dumps(manifest), flush=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, default=Path("data/training/runs/stochastic-crra-direct-horizons-gaussian-v2"))
    parser.add_argument("--output-directory", type=Path, default=Path("data/backtests"))
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 30, 120])
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument("--branches", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--count", type=int)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    for horizon in args.horizons:
        selection = json.loads((args.runs_root / f"h{horizon}" / "model-selection.json").read_text())
        output = args.output_directory / f"oracle-crra-horizon-h{horizon}.jsonl"
        export(Path(selection["selectedCheckpoint"]), output, args.split, args.branches, args.batch_size, args.count)


if __name__ == "__main__":
    main()
