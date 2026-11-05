"""Export causal two-tick scenario trees for contiguous historical decisions."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from stochastic_price_forecast import CONTRACT, load_forecast, load_timeline_split, price_tree, two_tick_scenarios
from trading_storage import candle_times, read_candle_column
from train_stochastic_price_forecast import write_json


def load_close_prices(times: np.ndarray, source: Path) -> np.ndarray:
    """Input times are completed-second boundaries, one second after open."""
    open_times = times - 1000
    days = open_times // 86_400_000
    result = np.empty(len(times), dtype=np.float64)
    for day in np.unique(days):
        date = datetime.fromtimestamp(int(day * 86400), timezone.utc).strftime("%Y-%m-%d")
        file = source / (date + ".json")
        candle_time = candle_times(file)
        close = read_candle_column(file, "close")
        selected = np.flatnonzero(days == day)
        indices = np.searchsorted(candle_time, open_times[selected])
        if np.any(indices >= len(candle_time)) or np.any(candle_time[indices] != open_times[selected]):
            raise ValueError("missing observed candle at a forecast boundary")
        result[selected] = close[indices]
    return result


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("data/training/runs/stochastic-crra-production178-zeroatom-k256-v1/best.pt"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--count", type=int)
    parser.add_argument("--branches", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    model, payload = load_forecast(args.checkpoint, device)
    dataset = Path(payload["plan"]["dataset"])
    split = load_timeline_split(dataset, args.split)
    stop = None if args.count is None else args.offset + args.count
    origins = split.origins[args.offset:stop]
    if not origins.size:
        raise ValueError("requested forecast interval is empty")
    times = split.axis_start_ms + origins * 1000
    if np.any(np.diff(times) != 1000):
        raise ValueError("forecast decisions must be contiguous seconds")
    prices = load_close_prices(np.concatenate((times, times[-1:] + 1000)),
                              Path("data/market/immutable/refs/candles/spot-btcusdt/btcusdt/1s"))
    realized = np.log(prices[1:] / prices[:-1]).astype(np.float32)
    np.testing.assert_allclose(realized, split.targets(origins), atol=2e-11, rtol=1e-6,
                               err_msg="market prices differ from exported training target alignment")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    start_time = time.monotonic()
    maximum_first_mean_error = 0.0
    with args.output.open("w", encoding="utf-8") as output:
        for start in range(0, len(origins), args.batch_size):
            batch_origins = origins[start:start + args.batch_size]
            features = torch.from_numpy(split.features(batch_origins)).to(device)
            first_returns, first_probabilities, second_returns, second_probabilities = two_tick_scenarios(model, features, args.branches)
            exact_mean = model.expectation_from_logits(model(features)).cpu().numpy()
            maximum_first_mean_error = max(maximum_first_mean_error,
                float(np.max(np.abs((first_returns * first_probabilities).sum(axis=1) - exact_mean))))
            for local in range(len(batch_origins)):
                index = start + local
                observed_price = float(prices[index])
                row = {"timestamp": int(times[index]), "observedPrice": observed_price,
                       "realizedNextTimestamp": int(times[index] + 1000),
                       "realizedNextPrice": float(prices[index + 1]),
                       "tree": price_tree(observed_price, first_returns[local], first_probabilities[local],
                                          second_returns[local], second_probabilities[local])}
                output.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
            if (start // args.batch_size) % 20 == 0:
                print(json.dumps({"rows": min(start + args.batch_size, len(origins)), "total": len(origins),
                                  "seconds": time.monotonic() - start_time}), flush=True)
    manifest = {"contract": CONTRACT, "checkpoint": str(args.checkpoint),
                "checkpointSha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                "checkpointSelection": "validation mixed-law NLL; no test/backtest outcome selection",
                "forecastFile": str(args.output), "split": args.split, "rows": len(origins),
                "firstTimestamp": int(times[0]), "lastTimestamp": int(times[-1]), "tickMilliseconds": 1000,
                "horizon": 2, "continuousBranchesPerNode": args.branches, "exactZeroAtom": True,
                "representatives": "conditional mean in equal-probability active-return quantile bins",
                "quadrature": "64-point Gauss-Legendre per triangular-knot interval",
                "maximumFirstReturnMeanError": maximum_first_mean_error,
                "rolloutInputs": "update 120 return lags with the hypothetical return; freeze 58 other observed features, including derived price features",
                "sourceMarket": "historical BTCUSDT spot one-second closes; not actual USD-M futures execution data",
                "priceQuantization": "left to execution adapter; exported prices are unrounded",
                "optimalityScope": "finite forecast tree only; quadrature/compression and learned distribution are not exact true-law equivalents",
                "seconds": time.monotonic() - start_time}
    write_json(args.output.with_suffix(".manifest.json"), manifest)
    print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
