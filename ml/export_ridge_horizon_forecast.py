"""Export the already selected ridge mixture; no fitting or selection occurs.

Each active Gaussian log-return bin uses its analytic conditional price-ratio
mean. Actual intermediate prices are included solely for the Oracle replay.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import ndtr, ndtri
import torch

from export_stochastic_price_forecast import load_close_prices
from horizon_price_forecast import load_horizon_split


def digest(file: Path) -> str:
    return hashlib.sha256(file.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path,
        default=Path("data/training/runs/stochastic-crra-ridge-horizon-search-v1/validation-selection.json"))
    parser.add_argument("--output", type=Path,
        default=Path("data/backtests/oracle-horizon-search/ridge-h7200.jsonl"))
    parser.add_argument("--branches", type=int, default=7)
    args = parser.parse_args()
    if args.branches < 1:
        raise ValueError("at least one active Gaussian bin is required")
    torch.set_num_threads(2)
    selection_hash = digest(args.selection)
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    horizon = selection["selectedHorizonSeconds"]
    name = selection["selectedCandidate"]
    if not isinstance(horizon, int) or horizon < 1 or not name:
        raise ValueError("validation did not select an eligible horizon and candidate")
    if selection["testLoaded"] is not False or selection["plan"]["testUsedForSelection"] is not False:
        raise ValueError("expected a frozen pre-test validation choice")
    file = args.selection.parent / f"h{horizon}.npz"
    model_hash = digest(file)
    with np.load(file, allow_pickle=False) as saved:
        names = saved["candidateNames"].tolist()
        index = names.index(name)
        coefficient = saved["coefficients"][index]
        intercept = float(saved["intercepts"][index])
        variance = float(saved["residualVariances"][index])
        p0 = float(saved["zeroProbability"])
        feature_mean, feature_std = saved["featureMean"], saved["featureStd"]
    if not np.isfinite(variance) or variance <= 0 or not 0 < p0 < 1:
        raise ValueError("invalid saved mixture parameters")
    sigma = np.sqrt(variance)
    split = load_horizon_split(Path(selection["plan"]["dataset"]), "test", horizon)
    origins = split.origins[::horizon]
    features = (split.features(origins).astype(np.float64) - feature_mean) / feature_std
    mu = features @ coefficient + intercept
    active_mean = np.exp(mu + variance / 2)
    cuts = ndtri(np.arange(args.branches + 1) / args.branches)
    bin_factors = args.branches * np.diff(ndtr(cuts - sigma))
    gross = np.concatenate((np.ones((len(origins), 1)), active_mean[:, None] * bin_factors), axis=1)
    probabilities = np.array([p0] + [(1 - p0) / args.branches] * args.branches)
    represented_mean = gross @ probabilities
    analytic_mean = 1 + (1 - p0) * np.expm1(mu + variance / 2)
    np.testing.assert_allclose(represented_mean, analytic_mean, atol=2e-15, rtol=0)
    if not np.isfinite(gross).all() or np.any(gross <= 0) or np.any(np.diff(gross[:, 1:], axis=1) <= 0):
        raise ValueError("invalid active conditional price-ratio means")
    times = split.source.axis_start_ms + origins * 1000
    if not len(times) or np.any(np.diff(times) != horizon * 1000):
        raise ValueError("expected complete contiguous nonoverlapping origins")
    all_times = np.arange(times[0], times[-1] + horizon * 1000 + 1, 1000, dtype=np.int64)
    prices = load_close_prices(all_times, Path("data/market/immutable/refs/candles/spot-btcusdt/btcusdt/1s"))
    np.testing.assert_array_equal(np.log(prices[horizon::horizon] / prices[:-horizon:horizon]).astype(np.float32),
                                  split.targets()[::horizon])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as sink:
        for index in range(len(origins)):
            offset = index * horizon
            observed = float(prices[offset])
            row = {"horizonSeconds": horizon, "timestamp": int(times[index]), "observedPrice": observed,
                   "realizedNextTimestamp": int(times[index] + horizon * 1000),
                   "realizedNextPrice": float(prices[offset + horizon]),
                   "realizedPrices": prices[offset + 1:offset + horizon + 1].tolist(),
                   "tree": {"children": [{"probability": float(probabilities[branch]),
                                          "price": observed * float(gross[index, branch])}
                                         for branch in range(args.branches + 1)]}}
            sink.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
    if digest(args.selection) != selection_hash or digest(file) != model_hash:
        raise ValueError("selected model or selection file changed while exporting")
    manifest = {"contract": "causal-production178-fixed-zero-lognormal-ridge-endpoint-v1",
                "checkpoint": str(file), "checkpointSha256": model_hash, "selectedCandidate": name,
                "selectionFile": str(args.selection), "selectionSha256": selection_hash,
                "selectionFrozenBeforeTest": True, "refit": False, "reselection": False,
                "testUsedForSelection": False, "split": "test", "forecastFile": str(args.output),
                "forecastSha256": digest(args.output), "rows": len(origins), "horizonSeconds": horizon,
                "decisionStrideSeconds": horizon, "horizon": 1, "continuousBranchesPerNode": args.branches,
                "exactZeroAtom": True, "zeroProbability": p0, "activeLogReturnResidualVariance": variance,
                "firstTimestamp": int(times[0]), "lastTimestamp": int(times[-1]),
                "lastRealizedTimestamp": int(times[-1] + horizon * 1000),
                "observedSeconds": len(origins) * horizon,
                "representatives": "analytic conditional price-ratio means in equal-probability active Gaussian bins",
                "formula": "B exp(mu+sigma^2/2) [Phi(z_hi-sigma)-Phi(z_lo-sigma)], z_j=Phi^-1(j/B)",
                "maximumPriceRatioMeanError": float(np.max(np.abs(represented_mean - analytic_mean))),
                "predictedGrossExcessBpsMin": float((analytic_mean.min() - 1) * 1e4),
                "predictedGrossExcessBpsMax": float((analytic_mean.max() - 1) * 1e4),
                "predictedFullFeeLongCount": int((analytic_mean > 1 / .9995**2).sum()),
                "features": "178 causal inputs: 120 observed one-second log-return lags and 58 latest observed features, saved train-only means/scales",
                "priceQuantization": "exported endpoint representatives are unrounded; Oracle adapter applies its declared price grid",
                "sourceMarket": "BTCUSDT spot one-second closes, synthetic Oracle account parameters",
                "realizedPrices": "every actual intermediate one-second mark, replay-only and never a prediction input",
                "executionScope": "nonnegative-cash/nonnegative-asset committed hold, no pending orders, unit funding",
                "limitations": "mean conservation applies before price-grid rounding; capped-close payoffs can be nonlinear within a bin; no full-path or continuous-law optimum claimed"}
    args.output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
