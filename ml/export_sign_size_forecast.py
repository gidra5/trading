"""Export frozen sign/conditional-size selections as declared three-point laws.

Both variants retain the selected sign head. The control replaces each magnitude
head with its own sign-conditional training constant, isolating contextual size
prediction. No fitting, test-driven selection, or perfect-sign labels are used.
"""
from __future__ import annotations

import os
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "4"

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import torch

from export_stochastic_price_forecast import load_close_prices
from horizon_price_forecast import load_horizon_split
from sign_size_forecast import CONTRACT, moment_values, predict_laws


def sha(file: Path) -> str:
    return hashlib.sha256(file.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, default=Path("data/training/runs/stochastic-crra-sign-size-v1"))
    parser.add_argument("--output", type=Path, default=Path("data/backtests/oracle-sign-size"))
    args = parser.parse_args()
    torch.set_num_threads(4)
    frozen_file = args.runs_root / "selection.json"
    frozen_hash = sha(frozen_file)
    frozen = json.loads(frozen_file.read_text(encoding="utf-8"))
    if (frozen["contract"] != CONTRACT or frozen["allHeadSelectionsFrozen"] is not True or
            frozen["testLoaded"] is not False or frozen["testUsedForSelection"] is not False):
        raise ValueError("all horizon/head choices must be frozen before test export")
    plan = frozen["plan"]
    horizons = plan["horizonsSeconds"]
    if horizons != [900, 1800, 3600, 7200]:
        raise ValueError("unexpected prespecified horizon set")
    args.output.mkdir(parents=True, exist_ok=True)
    manifests = []
    for horizon in horizons:
        selected_file = args.runs_root / f"h{horizon}" / "selection.json"
        selected_hash = sha(selected_file)
        selected = json.loads(selected_file.read_text(encoding="utf-8"))
        frozen_rows = [row for row in frozen["horizons"] if row["horizonSeconds"] == horizon]
        if len(frozen_rows) != 1 or selected != frozen_rows[0]:
            raise ValueError("per-horizon selection differs from the frozen all-horizon record")
        if selected["testLoaded"] is not False or selected["testUsedForSelection"] is not False:
            raise ValueError("expected pre-test head selection")
        model_file = Path(selected["modelFile"])
        model_hash = sha(model_file)
        if model_hash != selected["modelSha256"]:
            raise ValueError("model content does not match frozen selection")
        artifact = joblib.load(model_file)
        if (artifact["contract"] != CONTRACT or artifact["horizonSeconds"] != horizon or
                artifact["selectedHeads"] != selected["selectedHeads"]):
            raise ValueError("saved model does not implement the selected heads")
        split = load_horizon_split(Path(plan["dataset"]), "test", horizon)
        count = plan["replayCounts"][str(horizon)]
        origins = split.origins[::horizon][:count]
        if len(origins) != count or count*horizon != 151200:
            raise ValueError("incomplete common 42-hour evaluation window")
        times = split.source.axis_start_ms + origins*1000
        if np.any(np.diff(times) != horizon*1000):
            raise ValueError("test intervals must be contiguous and nonoverlapping")
        laws = predict_laws(artifact, split.features(origins))
        all_times = np.arange(times[0], times[-1]+horizon*1000+1, 1000, dtype=np.int64)
        prices = load_close_prices(all_times, Path("data/market/immutable/refs/candles/spot-btcusdt/btcusdt/1s"))
        np.testing.assert_array_equal(np.log(prices[horizon::horizon]/prices[:-horizon:horizon]).astype(np.float32),
                                      split.targets()[::horizon][:count])
        for variant, prefix in (("full", "factorized"), ("signConstantConditionalSizes", "sign-constant-size")):
            returns, probabilities, clipping = laws[variant]
            mean, size = moment_values(returns, probabilities)
            gross = 1+returns
            represented = np.sum(probabilities*gross, axis=1)
            np.testing.assert_allclose(represented, 1+mean, atol=2e-15, rtol=0)
            np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=2e-15, rtol=0)
            if not np.isfinite(gross).all() or np.any(gross <= 0):
                raise ValueError("three-point support must have finite positive prices")
            file = args.output / f"{prefix}-h{horizon}.jsonl"
            with file.open("w", encoding="utf-8", newline="\n") as sink:
                for index in range(count):
                    offset = index*horizon
                    observed = float(prices[offset])
                    row = {"horizonSeconds": horizon, "timestamp": int(times[index]), "observedPrice": observed,
                           "realizedNextTimestamp": int(times[index]+horizon*1000),
                           "realizedNextPrice": float(prices[offset+horizon]),
                           "realizedPrices": prices[offset+1:offset+horizon+1].tolist(),
                           "tree": {"children": [{"probability": float(probabilities[index, branch]),
                                                  "price": observed*float(gross[index, branch])}
                                                 for branch in range(3)]}}
                    sink.write(json.dumps(row, separators=(",", ":"), allow_nan=False)+"\n")
            manifest = {"contract": CONTRACT, "variant": variant, "checkpoint": str(model_file),
                        "checkpointSha256": model_hash, "selectedHeads": selected["selectedHeads"],
                        "selectionFile": str(selected_file), "selectionSha256": selected_hash,
                        "allHorizonSelectionFile": str(frozen_file), "allHorizonSelectionSha256": frozen_hash,
                        "headSelectionsFrozenBeforeTest": True, "refit": False, "reselection": False,
                        "testUsedForSelection": False, "forecastFile": str(file), "forecastSha256": sha(file),
                        "split": "test", "rows": count, "horizonSeconds": horizon,
                        "decisionStrideSeconds": horizon, "horizon": 1, "branches": 3, "exactZeroAtom": True,
                        "firstTimestamp": int(times[0]), "lastTimestamp": int(times[-1]),
                        "lastRealizedTimestamp": int(times[-1]+horizon*1000), "observedSeconds": count*horizon,
                        "priceRatioSupport": "1-m_minus, 1, 1+m_plus",
                        "probabilities": "(1-p0)*(1-p_up), p0, (1-p0)*p_up",
                        "maximumPriceRatioMeanError": float(np.abs(represented-1-mean).max()),
                        "predictedSignedMeanBps": float(mean.mean()*1e4),
                        "predictedAverageAbsoluteSizeBps": float(size.mean()*1e4),
                        "predictedMinMeanBps": float(mean.min()*1e4), "predictedMaxMeanBps": float(mean.max()*1e4),
                        "fullFeeThresholdLongCount": int((mean > 1/.9995**2-1).sum()),
                        "clipping": clipping,
                        "features": "178 causal inputs: 120 observed return lags and 58 latest observed features; no future values",
                        "target": "ABS SIMPLE-return magnitude, separately conditional on positive and negative sign",
                        "sourceMarket": "BTCUSDT spot one-second closes, synthetic Oracle account parameters",
                        "realizedPrices": "all actual intermediate one-second marks, replay-only and never model inputs",
                        "priceQuantization": "unrounded representative prices; the Oracle adapter applies its declared price grid",
                        "executionScope": "committed long/cash hold with nonnegative balances, no pending orders, unit funding",
                        "limitations": "three-point law retains supplied conditional first moments but omits within-sign spread and intermediate path; capped-close payoffs can be nonlinear; no unrestricted continuous-law optimality claimed"}
            file.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False)+"\n", encoding="utf-8")
            manifests.append(manifest)
            print(json.dumps({"file": str(file), "rows": count, "variant": variant,
                              "meanBps": manifest["predictedSignedMeanBps"],
                              "sizeBps": manifest["predictedAverageAbsoluteSizeBps"],
                              "maxMeanBps": manifest["predictedMaxMeanBps"],
                              "thresholdLongCount": manifest["fullFeeThresholdLongCount"]}), flush=True)
        if sha(selected_file) != selected_hash or sha(model_file) != model_hash:
            raise ValueError("frozen head selection/model changed during export")
    if sha(frozen_file) != frozen_hash:
        raise ValueError("all-horizon selection changed during export")
    (args.output / "exports.json").write_text(json.dumps({"contract": CONTRACT, "manifests": manifests},
        indent=2, allow_nan=False)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
