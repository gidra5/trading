"""Export before/after jointly tuned ternary sign-size laws for Oracle replay."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from export_stochastic_price_forecast import load_close_prices
from horizon_price_forecast import load_horizon_split
from neural_sign_size_forecast import StructuredSpecialist
from ternary_sign_size_forecast import CONTRACT, CombinedSignSize, TernarySignSpecialist


def sha(file: Path) -> str:
    return hashlib.sha256(file.read_bytes()).hexdigest()


def record_for(horizon: dict, stage: str) -> dict:
    return horizon["pretrained"]["combined"] if stage == "pretrained" else horizon["finetuned"]


def frozen_selection(file: Path) -> dict:
    selection = json.loads(file.read_text(encoding="utf-8"))
    if (selection["contract"] != CONTRACT or selection["allHeadSelectionsFrozen"] is not True
            or selection["testLoaded"] is not False or selection["testUsedForSelection"] is not False):
        raise ValueError("all fine-tuning selections must be frozen before export")
    plan_file = file.parent / "plan.json"
    if (sha(plan_file) != selection["planSha256"] or
            json.loads(plan_file.read_text(encoding="utf-8")) != selection["plan"]):
        raise ValueError("frozen selection does not match its training plan")
    for horizon in selection["horizons"]:
        for stage in ("pretrained", "finetuned"):
            record = record_for(horizon, stage)
            if sha(Path(record["checkpoint"])) != record["sha256"]:
                raise ValueError("combined checkpoint changed since selection")
    return selection


def load_combined(record: dict, horizon: int, plan_sha: str, device: torch.device) -> CombinedSignSize:
    file = Path(record["checkpoint"])
    if sha(file) != record["sha256"]:
        raise ValueError("checkpoint hash mismatch")
    saved = torch.load(file, map_location="cpu", weights_only=True)
    if (saved["contract"] != CONTRACT or saved["task"] != "combined"
            or saved["horizonsSeconds"] != [horizon] or saved["epoch"] != record["epoch"]
            or saved["planSha256"] != plan_sha):
        raise ValueError("checkpoint does not match selected combined model")
    mean, std = torch.as_tensor(saved["inputMean"]), torch.as_tensor(saved["inputStd"])
    sign = TernarySignSpecialist(mean, std, horizons=[horizon])
    size = StructuredSpecialist("size", mean, std, horizons=[horizon])
    model = CombinedSignSize(sign, size, torch.as_tensor(saved["sizeScales"]))
    model.load_state_dict(saved["modelState"], strict=True)
    return model.to(device).eval()


def endpoint_law(probability: np.ndarray, up: np.ndarray, down: np.ndarray) -> tuple:
    """Normalize floating softmax roundoff; preserve its down/zero/up ordering."""
    p = np.asarray(probability, dtype=np.float64)
    up, down = np.asarray(up, dtype=np.float64), np.asarray(down, dtype=np.float64)
    if p.shape != (len(up), 3) or down.shape != up.shape or up.ndim != 1:
        raise ValueError("ternary components have inconsistent shapes")
    if any(not np.isfinite(v).all() for v in (p, up, down)) or np.any(p < 0) \
            or np.any(p.sum(axis=1) <= 0) or np.any(up < 0) or np.any((down < 0) | (down >= 1)):
        raise ValueError("invalid ternary probability or physical price support")
    probability_sum_error = float(np.max(np.abs(p.sum(axis=1)-1)))
    if probability_sum_error > 1e-5:
        raise ValueError("model softmax probabilities do not sum to one")
    p = p / p.sum(axis=1, keepdims=True)
    returns = np.stack((-down, np.zeros(len(up)), up), axis=1)
    mean = (returns*p).sum(axis=1)
    size = (np.abs(returns)*p).sum(axis=1)
    np.testing.assert_allclose(p.sum(axis=1), 1, atol=2e-15, rtol=0)
    return returns, p, mean, size, probability_sum_error


@torch.inference_mode()
def predict(model: CombinedSignSize, timeline: np.ndarray, origins: np.ndarray,
            device: torch.device, batch_size: int = 2560) -> tuple:
    probabilities, ups, downs, means = [], [], [], []
    for start in range(0, len(origins), batch_size):
        rows = origins[start:start+batch_size]
        raw = np.asarray(timeline[rows[:, None] + np.array([-1, 0])[None, :]], dtype=np.float32)
        x = torch.from_numpy(raw).to(device)
        p, up, down = model.components(x)
        probabilities.append(p[:, 0].cpu().numpy().astype(np.float64))
        ups.append(up[:, 0].cpu().numpy().astype(np.float64))
        downs.append(down[:, 0].cpu().numpy().astype(np.float64))
        means.append((p[:, 0, 2]*up[:, 0]-p[:, 0, 0]*down[:, 0]).cpu().numpy().astype(np.float64))
    return tuple(np.concatenate(values) for values in (probabilities, ups, downs, means))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, default=Path("data/training/runs/stochastic-crra-ternary-finetune-v1"))
    parser.add_argument("--output", type=Path, default=Path("data/backtests/oracle-ternary-finetune"))
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    file = args.runs_root / "selection.json"
    selection_sha = sha(file)
    frozen = frozen_selection(file)
    plan = json.loads((args.output / "replay-plan.json").read_text(encoding="utf-8"))
    manifests = []
    for selected in frozen["horizons"]:
        horizon = selected["horizonSeconds"]
        if horizon not in plan["horizonsSeconds"]:
            continue
        split = load_horizon_split(Path(frozen["plan"]["dataset"]), "test", horizon)
        count = plan["decisionCountsByHorizon"][str(horizon)]
        origins = split.origins[::horizon][:count]
        times = split.source.axis_start_ms + origins*1000
        if (len(origins) != count or count*horizon != plan["comparisonObservedSeconds"]
                or int(times[0]) != plan["firstTimestamp"]
                or int(times[-1])+horizon*1000 != plan["terminalTimestamp"]
                or np.any(np.diff(times) != horizon*1000)):
            raise ValueError("incomplete common nonoverlapping replay interval")
        all_times = np.arange(times[0], times[-1]+horizon*1000+1, 1000, dtype=np.int64)
        prices = load_close_prices(all_times, Path("data/market/immutable/refs/candles/spot-btcusdt/btcusdt/1s"))
        np.testing.assert_array_equal(np.log(prices[horizon::horizon]/prices[:-horizon:horizon]).astype(np.float32),
                                      split.targets()[::horizon][:count])
        for stage in ("pretrained", "finetuned"):
            record = record_for(selected, stage)
            model = load_combined(record, horizon, frozen["planSha256"], device)
            p, up, down, model_mean = predict(model, split.source.timeline, origins, device)
            downside_cap = model.downside_cap
            del model
            returns, probabilities, mean, size, probability_sum_error = endpoint_law(p, up, down)
            mean_error = float(np.max(np.abs(mean-model_mean)))
            if mean_error > 1e-7:
                raise ValueError("exported mean differs materially from trained composed mean")
            gross = 1+returns
            np.testing.assert_allclose(np.sum(probabilities*gross, axis=1), 1+mean, atol=2e-15, rtol=0)
            forecast_file = args.output / f"{stage}-h{horizon}.jsonl"
            with forecast_file.open("w", encoding="utf-8", newline="\n") as sink:
                for i in range(count):
                    offset, price = i*horizon, float(prices[i*horizon])
                    row = {"horizonSeconds": horizon, "timestamp": int(times[i]), "observedPrice": price,
                        "realizedNextTimestamp": int(times[i]+horizon*1000),
                        "realizedNextPrice": float(prices[offset+horizon]),
                        "realizedPrices": prices[offset+1:offset+horizon+1].tolist(),
                        "tree": {"children": [{"probability": float(probabilities[i, b]),
                            "price": price*float(gross[i, b])} for b in range(3)]}}
                    sink.write(json.dumps(row, separators=(",", ":"), allow_nan=False)+"\n")
            manifest = {"contract": CONTRACT, "variant": stage, "combined": record,
                "selectionFile": str(file), "selectionSha256": selection_sha,
                "headSelectionsFrozenBeforeTest": True, "refit": False, "reselection": False,
                "testUsedForSelection": False, "forecastFile": str(forecast_file), "forecastSha256": sha(forecast_file),
                "horizonSeconds": horizon, "split": "test", "rows": count,
                "firstTimestamp": int(times[0]), "lastRealizedTimestamp": int(times[-1]+horizon*1000),
                "observedSeconds": count*horizon, "probabilityOrder": ["down", "zero", "up"],
                "predictedZeroProbabilityMean": float(probabilities[:, 1].mean()),
                "predictedMinMeanBps": float(mean.min()*1e4), "predictedMaxMeanBps": float(mean.max()*1e4),
                "predictedSignedMeanBps": float(mean.mean()*1e4), "predictedAverageAbsoluteSizeBps": float(size.mean()*1e4),
                "fullFeeThresholdLongCount": int((mean > 1/.9995**2-1).sum()),
                "maximumSoftmaxSumErrorBeforeNormalization": probability_sum_error,
                "maximumExportVsTorchMeanErrorBps": mean_error*1e4,
                "downsideCap": downside_cap, "downsideCapHits": int((down >= downside_cap).sum()),
                "features": "same two observed production59 states at t-1,t; actual future prices only in replay fields",
                "sourceMarket": "BTCUSDT spot closes with synthetic Oracle execution parameters",
                "limitation": "Pure mean-MSE tuning does not identify the true probabilities or size law. The three-point support is an approximation consumed by the finite-menu endpoint policy."}
            forecast_file.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False)+"\n", encoding="utf-8")
            manifests.append(manifest)
            print(json.dumps({"horizon": horizon, "stage": stage, "maxMeanBps": manifest["predictedMaxMeanBps"],
                "thresholdLongCount": manifest["fullFeeThresholdLongCount"]}), flush=True)
    if sha(file) != selection_sha:
        raise ValueError("frozen selection changed during export")
    frozen_selection(file)
    (args.output / "exports.json").write_text(json.dumps({"contract": CONTRACT, "manifests": manifests},
        indent=2, allow_nan=False)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
