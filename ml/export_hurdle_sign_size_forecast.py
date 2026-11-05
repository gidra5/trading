"""Export both frozen hurdle stages through the existing Oracle endpoint menu.

The full magnitude likelihood is evaluated during forecasting audits. This
export deliberately uses the same three-point conditional-mean approximation
as the preceding specialist comparisons, so execution assumptions stay fixed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from export_ternary_sign_size_forecast import endpoint_law, predict, record_for, sha
from export_stochastic_price_forecast import load_close_prices
from horizon_price_forecast import load_horizon_split
from hurdle_sign_size_forecast import CONTRACT, MINIMUM_MEAN, HurdleSignSize
from neural_sign_size_forecast import StructuredSpecialist


def frozen_selection(file: Path) -> dict:
    selected = json.loads(file.read_text(encoding="utf-8"))
    if selected["contract"] != CONTRACT or selected["allHeadSelectionsFrozen"] is not True \
            or selected["testLoaded"] is not False or selected["testUsedForSelection"] is not False:
        raise ValueError("all choices must be frozen before test loading")
    plan_file = file.parent/"plan.json"
    if sha(plan_file) != selected["planSha256"] or json.loads(plan_file.read_text(encoding="utf-8")) != selected["plan"]:
        raise ValueError("frozen plan differs")
    for row in selected["horizons"]:
        for stage in ("pretrained", "finetuned"):
            record = record_for(row, stage)
            if sha(Path(record["checkpoint"])) != record["sha256"]:
                raise ValueError("checkpoint changed after selection")
    return selected


def load_combined(record: dict, horizon: int, plan_sha: str, device: torch.device) -> HurdleSignSize:
    file = Path(record["checkpoint"])
    if sha(file) != record["sha256"]:
        raise ValueError("checkpoint hash mismatch")
    saved = torch.load(file, map_location="cpu", weights_only=True)
    if saved["contract"] != CONTRACT or saved["task"] != "combined" or saved["horizonsSeconds"] != [horizon] \
            or saved["epoch"] != record["epoch"] or saved["planSha256"] != plan_sha:
        raise ValueError("checkpoint provenance mismatch")
    mean, std = saved["inputMean"], saved["inputStd"]
    experts = [StructuredSpecialist(task, mean, std, horizons=(horizon,)) for task in ("sign", "sign", "size")]
    model = HurdleSignSize(*experts, saved["sizeScales"])
    model.load_state_dict(saved["modelState"], strict=True)
    return model.to(device).eval()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, default=Path("data/training/runs/stochastic-crra-hurdle-nll-v1"))
    parser.add_argument("--output", type=Path, default=Path("data/backtests/oracle-hurdle-nll"))
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    file = args.runs_root/"selection.json"
    selected_sha, selected = sha(file), frozen_selection(file)
    plan = json.loads((args.output/"replay-plan.json").read_text(encoding="utf-8"))
    if Path(plan["runsRoot"]).resolve() != args.runs_root.resolve():
        raise ValueError("replay plan references another training run")
    manifests = []
    for row in selected["horizons"]:
        h = row["horizonSeconds"]
        if h not in plan["horizonsSeconds"]:
            continue
        split = load_horizon_split(Path(selected["plan"]["dataset"]), "test", h)
        count = plan["decisionCountsByHorizon"][str(h)]
        origins = split.origins[::h][:count]
        times = split.source.axis_start_ms+origins*1000
        if len(origins) != count or count*h != plan["comparisonObservedSeconds"] \
                or int(times[0]) != plan["firstTimestamp"] or int(times[-1])+h*1000 != plan["terminalTimestamp"] \
                or np.any(np.diff(times) != h*1000):
            raise ValueError("incomplete common nonoverlapping replay interval")
        tape_times = np.arange(times[0], times[-1]+h*1000+1, 1000, dtype=np.int64)
        prices = load_close_prices(tape_times, Path("data/market/immutable/refs/candles/spot-btcusdt/btcusdt/1s"))
        np.testing.assert_array_equal(np.log(prices[h::h]/prices[:-h:h]).astype(np.float32), split.targets()[::h][:count])
        for stage in ("pretrained", "finetuned"):
            record = record_for(row, stage)
            model = load_combined(record, h, selected["planSha256"], device)
            p, up, down, model_mean = predict(model, split.source.timeline, origins, device)
            shapes = model.shapes.detach().cpu().tolist()
            downside_cap = model.downside_cap
            del model
            returns, probabilities, mean, size, normalization_error = endpoint_law(p, up, down)
            error = float(np.max(np.abs(mean-model_mean)))
            if error > 1e-10:
                raise ValueError("endpoint mean differs materially from the likelihood model mean")
            forecast_file = args.output/f"{stage}-h{h}.jsonl"
            with forecast_file.open("w", encoding="utf-8", newline="\n") as sink:
                for i in range(count):
                    offset = i*h
                    current = float(prices[offset])
                    value = {"horizonSeconds": h, "timestamp": int(times[i]), "observedPrice": current,
                        "realizedNextTimestamp": int(times[i]+h*1000), "realizedNextPrice": float(prices[offset+h]),
                        "realizedPrices": prices[offset+1:offset+h+1].tolist(),
                        "tree": {"children": [{"probability": float(probabilities[i,k]),
                            "price": current*float(1+returns[i,k])} for k in range(3)]}}
                    sink.write(json.dumps(value, separators=(",", ":"), allow_nan=False)+"\n")
            manifest = {"contract": CONTRACT, "variant": stage, "combined": record,
                "selectionFile": str(file), "selectionSha256": selected_sha, "headSelectionsFrozenBeforeTest": True,
                "refit": False, "reselection": False, "testUsedForSelection": False,
                "forecastFile": str(forecast_file), "forecastSha256": sha(forecast_file), "horizonSeconds": h,
                "split": "test", "rows": count, "firstTimestamp": int(times[0]),
                "lastRealizedTimestamp": int(times[-1]+h*1000), "observedSeconds": count*h,
                "probabilityOrder": ["down", "zero", "up"], "conditionalShapesUpDown": shapes,
                "predictedZeroProbabilityMean": float(probabilities[:,1].mean()),
                "predictedMinMeanBps": float(mean.min()*1e4), "predictedMaxMeanBps": float(mean.max()*1e4),
                "predictedSignedMeanBps": float(mean.mean()*1e4), "predictedAverageAbsoluteSizeBps": float(size.mean()*1e4),
                "fullFeeThresholdLongCount": int((mean > 1/.9995**2-1).sum()),
                "maximumSoftmaxSumErrorBeforeNormalization": normalization_error,
                "maximumExportVsTorchMeanErrorBps": error*1e4,
                "downsideCap": downside_cap, "downsideCapHits": int((down >= downside_cap).sum()),
                "minimumMean": MINIMUM_MEAN, "meanFloorHits": int((up <= MINIMUM_MEAN).sum()+(down <= MINIMUM_MEAN).sum()),
                "features": "two observed production59 states t-1,t; future prices only in realized replay fields",
                "sourceMarket": "BTCUSDT spot closes with synthetic Oracle execution parameters",
                "limitation": "Continuous nonzero likelihood approximates tick-quantized returns; this endpoint export further replaces each conditional magnitude law by its mean, losing spread."}
            forecast_file.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False)+"\n", encoding="utf-8")
            manifests.append(manifest)
            print(json.dumps({"horizon": h, "stage": stage, "maxMeanBps": manifest["predictedMaxMeanBps"],
                              "thresholdLongCount": manifest["fullFeeThresholdLongCount"]}), flush=True)
    if sha(file) != selected_sha:
        raise ValueError("selection changed during export")
    frozen_selection(file)
    (args.output/"exports.json").write_text(json.dumps({"contract": CONTRACT, "manifests": manifests}, indent=2)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
