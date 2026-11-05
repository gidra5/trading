"""Combine frozen neural sign/size specialists and export Oracle endpoint laws.

The sign expert gates upward/downward conditional-mean size components. This
three-point approximation does not retain within-sign spread or intrahorizon
crossings. Only the restricted long/cash endpoint policy consumes these files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from export_stochastic_price_forecast import load_close_prices
from horizon_price_forecast import load_horizon_split
from neural_sign_size_forecast import CONTRACT, StructuredSpecialist
from sign_size_forecast import moment_values, three_point_law


def sha(file: Path) -> str:
    return hashlib.sha256(file.read_bytes()).hexdigest()


def read_frozen_selection(file: Path) -> dict:
    selection = json.loads(file.read_text(encoding="utf-8"))
    if (selection["contract"] != CONTRACT or selection["allHeadSelectionsFrozen"] is not True
            or selection["testLoaded"] is not False or selection["testUsedForSelection"] is not False):
        raise ValueError("both specialists must be frozen before test export")
    plan_file = file.parent / "plan.json"
    if (sha(plan_file) != selection["planSha256"] or
            json.loads(plan_file.read_text(encoding="utf-8")) != selection["plan"]):
        raise ValueError("frozen selection does not match the prespecified training plan")
    for horizon in selection["horizons"]:
        for task in ("sign", "size"):
            checkpoint = Path(horizon[task]["checkpoint"])
            if sha(checkpoint) != horizon[task]["sha256"]:
                raise ValueError(f"{task} checkpoint differs from frozen selection")
    return selection


def load_specialist(record: dict, task: str, device: torch.device) -> StructuredSpecialist:
    file = Path(record["checkpoint"])
    if sha(file) != record["sha256"]:
        raise ValueError("checkpoint content changed")
    payload = torch.load(file, map_location="cpu", weights_only=True)
    if (payload["contract"] != CONTRACT or payload["task"] != task
            or payload["epoch"] != record["epoch"]):
        raise ValueError("unexpected specialist checkpoint")
    model = StructuredSpecialist(task, torch.as_tensor(payload["inputMean"]),
        torch.as_tensor(payload["inputStd"]), horizons=payload["horizonsSeconds"])
    model.load_state_dict(payload["modelState"], strict=True)
    return model.to(device).eval()


@torch.inference_mode()
def predict_pair(selection: dict, horizon: dict, timeline: np.ndarray,
                 origins: np.ndarray, device: torch.device, batch_size: int = 512) -> tuple:
    index = selection["plan"]["horizonsSeconds"].index(horizon["horizonSeconds"])
    values = {}
    for task in ("sign", "size"):
        model = load_specialist(horizon[task], task, device)
        if list(model.horizons) != selection["plan"]["horizonsSeconds"]:
            raise ValueError("specialist heads do not match frozen horizon order")
        batches = []
        for start in range(0, len(origins), batch_size):
            rows = origins[start:start+batch_size]
            raw = np.asarray(timeline[rows[:, None] + np.array([-1, 0])[None, :]], dtype=np.float32)
            result = model(torch.from_numpy(raw).to(device))[:, index]
            if task == "sign":
                result = result.sigmoid()
            batches.append(result.cpu().numpy().astype(np.float64))
        values[task] = np.concatenate(batches)
        del model
    constants = horizon["constants"]
    return (values["sign"], values["size"][:, 0] * constants["upMeanSimple"],
            values["size"][:, 1] * constants["downMeanSimple"])


def combined_laws(constants: dict, p_up: np.ndarray, up: np.ndarray, down: np.ndarray) -> dict:
    """Fixed ablations isolate each specialist without fitting another model."""
    count = len(p_up)
    return {
        "specialists": three_point_law(constants["pZero"], p_up, up, down),
        "sign-constant-size": three_point_law(constants["pZero"], p_up,
            np.full(count, constants["upMeanSimple"]), np.full(count, constants["downMeanSimple"])),
        "constant-sign-size": three_point_law(constants["pZero"],
            np.full(count, constants["pUpGivenActive"]), up, down),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path,
        default=Path("data/training/runs/stochastic-crra-neural-specialists-v1"))
    parser.add_argument("--output", type=Path, default=Path("data/backtests/oracle-neural-specialists"))
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    file = args.runs_root / "selection.json"
    selection_hash = sha(file)
    selection = read_frozen_selection(file)
    plan = json.loads((args.output / "replay-plan.json").read_text(encoding="utf-8"))
    if selection["plan"]["horizonsSeconds"] != [1, 900, 1800, 7200]:
        raise ValueError("unexpected prespecified horizon set")
    manifests = []
    for horizon in selection["horizons"]:
        seconds = horizon["horizonSeconds"]
        if seconds not in plan["horizonsSeconds"]:
            continue  # native one-second control is scored, not part of the 42h replay comparison
        split = load_horizon_split(Path(selection["plan"]["dataset"]), "test", seconds)
        count = plan["decisionCountsByHorizon"][str(seconds)]
        origins = split.origins[::seconds][:count]
        if len(origins) != count or count * seconds != plan["comparisonObservedSeconds"]:
            raise ValueError("incomplete common evaluation interval")
        times = split.source.axis_start_ms + origins * 1000
        if (int(times[0]) != plan["firstTimestamp"] or
                int(times[-1]) + seconds * 1000 != plan["terminalTimestamp"] or
                np.any(np.diff(times) != seconds * 1000)):
            raise ValueError("nonoverlapping replay interval differs from plan")
        p_up, up, down = predict_pair(selection, horizon, split.source.timeline, origins, device)
        laws = combined_laws(horizon["constants"], p_up, up, down)
        all_times = np.arange(times[0], times[-1] + seconds * 1000 + 1, 1000, dtype=np.int64)
        prices = load_close_prices(all_times, Path("data/market/immutable/refs/candles/spot-btcusdt/btcusdt/1s"))
        np.testing.assert_array_equal(np.log(prices[seconds::seconds] / prices[:-seconds:seconds]).astype(np.float32),
                                      split.targets()[::seconds][:count])
        for variant, (returns, probabilities, clipping) in laws.items():
            mean, size = moment_values(returns, probabilities)
            gross = 1 + returns
            represented = np.sum(probabilities * gross, axis=1)
            np.testing.assert_allclose(represented, 1 + mean, atol=2e-15, rtol=0)
            np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=2e-15, rtol=0)
            if not np.isfinite(gross).all() or np.any(gross <= 0):
                raise ValueError("invalid price support")
            forecast_file = args.output / f"{variant}-h{seconds}.jsonl"
            with forecast_file.open("w", encoding="utf-8", newline="\n") as sink:
                for index in range(count):
                    offset = index * seconds
                    observed = float(prices[offset])
                    row = {"horizonSeconds": seconds, "timestamp": int(times[index]), "observedPrice": observed,
                        "realizedNextTimestamp": int(times[index] + seconds * 1000),
                        "realizedNextPrice": float(prices[offset + seconds]),
                        "realizedPrices": prices[offset + 1:offset + seconds + 1].tolist(),
                        "tree": {"children": [{"probability": float(probabilities[index, branch]),
                            "price": observed * float(gross[index, branch])} for branch in range(3)]}}
                    sink.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
            manifest = {"contract": CONTRACT, "variant": variant, "specialists": {
                    task: horizon[task] for task in ("sign", "size")}, "constants": horizon["constants"],
                "selectionFile": str(file), "selectionSha256": selection_hash,
                "headSelectionsFrozenBeforeTest": True, "refit": False, "reselection": False,
                "testUsedForSelection": False, "forecastFile": str(forecast_file),
                "forecastSha256": sha(forecast_file), "split": "test", "rows": count,
                "horizonSeconds": seconds, "firstTimestamp": int(times[0]),
                "lastRealizedTimestamp": int(times[-1] + seconds * 1000), "observedSeconds": count * seconds,
                "maximumPriceRatioMeanError": float(np.abs(represented - 1 - mean).max()),
                "predictedSignedMeanBps": float(mean.mean() * 1e4),
                "predictedAverageAbsoluteSizeBps": float(size.mean() * 1e4),
                "predictedMinMeanBps": float(mean.min() * 1e4), "predictedMaxMeanBps": float(mean.max() * 1e4),
                "fullFeeThresholdLongCount": int((mean > 1 / .9995**2 - 1).sum()), "clipping": clipping,
                "features": "two current/past raw production59 states; no future features",
                "target": "conditional upward/downward ABS SIMPLE-return means and active-up probability",
                "sourceMarket": "BTCUSDT spot one-second closes; synthetic Oracle account parameters",
                "realizedPrices": "every intermediate one-second mark, replay-only; never model inputs",
                "limitations": "three-point conditional-mean approximation; restricted long/cash endpoint menu; no full-path or continuous-law optimality claim"}
            forecast_file.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2,
                allow_nan=False) + "\n", encoding="utf-8")
            manifests.append(manifest)
            print(json.dumps({"horizon": seconds, "variant": variant,
                "maxMeanBps": manifest["predictedMaxMeanBps"],
                "thresholdLongCount": manifest["fullFeeThresholdLongCount"]}), flush=True)
    if sha(file) != selection_hash:
        raise ValueError("frozen selection changed during export")
    read_frozen_selection(file)
    (args.output / "exports.json").write_text(json.dumps({"contract": CONTRACT, "manifests": manifests},
        indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
