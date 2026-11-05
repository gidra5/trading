"""Validation-only economic gates, including pre-existing selected checkpoints."""
import argparse
import json
from pathlib import Path

import torch

from horizon_price_forecast import load_horizon_forecast, load_horizon_split
from train_horizon_price_forecast import economic_diagnostics
from train_stochastic_price_forecast import write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, default=Path("data/training/runs/stochastic-crra-direct-horizons-gaussian-v2"))
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 30, 120])
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    for horizon in args.horizons:
        root = args.runs_root / f"h{horizon}"
        selection = json.loads((root / "model-selection.json").read_text())
        models = {}
        for candidate in selection["candidates"]:
            model, payload = load_horizon_forecast(Path(candidate["selectedCheckpoint"]), device)
            validation = load_horizon_split(Path(payload["plan"]["dataset"]), "validation", horizon)
            x, y = torch.from_numpy(validation.features()).to(device), torch.from_numpy(validation.targets().copy()).to(device)
            models[f"lambda{candidate['meanLossWeight']:g}"] = {
                "checkpoint": candidate["selectedCheckpoint"],
                "validationScore": candidate["validation"],
                "economics": economic_diagnostics(model, x, y, 4096, horizon)}
        selected_economics = models[f"lambda{selection['selectedMeanLossWeight']:g}"]["economics"]
        gate_passed = bool(selected_economics["finitePredictedMeans"] and
                           selected_economics["longCashThresholds"]["0.0005"]["nonconstantPredictedLongCashDecision"])
        result = {"split": "validation", "horizonSeconds": horizon, "decisionStrideSeconds": horizon,
                  "selectedCheckpoint": selection["selectedCheckpoint"], "selectedMeanLossWeight": selection["selectedMeanLossWeight"],
                  "testLoadedByThisAudit": False, "changesModelSelection": False,
                  "selectedModelGatePassed": gate_passed,
                  "economicGate": "finite predicted means and both strict long and strict cash winners across nonoverlapping validation contexts; fee .0005 each side",
                  "calibrationEvidence": "empirical conditional realized returns only; no uniform action-value error bound is established",
                  "models": models}
        output = root / "validation-economic-gate.json"
        write_json(output, result)
        print(json.dumps({"horizonSeconds": horizon, "validationGate": str(output),
                          "selectedEconomics": models[f"lambda{selection['selectedMeanLossWeight']:g}"]["economics"]}), flush=True)


if __name__ == "__main__":
    main()
