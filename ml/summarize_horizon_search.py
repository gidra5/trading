"""Summarize a prespecified ascending horizon screen without changing selection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from train_stochastic_price_forecast import write_json


def economic_row(economics: dict) -> dict:
    threshold = economics["longCashThresholds"]["0.0005"]
    return {"contexts": economics["contexts"],
            "forecastMeanQuantilesBps": economics["forecastReturnQuantilesBps"],
            "forecastMeanAverageBps": economics["forecastReturnMeanBps"],
            "forecastMeanStdBps": economics["forecastReturnStdBps"],
            "positiveCrossingsAboveFeeThreshold": threshold["chooseLongCount"],
            "otherContexts": threshold["chooseCashCount"],
            "meanRealizedNetReturnWhenCrossingBps": threshold["realizedGainWhenLongBps"],
            "averageRealizedNetGainAcrossContextsBps": threshold["realizedMeanPolicyGainBps"],
            "predictedNonconstantLongCashDecision": threshold["nonconstantPredictedLongCashDecision"]}


def summarize(root: Path) -> dict:
    plan = json.loads((root / "search-plan.json").read_text())
    rows = []
    for horizon in plan["horizonsSecondsAscending"]:
        selected_file = root / f"h{horizon}" / "model-selection.json"
        if not selected_file.exists():
            continue
        selection = json.loads(selected_file.read_text())
        candidate_root = Path(selection["selectedCheckpoint"]).parent
        result = json.loads((candidate_root / "result.json").read_text())
        name = selection["selectedModel"]
        validation = economic_row(result["validationEconomics"][name])
        stride_validation = economic_row(result["nonoverlappingValidationEconomics"][name])
        test = result["test"][name]
        row = {"horizonSeconds": horizon, "selectedModel": name,
               "selectedMeanLossWeight": selection["selectedMeanLossWeight"],
               "selectedCheckpoint": selection["selectedCheckpoint"],
               "selectedEpoch": result["selectedEpoch"] if name == "conditionalGlu" else 0,
               "positiveValidationCrossing": validation["positiveCrossingsAboveFeeThreshold"] > 0,
               "allOverlappingValidation": validation,
               "strideHValidation": stride_validation,
               "validationDistribution": result["selection"]["validation"][name],
               "testDistribution": test["distribution"],
               "allOverlappingTest": economic_row(test["economics"]),
               "trainingCounts": result["plan"]["counts"],
               "candidateResults": selection["candidates"]}
        rows.append(row)
    crossing = next((row["horizonSeconds"] for row in rows if row["positiveValidationCrossing"]), None)
    amendment = plan.get("executionAmendment")
    stopped = bool(amendment and any(row["horizonSeconds"] == amendment["stopNeuralGridAfterSeconds"] for row in rows))
    return {"plan": plan, "completedHorizonsSeconds": [row["horizonSeconds"] for row in rows],
            "firstSelectedValidationCrossingSeconds": crossing,
            "prespecifiedStopReached": crossing is not None,
            "executionStoppedByAmendment": stopped,
            "remainingPlannedHorizonsNotRun": amendment["remainingPlannedHorizonsNotRun"] if stopped else [],
            "allPrespecifiedHorizonsComplete": len(rows) == len(plan["horizonsSecondsAscending"]),
            "selectionInputs": "only the declared validation scores and validation mean crossing; no test PnL",
            "inferenceScope": "research screen on reused validation data; not a true-edge or uniform-error guarantee",
            "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, default=Path("data/training/runs/stochastic-crra-horizon-search-v1"))
    args = parser.parse_args()
    result = summarize(args.runs_root)
    output = args.runs_root / "search-results.json"
    write_json(output, result)
    concise = {"output": str(output), "firstSelectedValidationCrossingSeconds": result["firstSelectedValidationCrossingSeconds"],
               "rows": [{"horizonSeconds": row["horizonSeconds"], "selectedModel": row["selectedModel"],
                         "selectedMeanLossWeight": row["selectedMeanLossWeight"],
                         "allOverlappingValidation": row["allOverlappingValidation"],
                         "strideHValidation": row["strideHValidation"]} for row in result["rows"]]}
    print(json.dumps(concise, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
