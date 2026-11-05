"""Independently inspect the smooth UP Gamma-scale bound in saved raw forecasts.

Inverts the recorded tanh relation using log1p identities. No model inference,
calibration, test-based choice, or modification of the forecast arrays occurs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from log_hurdle_sign_size_forecast import MAXIMUM_UP_MEAN_RATIO, MINIMUM_MEAN
from train_neural_sign_size_forecast import now, sha, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", type=Path, default=Path("data/backtests/oracle-log-hurdle/evaluation.json"))
    parser.add_argument("--output", type=Path, default=Path("data/backtests/oracle-log-hurdle/bound-diagnostics.json"))
    args = parser.parse_args()
    evaluation = json.loads(args.evaluation.read_text())
    rows = []
    for item in evaluation["results"]:
        for stage in ("pretrainedRaw", "finetunedRaw"):
            record = item["models"][stage]
            saved = record["savedStream"]
            file = Path(saved["parametersFile"])
            if sha(file) != saved["parametersSha256"]:
                raise ValueError("raw forecast stream changed")
            with np.load(file, allow_pickle=False) as data:
                packed = data["parameters"][data["accuracyIndices"]]
            mean, cap = packed[:, 2], packed[:, 4]*MAXIMUM_UP_MEAN_RATIO
            if not np.isfinite(packed).all() or np.any(mean > cap) or np.any(mean < MINIMUM_MEAN):
                raise ValueError("recorded raw mean violates the model's product bound")
            floor, saturated = mean <= MINIMUM_MEAN, mean >= cap
            interior = ~(floor | saturated)
            z = mean[interior]/cap[interior]
            inverse = .5*(np.log1p(z)-np.log1p(-z))
            uncapped = cap[interior]*inverse
            absolute_reduction = uncapped-mean[interior]
            relative_reduction = absolute_reduction/uncapped
            percentiles = [50, 95, 99, 100]
            ratio_quantiles = np.percentile(mean/cap, percentiles).tolist()
            relative_quantiles = np.percentile(relative_reduction, percentiles).tolist() if len(z) else None
            reference = record["upSmoothBound"]
            if reference["examples"] != len(mean) or reference["meanFloorCount"] != int(floor.sum()) \
                    or reference["saturationCount"] != int(saturated.sum()):
                raise ValueError("reported floor/saturation populations changed")
            np.testing.assert_allclose(reference["upMeanOverCapQuantilesP50P95P99Max"], ratio_quantiles, rtol=1e-13, atol=1e-15)
            if relative_quantiles is None:
                if reference["relativeMeanReductionQuantilesP50P95P99Max"] is not None:
                    raise ValueError("unidentifiable bound effect was reported as finite")
            else:
                np.testing.assert_allclose(reference["relativeMeanReductionQuantilesP50P95P99Max"], relative_quantiles, rtol=1e-8, atol=5e-16)
            rows.append({"split": item["split"], "horizonSeconds": item["horizonSeconds"], "stage": stage,
                "examples": len(mean), "rawForecastFile": str(file), "rawForecastSha256": sha(file),
                "meanFloorCount": int(floor.sum()), "saturationCount": int(saturated.sum()),
                "upMeanOverCapQuantilesP50P95P99Max": ratio_quantiles,
                "relativeMeanReductionQuantilesP50P95P99Max": relative_quantiles,
                "absoluteLogMeanReductionBpsQuantilesP50P95P99Max": np.percentile(absolute_reduction*1e4, percentiles).tolist() if len(z) else None,
                "reportedEvaluatorDiagnosticsVerified": True})
    result = {"contract": "raw-log-hurdle-up-bound-audit-v1", "createdAt": now(),
        "evaluationFile": str(args.evaluation), "evaluationSha256": sha(args.evaluation),
        "formula": "cap=alphaUP*.49; rawMean=cap*(log1p(mean/cap)-log1p(-mean/cap))/2 away from floor and saturation",
        "scope": "raw neural model means only; conditional UP log-magnitude mean distortion, not a realized-return cap and not a signed portfolio-return difference",
        "changesTrainingOrSelection": False, "neuralInferencePerformed": False, "results": rows}
    write_json(args.output, result)
    print(json.dumps({"event": "logBoundAuditComplete", "output": str(args.output), "sha256": sha(args.output), "cohorts": len(rows),
        "maximumRelativeMeanReduction": max(row["relativeMeanReductionQuantilesP50P95P99Max"][-1] for row in rows
            if row["relativeMeanReductionQuantilesP50P95P99Max"] is not None),
        "floorCount": sum(row["meanFloorCount"] for row in rows), "saturationCount": sum(row["saturationCount"] for row in rows)}), flush=True)


if __name__ == "__main__":
    main()
