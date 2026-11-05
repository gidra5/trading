"""Read-only post-run audit of selected online-calibration caches and traces.

No neural inference, calibration fitting, hyperparameter selection, or replay
is performed. Event maturity and transformed forecasts are checked directly
from immutable saved arrays, and accuracy metrics are recomputed on the
original common-origin population.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy.special import expit
import torch

from export_hurdle_sign_size_forecast import frozen_selection
from online_hurdle_calibration import COEFFICIENT_BOUNDS, MAXIMUM_DOWN_MEAN, MINIMUM_MEAN, MODES
from train_hurdle_sign_size_forecast import metrics
from train_neural_sign_size_forecast import HORIZONS, now, sha, write_json

CONTRACT = "online-hurdle-causal-artifact-audit-v1"


def reconstructed_parameters(raw: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Independent vectorized odds formula; no calibrator methods are called."""
    output = raw.copy()
    output[:, :2] += theta[:, :2]
    changed_up, changed_down = theta[:, 2] != 0, theta[:, 3] != 0
    output[changed_up, 2] = np.maximum(MINIMUM_MEAN, raw[changed_up, 2]*np.exp(theta[changed_up, 2]))
    m = raw[changed_down, 3]
    # Algebraically equal to sigmoid(logit(m)+theta), with different numerical
    # operations from the implementation being audited.
    output[changed_down, 3] = np.clip(m/(m+(1-m)*np.exp(-theta[changed_down, 3])),
                                    MINIMUM_MEAN, MAXIMUM_DOWN_MEAN)
    return output


def audit_trace(raw: np.ndarray, parameters: np.ndarray, timestamps: np.ndarray,
                trace: dict[str, np.ndarray], horizon: int, mode: str) -> dict:
    n = len(timestamps)
    if raw.shape != (n, 6) or parameters.shape != raw.shape or trace["coefficients"].shape != (n, 4):
        raise ValueError("inconsistent canonical forecast/trace shape")
    if not np.all(np.isfinite(raw)) or not np.all(np.isfinite(parameters)):
        raise ValueError("nonfinite cached forecasts")
    np.testing.assert_array_equal(np.diff(timestamps), np.full(n-1, 4000))
    maturity = timestamps+horizon*1000
    counts = np.searchsorted(maturity, timestamps, side="right")
    latest = np.where(counts > 0, maturity[np.maximum(counts-1, 0)], -1)
    np.testing.assert_array_equal(trace["maturedObservations"], counts)
    np.testing.assert_array_equal(trace["latestMaturityUsed"], latest)
    if np.any(latest > timestamps):
        raise ValueError("calibration used an endpoint from the future")
    theta = trace["coefficients"]
    if not np.all(np.isfinite(theta)) or np.any(np.abs(theta) > COEFFICIENT_BOUNDS):
        raise ValueError("coefficient outside the declared bounds")
    np.testing.assert_array_equal(theta[counts == 0], np.zeros((int((counts == 0).sum()), 4)))
    np.testing.assert_array_equal(parameters[counts == 0], raw[counts == 0])
    inactive = range(4) if mode == "none" else (1, 2, 3) if mode == "zero" else (2, 3) if mode == "probabilities" else ()
    inactive = list(inactive)
    if inactive:
        np.testing.assert_array_equal(theta[:, inactive], np.zeros((n, len(inactive))))
        np.testing.assert_array_equal(trace["componentUpdates"][:, inactive], np.zeros((n, len(inactive)), dtype=np.int64))
    if mode == "none":
        np.testing.assert_array_equal(parameters, raw)
    np.testing.assert_array_equal(parameters[:, 4:], raw[:, 4:])
    if np.any(trace["discountedInformation"] < 0) or not np.isfinite(trace["discountedInformation"]).all():
        raise ValueError("invalid discounted Fisher information")
    updates = trace["componentUpdates"]
    if np.any(updates < 0) or np.any(updates > counts[:, None]) or np.any(np.diff(updates, axis=0) < 0):
        raise ValueError("component update counts exceed matured observations or move backward")
    expected = reconstructed_parameters(raw, theta)
    np.testing.assert_allclose(parameters, expected, rtol=3e-14, atol=3e-15)
    return {"queries": n, "coldStartIdentityQueries": int((counts == 0).sum()),
        "maximumLatestMaturityMinusQueryMs": int(np.max(latest[counts > 0]-timestamps[counts > 0])) if np.any(counts > 0) else None,
        "maximumIndependentTransformAbsoluteError": float(np.max(np.abs(parameters-expected))),
        "maximumAbsoluteCoefficients": np.max(np.abs(theta), axis=0).tolist(),
        "shapeTensorsUnchanged": True, "lastQueryMaturedObservations": int(counts[-1])}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, default=Path("data/training/runs/stochastic-crra-online-calibration-v1"))
    parser.add_argument("--output", type=Path, default=Path("data/backtests/oracle-online-calibration/diagnostics.json"))
    args = parser.parse_args()
    torch.set_num_threads(4)
    tracked: dict[str, str] = {}

    def checked(file: Path, expected: str | None = None) -> Path:
        digest = sha(file)
        if expected is not None and digest != expected:
            raise ValueError(f"frozen artifact hash mismatch: {file}")
        tracked[str(file)] = digest
        return file

    def read(file: Path, expected: str | None = None) -> dict:
        return json.loads(checked(file, expected).read_text(encoding="utf-8"))

    def arrays(file: Path, expected: str) -> dict[str, np.ndarray]:
        with np.load(checked(file, expected), allow_pickle=False) as saved:
            return {name: saved[name] for name in saved.files}

    selection_file = args.runs_root/"selection.json"
    selected = read(selection_file)
    if not selected["allCalibrationSelectionsFrozen"] or selected["testLoaded"] or selected["testUsedForSelection"]:
        raise ValueError("online settings were not frozen before test")
    plan = read(args.runs_root/"plan.json", selected["planSha256"])
    if plan != selected["plan"]:
        raise ValueError("selected embedded plan changed")
    source_file = checked(Path(selected["sourceSelectionFile"]), selected["sourceSelectionSha256"])
    source = frozen_selection(source_file)
    dataset = Path(plan["dataset"])
    checked(dataset/"manifest.json", plan["datasetManifestSha256"])
    source_result = read(source_file.parent/"result.json")
    if source_result["selectionSha256"] != selected["sourceSelectionSha256"]:
        raise ValueError("neural baseline results do not match the frozen checkpoints")
    result = read(args.runs_root/"result.json")
    if result["selectionSha256"] != tracked[str(selection_file)] or result["testUsedForSelection"]:
        raise ValueError("online results do not match frozen settings")
    candidates = read(Path(selected["validationCandidatesFile"]), selected["validationCandidatesSha256"])
    if tuple(r["horizonSeconds"] for r in selected["horizons"]) != HORIZONS:
        raise ValueError("unexpected horizon selection")
    for j, entry in enumerate(selected["horizons"]):
        h = entry["horizonSeconds"]
        rows = candidates["horizons"][j]["candidates"]
        if candidates["horizons"][j]["horizonSeconds"] != h:
            raise ValueError("candidate horizon order changed")
        if [{k:row[k] for k in ("candidateId", "config")} for row in rows] != plan["candidateGrid"][str(h)]:
            raise ValueError("candidate grid differs from the registered grid")
        for mode in MODES:
            winner = min((row for row in rows if row["config"]["mode"] == mode), key=lambda row: row["validationNll"])
            if entry["modes"][mode] != {k:winner[k] for k in ("candidateId", "config", "validationNll")}:
                raise ValueError("selected calibration is not the within-mode validation minimum")
        if entry["validationPreferredMode"] != min(MODES, key=lambda mode: entry["modes"][mode]["validationNll"]):
            raise ValueError("preferred mode was not chosen by validation NLL")
        if entry["checkpoint"] != source["horizons"][j]["finetuned"]:
            raise ValueError("online calibration changed its source neural checkpoint")
    audit = {"contract": CONTRACT, "createdAt": now(), "selectionFile": str(selection_file),
        "selectionSha256": tracked[str(selection_file)], "sourceSelectionSha256": selected["sourceSelectionSha256"],
        "algorithm": "independent trace maturity formula, independent vectorized odds/scale transform, metric recomputation; no fitting or neural inference",
        "testUsedForHyperparameterSelection": False, "causalTestLabelUpdatesAllowed": True,
        "overlapScope": "weight min(1,4/H) is a heuristic; adjacent outcomes and forecast errors remain dependent",
        "selectionChecks": "all registered mode/horizon winners are minimum validation NLL with registered order tie breaks",
        "results": []}
    started = time.monotonic()
    total_queries = 0
    for split in ("validation", "test"):
        split_record = read(args.runs_root/f"{split}.json")
        for j, entry in enumerate(selected["horizons"]):
            h = entry["horizonSeconds"]
            manifest = split_record["cacheManifests"][j]
            if manifest["horizonSeconds"] != h or manifest["sourceSelectionSha256"] != selected["sourceSelectionSha256"] \
                    or manifest["planSha256"] != selected["planSha256"]:
                raise ValueError("cached raw stream provenance changed")
            checked(Path(manifest["checkpoint"]), manifest["checkpointSha256"])
            cached = arrays(Path(manifest["cacheFile"]), manifest["cacheSha256"])
            market = arrays(Path(manifest["marketFile"]), manifest["marketSha256"])
            origins, timestamps, raw = cached["origins"], cached["timestamps"], cached["raw"]
            accuracy = cached["accuracyIndices"]
            np.testing.assert_array_equal(origins[accuracy], market["commonOrigins"])
            np.testing.assert_array_equal(timestamps, int(market["axisStartMs"])+origins*1000)
            offsets = origins[accuracy]-int(market["firstPriceOrigin"])
            y = market["prices"][offsets+h]/market["prices"][offsets]-1
            terminal = int(market["axisStartMs"])+(int(market["firstPriceOrigin"])+len(market["prices"])-1)*1000
            for mode in MODES:
                reported = split_record["horizons"][j]["models"][mode]
                if reported["calibration"]["config"] != entry["modes"][mode]["config"]:
                    raise ValueError("scored mode is not the frozen calibration configuration")
                saved = reported["savedStream"]
                predictions = arrays(Path(saved["parametersFile"]), saved["parametersSha256"])
                trace = arrays(Path(saved["stateTraceFile"]), saved["stateTraceSha256"])
                for other in (predictions, trace):
                    np.testing.assert_array_equal(other["timestamps"], timestamps)
                    np.testing.assert_array_equal(other["origins"], origins)
                np.testing.assert_array_equal(predictions["accuracyIndices"], accuracy)
                checks = audit_trace(raw, predictions["parameters"], timestamps, trace, h, mode)
                if saved["maturedUpdates"] != len(origins) or saved["canonicalIssues"] != len(origins) \
                        or saved["finalState"]["maturedObservations"] != len(origins) \
                        or saved["finalState"]["latestMaturityUsed"] != int(timestamps[-1]+h*1000) \
                        or saved["terminalTimestamp"] != terminal or timestamps[-1]+h*1000 > terminal:
                    raise ValueError("final flush did not process every within-split endpoint exactly once")
                packed = predictions["parameters"][accuracy]
                measured = metrics(packed, y, entry["constants"])
                fields = ("fullHurdleReturnNegativeLogLikelihood", "allTickSignedMeanMseBpsSquared",
                          "allTickSignedMeanMseSkillVsZero", "threeClassLogLoss", "zeroBrier")
                for field in fields:
                    np.testing.assert_allclose(measured[field], reported[field], rtol=1e-11, atol=1e-11)
                p0, pup = expit(packed[:, 0]), expit(packed[:, 1])
                independent_mean = (1-p0)*(pup*packed[:, 2]-(1-pup)*packed[:, 3])
                independent_mse = float(np.square(independent_mean-y).mean()*1e8)
                np.testing.assert_allclose(independent_mse, measured["allTickSignedMeanMseBpsSquared"], rtol=1e-11, atol=1e-11)
                baseline_error = None
                if mode == "none":
                    baseline = source_result[split][j]["models"]["jointlyFinetuned"]
                    for field in fields:
                        np.testing.assert_allclose(measured[field], baseline[field], rtol=1e-8, atol=1e-9)
                    baseline_error = abs(measured["allTickSignedMeanMseSkillVsZero"]-baseline["allTickSignedMeanMseSkillVsZero"])
                total_queries += len(origins)
                audit["results"].append({"split": split, "horizonSeconds": h, "mode": mode, **checks,
                    "accuracyOrigins": len(accuracy), "selectedConfig": entry["modes"][mode]["config"],
                    "recomputedMetrics": {field:measured[field] for field in fields},
                    "noneVsFrozenNeuralMeanSkillAbsoluteDifference": baseline_error})
        print(json.dumps({"event": "splitAudited", "split": split, "queriesSoFar": total_queries}), flush=True)
    for file, expected in tracked.items():
        if sha(Path(file)) != expected:
            raise ValueError(f"artifact changed while being audited: {file}")
    frozen_selection(source_file)
    audit["artifactHashes"] = tracked
    audit["totalCanonicalQueriesChecked"] = total_queries
    audit["seconds"] = time.monotonic()-started
    write_json(args.output, audit)
    print(json.dumps({"event": "auditComplete", "output": str(args.output), "queries": total_queries,
                      "seconds": audit["seconds"]}), flush=True)


if __name__ == "__main__":
    main()
