"""Read-only independent checks of shared-horizon selections and forecast artifacts.

No training, calibration fitting, or hyperparameter selection is performed.
Stored causal state is checked against exact endpoint maturity; likelihood and
mean accuracy are recomputed with NumPy/SciPy independently of the trainer.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy.special import expit
from scipy.stats import beta, gamma
import torch

from audit_online_hurdle_calibration import audit_trace
from neural_sign_size_forecast import StructuredSpecialist
from online_hurdle_calibration import CalibrationConfig
from stochastic_price_forecast import load_timeline_split
from train_hurdle_sign_size_forecast import load_combined
from train_neural_sign_size_forecast import HORIZONS, now, sha, write_json

CONTRACT = "shared-horizon-hurdle-artifact-audit-v1"
VARIANTS = ("separate-raw", "separate-online", "shared-raw", "shared-online")


def independent_mean(parameters: np.ndarray) -> np.ndarray:
    p0, pu = expit(parameters[..., 0]), expit(parameters[..., 1])
    return (1-p0)*(pu*parameters[..., 2]-(1-pu)*parameters[..., 3])


def independent_metrics(parameters: np.ndarray, y: np.ndarray) -> dict:
    zero, sign, up, down, au, ad = parameters.T
    active, positive, negative = y != 0, y > 0, y < 0
    p0, pu = expit(zero), expit(sign)
    mean = independent_mean(parameters)
    nll = np.logaddexp(0, zero)-zero*(~active)
    nll[active] += np.logaddexp(0, sign[active])-sign[active]*positive[active]
    nll[positive] -= gamma.logpdf(y[positive], au[positive], scale=up[positive]/au[positive])
    b = ad[negative]*(1-down[negative])/down[negative]
    nll[negative] -= beta.logpdf(-y[negative], ad[negative], b)
    mse = np.square(mean-y).mean()
    return {"fullHurdleReturnNegativeLogLikelihood": float(nll.mean()),
        "allTickSignedMeanMseBpsSquared": float(mse*1e8),
        "allTickSignedMeanMseSkillVsZero": float(1-mse/np.square(y).mean()),
        "zeroBrier": float(np.square(p0-(~active)).mean()),
        "predictedZeroProbability": float(p0.mean()),
        "realizedZeroFrequency": float((~active).mean())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, default=Path("data/training/runs/stochastic-crra-shared-hurdle-v1"))
    parser.add_argument("--output", type=Path, default=Path("data/backtests/oracle-shared-hurdle/diagnostics.json"))
    parser.add_argument("--training-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    started = time.monotonic()
    tracked: dict[str, str] = {}

    def checked(file: Path | str, expected: str | None = None) -> Path:
        file = Path(file)
        digest = sha(file)
        if expected is not None and digest != expected:
            raise ValueError(f"frozen artifact changed: {file}")
        tracked[str(file)] = digest
        return file

    def read(file: Path | str, expected: str | None = None) -> dict:
        return json.loads(checked(file, expected).read_text(encoding="utf-8"))

    def arrays(file: Path | str, expected: str) -> dict[str, np.ndarray]:
        with np.load(checked(file, expected), allow_pickle=False) as saved:
            return {key: saved[key] for key in saved.files}

    selection_file = args.runs_root/"selection.json"
    selected = read(selection_file)
    plan = read(args.runs_root/"plan.json", selected["planSha256"])
    if selected["plan"] != plan or selected["testLoaded"] or selected["testUsedForSelection"] \
            or not selected["allHeadSelectionsFrozen"]:
        raise ValueError("shared selection was not frozen before test")
    if tuple(row["horizonSeconds"] for row in selected["horizons"]) != HORIZONS:
        raise ValueError("shared horizon order changed")
    source_file = checked(selected["sourceSelectionFile"], selected["sourceSelectionSha256"])
    source = read(source_file)
    checked(Path(plan["dataset"])/"manifest.json", plan["datasetManifestSha256"])
    initial = read(selected["initializerSelectionFile"], selected["initializerSelectionSha256"])
    if initial["testLoaded"] or initial["testUsedForSelection"] or not initial["allInitializerChoicesFrozen"]:
        raise ValueError("initializer choice used test")
    initializer_rows = []
    for task in ("zero", "sign", "size"):
        choice = initial["experts"][task]
        registered = list({record["sha256"]: record for record in plan["initialization"]["candidateRecords"][task]}.values())
        if [row["record"] for row in choice["candidates"]] != registered:
            raise ValueError("initializer candidate set changed")
        for row in choice["candidates"]:
            values = np.asarray(row["validationScoresByHorizon"])
            if values.shape != (4,) or not np.isfinite(values).all():
                raise ValueError("initializer scores do not cover all horizons")
            np.testing.assert_allclose(row["meanValidationScore"], values.mean(), rtol=1e-14, atol=1e-14)
            record = row["record"]
            checked(record["checkpoint"], record["sha256"])
            old_log = Path(record["checkpoint"]).parent/"training.jsonl"
            logs = [json.loads(line) for line in checked(old_log).read_text().splitlines() if line.strip()]
            expected = next(event for event in logs if event.get("epoch") == record["epoch"]
                and ((task == "zero" and event.get("stage") == "zero" and event.get("event") == "epoch")
                     or (task != "zero" and event.get("task") == task and event.get("stage") == "epoch")))
            np.testing.assert_allclose(values, expected["validationScores"], rtol=2e-6, atol=2e-7)
        winner = min(choice["candidates"], key=lambda row: row["meanValidationScore"])
        if choice["selected"] != winner["record"]:
            raise ValueError("retained initializer is not the validation aggregate minimum")
        initializer_rows.append({"task": task, "selectedEpoch": winner["record"]["epoch"],
                                 "meanValidationScore": winner["meanValidationScore"], "candidates": len(registered)})
    logs = [json.loads(line) for line in checked(args.runs_root/"training.jsonl").read_text().splitlines() if line.strip()]
    epochs = [row for row in logs if row["event"] == "epoch" and row["stage"] == "finetuned"]
    for row in epochs:
        np.testing.assert_allclose(row["meanValidationScore"], np.mean(row["validationScores"]), rtol=1e-14, atol=1e-14)
    best = min(epochs, key=lambda row: np.mean(row["validationScores"]))
    fine = selected["finetuned"]
    if fine["epoch"] != best["epoch"] or fine["validationScoresByHorizon"] != best["validationScores"]:
        raise ValueError("fine selection is not the single global EMA minimum")
    checkpoint_states = {}
    for stage in ("pretrained", "finetuned"):
        record = selected[stage]
        saved = torch.load(checked(record["checkpoint"], record["sha256"]), map_location="cpu", weights_only=True)
        if saved["horizonsSeconds"] != list(HORIZONS) or saved["planSha256"] != selected["planSha256"] \
                or saved["epoch"] != record["epoch"] or saved["task"] != "combined":
            raise ValueError("shared checkpoint horizon or selection provenance changed")
        for expert, width in (("zero", 4), ("sign", 4), ("size", 8)):
            if saved["modelState"][expert+".readout.weight"].shape[0] != width:
                raise ValueError("a shared expert lost horizon readout rows")
        if saved["modelState"]["raw_shapes"].shape != (4, 2):
            raise ValueError("shape parameters lost a horizon or direction")
        checkpoint_states[stage] = saved["modelState"]
    for task in ("zero", "sign", "size"):
        record = initial["experts"][task]["selected"]
        original = torch.load(checked(record["checkpoint"], record["sha256"]), map_location="cpu", weights_only=True)
        for name, value in original["modelState"].items():
            torch.testing.assert_close(checkpoint_states["pretrained"][task+"."+name], value, rtol=0, atol=0)
    model = load_combined(fine, torch.device("cpu"))
    if sum(isinstance(module, StructuredSpecialist) for module in model.modules()) != 3 \
            or sum(p.numel() for p in model.parameters()) != selected["parameters"]:
        raise ValueError("retained architecture is not exactly three shared experts")
    changed = {task: sum(not torch.equal(checkpoint_states["pretrained"][name], value)
                        for name, value in checkpoint_states["finetuned"].items() if name.startswith(task+"."))
               for task in ("zero", "sign", "size")}
    del checkpoint_states
    training_checks = {"initializerSelections": initializer_rows, "selectedEpoch": fine["epoch"],
        "completedEpochs": fine["completedEpochs"], "globalCheckpointSelectionVerified": True,
        "intactPretrainedExpertTransferVerified": True, "sharedExpertBackbones": 3,
        "sharedParameters": selected["parameters"], "changedStateTensorsByExpert": changed}
    if args.training_only:
        output = {"contract": CONTRACT, "createdAt": now(), "trainingOnly": True,
            "selectionSha256": sha(selection_file), **training_checks, "trackedArtifactHashes": tracked,
            "seconds": time.monotonic()-started}
        write_json(args.output, output)
        print(json.dumps({"event": "sharedTrainingAuditComplete", "output": str(args.output),
                          "sha256": sha(args.output), "artifacts": len(tracked)}), flush=True)
        return
    evaluation = read(args.runs_root/"evaluation-plan.json")
    result = read(args.runs_root/"result.json")
    if result["selectionSha256"] != sha(selection_file) or result["testUsedForSelection"] or result["calibrationReselection"]:
        raise ValueError("result used different selection or reselected calibration")
    checked(result["evaluationPlanFile"], result["evaluationPlanSha256"])
    calibration_file = checked(evaluation["calibrationSelectionFile"], evaluation["calibrationSelectionSha256"])
    calibration = read(calibration_file)
    if evaluation["selectionSha256"] != sha(selection_file) or evaluation["testUsedForSelection"]:
        raise ValueError("evaluation was not registered on the frozen shared checkpoint")
    for row in calibration["horizons"]:
        if evaluation["frozenCalibration"][str(row["horizonSeconds"])] != row["modes"][row["validationPreferredMode"]]:
            raise ValueError("calibration settings differ from the previous validation choice")
    results, neural_checks, total_queries = [], [], 0
    for split in ("validation", "test"):
        reported = read(args.runs_root/f"{split}.json")
        shared_cache = reported["sharedCache"]
        matrix = arrays(shared_cache["finetuned"]["cacheFile"], shared_cache["finetuned"]["cacheSha256"])
        if matrix["raw"].shape != (len(matrix["origins"]), 4, 6):
            raise ValueError("shared cache was not one all-horizon matrix")
        np.testing.assert_array_equal(matrix["horizons"], HORIZONS)
        timeline = load_timeline_split(Path(plan["dataset"]), split)
        samples = np.linspace(0, len(matrix["origins"])-1, 16, dtype=np.int64)
        features = np.asarray(timeline.timeline[matrix["origins"][samples, None]+np.array([-1, 0])], dtype=np.float32)
        model.to("cpu")
        with torch.inference_mode():
            fresh = model(torch.from_numpy(features)).numpy()
        expected = matrix["raw"][samples]
        if not np.isfinite(fresh).all():
            raise ValueError("independent CPU inference became nonfinite")
        np.testing.assert_allclose(fresh[..., 4:], expected[..., 4:], rtol=1e-14, atol=1e-14)
        # Torch's registered float32 matmul precision="high" can select a
        # different reduced-precision kernel at a different batch width. Audit
        # exact checkpoint/cache identity using the original GPU batch contexts;
        # separately report CPU-vs-GPU differences without hiding them in a tol.
        if not torch.cuda.is_available():
            raise RuntimeError("original GPU execution is required for exact numerical cache audit")
        model.to("cuda")
        original_batches = set()
        common_count = len(matrix["commonOrigins"])
        for index in samples:
            lower, upper = (0, common_count) if index < common_count else (common_count, len(matrix["origins"]))
            begin = lower+(int(index)-lower)//evaluation["batchSize"]*evaluation["batchSize"]
            original_batches.add((begin, min(begin+evaluation["batchSize"], upper)))
        audited_rows = 0
        for begin, end in sorted(original_batches):
            batch_origins = matrix["origins"][begin:end]
            batch_features = np.asarray(timeline.timeline[batch_origins[:, None]+np.array([-1, 0])], dtype=np.float32)
            with torch.inference_mode():
                reproduced = model(torch.from_numpy(batch_features).to("cuda")).cpu().numpy()
            np.testing.assert_array_equal(reproduced, matrix["raw"][begin:end])
            audited_rows += end-begin
        neural_checks.append({"split": split, "samples": len(samples),
            "maximumCpuVsGpuLogitError": float(np.max(np.abs(fresh[..., :2]-expected[..., :2]))),
            "maximumCpuVsGpuConditionalSizeErrorSimple": float(np.max(np.abs(fresh[..., 2:4]-expected[..., 2:4]))),
            "maximumCpuVsGpuConditionalSizeErrorBps": float(np.max(np.abs(fresh[..., 2:4]-expected[..., 2:4]))*1e4),
            "maximumCpuVsGpuSignedMeanErrorBps": float(np.max(np.abs(independent_mean(fresh)-independent_mean(expected)))*1e4),
            "originalGpuBatchContexts": len(original_batches), "originalGpuRowsChecked": audited_rows,
            "originalGpuCacheBitExact": True,
            "numericalScope": "precision=high allows batch/device-dependent matmul rounding; exact identity checked at the registered GPU batch shapes"})
        old_report = read(calibration_file.parent/f"{split}.json")
        for j, h in enumerate(HORIZONS):
            source_cache = reported["baselineCaches"][j]
            old_raw = arrays(source_cache["cacheFile"], source_cache["cacheSha256"])
            market = arrays(source_cache["marketFile"], source_cache["marketSha256"])
            origins, timestamps, accuracy = old_raw["origins"], old_raw["timestamps"], old_raw["accuracyIndices"]
            np.testing.assert_array_equal(matrix["origins"][:len(origins)], origins)
            np.testing.assert_array_equal(origins[accuracy], market["commonOrigins"])
            np.testing.assert_array_equal(timestamps, int(market["axisStartMs"])+origins*1000)
            offsets = origins[accuracy]-int(market["firstPriceOrigin"])
            y = market["prices"][offsets+h]/market["prices"][offsets]-1
            shared_raw = matrix["raw"][:len(origins), j]
            chosen = evaluation["frozenCalibration"][str(h)]["config"]
            for variant in VARIANTS:
                score = reported["horizons"][j]["models"][variant]
                config = score["calibrationConfig"]
                if variant.endswith("online") and config != chosen:
                    raise ValueError("an architecture used different online settings")
                if variant.endswith("raw") and config["mode"] != "none":
                    raise ValueError("raw forecast unexpectedly calibrated")
                CalibrationConfig(**config)
                saved = score["savedStream"]
                forecast = arrays(saved["parametersFile"], saved["parametersSha256"])
                trace = arrays(saved["stateTraceFile"], saved["stateTraceSha256"])
                for array in (forecast, trace):
                    np.testing.assert_array_equal(array["origins"], origins)
                    np.testing.assert_array_equal(array["timestamps"], timestamps)
                np.testing.assert_array_equal(forecast["accuracyIndices"], accuracy)
                raw = old_raw["raw"] if variant.startswith("separate") else shared_raw
                checks = audit_trace(raw, forecast["parameters"], timestamps, trace, h, config["mode"])
                independent = independent_metrics(forecast["parameters"][accuracy], y)
                for name, value in independent.items():
                    np.testing.assert_allclose(score[name], value, rtol=1e-8, atol=1e-7 if "Likelihood" in name else 1e-10)
                if variant.startswith("separate"):
                    old = old_report["horizons"][j]["models"][config["mode"]]
                    if old["savedStream"] != saved:
                        raise ValueError("separate baseline was changed or regenerated")
                    for name in independent:
                        if score[name] != old[name]:
                            raise ValueError("separate baseline metric parity failed")
                results.append({"split": split, "horizonSeconds": h, "variant": variant,
                                "causality": checks, "independentMetrics": independent})
                total_queries += checks["queries"]
    output = {"contract": CONTRACT, "createdAt": now(), "selectionSha256": sha(selection_file),
        **training_checks,
        "frozenCalibrationReusedWithoutReselection": True, "baselineArtifactAndMetricParity": True,
        "independentInference": neural_checks, "canonicalQueriesAudited": total_queries,
        "results": results, "trackedArtifactHashes": tracked, "seconds": time.monotonic()-started,
        "scope": "Artifact/selection/causality and numerical audit; no independence claim for overlapping outcomes or fresh-holdout claim."}
    write_json(args.output, output)
    print(json.dumps({"event": "sharedAuditComplete", "output": str(args.output), "sha256": sha(args.output),
        "canonicalQueries": total_queries, "artifacts": len(tracked), "seconds": output["seconds"]}), flush=True)


if __name__ == "__main__":
    main()
