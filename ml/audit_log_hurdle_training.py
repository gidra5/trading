"""Read-only provenance, population and global-selection audit of log training.

Reconstructs train/validation labels and statistics only. Does not load test,
run neural inference, modify weights, fit calibration, or choose a checkpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from log_hurdle_sign_size_forecast import CONTRACT as LAW_CONTRACT, MAXIMUM_UP_MEAN_RATIO
from neural_sign_size_forecast import StructuredSpecialist
from stochastic_price_forecast import load_timeline_split
from train_log_hurdle_sign_size_forecast import CONTRACT, load_combined
from train_neural_sign_size_forecast import HORIZONS, examples, now, select_origins, sha, write_json

AUDIT_CONTRACT = "shared-log-hurdle-training-artifact-audit-v1"


def array_sha(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def independent_constants(targets: np.ndarray) -> list[dict]:
    result = []
    for j, horizon in enumerate(HORIZONS):
        y = targets[:, j]
        positive, negative, active = y > 0, y < 0, y != 0
        result.append({"horizonSeconds": horizon,
            "pZero": float(((~active).sum()+.5)/(len(y)+1)),
            "pUpGivenActive": float((positive.sum()+.5)/(active.sum()+1)),
            "upMeanLog": float(y[positive].mean()), "downMeanLog": float(-y[negative].mean()),
            "pooledActiveMeanLog": float(np.abs(y[active]).mean()),
            "activeTrainingExamples": int(active.sum()), "upTrainingExamples": int(positive.sum()),
            "downTrainingExamples": int(negative.sum())})
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, default=Path("data/training/runs/stochastic-crra-log-hurdle-v1"))
    parser.add_argument("--output", type=Path, default=Path("data/backtests/oracle-log-hurdle/training-diagnostics.json"))
    args = parser.parse_args()
    torch.set_num_threads(4)
    started = time.monotonic()
    tracked: dict[str, str] = {}

    def checked(file: str | Path, expected: str | None = None) -> Path:
        file = Path(file)
        digest = sha(file)
        if expected is not None and digest != expected:
            raise ValueError(f"frozen artifact hash mismatch: {file}")
        tracked[str(file)] = digest
        return file

    def read(file: str | Path, expected: str | None = None) -> dict:
        return json.loads(checked(file, expected).read_text(encoding="utf-8"))

    selection_file = args.runs_root/"selection.json"
    selected = read(selection_file)
    plan = read(args.runs_root/"plan.json", selected["planSha256"])
    metadata = read(args.runs_root/"metadata.json")
    training_result = read(args.runs_root/"training-result.json")
    pretraining = read(args.runs_root/"pretraining-selection.json")
    runtime = read(args.runs_root/"runtime-provenance.json")
    for file, digest in runtime["sourceHashes"].items():
        checked(file, digest)
    for item in (selected, metadata, training_result, pretraining):
        if item["testLoaded"] or item["testUsedForSelection"]:
            raise ValueError("training or checkpoint selection reports test use")
    if selected["contract"] != CONTRACT or selected["plan"] != plan or not selected["allHeadSelectionsFrozen"] \
            or metadata["planSha256"] != selected["planSha256"] \
            or training_result["selectionSha256"] != sha(selection_file):
        raise ValueError("log training selection provenance changed")
    if pretraining["experts"] != selected["pretraining"] or training_result["pretraining"] != selected["pretraining"] \
            or training_result["finetuned"] != selected["finetuned"]:
        raise ValueError("global checkpoint records disagree across artifacts")
    dataset = Path(plan["dataset"])
    checked(dataset/"manifest.json", plan["datasetManifestSha256"])
    source = read(plan["sourceSelectionFile"], plan["sourceSelectionSha256"])
    online = read(plan["onlineSelectionFile"], plan["onlineSelectionSha256"])
    if online["sourceSelectionSha256"] != plan["sourceSelectionSha256"] or online["testUsedForSelection"]:
        raise ValueError("retained calibration does not belong to the retained source model")
    retained = []
    for entry in source["horizons"]:
        record = entry["finetuned"]
        checked(record["checkpoint"], record["sha256"])
        retained.append({"horizonSeconds": entry["horizonSeconds"], "checkpointSha256": record["sha256"]})
    populations = {}
    means = stds = scales = None
    for split in ("train", "validation"):
        timeline = load_timeline_split(dataset, split)
        origins = select_origins(timeline, plan["strideSeconds"], plan["trainingLimit"] if split == "train" else None)
        simple = examples(timeline, origins)
        log = np.log1p(simple.targets)
        np.testing.assert_array_equal(np.sign(log), np.sign(simple.targets))
        if log.dtype != np.float64 or not np.isfinite(log).all():
            raise ValueError("log targets lost float64 precision or became nonfinite")
        expected_count = metadata["trainingExamples" if split == "train" else "validationExamples"]
        if len(origins) != expected_count or np.any(origins+max(HORIZONS) >= len(timeline.timeline)):
            raise ValueError("split population changed or a horizon crosses its boundary")
        first = int(timeline.axis_start_ms+origins[0]*1000)
        final = int(timeline.axis_start_ms+(origins[-1]+max(HORIZONS))*1000)
        prefix = "training" if split == "train" else "validation"
        if first != metadata[prefix+"FirstTimestamp"] or final != metadata[prefix+"LastEndpointTimestamp"]:
            raise ValueError("recorded population time bounds differ from the actual origins")
        populations[split] = {"examples": len(origins), "firstTimestamp": first, "lastEndpointTimestamp": final,
            "originSha256": array_sha(origins), "logTargetsSha256": array_sha(log),
            "simpleTargetsSha256": array_sha(simple.targets), "featuresSha256": array_sha(simple.features),
            "horizons": [{"horizonSeconds": horizon, "logTargetsSha256": array_sha(log[:, j]),
                "activeExamples": int((log[:, j] != 0).sum()), "zeroExamples": int((log[:, j] == 0).sum())}
                for j, horizon in enumerate(HORIZONS)]}
        if split == "train":
            constants = independent_constants(log)
            if constants != metadata["constants"] or constants != selected["constants"]:
                raise ValueError("model scales/probability priors do not come from train log targets")
            means = simple.features.reshape(-1, 59).mean(0, dtype=np.float64).astype(np.float32)
            stds = np.maximum(simple.features.reshape(-1, 59).std(0, dtype=np.float64), 1e-10).astype(np.float32)
            np.testing.assert_array_equal(means, np.asarray(metadata["inputMean"], dtype=np.float32))
            np.testing.assert_array_equal(stds, np.asarray(metadata["inputStd"], dtype=np.float32))
            scales = np.array([[row["upMeanLog"], row["downMeanLog"]] for row in constants])
        del timeline, simple, log
    if populations["train"]["lastEndpointTimestamp"] >= populations["validation"]["firstTimestamp"]:
        raise ValueError("training labels overlap the validation split")
    log_rows = [json.loads(line) for line in checked(args.runs_root/"training.jsonl").read_text().splitlines() if line.strip()]
    selected_states, minima = {}, []
    for task in ("zero", "sign", "size", "combined"):
        record = selected["finetuned"] if task == "combined" else selected["pretraining"][task]
        epochs = [row for row in log_rows if row["task"] == task and row["event"] == "epoch"]
        complete = next(row for row in log_rows if row["task"] == task and row["event"] == "complete")
        if [row["epoch"] for row in epochs] != list(range(record["completedEpochs"]+1)) \
                or complete["completedEpochs"] != record["completedEpochs"]:
            raise ValueError("epoch history is incomplete or differs from the selected record")
        best = None
        for row in epochs:
            values = np.asarray(row["validationScores"])
            if values.shape != (4,) or not np.isfinite(values).all():
                raise ValueError("global selection did not score every horizon")
            np.testing.assert_allclose(row["meanValidationScore"], values.mean(), rtol=1e-14, atol=1e-14)
            if best is None or row["meanValidationScore"] < best["meanValidationScore"]:
                best = row
            if row["bestEpoch"] != best["epoch"]:
                raise ValueError("running best epoch is not the strict global mean minimum")
        if record["epoch"] != best["epoch"] or record["validationScoresByHorizon"] != best["validationScores"] \
                or record["validationScore"] != best["meanValidationScore"]:
            raise ValueError("retained checkpoint is not the global validation minimum")
        limits = plan["fineTuning" if task == "combined" else "pretraining"]
        completed = record["completedEpochs"]
        if completed > limits["maximumEpochs"] or (completed < limits["maximumEpochs"] and
                (completed < limits["minimumEpochs"] or completed-record["epoch"] < limits["patience"])):
            raise ValueError("training stopped outside the registered bounds/patience")
        saved = torch.load(checked(record["checkpoint"], record["sha256"]), map_location="cpu", weights_only=True)
        if saved["contract"] != CONTRACT or saved["targetCoordinate"] != "log-return" or saved["task"] != task \
                or saved["horizonsSeconds"] != list(HORIZONS) or saved["planSha256"] != selected["planSha256"] \
                or saved["epoch"] != record["epoch"]:
            raise ValueError("selected checkpoint target/task/horizon/plan changed")
        if saved["validationScore"] != record["validationScore"] \
                or saved["validationScoresByHorizon"] != record["validationScoresByHorizon"]:
            raise ValueError("checkpoint validation metadata differs from its global selection")
        np.testing.assert_array_equal(saved["inputMean"].numpy(), means)
        np.testing.assert_array_equal(saved["inputStd"].numpy(), stds)
        np.testing.assert_array_equal(saved["sizeScales"].numpy(), scales)
        if task != "combined" and saved["initialization"] != {"freshSeed": 1337}:
            raise ValueError("pretraining did not record its registered fresh initialization")
        selected_states[task] = saved["modelState"]
        minima.append({"task": task, "selectedEpoch": record["epoch"], "completedEpochs": completed,
            "validationScoresByHorizon": record["validationScoresByHorizon"], "meanValidationScore": record["validationScore"]})
    baseline = selected["pretrained"]
    pretrained = torch.load(checked(baseline["checkpoint"], baseline["sha256"]), map_location="cpu", weights_only=True)
    if pretrained["contract"] != CONTRACT or pretrained["targetCoordinate"] != "log-return" \
            or pretrained["horizonsSeconds"] != list(HORIZONS) or pretrained["planSha256"] != selected["planSha256"]:
        raise ValueError("combined pretrained checkpoint provenance changed")
    for task in ("zero", "sign", "size"):
        for name, value in selected_states[task].items():
            torch.testing.assert_close(pretrained["modelState"][task+"."+name], value, rtol=0, atol=0)
    initial = pretrained["initialization"]
    if initial["experts"] != selected["pretraining"] or initial["shapeInitialization"]["testUsed"]:
        raise ValueError("combined initializer used different experts or test shapes")
    shape_initialization = initial["shapeInitialization"]
    expected_counts = np.array([[row["upTrainingExamples"], row["downTrainingExamples"]] for row in selected["constants"]])
    np.testing.assert_array_equal(shape_initialization["examplesByDirection"], expected_counts)
    residual_second = np.asarray(shape_initialization["relativeResidualSecondMoments"])
    expected_shapes = np.clip(np.divide(1., residual_second, out=np.full_like(residual_second, 50.), where=residual_second > 0), .01, 50.)
    np.testing.assert_array_equal(shape_initialization["shapes"], expected_shapes)
    model = load_combined(selected["finetuned"], torch.device("cpu"))
    if model.contract != LAW_CONTRACT or sum(isinstance(module, StructuredSpecialist) for module in model.modules()) != 3 \
            or sum(p.numel() for p in model.parameters()) != selected["parameters"]:
        raise ValueError("final model is not the declared three-expert shared log law")
    if [expert.readout.out_features for expert in (model.zero, model.sign, model.size)] != [4, 4, 8] \
            or model.raw_shapes.shape != (4, 2):
        raise ValueError("a shared expert or shape matrix lost horizon outputs")
    output = {"contract": AUDIT_CONTRACT, "createdAt": now(), "selectionSha256": sha(selection_file),
        "trainingPlanSha256": selected["planSha256"], "testLoadedByAudit": False, "neuralInferencePerformed": False,
        "allGlobalMinimaVerified": True, "trainingOnlyStatisticsVerified": True,
        "pretrainedStateTransferExact": True, "retainedSourceCheckpointsUnchanged": retained,
        "sharedExpertBackbones": 3, "sharedParameters": selected["parameters"], "selectedTasks": minima,
        "runtimeSourceHashesVerified": True, "runtimeProvenance": runtime,
        "populations": populations, "upGammaScaleCeiling": MAXIMUM_UP_MEAN_RATIO,
        "finiteFirstAndSecondPriceMoments": True,
        "momentScope": "finite moments follow from the declared signed-Gamma parameter map; this audit does not infer forecasts or measure bound distortion",
        "trackedArtifactHashes": tracked, "seconds": time.monotonic()-started}
    write_json(args.output, output)
    print(json.dumps({"event": "logTrainingAuditComplete", "output": str(args.output), "sha256": sha(args.output),
        "artifacts": len(tracked), "selectedTasks": minima, "seconds": output["seconds"]}), flush=True)


if __name__ == "__main__":
    main()
