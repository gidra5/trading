"""Post-selection ternary classification and active-move diagnostics.

This audit does not select models. Conditioning a forecast and its score on a
realized nonzero move uses an unavailable future-activity gate; it is a separate
diagnostic, not the all-tick forecast deployed by the policy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from audit_joint_vs_neural_specialists import score_components
from export_ternary_sign_size_forecast import (
    endpoint_law, frozen_selection, load_combined, predict, record_for, sha,
)
from stochastic_price_forecast import load_timeline_split
from ternary_sign_size_forecast import physical_downside_cap
from train_neural_sign_size_forecast import HORIZONS, examples, now, select_origins, write_json

CONTRACT = "frozen-ternary-finetune-diagnostics-v1"
CLASS_ORDER = ("DOWN", "ZERO", "UP")


def class_metrics(y: np.ndarray, probability: np.ndarray) -> dict:
    y, probability = np.asarray(y), np.asarray(probability, dtype=np.float64)
    if y.ndim != 1 or probability.shape != (len(y), 3) or not len(y):
        raise ValueError("one ternary probability vector per realized endpoint is required")
    if not np.isfinite(y).all() or not np.isfinite(probability).all() or np.any(probability < 0):
        raise ValueError("invalid class targets or probabilities")
    np.testing.assert_allclose(probability.sum(axis=1), 1, rtol=0, atol=2e-15)
    actual = (y > 0).astype(np.int64)+(y >= 0).astype(np.int64)
    predicted = probability.argmax(axis=1)
    confusion = np.bincount(3*actual+predicted, minlength=9).reshape(3, 3)
    counts = confusion.sum(axis=1)
    recall = [float(confusion[k, k]/counts[k]) if counts[k] else None for k in range(3)]
    realized_probability = probability[np.arange(len(y)), actual]
    return {"classOrder": list(CLASS_ORDER), "confusionRowsActualColumnsPredicted": confusion.tolist(),
        "actualCounts": counts.tolist(), "predictedCounts": confusion.sum(axis=0).tolist(),
        "accuracy": float(np.trace(confusion)/len(y)), "classRecall": dict(zip(CLASS_ORDER, recall)),
        "balancedAccuracyOverPresentClasses": float(np.mean([v for v in recall if v is not None])),
        "actualClassProbabilityBelowScoreFloorCount": int((realized_probability < 1e-12).sum()),
        "argmaxTieRule": "first maximum in DOWN, ZERO, UP order"}


def stable_class_logloss(y: np.ndarray, logits: np.ndarray) -> float:
    """Score original logits without flooring tiny true-class probabilities."""
    y, logits = np.asarray(y), np.asarray(logits, dtype=np.float64)
    if logits.shape != (len(y), 3) or not np.isfinite(logits).all():
        raise ValueError("finite ternary logits must match the evaluated targets")
    maximum = logits.max(axis=1)
    actual = (y > 0).astype(np.int64)+(y >= 0).astype(np.int64)
    return float(np.mean(maximum+np.log(np.exp(logits-maximum[:, None]).sum(axis=1))
                         -logits[np.arange(len(y)), actual]))


def diagnostics(y: np.ndarray, probability: np.ndarray, up: np.ndarray, down: np.ndarray,
                constants: dict) -> dict:
    _, p, _, _, numerical_error = endpoint_law(probability, up, down)
    active_mass = p[:, 0]+p[:, 2]
    active_up = np.divide(p[:, 2], active_mass, out=np.full_like(active_mass, .5), where=active_mass > 0)
    result = score_components(y, p[:, 1], active_up, up, down, constants)
    undefined = int(((active_mass == 0) & (y != 0)).sum())
    result["activeOnly"]["undefinedConditionalForecastOnRealizedActiveCount"] = undefined
    if undefined:
        # A conditional distribution is not identified on an event assigned zero
        # probability; do not score the arbitrary numerical placeholder above.
        result["activeOnly"]["conditionalActiveMeanMseBpsSquared"] = None
        result["activeOnly"]["conditionalActiveMeanMseSkillVsZero"] = None
    result["threeClass"] = class_metrics(y, p)
    result["maximumSoftmaxSumErrorBeforeNormalization"] = numerical_error
    return result


def array_sha(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def archived_specialists(archive: dict, horizon: int, split: str, population: dict) -> dict:
    matches = [row for row in archive["results"] if row["horizonSeconds"] == horizon
               and row["role"] == "previouslySelectedDensity"]
    if len(matches) != 1:
        raise ValueError("archived specialist baseline must have exactly one selected row per horizon")
    previous = matches[0]["splits"][split]
    for field in ("firstTimestamp", "lastOriginTimestamp", "examples"):
        if previous[field] != population[field]:
            raise ValueError("archived specialist baseline uses a different origin population")
    return previous["mixtures"]["specialists"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, default=Path("data/training/runs/stochastic-crra-ternary-finetune-v1"))
    parser.add_argument("--old-audit", type=Path, default=Path("data/backtests/oracle-neural-specialists/joint-marginal-audit.json"))
    parser.add_argument("--output", type=Path, default=Path("data/backtests/oracle-ternary-finetune/diagnostics.json"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=2560)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    device = torch.device(args.device)
    selection_file = args.runs_root/"selection.json"
    selected_sha = sha(selection_file)
    selected = frozen_selection(selection_file)
    if tuple(entry["horizonSeconds"] for entry in selected["horizons"]) != HORIZONS:
        raise ValueError("unexpected selected horizon order")
    if selected["plan"]["debugLimits"] != {"train": None, "validation": None}:
        raise ValueError("matched audit requires the full frozen experiment")
    source_file = Path(selected["plan"]["sourceSelection"])
    source_sha = sha(source_file)
    if source_sha != selected["plan"]["sourceSelectionSha256"]:
        raise ValueError("source specialist selection changed")
    source = json.loads(source_file.read_text(encoding="utf-8"))
    plan = source["plan"]
    dataset = Path(plan["dataset"])
    dataset_sha = sha(dataset/"manifest.json")
    if dataset_sha != plan["datasetManifestSha256"] or dataset_sha != selected["plan"]["datasetManifestSha256"]:
        raise ValueError("dataset differs from the frozen experiments")
    archive_sha = sha(args.old_audit)
    archive = json.loads(args.old_audit.read_text(encoding="utf-8"))
    if archive["specialistSelectionSha256"] != source_sha or archive["datasetManifestSha256"] != dataset_sha:
        raise ValueError("archived baseline is not from the frozen source specialists")
    if archive["evaluationBatchSize"] != args.batch_size:
        raise ValueError("use the same inference batch size as the archived baseline")
    result_file = args.runs_root/"result.json"
    result_sha = sha(result_file)  # Require the full trainer, including test evaluation, to have completed.
    trainer_result = json.loads(result_file.read_text(encoding="utf-8"))
    if trainer_result["selectionSha256"] != selected_sha or trainer_result["testUsedForSelection"]:
        raise ValueError("trainer evaluation is not attached to this frozen selection")
    output = {"contract": CONTRACT, "createdAt": now(), "selectionFile": str(selection_file),
        "selectionSha256": selected_sha, "sourceSelectionFile": str(source_file), "sourceSelectionSha256": source_sha,
        "datasetManifestSha256": dataset_sha, "archivedBaselineFile": str(args.old_audit), "archivedBaselineSha256": archive_sha,
        "trainerResultFile": str(result_file), "trainerResultSha256": result_sha,
        "changesTrainingOrSelection": False, "testUsedForSelection": False,
        "evaluationBatchSize": args.batch_size,
        "population": "original specialist common stride-4 validation/test origins with every horizon through 7200s inside its split; overlapping outcomes, not independent samples",
        "target": "SIMPLE endpoint price return; exact equal prices are ZERO",
        "activeDiagnosticScope": "conditional-active mean scores use the realized future nonzero-activity gate and are not deployed all-tick policy scores; historical deleted 7% models had different labels/populations",
        "componentScope": "pure composed-mean MSE does not identify calibrated sign probabilities or true conditional sizes",
        "archivedBaselineMatching": "identical source selection/dataset hashes and original deterministic origin selection; count and first/last timestamp verified; baseline is not retrained or selected here",
        "results": []}
    started = time.monotonic()
    for split_name in ("validation", "test"):
        timeline = load_timeline_split(dataset, split_name)
        origins = select_origins(timeline, plan["strideSeconds"], plan["validationLimit"] if split_name == "validation" else None)
        data = examples(timeline, origins)
        population = {"firstTimestamp": int(data.axis_start_ms+origins[0]*1000),
            "lastOriginTimestamp": int(data.axis_start_ms+origins[-1]*1000), "examples": len(origins),
            "originIndicesInt64Sha256": array_sha(origins.astype("<i8")),
            "targetSimpleReturnsFloat64Sha256": array_sha(data.targets.astype("<f8"))}
        for j, entry in enumerate(selected["horizons"]):
            h = entry["horizonSeconds"]
            old = archived_specialists(archive, h, split_name, population)
            if old["activeExamples"] != int((data.targets[:, j] != 0).sum()):
                raise ValueError("archived baseline has different target activity labels")
            scored = {"originalBinarySpecialists": old}
            records = {}
            for stage, trainer_name in (("pretrained", "separatePretrained"), ("finetuned", "jointlyFinetuned")):
                record = record_for(entry, stage)
                model = load_combined(record, h, selected["planSha256"], device)
                logits = []
                def collect_logits(_module, _inputs, value):
                    logits.append(value[:, 0].detach().cpu().numpy().astype(np.float64))
                handle = model.sign.register_forward_hook(collect_logits)
                p, up, down, _ = predict(model, timeline.timeline, origins, device, args.batch_size)
                handle.remove()
                measured = diagnostics(data.targets[:, j], p, up, down, entry["constants"])
                measured["threeClass"]["stableLogitLogLoss"] = stable_class_logloss(data.targets[:, j], np.concatenate(logits))
                measured["physicalDownsideCap"] = model.downside_cap
                measured["physicalDownsideCapHits"] = int((down >= model.downside_cap).sum())
                measured["trainerFloat32BoundaryDiagnosticCount"] = int((down >= physical_downside_cap(torch.float32)).sum())
                original = trainer_result[split_name][j]["models"][trainer_name]
                if trainer_result[split_name][j]["horizonSeconds"] != h:
                    raise ValueError("trainer result horizon order changed")
                for field in ("allTickSignedMeanMseBpsSquared", "allTickSignedMeanMseSkillVsZero"):
                    np.testing.assert_allclose(measured[field], original[field], rtol=1e-10, atol=1e-10)
                scored[stage], records[stage] = measured, record
                del model
            row = {"split": split_name, "horizonSeconds": h, **population, "checkpoints": records, "models": scored}
            output["results"].append(row)
            print(json.dumps({"split": split_name, "horizon": h, "models": {name: {
                "allTickMeanSkill": metrics["allTickSignedMeanMseSkillVsZero"],
                "activeConditionalMeanSkill": metrics["activeOnly"]["conditionalActiveMeanMseSkillVsZero"],
                "threeClassAccuracy": metrics.get("threeClass", {}).get("accuracy")}
                for name, metrics in scored.items()}}, allow_nan=False), flush=True)
    for file, expected in ((selection_file, selected_sha), (source_file, source_sha),
                           (args.old_audit, archive_sha), (result_file, result_sha),
                           (dataset/"manifest.json", dataset_sha)):
        if sha(file) != expected:
            raise ValueError("frozen audit input changed during evaluation")
    frozen_selection(selection_file)
    output["seconds"] = time.monotonic()-started
    write_json(args.output, output)


if __name__ == "__main__":
    main()
