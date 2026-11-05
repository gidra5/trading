"""Matched frozen hurdle/binary/ternary evaluation without model selection.

The new trainer already scores fixed checkpoints on train, validation and test.
This audit reproduces held-out scores through the export loader and verifies
that the pretrained hurdle's conditional sign/size tensors still equal the
original specialists. Archived earlier scores are joined by origin/target hash.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy.special import expit
import torch

from audit_joint_vs_neural_specialists import score_components
from audit_ternary_finetune import array_sha, class_metrics, diagnostics as ternary_diagnostics
from export_hurdle_sign_size_forecast import frozen_selection, load_combined
from export_ternary_sign_size_forecast import (
    endpoint_law, frozen_selection as frozen_ternary_selection, load_combined as load_ternary_combined,
    predict as predict_ternary, record_for, sha,
)
from hurdle_sign_size_forecast import hurdle_components, hurdle_mean
from stochastic_price_forecast import load_timeline_split
from train_hurdle_sign_size_forecast import load_combined as training_load_combined, metrics
from train_neural_sign_size_forecast import HORIZONS, examples, now, select_origins, write_json
from train_ternary_sign_size_forecast import load_verified, singleton_state

CONTRACT = "frozen-hurdle-versus-specialists-audit-v1"


@torch.inference_mode()
def predictions(model, features: np.ndarray, device: torch.device, batch: int) -> np.ndarray:
    return np.concatenate([model(torch.from_numpy(features[start:start+batch]).to(device))[:, 0].cpu().numpy()
                           for start in range(0, len(features), batch)])


def frozen_conditional_experts(model, entry: dict, j: int) -> dict:
    """All copied state tensors match the old selected models, not just metadata."""
    result = {}
    for task, width in (("sign", 1), ("size", 2)):
        record = entry["pretrained"][task]
        saved = load_verified(record)
        if saved["horizonsSeconds"] != list(HORIZONS) or saved["task"] != task:
            raise ValueError("original conditional expert contract mismatch")
        expected = singleton_state(saved["modelState"], j, width)
        actual = getattr(model, task).state_dict()
        if actual.keys() != expected.keys():
            raise ValueError("copied conditional expert state layout changed")
        for key in actual:
            torch.testing.assert_close(actual[key].cpu(), expected[key], rtol=0, atol=0)
        result[task] = {"sourceCheckpoint": record["checkpoint"], "sha256": record["sha256"],
                        "exactStateTensorMatches": len(actual)}
    return result


def matched_archive(archive: dict, split: str, h: int, population: dict) -> dict:
    rows = [r for r in archive["results"] if r["split"] == split and r["horizonSeconds"] == h]
    if len(rows) != 1:
        raise ValueError("archived ternary/binary baseline has ambiguous horizon/split")
    row = rows[0]
    for field in ("firstTimestamp", "lastOriginTimestamp", "examples", "originIndicesInt64Sha256",
                  "targetSimpleReturnsFloat64Sha256"):
        if row[field] != population[field]:
            raise ValueError("prior baseline uses a different origin/target population")
    return row["models"]


def new_metrics(output: np.ndarray, y: np.ndarray, constants: dict) -> dict:
    scored = metrics(output, y, constants)
    packed = torch.from_numpy(output[:, None, :])
    probability, up, down = hurdle_components(packed)
    p, up, down = probability[:, 0].numpy(), up[:, 0].numpy(), down[:, 0].numpy()
    _, normalized, mean, _, error = endpoint_law(p, up, down)
    model_mean = hurdle_mean(packed)[:, 0].numpy()
    mean_error = float(np.max(np.abs(mean-model_mean)))
    if mean_error > 1e-10:
        raise ValueError("export conditional-mean law changed the trained mean")
    scored["threeClass"] = class_metrics(y, normalized)
    scored["zeroTickContributionToTotalMseBpsSquared"] = float(np.sum(np.square(model_mean[y == 0]))/len(y)*1e8)
    scored["zeroOnlyMeanSquaredPredictionBpsSquared"] = float(np.mean(np.square(model_mean[y == 0]))*1e8) if np.any(y == 0) else None
    scored["maximumExportVsModelMeanErrorBps"] = mean_error*1e4
    scored["probabilityNormalizationError"] = error
    return scored


def old_mse_train_audit(selected: dict, source_plan: dict, device: torch.device, batch: int) -> dict:
    """Evaluate both old selected EMA checkpoints on one fixed training set."""
    if selected["plan"]["debugLimits"] != {"train": None, "validation": None}:
        raise ValueError("old MSE comparison requires the original full training population")
    timeline = load_timeline_split(Path(source_plan["dataset"]), "train")
    origins = select_origins(timeline, source_plan["strideSeconds"], source_plan["trainingLimit"])
    data = examples(timeline, origins)
    entry = selected["horizons"][0]
    if entry["horizonSeconds"] != 1:
        raise ValueError("old fixed-checkpoint training audit is restricted to H1")
    output = {"horizonSeconds": 1, "examples": len(origins),
        "firstTimestamp": int(data.axis_start_ms+origins[0]*1000),
        "lastOriginTimestamp": int(data.axis_start_ms+origins[-1]*1000),
        "originIndicesInt64Sha256": array_sha(origins.astype("<i8")),
        "targetSimpleReturnsFloat64Sha256": array_sha(data.targets.astype("<f8")),
        "checkpointInterpretation": "fixed previously selected EMA checkpoints; not online losses of changing raw weights",
        "trainingUsedForNewSelection": False, "models": {}}
    y = data.targets[:, 0]
    for stage in ("pretrained", "finetuned"):
        record = record_for(entry, stage)
        model = load_ternary_combined(record, 1, selected["planSha256"], device)
        p, up, down, raw_mean = predict_ternary(model, timeline.timeline, origins, device, batch)
        measured = ternary_diagnostics(y, p, up, down, entry["constants"])
        _, _, normalized_mean, _, _ = endpoint_law(p, up, down)
        raw_error = np.square(raw_mean-y)
        active = y != 0
        raw_mse = float(raw_error.mean()*1e8)
        raw_zero = float(raw_error[~active].sum()/len(y)*1e8)
        raw_active = float(raw_error[active].sum()/len(y)*1e8)
        measured["activeTickContributionToTotalMseBpsSquared"] = measured["allTickSignedMeanMseBpsSquared"]-measured["zeroTickContributionToTotalMseBpsSquared"]
        measured["exactTrainingForwardMean"] = {"allTickMseBpsSquared": raw_mse,
            "zeroTickContributionToTotalMseBpsSquared": raw_zero,
            "activeTickContributionToTotalMseBpsSquared": raw_active,
            "allTickMseSkillVsZero": 1-float(raw_error.mean()/np.square(y).mean()),
            "maximumNumericalNormalizationMeanDifferenceBps": float(np.max(np.abs(normalized_mean-raw_mean))*1e4)}
        measured["checkpoint"] = record
        output["models"][stage] = measured
        del model
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, default=Path("data/training/runs/stochastic-crra-hurdle-nll-v1"))
    parser.add_argument("--previous-audit", type=Path, default=Path("data/backtests/oracle-ternary-finetune/diagnostics.json"))
    parser.add_argument("--output", type=Path, default=Path("data/backtests/oracle-hurdle-nll/diagnostics.json"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=2560)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    device = torch.device(args.device)
    selection_file, result_file = args.runs_root/"selection.json", args.runs_root/"result.json"
    selected = frozen_selection(selection_file)
    selected_sha, result_sha = sha(selection_file), sha(result_file)
    result = json.loads(result_file.read_text(encoding="utf-8"))
    if result["selectionSha256"] != selected_sha or result["testUsedForSelection"]:
        raise ValueError("trainer result does not match the frozen selection")
    if tuple(entry["horizonSeconds"] for entry in selected["horizons"]) != HORIZONS \
            or selected["plan"]["debugLimits"] != {"train": None, "validation": None}:
        raise ValueError("matched audit requires all original horizons and complete populations")
    source_file = Path(selected["plan"]["sourceSelection"])
    source_sha = sha(source_file)
    if source_sha != selected["plan"]["sourceSelectionSha256"]:
        raise ValueError("source binary specialist selection changed")
    source = json.loads(source_file.read_text(encoding="utf-8"))
    source_plan = source["plan"]
    dataset = Path(source_plan["dataset"])
    manifest_file, manifest_sha = dataset/"manifest.json", source_plan["datasetManifestSha256"]
    if sha(manifest_file) != manifest_sha or selected["plan"]["datasetManifestSha256"] != manifest_sha:
        raise ValueError("dataset differs from either frozen experiment")
    archive_sha = sha(args.previous_audit)
    archive = json.loads(args.previous_audit.read_text(encoding="utf-8"))
    if archive["sourceSelectionSha256"] != source_sha or archive["datasetManifestSha256"] != manifest_sha \
            or archive["evaluationBatchSize"] != args.batch_size:
        raise ValueError("archived binary/ternary baseline is not the matched source experiment")
    prior_selection_file = Path(archive["selectionFile"])
    if sha(prior_selection_file) != archive["selectionSha256"]:
        raise ValueError("archived ternary selection changed")
    prior_selected = frozen_ternary_selection(prior_selection_file)
    if prior_selected["plan"]["sourceSelectionSha256"] != source_sha:
        raise ValueError("old ternary model does not use the same original specialist source")
    output = {"contract": CONTRACT, "createdAt": now(), "selectionFile": str(selection_file),
        "selectionSha256": selected_sha, "trainerResultFile": str(result_file), "trainerResultSha256": result_sha,
        "sourceSelectionFile": str(source_file), "sourceSelectionSha256": source_sha,
        "previousAuditFile": str(args.previous_audit), "previousAuditSha256": archive_sha,
        "datasetManifestSha256": manifest_sha, "changesTrainingOrSelection": False, "testUsedForSelection": False,
        "evaluationBatchSize": args.batch_size,
        "population": "same common stride4 held-out origins and exact SIMPLE endpoint targets, verified by SHA256; overlapping outcomes are not independent samples",
        "newFixedCheckpointTrainMetrics": result["train"],
        "trainingMetricsScope": "new trainer records fixed-checkpoint train scores; this audit additionally evaluates old H1 ternary pre/MSE-fine checkpoints on the same full training population. Other old-model train scores are unavailable",
        "comparisonScope": "binary-to-hurdle-pretrained comparison isolates the zero-probability head after exact conditional-expert tensor checks, provided physical bound hit counts are zero",
        "nllScope": "full NLL is only compared between the two Gamma/Beta hurdle stages; earlier point/mean models did not supply this conditional density",
        "activeDiagnosticScope": "active-only conditional-mean diagnostics use a realized future-activity gate and are separate from deployed all-tick forecasts",
        "results": []}
    started = time.monotonic()
    output["oldMseFinetuneTrainAudit"] = {"selectionFile": str(prior_selection_file),
        "selectionSha256": archive["selectionSha256"],
        **old_mse_train_audit(prior_selected, source_plan, device, args.batch_size)}
    print(json.dumps({"event": "oldMseFixedCheckpointTrainAudit", "meanSkills": {name:value["allTickSignedMeanMseSkillVsZero"]
        for name,value in output["oldMseFinetuneTrainAudit"]["models"].items()}}, allow_nan=False), flush=True)
    for split in ("validation", "test"):
        timeline = load_timeline_split(dataset, split)
        origins = select_origins(timeline, source_plan["strideSeconds"], source_plan["validationLimit"] if split == "validation" else None)
        data = examples(timeline, origins)
        population = {"firstTimestamp": int(data.axis_start_ms+origins[0]*1000),
            "lastOriginTimestamp": int(data.axis_start_ms+origins[-1]*1000), "examples": len(origins),
            "originIndicesInt64Sha256": array_sha(origins.astype("<i8")),
            "targetSimpleReturnsFloat64Sha256": array_sha(data.targets.astype("<f8"))}
        for j, entry in enumerate(selected["horizons"]):
            h = entry["horizonSeconds"]
            prior = matched_archive(archive, split, h, population)
            scored = {"originalBinarySpecialists": prior["originalBinarySpecialists"],
                      "ternaryPretrained": prior["pretrained"], "ternaryMeanMseFinetuned": prior["finetuned"]}
            checks = {}
            for stage, trainer_name in (("pretrained", "separatePretrained"), ("finetuned", "jointlyFinetuned")):
                record = record_for(entry, stage)
                model = load_combined(record, h, selected["planSha256"], device)
                if stage == "pretrained":
                    checks["frozenConditionalExperts"] = frozen_conditional_experts(model, entry, j)
                # Different constructors/loaders must preserve every packed
                # likelihood parameter, not merely its mean used by the policy.
                training_model = training_load_combined(record, device)
                with torch.inference_mode():
                    probe = torch.from_numpy(data.features[:17]).to(device)
                    torch.testing.assert_close(model(probe), training_model(probe), rtol=0, atol=0)
                del training_model
                parameters = predictions(model, data.features, device, args.batch_size)
                measured = new_metrics(parameters, data.targets[:, j], entry["constants"])
                original = result[split][j]["models"][trainer_name]
                if result[split][j]["horizonSeconds"] != h:
                    raise ValueError("trainer result horizon order changed")
                for field in ("fullHurdleReturnNegativeLogLikelihood", "threeClassLogLoss",
                              "allTickSignedMeanMseBpsSquared", "allTickSignedMeanMseSkillVsZero"):
                    np.testing.assert_allclose(measured[field], original[field], rtol=1e-10, atol=1e-10)
                if stage == "pretrained":
                    reconstructed = score_components(data.targets[:, j], entry["constants"]["pZero"],
                        expit(parameters[:, 1]), parameters[:, 2], parameters[:, 3], entry["constants"])
                    archived_binary = prior["originalBinarySpecialists"]
                    for field in ("allTickSignedMeanMseBpsSquared", "allTickSignedMeanMseSkillVsZero"):
                        np.testing.assert_allclose(reconstructed[field], archived_binary[field], rtol=1e-6, atol=1e-9)
                    for field in ("positiveSize", "negativeSize"):
                        np.testing.assert_allclose(reconstructed[field]["mseBpsSquared"],
                            archived_binary[field]["mseBpsSquared"], rtol=1e-6, atol=1e-9)
                    checks["originalFixedZeroMeanSkillReconstructed"] = reconstructed["allTickSignedMeanMseSkillVsZero"]
                    checks["originalFixedZeroMeanSkillAbsoluteDifference"] = abs(reconstructed["allTickSignedMeanMseSkillVsZero"]-archived_binary["allTickSignedMeanMseSkillVsZero"])
                    checks["onlyZeroHeadChangesBeforeFinetuning"] = measured["minimumMeanFloorHits"] == {"up": 0, "down": 0} and measured["downsideAtPhysicalBoundaryCount"] == 0
                scored["hurdlePretrained" if stage == "pretrained" else "hurdleNllFinetuned"] = measured
                del model
            row = {"split": split, "horizonSeconds": h, **population, "checks": checks, "models": scored}
            output["results"].append(row)
            print(json.dumps({"split": split, "horizon": h, "meanSkills": {name:value["allTickSignedMeanMseSkillVsZero"]
                for name,value in scored.items()}, "onlyZeroChanges": checks["onlyZeroHeadChangesBeforeFinetuning"]}, allow_nan=False), flush=True)
    for file, expected in ((selection_file, selected_sha), (result_file, result_sha), (source_file, source_sha),
                           (manifest_file, manifest_sha), (args.previous_audit, archive_sha),
                           (prior_selection_file, archive["selectionSha256"])):
        if sha(file) != expected:
            raise ValueError("frozen audit input changed during evaluation")
    frozen_selection(selection_file)
    frozen_ternary_selection(prior_selection_file)
    output["seconds"] = time.monotonic()-started
    write_json(args.output, output)


if __name__ == "__main__":
    main()
