"""Fit and apply executable-policy temperature scaling on validation only.

This command deliberately exposes no split selector.  It runs the ordered
action evaluator on the best checkpoint's chronological validation split,
writes the complete sweep beside the exported ONNX model, and records its
hash plus the selected action metrics in the artifact manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from evaluate_joint_price_oracle_actions import (
    ACTION_TEMPERATURE_CALIBRATION_METHOD,
    ACTION_TEMPERATURE_OBJECTIVE,
    ACTION_TEMPERATURE_TIE_BREAKERS,
    EXACT_STATE_ACTION_TEMPERATURE_OBJECTIVE,
    EXACT_STATE_ACTION_TEMPERATURE_TIE_BREAKERS,
    atomic_json,
    evaluate_plan_checkpoint,
    iso_now,
    parse_positive_temperatures,
    select_action_temperature,
)
from joint_price_oracle_actions import resolve_execution_policy_config
from trading_storage import require_under
from train_joint_price_oracle import resolve, validate_plan


DEFAULT_LOGIT_TEMPERATURES = (
    "0.05,0.075,0.1,0.125,0.15,0.2,0.25,0.3,0.4,0.5,0.6,0.75,"
    "1,1.25,1.5,2"
)
RUNTIME_CALIBRATION_METHOD = "validation-action-temperature-scaling"
CALIBRATION_REPORT_FILE = "calibration.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select and apply a joint price-oracle logit temperature from "
            "chronological validation action rollouts."
        )
    )
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--logit-temperatures",
        default=DEFAULT_LOGIT_TEMPERATURES,
        help="Positive comma-separated validation temperature candidates.",
    )
    return parser.parse_args()


def calibrate_artifact(
    plan_file: Path,
    *,
    requested_device: str = "auto",
    batch_size: int | None = None,
    logit_temperatures: tuple[float, ...],
) -> dict:
    """Evaluate validation and apply its selected temperature to an artifact."""
    report = evaluate_plan_checkpoint(
        plan_file,
        checkpoint_kind="best",
        split="validation",
        allow_test=False,
        output=None,
        requested_device=requested_device,
        batch_size=batch_size,
        logit_temperatures=logit_temperatures,
    )
    return apply_action_calibration(plan_file, report)


def apply_action_calibration(plan_file: Path, report: dict) -> dict:
    """Persist one verified evaluator report and update the model manifest."""
    repo_root = Path(__file__).resolve().parents[1]
    resolved_plan_file = resolve(repo_root, plan_file).resolve()
    plan = json.loads(resolved_plan_file.read_text(encoding="utf-8"))
    validate_plan(plan)
    artifact_dir = require_under(
        resolve(repo_root, Path(plan["artifactDir"])),
        repo_root / "data" / "models" / "joint-price-oracle",
        "artifactDir",
    )
    manifest_file = artifact_dir / "manifest.json"
    if not manifest_file.is_file():
        raise FileNotFoundError(
            "export the best ONNX artifact before applying calibration"
        )
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    calibration = validate_action_calibration_report(
        report,
        plan,
        resolved_plan_file,
        manifest,
    )
    report_file = artifact_dir / CALIBRATION_REPORT_FILE
    atomic_json(report, report_file)
    report_hash = hashlib.sha256(report_file.read_bytes()).hexdigest()
    calibrated_at = iso_now()
    manifest["version"] = 3
    manifest["calibration"] = {
        "method": RUNTIME_CALIBRATION_METHOD,
        "logitTemperature": calibration["selectedLogitTemperature"],
        "objective": calibration["objective"]["metric"],
        "tieBreakers": calibration["tieBreakers"],
        "calibratedAt": calibrated_at,
        "validationReportFile": report_file.name,
        "validationReportSha256": report_hash,
        "validationReportVersion": int(report["version"]),
        "validationExamples": int(report["split"]["examples"]),
        "candidateCount": int(calibration["candidateCount"]),
        "checkpoint": {
            "kind": "best",
            "epoch": int(report["checkpoint"]["epoch"]),
            "globalStep": int(report["checkpoint"]["globalStep"]),
        },
        "datasetFingerprint": report["datasetFingerprint"],
        "executionPolicy": report["actionEvaluation"]["executionPolicy"],
        "rolloutScoreVersion": report["actionEvaluation"][
            "rolloutScoreVersion"
        ],
        "rolloutResetPolicy": report["actionEvaluation"][
            "rolloutResetPolicy"
        ],
        "selectedMetrics": calibration["selectedMetrics"],
        "identityMetrics": calibration["identityMetrics"],
    }
    atomic_json(manifest, manifest_file)
    return manifest


def validate_action_calibration_report(
    report: dict,
    plan: dict,
    resolved_plan_file: Path,
    manifest: dict,
) -> dict:
    """Reject reports that are not the matching best validation evaluation."""
    if not isinstance(report, dict) or report.get("version") != 3:
        raise ValueError("action calibration requires evaluator report version 3")
    report_plan = report.get("plan")
    if not isinstance(report_plan, dict) \
            or report_plan.get("id") != plan.get("id") \
            or Path(str(report_plan.get("file", ""))).resolve() \
            != resolved_plan_file:
        raise ValueError("calibration report does not match the resolved plan")
    split = report.get("split")
    if not isinstance(split, dict) or split.get("name") != "validation":
        raise PermissionError(
            "temperature calibration accepts validation reports only"
        )
    examples = _positive_integer(split.get("examples"), "validation examples")
    if examples < 2:
        raise ValueError("validation calibration requires multiple decisions")
    checkpoint = report.get("checkpoint")
    training = manifest.get("training")
    if not isinstance(checkpoint, dict) \
            or checkpoint.get("kind") != "best" \
            or not isinstance(training, dict) \
            or checkpoint.get("epoch") != training.get("bestEpoch") \
            or checkpoint.get("globalStep") != training.get("globalStep"):
        raise ValueError(
            "calibration report is not from the exported best checkpoint"
        )
    if manifest.get("id") != plan.get("id") \
            or report.get("datasetFingerprint") \
            != training.get("datasetFingerprint"):
        raise ValueError(
            "calibration report artifact or dataset fingerprint is incompatible"
        )
    report_oracle = report.get("oracle")
    manifest_oracle = manifest.get("oracle")
    manifest_output = manifest.get("output")
    if not isinstance(report_oracle, dict) \
            or not isinstance(manifest_oracle, dict) \
            or not isinstance(manifest_output, dict) \
            or report_oracle.get("actionCount") \
            != manifest_output.get("actionCount") \
            or report_oracle.get("friction") \
            != manifest_oracle.get("options", {}).get("friction") \
            or report_oracle.get("temperature") \
            != manifest_oracle.get("options", {}).get("temperature"):
        raise ValueError("calibration report oracle contract is incompatible")
    selection = report.get("actionTemperatureCalibration")
    sweep = report.get("temperatureSweep")
    if not isinstance(selection, dict) \
            or selection.get("method") \
            != ACTION_TEMPERATURE_CALIBRATION_METHOD \
            or not isinstance(sweep, list) or not sweep:
        raise ValueError("calibration selection contract is incompatible")
    objective_metric = selection.get("objective", {}).get("metric")
    expected_tie_breakers = (
        EXACT_STATE_ACTION_TEMPERATURE_TIE_BREAKERS
        if objective_metric == EXACT_STATE_ACTION_TEMPERATURE_OBJECTIVE
        else ACTION_TEMPERATURE_TIE_BREAKERS
    )
    if objective_metric not in {
        ACTION_TEMPERATURE_OBJECTIVE,
        EXACT_STATE_ACTION_TEMPERATURE_OBJECTIVE,
    } or selection.get("objective", {}).get("direction") != "maximize" \
            or selection.get("tieBreakers") != list(expected_tie_breakers):
        raise ValueError("calibration selection objective is incompatible")
    if selection != select_action_temperature(sweep):
        raise ValueError(
            "calibration selection was not deterministically derived from "
            "the recorded temperature sweep"
        )
    selected_index = selection.get("selectedIndex")
    identity_index = selection.get("identityIndex")
    if not isinstance(selected_index, int) \
            or not 0 <= selected_index < len(sweep) \
            or selection.get("candidateCount") != len(sweep) \
            or not isinstance(identity_index, int) \
            or not 0 <= identity_index < len(sweep):
        raise ValueError("calibration selection indexes are invalid")
    selected_item = sweep[selected_index]
    identity_item = sweep[identity_index]
    selected_metrics = _recorded_selection_metrics(selected_item)
    identity_metrics = _recorded_selection_metrics(identity_item)
    selected_temperature = _positive_finite(
        selection.get("selectedLogitTemperature"),
        "selected logit temperature",
    )
    if not isinstance(selected_item, dict) \
            or selected_item.get("logitTemperature") != selected_temperature \
            or selection.get("selectedMetrics") != selected_metrics \
            or not isinstance(identity_item, dict) \
            or identity_item.get("logitTemperature") != 1.0 \
            or selection.get("identityMetrics") != identity_metrics:
        raise ValueError("calibration selected or identity metrics are inconsistent")
    if not isinstance(report.get("evaluatedAt"), str) \
            or not report["evaluatedAt"]:
        raise ValueError("calibration report evaluatedAt is missing")
    action_evaluation = report.get("actionEvaluation")
    if not isinstance(action_evaluation, dict) \
            or action_evaluation.get("rolloutResetPolicy") \
            != "reset-only-at-true-timeline-gaps" \
            or action_evaluation.get("rolloutScoreVersion") not in {1, 2} \
            or "executionPolicy" not in action_evaluation:
        raise ValueError("calibration action-evaluation provenance is missing")
    execution_policy = action_evaluation["executionPolicy"]
    if action_evaluation["rolloutScoreVersion"] == 2 \
            and execution_policy is None:
        raise ValueError("rollout score version 2 requires an execution policy")
    if execution_policy is not None \
            and resolve_execution_policy_config(execution_policy) \
            != execution_policy:
        raise ValueError("calibration execution policy is not canonical")
    return selection


def _recorded_selection_metrics(item: dict) -> dict:
    result = {
        "raw": item.get("raw"),
        "actions": item.get("actions"),
    }
    for name in ("surrogateRolloutActions", "exactStateActions"):
        if name in item:
            result[name] = item[name]
    return result


def _positive_finite(value, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be finite and positive") from error
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _positive_integer(value, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def main() -> None:
    arguments = parse_args()
    manifest = calibrate_artifact(
        arguments.plan,
        requested_device=arguments.device,
        batch_size=arguments.batch_size,
        logit_temperatures=parse_positive_temperatures(
            arguments.logit_temperatures
        ),
    )
    print(json.dumps({
        "artifactId": manifest["id"],
        "method": manifest["calibration"]["method"],
        "selectedLogitTemperature": manifest["calibration"][
            "logitTemperature"
        ],
        "validationExamples": manifest["calibration"][
            "validationExamples"
        ],
        "validationReportSha256": manifest["calibration"][
            "validationReportSha256"
        ],
    }, allow_nan=False))


if __name__ == "__main__":
    main()
