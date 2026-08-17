from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time


MATRIX_ID = "next-return-knot-density-scaling-v1"
BASE_PLANS = (
    ("65k", "next-second-4-layer-memorization-65k-static-c-v1.json"),
    ("128k", "next-second-4-layer-memorization-128k-static-c-v1.json"),
    ("256k", "next-second-4-layer-memorization-256k-static-c-v1.json"),
)
KNOT_COUNTS = (8, 16, 32, 64, 128, 256, 512, 1024)
REUSED_RUN = "next-second-4-layer-knot-density-65k-v1"
SPLIT_PLAN = "direct-glu-to-next-1s-recent-4m-1-layer-long-v2.json"
PAUSE_EXIT_CODE = 75


def run_id(dataset_label: str, knot_count: int) -> str:
    if dataset_label == "65k" and knot_count == 64:
        return REUSED_RUN
    return (
        f"next-second-4-layer-knot-density-{dataset_label}-"
        f"k{knot_count}-v1"
    )


def derived_plan(
    source: dict,
    dataset_label: str,
    knot_count: int,
) -> dict:
    identifier = run_id(dataset_label, knot_count)
    plan = json.loads(json.dumps(source))
    plan["id"] = identifier
    plan["label"] = (
        "Distribution scaling - four-layer width-512 next-1s GLU - "
        f"{dataset_label} clean examples - {knot_count} fixed global knots - NLL"
    )
    plan["datasetDir"] = f"data/training/datasets/{identifier}"
    plan["runDir"] = f"data/training/runs/{identifier}"
    plan["subset"]["type"] = "fixed-clean-count"
    plan["evaluation"] = {
        "type": "fixed-clean-heldout-count",
        "validationStart": "2026-06-01",
        "testStart": "2026-07-01",
        "examplesPerSplit": 65_536,
    }
    plan["datasetFilter"] = {
        "type": "exclude-exact-zero-target-return",
        "appliesTo": ["normalization", "training", "validation", "test"],
        "comparison": "float32-exact-zero-after-log-return-construction",
    }
    plan["density"] = {
        "source": "data/benchmarks/one-second-return-knot-scaling-v1.json",
        "fit": str(knot_count),
        "knotCount": knot_count,
        "transform": "frozen-asinh-logistic",
        "basis": "normalized-piecewise-linear-triangular",
        "output": "conditional-log-density-height",
        "controlledPrior": "same global KL density across knot counts",
    }
    plan["training"]["evaluationBatchSize"] = 8_192
    plan["training"]["selection"] = (
        "lowest-validation-return-space-negative-log-likelihood"
    )
    plan["training"].pop("targetNormalizedMse", None)
    return plan


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    plans_root = repo / "ml/training-plans"
    matrix_root = repo / f"data/training/matrices/{MATRIX_ID}"
    generated_plans = matrix_root / "plans"
    generated_plans.mkdir(parents=True, exist_ok=True)
    pause_file = matrix_root / "control/PAUSE"
    calibrator = repo / "ml/calibrate_next_return_density.py"
    checkpoint_recovery = (
        repo / "ml/recover_return_density_checkpoint_selections.py"
    )
    split_plan = plans_root / SPLIT_PLAN
    environment = os.environ.copy()
    environment["TRADING_STORAGE_GC_INTERVAL_MINUTES"] = "1"
    environment["TRADING_STORAGE_ORPHAN_GRACE_HOURS"] = "0.001"

    subprocess.run(
        (sys.executable, str(repo / "ml/build_return_density_knot_scaling.py")),
        cwd=repo,
        env=environment,
        check=True,
    )

    for dataset_label, base_name in BASE_PLANS:
        source = json.loads((plans_root / base_name).read_text(encoding="utf-8"))
        for knot_count in KNOT_COUNTS:
            identifier = run_id(dataset_label, knot_count)
            result_file = repo / f"data/training/runs/{identifier}/state/result.json"
            run_root = result_file.parents[1]
            if not result_file.is_file():
                plan = derived_plan(source, dataset_label, knot_count)
                plan_file = generated_plans / f"{identifier}.json"
                plan_file.write_text(
                    json.dumps(plan, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8",
                )
                command = (
                    sys.executable,
                    str(repo / "ml/train_next_return_knot_density.py"),
                    "--plan",
                    str(plan_file),
                    "--pause-file",
                    str(pause_file),
                )
                while not result_file.is_file():
                    while pause_file.is_file():
                        time.sleep(2)
                    completed = subprocess.run(command, cwd=repo, env=environment)
                    if completed.returncode == PAUSE_EXIT_CODE:
                        continue
                    completed.check_returncode()
                    if not result_file.is_file():
                        raise RuntimeError(
                            f"density trainer exited without result: {identifier}"
                        )
            calibration_file = (
                run_root / "state/output-calibration-pre-validation-7d.json"
            )
            if not calibration_file.is_file():
                subprocess.run((
                    sys.executable,
                    str(calibrator),
                    "--training-plan",
                    str(run_root / "state/plan.json"),
                    "--split-plan",
                    str(split_plan),
                    "--batch-size",
                    "8192",
                ), cwd=repo, env=environment, check=True)
            selection_file = (
                run_root / "state/checkpoint-selection-comparison.json"
            )
            if not selection_file.is_file():
                command = (
                    sys.executable,
                    str(checkpoint_recovery),
                    "--training-plan",
                    str(run_root / "state/plan.json"),
                    "--batch-size",
                    "8192",
                    "--pause-file",
                    str(pause_file),
                )
                while not selection_file.is_file():
                    while pause_file.is_file():
                        time.sleep(2)
                    completed = subprocess.run(
                        command, cwd=repo, env=environment
                    )
                    if completed.returncode == PAUSE_EXIT_CODE:
                        continue
                    completed.check_returncode()
                    if not selection_file.is_file():
                        raise RuntimeError(
                            "checkpoint recovery exited without comparison: "
                            f"{identifier}"
                        )
            selection_calibration_file = (
                run_root / "state/checkpoint-selection-calibrations.json"
            )
            if not selection_calibration_file.is_file():
                subprocess.run((
                    sys.executable,
                    str(calibrator),
                    "--training-plan",
                    str(run_root / "state/plan.json"),
                    "--split-plan",
                    str(split_plan),
                    "--batch-size",
                    "8192",
                    "--checkpoint-policies",
                    "train-mse",
                    "validation-mse",
                    "train-correlation",
                    "validation-correlation",
                    "train-nll",
                    "validation-nll",
                ), cwd=repo, env=environment, check=True)


if __name__ == "__main__":
    main()
