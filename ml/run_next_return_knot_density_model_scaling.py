from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time


MATRIX_ID = "next-return-knot-density-model-scaling-k256-v1"
KNOT_COUNT = 256
SPLIT_PLAN = "direct-glu-to-next-1s-recent-4m-1-layer-long-v2.json"
PAUSE_EXIT_CODE = 75
POLL_SECONDS = 2
PREDECESSOR_ARTIFACT = (
    "data/training/runs/"
    "next-second-4-layer-knot-density-256k-k1024-v1/"
    "state/checkpoint-selection-calibrations.json"
)
CASES = (
    (
        "next-second-4-layer-knot-density-65k-k256-v1",
        "next-second-4-layer-memorization-65k-static-c-v1.json",
        "four-layer width-512",
        "65k",
        True,
    ),
    (
        "next-second-4-layer-knot-density-128k-k256-v1",
        "next-second-4-layer-memorization-128k-static-c-v1.json",
        "four-layer width-512",
        "128k",
        True,
    ),
    (
        "next-second-4-layer-knot-density-256k-k256-v1",
        "next-second-4-layer-memorization-256k-static-c-v1.json",
        "four-layer width-512",
        "256k",
        True,
    ),
    (
        "next-second-4-layer-knot-density-512k-k256-v1",
        "next-second-4-layer-memorization-512k-static-c-v1.json",
        "four-layer width-512",
        "512k",
        False,
    ),
    (
        "next-second-8-layer-knot-density-512k-k256-v1",
        "next-second-8-layer-memorization-512k-static-c-v1.json",
        "eight-layer width-512",
        "512k",
        False,
    ),
    (
        "next-second-8-layer-width1024-knot-density-1024k-k256-v1",
        "next-second-8-layer-memorization-1024k-width1024-static-c-v1.json",
        "eight-layer width-1024",
        "1024k",
        False,
    ),
)


def derived_plan(
    source: dict,
    identifier: str,
    architecture_label: str,
    dataset_label: str,
) -> dict:
    plan = json.loads(json.dumps(source))
    plan["id"] = identifier
    plan["label"] = (
        "Distribution model scaling - "
        f"{architecture_label} next-1s GLU - {dataset_label} clean examples - "
        f"{KNOT_COUNT} fixed global knots - NLL"
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
        "fit": str(KNOT_COUNT),
        "knotCount": KNOT_COUNT,
        "transform": "frozen-asinh-logistic",
        "basis": "normalized-piecewise-linear-triangular",
        "output": "conditional-log-density-height",
        "controlledPrior": "same global KL density across model scales",
    }
    plan["training"]["evaluationBatchSize"] = 8_192
    plan["training"]["selection"] = (
        "lowest-validation-return-space-negative-log-likelihood"
    )
    plan["training"].pop("targetNormalizedMse", None)
    return plan


def wait_while_paused(pause_file: Path) -> None:
    while pause_file.is_file():
        time.sleep(POLL_SECONDS)


def run_until_artifact(
    command: tuple[str, ...],
    artifact: Path,
    pause_file: Path,
    repo: Path,
    environment: dict[str, str],
) -> None:
    while not artifact.is_file():
        wait_while_paused(pause_file)
        completed = subprocess.run(command, cwd=repo, env=environment)
        if completed.returncode == PAUSE_EXIT_CODE:
            continue
        completed.check_returncode()
        if not artifact.is_file():
            raise RuntimeError(f"command exited without artifact: {artifact}")


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    plans_root = repo / "ml/training-plans"
    matrix_root = repo / f"data/training/matrices/{MATRIX_ID}"
    generated_plans = matrix_root / "plans"
    generated_plans.mkdir(parents=True, exist_ok=True)
    pause_file = matrix_root / "control/PAUSE"
    split_plan = plans_root / SPLIT_PLAN
    calibrator = repo / "ml/calibrate_next_return_density.py"
    checkpoint_recovery = repo / "ml/recover_return_density_checkpoint_selections.py"
    environment = os.environ.copy()
    environment["TRADING_STORAGE_GC_INTERVAL_MINUTES"] = "1"
    environment["TRADING_STORAGE_ORPHAN_GRACE_HOURS"] = "0.001"

    subprocess.run(
        (sys.executable, str(repo / "ml/build_return_density_knot_scaling.py")),
        cwd=repo,
        env=environment,
        check=True,
    )

    plans: dict[str, Path] = {}
    for identifier, base_name, architecture_label, dataset_label, _ in CASES:
        source = json.loads((plans_root / base_name).read_text(encoding="utf-8"))
        plan = derived_plan(source, identifier, architecture_label, dataset_label)
        plan_file = generated_plans / f"{identifier}.json"
        plan_file.write_text(
            json.dumps(plan, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        plans[identifier] = plan_file

    predecessor = repo / PREDECESSOR_ARTIFACT
    while not predecessor.is_file():
        time.sleep(POLL_SECONDS)

    for identifier, _, _, _, reused in CASES:
        run_root = repo / f"data/training/runs/{identifier}"
        result_file = run_root / "state/result.json"
        if not result_file.is_file():
            if reused:
                raise RuntimeError(f"reused density run is missing: {identifier}")
            run_until_artifact(
                (
                    sys.executable,
                    str(repo / "ml/train_next_return_knot_density.py"),
                    "--plan",
                    str(plans[identifier]),
                    "--pause-file",
                    str(pause_file),
                ),
                result_file,
                pause_file,
                repo,
                environment,
            )

        calibration_file = run_root / "state/output-calibration-pre-validation-7d.json"
        if not calibration_file.is_file():
            wait_while_paused(pause_file)
            subprocess.run(
                (
                    sys.executable,
                    str(calibrator),
                    "--training-plan",
                    str(run_root / "state/plan.json"),
                    "--split-plan",
                    str(split_plan),
                    "--batch-size",
                    "8192",
                ),
                cwd=repo,
                env=environment,
                check=True,
            )

        selection_file = run_root / "state/checkpoint-selection-comparison.json"
        if not selection_file.is_file():
            run_until_artifact(
                (
                    sys.executable,
                    str(checkpoint_recovery),
                    "--training-plan",
                    str(run_root / "state/plan.json"),
                    "--batch-size",
                    "8192",
                    "--pause-file",
                    str(pause_file),
                ),
                selection_file,
                pause_file,
                repo,
                environment,
            )

        selection_calibration_file = (
            run_root / "state/checkpoint-selection-calibrations.json"
        )
        if not selection_calibration_file.is_file():
            wait_while_paused(pause_file)
            subprocess.run(
                (
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
                ),
                cwd=repo,
                env=environment,
                check=True,
            )


if __name__ == "__main__":
    main()
