from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time


BASE_PLANS = (
    "next-second-4-layer-memorization-65k-static-c-v1.json",
    "next-second-4-layer-memorization-128k-static-c-v1.json",
    "next-second-4-layer-memorization-256k-static-c-v1.json",
    "next-second-4-layer-memorization-512k-static-c-v1.json",
    "next-second-8-layer-memorization-512k-static-c-v1.json",
    "next-second-8-layer-memorization-1024k-width1024-static-c-v1.json",
)
SWEEP_BASE_PLAN = BASE_PLANS[0]
SPLIT_PLAN = "direct-glu-to-next-1s-recent-4m-1-layer-long-v2.json"
MATRIX_ID = "nonzero-eiil-irm-annealing-v1"
BASELINE_SUFFIX = "nonzero-clean-count-validation-curve-v1"
ANNEALING_FRACTIONS = (0.0, 0.25, 0.5, 0.75)
PENALTY_WEIGHTS = (1.0, 10.0, 30.0, 100.0, 300.0, 1_000.0)
VALIDATION_EXAMPLES = 65_536
PAUSE_EXIT_CODE = 75
WAIT_FOR_RESULT = (
    "data/training/runs/"
    "next-second-4-layer-memorization-256k-static-c-v1-"
    "nonzero-eiil-irmv1-validation-curve-v1/state/result.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Finish the six reusable clean ERM baselines, then sweep IRMv1 "
            "annealing fraction and penalty weight on 65,536 clean examples."
        )
    )
    parser.add_argument("--poll-seconds", type=float, default=2)
    return parser.parse_args()


def token(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def sweep_suffix(fraction: float, penalty_weight: float) -> str:
    return (
        "nonzero-eiil-irmv1-anneal-"
        f"{token(fraction)}-penalty-{token(penalty_weight)}-"
        "validation-curve-v1"
    )


def total_optimizer_steps(plan: dict) -> int:
    subset = plan["subset"]
    if subset.get("type") != "fixed-contiguous":
        raise ValueError("the IRM annealing sweep requires a fixed subset")
    examples = int(subset["examples"])
    batch_size = int(plan["training"]["batchSize"])
    epochs = int(plan["training"]["epochs"])
    return epochs * math.ceil(examples / batch_size)


def main() -> None:
    args = parse_args()
    if args.poll_seconds <= 0:
        raise ValueError("poll interval must be positive")
    repo = Path(__file__).resolve().parents[1]
    plans_root = repo / "ml/training-plans"
    split_plan = plans_root / SPLIT_PLAN
    split_plan_argument = str(split_plan.relative_to(repo))
    trainer = repo / "ml/train_next_return_memorization.py"
    evaluator = repo / "ml/evaluate_next_return_checkpoint.py"
    matrix_root = repo / "data/training/matrices" / MATRIX_ID
    pause_file = matrix_root / "control/PAUSE"
    wait_for_result = repo / WAIT_FOR_RESULT
    environment = os.environ.copy()
    environment["TRADING_STORAGE_GC_INTERVAL_MINUTES"] = "1"
    environment["TRADING_STORAGE_ORPHAN_GRACE_HOURS"] = "0.001"

    def wait_until_resumed() -> None:
        while pause_file.is_file():
            time.sleep(args.poll_seconds)

    def run_resumable(command: tuple[str, ...], completion_file: Path) -> None:
        if completion_file.is_file():
            return
        attempts = 0
        while not completion_file.is_file():
            wait_until_resumed()
            completed = subprocess.run(command, cwd=repo, env=environment)
            if completed.returncode == PAUSE_EXIT_CODE:
                wait_until_resumed()
                continue
            if completed.returncode == 0 and completion_file.is_file():
                return
            attempts += 1
            if attempts >= 3:
                completed.check_returncode()
                raise RuntimeError(
                    f"command completed without durable output: {completion_file}"
                )
            time.sleep(args.poll_seconds)

    # The user requested that the already-active historical run finish, while
    # preventing the old interleaved runner from launching another job.
    while not wait_for_result.is_file():
        wait_until_resumed()
        time.sleep(args.poll_seconds)

    validation_arguments = (
        "--validation-curve-split-plan",
        split_plan_argument,
        "--validation-curve-examples",
        str(VALIDATION_EXAMPLES),
    )

    # Phase 1: train every reusable ERM reference before starting any sweep.
    for base_plan_name in BASE_PLANS:
        source_plan_file = plans_root / base_plan_name
        source_plan = json.loads(source_plan_file.read_text(encoding="utf-8"))
        run_root = repo / f'{source_plan["runDir"]}-{BASELINE_SUFFIX}'
        snapshot = run_root / "state/plan.json"
        run_resumable((
            sys.executable,
            str(trainer),
            "--plan", str(source_plan_file),
            "--exclude-zero-targets",
            "--variant-suffix", BASELINE_SUFFIX,
            *validation_arguments,
            "--pause-file", str(pause_file),
        ), run_root / "state/result.json")
        # A completed run is not complete as an experiment until its held-out
        # metrics and standard output calibration are durable and visible.
        run_resumable((
            sys.executable,
            str(evaluator),
            "--training-plan", str(snapshot),
            "--split-plan", str(split_plan),
            "--split", "validation",
            "--batch-size", "8192",
        ), run_root / "state/output-calibration-pre-validation-7d.json")

    # Phase 2: the requested 4 x 6 IRMv1 grid on 65,536 clean examples.
    source_plan_file = plans_root / SWEEP_BASE_PLAN
    source_plan = json.loads(source_plan_file.read_text(encoding="utf-8"))
    reference_root = repo / f'{source_plan["runDir"]}-{BASELINE_SUFFIX}'
    reference_snapshot = reference_root / "state/plan.json"
    reference_argument = str(reference_snapshot.relative_to(repo))
    total_steps = total_optimizer_steps(source_plan)
    schedule_manifest = {
        "totalOptimizerSteps": total_steps,
        "stepsPerEpoch": math.ceil(
            int(source_plan["subset"]["examples"])
            / int(source_plan["training"]["batchSize"])
        ),
        "cells": [],
    }
    penalty_one_run_id = (
        f'{source_plan["id"]}-{sweep_suffix(0.0, 1.0)}'
    )
    for fraction in ANNEALING_FRACTIONS:
        anneal_steps = int(round(fraction * total_steps))
        for penalty_weight in PENALTY_WEIGHTS:
            suffix = sweep_suffix(fraction, penalty_weight)
            run_root = repo / f'{source_plan["runDir"]}-{suffix}'
            cell = {
                "annealingFraction": fraction,
                "annealingSteps": anneal_steps,
                "penaltyWeight": penalty_weight,
                "runId": f'{source_plan["id"]}-{suffix}',
            }
            # Before the boundary IRMv1 uses weight 1. If the selected weight
            # is also 1, every annealing fraction has exactly the same
            # objective. Train it once and reuse it in all four matrix cells.
            if penalty_weight == 1.0 and fraction != 0.0:
                cell["runId"] = penalty_one_run_id
                cell["reusedFrom"] = {
                    "annealingFraction": 0.0,
                    "reason": "penalty weight is 1 on both sides of boundary",
                }
                schedule_manifest["cells"].append(cell)
                continue
            schedule_manifest["cells"].append(cell)
            run_resumable((
                sys.executable,
                str(trainer),
                "--plan", str(source_plan_file),
                "--eiil-reference-plan", reference_argument,
                "--irm-penalty-weight", str(penalty_weight),
                "--irm-penalty-anneal-steps", str(anneal_steps),
                "--variant-suffix", suffix,
                *validation_arguments,
                "--pause-file", str(pause_file),
            ), run_root / "state/result.json")
            run_resumable((
                sys.executable,
                str(evaluator),
                "--training-plan", str(run_root / "state/plan.json"),
                "--split-plan", str(split_plan),
                "--split", "validation",
                "--batch-size", "8192",
            ), run_root / "state/output-calibration-pre-validation-7d.json")

    matrix_root.mkdir(parents=True, exist_ok=True)
    (matrix_root / "state").mkdir(parents=True, exist_ok=True)
    (matrix_root / "state/schedule.json").write_text(
        json.dumps(schedule_manifest, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
