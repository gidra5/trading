from __future__ import annotations

import argparse
import json
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
SPLIT_PLAN = "direct-glu-to-next-1s-recent-4m-1-layer-long-v2.json"
MATRIX_ID = "nonzero-eiil-irm-scaling-v1"
REFERENCE_SUFFIX = "nonzero-clean-count-v2"
EIIL_SUFFIX = "nonzero-eiil-irmv1-v1"
PAUSE_EXIT_CODE = 75


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare clean-count ERM references with EIIL-inferred binary "
            "environments and canonical IRMv1 across six scaling cases."
        )
    )
    parser.add_argument("--poll-seconds", type=float, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.poll_seconds <= 0:
        raise ValueError("poll interval must be positive")
    repo = Path(__file__).resolve().parents[1]
    plans_root = repo / "ml/training-plans"
    split_plan = plans_root / SPLIT_PLAN
    trainer = repo / "ml/train_next_return_memorization.py"
    evaluator = repo / "ml/evaluate_next_return_checkpoint.py"
    pause_file = repo / "data/training/matrices" / MATRIX_ID / "control/PAUSE"
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

    for base_plan_name in BASE_PLANS:
        source_plan_file = plans_root / base_plan_name
        source_plan = json.loads(source_plan_file.read_text(encoding="utf-8"))
        reference_root = repo / f'{source_plan["runDir"]}-{REFERENCE_SUFFIX}'
        reference_snapshot = reference_root / "state/plan.json"
        reference_plan_argument = str(reference_snapshot.relative_to(repo))
        eiil_root = repo / f'{source_plan["runDir"]}-{EIIL_SUFFIX}'
        commands = (
            ((
                sys.executable,
                str(trainer),
                "--plan",
                str(source_plan_file),
                "--exclude-zero-targets",
                "--variant-suffix",
                REFERENCE_SUFFIX,
                "--pause-file",
                str(pause_file),
            ), reference_root / "state/result.json"),
            ((
                sys.executable,
                str(evaluator),
                "--training-plan",
                str(reference_snapshot),
                "--split-plan",
                str(split_plan),
                "--split",
                "validation",
                "--batch-size",
                "8192",
            ), reference_root / "state/output-calibration-pre-validation-7d.json"),
            ((
                sys.executable,
                str(trainer),
                "--plan",
                str(source_plan_file),
                "--eiil-reference-plan",
                reference_plan_argument,
                "--variant-suffix",
                EIIL_SUFFIX,
                "--pause-file",
                str(pause_file),
            ), eiil_root / "state/result.json"),
            ((
                sys.executable,
                str(evaluator),
                "--training-plan",
                str(eiil_root / "state/plan.json"),
                "--split-plan",
                str(split_plan),
                "--split",
                "validation",
                "--batch-size",
                "8192",
            ), eiil_root / "state/output-calibration-pre-validation-7d.json"),
        )
        for command, completion_file in commands:
            run_resumable(command, completion_file)


if __name__ == "__main__":
    main()
