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
MATRIX_ID = "nonzero-daily-cvar-scaling-v1"
VARIANTS = (
    (0.2, "nonzero-daily-cvar-20pct-v1"),
    (0.3, "nonzero-daily-cvar-30pct-v1"),
    (0.5, "nonzero-daily-cvar-50pct-v1"),
    (None, "nonzero-clean-count-v2"),
)
PAUSE_EXIT_CODE = 75


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run cleaned ERM and daily CVaR-DRO at three tail fractions over "
            "the six standard architecture/data scaling cases."
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

    for base_plan in BASE_PLANS:
        source_plan_file = plans_root / base_plan
        source_plan = json.loads(source_plan_file.read_text(encoding="utf-8"))
        for tail_fraction, suffix in VARIANTS:
            run_root = repo / f'{source_plan["runDir"]}-{suffix}'
            training_plan = run_root / "state/plan.json"
            variant_arguments = (
                ("--exclude-zero-targets",)
                if tail_fraction is None
                else ("--cvar-tail-fraction", str(tail_fraction))
            )
            commands = (
                ((
                    sys.executable,
                    str(trainer),
                    "--plan",
                    str(source_plan_file),
                    *variant_arguments,
                    "--variant-suffix",
                    suffix,
                    "--pause-file",
                    str(pause_file),
                ), run_root / "state/result.json"),
                ((
                    sys.executable,
                    str(evaluator),
                    "--training-plan",
                    str(training_plan),
                    "--split-plan",
                    str(split_plan),
                    "--split",
                    "validation",
                    "--batch-size",
                    "8192",
                ), run_root / "state/output-calibration-pre-validation-7d.json"),
            )
            for command, completion_file in commands:
                run_resumable(command, completion_file)


if __name__ == "__main__":
    main()
