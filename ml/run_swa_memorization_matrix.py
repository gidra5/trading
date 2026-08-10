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
VARIANT_SUFFIX = "swa-sweep-v1"
PREREQUISITE = (
    "data/training/runs/"
    "next-second-4-layer-memorization-256k-static-c-v1-"
    "causal-volatility-w14400-v1/state/result.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SWA sweeps over the existing uniform diagnostics."
    )
    parser.add_argument("--poll-seconds", type=float, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.poll_seconds <= 0:
        raise ValueError("poll interval must be positive")
    repo = Path(__file__).resolve().parents[1]
    plans_root = repo / "ml/training-plans"
    trainer = repo / "ml/train_next_return_memorization.py"
    evaluator = repo / "ml/evaluate_next_return_checkpoint.py"
    split_plan = plans_root / SPLIT_PLAN
    prerequisite = repo / PREREQUISITE
    while not prerequisite.is_file():
        time.sleep(args.poll_seconds)

    environment = os.environ.copy()
    environment["TRADING_STORAGE_GC_INTERVAL_MINUTES"] = "1"
    environment["TRADING_STORAGE_ORPHAN_GRACE_HOURS"] = "0.001"
    for name in BASE_PLANS:
        source_plan_file = plans_root / name
        source_plan = json.loads(source_plan_file.read_text(encoding="utf-8"))
        run_root = repo / f'{source_plan["runDir"]}-{VARIANT_SUFFIX}'
        training_plan = run_root / "state/plan.json"
        sweep_checkpoint = run_root / "checkpoints/swa-sweep.json"
        commands = (
            (
                sys.executable,
                str(trainer),
                "--plan",
                str(source_plan_file),
                "--swa-sweep",
                "--variant-suffix",
                VARIANT_SUFFIX,
            ),
            (
                sys.executable,
                str(evaluator),
                "--training-plan",
                str(training_plan),
                "--split-plan",
                str(split_plan),
                "--checkpoint",
                str(sweep_checkpoint),
                "--split",
                "validation",
                "--batch-size",
                "8192",
            ),
        )
        for attempt in range(1, 4):
            for command in commands:
                completed = subprocess.run(
                    command, cwd=repo, env=environment
                )
                if completed.returncode != 0:
                    break
            if completed.returncode == 0:
                break
            if attempt == 3:
                completed.check_returncode()
            time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
