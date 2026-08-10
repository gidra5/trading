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
)
VOLATILITY_WINDOWS = (2, 5, 10, 15, 30, 60, 7_200, 14_400)
SPLIT_PLAN = "direct-glu-to-next-1s-recent-4m-1-layer-long-v2.json"
DROPOUT_PREREQUISITE = (
    "data/training/runs/"
    "next-second-8-layer-memorization-1024k-width1024-static-c-v1-"
    "dropout-p05-rate50-v1/state/result.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the causal-volatility normalization memorization matrix."
    )
    parser.add_argument("--poll-seconds", type=float, default=30)
    parser.add_argument(
        "--windows",
        type=int,
        nargs="+",
        default=VOLATILITY_WINDOWS,
        help="Causal RMS windows in seconds (default: full matrix).",
    )
    parser.add_argument(
        "--prerequisite",
        type=Path,
        default=Path(DROPOUT_PREREQUISITE),
        help="Repository-relative result that must exist before this runner starts.",
    )
    return parser.parse_args()


def variant_id(base_plan: dict, window: int) -> str:
    return f'{base_plan["id"]}-causal-volatility-w{window}-v1'


def main() -> None:
    args = parse_args()
    if args.poll_seconds <= 0:
        raise ValueError("poll interval must be positive")
    windows = tuple(dict.fromkeys(int(value) for value in args.windows))
    if not windows or any(not 1 <= value <= 86_400 for value in windows):
        raise ValueError("volatility windows must be in [1, 86,400]")
    repo = Path(__file__).resolve().parents[1]
    plans_root = repo / "ml/training-plans"
    trainer = repo / "ml/train_next_return_memorization.py"
    evaluator = repo / "ml/evaluate_next_return_checkpoint.py"
    split_plan = plans_root / SPLIT_PLAN

    prerequisite = (
        args.prerequisite.resolve()
        if args.prerequisite.is_absolute()
        else (repo / args.prerequisite).resolve()
    )
    while not prerequisite.is_file():
        time.sleep(args.poll_seconds)

    environment = os.environ.copy()
    environment["TRADING_STORAGE_GC_INTERVAL_MINUTES"] = "1"
    environment["TRADING_STORAGE_ORPHAN_GRACE_HOURS"] = "0.001"
    for name in BASE_PLANS:
        source_plan_file = plans_root / name
        source_plan = json.loads(source_plan_file.read_text(encoding="utf-8"))
        for window in windows:
            suffix = f"causal-volatility-w{window}-v1"
            run_root = repo / "data/training/runs" / variant_id(
                source_plan, window
            )
            training_plan = run_root / "state/plan.json"
            commands = (
                (
                    sys.executable,
                    str(trainer),
                    "--plan",
                    str(source_plan_file),
                    "--causal-volatility-window",
                    str(window),
                    "--variant-suffix",
                    suffix,
                ),
                (
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
