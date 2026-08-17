from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


SMALL_BASE_PLANS = (
    "next-second-4-layer-memorization-65k-static-c-v1.json",
    "next-second-4-layer-memorization-128k-static-c-v1.json",
    "next-second-4-layer-memorization-256k-static-c-v1.json",
)
LARGE_BASE_PLANS = (
    "next-second-4-layer-memorization-512k-static-c-v1.json",
    "next-second-8-layer-memorization-512k-static-c-v1.json",
    "next-second-8-layer-memorization-1024k-width1024-static-c-v1.json",
)
WEAK_RATES = (
    (1e-6, "1e-6"),
    (1e-5, "1e-5"),
    (1e-4, "1e-4"),
    (1e-3, "1e-3"),
)
STRONG_RATES = (
    (0.003, "3e-3"),
    (0.01, "1e-2"),
    (0.03, "3e-2"),
    (0.1, "1e-1"),
)
HIGH_RATES = (
    (0.3, "3e-1"),
    (1.0, "1e0"),
)
MODES = (
    ("l2", "--l2-rate"),
    ("optimizer-wd", "--optimizer-weight-decay"),
)
SPLIT_PLAN = "direct-glu-to-next-1s-recent-4m-1-layer-long-v2.json"
PREREQUISITE = (
    "data/training/runs/"
    "next-second-8-layer-memorization-1024k-width1024-static-c-v1-"
    "swa-sweep-v1/state/result.json"
)
STRONG_PREREQUISITE = (
    "data/training/runs/"
    "next-second-4-layer-memorization-256k-static-c-v1-"
    "optimizer-wd-rate-1e-3-v1/state/validation-current-best.json"
)
EXPANDED_PREREQUISITE = (
    "data/training/runs/"
    "next-second-4-layer-memorization-256k-static-c-v1-"
    "optimizer-wd-rate-1e-1-v1/state/validation-current-best.json"
)
PAUSE_EXIT_CODE = 75


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare explicit L2 with optimizer-native weight decay."
    )
    parser.add_argument("--poll-seconds", type=float, default=2)
    parser.add_argument(
        "--strong-optimizer-only",
        action="store_true",
        help="Run only optimizer-native rates 0.003 through 0.1.",
    )
    parser.add_argument(
        "--expanded-optimizer",
        action="store_true",
        help="Add high rates on small data and all strong/high rates on large data.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.poll_seconds <= 0:
        raise ValueError("poll interval must be positive")
    if args.strong_optimizer_only and args.expanded_optimizer:
        raise ValueError("optimizer sweep profiles are mutually exclusive")
    repo = Path(__file__).resolve().parents[1]
    plans_root = repo / "ml/training-plans"
    trainer = repo / "ml/train_next_return_memorization.py"
    evaluator = repo / "ml/evaluate_next_return_checkpoint.py"
    split_plan = plans_root / SPLIT_PLAN
    modes = MODES if not (
        args.strong_optimizer_only or args.expanded_optimizer
    ) else (
        ("optimizer-wd", "--optimizer-weight-decay"),
    )
    if args.expanded_optimizer:
        plan_groups = (
            (SMALL_BASE_PLANS, HIGH_RATES),
            (LARGE_BASE_PLANS, (*STRONG_RATES, *HIGH_RATES)),
        )
        prerequisite_value = EXPANDED_PREREQUISITE
        matrix_id = "optimizer-weight-decay-expanded-v1"
    elif args.strong_optimizer_only:
        plan_groups = ((SMALL_BASE_PLANS, STRONG_RATES),)
        prerequisite_value = STRONG_PREREQUISITE
        matrix_id = "optimizer-weight-decay-strong-v1"
    else:
        plan_groups = ((SMALL_BASE_PLANS, WEAK_RATES),)
        prerequisite_value = PREREQUISITE
        matrix_id = "weight-decay-memorization-v1"
    prerequisite = repo / prerequisite_value
    while not prerequisite.is_file():
        time.sleep(args.poll_seconds)

    environment = os.environ.copy()
    environment["TRADING_STORAGE_GC_INTERVAL_MINUTES"] = "1"
    environment["TRADING_STORAGE_ORPHAN_GRACE_HOURS"] = "0.001"
    pause_file = (
        repo / "data/training/matrices" / matrix_id / "control/PAUSE"
    )

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
                    f"command completed without its durable output: {completion_file}"
                )
            time.sleep(args.poll_seconds)

    for base_plans, rates in plan_groups:
        for name in base_plans:
            source_plan_file = plans_root / name
            source_plan = json.loads(source_plan_file.read_text(encoding="utf-8"))
            for rate, rate_label in rates:
                for mode, option in modes:
                    suffix = f"{mode}-rate-{rate_label}-v1"
                    run_root = repo / f'{source_plan["runDir"]}-{suffix}'
                    training_plan = run_root / "state/plan.json"
                    commands = (
                        ((
                            sys.executable,
                            str(trainer),
                            "--plan",
                            str(source_plan_file),
                            option,
                            str(rate),
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
