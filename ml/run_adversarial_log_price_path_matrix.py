from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


BASE_PLAN = "next-second-4-layer-memorization-65k-static-c-v1.json"
SPLIT_PLAN = "direct-glu-to-next-1s-recent-4m-1-layer-long-v2.json"
MATRIX_ID = "adversarial-log-price-path-screen-v1"
EPSILON_RMS_VALUES = (
    (0.01, "1e-2"),
    (0.03, "3e-2"),
    (0.1, "1e-1"),
    (0.3, "3e-1"),
    (1.0, "1e0"),
)
ATTACK_STEPS = 1
ADVERSARIAL_WEIGHT = 0.5
PAUSE_EXIT_CODE = 75


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Screen one-step joint adversarial log-price-path radii on the "
            "fixed 65k next-second diagnostic."
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
    source_plan_file = plans_root / BASE_PLAN
    source_plan = json.loads(source_plan_file.read_text(encoding="utf-8"))
    split_plan = plans_root / SPLIT_PLAN
    trainer = repo / "ml/train_next_return_memorization.py"
    evaluator = repo / "ml/evaluate_next_return_checkpoint.py"
    pause_file = (
        repo / "data/training/matrices" / MATRIX_ID / "control/PAUSE"
    )
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

    for epsilon_rms, epsilon_label in EPSILON_RMS_VALUES:
        suffix = f"adversarial-log-price-eps-{epsilon_label}-v1"
        run_root = repo / f'{source_plan["runDir"]}-{suffix}'
        training_plan = run_root / "state/plan.json"
        commands = (
            ((
                sys.executable,
                str(trainer),
                "--plan",
                str(source_plan_file),
                "--adversarial-log-price-epsilon-rms",
                str(epsilon_rms),
                "--adversarial-log-price-steps",
                str(ATTACK_STEPS),
                "--adversarial-log-price-weight",
                str(ADVERSARIAL_WEIGHT),
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
