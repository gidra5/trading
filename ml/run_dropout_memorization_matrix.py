from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


BASE_PLANS = (
    "next-second-4-layer-memorization-65k-v1.json",
    "next-second-4-layer-memorization-65k-static-c-v1.json",
    "next-second-4-layer-memorization-128k-static-c-v1.json",
    "next-second-4-layer-memorization-256k-static-c-v1.json",
    "next-second-4-layer-memorization-512k-static-c-v1.json",
    "next-second-8-layer-memorization-512k-static-c-v1.json",
    "next-second-8-layer-memorization-512k-static-c-quadrants-v1.json",
    "next-second-8-layer-memorization-1024k-width1024-static-c-v1.json",
)
VARIANT_SUFFIX = "dropout-p05-rate50-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the paired 5%-dropout memorization diagnostic matrix."
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

    # The largest pure run was still active when this paired matrix was queued.
    # Its completion gates the matrix so two trainers never contend for the GPU.
    largest = json.loads(
        (plans_root / BASE_PLANS[-1]).read_text(encoding="utf-8")
    )
    prerequisite = repo / largest["runDir"] / "state/result.json"
    while not prerequisite.is_file():
        time.sleep(args.poll_seconds)

    for name in BASE_PLANS:
        command = (
            sys.executable,
            str(trainer),
            "--plan",
            str(plans_root / name),
            "--dropout",
            "0.05",
            "--dropout-rate",
            "0.5",
            "--variant-suffix",
            VARIANT_SUFFIX,
        )
        environment = os.environ.copy()
        # Large checkpoints otherwise accumulate for the default one-hour
        # orphan grace window and can exhaust the experiment drive. References
        # to best and last checkpoints remain protected by the storage GC.
        environment["TRADING_STORAGE_GC_INTERVAL_MINUTES"] = "1"
        environment["TRADING_STORAGE_ORPHAN_GRACE_HOURS"] = "0.001"
        for attempt in range(1, 4):
            completed = subprocess.run(command, cwd=repo, env=environment)
            if completed.returncode == 0:
                break
            if attempt == 3:
                completed.check_returncode()
            time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
