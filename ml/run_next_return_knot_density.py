from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import time


MATRIX_ID = "next-return-knot-density-v1"
PLAN = "ml/training-plans/next-second-4-layer-knot-density-65k-v1.json"
PAUSE_EXIT_CODE = 75


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    pause_file = repo / f"data/training/matrices/{MATRIX_ID}/control/PAUSE"
    result_file = repo / (
        "data/training/runs/next-second-4-layer-knot-density-65k-v1/"
        "state/result.json"
    )
    command = (
        sys.executable,
        str(repo / "ml/train_next_return_knot_density.py"),
        "--plan",
        PLAN,
        "--pause-file",
        str(pause_file),
    )
    environment = os.environ.copy()
    environment["TRADING_STORAGE_GC_INTERVAL_MINUTES"] = "1"
    environment["TRADING_STORAGE_ORPHAN_GRACE_HOURS"] = "0.001"
    while not result_file.is_file():
        while pause_file.is_file():
            time.sleep(2)
        completed = subprocess.run(command, cwd=repo, env=environment)
        if completed.returncode == PAUSE_EXIT_CODE:
            continue
        completed.check_returncode()
        if not result_file.is_file():
            raise RuntimeError("density trainer exited without a durable result")


if __name__ == "__main__":
    main()
