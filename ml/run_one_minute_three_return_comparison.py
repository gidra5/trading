from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import time


PAUSE_EXIT_CODE = 75
PLAN_IDS = (
    "one-minute-active-return-path-direct-glu-65k-h3-v1",
    "one-minute-exact-tensor-density-65k-h3-k32-v1",
)
TRAINERS = (
    "train_active_return_path_glu.py",
    "train_exact_tensor_return_path.py",
)


def run_resumable(
    command: tuple[str, ...],
    pause_file: Path,
    *,
    cwd: Path,
    environment: dict[str, str],
) -> None:
    while True:
        completed = subprocess.run(command, cwd=cwd, env=environment)
        if completed.returncode != PAUSE_EXIT_CODE:
            completed.check_returncode()
            return
        while pause_file.is_file():
            time.sleep(2)


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    matrix_root = repo / "data/training/matrices/one-minute-h3-comparison-v1"
    pause_file = matrix_root / "control/PAUSE"
    pause_file.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["TRADING_STORAGE_GC_INTERVAL_MINUTES"] = "1"
    environment["TRADING_STORAGE_ORPHAN_GRACE_HOURS"] = "0.001"
    subprocess.run((
        sys.executable,
        str(repo / "ml/build_one_minute_return_knots.py"),
    ), cwd=repo, check=True)
    for plan_id, trainer in zip(PLAN_IDS, TRAINERS, strict=True):
        plan_file = repo / f"ml/training-plans/{plan_id}.json"
        result_file = repo / f"data/training/runs/{plan_id}/state/result.json"
        if result_file.is_file():
            continue
        run_resumable((
            sys.executable,
            str(repo / f"ml/{trainer}"),
            "--plan",
            str(plan_file),
            "--pause-file",
            str(pause_file),
        ), pause_file, cwd=repo, environment=environment)


if __name__ == "__main__":
    main()
