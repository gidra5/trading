from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from multiscale_candle_resolution import allowed_max_windows
from multiscale_next_return import window_path_slug
from train_normalized_glu_next_return import atomic_json, iso_now


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the resumable joint telescoping-MA depth screen at the "
            "aligned candle resolutions selected by the plan."
        )
    )
    parser.add_argument("--plan", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    plan_file = args.plan if args.plan.is_absolute() else repo_root / args.plan
    plan = json.loads(plan_file.read_text(encoding="utf-8"))
    plan_argument = str(plan_file.relative_to(repo_root))
    order = [
        (resolution, max_window)
        for resolution in plan["resolutions"]
        for max_window in allowed_max_windows(resolution)
    ]
    matrix_root = repo_root / "data" / "training" / "runs" / plan["id"]
    status_file = matrix_root / "state" / "matrix-status.json"
    completed: list[str] = []
    for index, (resolution, max_window) in enumerate(order):
        run_dir = repo_root / plan["runDirTemplate"].format(
            resolution=resolution,
            maxWindow=window_path_slug(max_window),
        )
        result_file = run_dir / "state" / "result.json"
        combo = f"{resolution}-{max_window}"
        if result_file.is_file():
            completed.append(combo)
            continue
        atomic_json({
            "stage": "running",
            "updatedAt": iso_now(),
            "current": combo,
            "position": index + 1,
            "total": len(order),
            "completed": completed,
        }, status_file)
        completed_process = subprocess.run([
            sys.executable,
            str(repo_root / "ml" / "train_multiscale_candle_resolution.py"),
            "--plan", plan_argument,
            "--resolution", resolution,
            "--max-window", max_window,
        ], cwd=repo_root / "ml", check=False)
        if completed_process.returncode:
            atomic_json({
                "stage": "failed",
                "updatedAt": iso_now(),
                "current": combo,
                "returnCode": completed_process.returncode,
                "completed": completed,
            }, status_file)
            raise SystemExit(completed_process.returncode)
        if not result_file.is_file():
            raise RuntimeError(f"matrix run did not produce a result: {combo}")
        completed.append(combo)
    atomic_json({
        "stage": "complete",
        "updatedAt": iso_now(),
        "current": None,
        "total": len(order),
        "completed": completed,
    }, status_file)


if __name__ == "__main__":
    main()
