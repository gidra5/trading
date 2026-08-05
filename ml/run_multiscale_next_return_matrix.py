from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from multiscale_next_return import MAX_WINDOW_LABELS, window_path_slug
from train_normalized_glu_next_return import atomic_json, iso_now


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the resumable telescoping moving-average GLU matrix."
    )
    parser.add_argument("--plan", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    plan_file = args.plan if args.plan.is_absolute() else repo_root / args.plan
    plan = json.loads(plan_file.read_text(encoding="utf-8"))
    plan_argument = str(plan_file.relative_to(repo_root))
    joint = [(window, "joint-input") for window in MAX_WINDOW_LABELS]
    separate = [
        (window, "separate-components") for window in MAX_WINDOW_LABELS
    ]
    order = [separate[0], *joint, *separate[1:]]
    matrix_root = repo_root / "data" / "training" / "runs" / plan["id"]
    status_file = matrix_root / "state" / "matrix-status.json"
    completed: list[str] = []
    for index, (max_window, variant) in enumerate(order):
        run_dir = repo_root / plan["runDirTemplate"].format(
            maxWindow=window_path_slug(max_window), variant=variant
        )
        result_file = run_dir / "state" / "result.json"
        combo = f"{max_window}-{variant}"
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
        command = [
            sys.executable,
            str(repo_root / "ml" / "train_multiscale_next_return.py"),
            "--plan",
            plan_argument,
            "--max-window",
            max_window,
            "--variant",
            variant,
        ]
        completed_process = subprocess.run(
            command, cwd=repo_root / "ml", check=False
        )
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
