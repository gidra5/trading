from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Wait for the temporal-feature run, then evaluate and calibrate it."
    )
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    return parser.parse_args()


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_step(status_file: Path, stage: str, command: list[str], repo: Path) -> None:
    write_json(status_file, {"stage": stage, "command": command})
    subprocess.run(command, cwd=repo, check=True)


def main() -> None:
    args = parse_args()
    repo = Path(__file__).resolve().parents[1]
    plan_file = (repo / args.plan).resolve() \
        if not args.plan.is_absolute() else args.plan.resolve()
    plan = json.loads(plan_file.read_text("utf-8"))
    run_root = (repo / plan["runDir"]).resolve()
    training_status = run_root / "state/status.json"
    status_file = run_root / "state/postprocess-status.json"
    write_json(status_file, {"stage": "waiting-for-training"})
    try:
        while True:
            if training_status.is_file():
                status = json.loads(training_status.read_text("utf-8"))
                stage = status.get("stage")
                if stage == "complete":
                    break
                if stage in {"failed", "paused"}:
                    raise RuntimeError(f"training ended in stage {stage}")
            time.sleep(args.poll_seconds)
        run_step(status_file, "exporting-calibration", [
            "node", "--conditions=development", "--import", "tsx",
            "scripts/export-next-return-production-basis.ts",
            "--all-feature-history", "--calibration-only",
        ], repo)
        run_step(status_file, "evaluating", [
            sys.executable, "ml/evaluate_feature_augmented_next_return.py",
            "--plan", str(plan_file), "--device", "cuda",
        ], repo)
        run_step(status_file, "calibrating", [
            sys.executable, "ml/calibrate_feature_augmented_next_return.py",
            "--plan", str(plan_file),
            "--calibration-dir",
            "data/training/datasets/next-return-production-basis-history120-calibration-16k-v1",
            "--device", "cuda",
        ], repo)
        write_json(status_file, {"stage": "complete"})
    except Exception as error:
        write_json(status_file, {
            "stage": "failed",
            "error": f"{type(error).__name__}: {error}",
        })
        raise


if __name__ == "__main__":
    main()
