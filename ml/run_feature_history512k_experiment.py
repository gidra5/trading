from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

from train_feature_augmented_next_return import FeatureMatrixDataset
from trading_storage import _prune_checkpoint_orphans


REFERENCE_RUN = "next-return-production-basis-history120-4l-65k-v1"
DEFAULT_PLAN = (
    "ml/training-plans/"
    "next-return-production-basis-history120-4l-train512k-v1.json"
)
CALIBRATION = (
    "data/training/datasets/"
    "next-return-production-basis-history120-calibration-16k-v1"
)
MINIMUM_PIPELINE_FREE_BYTES = 3 * 1024**3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export, train, evaluate, and calibrate a temporal-feature run."
    )
    parser.add_argument("--plan", default=DEFAULT_PLAN)
    parser.add_argument("--train-examples", type=int, default=512_000)
    parser.add_argument("--validation-examples", type=int, default=65_536)
    parser.add_argument("--test-examples", type=int, default=65_536)
    parser.add_argument(
        "--reuse-existing-dataset",
        action="store_true",
        help="Skip export when an exact compact dataset is already present.",
    )
    return parser.parse_args()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def wait_for_reference(repo: Path, status_file: Path) -> None:
    source = (
        repo / "data/training/runs" / REFERENCE_RUN
        / "state/postprocess-status.json"
    )
    while True:
        if source.is_file():
            status = json.loads(source.read_text("utf-8"))
            stage = status.get("stage")
            if stage == "complete":
                return
            if stage == "failed":
                raise RuntimeError("the 65k reference postprocessing failed")
        write_json(status_file, {"stage": "waiting-for-65k-reference"})
        time.sleep(5)


def wait_for_storage(repo: Path, status_file: Path, next_stage: str) -> None:
    store_root = (repo / "data/training/immutable").resolve()
    while True:
        _prune_checkpoint_orphans(store_root)
        free = shutil.disk_usage(repo).free
        if free >= MINIMUM_PIPELINE_FREE_BYTES:
            return
        write_json(status_file, {
            "stage": "waiting-for-storage",
            "nextStage": next_stage,
            "freeBytes": free,
            "requiredFreeBytes": MINIMUM_PIPELINE_FREE_BYTES,
            "message": "The run will resume automatically when storage is available.",
        })
        time.sleep(10)


def run_step(
    repo: Path, status_file: Path, stage: str, command: list[str],
    *, environment: dict[str, str] | None = None,
) -> None:
    write_json(status_file, {"stage": stage, "command": command})
    subprocess.run(
        command, cwd=repo, check=True,
        env={**os.environ, **(environment or {})},
    )


def verify_compact_equivalence(
    repo: Path,
    status_file: Path,
    compact_root: Path,
) -> None:
    write_json(status_file, {"stage": "verifying-compact-equivalence"})
    reference = FeatureMatrixDataset(
        repo / "data/training/datasets/"
        "next-return-production-basis-history120-4l-65k-v1"
    )
    compact = FeatureMatrixDataset(compact_root)
    if reference.manifest["features"] != compact.manifest["features"]:
        raise ValueError("compact and flattened feature definitions differ")
    for split in ("train", "validation", "test"):
        expected = reference.splits[split]
        actual = compact.splits[split]
        count = expected.count
        if not np.array_equal(expected.times, actual.times[:count]) \
                or not np.array_equal(expected.targets, actual.targets[:count]):
            raise ValueError(f"compact {split} targets or origins differ")
        for start in range(0, count, 512):
            stop = min(count, start + 512)
            if not np.array_equal(
                np.asarray(expected.features[start:stop]),
                np.asarray(actual.features[start:stop]),
            ):
                raise ValueError(
                    f"compact {split} feature windows differ at row {start}"
                )


def main() -> None:
    args = parse_args()
    repo = Path(__file__).resolve().parents[1]
    plan_file = repo / args.plan
    plan = json.loads(plan_file.read_text("utf-8"))
    dataset_root = repo / plan["datasetDir"]
    run_root = repo / plan["runDir"]
    status_file = run_root / "state/pipeline-status.json"
    try:
        wait_for_reference(repo, status_file)
        if not args.reuse_existing_dataset:
            wait_for_storage(repo, status_file, "exporting-compact-dataset")
            run_step(repo, status_file, "exporting-compact-dataset", [
                "node", "--conditions=development", "--import", "tsx",
                "scripts/export-next-return-production-basis.ts",
                "--all-feature-history", "--compact-temporal",
                "--train-examples", str(args.train_examples),
                "--validation-examples", str(args.validation_examples),
                "--test-examples", str(args.test_examples),
                "--output-dir", str(dataset_root.relative_to(repo)),
            ])
        elif not (dataset_root / "manifest.json").is_file():
            raise FileNotFoundError(
                f"reused compact dataset is missing: {dataset_root}"
            )
        verify_compact_equivalence(repo, status_file, dataset_root)
        wait_for_storage(repo, status_file, "training")
        run_step(repo, status_file, "training", [
            sys.executable, "ml/train_feature_augmented_next_return.py",
            "--plan", args.plan,
        ], environment={
            "PYTHONPATH": "ml",
            "TRADING_STORAGE_GC_INTERVAL_MINUTES": "0.05",
            "TRADING_STORAGE_ORPHAN_GRACE_HOURS": "0.01",
        })
        run_step(repo, status_file, "evaluating", [
            sys.executable, "ml/evaluate_feature_augmented_next_return.py",
            "--plan", args.plan, "--device", "cuda",
        ], environment={"PYTHONPATH": "ml"})
        run_step(repo, status_file, "calibrating", [
            sys.executable, "ml/calibrate_feature_augmented_next_return.py",
            "--plan", args.plan, "--calibration-dir", CALIBRATION,
            "--device", "cuda",
        ], environment={"PYTHONPATH": "ml"})
        write_json(status_file, {"stage": "complete"})
    except Exception as error:
        write_json(status_file, {
            "stage": "failed",
            "error": f"{type(error).__name__}: {error}",
        })
        raise


if __name__ == "__main__":
    main()
