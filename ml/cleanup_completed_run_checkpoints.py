from __future__ import annotations

import argparse
import json
from pathlib import Path

from trading_storage import require_under, training_storage_layout


POINTERS = (
    "checkpoints/last.json",
    "checkpoints/selections/validation-nll.json",
    "checkpoints/selections/validation-mse.json",
    "checkpoints/selections/validation-correlation.json",
    "checkpoints/selections/validation-crps.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Delete a completed run's checkpoint pointers and only the immutable "
            "checkpoint objects that become unreferenced."
        )
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    return parser.parse_args()


def artifact_object(pointer: Path, immutable_root: Path) -> Path:
    value = json.loads(pointer.read_text(encoding="utf-8"))
    if value.get("kind") != "trading-immutable-artifact":
        raise ValueError(f"invalid checkpoint artifact pointer: {pointer}")
    relative = value.get("object", {}).get("file")
    if not isinstance(relative, str) or not relative.endswith(".bin"):
        raise ValueError(f"invalid checkpoint object path: {pointer}")
    return require_under(
        immutable_root / relative, immutable_root, "checkpoint object"
    )


def main() -> None:
    args = parse_args()
    repo = Path(__file__).resolve().parents[1]
    layout = training_storage_layout(repo)
    requested = args.run_dir if args.run_dir.is_absolute() else repo / args.run_dir
    run_root = require_under(requested, layout.runs, "run directory")
    if not (run_root / "state/stopped-evaluation.json").is_file() \
            or not (run_root / "state/result.json").is_file() \
            or not (run_root / "logs/training.jsonl").is_file():
        raise ValueError(
            "checkpoint cleanup requires preserved evaluation, result, and training log"
        )

    pointer_files = tuple(run_root / relative for relative in POINTERS)
    candidates = {
        artifact_object(pointer, layout.immutable)
        for pointer in pointer_files
        if pointer.is_file()
    }
    for pointer in pointer_files:
        pointer.unlink(missing_ok=True)

    referenced: set[Path] = set()
    for pointer in layout.runs.glob("**/checkpoints/**/*.json"):
        try:
            referenced.add(artifact_object(pointer, layout.immutable))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
            continue
    removed = []
    for candidate in sorted(candidates):
        if candidate not in referenced and candidate.is_file():
            candidate.unlink()
            removed.append(candidate)

    print(json.dumps({
        "runDir": str(run_root.relative_to(repo)),
        "checkpointPointersRemoved": sum(
            not pointer.exists() for pointer in pointer_files
        ),
        "candidateObjects": len(candidates),
        "unreferencedObjectsRemoved": len(removed),
        "trainingLogPreserved": (run_root / "logs/training.jsonl").is_file(),
        "stoppedEvaluationPreserved": (
            run_root / "state/stopped-evaluation.json"
        ).is_file(),
        "resultPreserved": (run_root / "state/result.json").is_file(),
    }))


if __name__ == "__main__":
    main()
