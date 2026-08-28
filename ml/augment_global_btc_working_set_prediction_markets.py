from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from global_feature_registry import ROOT


DEFAULT_BASE = ROOT / "data/runtime-cache/global-btc-expanded-working-set-v4"
DEFAULT_AXIS = ROOT / "data/runtime-cache/prediction-market-candidate-axis-30d"
DEFAULT_OUTPUT = ROOT / "data/runtime-cache/global-btc-prediction-market-working-set-v1"


def resolved(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description="Append causal prediction-market candidates to the certified BTC working set.")
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--axis", type=Path, default=DEFAULT_AXIS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    base_dir, axis_dir, output_dir = map(resolved, (args.base, args.axis, args.output))
    base = json.loads((base_dir / "manifest.json").read_text(encoding="utf-8"))
    axis = json.loads((axis_dir / "manifest.json").read_text(encoding="utf-8"))
    if int(base["rows"]) != int(axis["rows"]):
        raise ValueError("Prediction-market and global working-set rows differ")
    base_ids = [str(row["id"]) for row in base["coordinates"]]
    base_set = set(base_ids)
    prediction_rows = [row for row in axis["coordinates"] if str(row["id"]) not in base_set]
    prediction_indices = [
        index for index, row in enumerate(axis["coordinates"])
        if str(row["id"]) not in base_set
    ]
    rows = int(base["rows"])
    columns = int(base["columns"]) + len(prediction_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    partial = output_dir / "working-set.raw.f32.partial"
    final = output_dir / "working-set.raw.f32"
    target = np.memmap(partial, dtype="<f4", mode="w+", shape=(rows, columns))
    base_values = np.memmap(
        base_dir / base["file"], dtype=base["dtype"], mode="r",
        shape=(rows, int(base["columns"])),
    )
    prediction_values = np.memmap(
        axis_dir / axis["file"], dtype=axis["dtype"], mode="r",
        shape=(rows, int(axis["columns"])),
    )
    for start in range(0, rows, 4_096):
        end = min(rows, start + 4_096)
        target[start:end, : int(base["columns"])] = base_values[start:end]
        target[start:end, int(base["columns"]):] = prediction_values[start:end, prediction_indices]
    target.flush()
    del target, base_values, prediction_values
    partial.replace(final)
    manifest = {
        "version": 1,
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "purpose": "Certified global working set augmented with all causal prediction-market summaries",
        "baseWorkingSet": str(base_dir.relative_to(ROOT)).replace("\\", "/"),
        "baseWorkingSetSha256": hashlib.sha256((base_dir / "manifest.json").read_bytes()).hexdigest(),
        "predictionMarketAxis": str(axis_dir.relative_to(ROOT)).replace("\\", "/"),
        "predictionMarketAxisSha256": hashlib.sha256((axis_dir / "manifest.json").read_bytes()).hexdigest(),
        "rows": rows,
        "columns": columns,
        "file": final.name,
        "dtype": "<f4",
        "baseCoordinatesRetained": int(base["columns"]),
        "newCoordinates": len(prediction_rows),
        "newCoordinatesBySource": {"prediction-market": len(prediction_rows)},
        "coordinates": [dict(row) for row in base["coordinates"]] + [
            {**row, "source": "prediction-market"} for row in prediction_rows
        ],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: manifest[key] for key in ("rows", "columns", "baseCoordinatesRetained", "newCoordinates")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
