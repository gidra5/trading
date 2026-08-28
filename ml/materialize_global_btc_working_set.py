from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from global_feature_candidate_axis import BASE_DIR, BaseRecentBatchProvider, DenseMinuteBatchProvider
from global_feature_registry import ROOT


DEFAULT_SCREEN = ROOT / "data/benchmarks/global-btc-dense-kkt-screen.json"
DEFAULT_OUTPUT = ROOT / "data/runtime-cache/global-btc-working-set"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize a reusable raw active working set.")
    parser.add_argument("--screen", type=Path, default=DEFAULT_SCREEN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--retain", type=int, default=4_096)
    parser.add_argument("--batch-size", type=int, default=512)
    return parser.parse_args()


def resolved(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    args = parse_args()
    screen_path = resolved(args.screen)
    output = resolved(args.output)
    screen = json.loads(screen_path.read_text(encoding="utf-8"))
    dense_ids = [
        str(row["canonicalId"])
        for row in screen["topGradientGroups"][: args.retain]
    ]
    if len(dense_ids) != args.retain or len(set(dense_ids)) != len(dense_ids):
        raise ValueError("The requested dense working set is incomplete or contains duplicate IDs.")

    base_provider = BaseRecentBatchProvider()
    base_batch = next(base_provider.raw_batches())
    base_ids = [group.id for group in base_batch.groups]
    ordered_ids = base_ids + [feature_id for feature_id in dense_ids if feature_id not in set(base_ids)]
    rows = int(base_batch.values.shape[0])
    columns = len(ordered_ids)
    index_by_id = {feature_id: index for index, feature_id in enumerate(ordered_ids)}
    output.mkdir(parents=True, exist_ok=True)
    partial = output / "working-set.raw.f32.partial"
    final = output / "working-set.raw.f32"
    values = np.memmap(partial, dtype="<f4", mode="w+", shape=(rows, columns))
    values[:, : len(base_ids)] = base_batch.values

    provider = DenseMinuteBatchProvider(
        batch_size=args.batch_size,
        selected_ids=set(dense_ids),
        progress=True,
    )
    seen: set[str] = set()
    started = time.perf_counter()
    for batch in provider.raw_batches():
        for source_index, group in enumerate(batch.groups):
            seen.add(group.id)
            target_index = index_by_id[group.id]
            values[:, target_index] = batch.values[:, source_index]
    values.flush()
    del values
    missing = set(dense_ids) - seen
    unexpected = seen - set(dense_ids)
    if missing or unexpected:
        partial.unlink(missing_ok=True)
        raise RuntimeError(
            f"Working-set reconstruction mismatch: {len(missing)} missing, {len(unexpected)} unexpected."
        )
    partial.replace(final)

    elapsed = time.perf_counter() - started
    screen_rank = {feature_id: rank + 1 for rank, feature_id in enumerate(dense_ids)}
    manifest = {
        "version": 1,
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "purpose": "Reusable raw values for the incumbent plus initial dense global active set",
        "sourceScreen": str(screen_path.relative_to(ROOT)),
        "sourceScreenSha256": hashlib.sha256(screen_path.read_bytes()).hexdigest(),
        "registrySha256": screen["registrySha256"],
        "rows": rows,
        "columns": columns,
        "baseCoordinates": len(base_ids),
        "denseCoordinatesRequested": len(dense_ids),
        "denseCoordinatesReconstructed": len(seen),
        "duplicateCoordinatesMerged": len(base_ids) + len(dense_ids) - columns,
        "file": final.name,
        "dtype": "<f4",
        "elapsedSeconds": elapsed,
        "coordinates": [
            {
                "id": feature_id,
                "source": "existing-recent" if feature_id in set(base_ids) else "dense-minute",
                "denseGradientRank": screen_rank.get(feature_id),
            }
            for feature_id in ordered_ids
        ],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {final.relative_to(ROOT)}", flush=True)
    print(json.dumps({key: manifest[key] for key in (
        "rows", "columns", "baseCoordinates", "denseCoordinatesRequested",
        "denseCoordinatesReconstructed", "duplicateCoordinatesMerged", "elapsedSeconds",
    )}, indent=2), flush=True)


if __name__ == "__main__":
    main()
