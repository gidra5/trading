from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from global_feature_basis_search import scan_quantized_batches_residual
from global_feature_candidate_axis import BASE_DIR, DenseMinuteBatchProvider
from global_feature_registry import ROOT


DEFAULT_OUTPUT = ROOT / "data/benchmarks/global-btc-dense-kkt-screen.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Initial all-coordinate KKT scan of the dense cross-asset minute bank.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit-assets", type=int)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--quantile-sample-rows", type=int, default=4_096)
    parser.add_argument("--retain", type=int, default=4_096)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def joint_labels(returns_bps: np.ndarray, train: np.ndarray) -> tuple[np.ndarray, list[float]]:
    active = np.abs(returns_bps[train & (returns_bps != 0)])
    thresholds = np.quantile(active, (0.25, 0.5, 0.75)) if active.size else np.zeros(3)
    magnitude = np.searchsorted(thresholds, np.abs(returns_bps), side="right")
    labels = np.where(
        returns_bps == 0,
        0,
        np.where(returns_bps > 0, 1 + magnitude, 5 + magnitude),
    ).astype(np.int64)
    return labels, thresholds.astype(float).tolist()


def multitask_intercept_residual() -> tuple[np.ndarray, dict]:
    manifest = json.loads((BASE_DIR / "manifest.json").read_text(encoding="utf-8"))
    dataset = manifest["datasets"][0]
    rows = int(dataset["rows"])
    targets = np.memmap(
        BASE_DIR / dataset["files"]["targets"],
        dtype="<f4",
        mode="r",
        shape=(rows, int(dataset["targetCount"])),
    )
    splits = np.asarray(np.memmap(
        BASE_DIR / dataset["files"]["splits"], dtype="u1", mode="r", shape=(rows,)
    ))
    train = splits == 0
    residuals = []
    horizons = []
    task_weight = 1.0 / int(dataset["targetCount"])
    scale = rows / int(np.count_nonzero(train))
    for index, definition in enumerate(dataset["targets"]):
        labels, thresholds = joint_labels(np.asarray(targets[:, index], dtype=np.float64), train)
        counts = np.bincount(labels[train], minlength=9).astype(np.float64) + 1.0
        probability = counts / counts.sum()
        residual = np.zeros((rows, 9), dtype=np.float64)
        residual[train] = probability
        residual[np.flatnonzero(train), labels[train]] -= 1.0
        residual *= task_weight * scale
        residuals.append(residual)
        horizons.append({
            "id": definition["id"],
            "classes": 9,
            "thresholdsBps": thresholds,
            "trainClassCounts": counts.astype(int).tolist(),
            "weight": task_weight,
        })
    return np.column_stack(residuals), {
        "rows": rows,
        "trainRows": int(np.count_nonzero(train)),
        "horizons": horizons,
        "split": manifest["split"],
    }


def main() -> None:
    args = parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    residual, target = multitask_intercept_residual()
    provider = DenseMinuteBatchProvider(
        batch_size=args.batch_size,
        limit_assets=args.limit_assets,
        quantile_sample_rows=args.quantile_sample_rows,
        progress=True,
    )
    started = time.perf_counter()
    result = scan_quantized_batches_residual(
        residual,
        lambda: provider.quantized_batches(),
        0.0,
        set(),
        add_limit=args.retain,
        device=args.device,
    )
    elapsed = time.perf_counter() - started
    registry_path = ROOT / "data/benchmarks/global-feature-registry.json"
    artifact = {
        "version": 1,
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "objective": "Initial exact-gradient working-set scan over every source-supported dense minute indicator coordinate",
        "modelScope": "shared-group additive 9-state return distributions at 1s, 1m, 15m, and 1h",
        "registrySha256": hashlib.sha256(registry_path.read_bytes()).hexdigest(),
        "target": target,
        "screen": {
            "assets": len(provider.assets),
            "coordinatesExpected": provider.coordinate_count,
            "coordinatesScanned": result.scanned_groups,
            "batchSize": args.batch_size,
            "quantileSampleRows": args.quantile_sample_rows,
            "retained": len(result.violating_groups),
            "lambdaMaximum": result.maximum_violation,
            "elapsedSeconds": elapsed,
            "coordinatesPerSecond": result.scanned_groups / elapsed,
            "complete": result.scanned_groups == provider.coordinate_count,
        },
        "topGradientGroups": [
            {"canonicalId": feature_id, "gradientNorm": value}
            for feature_id, value in result.violating_groups
        ],
        "use": (
            "This is the lambda-max/initial working-set pass, not a selected basis. "
            "Every excluded coordinate will be rescanned after active-model fitting for the final KKT certificate."
        ),
    }
    if not artifact["screen"]["complete"]:
        raise RuntimeError("The dense KKT screen did not visit every expected coordinate.")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output.relative_to(ROOT)}", flush=True)
    print(json.dumps(artifact["screen"], indent=2), flush=True)


if __name__ == "__main__":
    main()
