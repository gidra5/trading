"""Prepare matched candle-control and forward-market 15m oracle datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from forward_market_features import (
    DAY_ROWS,
    FORWARD_FEATURE_COUNT,
    build_forward_feature_day,
    has_forward_feature_day,
)


BASE_FEATURE_COUNT = 771
ACTION_COUNT = 101


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--refresh-normalization", action="store_true")
    return parser.parse_args()


def atomic_json(value: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def feature_normalization(
    features: np.ndarray,
    feature_count: int,
    mode: str,
) -> tuple[np.ndarray, np.ndarray, int]:
    feature_sum = np.zeros(feature_count, dtype=np.float64)
    feature_square_sum = np.zeros(feature_count, dtype=np.float64)
    for start in range(0, features.shape[0], 16_384):
        batch = np.asarray(features[start:start + 16_384], dtype=np.float64)
        feature_sum += batch.sum(axis=0)
        feature_square_sum += np.square(batch).sum(axis=0)
    mean = feature_sum / features.shape[0]
    variance = np.maximum(
        feature_square_sum / features.shape[0] - np.square(mean), 0,
    )
    std = np.sqrt(variance)
    low_variance = np.zeros(feature_count, dtype=bool)
    if mode == "joint":
        low_variance[BASE_FEATURE_COUNT:] = std[BASE_FEATURE_COUNT:] < 1e-4
        std[low_variance] = 1.0
    std = np.maximum(std, 1e-6)
    return mean, std, int(np.count_nonzero(low_variance))


def prepare(plan_file: Path, *, refresh_normalization: bool = False) -> dict:
    repo = Path(__file__).resolve().parent.parent
    resolved_plan = plan_file if plan_file.is_absolute() else repo / plan_file
    plan = json.loads(resolved_plan.read_text(encoding="utf-8"))
    dataset = plan["dataset"]
    mode = str(dataset["forwardFeatureMode"])
    if mode not in {"base-control", "joint"}:
        raise ValueError("forwardFeatureMode must be base-control or joint")
    feature_count = BASE_FEATURE_COUNT + (FORWARD_FEATURE_COUNT if mode == "joint" else 0)
    if int(dataset["featureCount"]) != feature_count:
        raise ValueError("plan featureCount disagrees with forward feature mode")
    base_root = repo / dataset["baseDatasetDir"]
    output_root = repo / dataset["datasetDir"]
    data_root = repo / "data"
    base_manifest_file = base_root / "dataset.json"
    source_sha = hashlib.sha256(base_manifest_file.read_bytes()).hexdigest()
    manifest_file = output_root / "dataset.json"
    if manifest_file.is_file():
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        if manifest.get("sourceSha256") != source_sha \
                or manifest.get("forwardFeatureMode") != mode \
                or int(manifest.get("featureCount", 0)) != feature_count:
            raise ValueError("prepared forward dataset differs from its plan")
        if refresh_normalization:
            features = np.load(
                output_root / manifest["files"]["train"]["features"],
                mmap_mode="r",
            )
            mean, std, low_variance_count = feature_normalization(
                features, feature_count, mode,
            )
            manifest["normalization"] = {
                "mean": mean.tolist(),
                "std": std.tolist(),
                "policy": "training-mean-std; forward std<1e-4 uses neutral scale 1",
                "neutralizedLowVarianceForwardFeatures": low_variance_count,
            }
            atomic_json(manifest, manifest_file)
        return manifest

    base_manifest = json.loads(base_manifest_file.read_text(encoding="utf-8"))
    output_root.mkdir(parents=True, exist_ok=False)
    counts: dict[str, int] = {}
    eligible_days: dict[str, list[str]] = {}
    arrays: dict[str, tuple[np.memmap, np.memmap, np.memmap]] = {}
    selected_indexes: dict[str, np.ndarray] = {}
    for split in ("train", "validation", "test"):
        files = base_manifest["files"][split]
        timestamps = np.load(base_root / files["timestamps"], mmap_mode="r")
        dates = np.datetime_as_string(
            timestamps.astype("datetime64[ms]").astype("datetime64[D]")
        )
        days = [
            str(day) for day in np.unique(dates)
            if has_forward_feature_day(data_root, str(day))
        ]
        mask = np.isin(dates, days)
        indexes = np.flatnonzero(mask)
        if indexes.size == 0:
            raise ValueError(f"no eligible {split} forward-feature rows")
        selected_indexes[split] = indexes
        eligible_days[split] = days
        counts[split] = int(indexes.size)
        arrays[split] = (
            np.lib.format.open_memmap(
                output_root / f"{split}.features.f16.npy",
                mode="w+", dtype="<f2", shape=(indexes.size, feature_count),
            ),
            np.lib.format.open_memmap(
                output_root / f"{split}.targets.f32.npy",
                mode="w+", dtype="<f4", shape=(indexes.size, ACTION_COUNT),
            ),
            np.lib.format.open_memmap(
                output_root / f"{split}.timestamps.i64.npy",
                mode="w+", dtype="<i8", shape=(indexes.size,),
            ),
        )

    for split in ("train", "validation", "test"):
        files = base_manifest["files"][split]
        base_features = np.load(base_root / files["features"], mmap_mode="r")
        base_targets = np.load(base_root / files["targets"], mmap_mode="r")
        base_timestamps = np.load(base_root / files["timestamps"], mmap_mode="r")
        indexes = selected_indexes[split]
        output_features, output_targets, output_timestamps = arrays[split]
        output_features[:, :BASE_FEATURE_COUNT] = base_features[indexes, :BASE_FEATURE_COUNT]
        output_targets[:] = base_targets[indexes]
        output_timestamps[:] = base_timestamps[indexes]
        if mode == "joint":
            selected_times = np.asarray(base_timestamps[indexes], dtype=np.int64)
            selected_dates = np.datetime_as_string(
                selected_times.astype("datetime64[ms]").astype("datetime64[D]")
            )
            for day_index, day in enumerate(eligible_days[split], 1):
                positions = np.flatnonzero(selected_dates == day)
                day_start = int(np.datetime64(day, "ms").astype(np.int64))
                elapsed = selected_times[positions] - day_start
                if bool((elapsed % 60_000 != 59_999).any()):
                    raise ValueError(f"{day} timestamps are not completed-minute closes")
                minute_rows = elapsed // 60_000
                if bool((minute_rows < 0).any()) or bool((minute_rows >= DAY_ROWS).any()):
                    raise ValueError(f"{day} minute indexes are out of range")
                forward = build_forward_feature_day(data_root, day)
                output_features[positions, BASE_FEATURE_COUNT:] = forward[minute_rows]
                if day_index % 25 == 0 or day_index == len(eligible_days[split]):
                    print(json.dumps({
                        "event": "forward-feature-progress",
                        "split": split,
                        "days": day_index,
                        "totalDays": len(eligible_days[split]),
                    }), flush=True)
        for output in arrays[split]:
            output.flush()

    mean, std, low_variance_count = feature_normalization(
        arrays["train"][0], feature_count, mode,
    )
    manifest = {
        "schemaVersion": 1,
        "prebuilt": True,
        "sourceDataset": str(base_root),
        "sourceSha256": source_sha,
        "forwardFeatureMode": mode,
        "samplingIntervalMs": 60_000,
        "predictionHorizonMs": 900_000,
        "pairing": "features through completed minute t to next-15m oracle at t",
        "featureContract": (
            "schema-6 multiscale OHLCV through daily plus causal spot trade flow, "
            "USD-M basis/flow, and USD-M positioning metrics"
            if mode == "joint" else
            "matched schema-6 multiscale OHLCV through daily control"
        ),
        "featureCount": feature_count,
        "baseFeatureCount": BASE_FEATURE_COUNT,
        "forwardFeatureCount": feature_count - BASE_FEATURE_COUNT,
        "actionCount": ACTION_COUNT,
        "oracle": base_manifest["oracle"],
        "counts": counts,
        "eligibleDays": eligible_days,
        "normalization": {
            "mean": mean.tolist(),
            "std": std.tolist(),
            "policy": "training-mean-std; forward std<1e-4 uses neutral scale 1",
            "neutralizedLowVarianceForwardFeatures": low_variance_count,
        },
        "files": {
            split: {
                "features": f"{split}.features.f16.npy",
                "targets": f"{split}.targets.f32.npy",
                "timestamps": f"{split}.timestamps.i64.npy",
            }
            for split in ("train", "validation", "test")
        },
    }
    atomic_json(manifest, manifest_file)
    return manifest


if __name__ == "__main__":
    arguments = parse_args()
    result = prepare(
        arguments.plan,
        refresh_normalization=arguments.refresh_normalization,
    )
    print(json.dumps({"event": "dataset-ready", "counts": result["counts"]}))
