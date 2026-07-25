from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from train_mlp import FittedPolicyDataset, cached_training_normalization


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure direct-oracle target variation among similar encoded inputs."
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="validation")
    parser.add_argument("--examples", type=int, default=8192)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = json.loads((args.dataset / "dataset.json").read_text())
    dataset = FittedPolicyDataset(manifest, args.dataset, args.split)
    training = FittedPolicyDataset(manifest, args.dataset, "train")
    mean, std = cached_training_normalization(training, None)
    count = min(args.examples, len(dataset))
    indices = np.linspace(0, len(dataset) - 1, count, dtype=np.int64)
    feature_rows = []
    target_rows = []
    for index in indices:
        features, target, *_ = dataset[int(index)]
        feature_rows.append(features.numpy())
        target_rows.append(target.numpy())
    features = np.stack(feature_rows).astype(np.float32)
    targets = np.stack(target_rows).astype(np.float32)

    visible = np.asarray([
        manifest["policySupport"]["visible_lower"] <= action
        <= manifest["policySupport"]["visible_upper"]
        for action in manifest["grid"]
    ])
    targets = targets[:, visible]
    targets /= np.maximum(targets.sum(axis=1, keepdims=True), np.finfo(np.float32).tiny)

    duplicate_groups: dict[bytes, list[int]] = {}
    for index, row in enumerate(features):
        duplicate_groups.setdefault(row.tobytes(), []).append(index)
    duplicate_pairs = sum(len(group) - 1 for group in duplicate_groups.values() if len(group) > 1)

    device = torch.device(args.device)
    normalized = (
        torch.from_numpy(features).to(device)
        - mean.to(device)
    ) / std.to(device)
    squared_norm = normalized.square().sum(dim=1)
    nearest_index = torch.empty(count, dtype=torch.long, device=device)
    nearest_distance = torch.empty(count, dtype=torch.float32, device=device)
    for start in range(0, count, 512):
        stop = min(count, start + 512)
        distance = (
            squared_norm[start:stop, None]
            + squared_norm[None, :]
            - 2 * normalized[start:stop] @ normalized.T
        ).clamp_min(0)
        row = torch.arange(stop - start, device=device)
        distance[row, torch.arange(start, stop, device=device)] = torch.inf
        nearest_distance[start:stop], nearest_index[start:stop] = distance.min(dim=1)

    neighbor = nearest_index.cpu().numpy()
    feature_rms = (nearest_distance / features.shape[1]).sqrt().cpu().numpy()
    left = targets
    right = targets[neighbor]
    midpoint = 0.5 * (left + right)
    tiny = np.finfo(np.float32).tiny
    jsd = 0.5 * (
        np.sum(left * (np.log(np.maximum(left, tiny)) - np.log(np.maximum(midpoint, tiny))), axis=1)
        + np.sum(right * (np.log(np.maximum(right, tiny)) - np.log(np.maximum(midpoint, tiny))), axis=1)
    )

    quantiles = {}
    order = np.argsort(feature_rms)
    for fraction in (0.001, 0.01, 0.05, 0.25, 1.0):
        selected = order[:max(1, round(count * fraction))]
        quantiles[str(fraction)] = {
            "count": int(selected.size),
            "maximumFeatureRms": float(feature_rms[selected].max()),
            "meanTargetJsd": float(jsd[selected].mean()),
            "p95TargetJsd": float(np.quantile(jsd[selected], 0.95)),
            "maximumTargetJsd": float(jsd[selected].max()),
            "fractionTargetJsdAbove0.1": float(np.mean(jsd[selected] > 0.1)),
        }
    print(json.dumps({
        "split": args.split,
        "examples": count,
        "exactDuplicatePairs": duplicate_pairs,
        "nearestNeighborFeatureRms": {
            "minimum": float(feature_rms.min()),
            "median": float(np.median(feature_rms)),
            "p95": float(np.quantile(feature_rms, 0.95)),
        },
        "similarityBuckets": quantiles,
    }, indent=2))


if __name__ == "__main__":
    main()
