from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from calibrate_feature_augmented_return_density import load_calibration
from calibrate_next_return_output import AffineStatistics
from evaluate_feature_augmented_next_return import load_model
from normalized_glu_next_return import NormalizedGluNextReturn
from trading_storage import load_torch_checkpoint
from train_feature_augmented_next_return import ArraySplit, FeatureMatrixDataset
from train_normalized_glu_next_return import MetricAccumulator, atomic_json


CONTRACT = "feature-regression-pre-validation-clean-calibration-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit scale-only and affine calibration for every selected feature "
            "regression checkpoint on an unseen pre-validation matrix."
        )
    )
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--calibration-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4_096)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser.parse_args()


def canonical_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


@torch.no_grad()
def fit_transforms(
    model: NormalizedGluNextReturn,
    values: ArraySplit,
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[dict, dict]:
    statistics = AffineStatistics(device)
    raw = MetricAccumulator(float(model.target_std.item()), device)
    for start in range(0, values.count, batch_size):
        stop = min(values.count, start + batch_size)
        features = torch.from_numpy(np.asarray(
            values.features[start:stop], dtype=np.float32
        ).copy()).to(device)
        targets = torch.from_numpy(np.asarray(
            values.targets[start:stop], dtype=np.float32
        ).copy()).to(device)
        weights = torch.ones(stop - start, device=device)
        prediction = model(features)
        statistics.add(prediction, targets, weights)
        raw.add(prediction, targets, weights)
    return statistics.fit(), raw.result()


@torch.no_grad()
def evaluate_transforms(
    model: NormalizedGluNextReturn,
    values: ArraySplit,
    transforms: dict,
    *,
    batch_size: int,
    device: torch.device,
) -> dict:
    metrics = {
        name: MetricAccumulator(float(model.target_std.item()), device)
        for name in transforms
    }
    for start in range(0, values.count, batch_size):
        stop = min(values.count, start + batch_size)
        features = torch.from_numpy(np.asarray(
            values.features[start:stop], dtype=np.float32
        ).copy()).to(device)
        targets = torch.from_numpy(np.asarray(
            values.targets[start:stop], dtype=np.float32
        ).copy()).to(device)
        weights = torch.ones(stop - start, device=device)
        prediction = model(features)
        for name, transform in transforms.items():
            metrics[name].add(
                prediction * transform.scale + transform.intercept,
                targets,
                weights,
            )
    return {name: metric.result() for name, metric in metrics.items()}


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("batch size must be positive")
    repo = Path(__file__).resolve().parents[1]
    plan_file = (repo / args.plan).resolve() \
        if not args.plan.is_absolute() else args.plan.resolve()
    calibration_root = (repo / args.calibration_dir).resolve() \
        if not args.calibration_dir.is_absolute() else args.calibration_dir.resolve()
    plan = json.loads(plan_file.read_text("utf-8"))
    plan_hash = canonical_hash(plan)
    run_root = (repo / plan["runDir"]).resolve()
    dataset = FeatureMatrixDataset((repo / plan["datasetDir"]).resolve())
    calibration_manifest, calibration = load_calibration(
        calibration_root, dataset.feature_count
    )
    if not (
        dataset.splits["train"].times[-1] < calibration.times[0]
        and calibration.times[-1] < dataset.splits["validation"].times[0]
    ):
        raise ValueError("calibration split is not between train and validation")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA calibration was requested but unavailable")
    comparison_file = run_root / "state/checkpoint-selection-comparison.json"
    comparison = json.loads(comparison_file.read_text("utf-8"))
    output = {
        "contract": CONTRACT,
        "trainingPlanId": plan["id"],
        "trainingPlanSha256": plan_hash,
        "calibrationDataset": {
            "directory": str(calibration_root.relative_to(repo)),
            "contract": calibration_manifest["contract"],
            "examples": calibration.count,
            "interval": calibration_manifest["interval"],
            "firstOriginMs": float(calibration.times[0]),
            "lastOriginMs": float(calibration.times[-1]),
            "testPolicy": "untouched until all transforms were frozen",
        },
        "policies": {},
    }
    status_file = run_root / "state/feature-regression-calibration-status.json"
    for index, (policy, selected) in enumerate(comparison["policies"].items()):
        atomic_json({
            "stage": "calibrating-checkpoints",
            "policy": policy,
            "completedPolicies": index,
            "totalPolicies": len(comparison["policies"]),
        }, status_file)
        checkpoint = load_torch_checkpoint(
            repo / selected["checkpoint"], map_location=device,
            weights_only=False,
        )
        if checkpoint.get("planSha256") != plan_hash:
            raise ValueError(f"checkpoint plan mismatch: {policy}")
        model = load_model(repo, plan, checkpoint, device)
        if not isinstance(model, NormalizedGluNextReturn):
            raise TypeError("regression calibration requires a scalar model")
        transforms, calibration_raw = fit_transforms(
            model, calibration, batch_size=args.batch_size, device=device
        )
        calibration_metrics = evaluate_transforms(
            model, calibration, transforms,
            batch_size=args.batch_size, device=device,
        )
        validation_metrics = evaluate_transforms(
            model, dataset.splits["validation"], transforms,
            batch_size=args.batch_size, device=device,
        )
        test_metrics = evaluate_transforms(
            model, dataset.splits["test"], transforms,
            batch_size=args.batch_size, device=device,
        )
        output["policies"][policy] = {
            "contract": CONTRACT,
            "trainingPlanId": plan["id"],
            "trainingPlanSha256": plan_hash,
            "checkpointPolicy": policy,
            "checkpointEpoch": int(checkpoint["epoch"]),
            "transforms": {
                name: transform.as_dict()
                for name, transform in transforms.items()
            },
            "calibrationRaw": calibration_raw,
            "calibration": calibration_metrics,
            "validation": validation_metrics,
            "test": test_metrics,
        }
        atomic_json(
            output,
            run_root / "state/checkpoint-selection-calibrations.json",
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    atomic_json({
        "stage": "complete",
        "completedPolicies": len(comparison["policies"]),
        "totalPolicies": len(comparison["policies"]),
    }, status_file)


if __name__ == "__main__":
    main()
