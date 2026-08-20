from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from calibrate_next_return_density import (
    affine_dict,
    calibrated_log_masses,
    distribution_metrics,
    fit_density_temperature,
    point_transform_metrics,
)
from calibrate_next_return_output import AffineStatistics
from evaluate_feature_augmented_next_return import load_model
from normalized_glu_return_density import NormalizedGluReturnDensity
from return_knot_density import transform_returns_to_unit
from trading_storage import load_torch_checkpoint
from train_feature_augmented_next_return import ArraySplit, FeatureMatrixDataset
from train_normalized_glu_next_return import atomic_json


CONTRACT = "feature-density-pre-validation-clean-calibration-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate feature-density expectations and mass temperature on "
            "an unseen pre-validation feature matrix."
        )
    )
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--calibration-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8_192)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser.parse_args()


def canonical_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_calibration(root: Path, feature_count: int) -> tuple[dict, ArraySplit]:
    manifest = json.loads((root / "manifest.json").read_text("utf-8"))
    if int(manifest["featureCount"]) != feature_count:
        raise ValueError("calibration feature width differs from training")
    count = int(manifest["examples"])
    features = np.memmap(
        root / "calibration.features.f32", dtype="<f4", mode="r",
        shape=(count, feature_count),
    )
    targets = np.memmap(
        root / "calibration.targets.f32", dtype="<f4", mode="r",
        shape=(count,),
    )
    times = np.memmap(
        root / "calibration.times.f64", dtype="<f8", mode="r",
        shape=(count,),
    )
    if np.any(targets == 0) or np.any(np.diff(times) <= 0):
        raise ValueError("calibration split is not clean and chronological")
    return manifest, ArraySplit(features, targets, times)


@torch.no_grad()
def collect(
    model: NormalizedGluReturnDensity,
    values: ArraySplit,
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logits: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    weights: list[torch.Tensor] = []
    log_areas = torch.log(model.density_basis_areas)
    for start in range(0, values.count, batch_size):
        stop = min(values.count, start + batch_size)
        features = torch.from_numpy(np.asarray(
            values.features[start:stop], dtype=np.float32
        ).copy()).to(device)
        target = torch.from_numpy(np.asarray(
            values.targets[start:stop], dtype=np.float32
        ).copy()).to(device)
        logits.append(model.raw_density_logits(features) + log_areas)
        targets.append(target)
        weights.append(torch.ones(stop - start, device=device))
    return torch.cat(logits), torch.cat(targets), torch.cat(weights)


def evaluate_split(
    model: NormalizedGluReturnDensity,
    values: ArraySplit,
    transforms: dict,
    temperature: float,
    *,
    batch_size: int,
    target_std: float,
    device: torch.device,
) -> tuple[dict, dict]:
    logits, targets, weights = collect(
        model, values, batch_size=batch_size, device=device
    )
    points = point_transform_metrics(
        model, logits, targets, weights, transforms, target_std, device
    )
    distribution = distribution_metrics(
        model, logits, targets, weights, temperature, target_std, device
    )
    return points, distribution


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
    target_std = float(np.asarray(
        dataset.splits["train"].targets, dtype=np.float64
    ).std())
    comparison = json.loads(
        (run_root / "state/checkpoint-selection-comparison.json").read_text(
            "utf-8"
        )
    )
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
    status_file = run_root / "state/feature-density-calibration-status.json"
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
        if not isinstance(model, NormalizedGluReturnDensity):
            raise TypeError("calibration requires a density model")
        calibration_logits, calibration_targets, calibration_weights = collect(
            model, calibration, batch_size=args.batch_size, device=device
        )
        raw_expectation = (
            calibrated_log_masses(calibration_logits, 1.0).exp()
            @ model.density_component_return_means
        )
        statistics = AffineStatistics(device)
        statistics.add(
            raw_expectation, calibration_targets, calibration_weights
        )
        transforms = statistics.fit()
        unit, log_jacobian = transform_returns_to_unit(
            calibration_targets, model.density_transform
        )
        temperature, fit = fit_density_temperature(
            calibration_logits, unit, log_jacobian, calibration_weights,
            model.density_knots_unit, model.density_basis_areas,
        )
        calibration_points = point_transform_metrics(
            model, calibration_logits, calibration_targets,
            calibration_weights, transforms, target_std, device,
        )
        calibration_distribution = distribution_metrics(
            model, calibration_logits, calibration_targets,
            calibration_weights, temperature, target_std, device,
        )
        validation_points, validation_distribution = evaluate_split(
            model, dataset.splits["validation"], transforms, temperature,
            batch_size=args.batch_size, target_std=target_std, device=device,
        )
        test_points, test_distribution = evaluate_split(
            model, dataset.splits["test"], transforms, temperature,
            batch_size=args.batch_size, target_std=target_std, device=device,
        )
        output["policies"][policy] = {
            "contract": CONTRACT,
            "trainingPlanId": plan["id"],
            "trainingPlanSha256": plan_hash,
            "checkpointPolicy": policy,
            "checkpointEpoch": int(checkpoint["epoch"]),
            "transforms": affine_dict(transforms),
            "calibrationRaw": calibration_points["identity"],
            "calibration": calibration_points,
            "validation": validation_points,
            "test": test_points,
            "densityTemperature": {
                "temperature": temperature,
                "fit": fit,
                "calibration": calibration_distribution,
                "validation": validation_distribution,
                "test": test_distribution,
            },
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
