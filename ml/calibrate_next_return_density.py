from __future__ import annotations

import argparse
from datetime import date
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
import torch

from active_return_path_dataset import ActiveReturnPathDataset
from calibrate_next_return_output import AffineStatistics, AffineTransform
from normalized_glu_return_density import NormalizedGluReturnDensity
from return_knot_density import (
    KnotDensityContract,
    interpolated_log_density_unit,
    transform_returns_to_unit,
)
from trading_storage import load_torch_checkpoint
from train_autoregressive_minute_return import direct_calendar_shards
from train_next_return_memorization import fixed_nonzero_subset_shards
from train_next_return_knot_density import SELECTION_POLICIES
from train_normalized_glu_next_return import (
    MetricAccumulator,
    NextReturnDataset,
    atomic_json,
    canonical_fingerprint,
    iter_device_batches,
    resolve,
)


CALIBRATION_CONTRACT = (
    "fixed-knot-density-affine-expectation-and-mass-temperature-"
    "pre-validation-7d-v1"
)
CHECKPOINT_CALIBRATION_CONTRACT = (
    "six-policy-fixed-knot-density-calibration-pre-validation-7d-v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit affine expectation and full-distribution temperature "
            "calibration on the seven clean days before validation."
        )
    )
    parser.add_argument("--training-plan", required=True, type=Path, nargs="+")
    parser.add_argument("--split-plan", required=True, type=Path)
    parser.add_argument("--calibration-days", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=8_192)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument(
        "--checkpoint-policies",
        nargs="+",
        choices=tuple(SELECTION_POLICIES),
        help="Calibrate saved checkpoint-selection policies instead of best.json",
    )
    return parser.parse_args()


def calibrated_log_masses(
    mass_logits: torch.Tensor,
    temperature: float | torch.Tensor,
) -> torch.Tensor:
    value = torch.as_tensor(
        temperature, dtype=mass_logits.dtype, device=mass_logits.device
    )
    if value.numel() != 1 or not bool(torch.isfinite(value)) \
            or not bool(value > 0):
        raise ValueError("density temperature must be finite and positive")
    return torch.log_softmax(mass_logits / value, dim=-1)


@torch.no_grad()
def weighted_temperature_nll(
    mass_logits: torch.Tensor,
    unit_targets: torch.Tensor,
    log_jacobian: torch.Tensor,
    weights: torch.Tensor,
    knots: torch.Tensor,
    areas: torch.Tensor,
    temperature: float,
) -> float:
    log_masses = calibrated_log_masses(mass_logits, temperature)
    log_density = interpolated_log_density_unit(
        log_masses, unit_targets, knots, areas
    )
    return float(
        (weights.double() * -(log_density + log_jacobian).double()).sum()
        / weights.double().sum()
    )


def fit_density_temperature(
    mass_logits: torch.Tensor,
    unit_targets: torch.Tensor,
    log_jacobian: torch.Tensor,
    weights: torch.Tensor,
    knots: torch.Tensor,
    areas: torch.Tensor,
) -> tuple[float, dict[str, float | int | bool | str]]:
    if mass_logits.ndim != 2 or mass_logits.shape[0] != unit_targets.numel() \
            or unit_targets.shape != log_jacobian.shape \
            or unit_targets.shape != weights.shape:
        raise ValueError("invalid cached density-calibration tensors")

    evaluations = 0

    def objective(log_temperature: float) -> float:
        nonlocal evaluations
        evaluations += 1
        return weighted_temperature_nll(
            mass_logits,
            unit_targets,
            log_jacobian,
            weights,
            knots,
            areas,
            math.exp(log_temperature),
        )

    raw_nll = objective(0.0)
    optimized = minimize_scalar(
        objective,
        method="bounded",
        bounds=(math.log(0.05), math.log(20.0)),
        options={"xatol": 1e-4, "maxiter": 48},
    )
    candidate_temperature = math.exp(float(optimized.x))
    candidate_nll = float(optimized.fun)
    if not math.isfinite(candidate_nll) or candidate_nll > raw_nll:
        candidate_temperature = 1.0
        candidate_nll = raw_nll
    return candidate_temperature, {
        "optimizer": "bounded-scalar-search-in-log-temperature",
        "lowerBound": 0.05,
        "upperBound": 20.0,
        "evaluations": evaluations,
        "success": bool(optimized.success),
        "rawNegativeLogLikelihood": raw_nll,
        "calibratedNegativeLogLikelihood": candidate_nll,
        "nllImprovementVsRaw": raw_nll - candidate_nll,
    }


def target_std_from_result(result: dict) -> float:
    train = result["train"]
    mse = float(train["mse"])
    normalized = float(train["normalizedMse"])
    if not math.isfinite(mse) or not math.isfinite(normalized) \
            or mse <= 0 or normalized <= 0:
        raise ValueError("density result cannot recover its training target scale")
    return math.sqrt(mse / normalized)


def load_model(
    repo: Path,
    plan: dict,
    checkpoint: dict,
    device: torch.device,
) -> NormalizedGluReturnDensity:
    density = KnotDensityContract.load(
        resolve(repo, Path(plan["density"]["source"])),
        fit=str(plan["density"]["fit"]),
    )
    state = checkpoint["model"]
    architecture = plan["architecture"]
    model = NormalizedGluReturnDensity(
        state["feature_mean"],
        state["feature_std"],
        density,
        return_count=int(plan["density"].get("returnCount", 1)),
        widths=tuple(int(value) for value in architecture["widths"]),
        dropout=0,
        dropout_rate=0,
        initial_radius=float(architecture["initialRadius"]),
        minimum_radius=float(architecture["minimumRadius"]),
        learnable_centering=bool(architecture["learnableCentering"]),
    )
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.unexpected_keys \
            or set(incompatible.missing_keys) - {"prefix_conditioner"} \
            or ("prefix_conditioner" in incompatible.missing_keys
                and model.prefix_conditioner.numel() != 0):
        raise ValueError("density checkpoint is incompatible with its plan")
    model.eval()
    return model.to(device)


def affine_dict(
    transforms: dict[str, AffineTransform],
) -> dict[str, dict[str, float]]:
    return {name: value.as_dict() for name, value in transforms.items()}


def distribution_metrics(
    model: NormalizedGluReturnDensity,
    mass_logits: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
    temperature: float,
    target_std: float,
    device: torch.device,
) -> dict[str, object]:
    unit, log_jacobian = transform_returns_to_unit(
        targets, model.density_transform
    )
    raw_log_masses = calibrated_log_masses(mass_logits, 1.0)
    log_masses = calibrated_log_masses(mass_logits, temperature)
    raw_log_density = interpolated_log_density_unit(
        raw_log_masses,
        unit,
        model.density_knots_unit,
        model.density_basis_areas,
    )
    log_density = interpolated_log_density_unit(
        log_masses,
        unit,
        model.density_knots_unit,
        model.density_basis_areas,
    )
    prior_log_masses = torch.log(model.density_prior_masses)[None, :].expand(
        targets.shape[0], -1
    )
    prior_log_density = interpolated_log_density_unit(
        prior_log_masses,
        unit,
        model.density_knots_unit,
        model.density_basis_areas,
    )
    weight = weights.double().sum()
    raw_nll = float(
        (weights.double() * -(raw_log_density + log_jacobian).double()).sum()
        / weight
    )
    nll = float(
        (weights.double() * -(log_density + log_jacobian).double()).sum()
        / weight
    )
    prior_nll = float(
        (weights.double() * -(prior_log_density + log_jacobian).double()).sum()
        / weight
    )
    expectation = log_masses.exp() @ model.density_component_return_means
    point = MetricAccumulator(target_std, device)
    point.add(expectation, targets, weights)
    return {
        "examples": int(round(float(weight))),
        "temperature": temperature,
        "negativeLogLikelihood": nll,
        "bitsPerExample": nll / math.log(2),
        "rawNegativeLogLikelihood": raw_nll,
        "nllImprovementVsRaw": raw_nll - nll,
        "globalBaselineNegativeLogLikelihood": prior_nll,
        "nllImprovementVsGlobal": prior_nll - nll,
        "expectation": point.result(),
    }


@torch.no_grad()
def collect_split(
    model: NormalizedGluReturnDensity,
    dataset: NextReturnDataset | ActiveReturnPathDataset,
    split: str,
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    forecast_logits: list[torch.Tensor] = []
    joint_logits: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    weights: list[torch.Tensor] = []
    log_areas = torch.log(model.density_basis_areas)
    for features, batch_targets, batch_weights in iter_device_batches(
        dataset.iter_batches(
            split, batch_size, shuffle=False, seed=0, reuse_buffers=True
        ),
        device,
    ):
        batch_forecast_logits = model.raw_density_logits(features) + log_areas
        batch_joint_logits = (
            model.teacher_forced_density_logits(features, batch_targets)
            + log_areas
        )
        if batch_forecast_logits.ndim == 3:
            path_length = batch_forecast_logits.shape[1]
            batch_forecast_logits = batch_forecast_logits.reshape(
                -1, batch_forecast_logits.shape[-1]
            )
            batch_joint_logits = batch_joint_logits.reshape(
                -1, batch_joint_logits.shape[-1]
            )
            batch_targets = batch_targets.reshape(-1)
            batch_weights = batch_weights[:, None].expand(
                -1, path_length
            ).reshape(-1)
        forecast_logits.append(batch_forecast_logits)
        joint_logits.append(batch_joint_logits)
        targets.append(batch_targets)
        weights.append(batch_weights)
    return (
        torch.cat(forecast_logits),
        torch.cat(joint_logits),
        torch.cat(targets),
        torch.cat(weights),
    )


def point_transform_metrics(
    model: NormalizedGluReturnDensity,
    mass_logits: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
    transforms: dict[str, AffineTransform],
    target_std: float,
    device: torch.device,
) -> dict[str, dict[str, float | int | None]]:
    expectation = (
        calibrated_log_masses(mass_logits, 1.0).exp()
        @ model.density_component_return_means
    )
    result: dict[str, dict[str, float | int | None]] = {}
    for name, transform in transforms.items():
        accumulator = MetricAccumulator(target_std, device)
        accumulator.add(
            expectation * transform.scale + transform.intercept,
            targets,
            weights,
        )
        result[name] = accumulator.result()
    return result


def calibrate_plan(
    repo: Path,
    training_plan_file: Path,
    split_plan: dict,
    split_plan_file: Path,
    *,
    calibration_days: int,
    batch_size: int,
    device: torch.device,
    checkpoint_file: Path | None = None,
    checkpoint_policy: str | None = None,
    persist: bool = True,
) -> dict[str, object]:
    plan_value = json.loads(training_plan_file.read_text(encoding="utf-8"))
    plan = plan_value.get("plan", plan_value)
    run_root = resolve(repo, Path(plan["runDir"]))
    result_file = run_root / "state/result.json"
    result = json.loads(result_file.read_text(encoding="utf-8"))
    checkpoint_file = checkpoint_file or run_root / "checkpoints/best.json"
    checkpoint = load_torch_checkpoint(
        checkpoint_file, map_location="cpu", weights_only=False
    )
    plan_hash = canonical_fingerprint(plan)
    if checkpoint.get("planSha256") != plan_hash \
            or result.get("planSha256") != plan_hash:
        raise ValueError("density calibration inputs belong to another plan")

    history_root = resolve(repo, Path(plan["historyDir"]))
    calendar = direct_calendar_shards(
        split_plan["split"], history_root,
        horizon_seconds=1, decision_stride_seconds=1,
    )
    if calibration_days < 1 or calibration_days >= len(calendar["train"]):
        raise ValueError("calibration days must select a proper training tail")
    calibration_shards = calendar["train"][-calibration_days:]
    train_shards = fixed_nonzero_subset_shards(
        history_root,
        date.fromisoformat(str(plan["subset"]["date"])),
        int(plan["subset"]["examples"]),
    )["train"]
    if train_shards[-1].decision_time_end >= calibration_shards[0].decision_time_start:
        raise ValueError("calibration tail overlaps the density training subset")
    heldout = int(plan["evaluation"]["examplesPerSplit"])
    validation_shards = fixed_nonzero_subset_shards(
        history_root,
        date.fromisoformat(str(plan["evaluation"]["validationStart"])),
        heldout,
    )["train"]
    test_shards = fixed_nonzero_subset_shards(
        history_root,
        date.fromisoformat(str(plan["evaluation"]["testStart"])),
        heldout,
    )["train"]
    return_count = int(plan["density"].get("returnCount", 1))
    dataset_type = NextReturnDataset if return_count == 1 \
        else ActiveReturnPathDataset
    dataset_kwargs = {"exclude_zero_targets": True} \
        if return_count == 1 else {"return_count": return_count}
    dataset = dataset_type({
        "calibration": calibration_shards,
        "validation": validation_shards,
        "test": test_shards,
    }, history_root, **dataset_kwargs)
    target_std = target_std_from_result(result)
    model = load_model(repo, plan, checkpoint, device)

    (
        calibration_forecast_logits,
        calibration_joint_logits,
        calibration_targets,
        calibration_weights,
    ) = collect_split(
        model, dataset, "calibration", batch_size=batch_size, device=device
    )
    raw_log_masses = calibrated_log_masses(
        calibration_forecast_logits, 1.0
    )
    raw_expectation = (
        raw_log_masses.exp() @ model.density_component_return_means
    )
    affine_statistics = AffineStatistics(device)
    affine_statistics.add(
        raw_expectation, calibration_targets, calibration_weights
    )
    transforms = affine_statistics.fit()
    unit, log_jacobian = transform_returns_to_unit(
        calibration_targets, model.density_transform
    )
    temperature, fit = fit_density_temperature(
        calibration_joint_logits,
        unit,
        log_jacobian,
        calibration_weights,
        model.density_knots_unit,
        model.density_basis_areas,
    )

    calibration_points = point_transform_metrics(
        model,
        calibration_forecast_logits,
        calibration_targets,
        calibration_weights,
        transforms, target_std, device,
    )
    calibration_distribution = distribution_metrics(
        model,
        calibration_joint_logits,
        calibration_targets,
        calibration_weights,
        temperature, target_std, device,
    )
    (
        validation_forecast_logits,
        validation_joint_logits,
        validation_targets,
        validation_weights,
    ) = collect_split(
        model, dataset, "validation", batch_size=batch_size, device=device
    )
    validation_points = point_transform_metrics(
        model,
        validation_forecast_logits,
        validation_targets,
        validation_weights,
        transforms, target_std, device,
    )
    validation_distribution = distribution_metrics(
        model,
        validation_joint_logits,
        validation_targets,
        validation_weights,
        temperature, target_std, device,
    )
    del validation_forecast_logits, validation_joint_logits
    del validation_targets, validation_weights
    (
        test_forecast_logits,
        test_joint_logits,
        test_targets,
        test_weights,
    ) = collect_split(
        model, dataset, "test", batch_size=batch_size, device=device
    )
    test_points = point_transform_metrics(
        model, test_forecast_logits, test_targets, test_weights,
        transforms, target_std, device,
    )
    test_distribution = distribution_metrics(
        model, test_joint_logits, test_targets, test_weights,
        temperature, target_std, device,
    )

    output: dict[str, object] = {
        "contract": CALIBRATION_CONTRACT,
        "trainingPlanId": plan["id"],
        "trainingPlanSha256": plan_hash,
        "checkpoint": str(checkpoint_file.relative_to(repo)),
        "checkpointEpoch": int(checkpoint["epoch"]),
        **({} if checkpoint_policy is None else {
            "checkpointPolicy": checkpoint_policy,
        }),
        "splitPlanId": split_plan["id"],
        "splitPlanSha256": canonical_fingerprint(split_plan),
        "splitPlan": str(split_plan_file.relative_to(repo)),
        "provenance": {
            "trainingSubsetStart": plan["subset"]["date"],
            "trainingExamples": int(plan["subset"]["examples"]),
            "calibrationStart": calibration_shards[0].date,
            "calibrationEnd": calibration_shards[-1].date,
            "calibrationDays": calibration_days,
            "calibrationExamples": dataset.logical_count("calibration"),
            "calibrationTargetReturns": (
                dataset.logical_count("calibration") * return_count
            ),
            "validationStart": plan["evaluation"]["validationStart"],
            "validationExamples": dataset.logical_count("validation"),
            "testStart": plan["evaluation"]["testStart"],
            "testExamples": dataset.logical_count("test"),
            "testPolicy": "untouched until both transforms were frozen",
        },
        "datasetFilter": plan.get("datasetFilter"),
        "fitObjective": {
            "expectation": (
                "shared weighted ordinary least squares across all path leads"
            ),
            "distribution": (
                "shared-temperature weighted return-space negative log "
                "likelihood across all path leads"
            ),
        },
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
    if persist:
        atomic_json(
            output,
            run_root / "state/output-calibration-pre-validation-7d.json",
        )
    del model, calibration_forecast_logits, calibration_joint_logits
    del calibration_targets, calibration_weights
    del test_forecast_logits, test_joint_logits, test_targets, test_weights
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return output


def calibrate_checkpoint_selections(
    repo: Path,
    training_plan_file: Path,
    split_plan: dict,
    split_plan_file: Path,
    *,
    policies: tuple[str, ...],
    calibration_days: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, object]:
    plan_value = json.loads(training_plan_file.read_text(encoding="utf-8"))
    plan = plan_value.get("plan", plan_value)
    plan_hash = canonical_fingerprint(plan)
    run_root = resolve(repo, Path(plan["runDir"]))
    output_file = (
        run_root / "state/checkpoint-selection-calibrations.json"
    )
    output: dict[str, object] = {
        "contract": CHECKPOINT_CALIBRATION_CONTRACT,
        "trainingPlanId": plan["id"],
        "trainingPlanSha256": plan_hash,
        "policies": {},
    }
    if output_file.is_file():
        existing = json.loads(output_file.read_text(encoding="utf-8"))
        if existing.get("trainingPlanSha256") == plan_hash:
            output = existing
    calibrated = output.setdefault("policies", {})
    if not isinstance(calibrated, dict):
        raise ValueError("invalid checkpoint calibration policy map")
    for policy in policies:
        checkpoint_file = (
            run_root / f"checkpoints/selections/{policy}.json"
        )
        checkpoint = load_torch_checkpoint(
            checkpoint_file, map_location="cpu", weights_only=False
        )
        checkpoint_epoch = int(checkpoint["epoch"])
        previous = calibrated.get(policy)
        if isinstance(previous, dict) \
                and int(previous.get("checkpointEpoch", -1)) == checkpoint_epoch \
                and previous.get("trainingPlanSha256") == plan_hash:
            continue
        calibrated[policy] = calibrate_plan(
            repo,
            training_plan_file,
            split_plan,
            split_plan_file,
            calibration_days=calibration_days,
            batch_size=batch_size,
            device=device,
            checkpoint_file=checkpoint_file,
            checkpoint_policy=policy,
            persist=False,
        )
        atomic_json(output, output_file)
    return output


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("calibration batch size must be positive")
    repo = Path(__file__).resolve().parents[1]
    split_plan_file = resolve(repo, args.split_plan)
    split_plan = json.loads(split_plan_file.read_text(encoding="utf-8"))
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA density calibration was requested but unavailable")
    torch.set_float32_matmul_precision("high")
    completed: list[str] = []
    for value in args.training_plan:
        training_plan_file = resolve(repo, value)
        if args.checkpoint_policies:
            result = calibrate_checkpoint_selections(
                repo,
                training_plan_file,
                split_plan,
                split_plan_file,
                policies=tuple(args.checkpoint_policies),
                calibration_days=int(args.calibration_days),
                batch_size=int(args.batch_size),
                device=device,
            )
            completed.append(str(result["trainingPlanId"]))
            print(json.dumps({
                "event": "density-checkpoint-calibrations-complete",
                "trainingPlanId": result["trainingPlanId"],
                "policies": {
                    policy: {
                        "epoch": value["checkpointEpoch"],
                        "affine": value["transforms"]["affine"],
                        "temperature": value["densityTemperature"]["temperature"],
                    }
                    for policy, value in result["policies"].items()
                },
            }, separators=(",", ":")), flush=True)
        else:
            result = calibrate_plan(
                repo,
                training_plan_file,
                split_plan,
                split_plan_file,
                calibration_days=int(args.calibration_days),
                batch_size=int(args.batch_size),
                device=device,
            )
            completed.append(str(result["trainingPlanId"]))
            print(json.dumps({
                "event": "density-output-calibration-complete",
                "trainingPlanId": result["trainingPlanId"],
                "affine": result["transforms"]["affine"],
                "temperature": result["densityTemperature"]["temperature"],
                "validation": result["densityTemperature"]["validation"],
                "test": result["densityTemperature"]["test"],
            }, separators=(",", ":")), flush=True)
    print(json.dumps({
        "contract": CALIBRATION_CONTRACT,
        "completedPlans": completed,
    }, separators=(",", ":")))


if __name__ == "__main__":
    main()
