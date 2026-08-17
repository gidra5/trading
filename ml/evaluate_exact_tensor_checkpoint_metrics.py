from __future__ import annotations

import argparse
from datetime import date
import json
import math
import os
from pathlib import Path

import numpy as np
import torch

from active_return_path_dataset import ActiveReturnPathDataset
from calibrate_next_return_output import AffineStatistics, AffineTransform
from evaluate_autoregressive_density_episodes import (
    evaluate_exact_tensor_autoregressive_episodes,
    evaluate_exact_tensor_sobol_expected_episodes,
)
from exact_tensor_return_density import (
    ExactTensorReturnDensity,
    exact_tensor_path_log_density,
    temperature_scaled_output,
)
from return_knot_density import KnotDensityContract
from trading_storage import load_torch_checkpoint
from train_autoregressive_minute_return import direct_calendar_shards
from train_exact_tensor_return_path import TensorMetrics
from train_next_return_knot_density import SELECTION_POLICIES, canonical_hash
from train_next_return_memorization import fixed_nonzero_subset_shards
from train_normalized_glu_next_return import (
    MetricAccumulator,
    Reporter,
    atomic_json,
    iter_device_batches,
)


CONTRACT = "exact-tensor-six-policy-calibration-and-path-evaluation-v1"
CALIBRATION_CONTRACT = "exact-tensor-pre-validation-calibration-v1"
EPISODE_SECONDS = 15 * 60
MAXIMUM_AR_EPISODES = 128
MAXIMUM_SOBOL_EPISODES = 16
SOBOL_TRAJECTORIES = 4_096
SOBOL_REPLICATES = 16
CALIBRATION_DAYS = 7
TEMPERATURE_FIT_EXAMPLES = 16_384
TEMPERATURE_GRID = (0.16, 0.25, 0.4, 0.63, 0.8, 1.0, 1.25, 1.6, 2.5, 4.0, 6.3)
PAUSE_EXIT_CODE = 75
DEFAULT_SPLIT_PLAN = (
    "ml/training-plans/direct-glu-to-next-1s-recent-4m-1-layer-long-v2.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill calibration and path metrics for exact tensors."
    )
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--split-plan", type=Path, default=DEFAULT_SPLIT_PLAN)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--trajectory-batch-size", type=int, default=4_096)
    parser.add_argument("--pause-file", type=Path)
    parser.add_argument(
        "--checkpoint-policies", nargs="+", choices=tuple(SELECTION_POLICIES),
        default=tuple(SELECTION_POLICIES),
    )
    parser.add_argument("--skip-sobol", action="store_true")
    return parser.parse_args()


def resolve(repo: Path, value: Path) -> Path:
    return value.resolve() if value.is_absolute() else (repo / value).resolve()


def load_plan(file: Path) -> dict:
    value = json.loads(file.read_text(encoding="utf-8"))
    return value.get("plan", value)


def build_evaluation_datasets(
    plan: dict,
    history_root: Path,
) -> tuple[ActiveReturnPathDataset, ActiveReturnPathDataset]:
    heldout = int(plan["evaluation"]["examplesPerSplit"])
    validation = fixed_nonzero_subset_shards(
        history_root,
        date.fromisoformat(str(plan["evaluation"]["validationStart"])),
        heldout,
    )["train"]
    test = fixed_nonzero_subset_shards(
        history_root,
        date.fromisoformat(str(plan["evaluation"]["testStart"])),
        heldout,
    )["train"]
    return (
        ActiveReturnPathDataset(
            {"validation": validation}, history_root, return_count=3
        ),
        ActiveReturnPathDataset(
            {"test": test}, history_root, return_count=3
        ),
    )


def build_calibration_dataset(
    plan: dict,
    split_plan: dict,
    history_root: Path,
) -> tuple[ActiveReturnPathDataset, dict[str, object]]:
    calendar = direct_calendar_shards(
        split_plan["split"], history_root,
        horizon_seconds=1, decision_stride_seconds=1,
    )
    selected = calendar["train"][-CALIBRATION_DAYS:]
    training = fixed_nonzero_subset_shards(
        history_root,
        date.fromisoformat(str(plan["subset"]["date"])),
        int(plan["subset"]["examples"]),
    )["train"]
    if training[-1].decision_time_end >= selected[0].decision_time_start:
        raise ValueError("exact tensor calibration overlaps training")
    dataset = ActiveReturnPathDataset(
        {"calibration": selected}, history_root, return_count=3
    )
    return dataset, {
        "calibrationStart": selected[0].date,
        "calibrationEnd": selected[-1].date,
        "calibrationDays": CALIBRATION_DAYS,
        "calibrationExamples": dataset.logical_count("calibration"),
        "temperatureFitExamples": min(
            TEMPERATURE_FIT_EXAMPLES,
            dataset.logical_count("calibration"),
        ),
        "testPolicy": "untouched until calibration parameters were frozen",
    }


def target_scales(result: dict) -> tuple[float, float]:
    point = result["train"]
    cumulative = result["distribution"]["train"]["cumulativeExpectation"]
    return (
        math.sqrt(float(point["mse"]) / float(point["normalizedMse"])),
        math.sqrt(
            float(cumulative["mse"]) / float(cumulative["normalizedMse"])
        ),
    )


def load_model(
    plan: dict,
    density: KnotDensityContract,
    checkpoint: dict,
    device: torch.device,
) -> ExactTensorReturnDensity:
    state = checkpoint["model"]
    architecture = plan["architecture"]
    model = ExactTensorReturnDensity(
        state["feature_mean"], state["feature_std"], density,
        hidden_width=int(architecture["hiddenWidth"]),
        initial_radius=float(architecture["initialRadius"]),
        minimum_radius=float(architecture["minimumRadius"]),
        learnable_centering=bool(architecture["learnableCentering"]),
    )
    model.load_state_dict(state)
    model.eval()
    return model.to(device)


@torch.no_grad()
def fit_point_transforms(
    model: ExactTensorReturnDensity,
    dataset: ActiveReturnPathDataset,
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[dict[str, AffineTransform], dict[str, float | int | None]]:
    statistics = AffineStatistics(device)
    raw = MetricAccumulator(1.0, device)
    for features, targets, weights in iter_device_batches(
        dataset.iter_batches(
            "calibration", batch_size, shuffle=False, seed=0,
            reuse_buffers=True,
        ),
        device,
    ):
        prediction = model(features).expectations
        expanded = weights[:, None].expand_as(targets)
        statistics.add(
            prediction.reshape(-1), targets.reshape(-1), expanded.reshape(-1)
        )
        raw.add(
            prediction.reshape(-1), targets.reshape(-1), expanded.reshape(-1)
        )
    return statistics.fit(), raw.result()


@torch.no_grad()
def fit_temperature(
    model: ExactTensorReturnDensity,
    dataset: ActiveReturnPathDataset,
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[float, dict[str, object]]:
    weighted_nll = torch.zeros(
        len(TEMPERATURE_GRID), dtype=torch.float64, device=device
    )
    weight_sum = torch.zeros((), dtype=torch.float64, device=device)
    seen = 0
    for features, targets, weights in iter_device_batches(
        dataset.iter_batches(
            "calibration", batch_size, shuffle=False, seed=0,
            reuse_buffers=True,
        ),
        device,
    ):
        remaining = TEMPERATURE_FIT_EXAMPLES - seen
        if remaining <= 0:
            break
        if features.shape[0] > remaining:
            features = features[:remaining]
            targets = targets[:remaining]
            weights = weights[:remaining]
        output = model(features)
        for index, temperature in enumerate(TEMPERATURE_GRID):
            scaled = temperature_scaled_output(output, model, temperature)
            terms = exact_tensor_path_log_density(scaled, targets, model)
            weighted_nll[index] += (
                weights.double()[:, None] * -terms.double()
            ).sum()
        weight_sum += weights.double().sum() * model.return_count
        seen += int(features.shape[0])
    values = weighted_nll / weight_sum
    best = int(torch.argmin(values))
    temperature = float(TEMPERATURE_GRID[best])
    return temperature, {
        "optimizer": "held-out-log-grid-search",
        "candidates": list(TEMPERATURE_GRID),
        "negativeLogLikelihoods": [float(value) for value in values],
        "examples": seen,
        "temperature": temperature,
        "rawNegativeLogLikelihood": float(values[TEMPERATURE_GRID.index(1.0)]),
        "calibratedNegativeLogLikelihood": float(values[best]),
        "nllImprovementVsRaw": float(
            values[TEMPERATURE_GRID.index(1.0)] - values[best]
        ),
    }


@torch.no_grad()
def evaluate_calibrated_split(
    model: ExactTensorReturnDensity,
    dataset: ActiveReturnPathDataset,
    split: str,
    transforms: dict[str, AffineTransform],
    temperature: float,
    *,
    batch_size: int,
    target_std: float,
    cumulative_std: float,
    device: torch.device,
) -> tuple[dict[str, dict], dict[str, object]]:
    points = {
        name: MetricAccumulator(target_std, device) for name in transforms
    }
    density = TensorMetrics(target_std, cumulative_std, device)
    for features, targets, weights in iter_device_batches(
        dataset.iter_batches(
            split, batch_size, shuffle=False, seed=0, reuse_buffers=True
        ),
        device,
    ):
        output = model(features)
        expanded = weights[:, None].expand_as(targets)
        for name, transform in transforms.items():
            points[name].add(
                (output.expectations * transform.scale + transform.intercept)
                .reshape(-1),
                targets.reshape(-1),
                expanded.reshape(-1),
            )
        scaled = temperature_scaled_output(output, model, temperature)
        terms = exact_tensor_path_log_density(scaled, targets, model)
        density.add(scaled.expectations, targets, weights, terms)
    distribution = density.result()
    distribution["temperature"] = temperature
    return (
        {name: metric.result() for name, metric in points.items()},
        distribution,
    )


def calibration_record(
    model: ExactTensorReturnDensity,
    calibration_dataset: ActiveReturnPathDataset,
    validation_dataset: ActiveReturnPathDataset,
    test_dataset: ActiveReturnPathDataset,
    provenance: dict[str, object],
    *,
    checkpoint_epoch: int,
    checkpoint_policy: str,
    batch_size: int,
    target_std: float,
    cumulative_std: float,
    device: torch.device,
) -> dict[str, object]:
    transforms, calibration_raw = fit_point_transforms(
        model, calibration_dataset, batch_size=batch_size, device=device
    )
    temperature, temperature_fit = fit_temperature(
        model, calibration_dataset, batch_size=batch_size, device=device
    )
    calibration_points, calibration_density = evaluate_calibrated_split(
        model, calibration_dataset, "calibration", transforms, temperature,
        batch_size=batch_size, target_std=target_std,
        cumulative_std=cumulative_std, device=device,
    )
    validation_points, validation_density = evaluate_calibrated_split(
        model, validation_dataset, "validation", transforms, temperature,
        batch_size=batch_size, target_std=target_std,
        cumulative_std=cumulative_std, device=device,
    )
    test_points, test_density = evaluate_calibrated_split(
        model, test_dataset, "test", transforms, temperature,
        batch_size=batch_size, target_std=target_std,
        cumulative_std=cumulative_std, device=device,
    )
    return {
        "contract": CALIBRATION_CONTRACT,
        "checkpointEpoch": checkpoint_epoch,
        "checkpointPolicy": checkpoint_policy,
        "provenance": provenance,
        "fitObjective": {
            "expectation": "shared weighted least squares across three leads",
            "distribution": "mean exact joint conditional NLL",
        },
        "transforms": {
            name: transform.as_dict() for name, transform in transforms.items()
        },
        "calibrationRaw": calibration_raw,
        "calibration": calibration_points,
        "validation": validation_points,
        "test": test_points,
        "densityTemperature": {
            "temperature": temperature,
            "fit": temperature_fit,
            "calibration": calibration_density,
            "validation": validation_density,
            "test": test_density,
        },
    }


def main() -> None:
    args = parse_args()
    repo = Path(__file__).resolve().parents[1]
    plan_file = resolve(repo, args.plan)
    split_plan_file = resolve(repo, args.split_plan)
    plan = load_plan(plan_file)
    split_plan = load_plan(split_plan_file)
    plan_hash = canonical_hash(plan)
    run_root = resolve(repo, Path(plan["runDir"]))
    pause_file = None if args.pause_file is None else resolve(
        repo, args.pause_file
    )
    result = json.loads(
        (run_root / "state/result.json").read_text(encoding="utf-8")
    )
    comparison_file = run_root / "state/checkpoint-selection-comparison.json"
    comparison = json.loads(comparison_file.read_text(encoding="utf-8"))
    calibration_file = run_root / "state/checkpoint-selection-calibrations.json"
    calibrations: dict[str, object] = {
        "contract": CONTRACT,
        "trainingPlanId": plan["id"],
        "trainingPlanSha256": plan_hash,
        "policies": {},
    }
    if calibration_file.is_file():
        existing = json.loads(calibration_file.read_text(encoding="utf-8"))
        if existing.get("trainingPlanSha256") == plan_hash:
            calibrations = existing
    history_root = resolve(repo, Path(plan["historyDir"]))
    calibration_dataset, calibration_provenance = build_calibration_dataset(
        plan, split_plan, history_root
    )
    validation_dataset, test_dataset = build_evaluation_datasets(
        plan, history_root
    )
    density = KnotDensityContract.load(
        resolve(repo, Path(plan["density"]["source"])),
        fit=str(plan["density"]["fit"]),
    )
    target_std, cumulative_std = target_scales(result)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA exact tensor evaluation was requested but unavailable")
    reporter = Reporter(run_root)
    cache: dict[int, dict[str, object]] = {}
    requested = tuple(args.checkpoint_policies)

    def pause_if_requested(latest: dict[str, object]) -> None:
        if pause_file is None or not pause_file.is_file():
            return
        reporter.status("paused", planId=plan["id"], latest=latest)
        raise SystemExit(PAUSE_EXIT_CODE)

    try:
        for policy_index, policy in enumerate(requested):
            checkpoint_file = run_root / f"checkpoints/selections/{policy}.json"
            checkpoint = load_torch_checkpoint(
                checkpoint_file, map_location="cpu", weights_only=False
            )
            epoch = int(checkpoint["epoch"])
            pause_if_requested({
                "stage": "calibrating-checkpoints", "policy": policy,
                "epoch": epoch, "completedPolicies": policy_index,
                "totalPolicies": len(requested),
            })
            reporter.status(
                "calibrating-checkpoints", planId=plan["id"],
                latest={
                    "policy": policy, "epoch": epoch,
                    "completedPolicies": policy_index,
                    "totalPolicies": len(requested),
                },
            )
            shared = cache.get(epoch)
            previous_calibration = calibrations.get("policies", {}).get(policy)
            previous_policy = comparison["policies"].get(policy, {})
            if shared is None \
                    and isinstance(previous_calibration, dict) \
                    and int(previous_calibration.get("checkpointEpoch", -1)) == epoch \
                    and isinstance(
                        previous_policy.get("autoregressiveEpisodes"), dict
                    ):
                shared = {
                    "calibration": previous_calibration,
                    "autoregressiveEpisodes": previous_policy[
                        "autoregressiveEpisodes"
                    ],
                }
                cache[epoch] = shared
            if shared is None:
                model = load_model(plan, density, checkpoint, device)
                calibration = calibration_record(
                    model,
                    calibration_dataset,
                    validation_dataset,
                    test_dataset,
                    calibration_provenance,
                    checkpoint_epoch=epoch,
                    checkpoint_policy=policy,
                    batch_size=int(args.batch_size),
                    target_std=target_std,
                    cumulative_std=cumulative_std,
                    device=device,
                )
                autoregressive = {
                    "validation": evaluate_exact_tensor_autoregressive_episodes(
                        model, validation_dataset, "validation",
                        episode_seconds=EPISODE_SECONDS,
                        maximum_episodes=MAXIMUM_AR_EPISODES,
                        device=device,
                    ),
                    "test": evaluate_exact_tensor_autoregressive_episodes(
                        model, test_dataset, "test",
                        episode_seconds=EPISODE_SECONDS,
                        maximum_episodes=MAXIMUM_AR_EPISODES,
                        device=device,
                    ),
                }
                shared = {
                    "calibration": calibration,
                    "autoregressiveEpisodes": autoregressive,
                }
                cache[epoch] = shared
            calibration = json.loads(json.dumps(shared["calibration"]))
            calibration["checkpointPolicy"] = policy
            calibrations["policies"][policy] = calibration
            comparison["policies"][policy]["autoregressiveEpisodes"] = shared[
                "autoregressiveEpisodes"
            ]
            atomic_json(calibrations, calibration_file)
            atomic_json(comparison, comparison_file)

        if not args.skip_sobol:
            sobol_cache: dict[int, dict[str, object]] = {}
            for policy_index, policy in enumerate(requested):
                checkpoint_file = run_root / f"checkpoints/selections/{policy}.json"
                checkpoint = load_torch_checkpoint(
                    checkpoint_file, map_location="cpu", weights_only=False
                )
                epoch = int(checkpoint["epoch"])
                pause_if_requested({
                    "stage": "evaluating-path-metrics", "policy": policy,
                    "epoch": epoch, "completedPolicies": policy_index,
                    "totalPolicies": len(requested),
                })
                sobol = sobol_cache.get(epoch)
                previous_policy = comparison["policies"].get(policy, {})
                if sobol is None and isinstance(
                    previous_policy.get("sobolExpectedEpisodes"), dict
                ):
                    sobol = previous_policy["sobolExpectedEpisodes"]
                    sobol_cache[epoch] = sobol
                if sobol is None:
                    model = load_model(plan, density, checkpoint, device)
                    reporter.status(
                        "evaluating-path-metrics", planId=plan["id"],
                        latest={
                            "policy": policy, "epoch": epoch,
                            "split": "validation",
                            "completedEpisodes": 0,
                            "totalEpisodes": MAXIMUM_SOBOL_EPISODES,
                            "completedPolicies": policy_index,
                            "totalPolicies": len(requested),
                            "trajectories": SOBOL_TRAJECTORIES,
                        },
                    )

                    def progress(split: str):
                        def update(completed: int, total: int) -> None:
                            pause_if_requested({
                                "stage": "evaluating-path-metrics",
                                "policy": policy, "epoch": epoch,
                                "split": split,
                                "completedEpisodes": completed,
                                "totalEpisodes": total,
                                "completedPolicies": policy_index,
                                "totalPolicies": len(requested),
                                "trajectories": SOBOL_TRAJECTORIES,
                            })
                            reporter.status(
                                "evaluating-path-metrics", planId=plan["id"],
                                latest={
                                    "policy": policy, "epoch": epoch,
                                    "split": split,
                                    "completedEpisodes": completed,
                                    "totalEpisodes": total,
                                    "completedPolicies": policy_index,
                                    "totalPolicies": len(requested),
                                    "trajectories": SOBOL_TRAJECTORIES,
                                },
                            )
                        return update

                    sobol = {
                        "validation": evaluate_exact_tensor_sobol_expected_episodes(
                            model, validation_dataset, "validation",
                            episode_seconds=EPISODE_SECONDS,
                            maximum_episodes=MAXIMUM_SOBOL_EPISODES,
                            trajectories=SOBOL_TRAJECTORIES,
                            randomized_replicates=SOBOL_REPLICATES,
                            seed=int(plan["training"]["seed"]) + 100_000,
                            device=device,
                            trajectory_batch_size=int(args.trajectory_batch_size),
                            progress=progress("validation"),
                        ),
                        "test": evaluate_exact_tensor_sobol_expected_episodes(
                            model, test_dataset, "test",
                            episode_seconds=EPISODE_SECONDS,
                            maximum_episodes=MAXIMUM_SOBOL_EPISODES,
                            trajectories=SOBOL_TRAJECTORIES,
                            randomized_replicates=SOBOL_REPLICATES,
                            seed=int(plan["training"]["seed"]) + 200_000,
                            device=device,
                            trajectory_batch_size=int(args.trajectory_batch_size),
                            progress=progress("test"),
                        ),
                    }
                    sobol_cache[epoch] = sobol
                comparison["policies"][policy]["sobolExpectedEpisodes"] = sobol
                atomic_json(comparison, comparison_file)
        episode_evaluation = {
            "contract": CONTRACT,
            "planId": plan["id"],
            "planSha256": plan_hash,
            "policies": {
                policy: {
                    "epoch": comparison["policies"][policy]["epoch"],
                    "autoregressiveEpisodes": comparison["policies"][policy].get(
                        "autoregressiveEpisodes"
                    ),
                    "sobolExpectedEpisodes": comparison["policies"][policy].get(
                        "sobolExpectedEpisodes"
                    ),
                }
                for policy in requested
            },
        }
        complete = set(requested) == set(SELECTION_POLICIES) and all(
            isinstance(comparison["policies"][policy].get(
                "autoregressiveEpisodes"
            ), dict)
            and isinstance(comparison["policies"][policy].get(
                "sobolExpectedEpisodes"
            ), dict)
            for policy in SELECTION_POLICIES
        )
        atomic_json(
            episode_evaluation,
            run_root / "state" / (
                "autoregressive-episode-evaluation.json"
                if complete else "autoregressive-episode-evaluation.partial.json"
            ),
        )
        reporter.status("complete", planId=plan["id"], latest=result)
    except Exception as error:
        reporter.status("failed", planId=plan["id"], error=str(error))
        raise


if __name__ == "__main__":
    main()
