from __future__ import annotations

import argparse
from datetime import date
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

from active_return_path_dataset import ActiveReturnPathDataset
from normalized_glu_return_density import NormalizedGluReturnDensity
from return_knot_density import (
    KnotDensityContract,
    component_log_masses,
    interpolated_log_density_unit,
    return_negative_log_likelihood,
    transform_returns_to_unit,
)
from trading_storage import (
    checkpoint_exists,
    load_torch_checkpoint,
    save_torch_checkpoint,
)
from train_autoregressive_minute_return import (
    build_optimizers,
    training_normalization,
)
from train_next_return_memorization import fixed_nonzero_subset_shards
from train_normalized_glu_next_return import (
    MetricAccumulator,
    NextReturnDataset,
    Reporter,
    atomic_json,
    iter_device_batches,
)


SINGLE_RETURN_RUNNER_CONTRACT = (
    "next-second-fixed-knot-conditional-density-nll-v1"
)
PATH_RUNNER_CONTRACT = (
    "next-second-direct-active-return-path-fixed-knot-density-nll-v1"
)
PAUSE_EXIT_CODE = 75
SELECTION_POLICIES = {
    "train-mse": ("train", "mse", "min"),
    "validation-mse": ("validation", "mse", "min"),
    "train-correlation": ("train", "correlation", "max"),
    "validation-correlation": ("validation", "correlation", "max"),
    "train-nll": ("trainDistribution", "negativeLogLikelihood", "min"),
    "validation-nll": (
        "validationDistribution", "negativeLogLikelihood", "min"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a fixed-knot conditional next-return density by NLL."
    )
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--pause-file", type=Path)
    return parser.parse_args()


def canonical_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


class DistributionAccumulator:
    def __init__(
        self,
        model: NormalizedGluReturnDensity,
        target_std: float,
        device: torch.device,
        *,
        include_mode: bool,
        cumulative_target_std: float | None = None,
    ) -> None:
        self.model = model
        self.expectation = MetricAccumulator(target_std, device)
        self.per_lead_expectation = tuple(
            MetricAccumulator(target_std, device)
            for _ in range(model.return_count)
        )
        self.cumulative_expectation = MetricAccumulator(
            cumulative_target_std
            if cumulative_target_std is not None
            else target_std * math.sqrt(model.return_count),
            device,
        )
        self.mode = MetricAccumulator(target_std, device) if include_mode else None
        self.values = torch.zeros(7, dtype=torch.float64, device=device)
        self.prior_log_masses = torch.log(model.density_prior_masses)
        prior_expectation, prior_mode = model.prior_point_predictions()
        self.prior_expectation = prior_expectation
        self.prior_mode = prior_mode

    def add(
        self,
        forecast_logits: torch.Tensor,
        targets: torch.Tensor,
        weights: torch.Tensor,
        *,
        joint_logits: torch.Tensor | None = None,
    ) -> None:
        joint_logits = forecast_logits if joint_logits is None else joint_logits
        nll, log_masses, log_density_unit = return_negative_log_likelihood(
            joint_logits,
            targets,
            self.model.density_knots_unit,
            self.model.density_basis_areas,
            self.model.density_transform,
        )
        unit, log_jacobian = transform_returns_to_unit(
            targets, self.model.density_transform
        )
        prior_log_masses = self.prior_log_masses[None, :].expand(
            *targets.shape, -1
        )
        prior_log_density_unit = interpolated_log_density_unit(
            prior_log_masses,
            unit,
            self.model.density_knots_unit,
            self.model.density_basis_areas,
        )
        prior_nll = -(prior_log_density_unit + log_jacobian)
        forecast_log_masses = component_log_masses(
            forecast_logits, self.model.density_basis_areas
        )
        expectation, mode = self.model.point_predictions_from_log_masses(
            forecast_log_masses, include_mode=self.mode is not None
        )
        path_targets = targets[:, None] if targets.ndim == 1 else targets
        path_expectation = expectation[:, None] \
            if expectation.ndim == 1 else expectation
        path_weights = weights[:, None].expand_as(path_targets)
        self.expectation.add(
            path_expectation.reshape(-1),
            path_targets.reshape(-1),
            path_weights.reshape(-1),
        )
        for lead, accumulator in enumerate(self.per_lead_expectation):
            accumulator.add(
                path_expectation[:, lead], path_targets[:, lead], weights
            )
        self.cumulative_expectation.add(
            path_expectation.sum(dim=1), path_targets.sum(dim=1), weights
        )
        if self.mode is not None:
            if mode is None:
                raise RuntimeError("mode evaluation returned no prediction")
            path_mode = mode[:, None] if mode.ndim == 1 else mode
            self.mode.add(
                path_mode.reshape(-1),
                path_targets.reshape(-1),
                path_weights.reshape(-1),
            )
        weights64 = weights.double()
        path_nll = nll[:, None] if nll.ndim == 1 else nll
        path_unit_nll = (-log_density_unit)[:, None] \
            if log_density_unit.ndim == 1 else -log_density_unit
        path_prior_nll = prior_nll[:, None] \
            if prior_nll.ndim == 1 else prior_nll
        path_prior_unit_nll = (-prior_log_density_unit)[:, None] \
            if prior_log_density_unit.ndim == 1 else -prior_log_density_unit
        return_weights64 = weights64[:, None].expand_as(path_nll)
        self.values += torch.stack((
            weights64.sum(),
            return_weights64.sum(),
            (return_weights64 * path_nll.double()).sum(),
            (return_weights64 * path_unit_nll.double()).sum(),
            (return_weights64 * path_prior_nll.double()).sum(),
            (return_weights64 * path_prior_unit_nll.double()).sum(),
            (weights64 * path_nll.double().sum(dim=1)).sum(),
        ))

    def result(self) -> dict:
        (
            path_weight, return_weight, nll, unit_nll,
            prior_nll, prior_unit_nll, path_nll,
        ) = (
            float(value) for value in self.values
        )
        if path_weight <= 0 or return_weight <= 0:
            raise RuntimeError("cannot finalize empty density metrics")
        nll /= return_weight
        unit_nll /= return_weight
        prior_nll /= return_weight
        prior_unit_nll /= return_weight
        path_nll /= path_weight
        prior_expectation = self.prior_expectation
        prior_mode = self.prior_mode
        if isinstance(prior_expectation, torch.Tensor):
            prior_expectation = prior_expectation.detach().cpu().tolist()
        if isinstance(prior_mode, torch.Tensor):
            prior_mode = prior_mode.detach().cpu().tolist()
        return {
            "examples": int(round(path_weight)),
            "targetReturns": int(round(return_weight)),
            "returnCount": self.model.return_count,
            "negativeLogLikelihood": nll,
            "pathNegativeLogLikelihood": path_nll,
            "unitNegativeLogLikelihood": unit_nll,
            "bitsPerExample": nll / math.log(2),
            "globalBaselineNegativeLogLikelihood": prior_nll,
            "globalBaselineUnitNegativeLogLikelihood": prior_unit_nll,
            "nllImprovementVsGlobal": prior_nll - nll,
            "expectation": self.expectation.result(),
            "perLeadExpectation": [
                value.result() for value in self.per_lead_expectation
            ],
            "cumulativeExpectation": self.cumulative_expectation.result(),
            "mode": self.mode.result() if self.mode is not None else None,
            "globalPointPredictions": {
                "expectation": prior_expectation,
                "mode": prior_mode,
            },
        }


@torch.no_grad()
def evaluate(
    model: NormalizedGluReturnDensity,
    dataset: NextReturnDataset | ActiveReturnPathDataset,
    split: str,
    *,
    batch_size: int,
    target_std: float,
    device: torch.device,
    include_mode: bool = False,
    cumulative_target_std: float | None = None,
) -> dict:
    model.eval()
    metrics = DistributionAccumulator(
        model,
        target_std,
        device,
        include_mode=include_mode,
        cumulative_target_std=cumulative_target_std,
    )
    for features, targets, weights in iter_device_batches(
        dataset.iter_batches(
            split,
            batch_size,
            shuffle=False,
            seed=0,
            reuse_buffers=True,
        ),
        device,
    ):
        forecast_logits = model.raw_density_logits(features)
        joint_logits = model.teacher_forced_density_logits(features, targets)
        metrics.add(
            forecast_logits,
            targets,
            weights,
            joint_logits=joint_logits,
        )
    return metrics.result()


def weighted_nll(
    model: NormalizedGluReturnDensity,
    features: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    nll, _log_masses, _log_density_unit = return_negative_log_likelihood(
        model.teacher_forced_density_logits(features, targets),
        targets,
        model.density_knots_unit,
        model.density_basis_areas,
        model.density_transform,
    )
    if nll.ndim == 1:
        return (nll * weights).sum() / weights.sum()
    return (nll * weights[:, None]).sum() / (
        weights.sum() * nll.shape[1]
    )


def selection_policy_values(
    train_metrics: dict,
    validation_metrics: dict,
) -> dict[str, float]:
    sources = {
        "train": train_metrics["expectation"],
        "validation": validation_metrics["expectation"],
        "trainDistribution": train_metrics,
        "validationDistribution": validation_metrics,
    }
    return {
        policy: float(sources[source][metric])
        for policy, (source, metric, _direction) in SELECTION_POLICIES.items()
    }


def policy_improved(policy: str, score: float, previous: float) -> bool:
    direction = SELECTION_POLICIES[policy][2]
    return score < previous if direction == "min" else score > previous


def density_training_normalization(
    dataset: NextReturnDataset | ActiveReturnPathDataset,
    *,
    batch_size: int,
) -> dict[str, np.ndarray | float]:
    normalization = training_normalization(dataset, batch_size=batch_size)
    return_sum = 0.0
    return_square_sum = 0.0
    return_weight = 0.0
    for _features, targets, weights in dataset.iter_batches(
        "train", batch_size, shuffle=False, seed=0, reuse_buffers=True
    ):
        values = targets.numpy().astype(np.float64, copy=False)
        if values.ndim == 1:
            values = values[:, None]
        sample_weights = weights.numpy().astype(np.float64, copy=False)
        expanded = sample_weights[:, None]
        return_sum += float((expanded * values).sum(dtype=np.float64))
        return_square_sum += float(
            (expanded * np.square(values)).sum(dtype=np.float64)
        )
        return_weight += float(sample_weights.sum()) * values.shape[1]
    mean = return_sum / return_weight
    variance = max(return_square_sum / return_weight - mean * mean, 1e-14)
    return {**normalization, "returnMean": mean, "returnStd": math.sqrt(variance)}


def evaluate_selection_checkpoints(
    repo: Path,
    run_root: Path,
    model: NormalizedGluReturnDensity,
    train_dataset: NextReturnDataset | ActiveReturnPathDataset,
    validation_dataset: NextReturnDataset | ActiveReturnPathDataset,
    test_dataset: NextReturnDataset | ActiveReturnPathDataset,
    *,
    plan_id: str,
    plan_hash: str,
    batch_size: int,
    target_std: float,
    cumulative_target_std: float,
    device: torch.device,
) -> dict:
    policies: dict[str, dict] = {}
    for policy in SELECTION_POLICIES:
        file = run_root / f"checkpoints/selections/{policy}.json"
        checkpoint = load_torch_checkpoint(
            file, map_location=device, weights_only=False
        )
        if checkpoint.get("planSha256") != plan_hash:
            raise ValueError(f"selection checkpoint belongs to another plan: {policy}")
        model.load_state_dict(checkpoint["model"])
        train_metrics = evaluate(
            model, train_dataset, "train", batch_size=batch_size,
            target_std=target_std, device=device,
            cumulative_target_std=cumulative_target_std,
        )
        validation_metrics = evaluate(
            model, validation_dataset, "validation", batch_size=batch_size,
            target_std=target_std, device=device,
            cumulative_target_std=cumulative_target_std,
        )
        test_metrics = evaluate(
            model, test_dataset, "test", batch_size=batch_size,
            target_std=target_std, device=device,
            cumulative_target_std=cumulative_target_std,
        )
        policies[policy] = {
            "label": policy.replace("-", " ").title(),
            "epoch": int(checkpoint["epoch"]),
            "selectionScore": float(checkpoint["score"]),
            "recoveredSelectionScore": float(checkpoint["score"]),
            "replayAbsoluteDifference": 0.0,
            "selection": checkpoint["selection"],
            "train": train_metrics["expectation"],
            "validation": validation_metrics["expectation"],
            "test": test_metrics["expectation"],
            "distribution": {
                "train": train_metrics,
                "validation": validation_metrics,
                "test": test_metrics,
            },
            "checkpoint": str(file.relative_to(repo)),
        }
    comparison = {
        "contract": "six-policy-density-path-checkpoint-comparison-v1",
        "planId": plan_id,
        "planSha256": plan_hash,
        "policies": policies,
    }
    atomic_json(
        comparison, run_root / "state/checkpoint-selection-comparison.json"
    )
    return comparison


def main() -> None:
    args = parse_args()
    repo = Path(__file__).resolve().parents[1]
    plan_file = (repo / args.plan).resolve() if not args.plan.is_absolute() \
        else args.plan.resolve()
    plan = json.loads(plan_file.read_text(encoding="utf-8"))
    plan_hash = canonical_hash(plan)
    run_root = (repo / plan["runDir"]).resolve()
    history_root = (repo / plan["historyDir"]).resolve()
    pause_file = None if args.pause_file is None else (
        (repo / args.pause_file).resolve()
        if not args.pause_file.is_absolute() else args.pause_file.resolve()
    )
    reporter = Reporter(run_root)
    try:
        seed = int(plan["training"]["seed"])
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        device = torch.device(plan["training"]["device"])
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")

        examples = int(plan["subset"]["examples"])
        heldout_examples = int(plan["evaluation"]["examplesPerSplit"])
        return_count = int(plan["density"].get("returnCount", 1))
        runner_contract = SINGLE_RETURN_RUNNER_CONTRACT \
            if return_count == 1 else PATH_RUNNER_CONTRACT
        train_shards = fixed_nonzero_subset_shards(
            history_root, date.fromisoformat(plan["subset"]["date"]), examples
        )
        validation_shards = fixed_nonzero_subset_shards(
            history_root,
            date.fromisoformat(plan["evaluation"]["validationStart"]),
            heldout_examples,
        )
        test_shards = fixed_nonzero_subset_shards(
            history_root,
            date.fromisoformat(plan["evaluation"]["testStart"]),
            heldout_examples,
        )
        validation_shards = {
            "train": [],
            "validation": validation_shards["train"],
            "test": [],
        }
        test_shards = {
            "train": [],
            "validation": [],
            "test": test_shards["train"],
        }
        dataset_type = NextReturnDataset if return_count == 1 \
            else ActiveReturnPathDataset
        dataset_kwargs = {"exclude_zero_targets": True} \
            if return_count == 1 else {"return_count": return_count}
        train_dataset = dataset_type(
            train_shards, history_root, **dataset_kwargs
        )
        validation_dataset = dataset_type(
            validation_shards, history_root, **dataset_kwargs
        )
        test_dataset = dataset_type(
            test_shards, history_root, **dataset_kwargs
        )
        counts = {
            "train": train_dataset.logical_count("train"),
            "validation": validation_dataset.logical_count("validation"),
            "test": test_dataset.logical_count("test"),
        }
        if counts != {
            "train": examples,
            "validation": heldout_examples,
            "test": heldout_examples,
        }:
            raise RuntimeError(f"density dataset counts changed: {counts}")
        reporter.emit({
            "event": "minute-return-dataset-selected",
            "planId": plan["id"],
            "counts": counts,
            "datasetFilter": plan["datasetFilter"],
        })

        snapshot = {"planSha256": plan_hash, "plan": plan}
        snapshot_file = run_root / "state/plan.json"
        if snapshot_file.is_file():
            if json.loads(snapshot_file.read_text(encoding="utf-8")) != snapshot:
                raise ValueError("run directory belongs to a different plan")
        else:
            atomic_json(snapshot, snapshot_file)

        training = plan["training"]
        reporter.status("computing-training-statistics", planId=plan["id"])
        normalization = density_training_normalization(
            train_dataset, batch_size=int(training["evaluationBatchSize"])
        )
        target_std = float(normalization["returnStd"])
        cumulative_target_std = float(normalization["minuteStd"])
        density_file = (repo / plan["density"]["source"]).resolve()
        density = KnotDensityContract.load(
            density_file, fit=str(plan["density"]["fit"])
        )
        architecture = plan["architecture"]
        model = NormalizedGluReturnDensity(
            torch.from_numpy(normalization["featureMean"]),
            torch.from_numpy(normalization["featureStd"]),
            density,
            return_count=return_count,
            widths=tuple(int(value) for value in architecture["widths"]),
            dropout=float(architecture["dropout"]),
            dropout_rate=float(architecture["dropoutRate"]),
            initial_radius=float(architecture["initialRadius"]),
            minimum_radius=float(architecture["minimumRadius"]),
            learnable_centering=bool(architecture["learnableCentering"]),
        ).to(device)
        parameter_count = sum(value.numel() for value in model.parameters())
        trainable_parameter_count = sum(
            value.numel() for value in model.parameters()
            if value.requires_grad
        )
        optimizers = build_optimizers(model, training, device)
        maximum_epochs = int(training["epochs"])
        batch_size = int(training["batchSize"])
        evaluation_batch_size = int(training["evaluationBatchSize"])
        last_file = run_root / "checkpoints/last.json"
        best_file = run_root / "checkpoints/best.json"
        start_epoch = 0
        global_step = 0
        best_validation_nll = math.inf
        best_epoch = -1
        selection_best: dict[str, dict[str, float | int]] = {}
        selection_root = run_root / "checkpoints/selections"
        for policy, (_source, _metric, direction) in SELECTION_POLICIES.items():
            selection_file = selection_root / f"{policy}.json"
            if checkpoint_exists(selection_file):
                selected = load_torch_checkpoint(
                    selection_file, map_location="cpu", weights_only=False
                )
                if selected.get("planSha256") == plan_hash:
                    selection_best[policy] = {
                        "score": float(selected["score"]),
                        "epoch": int(selected["epoch"]),
                    }
                    continue
            selection_best[policy] = {
                "score": math.inf if direction == "min" else -math.inf,
                "epoch": -1,
            }
        if checkpoint_exists(last_file):
            checkpoint = load_torch_checkpoint(
                last_file, map_location=device, weights_only=False
            )
            if checkpoint.get("planSha256") != plan_hash \
                    or checkpoint.get("runnerContract") != runner_contract:
                raise ValueError("density checkpoint contract changed")
            model.load_state_dict(checkpoint["model"])
            for optimizer, state in zip(
                optimizers, checkpoint["optimizers"], strict=True
            ):
                optimizer.load_state_dict(state)
            start_epoch = int(checkpoint["epoch"]) + 1
            global_step = int(checkpoint["globalStep"])
            best_validation_nll = float(checkpoint["bestValidationNll"])
            best_epoch = int(checkpoint["bestEpoch"])

        reporter.emit({
            "event": "training-start",
            "planId": plan["id"],
            "startEpoch": start_epoch,
            "epochs": maximum_epochs,
            "parameters": parameter_count,
            "objective": "conditional-return-negative-log-likelihood",
            "returnCount": return_count,
        })
        reporter.status(
            "training",
            planId=plan["id"],
            startEpoch=start_epoch,
            parameters=parameter_count,
            bestEpoch=best_epoch,
        )
        started = time.monotonic()
        for epoch in range(start_epoch, maximum_epochs):
            if pause_file is not None and pause_file.is_file():
                reporter.status(
                    "paused", planId=plan["id"], epoch=epoch,
                    bestEpoch=best_epoch, bestValidationNll=best_validation_nll,
                )
                raise SystemExit(PAUSE_EXIT_CODE)
            model.train()
            online_numerator = 0.0
            online_denominator = 0.0
            for features, targets, weights in iter_device_batches(
                train_dataset.iter_batches(
                    "train",
                    batch_size,
                    shuffle=True,
                    shuffle_rows=True,
                    seed=seed + epoch,
                    reuse_buffers=True,
                ),
                device,
            ):
                for optimizer in optimizers:
                    optimizer.zero_grad(set_to_none=True)
                loss = weighted_nll(model, features, targets, weights)
                loss.backward()
                clip_grad_norm_(
                    model.parameters(),
                    float(training["gradientClip"]),
                    foreach=device.type == "cuda",
                )
                for optimizer in optimizers:
                    optimizer.step()
                batch_weight = float(weights.sum())
                online_numerator += float(loss.detach()) * batch_weight
                online_denominator += batch_weight
                global_step += 1

            train_metrics = evaluate(
                model,
                train_dataset,
                "train",
                batch_size=evaluation_batch_size,
                target_std=target_std,
                cumulative_target_std=cumulative_target_std,
                device=device,
            )
            validation_metrics = evaluate(
                model,
                validation_dataset,
                "validation",
                batch_size=evaluation_batch_size,
                target_std=target_std,
                cumulative_target_std=cumulative_target_std,
                device=device,
            )
            score = float(validation_metrics["negativeLogLikelihood"])
            if not math.isfinite(score):
                raise FloatingPointError("validation NLL is non-finite")
            improved = score < best_validation_nll
            if improved:
                best_validation_nll = score
                best_epoch = epoch
            policy_values = selection_policy_values(
                train_metrics, validation_metrics
            )
            for policy, policy_score in policy_values.items():
                previous = float(selection_best[policy]["score"])
                if not policy_improved(policy, policy_score, previous):
                    continue
                selection_best[policy] = {
                    "score": policy_score,
                    "epoch": epoch,
                }
                save_torch_checkpoint({
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "score": policy_score,
                    "policy": policy,
                    "selection": {
                        "source": SELECTION_POLICIES[policy][0],
                        "metric": SELECTION_POLICIES[policy][1],
                        "direction": SELECTION_POLICIES[policy][2],
                    },
                    "trainDistribution": train_metrics,
                    "validationDistribution": validation_metrics,
                    "planSha256": plan_hash,
                    "runnerContract": runner_contract,
                }, selection_root / f"{policy}.json")
            checkpoint = {
                "model": model.state_dict(),
                "optimizers": [value.state_dict() for value in optimizers],
                "epoch": epoch,
                "globalStep": global_step,
                "bestEpoch": best_epoch,
                "bestValidationNll": best_validation_nll,
                "trainDistribution": train_metrics,
                "validationDistribution": validation_metrics,
                "planSha256": plan_hash,
                "runnerContract": runner_contract,
                "selectionPolicies": selection_best,
            }
            save_torch_checkpoint(checkpoint, last_file)
            if improved:
                save_torch_checkpoint(checkpoint, best_file)
            event = {
                "event": "minute-return-epoch",
                "epoch": epoch,
                "epochs": maximum_epochs,
                "globalStep": global_step,
                "seconds": time.monotonic() - started,
                # Expectation is the distribution's canonical MSE point
                # prediction, so expose it through the existing comparison UI.
                "train": train_metrics["expectation"],
                "validation": validation_metrics["expectation"],
                "trainDistribution": train_metrics,
                "validationDistribution": validation_metrics,
                "onlineNegativeLogLikelihood": (
                    online_numerator / online_denominator
                ),
                "bestValidationNll": best_validation_nll,
                "bestTrainScore": best_validation_nll,
                "bestEpoch": best_epoch,
                "improved": improved,
                "learningRate": float(optimizers[0].param_groups[0]["lr"]),
            }
            reporter.emit(event)
            reporter.status("training", planId=plan["id"], latest=event)

        best = load_torch_checkpoint(
            best_file, map_location=device, weights_only=False
        )
        model.load_state_dict(best["model"])
        train_metrics = evaluate(
            model, train_dataset, "train",
            batch_size=evaluation_batch_size,
            target_std=target_std, device=device,
            cumulative_target_std=cumulative_target_std,
            include_mode=True,
        )
        validation_metrics = evaluate(
            model, validation_dataset, "validation",
            batch_size=evaluation_batch_size,
            target_std=target_std, device=device,
            cumulative_target_std=cumulative_target_std,
            include_mode=True,
        )
        test_metrics = evaluate(
            model, test_dataset, "test",
            batch_size=evaluation_batch_size,
            target_std=target_std, device=device,
            cumulative_target_std=cumulative_target_std,
            include_mode=True,
        )
        result = {
            "version": 1,
            "planId": plan["id"],
            "planSha256": plan_hash,
            "runnerContract": runner_contract,
            "examples": examples,
            "counts": counts,
            "parameterCount": parameter_count,
            "trainableParameterCount": trainable_parameter_count,
            "bestEpoch": int(best["bestEpoch"]),
            "bestValidationNll": float(best["bestValidationNll"]),
            "train": train_metrics["expectation"],
            "validation": validation_metrics["expectation"],
            "test": test_metrics["expectation"],
            "pointPredictions": {
                "expectation": {
                    "train": train_metrics["expectation"],
                    "validation": validation_metrics["expectation"],
                    "test": test_metrics["expectation"],
                },
                "mode": {
                    "train": train_metrics["mode"],
                    "validation": validation_metrics["mode"],
                    "test": test_metrics["mode"],
                },
            },
            "distribution": {
                "train": train_metrics,
                "validation": validation_metrics,
                "test": test_metrics,
            },
            "densityContract": {
                "source": plan["density"]["source"],
                "fit": density.source_fit,
                "knotCount": int(density.knots_unit.size),
                "returnCount": return_count,
                "outputCount": int(density.knots_unit.size) * return_count,
                "jointFactorization": (
                    "autoregressive chain product of per-lead knot densities"
                ),
                "knotsStoredInModelState": True,
                "transform": {
                    "alpha": density.transform.alpha,
                    "locationBps": density.transform.location_bps,
                    "scaleBps": density.transform.scale_bps,
                },
                "output": (
                    "normalized knot heights h_i=exp(logit_i)/"
                    "sum_j(A_j*exp(logit_j)); q_i=A_i*h_i is retained only "
                    "as the equivalent component-mass representation"
                ),
                "trainingLoss": (
                    "teacher-forced mean per-return joint NLL; path NLL is "
                    "the sum of causal conditional NLLs across leads"
                ),
                "pointForecast": (
                    "recursive conditional expectation without realized-prefix "
                    "access"
                ),
            },
            "checkpoint": str(best_file.relative_to(repo)),
            "selectionPolicies": selection_best,
        }
        atomic_json(result, run_root / "state/result.json")
        reporter.status(
            "evaluating-checkpoint-selections",
            planId=plan["id"],
            bestEpoch=int(best["bestEpoch"]),
        )
        evaluate_selection_checkpoints(
            repo,
            run_root,
            model,
            train_dataset,
            validation_dataset,
            test_dataset,
            plan_id=plan["id"],
            plan_hash=plan_hash,
            batch_size=evaluation_batch_size,
            target_std=target_std,
            cumulative_target_std=cumulative_target_std,
            device=device,
        )
        reporter.emit({"event": "minute-return-complete", **result})
        reporter.status("complete", planId=plan["id"], latest=result)
    except SystemExit:
        raise
    except BaseException as error:
        reporter.status(
            "failed", planId=plan.get("id", "unknown"),
            error=f"{type(error).__name__}: {error}",
        )
        raise


if __name__ == "__main__":
    main()
