from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

from normalized_glu_return_density import NormalizedGluReturnDensity
from return_knot_density import KnotDensityContract
from trading_storage import load_torch_checkpoint, save_torch_checkpoint
from train_autoregressive_minute_return import build_optimizers
from train_feature_augmented_next_return import FeatureMatrixDataset, normalization
from train_next_return_knot_density import (
    DistributionAccumulator,
    SELECTION_POLICIES,
    policy_improved,
    selection_policy_values,
    weighted_nll,
)
from train_normalized_glu_next_return import Reporter, atomic_json


RUNNER_CONTRACT = "feature-augmented-next-second-fixed-knot-density-nll-v1"
PAUSE_EXIT_CODE = 75


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a fixed-knot next-return density from a feature matrix."
    )
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--pause-file", type=Path)
    return parser.parse_args()


def canonical_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def validate_plan(plan: dict) -> None:
    required = (
        "id", "label", "datasetDir", "runDir", "density",
        "architecture", "training",
    )
    if any(key not in plan for key in required):
        raise ValueError("feature-density training plan is incomplete")
    if int(plan["density"].get("returnCount", 1)) != 1:
        raise ValueError("this feature-density trainer predicts one return")
    if int(plan["density"]["knotCount"]) < 3:
        raise ValueError("density requires at least three knots")
    if plan["training"]["device"] != "cuda":
        raise ValueError("feature-density experiment requires CUDA")


@torch.no_grad()
def evaluate(
    model: NormalizedGluReturnDensity,
    dataset: FeatureMatrixDataset,
    split: str,
    *,
    batch_size: int,
    target_std: float,
    device: torch.device,
    include_mode: bool = False,
) -> dict:
    model.eval()
    metrics = DistributionAccumulator(
        model, target_std, device, include_mode=include_mode
    )
    for features, targets, weights in dataset.iter_batches(
        split, batch_size, shuffle=False, seed=0
    ):
        features = features.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        weights = weights.to(device, non_blocking=True)
        logits = model.raw_density_logits(features)
        metrics.add(logits, targets, weights, joint_logits=logits)
    return metrics.result()


def checkpoint_payload(
    model: NormalizedGluReturnDensity,
    optimizers: tuple[torch.optim.Optimizer, ...],
    *,
    epoch: int,
    global_step: int,
    train_distribution: dict,
    validation_distribution: dict,
    parameter_count: int,
    plan_hash: str,
    dataset_hash: str,
) -> dict:
    return {
        "model": model.state_dict(),
        "optimizers": [optimizer.state_dict() for optimizer in optimizers],
        "epoch": epoch,
        "globalStep": global_step,
        "trainDistribution": train_distribution,
        "validationDistribution": validation_distribution,
        "parameterCount": parameter_count,
        "planSha256": plan_hash,
        "datasetSha256": dataset_hash,
        "runnerContract": RUNNER_CONTRACT,
    }


def main() -> None:
    args = parse_args()
    repo = Path(__file__).resolve().parents[1]
    plan_file = (repo / args.plan).resolve() \
        if not args.plan.is_absolute() else args.plan.resolve()
    plan = json.loads(plan_file.read_text("utf-8"))
    validate_plan(plan)
    dataset_root = (repo / plan["datasetDir"]).resolve()
    run_root = (repo / plan["runDir"]).resolve()
    pause_file = None if args.pause_file is None else (
        (repo / args.pause_file).resolve()
        if not args.pause_file.is_absolute() else args.pause_file.resolve()
    )
    reporter = Reporter(run_root)
    plan_hash = canonical_hash(plan)
    dataset_hash = hashlib.sha256(
        (dataset_root / "manifest.json").read_bytes()
    ).hexdigest()
    run_root.mkdir(parents=True, exist_ok=True)
    snapshot = {"plan": plan, "planSha256": plan_hash}
    snapshot_file = run_root / "state/plan.json"
    if snapshot_file.is_file():
        if json.loads(snapshot_file.read_text("utf-8")) != snapshot:
            raise ValueError("run directory belongs to a different plan")
    else:
        atomic_json(snapshot, snapshot_file)
    reporter.status("loading-data", planId=plan["id"])
    try:
        dataset = FeatureMatrixDataset(dataset_root)
        counts = {
            split: dataset.logical_count(split)
            for split in ("train", "validation", "test")
        }
        stats = normalization(dataset)
        training = plan["training"]
        architecture = plan["architecture"]
        density_plan = plan["density"]
        density = KnotDensityContract.load(
            (repo / density_plan["source"]).resolve(),
            fit=str(density_plan["fit"]),
        )
        if density.knots_unit.size != int(density_plan["knotCount"]):
            raise ValueError("loaded density knot count differs from the plan")
        device = torch.device("cuda")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        seed = int(training["seed"])
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.set_float32_matmul_precision("high")
        model = NormalizedGluReturnDensity(
            torch.from_numpy(stats["featureMean"]),
            torch.from_numpy(stats["featureStd"]),
            density,
            return_count=1,
            widths=tuple(int(value) for value in architecture["widths"]),
            learnable_centering=bool(architecture["learnableCentering"]),
            dropout=float(architecture["dropout"]),
            dropout_rate=float(architecture["dropoutRate"]),
            initial_radius=float(architecture["initialRadius"]),
            minimum_radius=float(architecture["minimumRadius"]),
        ).to(device)
        optimizers = build_optimizers(model, training, device)
        parameter_count = sum(value.numel() for value in model.parameters())
        trainable_parameter_count = sum(
            value.numel() for value in model.parameters() if value.requires_grad
        )
        batch_size = int(training["batchSize"])
        evaluation_batch_size = int(training["evaluationBatchSize"])
        epochs = int(training["epochs"])
        target_std = float(stats["targetStd"])
        gradient_clip = float(training["gradientClip"])
        checkpoints = run_root / "checkpoints"
        selection_root = checkpoints / "selections"
        selection_root.mkdir(parents=True, exist_ok=True)
        last_file = checkpoints / "last.json"
        best_file = checkpoints / "best.json"
        selection_scores = {
            policy: math.inf if direction == "min" else -math.inf
            for policy, (_source, _metric, direction) in SELECTION_POLICIES.items()
        }
        selection_epochs = {policy: -1 for policy in SELECTION_POLICIES}
        start_epoch = 0
        global_step = 0
        if last_file.is_file():
            saved = load_torch_checkpoint(
                last_file, map_location=device, weights_only=False
            )
            if saved.get("planSha256") != plan_hash \
                    or saved.get("datasetSha256") != dataset_hash \
                    or saved.get("runnerContract") != RUNNER_CONTRACT:
                raise ValueError("existing checkpoint does not match this run")
            model.load_state_dict(saved["model"])
            for optimizer, state in zip(
                optimizers, saved["optimizers"], strict=True
            ):
                optimizer.load_state_dict(state)
            start_epoch = int(saved["epoch"]) + 1
            global_step = int(saved["globalStep"])
            selection_scores.update(saved.get("selectionScores", {}))
            selection_epochs.update(saved.get("selectionEpochs", {}))
        reporter.emit({
            "event": "training-start",
            "planId": plan["id"],
            "startEpoch": start_epoch,
            "epochs": epochs,
            "parameters": parameter_count,
            "featureCount": dataset.feature_count,
            "knotCount": model.knot_count,
            "objective": "conditional-return-negative-log-likelihood",
        })
        reporter.status(
            "training", planId=plan["id"], epochs=epochs,
            startEpoch=start_epoch, featureCount=dataset.feature_count,
            examples=counts["train"], parameterCount=parameter_count,
        )
        started = time.monotonic()
        for epoch in range(start_epoch, epochs):
            if pause_file is not None and pause_file.is_file():
                reporter.status(
                    "paused", planId=plan["id"], epoch=epoch,
                    selectionEpochs=selection_epochs,
                )
                raise SystemExit(PAUSE_EXIT_CODE)
            model.train()
            online_numerator = 0.0
            online_denominator = 0.0
            for features, targets, weights in dataset.iter_batches(
                "train", batch_size, shuffle=True, seed=seed + epoch
            ):
                features = features.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                weights = weights.to(device, non_blocking=True)
                for optimizer in optimizers:
                    optimizer.zero_grad(set_to_none=True)
                loss = weighted_nll(model, features, targets, weights)
                loss.backward()
                clip_grad_norm_(
                    model.parameters(), gradient_clip, foreach=True
                )
                for optimizer in optimizers:
                    optimizer.step()
                batch_weight = float(weights.sum())
                online_numerator += float(loss.detach()) * batch_weight
                online_denominator += batch_weight
                global_step += 1
            train_distribution = evaluate(
                model, dataset, "train", batch_size=evaluation_batch_size,
                target_std=target_std, device=device,
            )
            validation_distribution = evaluate(
                model, dataset, "validation", batch_size=evaluation_batch_size,
                target_std=target_std, device=device,
            )
            payload = checkpoint_payload(
                model, optimizers, epoch=epoch, global_step=global_step,
                train_distribution=train_distribution,
                validation_distribution=validation_distribution,
                parameter_count=parameter_count, plan_hash=plan_hash,
                dataset_hash=dataset_hash,
            )
            candidate_scores = selection_policy_values(
                train_distribution, validation_distribution
            )
            for policy, score in candidate_scores.items():
                if not policy_improved(
                    policy, score, float(selection_scores[policy])
                ):
                    continue
                selection_scores[policy] = score
                selection_epochs[policy] = epoch
                selected = {
                    **payload,
                    "score": score,
                    "policy": policy,
                    "selection": {
                        "source": SELECTION_POLICIES[policy][0],
                        "metric": SELECTION_POLICIES[policy][1],
                        "direction": SELECTION_POLICIES[policy][2],
                    },
                }
                save_torch_checkpoint(
                    selected, selection_root / f"{policy}.json"
                )
                if policy == "validation-nll":
                    save_torch_checkpoint(selected, best_file)
            payload["selectionScores"] = selection_scores
            payload["selectionEpochs"] = selection_epochs
            save_torch_checkpoint(payload, last_file)
            event = {
                "event": "minute-return-epoch",
                "epoch": epoch,
                "epochs": epochs,
                "seconds": time.monotonic() - started,
                "globalStep": global_step,
                "train": train_distribution["expectation"],
                "validation": validation_distribution["expectation"],
                "trainDistribution": train_distribution,
                "validationDistribution": validation_distribution,
                "onlineNegativeLogLikelihood": (
                    online_numerator / online_denominator
                ),
                "bestValidationNll": selection_scores["validation-nll"],
                "bestTrainScore": selection_scores["train-nll"],
                "bestEpoch": selection_epochs["validation-nll"],
                "parameterCount": parameter_count,
                "featureCount": dataset.feature_count,
                "knotCount": model.knot_count,
            }
            reporter.emit(event)
            reporter.status("training", planId=plan["id"], latest=event)

        reporter.status(
            "evaluating-checkpoint-selections", planId=plan["id"],
            completedPolicies=0, totalPolicies=len(SELECTION_POLICIES),
        )
        policies: dict[str, dict] = {}
        for index, policy in enumerate(SELECTION_POLICIES):
            file = selection_root / f"{policy}.json"
            saved = load_torch_checkpoint(
                file, map_location=device, weights_only=False
            )
            model.load_state_dict(saved["model"])
            split_distribution = {
                split: evaluate(
                    model, dataset, split, batch_size=evaluation_batch_size,
                    target_std=target_std, device=device, include_mode=True,
                )
                for split in ("train", "validation", "test")
            }
            policies[policy] = {
                "label": policy.replace("-", " ").title(),
                "epoch": int(saved["epoch"]),
                "selectionScore": float(saved["score"]),
                "selection": saved["selection"],
                "train": split_distribution["train"]["expectation"],
                "validation": split_distribution["validation"]["expectation"],
                "test": split_distribution["test"]["expectation"],
                "distribution": split_distribution,
                "checkpoint": str(file.relative_to(repo)),
            }
            atomic_json({
                "contract": "feature-density-six-policy-comparison-v1",
                "planId": plan["id"],
                "planSha256": plan_hash,
                "policies": policies,
            }, run_root / "state/checkpoint-selection-comparison.json")
            reporter.status(
                "evaluating-checkpoint-selections", planId=plan["id"],
                completedPolicies=index + 1,
                totalPolicies=len(SELECTION_POLICIES),
            )
        selected = policies["validation-nll"]
        result = {
            "version": 1,
            "planId": plan["id"],
            "planSha256": plan_hash,
            "datasetSha256": dataset_hash,
            "runnerContract": RUNNER_CONTRACT,
            "examples": counts["train"],
            "counts": counts,
            "featureCount": dataset.feature_count,
            "parameterCount": parameter_count,
            "trainableParameterCount": trainable_parameter_count,
            "bestEpoch": selected["epoch"],
            "bestValidationNll": selected["selectionScore"],
            "train": selected["train"],
            "validation": selected["validation"],
            "test": selected["test"],
            "pointPredictions": {
                "expectation": {
                    split: selected[split]
                    for split in ("train", "validation", "test")
                },
                "mode": {
                    split: selected["distribution"][split]["mode"]
                    for split in ("train", "validation", "test")
                },
            },
            "distribution": selected["distribution"],
            "densityContract": {
                "source": density_plan["source"],
                "fit": density.source_fit,
                "knotCount": model.knot_count,
                "returnCount": 1,
                "knotsStoredInModelState": True,
                "pointForecast": "conditional distribution expectation",
                "trainingLoss": "return-space negative log likelihood",
            },
            "checkpoint": selected["checkpoint"],
            "selectionPolicies": {
                policy: {
                    "score": selection_scores[policy],
                    "epoch": selection_epochs[policy],
                }
                for policy in SELECTION_POLICIES
            },
            "dataset": {
                "contract": dataset.manifest["contract"],
                "targetFilter": dataset.manifest["targetFilter"],
                "includedSources": dataset.manifest["includedSources"],
                "omittedForCoverage": dataset.manifest["omittedForCoverage"],
            },
        }
        atomic_json(result, run_root / "state/result.json")
        reporter.emit({"event": "minute-return-complete", **result})
        reporter.status("complete", planId=plan["id"], latest=result)
    except SystemExit:
        raise
    except KeyboardInterrupt:
        reporter.status("paused", planId=plan["id"], message="Interrupted")
        raise
    except Exception as error:
        reporter.status(
            "failed", planId=plan["id"],
            error=f"{type(error).__name__}: {error}",
        )
        raise


if __name__ == "__main__":
    main()
