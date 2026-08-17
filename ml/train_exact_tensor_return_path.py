from __future__ import annotations

import argparse
from datetime import date
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

from active_return_path_dataset import (
    ActiveReturnPathDataset,
    fixed_nonzero_candle_subset_shards,
)
from exact_tensor_return_density import (
    ExactTensorReturnDensity,
    exact_tensor_path_log_density,
)
from normalized_glu_next_return import optimizer_parameter_groups
from return_knot_density import KnotDensityContract
from trading_storage import (
    checkpoint_exists,
    load_torch_checkpoint,
    save_torch_checkpoint,
)
from train_autoregressive_minute_return import build_optimizers
from train_next_return_knot_density import (
    PAUSE_EXIT_CODE,
    SELECTION_POLICIES,
    canonical_hash,
    density_training_normalization,
    policy_improved,
)
from train_next_return_memorization import fixed_nonzero_subset_shards
from train_normalized_glu_next_return import (
    MetricAccumulator,
    Reporter,
    atomic_json,
    iter_device_batches,
)


RUNNER_CONTRACT = "exact-three-return-32-knot-joint-tensor-nll-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an exact three-return 32-knot joint tensor."
    )
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--pause-file", type=Path)
    return parser.parse_args()


class TensorMetrics:
    def __init__(
        self,
        target_std: float,
        cumulative_std: float,
        device: torch.device,
    ) -> None:
        self.pooled = MetricAccumulator(target_std, device)
        self.per_lead = tuple(MetricAccumulator(target_std, device) for _ in range(3))
        self.cumulative = MetricAccumulator(cumulative_std, device)
        self.values = torch.zeros(3, dtype=torch.float64, device=device)

    def add(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        weights: torch.Tensor,
        log_density_terms: torch.Tensor,
    ) -> None:
        expanded_weights = weights[:, None].expand_as(targets)
        self.pooled.add(
            predictions.reshape(-1), targets.reshape(-1),
            expanded_weights.reshape(-1),
        )
        for lead, accumulator in enumerate(self.per_lead):
            accumulator.add(predictions[:, lead], targets[:, lead], weights)
        self.cumulative.add(
            predictions.sum(dim=1), targets.sum(dim=1), weights
        )
        weights64 = weights.double()
        self.values += torch.stack((
            weights64.sum(),
            (weights64[:, None] * -log_density_terms.double()).sum(),
            (weights64 * -log_density_terms.double().sum(dim=1)).sum(),
        ))

    def result(self) -> dict:
        paths, candle_nll, path_nll = (float(value) for value in self.values)
        if paths <= 0:
            raise RuntimeError("cannot finalize empty exact tensor metrics")
        expectation = self.pooled.result()
        return {
            "examples": int(round(paths)),
            "returnCount": 3,
            "negativeLogLikelihood": candle_nll / (paths * 3),
            "pathNegativeLogLikelihood": path_nll / paths,
            "expectation": expectation,
            "perLeadExpectation": [value.result() for value in self.per_lead],
            "cumulativeExpectation": self.cumulative.result(),
        }


@torch.no_grad()
def evaluate(
    model: ExactTensorReturnDensity,
    dataset: ActiveReturnPathDataset,
    split: str,
    *,
    batch_size: int,
    target_std: float,
    cumulative_std: float,
    device: torch.device,
) -> dict:
    model.eval()
    metrics = TensorMetrics(target_std, cumulative_std, device)
    for features, targets, weights in iter_device_batches(
        dataset.iter_batches(
            split, batch_size, shuffle=False, seed=0, reuse_buffers=True
        ),
        device,
    ):
        output = model(features)
        terms = exact_tensor_path_log_density(output, targets, model)
        metrics.add(output.expectations, targets, weights, terms)
    return metrics.result()


def policy_values(train: dict, validation: dict) -> dict[str, float]:
    sources = {
        "train": train["expectation"],
        "validation": validation["expectation"],
        "trainDistribution": train,
        "validationDistribution": validation,
    }
    return {
        policy: float(sources[source][metric])
        for policy, (source, metric, _direction) in SELECTION_POLICIES.items()
    }


def evaluate_selection_checkpoints(
    repo: Path,
    run_root: Path,
    model: ExactTensorReturnDensity,
    datasets: tuple[
        ActiveReturnPathDataset,
        ActiveReturnPathDataset,
        ActiveReturnPathDataset,
    ],
    *,
    plan_id: str,
    plan_hash: str,
    batch_size: int,
    target_std: float,
    cumulative_std: float,
    device: torch.device,
) -> dict:
    train_dataset, validation_dataset, test_dataset = datasets
    policies: dict[str, dict] = {}
    for policy in SELECTION_POLICIES:
        file = run_root / f"checkpoints/selections/{policy}.json"
        checkpoint = load_torch_checkpoint(
            file, map_location=device, weights_only=False
        )
        model.load_state_dict(checkpoint["model"])
        train = evaluate(
            model, train_dataset, "train", batch_size=batch_size,
            target_std=target_std, cumulative_std=cumulative_std, device=device,
        )
        validation = evaluate(
            model, validation_dataset, "validation", batch_size=batch_size,
            target_std=target_std, cumulative_std=cumulative_std, device=device,
        )
        test = evaluate(
            model, test_dataset, "test", batch_size=batch_size,
            target_std=target_std, cumulative_std=cumulative_std, device=device,
        )
        policies[policy] = {
            "label": policy.replace("-", " ").title(),
            "epoch": int(checkpoint["epoch"]),
            "selectionScore": float(checkpoint["score"]),
            "recoveredSelectionScore": float(checkpoint["score"]),
            "replayAbsoluteDifference": 0.0,
            "selection": checkpoint["selection"],
            "train": train["expectation"],
            "validation": validation["expectation"],
            "test": test["expectation"],
            "distribution": {
                "train": train, "validation": validation, "test": test,
            },
            "checkpoint": str(file.relative_to(repo)),
        }
    comparison = {
        "contract": "six-policy-exact-tensor-checkpoint-comparison-v1",
        "planId": plan_id, "planSha256": plan_hash, "policies": policies,
    }
    atomic_json(
        comparison, run_root / "state/checkpoint-selection-comparison.json"
    )
    return comparison


def main() -> None:
    args = parse_args()
    repo = Path(__file__).resolve().parents[1]
    plan_file = args.plan.resolve() if args.plan.is_absolute() \
        else (repo / args.plan).resolve()
    plan = json.loads(plan_file.read_text(encoding="utf-8"))
    plan_hash = canonical_hash(plan)
    run_root = (repo / plan["runDir"]).resolve()
    history_root = (repo / plan["historyDir"]).resolve()
    pause_file = None if args.pause_file is None else (
        args.pause_file.resolve() if args.pause_file.is_absolute()
        else (repo / args.pause_file).resolve()
    )
    reporter = Reporter(run_root)
    try:
        training = plan["training"]
        seed = int(training["seed"])
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        device = torch.device(training["device"])
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        examples = int(plan["subset"]["examples"])
        heldout = int(plan["evaluation"]["examplesPerSplit"])
        resolution = str(plan.get("candleResolution", "1s"))
        subset_selector = (
            lambda root, start, count: fixed_nonzero_candle_subset_shards(
                root, start, count, resolution=resolution
            )
        ) if resolution != "1s" else fixed_nonzero_subset_shards
        train_shards = subset_selector(
            history_root, date.fromisoformat(plan["subset"]["date"]), examples
        )
        validation = subset_selector(
            history_root,
            date.fromisoformat(plan["evaluation"]["validationStart"]), heldout,
        )["train"]
        test = subset_selector(
            history_root,
            date.fromisoformat(plan["evaluation"]["testStart"]), heldout,
        )["train"]
        train_dataset = ActiveReturnPathDataset(
            train_shards, history_root, return_count=3,
            resolution=resolution,
        )
        validation_dataset = ActiveReturnPathDataset({
            "train": [], "validation": validation, "test": []
        }, history_root, return_count=3, resolution=resolution)
        test_dataset = ActiveReturnPathDataset({
            "train": [], "validation": [], "test": test
        }, history_root, return_count=3, resolution=resolution)
        counts = {
            "train": train_dataset.logical_count("train"),
            "validation": validation_dataset.logical_count("validation"),
            "test": test_dataset.logical_count("test"),
        }
        expected_counts = {
            "train": examples, "validation": heldout, "test": heldout
        }
        if counts != expected_counts:
            raise RuntimeError(f"exact tensor dataset counts changed: {counts}")
        snapshot = {"planSha256": plan_hash, "plan": plan}
        snapshot_file = run_root / "state/plan.json"
        if snapshot_file.is_file():
            if json.loads(snapshot_file.read_text(encoding="utf-8")) != snapshot:
                raise ValueError("run directory belongs to another plan")
        else:
            atomic_json(snapshot, snapshot_file)
        reporter.emit({
            "event": "minute-return-dataset-selected",
            "planId": plan["id"], "counts": counts,
            "datasetFilter": plan["datasetFilter"],
        })
        reporter.status("computing-training-statistics", planId=plan["id"])
        normalization = density_training_normalization(
            train_dataset, batch_size=int(training["evaluationBatchSize"])
        )
        target_std = float(normalization["returnStd"])
        cumulative_std = float(normalization["minuteStd"])
        density = KnotDensityContract.load(
            (repo / plan["density"]["source"]).resolve(),
            fit=str(plan["density"]["fit"]),
        )
        architecture = plan["architecture"]
        model = ExactTensorReturnDensity(
            torch.from_numpy(normalization["featureMean"]),
            torch.from_numpy(normalization["featureStd"]),
            density,
            hidden_width=int(architecture["hiddenWidth"]),
            initial_radius=float(architecture["initialRadius"]),
            minimum_radius=float(architecture["minimumRadius"]),
            learnable_centering=bool(architecture["learnableCentering"]),
        ).to(device)
        parameter_count = sum(value.numel() for value in model.parameters())
        optimizers = build_optimizers(model, training, device)
        maximum_epochs = int(training["epochs"])
        batch_size = int(training["batchSize"])
        evaluation_batch_size = int(training["evaluationBatchSize"])
        last_file = run_root / "checkpoints/last.json"
        best_file = run_root / "checkpoints/best.json"
        selection_root = run_root / "checkpoints/selections"
        selection_best = {
            policy: {
                "score": math.inf if direction == "min" else -math.inf,
                "epoch": -1,
            }
            for policy, (_source, _metric, direction) in SELECTION_POLICIES.items()
        }
        start_epoch = 0
        global_step = 0
        best_validation_nll = math.inf
        best_epoch = -1
        if checkpoint_exists(last_file):
            checkpoint = load_torch_checkpoint(
                last_file, map_location=device, weights_only=False
            )
            if checkpoint.get("planSha256") != plan_hash \
                    or checkpoint.get("runnerContract") != RUNNER_CONTRACT:
                raise ValueError("exact tensor checkpoint contract changed")
            model.load_state_dict(checkpoint["model"])
            for optimizer, state in zip(
                optimizers, checkpoint["optimizers"], strict=True
            ):
                optimizer.load_state_dict(state)
            start_epoch = int(checkpoint["epoch"]) + 1
            global_step = int(checkpoint["globalStep"])
            best_validation_nll = float(checkpoint["bestValidationNll"])
            best_epoch = int(checkpoint["bestEpoch"])
            selection_best = checkpoint["selectionPolicies"]
        reporter.emit({
            "event": "training-start", "planId": plan["id"],
            "startEpoch": start_epoch, "epochs": maximum_epochs,
            "parameters": parameter_count,
            "objective": "exact-three-return-joint-tensor-negative-log-likelihood",
        })
        reporter.status(
            "training", planId=plan["id"], startEpoch=start_epoch,
            parameters=parameter_count, bestEpoch=best_epoch,
        )
        started = time.monotonic()
        for epoch in range(start_epoch, maximum_epochs):
            if pause_file is not None and pause_file.is_file():
                reporter.status(
                    "paused", planId=plan["id"], epoch=epoch,
                    bestEpoch=best_epoch,
                )
                raise SystemExit(PAUSE_EXIT_CODE)
            model.train()
            online_numerator = 0.0
            online_weight = 0.0
            for features, targets, weights in iter_device_batches(
                train_dataset.iter_batches(
                    "train", batch_size, shuffle=True, shuffle_rows=True,
                    seed=seed + epoch, reuse_buffers=True,
                ),
                device,
            ):
                for optimizer in optimizers:
                    optimizer.zero_grad(set_to_none=True)
                terms = exact_tensor_path_log_density(
                    model(features), targets, model
                )
                loss = (
                    weights[:, None] * -terms
                ).sum() / (weights.sum() * 3)
                loss.backward()
                clip_grad_norm_(
                    model.parameters(), float(training["gradientClip"]),
                    foreach=device.type == "cuda",
                )
                for optimizer in optimizers:
                    optimizer.step()
                weight = float(weights.sum())
                online_numerator += float(loss.detach()) * weight
                online_weight += weight
                global_step += 1
            train_metrics = evaluate(
                model, train_dataset, "train",
                batch_size=evaluation_batch_size, target_std=target_std,
                cumulative_std=cumulative_std, device=device,
            )
            validation_metrics = evaluate(
                model, validation_dataset, "validation",
                batch_size=evaluation_batch_size, target_std=target_std,
                cumulative_std=cumulative_std, device=device,
            )
            score = float(validation_metrics["negativeLogLikelihood"])
            improved = score < best_validation_nll
            if improved:
                best_validation_nll = score
                best_epoch = epoch
            values = policy_values(train_metrics, validation_metrics)
            for policy, policy_score in values.items():
                if not policy_improved(
                    policy, policy_score, float(selection_best[policy]["score"])
                ):
                    continue
                selection_best[policy] = {"score": policy_score, "epoch": epoch}
                save_torch_checkpoint({
                    "model": model.state_dict(), "epoch": epoch,
                    "score": policy_score, "policy": policy,
                    "selection": {
                        "source": SELECTION_POLICIES[policy][0],
                        "metric": SELECTION_POLICIES[policy][1],
                        "direction": SELECTION_POLICIES[policy][2],
                    },
                    "planSha256": plan_hash,
                    "runnerContract": RUNNER_CONTRACT,
                }, selection_root / f"{policy}.json")
            checkpoint = {
                "model": model.state_dict(),
                "optimizers": [value.state_dict() for value in optimizers],
                "epoch": epoch, "globalStep": global_step,
                "bestEpoch": best_epoch,
                "bestValidationNll": best_validation_nll,
                "selectionPolicies": selection_best,
                "planSha256": plan_hash,
                "runnerContract": RUNNER_CONTRACT,
            }
            save_torch_checkpoint(checkpoint, last_file)
            if improved:
                save_torch_checkpoint(checkpoint, best_file)
            event = {
                "event": "minute-return-epoch", "epoch": epoch,
                "epochs": maximum_epochs, "globalStep": global_step,
                "seconds": time.monotonic() - started,
                "train": train_metrics["expectation"],
                "validation": validation_metrics["expectation"],
                "trainDistribution": train_metrics,
                "validationDistribution": validation_metrics,
                "onlineNegativeLogLikelihood": online_numerator / online_weight,
                "bestValidationNll": best_validation_nll,
                "bestTrainScore": best_validation_nll,
                "bestEpoch": best_epoch, "improved": improved,
                "learningRate": float(optimizers[0].param_groups[0]["lr"]),
            }
            reporter.emit(event)
            reporter.status("training", planId=plan["id"], latest=event)

        best = load_torch_checkpoint(
            best_file, map_location=device, weights_only=False
        )
        model.load_state_dict(best["model"])
        train_metrics = evaluate(
            model, train_dataset, "train", batch_size=evaluation_batch_size,
            target_std=target_std, cumulative_std=cumulative_std, device=device,
        )
        validation_metrics = evaluate(
            model, validation_dataset, "validation",
            batch_size=evaluation_batch_size, target_std=target_std,
            cumulative_std=cumulative_std, device=device,
        )
        test_metrics = evaluate(
            model, test_dataset, "test", batch_size=evaluation_batch_size,
            target_std=target_std, cumulative_std=cumulative_std, device=device,
        )
        result = {
            "version": 1, "planId": plan["id"], "planSha256": plan_hash,
            "runnerContract": RUNNER_CONTRACT, "examples": examples,
            "counts": counts, "parameterCount": parameter_count,
            "bestEpoch": int(best["bestEpoch"]),
            "bestValidationNll": float(best["bestValidationNll"]),
            "train": train_metrics["expectation"],
            "validation": validation_metrics["expectation"],
            "test": test_metrics["expectation"],
            "distribution": {
                "train": train_metrics, "validation": validation_metrics,
                "test": test_metrics,
            },
            "densityContract": {
                "knotCount": 32, "returnCount": 3,
                "jointTensorShape": [32, 32, 32],
                "conditionalTensorEntries": [32, 1024, 32768],
                "factorization": "exact-full-prefix-chain-rule-tensors",
                "propagation": "p1-to-joint2-to-joint3-without-sampling",
                "trainingLoss": "mean conditional NLL across three returns",
            },
            "checkpoint": str(best_file.relative_to(repo)),
            "selectionPolicies": selection_best,
        }
        atomic_json(result, run_root / "state/result.json")
        reporter.status(
            "evaluating-checkpoint-selections", planId=plan["id"],
            bestEpoch=int(best["bestEpoch"]),
        )
        evaluate_selection_checkpoints(
            repo, run_root, model,
            (train_dataset, validation_dataset, test_dataset),
            plan_id=plan["id"], plan_hash=plan_hash,
            batch_size=evaluation_batch_size,
            target_std=target_std, cumulative_std=cumulative_std,
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
