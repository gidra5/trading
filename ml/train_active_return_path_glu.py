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
from next_return_sequence import training_sequence_normalization
from normalized_glu_next_return import NormalizedGluNextReturn
from trading_storage import (
    checkpoint_exists,
    load_torch_checkpoint,
    save_torch_checkpoint,
)
from train_next_return_knot_density import (
    PAUSE_EXIT_CODE,
    canonical_hash,
)
from train_next_return_memorization import fixed_nonzero_subset_shards
from train_normalized_glu_next_return import (
    MetricAccumulator,
    Reporter,
    atomic_json,
    build_optimizers,
    iter_device_batches,
)


RUNNER_CONTRACT = "direct-three-active-return-glu-candle-cumulative-mse-v1"
RETURN_COUNT = 3
SELECTION_POLICIES = {
    "train-mse": ("train", "mse", "min"),
    "validation-mse": ("validation", "mse", "min"),
    "train-correlation": ("train", "correlation", "max"),
    "validation-correlation": ("validation", "correlation", "max"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a direct three-output GLU on the same clean active-return "
            "paths used by the exact joint tensor model."
        )
    )
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--pause-file", type=Path)
    return parser.parse_args()


class DirectPathMetrics:
    def __init__(
        self,
        target_std: np.ndarray,
        cumulative_std: float,
        device: torch.device,
    ) -> None:
        pooled_std = float(np.sqrt(np.mean(np.square(target_std))))
        self.pooled = MetricAccumulator(pooled_std, device)
        self.per_lead = tuple(
            MetricAccumulator(float(value), device) for value in target_std
        )
        self.cumulative = MetricAccumulator(cumulative_std, device)
        self.paths = 0

    def add(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        weights: torch.Tensor,
    ) -> None:
        expanded_weights = weights[:, None].expand_as(target)
        self.pooled.add(
            prediction.reshape(-1), target.reshape(-1),
            expanded_weights.reshape(-1),
        )
        for lead, accumulator in enumerate(self.per_lead):
            accumulator.add(prediction[:, lead], target[:, lead], weights)
        self.cumulative.add(prediction.sum(dim=1), target.sum(dim=1), weights)
        self.paths += int(round(float(weights.sum())))

    def result(self) -> dict:
        return {
            "examples": self.paths,
            "returnCount": RETURN_COUNT,
            "expectation": self.pooled.result(),
            "perLeadExpectation": [value.result() for value in self.per_lead],
            "cumulativeExpectation": self.cumulative.result(),
        }


@torch.no_grad()
def evaluate(
    model: NormalizedGluNextReturn,
    dataset: ActiveReturnPathDataset,
    split: str,
    *,
    batch_size: int,
    target_std: np.ndarray,
    cumulative_std: float,
    device: torch.device,
) -> dict:
    model.eval()
    metrics = DirectPathMetrics(target_std, cumulative_std, device)
    for features, targets, weights in iter_device_batches(
        dataset.iter_batches(
            split, batch_size, shuffle=False, seed=0, reuse_buffers=True
        ),
        device,
    ):
        metrics.add(model(features), targets, weights)
    return metrics.result()


def policy_values(train: dict, validation: dict) -> dict[str, float]:
    values = {"train": train["expectation"],
              "validation": validation["expectation"]}
    return {
        policy: float(values[source][metric])
        for policy, (source, metric, _direction) in SELECTION_POLICIES.items()
    }


def policy_improved(policy: str, score: float, previous: float) -> bool:
    return score < previous if SELECTION_POLICIES[policy][2] == "min" \
        else score > previous


def checkpoint_selection_comparison(
    repo: Path,
    run_root: Path,
    model: NormalizedGluNextReturn,
    datasets: tuple[
        ActiveReturnPathDataset,
        ActiveReturnPathDataset,
        ActiveReturnPathDataset,
    ],
    *,
    plan_id: str,
    plan_hash: str,
    batch_size: int,
    target_std: np.ndarray,
    cumulative_std: float,
    device: torch.device,
) -> dict:
    policies: dict[str, dict] = {}
    metrics_by_epoch: dict[int, tuple[dict, dict, dict]] = {}
    for policy, (source, metric, direction) in SELECTION_POLICIES.items():
        file = run_root / f"checkpoints/selections/{policy}.json"
        checkpoint = load_torch_checkpoint(
            file, map_location=device, weights_only=False
        )
        epoch = int(checkpoint["epoch"])
        cached = metrics_by_epoch.get(epoch)
        if cached is None:
            model.load_state_dict(checkpoint["model"])
            cached = tuple(
                evaluate(
                    model, dataset, split, batch_size=batch_size,
                    target_std=target_std, cumulative_std=cumulative_std,
                    device=device,
                )
                for dataset, split in zip(
                    datasets, ("train", "validation", "test"), strict=True
                )
            )
            metrics_by_epoch[epoch] = cached
        train, validation, test = cached
        policies[policy] = {
            "label": policy.replace("-", " ").title(),
            "epoch": epoch,
            "selectionScore": float(checkpoint["score"]),
            "selection": {
                "source": source,
                "metric": metric,
                "direction": direction,
            },
            "train": train["expectation"],
            "validation": validation["expectation"],
            "test": test["expectation"],
            "distribution": {
                "train": train,
                "validation": validation,
                "test": test,
            },
            "checkpoint": str(file.relative_to(repo)),
        }
    result = {
        "contract": "four-policy-direct-active-return-path-comparison-v1",
        "planId": plan_id,
        "planSha256": plan_hash,
        "policies": policies,
    }
    atomic_json(
        result, run_root / "state/checkpoint-selection-comparison.json"
    )
    return result


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
        datasets = (
            ActiveReturnPathDataset(
                train_shards, history_root, return_count=RETURN_COUNT,
                resolution=resolution,
            ),
            ActiveReturnPathDataset({
                "train": [], "validation": validation, "test": []
            }, history_root, return_count=RETURN_COUNT, resolution=resolution),
            ActiveReturnPathDataset({
                "train": [], "validation": [], "test": test
            }, history_root, return_count=RETURN_COUNT, resolution=resolution),
        )
        train_dataset, validation_dataset, test_dataset = datasets
        counts = {
            "train": train_dataset.logical_count("train"),
            "validation": validation_dataset.logical_count("validation"),
            "test": test_dataset.logical_count("test"),
        }
        if counts != {"train": examples, "validation": heldout, "test": heldout}:
            raise RuntimeError(f"direct path dataset counts changed: {counts}")
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
        normalization = training_sequence_normalization(
            train_dataset, batch_size=int(training["evaluationBatchSize"])
        )
        target_std = normalization.target_std
        cumulative_std = float(normalization.summary_std[4])
        architecture = plan["architecture"]
        model = NormalizedGluNextReturn(
            torch.from_numpy(normalization.feature_mean),
            torch.from_numpy(normalization.feature_std),
            torch.from_numpy(normalization.target_mean),
            torch.from_numpy(normalization.target_std),
            widths=tuple(int(value) for value in architecture["widths"]),
            dropout=float(architecture["dropout"]),
            dropout_rate=float(architecture["dropoutRate"]),
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
            for policy, (_source, _metric, direction)
            in SELECTION_POLICIES.items()
        }
        start_epoch = 0
        global_step = 0
        best_validation_mse = math.inf
        best_epoch = -1
        if checkpoint_exists(last_file):
            checkpoint = load_torch_checkpoint(
                last_file, map_location=device, weights_only=False
            )
            if checkpoint.get("planSha256") != plan_hash \
                    or checkpoint.get("runnerContract") != RUNNER_CONTRACT:
                raise ValueError("direct path checkpoint contract changed")
            model.load_state_dict(checkpoint["model"])
            for optimizer, state in zip(
                optimizers, checkpoint["optimizers"], strict=True
            ):
                optimizer.load_state_dict(state)
            start_epoch = int(checkpoint["epoch"]) + 1
            global_step = int(checkpoint["globalStep"])
            best_validation_mse = float(checkpoint["bestValidationMse"])
            best_epoch = int(checkpoint["bestEpoch"])
            selection_best = checkpoint["selectionPolicies"]
        reporter.emit({
            "event": "training-start", "planId": plan["id"],
            "startEpoch": start_epoch, "epochs": maximum_epochs,
            "parameters": parameter_count,
            "objective": "per-lead-normalized-mse-plus-cumulative-return-mse",
        })
        reporter.status(
            "training", planId=plan["id"], startEpoch=start_epoch,
            parameters=parameter_count, bestEpoch=best_epoch,
        )
        target_std_tensor = torch.from_numpy(target_std).to(device)
        cumulative_std_tensor = torch.tensor(cumulative_std, device=device)
        started = time.monotonic()
        for epoch in range(start_epoch, maximum_epochs):
            if pause_file is not None and pause_file.is_file():
                reporter.status(
                    "paused", planId=plan["id"], epoch=epoch,
                    bestEpoch=best_epoch,
                )
                raise SystemExit(PAUSE_EXIT_CODE)
            model.train()
            for features, targets, weights in iter_device_batches(
                train_dataset.iter_batches(
                    "train", batch_size, shuffle=True, shuffle_rows=True,
                    seed=seed + epoch, reuse_buffers=True,
                ),
                device,
            ):
                for optimizer in optimizers:
                    optimizer.zero_grad(set_to_none=True)
                prediction = model(features)
                candle = (
                    (prediction - targets) / target_std_tensor
                ).square().mean(dim=1)
                cumulative = (
                    (
                        torch.expm1(prediction.sum(dim=1))
                        - torch.expm1(targets.sum(dim=1))
                    ) / cumulative_std_tensor
                ).square()
                loss = (weights * (candle + cumulative)).sum() / weights.sum()
                loss.backward()
                clip_grad_norm_(
                    model.parameters(), float(training["gradientClip"]),
                    foreach=device.type == "cuda",
                )
                for optimizer in optimizers:
                    optimizer.step()
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
            score = float(validation_metrics["expectation"]["mse"])
            improved = score < best_validation_mse
            if improved:
                best_validation_mse = score
                best_epoch = epoch
            values = policy_values(train_metrics, validation_metrics)
            for policy, policy_score in values.items():
                if not policy_improved(
                    policy, policy_score,
                    float(selection_best[policy]["score"]),
                ):
                    continue
                selection_best[policy] = {
                    "score": policy_score, "epoch": epoch
                }
                save_torch_checkpoint({
                    "model": model.state_dict(), "epoch": epoch,
                    "score": policy_score, "policy": policy,
                    "planSha256": plan_hash,
                    "runnerContract": RUNNER_CONTRACT,
                }, selection_root / f"{policy}.json")
            checkpoint = {
                "model": model.state_dict(),
                "optimizers": [value.state_dict() for value in optimizers],
                "epoch": epoch, "globalStep": global_step,
                "bestEpoch": best_epoch,
                "bestValidationMse": best_validation_mse,
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
                "bestEpoch": best_epoch,
                "bestTrainScore": best_validation_mse,
                "improved": improved,
                "learningRate": float(optimizers[0].param_groups[0]["lr"]),
            }
            reporter.emit(event)
            reporter.status("training", planId=plan["id"], latest=event)

        best = load_torch_checkpoint(
            best_file, map_location=device, weights_only=False
        )
        model.load_state_dict(best["model"])
        final_metrics = tuple(
            evaluate(
                model, dataset, split, batch_size=evaluation_batch_size,
                target_std=target_std, cumulative_std=cumulative_std,
                device=device,
            )
            for dataset, split in zip(
                datasets, ("train", "validation", "test"), strict=True
            )
        )
        train_metrics, validation_metrics, test_metrics = final_metrics
        result = {
            "version": 1, "planId": plan["id"], "planSha256": plan_hash,
            "runnerContract": RUNNER_CONTRACT, "examples": examples,
            "counts": counts, "parameterCount": parameter_count,
            "bestEpoch": int(best["bestEpoch"]),
            "train": train_metrics["expectation"],
            "validation": validation_metrics["expectation"],
            "test": test_metrics["expectation"],
            "distribution": {
                "train": train_metrics,
                "validation": validation_metrics,
                "test": test_metrics,
            },
            "checkpoint": str(best_file.relative_to(repo)),
            "selectionPolicies": selection_best,
        }
        atomic_json(result, run_root / "state/result.json")
        checkpoint_selection_comparison(
            repo, run_root, model, datasets,
            plan_id=plan["id"], plan_hash=plan_hash,
            batch_size=evaluation_batch_size, target_std=target_std,
            cumulative_std=cumulative_std, device=device,
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
