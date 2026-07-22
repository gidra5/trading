from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import onnx
import torch
from torch import Tensor
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset

from mlp_model import (
    HIDDEN_LAYER_COUNT,
    HIDDEN_WIDTH,
    INPUT_FEATURE_COUNT,
    OUTPUT_PARAMETER_COUNT,
    ExposureMlp,
    LossWeights,
    PolicySupport,
    TimeWeighting,
    conditional_policy_logits,
    distance_imbalance_advice,
    fitted_teacher_loss,
    parameter_count,
    persistent_distance_imbalance_time_weights,
    validate_time_weighting,
)


METRIC_NAMES = (
    "loss",
    "crossEntropy",
    "probabilityMse",
    "parameterMse",
    "excessEntropy",
    "stateMutualInformation",
    "oracleMutualInformation",
    "targetEntropy",
    "predictedEntropy",
    "rawParameterMae",
    "distanceImbalanceWeight",
    "timeWeightEffectiveSampleRatio",
)


@dataclass(frozen=True)
class Shard:
    root: Path
    count: int
    features: str
    teacher_parameters: str
    teacher_metrics: str
    times: str


class FittedPolicyDataset(Dataset[tuple[Tensor, Tensor, Tensor, Tensor, Tensor]]):
    def __init__(self, manifest: dict, root: Path, split: str) -> None:
        self.split = split
        self.feature_count = int(manifest["featureCount"])
        self.parameter_count = int(manifest["teacherParameterCount"])
        self.teacher_metric_count = int(manifest["teacherMetricCount"])
        self.parts: list[tuple[Shard, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        self.offsets: list[int] = []
        total = 0
        for value in manifest["shards"]:
            if value["split"] != split:
                continue
            shard = Shard(
                root,
                int(value["count"]),
                value["features"],
                value["teacherParameters"],
                value["teacherMetrics"],
                value["times"],
            )
            features = np.memmap(
                root / shard.features, mode="r", dtype="<f2", shape=(shard.count, self.feature_count)
            )
            targets = np.memmap(
                root / shard.teacher_parameters,
                mode="r",
                dtype="<f4",
                shape=(shard.count, self.parameter_count),
            )
            teacher_metrics = np.memmap(
                root / shard.teacher_metrics,
                mode="r",
                dtype="<f4",
                shape=(shard.count, self.teacher_metric_count),
            )
            times = np.memmap(root / shard.times, mode="r", dtype="<i8", shape=(shard.count,))
            self.parts.append((shard, features, targets, teacher_metrics, times))
            total += shard.count
            self.offsets.append(total)
        self.time_weights = np.ones(total, dtype=np.float32)

    def __len__(self) -> int:
        return self.offsets[-1] if self.offsets else 0

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        part_index = int(np.searchsorted(self.offsets, index, side="right"))
        previous = self.offsets[part_index - 1] if part_index else 0
        _, features, targets, teacher_metrics, times = self.parts[part_index]
        row = index - previous
        return (
            torch.from_numpy(np.array(features[row], dtype=np.float32, copy=True)),
            torch.from_numpy(np.array(targets[row], dtype=np.float32, copy=True)),
            torch.tensor(float(self.time_weights[index]), dtype=torch.float32),
            torch.tensor(int(times[row]), dtype=torch.int64),
            torch.from_numpy(np.array(teacher_metrics[row], dtype=np.float32, copy=True)),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the causal 16x1024 revised-fitter MLP.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--label", default="Conservative revised-fitter MLP")
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--epochs", type=int, default=240)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--accumulate", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--states-per-example", type=int, default=17)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--log-every-steps", type=int, default=25)
    parser.add_argument("--loss-weights-json", default="{}")
    parser.add_argument("--time-weighting-json", default="{}")
    parser.add_argument("--finalize-file", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--initialize-from-checkpoint", type=Path)
    parser.add_argument("--compile", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_args(args)
    set_determinism(args.seed)
    manifest = json.loads((args.dataset / "dataset.json").read_text())
    validate_dataset_manifest(manifest)
    train = FittedPolicyDataset(manifest, args.dataset, "train")
    validation = FittedPolicyDataset(manifest, args.dataset, "validation")
    test = FittedPolicyDataset(manifest, args.dataset, "test")
    if min(len(train), len(validation), len(test)) == 0:
        raise RuntimeError("train, validation, and test datasets must all be non-empty")

    device = resolve_device(args.device)
    feature_mean, feature_std = training_normalization(train)
    parameter_scale = training_parameter_scale(train).to(device)
    model = ExposureMlp(feature_mean, feature_std, args.dropout).to(device)
    execution_support = PolicySupport(**manifest["policySupport"])
    # The teacher parameters retain the complete effective-range fit, while
    # every distribution objective is evaluated on the executable surface.
    support = execution_support
    actions = torch.linspace(
        support.visible_lower,
        support.visible_upper,
        int(manifest["actionCount"]),
        dtype=torch.float32,
        device=device,
    )
    current = deterministic_current_states(args.states_per_example, support, device, visible=True)
    loss_weights = parse_loss_weights(args.loss_weights_json)
    time_weighting = parse_time_weighting(args.time_weighting_json)
    for dataset in (train, validation, test):
        prepare_dataset_time_weights(
            dataset,
            actions,
            current,
            support,
            int(manifest["samplingIntervalMs"]),
            time_weighting,
            device,
        )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, betas=(0.9, 0.95)
    )
    train_loader = loader(train, args, shuffle=True)
    validation_loader = loader(validation, args, shuffle=False)
    test_loader = loader(test, args, shuffle=False)
    steps_per_epoch = math.ceil(len(train_loader) / args.accumulate)
    total_steps = max(1, steps_per_epoch * args.epochs)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: learning_rate_multiplier(step, total_steps)
    )
    scaler = torch.amp.GradScaler("cuda", init_scale=256.0, enabled=device.type == "cuda")
    checkpoint_file = args.output / "checkpoint.pt"
    best_model_file = args.output / "best-model.pt"
    args.output.mkdir(parents=True, exist_ok=True)
    start_epoch = 0
    global_step = 0
    best_epoch = -1
    best_validation = math.inf
    best_validation_metrics: dict[str, float] = {}
    stale_epochs = 0
    initialization: dict[str, object] | None = None
    if args.resume and checkpoint_file.exists():
        checkpoint = torch.load(checkpoint_file, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = checkpoint["epoch"] + 1
        global_step = checkpoint.get("globalStep", start_epoch * steps_per_epoch)
        best_epoch = checkpoint["bestEpoch"]
        best_validation = checkpoint["bestValidation"]
        best_validation_metrics = checkpoint.get("bestValidationMetrics", {"loss": best_validation})
        stale_epochs = checkpoint.get("staleEpochs", 0)
        initialization = checkpoint.get("initialization")
        restore_rng(checkpoint["rng"])
    elif args.initialize_from_checkpoint is not None:
        checkpoint = torch.load(
            args.initialize_from_checkpoint, map_location=device, weights_only=False
        )
        transferred = load_compatible_initialization(model, checkpoint["model"])
        restore_rng(checkpoint["rng"])
        for group in optimizer.param_groups:
            group["lr"] = args.learning_rate
            group["initial_lr"] = args.learning_rate
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda step: learning_rate_multiplier(step, total_steps)
        )
        initialization = {
            "checkpoint": str(args.initialize_from_checkpoint),
            "sourceEpoch": int(checkpoint["epoch"]),
            "sourceGlobalStep": int(checkpoint.get("globalStep", 0)),
            "transferredTensors": transferred,
            "optimizer": "reset for the eight-parameter objective",
        }

    train_model = torch.compile(model) if args.compile else model
    emit({
        "event": "training-start",
        "device": str(device),
        "parameters": parameter_count(model),
        "trainExamples": len(train),
        "validationExamples": len(validation),
        "testExamples": len(test),
        "lossWeights": asdict(loss_weights),
        "timeWeighting": time_weighting_metadata(time_weighting),
        "distributionLossRange": [
            execution_support.visible_lower,
            execution_support.visible_upper,
        ],
        "currentStates": args.states_per_example,
        "startEpoch": start_epoch,
        "epochs": args.epochs,
        **({"initializedFrom": initialization} if initialization else {}),
    })

    if not best_model_file.exists():
        baseline = evaluate(
            model, validation_loader, actions, current, support,
            parameter_scale, loss_weights, time_weighting, device
        )
        best_validation = baseline["loss"]
        best_validation_metrics = baseline
        best_epoch = -1
        atomic_torch_save(model.state_dict(), best_model_file)
        emit({"event": "baseline", "validation": baseline})

    stopped = False
    interrupted = False
    last_epoch = start_epoch - 1
    try:
        for epoch in range(start_epoch, args.epochs):
            last_epoch = epoch
            started = time.monotonic()
            train_metrics, global_step, stopped = train_epoch(
                train_model,
                train_loader,
                optimizer,
                scheduler,
                scaler,
                actions,
                current,
                support,
                parameter_scale,
                loss_weights,
                time_weighting,
                args,
                device,
                epoch,
                global_step,
            )
            validation_metrics = evaluate(
                model, validation_loader, actions, current, support,
                parameter_scale, loss_weights, time_weighting, device
            )
            improved = validation_metrics["loss"] < best_validation - 1e-6
            if improved:
                best_validation = validation_metrics["loss"]
                best_validation_metrics = validation_metrics
                best_epoch = epoch
                stale_epochs = 0
                atomic_torch_save(model.state_dict(), best_model_file)
            else:
                stale_epochs += 1
            save_checkpoint(
                checkpoint_file,
                epoch,
                global_step,
                best_epoch,
                best_validation,
                best_validation_metrics,
                stale_epochs,
                model,
                optimizer,
                scheduler,
                scaler,
                initialization,
            )
            emit({
                "event": "epoch",
                "epoch": epoch,
                "epochs": args.epochs,
                "globalStep": global_step,
                "seconds": round(time.monotonic() - started, 2),
                "train": train_metrics,
                "validation": validation_metrics,
                "bestEpoch": best_epoch,
                "bestValidation": best_validation,
                "staleEpochs": stale_epochs,
                "stopRequested": stopped,
            })
            if stopped or stale_epochs >= args.patience:
                break
    except KeyboardInterrupt:
        interrupted = True
        emit({"event": "interrupt", "message": "Finalizing the best validated checkpoint."})
        validation_metrics = evaluate(
            model, validation_loader, actions, current, support,
            parameter_scale, loss_weights, time_weighting, device
        )
        if validation_metrics["loss"] < best_validation - 1e-6:
            best_validation = validation_metrics["loss"]
            best_validation_metrics = validation_metrics
            best_epoch = last_epoch
            atomic_torch_save(model.state_dict(), best_model_file)
        save_checkpoint(
            checkpoint_file,
            last_epoch,
            global_step,
            best_epoch,
            best_validation,
            best_validation_metrics,
            stale_epochs,
            model,
            optimizer,
            scheduler,
            scaler,
            initialization,
        )

    model.load_state_dict(torch.load(best_model_file, map_location=device, weights_only=True))
    test_metrics = evaluate(
        model, test_loader, actions, current, support,
        parameter_scale, loss_weights, time_weighting, device
    )
    teacher_metrics = teacher_fit_summary(manifest, args.dataset)
    export_artifact(
        model,
        args,
        manifest,
        len(train),
        len(validation),
        len(test),
        best_epoch,
        best_validation_metrics,
        test_metrics,
        teacher_metrics,
        loss_weights,
        time_weighting,
        device,
        stopped or interrupted,
        initialization,
    )
    emit({
        "event": "training-complete",
        "test": test_metrics,
        "bestValidation": best_validation_metrics,
        "bestEpoch": best_epoch,
        "artifact": str(args.output / "model.onnx"),
        "finalizedEarly": stopped or interrupted,
    })


def train_epoch(
    model,
    data,
    optimizer,
    scheduler,
    scaler,
    actions,
    current,
    support,
    parameter_scale,
    loss_weights,
    time_weighting,
    args,
    device,
    epoch,
    global_step,
) -> tuple[dict[str, float], int, bool]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    totals = {name: 0.0 for name in METRIC_NAMES}
    total_examples = 0
    time_weight_sum = 0.0
    time_weight_square_sum = 0.0
    started = time.monotonic()
    stopped = False
    for batch_step, (features, targets, time_weights, _, _) in enumerate(data):
        features = features.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        time_weights = time_weights.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            raw = model(features)
            batch_metrics = fitted_teacher_loss(
                raw, targets, actions, current.expand(features.shape[0], -1), support,
                parameter_scale, loss_weights, time_weighting, time_weights
            )
            loss = batch_metrics["loss"] / args.accumulate
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite training loss at epoch {epoch}, batch {batch_step}")
        scaler.scale(loss).backward()
        should_step = (batch_step + 1) % args.accumulate == 0 or batch_step + 1 == len(data)
        if should_step:
            scaler.unscale_(optimizer)
            gradient_norm = float(clip_grad_norm_(model.parameters(), 1.0))
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if scaler.get_scale() >= scale_before:
                scheduler.step()
            global_step += 1
            if global_step == 1 or global_step % args.log_every_steps == 0:
                emit({
                    "event": "train-step",
                    "epoch": epoch,
                    "epochs": args.epochs,
                    "batch": batch_step + 1,
                    "batches": len(data),
                    "globalStep": global_step,
                    "learningRate": optimizer.param_groups[0]["lr"],
                    "gradientNorm": gradient_norm,
                    "examplesPerSecond": round(total_examples / max(time.monotonic() - started, 1e-6), 1),
                    "gpuMemoryMiB": round(torch.cuda.max_memory_allocated() / 1_048_576, 1)
                    if device.type == "cuda" else 0,
                    "latest": detached_metrics(batch_metrics),
                })
            if args.finalize_file and args.finalize_file.exists():
                stopped = True
        count = features.shape[0]
        total_examples += count
        time_weight_sum += float(batch_metrics["timeWeightSum"].detach())
        time_weight_square_sum += float(batch_metrics["timeWeightSquareSum"].detach())
        for name in METRIC_NAMES:
            totals[name] += float(batch_metrics[name].detach()) * count
        if stopped:
            break
    result = {name: value / max(1, total_examples) for name, value in totals.items()}
    result["timeWeightEffectiveSampleRatio"] = (
        time_weight_sum * time_weight_sum
        / max(1e-12, total_examples * time_weight_square_sum)
    )
    return result, global_step, stopped


@torch.inference_mode()
def evaluate(model, data, actions, current, support,
             parameter_scale, loss_weights, time_weighting, device) -> dict[str, float]:
    model.eval()
    totals = {name: 0.0 for name in METRIC_NAMES}
    total_examples = 0
    time_weight_sum = 0.0
    time_weight_square_sum = 0.0
    for features, targets, time_weights, _, _ in data:
        features = features.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        time_weights = time_weights.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            raw = model(features)
        batch_metrics = fitted_teacher_loss(
            raw, targets, actions, current.expand(features.shape[0], -1), support,
            parameter_scale, loss_weights, time_weighting, time_weights
        )
        count = features.shape[0]
        total_examples += count
        time_weight_sum += float(batch_metrics["timeWeightSum"])
        time_weight_square_sum += float(batch_metrics["timeWeightSquareSum"])
        for name in METRIC_NAMES:
            totals[name] += float(batch_metrics[name]) * count
    result = {name: value / max(1, total_examples) for name, value in totals.items()}
    result["timeWeightEffectiveSampleRatio"] = (
        time_weight_sum * time_weight_sum
        / max(1e-12, total_examples * time_weight_square_sum)
    )
    return result


def loader(dataset: Dataset, args: argparse.Namespace, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.workers > 0,
        drop_last=False,
    )


@torch.inference_mode()
def prepare_dataset_time_weights(
    dataset: FittedPolicyDataset,
    actions: Tensor,
    current: Tensor,
    support: PolicySupport,
    sampling_interval_ms: int,
    weighting: TimeWeighting,
    device: torch.device,
) -> None:
    """Decode teacher advice once, then apply causal persistence before shuffling."""
    advice = np.empty(len(dataset), dtype=np.float32)
    times = np.empty(len(dataset), dtype=np.int64)
    offset = 0
    preparation_batch = 8_192
    for shard, _, targets, _, shard_times in dataset.parts:
        for start in range(0, shard.count, preparation_batch):
            end = min(shard.count, start + preparation_batch)
            target = torch.from_numpy(
                np.array(targets[start:end], dtype=np.float32, copy=True)
            ).to(device=device, non_blocking=True)
            target_rows = target[:, None, :].expand(-1, current.shape[-1], -1)
            logits = conditional_policy_logits(
                target_rows,
                actions,
                current.expand(target.shape[0], -1),
                support,
            )
            probability = torch.softmax(logits, dim=-1)
            block_advice = distance_imbalance_advice(
                probability,
                actions,
                current.expand(target.shape[0], -1),
                weighting.distance_epsilon,
            )
            advice[offset + start:offset + end] = block_advice.cpu().numpy()
        times[offset:offset + shard.count] = np.asarray(shard_times, dtype=np.int64)
        offset += shard.count

    order = np.argsort(times, kind="stable")
    ordered_times = times[order]
    if ordered_times.size > 1 and np.any(ordered_times[1:] <= ordered_times[:-1]):
        raise RuntimeError(f"{dataset.split} timestamps must be unique for persistence weighting")
    ordered_weights = persistent_distance_imbalance_time_weights(
        torch.from_numpy(advice[order]),
        torch.from_numpy(ordered_times),
        sampling_interval_ms,
        weighting,
    ).numpy()
    dataset.time_weights[order] = ordered_weights
    total = float(ordered_weights.sum(dtype=np.float64))
    squared = float(np.square(ordered_weights, dtype=np.float64).sum(dtype=np.float64))
    effective_ratio = total * total / max(1e-12, len(dataset) * squared)
    emit({
        "event": "time-weighting-ready",
        "split": dataset.split,
        "examples": len(dataset),
        "meanAdviceMagnitude": float(np.abs(advice).mean(dtype=np.float64)),
        "meanWeight": float(ordered_weights.mean(dtype=np.float64)),
        "maximumWeight": float(ordered_weights.max()),
        "effectiveSampleRatio": effective_ratio,
    })


def training_normalization(dataset: FittedPolicyDataset) -> tuple[Tensor, Tensor]:
    total = 0
    feature_sum = np.zeros(INPUT_FEATURE_COUNT, dtype=np.float64)
    square_sum = np.zeros(INPUT_FEATURE_COUNT, dtype=np.float64)
    for shard, features, _, _, _ in dataset.parts:
        for start in range(0, shard.count, 8192):
            block = np.asarray(features[start:start + 8192], dtype=np.float32)
            feature_sum += block.sum(axis=0, dtype=np.float64)
            square_sum += np.square(block, dtype=np.float64).sum(axis=0)
            total += block.shape[0]
    mean = feature_sum / total
    variance = np.maximum(1e-12, square_sum / total - mean * mean)
    return torch.from_numpy(mean.astype(np.float32)), torch.from_numpy(np.sqrt(variance).astype(np.float32))


def training_parameter_scale(dataset: FittedPolicyDataset) -> Tensor:
    total = 0
    value_sum = np.zeros(OUTPUT_PARAMETER_COUNT, dtype=np.float64)
    square_sum = np.zeros(OUTPUT_PARAMETER_COUNT, dtype=np.float64)
    for shard, _, targets, _, _ in dataset.parts:
        for start in range(0, shard.count, 8192):
            block = np.asarray(targets[start:start + 8192], dtype=np.float32)
            value_sum += block.sum(axis=0, dtype=np.float64)
            square_sum += np.square(block, dtype=np.float64).sum(axis=0)
            total += block.shape[0]
    mean = value_sum / total
    minimum_variance = np.full(OUTPUT_PARAMETER_COUNT, 1e-4, dtype=np.float64)
    minimum_variance[6:8] = 0.25 ** 2
    variance = np.maximum(minimum_variance, square_sum / total - mean * mean)
    return torch.from_numpy(np.sqrt(variance).astype(np.float32))


def load_compatible_initialization(model: ExposureMlp, source: dict[str, Tensor]) -> int:
    """Transfer the shared six-coordinate network while retaining new normalization/cutoff rows."""
    target = model.state_dict()
    transferred = 0
    for name, value in source.items():
        if name in ("feature_mean", "feature_std") or name not in target:
            continue
        destination = target[name]
        if value.shape == destination.shape:
            destination.copy_(value.to(device=destination.device, dtype=destination.dtype))
            transferred += 1
            continue
        if name in ("output.weight", "output.bias") \
                and value.ndim == destination.ndim \
                and value.shape[1:] == destination.shape[1:]:
            rows = min(value.shape[0], destination.shape[0], 6)
            destination[:rows].copy_(
                value[:rows].to(device=destination.device, dtype=destination.dtype)
            )
            transferred += 1
    model.load_state_dict(target)
    return transferred


def teacher_fit_summary(manifest: dict, root: Path) -> dict[str, float]:
    names = manifest["teacherMetricNames"]
    total = np.zeros(len(names), dtype=np.float64)
    count = 0
    for value in manifest["shards"]:
        row_count = int(value["count"])
        metrics = np.memmap(
            root / value["teacherMetrics"], mode="r", dtype="<f4", shape=(row_count, len(names))
        )
        total += np.asarray(metrics, dtype=np.float64).sum(axis=0)
        count += row_count
    return {name: float(total[index] / max(1, count)) for index, name in enumerate(names)}


def deterministic_current_states(
    count: int,
    support: PolicySupport,
    device: torch.device,
    *,
    visible: bool = False,
) -> Tensor:
    lower = support.visible_lower if visible else support.latent_lower
    upper = support.visible_upper if visible else support.latent_upper
    return torch.linspace(lower, upper, count, device=device).view(1, count)


def parse_loss_weights(value: str) -> LossWeights:
    parsed = json.loads(value)
    return LossWeights(
        cross_entropy=float(parsed.get("crossEntropy", 1)),
        probability_mse=float(parsed.get("probabilityMse", 1)),
        parameter_mse=float(parsed.get("parameterMse", 1)),
        excess_entropy=float(parsed.get("excessEntropy", 1)),
        state_mutual_information=float(parsed.get("stateMutualInformation", 1)),
        oracle_mutual_information=float(parsed.get("oracleMutualInformation", 1)),
    )


def parse_time_weighting(value: str) -> TimeWeighting:
    parsed = json.loads(value)
    if parsed.get("mode", "distanceImbalance") != "distanceImbalance":
        raise ValueError("time weighting mode must be distanceImbalance")
    if parsed.get("stateAggregation", "absoluteMean") != "absoluteMean":
        raise ValueError("distance imbalance state aggregation must be absoluteMean")
    weighting = TimeWeighting(
        distance_epsilon=float(parsed.get("distanceEpsilon", 1e-6)),
        minimum_weight=float(parsed.get("minimumWeight", 1e-6)),
        minimum_advice_magnitude=float(parsed.get("minimumAdviceMagnitude", 0.25)),
        memory_half_life_steps=float(parsed.get("memoryHalfLifeSteps", 15)),
        growth_per_prior_advice=float(parsed.get("growthPerPriorAdvice", 0.25)),
        maximum_multiplier=float(parsed.get("maximumMultiplier", 4)),
        reset_after_gap_steps=float(parsed.get("resetAfterGapSteps", 60)),
    )
    validate_time_weighting(weighting)
    return weighting


def time_weighting_metadata(weighting: TimeWeighting) -> dict[str, object]:
    return {
        "mode": "distanceImbalance",
        "distanceEpsilon": weighting.distance_epsilon,
        "minimumWeight": weighting.minimum_weight,
        "stateAggregation": "absoluteMean",
        "minimumAdviceMagnitude": weighting.minimum_advice_magnitude,
        "memoryHalfLifeSteps": weighting.memory_half_life_steps,
        "growthPerPriorAdvice": weighting.growth_per_prior_advice,
        "maximumMultiplier": weighting.maximum_multiplier,
        "resetAfterGapSteps": weighting.reset_after_gap_steps,
    }


def learning_rate_multiplier(step: int, total: int) -> float:
    warmup = max(1, int(total * 0.05))
    if step < warmup:
        return (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def save_checkpoint(file, epoch, global_step, best_epoch, best_validation,
                    best_validation_metrics, stale_epochs, model, optimizer, scheduler, scaler,
                    initialization) -> None:
    atomic_torch_save({
        "epoch": epoch,
        "globalStep": global_step,
        "bestEpoch": best_epoch,
        "bestValidation": best_validation,
        "bestValidationMetrics": best_validation_metrics,
        "staleEpochs": stale_epochs,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "rng": capture_rng(),
        "initialization": initialization,
    }, file)


def export_artifact(model, args, dataset_manifest, train_count, validation_count, test_count,
                    best_epoch, best_validation_metrics, test_metrics, teacher_metrics,
                    loss_weights, time_weighting, device, finalized_early, initialization) -> None:
    model.eval().cpu()
    verification_batch_size = 7
    verification_features = (
        torch.sin(torch.arange(verification_batch_size * INPUT_FEATURE_COUNT) * 0.173) * 0.3
        + torch.cos(torch.arange(verification_batch_size * INPUT_FEATURE_COUNT) * 0.019) * 0.1
    ).reshape(verification_batch_size, INPUT_FEATURE_COUNT).float()
    with torch.inference_mode():
        verification_output = model(verification_features)
    atomic_bytes(verification_features.numpy().astype("<f4").tobytes(), args.output / "verification-input.f32")
    atomic_bytes(verification_output.numpy().astype("<f4").tobytes(), args.output / "verification-output.f32")
    onnx_file = args.output / "model.onnx"
    temporary = onnx_file.with_suffix(".onnx.tmp")
    torch.onnx.export(
        model,
        (torch.zeros(2, INPUT_FEATURE_COUNT, dtype=torch.float32),),
        temporary,
        input_names=["features"],
        output_names=["raw_parameters"],
        dynamic_shapes={"features": {0: torch.export.Dim("batch", min=1)}},
        opset_version=18,
        dynamo=True,
        external_data=False,
    )
    graph = onnx.load(temporary)
    onnx.checker.check_model(graph, full_check=True)
    temporary.replace(onnx_file)
    manifest = {
        "id": args.model_id,
        "label": args.label,
        "createdAt": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "featureSchemaVersion": 4,
        "inputFeatureCount": INPUT_FEATURE_COUNT,
        "outputParameterCount": OUTPUT_PARAMETER_COUNT,
        "hiddenLayerCount": HIDDEN_LAYER_COUNT,
        "hiddenWidth": HIDDEN_WIDTH,
        "modelFile": "model.onnx",
        "checkpointFile": "checkpoint.pt",
        "verificationFixture": {
            "batchSize": verification_batch_size,
            "inputFile": "verification-input.f32",
            "outputFile": "verification-output.f32",
        },
        "policySupport": dataset_manifest["policySupport"],
        "training": {
            "datasetPlanId": dataset_manifest["planId"],
            "trainExamples": train_count,
            "validationExamples": validation_count,
            "testExamples": test_count,
            "bestEpoch": best_epoch,
            "bestValidationLoss": best_validation_metrics["loss"],
            "testLoss": test_metrics["loss"],
            "bestValidationMetrics": best_validation_metrics,
            "testMetrics": test_metrics,
            "teacherFitMetrics": teacher_metrics,
            "lossWeights": asdict(loss_weights),
            "timeWeighting": time_weighting_metadata(time_weighting),
            "distributionLossRange": [
                dataset_manifest["policySupport"]["visible_lower"],
                dataset_manifest["policySupport"]["visible_upper"],
            ],
            "finalizedEarly": finalized_early,
            "seed": args.seed,
            "device": str(device),
            **({"initializedFrom": initialization} if initialization else {}),
        },
    }
    if args.plan:
        manifest["training"]["planFile"] = str(args.plan)
    atomic_json(manifest, args.output / "manifest.json")


def validate_dataset_manifest(manifest: dict) -> None:
    if manifest.get("version") != 4 or manifest.get("featureSchemaVersion") != 4:
        raise ValueError("unsupported MLP fitted-policy dataset schema")
    if manifest.get("featureCount") != INPUT_FEATURE_COUNT:
        raise ValueError("dataset feature count does not match model")
    if manifest.get("teacherParameterCount") != OUTPUT_PARAMETER_COUNT:
        raise ValueError("dataset teacher parameter count does not match model")
    if len(manifest.get("grid", [])) != manifest.get("actionCount"):
        raise ValueError("dataset action grid is invalid")
    if not isinstance(manifest.get("samplingIntervalMs"), int) \
            or manifest["samplingIntervalMs"] <= 0:
        raise ValueError("dataset sampling interval is invalid")
    support = manifest.get("policySupport", {})
    grid = manifest.get("grid", [])
    if not grid or grid[0] != support.get("latent_lower") or grid[-1] != support.get("latent_upper"):
        raise ValueError("dataset teacher grid must cover the complete effective range")


def validate_args(args: argparse.Namespace) -> None:
    if min(args.epochs, args.batch_size, args.accumulate, args.states_per_example,
           args.patience, args.log_every_steps) < 1 or args.workers < 0:
        raise ValueError("training counts must be positive (workers may be zero)")
    if args.learning_rate <= 0 or args.weight_decay < 0 or not 0 <= args.dropout < 1:
        raise ValueError("invalid optimizer or dropout configuration")
    if args.initialize_from_checkpoint is not None and not args.initialize_from_checkpoint.is_file():
        raise ValueError(
            f"initial checkpoint does not exist: {args.initialize_from_checkpoint}"
        )


def detached_metrics(metrics: dict[str, Tensor]) -> dict[str, float]:
    return {name: float(metrics[name].detach()) for name in METRIC_NAMES}


def emit(value: dict) -> None:
    print(json.dumps(value), flush=True)


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but PyTorch cannot access a GPU")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def set_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def capture_rng() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    # Checkpoints are loaded with map_location=device so model/optimizer restore
    # directly onto CUDA. RNG generator state is the exception: both CPU and
    # CUDA generator APIs require a CPU uint8 state tensor.
    torch.set_rng_state(cpu_rng_state(state["torch"]))
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([
            cpu_rng_state(value) for value in state["cuda"]
        ])


def cpu_rng_state(value) -> Tensor:
    if isinstance(value, Tensor):
        return value.detach().to(device="cpu", dtype=torch.uint8)
    return torch.as_tensor(value, dtype=torch.uint8, device="cpu")


def atomic_torch_save(value, target: Path) -> None:
    temporary = target.with_suffix(target.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(target)


def atomic_json(value: dict, target: Path) -> None:
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(target)


def atomic_bytes(value: bytes, target: Path) -> None:
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_bytes(value)
    temporary.replace(target)


if __name__ == "__main__":
    main()
