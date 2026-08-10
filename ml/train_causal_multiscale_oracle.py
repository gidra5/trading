from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from torch import Tensor
from torch.nn.utils import clip_grad_norm_

from return_oracle_decoder_screen import (
    LearnedRadiusShrinkingDecoder,
    next_equal_entropy_step_training_temperature,
    next_stale_learning_rate,
    optimizer_parameter_groups,
    smooth_oracle_probabilities,
    weighted_target_entropy_log_temperature_derivative,
)
from differentiable_exposure_value_oracle import (
    DifferentiableExposureValueOracle,
    DifferentiableExposureValueOracleConfig,
)
from oracle_distribution_path import (
    oracle_forward_kl_per_example,
    oracle_mean_plus_p50_kl_loss,
)
from trading_storage import (
    checkpoint_exists,
    load_torch_checkpoint,
    read_candle_column,
    read_shard_array,
    save_torch_checkpoint,
)


DAY_ROWS = 86_400
MINUTE_TARGET_ROWS = 1_441
SOURCE_FEATURES = 901
DAILY_TRUNCATED_FEATURES = 771
ACTION_COUNT = 101
HORIZON_MINUTES = 15
PRODUCTION_TEMPERATURE = 0.01


@dataclass(frozen=True)
class MinuteExample:
    timestamp: int
    split: str
    date: str
    feature_row: int
    target_row: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the learned-radius oracle decoder on causal multiscale "
            "features sampled at completed-minute boundaries."
        )
    )
    parser.add_argument(
        "--plan",
        type=Path,
        default=Path(
            "ml/training-plans/causal-multiscale-oracle-daily-v1.json"
        ),
    )
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--stop-after-epoch", type=int)
    return parser.parse_args()


def atomic_json(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def plan_hash(plan: dict[str, Any]) -> str:
    payload = json.dumps(
        plan, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def source_examples(manifest: dict[str, Any]) -> list[MinuteExample]:
    examples: list[MinuteExample] = []
    for shard in sorted(manifest["shards"], key=lambda item: item["predictionTimeStart"]):
        split = str(shard["split"])
        if split not in {"train", "validation", "test"}:
            continue
        start = int(shard["featureRowOffset"])
        stop = start + int(shard["count"])
        first = start + ((59 - start) % 60)
        prediction_start = int(shard["predictionTimeStart"])
        for row in range(first, stop, 60):
            timestamp = prediction_start + (row - start) * 1_000
            examples.append(MinuteExample(
                timestamp=timestamp,
                split=split,
                date=str(shard["date"]),
                feature_row=row,
                target_row=row // 60 + 1,
            ))
    examples.sort(key=lambda value: value.timestamp)
    if not examples:
        raise ValueError("source manifest produced no completed-minute examples")

    # Each target observes the next hour. Remove the final hour from every
    # contiguous split run, including gaps between the sampled market windows.
    purged: list[MinuteExample] = []
    run: list[MinuteExample] = []
    for example in examples:
        contiguous = bool(run) and (
            example.split == run[-1].split
            and example.timestamp == run[-1].timestamp + 60_000
        )
        if run and not contiguous:
            purged.extend(run[:-HORIZON_MINUTES])
            run = []
        run.append(example)
    if run:
        purged.extend(run[:-HORIZON_MINUTES])
    if not purged:
        raise ValueError("future-horizon purge removed the complete dataset")
    return purged


def prepare_dataset(
    source_root: Path,
    minute_history_root: Path,
    compact_root: Path,
    feature_count: int,
    oracle_config: DifferentiableExposureValueOracleConfig,
) -> dict[str, Any]:
    source_file = source_root / "dataset.json"
    source_bytes = source_file.read_bytes()
    source_sha = hashlib.sha256(source_bytes).hexdigest()
    manifest_file = compact_root / "dataset.json"
    if manifest_file.is_file():
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        if manifest.get("sourceSha256") != source_sha \
                or int(manifest.get("featureCount", 0)) != feature_count \
                or int(manifest.get("actionCount", 0)) != ACTION_COUNT \
                or manifest.get("oracle") != oracle_config.__dict__:
            raise ValueError("prepared causal dataset does not match its source")
        return manifest

    source = json.loads(source_bytes)
    examples = source_examples(source)
    by_split = {
        split: [value for value in examples if value.split == split]
        for split in ("train", "validation", "test")
    }
    compact_root.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, tuple[np.memmap, np.memmap, np.memmap]] = {}
    for split, values in by_split.items():
        arrays[split] = (
            np.lib.format.open_memmap(
                compact_root / f"{split}.features.f16.npy",
                mode="w+", dtype="<f2", shape=(len(values), feature_count),
            ),
            np.lib.format.open_memmap(
                compact_root / f"{split}.targets.f32.npy",
                mode="w+", dtype="<f4", shape=(len(values), ACTION_COUNT),
            ),
            np.lib.format.open_memmap(
                compact_root / f"{split}.timestamps.i64.npy",
                mode="w+", dtype="<i8", shape=(len(values),),
            ),
        )

    positions: dict[tuple[str, str], list[tuple[int, MinuteExample]]] = defaultdict(list)
    split_offsets = {split: 0 for split in by_split}
    for split, values in by_split.items():
        for index, example in enumerate(values):
            positions[(example.date, split)].append((index, example))

    input_by_date = {
        value["date"]: source_root / value["features"]
        for value in source["componentLayout"]["inputComponents"]
    }
    dates = sorted({example.date for example in examples})
    oracle_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    oracle = DifferentiableExposureValueOracle(oracle_config).to(oracle_device)
    oracle.eval()
    started = time.monotonic()
    for date_index, day in enumerate(dates, 1):
        feature_file = input_by_date.get(day)
        current_close_file = minute_history_root / f"{day}.json"
        next_date = (date.fromisoformat(day) + timedelta(days=1)).isoformat()
        next_close_file = minute_history_root / f"{next_date}.json"
        if feature_file is None \
                or not current_close_file.is_file() \
                or not next_close_file.is_file():
            raise FileNotFoundError(f"missing same-time components for {day}")
        _feature_meta, full_features = read_shard_array(
            feature_file, "<f2", (DAY_ROWS, SOURCE_FEATURES)
        )
        current_closes = read_candle_column(current_close_file, "close")
        next_closes = read_candle_column(next_close_file, "close")
        if current_closes.shape != (1_440,) or next_closes.shape != (1_440,):
            raise ValueError(f"invalid one-minute close component for {day}")
        closes = np.concatenate((current_closes, next_closes[:HORIZON_MINUTES]))
        returns = np.diff(np.log(closes)).astype(np.float32)
        paths = np.ascontiguousarray(
            np.lib.stride_tricks.sliding_window_view(returns, HORIZON_MINUTES)
        )
        if paths.shape != (1_440, HORIZON_MINUTES):
            raise RuntimeError("fifteen-minute target path alignment failed")
        with torch.inference_mode():
            minute_targets = oracle.forward_from_log_returns(
                torch.from_numpy(paths).to(oracle_device)
            ).probabilities.float().cpu().numpy()
        for split in by_split:
            selected = positions.get((day, split), ())
            if not selected:
                continue
            output_rows = np.fromiter(
                (item[0] for item in selected), dtype=np.int64
            )
            feature_rows = np.fromiter(
                (item[1].feature_row for item in selected), dtype=np.int64
            )
            target_rows = np.fromiter(
                (item[1].feature_row // 60 for item in selected), dtype=np.int64
            )
            timestamps = np.fromiter(
                (item[1].timestamp for item in selected), dtype=np.int64
            )
            arrays[split][0][output_rows] = full_features[
                feature_rows, :feature_count
            ]
            arrays[split][1][output_rows] = minute_targets[target_rows]
            arrays[split][2][output_rows] = timestamps
        if date_index % 20 == 0 or date_index == len(dates):
            print(json.dumps({
                "event": "prepare-progress",
                "days": date_index,
                "totalDays": len(dates),
                "seconds": round(time.monotonic() - started, 2),
            }), flush=True)
    for values in arrays.values():
        for value in values:
            value.flush()

    train_features = arrays["train"][0]
    feature_sum = np.zeros(feature_count, dtype=np.float64)
    feature_square_sum = np.zeros(feature_count, dtype=np.float64)
    for start in range(0, train_features.shape[0], 16_384):
        batch = np.asarray(train_features[start:start + 16_384], dtype=np.float64)
        feature_sum += batch.sum(axis=0)
        feature_square_sum += np.square(batch).sum(axis=0)
    mean = feature_sum / train_features.shape[0]
    variance = np.maximum(
        feature_square_sum / train_features.shape[0] - np.square(mean), 1e-12
    )
    std = np.sqrt(variance)
    manifest = {
        "schemaVersion": 1,
        "sourceDataset": str(source_root),
        "sourceSha256": source_sha,
        "samplingIntervalMs": 60_000,
        "predictionHorizonMs": 900_000,
        "pairing": "causal features at t to next-fifteen-minute oracle at t",
        "splitBoundaryPurgeMinutes": HORIZON_MINUTES,
        "featureContract": "schema-6 multiscale OHLCV through daily scale",
        "featureCount": feature_count,
        "actionCount": ACTION_COUNT,
        "oracle": oracle_config.__dict__,
        "counts": {split: len(values) for split, values in by_split.items()},
        "normalization": {"mean": mean.tolist(), "std": std.tolist()},
        "files": {
            split: {
                "features": f"{split}.features.f16.npy",
                "targets": f"{split}.targets.f32.npy",
                "timestamps": f"{split}.timestamps.i64.npy",
            }
            for split in by_split
        },
    }
    atomic_json(manifest, manifest_file)
    return manifest


class CompactDataset:
    def __init__(self, root: Path, manifest: dict[str, Any]) -> None:
        self.features: dict[str, np.ndarray] = {}
        self.targets: dict[str, np.ndarray] = {}
        for split, files in manifest["files"].items():
            self.features[split] = np.load(root / files["features"], mmap_mode="r")
            self.targets[split] = np.load(root / files["targets"], mmap_mode="r")

    def batches(
        self, split: str, batch_size: int, *, shuffle: bool, seed: int
    ) -> Iterator[tuple[Tensor, Tensor]]:
        count = self.features[split].shape[0]
        starts = list(range(0, count, batch_size))
        if shuffle:
            random.Random(seed).shuffle(starts)
        for start in starts:
            stop = min(count, start + batch_size)
            yield (
                torch.from_numpy(np.asarray(
                    self.features[split][start:stop], dtype=np.float32
                ).copy()),
                torch.from_numpy(np.asarray(
                    self.targets[split][start:stop], dtype=np.float32
                ).copy()),
            )


class ScalarAccumulator:
    def __init__(self) -> None:
        self.sums: dict[str, float] = defaultdict(float)
        self.weight = 0

    def add(self, metrics: dict[str, Tensor], count: int) -> None:
        for key, value in metrics.items():
            if value.ndim == 0:
                self.sums[key] += float(value.detach()) * count
        self.weight += count

    def result(self) -> dict[str, float]:
        return {key: value / self.weight for key, value in self.sums.items()}


def model_forward(
    model: LearnedRadiusShrinkingDecoder,
    features: Tensor,
    regularizers: dict[str, Any],
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    logits, mean_penalty, variance_penalty = model.forward_with_regularizers(features)
    weight = regularizers["softWeightBound"]
    parameter_penalties = model.regularizer_parameter_penalties(
        desired_weight_magnitude=float(weight["desiredMagnitude"]),
        weight_bound_sharpness=float(weight["sharpness"]),
        absolute_epsilon=float(weight["absoluteEpsilon"]),
        include_centering_constraint=False,
    )
    return logits, mean_penalty, variance_penalty, *parameter_penalties


def distribution_batch_metrics(
    forward_result: tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor],
    raw_targets: Tensor,
    train_temperature: float,
    objective: dict[str, Any],
) -> dict[str, Tensor]:
    logits, mean_penalties, variance_penalties, weight_bound, _, _ = forward_result
    raw_targets = raw_targets.float().clamp_min(0)
    raw_targets = raw_targets / raw_targets.sum(dim=-1, keepdim=True).clamp_min(
        torch.finfo(torch.float32).tiny
    )
    curriculum_targets = smooth_oracle_probabilities(
        raw_targets,
        source_temperature=PRODUCTION_TEMPERATURE,
        target_temperature=train_temperature,
    )
    predicted = torch.softmax(logits.float(), dim=-1)
    weights = torch.ones(predicted.shape[0], device=predicted.device)
    probability_floor = float(objective["probabilityFloor"])
    raw_kl = oracle_forward_kl_per_example(
        predicted, raw_targets, probability_floor=probability_floor
    )
    curriculum_kl = oracle_forward_kl_per_example(
        predicted, curriculum_targets, probability_floor=probability_floor
    )
    distribution_loss = oracle_mean_plus_p50_kl_loss(
        predicted,
        curriculum_targets,
        weights,
        probability_floor=probability_floor,
        mean_weight=float(objective["meanKlWeight"]),
        p50_weight=float(objective["p50KlWeight"]),
    )
    regularizers = objective["regularizers"]
    layer = regularizers["softLayerNorm"]
    soft_layer_norm = (
        mean_penalties.float().mean()
        + float(layer["varianceWeight"]) * variance_penalties.float().mean()
    )
    regularization_loss = (
        float(layer["weight"]) * soft_layer_norm
        + float(regularizers["softWeightBound"]["weight"])
        * weight_bound.float()
    )

    def entropy(probabilities: Tensor) -> Tensor:
        log = torch.where(
            probabilities > 0,
            probabilities.clamp_min(torch.finfo(torch.float32).tiny).log(),
            torch.zeros_like(probabilities),
        )
        return -(probabilities * log).sum(dim=-1).mean()

    return {
        "loss": distribution_loss + regularization_loss,
        "distributionLoss": distribution_loss,
        "rawBaseActionKl": raw_kl.mean(),
        "rawP50Kl": torch.quantile(raw_kl, 0.5),
        "curriculumTargetKl": curriculum_kl.mean(),
        "curriculumP50Kl": torch.quantile(curriculum_kl, 0.5),
        "rawTargetEntropy": entropy(raw_targets),
        "curriculumTargetEntropy": entropy(curriculum_targets),
        "curriculumTargetEntropyLogTemperatureDerivative": (
            weighted_target_entropy_log_temperature_derivative(
                raw_targets,
                curriculum_targets,
                weights,
                source_temperature=PRODUCTION_TEMPERATURE,
                target_temperature=train_temperature,
            )
        ),
        "softLayerNorm": soft_layer_norm,
        "softWeightBound": weight_bound.float(),
        "regularizationLoss": regularization_loss,
        "rawKlPerExample": raw_kl,
        "curriculumKlPerExample": curriculum_kl,
    }


@torch.inference_mode()
def evaluate(
    model: LearnedRadiusShrinkingDecoder,
    dataset: CompactDataset,
    split: str,
    batch_size: int,
    device: torch.device,
    train_temperature: float,
    objective: dict[str, Any],
) -> dict[str, float]:
    model.eval()
    accumulator = ScalarAccumulator()
    raw_samples: list[Tensor] = []
    curriculum_samples: list[Tensor] = []
    for features, targets in dataset.batches(split, batch_size, shuffle=False, seed=0):
        features = features.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            metrics = distribution_batch_metrics(
                model_forward(model, features, objective["regularizers"]),
                targets,
                train_temperature,
                objective,
            )
        accumulator.add(metrics, targets.shape[0])
        raw_samples.append(metrics["rawKlPerExample"].detach().cpu())
        curriculum_samples.append(
            metrics["curriculumKlPerExample"].detach().cpu()
        )
    result = accumulator.result()
    raw = torch.cat(raw_samples)
    curriculum = torch.cat(curriculum_samples)
    for label, quantile in (("P50", 0.5), ("P90", 0.9), ("P95", 0.95)):
        result[f"raw{label}Kl"] = float(torch.quantile(raw, quantile))
        result[f"curriculum{label}Kl"] = float(
            torch.quantile(curriculum, quantile)
        )
    result["selectionObjective"] = (
        float(objective["meanKlWeight"]) * float(result["rawBaseActionKl"])
        + float(objective["p50KlWeight"]) * float(result["rawP50Kl"])
    )
    return result


def create_optimizers(
    model: LearnedRadiusShrinkingDecoder,
    training: dict[str, Any],
) -> tuple[torch.optim.Optimizer, torch.optim.Optimizer]:
    muon_parameters, adamw_parameters = optimizer_parameter_groups(model)
    learning_rate = float(training["learningRate"])
    return (
        torch.optim.Muon(
            muon_parameters,
            lr=learning_rate,
            momentum=0.95,
            nesterov=True,
            ns_steps=3,
            eps=1e-7,
            adjust_lr_fn="match_rms_adamw",
        ),
        torch.optim.AdamW(
            adamw_parameters,
            lr=learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            fused=torch.cuda.is_available(),
        ),
    )


def train(
    plan: dict[str, Any],
    manifest: dict[str, Any],
    root: Path,
    stop_after_epoch: int | None = None,
) -> dict[str, Any]:
    training = plan["training"]
    run_root = Path(plan["runDir"])
    checkpoint_root = run_root / "checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    best_file = checkpoint_root / "best.json"
    last_file = checkpoint_root / "last.json"
    event_file = run_root / "events.jsonl"
    fingerprint = plan_hash(plan)
    device = torch.device(str(training["device"]))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA training requested but unavailable")
    seed = int(training["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    torch.set_float32_matmul_precision("high")

    normalization = manifest["normalization"]
    model = LearnedRadiusShrinkingDecoder(
        torch.tensor(normalization["mean"], dtype=torch.float32),
        torch.tensor(normalization["std"], dtype=torch.float32),
        dropout=float(plan["architecture"]["dropout"]),
        dropout_rate=float(plan["architecture"]["dropoutRate"]),
        initial_radius=float(plan["architecture"]["initialRadius"]),
        minimum_radius=float(plan["architecture"]["minimumRadius"]),
        output_count=int(manifest["actionCount"]),
    ).to(device)
    optimizers = create_optimizers(model, training)
    dataset = CompactDataset(root, manifest)
    start_epoch = 0
    global_step = 0
    best_score = math.inf
    best_epoch = -1
    stale_epochs = 0
    lr_stale_epochs = 0
    temperature_stale_epochs = 0
    temperature = float(plan["curriculum"]["startTemperature"])
    start_validation_entropy: float | None = None
    if checkpoint_exists(last_file):
        checkpoint = load_torch_checkpoint(last_file, map_location=device, weights_only=False)
        if checkpoint["planSha256"] != fingerprint:
            raise ValueError("checkpoint belongs to a different plan")
        model.load_state_dict(checkpoint["model"])
        for optimizer, state in zip(optimizers, checkpoint["optimizers"], strict=True):
            optimizer.load_state_dict(state)
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint["globalStep"])
        best_score = float(checkpoint["bestValidationScore"])
        best_epoch = int(checkpoint["bestEpoch"])
        stale_epochs = int(checkpoint["staleEpochs"])
        lr_stale_epochs = int(checkpoint["learningRateStaleEpochs"])
        temperature_stale_epochs = int(checkpoint["temperatureStaleEpochs"])
        temperature = float(checkpoint["nextTrainingTargetTemperature"])
        start_validation_entropy = checkpoint["curriculumStartValidationEntropy"]

    batch_size = int(training["batchSize"])
    validation_batch_size = int(training["evaluationBatchSize"])
    objective = plan["objective"]
    regularizers = objective["regularizers"]
    schedule = training["learningRateSchedule"]
    minimum_lr = float(schedule["finalLearningRate"])
    temperature_patience = int(training["temperatureStalePatience"])
    selection_patience = int(training["earlyStoppingPatience"])
    maximum_epochs = int(training["epochs"])
    stop_ready = False

    for epoch in range(start_epoch, maximum_epochs):
        epoch_started = time.monotonic()
        model.train()
        accumulator = ScalarAccumulator()
        for features, targets in dataset.batches(
            "train", batch_size, shuffle=True, seed=seed + epoch
        ):
            features = features.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            for optimizer in optimizers:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                metrics = distribution_batch_metrics(
                    model_forward(model, features, regularizers),
                    targets,
                    temperature,
                    objective,
                )
            metrics["loss"].backward()
            clip_grad_norm_(model.parameters(), float(training["gradientClip"]), foreach=True)
            for optimizer in optimizers:
                optimizer.step()
            accumulator.add(metrics, targets.shape[0])
            global_step += 1
        validation = evaluate(
            model, dataset, "validation", validation_batch_size,
            device, temperature, objective,
        )
        if start_validation_entropy is None:
            start_validation_entropy = validation["curriculumTargetEntropy"]
        next_temperature, _entropy_goal = next_equal_entropy_step_training_temperature(
            plan,
            temperature,
            validation["curriculumTargetKl"],
            validation["curriculumTargetEntropy"],
            validation["rawTargetEntropy"],
            start_validation_entropy,
            validation["curriculumTargetEntropyLogTemperatureDerivative"],
        )
        temperature_decreased = next_temperature < temperature - 1e-15
        temperature_stale_epochs = 0 if temperature_decreased else temperature_stale_epochs + 1
        score = validation["selectionObjective"]
        improved = score < best_score
        if improved:
            best_score = score
            best_epoch = epoch
            stale_epochs = 0
            lr_stale_epochs = 0
        else:
            stale_epochs += 1
            lr_stale_epochs += 1
        current_lr = float(optimizers[0].param_groups[0]["lr"])
        next_lr = next_stale_learning_rate(schedule, current_lr, lr_stale_epochs)
        learning_rate_reduced = next_lr < current_lr
        restored_from_best_epoch: int | None = None
        if learning_rate_reduced:
            if not checkpoint_exists(best_file):
                raise FileNotFoundError(
                    "learning-rate reduction requires a durable best checkpoint"
                )
            durable_best = load_torch_checkpoint(
                best_file, map_location=device, weights_only=False
            )
            model.load_state_dict(durable_best["model"])
            for optimizer, state in zip(
                optimizers, durable_best["optimizers"], strict=True
            ):
                optimizer.load_state_dict(state)
            for optimizer in optimizers:
                for group in optimizer.param_groups:
                    group["lr"] = next_lr
            restored_from_best_epoch = int(durable_best["epoch"])
            stale_epochs = 0
            lr_stale_epochs = 0
            next_temperature = float(
                durable_best["nextTrainingTargetTemperature"]
            )
            temperature_stale_epochs = int(
                durable_best["temperatureStaleEpochs"]
            )
            start_validation_entropy = durable_best[
                "curriculumStartValidationEntropy"
            ]

        stop_ready = (
            next_lr <= minimum_lr * (1 + 1e-9)
            and temperature_stale_epochs >= temperature_patience
            and stale_epochs >= selection_patience
        )
        pause_requested = (
            stop_after_epoch is not None and epoch >= stop_after_epoch
        )

        checkpoint = {
            "model": model.state_dict(),
            "optimizers": [optimizer.state_dict() for optimizer in optimizers],
            "epoch": epoch,
            "globalStep": global_step,
            "bestValidationScore": best_score,
            "bestEpoch": best_epoch,
            "staleEpochs": stale_epochs,
            "learningRateStaleEpochs": lr_stale_epochs,
            "temperatureStaleEpochs": temperature_stale_epochs,
            "trainingTargetTemperature": temperature,
            "nextTrainingTargetTemperature": next_temperature,
            "curriculumStartValidationEntropy": start_validation_entropy,
            "validation": validation,
            "planSha256": fingerprint,
            "featureCount": int(manifest["featureCount"]),
        }
        # Optimizer checkpoints are large content-addressed objects. Persist a
        # durable resume point every eight epochs, and immediately for every
        # new best or terminal boundary; writing unchanged-size optimizer
        # state every epoch otherwise roughly doubles epoch time on this host.
        if improved or epoch % 8 == 0 or stop_ready or pause_requested:
            save_torch_checkpoint(checkpoint, last_file)
        if improved:
            save_torch_checkpoint(checkpoint, best_file)
        event = {
            "event": "epoch",
            "epoch": epoch,
            "seconds": time.monotonic() - epoch_started,
            "globalStep": global_step,
            "temperature": temperature,
            "nextTemperature": next_temperature,
            "temperatureStaleEpochs": temperature_stale_epochs,
            "learningRate": next_lr,
            "learningRateReduced": learning_rate_reduced,
            "restoredFromBestEpoch": restored_from_best_epoch,
            "train": accumulator.result(),
            "validation": validation,
            "bestEpoch": best_epoch,
            "bestValidationScore": best_score,
            "staleEpochs": stale_epochs,
        }
        run_root.mkdir(parents=True, exist_ok=True)
        with event_file.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, allow_nan=False) + "\n")
        atomic_json(event, run_root / "status.json")
        print(json.dumps(event, allow_nan=False), flush=True)
        temperature = next_temperature

        if stop_ready:
            break
        if pause_requested:
            break

    if not checkpoint_exists(best_file):
        raise RuntimeError("training completed without a best checkpoint")
    if stop_after_epoch is not None and not stop_ready:
        result = {
            "pausedAtEpoch": int(checkpoint["epoch"]),
            "bestEpoch": int(checkpoint["bestEpoch"]),
            "bestValidationScore": float(checkpoint["bestValidationScore"]),
            "checkpoint": str(last_file),
            "sealedTestEvaluated": False,
        }
        atomic_json(result, run_root / "paused.json")
        print(json.dumps({"event": "paused", **result}), flush=True)
        return result
    best = load_torch_checkpoint(best_file, map_location=device, weights_only=False)
    model.load_state_dict(best["model"])
    test = evaluate(
        model, dataset, "test", validation_batch_size, device,
        float(best["trainingTargetTemperature"]), objective,
    )
    result = {
        "completedAtEpoch": int(checkpoint["epoch"]),
        "bestEpoch": int(best["epoch"]),
        "bestValidation": best["validation"],
        "test": test,
        "checkpoint": str(best_file),
        "stoppingCondition": {
            "minimumLearningRate": minimum_lr,
            "temperatureStalePatience": temperature_patience,
            "validationPatience": selection_patience,
        },
    }
    atomic_json(result, run_root / "result.json")
    print(json.dumps({"event": "complete", **result}, allow_nan=False), flush=True)
    return result


def main() -> None:
    args = parse_args()
    repo = Path(__file__).resolve().parent.parent
    plan_file = args.plan if args.plan.is_absolute() else repo / args.plan
    plan = json.loads(plan_file.read_text(encoding="utf-8"))
    compact_root = repo / plan["dataset"]["datasetDir"]
    plan["runDir"] = str(repo / plan["runDir"])
    oracle_value = plan["oracle"]
    oracle_config = DifferentiableExposureValueOracleConfig(
        holding_period_steps=int(oracle_value["holdingPeriodCandles"]),
        decision_delay_steps=int(oracle_value["decisionDelayCandles"]),
        value_horizon_steps=int(oracle_value["valueHorizonCandles"]),
        friction=float(oracle_value["friction"]),
        grid_size=int(oracle_value["gridSize"]),
        temperature=float(oracle_value["temperature"]),
        min_exposure=float(oracle_value["minExposure"]),
        max_exposure=float(oracle_value["maxExposure"]),
        max_effective_exposure=float(oracle_value["maxEffectiveExposure"]),
        quote_borrow_rate=float(oracle_value["quoteBorrowRatePerCandle"]),
        asset_borrow_rate=float(oracle_value["assetBorrowRatePerCandle"]),
    )
    if bool(plan["dataset"].get("prebuilt", False)):
        manifest = json.loads(
            (compact_root / "dataset.json").read_text(encoding="utf-8")
        )
        if manifest.get("prebuilt") is not True \
                or int(manifest.get("featureCount", 0)) \
                != int(plan["dataset"]["featureCount"]) \
                or int(manifest.get("actionCount", 0)) != ACTION_COUNT \
                or manifest.get("oracle") != oracle_config.__dict__:
            raise ValueError("prebuilt causal dataset does not match its plan")
    else:
        source_root = repo / plan["dataset"]["sourceDatasetDir"]
        minute_history_root = repo / plan["dataset"]["minuteHistoryDir"]
        manifest = prepare_dataset(
            source_root,
            minute_history_root,
            compact_root,
            int(plan["dataset"]["featureCount"]),
            oracle_config,
        )
    print(json.dumps({"event": "dataset-ready", "counts": manifest["counts"]}), flush=True)
    if not args.prepare_only:
        train(plan, manifest, compact_root, args.stop_after_epoch)


if __name__ == "__main__":
    main()
