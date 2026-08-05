from __future__ import annotations

import argparse
from collections import OrderedDict
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Iterable, Iterator

import numpy as np
import torch
from torch import Tensor
from torch.nn.utils import clip_grad_norm_

from next_return_dataset import (
    FEATURE_CONTRACT,
    HISTORY_RETURN_COUNT,
    ROWS_PER_DAY,
    SPLIT_CONTRACT,
    ExampleShard,
    count_examples,
    daily_log_return_examples,
    example_span_ms,
    example_rows,
    feature_contract,
    select_example_shards,
    split_contract,
    validate_horizon_return_count,
)
from next_return_sequence import (
    OBJECTIVE_CONTRACT,
    SUMMARY_NAMES,
    SequenceMetricAccumulator,
    SequenceNormalization,
    sequence_objective_loss,
    training_sequence_normalization,
)
from normalized_glu_next_return import (
    ARCHITECTURE_CONTRACT as GLU_ARCHITECTURE_CONTRACT,
    INPUT_NORMALIZATION_MODES,
    NormalizedGluNextReturn,
    TRAINING_POSITION_INPUT_NORMALIZATION,
    optimizer_parameter_groups,
)
from linear_next_return_path import (
    ARCHITECTURE_CONTRACT as LINEAR_ARCHITECTURE_CONTRACT,
    LinearNextReturnPath,
)
from trading_storage import (
    checkpoint_exists,
    load_torch_checkpoint,
    read_candle_column,
    require_under,
    save_torch_checkpoint,
    training_storage_layout,
)


RUNNER_CONTRACT = (
    "cross-shard-batched-distinct-second-configurable-horizon-multi-metric-"
    "reusable-host-buffer-validation-only-optional-summary-loss-glu-stack-"
    "reversible-input-normalization-side-features-v13"
)
SELECTION_CONTRACT = "validation-composite-path-objective-best-checkpoint-v2"


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def atomic_json(value: dict, file: Path) -> None:
    file.parent.mkdir(parents=True, exist_ok=True)
    temporary = file.with_name(f"{file.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    for attempt in range(10):
        try:
            os.replace(temporary, file)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(0.05 * (attempt + 1))


class Reporter:
    def __init__(self, run_dir: Path) -> None:
        self.log_file = run_dir / "logs" / "training.jsonl"
        self.status_file = run_dir / "state" / "status.json"

    def emit(self, event: dict) -> None:
        value = {"at": iso_now(), **event}
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        with self.log_file.open("a", encoding="utf-8", newline="\n") as output:
            output.write(json.dumps(value, separators=(",", ":")) + "\n")
        print(json.dumps(value, separators=(",", ":")), flush=True)

    def status(self, stage: str, **values) -> None:
        atomic_json({
            "pid": os.getpid(),
            "stage": stage,
            "updatedAt": iso_now(),
            **values,
        }, self.status_file)


class CloseCache:
    def __init__(self, history_root: Path, max_entries: int = 5) -> None:
        self.history_root = history_root
        self.max_entries = max_entries
        self.values: OrderedDict[str, np.ndarray] = OrderedDict()

    def load(self, day: str) -> np.ndarray:
        cached = self.values.pop(day, None)
        if cached is not None:
            self.values[day] = cached
            return cached
        file = self.history_root / f"{day}.json"
        if not file.is_file():
            raise FileNotFoundError(f"missing one-second close history: {file}")
        values = read_candle_column(file, "close")
        if values.shape != (86_400,) \
                or not np.isfinite(values).all() \
                or bool((values <= 0).any()):
            raise ValueError(f"invalid one-second close history: {file}")
        self.values[day] = values
        while len(self.values) > self.max_entries:
            self.values.popitem(last=False)
        return values


class NextReturnDataset:
    def __init__(
        self,
        shards: dict[str, list[ExampleShard]],
        history_root: Path,
        *,
        horizon_return_count: int = 1,
    ) -> None:
        self.shards = shards
        self.horizon_return_count = validate_horizon_return_count(
            horizon_return_count
        )
        self.close_cache = CloseCache(history_root)
        self.component_cache: OrderedDict[
            str,
            tuple[np.ndarray, np.ndarray],
        ] = OrderedDict()

    def logical_count(self, split: str) -> int:
        return sum(shard.count for shard in self.shards[split])

    def _component(self, day: str) -> tuple[np.ndarray, np.ndarray]:
        cached = self.component_cache.pop(day, None)
        if cached is not None:
            self.component_cache[day] = cached
            return cached
        current = date.fromisoformat(day)
        value = daily_log_return_examples(
            self.close_cache.load((current - timedelta(days=1)).isoformat()),
            self.close_cache.load(day),
            self.close_cache.load((current + timedelta(days=1)).isoformat()),
            horizon_return_count=self.horizon_return_count,
        )
        self.component_cache[day] = value
        return value

    def iter_batches(
        self,
        split: str,
        batch_size: int,
        *,
        shuffle: bool,
        seed: int,
        shuffle_rows: bool = False,
        reuse_buffers: bool = False,
    ) -> Iterator[tuple[Tensor, Tensor, Tensor]]:
        if batch_size < 1:
            raise ValueError("batch size must be positive")
        generator = np.random.default_rng(seed)
        shards = list(self.shards[split])
        if shuffle:
            generator.shuffle(shards)
        feature_buffer = np.empty(
            (batch_size, HISTORY_RETURN_COUNT), dtype=np.float32
        )
        target_buffer = np.empty(
            (batch_size,) if self.horizon_return_count == 1
            else (batch_size, self.horizon_return_count),
            dtype=np.float32,
        )
        weight_buffer = np.empty(batch_size, dtype=np.float32)
        filled = 0
        for shard in shards:
            history, target = self._component(shard.date)
            rows, weights = example_rows(shard.row_offset, shard.count)
            if rows[0] < 0 or rows[-1] >= ROWS_PER_DAY:
                raise IndexError("one-second row falls outside its UTC day")
            if shuffle_rows:
                order = generator.permutation(rows.shape[0])
                rows = rows[order]
                weights = weights[order]
            position = 0
            while position < rows.shape[0]:
                take = min(batch_size - filled, rows.shape[0] - position)
                selection = rows[position:position + take] if shuffle_rows else slice(
                    int(rows[position]), int(rows[position]) + take
                )
                destination = slice(filled, filled + take)
                feature_buffer[destination] = history[selection]
                target_buffer[destination] = target[selection]
                weight_buffer[destination] = weights[position:position + take]
                filled += take
                position += take
                if filled == batch_size:
                    yield (
                        torch.from_numpy(feature_buffer),
                        torch.from_numpy(target_buffer),
                        torch.from_numpy(weight_buffer),
                    )
                    if not reuse_buffers:
                        feature_buffer = np.empty(
                            (batch_size, HISTORY_RETURN_COUNT), dtype=np.float32
                        )
                        target_buffer = np.empty(
                            (batch_size,) if self.horizon_return_count == 1
                            else (batch_size, self.horizon_return_count),
                            dtype=np.float32,
                        )
                        weight_buffer = np.empty(batch_size, dtype=np.float32)
                    filled = 0
        if filled:
            yield (
                torch.from_numpy(feature_buffer[:filled]),
                torch.from_numpy(target_buffer[:filled]),
                torch.from_numpy(weight_buffer[:filled]),
            )


@dataclass(frozen=True)
class Normalization:
    feature_mean: np.ndarray
    feature_std: np.ndarray
    target_mean: float
    target_std: float

    def validate(self) -> None:
        if self.feature_mean.shape != (HISTORY_RETURN_COUNT,) \
                or self.feature_std.shape != self.feature_mean.shape \
                or not np.isfinite(self.feature_mean).all() \
                or not np.isfinite(self.feature_std).all() \
                or bool((self.feature_std <= 0).any()) \
                or not math.isfinite(self.target_mean) \
                or not math.isfinite(self.target_std) \
                or self.target_std <= 0:
            raise ValueError("next-return normalization is invalid")


def training_normalization(
    dataset: NextReturnDataset,
    *,
    batch_size: int,
) -> Normalization:
    feature_sum = np.zeros(HISTORY_RETURN_COUNT, dtype=np.float64)
    feature_square_sum = np.zeros(HISTORY_RETURN_COUNT, dtype=np.float64)
    target_sum = 0.0
    target_square_sum = 0.0
    total = 0.0
    for feature_tensor, target_tensor, weight_tensor in dataset.iter_batches(
        "train", batch_size, shuffle=False, seed=0
    ):
        features = feature_tensor.numpy().astype(np.float64, copy=False)
        targets = target_tensor.numpy().astype(np.float64, copy=False)
        weights = weight_tensor.numpy().astype(np.float64, copy=False)
        feature_sum += np.einsum("i,ij->j", weights, features)
        feature_square_sum += np.einsum(
            "i,ij->j", weights, np.square(features)
        )
        target_sum += float((weights * targets).sum(dtype=np.float64))
        target_square_sum += float(
            (weights * np.square(targets)).sum(dtype=np.float64)
        )
        total += float(weights.sum(dtype=np.float64))
    if int(round(total)) != dataset.logical_count("train"):
        raise RuntimeError("training normalization did not cover the corpus")
    feature_mean = feature_sum / total
    feature_variance = feature_square_sum / total - np.square(feature_mean)
    target_mean = target_sum / total
    target_variance = target_square_sum / total - target_mean**2
    result = Normalization(
        feature_mean=feature_mean.astype(np.float32),
        feature_std=np.sqrt(np.maximum(feature_variance, 1e-14)).astype(np.float32),
        target_mean=float(target_mean),
        target_std=math.sqrt(max(target_variance, 1e-14)),
    )
    result.validate()
    return result


class MetricAccumulator:
    def __init__(self, target_std: float, device: torch.device) -> None:
        self.target_std = float(target_std)
        self.values = torch.zeros(10, dtype=torch.float64, device=device)

    def add(self, prediction: Tensor, target: Tensor, weights: Tensor) -> None:
        prediction = prediction.detach().to(dtype=torch.float64)
        target = target.detach().to(dtype=torch.float64)
        weights = weights.detach().to(dtype=torch.float64)
        error = prediction - target
        self.values += torch.stack((
            weights.sum(),
            (weights * error.square()).sum(),
            (weights * error.abs()).sum(),
            (weights * ((prediction >= 0) == (target >= 0))).sum(),
            (weights * prediction).sum(),
            (weights * target).sum(),
            (weights * prediction.square()).sum(),
            (weights * target.square()).sum(),
            (weights * prediction * target).sum(),
            (weights * (error / self.target_std).square()).sum(),
        ))

    def result(self) -> dict[str, float | int | None]:
        (
            weight,
            squared,
            absolute,
            direction,
            prediction_sum,
            target_sum,
            prediction_square,
            target_square,
            product,
            normalized_squared,
        ) = (float(value) for value in self.values)
        if weight <= 0:
            raise RuntimeError("cannot finalize empty metrics")
        prediction_mean = prediction_sum / weight
        target_mean = target_sum / weight
        prediction_variance = max(0.0, prediction_square / weight - prediction_mean**2)
        target_variance = max(0.0, target_square / weight - target_mean**2)
        covariance = product / weight - prediction_mean * target_mean
        denominator = math.sqrt(prediction_variance * target_variance)
        mse = squared / weight
        zero_mse = target_square / weight
        return {
            "examples": int(round(weight)),
            "normalizedMse": normalized_squared / weight,
            "mse": mse,
            "rmse": math.sqrt(mse),
            "mae": absolute / weight,
            "directionAccuracy": direction / weight,
            "correlation": covariance / denominator if denominator > 0 else None,
            "predictionMean": prediction_mean,
            "predictionStd": math.sqrt(prediction_variance),
            "targetMean": target_mean,
            "targetStd": math.sqrt(target_variance),
            "zeroBaselineMse": zero_mse,
            "mseSkillVsZero": 1.0 - mse / zero_mse if zero_mse > 0 else 0.0,
        }


def move_batch(
    batch: tuple[Tensor, Tensor, Tensor],
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor]:
    return tuple(
        value.to(
            device,
            non_blocking=device.type == "cuda" and value.is_pinned(),
        )
        for value in batch
    )  # type: ignore[return-value]


def iter_device_batches(
    batches: Iterable[tuple[Tensor, Tensor, Tensor]],
    device: torch.device,
) -> Iterator[tuple[Tensor, Tensor, Tensor]]:
    """Copy reusable host batches without retaining CUDA staging buffers."""
    if device.type != "cuda":
        for batch in batches:
            yield move_batch(batch, device)
        return

    transfer_stream = torch.cuda.Stream(device=device)
    current_stream = torch.cuda.current_stream(device)
    for cpu_batch in batches:
        with torch.cuda.stream(transfer_stream):
            device_batch = move_batch(cpu_batch, device)
        current_stream.wait_stream(transfer_stream)
        for value in device_batch:
            value.record_stream(current_stream)
        yield device_batch


@torch.no_grad()
def evaluate(
    model: NormalizedGluNextReturn | LinearNextReturnPath,
    dataset: NextReturnDataset,
    split: str,
    *,
    batch_size: int,
    target_std: float,
    device: torch.device,
    amp_dtype: torch.dtype,
) -> dict[str, float | int | None]:
    model.eval()
    metrics = MetricAccumulator(target_std, device)
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
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=device.type == "cuda",
        ):
            prediction = model(features)
        metrics.add(prediction, targets, weights)
    return metrics.result()


@torch.no_grad()
def evaluate_sequence(
    model: NormalizedGluNextReturn | LinearNextReturnPath,
    dataset: NextReturnDataset,
    split: str,
    *,
    batch_size: int,
    normalization: SequenceNormalization,
    candle_weight: float,
    summary_weight: float,
    summary_metric_weights: tuple[float, ...],
    device: torch.device,
    amp_dtype: torch.dtype,
    include_per_lead: bool = True,
) -> dict:
    model.eval()
    metrics = SequenceMetricAccumulator(
        normalization,
        candle_weight=candle_weight,
        summary_weight=summary_weight,
        summary_metric_weights=summary_metric_weights,
        device=device,
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
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=device.type == "cuda",
        ):
            prediction = model(features)
        metrics.add(prediction, targets, weights)
    return metrics.result(include_per_lead=include_per_lead)


def build_optimizers(
    model: NormalizedGluNextReturn | LinearNextReturnPath,
    training: dict,
    device: torch.device,
) -> tuple[torch.optim.Optimizer, ...]:
    if isinstance(model, LinearNextReturnPath):
        config = training["optimizer"]
        if config.get("type") != "adamw":
            raise ValueError("linear path model requires AdamW")
        return (torch.optim.AdamW(
            model.parameters(),
            lr=float(training["learningRate"]),
            betas=tuple(float(value) for value in config["betas"]),
            eps=float(config["epsilon"]),
            weight_decay=float(config["weightDecay"]),
            fused=device.type == "cuda",
        ),)
    muon_parameters, adamw_parameters = optimizer_parameter_groups(model)
    config = training["optimizer"]
    muon = config["muon"]
    adamw = config["adamw"]
    return (
        torch.optim.Muon(
            muon_parameters,
            lr=float(training["learningRate"]),
            weight_decay=float(muon["weightDecay"]),
            momentum=float(muon["momentum"]),
            nesterov=bool(muon["nesterov"]),
            ns_steps=int(muon["newtonSchulzSteps"]),
            eps=float(muon["epsilon"]),
            adjust_lr_fn=str(muon["adjustLearningRate"]),
        ),
        torch.optim.AdamW(
            adamw_parameters,
            lr=float(training["learningRate"]),
            betas=tuple(float(value) for value in adamw["betas"]),
            eps=float(adamw["epsilon"]),
            weight_decay=float(adamw["weightDecay"]),
            fused=device.type == "cuda",
        ),
    )


def corpus_fingerprint(
    shards: dict[str, list[ExampleShard]],
    *,
    horizon_return_count: int = 1,
) -> str:
    horizon = validate_horizon_return_count(horizon_return_count)
    payload = {
        "featureContract": (
            FEATURE_CONTRACT if horizon == 1 else feature_contract(horizon)
        ),
        "splitContract": (
            SPLIT_CONTRACT if horizon == 1 else split_contract(horizon)
        ),
        "shards": [
            {
                "split": split,
                "decisionTimeStart": shard.decision_time_start,
                "count": shard.count,
                "date": shard.date,
                "rowOffset": shard.row_offset,
            }
            for split in ("train", "validation", "test")
            for shard in shards[split]
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def canonical_fingerprint(value: dict) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def validate_source_manifest(manifest: dict) -> None:
    if manifest.get("version") != 8 \
            or int(manifest.get("samplingIntervalMs", 0)) != 1_000 \
            or not manifest.get("shards"):
        raise ValueError("next-return source must be the schema-8 one-second corpus")


def validate_plan(plan: dict) -> None:
    required = (
        "id",
        "sourceDatasetDir",
        "datasetDir",
        "runDir",
        "historyDir",
        "testExamples",
        "architecture",
        "training",
    )
    if any(name not in plan or plan[name] in (None, "") for name in required):
        raise ValueError("normalized GLU next-return plan is missing required fields")
    horizon = validate_horizon_return_count(int(plan.get("horizonSeconds", 1)))
    if int(plan.get("testTailOffsetExamples", 0)) < 0:
        raise ValueError("test tail offset must be non-negative")
    if not isinstance(plan.get("evaluateTest", True), bool):
        raise ValueError("evaluateTest must be boolean")
    if horizon > 1:
        objective = plan.get("objective", {})
        metric_weights = objective.get("summaryMetricWeights", {
            name: 1.0 for name in SUMMARY_NAMES
        })
        if objective.get("contract") != OBJECTIVE_CONTRACT \
                or tuple(objective.get("summaryMetrics", ())) != SUMMARY_NAMES \
                or float(objective.get("candleWeight", 0)) <= 0 \
                or float(objective.get("summaryWeight", -1)) < 0 \
                or set(metric_weights) != set(SUMMARY_NAMES) \
                or any(
                    not math.isfinite(float(metric_weights[name]))
                    or float(metric_weights[name]) < 0
                    for name in SUMMARY_NAMES
                ) \
                or sum(float(metric_weights[name]) for name in SUMMARY_NAMES) <= 0:
            raise ValueError("sequence objective contract is invalid")
    architecture = plan["architecture"]
    architecture_contract = architecture.get("contract")
    if architecture_contract == GLU_ARCHITECTURE_CONTRACT:
        widths = architecture.get("widths")
        if not isinstance(widths, list) \
                or not widths \
                or any(
                    not isinstance(width, int)
                    or isinstance(width, bool)
                    or width < 2
                    for width in widths
                ) \
                or architecture.get(
                    "inputNormalization",
                    TRAINING_POSITION_INPUT_NORMALIZATION,
                ) not in INPUT_NORMALIZATION_MODES \
                or not 0 <= float(architecture.get("dropout", -1)) < 1 \
                or not 0 <= float(architecture.get("dropoutRate", -1)) <= 1 \
                or float(architecture.get("initialRadius", 0)) <= float(
                    architecture.get("minimumRadius", 0)
                ):
            raise ValueError("normalized GLU architecture contract is invalid")
    elif architecture_contract == LINEAR_ARCHITECTURE_CONTRACT:
        if architecture.get("type") != "linear":
            raise ValueError("linear path architecture type is invalid")
    else:
        raise ValueError("next-return architecture contract is invalid")
    training = plan["training"]
    if architecture_contract == LINEAR_ARCHITECTURE_CONTRACT:
        optimizer = training.get("optimizer", {})
        if optimizer.get("type") != "adamw" \
                or len(optimizer.get("betas", ())) != 2 \
                or float(optimizer.get("epsilon", 0)) <= 0 \
                or float(optimizer.get("weightDecay", -1)) < 0:
            raise ValueError("linear path AdamW settings are invalid")
    for name in (
        "epochs",
        "batchSize",
        "evaluationBatchSize",
        "earlyStoppingPatience",
        "seed",
    ):
        if int(training.get(name, 0)) < 1:
            raise ValueError(f"training {name} must be positive")
    if training.get("device") not in {"cpu", "cuda"} \
            or training.get("mixedPrecision") != "bfloat16" \
            or float(training.get("learningRate", 0)) <= 0 \
            or float(training.get("gradientClip", 0)) <= 0:
        raise ValueError("normalized GLU runtime settings are invalid")
    schedule = training.get("learningRateSchedule", {})
    if schedule.get("type") != "reduce-on-validation-plateau" \
            or not 0 < float(schedule.get("factor", 0)) < 1 \
            or int(schedule.get("patience", -1)) < 0 \
            or float(schedule.get("minimumLearningRate", 0)) <= 0:
        raise ValueError("normalized GLU learning-rate schedule is invalid")


def resolve(repo_root: Path, value: Path) -> Path:
    return value.resolve() if value.is_absolute() else (repo_root / value).resolve()


def build_predictor(
    architecture: dict,
    feature_mean: Tensor,
    feature_std: Tensor,
    target_mean: Tensor,
    target_std: Tensor,
) -> NormalizedGluNextReturn | LinearNextReturnPath:
    if architecture["contract"] == LINEAR_ARCHITECTURE_CONTRACT:
        return LinearNextReturnPath(
            feature_mean, feature_std, target_mean, target_std
        )
    return NormalizedGluNextReturn(
        feature_mean,
        feature_std,
        target_mean,
        target_std,
        widths=tuple(int(value) for value in architecture["widths"]),
        input_normalization=architecture.get(
            "inputNormalization", TRAINING_POSITION_INPUT_NORMALIZATION
        ),
        dropout=float(architecture["dropout"]),
        dropout_rate=float(architecture["dropoutRate"]),
        initial_radius=float(architecture["initialRadius"]),
        minimum_radius=float(architecture["minimumRadius"]),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train one learned-radius normalized GLU from 120 "
            "completed 1s returns to the next configurable T-second path."
        )
    )
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--stop-after-epoch", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stop_after_epoch is not None and args.stop_after_epoch < 1:
        raise ValueError("--stop-after-epoch must be positive")
    repo_root = Path(__file__).resolve().parents[1]
    plan_file = resolve(repo_root, args.plan)
    plan = json.loads(plan_file.read_text(encoding="utf-8"))
    validate_plan(plan)
    horizon_return_count = validate_horizon_return_count(
        int(plan.get("horizonSeconds", 1))
    )
    is_sequence = horizon_return_count > 1
    evaluate_test = bool(plan.get("evaluateTest", True))
    test_policy = (
        "sealed-until-training-completes"
        if evaluate_test
        else "sealed-validation-only-no-evaluation"
    )
    active_feature_contract = FEATURE_CONTRACT \
        if not is_sequence else feature_contract(horizon_return_count)
    active_split_contract = SPLIT_CONTRACT \
        if not is_sequence else split_contract(horizon_return_count)
    cross_split_purge_ms = example_span_ms(horizon_return_count)
    objective = plan.get("objective", {})
    candle_weight = float(objective.get("candleWeight", 1.0))
    summary_weight = float(objective.get("summaryWeight", 0.0))
    summary_metric_weights = tuple(
        float(objective.get("summaryMetricWeights", {}).get(name, 1.0))
        for name in SUMMARY_NAMES
    )
    plan_fingerprint = canonical_fingerprint(plan)
    layout = training_storage_layout(repo_root)
    source_root = require_under(
        resolve(repo_root, Path(plan["sourceDatasetDir"])),
        layout.datasets,
        "sourceDatasetDir",
    )
    dataset_root = require_under(
        resolve(repo_root, Path(plan["datasetDir"])),
        layout.datasets,
        "datasetDir",
    )
    run_dir = require_under(
        resolve(repo_root, Path(plan["runDir"])),
        layout.runs,
        "runDir",
    )
    history_root = require_under(
        resolve(repo_root, Path(plan["historyDir"])),
        repo_root / "data" / "market" / "immutable" / "refs" / "candles",
        "historyDir",
    )
    reporter = Reporter(run_dir)
    reporter.status("selecting-examples", planId=plan["id"])
    try:
        source_manifest = json.loads(
            (source_root / "dataset.json").read_text(encoding="utf-8")
        )
        validate_source_manifest(source_manifest)
        shards = select_example_shards(
            source_manifest,
            test_count=int(plan["testExamples"]),
            test_tail_offset=int(plan.get("testTailOffsetExamples", 0)),
            horizon_return_count=horizon_return_count,
        )
        counts = count_examples(shards)
        fingerprint = corpus_fingerprint(
            shards, horizon_return_count=horizon_return_count
        )
        architecture = plan["architecture"]
        active_architecture_contract = str(architecture["contract"])
        dummy_target_shape = (horizon_return_count,) if is_sequence else ()
        dummy_model = build_predictor(
            architecture,
            torch.zeros(HISTORY_RETURN_COUNT),
            torch.ones(HISTORY_RETURN_COUNT),
            torch.zeros(dummy_target_shape),
            torch.ones(dummy_target_shape),
        )
        model_parameters = sum(
            parameter.numel() for parameter in dummy_model.parameters()
        )
        if isinstance(dummy_model, NormalizedGluNextReturn):
            optimizer_parameter_groups(dummy_model)
        del dummy_model
        selection_event = {
            "event": "normalized-glu-dataset-selected",
            "planId": plan["id"],
            "corpusFingerprint": fingerprint,
            "counts": counts,
            "testTailOffsetExamples": int(
                plan.get("testTailOffsetExamples", 0)
            ),
            "horizonSeconds": horizon_return_count,
            "crossSplitPurgeMs": cross_split_purge_ms,
            "architectureContract": active_architecture_contract,
            "parameters": model_parameters,
            "testPolicy": test_policy,
        }
        reporter.emit(selection_event)
        if args.validate_only:
            reporter.status("paused", latest=selection_event)
            return

        snapshot = {"planSha256": plan_fingerprint, "plan": plan}
        snapshot_file = run_dir / "state" / "plan.json"
        if snapshot_file.is_file():
            if json.loads(snapshot_file.read_text(encoding="utf-8")) != snapshot:
                raise ValueError("run directory belongs to a different plan")
        else:
            atomic_json(snapshot, snapshot_file)

        reporter.status("computing-training-statistics", planId=plan["id"])
        dataset = NextReturnDataset(
            shards,
            history_root,
            horizon_return_count=horizon_return_count,
        )
        training = plan["training"]
        normalization_reused_from = None
        normalization_dataset_value = plan.get("normalizationDatasetDir")
        if is_sequence and normalization_dataset_value:
            normalization_root = require_under(
                resolve(repo_root, Path(normalization_dataset_value)),
                layout.datasets,
                "normalizationDatasetDir",
            )
            normalization_file = normalization_root / "dataset.json"
            normalization_source = json.loads(
                normalization_file.read_text(encoding="utf-8")
            )
            source_normalization = normalization_source.get("normalization", {})
            if normalization_source.get("featureContract") \
                    != active_feature_contract \
                    or int(normalization_source.get("horizonSeconds", 0)) \
                    != horizon_return_count \
                    or int(normalization_source.get("counts", {}).get("train", 0)) \
                    != counts["train"] \
                    or int(normalization_source.get("counts", {}).get(
                        "validation", 0
                    )) != counts["validation"]:
                raise ValueError("reused normalization corpus is incompatible")
            normalization = SequenceNormalization(
                feature_mean=np.asarray(
                    source_normalization["featureMean"], dtype=np.float32
                ),
                feature_std=np.asarray(
                    source_normalization["featureStd"], dtype=np.float32
                ),
                target_mean=np.asarray(
                    source_normalization["targetMeanByLead"], dtype=np.float32
                ),
                target_std=np.asarray(
                    source_normalization["targetStdByLead"], dtype=np.float32
                ),
                summary_mean=np.asarray(
                    source_normalization["summaryMean"], dtype=np.float32
                ),
                summary_std=np.asarray(
                    source_normalization["summaryStd"], dtype=np.float32
                ),
            )
            normalization.validate()
            normalization_reused_from = str(
                normalization_file.relative_to(repo_root)
            )
            normalization_manifest = {
                "source": "training split only",
                "featureMean": normalization.feature_mean.tolist(),
                "featureStd": normalization.feature_std.tolist(),
                "targetMeanByLead": normalization.target_mean.tolist(),
                "targetStdByLead": normalization.target_std.tolist(),
                "summaryNames": list(SUMMARY_NAMES),
                "summaryMean": normalization.summary_mean.tolist(),
                "summaryStd": normalization.summary_std.tolist(),
            }
        elif is_sequence:
            normalization = training_sequence_normalization(
                dataset,
                batch_size=int(training["evaluationBatchSize"]),
            )
            normalization_manifest = {
                "source": "training split only",
                "featureMean": normalization.feature_mean.tolist(),
                "featureStd": normalization.feature_std.tolist(),
                "targetMeanByLead": normalization.target_mean.tolist(),
                "targetStdByLead": normalization.target_std.tolist(),
                "summaryNames": list(SUMMARY_NAMES),
                "summaryMean": normalization.summary_mean.tolist(),
                "summaryStd": normalization.summary_std.tolist(),
            }
        else:
            normalization = training_normalization(
                dataset,
                batch_size=int(training["evaluationBatchSize"]),
            )
            normalization_manifest = {
                "source": "training split only",
                "featureMean": normalization.feature_mean.tolist(),
                "featureStd": normalization.feature_std.tolist(),
                "targetMean": normalization.target_mean,
                "targetStd": normalization.target_std,
            }
        dataset_manifest = {
            "version": 3,
            "createdAt": iso_now(),
            "planId": plan["id"],
            "sourceDataset": str(
                (source_root / "dataset.json").relative_to(repo_root)
            ),
            "featureContract": active_feature_contract,
            "splitContract": active_split_contract,
            "corpusFingerprint": fingerprint,
            "input": "120 adjacent completed close-to-close one-second log returns",
            "target": (
                f"the immediately following {horizon_return_count} completed "
                "one-second log returns"
            ),
            "horizonSeconds": horizon_return_count,
            "objective": objective if is_sequence else {"type": "per-candle-mse"},
            "normalization": normalization_manifest,
            "normalizationReusedFrom": normalization_reused_from,
            "crossSplitPurgeMs": cross_split_purge_ms,
            "counts": counts,
            "storage": "streamed strided daily return windows; no repeated-row compaction",
        }
        atomic_json(dataset_manifest, dataset_root / "dataset.json")

        device = torch.device(training["device"])
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA training was requested but is unavailable")
        seed = int(training["seed"])
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        torch.set_float32_matmul_precision("high")
        model = build_predictor(
            architecture,
            torch.from_numpy(normalization.feature_mean),
            torch.from_numpy(normalization.feature_std),
            torch.as_tensor(normalization.target_mean),
            torch.as_tensor(normalization.target_std),
        ).to(device)
        summary_std_tensor = torch.as_tensor(
            normalization.summary_std,
            dtype=torch.float32,
            device=device,
        ) if is_sequence else None
        summary_metric_weight_tensor = torch.as_tensor(
            summary_metric_weights,
            dtype=torch.float32,
            device=device,
        ) if is_sequence else None
        optimizers = build_optimizers(model, training, device)
        schedule = training["learningRateSchedule"]
        schedulers = tuple(
            torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=float(schedule["factor"]),
                patience=int(schedule["patience"]),
                threshold=float(schedule["threshold"]),
                threshold_mode="abs",
                min_lr=float(schedule["minimumLearningRate"]),
            )
            for optimizer in optimizers
        )
        start_epoch = 0
        global_step = 0
        best_validation = math.inf
        best_epoch = -1
        stale_epochs = 0
        last_checkpoint = run_dir / "checkpoints" / "last.json"
        best_checkpoint = run_dir / "checkpoints" / "best.json"
        if checkpoint_exists(last_checkpoint):
            checkpoint = load_torch_checkpoint(
                last_checkpoint, map_location=device, weights_only=False
            )
            if checkpoint.get("planSha256") != plan_fingerprint \
                    or checkpoint.get("corpusFingerprint") != fingerprint \
                    or checkpoint.get("architectureContract") \
                    != active_architecture_contract \
                    or checkpoint.get("runnerContract") != RUNNER_CONTRACT:
                raise ValueError("normalized GLU checkpoint contract changed")
            model.load_state_dict(checkpoint["model"])
            for optimizer, state in zip(
                optimizers, checkpoint["optimizers"], strict=True
            ):
                optimizer.load_state_dict(state)
            for scheduler, state in zip(
                schedulers, checkpoint["schedulers"], strict=True
            ):
                scheduler.load_state_dict(state)
            start_epoch = int(checkpoint["epoch"]) + 1
            global_step = int(checkpoint["globalStep"])
            best_validation = float(checkpoint.get(
                "bestValidationScore",
                checkpoint.get("bestValidationNormalizedMse", math.inf),
            ))
            best_epoch = int(checkpoint["bestEpoch"])
            stale_epochs = int(checkpoint["staleEpochs"])
            torch.set_rng_state(checkpoint["torchRngState"].cpu())
            if device.type == "cuda":
                torch.cuda.set_rng_state_all([
                    value.cpu() for value in checkpoint["cudaRngStates"]
                ])

        amp_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        batch_size = int(training["batchSize"])
        evaluation_batch_size = int(training["evaluationBatchSize"])
        maximum_epochs = int(training["epochs"])
        early_stopping_patience = int(training["earlyStoppingPatience"])
        reporter.status(
            "training",
            planId=plan["id"],
            startEpoch=start_epoch,
            parameters=model_parameters,
            horizonSeconds=horizon_return_count,
            bestEpoch=best_epoch,
            bestValidationScore=(
                best_validation if math.isfinite(best_validation) else None
            ),
            testPolicy=test_policy,
        )

        paused = False
        for epoch in range(start_epoch, maximum_epochs):
            epoch_started = time.monotonic()
            model.train()
            if is_sequence:
                train_metrics = SequenceMetricAccumulator(
                    normalization,
                    candle_weight=candle_weight,
                    summary_weight=summary_weight,
                    summary_metric_weights=summary_metric_weights,
                    device=device,
                    track_per_lead=False,
                )
            else:
                train_metrics = MetricAccumulator(normalization.target_std, device)
            for features, targets, weights in iter_device_batches(
                dataset.iter_batches(
                    "train",
                    batch_size,
                    shuffle=True,
                    seed=seed + epoch,
                    reuse_buffers=True,
                ),
                device,
            ):
                for optimizer in optimizers:
                    optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=device.type,
                    dtype=amp_dtype,
                    enabled=device.type == "cuda",
                ):
                    prediction = model(features)
                    if is_sequence:
                        loss = sequence_objective_loss(
                            prediction,
                            targets,
                            weights,
                            target_std=model.target_std,
                            summary_std=summary_std_tensor,
                            summary_metric_weights=summary_metric_weight_tensor,
                            candle_weight=candle_weight,
                            summary_weight=summary_weight,
                        )
                    else:
                        element_loss = (
                            (prediction - targets) / model.target_std
                        ).square()
                        loss = (element_loss * weights).sum() / weights.sum()
                loss.backward()
                clip_grad_norm_(
                    model.parameters(),
                    float(training["gradientClip"]),
                    foreach=device.type == "cuda",
                )
                for optimizer in optimizers:
                    optimizer.step()
                train_metrics.add(prediction, targets, weights)
                global_step += 1
            if is_sequence:
                validation = evaluate_sequence(
                    model,
                    dataset,
                    "validation",
                    batch_size=evaluation_batch_size,
                    normalization=normalization,
                    candle_weight=candle_weight,
                    summary_weight=summary_weight,
                    summary_metric_weights=summary_metric_weights,
                    device=device,
                    amp_dtype=amp_dtype,
                )
                validation_score = float(validation["objective"])
            else:
                validation = evaluate(
                    model,
                    dataset,
                    "validation",
                    batch_size=evaluation_batch_size,
                    target_std=normalization.target_std,
                    device=device,
                    amp_dtype=amp_dtype,
                )
                validation_score = float(validation["normalizedMse"])
            if not math.isfinite(validation_score):
                raise FloatingPointError("validation MSE is non-finite")
            improved = validation_score < best_validation
            if improved:
                best_validation = validation_score
                best_epoch = epoch
                stale_epochs = 0
            else:
                stale_epochs += 1
            learning_rate_before = float(optimizers[0].param_groups[0]["lr"])
            for scheduler in schedulers:
                scheduler.step(validation_score)
            learning_rate = float(optimizers[0].param_groups[0]["lr"])
            checkpoint = {
                "model": model.state_dict(),
                "optimizers": [value.state_dict() for value in optimizers],
                "schedulers": [value.state_dict() for value in schedulers],
                "epoch": epoch,
                "globalStep": global_step,
                "bestValidationScore": best_validation,
                "bestEpoch": best_epoch,
                "staleEpochs": stale_epochs,
                "validation": validation,
                "parameterCount": model_parameters,
                "planSha256": plan_fingerprint,
                "corpusFingerprint": fingerprint,
                "architectureContract": active_architecture_contract,
                "featureContract": active_feature_contract,
                "horizonSeconds": horizon_return_count,
                "objectiveContract": OBJECTIVE_CONTRACT if is_sequence else None,
                "runnerContract": RUNNER_CONTRACT,
                "selectionContract": SELECTION_CONTRACT,
                "torchRngState": torch.get_rng_state(),
                "cudaRngStates": (
                    torch.cuda.get_rng_state_all() if device.type == "cuda" else []
                ),
                "testEvaluated": False,
            }
            save_torch_checkpoint(checkpoint, last_checkpoint)
            if improved:
                save_torch_checkpoint(checkpoint, best_checkpoint)
            event = {
                "event": "epoch",
                "epoch": epoch,
                "epochs": maximum_epochs,
                "seconds": time.monotonic() - epoch_started,
                "globalStep": global_step,
                "train": train_metrics.result(include_per_lead=False)
                if is_sequence else train_metrics.result(),
                "validation": validation,
                "bestValidationScore": best_validation,
                "bestEpoch": best_epoch,
                "staleEpochs": stale_epochs,
                "improved": improved,
                "learningRate": learning_rate,
                "learningRateReduced": learning_rate < learning_rate_before,
                "testEvaluated": False,
            }
            reporter.emit(event)
            reporter.status(
                "training",
                planId=plan["id"],
                latest=event,
                bestEpoch=best_epoch,
                bestValidationScore=best_validation,
                testPolicy=test_policy,
            )
            if args.stop_after_epoch is not None \
                    and epoch + 1 >= args.stop_after_epoch:
                paused = True
                break
            if stale_epochs >= early_stopping_patience:
                break

        if paused:
            pause_event = {
                "event": "normalized-glu-paused",
                "completedEpochs": epoch + 1,
                "bestEpoch": best_epoch,
                "bestValidationScore": best_validation,
                "checkpoint": str(last_checkpoint.relative_to(repo_root)),
                "testEvaluated": False,
            }
            reporter.emit(pause_event)
            reporter.status("paused", planId=plan["id"], latest=pause_event)
            return

        best = load_torch_checkpoint(
            best_checkpoint, map_location=device, weights_only=False
        )
        model.load_state_dict(best["model"])
        if not evaluate_test:
            result = {
                "completedAt": iso_now(),
                "planId": plan["id"],
                "planSha256": plan_fingerprint,
                "corpusFingerprint": fingerprint,
                "architectureContract": active_architecture_contract,
                "parameterCount": model_parameters,
                "horizonSeconds": horizon_return_count,
                "bestEpoch": best_epoch,
                "bestValidationScore": best_validation,
                "bestValidation": best["validation"],
                "checkpoint": str(best_checkpoint.relative_to(repo_root)),
                "testEvaluated": False,
            }
            atomic_json(result, run_dir / "state" / "result.json")
            reporter.emit({
                "event": "normalized-glu-validation-only-complete",
                **result,
            })
            reporter.status("complete", planId=plan["id"], latest=result)
            return
        # Test rows are streamed exactly once, after every model and epoch
        # choice has been frozen by validation.
        if is_sequence:
            test_metrics = evaluate_sequence(
                model,
                dataset,
                "test",
                batch_size=evaluation_batch_size,
                normalization=normalization,
                candle_weight=candle_weight,
                summary_weight=summary_weight,
                summary_metric_weights=summary_metric_weights,
                device=device,
                amp_dtype=amp_dtype,
            )
        else:
            test_metrics = evaluate(
                model,
                dataset,
                "test",
                batch_size=evaluation_batch_size,
                target_std=normalization.target_std,
                device=device,
                amp_dtype=amp_dtype,
            )
        result = {
            "completedAt": iso_now(),
            "planId": plan["id"],
            "planSha256": plan_fingerprint,
            "corpusFingerprint": fingerprint,
            "architectureContract": active_architecture_contract,
            "parameterCount": model_parameters,
            "horizonSeconds": horizon_return_count,
            "bestEpoch": best_epoch,
            "bestValidationScore": best_validation,
            "bestValidation": best["validation"],
            "test": test_metrics,
            "checkpoint": str(best_checkpoint.relative_to(repo_root)),
            "testEvaluated": True,
        }
        atomic_json(result, run_dir / "state" / "result.json")
        reporter.emit({"event": "normalized-glu-complete", **result})
        reporter.status("complete", planId=plan["id"], latest=result)
    except KeyboardInterrupt:
        reporter.status(
            "paused",
            planId=plan["id"],
            message="Interrupted; resume from the last completed epoch.",
        )
        raise
    except Exception as error:
        reporter.status("failed", error=f"{type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    main()
