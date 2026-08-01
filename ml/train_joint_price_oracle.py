from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import queue
import random
import signal
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch import Tensor
from torch.nn.utils import clip_grad_norm_
from trading_storage import (
    checkpoint_exists,
    is_storage_reference,
    load_torch_checkpoint,
    read_candle_column,
    read_shard_array,
    require_under,
    save_torch_checkpoint,
    training_storage_layout,
)

from joint_price_oracle import (
    ARCHITECTURE_CONTRACT,
    OUTPUT_ACTION_COUNT,
    JointLossWeights,
    JointPriceOracleModel,
    joint_price_oracle_objective,
    parameter_count,
)


SECOND_MS = 1_000
DAY_SECONDS = 86_400
DAY_MS = DAY_SECONDS * SECOND_MS
METRIC_NAMES = (
    "loss",
    "crossEntropy",
    "klDivergence",
    "probabilityMse",
    "forecastLoss",
    "nextMovementRmse",
    "directionAccuracy",
    "softLayerNorm",
    "softLayerNormMeanPenalty",
    "softLayerNormVariancePenalty",
)
DATA_CONTRACT = (
    "causal-1s-close-context-ending-at-minute-t-future-closes-t-plus-1-"
    "through-1h-verified-oracle-policy-at-t-hold-60-delay-60-v2"
)


@dataclass(frozen=True)
class CausalSegment:
    split: str
    prediction_time_start: int
    count: int
    target_file: Path
    target_row_offset: int
    step_ms: int = SECOND_MS

    @property
    def prediction_time_end(self) -> int:
        return (
            self.prediction_time_start
            + (self.count - 1) * self.step_ms
        )


@dataclass
class MetricAccumulator:
    examples: int = 0
    totals: dict[str, Tensor] | None = None

    def add(self, metrics: dict[str, Tensor], count: int) -> None:
        if self.totals is None:
            self.totals = {
                name: torch.zeros(
                    (),
                    device=metrics[name].device,
                    dtype=torch.float64,
                )
                for name in METRIC_NAMES
            }
        self.examples += count
        for name in METRIC_NAMES:
            self.totals[name].add_(
                metrics[name].detach().to(dtype=torch.float64),
                alpha=count,
            )

    def result(self) -> dict[str, float]:
        if self.examples < 1 or self.totals is None:
            raise RuntimeError("cannot finalize empty metrics")
        return {
            name: float(value / self.examples)
            for name, value in self.totals.items()
        }


@dataclass
class BatchIteratorFailure:
    error: BaseException


class PrefetchedBatchIterator:
    """Load/decompress future CPU batches while the current batch uses CUDA."""

    def __init__(
        self,
        source: Iterator[tuple[Tensor, Tensor, Tensor]],
        prefetch_batches: int,
    ) -> None:
        self.source = source
        self.queue: queue.Queue[
            tuple[Tensor, Tensor, Tensor] | BatchIteratorFailure | None
        ] = queue.Queue(maxsize=max(1, prefetch_batches))
        self.stop_requested = False
        self.thread = threading.Thread(
            target=self._produce,
            name="joint-price-oracle-batch-prefetch",
            daemon=True,
        )
        self.thread.start()

    def _put(
        self,
        value: tuple[Tensor, Tensor, Tensor]
        | BatchIteratorFailure
        | None,
    ) -> None:
        while not self.stop_requested:
            try:
                self.queue.put(value, timeout=0.1)
                return
            except queue.Full:
                continue

    def _produce(self) -> None:
        try:
            for batch in self.source:
                if self.stop_requested:
                    break
                self._put(batch)
        except BaseException as error:
            self._put(BatchIteratorFailure(error))
        finally:
            self._put(None)

    def __iter__(self) -> PrefetchedBatchIterator:
        return self

    def __next__(self) -> tuple[Tensor, Tensor, Tensor]:
        value = self.queue.get()
        if value is None:
            raise StopIteration
        if isinstance(value, BatchIteratorFailure):
            raise value.error
        return value

    def close(self) -> None:
        self.stop_requested = True
        self.thread.join(timeout=5)

    def __enter__(self) -> PrefetchedBatchIterator:
        return self

    def __exit__(self, _type, _value, _traceback) -> None:
        self.close()


class RunReporter:
    def __init__(self, run_dir: Path, plan_id: str) -> None:
        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.status_file = run_dir / "state" / "status.json"
        self.log_file = run_dir / "logs" / "training.jsonl"
        self.status_file.parent.mkdir(parents=True, exist_ok=True)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.base = {
            "pid": os.getpid(),
            "planId": plan_id,
            "startedAt": iso_now(),
        }

    def emit(self, event: dict) -> None:
        value = {"at": iso_now(), **event}
        with self.log_file.open("a", encoding="utf-8") as output:
            output.write(json.dumps(value, separators=(",", ":")) + "\n")
        print(json.dumps(value, separators=(",", ":")), flush=True)

    def status(self, stage: str, **values) -> None:
        atomic_json({
            **self.base,
            "stage": stage,
            "updatedAt": iso_now(),
            **values,
        }, self.status_file)


class MarketCloseCache:
    def __init__(self, history_root: Path, maximum_days: int = 10) -> None:
        self.history_root = history_root
        self.maximum_days = max(3, int(maximum_days))
        self.days: OrderedDict[str, np.ndarray] = OrderedDict()

    def load_day(self, date_value: str) -> np.ndarray:
        cached = self.days.pop(date_value, None)
        if cached is not None:
            self.days[date_value] = cached
            return cached
        file = self.history_root / f"{date_value}.json"
        if not file.is_file():
            raise FileNotFoundError(f"missing one-second history: {file}")
        closes = read_candle_column(file, "close")
        if closes.shape != (DAY_SECONDS,) \
                or not np.isfinite(closes).all() \
                or bool((closes <= 0).any()):
            raise ValueError(
                f"one-second history must contain {DAY_SECONDS:,} "
                f"positive closes: {file}"
            )
        self.days[date_value] = closes
        while len(self.days) > self.maximum_days:
            self.days.popitem(last=False)
        return closes

    def range(
        self,
        first_close_time: int,
        last_close_time: int,
    ) -> np.ndarray:
        if first_close_time % SECOND_MS != SECOND_MS - 1 \
                or last_close_time % SECOND_MS != SECOND_MS - 1 \
                or last_close_time < first_close_time:
            raise ValueError("close range must use ordered one-second endings")
        first_day = utc_day_ms(first_close_time)
        last_day = utc_day_ms(last_close_time)
        pieces: list[np.ndarray] = []
        day = first_day
        while day <= last_day:
            date_value = datetime.fromtimestamp(
                day / SECOND_MS,
                timezone.utc,
            ).date().isoformat()
            values = self.load_day(date_value)
            start = (
                (first_close_time - day) // SECOND_MS
                if day == first_day
                else 0
            )
            end = (
                (last_close_time - day) // SECOND_MS + 1
                if day == last_day
                else DAY_SECONDS
            )
            pieces.append(values[int(start):int(end)])
            day += DAY_MS
        result = (
            pieces[0]
            if len(pieces) == 1
            else np.concatenate(pieces)
        )
        expected = (
            (last_close_time - first_close_time) // SECOND_MS + 1
        )
        if result.shape != (expected,):
            raise RuntimeError("loaded close range is not contiguous")
        return result


class OracleTargetCache:
    def __init__(
        self,
        maximum_days: int = 3,
        *,
        rows_per_file: int,
        action_count: int,
        pin_memory: bool = False,
    ) -> None:
        self.maximum_days = max(1, int(maximum_days))
        self.pin_memory = bool(pin_memory)
        self.rows_per_file = int(rows_per_file)
        self.action_count = int(action_count)
        self.days: OrderedDict[Path, Tensor] = OrderedDict()

    def load(self, file: Path) -> Tensor:
        cached = self.days.pop(file, None)
        if cached is not None:
            self.days[file] = cached
            return cached
        if not file.is_file():
            raise FileNotFoundError(f"missing raw oracle target: {file}")
        if not is_storage_reference(file):
            raise ValueError(f"oracle target is not a canonical reference: {file}")
        _shard, array = read_shard_array(
            file,
            "<f4",
            (self.rows_per_file, self.action_count),
        )
        if not np.isfinite(array).all() or bool((array < 0).any()):
            raise ValueError(f"raw oracle target is invalid: {file}")
        row_sums = array.sum(axis=1)
        if not np.allclose(row_sums, 1.0, atol=2e-4, rtol=2e-4):
            raise ValueError(f"raw oracle target rows are not normalized: {file}")
        tensor = torch.from_numpy(array.copy())
        if self.pin_memory:
            tensor = tensor.pin_memory()
        self.days[file] = tensor
        while len(self.days) > self.maximum_days:
            self.days.popitem(last=False)
        return tensor


class CausalOracleDataset:
    def __init__(
        self,
        history_root: Path,
        segments: dict[str, list[CausalSegment]],
        context_length: int,
        forecast_horizon: int,
        *,
        target_rows_per_file: int,
        action_count: int,
        close_cache_days: int = 10,
        target_cache_days: int = 3,
        pin_memory: bool = False,
    ) -> None:
        self.segments = segments
        self.context_length = int(context_length)
        self.forecast_horizon = int(forecast_horizon)
        self.close_cache = MarketCloseCache(
            history_root,
            close_cache_days,
        )
        self.target_cache = OracleTargetCache(
            target_cache_days,
            rows_per_file=target_rows_per_file,
            action_count=action_count,
            pin_memory=pin_memory,
        )
        self.pin_memory = bool(pin_memory)

    def count(self, split: str) -> int:
        return sum(segment.count for segment in self.segments[split])

    def batch_count(self, split: str, batch_size: int) -> int:
        return sum(
            math.ceil(segment.count / batch_size)
            for segment in self.segments[split]
        )

    def iter_batches(
        self,
        split: str,
        batch_size: int,
        *,
        shuffle: bool,
        seed: int,
        maximum_batches: int | None = None,
    ) -> Iterator[tuple[Tensor, Tensor, Tensor]]:
        if batch_size < 1:
            raise ValueError("batch size must be positive")
        generator = random.Random(seed)
        segments = list(self.segments[split])
        if shuffle:
            generator.shuffle(segments)
        emitted = 0
        for segment in segments:
            starts = list(range(0, segment.count, batch_size))
            if shuffle:
                generator.shuffle(starts)
            targets = self.target_cache.load(segment.target_file)
            for local_start in starts:
                count = min(batch_size, segment.count - local_start)
                prediction_time_start = (
                    segment.prediction_time_start
                    + local_start * segment.step_ms
                )
                first_close_time = (
                    prediction_time_start
                    - (self.context_length - 1) * SECOND_MS
                )
                last_close_time = (
                    prediction_time_start
                    + (count - 1) * segment.step_ms
                    + self.forecast_horizon * SECOND_MS
                )
                close_range = self.close_cache.range(
                    first_close_time,
                    last_close_time,
                )
                close_windows, future_windows = causal_close_windows(
                    close_range,
                    count,
                    self.context_length,
                    self.forecast_horizon,
                    sample_step_seconds=segment.step_ms // SECOND_MS,
                )
                target_start = segment.target_row_offset + local_start
                target_end = target_start + count
                if target_start < 0 \
                        or target_end > self.target_cache.rows_per_file:
                    raise IndexError("oracle target rows leave their UTC day")
                input_tensor = torch.from_numpy(
                    close_windows[:, :, None]
                )
                future_tensor = torch.from_numpy(
                    future_windows[:, :, None]
                )
                target_tensor = targets[target_start:target_end]
                if self.pin_memory:
                    input_tensor = input_tensor.pin_memory()
                    future_tensor = future_tensor.pin_memory()
                yield input_tensor, future_tensor, target_tensor
                emitted += 1
                if maximum_batches is not None \
                        and emitted >= maximum_batches:
                    return


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the causal TiDE/RLinear/DLinear forecast-to-oracle model."
        )
    )
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument(
        "--check-data",
        action="store_true",
        help="Validate causal pairing and read one batch per split without training.",
    )
    parser.add_argument(
        "--maximum-batches",
        type=int,
        default=None,
        help="Bound batches per split for an explicit smoke run.",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    storage = training_storage_layout(repo_root)
    plan_file = resolve(repo_root, arguments.plan)
    plan = json.loads(plan_file.read_text(encoding="utf-8"))
    validate_plan(plan)
    model_config = plan["model"]
    training = plan["training"]
    run_dir = require_under(
        resolve(repo_root, Path(plan["runDir"])),
        storage.runs,
        "runDir",
    )
    reporter = RunReporter(run_dir, plan["id"])
    ensure_run_is_not_active(reporter.status_file)
    reporter.status("preparing", planFile=str(plan_file))

    target_root = require_under(
        resolve(repo_root, Path(plan["targetReferenceDir"])),
        storage.immutable / "refs" / "oracle",
        "targetReferenceDir",
    )
    history_root = require_under(
        resolve(repo_root, Path(plan["historyDir"])),
        repo_root / "data" / "market" / "immutable" / "refs" / "candles",
        "historyDir",
    )
    (
        segments,
        target_manifest,
        excluded_dates,
        dataset_fingerprint,
    ) = load_causal_segments(
        target_root,
        plan,
        int(model_config["contextLength"]),
        int(model_config["forecastHorizon"]),
    )
    pin_memory = (
        training["device"] == "cuda" and torch.cuda.is_available()
    )
    dataset = CausalOracleDataset(
        history_root,
        segments,
        int(model_config["contextLength"]),
        int(model_config["forecastHorizon"]),
        target_rows_per_file=1_440,
        action_count=int(model_config["actionCount"]),
        close_cache_days=int(training.get("closeCacheDays", 10)),
        target_cache_days=int(training.get("targetCacheDays", 3)),
        pin_memory=pin_memory,
    )
    counts = {
        split: dataset.count(split)
        for split in ("train", "validation", "test")
    }
    reporter.emit({
        "event": "dataset",
        "dataContract": DATA_CONTRACT,
        "predictionDelayMs": 0,
        "contextLength": model_config["contextLength"],
        "forecastHorizon": model_config["forecastHorizon"],
        "actionCount": model_config["actionCount"],
        "oracleContract": target_manifest["contract"],
        "counts": counts,
        "excludedTargetDates": excluded_dates,
        "datasetFingerprint": dataset_fingerprint,
    })

    if arguments.check_data:
        summaries = {}
        for split in ("train", "validation", "test"):
            batch = next(dataset.iter_batches(
                split,
                min(4, int(training["evaluationBatchSize"])),
                shuffle=False,
                seed=int(training["seed"]),
                maximum_batches=1,
            ))
            input_closes, future_closes, target_policy = batch
            summaries[split] = {
                "inputShape": list(input_closes.shape),
                "futureShape": list(future_closes.shape),
                "targetShape": list(target_policy.shape),
                "inputLastClose": float(input_closes[0, -1, 0]),
                "futureFirstClose": float(future_closes[0, 0, 0]),
                "targetProbabilitySum": float(target_policy[0].sum()),
            }
        reporter.status(
            "checked",
            counts=counts,
            summaries=summaries,
            dataContract=DATA_CONTRACT,
        )
        reporter.emit({
            "event": "data-check-complete",
            "counts": counts,
            "summaries": summaries,
        })
        return

    device = torch.device(training["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA training was requested but CUDA is unavailable")
    seed = int(training["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.set_float32_matmul_precision("high")

    model = build_model(model_config).to(device)
    model_parameters = parameter_count(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learningRate"]),
        betas=tuple(training.get("betas", (0.9, 0.999))),
        eps=float(training.get("epsilon", 1e-8)),
        weight_decay=float(training.get("weightDecay", 0.0)),
    )
    scheduler_config = training["learningRateSchedule"]
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(scheduler_config["factor"]),
        patience=int(scheduler_config["patience"]),
        threshold=float(scheduler_config["threshold"]),
        min_lr=float(scheduler_config["minimumLearningRate"]),
    )
    loss_weights = JointLossWeights(
        policy_cross_entropy=float(
            training["lossWeights"]["policyCrossEntropy"]
        ),
        forecast=float(training["lossWeights"]["forecast"]),
        soft_layer_norm=float(
            training["lossWeights"]["softLayerNorm"]
        ),
    )
    last_checkpoint = run_dir / "checkpoints" / "last.json"
    best_checkpoint = run_dir / "checkpoints" / "best.json"
    (
        start_epoch,
        global_step,
        best_validation,
        best_epoch,
        stale_epochs,
    ) = load_resume_checkpoint(
        last_checkpoint,
        model,
        optimizer,
        scheduler,
        plan,
        model_config,
        dataset_fingerprint,
        model_parameters,
        device,
    )
    reporter.status(
        "training",
        counts=counts,
        epoch=start_epoch,
        epochs=training["epochs"],
        globalStep=global_step,
        parameterCount=model_parameters,
        architectureContract=ARCHITECTURE_CONTRACT,
        dataContract=DATA_CONTRACT,
        bestValidation=best_validation,
        bestEpoch=best_epoch,
        resumableCheckpoint=str(last_checkpoint),
    )
    reporter.emit({
        "event": "training-start",
        "epoch": start_epoch,
        "epochs": training["epochs"],
        "globalStep": global_step,
        "parameterCount": model_parameters,
        "device": str(device),
        "architectureContract": ARCHITECTURE_CONTRACT,
        "dataContract": DATA_CONTRACT,
        "resumed": start_epoch > 1,
    })

    stop_requested = False

    def request_stop(_signal_number, _frame) -> None:
        nonlocal stop_requested
        stop_requested = True

    previous_handlers = {
        signal_value: signal.signal(signal_value, request_stop)
        for signal_value in (signal.SIGINT, signal.SIGTERM)
    }
    epochs = int(training["epochs"])
    batch_size = int(training["batchSize"])
    evaluation_batch_size = int(training["evaluationBatchSize"])
    accumulate = int(training.get("gradientAccumulationSteps", 1))
    gradient_clip = float(training["gradientClip"])
    patience = int(training["patience"])
    selection_metric = training.get(
        "selectionMetric",
        "klDivergence",
    )
    maximum_batches = arguments.maximum_batches
    try:
        for epoch in range(start_epoch, epochs + 1):
            epoch_started = time.monotonic()
            model.train()
            optimizer.zero_grad(set_to_none=True)
            train_metrics = MetricAccumulator()
            batches = dataset.batch_count("train", batch_size)
            if maximum_batches is not None:
                batches = min(batches, maximum_batches)
            microbatch_count = 0
            batch_source = dataset.iter_batches(
                "train",
                batch_size,
                shuffle=True,
                seed=seed + epoch,
                maximum_batches=maximum_batches,
            )
            with PrefetchedBatchIterator(
                batch_source,
                int(training.get("prefetchBatches", 2)),
            ) as prefetched_batches:
                for batch_index, batch in enumerate(
                    prefetched_batches,
                    start=1,
                ):
                    input_closes, future_closes, target_policy = move_batch(
                        batch,
                        device,
                    )
                    with autocast_context(device, training):
                        output = model.forward_with_forecast(input_closes)
                        metrics = joint_price_oracle_objective(
                            output,
                            input_closes,
                            future_closes,
                            target_policy,
                            loss_weights,
                            forecast_huber_delta=float(
                                training.get("forecastHuberDelta", 1.0)
                            ),
                            volatility_floor=float(
                                training.get("volatilityFloor", 1e-5)
                            ),
                        )
                        scaled_loss = metrics["loss"] / accumulate
                    scaled_loss.backward()
                    count = input_closes.shape[0]
                    train_metrics.add(metrics, count)
                    microbatch_count += 1
                    update = (
                        microbatch_count % accumulate == 0
                        or batch_index == batches
                    )
                    gradient_norm = None
                    if update:
                        gradient_norm = clip_grad_norm_(
                            model.parameters(),
                            gradient_clip,
                            error_if_nonfinite=True,
                        )
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                        global_step += 1
                    if update and (
                        global_step % int(training["logEverySteps"]) == 0
                        or batch_index == batches
                    ):
                        latest = {
                            name: float(value.detach())
                            for name, value in metrics.items()
                        }
                        reporter.emit({
                            "event": "train-step",
                            "epoch": epoch,
                            "epochs": epochs,
                            "batch": batch_index,
                            "batches": batches,
                            "globalStep": global_step,
                            "learningRate": optimizer.param_groups[0]["lr"],
                            "gradientNorm": (
                                float(gradient_norm)
                                if gradient_norm is not None
                                else None
                            ),
                            "latest": latest,
                        })
                        reporter.status(
                            "training",
                            counts=counts,
                            epoch=epoch,
                            epochs=epochs,
                            globalStep=global_step,
                            latest=latest,
                            bestValidation=best_validation,
                            bestEpoch=best_epoch,
                            resumableCheckpoint=str(last_checkpoint),
                        )
                    if stop_requested:
                        break

            train_result = train_metrics.result()
            if stop_requested:
                checkpoint = build_training_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    epoch=epoch,
                    global_step=global_step,
                    best_validation=best_validation,
                    best_epoch=best_epoch,
                    stale_epochs=stale_epochs,
                    validation=None,
                    model_parameters=model_parameters,
                    dataset_fingerprint=dataset_fingerprint,
                    model_config=model_config,
                    plan_id=plan["id"],
                    device=device,
                    interrupted=True,
                )
                atomic_torch_save(checkpoint, last_checkpoint)
                reporter.emit({
                    "event": "training-paused",
                    "epoch": epoch,
                    "globalStep": global_step,
                    "train": train_result,
                    "checkpoint": str(last_checkpoint),
                })
                reporter.status(
                    "paused",
                    epoch=epoch,
                    globalStep=global_step,
                    bestValidation=best_validation,
                    bestEpoch=best_epoch,
                    resumableCheckpoint=str(last_checkpoint),
                    message=(
                        "Training was interrupted after a durable checkpoint "
                        "and can resume from the canonical last checkpoint pointer."
                    ),
                )
                return
            validation_result = evaluate(
                model,
                dataset,
                "validation",
                evaluation_batch_size,
                device,
                training,
                loss_weights,
                maximum_batches=maximum_batches,
            )
            if selection_metric not in validation_result:
                raise ValueError(
                    f"unknown validation selection metric: {selection_metric}"
                )
            selection_value = validation_result[selection_metric]
            improved = selection_value < best_validation
            if improved:
                best_validation = selection_value
                best_epoch = epoch
                stale_epochs = 0
            else:
                stale_epochs += 1
            scheduler.step(selection_value)
            checkpoint = build_training_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch=epoch,
                global_step=global_step,
                best_validation=best_validation,
                best_epoch=best_epoch,
                stale_epochs=stale_epochs,
                validation=validation_result,
                model_parameters=model_parameters,
                dataset_fingerprint=dataset_fingerprint,
                model_config=model_config,
                plan_id=plan["id"],
                device=device,
                interrupted=False,
            )
            atomic_torch_save(checkpoint, last_checkpoint)
            if improved:
                atomic_torch_save({
                    "model": checkpoint["model"],
                    "epoch": epoch,
                    "globalStep": global_step,
                    "validation": validation_result,
                    "parameterCount": model_parameters,
                    "architectureContract": ARCHITECTURE_CONTRACT,
                    "dataContract": DATA_CONTRACT,
                    "datasetFingerprint": dataset_fingerprint,
                    "modelConfig": model_config,
                    "planId": plan["id"],
                }, best_checkpoint)
            epoch_seconds = time.monotonic() - epoch_started
            reporter.emit({
                "event": "epoch",
                "epoch": epoch,
                "epochs": epochs,
                "globalStep": global_step,
                "seconds": epoch_seconds,
                "train": train_result,
                "validation": validation_result,
                "selectionMetric": selection_metric,
                "bestValidation": best_validation,
                "bestEpoch": best_epoch,
                "staleEpochs": stale_epochs,
                "learningRate": optimizer.param_groups[0]["lr"],
                "improved": improved,
                "stopRequested": stop_requested,
            })
            if stale_epochs > patience:
                break
    finally:
        for signal_value, previous_handler in previous_handlers.items():
            signal.signal(signal_value, previous_handler)

    if not checkpoint_exists(best_checkpoint):
        raise RuntimeError("training completed without a best checkpoint")
    best = load_torch_checkpoint(
        best_checkpoint,
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(best["model"])
    test_result = evaluate(
        model,
        dataset,
        "test",
        evaluation_batch_size,
        device,
        training,
        loss_weights,
        maximum_batches=maximum_batches,
    )
    result = {
        "event": "training-complete",
        "bestEpoch": best["epoch"],
        "bestValidation": best_validation,
        "test": test_result,
        "checkpoint": str(best_checkpoint),
        "parameterCount": model_parameters,
        "architectureContract": ARCHITECTURE_CONTRACT,
        "dataContract": DATA_CONTRACT,
    }
    reporter.emit(result)
    reporter.status("complete", **result)


@torch.no_grad()
def evaluate(
    model: JointPriceOracleModel,
    dataset: CausalOracleDataset,
    split: str,
    batch_size: int,
    device: torch.device,
    training: dict,
    loss_weights: JointLossWeights,
    *,
    maximum_batches: int | None,
) -> dict[str, float]:
    model.eval()
    accumulator = MetricAccumulator()
    batch_source = dataset.iter_batches(
        split,
        batch_size,
        shuffle=False,
        seed=int(training["seed"]),
        maximum_batches=maximum_batches,
    )
    with PrefetchedBatchIterator(
        batch_source,
        int(training.get("prefetchBatches", 2)),
    ) as prefetched_batches:
        for batch in prefetched_batches:
            input_closes, future_closes, target_policy = move_batch(
                batch,
                device,
            )
            with autocast_context(device, training):
                output = model.forward_with_forecast(input_closes)
                metrics = joint_price_oracle_objective(
                    output,
                    input_closes,
                    future_closes,
                    target_policy,
                    loss_weights,
                    forecast_huber_delta=float(
                        training.get("forecastHuberDelta", 1.0)
                    ),
                    volatility_floor=float(
                        training.get("volatilityFloor", 1e-5)
                    ),
                )
            accumulator.add(metrics, input_closes.shape[0])
    return accumulator.result()


def build_model(config: dict) -> JointPriceOracleModel:
    normalization = config["branchNormalization"]
    return JointPriceOracleModel(
        context_length=int(config["contextLength"]),
        forecast_horizon=int(config["forecastHorizon"]),
        variable_count=int(config.get("variableCount", 1)),
        moving_average_window=int(config["movingAverageWindow"]),
        patch_length=int(config["patchLength"]),
        linear_rank=int(config["linearRank"]),
        aggregate_mode=config["aggregateMode"],
        tide_hidden_width=int(config["tideHiddenWidth"]),
        tide_layer_count=int(config["tideLayerCount"]),
        policy_hidden_width=int(config["policyHiddenWidth"]),
        policy_layer_count=int(config["policyLayerCount"]),
        action_count=int(config["actionCount"]),
        dropout=float(config["dropout"]),
        normalization_epsilon=float(config["normalizationEpsilon"]),
        normalization_family=normalization["family"],
        normalization_initial_scale=float(
            normalization["initialScale"]
        ),
        normalization_minimum_scale=float(
            normalization["minimumScale"]
        ),
    )


def move_batch(
    batch: tuple[Tensor, Tensor, Tensor],
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor]:
    return tuple(
        value.to(device, non_blocking=True)
        for value in batch
    )


def autocast_context(device: torch.device, training: dict):
    precision = training.get("mixedPrecision", "float32")
    if device.type != "cuda" or precision == "float32":
        return torch.autocast(
            device_type=device.type,
            enabled=False,
        )
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }.get(precision)
    if dtype is None:
        raise ValueError(f"unsupported mixed precision: {precision}")
    return torch.autocast(
        device_type="cuda",
        dtype=dtype,
    )


def load_causal_segments(
    target_root: Path,
    plan: dict,
    context_length: int,
    forecast_horizon: int,
) -> tuple[
    dict[str, list[CausalSegment]],
    dict,
    list[str],
    str,
]:
    target_files = sorted(target_root.glob("*.json"))
    split_config = plan["dataSplit"]
    validation_days = int(split_config["validationDays"])
    test_days = int(split_config["testDays"])
    if len(target_files) <= validation_days + test_days:
        raise ValueError("verified oracle cache does not cover every data split")
    model_action_count = int(plan["model"]["actionCount"])
    oracle_target = plan["oracleTarget"]
    expected_options = {
        "holdingPeriodSteps": 60,
        "decisionDelaySteps": 60,
        "valueHorizonSteps": forecast_horizon,
        "gridSize": 255,
    }
    expected_options.update(oracle_target.get("options", {}))
    first_contract: dict | None = None
    contract_hash: str | None = None
    reference_fingerprint: list[tuple[str, str]] = []
    selected: dict[str, list[CausalSegment]] = {
        split: [] for split in ("train", "validation", "test")
    }
    train_count = len(target_files) - validation_days - test_days
    validation_end = len(target_files) - test_days
    for index, target_file in enumerate(target_files):
        reference = json.loads(target_file.read_text(encoding="utf-8"))
        if not is_storage_reference(target_file):
            raise ValueError(f"oracle target is not canonical: {target_file}")
        sequence = reference.get("sequence", {})
        layout = reference.get("layout", {})
        metadata = reference.get("metadata", {})
        contract = metadata.get("contract", {})
        options = contract.get("options", {})
        date_value = target_file.stem
        day = int(datetime.fromisoformat(
            f"{date_value}T00:00:00+00:00"
        ).timestamp() * SECOND_MS)
        if sequence != {
            "start": day + SECOND_MS - 1,
            "step": 60 * SECOND_MS,
            "count": 1_440,
            "unit": "unix-ms",
        }:
            raise ValueError(f"oracle target timeline is incompatible: {target_file}")
        if layout.get("dtype") != "float32-le" \
                or int(layout.get("rows", -1)) != 1_440 \
                or int(layout.get("columns", -1)) != model_action_count:
            raise ValueError(f"oracle target layout is incompatible: {target_file}")
        if contract.get("intervalMs") != SECOND_MS \
                or contract.get("decisionIntervalMs") != 60 * SECOND_MS \
                or any(options.get(key) != value for key, value in expected_options.items()) \
                or len(contract.get("usableGrid", [])) != model_action_count:
            raise ValueError(f"oracle target contract is incompatible: {target_file}")
        current_hash = metadata.get("contractHash")
        if not isinstance(current_hash, str):
            raise ValueError(f"oracle target contract hash is missing: {target_file}")
        if first_contract is None:
            first_contract = contract
            contract_hash = current_hash
        elif current_hash != contract_hash or contract != first_contract:
            raise ValueError("verified oracle references mix different contracts")
        split = (
            "train" if index < train_count
            else "validation" if index < validation_end
            else "test"
        )
        object_hash = reference.get("object", {}).get("contentHash")
        if not isinstance(object_hash, str):
            raise ValueError(f"oracle target object hash is missing: {target_file}")
        reference_fingerprint.append((date_value, object_hash))
        selected[split].append(CausalSegment(
            split=split,
            prediction_time_start=day + SECOND_MS - 1,
            count=1_440,
            target_file=target_file,
            target_row_offset=0,
            step_ms=60 * SECOND_MS,
        ))
    selected = purge_cross_split_windows(
        selected,
        context_length,
        forecast_horizon,
    )
    for split, segments in selected.items():
        if not segments:
            raise ValueError(f"causal {split} split is empty")
    fingerprint_value = {
        "oracleContractHash": contract_hash,
        "targetReferences": reference_fingerprint,
        "dataContract": DATA_CONTRACT,
        "contextLength": context_length,
        "forecastHorizon": forecast_horizon,
        "validationDays": validation_days,
        "testDays": test_days,
        "counts": {
            split: sum(segment.count for segment in segments)
            for split, segments in selected.items()
        },
        "segments": {
            split: [
                (
                    segment.prediction_time_start,
                    segment.count,
                    str(segment.target_file.relative_to(target_root)),
                    segment.target_row_offset,
                    segment.step_ms,
                )
                for segment in segments
            ]
            for split, segments in selected.items()
        },
    }
    fingerprint = hashlib.sha256(json.dumps(
        fingerprint_value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")).hexdigest()
    return selected, {"contract": first_contract}, [], fingerprint


def purge_cross_split_windows(
    segments_by_split: dict[str, list[CausalSegment]],
    context_length: int,
    forecast_horizon: int,
) -> dict[str, list[CausalSegment]]:
    """Keep every model input/forecast window disjoint across source splits."""
    all_segments = sorted(
        (
            segment
            for segments in segments_by_split.values()
            for segment in segments
        ),
        key=lambda segment: segment.prediction_time_start,
    )
    result: dict[str, list[CausalSegment]] = {
        split: [] for split in segments_by_split
    }
    for index, segment in enumerate(all_segments):
        start = segment.prediction_time_start
        end = segment.prediction_time_end
        for previous in reversed(all_segments[:index]):
            if previous.split == segment.split:
                continue
            if previous.prediction_time_end \
                    < start - context_length * SECOND_MS:
                break
            start = max(
                start,
                previous.prediction_time_end
                + context_length * SECOND_MS,
            )
        for following in all_segments[index + 1:]:
            if following.split == segment.split:
                continue
            if following.prediction_time_start \
                    > end + forecast_horizon * SECOND_MS:
                break
            end = min(
                end,
                following.prediction_time_start
                - (forecast_horizon + 1) * SECOND_MS,
            )
        skipped = max(0, math.ceil(
            (start - segment.prediction_time_start) / segment.step_ms
        ))
        start = segment.prediction_time_start + skipped * segment.step_ms
        if end < start:
            continue
        count = (end - start) // segment.step_ms + 1
        result[segment.split].append(CausalSegment(
            split=segment.split,
            prediction_time_start=start,
            count=int(count),
            target_file=segment.target_file,
            target_row_offset=segment.target_row_offset + int(skipped),
            step_ms=segment.step_ms,
        ))
    return {
        split: merge_causal_segments(segments)
        for split, segments in result.items()
    }


def causal_close_windows(
    close_range: np.ndarray,
    count: int,
    context_length: int,
    forecast_horizon: int,
    *,
    sample_step_seconds: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Split a contiguous range into inputs through t and labels after t."""
    if close_range.ndim != 1 \
            or min(
                count,
                context_length,
                forecast_horizon,
                sample_step_seconds,
            ) < 1:
        raise ValueError("causal close window dimensions are invalid")
    window_width = context_length + forecast_horizon
    expected_range = (
        context_length
        + (count - 1) * sample_step_seconds
        + forecast_horizon
    )
    if close_range.shape != (expected_range,):
        raise ValueError(
            "close range does not match the causal window dimensions"
        )
    all_windows = np.lib.stride_tricks.sliding_window_view(
        close_range,
        window_width,
    )
    windows = all_windows[::sample_step_seconds]
    if windows.shape != (count, window_width):
        raise RuntimeError("causal close windows are not aligned to predictions")
    return (
        np.asarray(
            windows[:, :context_length],
            dtype=np.float32,
        ).copy(),
        np.asarray(
            windows[:, context_length:],
            dtype=np.float32,
        ).copy(),
    )


def merge_causal_segments(
    segments: list[CausalSegment],
) -> list[CausalSegment]:
    merged: list[CausalSegment] = []
    for segment in sorted(
        segments,
        key=lambda value: value.prediction_time_start,
    ):
        if merged:
            previous = merged[-1]
            if previous.split == segment.split \
                    and previous.target_file == segment.target_file \
                    and previous.step_ms == segment.step_ms \
                    and previous.prediction_time_end + previous.step_ms \
                    == segment.prediction_time_start \
                    and previous.target_row_offset + previous.count \
                    == segment.target_row_offset:
                merged[-1] = CausalSegment(
                    split=previous.split,
                    prediction_time_start=previous.prediction_time_start,
                    count=previous.count + segment.count,
                    target_file=previous.target_file,
                    target_row_offset=previous.target_row_offset,
                    step_ms=previous.step_ms,
                )
                continue
        merged.append(segment)
    return merged


def take_tail(
    segments: list[CausalSegment],
    count: int,
) -> list[CausalSegment]:
    if count < 1:
        raise ValueError("test example count must be positive")
    remaining = count
    result: list[CausalSegment] = []
    for segment in reversed(segments):
        take = min(remaining, segment.count)
        offset = segment.count - take
        result.append(CausalSegment(
            split=segment.split,
            prediction_time_start=(
                segment.prediction_time_start + offset * segment.step_ms
            ),
            count=take,
            target_file=segment.target_file,
            target_row_offset=segment.target_row_offset + offset,
            step_ms=segment.step_ms,
        ))
        remaining -= take
        if remaining == 0:
            break
    if remaining:
        raise ValueError(
            f"test split has {count - remaining:,}/{count:,} requested examples"
        )
    return list(reversed(result))


def validate_plan(plan: dict) -> None:
    required = (
        "id",
        "label",
        "targetReferenceDir",
        "historyDir",
        "runDir",
        "samplingIntervalMs",
        "predictionDelayMs",
        "oracleTarget",
        "dataSplit",
        "model",
        "training",
    )
    if any(key not in plan for key in required):
        raise ValueError("joint price-oracle plan is incomplete")
    if plan["samplingIntervalMs"] != SECOND_MS \
            or plan["predictionDelayMs"] != 0:
        raise ValueError(
            "joint price-oracle training requires exact causal one-second pairing"
        )
    model = plan["model"]
    model_required = (
        "contextLength",
        "forecastHorizon",
        "movingAverageWindow",
        "patchLength",
        "linearRank",
        "aggregateMode",
        "tideHiddenWidth",
        "tideLayerCount",
        "policyHiddenWidth",
        "policyLayerCount",
        "actionCount",
        "dropout",
        "normalizationEpsilon",
        "branchNormalization",
    )
    if any(key not in model for key in model_required):
        raise ValueError("joint model configuration is incomplete")
    if int(model.get("variableCount", 1)) != 1:
        raise ValueError(
            "the current BTCUSDT source supplies exactly one close variable"
        )
    if min(
        int(model["contextLength"]),
        int(model["forecastHorizon"]),
        int(model["movingAverageWindow"]),
        int(model["patchLength"]),
        int(model["linearRank"]),
        int(model["tideHiddenWidth"]),
        int(model["tideLayerCount"]),
        int(model["policyHiddenWidth"]),
        int(model["policyLayerCount"]),
        int(model["actionCount"]),
    ) < 1:
        raise ValueError("joint model dimensions must be positive")
    oracle_target = plan["oracleTarget"]
    if int(model["actionCount"]) != OUTPUT_ACTION_COUNT \
            or int(model["forecastHorizon"]) != 3_600 \
            or oracle_target.get("holdingPeriodSteps") != 60 \
            or oracle_target.get("decisionDelaySteps") != 60 \
            or oracle_target.get("valueHorizonSteps") != 3_600:
        raise ValueError(
            "joint training requires the verified 1h horizon, 1m delay, "
            "1m hold oracle contract"
        )
    split = plan["dataSplit"]
    if min(int(split["validationDays"]), int(split["testDays"])) < 1:
        raise ValueError("validationDays and testDays must be positive")
    if model["contextLength"] < max(
        model["movingAverageWindow"],
        model["patchLength"],
    ):
        raise ValueError("joint model context is shorter than its aggregate")
    if int(model["linearRank"]) > min(
        int(model["contextLength"]),
        int(model["forecastHorizon"]),
    ):
        raise ValueError("joint temporal linear rank is too large")
    training = plan["training"]
    training_required = (
        "targetRepresentation",
        "epochs",
        "batchSize",
        "evaluationBatchSize",
        "learningRate",
        "learningRateSchedule",
        "gradientClip",
        "patience",
        "seed",
        "device",
        "mixedPrecision",
        "logEverySteps",
        "lossWeights",
    )
    if any(key not in training for key in training_required):
        raise ValueError("joint training configuration is incomplete")
    if training["targetRepresentation"] != "verifiedOracleProbabilities":
        raise ValueError(
            "training requires the canonical verified oracle probabilities"
        )
    loss_weights = training["lossWeights"]
    if any(
        key not in loss_weights
        for key in (
            "policyCrossEntropy",
            "forecast",
            "softLayerNorm",
        )
    ) or any(float(value) < 0 for value in loss_weights.values()) \
            or float(loss_weights["policyCrossEntropy"]) <= 0 \
            or float(loss_weights["forecast"]) <= 0:
        raise ValueError("joint loss weights are invalid")


def build_training_checkpoint(
    model: JointPriceOracleModel,
    optimizer: torch.optim.Optimizer,
    scheduler,
    *,
    epoch: int,
    global_step: int,
    best_validation: float,
    best_epoch: int,
    stale_epochs: int,
    validation: dict[str, float] | None,
    model_parameters: int,
    dataset_fingerprint: str,
    model_config: dict,
    plan_id: str,
    device: torch.device,
    interrupted: bool,
) -> dict:
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "globalStep": global_step,
        "bestValidation": best_validation,
        "bestEpoch": best_epoch,
        "staleEpochs": stale_epochs,
        "validation": validation,
        "parameterCount": model_parameters,
        "architectureContract": ARCHITECTURE_CONTRACT,
        "dataContract": DATA_CONTRACT,
        "datasetFingerprint": dataset_fingerprint,
        "modelConfig": model_config,
        "planId": plan_id,
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.random.get_rng_state(),
            "cuda": (
                torch.cuda.get_rng_state_all()
                if device.type == "cuda"
                else None
            ),
        },
        "interrupted": interrupted,
    }


def load_resume_checkpoint(
    file: Path,
    model: JointPriceOracleModel,
    optimizer: torch.optim.Optimizer,
    scheduler,
    plan: dict,
    model_config: dict,
    dataset_fingerprint: str,
    model_parameters: int,
    device: torch.device,
) -> tuple[int, int, float, int, int]:
    if not checkpoint_exists(file):
        return 1, 0, math.inf, 0, 0
    checkpoint = load_torch_checkpoint(
        file, map_location=device, weights_only=False
    )
    if checkpoint.get("planId") != plan["id"] \
            or checkpoint.get("architectureContract") \
            != ARCHITECTURE_CONTRACT \
            or checkpoint.get("dataContract") != DATA_CONTRACT \
            or checkpoint.get("datasetFingerprint") \
            != dataset_fingerprint \
            or checkpoint.get("modelConfig") != model_config \
            or checkpoint.get("parameterCount") != model_parameters:
        raise ValueError("resume checkpoint architecture is incompatible")
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    rng = checkpoint.get("rng", {})
    if "python" in rng:
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.random.set_rng_state(rng["torch"].cpu())
        if device.type == "cuda" and rng.get("cuda") is not None:
            torch.cuda.set_rng_state_all([
                value.cpu() for value in rng["cuda"]
            ])
    return (
        int(checkpoint["epoch"]) + 1,
        int(checkpoint["globalStep"]),
        float(checkpoint["bestValidation"]),
        int(checkpoint["bestEpoch"]),
        int(checkpoint["staleEpochs"]),
    )


def ensure_run_is_not_active(status_file: Path) -> None:
    if not status_file.is_file():
        return
    try:
        status = json.loads(status_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    pid = status.get("pid")
    if not isinstance(pid, int) or pid == os.getpid() \
            or status.get("stage") in {
                "complete",
                "failed",
                "paused",
                "checked",
            }:
        return
    try:
        os.kill(pid, 0)
    except OSError:
        return
    raise RuntimeError(f"joint price-oracle training is already PID {pid}")


def atomic_torch_save(value: dict, file: Path) -> None:
    save_torch_checkpoint(value, file)


def atomic_json(value: dict, file: Path) -> None:
    file.parent.mkdir(parents=True, exist_ok=True)
    temporary = file.with_suffix(file.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, file)


def utc_day_ms(timestamp: int) -> int:
    return timestamp - timestamp % DAY_MS


def resolve(repo_root: Path, value: Path) -> Path:
    return value if value.is_absolute() else repo_root / value


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    main()
