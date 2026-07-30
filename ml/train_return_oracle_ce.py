from __future__ import annotations

import argparse
import copy
import gzip
import json
import math
import os
import queue
import random
import re
import threading
import time
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch import Tensor
from torch.nn.utils import clip_grad_norm_
import zstandard

from return_oracle_ce import (
    BRANCH_NORMALIZATION_EPSILON,
    HIDDEN_LAYER_COUNT,
    HIDDEN_WIDTHS,
    INPUT_RETURN_COUNT,
    OUTPUT_ACTION_COUNT,
    ReturnOracleMlp,
    centering_matrix_constraint_components,
    oracle_policy_objective,
    parameter_count,
    soft_weight_bound_penalty,
)


SECOND_MS = 1_000
MINUTE_SECONDS = 60
INPUT_HORIZON_SECONDS = INPUT_RETURN_COUNT * MINUTE_SECONDS
DAY_SECONDS = 86_400
MINUTE_ROWS_PER_DAY = 1_441
TEST_EXAMPLES = 1_000_000
DENSE_RESIDUAL_PATH_COUNT = sum(
    max(layer_index - 1, 0)
    for layer_index in range(HIDDEN_LAYER_COUNT)
)
FEATURE_CONTRACT = (
    "train-position-standardized-completed-minute-close-only-simple-returns-v2"
)
ARCHITECTURE_CONTRACT = (
    "eight-uniform-256-fused-glu-shared-layer-centering-"
    "tanh-learned-radius-full-a-post-bias-full-normalized-global-input-glu-"
    "dense-all-prior-layer-residuals-fixed-c-path-shared-value-gate-a-v38"
)
OBJECTIVE_CONTRACT = (
    "soft-target-ce-plus-independent-skew-reverse-kl-p1-entropy-sharpness-"
    "p1-weight-point1-soft-layernorm-centering-weight-regularizers-v16"
)
RUNNER_CONTRACT = (
    "compact-minute-multiplicity-weighted-bf16-hybrid-muon-adamw-v10"
)
RETURN_COMPONENT_SUFFIX = (
    ".completed-minute-simple-returns-60.compact.f16.zst"
)
FLOAT16_ROW_BYTES = INPUT_RETURN_COUNT * np.dtype("<f2").itemsize
CLOSE_PATTERN = re.compile(rb'"close":([-+0-9.eE]+),')
METRIC_NAMES = (
    "loss",
    "crossEntropy",
    "baseKlDivergence",
    "reverseKlDivergence",
    "probabilityMse",
    "targetEntropy",
    "predictedEntropy",
    "entropyGap",
    "entropySharpness",
    "reverseKlGate",
    "entropySharpnessGate",
    "softLayerNorm",
    "softLayerNormMeanPenalty",
    "softLayerNormVariancePenalty",
    "distributionLayer",
    "distributionLayerSumPenalty",
    "distributionLayerNegativePenalty",
    "softWeightBound",
    "centeringConstraint",
    "centeringIdempotence",
    "centeringSymmetry",
)


def hybrid_optimizer_parameters(
    model: ReturnOracleMlp,
) -> tuple[tuple[Tensor, ...], tuple[Tensor, ...]]:
    """Route projection/A matrices to Muon and constrained C to AdamW."""
    muon_parameters = (
        tuple(layer.weight for layer in model.layers)
        + tuple(layer.weight for layer in model.residual_glu_layers)
        + tuple(
            layer.weight
            for target_layers in model.dense_residual_layers
            for layer in target_layers
        )
        + tuple(transform.weight for transform in model.value_transforms)
        + tuple(transform.weight for transform in model.gate_transforms)
    )
    muon_parameter_ids = {id(parameter) for parameter in muon_parameters}
    adamw_parameters = tuple(
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
        and id(parameter) not in muon_parameter_ids
    )
    all_parameters = tuple(
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    if not muon_parameters or not adamw_parameters \
            or len(muon_parameter_ids) != len(muon_parameters) \
            or len(muon_parameters) + len(adamw_parameters) \
            != len(all_parameters) \
            or {
                id(parameter)
                for parameter in (*muon_parameters, *adamw_parameters)
            } != {id(parameter) for parameter in all_parameters}:
        raise RuntimeError("hybrid optimizer parameter routing is invalid")
    if any(parameter.ndim != 2 for parameter in muon_parameters):
        raise RuntimeError("Muon parameters must all be two-dimensional")
    return muon_parameters, adamw_parameters


@dataclass(frozen=True)
class SourceShard:
    split: str
    date: str
    count: int
    feature_file: Path
    feature_row_offset: int
    feature_row_stride: int
    target_file: Path
    target_row_offset: int
    target_row_stride: int
    prediction_time_start: int
    oracle_target_time_start: int

    @property
    def prediction_time_end(self) -> int:
        return self.prediction_time_start + (self.count - 1) * SECOND_MS


@dataclass(frozen=True)
class Segment:
    shard: SourceShard
    local_offset: int
    count: int

    @property
    def split(self) -> str:
        return self.shard.split

    @property
    def prediction_time_start(self) -> int:
        return self.shard.prediction_time_start + self.local_offset * SECOND_MS

    @property
    def prediction_time_end(self) -> int:
        return self.prediction_time_start + (self.count - 1) * SECOND_MS


@dataclass
class LoadedComponentGroup:
    segments: list[Segment]
    features_by_date: dict[str, Tensor]
    targets: Tensor


@dataclass
class MetricAccumulator:
    examples: int = 0
    sums: dict[str, Tensor] | None = None

    def add(self, metrics: dict[str, Tensor], count: int) -> None:
        if self.sums is None:
            self.sums = {
                name: torch.zeros(
                    (),
                    device=metrics[name].device,
                    dtype=torch.float64,
                )
                for name in METRIC_NAMES
            }
        self.examples += count
        for name in METRIC_NAMES:
            self.sums[name].add_(
                metrics[name].detach().to(dtype=torch.float64),
                alpha=count,
            )

    def result(self) -> dict[str, float]:
        if self.examples < 1 or self.sums is None:
            raise RuntimeError("cannot finalize empty metrics")
        return {
            name: float(value / self.examples)
            for name, value in self.sums.items()
        }


def add_batch_invariant_regularizers(
    metrics: dict[str, float],
    regularizers: dict[str, float],
) -> dict[str, float]:
    """Restore regularizers computed once for an evaluation pass."""
    result = dict(metrics)
    for name in (
        "softWeightBound",
        "centeringIdempotence",
        "centeringSymmetry",
        "centeringConstraint",
    ):
        result[name] = float(regularizers[name])
    result["loss"] += float(regularizers["lossAddition"])
    return result


DeviceBatch = tuple[Tensor, Tensor, Tensor, int]


def group_batches(
    source: Iterator[DeviceBatch],
    group_size: int,
) -> Iterator[tuple[DeviceBatch, ...]]:
    """Group microbatches for one multiplicity-weighted optimizer update."""
    if group_size < 1:
        raise ValueError("gradient accumulation group size must be positive")
    pending: list[DeviceBatch] = []
    for batch in source:
        pending.append(batch)
        if len(pending) == group_size:
            yield tuple(pending)
            pending.clear()
    if pending:
        yield tuple(pending)


class DeviceBatchPipeline:
    """Overlap pinned-host copies with compute using the old MLP runner path."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.copy_stream = (
            torch.cuda.Stream(device=device)
            if device.type == "cuda"
            else None
        )

    def batches(
        self,
        source: Iterator[DeviceBatch],
    ) -> Iterator[DeviceBatch]:
        if self.copy_stream is None:
            yield from source
            return
        iterator = iter(source)

        def copy_batch(batch: DeviceBatch) -> DeviceBatch:
            features, targets, example_weights, original_count = batch
            with torch.cuda.stream(self.copy_stream):
                return (
                    features.to(self.device, non_blocking=True),
                    targets.to(self.device, non_blocking=True),
                    example_weights.to(self.device, non_blocking=True),
                    original_count,
                )

        try:
            pending = copy_batch(next(iterator))
        except StopIteration:
            return
        while True:
            current_stream = torch.cuda.current_stream(self.device)
            current_stream.wait_stream(self.copy_stream)
            batch = pending
            for value in batch[:3]:
                value.record_stream(current_stream)
            try:
                pending = copy_batch(next(iterator))
            except StopIteration:
                pending = None
            yield batch
            if pending is None:
                break


@dataclass
class BatchIteratorFailure:
    error: BaseException


class PreparedBatchIterator:
    """Start the next epoch's CPU pipeline while validation uses the GPU."""

    def __init__(
        self,
        source: Iterator[DeviceBatch],
        prefetch_batches: int,
    ) -> None:
        self.source = source
        self.queue: queue.Queue[DeviceBatch | BatchIteratorFailure | None] = \
            queue.Queue(maxsize=max(1, prefetch_batches))
        self.stop_requested = False
        self.thread = threading.Thread(
            target=self._produce,
            name="return-oracle-next-epoch-prefetch",
            daemon=True,
        )
        self.thread.start()

    def _put(
        self,
        value: DeviceBatch | BatchIteratorFailure | None,
    ) -> bool:
        while not self.stop_requested:
            try:
                self.queue.put(value, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def _produce(self) -> None:
        try:
            for batch in self.source:
                if not self._put(batch):
                    return
        except BaseException as error:
            self._put(BatchIteratorFailure(error))
        finally:
            self._put(None)

    def __iter__(self) -> PreparedBatchIterator:
        return self

    def __next__(self) -> DeviceBatch:
        value = self.queue.get()
        if value is None:
            self.thread.join()
            raise StopIteration
        if isinstance(value, BatchIteratorFailure):
            self.thread.join()
            raise value.error
        return value

    def close(self) -> None:
        self.stop_requested = True
        while self.thread.is_alive():
            try:
                self.queue.get_nowait()
            except queue.Empty:
                pass
            self.thread.join(timeout=0.1)


def frozen_cpu_copy(value):
    if isinstance(value, Tensor):
        return value.detach().to(device="cpu", copy=True)
    if isinstance(value, dict):
        return {
            key: frozen_cpu_copy(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [frozen_cpu_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(frozen_cpu_copy(item) for item in value)
    return copy.deepcopy(value)


class AsyncCheckpointWriter:
    """Serialize a frozen checkpoint while the next epoch uses the GPU."""

    def __init__(self) -> None:
        self.executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="return-oracle-checkpoint-writer",
        )
        self.pending: Future | None = None

    @staticmethod
    def _write(
        checkpoint: dict,
        checkpoint_file: Path,
        best_checkpoint_file: Path | None,
    ) -> dict[str, float]:
        started = time.monotonic()
        if best_checkpoint_file is not None:
            atomic_torch_save({
                "model": checkpoint["model"],
                "epoch": checkpoint["epoch"],
                "globalStep": checkpoint["globalStep"],
                "validation": checkpoint["validation"],
                "parameterCount": checkpoint["parameterCount"],
                "featureContract": checkpoint["featureContract"],
                "objectiveContract": checkpoint["objectiveContract"],
            }, best_checkpoint_file)
        best_seconds = time.monotonic() - started
        atomic_torch_save(checkpoint, checkpoint_file)
        return {
            "bestCheckpointWriteSeconds": (
                best_seconds if best_checkpoint_file is not None else 0.0
            ),
            "checkpointWriteSeconds": (
                time.monotonic() - started - best_seconds
            ),
        }

    def submit(
        self,
        checkpoint: dict,
        checkpoint_file: Path,
        best_checkpoint_file: Path | None,
    ) -> dict[str, float | bool | None]:
        wait_started = time.monotonic()
        previous = self.flush()
        wait_seconds = time.monotonic() - wait_started
        snapshot_started = time.monotonic()
        frozen = frozen_cpu_copy(checkpoint)
        snapshot_seconds = time.monotonic() - snapshot_started
        self.pending = self.executor.submit(
            self._write,
            frozen,
            checkpoint_file,
            best_checkpoint_file,
        )
        return {
            "asynchronous": True,
            "snapshotSeconds": snapshot_seconds,
            "previousWriteWaitSeconds": wait_seconds,
            "previousCheckpointWriteSeconds": (
                previous["checkpointWriteSeconds"]
                if previous is not None
                else None
            ),
        }

    def flush(self) -> dict[str, float] | None:
        if self.pending is None:
            return None
        pending = self.pending
        self.pending = None
        return pending.result()

    def close(self) -> None:
        try:
            self.flush()
        finally:
            self.executor.shutdown(wait=True)


class RunReporter:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = run_dir / "training.log"
        self.status_file = run_dir / "status.json"

    def emit(self, event: dict) -> None:
        line = json.dumps(event, separators=(",", ":"), allow_nan=False)
        with self.log_file.open("a", encoding="utf-8", newline="\n") as output:
            output.write(line + "\n")
        print(line, flush=True)

    def status(self, stage: str, **values) -> None:
        try:
            atomic_json({
                "pid": os.getpid(),
                "stage": stage,
                "updatedAt": iso_now(),
                **values,
            }, self.status_file)
        except PermissionError as error:
            # The MLP page polls this file. Windows can briefly deny replacing
            # an open destination even though the read is non-mutating. Status
            # reporting must never terminate an otherwise healthy training run.
            self.emit({
                "event": "status-write-warning",
                "stage": stage,
                "error": f"{type(error).__name__}: {error}",
            })


class ComponentCache:
    def __init__(self, max_entries: int = 4) -> None:
        self.max_entries = max_entries
        self.values: OrderedDict[
            tuple[Path, str, tuple[int, int]], np.ndarray
        ] = OrderedDict()
        self.decompressor = zstandard.ZstdDecompressor()

    def load(
        self,
        file: Path,
        dtype: str,
        shape: tuple[int, int],
    ) -> np.ndarray:
        key = (file, dtype, shape)
        cached = self.values.pop(key, None)
        if cached is not None:
            self.values[key] = cached
            return cached
        expected_bytes = math.prod(shape) * np.dtype(dtype).itemsize
        if file.name.endswith(".zst"):
            decoded = self.decompressor.decompress(
                file.read_bytes(),
                max_output_size=expected_bytes,
            )
            if len(decoded) != expected_bytes:
                raise ValueError(
                    f"decoded component has invalid size: {file} "
                    f"({len(decoded)} != {expected_bytes})"
                )
            value = np.frombuffer(decoded, dtype=dtype).reshape(shape)
        else:
            if file.stat().st_size != expected_bytes:
                raise ValueError(f"component has invalid size: {file}")
            value = np.memmap(file, mode="r", dtype=dtype, shape=shape)
        self.values[key] = value
        while len(self.values) > self.max_entries:
            self.values.popitem(last=False)
        return value


class ExperimentDataset:
    def __init__(
        self,
        feature_root: Path,
        segments: dict[str, list[Segment]],
        workers: int,
        prefetch_factor: int,
    ) -> None:
        self.feature_root = feature_root
        self.segments = segments
        self.pin_memory = torch.cuda.is_available()
        self.workers = max(1, workers)
        self.prefetch_factor = max(1, prefetch_factor)

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
    ) -> Iterator[DeviceBatch]:
        generator = random.Random(seed)
        groups = group_segments_by_target(self.segments[split])
        if shuffle:
            generator.shuffle(groups)
        if not groups:
            return

        worker_count = min(self.workers, len(groups))
        prefetch_groups = min(
            len(groups),
            worker_count * self.prefetch_factor,
        )
        with ThreadPoolExecutor(
            max_workers=worker_count,
            thread_name_prefix="return-oracle-component-prefetch",
        ) as executor:
            pending: dict[int, Future] = {
                group_index: executor.submit(
                    self._load_component_group,
                    groups[group_index],
                )
                for group_index in range(prefetch_groups)
            }
            next_group_to_submit = prefetch_groups
            for group_index, group in enumerate(groups):
                loaded = pending.pop(group_index).result()
                if next_group_to_submit < len(groups):
                    pending[next_group_to_submit] = executor.submit(
                        self._load_component_group,
                        groups[next_group_to_submit],
                    )
                    next_group_to_submit += 1
                segments = list(loaded.segments)
                if shuffle:
                    generator.shuffle(segments)
                for segment in segments:
                    starts = list(range(0, segment.count, batch_size))
                    if shuffle:
                        generator.shuffle(starts)
                    for batch_start in starts:
                        batch_count = min(
                            batch_size,
                            segment.count - batch_start,
                        )
                        local_start = segment.local_offset + batch_start
                        feature_start = (
                            segment.shard.feature_row_offset + local_start
                        )
                        target_start = (
                            segment.shard.target_row_offset + local_start
                        )
                        feature_end = feature_start + batch_count
                        target_end = target_start + batch_count
                        if feature_start < 0 or feature_end > DAY_SECONDS \
                                or target_start < 0 or target_end > DAY_SECONDS:
                            raise IndexError(
                                "source component row is outside its UTC day"
                            )
                        feature_rows, target_rows, multiplicities = (
                            compact_completed_minute_batch_rows(
                                feature_start,
                                target_start,
                                batch_count,
                            )
                        )
                        if bool((np.diff(feature_rows) != 1).any()) \
                                or bool((np.diff(target_rows) != 1).any()):
                            raise RuntimeError(
                                "completed-minute batch rows are not consecutive"
                            )
                        yield (
                            loaded.features_by_date[segment.shard.date][
                                feature_rows[0]:feature_rows[-1] + 1
                            ],
                            loaded.targets[
                                target_rows[0]:target_rows[-1] + 1
                            ],
                            copy_component_to_tensor(
                                multiplicities,
                                torch.float32,
                                self.pin_memory,
                            ),
                            batch_count,
                        )

    def _load_component_group(
        self,
        segments: list[Segment],
    ) -> LoadedComponentGroup:
        if not segments:
            raise ValueError("cannot load an empty component group")
        target_file = segments[0].shard.target_file
        if any(
            segment.shard.target_file != target_file
            for segment in segments
        ):
            raise ValueError("component group contains multiple target files")
        cache = ComponentCache(max_entries=len(segments) + 1)
        minute_targets = cache.load(
            target_file,
            "<f4",
            (MINUTE_ROWS_PER_DAY, OUTPUT_ACTION_COUNT),
        )
        targets = copy_component_to_tensor(
            minute_targets,
            torch.float32,
            self.pin_memory,
        )
        features_by_date = {
            component_date: copy_component_to_tensor(
                cache.load(
                    self.feature_root
                    / f"{component_date}{RETURN_COMPONENT_SUFFIX}",
                    "<f2",
                    (MINUTE_ROWS_PER_DAY, INPUT_RETURN_COUNT),
                ),
                torch.float16,
                self.pin_memory,
            )
            for component_date in sorted({
                segment.shard.date
                for segment in segments
            })
        }
        return LoadedComponentGroup(
            segments=segments,
            features_by_date=features_by_date,
            targets=targets,
        )


def group_segments_by_target(
    segments: list[Segment],
) -> list[list[Segment]]:
    groups: OrderedDict[Path, list[Segment]] = OrderedDict()
    for segment in segments:
        groups.setdefault(segment.shard.target_file, []).append(segment)
    return list(groups.values())


def copy_component_to_tensor(
    value: np.ndarray,
    dtype: torch.dtype,
    pin_memory: bool,
) -> Tensor:
    tensor = torch.empty(
        value.shape,
        dtype=dtype,
        pin_memory=pin_memory,
    )
    np.copyto(tensor.numpy(), value, casting="no")
    return tensor


def completed_minute_row_indices(rows: np.ndarray) -> np.ndarray:
    """Map one-second rows to the latest completed UTC-minute row."""
    rows = np.asarray(rows, dtype=np.int64)
    return rows // MINUTE_SECONDS + (rows % MINUTE_SECONDS == 59)


def compact_completed_minute_batch_rows(
    feature_start: int,
    target_start: int,
    count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collapse repeated second rows while preserving their exact weight."""
    if count < 1:
        raise ValueError("cannot compact an empty batch")
    offsets = np.arange(count, dtype=np.int64)
    feature_minutes = completed_minute_row_indices(feature_start + offsets)
    target_minutes = completed_minute_row_indices(target_start + offsets)
    feature_changes = np.flatnonzero(np.concatenate((
        np.asarray([True]),
        feature_minutes[1:] != feature_minutes[:-1],
    )))
    target_changes = np.flatnonzero(np.concatenate((
        np.asarray([True]),
        target_minutes[1:] != target_minutes[:-1],
    )))
    if not np.array_equal(feature_changes, target_changes):
        raise ValueError(
            "feature and oracle rows do not share completed-minute boundaries"
        )
    multiplicities = np.diff(np.append(feature_changes, count)).astype(
        "<f4",
        copy=False,
    )
    return (
        feature_minutes[feature_changes],
        target_minutes[target_changes],
        multiplicities,
    )


def completed_minute_simple_return_rows(
    previous_close: np.ndarray,
    current_close: np.ndarray,
) -> np.ndarray:
    """Build the 1,441 close-only minute paths used by the stored oracle."""
    if previous_close.shape != (DAY_SECONDS,) \
            or current_close.shape != (DAY_SECONDS,):
        raise ValueError("daily close arrays must contain 86,400 one-second rows")
    close = np.concatenate((
        previous_close[-(INPUT_HORIZON_SECONDS + 1):],
        current_close,
    ))
    offsets = np.arange(
        0,
        INPUT_HORIZON_SECONDS + 1,
        MINUTE_SECONDS,
        dtype=np.int64,
    )
    minute_rows = (
        np.arange(MINUTE_ROWS_PER_DAY, dtype=np.int64)[:, None]
        * MINUTE_SECONDS
    )
    boundaries = close[minute_rows + offsets[None, :]]
    features = boundaries[:, 1:] / boundaries[:, :-1] - 1.0
    if not np.isfinite(features).all():
        raise ValueError("completed-minute return features contain non-finite values")
    return features


def cached_training_normalization(
    dataset: ExperimentDataset,
    cache_file: Path,
    reporter: RunReporter,
) -> tuple[Tensor, Tensor]:
    """Compute expanded-corpus per-position moments from compact minute rows."""
    expected_count = dataset.count("train")
    if cache_file.is_file():
        with np.load(cache_file, allow_pickle=False) as cached:
            mean = cached["mean"]
            std = cached["std"]
            count = int(cached["count"])
            contract = str(cached["featureContract"])
        if count != expected_count \
                or contract != FEATURE_CONTRACT \
                or mean.shape != (INPUT_RETURN_COUNT,) \
                or std.shape != (INPUT_RETURN_COUNT,) \
                or not np.isfinite(mean).all() \
                or not np.isfinite(std).all() \
                or bool((std <= 0).any()):
            raise ValueError(
                f"invalid cached feature normalization: {cache_file}"
            )
        reporter.emit({
            "event": "training-statistics-cache",
            "kind": "features",
            "hit": True,
            "file": str(cache_file),
            "examples": expected_count,
        })
        return (
            torch.from_numpy(mean.astype(np.float32)),
            torch.from_numpy(std.astype(np.float32)),
        )

    feature_sum = np.zeros(INPUT_RETURN_COUNT, dtype=np.float64)
    square_sum = np.zeros(INPUT_RETURN_COUNT, dtype=np.float64)
    total = 0
    segments_by_date: OrderedDict[str, list[Segment]] = OrderedDict()
    for segment in dataset.segments["train"]:
        segments_by_date.setdefault(segment.shard.date, []).append(segment)
    cache = ComponentCache(max_entries=2)
    date_count = len(segments_by_date)
    for date_index, (component_date, segments) in enumerate(
        segments_by_date.items(),
        start=1,
    ):
        source = cache.load(
            dataset.feature_root
            / f"{component_date}{RETURN_COMPONENT_SUFFIX}",
            "<f2",
            (MINUTE_ROWS_PER_DAY, INPUT_RETURN_COUNT),
        )
        for segment in segments:
            segment_feature_start = (
                segment.shard.feature_row_offset + segment.local_offset
            )
            segment_target_start = (
                segment.shard.target_row_offset + segment.local_offset
            )
            for local_start in range(0, segment.count, DAY_SECONDS):
                block_count = min(
                    DAY_SECONDS,
                    segment.count - local_start,
                )
                feature_rows, _target_rows, multiplicities = (
                    compact_completed_minute_batch_rows(
                        segment_feature_start + local_start,
                        segment_target_start + local_start,
                        block_count,
                    )
                )
                if feature_rows[0] < 0 \
                        or feature_rows[-1] >= MINUTE_ROWS_PER_DAY \
                        or bool((np.diff(feature_rows) != 1).any()):
                    raise IndexError(
                        "training normalization row is outside its UTC day"
                    )
                block = np.asarray(
                    source[feature_rows[0]:feature_rows[-1] + 1],
                    dtype=np.float64,
                )
                weights = multiplicities.astype(np.float64, copy=False)
                if block.shape[0] != weights.shape[0]:
                    raise RuntimeError(
                        "training normalization compaction is misaligned"
                    )
                feature_sum += np.einsum(
                    "i,ij->j",
                    weights,
                    block,
                )
                square_sum += np.einsum(
                    "i,ij->j",
                    weights,
                    np.square(block),
                )
                total += int(weights.sum())
        if date_index == date_count or date_index % 10 == 0:
            reporter.emit({
                "event": "dataset-progress",
                "split": "training-input-normalization",
                "day": date_index,
                "days": date_count,
                "examples": total,
            })
    if total != expected_count:
        raise RuntimeError(
            "training normalization did not cover the selected training corpus"
        )
    mean = feature_sum / total
    variance = np.maximum(
        1e-12,
        square_sum / total - mean * mean,
    )
    std = np.sqrt(variance)
    temporary = cache_file.with_suffix(cache_file.suffix + ".tmp")
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("wb") as output:
        np.savez(
            output,
            mean=mean.astype(np.float32),
            std=std.astype(np.float32),
            count=np.asarray(total, dtype=np.int64),
            featureContract=np.asarray(FEATURE_CONTRACT),
        )
    replace_file_with_retry(temporary, cache_file)
    reporter.emit({
        "event": "training-statistics-cache",
        "kind": "features",
        "hit": False,
        "file": str(cache_file),
        "examples": total,
        "minimumStd": float(std.min()),
        "maximumStd": float(std.max()),
    })
    return (
        torch.from_numpy(mean.astype(np.float32)),
        torch.from_numpy(std.astype(np.float32)),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the standalone 60-return expanding-width MLP directly "
            "against "
            "stored close-only minute oracle distributions with soft-target "
            "cross-entropy."
        )
    )
    parser.add_argument(
        "--plan",
        type=Path,
        default=Path(
            "ml/training-plans/"
            "return-oracle-ce-shrinking-v1.json"
        ),
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Build return components and validate the data contract without training.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    plan_file = resolve(repo_root, args.plan)
    plan = json.loads(plan_file.read_text(encoding="utf-8"))
    source_root = resolve(repo_root, Path(plan["sourceDatasetDir"]))
    dataset_root = resolve(repo_root, Path(plan["datasetDir"]))
    run_dir = resolve(repo_root, Path(plan["runDir"]))
    history_root = resolve(repo_root, Path(plan["historyDir"]))
    feature_root = dataset_root / "components" / "returns"
    reporter = RunReporter(run_dir)
    training = plan["training"]
    validate_plan(plan)
    started_at = iso_now()
    reporter.status(
        "dataset-preparation",
        startedAt=started_at,
        planId=plan["id"],
        message="Preparing 60 simple-return inputs from the full inspector corpus.",
    )
    try:
        source_manifest = json.loads(
            (source_root / "dataset.json").read_text(encoding="utf-8")
        )
        validate_source_manifest(source_manifest)
        selected_segments = select_segments(source_manifest, source_root)
        counts = {
            split: sum(segment.count for segment in values)
            for split, values in selected_segments.items()
        }
        feature_dates = sorted({
            segment.shard.date
            for values in selected_segments.values()
            for segment in values
        })
        feature_root.mkdir(parents=True, exist_ok=True)
        prepare_return_components(
            history_root,
            feature_root,
            feature_dates,
            reporter,
        )
        progress = {
            "featureComponents": [
                {"date": component_date}
                for component_date in feature_dates
            ],
            "oracleComponents": sorted({
                str(segment.shard.target_file.relative_to(source_root))
                for values in selected_segments.values()
                for segment in values
            }),
        }
        atomic_json(progress, dataset_root / "progress.json")
        dataset = ExperimentDataset(
            feature_root,
            selected_segments,
            workers=int(training["workers"]),
            prefetch_factor=int(training["prefetchFactor"]),
        )
        reporter.status(
            "dataset-preparation",
            startedAt=started_at,
            planId=plan["id"],
            message=(
                "Computing training-only per-position simple-return "
                "normalization."
            ),
        )
        feature_mean, feature_std = cached_training_normalization(
            dataset,
            dataset_root
            / "training-feature-statistics-position-v3.npz",
            reporter,
        )
        model_definition = ReturnOracleMlp(
            feature_mean,
            feature_std,
            dropout=float(training["dropout"]),
            dropout_rate=float(training["dropoutRate"]),
            normalization_family=str(
                training["branchNormalization"]["family"]
            ),
            normalization_initial_scale=float(
                training["branchNormalization"]["initialScale"]
            ),
            normalization_minimum_scale=float(
                training["branchNormalization"]["minimumScale"]
            ),
            learnable_centering=bool(
                training["branchNormalization"]["learnableCentering"]
            ),
        )
        model_parameters = parameter_count(model_definition)
        if counts["train"] < model_parameters:
            raise ValueError(
                "the expanded training corpus must contain at least as many "
                f"examples as model parameters ({counts['train']:,} < "
                f"{model_parameters:,})"
            )
        dataset_manifest = {
            "version": 2,
            "createdAt": iso_now(),
            "planId": plan["id"],
            "sourceDataset": str(
                (source_root / "dataset.json").relative_to(repo_root)
            ),
            "featureSemantics": (
                "60 adjacent simple returns from completed UTC-minute close "
                "values over the exact close-only path used by the stored "
                "minute oracle; compact storage keeps one row per completed "
                "minute and training multiplicity weights exactly preserve "
                "the source example measure"
            ),
            "targetSemantics": (
                "stored minuteOracleProbabilities computed from the same 60 "
                "completed one-minute close candles on the source 255-action grid"
            ),
            "inputReturnCount": INPUT_RETURN_COUNT,
            "inputTransform": FEATURE_CONTRACT,
            "featureStandardization": {
                "source": "selected training split only",
                "axis": "each of the 60 return positions independently",
                "varianceCorrection": 0,
                "mean": feature_mean.tolist(),
                "std": feature_std.tolist(),
            },
            "trainingObjective": OBJECTIVE_CONTRACT,
            "reverseKl": {
                "direction": (
                    "KL(predicted || (1-epsilon)*oracle + "
                    "epsilon*predicted)"
                ),
                "lossWeight": float(
                    training["lossWeights"]["reverseKl"]
                ),
                "predictionMixtureWeight": float(
                    training["reverseKl"]["predictionMixtureWeight"]
                ),
                "referenceTransform": (
                    "(1-epsilon)*oracle + epsilon*predicted"
                ),
                "applicationProbability": float(
                    training["outputRegularizer"][
                        "applicationProbabilities"
                    ][
                        "reverseKl"
                    ]
                ),
            },
            "entropySharpness": {
                "lossWeight": float(
                    training["lossWeights"]["entropySharpness"]
                ),
                "formula": "relu(H(predicted)-H(oracle))^2",
                "exampleAggregation": (
                    "source-multiplicity-weighted mean"
                ),
                "applicationRate": float(
                    training["outputRegularizer"][
                        "applicationProbabilities"
                    ][
                        "entropySharpness"
                    ]
                ),
                "margin": 0.0,
            },
            "outputRegularizer": training["outputRegularizer"],
            "softLayerNorm": {
                "layerAggregation": (
                    f"mean across {HIDDEN_LAYER_COUNT} layers"
                ),
                "exampleAggregation": "source-multiplicity-weighted mean",
                "measurement": "raw value and gate-logit affine branches",
                "diagnosticOnly": (
                    float(training["lossWeights"]["softLayerNorm"]) == 0
                ),
                "varianceCorrection": 0,
                "varianceWeight": float(
                    training["softLayerNorm"]["varianceWeight"]
                ),
                "lossWeight": float(
                    training["lossWeights"]["softLayerNorm"]
                ),
            },
            "actionCount": OUTPUT_ACTION_COUNT,
            "hiddenWidths": list(HIDDEN_WIDTHS),
            "hiddenLayerCount": HIDDEN_LAYER_COUNT,
            "branchNormalization": {
                "placement": (
                    "separately on raw value and gate-logit branches after "
                    "the fused projection, before full A transforms and GLU"
                ),
                "family": training["branchNormalization"]["family"],
                "denominator": training["branchNormalization"][
                    "denominator"
                ],
                "scale": {
                    "learnable": True,
                    "sharing": "one-scalar-per-value-or-gate-branch",
                    "count": (
                        4 * HIDDEN_LAYER_COUNT
                        + 2 * DENSE_RESIDUAL_PATH_COUNT
                    ),
                    "role": training["branchNormalization"]["scaleRole"],
                    "initialValue": float(
                        training["branchNormalization"]["initialScale"]
                    ),
                    "initialSquaredValue": float(
                        training["branchNormalization"][
                            "initialSquaredScale"
                        ]
                    ),
                    "minimumValue": float(
                        training["branchNormalization"]["minimumScale"]
                    ),
                    "parameterization": training["branchNormalization"][
                        "scaleParameterization"
                    ],
                },
                "learnablePostTransformBias": True,
                "centering": (
                    "one fixed full C per target layer, shared across every "
                    "incoming value/gate path at I - 11^T / d"
                ),
                "normalization": (
                    "C h / (s g(r/s)), r=sqrt(mean((C h)^2)), "
                    f"g(u)={training['branchNormalization']['denominator']}"
                ),
                "branchWidths": list(HIDDEN_WIDTHS),
                "absorbedInputTransform": (
                    "B is absorbed into both halves of the fused projection"
                ),
                "postNormalizationTransform": (
                    "one value A and one gate A, each shared across the main "
                    "and residual paths, initialized to identity, followed by "
                    "branch-specific learnable bias vectors"
                ),
                "globalInputResidual": {
                    "formula": (
                        "hidden=GLU(main_projection)+GLU(input_projection)"
                    ),
                    "source": "shared-standardized-60-return-model-input",
                    "matrixCount": HIDDEN_LAYER_COUNT,
                    "initialization": (
                        "zero-value-half; kaiming-gate-half; zero-bias"
                    ),
                    "normalization": (
                        "independent-branches-sharing-layer-C-and-A"
                    ),
                    "optimizer": "Muon-matrix-AdamW-bias",
                },
                "denseLayerResiduals": {
                    "formula": (
                        "hidden_l=main_l(previous)+input_l(x0)+"
                        "sum_{j<l-1} residual_{l,j}(hidden_j)"
                    ),
                    "pathCount": DENSE_RESIDUAL_PATH_COUNT,
                    "aggregation": "additive",
                    "normalization": (
                        "path-specific-radius-and-bias; target-layer-shared-"
                        "C-value-A-gate-A"
                    ),
                },
                "centeringConstraints": {
                    "idempotence": "mean((C^2 - C)^2)",
                    "symmetry": "mean((C^T - C)^2)",
                    "layerAggregation": "mean",
                    "idempotenceWeight": float(
                        training["lossWeights"]["centeringIdempotence"]
                    ),
                    "symmetryWeight": float(
                        training["lossWeights"]["centeringSymmetry"]
                    ),
                },
            },
            "parameterCount": model_parameters,
            "dropout": {
                "activationProbability": float(training["dropout"]),
                "applicationRate": float(training["dropoutRate"]),
                "passGateProbability": (
                    float(training["dropoutRate"]) ** 0.5
                ),
                "layerGateProbability": (
                    float(training["dropoutRate"]) ** 0.5
                ),
            },
            "counts": counts,
            "testSelection": "last 1,000,000 chronological source test examples",
            "crossSplitPurgeMs": INPUT_HORIZON_SECONDS * SECOND_MS,
            "splitPolicy": (
                "source inspector split assignment, with later cross-split "
                "ranges purged until their 60-minute input path cannot overlap "
                "the preceding split"
            ),
            "featureDtype": "float16",
            "featureStorageRowsPerUtcDay": MINUTE_ROWS_PER_DAY,
            "batchCompaction": RUNNER_CONTRACT,
            "targetDtype": "float32",
            "actionGrid": source_manifest["grid"],
        }
        atomic_json(dataset_manifest, dataset_root / "dataset.json")
        reporter.emit({
            "event": "dataset-complete",
            "examples": sum(counts.values()),
            "trainExamples": counts["train"],
            "validationExamples": counts["validation"],
            "testExamples": counts["test"],
            "parameters": model_parameters,
            "featureComponents": len(feature_dates),
        })
        if args.prepare_only:
            reporter.status(
                "paused",
                startedAt=started_at,
                pausedAt=iso_now(),
                planId=plan["id"],
                message="Dataset preparation completed; training was not requested.",
            )
            return
        train(
            plan,
            plan_file,
            dataset,
            model_parameters,
            reporter,
            started_at,
            feature_mean,
            feature_std,
        )
    except KeyboardInterrupt:
        reporter.status(
            "paused",
            startedAt=started_at,
            pausedAt=iso_now(),
            planId=plan["id"],
            message="Training was interrupted and can resume from last.pt.",
        )
        raise
    except Exception as error:
        reporter.status(
            "failed",
            startedAt=started_at,
            failedAt=iso_now(),
            planId=plan["id"],
            error=f"{type(error).__name__}: {error}",
        )
        raise


def validate_plan(plan: dict) -> None:
    required = (
        "id",
        "label",
        "sourceDatasetDir",
        "datasetDir",
        "runDir",
        "historyDir",
        "training",
    )
    if any(not plan.get(name) for name in required):
        raise ValueError("experiment plan is missing required fields")
    training = plan["training"]
    if training.get("targetRepresentation") \
            != "minuteOracleProbabilities" \
            or training.get("inputStandardization") \
            != (
                "training-split per-position population mean/std "
                "across 60 return coordinates"
            ) \
            or training.get("architecture") != ARCHITECTURE_CONTRACT \
            or training.get("objective") != OBJECTIVE_CONTRACT:
        raise ValueError(
            "training plan does not match the aligned standardized CE contract"
        )
    numeric_positive = (
        "batchSize",
        "evaluationBatchSize",
        "gradientAccumulationSteps",
        "learningRate",
        "patience",
        "logEverySteps",
        "workers",
        "prefetchFactor",
    )
    if any(float(training.get(name, 0)) <= 0 for name in numeric_positive):
        raise ValueError("training plan contains invalid positive settings")
    schedule = training.get("learningRateSchedule", {})
    if schedule.get("type") != "reduce-on-validation-plateau" \
            or not 0 < float(schedule.get("factor", 0)) < 1 \
            or int(schedule.get("patience", 0)) < 1 \
            or float(schedule.get("threshold", -1)) < 0 \
            or not 0 < float(schedule.get("minimumLearningRate", 0)) \
            <= float(training["learningRate"]):
        raise ValueError("training plan contains an invalid LR schedule")
    optimizer = training.get("optimizer", {})
    muon = optimizer.get("muon", {})
    adamw = optimizer.get("adamw", {})
    adamw_betas = adamw.get("betas", ())
    if optimizer.get("type") != "hybrid-muon-adamw" \
            or not 0 <= float(muon.get("momentum", -1)) < 1 \
            or not isinstance(muon.get("nesterov"), bool) \
            or int(muon.get("newtonSchulzSteps", 0)) < 1 \
            or muon.get("adjustLearningRate") != "match_rms_adamw" \
            or float(muon.get("epsilon", 0)) <= 0 \
            or float(muon.get("weightDecay", -1)) < 0 \
            or not isinstance(adamw_betas, list) \
            or len(adamw_betas) != 2 \
            or not 0 <= float(adamw_betas[0]) < float(adamw_betas[1]) < 1 \
            or float(adamw.get("epsilon", 0)) <= 0 \
            or float(adamw.get("weightDecay", -1)) < 0:
        raise ValueError("training plan contains an invalid hybrid optimizer")
    loss_weights = training.get("lossWeights", {})
    reverse_kl = training.get("reverseKl", {})
    output_regularizer = training.get("outputRegularizer", {})
    application_probabilities = output_regularizer.get(
        "applicationProbabilities",
        {},
    )
    soft_layer_norm = training.get("softLayerNorm", {})
    soft_weight_bound = training.get("softWeightBound", {})
    branch_normalization = training.get("branchNormalization", {})
    if float(loss_weights.get("crossEntropy", 0)) <= 0 \
            or float(loss_weights.get("reverseKl", -1)) < 0 \
            or float(loss_weights.get("entropySharpness", -1)) < 0 \
            or float(loss_weights.get("softLayerNorm", -1)) < 0 \
            or float(loss_weights.get("softWeightBound", -1)) < 0 \
            or float(loss_weights.get("distributionLayerSum", -1)) < 0 \
            or float(
                loss_weights.get("distributionLayerNegative", -1)
            ) < 0 \
            or float(loss_weights.get("centeringIdempotence", -1)) < 0 \
            or float(loss_weights.get("centeringSymmetry", -1)) < 0 \
            or float(soft_layer_norm.get("varianceWeight", -1)) < 0 \
            or float(soft_weight_bound.get("desiredMagnitude", 0)) <= 0 \
            or float(soft_weight_bound.get("sharpness", 0)) <= 0 \
            or float(soft_weight_bound.get("absoluteEpsilon", 0)) <= 0 \
            or not 0 < float(
                reverse_kl.get("predictionMixtureWeight", 0)
            ) < 1 \
            or not 0 < float(
                application_probabilities.get("reverseKl", 0)
            ) <= 1 \
            or not 0 < float(
                application_probabilities.get("entropySharpness", 0)
            ) <= 1 \
            or output_regularizer.get("samplingUnit") \
            != "optimizer-update" \
            or output_regularizer.get("independentGates") is not True \
            or output_regularizer.get("inverseProbabilityScaling") \
            is not True \
            or branch_normalization.get("family") != "tanh" \
            or branch_normalization.get("denominator") \
            != "u/tanh(u)" \
            or branch_normalization.get("scaleRole") \
            != "denominatorRadius" \
            or branch_normalization.get("scaleParameterization") \
            != "softplus(rawScale)+minimumScale" \
            or branch_normalization.get("learnableCentering") is not False \
            or float(branch_normalization.get("minimumScale", 0)) <= 0 \
            or float(branch_normalization.get("initialScale", 0)) \
            <= float(branch_normalization.get("minimumScale", 0)) \
            or not math.isclose(
                float(branch_normalization.get("initialScale", 0)) ** 2,
                BRANCH_NORMALIZATION_EPSILON,
                rel_tol=1e-12,
                abs_tol=1e-15,
            ) \
            or not math.isclose(
                float(
                    branch_normalization.get("initialSquaredScale", 0)
                ),
                BRANCH_NORMALIZATION_EPSILON,
                rel_tol=1e-12,
                abs_tol=1e-15,
            ):
        raise ValueError("training plan contains invalid objective weights")
    if int(training["patience"]) != 1024:
        raise ValueError("the experiment patience must be exactly 1024 stale epochs")
    if training.get("mixedPrecision") != "bfloat16":
        raise ValueError("the experiment mixed precision must be bfloat16")
    if not 0 <= float(training["dropout"]) < 1 \
            or not 0 <= float(training["dropoutRate"]) <= 1:
        raise ValueError(
            "training dropout or dropout rate is invalid"
        )


def validate_source_manifest(manifest: dict) -> None:
    if manifest.get("version") != 8:
        raise ValueError("the source oracle dataset must use schema 8")
    if manifest.get("samplingIntervalMs") != SECOND_MS:
        raise ValueError("the source corpus must contain every one-second candle")
    if manifest.get("predictionDelayMs") != INPUT_HORIZON_SECONDS * SECOND_MS:
        raise ValueError(
            "the source target must be delayed by the exact 60-minute input horizon"
        )
    if manifest.get("actionCount") != OUTPUT_ACTION_COUNT \
            or len(manifest.get("grid", ())) != OUTPUT_ACTION_COUNT:
        raise ValueError("the source raw-oracle action grid must contain 255 cells")
    minute_oracle = manifest.get("minuteOracleMap", {})
    if minute_oracle.get("factorFileField") \
            != "minuteOracleProbabilities" \
            or minute_oracle.get("factorDtype") != "float32" \
            or minute_oracle.get("factorCompression") != "zstd" \
            or minute_oracle.get("storedRowsPerUtcDay") \
            != MINUTE_ROWS_PER_DAY \
            or minute_oracle.get("holdingPeriodSteps") != 1 \
            or minute_oracle.get("valueHorizonSteps") != INPUT_RETURN_COUNT \
            or minute_oracle.get("normalized") is not True \
            or "close-only one-minute path" not in str(
                minute_oracle.get("sampling", "")
            ):
        raise ValueError(
            "the source completed-minute oracle distribution contract is invalid"
        )


def select_segments(
    manifest: dict,
    source_root: Path,
) -> dict[str, list[Segment]]:
    shards: list[SourceShard] = []
    for value in manifest["shards"]:
        split = value["split"]
        if split not in ("train", "validation", "test"):
            continue
        shard = SourceShard(
            split=split,
            date=value["date"],
            count=int(value["count"]),
            feature_file=source_root / value["features"],
            feature_row_offset=int(value["featureRowOffset"]),
            feature_row_stride=int(value["featureRowStride"]),
            target_file=source_root / value["minuteOracleProbabilities"],
            target_row_offset=int(value["oracleRowOffset"]),
            target_row_stride=int(value["oracleRowStride"]),
            prediction_time_start=int(value["predictionTimeStart"]),
            oracle_target_time_start=int(value["oracleTargetTimeStart"]),
        )
        if shard.prediction_time_start - shard.oracle_target_time_start \
                != INPUT_HORIZON_SECONDS * SECOND_MS:
            raise ValueError("source shard does not preserve the 60-minute pairing")
        if shard.feature_row_stride != 1 or shard.target_row_stride != 1:
            raise ValueError(
                "the standalone experiment requires contiguous source components"
            )
        shards.append(shard)
    shards.sort(key=lambda item: item.prediction_time_start)
    previous_end: int | None = None
    previous_split: str | None = None
    selected = {"train": [], "validation": [], "test": []}
    for shard in shards:
        local_offset = 0
        if previous_end is not None:
            if shard.prediction_time_start <= previous_end:
                raise ValueError("source shard timestamps overlap")
            if shard.split != previous_split:
                earliest_nonoverlap = (
                    previous_end + INPUT_HORIZON_SECONDS * SECOND_MS + SECOND_MS
                )
                if shard.prediction_time_start < earliest_nonoverlap:
                    local_offset = math.ceil(
                        (earliest_nonoverlap - shard.prediction_time_start)
                        / SECOND_MS
                    )
        if local_offset < shard.count:
            selected[shard.split].append(
                Segment(shard, local_offset, shard.count - local_offset)
            )
        previous_end = shard.prediction_time_end
        previous_split = shard.split

    test_segments = take_tail(selected["test"], TEST_EXAMPLES)
    if sum(segment.count for segment in test_segments) != TEST_EXAMPLES:
        raise ValueError(
            f"the source test split must contain at least {TEST_EXAMPLES:,} "
            "post-purge examples"
        )
    selected["test"] = test_segments
    if not selected["train"] or not selected["validation"]:
        raise ValueError("train and validation splits must be non-empty")
    validate_split_disjointness(selected)
    return selected


def take_tail(segments: list[Segment], count: int) -> list[Segment]:
    remaining = count
    result: list[Segment] = []
    for segment in reversed(segments):
        selected_count = min(segment.count, remaining)
        if selected_count:
            result.append(Segment(
                segment.shard,
                segment.local_offset + segment.count - selected_count,
                selected_count,
            ))
            remaining -= selected_count
        if remaining == 0:
            break
    result.reverse()
    return result


def validate_split_disjointness(segments: dict[str, list[Segment]]) -> None:
    ranges = sorted(
        (
            segment.prediction_time_start - INPUT_HORIZON_SECONDS * SECOND_MS,
            segment.prediction_time_end,
            split,
        )
        for split, values in segments.items()
        for segment in values
    )
    previous_end: int | None = None
    previous_split: str | None = None
    for start, end, split in ranges:
        if previous_end is not None and split != previous_split \
                and start <= previous_end:
            raise ValueError(
                f"{previous_split} and {split} use overlapping one-minute input candles"
            )
        if previous_end is None or end > previous_end:
            previous_end = end
            previous_split = split


def prepare_return_components(
    history_root: Path,
    feature_root: Path,
    feature_dates: list[str],
    reporter: RunReporter,
) -> None:
    feature_root.mkdir(parents=True, exist_ok=True)
    close_cache: OrderedDict[str, np.ndarray] = OrderedDict()
    compressor = zstandard.ZstdCompressor(level=7, threads=0)
    total = len(feature_dates)
    started = time.monotonic()
    completed = 0
    for index, component_date in enumerate(feature_dates, start=1):
        output_file = feature_root / f"{component_date}{RETURN_COMPONENT_SUFFIX}"
        if valid_zstd_component(
            output_file,
            MINUTE_ROWS_PER_DAY * FLOAT16_ROW_BYTES,
        ):
            completed += 1
        else:
            current_day = date.fromisoformat(component_date)
            previous_date = (current_day - timedelta(days=1)).isoformat()
            previous_close = load_daily_close(
                history_root,
                previous_date,
                close_cache,
            )
            current_close = load_daily_close(
                history_root,
                component_date,
                close_cache,
            )
            minute_features = completed_minute_simple_return_rows(
                previous_close,
                current_close,
            )
            temporary = output_file.with_suffix(output_file.suffix + ".tmp")
            with temporary.open("wb") as raw_output:
                raw_output.write(compressor.compress(
                    minute_features.astype("<f2", copy=False).tobytes()
                ))
            os.replace(temporary, output_file)
            completed += 1
        elapsed = max(time.monotonic() - started, 1e-6)
        if index == 1 or index % 10 == 0 or index == total:
            reporter.status(
                "dataset-preparation",
                message=(
                    f"Prepared {completed:,}/{total:,} daily 60-return components."
                ),
                latest={
                    "date": component_date,
                    "components": completed,
                    "totalComponents": total,
                },
            )
            reporter.emit({
                "event": "dataset-progress",
                "date": component_date,
                "day": index,
                "days": total,
                "split": "simple-returns",
                "examplesPerSecond": completed * DAY_SECONDS / elapsed,
            })


def load_daily_close(
    history_root: Path,
    component_date: str,
    cache: OrderedDict[str, np.ndarray],
) -> np.ndarray:
    cached = cache.pop(component_date, None)
    if cached is not None:
        cache[component_date] = cached
        return cached
    file = history_root / f"{component_date}.jsonl.gz"
    if not file.is_file():
        raise FileNotFoundError(f"missing one-second history: {file}")
    with gzip.open(file, "rb") as source:
        decoded = source.read()
    values = np.fromiter(
        (float(match.group(1)) for match in CLOSE_PATTERN.finditer(decoded)),
        dtype=np.float64,
    )
    if values.shape != (DAY_SECONDS,) or not np.isfinite(values).all() \
            or bool((values <= 0).any()):
        raise ValueError(
            f"one-second close history must contain {DAY_SECONDS:,} "
            f"positive values: {file}"
        )
    cache[component_date] = values
    while len(cache) > 3:
        cache.popitem(last=False)
    return values


def valid_zstd_component(file: Path, expected_bytes: int) -> bool:
    if not file.is_file() or file.stat().st_size < 1:
        return False
    try:
        value = zstandard.frame_content_size(file.read_bytes())
        # python-zstandard returns -1 for a streaming frame with unknown
        # content size even though its exported constant is unsigned.
        return value in (
            expected_bytes,
            -1,
            zstandard.CONTENTSIZE_UNKNOWN,
        )
    except zstandard.ZstdError:
        return False


def train(
    plan: dict,
    plan_file: Path,
    dataset: ExperimentDataset,
    model_parameters: int,
    reporter: RunReporter,
    started_at: str,
    feature_mean: Tensor,
    feature_std: Tensor,
) -> None:
    training = plan["training"]
    device = torch.device(training["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA training was requested but is unavailable")
    torch.manual_seed(int(training["seed"]))
    np.random.seed(int(training["seed"]))
    random.seed(int(training["seed"]))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(training["seed"]))
    torch.set_float32_matmul_precision("high")
    model = ReturnOracleMlp(
        feature_mean,
        feature_std,
        dropout=float(training["dropout"]),
        dropout_rate=float(training["dropoutRate"]),
        normalization_family=str(
            training["branchNormalization"]["family"]
        ),
        normalization_initial_scale=float(
            training["branchNormalization"]["initialScale"]
        ),
        normalization_minimum_scale=float(
            training["branchNormalization"]["minimumScale"]
        ),
        learnable_centering=bool(
            training["branchNormalization"]["learnableCentering"]
        ),
    ).to(device)
    if parameter_count(model) != model_parameters:
        raise RuntimeError("model parameter count changed after device placement")
    optimizer_config = training["optimizer"]
    muon_config = optimizer_config["muon"]
    adamw_config = optimizer_config["adamw"]
    muon_parameters, adamw_parameters = hybrid_optimizer_parameters(model)
    muon_optimizer = torch.optim.Muon(
        muon_parameters,
        lr=float(training["learningRate"]),
        weight_decay=float(muon_config["weightDecay"]),
        momentum=float(muon_config["momentum"]),
        nesterov=bool(muon_config["nesterov"]),
        ns_steps=int(muon_config["newtonSchulzSteps"]),
        eps=float(muon_config["epsilon"]),
        adjust_lr_fn=str(muon_config["adjustLearningRate"]),
    )
    adamw_optimizer = torch.optim.AdamW(
        adamw_parameters,
        lr=float(training["learningRate"]),
        betas=tuple(float(value) for value in adamw_config["betas"]),
        eps=float(adamw_config["epsilon"]),
        weight_decay=float(adamw_config["weightDecay"]),
        fused=device.type == "cuda",
    )
    optimizers = (muon_optimizer, adamw_optimizer)
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
    amp_dtype = (
        torch.bfloat16
        if device.type == "cuda"
        else torch.float32
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=False,
    )
    start_epoch = 0
    global_step = 0
    stale_epochs = 0
    best_validation = math.inf
    best_epoch = -1
    loss_weights = training["lossWeights"]
    reverse_kl = training["reverseKl"]
    output_regularizer = training["outputRegularizer"]
    output_regularizer_probabilities = output_regularizer[
        "applicationProbabilities"
    ]
    reverse_kl_probability = float(
        output_regularizer_probabilities["reverseKl"]
    )
    entropy_sharpness_probability = float(
        output_regularizer_probabilities["entropySharpness"]
    )
    reverse_kl_scale = 1.0 / reverse_kl_probability
    entropy_sharpness_scale = 1.0 / entropy_sharpness_probability
    reverse_kl_generator = torch.Generator(device="cpu")
    reverse_kl_generator.manual_seed(
        int(training["seed"]) + 17_003
    )
    entropy_sharpness_generator = torch.Generator(device="cpu")
    entropy_sharpness_generator.manual_seed(
        int(training["seed"]) + 17_004
    )
    soft_layer_norm = training["softLayerNorm"]
    soft_weight_bound = training["softWeightBound"]
    bounded_weights = tuple(
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.ndim == 2
    )
    centering_matrices = tuple(
        normalizer.weight
        for normalizer in model.value_centering_normalizers
    )
    centering_is_learnable = any(
        matrix.requires_grad for matrix in centering_matrices
    )
    last_checkpoint = reporter.run_dir / "last.pt"
    best_checkpoint = reporter.run_dir / "best.pt"
    if last_checkpoint.is_file():
        checkpoint = torch.load(
            last_checkpoint,
            map_location=device,
            weights_only=False,
        )
        if checkpoint.get("parameterCount") != model_parameters \
                or checkpoint.get("featureContract") != FEATURE_CONTRACT \
                or checkpoint.get(
                    "architectureContract"
                ) != ARCHITECTURE_CONTRACT \
                or checkpoint.get("objectiveContract") != OBJECTIVE_CONTRACT \
                or float(
                    checkpoint.get("reverseKl", {}).get(
                        "lossWeight",
                        -1,
                    )
                ) != float(loss_weights["reverseKl"]) \
                or float(
                    checkpoint.get("reverseKl", {}).get(
                        "predictionMixtureWeight",
                        -1,
                    )
                ) != float(reverse_kl["predictionMixtureWeight"]) \
                or float(
                    checkpoint.get("entropySharpness", {}).get(
                        "lossWeight",
                        -1,
                    )
                ) != float(loss_weights["entropySharpness"]) \
                or checkpoint.get("outputRegularizer") \
                != output_regularizer \
                or "reverseKlGeneratorState" not in checkpoint \
                or "entropySharpnessGeneratorState" not in checkpoint \
                or checkpoint.get("batchSize") != int(
                    training["batchSize"]
                ) \
                or checkpoint.get("evaluationBatchSize") != int(
                    training["evaluationBatchSize"]
                ) \
                or checkpoint.get("gradientAccumulationSteps") != int(
                    training["gradientAccumulationSteps"]
                ) \
                or checkpoint.get("optimizerContract") != optimizer_config \
                or checkpoint.get("runnerContract") != RUNNER_CONTRACT:
            raise ValueError("resume checkpoint architecture is incompatible")
        model.load_state_dict(checkpoint["model"])
        optimizer_states = checkpoint["optimizers"]
        muon_optimizer.load_state_dict(optimizer_states["muon"])
        adamw_optimizer.load_state_dict(optimizer_states["adamw"])
        if device.type == "cuda":
            for group in adamw_optimizer.param_groups:
                group["fused"] = True
                group["foreach"] = None
        if "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint["globalStep"])
        stale_epochs = int(checkpoint["staleEpochs"])
        best_validation = float(checkpoint["bestValidation"])
        best_epoch = int(checkpoint["bestEpoch"])
        reverse_kl_generator.set_state(
            checkpoint["reverseKlGeneratorState"].cpu()
        )
        entropy_sharpness_generator.set_state(
            checkpoint["entropySharpnessGeneratorState"].cpu()
        )
        if "schedulers" in checkpoint:
            scheduler_states = checkpoint["schedulers"]
            schedulers[0].load_state_dict(scheduler_states["muon"])
            schedulers[1].load_state_dict(scheduler_states["adamw"])
        else:
            for scheduler in schedulers:
                scheduler.best = best_validation
                scheduler.num_bad_epochs = min(
                    stale_epochs,
                    int(schedule["patience"]),
                )
        configured_learning_rate = float(training["learningRate"])
        for optimizer, scheduler in zip(optimizers, schedulers):
            for group in optimizer.param_groups:
                group["lr"] = min(
                    float(group["lr"]),
                    configured_learning_rate,
                )
            scheduler._last_lr = [
                float(group["lr"])
                for group in optimizer.param_groups
            ]

    output_regularizer_gate_on = torch.ones(
        (),
        device=device,
        dtype=torch.float32,
    )
    output_regularizer_gate_off = torch.zeros(
        (),
        device=device,
        dtype=torch.float32,
    )
    validation_reverse_kl_gate = torch.full(
        (),
        reverse_kl_probability,
        device=device,
        dtype=torch.float32,
    )
    validation_entropy_sharpness_gate = torch.full(
        (),
        entropy_sharpness_probability,
        device=device,
        dtype=torch.float32,
    )

    def training_objective(
        features: Tensor,
        targets: Tensor,
        example_weights: Tensor,
        reverse_kl_gate: Tensor,
        entropy_sharpness_gate: Tensor,
    ) -> dict[str, Tensor]:
        (
            logits,
            mean_penalties,
            variance_penalties,
            distribution_sum_penalties,
            distribution_negative_penalties,
        ) = (
            model.forward_with_regularizers(features)
        )
        weight_bound_penalty = soft_weight_bound_penalty(
            bounded_weights,
            desired_magnitude=float(
                soft_weight_bound["desiredMagnitude"]
            ),
            sharpness=float(soft_weight_bound["sharpness"]),
            absolute_epsilon=float(
                soft_weight_bound["absoluteEpsilon"]
            ),
        )
        if centering_is_learnable:
            (
                centering_idempotence_penalty,
                centering_symmetry_penalty,
            ) = centering_matrix_constraint_components(centering_matrices)
        else:
            centering_idempotence_penalty = logits.new_zeros(
                (), dtype=torch.float32
            )
            centering_symmetry_penalty = logits.new_zeros(
                (), dtype=torch.float32
            )
        return oracle_policy_objective(
            logits,
            targets,
            mean_penalties,
            variance_penalties,
            distribution_sum_penalties,
            distribution_negative_penalties,
            weight_bound_penalty,
            centering_idempotence_penalty,
            centering_symmetry_penalty,
            example_weights,
            cross_entropy_weight=float(loss_weights["crossEntropy"]),
            reverse_kl_weight=float(loss_weights["reverseKl"]),
            reverse_kl_prediction_mixture=float(
                reverse_kl["predictionMixtureWeight"]
            ),
            entropy_sharpness_weight=float(
                loss_weights["entropySharpness"]
            ),
            reverse_kl_gate=reverse_kl_gate,
            entropy_sharpness_gate=entropy_sharpness_gate,
            reverse_kl_scale=reverse_kl_scale,
            entropy_sharpness_scale=entropy_sharpness_scale,
            soft_layer_norm_weight=float(loss_weights["softLayerNorm"]),
            soft_weight_bound_weight=float(
                loss_weights["softWeightBound"]
            ),
            variance_weight=float(soft_layer_norm["varianceWeight"]),
            distribution_sum_weight=float(
                loss_weights["distributionLayerSum"]
            ),
            distribution_negative_weight=float(
                loss_weights["distributionLayerNegative"]
            ),
            centering_idempotence_weight=float(
                loss_weights["centeringIdempotence"]
            ),
            centering_symmetry_weight=float(
                loss_weights["centeringSymmetry"]
            ),
        )

    def evaluation_objective(
        features: Tensor,
        targets: Tensor,
        example_weights: Tensor,
    ) -> dict[str, Tensor]:
        (
            logits,
            mean_penalties,
            variance_penalties,
            distribution_sum_penalties,
            distribution_negative_penalties,
        ) = (
            model.forward_with_regularizers(features)
        )
        zero = logits.new_zeros((), dtype=torch.float32)
        return oracle_policy_objective(
            logits,
            targets,
            mean_penalties,
            variance_penalties,
            distribution_sum_penalties,
            distribution_negative_penalties,
            zero,
            zero,
            zero,
            example_weights,
            cross_entropy_weight=float(loss_weights["crossEntropy"]),
            reverse_kl_weight=float(loss_weights["reverseKl"]),
            reverse_kl_prediction_mixture=float(
                reverse_kl["predictionMixtureWeight"]
            ),
            entropy_sharpness_weight=float(
                loss_weights["entropySharpness"]
            ),
            reverse_kl_gate=validation_reverse_kl_gate,
            entropy_sharpness_gate=validation_entropy_sharpness_gate,
            reverse_kl_scale=reverse_kl_scale,
            entropy_sharpness_scale=entropy_sharpness_scale,
            soft_layer_norm_weight=float(loss_weights["softLayerNorm"]),
            soft_weight_bound_weight=float(
                loss_weights["softWeightBound"]
            ),
            variance_weight=float(soft_layer_norm["varianceWeight"]),
            distribution_sum_weight=float(
                loss_weights["distributionLayerSum"]
            ),
            distribution_negative_weight=float(
                loss_weights["distributionLayerNegative"]
            ),
            centering_idempotence_weight=float(
                loss_weights["centeringIdempotence"]
            ),
            centering_symmetry_weight=float(
                loss_weights["centeringSymmetry"]
            ),
        )

    @torch.inference_mode()
    def evaluation_constant_regularizers() -> dict[str, float]:
        weight_bound_penalty = soft_weight_bound_penalty(
            bounded_weights,
            desired_magnitude=float(
                soft_weight_bound["desiredMagnitude"]
            ),
            sharpness=float(soft_weight_bound["sharpness"]),
            absolute_epsilon=float(
                soft_weight_bound["absoluteEpsilon"]
            ),
        )
        if centering_is_learnable:
            (
                centering_idempotence_penalty,
                centering_symmetry_penalty,
            ) = centering_matrix_constraint_components(centering_matrices)
        else:
            centering_idempotence_penalty = weight_bound_penalty.new_zeros(())
            centering_symmetry_penalty = weight_bound_penalty.new_zeros(())
        centering_constraint = (
            float(loss_weights["centeringIdempotence"])
            * centering_idempotence_penalty
            + float(loss_weights["centeringSymmetry"])
            * centering_symmetry_penalty
        )
        loss_addition = (
            float(loss_weights["softWeightBound"]) * weight_bound_penalty
            + centering_constraint
        )
        return {
            "softWeightBound": float(weight_bound_penalty),
            "centeringIdempotence": float(
                centering_idempotence_penalty
            ),
            "centeringSymmetry": float(centering_symmetry_penalty),
            "centeringConstraint": float(centering_constraint),
            "lossAddition": float(loss_addition),
        }

    if bool(training.get("compile", False)):
        compile_options = {"triton.cudagraphs": False}
        training_objective = torch.compile(
            training_objective,
            options=compile_options,
            dynamic=True,
            fullgraph=False,
        )
        evaluation_objective = torch.compile(
            evaluation_objective,
            options=compile_options,
            dynamic=True,
            fullgraph=False,
        )
    batch_size = int(training["batchSize"])
    evaluation_batch_size = int(training["evaluationBatchSize"])
    gradient_accumulation_steps = int(
        training["gradientAccumulationSteps"]
    )
    micro_batches_per_epoch = dataset.batch_count("train", batch_size)
    optimizer_steps_per_epoch = math.ceil(
        micro_batches_per_epoch / gradient_accumulation_steps
    )
    patience = int(training["patience"])
    log_every = int(training["logEverySteps"])
    maximum_epochs = int(training.get("epochs", 1_000_000))
    batch_pipeline = DeviceBatchPipeline(device)
    reporter.status(
        "training",
        startedAt=started_at,
        planId=plan["id"],
        startEpoch=start_epoch,
        parameters=model_parameters,
        trainExamples=dataset.count("train"),
        validationExamples=dataset.count("validation"),
        testExamples=dataset.count("test"),
        batchSize=batch_size,
        gradientAccumulationSteps=gradient_accumulation_steps,
        microBatchesPerEpoch=micro_batches_per_epoch,
        batchesPerEpoch=optimizer_steps_per_epoch,
        message=(
            f"Training the {HIDDEN_LAYER_COUNT}-layer uniform-"
            f"{HIDDEN_WIDTHS[0]} fused-GLU MLP in "
            f"{batch_size:,}-example "
            "batches "
            "with fully normalized additive GLU residuals from the original "
            "input and every earlier hidden layer, sharing one C and "
            "separate value/gate A matrices across paths at each target, "
            "separate post-projection learned-radius tanh "
            "normalizers initialized with radius sqrt(1e-5), with "
            "learnable value/gate radius s and fixed canonical C per layer, "
            "then "
            "full A transforms and post-A biases against "
            "soft-target cross-entropy plus independent optimizer-update "
            "gates for inverse-probability-corrected skew reverse KL at 10% "
            "and one-sided output entropy sharpness at 10%, alongside soft "
            "LayerNorm and soft weight-bound losses; fixed-C projector "
            "diagnostics are skipped; distribution-layer metrics are "
            "diagnostic only; stop "
            "occurs when stale epochs exceed 1024."
        ),
    )
    reporter.emit({
        "event": "training-start",
        "startEpoch": start_epoch,
        "epochs": maximum_epochs,
        "patience": patience,
        "parameters": model_parameters,
        "batchSize": batch_size,
        "evaluationBatchSize": evaluation_batch_size,
        "gradientAccumulationSteps": gradient_accumulation_steps,
        "microBatchesPerEpoch": micro_batches_per_epoch,
        "batchesPerEpoch": optimizer_steps_per_epoch,
        "featureContract": FEATURE_CONTRACT,
        "architectureContract": ARCHITECTURE_CONTRACT,
        "objectiveContract": OBJECTIVE_CONTRACT,
        "runnerContract": RUNNER_CONTRACT,
        "inputStandardization": training["inputStandardization"],
        "inputNormalization": {
            "source": "selected-training-split-only",
            "axis": "per-return-position",
            "varianceCorrection": 0,
        },
        "hiddenActivation": "fused-glu",
        "branchNormalization": {
            "placement": "post-projection-pre-glu",
            "branches": "separate-value-and-gate-logit",
            "family": training["branchNormalization"]["family"],
            "formula": "A*(C*h/(s*g(r/s)))+b",
            "tanhFormula": "A*(C*h*tanh(r/s)/r)+b",
            "rms": "r=sqrt(mean((C*h)^2))",
            "denominator": training["branchNormalization"]["denominator"],
            "limits": {
                "atZero": "g(0)=1",
                "atInfinity": "g(u)/u->1",
                "positive": True,
            },
            "scale": {
                "learnable": True,
                "sharing": "one-scalar-per-value-or-gate-branch",
                "count": (
                    4 * HIDDEN_LAYER_COUNT
                    + 2 * DENSE_RESIDUAL_PATH_COUNT
                ),
                "role": training["branchNormalization"]["scaleRole"],
                "initialValue": float(
                    training["branchNormalization"]["initialScale"]
                ),
                "initialSquaredValue": float(
                    training["branchNormalization"]["initialSquaredScale"]
                ),
                "minimumValue": float(
                    training["branchNormalization"]["minimumScale"]
                ),
                "parameterization": training["branchNormalization"][
                    "scaleParameterization"
                ],
            },
            "learnablePostTransformBias": True,
            "centeringMatrix": {
                "learnable": False,
                "sharing": "one-shared-matrix-across-four-paths-per-layer",
                "matrixCount": HIDDEN_LAYER_COUNT,
                "shape": "full-square-at-each-hidden-width",
                "initialization": "I-minus-11T-over-width",
                "fixedValue": "I-minus-11T-over-width",
                "normalization": "C-h-over-learned-tanh-rms-radius",
            },
            "softLossMeasurement": "raw-pre-branch-norm-projections",
            "absorbedInputTransform": (
                "B-absorbed-into-fused-value-gate-projection"
            ),
            "postNormalizationTransform": {
                "branches": (
                    "value-shared-main-residual-and-separate-gate-shared-"
                    "main-residual"
                ),
                "shape": "full-square-at-each-hidden-width",
                "bias": "separate-post-A-vector",
                "initialization": "identity",
            },
            "globalInputResidual": {
                "formula": (
                    "hidden=GLU(main_projection)+GLU(input_projection)"
                ),
                "source": "shared-standardized-60-return-model-input",
                "matrixCount": HIDDEN_LAYER_COUNT,
                "initialization": (
                    "zero-value-half; kaiming-gate-half; zero-bias"
                ),
                "normalization": "independent-branches-sharing-layer-C-and-A",
            },
            "denseLayerResiduals": {
                "formula": (
                    "hidden_l=main_l(previous)+input_l(x0)+"
                    "sum_{j<l-1} residual_{l,j}(hidden_j)"
                ),
                "pathCount": DENSE_RESIDUAL_PATH_COUNT,
                "aggregation": "additive",
                "normalization": (
                    "path-specific-radius-and-bias; target-layer-shared-"
                    "C-value-A-gate-A"
                ),
            },
            "centeringConstraints": {
                "idempotence": "mean-square-C2-minus-C",
                "symmetry": "mean-square-CT-minus-C",
                "layerAggregation": "mean",
                "idempotenceWeight": float(
                    loss_weights["centeringIdempotence"]
                ),
                "symmetryWeight": float(
                    loss_weights["centeringSymmetry"]
                ),
            },
        },
        "optimizer": {
            "type": optimizer_config["type"],
            "muon": {
                **muon_config,
                "parameters": sum(
                    parameter.numel() for parameter in muon_parameters
                ),
                "routing": (
                    f"all-{HIDDEN_LAYER_COUNT}-fused-value-gate-"
                    f"projection-{HIDDEN_LAYER_COUNT}-global-input-GLU-"
                    f"residual-{DENSE_RESIDUAL_PATH_COUNT}-dense-residual-"
                    f"projections-and-{2 * HIDDEN_LAYER_COUNT}-path-shared-"
                    "post-normalization-A-matrices"
                ),
            },
            "adamw": {
                **adamw_config,
                "parameters": sum(
                    parameter.numel() for parameter in adamw_parameters
                ),
                "routing": (
                    "output-head-all-affine-and-residual-biases-"
                    f"{4 * HIDDEN_LAYER_COUNT + 2 * DENSE_RESIDUAL_PATH_COUNT}"
                    "-post-A-offsets-and-soft-normalization-scales; "
                    "fixed-C-matrices-excluded"
                ),
            },
        },
        "compiledObjective": bool(training.get("compile", False)),
        "mixedPrecision": training["mixedPrecision"],
        "gradientScaling": scaler.is_enabled(),
        "compileMode": (
            "dynamic-shapes-no-cudagraphs"
            if bool(training.get("compile", False))
            else "disabled"
        ),
        "dataLoaderWorkers": int(training["workers"]),
        "prefetchFactor": int(training["prefetchFactor"]),
        "nextEpochPrefetchDuringValidation": True,
        "asynchronousCheckpointing": True,
        "validationBatchInvariantRegularizers": (
            "computed-once-per-evaluation-pass"
        ),
        "dropoutProbability": float(training["dropout"]),
        "dropoutRate": float(training["dropoutRate"]),
        "dropoutGateProbability": (
            float(training["dropoutRate"]) ** 0.5
        ),
        "reverseKl": {
            "direction": (
                "KL(predicted || (1-epsilon)*oracle + "
                "epsilon*predicted)"
            ),
            "lossWeight": float(loss_weights["reverseKl"]),
            "predictionMixtureWeight": float(
                reverse_kl["predictionMixtureWeight"]
            ),
            "referenceTransform": (
                "(1-epsilon)*oracle + epsilon*predicted"
            ),
            "applicationProbability": (
                reverse_kl_probability
            ),
        },
        "entropySharpness": {
            "lossWeight": float(loss_weights["entropySharpness"]),
            "formula": "relu(H(predicted)-H(oracle))^2",
            "exampleAggregation": "source-multiplicity-weighted-mean",
            "applicationRate": entropy_sharpness_probability,
            "margin": 0.0,
        },
        "outputRegularizer": {
            **output_regularizer,
            "inverseProbabilityScales": {
                "reverseKl": reverse_kl_scale,
                "entropySharpness": entropy_sharpness_scale,
            },
            "expectedWeights": {
                "reverseKl": float(loss_weights["reverseKl"]),
                "entropySharpness": float(loss_weights["entropySharpness"]),
            },
            "activeWeights": {
                "reverseKl": (
                    float(loss_weights["reverseKl"])
                    * reverse_kl_scale
                ),
                "entropySharpness": (
                    float(loss_weights["entropySharpness"])
                    * entropy_sharpness_scale
                ),
            },
            "validationGates": {
                "reverseKl": reverse_kl_probability,
                "entropySharpness": entropy_sharpness_probability,
            },
        },
        "softLayerNorm": {
            "lossWeight": float(loss_weights["softLayerNorm"]),
            "diagnosticOnly": (
                float(loss_weights["softLayerNorm"]) == 0
            ),
            "varianceWeight": float(soft_layer_norm["varianceWeight"]),
            "measurement": "raw-value-and-gate-affine-branches",
            "branchAggregation": "mean-of-value-and-gate-logit",
            "layerAggregation": "mean",
        },
        "distributionLayer": {
            "sumWeight": float(
                loss_weights["distributionLayerSum"]
            ),
            "negativeWeight": float(
                loss_weights["distributionLayerNegative"]
            ),
            "diagnosticOnly": (
                float(loss_weights["distributionLayerSum"]) == 0
                and float(loss_weights["distributionLayerNegative"]) == 0
            ),
            "activation": "post-glu-pre-dropout",
            "sumNormalization": "divide-error-by-layer-width",
            "negativeNormalization": "mean-across-layer-width",
            "layerAggregation": "mean",
        },
        "softWeightBound": {
            "lossWeight": float(loss_weights["softWeightBound"]),
            "desiredMagnitude": float(
                soft_weight_bound["desiredMagnitude"]
            ),
            "sharpness": float(soft_weight_bound["sharpness"]),
            "absoluteEpsilon": float(
                soft_weight_bound["absoluteEpsilon"]
            ),
            "parameterAggregation": "mean",
            "muonWeightDecay": float(muon_config["weightDecay"]),
            "adamwWeightDecay": float(adamw_config["weightDecay"]),
        },
        "learningRateSchedule": schedule,
        "trainExamples": dataset.count("train"),
        "validationExamples": dataset.count("validation"),
        "testExamples": dataset.count("test"),
        "planFile": str(plan_file),
    })

    stopped_for_patience = False
    prepared_train_iterator: PreparedBatchIterator | None = None
    checkpoint_writer = AsyncCheckpointWriter()
    try:
        for epoch in range(start_epoch, maximum_epochs):
            epoch_started = time.monotonic()
            model.train()
            train_metrics = MetricAccumulator()
            examples_since_log = 0
            network_rows_since_log = 0
            log_started = time.monotonic()
            batches = optimizer_steps_per_epoch
            train_source = (
                prepared_train_iterator
                if prepared_train_iterator is not None
                else dataset.iter_batches(
                    "train",
                    batch_size,
                    shuffle=True,
                    seed=int(training["seed"]) + epoch,
                )
            )
            prepared_train_iterator = None
            device_batches = batch_pipeline.batches(train_source)
            for batch_index, batch_group in enumerate(group_batches(
                device_batches,
                gradient_accumulation_steps,
            )):
                reverse_kl_applied = bool(
                    torch.rand(
                        (),
                        generator=reverse_kl_generator,
                    ).item()
                    < reverse_kl_probability
                )
                entropy_sharpness_applied = bool(
                    torch.rand(
                        (),
                        generator=entropy_sharpness_generator,
                    ).item()
                    < entropy_sharpness_probability
                )
                reverse_kl_gate = (
                    output_regularizer_gate_on
                    if reverse_kl_applied
                    else output_regularizer_gate_off
                )
                entropy_sharpness_gate = (
                    output_regularizer_gate_on
                    if entropy_sharpness_applied
                    else output_regularizer_gate_off
                )
                for optimizer in optimizers:
                    optimizer.zero_grad(set_to_none=True)
                count = sum(batch[3] for batch in batch_group)
                step_network_rows = sum(
                    batch[0].shape[0] for batch in batch_group
                )
                step_metrics = MetricAccumulator()
                for micro_batch_index, (
                    features,
                    targets,
                    example_weights,
                    original_count,
                ) in enumerate(batch_group):
                    with torch.autocast(
                        device_type=device.type,
                        dtype=amp_dtype,
                        enabled=device.type == "cuda",
                    ):
                        metrics = training_objective(
                            features,
                            targets,
                            example_weights,
                            reverse_kl_gate,
                            entropy_sharpness_gate,
                        )
                    loss = metrics["loss"]
                    if device.type == "cuda":
                        torch._assert_async(
                            torch.isfinite(loss),
                            f"non-finite training loss at epoch {epoch}, "
                            f"batch {batch_index}, microbatch "
                            f"{micro_batch_index}",
                        )
                    elif not bool(torch.isfinite(loss)):
                        raise FloatingPointError(
                            f"non-finite training loss at epoch {epoch}, "
                            f"batch {batch_index}, microbatch "
                            f"{micro_batch_index}"
                        )
                    scaled_loss = loss * (original_count / count)
                    scaler.scale(scaled_loss).backward()
                    train_metrics.add(metrics, original_count)
                    step_metrics.add(metrics, original_count)
                for optimizer in optimizers:
                    scaler.unscale_(optimizer)
                gradient_norm = clip_grad_norm_(
                    model.parameters(),
                    float(training["gradientClip"]),
                    foreach=device.type == "cuda",
                )
                for optimizer in optimizers:
                    scaler.step(optimizer)
                scaler.update()
                global_step += 1
                examples_since_log += count
                network_rows_since_log += step_network_rows
                if reverse_kl_applied \
                        or entropy_sharpness_applied \
                        or global_step == 1 \
                        or global_step % log_every == 0:
                    elapsed = max(time.monotonic() - log_started, 1e-6)
                    gradient_norm_value = float(gradient_norm)
                    gradient_overflow = not math.isfinite(
                        gradient_norm_value
                    )
                    latest = step_metrics.result()
                    memory = gpu_memory(device)
                    event = {
                        "event": "train-step",
                        "epoch": epoch,
                        "epochs": maximum_epochs,
                        "batch": batch_index,
                        "batches": batches,
                        "microBatches": len(batch_group),
                        "effectiveBatchExamples": count,
                        "globalStep": global_step,
                        "learningRate": (
                            muon_optimizer.param_groups[0]["lr"]
                        ),
                        "muonLearningRate": (
                            muon_optimizer.param_groups[0]["lr"]
                        ),
                        "adamwLearningRate": (
                            adamw_optimizer.param_groups[0]["lr"]
                        ),
                        "gradientNorm": (
                            None
                            if gradient_overflow
                            else gradient_norm_value
                        ),
                        "gradientOverflow": gradient_overflow,
                        "gradientScale": scaler.get_scale(),
                        "reverseKlApplied": reverse_kl_applied,
                        "entropySharpnessApplied": (
                            entropy_sharpness_applied
                        ),
                        "examplesPerSecond": examples_since_log / elapsed,
                        "networkRowsPerSecond": (
                            network_rows_since_log / elapsed
                        ),
                        "batchCompactionRatio": (
                            count / step_network_rows
                        ),
                        **memory,
                        "latest": latest,
                    }
                    reporter.emit(event)
                    reporter.status(
                        "training",
                        startedAt=started_at,
                        planId=plan["id"],
                        latest=event,
                        bestEpoch=best_epoch,
                        staleEpochs=stale_epochs,
                        **(
                            {"bestValidation": best_validation}
                            if math.isfinite(best_validation)
                            else {}
                        ),
                    )
                    examples_since_log = 0
                    network_rows_since_log = 0
                    log_started = time.monotonic()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            train_result = train_metrics.result()
            training_completed = time.monotonic()

            if epoch + 1 < maximum_epochs:
                prepared_train_iterator = PreparedBatchIterator(
                    dataset.iter_batches(
                        "train",
                        batch_size,
                        shuffle=True,
                        seed=int(training["seed"]) + epoch + 1,
                    ),
                    prefetch_batches=(
                        int(training["workers"])
                        * int(training["prefetchFactor"])
                    ),
                )
            prefetch_enqueued = time.monotonic()
            validation = evaluate(
                evaluation_objective,
                model,
                dataset,
                "validation",
                evaluation_batch_size,
                device,
                amp_dtype,
                batch_pipeline,
                evaluation_constant_regularizers(),
            )
            validation_completed = time.monotonic()
            validation_loss = validation["loss"]
            if not math.isfinite(validation_loss):
                raise FloatingPointError(
                    "combined validation objective is non-finite"
                )
            improved = validation_loss < best_validation
            if improved:
                best_validation = validation_loss
                best_epoch = epoch
                stale_epochs = 0
            else:
                stale_epochs += 1
            learning_rate_before = float(
                muon_optimizer.param_groups[0]["lr"]
            )
            for scheduler in schedulers:
                scheduler.step(validation_loss)
            learning_rate = float(
                muon_optimizer.param_groups[0]["lr"]
            )
            adamw_learning_rate = float(
                adamw_optimizer.param_groups[0]["lr"]
            )
            learning_rate_reduced = learning_rate < learning_rate_before
            checkpoint = {
                "model": model.state_dict(),
                "optimizers": {
                    "muon": muon_optimizer.state_dict(),
                    "adamw": adamw_optimizer.state_dict(),
                },
                "schedulers": {
                    "muon": schedulers[0].state_dict(),
                    "adamw": schedulers[1].state_dict(),
                },
                "scaler": scaler.state_dict(),
                "reverseKlGeneratorState": reverse_kl_generator.get_state(),
                "entropySharpnessGeneratorState": (
                    entropy_sharpness_generator.get_state()
                ),
                "epoch": epoch,
                "globalStep": global_step,
                "staleEpochs": stale_epochs,
                "bestValidation": best_validation,
                "bestEpoch": best_epoch,
                "validation": validation,
                "parameterCount": model_parameters,
                "batchSize": batch_size,
                "evaluationBatchSize": evaluation_batch_size,
                "gradientAccumulationSteps": (
                    gradient_accumulation_steps
                ),
                "featureContract": FEATURE_CONTRACT,
                "architectureContract": ARCHITECTURE_CONTRACT,
                "objectiveContract": OBJECTIVE_CONTRACT,
                "runnerContract": RUNNER_CONTRACT,
                "mixedPrecision": training["mixedPrecision"],
                "optimizerContract": optimizer_config,
                "dropoutProbability": float(training["dropout"]),
                "dropoutRate": float(training["dropoutRate"]),
                "reverseKl": {
                    "direction": (
                        "KL(predicted || (1-epsilon)*oracle + "
                        "epsilon*predicted)"
                    ),
                    "lossWeight": float(loss_weights["reverseKl"]),
                    "predictionMixtureWeight": float(
                        reverse_kl["predictionMixtureWeight"]
                    ),
                },
                "entropySharpness": {
                    "lossWeight": float(
                        loss_weights["entropySharpness"]
                    ),
                    "formula": (
                        "relu(H(predicted)-H(oracle))^2"
                    ),
                    "applicationRate": (
                        entropy_sharpness_probability
                    ),
                    "margin": 0.0,
                },
                "outputRegularizer": copy.deepcopy(
                    output_regularizer
                ),
                "softLayerNorm": {
                    "lossWeight": float(loss_weights["softLayerNorm"]),
                    "diagnosticOnly": (
                        float(loss_weights["softLayerNorm"]) == 0
                    ),
                    "varianceWeight": float(
                        soft_layer_norm["varianceWeight"]
                    ),
                },
                "distributionLayer": {
                    "sumWeight": float(
                        loss_weights["distributionLayerSum"]
                    ),
                    "negativeWeight": float(
                        loss_weights["distributionLayerNegative"]
                    ),
                    "diagnosticOnly": (
                        float(loss_weights["distributionLayerSum"]) == 0
                        and float(
                            loss_weights["distributionLayerNegative"]
                        ) == 0
                    ),
                    "activation": "post-glu-pre-dropout",
                    "sumNormalization": "divide-error-by-layer-width",
                    "negativeNormalization": "mean-across-layer-width",
                },
                "softWeightBound": {
                    "lossWeight": float(loss_weights["softWeightBound"]),
                    "desiredMagnitude": float(
                        soft_weight_bound["desiredMagnitude"]
                    ),
                    "sharpness": float(
                        soft_weight_bound["sharpness"]
                    ),
                    "absoluteEpsilon": float(
                        soft_weight_bound["absoluteEpsilon"]
                    ),
                },
            }
            checkpoint_io = checkpoint_writer.submit(
                checkpoint,
                last_checkpoint,
                best_checkpoint if improved else None,
            )
            checkpoint_enqueued = time.monotonic()
            epoch_event = {
                "event": "epoch",
                "epoch": epoch,
                "epochs": maximum_epochs,
                "globalStep": global_step,
                "seconds": checkpoint_enqueued - epoch_started,
                "trainingSeconds": training_completed - epoch_started,
                "validationSeconds": (
                    validation_completed - prefetch_enqueued
                ),
                "phaseSeconds": {
                    "training": training_completed - epoch_started,
                    "nextEpochPrefetchSetup": (
                        prefetch_enqueued - training_completed
                    ),
                    "validation": (
                        validation_completed - prefetch_enqueued
                    ),
                    "checkpointSnapshot": (
                        checkpoint_enqueued - validation_completed
                    ),
                },
                "train": train_result,
                "validation": validation,
                "bestValidation": best_validation,
                "bestEpoch": best_epoch,
                "staleEpochs": stale_epochs,
                "learningRate": learning_rate,
                "muonLearningRate": learning_rate,
                "adamwLearningRate": adamw_learning_rate,
                "learningRateReduced": learning_rate_reduced,
                "stopRequested": stale_epochs > patience,
                "checkpointIo": checkpoint_io,
            }
            reporter.emit(epoch_event)
            reporter.status(
                "training",
                startedAt=started_at,
                planId=plan["id"],
                latest=epoch_event,
                bestValidation=best_validation,
                bestEpoch=best_epoch,
                staleEpochs=stale_epochs,
            )
            if stale_epochs > patience:
                stopped_for_patience = True
                if prepared_train_iterator is not None:
                    prepared_train_iterator.close()
                    prepared_train_iterator = None
                break
    finally:
        if prepared_train_iterator is not None:
            prepared_train_iterator.close()
        checkpoint_writer.close()

    if not stopped_for_patience:
        raise RuntimeError(
            "maximum epochs were exhausted before validation staleness exceeded 1024"
        )
    best = torch.load(best_checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(best["model"])
    test_metrics = evaluate(
        evaluation_objective,
        model,
        dataset,
        "test",
        evaluation_batch_size,
        device,
        amp_dtype,
        batch_pipeline,
        evaluation_constant_regularizers(),
    )
    result = {
        "completedAt": iso_now(),
        "stopReason": "validation loss did not improve for more than 1024 epochs",
        "bestEpoch": best_epoch,
        "bestValidation": best_validation,
        "staleEpochs": stale_epochs,
        "testExamples": dataset.count("test"),
        "test": test_metrics,
        "checkpoint": str(best_checkpoint),
        "deployed": False,
    }
    atomic_json(result, reporter.run_dir / "result.json")
    reporter.emit({
        "event": "training-complete",
        **result,
    })
    reporter.status(
        "complete",
        startedAt=started_at,
        completedAt=result["completedAt"],
        planId=plan["id"],
        latest=result,
        message=(
            "Standalone experiment completed; the checkpoint is intentionally "
            "not exposed to the inspector or runtime."
        ),
    )


@torch.inference_mode()
def evaluate(
    objective,
    model: ReturnOracleMlp,
    dataset: ExperimentDataset,
    split: str,
    batch_size: int,
    device: torch.device,
    amp_dtype: torch.dtype,
    batch_pipeline: DeviceBatchPipeline,
    batch_invariant_regularizers: dict[str, float] | None = None,
) -> dict[str, float]:
    model.eval()
    accumulator = MetricAccumulator()
    source = dataset.iter_batches(
        split,
        batch_size,
        shuffle=False,
        seed=0,
    )
    for features, targets, example_weights, original_count \
            in batch_pipeline.batches(source):
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=device.type == "cuda",
        ):
            metrics = objective(features, targets, example_weights)
        accumulator.add(metrics, original_count)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    result = accumulator.result()
    if batch_invariant_regularizers is not None:
        result = add_batch_invariant_regularizers(
            result,
            batch_invariant_regularizers,
        )
    return result


def gpu_memory(device: torch.device) -> dict[str, float]:
    if device.type != "cuda":
        return {}
    free, total = torch.cuda.mem_get_info(device)
    return {
        "gpuAllocatedMiB": torch.cuda.memory_allocated(device) / (1024 * 1024),
        "gpuReservedMiB": torch.cuda.memory_reserved(device) / (1024 * 1024),
        "gpuDeviceUsedMiB": (total - free) / (1024 * 1024),
    }


def atomic_torch_save(value: dict, file: Path) -> None:
    file.parent.mkdir(parents=True, exist_ok=True)
    temporary = file.with_suffix(file.suffix + ".tmp")
    torch.save(value, temporary)
    replace_file_with_retry(temporary, file)


def atomic_json(value: dict, file: Path) -> None:
    file.parent.mkdir(parents=True, exist_ok=True)
    temporary = file.with_suffix(file.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    replace_file_with_retry(temporary, file)


def replace_file_with_retry(
    temporary: Path,
    destination: Path,
    attempts: int = 40,
) -> None:
    if attempts < 1:
        raise ValueError("atomic replacement attempts must be positive")
    for attempt in range(attempts):
        try:
            os.replace(temporary, destination)
            return
        except PermissionError:
            if attempt + 1 == attempts:
                raise
            time.sleep(min(0.01 * (attempt + 1), 0.25))


def resolve(repo_root: Path, value: Path) -> Path:
    return value if value.is_absolute() else repo_root / value


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    main()
