from __future__ import annotations

import argparse
import json
import math
import os
import random
import struct
import subprocess
import tempfile
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import onnx
import torch
from torch import Tensor
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset

try:
    import zstandard
except ImportError:
    zstandard = None

if os.name == "posix" and os.environ.get("TMPDIR", "").startswith("/mnt/"):
    os.environ["TMPDIR"] = "/tmp"
    tempfile.tempdir = "/tmp"

from mlp_model import (
    DirectLossWeights,
    FEATURE_SCHEMA_VERSION,
    HIDDEN_LAYER_COUNT,
    HIDDEN_WIDTH,
    INPUT_FEATURE_COUNT,
    OUTPUT_ACTION_COUNT,
    TEACHER_PARAMETER_COUNT,
    ExposureMlp,
    PolicySupport,
    TimeWeighting,
    conditional_transaction_transition,
    direct_oracle_loss,
    parameter_count,
    validate_time_weighting,
)


METRIC_NAMES = (
    "loss",
    "klDivergence",
    "klDivergenceVariance",
    "klDivergenceStdDev",
    "baseKlDivergence",
    "probabilityMse",
    "probabilityMseVariance",
    "probabilityMseStdDev",
    "excessEntropy",
    "oracleMutualInformation",
    "targetEntropy",
    "predictedEntropy",
    "distanceImbalanceWeight",
    "timeWeightEffectiveSampleRatio",
)
TRAIN_METRIC_NAMES = METRIC_NAMES
KL_MOMENT_METRIC_NAMES = frozenset((
    "klDivergence",
    "klDivergenceVariance",
    "klDivergenceStdDev",
))
PROBABILITY_MSE_MOMENT_METRIC_NAMES = frozenset((
    "probabilityMse",
    "probabilityMseVariance",
    "probabilityMseStdDev",
))
TIME_BLOCK_METRIC_NAMES = frozenset((
    "oracleMutualInformation",
))


def merge_weighted_moments(
    total_weight: Tensor,
    total_mean: Tensor,
    total_centered_square_sum: Tensor,
    batch_weight: Tensor,
    batch_mean: Tensor,
    batch_centered_square_sum: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Merge weighted batch moments without losing precision to raw E[x²] sums."""
    combined_weight = total_weight + batch_weight
    safe_weight = combined_weight.clamp_min(1e-12)
    delta = batch_mean - total_mean
    combined_mean = total_mean + delta * batch_weight / safe_weight
    combined_centered_square_sum = (
        total_centered_square_sum
        + batch_centered_square_sum
        + delta.square() * total_weight * batch_weight / safe_weight
    )
    return combined_weight, combined_mean, combined_centered_square_sum


def weighted_standard_deviation(
    centered_square_sum: Tensor,
    weight_sum: Tensor,
) -> Tensor:
    """Population standard deviation under the persisted example weights."""
    return (
        centered_square_sum / weight_sum.clamp_min(1e-12)
    ).clamp_min(0).sqrt()


def weighted_variance(
    centered_square_sum: Tensor,
    weight_sum: Tensor,
) -> Tensor:
    """Population variance under the persisted example weights."""
    return (
        centered_square_sum / weight_sum.clamp_min(1e-12)
    ).clamp_min(0)


@dataclass(frozen=True)
class Shard:
    root: Path
    count: int
    features: str
    feature_row_offset: int
    feature_row_stride: int
    teacher_parameters: str
    teacher_metrics: str
    raw_oracle_probabilities: str
    minute_oracle_probabilities: str
    resolution_divergence: str
    oracle_row_offset: int
    oracle_row_stride: int
    base_time_weights: str
    time_weights: str
    prediction_time_start: int
    oracle_target_time_start: int


class RuntimeMinuteOracleRows:
    """A per-second view over one runtime-computed oracle row per completed minute."""

    def __init__(
        self,
        probabilities_by_day: dict[int, np.ndarray],
        shard: Shard,
        action_count: int,
    ) -> None:
        self.probabilities_by_day = probabilities_by_day
        self.day_start = shard.oracle_target_time_start // 86_400_000 * 86_400_000
        self.row_offset = shard.oracle_row_offset
        self.row_stride = shard.oracle_row_stride
        self.count = shard.count
        self.action_count = action_count
        if self.day_start not in probabilities_by_day:
            raise ValueError(
                "runtime one-minute oracle is missing target day "
                f"{utc_date(self.day_start)}"
            )

    def __getitem__(self, key) -> np.ndarray:
        if isinstance(key, slice):
            start, stop, step = key.indices(self.count)
            local_rows = np.arange(start, stop, step, dtype=np.int64)
            return self._rows(local_rows)
        local_row = int(key)
        if local_row < 0:
            local_row += self.count
        if local_row < 0 or local_row >= self.count:
            raise IndexError(local_row)
        return self._rows(np.asarray([local_row], dtype=np.int64))[0]

    def _rows(self, local_rows: np.ndarray) -> np.ndarray:
        oracle_rows = self.row_offset + local_rows * self.row_stride
        coarse_rows = oracle_rows // 60 + (oracle_rows % 60 == 59)
        source = self.probabilities_by_day[self.day_start]
        if coarse_rows.size and (
            int(coarse_rows.min()) < 0 or int(coarse_rows.max()) >= source.shape[0]
        ):
            raise IndexError("runtime one-minute oracle row is outside its UTC day")
        return source[coarse_rows]


class PersistedMinuteOracleRows:
    """A per-second view over one persisted distribution per completed minute."""

    def __init__(
        self,
        probabilities: np.ndarray,
        shard: Shard,
        action_count: int,
    ) -> None:
        if probabilities.shape != (1_441, action_count):
            raise ValueError("persisted one-minute oracle has an invalid shape")
        self.probabilities = probabilities
        self.row_offset = shard.oracle_row_offset
        self.row_stride = shard.oracle_row_stride
        self.count = shard.count

    def __getitem__(self, key) -> np.ndarray:
        if isinstance(key, slice):
            start, stop, step = key.indices(self.count)
            local_rows = np.arange(start, stop, step, dtype=np.int64)
            return self._rows(local_rows)
        local_row = int(key)
        if local_row < 0:
            local_row += self.count
        if local_row < 0 or local_row >= self.count:
            raise IndexError(local_row)
        return self._rows(np.asarray([local_row], dtype=np.int64))[0]

    def _rows(self, local_rows: np.ndarray) -> np.ndarray:
        coarse_rows = self.row_indices(local_rows)
        if coarse_rows.size and (
            int(coarse_rows.min()) < 0
            or int(coarse_rows.max()) >= self.probabilities.shape[0]
        ):
            raise IndexError("persisted one-minute oracle row is outside its UTC day")
        return self.probabilities[coarse_rows]

    def row_indices(self, local_rows: np.ndarray) -> np.ndarray:
        oracle_rows = self.row_offset + local_rows * self.row_stride
        return oracle_rows // 60 + (oracle_rows % 60 == 59)

    def compact_rows(self, start: int, stop: int) -> tuple[np.ndarray, np.ndarray]:
        """Return unique consecutive minute targets and one row index per example."""
        local_rows = np.arange(start, stop, dtype=np.int64)
        coarse_rows = self.row_indices(local_rows)
        unique_rows, inverse = np.unique(coarse_rows, return_inverse=True)
        if unique_rows.size and (
            int(unique_rows.min()) < 0
            or int(unique_rows.max()) >= self.probabilities.shape[0]
        ):
            raise IndexError("persisted one-minute oracle row is outside its UTC day")
        return self.probabilities[unique_rows], inverse


class FittedPolicyDataset(
    Dataset[tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]]
):
    def __init__(
        self,
        manifest: dict,
        root: Path,
        split: str,
        *,
        target: str = "rawOracleProbabilities",
        runtime_minute_oracles: dict[int, np.ndarray] | None = None,
        compact_minute_targets: bool = False,
        compact_minute_features: dict[
            Path, CompactMinuteFeatureComponent
        ] | None = None,
    ) -> None:
        if target not in (
            "rawOracleProbabilities",
            "minuteOracleProbabilities",
            "teacherParameters",
        ):
            raise ValueError("unknown MLP dataset target representation")
        self.split = split
        self.target = target
        self.compact_minute_targets = compact_minute_targets \
            and target == "minuteOracleProbabilities"
        self.feature_count = int(manifest["featureCount"])
        self.parameter_count = int(manifest["teacherParameterCount"])
        self.action_count = int(manifest["actionCount"])
        self.teacher_metric_count = int(manifest["teacherMetricCount"])
        self.manifest_sampling_interval_ms = int(manifest["samplingIntervalMs"])
        self.parts: list[
            tuple[Shard, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        ] = []
        self.offsets: list[int] = []
        self.temporal_runs: list[tuple[int, int]] = []
        self.storage_runs: list[tuple[int, int, str]] = []
        self.group_batches_by_storage = False
        self.coalesce_batches_across_storage = (
            target == "minuteOracleProbabilities"
            and self.manifest_sampling_interval_ms == 60_000
            and compact_minute_features is not None
        )
        total = 0
        run_start = 0
        previous_prediction_time: int | None = None
        minute_components: dict[Path, np.ndarray] = {}
        for value in manifest["shards"]:
            if value["split"] != split:
                continue
            shard = Shard(
                root,
                int(value["count"]),
                value["features"],
                int(value["featureRowOffset"]),
                int(value["featureRowStride"]),
                value["teacherParameters"],
                value["teacherMetrics"],
                value["rawOracleProbabilities"],
                value["minuteOracleProbabilities"],
                value["resolutionDivergence"],
                int(value["oracleRowOffset"]),
                int(value["oracleRowStride"]),
                value["baseTimeWeights"],
                value["timeWeights"],
                int(value["predictionTimeStart"]),
                int(value.get(
                    "oracleTargetTimeStart",
                    int(value["predictionTimeStart"])
                    - int(manifest.get("predictionDelayMs", 0)),
                )),
            )
            if previous_prediction_time is not None \
                    and shard.prediction_time_start != previous_prediction_time \
                    + self.manifest_sampling_interval_ms:
                self.temporal_runs.append((run_start, total))
                run_start = total
            feature_file = root / shard.features
            compact_feature = (
                compact_minute_features.get(feature_file)
                if compact_minute_features is not None
                else None
            )
            if compact_feature is None:
                features = component_rows(
                    feature_file, "<f2", self.feature_count,
                    shard.feature_row_offset, shard.feature_row_stride, shard.count,
                )
                feature_storage = shard.features
            else:
                relative_offset = (
                    shard.feature_row_offset - compact_feature.row_phase
                )
                if relative_offset < 0 \
                        or relative_offset % compact_feature.row_stride != 0 \
                        or shard.feature_row_stride % compact_feature.row_stride != 0:
                    raise ValueError(
                        f"minute feature rows are incompatible with "
                        f"{compact_feature.file}"
                    )
                compact_offset = (
                    relative_offset // compact_feature.row_stride
                )
                compact_stride = (
                    shard.feature_row_stride // compact_feature.row_stride
                )
                features = (
                    CompressedComponentRows(
                        compact_feature.file,
                        "<f2",
                        self.feature_count,
                        compact_offset,
                        compact_stride,
                        shard.count,
                        total_rows=compact_feature.total_rows,
                    )
                    if compact_feature.file.name.endswith(".zst")
                    else component_rows(
                        compact_feature.file,
                        "<f2",
                        self.feature_count,
                        compact_offset,
                        compact_stride,
                        shard.count,
                    )
                )
                feature_storage = str(compact_feature.file)
            teacher_parameters = component_rows(
                root / shard.teacher_parameters, "<f4", self.parameter_count,
                shard.oracle_row_offset, shard.oracle_row_stride, shard.count,
            )
            teacher_metrics = component_rows(
                root / shard.teacher_metrics, "<f4", self.teacher_metric_count,
                shard.oracle_row_offset, shard.oracle_row_stride, shard.count,
            )
            if target == "minuteOracleProbabilities":
                minute_storage = manifest.get("minuteOracleMap", {}).get(
                    "storage", "persisted-per-example"
                )
                if minute_storage \
                        == "computed-directly-at-training-startup":
                    if runtime_minute_oracles is None:
                        raise ValueError(
                            "runtime one-minute targets were not prepared"
                        )
                    direct_target_probabilities = RuntimeMinuteOracleRows(
                        runtime_minute_oracles, shard, self.action_count
                    )
                elif minute_storage == "persisted-per-minute-day":
                    minute_file = root / shard.minute_oracle_probabilities
                    minute_component = minute_components.get(minute_file)
                    if minute_component is None:
                        minute_component = (
                            load_compressed_component(
                                minute_file,
                                "<f4",
                                self.action_count,
                                1_441,
                            )
                            if minute_file.name.endswith(".zst")
                            else np.memmap(
                                minute_file,
                                mode="r",
                                dtype="<f4",
                                shape=(1_441, self.action_count),
                            )
                        )
                        minute_components[minute_file] = minute_component
                    direct_target_probabilities = PersistedMinuteOracleRows(
                        minute_component,
                        shard,
                        self.action_count,
                    )
                else:
                    minute_file = root / shard.minute_oracle_probabilities
                    if minute_file.name.endswith(".zst"):
                        direct_target_probabilities = CompressedComponentRows(
                            minute_file,
                            "<f4",
                            self.action_count,
                            0,
                            1,
                            shard.count,
                            total_rows=shard.count,
                        )
                    else:
                        direct_target_probabilities = np.memmap(
                            minute_file,
                            mode="r",
                            dtype="<f4",
                            shape=(shard.count, self.action_count),
                        )
            else:
                direct_target_probabilities = component_rows(
                    root / shard.raw_oracle_probabilities, "<f4", self.action_count,
                    shard.oracle_row_offset, shard.oracle_row_stride, shard.count,
                )
            time_weights = np.memmap(
                root / shard.time_weights, mode="r", dtype="<f4", shape=(shard.count,)
            )
            self.parts.append((
                shard,
                features,
                teacher_parameters,
                teacher_metrics,
                direct_target_probabilities,
                time_weights,
            ))
            total += shard.count
            self.offsets.append(total)
            self.storage_runs.append((total - shard.count, total, feature_storage))
            self.group_batches_by_storage = self.group_batches_by_storage \
                or shard.features.endswith(".zst") \
                or shard.raw_oracle_probabilities.endswith(".zst") \
                or shard.minute_oracle_probabilities.endswith(".zst")
            previous_prediction_time = shard.prediction_time_start \
                + (shard.count - 1) * self.manifest_sampling_interval_ms
        if total > run_start:
            self.temporal_runs.append((run_start, total))

    def __len__(self) -> int:
        return self.offsets[-1] if self.offsets else 0

    def __getitem__(
        self,
        index: int,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        part_index = int(np.searchsorted(self.offsets, index, side="right"))
        previous = self.offsets[part_index - 1] if part_index else 0
        shard, features, teacher_parameters, teacher_metrics, direct_target, time_weights = \
            self.parts[part_index]
        row = index - previous
        targets = teacher_parameters \
            if self.target == "teacherParameters" else direct_target
        return (
            torch.from_numpy(np.array(features[row], dtype=np.float32, copy=True)),
            torch.from_numpy(np.array(targets[row], dtype=np.float32, copy=True)),
            torch.tensor(float(time_weights[row]), dtype=torch.float32),
            torch.tensor(
                shard.prediction_time_start + row * int(self.manifest_sampling_interval_ms),
                dtype=torch.int64,
            ),
            torch.from_numpy(np.array(teacher_metrics[row], dtype=np.float32, copy=True)),
            torch.tensor([0], dtype=torch.int64)
            if self.compact_minute_targets
            else torch.empty(0, dtype=torch.int64),
        )

    def __getitems__(
        self, indices: list[int] | range
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Read one contiguous temporal batch with one copy per component."""
        if not indices:
            raise ValueError("cannot load an empty training batch")
        start = int(indices[0])
        stop = int(indices[-1]) + 1
        if stop - start != len(indices) or any(
            int(index) != start + offset for offset, index in enumerate(indices)
        ):
            raise ValueError("training batch indices must be contiguous")

        feature_chunks: list[np.ndarray] = []
        target_chunks: list[np.ndarray] = []
        metric_chunks: list[np.ndarray] = []
        weight_chunks: list[np.ndarray] = []
        time_chunks: list[np.ndarray] = []
        target_row_index_chunks: list[np.ndarray] = []
        compact_target_rows = 0
        compact_targets = self.compact_minute_targets
        cursor = start
        while cursor < stop:
            part_index = int(np.searchsorted(self.offsets, cursor, side="right"))
            previous = self.offsets[part_index - 1] if part_index else 0
            shard, features, teacher_parameters, teacher_metrics, direct_target, time_weights = \
                self.parts[part_index]
            local_start = cursor - previous
            count = min(stop - cursor, shard.count - local_start)
            local_stop = local_start + count
            feature_chunks.append(features[local_start:local_stop])
            targets = teacher_parameters \
                if self.target == "teacherParameters" else direct_target
            if compact_targets and isinstance(targets, PersistedMinuteOracleRows):
                compact_rows, row_indices = targets.compact_rows(
                    local_start,
                    local_stop,
                )
                target_chunks.append(compact_rows)
                target_row_index_chunks.append(
                    row_indices + compact_target_rows
                )
                compact_target_rows += compact_rows.shape[0]
            else:
                if compact_targets:
                    raise RuntimeError(
                        "compact minute targets require persisted per-minute rows"
                    )
                target_chunks.append(targets[local_start:local_stop])
            metric_chunks.append(teacher_metrics[local_start:local_stop])
            weight_chunks.append(time_weights[local_start:local_stop])
            time_chunks.append(
                shard.prediction_time_start
                + np.arange(local_start, local_stop, dtype=np.int64)
                * self.manifest_sampling_interval_ms
            )
            cursor += count

        if compact_targets:
            target_values = copy_chunks(target_chunks, np.float32)
            examples_per_minute = 60_000 // self.manifest_sampling_interval_ms
            maximum_rows = math.ceil(
                (len(indices) + examples_per_minute - 1)
                / examples_per_minute
            )
            if target_values.shape[0] > maximum_rows:
                raise RuntimeError("compact minute target batch exceeded its row bound")
            if target_values.shape[0] < maximum_rows:
                padding = np.repeat(
                    target_values[-1:],
                    maximum_rows - target_values.shape[0],
                    axis=0,
                )
                target_values = np.concatenate((target_values, padding))
            target_row_indices = np.concatenate(target_row_index_chunks).astype(
                np.int64,
                copy=False,
            )
        else:
            target_values = copy_chunks(target_chunks, np.float32)
            target_row_indices = np.empty(0, dtype=np.int64)

        return (
            torch.from_numpy(copy_chunks(feature_chunks, np.float32)),
            torch.from_numpy(target_values),
            torch.from_numpy(copy_chunks(weight_chunks, np.float32)),
            torch.from_numpy(copy_chunks(time_chunks, np.int64)),
            torch.from_numpy(copy_chunks(metric_chunks, np.float32)),
            torch.from_numpy(target_row_indices),
        )

    def mean_time_weight(self, block: range) -> float:
        """Mean persisted weight for a contiguous block without materializing examples."""
        if not block or block.step != 1:
            raise ValueError("time-weight scoring requires a non-empty contiguous block")
        cursor = block.start
        total = 0.0
        count = 0
        while cursor < block.stop:
            part_index = int(np.searchsorted(self.offsets, cursor, side="right"))
            previous = self.offsets[part_index - 1] if part_index else 0
            shard, _, _, _, _, time_weights = self.parts[part_index]
            local_start = cursor - previous
            take = min(block.stop - cursor, shard.count - local_start)
            total += float(np.asarray(
                time_weights[local_start:local_start + take],
                dtype=np.float32,
            ).sum(dtype=np.float64))
            count += take
            cursor += take
        return total / count

    def close(self) -> None:
        """Release memory-mapped component files before deleting/moving a dataset."""
        closed: set[int] = set()
        for part in self.parts:
            for component in part[1:]:
                values = (
                    component.probabilities
                    if isinstance(component, PersistedMinuteOracleRows)
                    else component
                )
                mapping = getattr(values, "_mmap", None)
                if mapping is not None and id(mapping) not in closed:
                    mapping.close()
                    closed.add(id(mapping))

    def __enter__(self) -> FittedPolicyDataset:
        return self

    def __exit__(self, _type, _value, _traceback) -> None:
        self.close()


def copy_chunks(chunks: list[np.ndarray], dtype: np.dtype) -> np.ndarray:
    if len(chunks) == 1:
        return np.array(chunks[0], dtype=dtype, copy=True)
    return np.concatenate(chunks).astype(dtype, copy=False)


def passthrough_batch(batch):
    return batch


def component_rows(
    file: Path,
    dtype: str,
    columns: int,
    row_offset: int,
    row_stride: int,
    count: int,
) -> np.ndarray:
    if file.name.endswith(".zst"):
        return CompressedComponentRows(
            file, dtype, columns, row_offset, row_stride, count
        )
    item_size = np.dtype(dtype).itemsize
    row_bytes = columns * item_size
    file_bytes = file.stat().st_size
    if row_bytes <= 0 or file_bytes % row_bytes != 0:
        raise ValueError(f"component file is not row aligned: {file}")
    component_count = file_bytes // row_bytes
    final_row = row_offset + max(0, count - 1) * row_stride
    if min(row_offset, row_stride, count) < 0 or row_stride < 1 or final_row >= component_count:
        raise ValueError(f"component row view is outside {file}")
    values = np.memmap(file, mode="r", dtype=dtype, shape=(component_count, columns))
    return values[row_offset:row_offset + count * row_stride:row_stride]


_COMPRESSED_COMPONENT_CACHE: OrderedDict[Path, np.ndarray] = OrderedDict()
_COMPRESSED_COMPONENT_CACHE_DAYS = max(
    1, int(os.environ.get("MLP_FEATURE_CACHE_DAYS", "2"))
)


@dataclass(frozen=True)
class CompactMinuteFeatureComponent:
    file: Path
    row_phase: int
    row_stride: int = 60
    total_rows: int = 1_440


class CompressedComponentRows:
    """Lazy row-addressable view over one lossless zstd component chunk."""

    def __init__(
        self,
        file: Path,
        dtype: str,
        columns: int,
        row_offset: int,
        row_stride: int,
        count: int,
        *,
        total_rows: int = 86_400,
    ) -> None:
        final_row = row_offset + max(0, count - 1) * row_stride
        if min(row_offset, row_stride, count) < 0 or row_stride < 1 \
                or total_rows < 1 or final_row >= total_rows:
            raise ValueError(f"compressed component row view is outside {file}")
        self.file = file
        self.dtype = dtype
        self.columns = columns
        self.row_offset = row_offset
        self.row_stride = row_stride
        self.count = count
        self.total_rows = total_rows

    def __getitem__(self, key) -> np.ndarray:
        values = load_compressed_component(
            self.file, self.dtype, self.columns, self.total_rows
        )
        rows = values[
            self.row_offset:
            self.row_offset + self.count * self.row_stride:
            self.row_stride
        ]
        return rows[key]


def load_compressed_component(
    file: Path,
    dtype: str,
    columns: int,
    total_rows: int,
) -> np.ndarray:
    cached = _COMPRESSED_COMPONENT_CACHE.pop(file, None)
    if cached is not None:
        _COMPRESSED_COMPONENT_CACHE[file] = cached
        return cached
    if zstandard is None:
        raise RuntimeError(
            "zstandard is required for compressed training components; "
            "run `npm run mlp:bootstrap`"
        )
    expected_bytes = total_rows * columns * np.dtype(dtype).itemsize
    decoded = zstandard.ZstdDecompressor().decompress(
        file.read_bytes(), max_output_size=expected_bytes
    )
    if len(decoded) != expected_bytes:
        raise ValueError(
            f"compressed component has {len(decoded)} decoded bytes, "
            f"expected {expected_bytes}: {file}"
        )
    values = np.frombuffer(decoded, dtype=dtype).reshape(total_rows, columns)
    _COMPRESSED_COMPONENT_CACHE[file] = values
    while len(_COMPRESSED_COMPONENT_CACHE) > _COMPRESSED_COMPONENT_CACHE_DAYS:
        _COMPRESSED_COMPONENT_CACHE.popitem(last=False)
    return values


def prepare_compact_minute_features(
    manifest: dict,
    dataset_root: Path,
    cache_root: Path,
) -> dict[Path, CompactMinuteFeatureComponent]:
    """Materialize exact minute-cadence views of compressed one-second features."""
    if int(manifest["samplingIntervalMs"]) != 60_000:
        return {}
    if zstandard is None:
        raise RuntimeError(
            "zstandard is required for compact minute feature components; "
            "run `npm run mlp:bootstrap`"
        )

    columns = int(manifest["featureCount"])
    schema_version = int(manifest["featureSchemaVersion"])
    source_phases: dict[Path, int] = {}
    for value in manifest["shards"]:
        source = dataset_root / value["features"]
        if not source.name.endswith(".zst"):
            continue
        row_stride = int(value["featureRowStride"])
        if row_stride != 60:
            raise ValueError(
                "one-minute feature compaction requires featureRowStride = 60"
            )
        phase = int(value["featureRowOffset"]) % row_stride
        previous = source_phases.setdefault(source, phase)
        if previous != phase:
            raise ValueError(
                f"compressed feature component uses multiple minute phases: {source}"
            )
    if not source_phases:
        return {}

    cache_root.mkdir(parents=True, exist_ok=True)
    expected_bytes = 1_440 * columns * np.dtype("<f2").itemsize
    source_bytes = 86_400 * columns * np.dtype("<f2").itemsize

    def cache_file(source: Path, phase: int) -> Path:
        return cache_root / (
            f"{source.name}.phase-{phase}.schema-{schema_version}."
            f"columns-{columns}.minute.f16.zst"
        )

    def prepare_one(
        source: Path,
        phase: int,
    ) -> tuple[Path, CompactMinuteFeatureComponent, bool, int]:
        target = cache_file(source, phase)
        source_stat = source.stat()
        try:
            target_stat = target.stat()
            if target_stat.st_size > 0 \
                    and target_stat.st_mtime_ns >= source_stat.st_mtime_ns:
                return (
                    source,
                    CompactMinuteFeatureComponent(target, phase),
                    True,
                    target_stat.st_size,
                )
        except FileNotFoundError:
            pass

        decoded = zstandard.ZstdDecompressor().decompress(
            source.read_bytes(),
            max_output_size=source_bytes,
        )
        if len(decoded) != source_bytes:
            raise ValueError(
                f"compressed feature component has {len(decoded)} decoded bytes, "
                f"expected {source_bytes}: {source}"
            )
        values = np.frombuffer(decoded, dtype="<f2").reshape(86_400, columns)
        compact = np.ascontiguousarray(values[phase::60])
        if compact.nbytes != expected_bytes:
            raise ValueError(
                f"minute feature component has {compact.nbytes} bytes, "
                f"expected {expected_bytes}: {source}"
            )
        compressed = zstandard.ZstdCompressor(level=3).compress(compact.tobytes())
        temporary = target.with_name(
            f"{target.name}.tmp-{os.getpid()}-{random.randrange(1 << 30)}"
        )
        try:
            temporary.write_bytes(compressed)
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        return (
            source,
            CompactMinuteFeatureComponent(target, phase),
            False,
            len(compressed),
        )

    worker_count = min(
        len(source_phases),
        max(1, int(os.environ.get("MLP_MINUTE_FEATURE_CACHE_WORKERS", "4"))),
    )
    emit({
        "event": "minute-feature-cache-start",
        "components": len(source_phases),
        "rowsPerComponent": 1_440,
        "sourceRowsPerComponent": 86_400,
        "columns": columns,
        "workers": worker_count,
        "cache": str(cache_root),
    })
    result: dict[Path, CompactMinuteFeatureComponent] = {}
    cached = 0
    compressed_bytes = 0
    with ThreadPoolExecutor(
        max_workers=worker_count,
        thread_name_prefix="minute-features",
    ) as executor:
        futures = {
            executor.submit(prepare_one, source, phase): source
            for source, phase in source_phases.items()
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            source, component, reused, size = future.result()
            result[source] = component
            cached += int(reused)
            compressed_bytes += size
            if completed == 1 or completed % 25 == 0 \
                    or completed == len(futures):
                emit({
                    "event": "minute-feature-cache-progress",
                    "completedComponents": completed,
                    "components": len(futures),
                    "reusedComponents": cached,
                })
    obsolete_bytes = 0
    obsolete_components = 0
    for obsolete in cache_root.glob("*.minute.f16"):
        if not obsolete.is_file():
            continue
        obsolete_bytes += obsolete.stat().st_size
        obsolete.unlink()
        obsolete_components += 1
    emit({
        "event": "minute-feature-cache-complete",
        "components": len(result),
        "reusedComponents": cached,
        "materializedComponents": len(result) - cached,
        "storage": "shared-compressed-minute-view-cache",
        "compressedMiB": compressed_bytes / (1024 * 1024),
        "decodedMiBPerEpoch": len(result) * expected_bytes / (1024 * 1024),
        "avoidedDecodedMiBPerEpoch": len(result)
        * (source_bytes - expected_bytes) / (1024 * 1024),
        "prunedObsoleteComponents": obsolete_components,
        "prunedObsoleteMiB": obsolete_bytes / (1024 * 1024),
        "cache": str(cache_root),
    })
    return result


class TemporalBlockBatchSampler:
    """Shuffle contiguous time blocks without inheriting component boundaries."""

    def __init__(
        self,
        dataset: FittedPolicyDataset,
        batch_size: int,
        shuffle: bool,
        sample_fraction: float = 1.0,
        weighted_sample: bool = False,
        seed: int = 1337,
    ) -> None:
        self.blocks: list[range] = []
        groups_by_storage: OrderedDict[str, list[range]] = OrderedDict()
        source_runs = dataset.storage_runs \
            if getattr(dataset, "group_batches_by_storage", False) \
            and not getattr(dataset, "coalesce_batches_across_storage", False) \
            else [
                (start, end, f"temporal-{index}")
                for index, (start, end) in enumerate(dataset.temporal_runs)
            ]
        for start, end, storage_key in source_runs:
            all_run_blocks = [
                range(block_start, min(end, block_start + batch_size))
                for block_start in range(start, end, batch_size)
            ]
            if sample_fraction >= 1.0:
                run_blocks = all_run_blocks
            elif weighted_sample:
                run_length = end - start
                keep_examples = max(1, round(run_length * sample_fraction))
                segment_count = math.ceil(keep_examples / batch_size)
                base_size, extra = divmod(keep_examples, segment_count)
                generator = random.Random(seed + start)
                run_blocks = []
                for segment in range(segment_count):
                    segment_size = base_size + (1 if segment < extra else 0)
                    stratum_start = start + segment * run_length // segment_count
                    stratum_end = start + (segment + 1) * run_length // segment_count
                    available = stratum_end - stratum_start - segment_size
                    candidates = [
                        stratum_start + available * candidate // 15
                        for candidate in range(16)
                    ] if available > 0 else [stratum_start]
                    scores = [
                        dataset.mean_time_weight(
                            range(candidate, candidate + segment_size)
                        )
                        for candidate in candidates
                    ]
                    threshold = generator.random() * sum(scores)
                    selected_start = candidates[-1]
                    for candidate, score in zip(candidates, scores, strict=True):
                        threshold -= score
                        if threshold <= 0:
                            selected_start = candidate
                            break
                    run_blocks.append(
                        range(selected_start, selected_start + segment_size)
                    )
            else:
                run_length = end - start
                keep_examples = max(1, round(run_length * sample_fraction))
                segment_count = math.ceil(keep_examples / batch_size)
                base_size, extra = divmod(keep_examples, segment_count)
                run_blocks = []
                for segment in range(segment_count):
                    segment_size = base_size + (1 if segment < extra else 0)
                    stratum_start = start + segment * run_length // segment_count
                    stratum_end = start + (segment + 1) * run_length // segment_count
                    block_start = stratum_start + (stratum_end - stratum_start - segment_size) // 2
                    run_blocks.append(range(block_start, block_start + segment_size))
            self.blocks.extend(run_blocks)
            groups_by_storage.setdefault(storage_key, []).extend(run_blocks)
        self.groups = list(groups_by_storage.values())
        self.shuffle = shuffle
        self.example_count = sum(len(block) for block in self.blocks)

    def __iter__(self):
        if not self.shuffle:
            yield from self.blocks
            return
        group_order = torch.randperm(len(self.groups)).tolist()
        for group_index in group_order:
            group = self.groups[group_index]
            block_order = torch.randperm(len(group)).tolist()
            for block_index in block_order:
                yield group[block_index]

    def __len__(self) -> int:
        return len(self.blocks)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the causal 16x1024 direct oracle-distribution MLP."
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--label", default="Direct oracle-distribution MLP")
    parser.add_argument("--plan", type=Path)
    parser.add_argument(
        "--target",
        choices=("rawOracleProbabilities", "minuteOracleProbabilities"),
        default="rawOracleProbabilities",
        help="Stored direct-distribution target to learn.",
    )
    parser.add_argument("--epochs", type=int, default=240)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--evaluation-batch-size", type=int, default=0)
    parser.add_argument("--validation-fraction", type=float, default=1.0)
    parser.add_argument("--training-fraction", type=float, default=1.0)
    parser.add_argument("--weighted-training-sample", action="store_true")
    parser.add_argument("--accumulate", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--states-per-example", type=int, default=31)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument(
        "--disable-patience",
        action="store_true",
        help="Disable plateau stopping; validation targets are the only automatic stop.",
    )
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--log-every-steps", type=int, default=25)
    parser.add_argument("--loss-weights-json", default="{}")
    parser.add_argument("--time-weighting-json", default="{}")
    parser.add_argument(
        "--selection-metric",
        choices=("loss", "klDivergence", "baseKlDivergence"),
        default="loss",
    )
    parser.add_argument("--target-validation-kl", type=float)
    parser.add_argument("--target-validation-base-kl", type=float)
    parser.add_argument("--target-validation-kl-stddev", type=float)
    parser.add_argument("--study-file", type=Path)
    parser.add_argument("--feature-statistics-cache", type=Path)
    parser.add_argument(
        "--reuse-feature-statistics-cache",
        action="store_true",
        help=(
            "Reuse an existing feature normalization cache when delayed pairing "
            "changes only the number of examples."
        ),
    )
    parser.add_argument("--finalize-file", type=Path)
    parser.add_argument("--initialize-from-checkpoint", type=Path)
    parser.add_argument("--inherited-best-epoch", type=int, default=-1)
    parser.add_argument("--evaluation-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--skip-baseline", action="store_true")
    return parser.parse_args()


def prepare_runtime_minute_oracles(
    manifest: dict,
    plan_file: Path,
) -> dict[int, np.ndarray]:
    """Compute each unique UTC target day once, before DataLoader workers fork."""
    target_days = sorted({
        int(shard["oracleTargetTimeStart"]) // 86_400_000 * 86_400_000
        for shard in manifest["shards"]
    })
    if not target_days:
        raise ValueError("runtime one-minute target set is empty")
    worker_count = min(
        len(target_days),
        max(1, int(os.environ.get("MLP_MINUTE_ORACLE_WORKERS", "4"))),
    )
    emit({
        "event": "runtime-minute-oracle-start",
        "days": len(target_days),
        "examples": sum(int(shard["count"]) for shard in manifest["shards"]),
        "storage": "memory-only",
        "backend": "deterministic-cpu",
        "workers": worker_count,
    })
    result: dict[int, np.ndarray] = {}
    day_groups = [
        target_days[worker::worker_count]
        for worker in range(worker_count)
    ]
    with ThreadPoolExecutor(
        max_workers=worker_count,
        thread_name_prefix="minute-oracle",
    ) as executor:
        futures = {
            executor.submit(
                load_runtime_minute_oracle_days,
                manifest,
                plan_file,
                days,
            ): worker
            for worker, days in enumerate(day_groups)
        }
        completed = 0
        for future in as_completed(futures):
            values = future.result()
            result.update(values)
            completed += len(values)
            emit({
                "event": "runtime-minute-oracle-progress",
                "daysCompleted": completed,
                "days": len(target_days),
                "worker": futures[future],
                "residentMiB": sum(
                    value.nbytes for value in result.values()
                ) / (1024 * 1024),
            })
    emit({
        "event": "runtime-minute-oracle-complete",
        "days": len(result),
        "residentMiB": sum(value.nbytes for value in result.values())
        / (1024 * 1024),
    })
    return result


def load_runtime_minute_oracle_days(
    manifest: dict,
    plan_file: Path,
    target_days: list[int],
) -> dict[int, np.ndarray]:
    repository = Path(__file__).resolve().parents[1]
    bundled_node = repository / (
        ".node-22/node.exe" if os.name == "nt" else ".node-22/bin/node"
    )
    node = bundled_node if bundled_node.is_file() else Path("node")
    process = subprocess.Popen(
        [
            str(node),
            str(repository / "node_modules/tsx/dist/cli.mjs"),
            str(repository / "scripts/stream-minute-oracle-targets.ts"),
            "--plan",
            str(plan_file.resolve()),
        ],
        cwd=repository,
        env=os.environ.copy(),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=None,
        bufsize=0,
    )
    if process.stdin is None or process.stdout is None:
        process.kill()
        raise RuntimeError("failed to open runtime minute-oracle provider pipes")
    result: dict[int, np.ndarray] = {}
    try:
        process.stdin.write(
            ("".join(f"{utc_date(day)}\n" for day in target_days)).encode()
        )
        process.stdin.close()
        for day in target_days:
            date = utc_date(day)
            rows, actions = struct.unpack("<II", read_exact(process.stdout, 8))
            if actions != int(manifest["actionCount"]) or rows < 1_441:
                raise RuntimeError(
                    f"runtime one-minute oracle shape for {date} is "
                    f"{rows}x{actions}, expected at least "
                    f"1441x{manifest['actionCount']}"
                )
            payload = read_exact(process.stdout, rows * actions * 4)
            probabilities = np.frombuffer(payload, dtype="<f4").reshape(
                rows, actions
            ).copy()
            if not np.isfinite(probabilities).all() \
                    or np.any(probabilities < 0) \
                    or not np.allclose(
                        probabilities.sum(axis=1), 1, rtol=1e-4, atol=1e-5
                    ):
                raise RuntimeError(
                    f"runtime one-minute oracle contains invalid probabilities for {date}"
                )
            result[day] = probabilities
        exit_code = process.wait()
        if exit_code != 0:
            raise RuntimeError(
                f"runtime one-minute oracle provider exited with code {exit_code}"
            )
    except BaseException:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        raise
    return result


def read_exact(stream, count: int) -> bytes:
    chunks: list[bytes] = []
    remaining = count
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            raise EOFError(
                f"runtime one-minute oracle provider ended with "
                f"{remaining}/{count} bytes unread"
            )
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def utc_date(time_ms: int) -> str:
    return datetime.fromtimestamp(time_ms / 1000, timezone.utc).strftime("%Y-%m-%d")


def main() -> None:
    args = parse_args()
    validate_args(args)
    set_determinism(args.seed)
    manifest = json.loads((args.dataset / "dataset.json").read_text())
    validate_dataset_manifest(manifest, args.dataset)
    sampling_interval_ms = int(manifest["samplingIntervalMs"])
    runtime_minute_oracles = None
    if args.target == "minuteOracleProbabilities" \
            and manifest.get("minuteOracleMap", {}).get("storage") \
            == "computed-directly-at-training-startup":
        if args.plan is None:
            raise ValueError(
                "--plan is required to compute runtime one-minute oracle targets"
            )
        runtime_minute_oracles = prepare_runtime_minute_oracles(
            manifest, args.plan
        )
    compact_training_targets = (
        args.target == "minuteOracleProbabilities"
        and manifest.get("minuteOracleMap", {}).get("storage")
        == "persisted-per-minute-day"
        and 60_000 % sampling_interval_ms == 0
    )
    minute_feature_cache_root = (
        args.feature_statistics_cache.parent / "minute-feature-components"
        if args.feature_statistics_cache is not None
        else args.dataset / ".training-cache" / "minute-feature-components"
    )
    compact_minute_features = (
        prepare_compact_minute_features(
            manifest,
            args.dataset,
            minute_feature_cache_root,
        )
        if args.target == "minuteOracleProbabilities"
        and sampling_interval_ms == 60_000
        else None
    )
    train = FittedPolicyDataset(
        manifest,
        args.dataset,
        "train",
        target=args.target,
        runtime_minute_oracles=runtime_minute_oracles,
        compact_minute_targets=compact_training_targets,
        compact_minute_features=compact_minute_features,
    )
    validation = FittedPolicyDataset(
        manifest,
        args.dataset,
        "validation",
        target=args.target,
        runtime_minute_oracles=runtime_minute_oracles,
        compact_minute_features=compact_minute_features,
    )
    test = FittedPolicyDataset(
        manifest,
        args.dataset,
        "test",
        target=args.target,
        runtime_minute_oracles=runtime_minute_oracles,
        compact_minute_features=compact_minute_features,
    )
    if min(len(train), len(validation), len(test)) == 0:
        raise RuntimeError("train, validation, and test datasets must all be non-empty")

    device = resolve_device(args.device)
    feature_mean, feature_std = cached_training_normalization(
        train,
        args.feature_statistics_cache,
        allow_count_mismatch=args.reuse_feature_statistics_cache,
    )
    model = ExposureMlp(feature_mean, feature_std, args.dropout).to(device)
    if args.initialize_from_checkpoint is not None:
        parent = torch.load(
            args.initialize_from_checkpoint,
            map_location=device,
            # Project checkpoints also contain NumPy RNG/optimizer metadata. PyTorch
            # 2.6's weights-only unpickler rejects that trusted local metadata before
            # we can select the model state below.
            weights_only=False,
        )
        if isinstance(parent, dict) and isinstance(parent.get("model"), dict):
            parent = parent["model"]
        if not isinstance(parent, dict):
            raise ValueError("warm-start checkpoint does not contain a model state")
        model.load_state_dict(parent)
        emit({
            "event": "training-warm-start",
            "checkpoint": str(args.initialize_from_checkpoint),
            "semantics": "model weights inherited; optimizer, scheduler, and patience reset",
        })
    execution_support = PolicySupport(**manifest["policySupport"])
    support = execution_support
    actions = torch.tensor(
        manifest["grid"],
        dtype=torch.float32,
        device=device,
    )
    current = deterministic_current_states(args.states_per_example, support, device, visible=True)
    transaction_transition = conditional_transaction_transition(
        actions,
        current,
        support,
    )
    loss_weights = parse_loss_weights(args.loss_weights_json)
    runtime_loss_weights = loss_weights_on_device(loss_weights, device)
    time_weighting = parse_time_weighting(args.time_weighting_json)
    training_contract = {
        "datasetPlanId": manifest["planId"],
        "componentStoreId": manifest["componentLayout"].get(
            "storeId", manifest["planId"]
        ),
        "datasetVersion": manifest["version"],
        "featureSchemaVersion": FEATURE_SCHEMA_VERSION,
        "inputFeatureCount": INPUT_FEATURE_COUNT,
        "outputRepresentation": "base-action-logits",
        "targetRepresentation": args.target,
        "outputActionCount": OUTPUT_ACTION_COUNT,
        "samplingIntervalMs": sampling_interval_ms,
        "predictionDelayMs": int(manifest["predictionDelayMs"]),
        "statesPerExample": args.states_per_example,
        "batchSize": args.batch_size,
        "evaluationBatchSize": args.evaluation_batch_size,
        "validationFraction": args.validation_fraction,
        "trainingFraction": args.training_fraction,
        "weightedTrainingSample": args.weighted_training_sample,
        "gradientAccumulation": args.accumulate,
        "epochs": args.epochs,
        "patience": None if args.disable_patience else args.patience,
        "earlyStopping": "target-only" if args.disable_patience else "patience",
        "learningRate": args.learning_rate,
        "weightDecay": args.weight_decay,
        "dropout": args.dropout,
        "seed": args.seed,
        "selectionMetric": args.selection_metric,
        "targetValidationKl": args.target_validation_kl,
        "targetValidationBaseKl": args.target_validation_base_kl,
        "targetValidationKlStdDev": args.target_validation_kl_stddev,
        "initializeFromCheckpoint": checkpoint_identity(
            args.initialize_from_checkpoint
        ),
        "evaluationOnly": args.evaluation_only,
        "skipBaseline": args.skip_baseline,
        "compile": args.compile,
        "lossWeights": asdict(loss_weights),
        "distributionObjective": "conditional-plus-action-only-ce-pmse-v1",
        "oracleObjective": "base-action-gaussian-time-correlation-mi-v1",
    }
    if manifest["exampleWeighting"]["timeWeighting"] != time_weighting_metadata(time_weighting):
        raise ValueError("training time-weighting configuration does not match stored example weights")
    for dataset in (train, validation, test):
        report_stored_example_weights(dataset)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
        fused=device.type == "cuda",
    )
    evaluation_batch_size = args.evaluation_batch_size or args.batch_size
    train_loader = loader(
        train,
        args,
        shuffle=True,
        batch_size=args.batch_size,
        sample_fraction=args.training_fraction,
        weighted_sample=args.weighted_training_sample,
    )
    validation_loader = loader(
        validation,
        args,
        shuffle=False,
        batch_size=evaluation_batch_size,
        sample_fraction=args.validation_fraction,
    )
    test_loader = loader(test, args, shuffle=False, batch_size=evaluation_batch_size)
    steps_per_epoch = math.ceil(len(train_loader) / args.accumulate)
    total_steps = max(1, steps_per_epoch * args.epochs)
    checkpoint_file = args.output / "checkpoint.pt"
    best_model_file = args.output / "best-model.pt"
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = (
        torch.load(checkpoint_file, map_location=device, weights_only=False)
        if args.resume and checkpoint_file.exists()
        else None
    )
    resume_contract_extended = False
    if checkpoint is not None:
        checkpoint_contract = checkpoint.get("trainingContract")
        if not isinstance(checkpoint_contract, dict):
            raise RuntimeError("checkpoint does not contain a training contract")
        checkpoint_schedule = checkpoint_contract.get("learningRateSchedule")
        if checkpoint_schedule is not None:
            training_contract["learningRateSchedule"] = checkpoint_schedule
        if checkpoint_contract != training_contract:
            if not resume_contract_is_monotonic_extension(
                checkpoint_contract,
                training_contract,
            ):
                raise RuntimeError(
                    "checkpoint does not match the current training contract"
                )
            resume_contract_extended = True
            scheduler_state = checkpoint.get("scheduler", {})
            optimizer_groups = checkpoint.get("optimizer", {}).get(
                "param_groups", []
            )
            if not optimizer_groups:
                raise RuntimeError("checkpoint optimizer does not contain parameter groups")
            schedule_start_step = int(
                scheduler_state.get(
                    "last_epoch",
                    checkpoint.get("globalStep", 0),
                )
            )
            start_multiplier = float(optimizer_groups[0]["lr"]) \
                / args.learning_rate
            training_contract["learningRateSchedule"] = {
                "mode": "cosine-continuation",
                "startStep": schedule_start_step,
                "startMultiplier": start_multiplier,
                "endMultiplier": min(0.05, start_multiplier),
            }
    schedule_contract = training_contract.get("learningRateSchedule")
    if isinstance(schedule_contract, dict) \
            and schedule_contract.get("mode") == "cosine-continuation":
        scheduler_multiplier = lambda step: continuation_learning_rate_multiplier(
            step,
            total_steps,
            int(schedule_contract["startStep"]),
            float(schedule_contract["startMultiplier"]),
            float(schedule_contract["endMultiplier"]),
        )
    else:
        scheduler_multiplier = lambda step: learning_rate_multiplier(
            step,
            total_steps,
        )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        scheduler_multiplier,
    )
    scaler = torch.amp.GradScaler("cuda", init_scale=256.0, enabled=device.type == "cuda")
    start_epoch = 0
    global_step = 0
    best_epoch = -1
    best_validation = math.inf
    best_validation_metrics: dict[str, float] = {}
    stale_epochs = 0
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if device.type == "cuda":
            # Optimizer state dictionaries include backend-selection fields.
            # A checkpoint written before fused AdamW was enabled would
            # otherwise silently replace the CUDA backend chosen above.
            for group in optimizer.param_groups:
                group["fused"] = True
                group["foreach"] = None
        saved_scheduler_step, resumed_scheduler_step = scheduler_resume_steps(
            checkpoint,
            steps_per_epoch,
        )
        if saved_scheduler_step == resumed_scheduler_step:
            scheduler.load_state_dict(checkpoint["scheduler"])
        else:
            scheduler.last_epoch = resumed_scheduler_step
            scheduler._step_count = resumed_scheduler_step + 1
            resumed_learning_rates = [
                base_lr * learning_rate_lambda(resumed_scheduler_step)
                for base_lr, learning_rate_lambda in zip(
                    scheduler.base_lrs,
                    scheduler.lr_lambdas,
                    strict=True,
                )
            ]
            for group, learning_rate in zip(
                optimizer.param_groups,
                resumed_learning_rates,
                strict=True,
            ):
                group["lr"] = learning_rate
            scheduler._last_lr = resumed_learning_rates
            emit({
                "event": "learning-rate-schedule-rebased",
                "reason": "optimizer batches per epoch changed",
                "checkpointStep": saved_scheduler_step,
                "resumedStep": resumed_scheduler_step,
                "optimizerStepsPerEpoch": steps_per_epoch,
                "learningRates": resumed_learning_rates,
            })
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = checkpoint["epoch"] + 1
        global_step = checkpoint.get("globalStep", start_epoch * steps_per_epoch)
        best_epoch = checkpoint["bestEpoch"]
        best_validation = checkpoint["bestValidation"]
        best_validation_metrics = checkpoint.get("bestValidationMetrics", {"loss": best_validation})
        stale_epochs = checkpoint.get("staleEpochs", 0)
        restore_rng(checkpoint["rng"])

    required_loss_terms = frozenset(
        name for name, value in asdict(loss_weights).items() if value != 0
    )

    def objective(
        *,
        include_diagnostics: bool,
        compact_targets: bool,
        precompute_transition: bool,
    ):
        def compute(features, targets, time_weights, times, target_row_indices):
            return direct_oracle_loss(
                model(features), targets, actions,
                current
                if precompute_transition
                else current.expand(features.shape[0], -1),
                support,
                runtime_loss_weights, time_weights, times,
                sampling_interval_ms,
                include_diagnostics=include_diagnostics,
                required_loss_terms=required_loss_terms,
                target_row_indices=target_row_indices
                if compact_targets else None,
                transaction_transition=transaction_transition
                if precompute_transition else None,
            )
        return compute

    training_objective = objective(
        include_diagnostics=False,
        compact_targets=compact_training_targets,
        precompute_transition=True,
    )
    training_diagnostic_objective = objective(
        include_diagnostics=True,
        compact_targets=compact_training_targets,
        precompute_transition=True,
    )
    evaluation_objective = objective(
        include_diagnostics=True,
        compact_targets=False,
        precompute_transition=False,
    )
    if args.compile:
        training_objective = torch.compile(
            training_objective,
            mode="reduce-overhead",
            fullgraph=False,
        )
        training_diagnostic_objective = torch.compile(
            training_diagnostic_objective,
            mode="reduce-overhead",
            fullgraph=False,
        )
        evaluation_objective = torch.compile(
            evaluation_objective,
            mode="reduce-overhead",
            fullgraph=False,
        )
    emit({
        "event": "training-start",
        "device": str(device),
        "parameters": parameter_count(model),
        "trainExamples": len(train),
        "selectedTrainingExamples": train_loader.batch_sampler.example_count,
        "validationExamples": len(validation),
        "screeningValidationExamples": validation_loader.batch_sampler.example_count,
        "batchSize": args.batch_size,
        "evaluationBatchSize": evaluation_batch_size,
        "validationFraction": args.validation_fraction,
        "compiledObjective": args.compile,
        "testExamples": len(test),
        "targetRepresentation": args.target,
        "compactMinuteFeatureComponents": (
            len(compact_minute_features)
            if compact_minute_features is not None
            else 0
        ),
        "lossWeights": asdict(loss_weights),
        "distributionObjective": training_contract["distributionObjective"],
        "oracleObjective": training_contract["oracleObjective"],
        "timeWeighting": time_weighting_metadata(time_weighting),
        "predictionDelayMs": int(manifest["predictionDelayMs"]),
        "distributionLossRange": [
            execution_support.visible_lower,
            execution_support.visible_upper,
        ],
        "currentStates": args.states_per_example,
        "startEpoch": start_epoch,
        "epochs": args.epochs,
        "evaluationOnly": args.evaluation_only,
        "targetValidation": {
            "klDivergence": args.target_validation_kl,
            "baseKlDivergence": args.target_validation_base_kl,
            "klDivergenceStdDev": args.target_validation_kl_stddev,
        },
        "resumeContractExtended": resume_contract_extended,
        "learningRateSchedule": training_contract.get("learningRateSchedule", {
            "mode": "cosine",
        }),
    })

    if not best_model_file.exists() and not args.skip_baseline:
        baseline = evaluate(
            model, validation_loader, actions, current, support,
            loss_weights, device, sampling_interval_ms,
            objective=evaluation_objective,
        )
        best_validation = baseline[args.selection_metric]
        best_validation_metrics = baseline
        best_epoch = args.inherited_best_epoch
        atomic_torch_save(model.state_dict(), best_model_file)
        emit({"event": "baseline", "validation": baseline})
    elif not best_model_file.exists():
        emit({
            "event": "baseline-skipped",
            "reason": "study variants select among trained epochs only",
        })

    stopped = False
    interrupted = False
    quality_target_reached = validation_target_reached(best_validation_metrics, args)
    patience_exhausted = (
        not args.disable_patience and stale_epochs >= args.patience
    )
    if quality_target_reached:
        emit({
            "event": "validation-target-reached",
            "epoch": best_epoch,
            "validation": best_validation_metrics,
            "targetKlDivergence": args.target_validation_kl,
            "targetBaseKlDivergence": args.target_validation_base_kl,
            "targetKlDivergenceStdDev": args.target_validation_kl_stddev,
        })
    if patience_exhausted:
        emit({
            "event": "training-patience-exhausted",
            "bestEpoch": best_epoch,
            "staleEpochs": stale_epochs,
            "patience": args.patience,
            "message": "Finalizing the saved best validated checkpoint.",
        })
    last_epoch = start_epoch - 1
    try:
        for epoch in range(start_epoch, args.epochs):
            if args.evaluation_only or quality_target_reached or patience_exhausted:
                break
            last_epoch = epoch
            started = time.monotonic()
            train_metrics, global_step, stopped = train_epoch(
                model,
                train_loader,
                optimizer,
                scheduler,
                scaler,
                actions,
                current,
                support,
                loss_weights,
                args,
                device,
                epoch,
                global_step,
                sampling_interval_ms,
                objective=training_objective,
                diagnostic_objective=training_diagnostic_objective,
            )
            validation_metrics = evaluate(
                model, validation_loader, actions, current, support,
                loss_weights, device, sampling_interval_ms,
                objective=evaluation_objective,
            )
            improved = validation_metrics[args.selection_metric] < best_validation - 1e-6
            if improved:
                best_validation = validation_metrics[args.selection_metric]
                best_validation_metrics = validation_metrics
                best_epoch = epoch
                stale_epochs = 0
                atomic_torch_save(model.state_dict(), best_model_file)
            else:
                stale_epochs += 1
            patience_exhausted = (
                not args.disable_patience and stale_epochs >= args.patience
            )
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
                training_contract,
            )
            quality_target_reached = validation_target_reached(validation_metrics, args)
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
                "bestValidationMetric": args.selection_metric,
                "staleEpochs": stale_epochs,
                "stopRequested": stopped,
                "validationTargetReached": quality_target_reached,
            })
            if quality_target_reached:
                emit({
                    "event": "validation-target-reached",
                    "epoch": epoch,
                    "validation": validation_metrics,
                    "targetKlDivergence": args.target_validation_kl,
                    "targetBaseKlDivergence": args.target_validation_base_kl,
                    "targetKlDivergenceStdDev": args.target_validation_kl_stddev,
                })
            if stopped or quality_target_reached or patience_exhausted:
                break
    except KeyboardInterrupt:
        interrupted = True
        emit({"event": "interrupt", "message": "Finalizing the best validated checkpoint."})
        validation_metrics = evaluate(
            model, validation_loader, actions, current, support,
            loss_weights, device, sampling_interval_ms,
            objective=evaluation_objective,
        )
        if validation_metrics[args.selection_metric] < best_validation - 1e-6:
            best_validation = validation_metrics[args.selection_metric]
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
            training_contract,
        )

    if not best_model_file.exists():
        raise RuntimeError("training finished without a validated checkpoint")
    model.load_state_dict(torch.load(best_model_file, map_location=device, weights_only=True))
    # Re-evaluate the materialized best state so persisted metrics always
    # describe the actual checkpoint under the current metric definitions,
    # including metrics added after a resumable checkpoint was written.
    best_validation_metrics = evaluate(
        model, validation_loader, actions, current, support,
        loss_weights, device, sampling_interval_ms,
        objective=evaluation_objective,
    )
    best_validation = best_validation_metrics[args.selection_metric]
    if args.study_file is not None:
        args.study_file.parent.mkdir(parents=True, exist_ok=True)
        study = {
            "modelId": args.model_id,
            "datasetPlanId": manifest["planId"],
            "componentStoreId": manifest["componentLayout"].get(
                "storeId", manifest["planId"]
            ),
            "predictionDelayMs": int(manifest["predictionDelayMs"]),
            "targetRepresentation": args.target,
            "selectionMetric": args.selection_metric,
            "policyMetricDefinitions": {
                "klDivergence":
                    "conditional KL(target || prediction) on the visible range",
                "klDivergenceVariance":
                    "weighted population variance of per-example visible-range conditional KL",
                "baseKlDivergence":
                    "KL between stored and predicted base-action distributions",
                "probabilityMse":
                    "weighted mean per-example probability MSE across all visible-range conditional rows",
                "probabilityMseVariance":
                    "weighted population variance of per-example visible-range conditional probability MSE",
            },
            "bestEpoch": best_epoch,
            "bestValidationScore": best_validation,
            "bestValidationMetrics": best_validation_metrics,
            "screeningBestValidationScore": best_validation,
            "screeningBestValidationMetrics": best_validation_metrics,
            "screeningValidationExamples": validation_loader.batch_sampler.example_count,
            "validationFraction": args.validation_fraction,
            "lossWeights": asdict(loss_weights),
            "distributionObjective": "conditional-plus-action-only-ce-pmse-v1",
            "oracleObjective": "base-action-gaussian-time-correlation-mi-v1",
            "trainExamples": len(train),
            "validationExamples": len(validation),
            "epochs": args.epochs,
            "patience": None if args.disable_patience else args.patience,
            "earlyStopping": (
                "target-only" if args.disable_patience else "patience"
            ),
            "seed": args.seed,
            "device": str(device),
            "finalizedEarly":
                stopped or interrupted or quality_target_reached
                or patience_exhausted or args.evaluation_only,
            "validationTargetReached": quality_target_reached,
        }
        atomic_json(study, args.study_file)
        emit({"event": "training-study-complete", **study, "studyFile": str(args.study_file)})
        return
    test_metrics = evaluate(
        model, test_loader, actions, current, support,
        loss_weights, device, sampling_interval_ms,
        objective=evaluation_objective,
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
        stopped or interrupted or quality_target_reached
        or patience_exhausted or args.evaluation_only,
    )
    emit({
        "event": "training-complete",
        "test": test_metrics,
        "bestValidation": best_validation_metrics,
        "bestEpoch": best_epoch,
        "artifact": str(args.output / "model.onnx"),
        "finalizedEarly":
            stopped or interrupted or quality_target_reached
            or patience_exhausted or args.evaluation_only,
        "validationTargetReached": quality_target_reached,
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
    loss_weights,
    args,
    device,
    epoch,
    global_step,
    sampling_interval_ms,
    objective=None,
    diagnostic_objective=None,
) -> tuple[dict[str, float], int, bool]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    totals = {
        name: torch.zeros((), device=device)
        for name in TRAIN_METRIC_NAMES
    }
    metric_counts = {
        name: torch.zeros((), device=device)
        for name in TRAIN_METRIC_NAMES
    }
    total_examples = 0
    time_weight_sum = torch.zeros((), device=device)
    time_weight_square_sum = torch.zeros((), device=device)
    kl_weight_sum = torch.zeros((), device=device)
    kl_mean = torch.zeros((), device=device)
    kl_centered_square_sum = torch.zeros((), device=device)
    probability_mse_weight_sum = torch.zeros((), device=device)
    probability_mse_mean = torch.zeros((), device=device)
    probability_mse_centered_square_sum = torch.zeros((), device=device)
    started = time.monotonic()
    stopped = False
    for batch_step, (
        features,
        targets,
        time_weights,
        times,
        _,
        target_row_indices,
    ) in enumerate(
        device_batches(data, device)
    ):
        should_step = (batch_step + 1) % args.accumulate == 0 \
            or batch_step + 1 == len(data)
        next_global_step = global_step + int(should_step)
        will_log = should_step and (
            next_global_step == 1
            or next_global_step % args.log_every_steps == 0
        )
        include_diagnostics = batch_step == 0 or will_log
        selected_objective = (
            diagnostic_objective
            if include_diagnostics and diagnostic_objective is not None
            else objective
        )
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            batch_metrics = selected_objective(
                features,
                targets,
                time_weights,
                times,
                target_row_indices,
            ) if selected_objective else (
                direct_oracle_loss(
                    model(features), targets, actions,
                    current.expand(features.shape[0], -1), support,
                    loss_weights, time_weights, times, sampling_interval_ms
                )
            )
            loss = batch_metrics["loss"] / args.accumulate
        if device.type == "cuda":
            torch._assert_async(
                torch.isfinite(loss),
                f"non-finite training loss at epoch {epoch}, batch {batch_step}",
            )
        elif not bool(torch.isfinite(loss)):
            raise RuntimeError(f"non-finite training loss at epoch {epoch}, batch {batch_step}")
        scaler.scale(loss).backward()
        if should_step:
            scaler.unscale_(optimizer)
            gradient_norm = clip_grad_norm_(
                model.parameters(),
                1.0,
                foreach=device.type == "cuda",
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            # Global steps already count attempted updates. Advancing the
            # step-based schedule here avoids two GradScaler.get_scale() calls,
            # each of which synchronizes the CPU with CUDA. Fused AdamW consumes
            # found_inf on-device and still skips an overflowing parameter update.
            scheduler.step()
            global_step += 1
            if will_log:
                emit({
                    "event": "train-step",
                    "epoch": epoch,
                    "epochs": args.epochs,
                    "batch": batch_step + 1,
                    "batches": len(data),
                    "globalStep": global_step,
                    "learningRate": optimizer.param_groups[0]["lr"],
                    "gradientNorm": float(gradient_norm),
                    "examplesPerSecond": round(total_examples / max(time.monotonic() - started, 1e-6), 1),
                    "gpuMemoryMiB": round(torch.cuda.max_memory_allocated() / 1_048_576, 1)
                    if device.type == "cuda" else 0,
                    "latest": detached_metrics(batch_metrics),
                })
            if args.finalize_file and args.finalize_file.exists():
                stopped = True
        count = features.shape[0]
        total_examples += count
        time_weight_sum += batch_metrics["timeWeightSum"].detach()
        time_weight_square_sum += batch_metrics["timeWeightSquareSum"].detach()
        if "klWeightSum" in batch_metrics:
            kl_weight_sum, kl_mean, kl_centered_square_sum = merge_weighted_moments(
                kl_weight_sum,
                kl_mean,
                kl_centered_square_sum,
                batch_metrics["klWeightSum"].detach(),
                batch_metrics["klDivergence"].detach(),
                batch_metrics["klCenteredSquareSum"].detach(),
            )
        if "probabilityMseWeightSum" in batch_metrics:
            probability_mse_weight_sum, probability_mse_mean, \
                probability_mse_centered_square_sum = merge_weighted_moments(
                    probability_mse_weight_sum,
                    probability_mse_mean,
                    probability_mse_centered_square_sum,
                    batch_metrics["probabilityMseWeightSum"].detach(),
                    batch_metrics["probabilityMse"].detach(),
                    batch_metrics["probabilityMseCenteredSquareSum"].detach(),
                )
        information_count = batch_metrics["informationExampleCount"].detach()
        for name in TRAIN_METRIC_NAMES:
            if name in KL_MOMENT_METRIC_NAMES \
                    or name in PROBABILITY_MSE_MOMENT_METRIC_NAMES:
                continue
            if name not in batch_metrics:
                continue
            metric_count = information_count if name in TIME_BLOCK_METRIC_NAMES else count
            totals[name] += batch_metrics[name].detach() * metric_count
            metric_counts[name] += metric_count
        if stopped:
            break
    result = {
        name: float(value) / max(1.0, float(metric_counts[name]))
        for name, value in totals.items()
    }
    result["timeWeightEffectiveSampleRatio"] = float(
        time_weight_sum * time_weight_sum
        / max(1e-12, total_examples * time_weight_square_sum)
    )
    result["klDivergence"] = float(kl_mean)
    result["klDivergenceVariance"] = float(weighted_variance(
        kl_centered_square_sum,
        kl_weight_sum,
    ))
    result["klDivergenceStdDev"] = float(weighted_standard_deviation(
        kl_centered_square_sum,
        kl_weight_sum,
    ))
    result["probabilityMse"] = float(probability_mse_mean)
    result["probabilityMseVariance"] = float(weighted_variance(
        probability_mse_centered_square_sum,
        probability_mse_weight_sum,
    ))
    result["probabilityMseStdDev"] = float(weighted_standard_deviation(
        probability_mse_centered_square_sum,
        probability_mse_weight_sum,
    ))
    return result, global_step, stopped


@torch.inference_mode()
def evaluate(model, data, actions, current, support,
             loss_weights, device, sampling_interval_ms,
             objective=None) -> dict[str, float]:
    model.eval()
    totals = {name: torch.zeros((), device=device) for name in METRIC_NAMES}
    total_examples = 0
    time_weight_sum = torch.zeros((), device=device)
    time_weight_square_sum = torch.zeros((), device=device)
    kl_weight_sum = torch.zeros((), device=device)
    kl_mean = torch.zeros((), device=device)
    kl_centered_square_sum = torch.zeros((), device=device)
    probability_mse_weight_sum = torch.zeros((), device=device)
    probability_mse_mean = torch.zeros((), device=device)
    probability_mse_centered_square_sum = torch.zeros((), device=device)
    information_example_count = torch.zeros((), device=device)
    for (
        features,
        targets,
        time_weights,
        times,
        _,
        target_row_indices,
    ) in device_batches(data, device):
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            batch_metrics = objective(
                features,
                targets,
                time_weights,
                times,
                target_row_indices,
            ) if objective else (
                direct_oracle_loss(
                    model(features), targets, actions,
                    current.expand(features.shape[0], -1), support,
                    loss_weights, time_weights, times, sampling_interval_ms
                )
            )
        count = features.shape[0]
        total_examples += count
        time_weight_sum += batch_metrics["timeWeightSum"]
        time_weight_square_sum += batch_metrics["timeWeightSquareSum"]
        kl_weight_sum, kl_mean, kl_centered_square_sum = merge_weighted_moments(
            kl_weight_sum,
            kl_mean,
            kl_centered_square_sum,
            batch_metrics["klWeightSum"],
            batch_metrics["klDivergence"],
            batch_metrics["klCenteredSquareSum"],
        )
        probability_mse_weight_sum, probability_mse_mean, \
            probability_mse_centered_square_sum = merge_weighted_moments(
                probability_mse_weight_sum,
                probability_mse_mean,
                probability_mse_centered_square_sum,
                batch_metrics["probabilityMseWeightSum"],
                batch_metrics["probabilityMse"],
                batch_metrics["probabilityMseCenteredSquareSum"],
            )
        information_count = batch_metrics["informationExampleCount"]
        information_example_count += information_count
        for name in METRIC_NAMES:
            if name in KL_MOMENT_METRIC_NAMES \
                    or name in PROBABILITY_MSE_MOMENT_METRIC_NAMES:
                continue
            metric_count = information_count if name in TIME_BLOCK_METRIC_NAMES else count
            totals[name] += batch_metrics[name] * metric_count
    information_count_value = max(1.0, float(information_example_count))
    result = {
        name: float(value) / (
            information_count_value if name in TIME_BLOCK_METRIC_NAMES else max(1, total_examples)
        ) for name, value in totals.items()
    }
    result["timeWeightEffectiveSampleRatio"] = float(
        time_weight_sum * time_weight_sum
        / max(1e-12, total_examples * time_weight_square_sum)
    )
    result["klDivergence"] = float(kl_mean)
    result["klDivergenceVariance"] = float(weighted_variance(
        kl_centered_square_sum,
        kl_weight_sum,
    ))
    result["klDivergenceStdDev"] = float(weighted_standard_deviation(
        kl_centered_square_sum,
        kl_weight_sum,
    ))
    result["probabilityMse"] = float(probability_mse_mean)
    result["probabilityMseVariance"] = float(weighted_variance(
        probability_mse_centered_square_sum,
        probability_mse_weight_sum,
    ))
    result["probabilityMseStdDev"] = float(weighted_standard_deviation(
        probability_mse_centered_square_sum,
        probability_mse_weight_sum,
    ))
    return result


def loader(
    dataset: FittedPolicyDataset,
    args: argparse.Namespace,
    shuffle: bool,
    *,
    batch_size: int,
    sample_fraction: float = 1.0,
    weighted_sample: bool = False,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_sampler=TemporalBlockBatchSampler(
            dataset,
            batch_size,
            shuffle,
            sample_fraction=sample_fraction,
            weighted_sample=weighted_sample,
            seed=args.seed,
        ),
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.workers > 0,
        prefetch_factor=4 if args.workers > 0 else None,
        collate_fn=passthrough_batch,
    )


def device_batches(data: DataLoader, device: torch.device):
    """Overlap pinned host-to-device copies with the preceding CUDA batch."""
    if device.type != "cuda":
        yield from data
        return

    copy_stream = torch.cuda.Stream(device=device)
    iterator = iter(data)

    def copy(batch):
        (
            features,
            targets,
            time_weights,
            times,
            diagnostics,
            target_row_indices,
        ) = batch
        with torch.cuda.stream(copy_stream):
            return (
                features.to(device, non_blocking=True),
                targets.to(device, non_blocking=True),
                time_weights.to(device, non_blocking=True),
                times.to(device, non_blocking=True),
                diagnostics,
                target_row_indices.to(device, non_blocking=True),
            )

    try:
        pending = copy(next(iterator))
    except StopIteration:
        return

    while True:
        torch.cuda.current_stream(device).wait_stream(copy_stream)
        batch = pending
        for value in (*batch[:4], batch[5]):
            value.record_stream(torch.cuda.current_stream(device))
        try:
            pending = copy(next(iterator))
        except StopIteration:
            pending = None
        yield batch
        if pending is None:
            break


def report_stored_example_weights(dataset: FittedPolicyDataset) -> None:
    """Validate and summarize persisted whole-example weights without changing them."""
    total = 0.0
    squared = 0.0
    maximum = 0.0
    count = 0
    for shard, _, _, _, _, time_weights in dataset.parts:
        values = np.asarray(time_weights, dtype=np.float32)
        if values.shape != (shard.count,) or not np.isfinite(values).all() \
                or np.any(values <= 0):
            raise RuntimeError(f"{dataset.split} contains invalid stored example weights")
        total += float(values.sum(dtype=np.float64))
        squared += float(np.square(values, dtype=np.float64).sum(dtype=np.float64))
        maximum = max(maximum, float(values.max(initial=0)))
        count += shard.count
    effective_ratio = total * total / max(1e-12, len(dataset) * squared)
    emit({
        "event": "time-weighting-ready",
        "split": dataset.split,
        "examples": len(dataset),
        "source": "dataset",
        "normalized": False,
        "meanWeight": total / max(1, count),
        "maximumWeight": maximum,
        "effectiveSampleRatio": effective_ratio,
    })


def training_normalization(dataset: FittedPolicyDataset) -> tuple[Tensor, Tensor]:
    total = 0
    feature_sum = np.zeros(INPUT_FEATURE_COUNT, dtype=np.float64)
    square_sum = np.zeros(INPUT_FEATURE_COUNT, dtype=np.float64)
    for shard, features, _, _, _, _ in dataset.parts:
        for start in range(0, shard.count, 8192):
            block = np.asarray(features[start:start + 8192], dtype=np.float32)
            feature_sum += block.sum(axis=0, dtype=np.float64)
            square_sum += np.square(block, dtype=np.float64).sum(axis=0)
            total += block.shape[0]
    mean = feature_sum / total
    variance = np.maximum(1e-12, square_sum / total - mean * mean)
    return torch.from_numpy(mean.astype(np.float32)), torch.from_numpy(np.sqrt(variance).astype(np.float32))


def cached_training_normalization(
    dataset: FittedPolicyDataset,
    cache_file: Path | None,
    *,
    allow_count_mismatch: bool = False,
) -> tuple[Tensor, Tensor]:
    if cache_file is not None and cache_file.exists():
        with np.load(cache_file, allow_pickle=False) as cache:
            mean = cache["mean"]
            std = cache["std"]
            count = int(cache["count"])
        if (count != len(dataset) and not allow_count_mismatch) \
                or mean.shape != (INPUT_FEATURE_COUNT,) \
                or std.shape != (INPUT_FEATURE_COUNT,) \
                or not np.isfinite(mean).all() or not np.isfinite(std).all() \
                or np.any(std <= 0):
            raise ValueError(f"invalid cached feature statistics: {cache_file}")
        emit({"event": "training-statistics-cache", "kind": "features", "hit": True,
              "file": str(cache_file), "examples": len(dataset),
              "cachedExamples": count,
              "countMismatchAccepted": count != len(dataset)})
        return torch.from_numpy(mean.astype(np.float32)), torch.from_numpy(std.astype(np.float32))
    mean, std = training_normalization(dataset)
    if cache_file is not None:
        atomic_numpy_archive(
            cache_file,
            mean=mean.numpy(),
            std=std.numpy(),
            count=np.asarray(len(dataset), dtype=np.int64),
        )
    emit({"event": "training-statistics-cache", "kind": "features", "hit": False,
          "file": str(cache_file) if cache_file is not None else None,
          "examples": len(dataset)})
    return mean, std


def training_parameter_scale(dataset: FittedPolicyDataset) -> Tensor:
    total = 0
    value_sum = np.zeros(TEACHER_PARAMETER_COUNT, dtype=np.float64)
    square_sum = np.zeros(TEACHER_PARAMETER_COUNT, dtype=np.float64)
    for shard, _, targets, _, _, _ in dataset.parts:
        for start in range(0, shard.count, 8192):
            block = np.asarray(targets[start:start + 8192], dtype=np.float32)
            value_sum += block.sum(axis=0, dtype=np.float64)
            square_sum += np.square(block, dtype=np.float64).sum(axis=0)
            total += block.shape[0]
    mean = value_sum / total
    minimum_variance = np.full(TEACHER_PARAMETER_COUNT, 1e-4, dtype=np.float64)
    minimum_variance[6:8] = 0.25 ** 2
    variance = np.maximum(minimum_variance, square_sum / total - mean * mean)
    return torch.from_numpy(np.sqrt(variance).astype(np.float32))


def cached_training_parameter_scale(
    dataset: FittedPolicyDataset, cache_file: Path | None
) -> Tensor:
    if cache_file is not None and cache_file.exists():
        with np.load(cache_file, allow_pickle=False) as cache:
            scale = cache["scale"]
            count = int(cache["count"])
        if count != len(dataset) or scale.shape != (TEACHER_PARAMETER_COUNT,) \
                or not np.isfinite(scale).all() or np.any(scale <= 0):
            raise ValueError(f"invalid cached target statistics: {cache_file}")
        emit({"event": "training-statistics-cache", "kind": "targets", "hit": True,
              "file": str(cache_file), "examples": count})
        return torch.from_numpy(scale.astype(np.float32))
    scale = training_parameter_scale(dataset)
    if cache_file is not None:
        atomic_numpy_archive(
            cache_file,
            scale=scale.numpy(),
            count=np.asarray(len(dataset), dtype=np.int64),
        )
    emit({"event": "training-statistics-cache", "kind": "targets", "hit": False,
          "file": str(cache_file) if cache_file is not None else None,
          "examples": len(dataset)})
    return scale


def teacher_fit_summary(manifest: dict, root: Path) -> dict[str, float]:
    names = manifest["teacherMetricNames"]
    total = np.zeros(len(names), dtype=np.float64)
    count = 0
    for value in manifest["shards"]:
        row_count = int(value["count"])
        metrics = component_rows(
            root / value["teacherMetrics"], "<f4", len(names),
            int(value["oracleRowOffset"]), int(value["oracleRowStride"]), row_count,
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


def parse_loss_weights(value: str) -> DirectLossWeights:
    parsed = json.loads(value)
    if "parameterMse" in parsed:
        raise ValueError(
            "parameterMse is not part of the direct-distribution objective"
        )
    return DirectLossWeights(
        cross_entropy=float(parsed.get("crossEntropy", 1)),
        probability_mse=float(parsed.get("probabilityMse", 0.1)),
        action_cross_entropy=float(parsed.get("actionCrossEntropy", 0)),
        action_probability_mse=float(parsed.get("actionProbabilityMse", 0)),
        excess_entropy=float(parsed.get("excessEntropy", 0)),
        oracle_mutual_information=float(parsed.get("oracleMutualInformation", 1)),
    )


def loss_weights_on_device(
    weights: DirectLossWeights,
    device: torch.device,
) -> DirectLossWeights:
    return DirectLossWeights(**{
        field: torch.tensor(value, dtype=torch.float32, device=device)
        for field, value in asdict(weights).items()
    })


def parse_time_weighting(value: str) -> TimeWeighting:
    parsed = json.loads(value)
    if parsed.get("mode", "distanceImbalance") != "distanceImbalance":
        raise ValueError("time weighting mode must be distanceImbalance")
    if parsed.get("stateAggregation", "globalDistanceRatio") != "globalDistanceRatio":
        raise ValueError("distance imbalance state aggregation must be globalDistanceRatio")
    weighting = TimeWeighting(
        distance_epsilon=float(parsed.get("distanceEpsilon", 1e-6)),
        minimum_weight=float(parsed.get("minimumWeight", 1e-6)),
        minimum_advice_magnitude=float(parsed.get("minimumAdviceMagnitude", 0.25)),
        memory_half_life_steps=float(parsed.get("memoryHalfLifeSteps", 15)),
        growth_per_prior_advice=float(parsed.get("growthPerPriorAdvice", 0.25)),
        maximum_multiplier=float(parsed.get("maximumMultiplier", 4)),
        reset_after_gap_steps=float(parsed.get("resetAfterGapSteps", 60)),
        resolution_divergence_multiplier=float(
            parsed.get("resolutionDivergenceMultiplier", 0)
        ),
    )
    validate_time_weighting(weighting)
    return weighting


def time_weighting_metadata(weighting: TimeWeighting) -> dict[str, object]:
    return {
        "mode": "distanceImbalance",
        "distanceEpsilon": weighting.distance_epsilon,
        "minimumWeight": weighting.minimum_weight,
        "stateAggregation": "globalDistanceRatio",
        "minimumAdviceMagnitude": weighting.minimum_advice_magnitude,
        "memoryHalfLifeSteps": weighting.memory_half_life_steps,
        "growthPerPriorAdvice": weighting.growth_per_prior_advice,
        "maximumMultiplier": weighting.maximum_multiplier,
        "resetAfterGapSteps": weighting.reset_after_gap_steps,
        "resolutionDivergenceMultiplier":
            weighting.resolution_divergence_multiplier,
    }


def learning_rate_multiplier(step: int, total: int) -> float:
    warmup = max(1, int(total * 0.05))
    if step < warmup:
        return (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def continuation_learning_rate_multiplier(
    step: int,
    total: int,
    start_step: int,
    start_multiplier: float,
    end_multiplier: float = 0.05,
) -> float:
    """Continue a completed schedule without raising its saved learning rate."""
    if total < 1 or start_step < 0 \
            or start_multiplier <= 0 or end_multiplier <= 0 \
            or end_multiplier > start_multiplier:
        raise ValueError("invalid continuation learning-rate schedule")
    if step <= start_step:
        return start_multiplier
    progress = (step - start_step) / max(1, total - start_step)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return end_multiplier + (start_multiplier - end_multiplier) * cosine


def scheduler_resume_steps(
    checkpoint: dict,
    optimizer_steps_per_epoch: int,
) -> tuple[int, int]:
    """Translate an old loader-sized schedule onto the current epoch geometry."""
    saved_step = int(
        checkpoint.get("scheduler", {}).get(
            "last_epoch",
            checkpoint.get("globalStep", 0),
        )
    )
    checkpoint_epoch = int(checkpoint["epoch"])
    current_epoch_start = checkpoint_epoch * optimizer_steps_per_epoch
    current_epoch_end = current_epoch_start + optimizer_steps_per_epoch
    if current_epoch_start <= saved_step <= current_epoch_end:
        return saved_step, saved_step
    return saved_step, current_epoch_end


def resume_contract_is_monotonic_extension(
    checkpoint_contract: dict,
    requested_contract: dict,
) -> bool:
    """Allow only a larger epoch or patience horizon for an exact run contract."""
    previous = dict(checkpoint_contract)
    requested = dict(requested_contract)
    previous_epochs = previous.pop("epochs", None)
    requested_epochs = requested.pop("epochs", None)
    previous_patience = previous.pop("patience", None)
    requested_patience = requested.pop("patience", None)
    previous.pop("learningRateSchedule", None)
    requested.pop("learningRateSchedule", None)
    if previous != requested \
            or not all(isinstance(value, int) for value in (
                previous_epochs,
                requested_epochs,
            )) \
            or not (
                previous_patience is None and requested_patience is None
                or isinstance(previous_patience, int)
                and isinstance(requested_patience, int)
            ):
        return False
    patience_extended = (
        previous_patience is not None
        and requested_patience is not None
        and requested_patience > previous_patience
    )
    patience_compatible = (
        previous_patience is None and requested_patience is None
        or requested_patience >= previous_patience
    )
    return (
        requested_epochs >= previous_epochs
        and patience_compatible
        and (
            requested_epochs > previous_epochs
            or patience_extended
        )
    )


def save_checkpoint(file, epoch, global_step, best_epoch, best_validation,
                    best_validation_metrics, stale_epochs, model, optimizer, scheduler,
                    scaler, training_contract) -> None:
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
        "trainingContract": training_contract,
    }, file)


def export_artifact(model, args, dataset_manifest, train_count, validation_count, test_count,
                    best_epoch, best_validation_metrics, test_metrics, teacher_metrics,
                    loss_weights, time_weighting, device, finalized_early) -> None:
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
        output_names=["action_logits"],
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
        "featureSchemaVersion": FEATURE_SCHEMA_VERSION,
        "inputFeatureCount": INPUT_FEATURE_COUNT,
        "outputRepresentation": "base-action-logits",
        "outputActionCount": OUTPUT_ACTION_COUNT,
        "actionGrid": dataset_manifest["grid"],
        "hiddenLayerCount": HIDDEN_LAYER_COUNT,
        "hiddenWidth": HIDDEN_WIDTH,
        "modelFile": "model.onnx",
        **({} if args.evaluation_only else {"checkpointFile": "checkpoint.pt"}),
        "predictionDelayMs": int(dataset_manifest["predictionDelayMs"]),
        "verificationFixture": {
            "batchSize": verification_batch_size,
            "inputFile": "verification-input.f32",
            "outputFile": "verification-output.f32",
        },
        "policySupport": dataset_manifest["policySupport"],
        "training": {
            "datasetPlanId": dataset_manifest["planId"],
            "targetRepresentation": args.target,
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
            "distributionObjective": "conditional-plus-action-only-ce-pmse-v1",
            "oracleObjective": "base-action-gaussian-time-correlation-mi-v1",
            "selectionMetric": args.selection_metric,
            "patience": None if args.disable_patience else args.patience,
            "earlyStopping": (
                "target-only" if args.disable_patience else "patience"
            ),
            "initializeFromCheckpoint": (
                str(args.initialize_from_checkpoint)
                if args.initialize_from_checkpoint is not None
                else None
            ),
            "evaluationOnly": args.evaluation_only,
            "targetValidation": {
                "klDivergence": args.target_validation_kl,
                "baseKlDivergence": args.target_validation_base_kl,
                "klDivergenceStdDev": args.target_validation_kl_stddev,
            },
            "policyMetricDefinitions": {
                "klDivergence":
                    "conditional KL(target || prediction) on the visible range",
                "klDivergenceVariance":
                    "weighted population variance of per-example visible-range conditional KL",
                "baseKlDivergence":
                    "KL between stored and predicted base-action distributions",
                "probabilityMse":
                    "weighted mean per-example probability MSE on the visible-range base-action distribution",
                "probabilityMseVariance":
                    "weighted population variance of per-example visible-range base-action probability MSE",
            },
            "timeWeighting": time_weighting_metadata(time_weighting),
            "predictionDelayMs": int(dataset_manifest["predictionDelayMs"]),
            "distributionLossRange": [
                dataset_manifest["policySupport"]["visible_lower"],
                dataset_manifest["policySupport"]["visible_upper"],
            ],
            "finalizedEarly": finalized_early,
            "seed": args.seed,
            "device": str(device),
        },
    }
    if args.plan:
        manifest["training"]["planFile"] = str(args.plan)
    atomic_json(manifest, args.output / "manifest.json")


def validate_dataset_manifest(manifest: dict, root: Path | None = None) -> None:
    if manifest.get("version") != 8 \
            or manifest.get("featureSchemaVersion") != FEATURE_SCHEMA_VERSION:
        raise ValueError("unsupported MLP fitted-policy dataset schema")
    if manifest.get("featureCount") != INPUT_FEATURE_COUNT:
        raise ValueError("dataset feature count does not match model")
    if manifest.get("teacherParameterCount") != TEACHER_PARAMETER_COUNT:
        raise ValueError("dataset diagnostic teacher parameter count is invalid")
    if manifest.get("actionCount") != OUTPUT_ACTION_COUNT:
        raise ValueError("dataset action count does not match direct model output")
    delay = manifest.get("predictionDelayMs")
    pairing = manifest.get("timestampPairing", {})
    component_layout = manifest.get("componentLayout", {})
    if not isinstance(delay, int) or delay < 0 \
            or pairing.get("oracleTargetTime") != "predictionTime - predictionDelayMs" \
            or pairing.get("splitAssignment") != "predictionTime" \
            or pairing.get("responseLagMs") != delay \
            or component_layout.get("version") != 1 \
            or ("storeId" in component_layout
                and (not isinstance(component_layout["storeId"], str)
                     or not component_layout["storeId"])):
        raise ValueError("dataset delayed timestamp-pairing contract is invalid")
    expected_teacher_metrics = [
        "crossEntropy",
        "klDivergence",
        "meanSquaredError",
        "iterations",
        "restarts",
        "converged",
        "distanceImbalance",
    ]
    if manifest.get("teacherMetricCount") != len(expected_teacher_metrics) \
            or manifest.get("teacherMetricNames") != expected_teacher_metrics:
        raise ValueError("dataset teacher metric metadata is invalid")
    if len(manifest.get("grid", [])) != manifest.get("actionCount"):
        raise ValueError("dataset action grid is invalid")
    if not is_centered_power_of_two_grid(manifest["actionCount"]):
        raise ValueError("dataset action grid must contain 2^n-1 cells")
    raw_oracle = manifest.get("rawOracleMap", {})
    minute_oracle = manifest.get("minuteOracleMap", {})
    expected_shape = [len(manifest.get("currentGrid", [])), manifest["actionCount"]]
    if expected_shape != [manifest["actionCount"], manifest["actionCount"]] \
            or raw_oracle.get("materializedDtype") != "float32" \
            or raw_oracle.get("materializedLayout") \
            != "row-major [example, currentExposure, targetExposure]" \
            or raw_oracle.get("shape") != expected_shape \
            or raw_oracle.get("currentExposureGrid") != "currentGrid" \
            or raw_oracle.get("targetExposureGrid") != "grid" \
            or raw_oracle.get("normalized") is not True \
            or raw_oracle.get("losslessEncoding") \
            != "base-probabilities-plus-deterministic-transaction-transition-v1" \
            or raw_oracle.get("factorDtype") != "float32" \
            or raw_oracle.get("factorLayout") \
            != "row-major [example, targetExposure]" \
            or raw_oracle.get("factorShape") != [manifest["actionCount"]] \
            or raw_oracle.get("factorFileField") != "rawOracleProbabilities" \
            or raw_oracle.get("factorCompression", "none") not in ("none", "zstd") \
            or raw_oracle.get("optionalHardCutoffCoordinates") \
            != "teacherParameters[6:8]" \
            or any(not isinstance(shard.get("rawOracleProbabilities"), str)
                   for shard in manifest.get("shards", [])):
        raise ValueError("dataset raw oracle map contract is invalid")
    resolution = minute_oracle.get("resolutionDivergence", {})
    minute_storage = minute_oracle.get("storage", "persisted-per-example")
    if minute_oracle.get("factorDtype") != "float32" \
            or minute_oracle.get("factorLayout") \
            != "row-major [example, targetExposure]" \
            or minute_oracle.get("factorShape") != [manifest["actionCount"]] \
            or minute_oracle.get("factorFileField") \
            != "minuteOracleProbabilities" \
            or minute_oracle.get("factorCompression", "none") \
            not in ("none", "zstd") \
            or minute_oracle.get("normalized") is not True \
            or minute_storage not in (
                "persisted-per-example",
                "persisted-per-minute-day",
                "computed-directly-at-training-startup",
            ) \
            or (minute_storage == "persisted-per-minute-day"
                and minute_oracle.get("storedRowsPerUtcDay") != 1_441) \
            or resolution.get("metric") != "Jensen-Shannon divergence" \
            or resolution.get("fileField") != "resolutionDivergence" \
            or any(not isinstance(shard.get("minuteOracleProbabilities"), str)
                   or not isinstance(shard.get("resolutionDivergence"), str)
                   for shard in manifest.get("shards", [])):
        raise ValueError("dataset one-minute oracle contract is invalid")
    example_weighting = manifest.get("exampleWeighting", {})
    if example_weighting.get("dtype") != "float32" \
            or example_weighting.get("layout") != "row-major [example]" \
            or example_weighting.get("fileField") != "timeWeights" \
            or example_weighting.get("baseFileField") != "baseTimeWeights" \
            or example_weighting.get("distanceImbalanceMetadataField") \
            != "teacherMetrics.distanceImbalance" \
            or example_weighting.get("storedWeights") \
            != "causal unnormalized example weights" \
            or example_weighting.get("trainingTransform") \
            != "divide each batch by its mean only" \
            or any(not isinstance(shard.get("timeWeights"), str)
                   for shard in manifest.get("shards", [])):
        raise ValueError("dataset example-weighting contract is invalid")
    if root is not None:
        expected_row_bytes = manifest["actionCount"] * np.dtype("<f4").itemsize
        input_components = {
            component["features"]: component
            for component in component_layout.get("inputComponents", [])
            if isinstance(component, dict)
            and isinstance(component.get("features"), str)
        }
        oracle_components = {
            component["rawOracleProbabilities"]: component
            for component in component_layout.get("oracleComponents", [])
            if isinstance(component, dict)
            and isinstance(component.get("rawOracleProbabilities"), str)
        }
        for shard in manifest["shards"]:
            count = int(shard["count"])
            feature_offset = int(shard.get("featureRowOffset", -1))
            feature_stride = int(shard.get("featureRowStride", 0))
            oracle_offset = int(shard.get("oracleRowOffset", -1))
            oracle_stride = int(shard.get("oracleRowStride", 0))
            prediction_start = shard.get("predictionTimeStart")
            oracle_start = shard.get("oracleTargetTimeStart")
            if count < 1 or min(feature_offset, oracle_offset) < 0 \
                    or min(feature_stride, oracle_stride) < 1 \
                    or not isinstance(prediction_start, int) \
                    or not isinstance(oracle_start, int) \
                    or prediction_start - oracle_start != delay:
                raise ValueError("dataset component row view is invalid")
            file = root / shard["rawOracleProbabilities"]
            minimum_oracle_rows = oracle_offset + (count - 1) * oracle_stride + 1
            oracle_component = oracle_components.get(
                shard["rawOracleProbabilities"], {}
            )
            if file.name.endswith(".zst") \
                    or raw_oracle.get("factorCompression", "none") == "zstd":
                expected_decoded_bytes = 86_400 * expected_row_bytes
                invalid_raw_oracle = (
                    not file.is_file()
                    or file.stat().st_size < 1
                    or not file.name.endswith(".zst")
                    or raw_oracle.get("factorCompression") != "zstd"
                    or oracle_component.get(
                        "rawOracleProbabilitiesCompression"
                    ) != "zstd"
                    or oracle_component.get(
                        "rawOracleProbabilitiesUncompressedBytes"
                    ) != expected_decoded_bytes
                    or minimum_oracle_rows > 86_400
                )
            else:
                invalid_raw_oracle = (
                    not file.is_file()
                    or file.stat().st_size % expected_row_bytes != 0
                    or file.stat().st_size
                    < minimum_oracle_rows * expected_row_bytes
                )
            if invalid_raw_oracle:
                raise ValueError(
                    f"dataset raw oracle grid file has an invalid size: {file}"
                )
            feature_file = root / shard["features"]
            feature_row_bytes = manifest["featureCount"] * np.dtype("<f2").itemsize
            minimum_feature_rows = feature_offset + (count - 1) * feature_stride + 1
            feature_component = input_components.get(shard["features"], {})
            if feature_component.get("featuresCompression") == "zstd" \
                    or feature_file.name.endswith(".zst"):
                expected_decoded_bytes = 86_400 * feature_row_bytes
                if not feature_file.is_file() \
                        or feature_file.stat().st_size < 1 \
                        or feature_component.get("featuresCompression") != "zstd" \
                        or feature_component.get("featuresUncompressedBytes") \
                        != expected_decoded_bytes \
                        or minimum_feature_rows > 86_400:
                    raise ValueError(
                        "dataset compressed input feature component is invalid: "
                        f"{feature_file}"
                    )
            elif not feature_file.is_file() \
                    or feature_file.stat().st_size % feature_row_bytes != 0 \
                    or feature_file.stat().st_size \
                    < minimum_feature_rows * feature_row_bytes:
                raise ValueError(
                    f"dataset input feature component has an invalid size: {feature_file}"
                )
            weight_file = root / shard["timeWeights"]
            base_weight_file = root / shard["baseTimeWeights"]
            expected_weight_bytes = count * np.dtype("<f4").itemsize
            if not weight_file.is_file() \
                    or weight_file.stat().st_size != expected_weight_bytes \
                    or not base_weight_file.is_file() \
                    or base_weight_file.stat().st_size != expected_weight_bytes:
                raise ValueError(
                    f"dataset example-weight file has an invalid size: {weight_file}"
                )
            minute_file = root / shard["minuteOracleProbabilities"]
            resolution_file = root / shard["resolutionDivergence"]
            minute_compression = minute_oracle.get("factorCompression", "none")
            minute_persisted = minute_storage in (
                "persisted-per-example",
                "persisted-per-minute-day",
            )
            minute_rows = 1_441 \
                if minute_storage == "persisted-per-minute-day" else count
            if minute_persisted \
                    and minute_compression == "zstd":
                invalid_minute_oracle = (
                    not minute_file.is_file()
                    or minute_file.stat().st_size < 1
                    or not minute_file.name.endswith(".zst")
                )
            else:
                invalid_minute_oracle = (
                    minute_persisted
                    and (
                        not minute_file.is_file()
                        or minute_file.stat().st_size
                        != minute_rows * expected_row_bytes
                    )
                )
            if invalid_minute_oracle:
                raise ValueError(
                    f"dataset one-minute oracle file has an invalid size: {minute_file}"
                )
            if not resolution_file.is_file() \
                    or resolution_file.stat().st_size != count * np.dtype("<f4").itemsize:
                raise ValueError(
                    f"dataset resolution divergence file has an invalid size: {resolution_file}"
                )
    if not isinstance(manifest.get("samplingIntervalMs"), int) \
            or manifest["samplingIntervalMs"] <= 0:
        raise ValueError("dataset sampling interval is invalid")
    support = manifest.get("policySupport", {})
    grid = manifest.get("grid", [])
    if not grid or grid[0] != support.get("latent_lower") or grid[-1] != support.get("latent_upper"):
        raise ValueError("dataset teacher grid must cover the complete effective range")


def validate_args(args: argparse.Namespace) -> None:
    if min(args.epochs, args.batch_size, args.accumulate, args.states_per_example,
           args.log_every_steps) < 1 or args.workers < 0:
        raise ValueError("training counts must be positive (workers may be zero)")
    if not args.disable_patience and args.patience < 1:
        raise ValueError("training patience must be positive when enabled")
    if args.evaluation_batch_size < 0:
        raise ValueError("evaluation batch size must be non-negative")
    if not 0 < args.validation_fraction <= 1 or not 0 < args.training_fraction <= 1:
        raise ValueError("training and validation fractions must be in (0, 1]")
    if args.learning_rate <= 0 or args.weight_decay < 0 or not 0 <= args.dropout < 1:
        raise ValueError("invalid optimizer or dropout configuration")
    if args.target_validation_kl is not None and args.target_validation_kl <= 0:
        raise ValueError("target validation KL must be positive")
    if args.target_validation_base_kl is not None \
            and args.target_validation_base_kl <= 0:
        raise ValueError("target validation base KL must be positive")
    if args.target_validation_kl is not None \
            and args.target_validation_base_kl is not None:
        raise ValueError("choose either conditional or base validation KL target")
    if args.target_validation_kl_stddev is not None \
            and args.target_validation_kl_stddev <= 0:
        raise ValueError("target validation KL standard deviation must be positive")
    if args.target_validation_kl_stddev is not None \
            and args.target_validation_kl is None:
        raise ValueError("target validation KL is required when its standard deviation is set")
    if args.initialize_from_checkpoint is not None \
            and not args.initialize_from_checkpoint.is_file():
        raise ValueError("warm-start checkpoint does not exist")
    if args.evaluation_only and args.initialize_from_checkpoint is None:
        raise ValueError("evaluation-only export requires a warm-start checkpoint")
    if args.inherited_best_epoch < -1:
        raise ValueError("inherited best epoch must be -1 or non-negative")
    if not is_centered_power_of_two_grid(args.states_per_example):
        raise ValueError("training current-state grid must contain 2^n-1 cells")


def validation_target_reached(
    metrics: dict[str, float],
    args: argparse.Namespace,
) -> bool:
    if args.target_validation_base_kl is not None:
        return (
            metrics.get("baseKlDivergence", math.inf)
            <= args.target_validation_base_kl
        )
    if args.target_validation_kl is None:
        return False
    if metrics.get("klDivergence", math.inf) > args.target_validation_kl:
        return False
    if args.target_validation_kl_stddev is not None \
            and metrics.get("klDivergenceStdDev", math.inf) \
            > args.target_validation_kl_stddev:
        return False
    return True


def checkpoint_identity(file: Path | None) -> dict | None:
    if file is None:
        return None
    resolved = file.resolve()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size": stat.st_size,
        "modifiedNs": stat.st_mtime_ns,
    }
def is_centered_power_of_two_grid(count: int) -> bool:
    return count >= 3 and (count & (count + 1)) == 0


def detached_metrics(metrics: dict[str, Tensor]) -> dict[str, float]:
    return {
        name: float(metrics[name].detach())
        for name in METRIC_NAMES
        if name in metrics
    }


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


def atomic_numpy_archive(target: Path, **values) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, **values)
    temporary.replace(target)


def atomic_bytes(value: bytes, target: Path) -> None:
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_bytes(value)
    temporary.replace(target)


if __name__ == "__main__":
    main()
