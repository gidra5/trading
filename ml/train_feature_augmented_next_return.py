from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

from global_feature_training_dataset import IndexedGlobalFeatureDataset
from normalized_glu_next_return import NormalizedGluNextReturn
from trading_storage import (
    _prune_checkpoint_orphans,
    load_torch_checkpoint,
    save_torch_checkpoint,
)
from train_autoregressive_minute_return import build_optimizers
from train_normalized_glu_next_return import MetricAccumulator, Reporter, atomic_json
from train_next_return_memorization import (
    adversarial_input_examples,
    sam_perturb_parameters,
    sam_restore_parameters,
)


RUNNER_CONTRACT = "feature-augmented-clean-next-second-glu-regression-v1"
MINIMUM_CHECKPOINT_FREE_BYTES = 2 * 1024**3


def wait_for_checkpoint_storage(
    repo: Path,
    reporter: Reporter,
    plan_id: str,
    *,
    epoch: int,
) -> None:
    """Pause safely before serialization instead of dying on a full volume."""
    store_root = (repo / "data/training/immutable").resolve()
    while True:
        _prune_checkpoint_orphans(store_root)
        free = shutil.disk_usage(repo).free
        if free >= MINIMUM_CHECKPOINT_FREE_BYTES:
            return
        reporter.status(
            "waiting-for-storage",
            planId=plan_id,
            epoch=epoch,
            freeBytes=free,
            requiredFreeBytes=MINIMUM_CHECKPOINT_FREE_BYTES,
            message=(
                "Training is paused before checkpoint serialization; it will "
                "resume automatically after checkpoint-only GC or free space."
            ),
        )
        time.sleep(10)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a GLU next-return regressor from an exported feature matrix."
    )
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument(
        "--recover-validation-selections",
        action="store_true",
        help="Replay only through validation-selected epochs recorded in the log.",
    )
    return parser.parse_args()


def canonical_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class ArraySplit:
    features: object
    targets: np.memmap
    times: np.memmap

    @property
    def count(self) -> int:
        return int(self.targets.shape[0])


class TemporalFeatureMatrix:
    """Materialize overlapping channel-major windows from one stored timeline."""

    def __init__(
        self,
        timeline_file: Path,
        origins_file: Path,
        *,
        count: int,
        timeline_rows: int,
        channel_count: int,
        history_seconds: int,
    ) -> None:
        self.timeline = np.memmap(
            timeline_file, dtype="<f4", mode="r",
            shape=(timeline_rows, channel_count),
        )
        self.origins = np.memmap(
            origins_file, dtype="<i4", mode="r", shape=(count,)
        )
        self.channel_count = channel_count
        self.history_seconds = history_seconds
        self.shape = (count, channel_count * history_seconds)
        self._device_timeline: torch.Tensor | None = None
        self._device_origins: torch.Tensor | None = None
        self._device_offsets: torch.Tensor | None = None
        if np.any(self.origins < history_seconds - 1) \
                or np.any(self.origins >= timeline_rows):
            raise ValueError("temporal feature origins escape their timeline")

    def __getitem__(self, selected):
        scalar = isinstance(selected, (int, np.integer))
        origins = np.asarray(self.origins[selected], dtype=np.int64)
        if scalar:
            origins = origins.reshape(1)
        offsets = np.arange(
            1 - self.history_seconds, 1, dtype=np.int64
        )
        timeline_rows = origins[:, None] + offsets[None, :]
        values = np.asarray(self.timeline[timeline_rows], dtype=np.float32)
        flattened = values.transpose(0, 2, 1).reshape(
            origins.size, self.shape[1]
        )
        return flattened[0] if scalar else flattened

    def to_device(self, device: torch.device) -> None:
        """Cache the compact timeline on the accelerator for window gathers."""
        if self._device_timeline is not None \
                and self._device_timeline.device == device:
            return
        self._device_timeline = torch.tensor(
            np.asarray(self.timeline), dtype=torch.float32, device=device
        )
        self._device_origins = torch.tensor(
            np.asarray(self.origins, dtype=np.int64),
            dtype=torch.long,
            device=device,
        )
        self._device_offsets = torch.arange(
            1 - self.history_seconds, 1,
            dtype=torch.long, device=device,
        )

    def device_rows(
        self,
        selected: slice | np.ndarray,
        device: torch.device,
    ) -> torch.Tensor:
        self.to_device(device)
        assert self._device_timeline is not None
        assert self._device_origins is not None
        assert self._device_offsets is not None
        if isinstance(selected, slice):
            start, stop, step = selected.indices(self.shape[0])
            selected_tensor = torch.arange(
                start, stop, step, dtype=torch.long, device=device
            )
        else:
            selected_tensor = torch.as_tensor(
                selected, dtype=torch.long, device=device
            )
        origins = self._device_origins[selected_tensor]
        timeline_rows = origins[:, None] + self._device_offsets[None, :]
        values = self._device_timeline[timeline_rows]
        return values.transpose(1, 2).reshape(
            origins.numel(), self.shape[1]
        )


class IndexedGlobalFeatureMatrix:
    """Expose one chronological split of a compact global-feature dataset."""

    def __init__(
        self,
        dataset: IndexedGlobalFeatureDataset,
        logical_rows: np.ndarray,
    ) -> None:
        self.dataset = dataset
        self.logical_rows = np.asarray(logical_rows, dtype=np.int64)
        self.shape = (
            int(self.logical_rows.size),
            int(dataset.manifest["featureCount"]),
        )

    def __getitem__(self, selected):
        scalar = isinstance(selected, (int, np.integer))
        if isinstance(selected, slice):
            selected_rows = np.arange(*selected.indices(self.shape[0]), dtype=np.int64)
        else:
            selected_rows = np.asarray(selected, dtype=np.int64)
            if scalar:
                selected_rows = selected_rows.reshape(1)
        values = self.dataset.read(self.logical_rows[selected_rows])["features"]
        return values[0] if scalar else values


class UnionTemporalFeatureMatrix:
    """Join 59 production-basis and 471 global channels over one history."""

    def __init__(
        self,
        old_timeline: np.memmap,
        old_origin_rows: np.ndarray,
        global_dataset: IndexedGlobalFeatureDataset,
        global_physical_rows: np.ndarray,
        *,
        history_seconds: int,
        old_channel_count: int,
    ) -> None:
        self.old_timeline = old_timeline
        self.old_origin_rows = np.asarray(old_origin_rows, dtype=np.int64)
        self.global_dataset = global_dataset
        self.global_physical_rows = np.asarray(
            global_physical_rows, dtype=np.int64
        )
        self.history_seconds = int(history_seconds)
        self.old_channel_count = int(old_channel_count)
        self.global_channel_count = int(global_dataset.manifest["featureCount"])
        self.channel_count = self.old_channel_count + self.global_channel_count
        self.offsets = np.arange(
            1 - self.history_seconds, 1, dtype=np.int64
        )
        self.shape = (
            int(self.global_physical_rows.size),
            self.channel_count * self.history_seconds,
        )

    def __getitem__(self, selected):
        scalar = isinstance(selected, (int, np.integer))
        if isinstance(selected, slice):
            selected_rows = np.arange(*selected.indices(self.shape[0]), dtype=np.int64)
        else:
            selected_rows = np.asarray(selected, dtype=np.int64)
            if scalar:
                selected_rows = selected_rows.reshape(1)
        old_current = self.old_origin_rows[selected_rows]
        global_current = self.global_physical_rows[selected_rows]
        old_history_rows = old_current[:, None] + self.offsets[None, :]
        global_history_rows = global_current[:, None] + self.offsets[None, :]
        old_history = np.asarray(
            self.old_timeline[old_history_rows], dtype=np.float32
        )
        source_rows = np.asarray(
            self.global_dataset.source_rows[global_history_rows],
            dtype=np.int64,
        )
        global_history = np.asarray(
            self.global_dataset.matrix[
                source_rows[..., None],
                self.global_dataset.columns,
            ],
            dtype=np.float32,
        )
        output = np.empty(
            (
                selected_rows.size,
                self.channel_count,
                self.history_seconds,
            ),
            dtype=np.float32,
        )
        output[:, :self.old_channel_count] = old_history.transpose(0, 2, 1)
        global_end = self.old_channel_count + global_history.shape[2]
        output[:, self.old_channel_count:global_end] = global_history.transpose(
            0, 2, 1
        )
        if self.global_dataset.spread is not None:
            output[:, global_end] = np.asarray(
                self.global_dataset.spread[global_history_rows],
                dtype=np.float32,
            )
        elif global_end != self.channel_count:
            raise ValueError("global union history is missing its final channel")
        flattened = output.reshape(selected_rows.size, self.shape[1])
        return flattened[0] if scalar else flattened


class FeatureMatrixDataset:
    def __init__(
        self,
        root: Path,
        *,
        selection: str | None = None,
        examples_by_split: dict[str, int] | None = None,
        union_history_root: Path | None = None,
        feature_history_seconds: int | None = None,
    ) -> None:
        self.root = root
        indexed_manifest = root / "dataset.json"
        if union_history_root is not None:
            if not indexed_manifest.is_file():
                raise ValueError("union history requires an indexed global dataset")
            self._initialize_union_history(
                union_history_root,
                history_seconds=int(feature_history_seconds or 0),
                examples_by_split=examples_by_split,
            )
            return
        if indexed_manifest.is_file():
            self._initialize_indexed(
                indexed_manifest,
                selection=selection or "nonzero",
                examples_by_split=examples_by_split,
            )
            return
        if selection is not None or examples_by_split is not None:
            raise ValueError(
                "indexed dataset selection settings require dataset.json"
            )
        self.manifest = json.loads((root / "manifest.json").read_text("utf-8"))
        self.feature_count = int(self.manifest["featureCount"])
        compact_temporal = self.manifest.get("storageLayout") \
            == "temporal-channel-timeline-v1"
        self.splits: dict[str, ArraySplit] = {}
        for name in ("train", "validation", "test"):
            targets_file = root / f"{name}.targets.f32"
            times_file = root / f"{name}.times.f64"
            targets = np.memmap(targets_file, dtype="<f4", mode="r")
            if targets.size < 1:
                raise ValueError(f"{name} split is empty")
            if compact_temporal:
                timeline_rows = int(
                    self.manifest["timelineRowsBySplit"][name]
                )
                features = TemporalFeatureMatrix(
                    root / f"{name}.timeline-features.f32",
                    root / f"{name}.origins.i32",
                    count=int(targets.size),
                    timeline_rows=timeline_rows,
                    channel_count=int(self.manifest["temporalChannelCount"]),
                    history_seconds=int(self.manifest["featureHistorySeconds"]),
                )
            else:
                features = np.memmap(
                    root / f"{name}.features.f32",
                    dtype="<f4",
                    mode="r",
                    shape=(targets.size, self.feature_count),
                )
            times = np.memmap(
                times_file, dtype="<f8", mode="r", shape=(targets.size,)
            )
            finite_features = np.isfinite(
                features.timeline if isinstance(features, TemporalFeatureMatrix)
                else features
            ).all()
            if not finite_features or not np.isfinite(targets).all():
                raise ValueError(f"{name} split contains non-finite values")
            if np.any(targets == 0):
                raise ValueError(f"{name} split contains an exact-zero target")
            if np.any(np.diff(times) <= 0):
                raise ValueError(f"{name} timestamps are not strictly increasing")
            self.splits[name] = ArraySplit(features, targets, times)
        if not (
            self.splits["train"].times[-1]
            < self.splits["validation"].times[0]
            < self.splits["test"].times[0]
        ):
            raise ValueError("feature-matrix splits are not chronological")

    def _initialize_indexed(
        self,
        manifest_file: Path,
        *,
        selection: str,
        examples_by_split: dict[str, int] | None,
    ) -> None:
        self.manifest = json.loads(manifest_file.read_text("utf-8"))
        self.feature_count = int(self.manifest["featureCount"])
        self.indexed_dataset = IndexedGlobalFeatureDataset(
            self.root, selection=selection
        )
        limits = examples_by_split or {}
        unknown = set(limits) - {"train", "validation", "test"}
        if unknown:
            raise ValueError(f"unknown indexed split limits: {sorted(unknown)}")
        split_codes = {"train": 0, "validation": 1, "test": 2}
        selected_physical = self.indexed_dataset.rows
        selected_splits = np.asarray(
            self.indexed_dataset.splits[selected_physical], dtype=np.uint8
        )
        self.splits = {}
        for name, code in split_codes.items():
            logical_rows = np.flatnonzero(selected_splits == code)
            available = int(logical_rows.size)
            limit = int(limits.get(name, available))
            if limit < 1 or limit > available:
                raise ValueError(
                    f"indexed {name} split requested {limit:,} of "
                    f"{available:,} available {selection} examples"
                )
            logical_rows = logical_rows[:limit]
            physical_rows = selected_physical[logical_rows]
            features = IndexedGlobalFeatureMatrix(
                self.indexed_dataset, logical_rows
            )
            targets = np.asarray(
                self.indexed_dataset.targets[physical_rows], dtype=np.float32
            )
            times = np.asarray(
                self.indexed_dataset.origins[physical_rows], dtype=np.float64
            )
            if not np.isfinite(targets).all():
                raise ValueError(f"{name} split contains non-finite targets")
            if selection == "nonzero" and np.any(targets == 0):
                raise ValueError(f"{name} split contains an exact-zero target")
            if np.any(np.diff(times) <= 0):
                raise ValueError(f"{name} timestamps are not strictly increasing")
            self.splits[name] = ArraySplit(features, targets, times)
        if not (
            self.splits["train"].times[-1]
            < self.splits["validation"].times[0]
            < self.splits["test"].times[0]
        ):
            raise ValueError("indexed feature splits are not chronological")

    def _initialize_union_history(
        self,
        history_root: Path,
        *,
        history_seconds: int,
        examples_by_split: dict[str, int] | None,
    ) -> None:
        if history_seconds < 1:
            raise ValueError("union feature history must be positive")
        history_manifest = json.loads(
            (history_root / "manifest.json").read_text("utf-8")
        )
        old_channel_count = int(history_manifest["temporalChannelCount"])
        source_history_seconds = int(
            history_manifest["featureHistorySeconds"]
        )
        if history_seconds > source_history_seconds:
            raise ValueError(
                "union history length exceeds its source timeline capacity"
            )
        global_dataset = IndexedGlobalFeatureDataset(self.root, selection="all")
        global_channel_count = int(global_dataset.manifest["featureCount"])
        self.feature_count = (
            old_channel_count + global_channel_count
        ) * history_seconds
        self.indexed_dataset = global_dataset
        self.union_history_manifest = history_manifest
        self.manifest = {
            "schemaVersion": 1,
            "id": (
                f'{global_dataset.manifest["id"]}-union-'
                f'{old_channel_count + global_channel_count}x{history_seconds}'
            ),
            "purpose": (
                "Causal temporal union of the production-basis channels and "
                "the global BTC production feature union"
            ),
            "featureCount": self.feature_count,
            "baseFeatureCount": old_channel_count + global_channel_count,
            "featureHistorySeconds": history_seconds,
            "sourceHistorySeconds": source_history_seconds,
            "selectionModes": global_dataset.manifest.get("selectionModes", {}),
            "featureStorage": global_dataset.manifest.get("featureStorage", {}),
            "sources": {
                "productionBasis": str(history_root),
                "globalFeatures": str(self.root),
            },
        }
        limits = examples_by_split or {}
        unknown = set(limits) - {"train", "validation", "test"}
        if unknown:
            raise ValueError(f"unknown union split limits: {sorted(unknown)}")
        all_origins = np.asarray(global_dataset.origins, dtype=np.int64)
        all_splits = np.asarray(global_dataset.splits, dtype=np.uint8)
        all_nonzero = np.asarray(global_dataset.nonzero, dtype=np.uint8) != 0
        split_codes = {"train": 0, "validation": 1, "test": 2}
        self.splits = {}
        self.union_timelines: dict[str, np.memmap] = {}
        expected_span_ms = (history_seconds - 1) * 1_000
        for name, code in split_codes.items():
            candidates = np.flatnonzero((all_splits == code) & all_nonzero)
            candidates = candidates[candidates >= history_seconds - 1]
            candidates = candidates[
                all_origins[candidates]
                - all_origins[candidates - (history_seconds - 1)]
                == expected_span_ms
            ]
            history_times = np.memmap(
                history_root / f"{name}.times.f64", dtype="<f8", mode="r"
            )
            history_origins = np.memmap(
                history_root / f"{name}.origins.i32", dtype="<i4", mode="r"
            )
            # The compact production-basis exporter labels a completed candle
            # by its right boundary, while the indexed global dataset labels
            # that same causal state by the candle origin. Their targets agree
            # at a +1 second production-basis timestamp offset.
            history_target_times = all_origins[candidates] + 1_000
            positions = np.searchsorted(history_times, history_target_times)
            matched = positions < history_times.size
            matched[matched] &= (
                np.asarray(history_times[positions[matched]], dtype=np.int64)
                == history_target_times[matched]
            )
            candidates = candidates[matched]
            positions = positions[matched]
            available = int(candidates.size)
            limit = int(limits.get(name, available))
            if limit < 1 or limit > available:
                raise ValueError(
                    f"union {name} split requested {limit:,} of "
                    f"{available:,} aligned examples"
                )
            candidates = candidates[:limit]
            positions = positions[:limit]
            timeline_rows = int(history_manifest["timelineRowsBySplit"][name])
            timeline = np.memmap(
                history_root / f"{name}.timeline-features.f32",
                dtype="<f4",
                mode="r",
                shape=(timeline_rows, old_channel_count),
            )
            old_origin_rows = np.asarray(
                history_origins[positions], dtype=np.int64
            )
            if np.any(old_origin_rows < history_seconds - 1):
                raise ValueError(f"union {name} history escapes its source timeline")
            physical = np.asarray(candidates, dtype=np.int64)
            features = UnionTemporalFeatureMatrix(
                timeline,
                old_origin_rows,
                global_dataset,
                physical,
                history_seconds=history_seconds,
                old_channel_count=old_channel_count,
            )
            targets = np.asarray(global_dataset.targets[physical], dtype=np.float32)
            times = np.asarray(global_dataset.origins[physical], dtype=np.float64)
            if not np.isfinite(targets).all() or np.any(targets == 0):
                raise ValueError(f"union {name} targets are invalid")
            if np.any(np.diff(times) <= 0):
                raise ValueError(f"union {name} timestamps are not increasing")
            self.union_timelines[name] = timeline
            self.splits[name] = ArraySplit(features, targets, times)
        if not (
            self.splits["train"].times[-1]
            < self.splits["validation"].times[0]
            < self.splits["test"].times[0]
        ):
            raise ValueError("union feature splits are not chronological")

    def logical_count(self, split: str) -> int:
        return self.splits[split].count

    def iter_batches(
        self,
        split: str,
        batch_size: int,
        *,
        shuffle: bool,
        seed: int,
        device: torch.device | None = None,
    ):
        values = self.splits[split]
        if shuffle:
            batches: list[slice | np.ndarray] = []
            indices = np.arange(values.count, dtype=np.int64)
            np.random.default_rng(seed).shuffle(indices)
            for start in range(0, values.count, batch_size):
                # Row order inside a batch does not change its gradient. Sorting
                # restores sequential memmap access without changing which
                # globally shuffled examples share each stochastic update.
                batches.append(np.sort(indices[start:start + batch_size]))
        else:
            starts = np.arange(0, values.count, batch_size, dtype=np.int64)
            batches = [
                slice(int(start), min(int(start) + batch_size, values.count))
                for start in starts
            ]
        for selected in batches:
            targets = np.asarray(values.targets[selected], dtype=np.float32)
            if device is not None \
                    and isinstance(values.features, TemporalFeatureMatrix):
                feature_tensor = values.features.device_rows(selected, device)
            else:
                features = np.asarray(
                    values.features[selected], dtype=np.float32
                )
                feature_tensor = torch.from_numpy(features.copy())
            yield (
                feature_tensor,
                torch.from_numpy(targets.copy()),
                torch.ones(len(targets), dtype=torch.float32),
            )


def normalization(dataset: FeatureMatrixDataset) -> dict[str, np.ndarray | float]:
    features = dataset.splits["train"].features
    targets = np.asarray(dataset.splits["train"].targets, dtype=np.float64)
    feature_sum = np.zeros(dataset.feature_count, dtype=np.float64)
    feature_square_sum = np.zeros(dataset.feature_count, dtype=np.float64)
    normalization_batch_size = min(
        1_024, max(1, 8_000_000 // dataset.feature_count)
    )
    for start in range(0, features.shape[0], normalization_batch_size):
        batch = np.asarray(
            features[start:start + normalization_batch_size], dtype=np.float64
        )
        feature_sum += batch.sum(axis=0)
        feature_square_sum += np.square(batch).sum(axis=0)
    feature_mean = feature_sum / features.shape[0]
    feature_variance = np.maximum(
        feature_square_sum / features.shape[0] - np.square(feature_mean), 0.0
    )
    feature_std = np.sqrt(feature_variance)
    # Availability masks can legitimately be constant on a complete historical
    # tier. Retaining them with unit scale preserves the production contract
    # without introducing NaNs or an artificial signal.
    feature_std = np.where(feature_std > 1e-8, feature_std, 1.0)
    target_mean = float(targets.mean())
    target_std = float(targets.std())
    if not math.isfinite(target_std) or target_std <= 0:
        raise ValueError("training targets have no finite variance")
    return {
        "featureMean": feature_mean.astype(np.float32),
        "featureStd": feature_std.astype(np.float32),
        "targetMean": target_mean,
        "targetStd": target_std,
    }


@torch.no_grad()
def evaluate(
    model: NormalizedGluNextReturn,
    dataset: FeatureMatrixDataset,
    split: str,
    *,
    batch_size: int,
    target_std: float,
    device: torch.device,
) -> dict:
    model.eval()
    metrics = MetricAccumulator(target_std, device)
    for features, targets, weights in dataset.iter_batches(
        split, batch_size, shuffle=False, seed=0, device=device
    ):
        features = features.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        weights = weights.to(device, non_blocking=True)
        prediction = model(features)
        metrics.add(prediction, targets, weights)
    return metrics.result()


def train_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    target_std: float,
) -> torch.Tensor:
    return ((prediction - target) / target_std).square().mean()


def validate_plan(plan: dict) -> None:
    required = ("id", "label", "datasetDir", "runDir", "architecture", "training")
    if any(key not in plan for key in required):
        raise ValueError("feature-augmented training plan is incomplete")
    widths = tuple(int(value) for value in plan["architecture"]["widths"])
    if len(widths) != 4:
        raise ValueError("this experiment requires exactly four GLU layers")
    if plan["training"]["device"] != "cuda":
        raise ValueError("feature-augmented experiment requires CUDA")
    training = plan["training"]
    sam = training.get("sam")
    if sam is not None and (
        sam.get("type") != "sharpness-aware-minimization"
        or sam.get("adaptive") is not False
        or sam.get("gradientNorm") != "global-l2"
        or sam.get("perturbationUnit") != "optimizer-update"
        or sam.get("baseObjective") != "clean-plus-adversarial-input"
        or not math.isfinite(float(sam.get("rho", 0)))
        or not 0 < float(sam["rho"]) <= 1
    ):
        raise ValueError("feature-augmented SAM contract is invalid")
    adversarial = training.get("adversarialInput")
    if adversarial is not None:
        epsilon_rms = float(adversarial.get("epsilonRms", 0))
        steps = adversarial.get("steps")
        step_size_rms = float(adversarial.get("stepSizeRms", 0))
        adversarial_weight = float(adversarial.get("adversarialWeight", 0))
        if (
            adversarial.get("type") != "projected-gradient-ascent"
            or adversarial.get("space") != "training-feature-normalized-input"
            or adversarial.get("norm") != "rms-l2"
            or adversarial.get("randomStart") is not False
            or adversarial.get("target") != "unchanged-next-return"
            or adversarial.get("regenerateAtSamPerturbedWeights") is not True
            or not math.isfinite(epsilon_rms)
            or not 0 < epsilon_rms <= 1
            or not isinstance(steps, int)
            or isinstance(steps, bool)
            or not 1 <= steps <= 16
            or not math.isfinite(step_size_rms)
            or step_size_rms <= 0
            or not math.isfinite(adversarial_weight)
            or not 0 < adversarial_weight <= 1
        ):
            raise ValueError("feature-augmented adversarial-input contract is invalid")
    if (sam is None) != (adversarial is None):
        raise ValueError(
            "this feature experiment requires SAM and adversarial input together"
        )


def checkpoint_payload(
    model: NormalizedGluNextReturn,
    optimizers: tuple[torch.optim.Optimizer, ...],
    *,
    epoch: int,
    global_step: int,
    train: dict,
    validation: dict,
    parameter_count: int,
    plan_hash: str,
    dataset_hash: str,
) -> dict:
    return {
        "model": model.state_dict(),
        "optimizers": [optimizer.state_dict() for optimizer in optimizers],
        "epoch": epoch,
        "globalStep": global_step,
        "train": train,
        "validation": validation,
        "parameterCount": parameter_count,
        "planSha256": plan_hash,
        "datasetSha256": dataset_hash,
        "runnerContract": RUNNER_CONTRACT,
    }


def main() -> None:
    args = parse_args()
    repo = Path(__file__).resolve().parents[1]
    plan_file = (repo / args.plan).resolve() if not args.plan.is_absolute() else args.plan
    plan = json.loads(plan_file.read_text("utf-8"))
    validate_plan(plan)
    dataset_root = (repo / plan["datasetDir"]).resolve()
    run_root = (repo / plan["runDir"]).resolve()
    reporter = Reporter(run_root)
    plan_hash = canonical_hash(plan)
    dataset_manifest_file = (
        dataset_root / "dataset.json"
        if (dataset_root / "dataset.json").is_file()
        else dataset_root / "manifest.json"
    )
    dataset_hasher = hashlib.sha256(dataset_manifest_file.read_bytes())
    union_history_root = None
    if plan.get("unionHistoryDatasetDir") is not None:
        union_history_root = (
            repo / str(plan["unionHistoryDatasetDir"])
        ).resolve()
        dataset_hasher.update(
            (union_history_root / "manifest.json").read_bytes()
        )
        dataset_hasher.update(
            str(plan.get("featureHistorySeconds")).encode("ascii")
        )
    dataset_hash = dataset_hasher.hexdigest()
    run_root.mkdir(parents=True, exist_ok=True)
    atomic_json({"plan": plan, "planSha256": plan_hash}, run_root / "state/plan.json")
    reporter.status("loading-data", planId=plan["id"])
    try:
        dataset = FeatureMatrixDataset(
            dataset_root,
            selection=plan.get("datasetSelection"),
            examples_by_split=plan.get("examplesBySplit"),
            union_history_root=union_history_root,
            feature_history_seconds=plan.get("featureHistorySeconds"),
        )
        stats = normalization(dataset)
        training = plan["training"]
        architecture = plan["architecture"]
        device = torch.device("cuda")
        torch.manual_seed(int(training["seed"]))
        torch.cuda.manual_seed_all(int(training["seed"]))
        random.seed(int(training["seed"]))
        np.random.seed(int(training["seed"]))
        model = NormalizedGluNextReturn(
            torch.from_numpy(stats["featureMean"]),
            torch.from_numpy(stats["featureStd"]),
            torch.tensor(float(stats["targetMean"])),
            torch.tensor(float(stats["targetStd"])),
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
        sam = training.get("sam")
        adversarial_input = training.get("adversarialInput")
        sam_rho = float(sam["rho"]) if sam is not None else 0.0
        adversarial_epsilon_rms = (
            float(adversarial_input["epsilonRms"])
            if adversarial_input is not None else 0.0
        )
        adversarial_steps = (
            int(adversarial_input["steps"])
            if adversarial_input is not None else 0
        )
        adversarial_step_size_rms = (
            float(adversarial_input["stepSizeRms"])
            if adversarial_input is not None else 0.0
        )
        adversarial_weight = (
            float(adversarial_input["adversarialWeight"])
            if adversarial_input is not None else 0.0
        )
        checkpoints = run_root / "checkpoints"
        checkpoints.mkdir(parents=True, exist_ok=True)
        last_file = checkpoints / "last.json"
        policy_files = {
            "train-mse": checkpoints / "best-train-mse.json",
            "train-correlation": checkpoints / "best-train-correlation.json",
            "validation-mse": checkpoints / "best-validation-mse.json",
            "validation-correlation": checkpoints / "best-validation-correlation.json",
        }
        policy_scores = {
            "train-mse": math.inf,
            "train-correlation": -math.inf,
            "validation-mse": math.inf,
            "validation-correlation": -math.inf,
        }
        best_train_score = math.inf
        start_epoch = 0
        global_step = 0
        if last_file.is_file():
            saved = load_torch_checkpoint(last_file, map_location=device, weights_only=False)
            if saved.get("planSha256") != plan_hash or saved.get("datasetSha256") != dataset_hash:
                raise ValueError("existing checkpoint does not match this plan and dataset")
            model.load_state_dict(saved["model"])
            for optimizer, state in zip(optimizers, saved["optimizers"], strict=True):
                optimizer.load_state_dict(state)
            start_epoch = int(saved["epoch"]) + 1
            global_step = int(saved["globalStep"])
            policy_scores.update(saved.get("policyScores", {}))
            best_train_score = float(saved.get("bestTrainScore", math.inf))
        maximum_epoch_exclusive = epochs
        if args.recover_validation_selections:
            events = [
                json.loads(line)
                for line in (run_root / "logs/training.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()
                if line.strip()
            ]
            epoch_events = [
                value for value in events
                if value.get("event") == "minute-return-epoch"
            ]
            if not epoch_events:
                raise ValueError("checkpoint recovery requires the original epoch log")
            selected_epochs = (
                min(epoch_events, key=lambda value: value["validation"]["normalizedMse"])["epoch"],
                max(epoch_events, key=lambda value: value["validation"]["correlation"])["epoch"],
            )
            maximum_epoch_exclusive = max(int(value) for value in selected_epochs) + 1
            start_epoch = 0
            global_step = 0
            policy_scores = {
                "train-mse": math.inf,
                "train-correlation": -math.inf,
                "validation-mse": math.inf,
                "validation-correlation": -math.inf,
            }
            best_train_score = math.inf
            reporter.status(
                "recovering-validation-checkpoints",
                planId=plan["id"],
                selectedEpochs=[int(value) for value in selected_epochs],
                replayEpochs=maximum_epoch_exclusive,
            )
        reporter.status(
            "training",
            planId=plan["id"],
            epochs=epochs,
            featureCount=dataset.feature_count,
            examples=dataset.logical_count("train"),
            parameterCount=parameter_count,
        )
        started = time.monotonic()
        for epoch in range(start_epoch, maximum_epoch_exclusive):
            model.train()
            epoch_clean_numerator = 0.0
            epoch_adversarial_numerator = 0.0
            epoch_example_count = 0
            for features, targets, weights in dataset.iter_batches(
                "train", batch_size, shuffle=True,
                seed=int(training["seed"]) + epoch, device=device,
            ):
                features = features.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                weights = weights.to(device, non_blocking=True)
                for optimizer in optimizers:
                    optimizer.zero_grad(set_to_none=True)
                clean_loss = train_loss(model(features), targets, target_std)
                if adversarial_input is not None:
                    attacked_features, _normalized_delta = (
                        adversarial_input_examples(
                            model,
                            features,
                            targets,
                            weights,
                            target_std=target_std,
                            epsilon_rms=adversarial_epsilon_rms,
                            steps=adversarial_steps,
                            step_size_rms=adversarial_step_size_rms,
                        )
                    )
                    adversarial_loss = train_loss(
                        model(attacked_features), targets, target_std
                    )
                    loss = (
                        (1.0 - adversarial_weight) * clean_loss
                        + adversarial_weight * adversarial_loss
                    )
                else:
                    adversarial_loss = clean_loss
                    loss = clean_loss
                loss.backward()
                if sam is not None:
                    perturbations = sam_perturb_parameters(model, sam_rho)
                    try:
                        for optimizer in optimizers:
                            optimizer.zero_grad(set_to_none=True)
                        perturbed_clean_loss = train_loss(
                            model(features), targets, target_std
                        )
                        perturbed_attacked_features, _perturbed_delta = (
                            adversarial_input_examples(
                                model,
                                features,
                                targets,
                                weights,
                                target_std=target_std,
                                epsilon_rms=adversarial_epsilon_rms,
                                steps=adversarial_steps,
                                step_size_rms=adversarial_step_size_rms,
                            )
                        )
                        perturbed_adversarial_loss = train_loss(
                            model(perturbed_attacked_features),
                            targets,
                            target_std,
                        )
                        perturbed_loss = (
                            (1.0 - adversarial_weight) * perturbed_clean_loss
                            + adversarial_weight * perturbed_adversarial_loss
                        )
                        perturbed_loss.backward()
                    finally:
                        sam_restore_parameters(perturbations)
                clip_grad_norm_(model.parameters(), gradient_clip, foreach=True)
                for optimizer in optimizers:
                    optimizer.step()
                batch_count = int(targets.numel())
                epoch_clean_numerator += float(clean_loss.detach()) * batch_count
                epoch_adversarial_numerator += (
                    float(adversarial_loss.detach()) * batch_count
                )
                epoch_example_count += batch_count
                global_step += 1
            train_metrics = evaluate(
                model, dataset, "train", batch_size=evaluation_batch_size,
                target_std=target_std, device=device,
            )
            validation_metrics = evaluate(
                model, dataset, "validation", batch_size=evaluation_batch_size,
                target_std=target_std, device=device,
            )
            payload = checkpoint_payload(
                model, optimizers, epoch=epoch, global_step=global_step,
                train=train_metrics, validation=validation_metrics,
                parameter_count=parameter_count, plan_hash=plan_hash,
                dataset_hash=dataset_hash,
            )
            wait_for_checkpoint_storage(
                repo, reporter, plan["id"], epoch=epoch
            )
            candidates = {
                "train-mse": float(train_metrics["normalizedMse"]),
                "train-correlation": float(train_metrics["correlation"]),
                "validation-mse": float(validation_metrics["normalizedMse"]),
                "validation-correlation": float(validation_metrics["correlation"]),
            }
            for policy, score in candidates.items():
                improved = score < policy_scores[policy] if policy.endswith("mse") else score > policy_scores[policy]
                if improved:
                    policy_scores[policy] = score
                    # Selection checkpoints are inference artifacts. Optimizer
                    # state is only needed by `last` for exact resumption and
                    # is much larger than this model's weights.
                    selection_payload = {
                        key: value for key, value in payload.items()
                        if key != "optimizers"
                    }
                    save_torch_checkpoint(
                        selection_payload, policy_files[policy]
                    )
            best_train_score = min(
                best_train_score, float(train_metrics["normalizedMse"])
            )
            payload["policyScores"] = policy_scores
            payload["bestTrainScore"] = best_train_score
            save_torch_checkpoint(payload, last_file)
            event = {
                "event": "minute-return-epoch",
                "epoch": epoch,
                "epochs": epochs,
                "seconds": time.monotonic() - started,
                "globalStep": global_step,
                "train": train_metrics,
                "validation": validation_metrics,
                "bestTrainScore": best_train_score,
                "bestValidationScore": policy_scores["validation-mse"],
                "parameterCount": parameter_count,
                "featureCount": dataset.feature_count,
                "robustTraining": {
                    "samRho": sam_rho,
                    "adversarialInputEpsilonRms": adversarial_epsilon_rms,
                    "adversarialInputWeight": adversarial_weight,
                    "cleanNormalizedMse": (
                        epoch_clean_numerator / epoch_example_count
                    ),
                    "adversarialNormalizedMse": (
                        epoch_adversarial_numerator / epoch_example_count
                    ),
                },
            }
            if not args.recover_validation_selections:
                reporter.emit(event)
            reporter.status("training", planId=plan["id"], latest=event)
        policies: dict[str, dict] = {}
        for policy, file in policy_files.items():
            saved = load_torch_checkpoint(file, map_location=device, weights_only=False)
            model.load_state_dict(saved["model"])
            split_metrics = {
                split: evaluate(
                    model, dataset, split, batch_size=evaluation_batch_size,
                    target_std=target_std, device=device,
                )
                for split in ("train", "validation", "test")
            }
            policies[policy] = {
                "epoch": int(saved["epoch"]),
                "selectionScore": policy_scores[policy],
                **split_metrics,
                "checkpoint": str(file.relative_to(repo)),
            }
        comparison = {
            "contract": "next-return-checkpoint-selection-comparison-v1",
            "policies": policies,
        }
        atomic_json(comparison, run_root / "state/checkpoint-selection-comparison.json")
        selected = policies["validation-mse"]
        result = {
            "planId": plan["id"],
            "planSha256": plan_hash,
            "datasetSha256": dataset_hash,
            "runnerContract": RUNNER_CONTRACT,
            "examples": dataset.logical_count("train"),
            "featureCount": dataset.feature_count,
            "parameterCount": parameter_count,
            "trainableParameterCount": trainable_parameter_count,
            "bestEpoch": selected["epoch"],
            "bestValidationScore": selected["selectionScore"],
            "train": selected["train"],
            "validation": selected["validation"],
            "test": selected["test"],
            "checkpoint": selected["checkpoint"],
            "dataset": {
                "id": dataset.manifest.get("id"),
                "contract": dataset.manifest.get(
                    "contract", dataset.manifest.get("purpose")
                ),
                "targetFilter": dataset.manifest.get(
                    "targetFilter",
                    dataset.manifest.get("selectionModes", {}).get("nonzeroRule"),
                ),
                "includedSources": dataset.manifest.get(
                    "includedSources",
                    [dataset.manifest.get("featureStorage", {}).get("matrix")],
                ),
                "omittedForCoverage": dataset.manifest.get(
                    "omittedForCoverage", []
                ),
            },
        }
        atomic_json(result, run_root / "state/result.json")
        reporter.emit({"event": "minute-return-complete", **result})
        reporter.status("complete", planId=plan["id"], latest=result)
    except KeyboardInterrupt:
        reporter.status("paused", planId=plan["id"], message="Interrupted")
        raise
    except Exception as error:
        reporter.status("failed", planId=plan["id"], error=f"{type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    main()
