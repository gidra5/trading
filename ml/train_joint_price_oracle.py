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
from typing import Callable, Iterator

import numpy as np
import torch
from torch import Tensor, nn
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
    architecture_contract_with_policy_decoder,
    joint_price_oracle_objective,
    joint_price_oracle_policy_only_objective,
    parameter_count,
)
from joint_price_oracle_variants import (
    LEARNED_AGGREGATE_PATCH_MIXER_CONTRACT,
    LONG_CONTEXT_PATCH_MIXER_CONTRACT,
    MINUTE_RETURN_MLP_CONTRACT,
    MINUTE_SEQUENCE_BOUNDARY_MA_TCN_CONTRACT,
    MINUTE_SEQUENCE_BOUNDARY_LONG_TCN_CONTRACT,
    MINUTE_SEQUENCE_BOUNDARY_PROTOTYPE_MIXTURE_CONTRACT,
    MINUTE_SEQUENCE_BOUNDARY_TCN_CONTRACT,
    MINUTE_SEQUENCE_BOUNDARY_TCN_POLICY_DROPOUT_CONTRACT,
    MINUTE_SEQUENCE_TCN_CONTRACT,
    PATCH_TRANSFORMER_CONTRACT,
    RESIDUAL_MIXER_CONTRACT,
    TCN_CONTRACT,
    build_variant_model,
)
from joint_price_oracle_sequence import (
    DECISION_INTERVAL_MS,
    boundary_sequence_core_alignment,
    sequence_core_alignment,
)
from joint_price_oracle_actions import (
    SwitchBalancedActionLossWeights,
    actionable_policy_metrics_numpy,
    exact_state_actionable_policy_metrics_numpy,
    greedy_teacher_rollout_numpy,
    resolve_execution_policy_config,
    self_conditioned_current_exposures_tensor,
    switch_balanced_action_objective,
    teacher_actions_at_current_exposures_numpy,
)
from joint_price_oracle_teacher_trace import (
    ExactTraceExposureProvider,
    build_exact_trace_exposure_provider,
)


SECOND_MS = 1_000
DAY_SECONDS = 86_400
DAY_MS = DAY_SECONDS * SECOND_MS
EXACT_TEACHER_TRACE_DIRECTORY = Path(
    "data/training/derived/joint-price-oracle/teacher-traces"
)
METRIC_NAMES = (
    "loss",
    "crossEntropy",
    "klDivergence",
    "conditionedCrossEntropy",
    "conditionedKlDivergence",
    "probabilityMse",
    "forecastLoss",
    "nextMovementRmse",
    "directionAccuracy",
    "softLayerNorm",
    "softLayerNormMeanPenalty",
    "softLayerNormVariancePenalty",
)
ACTION_METRIC_NAMES = (
    "actionLoss",
    "actionHardCrossEntropy",
    "actionRankingLoss",
    "actionDirectionCrossEntropy",
    "actionConditionalKlDivergence",
    "actionHardAccuracy",
    "actionSwitchRate",
    "actionMeanExampleWeight",
    "mixedActionLoss",
    "selfActionLoss",
    "selfActionHardCrossEntropy",
    "selfActionRankingLoss",
    "selfActionDirectionCrossEntropy",
    "selfActionConditionalKlDivergence",
    "selfActionHardAccuracy",
    "selfActionSwitchRate",
    "selfActionMeanExampleWeight",
)
DATA_CONTRACT = (
    "causal-1s-close-context-ending-at-minute-t-future-closes-t-plus-1-"
    "through-1h-verified-oracle-policy-at-t-hold-60-delay-60-v2"
)
LEGACY_UNFINGERPRINTED_PLAN_IDS = frozenset({
    "joint-price-oracle-tide-rlinear-dlinear-v2-1h-1m-1m",
    "joint-price-oracle-decision-conditioned-v3-1h-1m-1m",
})
DISPOSABLE_SMOKE_DIRECTORY = "disposable-smoke"
SEQUENCE_REUSE_TRAINING_CONTRACT = (
    "minute-sequence-core-with-rf-halo-padded-batch-raw-kl-v1"
)
PACKED_SEQUENCE_REUSE_TRAINING_CONTRACT = (
    "minute-sequence-core-with-rf-halo-contiguous-packed-raw-kl-v2"
)


def uses_packed_sequence_cores(configuration: dict | None) -> bool:
    return configuration is not None and configuration.get("contract") \
        == PACKED_SEQUENCE_REUSE_TRAINING_CONTRACT

Batch = (
    tuple[Tensor, Tensor]
    | tuple[Tensor, Tensor, Tensor]
    | tuple[Tensor, Tensor, Tensor, Tensor]
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


@dataclass(frozen=True)
class SequenceCoreTargetSlice:
    segment: CausalSegment
    local_start: int
    count: int


@dataclass(frozen=True)
class SequenceCoreSpan:
    slices: tuple[SequenceCoreTargetSlice, ...]
    prediction_time_start: int
    count: int
    step_ms: int


@dataclass(frozen=True)
class SequenceCoreGroup:
    spans: tuple[SequenceCoreSpan, ...]
    count: int


@dataclass
class MetricAccumulator:
    examples: int = 0
    totals: dict[str, Tensor] | None = None
    metric_names: tuple[str, ...] | None = None

    def add(self, metrics: dict[str, Tensor], count: int) -> None:
        if self.totals is None:
            missing = tuple(name for name in METRIC_NAMES if name not in metrics)
            if missing:
                raise ValueError(f"training metrics are missing: {missing}")
            self.metric_names = METRIC_NAMES + tuple(
                name for name in ACTION_METRIC_NAMES if name in metrics
            )
            self.totals = {
                name: torch.zeros(
                    (),
                    device=metrics[name].device,
                    dtype=torch.float64,
                )
                for name in self.metric_names
            }
        elif self.metric_names is None \
                or set(metrics) != set(self.metric_names):
            raise ValueError("training metric names changed within one pass")
        self.examples += count
        assert self.metric_names is not None
        for name in self.metric_names:
            self.totals[name].add_(
                metrics[name].detach().to(dtype=torch.float64),
                alpha=count,
            )

    def result(self) -> dict[str, float]:
        if self.examples < 1 or self.totals is None:
            raise RuntimeError("cannot finalize empty metrics")
        assert self.metric_names is not None
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
        source: Iterator[Batch],
        prefetch_batches: int,
    ) -> None:
        self.source = source
        self.queue: queue.Queue[
            Batch | BatchIteratorFailure | None
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
        value: Batch
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

    def __next__(self) -> Batch:
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
        value = finite_metadata_value({"at": iso_now(), **event})
        encoded = json.dumps(
            value,
            separators=(",", ":"),
            allow_nan=False,
        )
        with self.log_file.open("a", encoding="utf-8") as output:
            output.write(encoded + "\n")
        print(encoded, flush=True)

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
        teacher_current_exposures: dict[Path, Tensor] | None = None,
        include_future_closes: bool = True,
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
        self.teacher_current_exposures = teacher_current_exposures
        self.include_future_closes = bool(include_future_closes)
        if not self.include_future_closes \
                and self.teacher_current_exposures is not None:
            raise ValueError(
                "policy-only data cannot include action-objective teacher state"
            )

    def count(self, split: str) -> int:
        return sum(segment.count for segment in self.segments[split])

    def batch_count(self, split: str, batch_size: int) -> int:
        return sum(
            math.ceil(segment.count / batch_size)
            for segment in self.segments[split]
        )

    def sequence_core_batch_count(
        self,
        split: str,
        core_rows: int,
        *,
        pack_contiguous_runs: bool = False,
    ) -> int:
        """Count contiguous core chunks without changing target weighting."""
        if core_rows < 1:
            raise ValueError("sequence core rows must be positive")
        if pack_contiguous_runs:
            return math.ceil(self.count(split) / core_rows)
        return sum(
            math.ceil(segment.count / core_rows)
            for segment in self.segments[split]
        )

    def _packed_sequence_core_groups(
        self,
        split: str,
        core_rows: int,
    ) -> list[SequenceCoreGroup]:
        """Pack causal spans into uniformly row-weighted optimizer groups.

        Adjacent segments may share one sequence even when their targets live
        in different immutable files.  True gaps start an independent batch
        dimension, so no synthetic history crosses missing market data.  A
        group may contain multiple independent spans, but always contains at
        most ``core_rows`` valid target rows in total.
        """
        if core_rows < 1:
            raise ValueError("sequence core rows must be positive")
        segments = list(self.segments[split])
        if not segments:
            return []
        for previous, following in zip(segments, segments[1:]):
            if following.prediction_time_start <= previous.prediction_time_end:
                raise ValueError(
                    "sequence core segments must be chronological and "
                    "non-overlapping"
                )

        runs: list[list[CausalSegment]] = []
        for segment in segments:
            if segment.count < 1:
                raise ValueError("sequence core segments must be non-empty")
            if runs:
                previous = runs[-1][-1]
                contiguous = (
                    segment.split == previous.split
                    and segment.step_ms == previous.step_ms
                    and segment.prediction_time_start
                    == previous.prediction_time_end + previous.step_ms
                )
            else:
                contiguous = False
            if not contiguous:
                runs.append([])
            runs[-1].append(segment)

        groups: list[SequenceCoreGroup] = []
        pending_spans: list[SequenceCoreSpan] = []
        pending_rows = 0
        for run in runs:
            segment_index = 0
            local_start = 0
            run_rows = sum(segment.count for segment in run)
            consumed_run_rows = 0
            while consumed_run_rows < run_rows:
                take = min(
                    core_rows - pending_rows,
                    run_rows - consumed_run_rows,
                )
                remaining = take
                slices: list[SequenceCoreTargetSlice] = []
                while remaining:
                    segment = run[segment_index]
                    available = segment.count - local_start
                    slice_rows = min(available, remaining)
                    slices.append(SequenceCoreTargetSlice(
                        segment=segment,
                        local_start=local_start,
                        count=slice_rows,
                    ))
                    local_start += slice_rows
                    remaining -= slice_rows
                    if local_start == segment.count:
                        segment_index += 1
                        local_start = 0
                first = slices[0]
                pending_spans.append(SequenceCoreSpan(
                    slices=tuple(slices),
                    prediction_time_start=(
                        first.segment.prediction_time_start
                        + first.local_start * first.segment.step_ms
                    ),
                    count=take,
                    step_ms=first.segment.step_ms,
                ))
                pending_rows += take
                consumed_run_rows += take
                if pending_rows == core_rows:
                    groups.append(SequenceCoreGroup(
                        spans=tuple(pending_spans),
                        count=pending_rows,
                    ))
                    pending_spans = []
                    pending_rows = 0
        if pending_rows:
            groups.append(SequenceCoreGroup(
                spans=tuple(pending_spans),
                count=pending_rows,
            ))
        expected_rows = self.count(split)
        if sum(group.count for group in groups) != expected_rows:
            raise RuntimeError("packed sequence cores lost target rows")
        return groups

    def _load_packed_sequence_core_group(
        self,
        group: SequenceCoreGroup,
        receptive_field_minutes: int,
    ) -> Batch:
        inputs: list[Tensor] = []
        targets_by_span: list[Tensor] = []
        for span in group.spans:
            if span.step_ms != DECISION_INTERVAL_MS:
                raise ValueError(
                    "sequence cores require adjacent one-minute targets"
                )
            alignment = boundary_sequence_core_alignment(
                span.prediction_time_start,
                span.count,
                receptive_field_minutes,
            )
            close_range = self.close_cache.range(
                alignment.input_close_time_start,
                alignment.input_close_time_end,
            )
            if close_range.shape != (alignment.close_count,):
                raise RuntimeError(
                    "packed sequence core close range is not contiguous"
                )
            inputs.append(torch.from_numpy(
                np.asarray(close_range, dtype=np.float32).copy()
            ).view(alignment.close_count, 1))

            target_slices: list[Tensor] = []
            for target_slice in span.slices:
                segment = target_slice.segment
                target_start = (
                    segment.target_row_offset + target_slice.local_start
                )
                target_end = target_start + target_slice.count
                if target_start < 0 \
                        or target_end > self.target_cache.rows_per_file:
                    raise IndexError(
                        "packed sequence core targets leave their UTC day"
                    )
                target_slices.append(self.target_cache.load(
                    segment.target_file
                )[target_start:target_end])
            span_targets = (
                target_slices[0]
                if len(target_slices) == 1
                else torch.cat(target_slices, dim=0)
            )
            if span_targets.shape != (
                span.count,
                self.target_cache.action_count,
            ):
                raise RuntimeError(
                    "packed sequence core targets are not aligned"
                )
            targets_by_span.append(span_targets)

        maximum_close_count = max(value.shape[0] for value in inputs)
        maximum_core_count = max(value.shape[0] for value in targets_by_span)
        batch_size = len(inputs)
        input_tensor = torch.empty(
            (batch_size, maximum_close_count, 1),
            dtype=torch.float32,
        )
        target_tensor = torch.zeros(
            (
                batch_size,
                maximum_core_count,
                self.target_cache.action_count,
            ),
            dtype=targets_by_span[0].dtype,
        )
        target_mask = torch.zeros(
            (batch_size, maximum_core_count),
            dtype=torch.bool,
        )
        for index, (input_values, target_values) in enumerate(zip(
            inputs,
            targets_by_span,
        )):
            input_count = input_values.shape[0]
            target_count = target_values.shape[0]
            input_tensor[index, :input_count] = input_values
            input_tensor[index, input_count:] = input_values[-1]
            target_tensor[index, :target_count] = target_values
            target_mask[index, :target_count] = True
        if int(target_mask.sum()) != group.count:
            raise RuntimeError("packed sequence core mask lost target rows")
        if self.pin_memory:
            input_tensor = input_tensor.pin_memory()
            target_tensor = target_tensor.pin_memory()
            target_mask = target_mask.pin_memory()
        return input_tensor, target_tensor, target_mask

    def iter_sequence_core_batches(
        self,
        split: str,
        core_rows: int,
        *,
        receptive_field_minutes: int,
        shuffle: bool,
        seed: int,
        maximum_batches: int | None = None,
        pack_contiguous_runs: bool = False,
    ) -> Iterator[Batch]:
        """Yield one halo-padded contiguous policy core per microbatch.

        Targets remain in their immutable source rows.  One shared close range
        starts at the first target's exact pre-return boundary and ends at the
        final target timestamp; no future close is requested.
        """
        if core_rows < 1 or receptive_field_minutes < 1:
            raise ValueError("sequence core dimensions must be positive")
        if self.include_future_closes \
                or self.teacher_current_exposures is not None:
            raise ValueError(
                "sequence core reuse supports direct policy-only data"
            )
        generator = random.Random(seed)
        if pack_contiguous_runs:
            groups = self._packed_sequence_core_groups(split, core_rows)
            if shuffle:
                generator.shuffle(groups)
            for emitted, group in enumerate(groups, start=1):
                yield self._load_packed_sequence_core_group(
                    group,
                    receptive_field_minutes,
                )
                if maximum_batches is not None \
                        and emitted >= maximum_batches:
                    return
            return
        segments = list(self.segments[split])
        if shuffle:
            generator.shuffle(segments)
        emitted = 0
        for segment in segments:
            if segment.step_ms != DECISION_INTERVAL_MS:
                raise ValueError(
                    "sequence cores require adjacent one-minute targets"
                )
            starts = list(range(0, segment.count, core_rows))
            if shuffle:
                generator.shuffle(starts)
            targets = self.target_cache.load(segment.target_file)
            for local_start in starts:
                count = min(core_rows, segment.count - local_start)
                prediction_time_start = (
                    segment.prediction_time_start
                    + local_start * segment.step_ms
                )
                alignment = boundary_sequence_core_alignment(
                    prediction_time_start,
                    count,
                    receptive_field_minutes,
                )
                close_range = self.close_cache.range(
                    alignment.input_close_time_start,
                    alignment.input_close_time_end,
                )
                if close_range.shape != (alignment.close_count,):
                    raise RuntimeError(
                        "sequence core close range is not contiguous"
                    )
                target_start = segment.target_row_offset + local_start
                target_end = target_start + count
                if target_start < 0 \
                        or target_end > self.target_cache.rows_per_file:
                    raise IndexError("sequence core targets leave their UTC day")
                input_tensor = torch.from_numpy(
                    np.asarray(close_range, dtype=np.float32).copy()
                ).view(1, alignment.close_count, 1)
                target_tensor = targets[target_start:target_end].unsqueeze(0)
                if self.pin_memory:
                    input_tensor = input_tensor.pin_memory()
                yield input_tensor, target_tensor
                emitted += 1
                if maximum_batches is not None \
                        and emitted >= maximum_batches:
                    return

    def iter_batches(
        self,
        split: str,
        batch_size: int,
        *,
        shuffle: bool,
        seed: int,
        maximum_batches: int | None = None,
    ) -> Iterator[Batch]:
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
                    + (
                        self.forecast_horizon * SECOND_MS
                        if self.include_future_closes else 0
                    )
                )
                close_range = self.close_cache.range(
                    first_close_time,
                    last_close_time,
                )
                sample_step_seconds = segment.step_ms // SECOND_MS
                future_windows = None
                if self.include_future_closes:
                    close_windows, future_windows = causal_close_windows(
                        close_range,
                        count,
                        self.context_length,
                        self.forecast_horizon,
                        sample_step_seconds=sample_step_seconds,
                    )
                else:
                    close_windows = causal_input_close_windows(
                        close_range,
                        count,
                        self.context_length,
                        sample_step_seconds=sample_step_seconds,
                    )
                target_start = segment.target_row_offset + local_start
                target_end = target_start + count
                if target_start < 0 \
                        or target_end > self.target_cache.rows_per_file:
                    raise IndexError("oracle target rows leave their UTC day")
                input_tensor = torch.from_numpy(
                    close_windows[:, :, None]
                )
                future_tensor = (
                    torch.from_numpy(future_windows[:, :, None])
                    if future_windows is not None else None
                )
                target_tensor = targets[target_start:target_end]
                teacher_exposure_tensor = None
                if self.teacher_current_exposures is not None:
                    teacher_rows = self.teacher_current_exposures.get(
                        segment.target_file
                    )
                    if teacher_rows is None:
                        raise KeyError(
                            "teacher exposure rows are missing for "
                            f"{segment.target_file}"
                        )
                    teacher_exposure_tensor = teacher_rows[
                        target_start:target_end
                    ]
                    if teacher_exposure_tensor.shape != (count,) \
                            or not bool(torch.isfinite(
                                teacher_exposure_tensor
                            ).all()):
                        raise ValueError(
                            "teacher exposure rows do not align with targets"
                        )
                if self.pin_memory:
                    input_tensor = input_tensor.pin_memory()
                    if future_tensor is not None:
                        future_tensor = future_tensor.pin_memory()
                if future_tensor is None:
                    yield input_tensor, target_tensor
                elif teacher_exposure_tensor is None:
                    yield input_tensor, future_tensor, target_tensor
                else:
                    yield (
                        input_tensor,
                        future_tensor,
                        target_tensor,
                        teacher_exposure_tensor,
                    )
                emitted += 1
                if maximum_batches is not None \
                        and emitted >= maximum_batches:
                    return


def precompute_teacher_current_exposures(
    segments_by_split: dict[str, list[CausalSegment]],
    target_cache: OracleTargetCache,
    action_grid: np.ndarray,
    *,
    splits: tuple[str, ...] = ("train", "validation"),
    friction: float,
    temperature: float,
    initial_exposure: float = 0.0,
    pin_memory: bool = False,
    switch_statistics: dict[str, dict[str, int | float]] | None = None,
    execution_policy: dict | None = None,
    teacher_current_exposure_provider: (
        Callable[[str, CausalSegment], np.ndarray] | None
    ) = None,
    teacher_current_exposure_providers: (
        dict[str, Callable[[str, CausalSegment], np.ndarray]] | None
    ) = None,
) -> dict[Path, Tensor]:
    """Build chronological teacher state indexed by immutable target row.

    Exposure carries across ordinary daily files when their timestamps are
    continuous.  Each split and each real market-history gap starts from the
    supplied initial exposure.  An optional simulator provider can instead
    supply exact marked current exposures for every timestamp, converted from
    execution exposure by dividing by the execution/native scale.  The segment
    exposes timestamp, target file, and row offset for trace lookup.  The
    resulting file/row lookup remains stable when segments and batches are
    later shuffled.
    """
    grid = np.asarray(action_grid, dtype=np.float64)
    if grid.ndim != 1 or grid.shape != (target_cache.action_count,):
        raise ValueError("teacher action grid does not match target columns")
    if not math.isfinite(initial_exposure):
        raise ValueError("teacher initial exposure must be finite")
    rows_by_file: dict[Path, np.ndarray] = {}
    if not splits or len(set(splits)) != len(splits):
        raise ValueError("teacher rollout splits must be unique and non-empty")
    if teacher_current_exposure_provider is not None \
            and teacher_current_exposure_providers is not None:
        raise ValueError(
            "use either one teacher provider or split-specific providers"
        )
    if teacher_current_exposure_providers is not None:
        unknown_provider_splits = (
            set(teacher_current_exposure_providers) - set(splits)
        )
        if unknown_provider_splits:
            raise ValueError(
                "teacher providers contain unloaded splits: "
                f"{sorted(unknown_provider_splits)}"
            )
    unknown_splits = set(splits) - set(segments_by_split)
    if unknown_splits:
        raise ValueError(
            f"unknown teacher rollout splits: {sorted(unknown_splits)}"
        )
    for split in splits:
        split_segments = segments_by_split.get(split)
        if not split_segments:
            raise ValueError(f"teacher rollout split is empty: {split}")
        previous: CausalSegment | None = None
        current_exposure = float(initial_exposure)
        split_decisions = 0
        split_switches = 0
        split_provider = (
            teacher_current_exposure_providers.get(split)
            if teacher_current_exposure_providers is not None
            else teacher_current_exposure_provider
        )
        for segment in split_segments:
            if previous is not None \
                    and segment.prediction_time_start \
                    <= previous.prediction_time_end:
                raise ValueError(
                    "teacher rollout segments must be chronological and "
                    "non-overlapping"
                )
            continuous = (
                previous is not None
                and segment.step_ms == previous.step_ms
                and segment.prediction_time_start
                == previous.prediction_time_end + previous.step_ms
            )
            if not continuous:
                current_exposure = float(initial_exposure)
            target_start = segment.target_row_offset
            target_end = target_start + segment.count
            if target_start < 0 or target_end > target_cache.rows_per_file:
                raise IndexError("teacher target rows leave their UTC day")
            target_rows = target_cache.load(segment.target_file)[
                target_start:target_end
            ].detach().cpu().numpy()
            if split_provider is None:
                rollout = greedy_teacher_rollout_numpy(
                    target_rows,
                    grid,
                    initial_exposure=current_exposure,
                    friction=friction,
                    temperature=temperature,
                    execution_policy=execution_policy,
                )
                current_rows = rollout.current_exposures
            else:
                current_rows = np.asarray(
                    split_provider(split, segment),
                    dtype=np.float64,
                )
                if current_rows.shape != (segment.count,) \
                        or not np.isfinite(current_rows).all():
                    raise ValueError(
                        "exact teacher exposure provider returned incompatible "
                        f"rows for {segment.target_file}"
                    )
                rollout = teacher_actions_at_current_exposures_numpy(
                    target_rows,
                    grid,
                    current_rows,
                    friction=friction,
                    temperature=temperature,
                    execution_policy=execution_policy,
                )
            file_rows = rows_by_file.setdefault(
                segment.target_file,
                np.full(
                    target_cache.rows_per_file,
                    np.nan,
                    dtype=np.float32,
                ),
            )
            destination = file_rows[target_start:target_end]
            if bool(np.isfinite(destination).any()):
                raise ValueError("teacher target rows overlap")
            destination[:] = current_rows.astype(
                np.float32,
                copy=False,
            )
            current_exposure = float(rollout.target_exposures[-1])
            split_decisions += int(rollout.switch_labels.size)
            split_switches += int(rollout.switch_labels.sum())
            previous = segment
        if switch_statistics is not None:
            switch_statistics[split] = {
                "decisions": split_decisions,
                "switches": split_switches,
                "switchFraction": split_switches / split_decisions,
            }
    result = {
        file: torch.from_numpy(rows)
        for file, rows in rows_by_file.items()
    }
    if pin_memory:
        result = {
            file: rows.pin_memory()
            for file, rows in result.items()
        }
    return result


def load_exact_state_trace_providers(
    repo_root: Path,
    target_root: Path,
    segments_by_split: dict[str, list[CausalSegment]],
    action_objective: dict | None,
    *,
    splits: tuple[str, ...],
    allow_test: bool = False,
) -> tuple[
    dict[str, ExactTraceExposureProvider],
    dict[str, dict[str, object]],
]:
    """Load configured exact traces and describe every requested split.

    Unconfigured splits remain explicitly labelled as chronological surrogate
    state.  A configured test trace is never opened unless ``allow_test`` is
    true and test is among ``splits``.
    """
    if action_objective is None:
        return {}, {}
    declarations = action_objective.get("exactStateTraces", {})
    execution_policy = action_objective.get("executionPolicy")
    providers: dict[str, ExactTraceExposureProvider] = {}
    provenance: dict[str, dict[str, object]] = {}
    trace_root = (repo_root / EXACT_TEACHER_TRACE_DIRECTORY).resolve()
    for split in splits:
        declaration = declarations.get(split)
        if declaration is None:
            provenance[split] = {
                "kind": (
                    "capped-target-rollout-surrogate-v2"
                    if execution_policy is not None
                    else "raw-modal-rollout-surrogate-v1"
                ),
                "exactSimulatorState": False,
                "coordinateSystem": "native-oracle-grid",
                "executionPolicy": execution_policy,
                "warning": (
                    "Targets are carried between decisions; marked exposure "
                    "drift is not represented for this split."
                ),
            }
            continue
        if split == "test" and not allow_test:
            raise PermissionError(
                "loading an exact test-state trace requires explicit test access"
            )
        trace_file = require_under(
            resolve(repo_root, Path(declaration["path"])),
            trace_root,
            f"actionObjective.exactStateTraces.{split}.path",
        )
        dates = _target_dates_for_segments(segments_by_split[split])
        provider = build_exact_trace_exposure_provider(
            trace_file,
            target_root,
            legacy_maximum_leverage=float(
                declaration["maximumLeverage"]
            ),
            allow_test=allow_test,
            expected_sha256=declaration["sha256"],
            expected_split=split,
            expected_schema_version=int(declaration["schemaVersion"]),
            expected_dates=dates,
            expected_row_count=int(declaration["rows"]),
            expected_execution_policy=execution_policy,
        )
        usable_rows = 0
        for segment in segments_by_split[split]:
            rows = provider(split, segment)
            if rows.shape != (segment.count,):
                raise RuntimeError("exact trace segment coverage is inconsistent")
            usable_rows += int(rows.size)
        providers[split] = provider
        provenance[split] = {
            "kind": "exact-simulator-current-exposure-trace-v1",
            "exactSimulatorState": True,
            "coordinateSystem": "native-oracle-grid",
            "sourceFile": str(provider.source_file),
            "configuredPath": declaration["path"],
            "sha256": provider.source_sha256,
            "schemaVersion": provider.schema_version,
            "dates": list(provider.dates),
            "rows": provider.row_count,
            "usableRows": usable_rows,
            "purgedRows": provider.row_count - usable_rows,
            "maximumLeverage": provider.maximum_leverage,
            "maximumLeverageSource": provider.maximum_leverage_source,
            "executionScale": provider.execution_scale,
            "executionPolicy": provider.execution_policy,
            "oracleContractHash": provider.oracle_contract_hash,
            "oracleDatasetFingerprint": (
                provider.oracle_dataset_fingerprint
            ),
        }
    return providers, provenance


def _target_dates_for_segments(
    segments: list[CausalSegment],
) -> tuple[str, ...]:
    if not segments:
        raise ValueError("exact teacher trace split is empty")
    dates: list[str] = []
    for segment in segments:
        date = segment.target_file.stem
        if not dates or date != dates[-1]:
            dates.append(date)
    if dates != sorted(set(dates)):
        raise ValueError("exact teacher trace target dates are not ordered")
    return tuple(dates)


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
        help=(
            "Bound batches per split for a disposable smoke run. Requires "
            "--disposable-smoke and a runDir below runs/disposable-smoke."
        ),
    )
    parser.add_argument(
        "--disposable-smoke",
        action="store_true",
        help=(
            "Permit --maximum-batches only for an isolated, non-production "
            "run directory below data/training/runs/disposable-smoke."
        ),
    )
    parser.add_argument(
        "--stop-after-epoch",
        type=int,
        default=None,
        help=(
            "Pause after validation and the durable checkpoint for this "
            "completed epoch."
        ),
    )
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help=(
            "Explicitly evaluate the selected best checkpoint on the sealed "
            "test split after training completes."
        ),
    )
    parser.add_argument(
        "--runtime-cache-days",
        type=int,
        default=None,
        help=(
            "Non-semantic in-memory override for decoded close and target "
            "cache capacities. It does not change checkpoint compatibility."
        ),
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
    architecture_contract = architecture_contract_for_model_config(
        model_config
    )
    training = plan["training"]
    run_dir = require_under(
        resolve(repo_root, Path(plan["runDir"])),
        storage.runs,
        "runDir",
    )
    validate_experiment_arguments(arguments, run_dir, storage.runs)
    reporter = RunReporter(run_dir, plan["id"])
    ensure_run_is_not_active(reporter.status_file)
    resolved_training = resolve_training_config(training)
    policy_only = bool(resolved_training.get("policyOnly", False))
    sequence_core_training = resolved_training.get("sequenceCoreTraining")
    pack_sequence_cores = uses_packed_sequence_cores(
        sequence_core_training
    )
    training_config_fingerprint = configuration_fingerprint(
        resolved_training
    )
    plan_snapshot = run_dir / "plans" / "resolved-plan.json"
    reporter.status(
        "preparing",
        planFile=str(plan_file),
        planSnapshot=str(plan_snapshot),
        trainingConfigFingerprint=training_config_fingerprint,
        stopAfterEpoch=arguments.stop_after_epoch,
        evaluateTest=arguments.evaluate_test,
        disposableSmoke=arguments.disposable_smoke,
        runtimeCacheDays=arguments.runtime_cache_days,
    )

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
    action_objective = resolved_training.get("actionObjective")
    pin_memory = (
        training["device"] == "cuda" and torch.cuda.is_available()
    )
    target_contract = target_manifest["contract"]
    action_grid_values = np.asarray(
        target_contract["usableGrid"],
        dtype=np.float64,
    )
    policy_friction = float(target_contract["options"]["friction"])
    policy_temperature = float(target_contract["options"]["temperature"])
    teacher_current_exposures = None
    teacher_switch_statistics: dict[
        str, dict[str, int | float]
    ] = {}
    source_switch_fraction = None
    teacher_state_provenance: dict[str, dict[str, object]] | None = None
    exact_state_providers: dict[str, ExactTraceExposureProvider] = {}
    if action_objective is not None:
        execution_policy = action_objective.get("executionPolicy")
        teacher_splits = (
            ("train", "validation", "test")
            if arguments.evaluate_test
            else ("train", "validation")
        )
        (
            exact_state_providers,
            teacher_state_provenance,
        ) = load_exact_state_trace_providers(
            repo_root,
            target_root,
            segments,
            action_objective,
            splits=teacher_splits,
            allow_test=arguments.evaluate_test,
        )
        teacher_current_exposures = precompute_teacher_current_exposures(
            segments,
            OracleTargetCache(
                int(training.get("targetCacheDays", 3)),
                rows_per_file=1_440,
                action_count=int(model_config["actionCount"]),
                pin_memory=False,
            ),
            action_grid_values,
            splits=teacher_splits,
            friction=policy_friction,
            temperature=policy_temperature,
            pin_memory=pin_memory,
            switch_statistics=teacher_switch_statistics,
            execution_policy=execution_policy,
            teacher_current_exposure_providers=exact_state_providers,
        )
        if action_objective.get("switchWeighting", "batch") == "global":
            source_switch_fraction = float(
                teacher_switch_statistics["train"]["switchFraction"]
            )
            if not 0 < source_switch_fraction < 1:
                raise ValueError(
                    "global action switch weighting requires both switch and "
                    "hold examples in the training split"
                )
    runtime_cache_days = getattr(arguments, "runtime_cache_days", None)
    close_cache_days = (
        runtime_cache_days
        if runtime_cache_days is not None
        else int(training.get("closeCacheDays", 10))
    )
    target_cache_days = (
        runtime_cache_days
        if runtime_cache_days is not None
        else int(training.get("targetCacheDays", 3))
    )
    dataset = CausalOracleDataset(
        history_root,
        segments,
        int(model_config["contextLength"]),
        int(model_config["forecastHorizon"]),
        target_rows_per_file=1_440,
        action_count=int(model_config["actionCount"]),
        close_cache_days=close_cache_days,
        target_cache_days=target_cache_days,
        pin_memory=pin_memory,
        teacher_current_exposures=teacher_current_exposures,
        include_future_closes=not policy_only,
    )
    checkpoint_teacher_state_provenance = (
        teacher_state_provenance if exact_state_providers else None
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
        "policyOnly": policy_only,
        "sequenceCoreTraining": sequence_core_training,
        "runtimeCacheDays": runtime_cache_days,
        "closeCacheDays": close_cache_days,
        "targetCacheDays": target_cache_days,
        "actionCount": model_config["actionCount"],
        "oracleContract": target_manifest["contract"],
        "counts": counts,
        "excludedTargetDates": excluded_dates,
        "datasetFingerprint": dataset_fingerprint,
        "teacherSwitchStatistics": (
            teacher_switch_statistics if action_objective is not None else None
        ),
        "teacherStateProvenance": teacher_state_provenance,
    })

    if arguments.check_data:
        summaries = {}
        # A data-contract check must not unseal the held-out test payload.
        for split in ("train", "validation"):
            check_rows = min(4, int(training["evaluationBatchSize"]))
            if sequence_core_training is None:
                batch = next(dataset.iter_batches(
                    split,
                    check_rows,
                    shuffle=False,
                    seed=int(training["seed"]),
                    maximum_batches=1,
                ))
            else:
                batch = next(dataset.iter_sequence_core_batches(
                    split,
                    check_rows,
                    receptive_field_minutes=int(
                        model_config["receptiveFieldMinutes"]
                    ),
                    shuffle=False,
                    seed=int(training["seed"]),
                    maximum_batches=1,
                    pack_contiguous_runs=pack_sequence_cores,
                ))
            teacher_current_exposure = None
            future_closes = None
            if policy_only:
                if sequence_core_training is None:
                    input_closes, target_policy = unpack_policy_only_batch(
                        batch
                    )
                    target_mask = None
                else:
                    (
                        input_closes,
                        target_policy,
                        target_mask,
                    ) = unpack_sequence_core_batch(batch)
            else:
                (
                    input_closes,
                    future_closes,
                    target_policy,
                    teacher_current_exposure,
                ) = unpack_batch(batch)
            summaries[split] = {
                "policyOnly": policy_only,
                "inputShape": list(input_closes.shape),
                "futureShape": (
                    list(future_closes.shape)
                    if future_closes is not None else None
                ),
                "targetShape": list(target_policy.shape),
                "targetRows": (
                    sequence_core_target_count(target_policy, target_mask)
                    if sequence_core_training is not None
                    else int(target_policy.shape[0])
                ),
                "inputLastClose": float(input_closes[0, -1, 0]),
                "futureFirstClose": (
                    float(future_closes[0, 0, 0])
                    if future_closes is not None else None
                ),
                "targetProbabilitySum": float(
                    target_policy.reshape(
                        -1,
                        target_policy.shape[-1],
                    )[0].sum()
                ),
            }
            if teacher_current_exposure is not None:
                summaries[split]["teacherExposureShape"] = list(
                    teacher_current_exposure.shape
                )
                summaries[split]["teacherCurrentExposure"] = float(
                    teacher_current_exposure[0]
                )
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
        conditioned_policy_cross_entropy=float(
            training["lossWeights"].get(
                "conditionedPolicyCrossEntropy",
                0,
            )
        ),
        forecast=float(training["lossWeights"]["forecast"]),
        soft_layer_norm=float(
            training["lossWeights"]["softLayerNorm"]
        ),
    )
    action_grid = torch.tensor(
        action_grid_values,
        device=device,
        dtype=torch.float32,
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
        training_config_fingerprint=training_config_fingerprint,
        teacher_state_provenance=checkpoint_teacher_state_provenance,
    )
    initialized_from = None
    initial_checkpoint_value = plan.get("initialCheckpoint")
    if start_epoch == 1 and global_step == 0 and initial_checkpoint_value:
        initial_checkpoint = require_under(
            resolve(repo_root, Path(initial_checkpoint_value)),
            storage.runs,
            "initialCheckpoint",
        )
        initial = load_torch_checkpoint(
            initial_checkpoint,
            map_location=device,
            weights_only=False,
        )
        if initial.get("architectureContract") != architecture_contract \
                or initial.get("dataContract") != DATA_CONTRACT \
                or initial.get("datasetFingerprint") != dataset_fingerprint \
                or initial.get("modelConfig") != model_config \
                or initial.get("parameterCount") != model_parameters:
            raise ValueError("initial checkpoint architecture is incompatible")
        model.load_state_dict(initial["model"])
        initialized_from = str(initial_checkpoint)
    plan_snapshot = snapshot_resolved_plan(
        run_dir,
        plan_file,
        plan,
        repo_root,
        resolved_training,
        training_config_fingerprint,
    )
    last_completed_epoch = start_epoch - 1
    durable_global_step = global_step
    if arguments.stop_after_epoch is not None \
            and last_completed_epoch >= arguments.stop_after_epoch:
        reporter.emit({
            "event": "training-paused",
            "reason": "stop-after-epoch-already-reached",
            "lastCompletedEpoch": last_completed_epoch,
            "globalStep": durable_global_step,
            "checkpoint": (
                str(last_checkpoint)
                if checkpoint_exists(last_checkpoint)
                else None
            ),
        })
        reporter.status(
            "paused",
            counts=counts,
            epoch=last_completed_epoch,
            epochs=training["epochs"],
            globalStep=durable_global_step,
            bestValidation=best_validation,
            bestEpoch=best_epoch,
            resumableCheckpoint=(
                str(last_checkpoint)
                if checkpoint_exists(last_checkpoint)
                else None
            ),
            trainingConfigFingerprint=training_config_fingerprint,
            stopAfterEpoch=arguments.stop_after_epoch,
            message=(
                "The requested completed-epoch boundary was already "
                "reached; no training or test evaluation ran."
            ),
        )
        return
    reporter.status(
        "training",
        counts=counts,
        epoch=start_epoch,
        epochs=training["epochs"],
        globalStep=global_step,
        parameterCount=model_parameters,
        architectureContract=architecture_contract,
        dataContract=DATA_CONTRACT,
        bestValidation=best_validation,
        bestEpoch=best_epoch,
        resumableCheckpoint=str(last_checkpoint),
        trainingConfigFingerprint=training_config_fingerprint,
        stopAfterEpoch=arguments.stop_after_epoch,
        evaluateTest=arguments.evaluate_test,
        disposableSmoke=arguments.disposable_smoke,
    )
    reporter.emit({
        "event": "training-start",
        "epoch": start_epoch,
        "epochs": training["epochs"],
        "globalStep": global_step,
        "parameterCount": model_parameters,
        "device": str(device),
        "architectureContract": architecture_contract,
        "dataContract": DATA_CONTRACT,
        "resumed": start_epoch > 1,
        "initializedFrom": initialized_from,
        "trainingConfigFingerprint": training_config_fingerprint,
        "stopAfterEpoch": arguments.stop_after_epoch,
        "evaluateTest": arguments.evaluate_test,
        "disposableSmoke": arguments.disposable_smoke,
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
            if stale_epochs > patience:
                break
            epoch_started = time.monotonic()
            model.train()
            optimizer.zero_grad(set_to_none=True)
            train_metrics = MetricAccumulator()
            batches = (
                dataset.batch_count("train", batch_size)
                if sequence_core_training is None
                else dataset.sequence_core_batch_count(
                    "train",
                    int(sequence_core_training["coreRows"]),
                    pack_contiguous_runs=pack_sequence_cores,
                )
            )
            if maximum_batches is not None:
                batches = min(batches, maximum_batches)
            microbatch_count = 0
            batch_source = (
                dataset.iter_batches(
                    "train",
                    batch_size,
                    shuffle=True,
                    seed=seed + epoch,
                    maximum_batches=maximum_batches,
                )
                if sequence_core_training is None
                else dataset.iter_sequence_core_batches(
                    "train",
                    int(sequence_core_training["coreRows"]),
                    receptive_field_minutes=int(
                        model_config["receptiveFieldMinutes"]
                    ),
                    shuffle=True,
                    seed=seed + epoch,
                    maximum_batches=maximum_batches,
                    pack_contiguous_runs=pack_sequence_cores,
                )
            )
            with PrefetchedBatchIterator(
                batch_source,
                int(training.get("prefetchBatches", 2)),
            ) as prefetched_batches:
                for batch_index, batch in enumerate(
                    prefetched_batches,
                    start=1,
                ):
                    sequence_core_count = None
                    if sequence_core_training is not None:
                        (
                            _cpu_input,
                            cpu_target_policy,
                            cpu_target_mask,
                        ) = unpack_sequence_core_batch(batch)
                        sequence_core_count = sequence_core_target_count(
                            cpu_target_policy,
                            cpu_target_mask,
                        )
                    moved_batch = move_batch(batch, device)
                    with autocast_context(device, training):
                        if policy_only:
                            if sequence_core_training is None:
                                input_closes, target_policy = (
                                    unpack_policy_only_batch(moved_batch)
                                )
                                _policy_logits, metrics = (
                                    policy_only_forward_objective(
                                        model,
                                        input_closes,
                                        target_policy,
                                        loss_weights,
                                    )
                                )
                                count = int(input_closes.shape[0])
                            else:
                                (
                                    input_closes,
                                    target_policy,
                                    target_mask,
                                ) = unpack_sequence_core_batch(moved_batch)
                                _policy_logits, metrics = (
                                    sequence_core_policy_forward_objective(
                                        model,
                                        input_closes,
                                        target_policy,
                                        loss_weights,
                                        target_mask=target_mask,
                                    )
                                )
                                assert sequence_core_count is not None
                                count = sequence_core_count
                        else:
                            (
                                input_closes,
                                future_closes,
                                target_policy,
                                teacher_current_exposure,
                            ) = unpack_batch(moved_batch)
                            output = model.forward_with_forecast(input_closes)
                            metrics = joint_price_oracle_objective(
                                output,
                                input_closes,
                                future_closes,
                                target_policy,
                                loss_weights,
                                action_grid=action_grid,
                                policy_friction=policy_friction,
                                policy_temperature=policy_temperature,
                                forecast_huber_delta=float(
                                    training.get("forecastHuberDelta", 1.0)
                                ),
                                volatility_floor=float(
                                    training.get("volatilityFloor", 1e-5)
                                ),
                            )
                            metrics = add_action_objective(
                                metrics,
                                output.policy_logits,
                                target_policy,
                                teacher_current_exposure,
                                action_grid,
                                action_objective,
                                source_switch_fraction=(
                                    source_switch_fraction
                                ),
                                policy_friction=policy_friction,
                                policy_temperature=policy_temperature,
                            )
                            count = int(input_closes.shape[0])
                        scaled_loss = metrics["loss"] / accumulate
                    scaled_loss.backward()
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
                            lastCompletedEpoch=last_completed_epoch,
                            durableGlobalStep=durable_global_step,
                            trainingConfigFingerprint=(
                                training_config_fingerprint
                            ),
                        )
                    if stop_requested:
                        break

            if stop_requested:
                partial_train = (
                    train_metrics.result()
                    if train_metrics.examples > 0
                    else None
                )
                reporter.emit({
                    "event": "training-paused",
                    "reason": "signal-discarded-partial-epoch",
                    "discardedEpoch": epoch,
                    "discardedGlobalSteps": (
                        global_step - durable_global_step
                    ),
                    "partialTrain": partial_train,
                    "lastCompletedEpoch": last_completed_epoch,
                    "globalStep": durable_global_step,
                    "checkpoint": (
                        str(last_checkpoint)
                        if checkpoint_exists(last_checkpoint)
                        else None
                    ),
                })
                reporter.status(
                    "paused",
                    counts=counts,
                    epoch=last_completed_epoch,
                    epochs=epochs,
                    globalStep=durable_global_step,
                    bestValidation=best_validation,
                    bestEpoch=best_epoch,
                    resumableCheckpoint=(
                        str(last_checkpoint)
                        if checkpoint_exists(last_checkpoint)
                        else None
                    ),
                    trainingConfigFingerprint=(
                        training_config_fingerprint
                    ),
                    message=(
                        f"Signal interrupted epoch {epoch}; its partial "
                        "updates were discarded. Resume starts after the "
                        "last fully validated, durable epoch."
                    ),
                )
                return
            train_result = train_metrics.result()
            validation_result = finite_metadata_value(evaluate(
                model,
                dataset,
                "validation",
                evaluation_batch_size,
                device,
                training,
                loss_weights,
                action_grid,
                policy_friction,
                policy_temperature,
                action_objective=action_objective,
                source_switch_fraction=source_switch_fraction,
                exact_state_provider=exact_state_providers.get("validation"),
                maximum_batches=maximum_batches,
            ))
            if not isinstance(validation_result, dict):
                raise RuntimeError("validation metrics must be an object")
            if selection_metric not in validation_result:
                raise ValueError(
                    f"unknown validation selection metric: {selection_metric}"
                )
            selection_value = validation_result[selection_metric]
            if isinstance(selection_value, bool) \
                    or not isinstance(selection_value, (int, float)) \
                    or not math.isfinite(float(selection_value)):
                raise ValueError(
                    f"validation selection metric is non-finite: "
                    f"{selection_metric}={selection_value!r}"
                )
            selection_value = float(selection_value)
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
                training_config_fingerprint=(
                    training_config_fingerprint
                ),
                teacher_state_provenance=(
                    checkpoint_teacher_state_provenance
                ),
            )
            atomic_torch_save(checkpoint, last_checkpoint)
            if improved:
                best_payload = {
                    "model": checkpoint["model"],
                    "epoch": epoch,
                    "globalStep": global_step,
                    "validation": validation_result,
                    "parameterCount": model_parameters,
                    "architectureContract": architecture_contract,
                    "dataContract": DATA_CONTRACT,
                    "datasetFingerprint": dataset_fingerprint,
                    "modelConfig": model_config,
                    "planId": plan["id"],
                    "trainingConfigFingerprint": (
                        training_config_fingerprint
                    ),
                }
                if checkpoint_teacher_state_provenance is not None:
                    best_payload["teacherStateProvenance"] = (
                        checkpoint_teacher_state_provenance
                    )
                atomic_torch_save(best_payload, best_checkpoint)
            last_completed_epoch = epoch
            durable_global_step = global_step
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
            pause_reason = None
            if stop_requested:
                pause_reason = "signal-after-durable-epoch"
            elif arguments.stop_after_epoch is not None \
                    and epoch >= arguments.stop_after_epoch:
                pause_reason = "stop-after-epoch"
            if pause_reason is not None:
                reporter.emit({
                    "event": "training-paused",
                    "reason": pause_reason,
                    "lastCompletedEpoch": last_completed_epoch,
                    "globalStep": durable_global_step,
                    "checkpoint": str(last_checkpoint),
                })
                reporter.status(
                    "paused",
                    counts=counts,
                    epoch=last_completed_epoch,
                    epochs=epochs,
                    globalStep=durable_global_step,
                    bestValidation=best_validation,
                    bestEpoch=best_epoch,
                    resumableCheckpoint=str(last_checkpoint),
                    trainingConfigFingerprint=(
                        training_config_fingerprint
                    ),
                    stopAfterEpoch=arguments.stop_after_epoch,
                    message=(
                        "Paused at a fully validated durable epoch boundary; "
                        "the canonical last checkpoint resumes at the next epoch."
                    ),
                )
                return
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
    test_result = None
    if arguments.evaluate_test:
        model.load_state_dict(best["model"])
        test_result = evaluate(
            model,
            dataset,
            "test",
            evaluation_batch_size,
            device,
            training,
            loss_weights,
            action_grid,
            policy_friction,
            policy_temperature,
            action_objective=action_objective,
            source_switch_fraction=source_switch_fraction,
            exact_state_provider=exact_state_providers.get("test"),
            maximum_batches=None,
        )
    result = {
        "event": "training-complete",
        "bestEpoch": best["epoch"],
        "bestValidation": best_validation,
        "testEvaluated": arguments.evaluate_test,
        "test": test_result,
        "checkpoint": str(best_checkpoint),
        "resumableCheckpoint": str(last_checkpoint),
        "parameterCount": model_parameters,
        "architectureContract": architecture_contract,
        "dataContract": DATA_CONTRACT,
        "trainingConfigFingerprint": training_config_fingerprint,
        "disposableSmoke": arguments.disposable_smoke,
    }
    reporter.emit(result)
    reporter.status("complete", **result)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataset: CausalOracleDataset,
    split: str,
    batch_size: int,
    device: torch.device,
    training: dict,
    loss_weights: JointLossWeights,
    action_grid: Tensor,
    policy_friction: float,
    policy_temperature: float,
    *,
    action_objective: dict | None = None,
    source_switch_fraction: float | None = None,
    exact_state_provider: ExactTraceExposureProvider | None = None,
    maximum_batches: int | None,
) -> dict[str, float | bool | None]:
    model.eval()
    policy_only = bool(training.get("policyOnly", False))
    sequence_core_training = training.get("sequenceCoreTraining")
    pack_sequence_cores = uses_packed_sequence_cores(
        sequence_core_training
    )
    accumulator = MetricAccumulator()
    ordered_logits: list[np.ndarray] = []
    ordered_targets: list[np.ndarray] = []
    batch_source = (
        dataset.iter_batches(
            split,
            batch_size,
            shuffle=False,
            seed=int(training["seed"]),
            maximum_batches=maximum_batches,
        )
        if sequence_core_training is None
        else dataset.iter_sequence_core_batches(
            split,
            int(sequence_core_training["coreRows"]),
            receptive_field_minutes=int(model.receptive_field_minutes),
            shuffle=False,
            seed=int(training["seed"]),
            maximum_batches=maximum_batches,
            pack_contiguous_runs=pack_sequence_cores,
        )
    )
    with PrefetchedBatchIterator(
        batch_source,
        int(training.get("prefetchBatches", 2)),
    ) as prefetched_batches:
        for batch in prefetched_batches:
            sequence_core_count = None
            if sequence_core_training is not None:
                (
                    _cpu_input,
                    cpu_target_policy,
                    cpu_target_mask,
                ) = unpack_sequence_core_batch(batch)
                sequence_core_count = sequence_core_target_count(
                    cpu_target_policy,
                    cpu_target_mask,
                )
            moved_batch = move_batch(batch, device)
            with autocast_context(device, training):
                if policy_only:
                    if sequence_core_training is None:
                        input_closes, target_policy = (
                            unpack_policy_only_batch(moved_batch)
                        )
                        policy_logits, metrics = policy_only_forward_objective(
                            model,
                            input_closes,
                            target_policy,
                            loss_weights,
                        )
                        count = int(input_closes.shape[0])
                    else:
                        (
                            input_closes,
                            target_policy,
                            target_mask,
                        ) = unpack_sequence_core_batch(moved_batch)
                        policy_logits, metrics = (
                            sequence_core_policy_forward_objective(
                                model,
                                input_closes,
                                target_policy,
                                loss_weights,
                                target_mask=target_mask,
                            )
                        )
                        assert sequence_core_count is not None
                        count = sequence_core_count
                else:
                    (
                        input_closes,
                        future_closes,
                        target_policy,
                        teacher_current_exposure,
                    ) = unpack_batch(moved_batch)
                    output = model.forward_with_forecast(input_closes)
                    policy_logits = output.policy_logits
                    metrics = joint_price_oracle_objective(
                        output,
                        input_closes,
                        future_closes,
                        target_policy,
                        loss_weights,
                        action_grid=action_grid,
                        policy_friction=policy_friction,
                        policy_temperature=policy_temperature,
                        forecast_huber_delta=float(
                            training.get("forecastHuberDelta", 1.0)
                        ),
                        volatility_floor=float(
                            training.get("volatilityFloor", 1e-5)
                        ),
                    )
                    metrics = add_action_objective(
                        metrics,
                        policy_logits,
                        target_policy,
                        teacher_current_exposure,
                        action_grid,
                        action_objective,
                        source_switch_fraction=source_switch_fraction,
                        policy_friction=policy_friction,
                        policy_temperature=policy_temperature,
                    )
                    count = int(input_closes.shape[0])
            if action_objective is not None:
                ordered_logits.append(
                    policy_logits.detach().float().cpu().numpy()
                )
                ordered_targets.append(
                    target_policy.detach().float().cpu().numpy()
                )
            accumulator.add(metrics, count)
    result = accumulator.result()
    if action_objective is not None:
        if not ordered_logits:
            raise RuntimeError("action validation produced no policy rows")
        result.update(ordered_rollout_metrics(
            np.concatenate(ordered_logits),
            np.concatenate(ordered_targets),
            action_grid.detach().float().cpu().numpy(),
            dataset.segments[split],
            friction=policy_friction,
            temperature=policy_temperature,
            execution_policy=action_objective.get("executionPolicy"),
            rollout_score_version=int(
                action_objective.get("rolloutScoreVersion", 1)
            ),
        ))
        if exact_state_provider is not None:
            exact_current = np.concatenate([
                exact_state_provider(split, segment)
                for segment in dataset.segments[split]
            ])[:np.concatenate(ordered_logits).shape[0]]
            exact_metrics = exact_state_actionable_policy_metrics_numpy(
                np.concatenate(ordered_logits),
                np.concatenate(ordered_targets),
                action_grid.detach().float().cpu().numpy(),
                exact_current,
                friction=policy_friction,
                temperature=policy_temperature,
                execution_policy=action_objective.get("executionPolicy"),
            )
            result["exactStateScore"] = float(
                exact_metrics["exactStateScore"]
            )
            for name, value in exact_metrics.items():
                if name == "exactStateScore":
                    continue
                metric_name = f"exactState{name[0].upper()}{name[1:]}"
                result[metric_name] = float(value)
    return result


def chronological_reset_mask(
    segments: list[CausalSegment],
) -> np.ndarray:
    """Mark split start and true gaps, but not batches or daily files."""
    if not segments:
        raise ValueError("cannot construct resets for an empty split")
    result = np.zeros(
        sum(segment.count for segment in segments),
        dtype=np.bool_,
    )
    cursor = 0
    previous: CausalSegment | None = None
    for segment in segments:
        if segment.count < 1:
            raise ValueError("causal segments must be non-empty")
        if previous is not None \
                and segment.prediction_time_start \
                <= previous.prediction_time_end:
            raise ValueError(
                "causal segments must be chronological and non-overlapping"
            )
        continuous = (
            previous is not None
            and segment.step_ms == previous.step_ms
            and segment.prediction_time_start
            == previous.prediction_time_end + previous.step_ms
        )
        if not continuous:
            result[cursor] = True
        cursor += segment.count
        previous = segment
    return result


def ordered_rollout_metrics(
    predicted_base_logits: np.ndarray,
    target_base_probabilities: np.ndarray,
    action_grid: np.ndarray,
    segments: list[CausalSegment],
    *,
    friction: float,
    temperature: float,
    execution_policy: dict | None = None,
    rollout_score_version: int = 1,
) -> dict[str, float | bool | None]:
    """Score one uninterrupted predicted policy path for model selection."""
    row_count = int(predicted_base_logits.shape[0])
    resets = chronological_reset_mask(segments)
    if row_count < 1 or row_count > resets.size:
        raise ValueError("ordered policy rows do not match the split timeline")
    actions = actionable_policy_metrics_numpy(
        predicted_base_logits,
        target_base_probabilities,
        action_grid,
        reset_mask=resets[:row_count],
        friction=friction,
        temperature=temperature,
        execution_policy=execution_policy,
    )
    grid = np.asarray(action_grid, dtype=np.float64)
    grid_span = float(grid[-1] - grid[0])
    if not math.isfinite(grid_span) or grid_span <= 0:
        raise ValueError("rollout score requires a finite action-grid span")
    if rollout_score_version not in {1, 2}:
        raise ValueError("rollout score version must be 1 or 2")
    transition_error = 1 - float(actions["transitionF1"])
    direction_error = 1 - float(actions["pathDirectionalAgreement"])
    normalized_path_error = (
        float(actions["pathMeanAbsoluteError"]) / grid_span
    )
    turnover_error = min(1.0, abs(
        float(actions["predictedTurnover"])
        - float(actions["targetTurnover"])
    ) / max(float(actions["targetTurnover"]), grid_span))
    if rollout_score_version == 1:
        rollout_score = (
            0.5 * transition_error
            + 0.25 * direction_error
            + 0.25 * normalized_path_error
        )
    else:
        signed_transition_error = 1 - float(actions["signedTransitionF1"])
        rollout_score = (
            0.6 * signed_transition_error
            + 0.15 * direction_error
            + 0.15 * normalized_path_error
            + 0.1 * turnover_error
        )
    if not math.isfinite(rollout_score):
        raise ValueError("ordered rollout score is non-finite")
    mean_regret = float(actions["meanConditionalRegret"])
    switch_regret = float(actions["switchConditionalRegret"])
    mean_regret_finite = math.isfinite(mean_regret)
    switch_regret_finite = math.isfinite(switch_regret)
    result = {
        "rolloutScore": rollout_score,
        "rolloutTransitionPrecision": float(actions["transitionPrecision"]),
        "rolloutTransitionRecall": float(actions["transitionRecall"]),
        "rolloutTransitionF1": float(actions["transitionF1"]),
        "rolloutSignedTransitionF1": float(actions["signedTransitionF1"]),
        "rolloutExactTransitionF1": float(actions["exactTransitionF1"]),
        "rolloutPathMeanAbsoluteError": float(
            actions["pathMeanAbsoluteError"]
        ),
        "rolloutPathDirectionalAgreement": float(
            actions["pathDirectionalAgreement"]
        ),
        "rolloutMeanConditionalRegret": (
            mean_regret if mean_regret_finite else None
        ),
        "rolloutMeanConditionalRegretFinite": mean_regret_finite,
        "rolloutSwitchConditionalRegret": (
            switch_regret if switch_regret_finite else None
        ),
        "rolloutSwitchConditionalRegretFinite": switch_regret_finite,
        "rolloutTargetSwitches": float(actions["targetSwitches"]),
        "rolloutPredictedSwitches": float(actions["predictedSwitches"]),
    }
    if rollout_score_version == 2:
        result.update({
            "rolloutScoreVersion": 2.0,
            "rolloutTurnoverRelativeError": turnover_error,
            "rolloutTargetTurnover": float(actions["targetTurnover"]),
            "rolloutPredictedTurnover": float(actions["predictedTurnover"]),
        })
    return result


def architecture_contract_for_model_config(config: dict) -> str:
    """Return the checkpoint/export contract selected by one model config."""
    policy_logit_rank = config.get("policyLogitRank")
    if "variant" not in config:
        return architecture_contract_with_policy_decoder(
            ARCHITECTURE_CONTRACT,
            policy_logit_rank,
        )
    variant = config.get("variant")
    direct_policy_contracts = {
        "minute_return_mlp": MINUTE_RETURN_MLP_CONTRACT,
        "minute_sequence_tcn": MINUTE_SEQUENCE_TCN_CONTRACT,
        "minute_sequence_boundary_tcn": (
            MINUTE_SEQUENCE_BOUNDARY_TCN_CONTRACT
        ),
        "minute_sequence_boundary_long_tcn": (
            MINUTE_SEQUENCE_BOUNDARY_LONG_TCN_CONTRACT
        ),
        "minute_sequence_boundary_prototype_mixture_tcn": (
            MINUTE_SEQUENCE_BOUNDARY_PROTOTYPE_MIXTURE_CONTRACT
        ),
        "minute_sequence_boundary_ma_tcn": (
            MINUTE_SEQUENCE_BOUNDARY_MA_TCN_CONTRACT
        ),
    }
    if variant in direct_policy_contracts:
        if policy_logit_rank is not None:
            raise ValueError(
                f"{variant} emits direct 101 logits and does not "
                "support policyLogitRank"
            )
        if variant == "minute_sequence_boundary_tcn" \
                and "policyDropout" in config:
            return MINUTE_SEQUENCE_BOUNDARY_TCN_POLICY_DROPOUT_CONTRACT
        return direct_policy_contracts[variant]
    contracts = {
        "patch_transformer": PATCH_TRANSFORMER_CONTRACT,
        "dilated_tcn": TCN_CONTRACT,
        "multiscale_residual_mixer": RESIDUAL_MIXER_CONTRACT,
        "long_context_patch_mixer": LONG_CONTEXT_PATCH_MIXER_CONTRACT,
        "learned_aggregate_patch_mixer": (
            LEARNED_AGGREGATE_PATCH_MIXER_CONTRACT
        ),
    }
    if variant not in contracts:
        raise ValueError(
            "variant must be 'patch_transformer', 'dilated_tcn', or "
            "'multiscale_residual_mixer', 'long_context_patch_mixer', or "
            "'learned_aggregate_patch_mixer', or "
            "'minute_return_mlp', 'minute_sequence_tcn', "
            "'minute_sequence_boundary_tcn', or "
            "'minute_sequence_boundary_long_tcn', or "
            "'minute_sequence_boundary_prototype_mixture_tcn', or "
            "'minute_sequence_boundary_ma_tcn'"
        )
    return architecture_contract_with_policy_decoder(
        contracts[variant],
        policy_logit_rank,
    )


def build_model(config: dict) -> nn.Module:
    if "variant" in config:
        # The variant factory owns validation of every variant-specific field.
        return build_variant_model(config)
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
        policy_logit_rank=config.get("policyLogitRank"),
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
    batch: Batch,
    device: torch.device,
) -> Batch:
    return tuple(
        value.to(device, non_blocking=True)
        for value in batch
    )


def unpack_batch(
    batch: Batch,
) -> tuple[Tensor, Tensor, Tensor, Tensor | None]:
    """Normalize legacy three-item and action-aware four-item batches."""
    if len(batch) == 3:
        input_closes, future_closes, target_policy = batch
        return input_closes, future_closes, target_policy, None
    if len(batch) == 4:
        input_closes, future_closes, target_policy, teacher_exposure = batch
        return (
            input_closes,
            future_closes,
            target_policy,
            teacher_exposure,
        )
    raise ValueError("joint training batch must contain three or four tensors")


def unpack_policy_only_batch(
    batch: Batch,
) -> tuple[Tensor, Tensor]:
    """Require the compact causal-input/target policy-only batch contract."""
    if len(batch) != 2:
        raise ValueError("policy-only training batch must contain two tensors")
    input_closes, target_policy = batch
    return input_closes, target_policy


def unpack_sequence_core_batch(
    batch: Batch,
) -> tuple[Tensor, Tensor, Tensor | None]:
    """Normalize legacy single-core and packed masked sequence batches."""
    if len(batch) == 2:
        input_closes, target_policy = batch
        return input_closes, target_policy, None
    if len(batch) == 3:
        input_closes, target_policy, target_mask = batch
        if target_mask.dtype != torch.bool:
            raise ValueError("sequence core target mask must be boolean")
        return input_closes, target_policy, target_mask
    raise ValueError(
        "sequence core training batch must contain two or three tensors"
    )


def policy_only_forward_objective(
    model: nn.Module,
    input_closes: Tensor,
    target_policy: Tensor,
    loss_weights: JointLossWeights,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Run a model's explicit forecast-free policy path and raw objective."""
    forward_policy_logits = getattr(model, "forward_policy_logits", None)
    if not callable(forward_policy_logits):
        raise ValueError(
            "policy-only training requires model.forward_policy_logits"
        )
    policy_logits = forward_policy_logits(input_closes)
    metrics = joint_price_oracle_policy_only_objective(
        policy_logits,
        target_policy,
        loss_weights,
    )
    return policy_logits, metrics


def sequence_core_policy_forward_objective(
    model: nn.Module,
    input_closes: Tensor,
    target_policy: Tensor,
    loss_weights: JointLossWeights,
    *,
    target_mask: Tensor | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Score contiguous core rows identically to flattened fixed windows."""
    forward_sequence_core = getattr(model, "forward_sequence_core", None)
    if not callable(forward_sequence_core):
        raise ValueError(
            "sequence core training requires model.forward_sequence_core"
        )
    policy_logits = forward_sequence_core(input_closes)
    if policy_logits.ndim != 3 \
            or target_policy.shape != policy_logits.shape:
        raise ValueError(
            "sequence core targets must match [batch, core, action] logits"
        )
    if target_mask is None:
        objective_logits = policy_logits.reshape(
            -1,
            policy_logits.shape[-1],
        )
        objective_targets = target_policy.reshape(
            -1,
            target_policy.shape[-1],
        )
    else:
        if target_mask.dtype != torch.bool \
                or target_mask.shape != policy_logits.shape[:2]:
            raise ValueError(
                "sequence core target mask must select [batch, core] rows"
            )
        objective_logits = policy_logits[target_mask]
        objective_targets = target_policy[target_mask]
    metrics = joint_price_oracle_policy_only_objective(
        objective_logits,
        objective_targets,
        loss_weights,
    )
    return policy_logits, metrics


def sequence_core_target_count(
    target_policy: Tensor,
    target_mask: Tensor | None = None,
) -> int:
    if target_policy.ndim != 3 or target_policy.shape[-1] != OUTPUT_ACTION_COUNT:
        raise ValueError(
            "sequence core target must be [batch, core, 101]"
        )
    if target_mask is None:
        return int(target_policy.shape[0] * target_policy.shape[1])
    if target_mask.dtype != torch.bool \
            or target_mask.shape != target_policy.shape[:2]:
        raise ValueError(
            "sequence core target mask must match [batch, core] targets"
        )
    return int(target_mask.sum())


def add_action_objective(
    base_metrics: dict[str, Tensor],
    predicted_base_logits: Tensor,
    target_base_probabilities: Tensor,
    teacher_current_exposures: Tensor | None,
    action_grid: Tensor,
    action_objective: dict | None,
    *,
    source_switch_fraction: float | None = None,
    policy_friction: float,
    policy_temperature: float,
) -> dict[str, Tensor]:
    """Optionally add execution-aligned action losses to a base objective."""
    if action_objective is None:
        if teacher_current_exposures is not None:
            raise ValueError(
                "legacy objective received unexpected teacher exposures"
            )
        return base_metrics
    if teacher_current_exposures is None:
        raise ValueError("action objective requires teacher current exposures")
    config = resolve_action_objective_config(action_objective)
    if config.get("switchWeighting", "batch") == "global":
        if source_switch_fraction is None:
            raise ValueError(
                "global action switch weighting requires a train-split "
                "switch fraction"
            )
    else:
        source_switch_fraction = None
    objective_weights = config["weights"]
    loss_weights = SwitchBalancedActionLossWeights(
        hard_action=float(objective_weights["hardAction"]),
        ranking=float(objective_weights["ranking"]),
        direction=float(objective_weights["direction"]),
        conditional_kl=float(objective_weights["conditionalKl"]),
    )
    logits = predicted_base_logits.float()
    targets = target_base_probabilities.float()
    grid = action_grid.float()

    def objective(current_exposures: Tensor) -> dict[str, Tensor]:
        return switch_balanced_action_objective(
            logits,
            targets,
            grid,
            current_exposures.float(),
            weights=loss_weights,
            target_switch_fraction=float(config["targetSwitchFraction"]),
            source_switch_fraction=source_switch_fraction,
            ranking_margin=float(config["rankingMargin"]),
            friction=policy_friction,
            temperature=policy_temperature,
            execution_policy=config.get("executionPolicy"),
        )

    action_metrics = objective(teacher_current_exposures)
    teacher_state_weight = float(config.get("teacherStateWeight", 1.0))
    self_state_weight = float(config.get("selfStateWeight", 0.0))
    mixed_action_loss = teacher_state_weight * action_metrics["loss"]
    self_metrics = None
    if self_state_weight > 0:
        self_current_exposures = self_conditioned_current_exposures_tensor(
            logits,
            grid,
            initial_exposure=float(
                teacher_current_exposures[0].detach().item()
            ),
            friction=policy_friction,
            temperature=policy_temperature,
            execution_policy=config.get("executionPolicy"),
        )
        self_metrics = objective(self_current_exposures)
        mixed_action_loss = (
            mixed_action_loss + self_state_weight * self_metrics["loss"]
        )
    result = dict(base_metrics)
    result["loss"] = (
        result["loss"] + float(config["lossWeight"]) * mixed_action_loss
    )
    result.update({
        "actionLoss": action_metrics["loss"],
        "actionHardCrossEntropy": action_metrics[
            "hardActionCrossEntropy"
        ],
        "actionRankingLoss": action_metrics["rankingLoss"],
        "actionDirectionCrossEntropy": action_metrics[
            "directionCrossEntropy"
        ],
        "actionConditionalKlDivergence": action_metrics[
            "conditionalKlDivergence"
        ],
        "actionHardAccuracy": action_metrics["hardActionAccuracy"],
        "actionSwitchRate": action_metrics["switchRate"],
        "actionMeanExampleWeight": action_metrics["meanExampleWeight"],
    })
    if "teacherStateWeight" in config or "selfStateWeight" in config:
        result["mixedActionLoss"] = mixed_action_loss
    if self_metrics is not None:
        result.update({
            "selfActionLoss": self_metrics["loss"],
            "selfActionHardCrossEntropy": self_metrics[
                "hardActionCrossEntropy"
            ],
            "selfActionRankingLoss": self_metrics["rankingLoss"],
            "selfActionDirectionCrossEntropy": self_metrics[
                "directionCrossEntropy"
            ],
            "selfActionConditionalKlDivergence": self_metrics[
                "conditionalKlDivergence"
            ],
            "selfActionHardAccuracy": self_metrics["hardActionAccuracy"],
            "selfActionSwitchRate": self_metrics["switchRate"],
            "selfActionMeanExampleWeight": self_metrics[
                "meanExampleWeight"
            ],
        })
    return result


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


def causal_input_close_windows(
    close_range: np.ndarray,
    count: int,
    context_length: int,
    *,
    sample_step_seconds: int = 1,
) -> np.ndarray:
    """Construct causal inputs ending at t without reading any close after t."""
    if close_range.ndim != 1 \
            or min(count, context_length, sample_step_seconds) < 1:
        raise ValueError("causal input window dimensions are invalid")
    expected_range = (
        context_length + (count - 1) * sample_step_seconds
    )
    if close_range.shape != (expected_range,):
        raise ValueError(
            "close range does not match the causal input dimensions"
        )
    all_windows = np.lib.stride_tricks.sliding_window_view(
        close_range,
        context_length,
    )
    windows = all_windows[::sample_step_seconds]
    if windows.shape != (count, context_length):
        raise RuntimeError("causal input windows are not aligned to predictions")
    return np.asarray(windows, dtype=np.float32).copy()


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


def validate_experiment_arguments(
    arguments: argparse.Namespace,
    run_dir: Path,
    runs_root: Path,
) -> None:
    maximum_batches = arguments.maximum_batches
    runtime_cache_days = getattr(arguments, "runtime_cache_days", None)
    if runtime_cache_days is not None and runtime_cache_days < 1:
        raise ValueError("--runtime-cache-days must be positive")
    if maximum_batches is not None and maximum_batches < 1:
        raise ValueError("--maximum-batches must be positive")
    if arguments.stop_after_epoch is not None \
            and arguments.stop_after_epoch < 1:
        raise ValueError("--stop-after-epoch must be positive")
    if arguments.evaluate_test and arguments.stop_after_epoch is not None:
        raise ValueError(
            "--evaluate-test and --stop-after-epoch are mutually exclusive"
        )
    if arguments.check_data and (
        arguments.evaluate_test
        or arguments.stop_after_epoch is not None
    ):
        raise ValueError(
            "--check-data cannot train, stop at an epoch, or evaluate test"
        )

    smoke_root = (runs_root / DISPOSABLE_SMOKE_DIRECTORY).resolve()
    try:
        smoke_relative = run_dir.resolve().relative_to(smoke_root)
        isolated_smoke_path = bool(smoke_relative.parts)
    except ValueError:
        isolated_smoke_path = False
    if maximum_batches is not None and not arguments.disposable_smoke:
        raise ValueError(
            "--maximum-batches requires --disposable-smoke"
        )
    if arguments.disposable_smoke and maximum_batches is None:
        raise ValueError(
            "--disposable-smoke requires --maximum-batches"
        )
    if arguments.disposable_smoke != isolated_smoke_path:
        raise ValueError(
            "disposable smoke runs must use --disposable-smoke and a unique "
            "runDir below data/training/runs/disposable-smoke"
        )
    if arguments.disposable_smoke and arguments.evaluate_test:
        raise ValueError("disposable smoke runs cannot evaluate the test split")
    if arguments.disposable_smoke \
            and (run_dir / "checkpoints" / "last.json").exists():
        raise ValueError(
            "disposable smoke runDir already has a checkpoint; use a new "
            "isolated runDir instead of resuming truncated-data training"
        )


def resolve_training_config(training: dict) -> dict:
    """Return the complete, JSON-canonical configuration that affects a run."""
    resolved = json.loads(json.dumps(training))
    if "policyOnly" in resolved \
            and not isinstance(resolved["policyOnly"], bool):
        raise ValueError("training.policyOnly must be a boolean")
    defaults = {
        "betas": [0.9, 0.999],
        "epsilon": 1e-8,
        "weightDecay": 0.0,
        "gradientAccumulationSteps": 1,
        "closeCacheDays": 10,
        "targetCacheDays": 3,
        "prefetchBatches": 2,
        "selectionMetric": "klDivergence",
        "forecastHuberDelta": 1.0,
        "volatilityFloor": 1e-5,
    }
    for key, value in defaults.items():
        resolved.setdefault(key, value)
    loss_weights = resolved.setdefault("lossWeights", {})
    loss_weights.setdefault("conditionedPolicyCrossEntropy", 0.0)
    if "actionObjective" in resolved:
        resolved["actionObjective"] = resolve_action_objective_config(
            resolved["actionObjective"]
        )
    return resolved


def resolve_action_objective_config(configuration: dict) -> dict:
    """Validate and canonicalize optional execution-aligned loss settings."""
    if not isinstance(configuration, dict):
        raise ValueError("actionObjective must be an object")
    allowed = {
        "lossWeight",
        "targetSwitchFraction",
        "rankingMargin",
        "weights",
        "teacherStateWeight",
        "selfStateWeight",
        "switchWeighting",
        "executionPolicy",
        "rolloutScoreVersion",
        "exactStateTraces",
        "allowValidationExactWithTrainSurrogate",
    }
    unknown = set(configuration) - allowed
    if unknown:
        raise ValueError(f"unknown actionObjective settings: {sorted(unknown)}")
    resolved = json.loads(json.dumps(configuration))
    resolved.setdefault("lossWeight", 1.0)
    resolved.setdefault("targetSwitchFraction", 0.5)
    resolved.setdefault("rankingMargin", 0.1)
    objective_weights = resolved.setdefault("weights", {})
    if not isinstance(objective_weights, dict):
        raise ValueError("actionObjective.weights must be an object")
    allowed_weights = {
        "hardAction",
        "ranking",
        "direction",
        "conditionalKl",
    }
    unknown_weights = set(objective_weights) - allowed_weights
    if unknown_weights:
        raise ValueError(
            "unknown actionObjective weights: "
            f"{sorted(unknown_weights)}"
        )
    defaults = {
        "hardAction": 1.0,
        "ranking": 1.0,
        "direction": 1.0,
        "conditionalKl": 0.1,
    }
    for key, value in defaults.items():
        objective_weights.setdefault(key, value)
    loss_weight = float(resolved["lossWeight"])
    switch_fraction = float(resolved["targetSwitchFraction"])
    ranking_margin = float(resolved["rankingMargin"])
    numeric_weights = [float(value) for value in objective_weights.values()]
    if not math.isfinite(loss_weight) or loss_weight <= 0:
        raise ValueError("actionObjective.lossWeight must be positive")
    if not math.isfinite(switch_fraction) \
            or not 0 < switch_fraction < 1:
        raise ValueError(
            "actionObjective.targetSwitchFraction must be in (0, 1)"
        )
    if not math.isfinite(ranking_margin) or ranking_margin < 0:
        raise ValueError(
            "actionObjective.rankingMargin must be finite and non-negative"
        )
    if any(not math.isfinite(value) or value < 0 for value in numeric_weights) \
            or not any(value > 0 for value in numeric_weights):
        raise ValueError(
            "actionObjective weights must be finite, non-negative, and nonzero"
        )
    resolved["lossWeight"] = loss_weight
    resolved["targetSwitchFraction"] = switch_fraction
    resolved["rankingMargin"] = ranking_margin
    resolved["weights"] = {
        key: float(objective_weights[key])
        for key in (
            "hardAction",
            "ranking",
            "direction",
            "conditionalKl",
        )
    }
    teacher_state_weight = float(resolved.get("teacherStateWeight", 1.0))
    self_state_weight = float(resolved.get("selfStateWeight", 0.0))
    if not math.isfinite(teacher_state_weight) \
            or teacher_state_weight < 0 \
            or not math.isfinite(self_state_weight) \
            or self_state_weight < 0 \
            or teacher_state_weight + self_state_weight <= 0:
        raise ValueError(
            "actionObjective state weights must be finite, non-negative, "
            "and nonzero"
        )
    if "teacherStateWeight" in resolved:
        resolved["teacherStateWeight"] = teacher_state_weight
    if "selfStateWeight" in resolved:
        resolved["selfStateWeight"] = self_state_weight
    switch_weighting = resolved.get("switchWeighting", "batch")
    if switch_weighting not in {"batch", "global"}:
        raise ValueError(
            "actionObjective.switchWeighting must be 'batch' or 'global'"
        )
    if "executionPolicy" in resolved:
        resolved["executionPolicy"] = resolve_execution_policy_config(
            resolved["executionPolicy"]
        )
    rollout_score_version = resolved.get("rolloutScoreVersion", 1)
    if isinstance(rollout_score_version, bool) \
            or rollout_score_version not in {1, 2}:
        raise ValueError("actionObjective.rolloutScoreVersion must be 1 or 2")
    if "rolloutScoreVersion" in resolved:
        resolved["rolloutScoreVersion"] = int(rollout_score_version)
    if rollout_score_version == 2 and "executionPolicy" not in resolved:
        raise ValueError(
            "actionObjective.rolloutScoreVersion 2 requires executionPolicy"
        )
    if "exactStateTraces" in resolved:
        if "executionPolicy" not in resolved:
            raise ValueError(
                "actionObjective.exactStateTraces requires executionPolicy"
            )
        resolved["exactStateTraces"] = resolve_exact_state_trace_config(
            resolved["exactStateTraces"],
            resolved["executionPolicy"],
        )
        allow_surrogate_train = resolved.get(
            "allowValidationExactWithTrainSurrogate",
            False,
        )
        if not isinstance(allow_surrogate_train, bool):
            raise ValueError(
                "allowValidationExactWithTrainSurrogate must be boolean"
            )
        traces = resolved["exactStateTraces"]
        if "validation" in traces and "train" not in traces \
                and allow_surrogate_train is not True:
            raise ValueError(
                "validation exact state with surrogate train state requires "
                "allowValidationExactWithTrainSurrogate=true"
            )
    elif "allowValidationExactWithTrainSurrogate" in resolved:
        raise ValueError(
            "allowValidationExactWithTrainSurrogate requires exactStateTraces"
        )
    return resolved


def resolve_exact_state_trace_config(
    configuration: object,
    execution_policy: dict,
) -> dict[str, dict[str, object]]:
    """Canonicalize SHA-bound per-split exact simulator traces."""
    if not isinstance(configuration, dict) or not configuration:
        raise ValueError("actionObjective.exactStateTraces must be an object")
    unknown_splits = set(configuration) - {"train", "validation", "test"}
    if unknown_splits:
        raise ValueError(
            f"unknown exact-state trace splits: {sorted(unknown_splits)}"
        )
    result: dict[str, dict[str, object]] = {}
    expected_leverage = float(execution_policy["maximumLeverage"])
    required = {
        "path",
        "sha256",
        "schemaVersion",
        "maximumLeverage",
        "rows",
    }
    for split in ("train", "validation", "test"):
        if split not in configuration:
            continue
        declaration = configuration[split]
        if not isinstance(declaration, dict) \
                or set(declaration) != required:
            raise ValueError(
                f"exactStateTraces.{split} must contain exactly "
                f"{sorted(required)}"
            )
        path = declaration["path"]
        digest = declaration["sha256"]
        schema = declaration["schemaVersion"]
        leverage = declaration["maximumLeverage"]
        rows = declaration["rows"]
        if not isinstance(path, str) or not path \
                or Path(path).is_absolute():
            raise ValueError(
                f"exactStateTraces.{split}.path must be a relative path"
            )
        if not isinstance(digest, str) or len(digest) != 64 \
                or any(value not in "0123456789abcdef" for value in digest):
            raise ValueError(
                f"exactStateTraces.{split}.sha256 must be lowercase hex"
            )
        if isinstance(schema, bool) or schema not in {1, 2}:
            raise ValueError(
                f"exactStateTraces.{split}.schemaVersion must be 1 or 2"
            )
        if isinstance(leverage, bool) \
                or not isinstance(leverage, (int, float)) \
                or not math.isfinite(float(leverage)) \
                or float(leverage) <= 0 \
                or float(leverage) != expected_leverage:
            raise ValueError(
                f"exactStateTraces.{split}.maximumLeverage must equal "
                "executionPolicy.maximumLeverage"
            )
        if isinstance(rows, bool) or not isinstance(rows, int) or rows < 1:
            raise ValueError(
                f"exactStateTraces.{split}.rows must be a positive integer"
            )
        result[split] = {
            "path": path.replace("\\", "/"),
            "sha256": digest,
            "schemaVersion": int(schema),
            "maximumLeverage": float(leverage),
            "rows": int(rows),
        }
    return result


def configuration_fingerprint(configuration: dict) -> str:
    encoded = json.dumps(
        configuration,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def snapshot_resolved_plan(
    run_dir: Path,
    plan_file: Path,
    plan: dict,
    repo_root: Path,
    resolved_training: dict,
    training_config_fingerprint: str,
) -> Path:
    resolved_plan = json.loads(json.dumps(plan))
    resolved_plan["training"] = resolved_training
    resolved_paths: dict[str, str] = {}
    for key in (
        "targetReferenceDir",
        "historyDir",
        "runDir",
        "artifactDir",
        "initialCheckpoint",
    ):
        value = plan.get(key)
        if isinstance(value, str):
            resolved_paths[key] = str(resolve(repo_root, Path(value)))
    snapshot = {
        "version": 1,
        "planId": plan["id"],
        "planFile": str(plan_file.resolve()),
        "trainingConfigFingerprint": training_config_fingerprint,
        "resolvedPaths": resolved_paths,
        "plan": resolved_plan,
    }
    snapshot_file = run_dir / "plans" / "resolved-plan.json"
    if snapshot_file.is_file():
        try:
            existing = json.loads(snapshot_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(
                f"resolved plan snapshot is unreadable: {snapshot_file}"
            ) from error
        if existing != snapshot:
            raise ValueError(
                "resolved plan differs from the immutable run snapshot: "
                f"{snapshot_file}"
            )
    else:
        atomic_json(snapshot, snapshot_file)
    return snapshot_file


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
    if "testPolicy" in plan \
            and plan["testPolicy"] != "sealed-never-load":
        raise ValueError("joint test policy must remain sealed-never-load")
    if model.get("variant") == "learned_aggregate_patch_mixer" \
            and plan.get("testPolicy") != "sealed-never-load":
        raise ValueError(
            "learned aggregate screen requires sealed-never-load test policy"
        )
    common_model_required = (
        "contextLength",
        "forecastHorizon",
        "actionCount",
    )
    legacy_model_required = (
        "movingAverageWindow",
        "patchLength",
        "linearRank",
        "aggregateMode",
        "tideHiddenWidth",
        "tideLayerCount",
        "policyHiddenWidth",
        "policyLayerCount",
        "dropout",
        "normalizationEpsilon",
        "branchNormalization",
    )
    model_required = (
        common_model_required
        if "variant" in model
        else common_model_required + legacy_model_required
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
        int(model["actionCount"]),
    ) < 1:
        raise ValueError("joint model dimensions must be positive")
    validated_variant_model = None
    if "variant" in model:
        # Construction is cheap and is the single source of truth for the
        # variant-specific divisibility, width, and RTX-sized token limits.
        validated_variant_model = build_variant_model(model)
    elif min(
        int(model["movingAverageWindow"]),
        int(model["patchLength"]),
        int(model["linearRank"]),
        int(model["tideHiddenWidth"]),
        int(model["tideLayerCount"]),
        int(model["policyHiddenWidth"]),
        int(model["policyLayerCount"]),
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
    if "variant" not in model:
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
            or float(loss_weights["policyCrossEntropy"]) <= 0:
        raise ValueError("joint loss weights are invalid")
    if "policyOnly" in training \
            and not isinstance(training["policyOnly"], bool):
        raise ValueError("training.policyOnly must be a boolean")
    policy_only = bool(training.get("policyOnly", False))
    policy_only_variants = {
        "minute_return_mlp",
        "minute_sequence_tcn",
        "minute_sequence_boundary_tcn",
        "minute_sequence_boundary_long_tcn",
        "minute_sequence_boundary_prototype_mixture_tcn",
        "minute_sequence_boundary_ma_tcn",
        "learned_aggregate_patch_mixer",
    }
    if model.get("variant") in policy_only_variants and not policy_only:
        raise ValueError(
            f"{model['variant']} is a policy-only variant and requires "
            "training.policyOnly=true"
        )
    sequence_core_training = training.get("sequenceCoreTraining")
    if sequence_core_training is not None:
        required_sequence_core = {"contract", "coreRows"}
        if not isinstance(sequence_core_training, dict) \
                or set(sequence_core_training) != required_sequence_core:
            raise ValueError(
                "training.sequenceCoreTraining must contain exactly "
                "contract and coreRows"
            )
        sequence_core_contract = sequence_core_training["contract"]
        if sequence_core_contract not in {
            SEQUENCE_REUSE_TRAINING_CONTRACT,
            PACKED_SEQUENCE_REUSE_TRAINING_CONTRACT,
        }:
            raise ValueError(
                "sequence core training contract is incompatible"
            )
        core_rows = sequence_core_training["coreRows"]
        if isinstance(core_rows, bool) \
                or not isinstance(core_rows, int) \
                or core_rows < 1:
            raise ValueError("sequence core rows must be a positive integer")
        if model.get("variant") not in {
            "minute_sequence_boundary_tcn",
            "minute_sequence_boundary_long_tcn",
            "minute_sequence_boundary_prototype_mixture_tcn",
        }:
            raise ValueError(
                "exact sequence core reuse is supported only by the "
                "boundary-complete non-MA minute TCN family"
            )
        if validated_variant_model is None \
                or not bool(getattr(
                    validated_variant_model,
                    "supports_sequence_core_reuse",
                    False,
                )) \
                or not callable(getattr(
                    validated_variant_model,
                    "forward_sequence_core",
                    None,
                )):
            raise ValueError(
                "sequence core model implementation is incompatible"
            )
        if float(model.get("dropout", 0.05)) != 0:
            raise ValueError(
                "exact sequence core training requires model.dropout=0 "
                "so the reused core is deterministic; configure "
                "model.policyDropout for head-only regularization"
            )
        if not policy_only:
            raise ValueError("sequence core training requires policyOnly=true")
        if int(training["batchSize"]) != core_rows \
                or int(training["evaluationBatchSize"]) != core_rows:
            raise ValueError(
                "sequence core rows must equal training and evaluation "
                "batch sizes to preserve fixed-window batch weighting"
            )
        if sequence_core_contract \
                == PACKED_SEQUENCE_REUSE_TRAINING_CONTRACT:
            if core_rows <= 1_440 or core_rows % 1_440 != 0:
                raise ValueError(
                    "packed sequence core rows must be a multiple of the "
                    "1,440-row reference optimizer batch"
                )
            if int(training.get("gradientAccumulationSteps", 1)) != 1:
                raise ValueError(
                    "packed sequence cores already define one row-weighted "
                    "optimizer batch and cannot use nested accumulation"
                )
    action_objective_enabled = "actionObjective" in training
    resolved_action_objective = None
    if action_objective_enabled:
        action_objective = training["actionObjective"]
        resolved_action_objective = resolve_action_objective_config(
            action_objective
        )
    selection_metric = training.get("selectionMetric", "klDivergence")
    if policy_only:
        incompatible_weights = {
            "conditionedPolicyCrossEntropy": float(
                loss_weights.get("conditionedPolicyCrossEntropy", 0)
            ),
            "forecast": float(loss_weights["forecast"]),
            "softLayerNorm": float(loss_weights["softLayerNorm"]),
        }
        enabled_incompatible = sorted(
            name
            for name, value in incompatible_weights.items()
            if value != 0
        )
        if enabled_incompatible:
            raise ValueError(
                "policy-only training requires zero incompatible loss "
                f"weights: {enabled_incompatible}"
            )
        if action_objective_enabled:
            raise ValueError(
                "policy-only training is incompatible with actionObjective"
            )
        if selection_metric != "klDivergence":
            raise ValueError(
                "policy-only training must select raw klDivergence"
            )
        if validated_variant_model is None \
                or not callable(getattr(
                    validated_variant_model,
                    "forward_policy_logits",
                    None,
                )):
            raise ValueError(
                "policy-only training requires a variant with an explicit "
                "forward_policy_logits path"
            )
    action_selection_metrics = {
        "actionLoss",
        "mixedActionLoss",
        "rolloutScore",
        "exactStateScore",
    }
    if selection_metric in action_selection_metrics \
            and not action_objective_enabled:
        raise ValueError(
            "action/rollout selection metrics require "
            "training.actionObjective"
        )
    if selection_metric == "mixedActionLoss" \
            and action_objective_enabled \
            and not any(
                key in training["actionObjective"]
                for key in ("teacherStateWeight", "selfStateWeight")
            ):
        raise ValueError(
            "selectionMetric mixedActionLoss requires explicit action state "
            "weights"
        )
    if selection_metric == "exactStateScore" \
            and (
                resolved_action_objective is None
                or "validation" not in resolved_action_objective.get(
                    "exactStateTraces",
                    {},
                )
            ):
        raise ValueError(
            "selectionMetric exactStateScore requires an exact validation trace"
        )


def build_training_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    *,
    epoch: int,
    global_step: int,
    best_validation: float,
    best_epoch: int,
    stale_epochs: int,
    validation: dict[str, object] | None,
    model_parameters: int,
    dataset_fingerprint: str,
    model_config: dict,
    plan_id: str,
    device: torch.device,
    interrupted: bool,
    training_config_fingerprint: str,
    teacher_state_provenance: dict[str, dict[str, object]] | None = None,
) -> dict:
    if not math.isfinite(best_validation):
        raise ValueError("checkpoint best validation must be finite")
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "globalStep": global_step,
        "bestValidation": best_validation,
        "bestEpoch": best_epoch,
        "staleEpochs": stale_epochs,
        "validation": finite_metadata_value(validation),
        "parameterCount": model_parameters,
        "architectureContract": architecture_contract_for_model_config(
            model_config
        ),
        "dataContract": DATA_CONTRACT,
        "datasetFingerprint": dataset_fingerprint,
        "modelConfig": model_config,
        "planId": plan_id,
        "trainingConfigFingerprint": training_config_fingerprint,
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
    if teacher_state_provenance is not None:
        checkpoint["teacherStateProvenance"] = finite_metadata_value(
            teacher_state_provenance
        )
    return checkpoint


def load_resume_checkpoint(
    file: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    plan: dict,
    model_config: dict,
    dataset_fingerprint: str,
    model_parameters: int,
    device: torch.device,
    *,
    training_config_fingerprint: str,
    teacher_state_provenance: dict[str, dict[str, object]] | None = None,
) -> tuple[int, int, float, int, int]:
    if not checkpoint_exists(file):
        return 1, 0, math.inf, 0, 0
    checkpoint = load_torch_checkpoint(
        file, map_location=device, weights_only=False
    )
    checkpoint_training_fingerprint = checkpoint.get(
        "trainingConfigFingerprint"
    )
    legacy_fingerprint_allowed = (
        checkpoint_training_fingerprint is None
        and plan["id"] in LEGACY_UNFINGERPRINTED_PLAN_IDS
    )
    if checkpoint.get("planId") != plan["id"] \
            or checkpoint.get("architectureContract") \
            != architecture_contract_for_model_config(model_config) \
            or checkpoint.get("dataContract") != DATA_CONTRACT \
            or checkpoint.get("datasetFingerprint") \
            != dataset_fingerprint \
            or checkpoint.get("modelConfig") != model_config \
            or checkpoint.get("parameterCount") != model_parameters \
            or (
                not legacy_fingerprint_allowed
                and checkpoint_training_fingerprint
                != training_config_fingerprint
            ) \
            or (
                teacher_state_provenance is not None
                and checkpoint.get("teacherStateProvenance")
                != finite_metadata_value(teacher_state_provenance)
            ):
        raise ValueError(
            "resume checkpoint model, data, or training configuration is "
            "incompatible"
        )
    if checkpoint.get("interrupted") is True:
        raise ValueError(
            "partial-epoch interrupted checkpoints cannot be resumed safely"
        )
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
        json.dumps(
            finite_metadata_value(value),
            indent=2,
            allow_nan=False,
        ) + "\n",
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


def finite_metadata_value(value):
    """Recursively encode non-finite scalar metadata as JSON-safe null."""
    if isinstance(value, dict):
        return {
            key: finite_metadata_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [finite_metadata_value(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def utc_day_ms(timestamp: int) -> int:
    return timestamp - timestamp % DAY_MS


def resolve(repo_root: Path, value: Path) -> Path:
    return value if value.is_absolute() else repo_root / value


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    main()
