"""Strict exact simulator-state trace loading for joint oracle models.

This production module owns the immutable schema/timeline/policy validation
shared by training, evaluation, calibration, and the standalone diagnostic.
Trace paths are not trusted: callers bind them to a declared SHA-256 digest.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from joint_price_oracle_actions import (
    DEFAULT_EXECUTION_CONFIDENCE_EXPOSURE_POWER,
    DEFAULT_EXECUTION_CONFIDENCE_LEVERAGE_FLOOR,
    DEFAULT_EXECUTION_MINIMUM_CONFIDENCE,
    execution_policy_native_scale,
    resolve_execution_policy_config,
    teacher_actions_at_current_exposures_numpy,
)
from trading_storage import implicit_times, read_shard_array, resolve_shard


TRACE_KIND = "exact-hindsight-oracle-bot-teacher-trace"


@dataclass(frozen=True)
class TraceRows:
    version: int
    split: str
    dates: tuple[str, ...]
    timestamps: np.ndarray
    current_exposures: np.ndarray
    raw_modal_exposures: np.ndarray | None
    conditioned_modal_exposures: np.ndarray
    requested_target_exposures: np.ndarray
    confidences: np.ndarray
    entropies: np.ndarray
    signal_emitted: np.ndarray
    oracle: dict[str, Any]
    maximum_leverage: float
    maximum_leverage_source: str


@dataclass(frozen=True)
class OracleRows:
    probabilities: np.ndarray
    grid: np.ndarray
    timestamps: np.ndarray
    reset_mask: np.ndarray
    contract: dict[str, Any]
    contract_hash: str
    dataset_fingerprint: str


@dataclass(frozen=True)
class ExactTraceExposureProvider:
    """Timestamp-indexed exact marked exposure in native oracle coordinates."""

    split: str
    timestamps: np.ndarray
    native_current_exposures: np.ndarray
    execution_scale: float
    source_file: Path
    source_sha256: str
    schema_version: int
    dates: tuple[str, ...]
    row_count: int
    maximum_leverage: float
    maximum_leverage_source: str
    execution_policy: dict[str, int | float]
    oracle_contract_hash: str
    oracle_dataset_fingerprint: str

    def __call__(self, split: str, segment: Any) -> np.ndarray:
        if split != self.split:
            raise ValueError(
                f"exact trace covers split {self.split!r}, not {split!r}"
            )
        count = int(segment.count)
        start = int(segment.prediction_time_start)
        step = int(segment.step_ms)
        if count < 1 or step < 1:
            raise ValueError("teacher segment count and step must be positive")
        requested = start + np.arange(count, dtype=np.int64) * step
        indexes = np.searchsorted(self.timestamps, requested)
        if bool((indexes >= self.timestamps.size).any()) \
                or not np.array_equal(self.timestamps[indexes], requested):
            raise ValueError(
                "exact teacher trace does not cover every segment timestamp"
            )
        return self.native_current_exposures[indexes].copy()


def build_exact_trace_exposure_provider(
    trace_file: Path,
    target_reference_dir: Path,
    *,
    legacy_maximum_leverage: float = 100.0,
    allow_test: bool = False,
    expected_sha256: str | None = None,
    expected_split: str | None = None,
    expected_schema_version: int | None = None,
    expected_dates: tuple[str, ...] | None = None,
    expected_row_count: int | None = None,
    expected_execution_policy: dict[str, Any] | None = None,
    absolute_tolerance: float = 1e-9,
    relative_tolerance: float = 1e-9,
) -> ExactTraceExposureProvider:
    """Load a trace only after all immutable and executable checks pass."""
    _validate_tolerance(absolute_tolerance, "absolute tolerance")
    _validate_tolerance(relative_tolerance, "relative tolerance")
    trace_path = trace_file.resolve()
    trace_bytes = trace_path.read_bytes()
    source_sha256 = hashlib.sha256(trace_bytes).hexdigest()
    if expected_sha256 is not None:
        if not isinstance(expected_sha256, str) \
                or len(expected_sha256) != 64 \
                or any(value not in "0123456789abcdef" for value in expected_sha256):
            raise ValueError("expected teacher trace sha256 must be lowercase hex")
        if source_sha256 != expected_sha256:
            raise ValueError("teacher trace sha256 does not match its declaration")
    trace = load_teacher_trace(
        json.loads(trace_bytes),
        legacy_maximum_leverage=legacy_maximum_leverage,
    )
    if trace.split == "test" and not allow_test:
        raise ValueError(
            "loading a locked test teacher trace requires explicit allow_test"
        )
    if expected_split is not None and trace.split != expected_split:
        raise ValueError("teacher trace split does not match its declaration")
    if expected_schema_version is not None \
            and trace.version != expected_schema_version:
        raise ValueError("teacher trace schema version does not match its declaration")
    if expected_dates is not None and trace.dates != tuple(expected_dates):
        raise ValueError("teacher trace dates do not match the selected split")
    if expected_row_count is not None \
            and trace.timestamps.size != expected_row_count:
        raise ValueError("teacher trace row count does not match its declaration")

    oracle = load_immutable_oracle_rows(
        target_reference_dir.resolve(),
        trace.dates,
    )
    _validate_trace_timeline(trace, oracle)
    _validate_oracle_contract(trace, oracle.contract)
    execution_policy = execution_policy_for_trace(trace)
    if expected_execution_policy is not None \
            and execution_policy != resolve_execution_policy_config(
                expected_execution_policy
            ):
        raise ValueError(
            "teacher trace leverage/execution policy does not match training"
        )
    execution_scale = execution_policy_native_scale(
        oracle.grid,
        execution_policy,
    )
    _validate_trace_decisions(
        trace,
        oracle,
        execution_policy,
        execution_scale,
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
    )
    return ExactTraceExposureProvider(
        split=trace.split,
        timestamps=trace.timestamps.copy(),
        native_current_exposures=(
            trace.current_exposures / execution_scale
        ),
        execution_scale=execution_scale,
        source_file=trace_path,
        source_sha256=source_sha256,
        schema_version=trace.version,
        dates=trace.dates,
        row_count=int(trace.timestamps.size),
        maximum_leverage=trace.maximum_leverage,
        maximum_leverage_source=trace.maximum_leverage_source,
        execution_policy=execution_policy,
        oracle_contract_hash=oracle.contract_hash,
        oracle_dataset_fingerprint=oracle.dataset_fingerprint,
    )


def load_teacher_trace(
    payload: Any,
    *,
    legacy_maximum_leverage: float,
) -> TraceRows:
    if not isinstance(payload, dict) or payload.get("kind") != TRACE_KIND:
        raise ValueError("not an exact hindsight-oracle bot teacher trace")
    version = payload.get("version")
    if isinstance(version, bool) or version not in {1, 2}:
        raise ValueError("teacher trace version must be 1 or 2")
    split = payload.get("split")
    if split not in {"train", "validation", "test"}:
        raise ValueError("teacher trace split must be train, validation, or test")
    raw_dates = payload.get("dates")
    if not isinstance(raw_dates, list) or not raw_dates \
            or any(not isinstance(value, str) or not value for value in raw_dates):
        raise ValueError("teacher trace dates must be a non-empty string list")
    dates = tuple(raw_dates)
    if len(set(dates)) != len(dates) or list(dates) != sorted(dates):
        raise ValueError("teacher trace dates must be unique and sorted")
    oracle = payload.get("oracle")
    if not isinstance(oracle, dict):
        raise ValueError("teacher trace oracle metadata is missing")
    decisions = payload.get("decisions")
    if not isinstance(decisions, list) or not decisions:
        raise ValueError("teacher trace decisions must be a non-empty list")
    if not all(isinstance(row, dict) for row in decisions):
        raise ValueError("every teacher trace decision must be an object")

    timestamps = _integer_field(decisions, "timestamp")
    if bool((np.diff(timestamps) <= 0).any()):
        raise ValueError("teacher trace timestamps must be strictly increasing")
    current = _finite_field(decisions, "currentExposure")
    target = _finite_field(decisions, "targetExposure")
    confidence = _finite_field(decisions, "confidence")
    entropy = _finite_field(decisions, "entropy")
    if bool(((confidence < 0) | (confidence > 1)).any()):
        raise ValueError("teacher trace confidence must be in [0, 1]")

    if version == 1:
        raw_modal = None
        conditioned_modal = _finite_field(decisions, "modalExposure")
        emitted = _boolean_field(decisions, "emitted")
        maximum_leverage = _finite_positive(
            legacy_maximum_leverage,
            "legacy maximum leverage",
        )
        maximum_leverage_source = "caller-verified-legacy-v1"
    else:
        raw_modal = _finite_field(decisions, "rawModalExposure")
        conditioned_modal = _finite_field(
            decisions,
            "conditionedModalExposure",
        )
        emitted = _boolean_field(decisions, "signalEmitted")
        maximum_leverage = _finite_positive(
            oracle.get("maximumLeverage"),
            "teacher trace oracle.maximumLeverage",
        )
        maximum_leverage_source = "trace-oracle-metadata"
    return TraceRows(
        version=int(version),
        split=split,
        dates=dates,
        timestamps=timestamps,
        current_exposures=current,
        raw_modal_exposures=raw_modal,
        conditioned_modal_exposures=conditioned_modal,
        requested_target_exposures=target,
        confidences=confidence,
        entropies=entropy,
        signal_emitted=emitted,
        oracle=oracle,
        maximum_leverage=maximum_leverage,
        maximum_leverage_source=maximum_leverage_source,
    )


def load_immutable_oracle_rows(
    target_reference_dir: Path,
    dates: tuple[str, ...],
) -> OracleRows:
    probabilities: list[np.ndarray] = []
    timestamps: list[np.ndarray] = []
    contract: dict[str, Any] | None = None
    contract_hash: str | None = None
    fingerprints: list[str] = []
    for date in dates:
        reference_file = target_reference_dir / f"{date}.json"
        shard = resolve_shard(reference_file)
        reference = shard.reference
        layout = reference.get("layout")
        metadata = reference.get("metadata")
        if reference.get("key") != date:
            raise ValueError(f"oracle shard key does not match date: {reference_file}")
        if not isinstance(layout, dict) \
                or layout.get("encoding") != "raw-row-major" \
                or layout.get("dtype") != "float32-le":
            raise ValueError(f"unsupported oracle row layout: {reference_file}")
        if not isinstance(metadata, dict) \
                or not isinstance(metadata.get("contract"), dict) \
                or not isinstance(metadata.get("contractHash"), str):
            raise ValueError(f"oracle contract metadata is missing: {reference_file}")
        rows = int(layout.get("rows", -1))
        columns = int(layout.get("columns", -1))
        if rows != shard.axis.count or rows < 1 or columns < 2:
            raise ValueError(f"oracle row dimensions are invalid: {reference_file}")
        _loaded, values = read_shard_array(
            reference_file,
            "<f4",
            (rows, columns),
        )
        row_contract = metadata["contract"]
        row_grid = row_contract.get("usableGrid")
        if not isinstance(row_grid, list) or len(row_grid) != columns:
            raise ValueError(f"oracle action grid is invalid: {reference_file}")
        if contract is None:
            contract = row_contract
            contract_hash = metadata["contractHash"]
        elif row_contract != contract or metadata["contractHash"] != contract_hash:
            raise ValueError("oracle contract changed within teacher trace dates")
        probabilities.append(values.astype(np.float64))
        timestamps.append(implicit_times(shard.axis))
        fingerprints.append(f"{date}:{reference['object']['contentHash']}")
    assert contract is not None and contract_hash is not None
    combined = np.concatenate(probabilities)
    if not np.isfinite(combined).all() or bool((combined < 0).any()):
        raise ValueError("immutable oracle probabilities must be finite and non-negative")
    mass = combined.sum(axis=-1)
    if bool((mass <= 0).any()) or bool((np.abs(mass - 1) > 1e-3).any()):
        raise ValueError("immutable oracle probability rows have invalid mass")
    combined /= mass[:, None]
    combined_timestamps = np.concatenate(timestamps).astype(np.int64, copy=False)
    expected_step = int(contract.get("decisionIntervalMs", 0))
    if expected_step <= 0:
        raise ValueError("oracle decision interval must be positive")
    reset_mask = np.ones(combined_timestamps.size, dtype=np.bool_)
    if combined_timestamps.size > 1:
        reset_mask[1:] = np.diff(combined_timestamps) != expected_step
    grid = np.asarray(contract["usableGrid"], dtype=np.float64)
    if not np.isfinite(grid).all() or bool((np.diff(grid) <= 0).any()):
        raise ValueError("oracle action grid must be finite and increasing")
    fingerprint = hashlib.sha256("\n".join(fingerprints).encode()).hexdigest()
    return OracleRows(
        probabilities=combined,
        grid=grid,
        timestamps=combined_timestamps,
        reset_mask=reset_mask,
        contract=contract,
        contract_hash=contract_hash,
        dataset_fingerprint=fingerprint,
    )


def execution_policy_for_trace(
    trace: TraceRows,
) -> dict[str, int | float]:
    return resolve_execution_policy_config({
        "version": 2,
        "maximumLeverage": trace.maximum_leverage,
        "minimumConfidence": trace.oracle.get(
            "minimumConfidence",
            DEFAULT_EXECUTION_MINIMUM_CONFIDENCE,
        ),
        "confidenceExposurePower": trace.oracle.get(
            "confidenceExposurePower",
            DEFAULT_EXECUTION_CONFIDENCE_EXPOSURE_POWER,
        ),
        "confidenceLeverageFloor": trace.oracle.get(
            "confidenceLeverageFloor",
            DEFAULT_EXECUTION_CONFIDENCE_LEVERAGE_FLOOR,
        ),
    })


def requested_execution_targets(
    target_indices: np.ndarray,
    confidences: np.ndarray,
    grid: np.ndarray,
    execution_policy: dict[str, Any],
    execution_scale: float,
) -> np.ndarray:
    modal = grid[target_indices] * execution_scale
    power = float(execution_policy["confidenceExposurePower"])
    floor = float(execution_policy["confidenceLeverageFloor"])
    maximum = float(execution_policy["maximumLeverage"])
    result = modal * np.power(np.clip(confidences, 0, 1), power)
    caps = maximum * (floor + (1 - floor) * np.clip(confidences, 0, 1))
    return np.clip(result, -caps, caps)


def _validate_trace_decisions(
    trace: TraceRows,
    oracle: OracleRows,
    execution_policy: dict[str, Any],
    execution_scale: float,
    *,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> None:
    options = oracle.contract["options"]
    one_step = teacher_actions_at_current_exposures_numpy(
        oracle.probabilities,
        oracle.grid,
        trace.current_exposures / execution_scale,
        friction=float(options["friction"]),
        temperature=float(options["temperature"]),
        execution_policy=execution_policy,
    )
    immutable_raw = oracle.grid[oracle.probabilities.argmax(axis=-1)]
    conditioned_modal = oracle.grid[one_step.target_indices]
    requested = requested_execution_targets(
        one_step.target_indices,
        one_step.confidences,
        oracle.grid,
        execution_policy,
        execution_scale,
    )

    def exact(expected: np.ndarray, actual: np.ndarray, name: str) -> None:
        matches = np.isclose(
            expected,
            actual,
            atol=absolute_tolerance,
            rtol=relative_tolerance,
        )
        if not bool(matches.all()):
            mismatch = int(np.flatnonzero(~matches)[0])
            raise ValueError(
                f"teacher trace {name} disagrees with the immutable oracle "
                f"at row {mismatch}"
            )

    if trace.raw_modal_exposures is not None:
        exact(immutable_raw, trace.raw_modal_exposures, "raw modal exposure")
    exact(
        conditioned_modal,
        trace.conditioned_modal_exposures,
        "conditioned modal exposure",
    )
    exact(requested, trace.requested_target_exposures, "requested target")
    exact(one_step.confidences, trace.confidences, "confidence")
    exact(one_step.conditional_entropies, trace.entropies, "entropy")
    if not np.array_equal(one_step.switch_labels, trace.signal_emitted):
        mismatch = int(np.flatnonzero(
            one_step.switch_labels != trace.signal_emitted
        )[0])
        raise ValueError(
            "teacher trace signal emission disagrees with the immutable "
            f"oracle at row {mismatch}"
        )


def _validate_trace_timeline(trace: TraceRows, oracle: OracleRows) -> None:
    if trace.timestamps.shape != oracle.timestamps.shape \
            or not np.array_equal(trace.timestamps, oracle.timestamps):
        raise ValueError(
            "teacher trace decisions do not exactly match immutable oracle timeline"
        )


def _validate_oracle_contract(trace: TraceRows, contract: dict[str, Any]) -> None:
    options = contract.get("options")
    if not isinstance(options, dict):
        raise ValueError("immutable oracle options are missing")
    expected = {
        "valueHorizonSteps": options.get("valueHorizonSteps"),
        "decisionDelaySteps": options.get("decisionDelaySteps"),
        "holdingPeriodSteps": options.get("holdingPeriodSteps"),
    }
    for name, value in expected.items():
        if trace.oracle.get(name) != value:
            raise ValueError(
                f"teacher trace {name} does not match immutable oracle contract"
            )
    _finite_positive(options.get("friction"), "oracle friction")
    _finite_positive(options.get("temperature"), "oracle temperature")


def _integer_field(rows: list[dict[str, Any]], name: str) -> np.ndarray:
    values = [row.get(name) for row in rows]
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise ValueError(f"teacher trace {name} must contain integers")
    return np.asarray(values, dtype=np.int64)


def _finite_field(rows: list[dict[str, Any]], name: str) -> np.ndarray:
    values = [row.get(name) for row in rows]
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in values):
        raise ValueError(f"teacher trace {name} must contain numbers")
    result = np.asarray(values, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError(f"teacher trace {name} must be finite")
    return result


def _boolean_field(rows: list[dict[str, Any]], name: str) -> np.ndarray:
    values = [row.get(name) for row in rows]
    if any(not isinstance(value, bool) for value in values):
        raise ValueError(f"teacher trace {name} must contain booleans")
    return np.asarray(values, dtype=np.bool_)


def _finite_positive(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{label} must be finite and positive")
    return result


def _validate_tolerance(value: float, label: str) -> None:
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{label} must be finite and non-negative")
