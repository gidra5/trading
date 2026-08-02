"""Check exact bot teacher traces against immutable oracle policy rows.

The diagnostic deliberately performs no model inference and never needs candle
history.  It compares three policy views on the trace's exact timeline:

* the immutable base distribution and its raw modal action;
* a chronological Python surrogate which carries its requested exposure; and
* one-step Python conditioning at the simulator's recorded current exposure.

The last view separates implementation disagreement from the expected drift
between a carried target and the simulator's marked exposure.  Test payloads
remain sealed unless the caller opts in explicitly.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from joint_price_oracle_actions import (
    execution_policy_native_scale,
    greedy_teacher_rollout_numpy,
    teacher_actions_at_current_exposures_numpy,
)
import joint_price_oracle_teacher_trace as teacher_trace


DEFAULT_TARGET_REFERENCE_DIR = Path(
    "data/training/immutable/refs/oracle/1s/"
    "hindsight-bot-71391c44b323e044e6ab"
)
TraceRows = teacher_trace.TraceRows
OracleRows = teacher_trace.OracleRows
ExactTraceExposureProvider = teacher_trace.ExactTraceExposureProvider


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument(
        "--target-reference-dir",
        type=Path,
        default=DEFAULT_TARGET_REFERENCE_DIR,
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--legacy-maximum-leverage",
        type=float,
        default=100.0,
        help="Execution leverage for v1 traces, whose metadata omitted it.",
    )
    parser.add_argument(
        "--allow-test",
        action="store_true",
        help="Explicitly unseal a trace whose split is test.",
    )
    parser.add_argument("--absolute-tolerance", type=float, default=1e-9)
    parser.add_argument("--relative-tolerance", type=float, default=1e-9)
    parser.add_argument("--example-limit", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    report = diagnose_teacher_trace(
        arguments.trace,
        arguments.target_reference_dir,
        legacy_maximum_leverage=arguments.legacy_maximum_leverage,
        allow_test=arguments.allow_test,
        absolute_tolerance=arguments.absolute_tolerance,
        relative_tolerance=arguments.relative_tolerance,
        example_limit=arguments.example_limit,
    )
    if arguments.output is not None:
        _atomic_json(arguments.output.resolve(), report)
    print(json.dumps(report, indent=2, allow_nan=False))


def diagnose_teacher_trace(
    trace_file: Path,
    target_reference_dir: Path,
    *,
    legacy_maximum_leverage: float = 100.0,
    allow_test: bool = False,
    absolute_tolerance: float = 1e-9,
    relative_tolerance: float = 1e-9,
    example_limit: int = 10,
) -> dict[str, Any]:
    """Return a strict-JSON comparison for one v1 or v2 teacher trace."""
    _validate_tolerance(absolute_tolerance, "absolute tolerance")
    _validate_tolerance(relative_tolerance, "relative tolerance")
    if isinstance(example_limit, bool) or not isinstance(example_limit, int) \
            or example_limit < 0:
        raise ValueError("example limit must be a non-negative integer")
    trace_path = trace_file.resolve()
    trace_payload = json.loads(trace_path.read_text(encoding="utf-8"))
    trace = teacher_trace.load_teacher_trace(
        trace_payload,
        legacy_maximum_leverage=legacy_maximum_leverage,
    )
    if trace.split == "test" and not allow_test:
        raise ValueError(
            "diagnosing a locked test teacher trace requires --allow-test"
        )
    oracle_rows = teacher_trace.load_immutable_oracle_rows(
        target_reference_dir.resolve(),
        trace.dates,
    )
    teacher_trace._validate_trace_timeline(trace, oracle_rows)
    teacher_trace._validate_oracle_contract(trace, oracle_rows.contract)

    contract_options = oracle_rows.contract["options"]
    friction = float(contract_options["friction"])
    temperature = float(contract_options["temperature"])
    execution_policy = teacher_trace.execution_policy_for_trace(trace)
    execution_scale = execution_policy_native_scale(
        oracle_rows.grid,
        execution_policy,
    )
    native_actual_current = trace.current_exposures / execution_scale

    surrogate = greedy_teacher_rollout_numpy(
        oracle_rows.probabilities,
        oracle_rows.grid,
        reset_mask=oracle_rows.reset_mask,
        friction=friction,
        temperature=temperature,
        execution_policy=execution_policy,
    )
    one_step = teacher_actions_at_current_exposures_numpy(
        oracle_rows.probabilities,
        oracle_rows.grid,
        native_actual_current,
        friction=friction,
        temperature=temperature,
        execution_policy=execution_policy,
    )

    immutable_raw_indices = oracle_rows.probabilities.argmax(axis=-1)
    immutable_raw_modal = oracle_rows.grid[immutable_raw_indices]
    surrogate_conditioned = oracle_rows.grid[surrogate.target_indices]
    one_step_conditioned = oracle_rows.grid[one_step.target_indices]
    surrogate_requested = teacher_trace.requested_execution_targets(
        surrogate.target_indices,
        surrogate.confidences,
        oracle_rows.grid,
        execution_policy,
        execution_scale,
    )
    one_step_requested = teacher_trace.requested_execution_targets(
        one_step.target_indices,
        one_step.confidences,
        oracle_rows.grid,
        execution_policy,
        execution_scale,
    )
    surrogate_applied = surrogate.target_exposures * execution_scale
    one_step_applied = one_step.target_exposures * execution_scale
    actual_signal_applied = np.where(
        trace.signal_emitted,
        trace.requested_target_exposures,
        trace.current_exposures,
    )
    native_threshold = abs(float(oracle_rows.grid[1] - oracle_rows.grid[0])) / 2
    execution_threshold = native_threshold * execution_scale
    actual_target_switch = (
        np.abs(
            trace.requested_target_exposures - trace.current_exposures
        ) >= execution_threshold
    )

    exposure = lambda expected, actual: _exposure_metrics(
        expected,
        actual,
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
    )
    raw_comparison: dict[str, Any]
    if trace.raw_modal_exposures is None:
        raw_comparison = {
            "available": False,
            "reason": (
                "v1 traces did not record the pre-transition raw modal "
                "exposure"
            ),
        }
    else:
        raw_comparison = {
            "available": True,
            **exposure(immutable_raw_modal, trace.raw_modal_exposures),
        }

    carried_mask = ~oracle_rows.reset_mask
    prior_requested = np.roll(trace.requested_target_exposures, 1)
    state_carry = exposure(
        prior_requested[carried_mask],
        trace.current_exposures[carried_mask],
    ) if bool(carried_mask.any()) else _empty_exposure_metrics()

    comparisons = {
        "immutableRawModalVsTrace": raw_comparison,
        "surrogateRolloutVsTrace": {
            "currentExposure": exposure(
                surrogate.current_exposures * execution_scale,
                trace.current_exposures,
            ),
            "conditionedModalExposure": exposure(
                surrogate_conditioned,
                trace.conditioned_modal_exposures,
            ),
            "requestedTargetExposure": exposure(
                surrogate_requested,
                trace.requested_target_exposures,
            ),
            "signalAppliedTargetExposure": exposure(
                surrogate_applied,
                actual_signal_applied,
            ),
            "confidence": exposure(
                surrogate.confidences,
                trace.confidences,
            ),
            "entropy": exposure(
                surrogate.conditional_entropies,
                trace.entropies,
            ),
            "signalEmission": _boolean_metrics(
                trace.signal_emitted,
                surrogate.switch_labels,
            ),
            "targetSwitch": _boolean_metrics(
                actual_target_switch,
                surrogate.switch_labels,
            ),
        },
        "oneStepAtActualCurrentVsTrace": {
            "conditionedActionIndex": _index_metrics(
                _nearest_grid_indices(
                    trace.conditioned_modal_exposures,
                    oracle_rows.grid,
                ),
                one_step.target_indices,
            ),
            "conditionedModalExposure": exposure(
                one_step_conditioned,
                trace.conditioned_modal_exposures,
            ),
            "requestedTargetExposure": exposure(
                one_step_requested,
                trace.requested_target_exposures,
            ),
            "signalAppliedTargetExposure": exposure(
                one_step_applied,
                actual_signal_applied,
            ),
            "confidence": exposure(one_step.confidences, trace.confidences),
            "entropy": exposure(
                one_step.conditional_entropies,
                trace.entropies,
            ),
            "signalEmission": _boolean_metrics(
                trace.signal_emitted,
                one_step.switch_labels,
            ),
            "targetSwitch": _boolean_metrics(
                actual_target_switch,
                one_step.switch_labels,
            ),
        },
        "traceInternalConsistency": {
            "signalEmissionVsRequestedTargetSwitch": _boolean_metrics(
                trace.signal_emitted,
                actual_target_switch,
            ),
            "actualCurrentVsPriorRequestedTarget": state_carry,
        },
    }
    examples = _divergence_examples(
        trace,
        oracle_rows,
        immutable_raw_modal=immutable_raw_modal,
        surrogate_conditioned=surrogate_conditioned,
        surrogate_requested=surrogate_requested,
        surrogate_emitted=surrogate.switch_labels,
        one_step_conditioned=one_step_conditioned,
        one_step_requested=one_step_requested,
        one_step_emitted=one_step.switch_labels,
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
        limit=example_limit,
    )
    report = {
        "version": 1,
        "kind": "joint-price-oracle-teacher-trace-alignment",
        "trace": {
            "file": str(trace_path),
            "schemaVersion": trace.version,
            "split": trace.split,
            "dates": list(trace.dates),
            "rows": int(trace.timestamps.size),
            "maximumLeverage": trace.maximum_leverage,
            "maximumLeverageSource": trace.maximum_leverage_source,
        },
        "immutableOracle": {
            "targetReferenceDir": str(target_reference_dir.resolve()),
            "contractHash": oracle_rows.contract_hash,
            "datasetFingerprint": oracle_rows.dataset_fingerprint,
            "rows": int(oracle_rows.timestamps.size),
            "actions": int(oracle_rows.grid.size),
        },
        "policy": {
            "friction": friction,
            "temperature": temperature,
            "executionScale": execution_scale,
            "nativeSwitchThreshold": native_threshold,
            "executionSwitchThreshold": execution_threshold,
            "executionPolicy": execution_policy,
            "surrogateState": (
                "previous signal-applied target; reset to zero at timeline gaps"
            ),
            "oneStepState": (
                "simulator-recorded current exposure mapped to native grid"
            ),
        },
        "comparisons": comparisons,
        "divergenceExamples": examples,
    }
    # Fail here rather than emit implementation-specific NaN/Infinity tokens.
    json.dumps(report, allow_nan=False)
    return report


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
    """Load and strictly validate a trace for exact-state model scoring.

    Besides the immutable-oracle timeline and contract checks used by the
    diagnostic, this entry point verifies the source hash, declared split,
    schema, dates, leverage/execution policy, row count, and every recorded
    executable decision.  Callers can therefore bind this provider into a
    training fingerprint without trusting mutable path contents.
    """
    return teacher_trace.build_exact_trace_exposure_provider(
        trace_file,
        target_reference_dir,
        legacy_maximum_leverage=legacy_maximum_leverage,
        allow_test=allow_test,
        expected_sha256=expected_sha256,
        expected_split=expected_split,
        expected_schema_version=expected_schema_version,
        expected_dates=expected_dates,
        expected_row_count=expected_row_count,
        expected_execution_policy=expected_execution_policy,
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
    )


_validate_trace_decisions = teacher_trace._validate_trace_decisions
load_teacher_trace = teacher_trace.load_teacher_trace
load_immutable_oracle_rows = teacher_trace.load_immutable_oracle_rows
_execution_policy = teacher_trace.execution_policy_for_trace


def _requested_execution_targets(
    target_indices: np.ndarray,
    confidences: np.ndarray,
    grid: np.ndarray,
    execution_policy: dict[str, Any],
    execution_scale: float,
) -> np.ndarray:
    return teacher_trace.requested_execution_targets(
        target_indices,
        confidences,
        grid,
        execution_policy,
        execution_scale,
    )


_validate_trace_timeline = teacher_trace._validate_trace_timeline
_validate_oracle_contract = teacher_trace._validate_oracle_contract


def _exposure_metrics(
    expected: np.ndarray,
    actual: np.ndarray,
    *,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> dict[str, Any]:
    expected_values = np.asarray(expected, dtype=np.float64)
    actual_values = np.asarray(actual, dtype=np.float64)
    if expected_values.shape != actual_values.shape:
        raise ValueError("comparison arrays must have matching shapes")
    if expected_values.size == 0:
        return _empty_exposure_metrics()
    if not np.isfinite(expected_values).all() or not np.isfinite(actual_values).all():
        raise ValueError("comparison arrays must be finite")
    error = actual_values - expected_values
    exact = np.isclose(
        expected_values,
        actual_values,
        atol=absolute_tolerance,
        rtol=relative_tolerance,
    )
    return {
        "count": int(expected_values.size),
        "exactCount": int(exact.sum()),
        "exactRate": float(exact.mean()),
        "meanAbsoluteError": float(np.abs(error).mean()),
        "rootMeanSquaredError": float(np.sqrt(np.square(error).mean())),
        "maximumAbsoluteError": float(np.abs(error).max()),
        "meanSignedError": float(error.mean()),
    }


def _empty_exposure_metrics() -> dict[str, Any]:
    return {
        "count": 0,
        "exactCount": 0,
        "exactRate": None,
        "meanAbsoluteError": None,
        "rootMeanSquaredError": None,
        "maximumAbsoluteError": None,
        "meanSignedError": None,
    }


def _boolean_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    actual_values = np.asarray(actual, dtype=np.bool_)
    predicted_values = np.asarray(predicted, dtype=np.bool_)
    if actual_values.shape != predicted_values.shape:
        raise ValueError("boolean comparison arrays must have matching shapes")
    true_positive = int((actual_values & predicted_values).sum())
    false_positive = int((~actual_values & predicted_values).sum())
    false_negative = int((actual_values & ~predicted_values).sum())
    true_negative = int((~actual_values & ~predicted_values).sum())
    precision = _safe_ratio(true_positive, true_positive + false_positive)
    recall = _safe_ratio(true_positive, true_positive + false_negative)
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None
        and precision + recall > 0 else None
    )
    count = int(actual_values.size)
    disagreements = false_positive + false_negative
    return {
        "count": count,
        "actualTrue": int(actual_values.sum()),
        "predictedTrue": int(predicted_values.sum()),
        "agreementCount": count - disagreements,
        "agreementRate": (count - disagreements) / count if count else None,
        "divergenceCount": disagreements,
        "divergenceRate": disagreements / count if count else None,
        "truePositive": true_positive,
        "falsePositive": false_positive,
        "falseNegative": false_negative,
        "trueNegative": true_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _index_metrics(expected: np.ndarray, actual: np.ndarray) -> dict[str, Any]:
    expected_values = np.asarray(expected, dtype=np.int64)
    actual_values = np.asarray(actual, dtype=np.int64)
    if expected_values.shape != actual_values.shape:
        raise ValueError("index comparison arrays must have matching shapes")
    exact = expected_values == actual_values
    return {
        "count": int(exact.size),
        "exactCount": int(exact.sum()),
        "exactRate": float(exact.mean()) if exact.size else None,
        "divergenceCount": int((~exact).sum()),
    }


def _nearest_grid_indices(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    distances = np.abs(np.asarray(values, dtype=np.float64)[:, None] - grid)
    return distances.argmin(axis=-1)


def _divergence_examples(
    trace: TraceRows,
    oracle: OracleRows,
    *,
    immutable_raw_modal: np.ndarray,
    surrogate_conditioned: np.ndarray,
    surrogate_requested: np.ndarray,
    surrogate_emitted: np.ndarray,
    one_step_conditioned: np.ndarray,
    one_step_requested: np.ndarray,
    one_step_emitted: np.ndarray,
    absolute_tolerance: float,
    relative_tolerance: float,
    limit: int,
) -> dict[str, list[dict[str, Any]]]:
    def close(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        return np.isclose(
            left,
            right,
            atol=absolute_tolerance,
            rtol=relative_tolerance,
        )

    surrogate_mismatch = (
        ~close(surrogate_conditioned, trace.conditioned_modal_exposures)
        | ~close(surrogate_requested, trace.requested_target_exposures)
        | (surrogate_emitted != trace.signal_emitted)
    )
    one_step_mismatch = (
        ~close(one_step_conditioned, trace.conditioned_modal_exposures)
        | ~close(one_step_requested, trace.requested_target_exposures)
        | (one_step_emitted != trace.signal_emitted)
    )
    if trace.raw_modal_exposures is not None:
        one_step_mismatch |= ~close(
            immutable_raw_modal,
            trace.raw_modal_exposures,
        )

    def rows(mask: np.ndarray, one_step_view: bool) -> list[dict[str, Any]]:
        result = []
        for index in np.flatnonzero(mask)[:limit]:
            predicted_conditioned = (
                one_step_conditioned[index]
                if one_step_view else surrogate_conditioned[index]
            )
            predicted_requested = (
                one_step_requested[index]
                if one_step_view else surrogate_requested[index]
            )
            predicted_emitted = (
                one_step_emitted[index]
                if one_step_view else surrogate_emitted[index]
            )
            result.append({
                "row": int(index),
                "timestamp": int(trace.timestamps[index]),
                "reset": bool(oracle.reset_mask[index]),
                "actualCurrentExposure": float(trace.current_exposures[index]),
                "traceConditionedModalExposure": float(
                    trace.conditioned_modal_exposures[index]
                ),
                "pythonConditionedModalExposure": float(predicted_conditioned),
                "traceRequestedTargetExposure": float(
                    trace.requested_target_exposures[index]
                ),
                "pythonRequestedTargetExposure": float(predicted_requested),
                "traceSignalEmitted": bool(trace.signal_emitted[index]),
                "pythonSignalEmitted": bool(predicted_emitted),
            })
        return result

    return {
        "surrogateRollout": rows(surrogate_mismatch, False),
        "oneStepAtActualCurrent": rows(one_step_mismatch, True),
    }


def _integer_field(rows: list[dict[str, Any]], name: str) -> np.ndarray:
    values = []
    for row in rows:
        value = row.get(name)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"teacher trace {name} must contain integers")
        values.append(value)
    return np.asarray(values, dtype=np.int64)


def _finite_field(rows: list[dict[str, Any]], name: str) -> np.ndarray:
    values = []
    for row in rows:
        value = row.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) \
                or not math.isfinite(float(value)):
            raise ValueError(f"teacher trace {name} must contain finite numbers")
        values.append(float(value))
    return np.asarray(values, dtype=np.float64)


def _boolean_field(rows: list[dict[str, Any]], name: str) -> np.ndarray:
    values = []
    for row in rows:
        value = row.get(name)
        if not isinstance(value, bool):
            raise ValueError(f"teacher trace {name} must contain booleans")
        values.append(value)
    return np.asarray(values, dtype=np.bool_)


def _finite_positive(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) \
            or not math.isfinite(float(value)) or float(value) <= 0:
        raise ValueError(f"{label} must be finite and positive")
    return float(value)


def _validate_tolerance(value: float, label: str) -> None:
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{label} must be finite and non-negative")


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _atomic_json(file: Path, value: Any) -> None:
    file.parent.mkdir(parents=True, exist_ok=True)
    temporary = file.with_name(f"{file.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, file)
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
