"""Leakage-safe UTC calendar audit for the causal oracle policy.

The audit uses only timestamps that are known at decision time.  Calendar
grouping and all empirical-Bayes shrinkage strengths are selected on the final
chronological 20% of the training split.  The selected estimator is then
refitted on the complete training split and scored once on the purged
validation split.  Held-out test reference files and payloads are never read.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Callable

import numpy as np

from audit_causal_oracle_predictability import (
    ACTION_COUNT,
    CALIBRATION_FRACTION,
    MINUTE_MS,
    SECOND_MS,
    TEST_DAYS,
    VALIDATION_DAYS,
    V18_BEST_RAW_VALIDATION_KL,
    mean_entropy,
    mean_kl,
    normalized_mean,
    read_target_day,
    split_and_purge,
    sufficient_table,
)
from train_joint_price_oracle import CausalSegment


FEATURE_DEFINITIONS = (
    ("minuteOfHour", 60),
    ("fiveMinuteOfHour", 12),
    ("hourOfDay", 24),
    ("sixHourSession", 4),
    ("dayOfWeek", 7),
    ("monthOfYear", 12),
    ("meteorologicalSeason", 4),
)
FEATURE_NAMES = tuple(name for name, _cardinality in FEATURE_DEFINITIONS)
FEATURE_INDEX = {name: index for index, name in enumerate(FEATURE_NAMES)}
FEATURE_CARDINALITY = dict(FEATURE_DEFINITIONS)
FINE_PRIOR_STRENGTHS = (16.0, 64.0, 256.0, 1_024.0)
BACKOFF_PRIOR_STRENGTHS = (64.0, 256.0, 1_024.0)
UTC_MINUTE_PHASE_MS = SECOND_MS - 1


@dataclass(frozen=True)
class CalendarRows:
    timestamps_ms: np.ndarray
    values: np.ndarray


@dataclass(frozen=True)
class CalendarSpec:
    name: str
    fields: tuple[str, ...]
    backoff_fields: tuple[str, ...] = ()


@dataclass(frozen=True)
class CalendarSelection:
    spec: CalendarSpec
    fine_prior_strength: float | None
    backoff_prior_strength: float | None
    calibration_kl: float


CALENDAR_SPECS = (
    CalendarSpec("full-train-prior-control", ()),
    CalendarSpec("minute-of-hour", ("minuteOfHour",)),
    CalendarSpec("five-minute-of-hour", ("fiveMinuteOfHour",)),
    CalendarSpec("hour-of-day", ("hourOfDay",)),
    CalendarSpec("six-hour-session", ("sixHourSession",)),
    CalendarSpec("day-of-week", ("dayOfWeek",)),
    CalendarSpec("month-of-year", ("monthOfYear",)),
    CalendarSpec("meteorological-season", ("meteorologicalSeason",)),
    CalendarSpec(
        "five-minute-of-day",
        ("hourOfDay", "fiveMinuteOfHour"),
        ("hourOfDay",),
    ),
    CalendarSpec(
        "minute-of-day",
        ("hourOfDay", "minuteOfHour"),
        ("hourOfDay",),
    ),
    CalendarSpec(
        "hour-by-weekday",
        ("dayOfWeek", "hourOfDay"),
        ("hourOfDay",),
    ),
    CalendarSpec(
        "hour-by-month",
        ("monthOfYear", "hourOfDay"),
        ("hourOfDay",),
    ),
    CalendarSpec(
        "hour-by-season",
        ("meteorologicalSeason", "hourOfDay"),
        ("hourOfDay",),
    ),
    CalendarSpec(
        "five-minute-of-week",
        ("dayOfWeek", "hourOfDay", "fiveMinuteOfHour"),
        ("dayOfWeek", "hourOfDay"),
    ),
    CalendarSpec(
        "minute-of-week",
        ("dayOfWeek", "hourOfDay", "minuteOfHour"),
        ("dayOfWeek", "hourOfDay"),
    ),
)


TargetReader = Callable[
    [Path, dict[Path, np.ndarray] | None, set[Path] | None],
    np.ndarray,
]


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    target_root = (
        repo_root
        / "data/training/immutable/refs/oracle/1s"
        / "hindsight-bot-71391c44b323e044e6ab"
    )
    target_files = sorted(target_root.glob("*.json"))
    if len(target_files) <= VALIDATION_DAYS + TEST_DAYS:
        raise ValueError("causal oracle corpus is too short")
    segments = split_and_purge(target_files)
    print(
        "Loading purged train/validation oracle targets only; sealed test "
        "reference files and payloads remain unopened.",
        file=sys.stderr,
        flush=True,
    )
    targets, opened = load_train_validation_targets(segments)
    calendar = {
        split: calendar_rows_for_segments(segments[split])
        for split in ("train", "validation")
    }
    for split in ("train", "validation"):
        if calendar[split].values.shape[0] != targets[split].shape[0]:
            raise RuntimeError(f"{split} timestamp/target alignment failed")

    selected, calibration_report = select_calendar_estimator(
        calendar["train"].values,
        targets["train"],
    )
    validation_report = evaluate_selected_estimator(
        selected,
        calendar["train"].values,
        targets["train"],
        calendar["validation"].values,
        targets["validation"],
    )
    candidate_validation_diagnostics = evaluate_train_selected_candidates(
        calibration_report,
        calendar["train"].values,
        targets["train"],
        calendar["validation"].values,
        targets["validation"],
    )
    train_files = {
        segment.target_file.resolve() for segment in segments["train"]
    }
    validation_files = {
        segment.target_file.resolve() for segment in segments["validation"]
    }
    test_files = {
        segment.target_file.resolve() for segment in segments["test"]
    }
    if opened & test_files:
        raise RuntimeError("sealed test reference or payload was opened")
    expected_opened = train_files | validation_files
    if opened != expected_opened:
        raise RuntimeError("unexpected target-reference access set")

    result = {
        "schemaVersion": 1,
        "audit": "causal-utc-calendar-policy-predictability",
        "oracleContract": {
            "valueHorizon": "1h",
            "decisionDelay": "1m",
            "minimumHold": "1m",
            "temperature": 0.01,
            "actionCount": ACTION_COUNT,
        },
        "accessContract": {
            "trainTargetReferenceFilesOpened": len(train_files),
            "validationTargetReferenceFilesOpened": len(validation_files),
            "testReferenceFilesOpened": 0,
            "testPayloadsOpened": 0,
            "candlePayloadsOpened": 0,
            "gpuUsed": False,
        },
        "corpus": {
            "targetFileCount": len(target_files),
            "trainFileCount": len(target_files) - VALIDATION_DAYS - TEST_DAYS,
            "validationFileCount": VALIDATION_DAYS,
            "heldoutTestFileCount": TEST_DAYS,
            "trainRowsAfterPurge": int(targets["train"].shape[0]),
            "validationRowsAfterPurge": int(targets["validation"].shape[0]),
            "trainTimestampStartUtc": timestamp_iso(
                int(calendar["train"].timestamps_ms[0])
            ),
            "trainTimestampEndUtc": timestamp_iso(
                int(calendar["train"].timestamps_ms[-1])
            ),
            "validationTimestampStartUtc": timestamp_iso(
                int(calendar["validation"].timestamps_ms[0])
            ),
            "validationTimestampEndUtc": timestamp_iso(
                int(calendar["validation"].timestamps_ms[-1])
            ),
        },
        "calendarContract": {
            "timezone": "UTC",
            "knownAtDecisionTime": True,
            "minuteTimestampPhaseMs": UTC_MINUTE_PHASE_MS,
            "dayOfWeekEncoding": "Monday=0..Sunday=6",
            "monthEncoding": "January=0..December=11",
            "meteorologicalSeasonEncoding": "DJF=0,MAM=1,JJA=2,SON=3",
            "candidateCount": len(CALENDAR_SPECS),
            "validationUsedForSelection": False,
        },
        "chronologicalTrainOnlySelection": calibration_report,
        "selectedEstimator": {
            "name": selected.spec.name,
            "fields": list(selected.spec.fields),
            "backoffFields": list(selected.spec.backoff_fields),
            "finePriorStrength": selected.fine_prior_strength,
            "backoffPriorStrength": selected.backoff_prior_strength,
            "calibrationKl": selected.calibration_kl,
        },
        "raw01Validation": validation_report,
        "postSelectionValidationDiagnostics": {
            "eligibleForEstimatorSelection": False,
            "reason": (
                "requested dimensionality diagnostic after the train-only "
                "winner was frozen; these scores must not replace it"
            ),
            "candidates": candidate_validation_diagnostics,
        },
    }
    print(json.dumps(result, indent=2, allow_nan=False))


def load_train_validation_targets(
    segments: dict[str, list[CausalSegment]],
    *,
    reader: TargetReader = read_target_day,
) -> tuple[dict[str, np.ndarray], set[Path]]:
    """Read only train/validation targets, never the supplied test paths."""
    if any(split not in segments for split in ("train", "validation", "test")):
        raise ValueError("train, validation, and sealed test segments are required")
    test_files = {
        segment.target_file.resolve() for segment in segments["test"]
    }
    cache: dict[Path, np.ndarray] = {}
    opened: set[Path] = set()
    result: dict[str, np.ndarray] = {}
    for split in ("train", "validation"):
        parts: list[np.ndarray] = []
        for segment in segments[split]:
            target_file = segment.target_file.resolve()
            if target_file in test_files:
                raise ValueError(
                    f"{split} contains sealed test reference {target_file}"
                )
            day = reader(target_file, cache, opened)
            start = segment.target_row_offset
            end = start + segment.count
            part = np.asarray(day[start:end], dtype=np.float32)
            if part.shape != (segment.count, ACTION_COUNT):
                raise ValueError(f"invalid {split} target slice shape")
            parts.append(part)
        if not parts:
            raise ValueError(f"{split} target split is empty")
        result[split] = np.concatenate(parts, axis=0)
        if not np.isfinite(result[split]).all() \
                or bool((result[split] < 0).any()) \
                or not np.allclose(
                    result[split].sum(axis=1),
                    1,
                    atol=2e-4,
                    rtol=2e-4,
                ):
            raise ValueError(f"{split} target probabilities are invalid")
    if opened & test_files:
        raise RuntimeError("sealed test target access detected")
    return result, opened


def calendar_rows_for_segments(
    segments: list[CausalSegment],
) -> CalendarRows:
    """Derive exact UTC calendar fields from causal prediction timestamps."""
    if not segments:
        raise ValueError("calendar segment list is empty")
    timestamp_parts: list[np.ndarray] = []
    for segment in segments:
        if segment.step_ms != MINUTE_MS:
            raise ValueError("calendar audit requires one-minute target rows")
        if segment.prediction_time_start % MINUTE_MS != UTC_MINUTE_PHASE_MS:
            raise ValueError("prediction timestamp is off the oracle minute phase")
        timestamps = (
            segment.prediction_time_start
            + np.arange(segment.count, dtype=np.int64) * segment.step_ms
        )
        timestamp_parts.append(timestamps)
    timestamps_ms = np.concatenate(timestamp_parts)
    if timestamps_ms.size > 1 and bool((np.diff(timestamps_ms) <= 0).any()):
        raise ValueError("calendar rows must be strictly chronological")

    epoch_minutes = np.floor_divide(timestamps_ms, MINUTE_MS)
    minute_of_hour = np.mod(epoch_minutes, 60)
    hour_of_day = np.mod(np.floor_divide(epoch_minutes, 60), 24)
    epoch_days = np.floor_divide(epoch_minutes, 24 * 60)
    day_of_week = np.mod(epoch_days + 3, 7)
    calendar_days = (
        np.datetime64("1970-01-01", "D")
        + epoch_days.astype("timedelta64[D]")
    )
    month_of_year = np.mod(
        calendar_days.astype("datetime64[M]").astype(np.int64),
        12,
    )
    season = np.floor_divide(np.mod(month_of_year + 1, 12), 3)
    values = np.column_stack((
        minute_of_hour,
        np.floor_divide(minute_of_hour, 5),
        hour_of_day,
        np.floor_divide(hour_of_day, 6),
        day_of_week,
        month_of_year,
        season,
    )).astype(np.int16, copy=False)
    for index, (name, cardinality) in enumerate(FEATURE_DEFINITIONS):
        column = values[:, index]
        if bool((column < 0).any()) or bool((column >= cardinality).any()):
            raise RuntimeError(f"invalid encoded calendar field {name}")
    return CalendarRows(timestamps_ms=timestamps_ms, values=values)


def encode_cells(
    calendar_values: np.ndarray,
    fields: tuple[str, ...],
) -> tuple[np.ndarray, int]:
    if calendar_values.ndim != 2 \
            or calendar_values.shape[1] != len(FEATURE_NAMES):
        raise ValueError("calendar feature matrix has invalid shape")
    ids = np.zeros(calendar_values.shape[0], dtype=np.int64)
    multiplier = 1
    for field in fields:
        if field not in FEATURE_INDEX:
            raise ValueError(f"unknown calendar field: {field}")
        cardinality = FEATURE_CARDINALITY[field]
        values = calendar_values[:, FEATURE_INDEX[field]].astype(
            np.int64,
            copy=False,
        )
        if bool((values < 0).any()) or bool((values >= cardinality).any()):
            raise ValueError(f"calendar values exceed {field} cardinality")
        ids += values * multiplier
        multiplier *= cardinality
    return ids, multiplier


def parent_ids_for_fine_cells(spec: CalendarSpec) -> np.ndarray:
    if not spec.backoff_fields:
        raise ValueError("calendar spec has no hierarchical backoff")
    if not set(spec.backoff_fields).issubset(spec.fields):
        raise ValueError("backoff fields must be a subset of fine fields")
    fine_count = int(np.prod([
        FEATURE_CARDINALITY[field] for field in spec.fields
    ], dtype=np.int64))
    fine_ids = np.arange(fine_count, dtype=np.int64)
    decoded: dict[str, np.ndarray] = {}
    multiplier = 1
    for field in spec.fields:
        cardinality = FEATURE_CARDINALITY[field]
        decoded[field] = np.mod(
            np.floor_divide(fine_ids, multiplier),
            cardinality,
        )
        multiplier *= cardinality
    result = np.zeros(fine_count, dtype=np.int64)
    multiplier = 1
    for field in spec.backoff_fields:
        result += decoded[field] * multiplier
        multiplier *= FEATURE_CARDINALITY[field]
    return result


def fit_calendar_table(
    calendar_values: np.ndarray,
    targets: np.ndarray,
    spec: CalendarSpec,
    fine_prior_strength: float | None,
    backoff_prior_strength: float | None,
    global_prior: np.ndarray | None = None,
) -> np.ndarray:
    if calendar_values.shape[0] != targets.shape[0] or targets.ndim != 2:
        raise ValueError("calendar rows and targets are not aligned")
    prior = normalized_mean(targets) if global_prior is None else global_prior
    if prior.shape != (targets.shape[1],):
        raise ValueError("calendar global prior has invalid shape")
    if not spec.fields:
        if fine_prior_strength is not None or backoff_prior_strength is not None:
            raise ValueError("global control does not accept shrinkage")
        return prior[None, :]
    if fine_prior_strength is None or fine_prior_strength <= 0:
        raise ValueError("fine calendar shrinkage must be positive")

    fine_ids, fine_count = encode_cells(calendar_values, spec.fields)
    fine_sums, fine_counts = sufficient_table(targets, fine_ids, fine_count)
    if spec.backoff_fields:
        if backoff_prior_strength is None or backoff_prior_strength <= 0:
            raise ValueError("hierarchical calendar shrinkage must be positive")
        backoff_ids, backoff_count = encode_cells(
            calendar_values,
            spec.backoff_fields,
        )
        backoff_sums, backoff_counts = sufficient_table(
            targets,
            backoff_ids,
            backoff_count,
        )
        backoff_table = (
            backoff_sums + backoff_prior_strength * prior[None, :]
        ) / (backoff_counts[:, None] + backoff_prior_strength)
        cell_prior = backoff_table[parent_ids_for_fine_cells(spec)]
    else:
        if backoff_prior_strength is not None:
            raise ValueError("non-hierarchical calendar spec has no backoff")
        cell_prior = np.broadcast_to(prior, fine_sums.shape)
    return (
        fine_sums + fine_prior_strength * cell_prior
    ) / (fine_counts[:, None] + fine_prior_strength)


def grouped_mean_kl(
    target_mean_entropy: float,
    target_sums_by_cell: np.ndarray,
    prediction_by_cell: np.ndarray,
    row_count: int,
) -> float:
    if target_sums_by_cell.shape != prediction_by_cell.shape \
            or row_count < 1:
        raise ValueError("grouped KL inputs are incompatible")
    prediction = np.clip(
        prediction_by_cell,
        np.finfo(np.float64).tiny,
        None,
    )
    cross_entropy = -float(
        np.sum(target_sums_by_cell * np.log(prediction), dtype=np.float64)
        / row_count
    )
    result = cross_entropy - target_mean_entropy
    return 0.0 if -1e-9 < result < 0 else result


def select_calendar_estimator(
    calendar_values: np.ndarray,
    targets: np.ndarray,
) -> tuple[CalendarSelection, dict[str, object]]:
    """Choose all calendar structure on a chronological internal train split."""
    if calendar_values.shape[0] != targets.shape[0] or targets.shape[0] < 10:
        raise ValueError("calendar selection corpus is invalid")
    fit_end = int(targets.shape[0] * (1 - CALIBRATION_FRACTION))
    fit_calendar = calendar_values[:fit_end]
    fit_targets = targets[:fit_end]
    calibration_calendar = calendar_values[fit_end:]
    calibration_targets = targets[fit_end:]
    fit_prior = normalized_mean(fit_targets)
    calibration_entropy = mean_entropy(calibration_targets)
    best: CalendarSelection | None = None
    candidate_reports: list[dict[str, object]] = []

    for spec in CALENDAR_SPECS:
        if not spec.fields:
            score = mean_kl(
                calibration_targets,
                np.broadcast_to(fit_prior, calibration_targets.shape),
            )
            selection = CalendarSelection(spec, None, None, score)
            candidate_reports.append({
                "name": spec.name,
                "fields": [],
                "backoffFields": [],
                "effectiveCells": 1,
                "internalFitMeanRowsPerCell": float(fit_end),
                "fullTrainMeanRowsPerCell": float(targets.shape[0]),
                "scores": [{
                    "finePriorStrength": None,
                    "backoffPriorStrength": None,
                    "calibrationKl": score,
                }],
                "bestCalibrationKl": score,
            })
            best = selection
            continue

        calibration_ids, cell_count = encode_cells(
            calibration_calendar,
            spec.fields,
        )
        calibration_sums, _counts = sufficient_table(
            calibration_targets,
            calibration_ids,
            cell_count,
        )
        score_rows: list[dict[str, float | None]] = []
        spec_best: CalendarSelection | None = None
        backoff_strengths: tuple[float | None, ...] = (
            BACKOFF_PRIOR_STRENGTHS
            if spec.backoff_fields
            else (None,)
        )
        for backoff_strength in backoff_strengths:
            for fine_strength in FINE_PRIOR_STRENGTHS:
                table = fit_calendar_table(
                    fit_calendar,
                    fit_targets,
                    spec,
                    fine_strength,
                    backoff_strength,
                    fit_prior,
                )
                score = grouped_mean_kl(
                    calibration_entropy,
                    calibration_sums,
                    table,
                    calibration_targets.shape[0],
                )
                score_rows.append({
                    "finePriorStrength": fine_strength,
                    "backoffPriorStrength": backoff_strength,
                    "calibrationKl": score,
                })
                selection = CalendarSelection(
                    spec,
                    fine_strength,
                    backoff_strength,
                    score,
                )
                if spec_best is None or score < spec_best.calibration_kl:
                    spec_best = selection
                if best is None or score < best.calibration_kl:
                    best = selection
        if spec_best is None:
            raise RuntimeError(f"no shrinkage candidate for {spec.name}")
        candidate_reports.append({
            "name": spec.name,
            "fields": list(spec.fields),
            "backoffFields": list(spec.backoff_fields),
            "effectiveCells": cell_count,
            "internalFitMeanRowsPerCell": fit_end / cell_count,
            "fullTrainMeanRowsPerCell": targets.shape[0] / cell_count,
            "scores": score_rows,
            "bestCalibrationKl": spec_best.calibration_kl,
            "bestFinePriorStrength": spec_best.fine_prior_strength,
            "bestBackoffPriorStrength": spec_best.backoff_prior_strength,
        })
    if best is None:
        raise RuntimeError("no calendar estimator was evaluated")
    return best, {
        "fitRows": fit_end,
        "calibrationRows": targets.shape[0] - fit_end,
        "fitFraction": 1 - CALIBRATION_FRACTION,
        "calibrationFraction": CALIBRATION_FRACTION,
        "candidateReports": candidate_reports,
        "selection": {
            "name": best.spec.name,
            "finePriorStrength": best.fine_prior_strength,
            "backoffPriorStrength": best.backoff_prior_strength,
            "calibrationKl": best.calibration_kl,
            "validationRowsObservedDuringSelection": 0,
        },
    }


def selections_from_calibration_report(
    calibration_report: dict[str, object],
) -> tuple[CalendarSelection, ...]:
    reports = calibration_report.get("candidateReports")
    if not isinstance(reports, list) or len(reports) != len(CALENDAR_SPECS):
        raise ValueError("calendar calibration report is incomplete")
    result: list[CalendarSelection] = []
    for spec, report in zip(CALENDAR_SPECS, reports, strict=True):
        if not isinstance(report, dict) or report.get("name") != spec.name:
            raise ValueError("calendar calibration report/spec mismatch")
        score = float(report["bestCalibrationKl"])
        fine_value = report.get("bestFinePriorStrength")
        backoff_value = report.get("bestBackoffPriorStrength")
        result.append(CalendarSelection(
            spec=spec,
            fine_prior_strength=(
                None if fine_value is None else float(fine_value)
            ),
            backoff_prior_strength=(
                None if backoff_value is None else float(backoff_value)
            ),
            calibration_kl=score,
        ))
    return tuple(result)


def evaluate_train_selected_candidates(
    calibration_report: dict[str, object],
    train_calendar: np.ndarray,
    train_targets: np.ndarray,
    validation_calendar: np.ndarray,
    validation_targets: np.ndarray,
) -> list[dict[str, object]]:
    """Score train-selected candidates for post-selection diagnostics only."""
    result: list[dict[str, object]] = []
    for selected in selections_from_calibration_report(calibration_report):
        validation = evaluate_selected_estimator(
            selected,
            train_calendar,
            train_targets,
            validation_calendar,
            validation_targets,
        )
        cell_count = int(np.prod([
            FEATURE_CARDINALITY[field] for field in selected.spec.fields
        ], dtype=np.int64))
        result.append({
            "name": selected.spec.name,
            "fields": list(selected.spec.fields),
            "backoffFields": list(selected.spec.backoff_fields),
            "effectiveCells": cell_count,
            "fullTrainMeanRowsPerCell": train_targets.shape[0] / cell_count,
            "trainSelectedFinePriorStrength": selected.fine_prior_strength,
            "trainSelectedBackoffPriorStrength": (
                selected.backoff_prior_strength
            ),
            "internalCalibrationKl": selected.calibration_kl,
            "rawValidationKl": validation["trainSelectedCalendarKl"],
            "absoluteValidationReductionFromPrior": (
                validation["absoluteCalendarReductionFromPrior"]
            ),
            "eligibleForEstimatorSelection": False,
        })
    return result


def evaluate_selected_estimator(
    selected: CalendarSelection,
    train_calendar: np.ndarray,
    train_targets: np.ndarray,
    validation_calendar: np.ndarray,
    validation_targets: np.ndarray,
) -> dict[str, float | bool]:
    """Refit one train-selected calendar estimator and score validation once."""
    train_prior = normalized_mean(train_targets)
    prior_kl = mean_kl(
        validation_targets,
        np.broadcast_to(train_prior, validation_targets.shape),
    )
    table = fit_calendar_table(
        train_calendar,
        train_targets,
        selected.spec,
        selected.fine_prior_strength,
        selected.backoff_prior_strength,
        train_prior,
    )
    validation_ids, _cell_count = encode_cells(
        validation_calendar,
        selected.spec.fields,
    )
    calendar_kl = mean_kl(validation_targets, table[validation_ids])
    return {
        "fullTrainPriorKl": prior_kl,
        "trainSelectedCalendarKl": calendar_kl,
        "absoluteCalendarReductionFromPrior": prior_kl - calendar_kl,
        "relativeCalendarReductionFromPrior": (
            (prior_kl - calendar_kl) / prior_kl
        ),
        "v18BestKl": V18_BEST_RAW_VALIDATION_KL,
        "calendarKlMinusV18": calendar_kl - V18_BEST_RAW_VALIDATION_KL,
        "materiallyBeatsV18ByAtLeast0.002": (
            calendar_kl < V18_BEST_RAW_VALIDATION_KL - 0.002
        ),
    }


def timestamp_iso(timestamp_ms: int) -> str:
    return datetime.fromtimestamp(
        timestamp_ms / 1_000,
        timezone.utc,
    ).isoformat(timespec="milliseconds").replace("+00:00", "Z")


if __name__ == "__main__":
    main()
