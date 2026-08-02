from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import unittest

import numpy as np

from audit_oracle_calendar_predictability import (
    ACTION_COUNT,
    CALENDAR_SPECS,
    CalendarSelection,
    FEATURE_INDEX,
    MINUTE_MS,
    calendar_rows_for_segments,
    load_train_validation_targets,
    selections_from_calibration_report,
)
from train_joint_price_oracle import CausalSegment


def timestamp(value: str) -> int:
    return int(
        datetime.fromisoformat(value)
        .replace(tzinfo=timezone.utc)
        .timestamp()
        * 1_000
    ) + 999


def segment(
    split: str,
    day: str,
    start: str,
    count: int,
    row_offset: int = 0,
) -> CausalSegment:
    return CausalSegment(
        split=split,
        prediction_time_start=timestamp(start),
        count=count,
        target_file=Path(f"{day}.json"),
        target_row_offset=row_offset,
        step_ms=MINUTE_MS,
    )


class CalendarTimestampTests(unittest.TestCase):
    def test_utc_fields_follow_timestamp_across_week_month_and_year(self) -> None:
        rows = calendar_rows_for_segments([
            segment(
                "validation",
                "2027-01-01",
                "2026-12-31T23:59:00",
                3,
            )
        ])
        minute = rows.values[:, FEATURE_INDEX["minuteOfHour"]]
        hour = rows.values[:, FEATURE_INDEX["hourOfDay"]]
        weekday = rows.values[:, FEATURE_INDEX["dayOfWeek"]]
        month = rows.values[:, FEATURE_INDEX["monthOfYear"]]
        season = rows.values[:, FEATURE_INDEX["meteorologicalSeason"]]
        np.testing.assert_array_equal(minute, [59, 0, 1])
        np.testing.assert_array_equal(hour, [23, 0, 0])
        # 2026-12-31 is Thursday and 2027-01-01 is Friday (Monday=0).
        np.testing.assert_array_equal(weekday, [3, 4, 4])
        np.testing.assert_array_equal(month, [11, 0, 0])
        np.testing.assert_array_equal(season, [0, 0, 0])

    def test_purged_row_offset_does_not_shift_timestamp_features(self) -> None:
        rows = calendar_rows_for_segments([
            segment(
                "train",
                "2026-08-02",
                "2026-08-02T02:03:00",
                2,
                row_offset=123,
            )
        ])
        self.assertEqual(rows.values[0, FEATURE_INDEX["hourOfDay"]], 2)
        self.assertEqual(rows.values[0, FEATURE_INDEX["minuteOfHour"]], 3)
        self.assertEqual(rows.values[1, FEATURE_INDEX["minuteOfHour"]], 4)

    def test_rejects_timestamp_off_oracle_minute_phase(self) -> None:
        bad = CausalSegment(
            split="train",
            prediction_time_start=timestamp("2026-08-02T02:03:00") + 1,
            count=1,
            target_file=Path("2026-08-02.json"),
            target_row_offset=123,
            step_ms=MINUTE_MS,
        )
        with self.assertRaisesRegex(ValueError, "minute phase"):
            calendar_rows_for_segments([bad])


class SealedTestAccessTests(unittest.TestCase):
    def test_loader_never_requests_test_reference_or_payload(self) -> None:
        splits = {
            "train": [segment(
                "train", "2026-01-01", "2026-01-01T00:00:00", 2
            )],
            "validation": [segment(
                "validation", "2026-01-02", "2026-01-02T00:00:00", 2
            )],
            "test": [segment(
                "test", "2026-01-03", "2026-01-03T00:00:00", 2
            )],
        }
        requested: list[str] = []

        def reader(path, _cache, opened):
            requested.append(path.stem)
            opened.add(path.resolve())
            result = np.zeros((1_440, ACTION_COUNT), dtype=np.float32)
            result[:, 50] = 1
            return result

        targets, opened = load_train_validation_targets(
            splits,
            reader=reader,
        )
        self.assertEqual(requested, ["2026-01-01", "2026-01-02"])
        self.assertNotIn("2026-01-03", requested)
        self.assertEqual(targets["train"].shape, (2, ACTION_COUNT))
        self.assertEqual(targets["validation"].shape, (2, ACTION_COUNT))
        self.assertEqual(
            {path.stem for path in opened},
            {"2026-01-01", "2026-01-02"},
        )

    def test_overlap_with_test_reference_is_rejected_before_read(self) -> None:
        shared = segment(
            "train", "2026-01-03", "2026-01-03T00:00:00", 1
        )
        splits = {
            "train": [shared],
            "validation": [segment(
                "validation", "2026-01-02", "2026-01-02T00:00:00", 1
            )],
            "test": [segment(
                "test", "2026-01-03", "2026-01-03T00:00:00", 1
            )],
        }
        requested: list[Path] = []

        def reader(path, _cache, _opened):
            requested.append(path)
            raise AssertionError("reader must not be called")

        with self.assertRaisesRegex(ValueError, "sealed test"):
            load_train_validation_targets(splits, reader=reader)
        self.assertEqual(requested, [])


class DiagnosticReplayTests(unittest.TestCase):
    def test_replays_each_train_selected_candidate_without_reselection(self) -> None:
        reports = []
        for index, spec in enumerate(CALENDAR_SPECS):
            report = {
                "name": spec.name,
                "bestCalibrationKl": 1 + index / 100,
            }
            if spec.fields:
                report["bestFinePriorStrength"] = 64.0
                if spec.backoff_fields:
                    report["bestBackoffPriorStrength"] = 256.0
            reports.append(report)
        selections = selections_from_calibration_report({
            "candidateReports": reports,
        })
        self.assertEqual(len(selections), len(CALENDAR_SPECS))
        self.assertIsInstance(selections[0], CalendarSelection)
        self.assertIsNone(selections[0].fine_prior_strength)
        self.assertEqual(selections[-1].fine_prior_strength, 64.0)
        self.assertEqual(selections[-1].backoff_prior_strength, 256.0)
        self.assertAlmostEqual(
            selections[-1].calibration_kl,
            1 + (len(CALENDAR_SPECS) - 1) / 100,
        )


if __name__ == "__main__":
    unittest.main()
