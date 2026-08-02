from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
import unittest

import numpy as np

from audit_v18_on_v28_validation_rows import (
    filename_only_segments,
    ordered_row_keys,
    ordered_subsequence_indexes,
    split_counts,
)
from train_joint_price_oracle import CausalSegment


class MatchedValidationAuditTests(unittest.TestCase):
    def test_v28_validation_rows_are_exact_300_row_v18_suffix(self) -> None:
        start = date(2026, 1, 1)
        target_files = [
            Path(f"{start + timedelta(days=index)}.json")
            for index in range(65)
        ]
        shared = {
            "dataSplit": {"validationDays": 30, "testDays": 30},
            "model": {"forecastHorizon": 3_600},
        }
        v18_plan = {
            **shared,
            "model": {
                **shared["model"],
                "contextLength": 3_601,
            },
        }
        v28_plan = {
            **shared,
            "model": {
                **shared["model"],
                "contextLength": 21_601,
            },
        }

        v18 = filename_only_segments(target_files, v18_plan)
        v28 = filename_only_segments(target_files, v28_plan)
        v18_keys = ordered_row_keys(v18["validation"])
        v28_keys = ordered_row_keys(v28["validation"])
        indexes = ordered_subsequence_indexes(v18_keys, v28_keys)

        self.assertEqual(split_counts(v18)["validation"], 43_080)
        self.assertEqual(split_counts(v28)["validation"], 42_780)
        self.assertEqual(v18["validation"][0].target_row_offset, 60)
        self.assertEqual(v28["validation"][0].target_row_offset, 360)
        self.assertEqual(v18["validation"][-1].count, 1_380)
        self.assertEqual(v28["validation"][-1].count, 1_380)
        np.testing.assert_array_equal(
            indexes,
            np.arange(300, 43_080, dtype=np.int64),
        )

    def test_subsequence_alignment_rejects_a_missing_target_row(self) -> None:
        segment = CausalSegment(
            split="validation",
            prediction_time_start=999,
            count=3,
            target_file=Path("2026-01-01.json"),
            target_row_offset=10,
            step_ms=60_000,
        )
        source = ordered_row_keys([segment])
        missing = [(source[1][0], source[1][1], 999)]
        with self.assertRaisesRegex(ValueError, "not a v18 subsequence"):
            ordered_subsequence_indexes(source, missing)


if __name__ == "__main__":
    unittest.main()
