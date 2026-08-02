from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np

from audit_causal_oracle_predictability import mean_kl
from audit_v18_calendar_fusion import select_scalar
from audit_v18_ohlcv_fusion import (
    log_ratio_feature_fusion,
    validation_timestamps,
)
from train_joint_price_oracle import CausalSegment


class OhlcvLogRatioFusionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.base = np.asarray([
            [0.65, 0.25, 0.10],
            [0.15, 0.35, 0.50],
            [0.45, 0.30, 0.25],
        ], dtype=np.float64)
        self.control = np.asarray([
            [0.40, 0.35, 0.25],
            [0.30, 0.40, 0.30],
            [0.35, 0.25, 0.40],
        ], dtype=np.float64)
        self.feature = np.asarray([
            [0.50, 0.30, 0.20],
            [0.25, 0.45, 0.30],
            [0.30, 0.35, 0.35],
        ], dtype=np.float64)

    def test_scalar_search_recovers_constructed_feature_weight(self) -> None:
        expected = 1.37
        targets = log_ratio_feature_fusion(
            self.base,
            self.feature,
            self.control,
            expected,
        )
        weight, score = select_scalar(
            lambda value: mean_kl(
                targets,
                log_ratio_feature_fusion(
                    self.base,
                    self.feature,
                    self.control,
                    value,
                ),
            ),
            0,
            4,
        )
        self.assertAlmostEqual(weight, expected, places=5)
        self.assertLess(score, 1e-10)

    def test_zero_weight_preserves_v18_exactly(self) -> None:
        np.testing.assert_allclose(
            log_ratio_feature_fusion(
                self.base,
                self.feature,
                self.control,
                0,
            ),
            self.base,
            atol=1e-15,
        )

    def test_equal_feature_and_control_are_neutral(self) -> None:
        np.testing.assert_allclose(
            log_ratio_feature_fusion(
                self.base,
                self.control,
                self.control,
                3.5,
            ),
            self.base,
            atol=1e-15,
        )


class OrderedTimestampTests(unittest.TestCase):
    def test_timestamp_rows_follow_segment_offsets_and_reject_reordering(self) -> None:
        first = CausalSegment(
            split="validation",
            prediction_time_start=1_000_999,
            count=2,
            target_file=Path("a.json"),
            target_row_offset=7,
            step_ms=60_000,
        )
        second = CausalSegment(
            split="validation",
            prediction_time_start=1_180_999,
            count=2,
            target_file=Path("b.json"),
            target_row_offset=0,
            step_ms=60_000,
        )
        np.testing.assert_array_equal(
            validation_timestamps([first, second]),
            np.asarray([1_000_999, 1_060_999, 1_180_999, 1_240_999]),
        )
        with self.assertRaisesRegex(ValueError, "chronological"):
            validation_timestamps([second, first])


if __name__ == "__main__":
    unittest.main()
