from __future__ import annotations

from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from audit_v18_second_microstructure_fusion import (
    ACTION_COUNT,
    AccessLog,
    DAY_ROWS,
    FEATURE_NAMES,
    HORIZONS,
    MICRO_FEATURE_NAMES,
    MICRO_METRIC_NAMES,
    SECOND_ROWS,
    SecondDayCache,
    causal_second_microstructure_features,
    load_split,
    minute_micro_metrics,
)
from train_joint_price_oracle import CausalSegment


def synthetic_days(seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    count = 2 * SECOND_ROWS
    log_returns = rng.normal(0, 2e-5, size=count)
    closes = 50_000 * np.exp(np.cumsum(log_returns))
    opens = np.concatenate(([50_000.0], closes[:-1]))
    wick = rng.uniform(0, 2e-5, size=count)
    highs = np.maximum(opens, closes) * np.exp(wick)
    lows = np.minimum(opens, closes) / np.exp(wick * 0.8)
    volumes = rng.lognormal(-2, 0.8, size=count)
    values = np.column_stack((opens, highs, lows, closes, volumes))
    return values[:SECOND_ROWS], values[SECOND_ROWS:]


class MicrostructureFeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.previous, cls.current = synthetic_days()

    def test_feature_set_covers_every_metric_and_horizon(self) -> None:
        expected = {
            f"{metric}{horizon}m"
            for metric in MICRO_METRIC_NAMES
            for horizon in HORIZONS
        }
        self.assertTrue(expected.issubset(MICRO_FEATURE_NAMES))
        self.assertEqual(len(FEATURE_NAMES), len(set(FEATURE_NAMES)))

    def test_features_are_price_and_volume_scale_invariant(self) -> None:
        expected = causal_second_microstructure_features(
            self.previous,
            self.current,
        )
        previous = self.previous.copy()
        current = self.current.copy()
        previous[:, :4] *= 37
        current[:, :4] *= 37
        previous[:, 4] *= 11
        current[:, 4] *= 11
        actual = causal_second_microstructure_features(previous, current)
        np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)

    def test_target_row_excludes_still_open_remainder_of_current_minute(self) -> None:
        expected = causal_second_microstructure_features(
            self.previous,
            self.current,
        )
        changed = self.current.copy()
        changed[1:60, :4] *= 1.01
        changed[1:60, 4] *= 10
        actual = causal_second_microstructure_features(
            self.previous,
            changed,
        )
        np.testing.assert_allclose(actual[0], expected[0], atol=0, rtol=0)
        self.assertFalse(np.array_equal(actual[1], expected[1]))

    def test_minute_metrics_are_finite_and_bounded(self) -> None:
        metrics = minute_micro_metrics(
            self.current,
            float(self.previous[-1, 3]),
        )
        self.assertEqual(metrics.shape, (DAY_ROWS, len(MICRO_METRIC_NAMES)))
        self.assertTrue(np.isfinite(metrics).all())
        for index in (0, 1, 2, 4, 7):
            self.assertTrue(bool((metrics[:, index] >= 0).all()))
            self.assertTrue(bool((metrics[:, index] <= 1).all()))
        for index in (3, 5, 6):
            self.assertTrue(bool((metrics[:, index] >= -1).all()))
            self.assertTrue(bool((metrics[:, index] <= 1).all()))


class SealedAccessTests(unittest.TestCase):
    def test_test_split_is_rejected_before_any_reader_runs(self) -> None:
        segment = CausalSegment(
            split="test",
            prediction_time_start=999,
            count=1,
            target_file=Path("2026-01-02.json"),
            target_row_offset=0,
            step_ms=60_000,
        )
        access = AccessLog(set(), set())
        cache = SecondDayCache(Path("candles"), access.candle_files)
        with patch.object(cache, "load") as candle_reader, patch(
            "audit_v18_second_microstructure_fusion.read_shard_array",
        ) as target_reader:
            with self.assertRaisesRegex(ValueError, "only loads train or validation"):
                load_split("test", [segment], cache, access)
        candle_reader.assert_not_called()
        target_reader.assert_not_called()
        self.assertEqual(access.target_files, set())
        self.assertEqual(access.candle_files, set())

    def test_validation_target_slice_preserves_explicit_offset(self) -> None:
        segment = CausalSegment(
            split="validation",
            prediction_time_start=999,
            count=2,
            target_file=Path("2026-01-02.json"),
            target_row_offset=7,
            step_ms=60_000,
        )
        access = AccessLog(set(), set())
        cache = SecondDayCache(Path("candles"), access.candle_files)
        features = np.arange(
            DAY_ROWS * len(FEATURE_NAMES), dtype=np.float32,
        ).reshape(DAY_ROWS, len(FEATURE_NAMES))
        targets = np.full(
            (DAY_ROWS, ACTION_COUNT), 1 / ACTION_COUNT, dtype=np.float32,
        )
        with patch(
            "audit_v18_second_microstructure_fusion.daily_features",
            return_value=features,
        ), patch(
            "audit_v18_second_microstructure_fusion.read_shard_array",
            return_value=(None, targets),
        ):
            actual_features, actual_targets = load_split(
                "validation", [segment], cache, access,
            )
        np.testing.assert_array_equal(actual_features, features[7:9])
        np.testing.assert_array_equal(actual_targets, targets[7:9])


if __name__ == "__main__":
    unittest.main()
