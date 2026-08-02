from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from audit_causal_oracle_ohlcv_predictability import (
    ACTION_COUNT,
    AccessLog,
    DAY_ROWS,
    MINUTE_MS,
    causal_ohlcv_features,
    completed_candle_windows,
    completed_close_windows,
    latest_closed_candle_start_ms,
    load_split,
    mean_kl_stacked_indexed,
)
from train_joint_price_oracle import CausalSegment


def day_start_ms(day_value: str) -> int:
    return int(datetime.combine(
        date.fromisoformat(day_value),
        datetime.min.time(),
        timezone.utc,
    ).timestamp() * 1_000)


def synthetic_day(marker: float) -> np.ndarray:
    index = np.arange(DAY_ROWS, dtype=np.float64)
    close = marker + index + 100
    return np.column_stack((
        close - 0.1,
        close + 0.5,
        close - 0.5,
        close,
        marker * 10 + index + 1,
    ))


class CompletedMinuteAlignmentTests(unittest.TestCase):
    def test_prediction_at_second_999_excludes_current_minute(self) -> None:
        start = day_start_ms("2026-01-02")
        self.assertEqual(
            latest_closed_candle_start_ms(start + 999),
            start - MINUTE_MS,
        )
        self.assertEqual(
            latest_closed_candle_start_ms(start + MINUTE_MS + 999),
            start,
        )
        self.assertEqual(
            latest_closed_candle_start_ms(start + 59_999),
            start,
        )

    def test_row_windows_end_on_previous_completed_candle(self) -> None:
        previous = synthetic_day(1_000)
        current = synthetic_day(10_000)
        candles = completed_candle_windows(previous, current)
        closes = completed_close_windows(previous, current)

        np.testing.assert_array_equal(candles[0], previous[-60:])
        np.testing.assert_array_equal(candles[1, :-1], previous[-59:])
        np.testing.assert_array_equal(candles[1, -1], current[0])
        np.testing.assert_array_equal(closes[0], previous[-61:, 3])
        self.assertEqual(candles[-1, -1, 3], current[-2, 3])
        self.assertNotEqual(candles[-1, -1, 3], current[-1, 3])

    def test_features_are_invariant_to_price_and_volume_units(self) -> None:
        rng = np.random.default_rng(7)
        rows = 4
        base = np.exp(rng.normal(10, 0.01, size=(rows, 60)))
        opens = base
        closes = opens * np.exp(rng.normal(0, 0.001, size=(rows, 60)))
        highs = np.maximum(opens, closes) * np.exp(
            rng.uniform(0, 0.001, size=(rows, 60))
        )
        lows = np.minimum(opens, closes) / np.exp(
            rng.uniform(0, 0.001, size=(rows, 60))
        )
        volumes = np.exp(rng.normal(3, 0.5, size=(rows, 60)))
        candles = np.stack((opens, highs, lows, closes, volumes), axis=2)
        close_windows = np.concatenate((
            opens[:, :1],
            closes,
        ), axis=1)
        expected = causal_ohlcv_features(candles, close_windows)

        scaled = candles.copy()
        scaled[:, :, :4] *= 37.0
        scaled[:, :, 4] *= 11.0
        actual = causal_ohlcv_features(scaled, close_windows * 37.0)
        np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-6)


class SealedTestAccessTests(unittest.TestCase):
    def test_loader_rejects_test_before_opening_any_payload(self) -> None:
        segment = CausalSegment(
            split="test",
            prediction_time_start=day_start_ms("2026-01-03") + 999,
            count=1,
            target_file=Path("2026-01-03.json"),
            target_row_offset=0,
            step_ms=MINUTE_MS,
        )
        access = AccessLog(set(), set())
        with patch(
            "audit_causal_oracle_ohlcv_predictability.read_shard_array",
        ) as target_reader, patch(
            "audit_causal_oracle_ohlcv_predictability.daily_causal_ohlcv_features",
        ) as feature_reader:
            with self.assertRaisesRegex(ValueError, "only load train or validation"):
                load_split(
                    "test",
                    [segment],
                    Path("candles"),
                    {},
                    access,
                )
        target_reader.assert_not_called()
        feature_reader.assert_not_called()
        self.assertEqual(access.target_files, set())
        self.assertEqual(access.candle_files, set())

        with patch(
            "audit_causal_oracle_ohlcv_predictability.read_shard_array",
        ) as target_reader, patch(
            "audit_causal_oracle_ohlcv_predictability.daily_causal_ohlcv_features",
        ) as feature_reader:
            with self.assertRaisesRegex(ValueError, "segment split"):
                load_split(
                    "validation",
                    [segment],
                    Path("candles"),
                    {},
                    access,
                )
        target_reader.assert_not_called()
        feature_reader.assert_not_called()

    def test_validation_loader_opens_only_its_explicit_segment(self) -> None:
        day = "2026-01-02"
        target_path = Path(f"{day}.json")
        segment = CausalSegment(
            split="validation",
            prediction_time_start=day_start_ms(day) + 999,
            count=2,
            target_file=target_path,
            target_row_offset=7,
            step_ms=MINUTE_MS,
        )
        targets = np.full(
            (DAY_ROWS, ACTION_COUNT),
            1 / ACTION_COUNT,
            dtype=np.float32,
        )
        features = np.zeros((DAY_ROWS, 32), dtype=np.float32)
        access = AccessLog(set(), set())
        with patch(
            "audit_causal_oracle_ohlcv_predictability.read_shard_array",
            return_value=(None, targets),
        ) as target_reader, patch(
            "audit_causal_oracle_ohlcv_predictability.daily_causal_ohlcv_features",
            return_value=features,
        ) as feature_reader:
            loaded_features, loaded_targets = load_split(
                "validation",
                [segment],
                Path("candles"),
                {},
                access,
            )
        self.assertEqual(loaded_features.shape, (2, 32))
        self.assertEqual(loaded_targets.shape, (2, ACTION_COUNT))
        target_reader.assert_called_once_with(
            target_path.resolve(),
            "<f4",
            (DAY_ROWS, ACTION_COUNT),
        )
        feature_reader.assert_called_once()


class StackedKlTests(unittest.TestCase):
    def test_indexed_stacking_has_exact_endpoints(self) -> None:
        targets = np.asarray([[0.8, 0.2], [0.3, 0.7]], dtype=np.float64)
        first = np.asarray([[0.6, 0.4], [0.4, 0.6]], dtype=np.float64)
        second = np.asarray([[0.9, 0.1], [0.2, 0.8]], dtype=np.float64)
        ids = np.asarray([0, 1], dtype=np.int64)
        first_score = mean_kl_stacked_indexed(
            targets, ids, first, ids, second, 0,
        )
        second_score = mean_kl_stacked_indexed(
            targets, ids, first, ids, second, 1,
        )
        expected_first = float(np.mean(np.sum(
            targets * (np.log(targets) - np.log(first)), axis=1,
        )))
        expected_second = float(np.mean(np.sum(
            targets * (np.log(targets) - np.log(second)), axis=1,
        )))
        self.assertAlmostEqual(first_score, expected_first)
        self.assertAlmostEqual(second_score, expected_second)


if __name__ == "__main__":
    unittest.main()
