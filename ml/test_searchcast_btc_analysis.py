from __future__ import annotations

import unittest

import numpy as np

from searchcast_btc_analysis import (
    Trial,
    aggregate_complete_weeks,
    aggregate_fixed,
    CandleFrame,
    effective_local_count,
    normalize_evaluation,
    normalize_training,
    ridge_alpha_grid,
    sample_windows,
)


class SearchCastBtcAnalysisTest(unittest.TestCase):
    def test_sample_windows_respects_segment_boundaries_and_time_range(self) -> None:
        segments = [
            (np.arange(10, dtype=np.int64) * 1_000, np.arange(10, dtype=np.float64)),
            (100_000 + np.arange(10, dtype=np.int64) * 1_000,
             100.0 + np.arange(10, dtype=np.float64)),
        ]
        batch = sample_windows(segments, 0, 200_000, 3, 2, 100)
        self.assertEqual(batch.x.shape, (12, 3))
        self.assertTrue(np.all(np.diff(batch.x, axis=1) == 1))
        self.assertFalse(bool(np.any((batch.x[:, -1] < 10) & (batch.y[:, 0] >= 100))))

    def test_local_normalization_round_trip_contract(self) -> None:
        x = np.asarray([[1.0, 2.0, 3.0, 4.0], [10.0, 12.0, 14.0, 16.0]])
        y = np.asarray([[5.0, 6.0], [18.0, 20.0]])
        trial = Trial(4, "local", "standard", 0.5, "none", 0.0)
        design, normalized_y, state, parameters = normalize_training(x, y, trial, seed=1)
        evaluation, evaluation_state = normalize_evaluation(x, trial, parameters)
        np.testing.assert_allclose(design, evaluation)
        np.testing.assert_allclose(
            normalized_y * evaluation_state.scales + evaluation_state.means,
            y,
        )
        self.assertEqual(effective_local_count(4, 0.01), 2)

    def test_ridge_grid_fits_simple_linear_relation(self) -> None:
        x = np.column_stack((np.arange(1.0, 8.0), np.ones(7)))
        y = (2.0 * x[:, :1]) + 3.0
        coefficient = ridge_alpha_grid(x, y, np.asarray([1e-9]))[0]
        np.testing.assert_allclose(x @ coefficient, y, atol=1e-7)

    def test_fixed_aggregation_preserves_ohlcv_semantics(self) -> None:
        frame = CandleFrame(
            scale="1m",
            timestamps=np.arange(6, dtype=np.int64) * 60_000,
            open=np.asarray([1, 2, 3, 4, 5, 6], dtype=np.float64),
            high=np.asarray([2, 3, 4, 5, 6, 7], dtype=np.float64),
            low=np.asarray([0, 1, 2, 3, 4, 5], dtype=np.float64),
            close=np.asarray([1.5, 2.5, 3.5, 4.5, 5.5, 6.5]),
            volume=np.ones(6),
            segment_bounds=[(0, 6)],
            source_rows=6,
            expected_source_rows=6,
            sampling_note="fixture",
        )
        aggregated = aggregate_fixed(frame, 3, "3m")
        np.testing.assert_allclose(aggregated.open, [1, 4])
        np.testing.assert_allclose(aggregated.high, [4, 7])
        np.testing.assert_allclose(aggregated.low, [0, 3])
        np.testing.assert_allclose(aggregated.close, [3.5, 6.5])
        np.testing.assert_allclose(aggregated.volume, [3, 3])

    def test_weekly_aggregation_keeps_only_complete_monday_sunday_weeks(self) -> None:
        dates = np.arange(
            np.datetime64("2024-01-07"),
            np.datetime64("2024-01-16"),
            dtype="datetime64[D]",
        )
        rows = dates.size
        frame = CandleFrame(
            scale="1d",
            timestamps=dates.astype("datetime64[ms]").astype(np.int64),
            open=np.arange(1, rows + 1, dtype=np.float64),
            high=np.arange(2, rows + 2, dtype=np.float64),
            low=np.arange(0, rows, dtype=np.float64),
            close=np.arange(1.5, rows + 1.5, dtype=np.float64),
            volume=np.ones(rows),
            segment_bounds=[(0, rows)],
            source_rows=rows,
            expected_source_rows=rows,
            sampling_note="fixture",
        )
        weekly = aggregate_complete_weeks(frame)
        self.assertEqual(weekly.timestamps.size, 1)
        self.assertEqual(
            weekly.timestamps[0],
            np.datetime64("2024-01-08", "ms").astype(np.int64),
        )
        np.testing.assert_allclose(weekly.open, [2])
        np.testing.assert_allclose(weekly.high, [9])
        np.testing.assert_allclose(weekly.low, [1])
        np.testing.assert_allclose(weekly.close, [8.5])
        np.testing.assert_allclose(weekly.volume, [7])


if __name__ == "__main__":
    unittest.main()
