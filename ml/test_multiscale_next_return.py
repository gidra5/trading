from __future__ import annotations

from datetime import datetime, timezone
import unittest

import numpy as np
import torch

from multiscale_candle_resolution import (
    aligned_candle_indices,
    allowed_max_windows,
    daily_component_windows,
    daily_prediction_indices,
    daily_resolution_examples,
    resolution_component_labels,
    resolution_telescoping_components,
)
from multiscale_next_return import (
    JointInputGlu,
    MAX_WINDOW_LABELS,
    MultiscaleNormalization,
    SeparateComponentGlu,
    component_labels,
    optimizer_parameter_groups,
    selected_windows,
    telescoping_components,
    window_path_slug,
)
from multiscale_next_return_dataset import (
    daily_multiscale_return_examples,
    trim_shards_for_moving_average_history,
)
from next_return_dataset import HISTORY_RETURN_COUNT, ExampleShard


def normalization(components: int, horizon: int = 5) -> MultiscaleNormalization:
    return MultiscaleNormalization(
        feature_mean=np.zeros((components, HISTORY_RETURN_COUNT), np.float32),
        feature_std=np.ones((components, HISTORY_RETURN_COUNT), np.float32),
        component_target_mean=np.arange(
            components, dtype=np.float32
        )[:, None].repeat(horizon, axis=1) * 1e-5,
        component_target_std=np.ones((components, horizon), np.float32),
        component_cumulative_mean=np.zeros(components, np.float32),
        component_cumulative_std=np.ones(components, np.float32),
        raw_target_mean=np.arange(horizon, dtype=np.float32) * 1e-5,
        raw_target_std=np.ones(horizon, np.float32),
        raw_summary_mean=np.zeros(5, np.float32),
        raw_summary_std=np.ones(5, np.float32),
    )


class MultiscaleNextReturnTest(unittest.TestCase):
    def test_resolution_window_choices_and_labels(self) -> None:
        self.assertEqual(
            allowed_max_windows("1m"),
            ("1m", "1h", "1d", "1w", "1M", "3M"),
        )
        self.assertEqual(
            allowed_max_windows("1h"),
            ("1h", "1d", "1w", "1M", "3M"),
        )
        self.assertEqual(
            allowed_max_windows("1d"),
            ("1d", "1w", "1M", "3M"),
        )
        self.assertEqual(
            resolution_component_labels("1h", "1w"),
            ("ma_1w", "ma_1d_minus_ma_1w", "return_1h_minus_ma_1d"),
        )
        self.assertEqual(
            resolution_component_labels("1d", "1d"),
            ("return_1d",),
        )

    def test_resolution_components_reconstruct_raw_candles(self) -> None:
        generator = np.random.default_rng(19)
        raw = generator.normal(0, 1e-3, 1000)
        averages = {
            label: generator.normal(0, 1e-4, raw.size)
            for label in ("1h", "1d", "1w", "1M", "3M")
        }
        values = resolution_telescoping_components(
            raw, averages, resolution="1h", max_window="3M"
        )
        tolerance = (
            2 * np.finfo(np.float32).eps
            * np.abs(values).sum(axis=1, dtype=np.float32)
            + 1e-12
        )
        error = np.abs(values.sum(axis=1) - raw.astype(np.float32))
        self.assertTrue(bool((error <= tolerance).all()))

    def test_resolution_examples_use_aligned_candles(self) -> None:
        class AnalyticCloseCache:
            def load_range(self, start: datetime, count: int) -> np.ndarray:
                seconds = start.timestamp() + np.arange(count)
                return np.exp(seconds * 1e-9)

        minute_history, minute_target = daily_resolution_examples(
            AnalyticCloseCache(),  # type: ignore[arg-type]
            "2025-01-01",
            resolution="1m",
            max_window="1d",
        )
        self.assertEqual(minute_history.shape, (1440, 3, 120))
        self.assertEqual(minute_target.shape, (1440, 5))
        np.testing.assert_allclose(
            minute_history.sum(axis=1), 60e-9, rtol=0, atol=3e-15
        )
        np.testing.assert_allclose(
            minute_target, 60e-9, rtol=0, atol=3e-15
        )
        hour_history, hour_target = daily_resolution_examples(
            AnalyticCloseCache(),  # type: ignore[arg-type]
            "2025-01-01",
            resolution="1h",
            max_window="1w",
        )
        self.assertEqual(hour_history.shape, (24, 3, 120))
        self.assertEqual(hour_target.shape, (24, 5))
        np.testing.assert_allclose(
            hour_history.sum(axis=1), 3600e-9, rtol=0, atol=3e-13
        )
        np.testing.assert_allclose(
            hour_target, 3600e-9, rtol=0, atol=3e-13
        )

    def test_resolution_indices_align_source_shards(self) -> None:
        start = int(datetime(
            2025, 1, 1, 0, 0, 17, tzinfo=timezone.utc
        ).timestamp() * 1000) + 999
        shard = ExampleShard("train", start, 8_000, "2025-01-01", 17)
        minute, _weights = aligned_candle_indices(shard, 60)
        np.testing.assert_array_equal(minute[:3], (1, 2, 3))
        hour, _weights = aligned_candle_indices(shard, 3600)
        np.testing.assert_array_equal(hour[:2], (1, 2))

    def test_daily_component_windows_align_and_reconstruct_returns(self) -> None:
        log_closes = np.arange(400, dtype=np.float64) * 1e-3
        prediction_indices = np.asarray([210, 211, 300], dtype=np.int64)
        histories, targets = daily_component_windows(
            log_closes, prediction_indices, max_window="3M"
        )
        self.assertEqual(histories.shape, (3, 4, 120))
        self.assertEqual(targets.shape, (3, 5))
        np.testing.assert_allclose(
            histories.sum(axis=1), 1e-3, rtol=0, atol=2e-10
        )
        np.testing.assert_allclose(targets, 1e-3, rtol=0, atol=2e-10)

    def test_daily_splits_purge_targets_at_boundaries(self) -> None:
        splits = daily_prediction_indices(
            500, comparison_max_window="3M"
        )
        self.assertEqual(splits["validation"][0], 300)
        self.assertEqual(splits["test"][0], 400)
        self.assertLessEqual(splits["train"][-1] + 5, splits["validation"][0])
        self.assertLessEqual(splits["validation"][-1] + 5, splits["test"][0])

    def test_component_hierarchies_telescope_exactly(self) -> None:
        generator = np.random.default_rng(7)
        raw = generator.normal(0, 1e-4, 1000)
        averages = {
            label: generator.normal(0, 1e-5, raw.shape[0])
            for label in MAX_WINDOW_LABELS
        }
        for max_window in MAX_WINDOW_LABELS:
            components = telescoping_components(raw, averages, max_window)
            self.assertEqual(
                components.shape, (raw.size, len(component_labels(max_window)))
            )
            error = np.abs(
                components.sum(axis=1, dtype=np.float32)
                - raw.astype(np.float32)
            )
            tolerance = (
                2 * np.finfo(np.float32).eps
                * np.abs(components).sum(axis=1, dtype=np.float32)
                + 1e-12
            )
            self.assertTrue(bool((error <= tolerance).all()))

    def test_selected_windows_are_largest_to_smallest(self) -> None:
        self.assertEqual(
            tuple(label for label, _seconds in selected_windows("1w")),
            ("1w", "1d", "1h", "1m"),
        )
        self.assertEqual(
            component_labels("1h"),
            ("ma_1h", "ma_1m_minus_ma_1h", "return_minus_ma_1m"),
        )
        self.assertEqual(window_path_slug("1m"), "1m")
        self.assertEqual(window_path_slug("1M"), "1mo")

    def test_daily_examples_align_histories_and_targets(self) -> None:
        class AnalyticCloseCache:
            def load_range(self, start: datetime, count: int) -> np.ndarray:
                seconds = start.timestamp() + np.arange(count)
                return np.exp(seconds * 1e-9)

        history, target = daily_multiscale_return_examples(
            AnalyticCloseCache(),  # type: ignore[arg-type]
            "2025-01-01",
            max_window="3M",
            horizon_return_count=5,
        )
        self.assertEqual(history.shape, (86_400, 7, 120))
        self.assertEqual(target.shape, (86_400, 7, 5))
        np.testing.assert_allclose(
            history.sum(axis=1), 1e-9, rtol=0, atol=3e-16
        )
        np.testing.assert_allclose(
            target.sum(axis=1), 1e-9, rtol=0, atol=3e-16
        )

    def test_lookback_trimming_preserves_only_available_rows(self) -> None:
        start = int(datetime(
            2021, 7, 25, tzinfo=timezone.utc
        ).timestamp() * 1000) + 999
        shard = ExampleShard("train", start, 10_000_000, "2021-07-25", 0)
        trimmed = trim_shards_for_moving_average_history(
            {"train": [shard], "validation": [shard], "test": [shard]},
            first_history_day="2021-07-25",
            max_window="1m",
        )
        expected = 60 + HISTORY_RETURN_COUNT - 1
        self.assertEqual(trimmed["train"][0].row_offset, expected)
        self.assertEqual(
            trimmed["train"][0].decision_time_start, start + expected * 1000
        )

    def test_separate_and_joint_models_return_raw_paths(self) -> None:
        values = normalization(3)
        features = torch.zeros(4, 3, HISTORY_RETURN_COUNT)
        separate = SeparateComponentGlu(
            values, widths=(8, 8), dropout=0, dropout_rate=0
        )
        component_prediction = separate.forward_components(features)
        self.assertEqual(component_prediction.shape, (4, 3, 5))
        torch.testing.assert_close(
            separate(features), component_prediction.sum(dim=1)
        )
        joint = JointInputGlu(
            values, widths=(8, 8), dropout=0, dropout_rate=0
        )
        torch.testing.assert_close(
            joint(features),
            torch.from_numpy(values.raw_target_mean).repeat(4, 1),
        )
        for model in (separate, joint):
            muon, adamw = optimizer_parameter_groups(model)
            self.assertTrue(muon)
            self.assertTrue(adamw)


if __name__ == "__main__":
    unittest.main()
