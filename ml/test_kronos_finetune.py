from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import torch

from finetune_kronos import (
    KronosWindowDataset,
    SeriesRange,
    load_exclusion_plan,
    predictor_losses,
)


class KronosFinetuneDatasetTest(unittest.TestCase):
    def test_normalization_uses_only_historical_lookback(self) -> None:
        timestamps = np.arange(6, dtype=np.int64) * 60_000
        history = np.arange(1, 5, dtype=np.float32)[:, None]
        future = np.array([[1_000], [2_000]], dtype=np.float32)
        values = np.repeat(np.concatenate((history, future)), 6, axis=1)
        dataset = KronosWindowDataset(
            SeriesRange(timestamps=timestamps, values=values),
            window=6,
            lookback=4,
            samples=1,
            seed=1,
            training=False,
        )

        normalized, _ = dataset[0]

        np.testing.assert_allclose(
            normalized[:4].numpy().mean(axis=0), 0, atol=1e-6
        )
        np.testing.assert_allclose(
            normalized[:4].numpy().std(axis=0), 1, atol=1e-5
        )
        np.testing.assert_array_equal(normalized[4:].numpy(), 5)

    def test_training_origin_is_deterministic_within_epoch(self) -> None:
        timestamps = np.arange(20, dtype=np.int64) * 60_000
        values = np.repeat(
            np.arange(1, 21, dtype=np.float32)[:, None], 6, axis=1
        )
        dataset = KronosWindowDataset(
            SeriesRange(timestamps=timestamps, values=values),
            window=6,
            lookback=4,
            samples=4,
            seed=1_337,
            training=True,
        )

        first, first_stamps = dataset[2]
        second, second_stamps = dataset[2]

        np.testing.assert_array_equal(first.numpy(), second.numpy())
        np.testing.assert_array_equal(first_stamps.numpy(), second_stamps.numpy())

    def test_exclusions_remove_every_intersecting_candidate_window(self) -> None:
        timestamps = np.arange(20, dtype=np.int64) * 60_000
        values = np.repeat(
            np.arange(1, 21, dtype=np.float32)[:, None], 6, axis=1
        )
        dataset = KronosWindowDataset(
            SeriesRange(timestamps=timestamps, values=values),
            window=4,
            lookback=2,
            samples=100,
            seed=1_337,
            training=True,
            excluded_ranges=((5 * 60_000, 9 * 60_000),),
        )

        self.assertEqual(dataset.excluded_candidate_starts, 7)
        self.assertEqual(
            set(dataset.start_indexes.tolist()),
            {0, 1, 9, 10, 11, 12, 13, 14, 15, 16},
        )
        for index in range(len(dataset)):
            start = dataset.sample_start_index(index)
            self.assertTrue(start + 4 <= 5 or start >= 9)

    def test_policy_holdout_plan_is_valid_and_content_addressed(self) -> None:
        plan = load_exclusion_plan(
            Path(__file__).resolve().parent
            / "training-plans/kronos-btcusdt-1m-policy-holdout-v1.json"
        )

        self.assertEqual(len(plan.ranges), 13)
        self.assertEqual(len(plan.fingerprint), 64)
        self.assertEqual(
            plan.source["contract"],
            "kronos-btcusdt-1m-finetune-exclusion-plan-v1",
        )

    def test_forecast_weight_targets_only_the_requested_future_positions(self) -> None:
        class MeanTargetHead:
            @staticmethod
            def compute_loss(_logits_0, _logits_1, targets_0, targets_1):
                loss = (targets_0.float().mean() + targets_1.float().mean()) / 2
                return loss, loss, loss

        tokens = torch.arange(7, dtype=torch.long).unsqueeze(0)
        logits = (
            torch.zeros((1, 6, 1)),
            torch.zeros((1, 6, 1)),
        )
        losses = predictor_losses(
            MeanTargetHead(),
            logits,
            tokens,
            tokens,
            lookback=4,
            horizon=2,
            forecast_loss_weight=3,
        )

        self.assertAlmostEqual(float(losses.full), 3.5)
        self.assertAlmostEqual(float(losses.forecast), 4.5)
        self.assertAlmostEqual(float(losses.objective), 4.25)


if __name__ == "__main__":
    unittest.main()
