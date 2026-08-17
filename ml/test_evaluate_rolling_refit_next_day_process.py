import unittest

import numpy as np

from evaluate_rolling_refit_next_day_process import (
    actual_archive_key,
    aggregate_history_blocks,
    counts_matching_period_activity,
    ensemble_archive_key,
    fit_ar1,
    generated_features,
    sample_local_period_targets,
)


class RollingRefitNextDayProcessTest(unittest.TestCase):
    def test_forecast_archive_keys_are_stable(self) -> None:
        self.assertEqual(
            actual_archive_key("15m", "periodReturnBps"),
            "actual__15m__periodReturnBps",
        )
        self.assertEqual(
            ensemble_archive_key(30, "4h", "activeSecondFraction"),
            "ensemble__30__4h__activeSecondFraction",
        )

    def test_history_is_aggregated_into_matching_horizon_blocks(self) -> None:
        returns = np.arange(15.0)
        variance = np.ones(15)
        counts = np.full(15, 30)
        block_return, block_variance, active = aggregate_history_blocks(
            returns,
            variance,
            counts,
            3,
        )
        np.testing.assert_array_equal(block_return, np.array([3.0, 12.0, 21.0, 30.0, 39.0]))
        np.testing.assert_array_equal(block_variance, np.full(5, 3.0))
        np.testing.assert_array_equal(active, np.full(5, 0.5))

    def test_activity_offset_matches_each_path_target(self) -> None:
        scores = np.tile(np.linspace(-2.0, 2.0, 1_000), (3, 1))
        probabilities = np.full(61, 1.0 / 61.0)
        targets = np.array([0.2, 0.5, 0.8])
        counts = counts_matching_period_activity(scores, probabilities, targets)
        np.testing.assert_allclose(np.mean(counts, axis=1) / 60.0, targets, atol=0.002)

    def test_local_targets_return_requested_ensemble_shape(self) -> None:
        rng = np.random.default_rng(4)
        variance = np.exp(rng.normal(8.0, 0.5, 100))
        returns = rng.normal(0.0, np.sqrt(variance))
        active = np.clip(rng.normal(0.6, 0.05, 100), 0.1, 0.9)
        targets = sample_local_period_targets(
            returns,
            variance,
            active,
            {
                "variance": rng.standard_t(8.0, 16),
                "return": rng.standard_t(8.0, 16),
                "activity": rng.standard_t(8.0, 16),
            },
        )
        for values in targets.values():
            self.assertEqual(values.shape, (16,))
            self.assertTrue(np.all(np.isfinite(values)))
        self.assertTrue(np.all(targets["variance"] > 0.0))

    def test_generated_features_use_period_sum(self) -> None:
        returns = np.array([[1.0, -0.5, 2.0], [-1.0, -1.0, 0.5]])
        q = np.ones_like(returns)
        counts = np.full_like(returns, 30, dtype=np.uint8)
        features = generated_features(returns, q, counts)
        np.testing.assert_allclose(features["periodReturnBps"], np.array([2.5, -1.5]))
        np.testing.assert_allclose(features["activeSecondFraction"], np.full(2, 0.5))

    def test_ar_fit_shrinks_short_sample_persistence(self) -> None:
        fit = fit_ar1(np.arange(10.0))
        self.assertLess(fit["phi"], 0.97)
        self.assertGreater(fit["scale"], 0.0)


if __name__ == "__main__":
    unittest.main()
