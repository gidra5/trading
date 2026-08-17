import unittest

import numpy as np

from evaluate_parametric_next_day_process import (
    GaussianPosterior,
    ensemble_metrics,
    factor_next_observation,
    filter_factor_observations,
)
from analyze_parametric_one_second_process import FactorMixture


class ParametricNextDayProcessTest(unittest.TestCase):
    def test_kalman_filter_conditions_persistent_factor(self) -> None:
        fit = FactorMixture(
            timescales=np.array([10.0]),
            weights=np.array([0.8]),
            white_variance=0.2,
            target_lags=np.array([1]),
            target_covariance=np.array([0.7]),
            fitted_covariance=np.array([0.7]),
        )
        posterior = filter_factor_observations(fit, np.full(20, 1.5))
        mean, variance = factor_next_observation(fit, posterior)
        self.assertGreater(mean, 0.5)
        self.assertLess(variance, 1.0)

    def test_zero_weight_factor_stays_unconditioned(self) -> None:
        fit = FactorMixture(
            timescales=np.array([5.0]),
            weights=np.array([0.0]),
            white_variance=1.0,
            target_lags=np.array([1]),
            target_covariance=np.array([0.0]),
            fitted_covariance=np.array([0.0]),
        )
        posterior = filter_factor_observations(fit, np.arange(20.0))
        mean, variance = factor_next_observation(fit, posterior)
        self.assertAlmostEqual(mean, 0.0)
        self.assertAlmostEqual(variance, 1.0)

    def test_ensemble_metrics_reward_centered_sharp_forecast(self) -> None:
        rng = np.random.default_rng(8)
        actual = rng.normal(0.0, 1.0, 500)
        good = actual[:, None] + rng.normal(0.0, 0.2, (500, 128))
        bad = rng.normal(5.0, 2.0, (500, 128))
        good_metrics = ensemble_metrics(actual, good)
        bad_metrics = ensemble_metrics(actual, bad)
        self.assertLess(good_metrics["meanCrps"], bad_metrics["meanCrps"])
        self.assertLess(
            good_metrics["meanAbsoluteMedianError"],
            bad_metrics["meanAbsoluteMedianError"],
        )


if __name__ == "__main__":
    unittest.main()
