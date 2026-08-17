import unittest

import numpy as np

from analyze_parametric_one_second_process import (
    ArFactorGenerator,
    MINUTES_PER_DAY,
    FactorMixture,
    activity_mask_from_scores,
    apply_aligned_daily_variance_budgets,
    bounded_histogram_counts,
    calibrate_discrete_location,
    constrained_second_returns,
    discrete_midpoint_gaussian_scores,
    dirichlet_alpha_from_simpson,
    fit_covariance_targets,
    fit_gaussian_mixture_1d,
    fit_factor_mixture,
    inverse_discrete_gaussian_copula,
    project_returns_to_daily_targets,
)


class ParametricOneSecondProcessTest(unittest.TestCase):
    def test_activity_mask_preserves_each_requested_count(self) -> None:
        scores = np.array([
            [0.1, 0.4, -0.2, 0.3],
            [4.0, 3.0, 2.0, 1.0],
            [0.0, 0.0, 0.0, 0.0],
        ])
        counts = np.array([2, 4, 0])
        mask = activity_mask_from_scores(scores, counts)
        np.testing.assert_array_equal(mask.sum(axis=1), counts)
        np.testing.assert_array_equal(mask[0], np.array([False, True, False, True]))

    def test_dirichlet_concentration_inverts_expected_simpson_index(self) -> None:
        counts = np.array([5.0, 20.0])
        alpha = 0.4
        simpson = (alpha + 1.0) / (counts * alpha + 1.0)
        np.testing.assert_allclose(
            dirichlet_alpha_from_simpson(counts, simpson),
            np.full(2, alpha),
        )

    def test_covariance_fit_is_nonnegative_and_bounded_by_variance(self) -> None:
        fit = fit_covariance_targets(
            target_variance=1.0,
            target_lags=np.array([1, 2, 5]),
            target_covariance=np.array([0.5, 0.3, 0.1]),
            timescales=np.array([1.0, 5.0, 20.0]),
        )
        self.assertTrue(np.all(fit.weights >= 0.0))
        self.assertGreaterEqual(fit.white_variance, 0.0)
        self.assertLessEqual(np.sum(fit.weights), 1.0)

    def test_factor_fit_discards_lags_longer_than_history(self) -> None:
        fit = fit_factor_mixture(
            np.arange(10.0),
            timescales=np.array([2.0, 10.0]),
            lags=np.array([1, 5, 10, 20]),
        )
        np.testing.assert_array_equal(fit.target_lags, np.array([1, 5]))

    def test_ar_factor_generator_has_requested_stationary_variance(self) -> None:
        fit = FactorMixture(
            timescales=np.array([3.0]),
            weights=np.array([0.7]),
            white_variance=0.3,
            target_lags=np.array([1]),
            target_covariance=np.array([0.5]),
            fitted_covariance=np.array([0.5]),
        )
        values = ArFactorGenerator(fit, np.random.default_rng(7)).draw(200_000)
        self.assertAlmostEqual(float(np.var(values)), 1.0, delta=0.03)

    def test_discrete_gaussian_copula_maps_to_ordered_categories(self) -> None:
        probabilities = np.array([0.2, 0.5, 0.3])
        midpoint_scores = discrete_midpoint_gaussian_scores(probabilities)
        np.testing.assert_array_equal(
            inverse_discrete_gaussian_copula(midpoint_scores, probabilities),
            np.arange(3),
        )

    def test_constrained_returns_match_sum_and_energy(self) -> None:
        active = np.array([
            [True, True, True, False],
            [False, True, False, False],
            [False, False, False, False],
        ])
        base = np.array([
            [0.2, -2.0, 0.7, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ])
        target_return = np.array([1.5, -2.0, 0.0])
        target_energy = np.array([4.0, 4.0, 0.0])
        returns = constrained_second_returns(
            base, active, target_return, target_energy
        )
        np.testing.assert_allclose(np.sum(returns, axis=1), target_return)
        np.testing.assert_allclose(np.sum(returns * returns, axis=1), target_energy)
        self.assertTrue(np.all(returns[~active] == 0.0))

    def test_discrete_location_calibration_hits_requested_mean(self) -> None:
        scores = np.linspace(-2.0, 2.0, 10_000)
        probabilities = np.array([0.2, 0.3, 0.5])
        offset = calibrate_discrete_location(scores, probabilities, 1.25)
        generated = inverse_discrete_gaussian_copula(scores + offset, probabilities)
        self.assertAlmostEqual(float(np.mean(generated)), 1.25, places=3)

    def test_daily_budget_allocation_preserves_intraday_ratios_and_totals(self) -> None:
        minute_variance = np.ones(2 * MINUTES_PER_DAY - 1)
        minute_variance[MINUTES_PER_DAY - 1:] = np.tile(
            np.array([1.0, 3.0]),
            MINUTES_PER_DAY // 2,
        )
        result = apply_aligned_daily_variance_budgets(
            minute_variance,
            np.array([800.0]),
        )
        day = result[MINUTES_PER_DAY - 1:]
        self.assertAlmostEqual(float(np.sum(day)), 800.0)
        self.assertAlmostEqual(float(day[1] / day[0]), 3.0)

    def test_gaussian_mixture_fit_is_stochastic_and_ordered(self) -> None:
        rng = np.random.default_rng(11)
        values = np.concatenate((
            rng.normal(-2.0, 0.2, 20_000),
            rng.normal(1.0, 0.4, 30_000),
            rng.normal(3.0, 0.3, 10_000),
        ))
        weights, means, stds = fit_gaussian_mixture_1d(values, 3)
        self.assertAlmostEqual(float(np.sum(weights)), 1.0)
        self.assertTrue(np.all(np.diff(means) > 0.0))
        self.assertTrue(np.all(stds > 0.0))

    def test_bounded_histogram_keeps_underflow_and_overflow(self) -> None:
        counts = bounded_histogram_counts(
            np.array([-10.0, 0.25, 0.75, 10.0]),
            np.array([0.0, 0.5, 1.0]),
        )
        np.testing.assert_array_equal(counts, np.array([2, 2]))

    def test_daily_return_projection_hits_target_and_respects_bounds(self) -> None:
        minute_returns = np.zeros(MINUTES_PER_DAY)
        minute_variance = np.ones(MINUTES_PER_DAY)
        activity_counts = np.full(MINUTES_PER_DAY, 4)
        projected, fallbacks = project_returns_to_daily_targets(
            minute_returns,
            minute_variance,
            activity_counts,
            np.array([144.0]),
            aligned=False,
        )
        self.assertEqual(fallbacks, 0)
        self.assertAlmostEqual(float(np.sum(projected)), 144.0)
        self.assertTrue(np.all(np.abs(projected) < 2.0))

    def test_return_projection_supports_shorter_forecast_blocks(self) -> None:
        minute_returns = np.zeros(6)
        minute_variance = np.ones(6)
        activity_counts = np.full(6, 4)
        projected, fallbacks = project_returns_to_daily_targets(
            minute_returns,
            minute_variance,
            activity_counts,
            np.array([0.6, -0.3]),
            aligned=False,
            block_minutes=3,
        )
        self.assertEqual(fallbacks, 0)
        np.testing.assert_allclose(
            np.sum(projected.reshape(2, 3), axis=1),
            np.array([0.6, -0.3]),
        )


if __name__ == "__main__":
    unittest.main()
