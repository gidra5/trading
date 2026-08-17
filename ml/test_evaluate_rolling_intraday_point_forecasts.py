import unittest

import numpy as np

from ml.evaluate_rolling_intraday_point_forecasts import (
    aggregate_complete_blocks,
    build_causal_minute_paths,
    day_bootstrap,
    deterministic_draws,
    forecast_window_medoid,
    round_capped_rows,
    rolling_ar1_parameters,
    selection_metrics,
    spread_ensemble,
)


class RollingIntradayPointForecastHelpersTest(unittest.TestCase):
    def test_complete_block_aggregation_preserves_feature_semantics(self) -> None:
        returns = np.arange(120, dtype=np.float64)
        variance = returns + 1.0
        counts = np.ones(120, dtype=np.uint8) * 2
        result = aggregate_complete_blocks(returns, variance, counts, 60)
        np.testing.assert_allclose(
            result["periodReturnBps"],
            [np.sum(returns[:60]), np.sum(returns[60:])],
        )
        np.testing.assert_allclose(
            result["oneSecondRealizedVarianceBpsSquared"],
            [np.sum(variance[:60]), np.sum(variance[60:])],
        )
        np.testing.assert_allclose(result["activeSeconds"], [120.0, 120.0])

    def test_deterministic_draws_repeat_per_origin_but_change_across_origins(self) -> None:
        left = deterministic_draws(1, 123, 16)
        repeat = deterministic_draws(1, 123, 16)
        right = deterministic_draws(1, 124, 16)
        for feature in left:
            np.testing.assert_allclose(left[feature], repeat[feature])
            self.assertFalse(np.array_equal(left[feature], right[feature]))

    def test_selection_metrics_prefer_exact_ensemble(self) -> None:
        actual = np.asarray([-2.0, 1.0, 3.0])
        exact = np.repeat(actual[:, None], 4, axis=1)
        diffuse = exact + np.asarray([-5.0, -2.0, 2.0, 5.0])[None, :]
        self.assertLess(
            selection_metrics(actual, exact)["selectionScore"],
            selection_metrics(actual, diffuse)["selectionScore"],
        )

    def test_vectorized_rolling_ar_parameters_match_direct_moments(self) -> None:
        values = np.asarray([0.2, -0.1, 0.5, 0.7, -0.4, 0.9, 0.3, -0.2])
        origins = np.asarray([6, 7, 8])
        result = rolling_ar1_parameters(values, origins, window=6)
        for row, origin in enumerate(origins):
            selected = values[origin - 6:origin]
            mean = float(np.mean(selected))
            centered = selected - mean
            raw_phi = float(
                np.sum(centered[1:] * centered[:-1])
                / np.sum(centered[:-1] * centered[:-1])
            )
            phi = float(np.clip(raw_phi * 6.0 / 26.0, -0.7, 0.97))
            residual = selected[1:] - (
                mean + phi * (selected[:-1] - mean)
            )
            self.assertAlmostEqual(result["mean"][row], mean)
            self.assertAlmostEqual(result["phi"][row], phi)
            self.assertAlmostEqual(result["last"][row], selected[-1])
            self.assertAlmostEqual(result["scale"][row], float(np.std(residual, ddof=1)))

    def test_day_bootstrap_reports_positive_perfect_forecast(self) -> None:
        actual = np.arange(12, dtype=np.float64) - 5.0
        result = day_bootstrap(actual, actual, blocks_per_day=4, draws=100)
        np.testing.assert_allclose(result["correlation95Interval"], [1.0, 1.0])
        np.testing.assert_allclose(result["mseSkill95Interval"], [1.0, 1.0])
        self.assertEqual(result["probabilityPositiveCorrelation"], 1.0)
        self.assertEqual(result["probabilityPositiveMseSkill"], 1.0)

    def test_spread_calibration_preserves_row_median(self) -> None:
        ensemble = np.asarray([
            [-3.0, -1.0, 1.0, 5.0],
            [10.0, 11.0, 12.0, 20.0],
        ])
        calibrated = spread_ensemble(ensemble, 1.5)
        np.testing.assert_allclose(
            np.median(calibrated, axis=1),
            np.median(ensemble, axis=1),
        )
        np.testing.assert_allclose(
            calibrated - np.median(calibrated, axis=1, keepdims=True),
            1.5 * (
                ensemble - np.median(ensemble, axis=1, keepdims=True)
            ),
        )

    def test_exact_capped_rounding_hits_requested_totals(self) -> None:
        values = np.asarray([[0.2, 1.7, 59.9], [20.4, 20.4, 20.4]])
        rounded = round_capped_rows(values, np.asarray([62.0, 61.0]))
        np.testing.assert_array_equal(np.sum(rounded, axis=1), [62, 61])
        self.assertLessEqual(int(np.max(rounded)), 60)

    def test_causal_minute_paths_reconcile_all_coarse_targets(self) -> None:
        rng = np.random.default_rng(81)
        minutes = 2_000
        returns = rng.normal(0.0, 0.3, minutes)
        variance = np.exp(rng.normal(-1.0, 0.2, minutes))
        counts = rng.integers(10, 50, minutes, dtype=np.uint8)
        target_return = np.asarray([[2.0, -1.0]])
        target_variance = np.asarray([[9.0, 12.0]])
        target_activity = np.asarray([[320.0, 500.0]])
        paths = build_causal_minute_paths(
            minute_returns=returns,
            minute_variance=variance,
            minute_counts=counts,
            origins=np.asarray([1_800]),
            horizon_minutes=15,
            history_days=1,
            target_return=target_return,
            target_variance=target_variance,
            target_activity=target_activity,
            seed=91,
        )
        np.testing.assert_allclose(
            np.sum(paths["periodReturnBps"], axis=2), target_return, atol=1e-6
        )
        np.testing.assert_allclose(
            np.sum(paths["oneSecondRealizedVarianceBpsSquared"], axis=2),
            target_variance,
            atol=1e-6,
        )
        np.testing.assert_array_equal(
            np.sum(paths["activeSeconds"], axis=2),
            np.rint(target_activity).astype(int),
        )

    def test_window_medoid_selects_central_cumulative_path(self) -> None:
        paths = np.zeros((3, 15 * 60), dtype=np.float64)
        paths[0, :60] = -1.0
        paths[1, :60] = 0.1
        paths[2, :60] = 1.0
        self.assertEqual(forecast_window_medoid(paths), 1)


if __name__ == "__main__":
    unittest.main()
