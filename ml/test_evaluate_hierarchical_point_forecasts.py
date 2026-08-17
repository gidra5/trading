import unittest

import numpy as np

from ml.evaluate_hierarchical_point_forecasts import (
    BinaryAccumulator,
    BlendCalibrationAccumulator,
    ProbabilisticAccumulator,
    RegressionAccumulator,
    aggregate_seconds,
    forecast_only_medoid_member,
    fit_blend_calibration,
    local_density_mode,
    point_summaries,
    sorted_quantile,
)


class HierarchicalPointForecastHelpersTest(unittest.TestCase):
    def test_regression_metrics_match_direct_calculation(self) -> None:
        actual = np.asarray([-2.0, 0.0, 1.0, 3.0])
        predicted = np.asarray([-1.0, 0.0, -1.0, 2.0])
        accumulator = RegressionAccumulator()
        accumulator.add(actual[:2], predicted[:2])
        accumulator.add(actual[2:], predicted[2:])
        result = accumulator.finish()
        error = predicted - actual
        self.assertAlmostEqual(result["maeBps"], float(np.mean(np.abs(error))))
        self.assertAlmostEqual(result["rmseBps"], float(np.sqrt(np.mean(error ** 2))))
        self.assertAlmostEqual(
            result["mseSkillVsZeroReturn"],
            1.0 - float(np.mean(error ** 2)) / float(np.mean(actual ** 2)),
        )
        self.assertAlmostEqual(
            result["pearsonCorrelation"],
            float(np.corrcoef(actual, predicted)[0, 1]),
        )

    def test_sorted_quantile_interpolates_member_order_statistics(self) -> None:
        ordered = np.asarray([[0.0, 2.0, 4.0, 10.0]])
        self.assertAlmostEqual(float(sorted_quantile(ordered, 0.5)[0]), 3.0)
        self.assertAlmostEqual(float(sorted_quantile(ordered, 0.25)[0]), 1.5)

    def test_local_density_mode_honors_zero_atom_and_dense_cluster(self) -> None:
        ensemble = np.asarray([
            [0.0, 0.0, 0.0, 0.0, 2.0, 5.0],
            [-5.0, 1.0, 1.1, 1.2, 1.3, 8.0],
        ])
        result = local_density_mode(ensemble, neighbors=4)
        self.assertEqual(result[0], 0.0)
        self.assertAlmostEqual(result[1], 1.15)

    def test_probabilistic_crps_and_coverage_are_exact_for_identical_members(self) -> None:
        actual = np.asarray([1.0, -2.0])
        ensemble = np.asarray([[1.0, 1.0, 1.0], [-2.0, -2.0, -2.0]])
        accumulator = ProbabilisticAccumulator()
        accumulator.add(actual, ensemble)
        result = accumulator.finish()
        self.assertAlmostEqual(result["meanCrpsBps"], 0.0)
        for interval in result["centralIntervals"].values():
            self.assertAlmostEqual(interval["empiricalCoverage"], 1.0)
            self.assertAlmostEqual(interval["meanWidthBps"], 0.0)

    def test_binary_auc_and_brier_recognize_perfect_activity_forecast(self) -> None:
        accumulator = BinaryAccumulator()
        accumulator.add(
            np.asarray([False, False, True, True]),
            np.asarray([0.0, 0.1, 0.9, 1.0]),
        )
        result = accumulator.finish()
        self.assertAlmostEqual(result["approximateRocAuc"], 1.0)
        self.assertAlmostEqual(result["precision"], 1.0)
        self.assertAlmostEqual(result["recall"], 1.0)

    def test_second_aggregation_preserves_member_axis_and_sums(self) -> None:
        values = np.ones((3, 86_400), dtype=np.float64)
        minute = aggregate_seconds(values, 60)
        self.assertEqual(minute.shape, (3, 1_440))
        np.testing.assert_allclose(minute, 60.0)

    def test_medoid_and_point_summaries_do_not_require_realized_values(self) -> None:
        paths = np.zeros((3, 86_400), dtype=np.float64)
        paths[0, :60] = -1.0
        paths[1, :60] = 0.1
        paths[2, :60] = 1.0
        medoid = forecast_only_medoid_member(paths)
        self.assertEqual(medoid, 1)
        ensemble = np.asarray([[-1.0, 0.1, 1.0], [0.0, 0.0, 2.0]])
        summaries = point_summaries(ensemble, medoid, ordered=np.sort(ensemble, axis=1))
        np.testing.assert_allclose(summaries["pathMedoid"], ensemble[:, 1])
        np.testing.assert_allclose(summaries["ensembleMedian"], [0.1, 0.0])

    def test_blend_is_selected_only_when_it_beats_zero_in_every_fold(self) -> None:
        values = BlendCalibrationAccumulator()
        for fold in range(3):
            actual = np.asarray([1.0, -2.0, 3.0])
            summaries = {
                "ensembleMean": actual.copy(),
                "ensembleMedian": np.zeros(3),
                "localDensityMode": -actual,
            }
            values.add(actual, summaries, phase="training")
            values.add(actual, summaries, phase="validation", fold=fold)
        result = fit_blend_calibration(values)
        self.assertEqual(result["selectedCandidate"], "mean")
        np.testing.assert_allclose(result["selectedCoefficients"], [1.0, 0.0, 0.0])

    def test_blend_falls_back_to_zero_when_validation_signal_reverses(self) -> None:
        values = BlendCalibrationAccumulator()
        training_actual = np.asarray([1.0, -2.0, 3.0])
        training = {
            "ensembleMean": training_actual.copy(),
            "ensembleMedian": training_actual.copy(),
            "localDensityMode": training_actual.copy(),
        }
        values.add(training_actual, training, phase="training")
        for fold in range(3):
            validation_actual = -training_actual
            values.add(validation_actual, training, phase="validation", fold=fold)
        result = fit_blend_calibration(values)
        self.assertEqual(result["selectedCandidate"], "zero")
        np.testing.assert_allclose(result["selectedCoefficients"], [0.0, 0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
