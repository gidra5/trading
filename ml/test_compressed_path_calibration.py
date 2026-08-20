from __future__ import annotations

import unittest

import numpy as np

from evaluate_compressed_path_checkpoint_metrics import (
    apply_affine_matrix,
    apply_polynomial,
    expectation_calibration_variants_from_arrays,
    fit_affine_matrix,
    fit_polynomial,
    per_lead_metric_results,
    rolling_affine_matrix_predictions,
    rolling_per_step_polynomial_predictions,
    rolling_polynomial_predictions,
)
from train_feature_compressed_path_density import rolling_online_per_step_affine


class CompressedPathCalibrationTests(unittest.TestCase):
    def test_training_online_affine_matches_checkpoint_evaluator(self) -> None:
        rng = np.random.default_rng(19)
        history_prediction = rng.normal(size=(80, 3))
        history_target = 0.4 * history_prediction + rng.normal(
            scale=0.2, size=(80, 3)
        )
        prediction = rng.normal(size=(25, 3))
        target = 0.4 * prediction + rng.normal(scale=0.2, size=(25, 3))
        input_scales = np.maximum(history_prediction.std(axis=0), 1e-12)
        output_scales = np.maximum(history_target.std(axis=0), 1e-12)
        expected = rolling_per_step_polynomial_predictions(
            history_prediction, history_target[:, 0], prediction, target[:, 0],
            degree=1, input_scales=input_scales,
            output_scales=output_scales, ridge=1e-8, window=64,
        )
        actual = rolling_online_per_step_affine(
            history_prediction, history_target, prediction, target,
            input_scales=input_scales, output_scales=output_scales,
            ridge=1e-8, window=64,
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-11)
    def test_calibration_window_uses_only_the_trailing_examples(self) -> None:
        prediction = np.arange(12, dtype=np.float64).reshape(6, 2) / 100
        target = prediction.copy()
        arrays = {
            split: {
                "logPrediction": prediction,
                "arithmeticPrediction": np.expm1(prediction),
                "logTarget": target,
                "arithmeticTarget": np.expm1(target),
            }
            for split in ("calibration", "validation", "test")
        }
        variants = expectation_calibration_variants_from_arrays(
            arrays, target_std=1.0, calibration_window=4
        )
        affine = variants["affine-log"]
        self.assertEqual(affine["calibrationWindow"], 4)
        self.assertEqual(affine["calibration"]["examples"], 8)

    def test_per_lead_metrics_preserve_each_forecast_horizon(self) -> None:
        target = np.asarray([[1.0, -1.0], [2.0, -2.0]], dtype=np.float64)
        prediction = np.asarray([[1.0, 1.0], [2.0, 2.0]], dtype=np.float64)
        metrics = per_lead_metric_results(prediction, target, target_std=1.0)
        self.assertEqual(len(metrics), 2)
        self.assertAlmostEqual(metrics[0]["mseSkillVsZero"], 1.0)
        self.assertLess(metrics[1]["mseSkillVsZero"], 0.0)

    def test_cubic_fit_recovers_standardized_mapping(self) -> None:
        prediction = np.linspace(-2, 2, 401, dtype=np.float64)
        expected = 0.1 + 0.7 * prediction - 0.2 * prediction**2 \
            + 0.05 * prediction**3
        coefficients = fit_polynomial(
            prediction, expected, degree=3, input_scale=1,
            output_scale=1, ridge=0,
        )
        actual = apply_polynomial(
            prediction, coefficients, input_scale=1, output_scale=1
        )
        np.testing.assert_allclose(actual, expected, atol=1e-11)

    def test_online_first_forecast_does_not_consume_current_target(self) -> None:
        history_prediction = np.linspace(-1, 1, 40).reshape(20, 2)
        history_target = 0.5 * history_prediction[:, 0]
        prediction = np.asarray([
            [0.2, -0.3], [0.4, 0.1], [-0.2, 0.5], [0.7, -0.1]
        ])
        target_a = np.asarray([0.1, 0.2, -0.3, 0.4])
        target_b = np.asarray([100.0, -200.0, 300.0, -400.0])
        first = rolling_polynomial_predictions(
            history_prediction, history_target, prediction, target_a,
            degree=1, input_scale=1, output_scale=1, ridge=0,
            window=16,
        )
        second = rolling_polynomial_predictions(
            history_prediction, history_target, prediction, target_b,
            degree=1, input_scale=1, output_scale=1, ridge=0,
            window=16,
        )
        np.testing.assert_allclose(first[0], second[0], atol=1e-12)

    def test_per_step_online_first_forecast_does_not_consume_current_target(self) -> None:
        history_prediction = np.linspace(-1, 1, 60).reshape(20, 3)
        history_target = 0.5 * history_prediction[:, 0]
        prediction = np.asarray([
            [0.2, -0.3, 0.1], [0.4, 0.1, -0.2],
            [-0.2, 0.5, 0.3], [0.7, -0.1, 0.6],
        ])
        target_a = np.asarray([0.1, 0.2, -0.3, 0.4])
        target_b = np.asarray([100.0, -200.0, 300.0, -400.0])
        arguments = {
            "degree": 1,
            "input_scales": np.ones(3),
            "output_scales": np.ones(3),
            "ridge": 0,
            "window": 16,
        }
        first = rolling_per_step_polynomial_predictions(
            history_prediction, history_target, prediction, target_a, **arguments
        )
        second = rolling_per_step_polynomial_predictions(
            history_prediction, history_target, prediction, target_b, **arguments
        )
        np.testing.assert_allclose(first[0], second[0], atol=1e-12)

    def test_matrix_affine_recovers_cross_step_linear_map(self) -> None:
        generator = np.random.default_rng(7)
        prediction = generator.normal(size=(500, 3))
        matrix = np.asarray([
            [0.8, -0.2, 0.1],
            [0.3, 0.5, -0.4],
            [-0.1, 0.2, 0.9],
        ])
        bias = np.asarray([0.1, -0.2, 0.05])
        target = prediction @ matrix.T + bias
        coefficients, input_scales, output_scales = fit_affine_matrix(
            prediction, target, ridge=0
        )
        actual = apply_affine_matrix(
            prediction, coefficients, input_scales, output_scales
        )
        np.testing.assert_allclose(actual, target, atol=1e-11)

    def test_online_matrix_does_not_consume_unresolved_current_path(self) -> None:
        generator = np.random.default_rng(11)
        history_prediction = generator.normal(size=(40, 3))
        history_target = 0.5 * history_prediction
        prediction = generator.normal(size=(8, 3))
        target_a = generator.normal(size=(8, 3))
        target_b = generator.normal(size=(8, 3)) * 1_000
        arguments = {
            "input_scales": np.ones(3),
            "output_scales": np.ones(3),
            "ridge": 1e-3,
            "window": 32,
        }
        first = rolling_affine_matrix_predictions(
            history_prediction, history_target, prediction, target_a, **arguments
        )
        second = rolling_affine_matrix_predictions(
            history_prediction, history_target, prediction, target_b, **arguments
        )
        np.testing.assert_allclose(first[:3], second[:3], atol=1e-12)


if __name__ == "__main__":
    unittest.main()
