from __future__ import annotations

import unittest

import numpy as np
import torch

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
from train_feature_compressed_path_density import (
    embedding_dropout_probability_at_epoch,
    expected_return_correlation_loss_weight_at_epoch,
    expected_return_mse_loss_weight_at_epoch,
    rolling_online_per_step_affine,
    weighted_expected_return_correlation,
    weighted_expected_return_normalized_mse,
)


class CompressedPathCalibrationTests(unittest.TestCase):
    def test_weighted_expected_return_correlation_is_directional(self) -> None:
        target = torch.tensor([[-2.0], [-1.0], [1.0], [2.0], [100.0]])
        weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0])
        positive = weighted_expected_return_correlation(
            target.clone(), target, weights, target_std=1.0
        )
        negative = weighted_expected_return_correlation(
            -target, target, weights, target_std=1.0
        )
        self.assertGreater(float(positive), 0.999999)
        self.assertLess(float(negative), -0.999999)

    def test_expected_return_correlation_has_finite_constant_start_gradient(self) -> None:
        prediction = torch.zeros((4, 1), requires_grad=True)
        target = torch.tensor([[-2.0], [-1.0], [1.0], [2.0]])
        correlation = weighted_expected_return_correlation(
            prediction, target, torch.ones(4), target_std=1.0
        )
        (1.0 - correlation).backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())
        self.assertGreater(float(prediction.grad.abs().sum()), 0.0)

    def test_expected_return_normalized_mse_ignores_padding(self) -> None:
        prediction = torch.tensor([[1.0], [3.0], [100.0]])
        target = torch.tensor([[0.0], [1.0], [-100.0]])
        weights = torch.tensor([1.0, 1.0, 0.0])
        loss = weighted_expected_return_normalized_mse(
            prediction, target, weights, target_std=2.0
        )
        self.assertAlmostEqual(float(loss), 0.625)

    def test_linear_expected_return_mse_weight_hits_exact_endpoints(self) -> None:
        training = {
            "expectedReturnMseLossWeight": 1.0,
            "expectedReturnMseLossWeightSchedule": {
                "type": "linear-v1",
                "startWeight": 1.0,
                "endWeight": 0.01,
                "endEpoch": 24,
            },
        }
        self.assertEqual(expected_return_mse_loss_weight_at_epoch(training, 0), 1.0)
        self.assertAlmostEqual(
            expected_return_mse_loss_weight_at_epoch(training, 12), 0.505
        )
        self.assertEqual(expected_return_mse_loss_weight_at_epoch(training, 24), 0.01)
        self.assertEqual(expected_return_mse_loss_weight_at_epoch(training, 48), 0.01)

    def test_geometric_expected_return_mse_weight_hits_exact_endpoints(self) -> None:
        training = {
            "expectedReturnMseLossWeight": 1.0,
            "expectedReturnMseLossWeightSchedule": {
                "type": "geometric-v1",
                "startWeight": 1.0,
                "endWeight": 0.01,
                "endEpoch": 48,
            },
        }
        self.assertEqual(expected_return_mse_loss_weight_at_epoch(training, 0), 1.0)
        self.assertAlmostEqual(
            expected_return_mse_loss_weight_at_epoch(training, 24), 0.1
        )
        self.assertEqual(expected_return_mse_loss_weight_at_epoch(training, 48), 0.01)
        self.assertEqual(expected_return_mse_loss_weight_at_epoch(training, 96), 0.01)

    def test_delayed_geometric_expected_return_mse_weight(self) -> None:
        training = {
            "expectedReturnMseLossWeight": 1.0,
            "expectedReturnMseLossWeightSchedule": {
                "type": "geometric-v1",
                "startWeight": 1.0,
                "endWeight": 0.01,
                "startEpoch": 16,
                "endEpoch": 32,
            },
        }
        self.assertEqual(expected_return_mse_loss_weight_at_epoch(training, 0), 1.0)
        self.assertEqual(expected_return_mse_loss_weight_at_epoch(training, 16), 1.0)
        self.assertAlmostEqual(
            expected_return_mse_loss_weight_at_epoch(training, 24), 0.1
        )
        self.assertEqual(expected_return_mse_loss_weight_at_epoch(training, 32), 0.01)
        self.assertEqual(expected_return_mse_loss_weight_at_epoch(training, 64), 0.01)

    def test_delayed_geometric_expected_return_correlation_weight(self) -> None:
        training = {
            "expectedReturnCorrelationLossWeight": 1.0,
            "expectedReturnCorrelationLossWeightSchedule": {
                "type": "geometric-v1",
                "startWeight": 1.0,
                "endWeight": 0.01,
                "startEpoch": 16,
                "endEpoch": 32,
            },
        }
        self.assertEqual(
            expected_return_correlation_loss_weight_at_epoch(training, 16), 1.0
        )
        self.assertAlmostEqual(
            expected_return_correlation_loss_weight_at_epoch(training, 24), 0.1
        )
        self.assertEqual(
            expected_return_correlation_loss_weight_at_epoch(training, 32), 0.01
        )

    def test_geometric_keep_dropout_schedule_hits_exact_endpoints(self) -> None:
        training = {
            "embeddingDropoutProbability": 0.8,
            "embeddingDropoutSchedule": {
                "type": "geometric-keep-probability-v1",
                "startProbability": 0.8,
                "endProbability": 0.0,
                "endEpoch": 256,
            },
        }
        self.assertAlmostEqual(
            embedding_dropout_probability_at_epoch(training, 0), 0.8
        )
        self.assertAlmostEqual(
            embedding_dropout_probability_at_epoch(training, 128),
            1.0 - np.sqrt(0.2),
        )
        self.assertEqual(
            embedding_dropout_probability_at_epoch(training, 256), 0.0
        )
        self.assertEqual(
            embedding_dropout_probability_at_epoch(training, 512), 0.0
        )

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
