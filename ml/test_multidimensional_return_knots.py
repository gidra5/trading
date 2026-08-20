from __future__ import annotations

import unittest

import numpy as np
from scipy.linalg import expm

from multidimensional_return_knots import (
    AsinhMatrixTransform,
    CovarianceTransform,
    conditional_operation_metrics,
    factorized_matrix,
    grid_centers,
    point_cloud_bin_probabilities,
    rectangular_grid_bin_probabilities,
    sample_each_point_cloud_component,
    sample_point_cloud,
    select_nonredundant_point_cloud_centers,
)
from search_multidimensional_return_knots import (
    acceptance_metrics,
    density_tempering_weights,
    fast_weighted_lloyd,
)
from joint_optimize_multidimensional_return_knots import (
    JointState,
    covariance_directions,
    factorize_matrix,
    joint_state_from_payload,
    joint_state_payload,
    transform_directions,
)


class MultidimensionalReturnKnotsTest(unittest.TestCase):
    def test_fast_weighted_lloyd_improves_warm_centers(self) -> None:
        rng = np.random.default_rng(31)
        values = np.vstack((
            rng.normal(0.25, 0.04, size=(250, 2)),
            rng.normal(0.75, 0.06, size=(250, 2)),
        ))
        values = np.clip(values, 0.0, 1.0)
        weights = np.linspace(0.5, 1.5, values.shape[0])
        initial = rng.uniform(0.0, 1.0, size=(12, 2))

        def distortion(centers: np.ndarray) -> float:
            distances = np.sum((values[:, None, :] - centers[None, :, :]) ** 2, axis=2)
            return float(np.sum(weights * np.min(distances, axis=1)))

        centers, labels = fast_weighted_lloyd(values, weights, 12, initial, seed=71)
        self.assertEqual(centers.shape, (12, 2))
        self.assertEqual(labels.shape, (500,))
        self.assertLessEqual(distortion(centers), distortion(initial))

    def test_joint_state_checkpoint_round_trip(self) -> None:
        rng = np.random.default_rng(37)
        raw = rng.normal(size=(200, 2))
        empirical = CovarianceTransform.fit(raw)
        matrix = np.asarray([[2.0, 0.2], [-0.1, 1.7]])
        state = JointState(
            empirical_covariance=empirical,
            covariance_log_shape=np.asarray([[0.1, 0.03], [0.03, -0.1]]),
            covariance=empirical,
            transform=AsinhMatrixTransform(matrix),
            beta=0.8,
            fit={"knotCount": 7, "jointObjective": 0.4, "passes": True},
        )
        restored = joint_state_from_payload(joint_state_payload(state))
        np.testing.assert_allclose(restored.empirical_covariance.whitening, empirical.whitening)
        np.testing.assert_allclose(restored.covariance_log_shape, state.covariance_log_shape)
        np.testing.assert_allclose(restored.transform.matrix, matrix)
        self.assertEqual(restored.beta, 0.8)
        self.assertEqual(restored.fit, state.fit)

    def test_covariance_and_matrix_transform_round_trip(self) -> None:
        rng = np.random.default_rng(7)
        source = rng.normal(size=(2000, 3)) @ np.asarray([
            [2.0, 0.3, -0.2],
            [0.0, 0.7, 0.1],
            [0.0, 0.0, 1.4],
        ])
        covariance = CovarianceTransform.fit(source)
        whitened = covariance.forward(source)
        np.testing.assert_allclose(
            np.cov(whitened, rowvar=False),
            np.eye(3),
            rtol=1e-10,
            atol=1e-10,
        )
        matrix = factorized_matrix(
            3,
            rotation=[0.2, -0.1, 0.05],
            shear=[0.1, -0.08, 0.04],
            log_scales=[0.4, 0.2, 0.3],
        )
        transform = AsinhMatrixTransform(matrix)
        unit = transform.forward(whitened)
        reconstructed = covariance.inverse(transform.inverse(unit))
        np.testing.assert_allclose(reconstructed, source, rtol=2e-12, atol=2e-12)

    def test_rectangular_grid_probabilities_are_normalized(self) -> None:
        knots = [
            np.asarray([0.0, 0.2, 0.7, 1.0]),
            np.asarray([0.0, 0.4, 1.0]),
            np.asarray([0.0, 0.15, 0.5, 0.8, 1.0]),
        ]
        weights = np.arange(1, 4 * 3 * 5 + 1, dtype=np.float64).reshape(4, 3, 5)
        weights /= np.sum(weights)
        edges = [np.linspace(0, 1, 18), np.linspace(0, 1, 13), np.linspace(0, 1, 11)]
        probabilities = rectangular_grid_bin_probabilities(edges, knots, weights)
        self.assertAlmostEqual(float(np.sum(probabilities)), 1.0, places=12)
        self.assertTrue(np.all(probabilities >= 0))
        self.assertEqual(grid_centers(knots).shape, (60, 3))

    def test_point_cloud_probabilities_are_normalized_at_boundaries(self) -> None:
        centers = np.asarray([
            [0.01, 0.5],
            [0.8, 0.99],
            [0.45, 0.25],
        ])
        widths = np.asarray([
            [0.2, 0.3],
            [0.4, 0.25],
            [0.3, 0.5],
        ])
        weights = np.asarray([0.2, 0.3, 0.5])
        edges = [np.linspace(0, 1, 33), np.linspace(0, 1, 29)]
        probabilities = point_cloud_bin_probabilities(edges, centers, widths, weights)
        self.assertAlmostEqual(float(np.sum(probabilities)), 1.0, places=12)
        self.assertTrue(np.all(probabilities >= 0))

    def test_point_cloud_sampler_matches_component_means(self) -> None:
        centers = np.asarray([[0.25, 0.4], [0.7, 0.8]])
        widths = np.asarray([[0.1, 0.2], [0.15, 0.1]])
        weights = np.asarray([0.3, 0.7])
        samples = sample_point_cloud(centers, widths, weights, 131_072, seed=19)
        np.testing.assert_allclose(
            np.mean(samples, axis=0),
            weights @ centers,
            atol=3e-4,
        )
        self.assertTrue(np.all((samples >= 0) & (samples <= 1)))

        component_samples, component_ids = sample_each_point_cloud_component(
            centers, widths, 4_096, seed=29,
        )
        for component in range(centers.shape[0]):
            np.testing.assert_allclose(
                np.mean(component_samples[component_ids == component], axis=0),
                centers[component],
                atol=3e-4,
            )

    def test_conditional_operations_measure_mean_and_median_errors(self) -> None:
        rng = np.random.default_rng(23)
        reference = rng.normal(size=(20_000, 2))
        approximation = reference.copy()
        approximation[:, 1] += 0.25
        metrics = conditional_operation_metrics(reference, approximation)
        self.assertEqual(len(metrics), 1)
        self.assertAlmostEqual(metrics[0]["conditionalMeanRmseBps"], 0.25, places=12)
        self.assertAlmostEqual(metrics[0]["conditionalMedianMaeBps"], 0.25, places=12)

    def test_density_tempering_flattens_center_allocation_without_moving_points(self) -> None:
        rng = np.random.default_rng(31)
        concentrated = np.clip(rng.normal(0.5, 0.035, size=(18_000, 2)), 0.0, 1.0)
        diffuse = rng.uniform(0.02, 0.98, size=(2_000, 2))
        values = np.vstack((concentrated, diffuse))
        weights, metadata = density_tempering_weights(values, beta=0.5)
        central = np.all(np.abs(values - 0.5) < 0.05, axis=1)
        outer = np.any(np.abs(values - 0.5) > 0.2, axis=1)
        self.assertGreater(float(np.mean(weights[outer])), float(np.mean(weights[central])))
        self.assertAlmostEqual(float(np.mean(weights)), 1.0, places=12)
        self.assertLess(float(metadata["effectiveSampleFraction"]), 1.0)

        identity, identity_metadata = density_tempering_weights(values, beta=1.0)
        np.testing.assert_array_equal(identity, np.ones(values.shape[0]))
        self.assertEqual(identity_metadata["effectiveSampleFraction"], 1.0)

    def test_redundancy_pruning_keeps_isolated_tail_center(self) -> None:
        centers = np.asarray([[0.49, 0.5], [0.5, 0.5], [0.51, 0.5], [0.95, 0.95]])
        weights = np.asarray([0.3, 0.3, 0.39, 0.01])
        retained = select_nonredundant_point_cloud_centers(centers, weights, 2)
        self.assertIn(3, retained.tolist())
        self.assertEqual(retained.size, 2)

    def test_disabled_conditional_mean_is_diagnostic_only(self) -> None:
        density = {"jensenShannonBitsPerDimension": 0.5}
        reference = {"densityThresholds": {"jensenShannonBits": 1.0}}
        baseline = [{"conditionalMeanRmseBps": 1.0, "conditionalMedianMaeBps": 1.0}]
        operations = [{
            "targetAxis": 1,
            "conditionalMeanRmseBps": 2.0,
            "conditionalMedianMaeBps": 0.25,
            "coveredConditioningMass": 1.0,
        }]
        enabled = acceptance_metrics(density, operations, reference, baseline)
        disabled = acceptance_metrics(
            density,
            operations,
            reference,
            baseline,
            frozenset({1}),
        )
        self.assertFalse(enabled["passes"])
        self.assertTrue(disabled["passes"])
        self.assertEqual(disabled["conditionalRatios"][0]["conditionalMeanRatioTo1d32"], 2.0)
        self.assertFalse(disabled["conditionalRatios"][0]["conditionalMeanEnabled"])

    def test_js_only_acceptance_keeps_all_conditionals_diagnostic(self) -> None:
        density = {"jensenShannonBitsPerDimension": 0.5}
        reference = {"densityThresholds": {"jensenShannonBits": 1.0}}
        baseline = [{"conditionalMeanRmseBps": 1.0, "conditionalMedianMaeBps": 1.0}]
        operations = [{
            "targetAxis": 1,
            "conditionalMeanRmseBps": 20.0,
            "conditionalMedianMaeBps": 30.0,
            "coveredConditioningMass": 0.5,
        }]
        result = acceptance_metrics(
            density,
            operations,
            reference,
            baseline,
            conditional_metrics_active=False,
        )
        self.assertTrue(result["passes"])
        self.assertEqual(result["acceptanceScore"], 0.5)
        self.assertEqual(result["jointObjective"], 0.5)
        self.assertFalse(result["conditionalMetricsActive"])
        self.assertFalse(result["conditionalRatios"][0]["conditionalMeanEnabled"])
        self.assertFalse(result["conditionalRatios"][0]["conditionalMedianEnabled"])
        self.assertFalse(result["conditionalRatios"][0]["coverageEnabled"])
        search_only = acceptance_metrics(
            density,
            [],
            reference,
            baseline,
            conditional_metrics_active=False,
        )
        self.assertTrue(search_only["passes"])
        self.assertEqual(search_only["jointObjective"], 0.5)
        self.assertEqual(search_only["conditionalRatios"], [])

    def test_joint_matrix_updates_are_invertible_and_factorizable(self) -> None:
        for direction in covariance_directions(3):
            self.assertAlmostEqual(float(np.trace(direction)), 0.0, places=14)
            self.assertAlmostEqual(float(np.linalg.det(expm(0.2 * direction))), 1.0, places=12)
        matrix = np.eye(3)
        for direction in transform_directions(3):
            matrix = expm(0.03 * direction) @ matrix
        factors = factorize_matrix(matrix)
        reconstructed = (
            np.asarray(factors["rotation"])
            @ np.asarray(factors["unitUpperShear"])
            @ np.diag(np.asarray(factors["positiveScales"]))
        )
        np.testing.assert_allclose(reconstructed, matrix, rtol=1e-12, atol=1e-12)
        self.assertGreater(float(np.linalg.det(matrix)), 0.0)


if __name__ == "__main__":
    unittest.main()
