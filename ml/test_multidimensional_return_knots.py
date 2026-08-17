from __future__ import annotations

import unittest

import numpy as np

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
)


class MultidimensionalReturnKnotsTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
