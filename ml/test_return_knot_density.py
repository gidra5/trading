from __future__ import annotations

import math
import unittest

import numpy as np
import torch

from return_knot_density import (
    ReturnTransform,
    component_log_masses,
    component_return_means,
    interpolated_log_density_unit,
    triangular_basis_areas,
)


class ReturnKnotDensityTest(unittest.TestCase):
    def test_area_biased_softmax_produces_constant_density(self) -> None:
        knots = torch.tensor([0.0, 0.2, 0.7, 1.0])
        areas = triangular_basis_areas(knots)
        logits = torch.zeros((5, knots.numel()))
        log_masses = component_log_masses(logits, areas)
        targets = torch.tensor([0.0, 0.1, 0.4, 0.85, 1.0])
        log_density = interpolated_log_density_unit(
            log_masses, targets, knots, areas
        )
        torch.testing.assert_close(log_density, torch.zeros_like(log_density))
        torch.testing.assert_close(log_masses.exp().sum(dim=1), torch.ones(5))

    def test_mass_form_equals_single_area_normalized_height_formula(self) -> None:
        knots = torch.tensor([0.0, 0.1, 0.45, 1.0])
        areas = triangular_basis_areas(knots)
        logits = torch.tensor([[0.7, -0.2, 1.1, 0.3]])
        log_masses = component_log_masses(logits, areas)
        heights_from_masses = log_masses.exp() / areas
        direct_heights = logits.exp() / torch.sum(areas * logits.exp(), dim=1)
        torch.testing.assert_close(heights_from_masses, direct_heights)

    def test_interpolated_density_integrates_to_one(self) -> None:
        knots = torch.tensor([0.0, 0.15, 0.6, 0.8, 1.0])
        areas = triangular_basis_areas(knots)
        logits = torch.tensor([[1.2, -0.4, 0.1, 2.0, -1.0]])
        log_masses = component_log_masses(logits, areas)
        grid = torch.linspace(0, 1, 100_001)
        tiled = log_masses.expand(grid.numel(), -1)
        density = interpolated_log_density_unit(
            tiled, grid, knots, areas
        ).exp()
        integral = torch.trapezoid(density, grid)
        self.assertAlmostEqual(float(integral), 1.0, places=5)

    def test_component_means_reconstruct_symmetric_mean(self) -> None:
        transform = ReturnTransform(
            alpha=4.0, location_bps=0.0, scale_bps=2.3
        )
        knots = np.linspace(0, 1, 17, dtype=np.float64)
        means = component_return_means(knots, transform, quadrature_order=96)
        areas = triangular_basis_areas(torch.from_numpy(knots)).numpy()
        expectation = float(np.sum(areas * means))
        self.assertTrue(math.isfinite(expectation))
        self.assertAlmostEqual(expectation, 0.0, places=10)

    def test_interpolation_accepts_batch_and_path_dimensions(self) -> None:
        knots = torch.tensor([0.0, 0.2, 0.7, 1.0])
        areas = triangular_basis_areas(knots)
        logits = torch.tensor([
            [[0.2, -0.1, 0.7, 0.3], [0.1, 0.5, -0.2, 0.4]],
            [[-0.3, 0.2, 0.6, 0.1], [0.9, -0.4, 0.3, 0.2]],
        ])
        targets = torch.tensor([[0.1, 0.4], [0.8, 0.95]])
        log_masses = component_log_masses(logits, areas)
        shaped = interpolated_log_density_unit(
            log_masses, targets, knots, areas
        )
        flattened = interpolated_log_density_unit(
            log_masses.reshape(-1, knots.numel()),
            targets.reshape(-1),
            knots,
            areas,
        ).reshape_as(targets)
        torch.testing.assert_close(shaped, flattened)


if __name__ == "__main__":
    unittest.main()
