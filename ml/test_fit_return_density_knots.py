from __future__ import annotations

import unittest

import numpy as np
import torch

from fit_return_density_knots import (
    FitState,
    KnotFitter,
    StaticTransform,
    TargetHistogram,
    model_probabilities_numpy,
)


class ReturnDensityKnotsTest(unittest.TestCase):
    def test_static_transform_round_trip(self) -> None:
        transform = StaticTransform(
            alpha=4.043721227,
            location_bps=-0.00243725,
            scale_bps=2.31355475,
        )
        returns = np.asarray([-100.0, -5.0, -0.01, 0.0, 0.01, 5.0, 100.0])
        reconstructed = transform.inverse(transform.forward(returns))
        np.testing.assert_allclose(reconstructed, returns, rtol=2e-9, atol=2e-12)

    def test_piecewise_linear_probabilities_are_normalized(self) -> None:
        knots = np.asarray([0.0, 0.2, 0.55, 1.0])
        weights = np.asarray([0.1, 0.25, 0.4, 0.25])
        gaps = np.diff(knots)
        areas = np.asarray([
            gaps[0] / 2.0,
            (gaps[0] + gaps[1]) / 2.0,
            (gaps[1] + gaps[2]) / 2.0,
            gaps[2] / 2.0,
        ])
        state = FitState(
            objective="js",
            knots=knots,
            component_weights=weights,
            knot_density_heights=weights / areas,
            objective_value=0.0,
            convergence={},
        )
        edges = torch.linspace(0.0, 1.0, 1001, dtype=torch.float64)
        probabilities = model_probabilities_numpy(state, edges)
        self.assertAlmostEqual(float(np.sum(probabilities)), 1.0, places=12)
        self.assertTrue(np.all(probabilities > 0.0))

    def test_small_fit_improves_over_initialization(self) -> None:
        edges = np.linspace(0.0, 1.0, 258)
        centers = (edges[:-1] + edges[1:]) / 2.0
        density = 0.65 * np.exp(-0.5 * ((centers - 0.3) / 0.07) ** 2) \
            + 0.35 * np.exp(-0.5 * ((centers - 0.75) / 0.12) ** 2)
        probabilities = density / np.sum(density)
        counts = np.maximum(np.rint(probabilities * 1_000_000), 1).astype(np.int64)
        target = TargetHistogram(
            counts=counts,
            unit_edges=edges,
            unit_centers=centers,
            return_widths_bps=np.ones_like(centers) / centers.size,
            observations=int(np.sum(counts)),
            active_observations=int(np.sum(counts)),
            zero_observations=0,
            active_standard_deviation_bps=1.0,
        )
        fitter = KnotFitter(target, knot_count=8, device_name="cpu")
        initial_knots, initial_weights, _ = fitter.initializations[0]
        initial_gaps = torch.nn.Parameter(torch.log(torch.as_tensor(
            np.diff(initial_knots), dtype=torch.float64,
        )))
        initial_weight_logits = torch.nn.Parameter(torch.log(torch.as_tensor(
            initial_weights, dtype=torch.float64,
        )))
        initial = float(fitter._objective(
            initial_gaps,
            initial_weight_logits,
            "js",
            None,
        ).detach())
        fitted = fitter.fit("js", adam_steps=80, lbfgs_steps=20, restarts=1)
        self.assertLess(fitted.objective_value, initial)


if __name__ == "__main__":
    unittest.main()
