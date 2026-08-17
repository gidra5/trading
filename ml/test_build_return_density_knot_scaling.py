from __future__ import annotations

import unittest

import numpy as np

from build_return_density_knot_scaling import (
    basis_areas,
    interval_masses,
    refined_knots,
)


class ReturnDensityKnotScalingTest(unittest.TestCase):
    def test_refinement_preserves_source_breakpoints_and_density(self) -> None:
        source_knots = np.asarray([0.0, 0.2, 0.7, 1.0])
        source_heights = np.asarray([0.5, 1.5, 0.8, 0.4])
        source_heights /= interval_masses(
            source_knots, source_heights
        ).sum()
        knots = refined_knots(13, source_knots, source_heights)
        for source_knot in source_knots:
            self.assertTrue(np.any(np.isclose(knots, source_knot)))
        heights = np.interp(knots, source_knots, source_heights)
        self.assertAlmostEqual(float(np.sum(basis_areas(knots) * heights)), 1)

    def test_compression_uses_ordered_global_quantiles(self) -> None:
        source_knots = np.asarray([0.0, 0.1, 0.4, 0.8, 1.0])
        source_heights = np.ones(5)
        knots = refined_knots(3, source_knots, source_heights)
        np.testing.assert_allclose(knots, [0, 0.5, 1], atol=1e-12)


if __name__ == "__main__":
    unittest.main()
