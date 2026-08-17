from __future__ import annotations

import unittest

import torch

from calibrate_next_return_output import AffineStatistics


class AffineStatisticsTest(unittest.TestCase):
    def test_fits_scale_and_intercept_in_closed_form(self) -> None:
        statistics = AffineStatistics(torch.device("cpu"))
        prediction = torch.tensor([1.0, 2.0, 4.0])
        target = 2.5 * prediction - 0.75
        weights = torch.tensor([1.0, 2.0, 3.0])
        statistics.add(prediction, target, weights)

        fitted = statistics.fit()

        self.assertAlmostEqual(fitted["affine"].scale, 2.5, places=6)
        self.assertAlmostEqual(fitted["affine"].intercept, -0.75, places=6)
        expected_scale_only = float(
            (weights * prediction * target).sum()
            / (weights * prediction.square()).sum()
        )
        self.assertAlmostEqual(
            fitted["scaleOnly"].scale, expected_scale_only, places=6
        )
        self.assertEqual(fitted["scaleOnly"].intercept, 0.0)


if __name__ == "__main__":
    unittest.main()
