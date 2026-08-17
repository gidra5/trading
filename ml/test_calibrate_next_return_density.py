from __future__ import annotations

import unittest

import torch

from calibrate_next_return_density import (
    calibrated_log_masses,
    fit_density_temperature,
    weighted_temperature_nll,
)
from return_knot_density import triangular_basis_areas


class DensityTemperatureCalibrationTest(unittest.TestCase):
    def test_temperature_one_is_standard_mass_softmax(self) -> None:
        logits = torch.tensor([[1.0, -0.5, 0.2]])
        torch.testing.assert_close(
            calibrated_log_masses(logits, 1.0),
            torch.log_softmax(logits, dim=-1),
        )

    def test_temperature_fit_softens_consistently_wrong_logits(self) -> None:
        knots = torch.tensor([0.0, 0.5, 1.0])
        areas = triangular_basis_areas(knots)
        logits = torch.tensor([[8.0, 0.0, -2.0]]).expand(64, -1).clone()
        unit = torch.full((64,), 0.9)
        jacobian = torch.zeros(64)
        weights = torch.ones(64)
        temperature, fit = fit_density_temperature(
            logits, unit, jacobian, weights, knots, areas
        )
        self.assertGreater(temperature, 1.0)
        raw = weighted_temperature_nll(
            logits, unit, jacobian, weights, knots, areas, 1.0
        )
        calibrated = weighted_temperature_nll(
            logits, unit, jacobian, weights, knots, areas, temperature
        )
        self.assertLess(calibrated, raw)
        self.assertGreater(float(fit["nllImprovementVsRaw"]), 0)

    def test_invalid_temperature_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            calibrated_log_masses(torch.zeros(2, 3), 0.0)


if __name__ == "__main__":
    unittest.main()
