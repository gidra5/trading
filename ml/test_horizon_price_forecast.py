"""Contract tests for causal direct horizons and decision-preserving means."""
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from horizon_price_forecast import HorizonPriceForecast, fit_horizon_basis, horizon_split, load_horizon_split
from stochastic_price_forecast import TimelineSplit


class HorizonForecastTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(22)
        targets = np.concatenate((np.zeros(300), rng.normal(0, .01, 700))).astype(np.float32)
        self.basis = fit_horizon_basis(targets, 32)
        self.model = HorizonPriceForecast(torch.zeros(178), torch.ones(178), self.basis, (16, 16))

    def test_horizon_target_and_split_boundary(self):
        timeline = np.zeros((130, 59), dtype=np.float32)
        timeline[:, 0] = np.arange(130)
        split = horizon_split(TimelineSplit(timeline, np.arange(119, 129), 0), 3)
        np.testing.assert_array_equal(split.origins, np.arange(119, 127))
        self.assertEqual(split.targets()[0], 120 + 121 + 122)
        self.assertEqual(split.targets()[-1], 127 + 128 + 129)
        before = split.features(np.array([119]))
        timeline[120:, 1:] = 1e8
        np.testing.assert_array_equal(split.features(np.array([119])), before)

    def test_horizon_requires_complete_positive_integer_target(self):
        source = TimelineSplit(np.zeros((130, 59), dtype=np.float32), np.arange(119, 129), 0)
        for invalid in (0, -1, True, 2.5, 20):
            with self.assertRaises(ValueError):
                horizon_split(source, invalid)

    def test_actual_equal_endpoints_restore_zero_atom_after_float_roundtrip(self):
        prices = np.array([100., 101., 102., 100.])
        timeline = np.zeros((123, 59), dtype=np.float32)
        timeline[120:123, 0] = np.log(prices[1:] / prices[:-1])
        source = TimelineSplit(timeline, np.arange(119, 122), 0)
        with patch("horizon_price_forecast.load_timeline_split", return_value=source), \
                patch("export_stochastic_price_forecast.load_close_prices", return_value=prices):
            actual = load_horizon_split(Path("unused"), "train", 3)
        self.assertEqual(actual.targets().tolist(), [0.])

    def test_zero_atom_and_finite_extreme_log_density(self):
        logits = self.model(torch.zeros(3, 178))
        likelihood = self.model.log_prob_from_logits(logits, torch.tensor([0., -100., 100.]))
        self.assertAlmostEqual(float(likelihood[0].detach().exp()), .3, places=6)
        self.assertTrue(bool(torch.isfinite(likelihood).all()))

    def test_gross_mean_preservation_is_distinct_from_log_mean(self):
        features = torch.zeros(3, 178)
        gross, probabilities = self.model.endpoint_scenarios(features, 7)
        zero, masses = self.model.probabilities(features)
        expected = zero + (1 - zero) * self.model.active_gross_mean(masses)
        np.testing.assert_allclose(probabilities.sum(axis=1), 1, rtol=0, atol=1e-14)
        np.testing.assert_allclose((gross * probabilities).sum(axis=1), expected, rtol=0, atol=5e-14)
        self.assertTrue(np.all(gross > 0))
        self.assertTrue(np.all(np.diff(gross[:, 1:], axis=1) >= 0))
        self.assertTrue(np.all(expected > np.exp((np.log(gross) * probabilities).sum(axis=1))))

    def test_uniform_unit_density_is_gaussian_with_known_gross_moment(self):
        # Constant unit height gives exactly N(0,sigma^2) in log returns.
        masses = np.asarray(self.basis["basisAreas"])[None, :]
        actual = self.model.active_gross_mean(masses)[0]
        expected = np.exp(self.model.sigma ** 2 / 2)
        self.assertAlmostEqual(actual, expected, places=7)

    def test_sub_basis_point_excess_mean_keeps_gradient_and_precision(self):
        logits = self.model(torch.zeros(1, 178))
        predicted = self.model.excess_expectation_from_logits(logits)
        zero, masses = self.model.probabilities(torch.zeros(1, 178))
        expected = (1 - zero) * (masses / masses.sum(axis=1, keepdims=True) @ self.model._component_moments("excess"))
        np.testing.assert_allclose(predicted.detach().numpy(), expected, atol=1e-9, rtol=1e-6)
        predicted.square().sum().backward()
        self.assertGreater(float(self.model.output.bias.grad.abs().sum()), 0)

    def test_legacy_log_representatives_cannot_be_silently_used(self):
        with self.assertRaises(NotImplementedError):
            self.model.scenarios(torch.zeros(1, 178), 3)


if __name__ == "__main__":
    unittest.main()
