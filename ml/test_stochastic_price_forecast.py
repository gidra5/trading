from __future__ import annotations

import unittest
import numpy as np
import torch

from stochastic_price_forecast import (
    StochasticPriceForecast, TimelineSplit, append_hypothetical_return, fit_training_basis,
)


class ForecastTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(22)
        self.targets = np.concatenate((np.zeros(300), rng.normal(0, 1e-4, 700))).astype(np.float32)
        self.basis = fit_training_basis(self.targets, 32)
        self.model = StochasticPriceForecast(torch.zeros(178), torch.ones(178), self.basis, (16, 16))

    def test_mixed_law_has_exact_zero_atom(self):
        logits = self.model(torch.zeros(1, 178))
        logp = self.model.log_prob_from_logits(logits, torch.zeros(1))
        self.assertAlmostEqual(float(logp.detach().exp()), 0.3, places=6)

    def test_conditional_scenarios_preserve_probability_and_mean(self):
        features = torch.zeros(4, 178)
        returns, probabilities = self.model.scenarios(features, 7)
        np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-12)
        expected = self.model.expectation_from_logits(self.model(features)).detach().numpy()
        np.testing.assert_allclose((returns * probabilities).sum(axis=1), expected, rtol=1e-5, atol=1e-10)
        self.assertTrue(np.all(np.diff(returns[:, 1:], axis=1) >= 0))
        self.assertTrue(np.isfinite(returns).all())

    def test_rollout_does_not_read_future_nonprice_features(self):
        features = torch.arange(356).reshape(2, 178).float()
        original = features.clone()
        updated = append_hypothetical_return(features, torch.tensor([0.01, -0.02]))
        torch.testing.assert_close(features, original)
        torch.testing.assert_close(updated[:, :119], original[:, 1:120])
        torch.testing.assert_close(updated[:, 120:], original[:, 120:])
        torch.testing.assert_close(updated[:, 119], torch.tensor([0.01, -0.02]))

    def test_timeline_features_use_only_current_and_past_rows(self):
        timeline = np.arange(200 * 59, dtype=np.float32).reshape(200, 59)
        split = TimelineSplit(timeline, np.array([119, 120]), 0)
        before = split.features()
        timeline[121:] = -1
        np.testing.assert_array_equal(split.features(), before)
        self.assertEqual(before.shape, (2, 178))
        self.assertEqual(before[0, 119], timeline[119, 0])

    def test_train_basis_is_finite_and_full_support(self):
        self.assertEqual(self.basis["fitSplit"], "train")
        self.assertEqual(self.basis["knots"][0], 0)
        self.assertEqual(self.basis["knots"][-1], 1)
        self.assertTrue(np.all(np.diff(self.basis["knots"]) > 0))
        self.assertAlmostEqual(sum(self.basis["priorMasses"]), 1)


if __name__ == "__main__":
    unittest.main()
