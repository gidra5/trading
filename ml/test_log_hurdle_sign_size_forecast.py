"""Independent likelihood, analytic price-moment and shared-gradient checks."""
from __future__ import annotations

import io
import unittest

import numpy as np
from scipy.integrate import quad
from scipy.stats import gamma as scipy_gamma
import torch
from torch.distributions import Gamma
from torch.nn import functional as F

from log_hurdle_sign_size_forecast import (MAXIMUM_SHAPE, MAXIMUM_UP_MEAN_RATIO,
    MINIMUM_MEAN, MINIMUM_SHAPE, LogHurdleSignSize, log_hurdle_components,
    log_hurdle_conditional_log_gross_means, log_hurdle_gross_components,
    log_hurdle_log_probabilities, log_hurdle_loss_parts, log_hurdle_mean,
    log_hurdle_nll_elements, log_hurdle_simple_components, log_hurdle_simple_mean)
from neural_sign_size_forecast import StructuredSpecialist
from normalized_glu_next_return import optimizer_parameter_groups


class LogHurdleSignSizeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def packed(self):
        return torch.tensor([[[.2, -.5, .03, .08, .7, 1.1], [-.4, .7, .2, 1.5, 2.2, .8]]]*3,
                            dtype=torch.float64, requires_grad=True)

    def model(self):
        torch.manual_seed(556)
        horizons = (1, 900, 1800, 7200)
        mean, std = torch.zeros(59), torch.ones(59)
        experts = [StructuredSpecialist(task, mean, std, horizons) for task in ("sign", "sign", "size")]
        return LogHurdleSignSize(*experts, torch.tensor([[.0001, .0002], [.002, .003],
            [.004, .005], [.02, .03]], dtype=torch.float64), torch.tensor([[.8, 1.2]]*4, dtype=torch.float64))

    def test_signed_gamma_likelihood_matches_torch_with_large_negative_log_returns(self):
        parameters = self.packed()
        target = torch.tensor([[0., -2.4], [.004, 0.], [-.002, .3]], dtype=torch.float64)
        logp = log_hurdle_log_probabilities(parameters)
        expected = torch.empty_like(target)
        for i in range(3):
            for j in range(2):
                x = float(target[i, j])
                if x == 0:
                    expected[i, j] = -logp[i, j, 1]
                else:
                    k = 2 if x > 0 else 3
                    alpha = parameters[i, j, k+2]
                    law = Gamma(alpha, alpha/parameters[i, j, k])
                    expected[i, j] = -logp[i, j, 2 if x > 0 else 0]-law.log_prob(target[i, j].abs())
        actual = log_hurdle_nll_elements(parameters, target)
        torch.testing.assert_close(actual, expected, rtol=2e-14, atol=2e-14)
        actual.mean().backward()
        self.assertTrue(bool(torch.isfinite(parameters.grad).all()))
        self.assertTrue(bool((parameters.grad[1, 1, 1:] == 0).all()))

    def test_masks_and_component_gradients_do_not_mix_expert_targets(self):
        target = torch.tensor([[0., -.2], [.004, 0.], [-.002, .3]], dtype=torch.float64)
        columns = {"zero": [0], "sign": [1], "up": [2, 4], "down": [3, 5]}
        for part, allowed in columns.items():
            parameters = self.packed()
            values = log_hurdle_loss_parts(parameters, target)[part]
            gradient = torch.autograd.grad(values.sum(), parameters)[0]
            other = [j for j in range(6) if j not in allowed]
            torch.testing.assert_close(gradient[..., other], torch.zeros_like(gradient[..., other]), rtol=0, atol=0)
            if part in ("sign", "up", "down"):
                active = target != 0 if part == "sign" else target > 0 if part == "up" else target < 0
                torch.testing.assert_close(values[~active], torch.zeros_like(values[~active]), rtol=0, atol=0)
                torch.testing.assert_close(gradient[~active], torch.zeros_like(gradient[~active]), rtol=0, atol=0)

    def test_analytic_log_and_price_means_match_independent_gamma_integrals(self):
        parameters = self.packed().detach()[:1]
        probability, mean_up, mean_down = log_hurdle_components(parameters)
        p, simple_up, simple_down = log_hurdle_simple_components(parameters)
        _, gross_up, gross_down = log_hurdle_gross_components(parameters)
        torch.testing.assert_close(p, probability, rtol=0, atol=0)
        torch.testing.assert_close(probability.sum(-1), torch.ones_like(mean_up), rtol=1e-14, atol=1e-14)
        for j in range(2):
            for k in (0, 1):
                alpha = float(parameters[0, j, 4+k])
                scale = float(parameters[0, j, 2+k])/alpha
                if k == 0:
                    integrand = lambda z: np.exp(scipy_gamma.logpdf(z, alpha)+scale*z)*(-np.expm1(-scale*z))
                    observed = float(simple_up[0, j])
                else:
                    integrand = lambda z: np.exp(scipy_gamma.logpdf(z, alpha))*(-np.expm1(-scale*z))
                    observed = float(simple_down[0, j])
                numerical, error = quad(integrand, 0, np.inf, epsabs=1e-12, epsrel=1e-12, limit=250)
                self.assertLess(error, 2e-11)
                self.assertAlmostEqual(observed, numerical, delta=2e-11)
        torch.testing.assert_close(gross_up-1, simple_up, rtol=2e-14, atol=2e-14)
        torch.testing.assert_close(1-gross_down, simple_down, rtol=2e-14, atol=2e-14)
        expected_log = probability[..., 2]*mean_up-probability[..., 0]*mean_down
        torch.testing.assert_close(log_hurdle_mean(parameters), expected_log, rtol=0, atol=0)
        expected_simple = probability[..., 2]*simple_up-probability[..., 0]*simple_down
        torch.testing.assert_close(log_hurdle_simple_mean(parameters), expected_simple, rtol=0, atol=0)
        self.assertGreater(float(torch.max(torch.abs(expected_simple-torch.expm1(expected_log)))), .02)

    def test_first_and_second_price_moments_stay_finite_at_up_scale_boundary(self):
        for alpha in (MINIMUM_SHAPE, 1., MAXIMUM_SHAPE):
            parameters = torch.tensor([[[0., 0., alpha*MAXIMUM_UP_MEAN_RATIO, 3., alpha, alpha]]], dtype=torch.float64)
            for power in (1., 2.):
                up, down = log_hurdle_conditional_log_gross_means(parameters, power)
                expected_up = -alpha*np.log1p(-power*MAXIMUM_UP_MEAN_RATIO)
                expected_down = -alpha*np.log1p(power*3/alpha)
                self.assertAlmostEqual(float(up), expected_up, delta=1e-11)
                self.assertAlmostEqual(float(down), expected_down, delta=1e-11)
                self.assertTrue(bool(torch.isfinite(up.exp()).all()))
                self.assertTrue(bool(torch.isfinite(down.exp()).all()))
                self.assertGreater(float(down.exp()), 0)
        for invalid in (0., -1., 3., float("inf"), float("nan")):
            with self.assertRaises(ValueError):
                log_hurdle_conditional_log_gross_means(parameters, invalid)

    def test_smooth_up_scale_bound_preserves_tiny_means_and_keeps_down_log_support(self):
        model = self.model().eval()
        inputs = torch.randn(2, 2, 59)
        with torch.no_grad():
            model.size.readout.weight.zero_()
            model.size.readout.bias.zero_()
            output = model(inputs)
            raw = model.size(inputs).double()*model.scales
        relative = float(torch.max(torch.abs(output[:, 0, 2]/raw[:, 0, 0]-1)))
        self.assertLess(relative, 3e-8)
        self.assertTrue(bool((output[..., 2]/output[..., 4] <= MAXIMUM_UP_MEAN_RATIO).all()))
        torch.testing.assert_close(output[..., 3], raw[..., 1], rtol=0, atol=0)
        with torch.no_grad():
            model.size.readout.bias.fill_(1e6)
            saturated = model(inputs)
        self.assertTrue(bool((saturated[..., 2]/saturated[..., 4] <= MAXIMUM_UP_MEAN_RATIO+1e-16).all()))
        self.assertTrue(bool((saturated[..., 3] > 1).all()))
        self.assertTrue(bool(torch.isfinite(log_hurdle_simple_mean(saturated)).all()))

    def test_shared_four_horizon_gradients_optimizer_partition_and_checkpoint_roundtrip(self):
        model = self.model()
        x = torch.randn(4, 2, 59)
        target = torch.tensor([[0., 0., 0., 0.], [.0001, .002, .004, .02],
            [-.0002, -.003, -.005, -.03], [.0003, -.004, .006, -.04]], dtype=torch.float64)
        parameters = model(x)
        self.assertEqual(parameters.shape, (4, 4, 6))
        self.assertEqual(parameters.dtype, torch.float64)
        log_hurdle_nll_elements(parameters, target).mean().backward()
        self.assertEqual(sum(isinstance(module, StructuredSpecialist) for module in model.modules()), 3)
        for expert in (model.zero, model.sign, model.size):
            self.assertTrue(bool((expert.readout.weight.grad != 0).any(dim=1).all()))
            self.assertTrue(any(p.grad is not None and bool((p.grad != 0).any()) for p in expert.backbone.parameters()))
        self.assertTrue(bool((model.raw_shapes.grad != 0).all()))
        self.assertTrue(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()))
        muon, adam = optimizer_parameter_groups(model)
        identifiers = [id(p) for p in (*muon, *adam)]
        self.assertEqual(len(identifiers), len(set(identifiers)))
        self.assertEqual(set(identifiers), {id(p) for p in model.parameters() if p.requires_grad})
        self.assertIn(id(model.raw_shapes), {id(p) for p in adam})
        serialized = io.BytesIO()
        torch.save(model.state_dict(), serialized)
        serialized.seek(0)
        restored = self.model().eval()
        restored.load_state_dict(torch.load(serialized, weights_only=True))
        model.eval()
        with torch.no_grad():
            torch.testing.assert_close(restored(x), model(x), rtol=0, atol=0)

    def test_minimum_means_extreme_probabilities_and_inactive_densities_remain_finite(self):
        parameters = torch.tensor([[[1000., -1000., MINIMUM_MEAN, MINIMUM_MEAN, MINIMUM_SHAPE, MAXIMUM_SHAPE]],
                                   [[-1000., 1000., MINIMUM_MEAN, MINIMUM_MEAN, MAXIMUM_SHAPE, MINIMUM_SHAPE]],
                                   [[0., 0., MINIMUM_MEAN, MINIMUM_MEAN, 1., 1.]]],
                                  dtype=torch.float64, requires_grad=True)
        target = torch.tensor([[0.], [1e-9], [-2.]], dtype=torch.float64)
        values = log_hurdle_nll_elements(parameters, target)
        self.assertTrue(bool(torch.isfinite(values).all()))
        values.sum().backward()
        self.assertTrue(bool(torch.isfinite(parameters.grad).all()))
        probability, _, _ = log_hurdle_components(parameters)
        torch.testing.assert_close(probability[:2, 0], torch.tensor([[0., 1., 0.], [0., 0., 1.]], dtype=torch.float64), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
