"""Independent likelihood, numerical and gradient checks for hurdle experts."""
from __future__ import annotations

import io
import unittest

import mpmath as mp
import numpy as np
import torch
from torch.distributions import Beta, Gamma

from hurdle_sign_size_forecast import (
    MAXIMUM_SHAPE, MINIMUM_MEAN, MINIMUM_SHAPE, HurdleSignSize,
    hurdle_components, hurdle_log_probabilities, hurdle_loss_parts, hurdle_mean,
    hurdle_nll_elements, log_gamma_increment,
)
from neural_sign_size_forecast import StructuredSpecialist
from normalized_glu_next_return import optimizer_parameter_groups


class HurdleSignSizeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def model(self, horizons=(1, 900)):
        torch.manual_seed(932)
        mean, std = torch.linspace(-.1, .1, 59), torch.linspace(.8, 1.2, 59)
        return HurdleSignSize(StructuredSpecialist("sign", mean, std, horizons),
            StructuredSpecialist("sign", mean, std, horizons),
            StructuredSpecialist("size", mean, std, horizons),
            torch.tensor([[.002, .003]]*len(horizons), dtype=torch.float64),
            torch.tensor([[.7, 2.3]]*len(horizons), dtype=torch.float64))

    def parameters(self):
        return torch.tensor([[[.2, -.7, .002, .003, .7, 2.3], [-.3, .4, .004, .006, 2., .3]],
                             [[-.4, .8, .003, .005, .2, 4.], [.7, -.3, .001, .002, 3., .8]],
                             [[.6, -.8, .006, .007, 1., .5], [-.9, .2, .002, .001, .4, 2.]]],
                            dtype=torch.float64, requires_grad=True)

    def test_nll_matches_torch_gamma_beta_and_exact_class_factorization(self):
        packed = self.parameters()
        y = torch.tensor([[.001, -.005], [-.003, 0.], [0., .003]], dtype=torch.float64)
        parts = hurdle_loss_parts(packed, y)
        labels = (y > 0).long()+(y >= 0).long()
        class_nll = -hurdle_log_probabilities(packed).gather(-1, labels[..., None])[..., 0]
        torch.testing.assert_close(parts["zero"]+parts["sign"], class_nll, rtol=1e-14, atol=1e-14)
        positive, negative = y > 0, y < 0
        torch_up = -Gamma(packed[..., 4][positive],
                          packed[..., 4][positive]/packed[..., 2][positive]).log_prob(y[positive])
        torch_down = -Beta(packed[..., 5][negative], packed[..., 5][negative]
                          *(1-packed[..., 3][negative])/packed[..., 3][negative]).log_prob(-y[negative])
        torch.testing.assert_close(parts["up"][positive], torch_up, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(parts["down"][negative], torch_down, rtol=1e-11, atol=1e-11)
        expected = class_nll.clone()
        expected[positive] += torch_up
        expected[negative] += torch_down
        torch.testing.assert_close(hurdle_nll_elements(packed, y), expected, rtol=1e-11, atol=1e-11)

    def test_inactive_branches_are_safe_and_have_exactly_zero_gradient(self):
        packed = self.parameters()
        y = torch.tensor([[0., .002], [-.004, -0.], [.001, -.003]], dtype=torch.float64)
        parts = hurdle_loss_parts(packed, y)
        self.assertTrue(all(bool(torch.isfinite(v).all()) for v in parts.values()))
        hurdle_nll_elements(packed, y).sum().backward()
        self.assertTrue(bool(torch.isfinite(packed.grad).all()))
        torch.testing.assert_close(packed.grad[..., 1][y == 0], torch.zeros_like(y[y == 0]), rtol=0, atol=0)
        for column, mask in ((2, y <= 0), (4, y <= 0), (3, y >= 0), (5, y >= 0)):
            torch.testing.assert_close(packed.grad[..., column][mask], torch.zeros_like(y[mask]), rtol=0, atol=0)
        for part, columns in (("zero", (0,)), ("sign", (1,)), ("up", (2, 4)), ("down", (3, 5))):
            probe = self.parameters()
            derivative = torch.autograd.grad(hurdle_loss_parts(probe, y)[part].sum(), probe)[0]
            other = [i for i in range(6) if i not in columns]
            torch.testing.assert_close(derivative[..., other], torch.zeros_like(derivative[..., other]), rtol=0, atol=0)

    def test_probability_law_means_and_extreme_logits(self):
        packed = self.parameters().detach()
        p, up, down = hurdle_components(packed)
        torch.testing.assert_close(p.sum(-1), torch.ones_like(up), rtol=1e-14, atol=1e-14)
        torch.testing.assert_close(Gamma(packed[..., 4], packed[..., 4]/up).mean, up)
        torch.testing.assert_close(Beta(packed[..., 5], packed[..., 5]*(1-down)/down).mean, down)
        numeric = np.sum(p.numpy()*np.stack((-down.numpy(), np.zeros_like(up.numpy()), up.numpy()), -1), -1)
        np.testing.assert_allclose(hurdle_mean(packed).numpy(), numeric, rtol=1e-14, atol=1e-14)
        packed[0, :, 0] = torch.tensor([1000., -1000.])
        packed[0, :, 1] = torch.tensor([-1000., 1000.])
        self.assertTrue(bool(torch.isfinite(hurdle_log_probabilities(packed)).all()))
        torch.testing.assert_close(hurdle_components(packed)[0][0],
                                   torch.tensor([[0., 1., 0.], [0., 0., 1.]], dtype=torch.float64), rtol=0, atol=0)

    def test_gamma_ratio_and_gradients_match_high_precision_across_branch_and_bounds(self):
        with mp.workdps(80):
            for base in (1e-14, .2, 1., 1e5, 1e6, 1e6+1, 1e9, 1e14):
                for increment in (.0011, .1, 1., 3.5, 99.9):
                    with self.subTest(base=base, increment=increment):
                        b = torch.tensor(base, dtype=torch.float64, requires_grad=True)
                        a = torch.tensor(increment, dtype=torch.float64, requires_grad=True)
                        actual = log_gamma_increment(b, a)
                        mb, ma = mp.mpf(base), mp.mpf(increment)
                        expected = float(mp.loggamma(mb+ma)-mp.loggamma(mb))
                        tolerance = 2e-8 if base <= 1e6 else 5e-12
                        self.assertAlmostEqual(float(actual.detach()), expected, delta=tolerance)
                        db, da = torch.autograd.grad(actual, (b, a))
                        expected_db = float(mp.digamma(mb+ma)-mp.digamma(mb))
                        expected_da = float(mp.digamma(mb+ma))
                        self.assertAlmostEqual(float(db), expected_db, delta=max(2e-14, abs(expected_db)*3e-8))
                        self.assertAlmostEqual(float(da), expected_da, delta=max(2e-14, abs(expected_da)*1e-12))

    def test_likelihood_and_gradients_are_finite_at_declared_mean_shape_bounds(self):
        for mean in (MINIMUM_MEAN, 1e-7, .5, 1-1e-12):
            for shape in (MINIMUM_SHAPE+1e-8, MAXIMUM_SHAPE-1e-8):
                packed = torch.tensor([[[.2, -.3, mean, mean, shape, shape]]]*3,
                                      dtype=torch.float64, requires_grad=True)
                y = torch.tensor([[0.], [1e-7], [-(1-1e-12)]], dtype=torch.float64)
                values = hurdle_nll_elements(packed, y)
                self.assertTrue(bool(torch.isfinite(values).all()))
                values.sum().backward()
                self.assertTrue(bool(torch.isfinite(packed.grad).all()))

    def test_joint_nll_reaches_all_three_backbones_and_shape_parameters(self):
        model = self.model()
        inputs = torch.randn(5, 2, 59, requires_grad=True)
        y = torch.tensor([[0., .001], [.003, -.002], [-.005, 0.], [.007, -.004], [-.001, .002]], dtype=torch.float64)
        output = model(inputs)
        self.assertEqual(output.shape, (5, 2, 6))
        self.assertEqual(output.dtype, torch.float64)
        hurdle_nll_elements(output, y).mean().backward()
        self.assertTrue(bool(torch.isfinite(inputs.grad).all()))
        for expert in (model.zero, model.sign, model.size):
            blocks = [getattr(expert.backbone, f"layer{i}") for i in range(1, 12)]
            blocks += [expert.backbone.initial_market, expert.backbone.initial_prefix,
                       *expert.backbone.layer4_source_branches]
            for block in blocks:
                for parameter in block.muon_parameters():
                    self.assertIsNotNone(parameter.grad)
                    self.assertTrue(bool(torch.isfinite(parameter.grad).all()))
                    self.assertTrue(bool((parameter.grad != 0).any()), type(block).__name__)
            self.assertTrue(bool((expert.readout.weight.grad != 0).any()))
        self.assertTrue(bool((model.raw_shapes.grad != 0).all()))

    def test_mean_floor_cap_optimizer_routing_and_checkpoint_reconstruction(self):
        model = self.model().eval()
        with torch.no_grad():
            model.size.readout.weight.zero_()
            model.size.readout.bias.copy_(torch.tensor([-1000., 1000., 1000., -1000.]))
        inputs = torch.randn(3, 2, 59)
        with torch.no_grad(): expected = model(inputs)
        self.assertEqual(float(expected[0, 0, 2]), MINIMUM_MEAN)
        self.assertEqual(float(expected[0, 0, 3]), model.downside_cap)
        self.assertTrue(bool((1-expected[..., 3] > 0).all()))
        muon, adam = optimizer_parameter_groups(model)
        routed = [id(p) for p in (*muon, *adam)]
        self.assertEqual(len(routed), len(set(routed)))
        self.assertEqual(set(routed), {id(p) for p in model.parameters() if p.requires_grad})
        self.assertIn(id(model.raw_shapes), {id(p) for p in adam})
        self.assertNotIn(id(model.scales), set(routed))
        saved = io.BytesIO()
        torch.save(model.state_dict(), saved)
        saved.seek(0)
        restored = self.model().eval()
        restored.load_state_dict(torch.load(saved, weights_only=True))
        with torch.no_grad(): torch.testing.assert_close(restored(inputs), expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
