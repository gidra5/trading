import unittest
import torch
from neural_stochastic_flow import NeuralStochasticFlow, Bridge, flow_loss


class FlowTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2); torch.manual_seed(21)
        self.model = NeuralStochasticFlow(torch.zeros(3), torch.ones(3), width=8).double()
        self.bridge = Bridge(torch.zeros(3), torch.ones(3), width=8).double()
        self.x, self.f = torch.randn(12, 2, dtype=torch.float64), torch.randn(12, 3, dtype=torch.float64)
        self.d, self.s = torch.ones(12, dtype=torch.float64)*.4, torch.zeros(12, dtype=torch.float64)

    def test_zero_duration_identity(self):
        self.assertTrue(torch.equal(self.model.rsample(self.x, self.f, self.s, self.s), self.x))
        with self.assertRaises(ValueError):
            self.model.log_prob(self.x, self.x, self.f, self.s, self.s)

    def test_coupling_inverse_and_jacobian(self):
        with torch.no_grad():
            for net in self.model.couplings.nets:
                net[-1].weight.normal_(0, .05)
        c = self.model.condition(self.x, self.f, self.d, self.s)
        y, jac = self.model.couplings.transform(self.x, c, self.d)
        z, invjac = self.model.couplings.transform(y, c, self.d, True)
        torch.testing.assert_close(z, self.x, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(jac+invjac, torch.zeros_like(jac), atol=1e-12, rtol=0)
        fn = lambda v: self.model.couplings.transform(v[None], c[:1], self.d[:1])[0][0]
        numerical = torch.linalg.slogdet(torch.autograd.functional.jacobian(fn, self.x[0]))[1]
        torch.testing.assert_close(numerical, jac[0], atol=1e-12, rtol=1e-12)

    def test_density_includes_base_scale(self):
        noise = torch.randn_like(self.x)
        y = self.model.rsample(self.x, self.f, self.d, self.s, noise)
        _, mu, std = self.model.parameters_at(self.x, self.f, self.d, self.s)
        offset = torch.stack((self.x[:, 0], torch.zeros_like(self.x[:, 0])), -1)
        expected = torch.distributions.Normal(mu+offset, std).log_prob(y).sum(-1)
        torch.testing.assert_close(self.model.log_prob(y, self.x, self.f, self.d, self.s), expected)

    def test_brownian_bridge_bounds_vanish(self):
        loss = flow_loss(self.model, self.bridge, self.x, self.f, self.d*.3, self.d*.7, self.s)
        self.assertLess(abs(float(loss.detach())), 1e-10)

    def test_price_translation_and_time_stationarity(self):
        eps = torch.randn_like(self.x)
        shift = torch.tensor([3., 0.], dtype=torch.float64)
        a = self.model.rsample(self.x, self.f, self.d, self.s, eps)
        b = self.model.rsample(self.x+shift, self.f, self.d, self.s+10, eps)
        torch.testing.assert_close(a+shift, b)

    def test_nontrivial_flow_gradients_are_finite(self):
        with torch.no_grad():
            self.model.base[-1].weight.normal_(0, .02)
        loss = flow_loss(self.model, self.bridge, self.x, self.f, self.d*.4, self.d*.6, self.s)
        loss.backward()
        for model in (self.model, self.bridge):
            grads = [p.grad for p in model.parameters() if p.grad is not None]
            self.assertTrue(all(bool(torch.isfinite(g).all()) for g in grads))
            self.assertGreater(sum(float(g.abs().sum()) for g in grads), 0)

    def test_arbitrary_horizon_distribution_and_price_grid(self):
        marginal = self.model.distribution(self.f, self.x[:, 1], torch.full_like(self.d, 100.), 1234.5)
        price = marginal.sample_prices(32)
        self.assertEqual(price.shape, (12, 32))
        torch.testing.assert_close(price*100, (price*100).round(), atol=1e-9, rtol=0)
        self.assertTrue(bool(torch.all((marginal.cdf(self.d*250) >= 0) & (marginal.cdf(self.d*250) <= 1))))
        zero = self.model.distribution(self.f, self.x[:, 1], torch.full_like(self.d, 100.), 0).sample_prices(8)
        self.assertTrue(torch.equal(zero, torch.full_like(zero, 100.)))
        with self.assertRaises(ValueError):
            self.model.distribution(self.f, self.x[:, 1], torch.full_like(self.d, 100.), 86401)


if __name__ == '__main__':
    unittest.main()
