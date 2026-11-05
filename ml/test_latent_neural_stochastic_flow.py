import unittest
import math
import torch
from latent_neural_stochastic_flow import LatentNSF, LatentTransition, LatentBridge, LatentPriceMixture, horizon_features, clock_features, sample_skip_pairs
from neural_stochastic_flow import flow_loss


class LatentFlowTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2); torch.manual_seed(8)
        self.model = LatentNSF(torch.zeros(59), torch.ones(59), dimensions=4, width=8)

    def test_encoder_is_causal_and_ignores_duplicate_time(self):
        with torch.no_grad():
            self.model.posterior.weight.normal_(0, .1)
        obs = torch.randn(3, 4, 60)
        elapsed = torch.tensor([[0., 1., 100., 100.]]).expand(3, -1)
        old = self.model.encode(obs, elapsed, elapsed.double().cumsum(1)+1700000000)
        changed = obs.clone(); changed[:, 2:] += 50
        new = self.model.encode(changed, elapsed, elapsed.double().cumsum(1)+1700000000)
        torch.testing.assert_close(old[0][:, :2], new[0][:, :2], atol=0, rtol=0)
        elapsed = elapsed.clone(); elapsed[:, 2] = 0
        mu, std = self.model.encode(obs, elapsed, elapsed.double().cumsum(1)+1700000000)
        torch.testing.assert_close(mu[:, 1], mu[:, 2], atol=0, rtol=0)
        torch.testing.assert_close(std[:, 1], std[:, 2], atol=0, rtol=0)

    def test_loss_and_gradients_with_one_second_endpoint(self):
        obs = torch.randn(3, 4, 60)*.01
        elapsed = torch.tensor([[0., 1., 0., 1.]]).expand(3, -1)
        loss, parts, z = self.model.variational_loss(obs, elapsed, elapsed.double().cumsum(1)+1700000000)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in self.model.parameters()))

    def test_skip_targets_cover_sources_uniformly(self):
        elapsed = torch.tensor([[0., 1., 1., 1., 1., 1., 1., 1.]]).expand(6, -1)
        u = ((torch.arange(6)+.5)/6)[:, None].expand(-1, 8)
        batch, source, target = sample_skip_pairs(elapsed, max_skip=4, uniforms=u)
        expected = {0: [2,3,4], 1: [3,4,5], 2: [4,5,6], 3: [5,6,7], 4: [6,7], 5: [7]}
        for i, options in expected.items():
            values, counts = target[source == i].unique(return_counts=True)
            self.assertEqual(values.tolist(), options)
            self.assertEqual(counts.tolist(), [6//len(options)]*len(options))
        self.assertEqual(len(batch), 36)
        # The sampled source-average has the same expectation as enumeration.
        sampled = (10*source+target).float().reshape(6, 6).mean()
        enumerated = sum(sum(10*i+j for j in js)/len(js) for i,js in expected.items())/len(expected)
        self.assertAlmostEqual(float(sampled), enumerated, places=5)

    def test_skips_exclude_duplicate_and_adjacent_times(self):
        elapsed = torch.tensor([[0., 1., 0., 0., 0., 0., 0., 1.]])
        b, i, j = sample_skip_pairs(elapsed)
        self.assertEqual((b.tolist(), i.tolist(), j.tolist()), ([0], [0], [7]))
        b, i, j = sample_skip_pairs(torch.tensor([[0., 1.]]))
        self.assertEqual(len(b), 0)
        with self.assertRaises(ValueError):
            sample_skip_pairs(elapsed, max_skip=1)

    def test_sampled_skip_loss_on_eight_observations(self):
        obs = torch.randn(3, 8, 60)*.01
        elapsed = torch.tensor([[0., 1., 100., 101., 100., 99., 100., 100.]]).expand(3, -1)
        times = elapsed.double().cumsum(1)+1700000000
        loss, parts, z = self.model.variational_loss(obs, elapsed, times)
        self.assertEqual(z.shape, (3, 4))
        self.assertTrue(all(torch.isfinite(v) for v in parts.values()))
        loss.backward()
        self.assertTrue(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in self.model.parameters()))
        self.assertGreater(float(self.model.posterior.weight.grad.abs().sum()), 0.)

    def test_latent_transition_inverse_and_bridge(self):
        t, b = LatentTransition(4, 8).double(), LatentBridge(4, 8).double()
        z = torch.randn(8, 4, dtype=torch.float64); d = torch.full((8,), .3, dtype=torch.float64)
        f = torch.empty(8, 0, dtype=torch.float64); s = torch.zeros_like(d)
        self.assertTrue(torch.equal(t.rsample(z, f, s, s), z))
        self.assertLess(abs(float(flow_loss(t, b, z, f, d*.4, d*.6, s).detach())), 1e-10)
        with torch.no_grad():
            t.couplings.nets[0][-1].weight.normal_(0, .1)
        c, _, _ = t.parameters_at(z, d, s)
        out, jac = t.couplings.transform(z, c, d)
        back, ijac = t.couplings.transform(out, c, d, True)
        torch.testing.assert_close(z, back, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(jac, -ijac, atol=1e-12, rtol=1e-12)

    def test_time_features_preserve_seconds_and_periods(self):
        s = torch.tensor([1700000000., 1700000001.], dtype=torch.float64)/21600
        f = clock_features(s, torch.float64)
        self.assertFalse(torch.equal(f[0], f[1]))
        torch.testing.assert_close(f, clock_features(s+604800/21600, torch.float64), atol=1e-10, rtol=0)
        d = torch.tensor([1., 7200., 86400.])/21600
        torch.testing.assert_close(horizon_features(d)[:, 2], torch.log1p(d*21600))
        self.assertTrue(bool((horizon_features(d)[1:] > horizon_features(d)[:-1]).all()))

    def test_composed_flow_advances_clock(self):
        seen = []
        t, b = LatentTransition(4, 8).double(), LatentBridge(4, 8).double()
        original = t.parameters_at
        def record(state, duration, start):
            seen.append(start.clone())
            return original(state, duration, start)
        t.parameters_at = record
        s = torch.tensor([1700000000.], dtype=torch.float64)/21600
        d1, d2 = torch.tensor([.125], dtype=torch.float64), torch.tensor([.375], dtype=torch.float64)
        flow_loss(t, b, torch.zeros(1, 4, dtype=torch.float64), torch.empty(1, 0), d1, d2, s)
        self.assertTrue(any(torch.equal(v, s+d1) for v in seen))
        self.assertTrue(all(torch.equal(v, s) or torch.equal(v, s+d1) for v in seen))
        with torch.no_grad():
            t.base[-1].weight.normal_(0, .1)
        a = t.parameters_at(torch.zeros(1, 4, dtype=torch.float64), d1, s)[1]
        c = t.parameters_at(torch.zeros(1, 4, dtype=torch.float64), d1, s+.5)[1]
        self.assertFalse(torch.equal(a, c))

    def test_decoder_uncertainty_is_integrated(self):
        law = LatentPriceMixture(torch.tensor([100.]), torch.tensor([[-.1, .1]], dtype=torch.float64),
            torch.tensor([[.02, .02]], dtype=torch.float64))
        torch.testing.assert_close(law.mean_log_return(), torch.zeros(1, dtype=torch.float64))
        self.assertGreater(float(law.mean_price()), 100.)
        torch.testing.assert_close(law.cdf(torch.tensor([100.]), tick_size=None), torch.tensor([.5], dtype=torch.float64))
        density = law.log_return_density(torch.tensor([0.]))
        expected = -.5*25-math.log(.02)-.5*math.log(2*math.pi)
        self.assertAlmostEqual(float(density), expected, places=6)

    def test_rejects_low_precision_and_inconsistent_clock(self):
        f = torch.zeros(1, 2, 59); p = torch.ones(1, 2)*100
        dt = torch.tensor([[0., 1.]])
        times = dt.double().cumsum(1)+1700000000
        with self.assertRaisesRegex(ValueError, 'float64'):
            self.model.distribution(f, p, dt, 1, times.float(), components=2)
        with self.assertRaisesRegex(ValueError, 'disagree'):
            self.model.distribution(f, p, dt, 1, times+torch.tensor([[0., 2.]]), components=2)

    def test_arbitrary_horizon_and_zero_horizon(self):
        f = torch.randn(2, 2, 59); p = torch.tensor([[100., 100.01], [101., 100.99]], dtype=torch.float64)
        dt = torch.tensor([[0., 1.], [0., 1.]])
        law = self.model.distribution(f, p, dt, 1234.5, dt.double().cumsum(1)+1700000000, components=32)
        self.assertEqual(law.log_means.shape, (2, 32))
        self.assertTrue(torch.isfinite(law.mean_price()).all())
        self.assertTrue((law.log_stds > 0).all())
        current = self.model.distribution(f, p, dt, 0, dt.double().cumsum(1)+1700000000, components=32)
        torch.testing.assert_close(current.sample_prices(16), p[:, -1, None].expand(-1, 16), atol=1e-12, rtol=0)
        with self.assertRaises(ValueError):
            self.model.distribution(f, p, dt, 90000, dt.double().cumsum(1)+1700000000)


if __name__ == '__main__':
    unittest.main()
