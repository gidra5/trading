"""Contract tests for direct Gaussian-coordinate mixed price laws."""
import math
import unittest

import torch

from knot_price_distribution import KnotPriceDistribution
from return_knot_density import triangular_basis_areas


class KnotPriceDistributionTests(unittest.TestCase):
    def law(self, *, zero=(.2,), sigma=(.1,), current=None, masses=None):
        knots = torch.tensor([0., .1, .35, .5, .8, 1.], dtype=torch.float64)
        areas = triangular_basis_areas(knots)
        batch = len(zero)
        return KnotPriceDistribution(
            torch.full((batch,), 100., dtype=torch.float64) if current is None else torch.tensor(current, dtype=torch.float64),
            torch.tensor(zero, dtype=torch.float64),
            areas.expand(batch, -1) if masses is None else masses, knots, areas,
            torch.tensor(sigma, dtype=torch.float64))

    def test_uniform_height_is_normal_with_exact_mixed_cdf(self):
        law = self.law(zero=(.2, .35), sigma=(.1, .25))
        y = torch.tensor([[-math.inf, -.2, 0., .1, math.inf], [-math.inf, -.2, 0., .1, math.inf]], dtype=torch.float64)
        expected = law.zero_probability[:, None] * (y >= 0) + (1 - law.zero_probability[:, None]) * torch.special.ndtr(y / law.log_return_sigma[:, None])
        torch.testing.assert_close(law.log_return_cdf(y), expected, atol=2e-15, rtol=0)
        torch.testing.assert_close(law.cdf(law.current_price[:, None] * y.exp(), tick_size=None), expected, atol=2e-15, rtol=0)
        self.assertEqual(law.cdf(torch.tensor([-1., 0.]), tick_size=None).tolist(), [0., 0.])

    def test_mixed_log_likelihood_and_gradients(self):
        base = self.law(zero=(.2, .3, .4), sigma=(.1, .2, .3))
        logits = torch.randn((3, 6), dtype=torch.float64, requires_grad=True)
        zero_logit = torch.tensor([-.5, -.3, -.2], dtype=torch.float64, requires_grad=True)
        sigma = torch.tensor([.1, .2, .3], dtype=torch.float64, requires_grad=True)
        law = KnotPriceDistribution(base.current_price, zero_logit.sigmoid(), logits.softmax(-1), base.knots, base.areas, sigma)
        values = torch.tensor([0., -.15, .2], dtype=torch.float64)
        loss = -law.log_prob(values).mean() + law.mean_price().mean() * .001 + law.log_return_cdf(values).mean()
        loss.backward()
        for gradient in (logits.grad, zero_logit.grad, sigma.grad):
            self.assertTrue(bool(torch.isfinite(gradient).all()))
            self.assertGreater(float(gradient.abs().sum()), 0)
        torch.testing.assert_close(law.log_return_density(values), law.log_prob(values))

    def test_tick_likelihood_has_finite_gradients_at_market_price_scale(self):
        torch.manual_seed(41)
        base = self.law(zero=(.5,) * 5, sigma=(1e-4,) * 5, current=(60000.,) * 5)
        logits = torch.randn((5, 6), dtype=torch.float64, requires_grad=True)
        zero_logit = torch.zeros(5, dtype=torch.float64, requires_grad=True)
        sigma = torch.full((5,), 1e-4, dtype=torch.float64, requires_grad=True)
        law = KnotPriceDistribution(base.current_price, zero_logit.sigmoid(), logits.softmax(-1), base.knots, base.areas, sigma)
        actual = torch.tensor([59999.97, 59999.99, 60000., 60000.01, 60000.03], dtype=torch.float64)
        probability = law.tick_probability(actual)
        self.assertTrue(bool((probability > 0).all()))
        loss = -probability.log().mean() + law.cdf(actual).square().mean()
        loss.backward()
        for gradient in (logits.grad, zero_logit.grad, sigma.grad):
            self.assertTrue(bool(torch.isfinite(gradient).all()))
            self.assertGreater(float(gradient.abs().sum()), 0)
        # Normal CDF integration provides an independent cell-mass reference.
        gaussian = self.law(zero=(0.,), sigma=(1e-4,), current=(60000.,))
        target = torch.tensor([60000.01], dtype=torch.float64)
        bounds = torch.tensor([60000.005, 60000.015], dtype=torch.float64)
        expected = torch.special.ndtr((bounds / 60000.).log() / 1e-4).diff()[0]
        self.assertAlmostEqual(float(gaussian.tick_probability(target)), float(expected), delta=2e-12)

    def test_known_density_and_zero_atom_are_distinct(self):
        law = self.law()
        y = torch.tensor([[0., -.2, .05]], dtype=torch.float64)
        normal = -.5 * (y / .1).square() - math.log(.1) - .5 * math.log(2 * math.pi)
        expected = torch.where(y == 0, math.log(.2), math.log(.8) + normal)
        torch.testing.assert_close(law.log_prob(y), expected, atol=2e-14, rtol=0)

    def test_zero_endpoint_heights_keep_nonzero_structural_tail_density(self):
        masses = torch.tensor([[0., 1., 0., 0., 0., 0.], [0., 0., 0., 0., 1., 0.]], dtype=torch.float64)
        law = self.law(zero=(0., 0.), sigma=(.1, .1), masses=masses)
        y = torch.tensor([-5., 5.], dtype=torch.float64)
        z = y / .1
        log_height = -law.areas[torch.tensor([1, 4])].log() + torch.special.log_ndtr(-z.abs()) - torch.tensor([.1, .2], dtype=torch.float64).log()
        expected = log_height - .5 * z.square() - math.log(.1) - .5 * math.log(2 * math.pi)
        self.assertTrue(bool(torch.isfinite(law.log_prob(y)).all()))
        torch.testing.assert_close(law.log_prob(y), expected, atol=1e-10, rtol=0)

    def test_zero_sigma_and_probability_one_are_exact_delta(self):
        law = self.law(zero=(.2, 1., .3), sigma=(0., .2, .1), current=(100., .004, 100.))
        values = torch.tensor([[-.1, 0., .1]] * 3, dtype=torch.float64)
        expected = torch.tensor([[-torch.inf, 0., -torch.inf]] * 2, dtype=torch.float64)
        torch.testing.assert_close(law.log_prob(values)[:2], expected)
        torch.testing.assert_close(law.log_return_cdf(values)[:2], torch.tensor([[0., 1., 1.]] * 2, dtype=torch.float64))
        samples = law.sample_prices(64)
        self.assertTrue(bool((samples[0] == 100).all()))
        self.assertTrue(bool((samples[1] == .01).all()))
        self.assertEqual(law.mean_log_return()[:2].tolist(), [0., 0.])
        torch.testing.assert_close(law.mean_price()[:2], law.current_price[:2])

    def test_whole_law_quantiles_invert_cdf_and_contain_atom(self):
        masses = torch.tensor([[.05, .1, .1, .2, .25, .3]], dtype=torch.float64)
        law = self.law(masses=masses)
        q = torch.linspace(.0001, .9999, 1001, dtype=torch.float64)[None]
        y = law.quantile_log_returns(q)
        nonzero = y != 0
        torch.testing.assert_close(law.log_return_cdf(y)[nonzero], q[nonzero], atol=2e-14, rtol=0)
        self.assertTrue(bool((y.diff() >= 0).all()))
        self.assertAlmostEqual(float((y == 0).double().mean()), .2, delta=.002)
        boundaries = law.quantile_log_returns(torch.tensor([[0., 1.]], dtype=torch.float64))
        self.assertEqual(boundaries.tolist(), [[-math.inf, math.inf]])

    def test_known_normal_quantiles_and_structural_bounded_support(self):
        law = self.law(zero=(0.,))
        q = torch.tensor([[.0001, .1, .5, .9, .9999]], dtype=torch.float64)
        torch.testing.assert_close(law.quantile_log_returns(q), .1 * torch.special.ndtri(q), atol=2e-13, rtol=0)
        masses = torch.tensor([[0., 0., 0., 0., 1., 0.]], dtype=torch.float64)
        compact = self.law(zero=(0.,), masses=masses)
        edges = compact.quantile_log_returns(torch.tensor([[0., 1.]], dtype=torch.float64))
        self.assertEqual(float(edges[0, 0]), 0.)
        self.assertEqual(float(edges[0, 1]), math.inf)
        q2 = torch.tensor([[.001, .25, .8]], dtype=torch.float64)
        torch.testing.assert_close(compact.log_return_cdf(compact.quantile_log_returns(q2)), q2, atol=1e-14, rtol=0)

    def test_price_moments_preserve_lognormal_mean_and_converge(self):
        law = self.law(zero=(.2, .35), sigma=(.1, .25))
        expected = law.current_price * (law.zero_probability + (1 - law.zero_probability) * (.5 * law.log_return_sigma.square()).exp())
        low, high = law.mean_price(96), law.mean_price(384)
        self.assertLess(float((high - expected).abs().max()), 2e-5)
        self.assertLess(float((high - expected).abs().max()), float((low - expected).abs().max()))
        torch.testing.assert_close(law.mean_log_return(384), torch.zeros(2, dtype=torch.float64), atol=1e-7, rtol=0)

    def test_tick_grid_normalizes_and_minimum_tick_collects_lower_tail(self):
        law = self.law(zero=(.15,), sigma=(.5,), current=(.025,))
        prices = torch.arange(1, 501, dtype=torch.float64)[None] * .01
        masses = law.tick_probability(prices)
        self.assertTrue(bool((masses >= 0).all()))
        self.assertAlmostEqual(float(masses.sum()), 1., places=12)
        self.assertEqual(float(law.cdf(torch.tensor([0.]))), 0.)
        expected_first = .85 * torch.special.ndtr(torch.tensor(math.log(.015 / .025) / .5, dtype=torch.float64))
        self.assertAlmostEqual(float(masses[0, 0]), float(expected_first), places=14)
        # Anchor 0.025 sits exactly at a tie and projects to 0.03.
        self.assertGreater(float(masses[0, 2]), .15)

    def test_seeded_sampling_matches_whole_law_and_keeps_prices_positive(self):
        law = self.law(zero=(.2,), sigma=(.1,))
        samples = law.sample_prices(4096, seed=21, tick_size=None)
        torch.testing.assert_close(samples, law.sample_prices(4096, seed=21, tick_size=None), rtol=0, atol=0)
        self.assertTrue(bool((samples > 0).all()))
        self.assertAlmostEqual(float((samples == 100).double().mean()), .2, delta=.001)
        probe = torch.tensor([90.], dtype=torch.float64)
        self.assertAlmostEqual(float((samples <= probe[:, None]).double().mean()), float(law.cdf(probe, tick_size=None)), delta=.001)

    def test_rejects_invalid_parameters_and_queries(self):
        for kwargs in ({'sigma': (-1.,)}, {'zero': (1.1,)}, {'current': (0.,)}, {'masses': torch.zeros((1, 6), dtype=torch.float64)}):
            with self.assertRaises(ValueError):
                self.law(**kwargs)
        law = self.law()
        for query in (torch.tensor([1., 2.]), torch.tensor([math.nan])):
            with self.assertRaises(ValueError):
                law.log_return_cdf(query)
        with self.assertRaises(ValueError):
            law.quantile_log_returns(-.1)
        with self.assertRaises(ValueError):
            law.cdf(100., tick_size=0)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA is unavailable')
    def test_cuda_agrees_with_cpu_including_gradients(self):
        cpu = self.law(zero=(.2, .3), sigma=(.1, .2))
        arguments = [getattr(cpu, name).cuda() for name in ('current_price', 'zero_probability', 'masses', 'knots', 'areas', 'log_return_sigma')]
        arguments[1].requires_grad_()
        gpu = KnotPriceDistribution(*arguments)
        y = torch.tensor([-.2, .1], dtype=torch.float64)
        torch.testing.assert_close(gpu.log_prob(y.cuda()).cpu(), cpu.log_prob(y))
        torch.testing.assert_close(gpu.mean_price().cpu(), cpu.mean_price())
        (-gpu.log_prob(y.cuda()).mean()).backward()
        self.assertTrue(bool(torch.isfinite(arguments[1].grad).all()))


if __name__ == '__main__':
    unittest.main()
