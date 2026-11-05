"""Behavioral contracts for direct price laws and retained NSF dynamics."""
from contextlib import ExitStack
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from direct_latent_nsf import CONTRACT, DirectLatentNSF, load_direct_latent_nsf
from horizon_price_forecast import fit_horizon_basis
from latent_neural_stochastic_flow import LatentNSF


class DirectLatentNSFTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        targets = np.concatenate((np.zeros(32), np.linspace(-.0002, .0003, 96)))
        cls.basis = fit_horizon_basis(targets, knot_count=16)

    def setUp(self):
        torch.manual_seed(38)
        nsf = LatentNSF(torch.zeros(59), torch.ones(59), dimensions=4, width=8)
        self.model = DirectLatentNSF(nsf, self.basis, torch.zeros(19), torch.ones(19), widths=(8,))
        # Nonzero final weights ensure conditioning/gradient tests exercise
        # learned behavior instead of the input-independent initialization.
        with torch.no_grad():
            self.model.head.output.weight.normal_(0, .2)
            nsf.posterior.weight.normal_(0, .1)
            nsf.transition.base[-1].weight.normal_(0, .1)
            nsf.transition.couplings.nets[0][-1].weight.normal_(0, .1)
        self.features = torch.randn(4, 4, 59)
        self.prices = torch.tensor([[100., 100.01, 100.02, 100.],
                                    [101., 101., 100.99, 100.98],
                                    [102., 101.99, 102.01, 102.03],
                                    [103., 103.01, 103., 103.02]], dtype=torch.float64)
        self.elapsed = torch.tensor([[0., 1., 11., 60.]]).expand(4, -1)
        self.times = self.elapsed.double().cumsum(1) + 1700000000.

    def distribution(self, horizon):
        return self.model.distribution(self.features, self.prices, self.elapsed, horizon, self.times)

    def test_direct_distribution_never_samples_or_advances_latent_state(self):
        # Changing the process RNG must not change a forecast, and forecasting
        # must not consume that RNG even while the module is in training mode.
        for training in (False, True):
            self.model.train(training)
            laws = []
            for seed in (19, 503):
                torch.manual_seed(seed)
                before = torch.get_rng_state().clone()
                with ExitStack() as stack:
                    stack.enter_context(patch.object(self.model.nsf.transition, 'rsample',
                                                     side_effect=AssertionError('latent simulation')))
                    stack.enter_context(patch.object(self.model.nsf, 'distribution',
                                                     side_effect=AssertionError('sampled marginal')))
                    for name in ('rand', 'rand_like', 'randn', 'randn_like', 'normal', 'multinomial'):
                        stack.enter_context(patch.object(torch, name, side_effect=AssertionError(name)))
                    stack.enter_context(patch.object(torch.quasirandom, 'SobolEngine',
                                                     side_effect=AssertionError('quasi Monte Carlo')))
                    law = self.distribution(1234.5)
                    laws.append((law.zero_probability, law.masses, law.mean_price(),
                                 law.cdf(self.prices[:, -1], tick_size=None)))
                self.assertTrue(torch.equal(before, torch.get_rng_state()))
            for left, right in zip(*laws):
                torch.testing.assert_close(left, right, atol=0, rtol=0)

    def test_scalar_and_batched_arbitrary_horizons_and_exact_present(self):
        h = torch.tensor([0., .25, 1234.5, 86400.])
        law = self.distribution(h)
        self.assertEqual(law.masses.shape, (4, len(self.basis['knots'])))
        self.assertEqual(law.zero_probability.shape, (4,))
        torch.testing.assert_close(law.masses.sum(-1), torch.ones(4, dtype=torch.float64))
        self.assertTrue(bool((law.masses >= 0).all()))
        self.assertTrue(bool(torch.isfinite(law.mean_price()).all()))
        self.assertEqual(float(law.zero_probability[0].detach()), 1.)
        self.assertEqual(float(law.mean_price()[0].detach()), float(self.prices[0, -1]))
        self.assertEqual(float(law.tick_probability(self.prices[:, -1])[0].detach()), 1.)
        torch.testing.assert_close(law.sample_prices(16)[0], self.prices[0, -1].expand(16),
                                   atol=1e-12, rtol=0)
        for i in range(4):
            scalar = self.model.distribution(self.features[i:i+1], self.prices[i:i+1],
                                            self.elapsed[i:i+1], float(h[i]), self.times[i:i+1])
            torch.testing.assert_close(scalar.zero_probability, law.zero_probability[i:i+1])
            torch.testing.assert_close(scalar.masses, law.masses[i:i+1], atol=1e-9, rtol=1e-6)
        zero, masses = self.model.probabilities(self.features, self.prices, self.elapsed, h, self.times)
        torch.testing.assert_close(zero, law.zero_probability, atol=0, rtol=0)
        torch.testing.assert_close(masses, law.masses, atol=0, rtol=0)

    def test_full_posterior_uncertainty_affects_forecast(self):
        means, stds = self.model.posterior(self.features, self.prices, self.elapsed, self.times)
        means = means.detach().requires_grad_()
        stds = stds.detach().requires_grad_()
        origin = self.times[:, -1]
        inputs = self.model.conditioning(means, stds, 12.5, origin)
        torch.testing.assert_close(inputs[:, :4], means, atol=0, rtol=0)
        torch.testing.assert_close(inputs[:, 4:8], stds, atol=0, rtol=0)
        before = self.model.logits_from_posterior(means, stds, 12.5, origin)
        after = self.model.logits_from_posterior(means, stds + .2, 12.5, origin)
        self.assertFalse(torch.equal(before, after))
        before[:, 0].sum().backward()
        for gradient in (means.grad, stds.grad):
            self.assertTrue(bool(torch.isfinite(gradient).all()))
            self.assertGreater(float(gradient.abs().sum()), 0.)

    def test_history_matches_causal_encoder_and_ignores_padding(self):
        prefix = (self.features[:, :3], self.prices[:, :3], self.elapsed[:, :3], self.times[:, :3])
        original = self.model.posterior(*prefix)
        observed = self.model.nsf.observations(prefix[0], (prefix[1] / prefix[1][:, -1:]).log())
        expected = self.model.nsf.encode(observed, prefix[2], prefix[3])
        for actual, encoded in zip(original, expected):
            torch.testing.assert_close(actual, encoded[:, -1], atol=0, rtol=0)
        # Duplicate timestamps are padding. A dummy feature row must not
        # update the causal state or use an unobserved later price as anchor.
        padded_features = torch.cat((prefix[0], torch.full_like(prefix[0][:, :1], 1e5)), 1)
        padded_prices = torch.cat((prefix[1], prefix[1][:, -1:]), 1)
        padded_elapsed = torch.cat((prefix[2], torch.zeros_like(prefix[2][:, :1])), 1)
        padded_times = torch.cat((prefix[3], prefix[3][:, -1:]), 1)
        padded = self.model.posterior(padded_features, padded_prices, padded_elapsed, padded_times)
        for actual, expected in zip(padded, original):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_direct_likelihood_trains_head_and_causal_encoder(self):
        law = self.distribution(torch.tensor([1., 17., 300., 1234.5]))
        targets = torch.tensor([0., -.0002, .0003, .0001], dtype=torch.float64)
        loss = -law.log_prob(targets).mean()
        self.assertTrue(bool(torch.isfinite(loss)))
        loss.backward()
        for parameter in (self.model.head.output.weight, self.model.nsf.posterior.weight,
                          self.model.nsf.encoder.weight_ih):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(bool(torch.isfinite(parameter.grad).all()))
            self.assertGreater(float(parameter.grad.abs().sum()), 0.)
        self.assertTrue(all(p.grad is None for p in self.model.nsf.transition.parameters()))

    def test_rejects_invalid_observations_and_lossy_timestamps(self):
        arguments = [self.features, self.prices, self.elapsed, self.times]
        invalid = [(0, self.features[:, :, :58]), (0, self.features[:, :0]),
                   (0, self.features * float('nan')), (1, self.prices.float()),
                   (1, self.prices * 0), (1, self.prices[:, :2]),
                   (2, -torch.ones_like(self.elapsed)), (2, self.elapsed * float('inf')),
                   (3, self.times.float()), (3, self.times * float('nan')),
                   (3, self.times + torch.tensor([0., 1., 0., 0.]))]
        for index, value in invalid:
            with self.subTest(index=index, shape=value.shape, dtype=value.dtype):
                bad = list(arguments)
                bad[index] = value
                with self.assertRaises(ValueError):
                    self.model.posterior(*bad)

    def test_rejects_invalid_horizons_and_posterior_parameters(self):
        means, stds = self.model.posterior(self.features, self.prices, self.elapsed, self.times)
        origin = self.times[:, -1]
        for horizon in (-1., 86400.1, float('nan'), float('inf'), [1., 2.], [[1.], [2.], [3.], [4.]]):
            with self.subTest(horizon=horizon), self.assertRaises(ValueError):
                self.model.conditioning(means, stds, horizon, origin)
        for bad_mean, bad_std, bad_origin in ((means[:, :3], stds, origin),
                                             (means, stds[:, :3], origin),
                                             (means * float('nan'), stds, origin),
                                             (means, -stds, origin),
                                             (means, stds * float('inf'), origin),
                                             (means, stds, origin.float()),
                                             (means, stds, origin[:2])):
            with self.subTest(shape=bad_std.shape), self.assertRaises(ValueError):
                self.model.conditioning(bad_mean, bad_std, 1., bad_origin)

    def test_configured_horizon_and_normalization_are_enforced(self):
        for maximum in (0., -1., 86401., float('nan'), float('inf')):
            with self.subTest(maximum=maximum), self.assertRaises(ValueError):
                DirectLatentNSF(self.model.nsf, self.basis, torch.zeros(19), torch.ones(19),
                                widths=(8,), maximum_seconds=maximum)
        for mean, std in ((torch.zeros(18), torch.ones(18)),
                          (torch.zeros(19), torch.ones(18)),
                          (torch.zeros(19), torch.zeros(19))):
            with self.subTest(shape=mean.shape), self.assertRaises(ValueError):
                DirectLatentNSF(self.model.nsf, self.basis, mean, std, widths=(8,))
        self.model.maximum_seconds = 60.
        self.distribution(60.)
        with self.assertRaises(ValueError):
            self.distribution(60.001)

    def test_checkpoint_preserves_direct_law_and_original_latent_transition(self):
        self.model.eval()
        h = torch.tensor([0., .5, 17., 300.])
        before = self.distribution(h)
        means, _ = self.model.posterior(self.features, self.prices, self.elapsed, self.times)
        noise = torch.arange(16, dtype=means.dtype).reshape(4, 4) / 10
        transition_args = (means, means.new_empty((4, 0)), h / self.model.nsf.time_scale,
                           self.times[:, -1] / self.model.nsf.time_scale, noise)
        evolved = self.model.nsf.transition.rsample(*transition_args)
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / 'direct.pt'
            torch.save(self.model.checkpoint(), file)
            restored, payload = load_direct_latent_nsf(file, torch.device('cpu'))
        self.assertEqual(payload['contract'], CONTRACT)
        self.assertFalse(restored.training)
        self.assertEqual(restored.maximum_seconds, self.model.maximum_seconds)
        self.assertEqual(restored.nsf.dimensions, 4)
        self.assertEqual(restored.head.widths, (8,))
        after = restored.distribution(self.features, self.prices, self.elapsed, h, self.times)
        for actual, expected in ((after.zero_probability, before.zero_probability),
                                 (after.masses, before.masses),
                                 (after.mean_price(), before.mean_price()),
                                 (after.cdf(self.prices[:, -1]), before.cdf(self.prices[:, -1]))):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        torch.testing.assert_close(restored.nsf.transition.rsample(*transition_args), evolved, atol=0, rtol=0)
        for name, value in self.model.nsf.state_dict().items():
            torch.testing.assert_close(restored.nsf.state_dict()[name], value, atol=0, rtol=0)

    def test_unrelated_checkpoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / 'wrong.pt'
            torch.save({'contract': 'latent-nsf-calendar-multiscale-v2'}, file)
            with self.assertRaisesRegex(ValueError, 'unsupported'):
                load_direct_latent_nsf(file, torch.device('cpu'))


if __name__ == '__main__':
    unittest.main()
