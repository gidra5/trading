"""Causal data and actual frozen-parent training checks for the direct head."""
import unittest

import numpy as np
import torch

from direct_latent_nsf import DirectLatentNSF
from horizon_price_forecast import fit_horizon_basis
from latent_neural_stochastic_flow import LatentNSF
from train_direct_latent_nsf import (ANCHORS, encode_data, evaluate_nll, history,
                                    law_at, posterior_tensors, sample_horizons, teacher_bank)
from train_target_horizon_forecast import targets


class RecordingTimeline:
    def __init__(self, values):
        self.values = values
        self.reads = []

    def __getitem__(self, indices):
        self.reads.append(np.array(indices, copy=True))
        return self.values[indices]


class DirectTrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(43)
        self.device = torch.device('cpu')
        self.nsf = LatentNSF(torch.zeros(59), torch.ones(59), dimensions=4, width=8).eval()
        self.nsf.requires_grad_(False)
        with torch.no_grad():
            self.nsf.posterior.weight.normal_(0, .1)
        self.origins = np.array([3, 7, 11, 16, 21, 28])
        timeline = np.random.default_rng(9).normal(0, .01, (100, 59)).astype(np.float32)
        # Decimal ticks include values whose float quotient by .01 is just
        # below an integer (1.13), exercising teacher/student tick agreement.
        prices = (113 + np.arange(90) % 7).astype(np.float64) / 100
        self.data = {'origins': self.origins, 'indices': self.origins - self.origins[0],
                     'timeline': timeline, 'prices': prices,
                     'pricesWithPast': np.r_[1.12, prices],
                     'timestamps': 1700000000000 + self.origins * 1000}

    def test_history_only_reads_two_completed_observations(self):
        recorder = RecordingTimeline(self.data['timeline'])
        self.data['timeline'] = recorder
        ids = np.array([0, 2, 5])
        f, p, dt, times = history(self.data, ids, self.device)
        expected_rows = self.origins[ids, None] + np.array([-1, 0])
        self.assertEqual(len(recorder.reads), 1)
        np.testing.assert_array_equal(recorder.reads[0], expected_rows)
        np.testing.assert_array_equal(f.numpy(), recorder.values[expected_rows])
        np.testing.assert_array_equal(p[:, -1].numpy(), self.data['prices'][self.data['indices'][ids]])
        np.testing.assert_array_equal(dt.numpy(), [[0., 1.]] * len(ids))
        np.testing.assert_array_equal(times[:, -1].numpy(), self.data['timestamps'][ids] / 1000)
        torch.testing.assert_close(times[:, 1] - times[:, 0], torch.ones(len(ids), dtype=torch.float64))
        self.assertEqual(p.dtype, torch.float64)
        self.assertEqual(times.dtype, torch.float64)
        # The same cached origin must survive arbitrary changes to future
        # feature channels; the training path cannot accidentally read them.
        first = history(self.data, np.array([0]), self.device)
        recorder.values[self.origins[0]+1:] = 1e9
        again = history(self.data, np.array([0]), self.device)
        for actual, expected in zip(again, first):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_targets_use_exact_endpoint_prices_and_preserve_zero_atom(self):
        horizons = np.array([7, 1, 5, 15, 30, 60])
        expected_prices = self.data['prices'][self.data['indices'] + horizons]
        current = self.data['prices'][self.data['indices']]
        actual = targets(self.data, horizons)
        np.testing.assert_allclose(actual, np.log(expected_prices / current), atol=1e-16, rtol=0)
        self.assertEqual(float(actual[0]), 0.)
        self.data['timeline'][:] = 1e20
        np.testing.assert_array_equal(targets(self.data, horizons), actual)

    def test_horizon_sampler_covers_short_anchors_and_full_valid_range(self):
        horizons = sample_horizons(np.random.default_rng(917), 32768)
        self.assertTrue(np.issubdtype(horizons.dtype, np.integer))
        self.assertEqual(horizons.shape, (32768,))
        self.assertTrue(np.all((horizons >= 1) & (horizons <= 86400)))
        self.assertTrue(set(ANCHORS).issubset(set(horizons)))
        self.assertGreater(np.mean(horizons <= 60), .15)
        self.assertGreater(np.mean(~np.isin(horizons, ANCHORS)), .4)
        np.testing.assert_array_equal(horizons, sample_horizons(np.random.default_rng(917), 32768))

    def test_teacher_bank_uses_identical_tick_boundaries_to_price_law(self):
        ids = np.array([0, 2, 5])
        horizons = np.array([1, 17, 300])
        queries, cdfs = teacher_bank(self.nsf, self.data, ids, horizons, .005, self.device)
        f, p, dt, times = history(self.data, ids, self.device)
        parent = self.nsf.distribution(f, p, dt, torch.from_numpy(horizons), times, components=64, seed=73)
        expected = torch.stack([parent.cdf(torch.from_numpy(queries[:, column]))
                                for column in range(queries.shape[1])], 1).numpy()
        self.assertEqual(queries.shape, (len(ids), 34))
        np.testing.assert_array_equal(queries[:, -1], p[:, -1].numpy())
        np.testing.assert_allclose(cdfs, expected, atol=1e-15, rtol=0)

    def test_tick_nll_and_projection_optimizer_step_preserves_frozen_parent(self):
        encoded = encode_data(self.nsf, self.data, self.device)
        posterior = posterior_tensors(encoded, self.data, self.device)
        self.assertTrue(all(not tensor.is_inference() for tensor in posterior))
        train_targets = np.r_[np.zeros(32), np.linspace(-.02, .02, 96)]
        basis = fit_horizon_basis(train_targets, knot_count=16)
        model = DirectLatentNSF(self.nsf, basis, torch.zeros(19), torch.ones(19), widths=(8,))
        ids = np.array([0, 2, 4, 5])
        horizons = np.array([1, 5, 15, 60])
        queries, cdfs = teacher_bank(self.nsf, self.data, ids, horizons, model.head.sigma, self.device)
        parent_before = {name: tensor.clone() for name, tensor in model.nsf.state_dict().items()}
        head_before = model.head.output.bias.detach().clone()
        optimizer = torch.optim.AdamW(model.head.parameters(), lr=.001)
        predicted = law_at(model, posterior, ids, horizons)
        actual = torch.from_numpy(self.data['prices'][self.data['indices'][ids] + horizons])
        nll = -predicted.tick_probability(actual).clamp_min(1e-300).log().mean()
        projection = (predicted.cdf(torch.from_numpy(queries)) - torch.from_numpy(cdfs)).square().mean()
        self.assertTrue(bool(torch.isfinite(nll + projection)))
        optimizer.zero_grad(set_to_none=True)
        (nll + projection).backward()
        self.assertTrue(all(p.grad is None or bool(torch.isfinite(p.grad).all())
                            for p in model.head.parameters()))
        self.assertGreater(float(model.head.output.bias.grad.abs().sum()), 0.)
        optimizer.step()
        self.assertFalse(torch.equal(model.head.output.bias, head_before))
        for name, tensor in model.nsf.state_dict().items():
            torch.testing.assert_close(tensor, parent_before[name], atol=0, rtol=0)
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in model.nsf.parameters()))
        # Validation scoring must be the actual endpoint tick likelihood,
        # with no sampled target or density/price-coordinate substitution.
        with torch.inference_mode():
            expected = -law_at(model, posterior, ids, horizons).tick_probability(actual).log().mean()
        self.assertAlmostEqual(evaluate_nll(model, posterior, self.data, ids, horizons), float(expected), places=12)


if __name__ == '__main__':
    unittest.main()
