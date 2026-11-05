import unittest
import json
import tempfile
from pathlib import Path
import numpy as np
from hawkes_latent import Diffusion, LatentModel, filter_model, seed_latent
from hawkes_latent_forecasts import evaluate_forecasts, exact_means, linear_operators, rollout
from hawkes_research import Corpus, Model, Mode, forecast_operators


class HiddenForecastTests(unittest.TestCase):
    def test_saved_predictive_loss_integrates_states_instead_of_scoring_the_mean(self):
        y = np.random.default_rng(28).poisson([3., 4.], (20000, 2)).astype(float)
        valid = np.ones(len(y), dtype=bool)
        corpus = Corpus(y, valid, valid, 12000, 16000, np.array([3., 4.]), {'toy': True})
        model = LatentModel((3., 4., .01), (), diffusions=(Diffusion(1.4, (1., .8, 0.)),))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            evaluate_forecasts(model, corpus, root, 'M2LZ', filter_particles=32, paths=16, split='test')
            metric = json.loads((root / 'test_one_step_metrics.json').read_text())
            with np.load(root / 'test_predictive_loss.npz') as saved:
                self.assertAlmostEqual(saved['marginal_nll'].mean(), metric['predictive_marginal_nll'], places=12)
                np.testing.assert_array_equal(saved['seconds'], np.arange(16000, 20000))
            self.assertGreater(abs(metric['predictive_marginal_nll'] - metric['poisson_mean_log_score']), .001)

    def test_linear_three_channel_operator_matches_direct_recursion_through_one_hour(self):
        observed = Model((.5, .8), (Mode(0, 0, .3, 2.), Mode(1, 1, .4, 40.)))
        model = seed_latent(observed, tau=5, strength=.1)
        phi = np.arange(len(model.modes), dtype=float) / 10
        decay = np.exp(-1 / np.array([m.tau for m in model.modes]))
        cumulative = np.zeros(2)
        horizons = [1, 5, 15, 60, 300, 900, 3600]
        reference = []
        for t in range(1, 3601):
            rate = np.array(model.background)
            for j, mode in enumerate(model.modes):
                rate[mode.target] += mode.mass * phi[j]
            cumulative += rate[:2]
            phi = decay * phi + (1 - decay) * np.array([rate[m.source] for m in model.modes])
            if t in horizons:
                reference.append(cumulative.copy())
        initial = np.r_[np.arange(len(model.modes)) / 10, 1.]
        np.testing.assert_allclose(linear_operators(model) @ initial, reference, rtol=1e-12)

    def test_snapshot_forecasts_do_not_read_current_or_future_counts(self):
        observed = Model((.5, .8), (Mode(0, 0, .3, 2.), Mode(1, 1, .4, 40.)))
        model = seed_latent(observed, tau=5, strength=.1)
        y = np.random.default_rng(1).poisson(3, (100, 2)).astype(float)
        valid, mean, origins = np.ones(100, dtype=bool), np.array([3., 3.]), np.array([50])
        original = filter_model(model, y, valid, mean, 64, 42, origins)
        changed = y.copy()
        changed[50:] = 500
        altered = filter_model(model, changed, valid, mean, 64, 42, origins)
        for left, right in zip(original[6], altered[6]):
            np.testing.assert_array_equal(left, right)
        np.testing.assert_allclose(exact_means(model, original[6])[:, 0], original[0][origins], atol=1e-12)

    def test_generative_rollout_agrees_with_exact_hidden_forecast(self):
        observed = Model((.5, .8), (Mode(0, 0, .3, 2.), Mode(1, 1, .4, 40.)))
        model = seed_latent(observed, tau=5, strength=.1)
        y = np.random.default_rng(1).poisson(3, (100, 2)).astype(float)
        mean, origins = np.array([3., 3.]), np.array([50])
        filtered = filter_model(model, y, np.ones(100, dtype=bool), mean, 256, 42, origins)
        horizons = [1, 5, 60]
        exact = exact_means(model, filtered[6], horizons)
        estimate, se, failed, samples = rollout(model, filtered[6], mean, 10000, horizons, origins=origins)
        np.testing.assert_array_equal(failed, 0)
        self.assertTrue(np.all(np.abs(estimate - exact) < 5 * se + .02))
        self.assertTrue(np.all(samples >= 0))
        np.testing.assert_array_equal(samples, np.floor(samples))

    def test_disconnected_hidden_operators_nest_two_channel_operators(self):
        observed = Model((.5, .8), (Mode(0, 0, .3, 2.), Mode(1, 1, .4, 40.)))
        model = LatentModel((*observed.background, .2), observed.modes, beta=tuple(() for _ in observed.modes))
        np.testing.assert_array_equal(linear_operators(model), forecast_operators(observed))

    def test_numerical_failed_paths_are_not_averaged_away(self):
        model = LatentModel((1e13, 1e13, .1), ())
        snapshots = (np.zeros((1, 1, 0)), np.zeros((1, 1, 0)), np.zeros((1, 1, 0)), np.zeros((1, 1, 0)), np.zeros((1, 1)), np.zeros((1, 1, 0)))
        estimate, _, failed, _ = rollout(model, snapshots, np.array([1., 1.]), 16, [1, 60])
        self.assertTrue(np.isnan(estimate).all())
        np.testing.assert_array_equal(failed, 1)

    def test_exact_ou_immigration_moments_match_closed_form_through_one_hour(self):
        model = LatentModel((.5, .8, .1), (), diffusions=(Diffusion(5., (.4, -.3, .2)),))
        snapshots = (np.zeros((1, 4, 0)), np.zeros((1, 4, 0)), np.zeros((1, 4, 0)),
                     np.zeros((1, 4, 0)), np.zeros((1, 4)), np.full((1, 4, 1), 1.5))
        horizons = [1, 5, 60, 3600]
        time = np.arange(1, 3601)
        d = np.exp(-time / 5.)
        loading = np.array([.4, -.3])
        rate = np.array([.5, .8]) * np.exp(1.5 * d[:, None] * loading - .5 * d[:, None] ** 2 * loading ** 2)
        expected = np.cumsum(rate, axis=0)[np.array(horizons) - 1]
        np.testing.assert_allclose(exact_means(model, snapshots, horizons)[0], expected, rtol=1e-12)

    def test_mixed_ou_and_hidden_hawkes_simulation_matches_exact_means(self):
        observed = Model((.5, .8), (Mode(0, 0, .3, 2.), Mode(1, 1, .4, 40.)))
        base = seed_latent(observed, tau=5., strength=.1)
        model = LatentModel(base.background, base.modes, beta=base.beta, diffusions=(Diffusion(10., (.3, -.2, .2)),))
        k = len(model.modes)
        fast = np.tile(np.arange(k) / 10., (1, 8, 1))
        slopes = np.zeros_like(fast)
        for j, mode in enumerate(model.modes):
            if mode.source == 2:
                slopes[:, :, j] = 1 - np.exp(-1 / mode.tau)
        snapshots = (fast, slopes, np.zeros((1, 8, 0)), np.zeros((1, 8, 0)),
                     np.full((1, 8), .3), np.full((1, 8, 1), .7))
        horizons = [1, 5, 60]
        exact = exact_means(model, snapshots, horizons)
        estimate, error, failed, _ = rollout(model, snapshots, np.array([3., 3.]), 10000, horizons)
        np.testing.assert_array_equal(failed, 0)
        self.assertTrue(np.all(np.abs(estimate - exact) < 5 * error + .02))


if __name__ == '__main__':
    unittest.main()
