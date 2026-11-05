import unittest
import numpy as np
from hawkes_count_experiment import branching_core, features, objective, particle_filter


class HawkesCountTests(unittest.TestCase):
    def test_synthetic_hidden_event_filter_beats_stationary_poisson(self):
        rng = np.random.default_rng(123)
        n = 3000
        y = np.zeros((n, 2))
        z = 0.
        decay = np.exp(-.5)
        for t in range(n):
            y[t] = rng.poisson(1 + 3 * z, 2)
            latent = rng.poisson(.2 + .7 * z)
            z = decay * z + (1 - decay) * latent
        baseline = np.zeros((n, 2, 2))
        baseline[:, :, 0] = 1.
        params = np.array([.2, 0., 0., .7, 3., 3., 1., 1., 1., 1.])
        _, losses, _ = particle_filter(y, np.ones(n, dtype=bool), baseline, np.zeros_like(y), params, 2., 1, 256, 42)
        stationary_mean = 1 + 3 * .2 / (1 - .7)
        control = (stationary_mean - y * np.log(stationary_mean)).sum(axis=1)
        self.assertLess(losses[500:].mean(), control[500:].mean())

    def test_feedback_loop_can_be_unstable_despite_stable_diagonals(self):
        weights = np.array([[1., 1.], [.8, 0.], [0., .8]])
        params = np.array([.1, .2, .2, .4, 2., 2., 1., 1., 1., 1.])
        matrix = branching_core(weights, params, 1)
        self.assertTrue(np.all(matrix.diagonal() < 1))
        self.assertGreater(np.max(np.abs(np.linalg.eigvals(matrix))), 1)

    def test_predictions_do_not_use_current_or_future_observations(self):
        y = np.ones((50, 2))
        changed = y.copy()
        changed[25:] = 100
        valid = np.ones(50, dtype=bool)
        for original, altered in zip(features(y, valid, np.array([10., 40.]), 5), features(changed, valid, np.array([10., 40.]), 5)):
            np.testing.assert_array_equal(original[:26], altered[:26])
        baseline = np.ones((50, 2, 2))
        state = features(y, valid, np.array([10.]), 5)[0][:, 1:3]
        params = np.array([.1, .1, .1, .2, 2., 2., 1., 1., 1., 1.])
        original = particle_filter(y, valid, baseline, state, params, 10., 5, 64, 42)[0]
        altered = particle_filter(changed, valid, baseline, state, params, 10., 5, 64, 42)[0]
        np.testing.assert_array_equal(original[:26], altered[:26])

    def test_disconnected_latent_recovers_observed_likelihood(self):
        y = np.array([[0., 1.], [4., 2.], [8., 0.]])
        baseline = np.full((3, 2, 2), 1.5)
        params = np.array([.1, .1, .1, .2, 0., 0., 1., 1., 1., 1.])
        prediction, losses, _ = particle_filter(y, np.ones(3, dtype=bool), baseline, np.ones_like(y), params, 10., 5, 32, 42)
        np.testing.assert_allclose(prediction, 3.)
        np.testing.assert_allclose(losses, (3. - y * np.log(3.)).sum(axis=1))

    def test_recording_gap_resets_memory(self):
        y = np.full((5, 2), 10.)
        valid = np.array([True, True, False, True, True])
        linear, modulation, age = features(y, valid, np.array([10., 40.]), 5)
        np.testing.assert_array_equal(linear[3, 1:], 0.)
        np.testing.assert_array_equal(modulation[3], 0.)
        self.assertEqual(age[3], 0)

    def test_poisson_gradient_matches_finite_differences(self):
        rng = np.random.default_rng(7)
        x = rng.uniform(.1, 2, (20, 3))
        y = rng.poisson(3, (20, 2)).astype(float)
        w = rng.uniform(.1, 1, (3, 2))
        _, grad = objective(w, x, y)
        for row in range(3):
            for side in range(2):
                step = np.zeros_like(w)
                step[row, side] = 1e-6
                finite = (objective(w + step, x, y)[0] - objective(w - step, x, y)[0]) / 2e-6
                self.assertAlmostEqual(grad[row, side], finite, places=6)


if __name__ == '__main__':
    unittest.main()
