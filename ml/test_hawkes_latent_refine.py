import unittest
from unittest.mock import patch
import numpy as np
from hawkes_latent import LatentModel
from hawkes_latent_refine import difference_gradient, window_gradient
from hawkes_research import Corpus, Mode


class JointDifferenceTests(unittest.TestCase):
    def test_complete_gradient_matches_coupled_quadratic(self):
        raw = np.array([.3, -.2, .7])
        matrix = np.array([[2., 1., 0.], [1., 4., .5], [0., .5, 3.]])
        gradient = difference_gradient(raw, np.full(3, .01), (np.full(3, -2.), np.full(3, 2.)),
                                       lambda x: .5 * x @ matrix @ x)
        np.testing.assert_allclose(gradient, matrix @ raw, atol=1e-12)

    def test_boundary_probe_denominator_uses_actual_interval(self):
        raw = np.array([0., 1.])
        gradient = difference_gradient(raw, np.full(2, .1), (np.zeros(2), np.ones(2)), lambda x: 2 * x[0] - 3 * x[1])
        np.testing.assert_allclose(gradient, [2., -3.], atol=1e-12)

    def test_all_coordinates_share_context_blocks_and_seeds(self):
        model = LatentModel((.4, .6, .035), (Mode(0, 2, 15., 2.),), beta=((),))
        y = np.ones((100, 2))
        corpus = Corpus(y, np.ones(100, bool), np.ones(100, bool), 80, 100, np.ones(2), {})
        chosen, seeds = np.array([[30, 40], [60, 70]]), np.array([12, 34])
        calls = []

        def loss(candidate, source, blocks, particles, random_seeds, minimum, context):
            calls.append((blocks, random_seeds, context))
            return np.full(len(blocks), float(np.sum(candidate.raw()))), {'full_prefix': False}

        with patch('hawkes_latent_refine.model_context', side_effect=lambda m, c, minimum: 1 + int(m.modes[0].tau * 2)), patch('hawkes_latent_refine.window_losses', side_effect=loss):
            gradient, info = window_gradient(model.raw(), model.layout(), model.link, corpus, chosen, seeds, 8,
                                             np.full(len(model.raw()), .1), 8, 80, minimum_context=1)
        self.assertEqual(len(calls), 2 * len(model.raw()))
        self.assertTrue(all(blocks is chosen and random_seeds is seeds for blocks, random_seeds, _ in calls))
        self.assertEqual(len({c for _, _, c in calls}), 1)
        self.assertTrue(np.isfinite(gradient).all())

    def test_observable_poisson_gradient_and_unobservable_hidden_coordinate(self):
        model = LatentModel((.4, .6, .035), ())
        y = np.random.default_rng(42).poisson([.4, .6], (120, 2)).astype(float)
        valid = np.ones(120, dtype=bool)
        mean = y[:100].mean(axis=0)
        for value in (y, valid, mean):
            value.setflags(write=False)
        corpus = Corpus(y, valid, valid, 100, 120, mean, {})
        gradient, _ = window_gradient(model.raw(), model.layout(), model.link, corpus,
                                      np.array([[0, 50], [50, 100]]), np.array([123, 456]), 8,
                                      np.full(3, .01), 2, 100)
        # Marginal observed likelihood is exactly two independent Poisson laws.
        # A disconnected hidden immigration rate cannot affect their density.
        expected = np.array([.4, .6]) * np.sinh(.01) / .01 - mean
        np.testing.assert_allclose(gradient[:2], expected, atol=1e-8)
        self.assertAlmostEqual(gradient[2], 0., places=8)


if __name__ == '__main__':
    unittest.main()
