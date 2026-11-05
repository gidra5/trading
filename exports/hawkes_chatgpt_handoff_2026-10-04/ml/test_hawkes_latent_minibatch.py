import unittest
import numpy as np
from hawkes_latent import LatentModel, Diffusion, filter_model, physical_arrays, seed_latent
from hawkes_latent_minibatch import conditioning_context, coordinate_scales, observed_initializations, partitions, window_losses
from hawkes_modulation import Anchor
from hawkes_research import Corpus, Model, Mode


class StochasticJointTests(unittest.TestCase):
    def corpus(self):
        y = np.random.default_rng(123).poisson(3, (500, 2)).astype(float)
        valid = np.ones(500, dtype=bool)
        valid[200:210] = False
        mean = np.array([3., 3.])
        return Corpus(y, valid, valid.copy(), 400, 500, mean, {'train_scored_seconds': int(valid[:400].sum()), 'validation_scored_seconds': 100})

    def test_long_observed_traces_are_exact_not_restarted_at_window_boundary(self):
        corpus = self.corpus()
        model = LatentModel((.5, .8, .2), (Mode(0, 0, .3, 5000.), Mode(1, 1, .4, 10000.)),
                            (Anchor(0, 3000.),), ((.1,), (.2,)))
        blocks = np.array([[100, 150], [300, 350]])
        result = filter_model(model, corpus.y[:400], corpus.valid[:400], corpus.initial_mean, 16, 123)
        values, info = window_losses(model, corpus, blocks, 16, np.array([124, 125]), minimum_context=1)
        self.assertFalse(info['full_prefix'])
        self.assertEqual(info['context_seconds'], 1)
        expected = [result[1][s:e][corpus.score[s:e]].sum() for s, e in blocks]
        np.testing.assert_allclose(values, expected, atol=1e-10)

    def test_context_expands_with_latent_scales_and_uses_full_prefix_when_needed(self):
        base = Model((.5, .8), (Mode(0, 0, .3, 2.), Mode(1, 1, .4, 40.)))
        short = seed_latent(base, tau=10.)
        long = seed_latent(base, tau=10000.)
        mean, maximum = np.array([3., 3.]), np.array([100., 100.])
        self.assertEqual(conditioning_context(short, mean, maximum), 3600)
        self.assertGreater(conditioning_context(long, mean, maximum), 400000)
        latent_anchor = LatentModel(short.background, short.modes, (Anchor(2, 30.),), tuple((.1,) for _ in short.modes))
        self.assertIsNone(conditioning_context(latent_anchor, mean, maximum))
        diffusion = LatentModel(short.background, short.modes, beta=short.beta, diffusions=(Diffusion(10000., (.1, .1, .1)),))
        self.assertGreaterEqual(conditioning_context(diffusion, mean, maximum), 400000)

    def test_latent_anchor_fallback_matches_full_particle_prefix(self):
        corpus = self.corpus()
        base = Model((.5, .8), (Mode(0, 0, .3, 2.), Mode(1, 1, .4, 40.)))
        model = seed_latent(base, tau=5.)
        model = LatentModel(model.background, model.modes, (Anchor(2, 30.),), tuple((.1,) for _ in model.modes))
        blocks = np.array([[100, 150], [300, 350]])
        values, info = window_losses(model, corpus, blocks, 16, np.array([124, 125]))
        self.assertTrue(info['full_prefix'])
        full = filter_model(model, corpus.y[:350], corpus.valid[:350], corpus.initial_mean, 16, 124)
        np.testing.assert_array_equal(values, [full[1][s:e][corpus.score[s:e]].sum() for s, e in blocks])

    def test_partition_weighting_covers_every_scored_second_once(self):
        score = np.ones(100, dtype=bool)
        score[20:40] = False
        blocks = partitions(score, 20)
        covered = np.zeros(100, dtype=int)
        for s, e in blocks:
            covered[s:e] += score[s:e]
        np.testing.assert_array_equal(covered, score)

    def test_all_parameter_coordinates_remain_in_joint_vector(self):
        model = LatentModel((.5, .8, .2), (Mode(0, 0, .3, 2.), Mode(1, 2, .4, 40.)),
                            (Anchor(0, 300.),), ((.1,), (.2,)), (Diffusion(1000., (.1, -.1, .2)),))
        scales = coordinate_scales(model.layout())
        self.assertEqual(len(scales), len(model.raw()))
        self.assertTrue(np.all(scales > 0))
        np.testing.assert_array_equal(scales[7:9], .03)


if __name__ == '__main__':
    unittest.main()
