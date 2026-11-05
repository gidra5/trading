from types import SimpleNamespace
import unittest
import numpy as np
from hawkes_latent import LatentModel, Diffusion, filter_model
from hawkes_latent_dispatch import evaluate_batches
from hawkes_research import Mode


class DispatchTests(unittest.TestCase):
    def corpus(self):
        y = np.random.default_rng(8).poisson(3, (180, 2)).astype(float)
        valid = np.ones(180, bool)
        valid[60:65] = False
        return SimpleNamespace(y=y, valid=valid, score=valid, train_end=180, initial_mean=np.array([3., 3.]))

    def test_cpu_windows_match_reference_and_repeated_masks(self):
        c, state = self.corpus(), object()
        model = LatentModel((1., 1., .1), (Mode(0, 2, 3., 2.), Mode(2, 0, .02, 3.)),
                            diffusions=(Diffusion(3., (.3, -.1, .01)),))
        blocks = np.array([[40, 90], [40, 90], [100, 140]])
        group = [(state, leg, model, 20, 150, 72, None, None, blocks) for leg in (0, 1)]
        losses, counts = evaluate_batches([group], c, 8)
        ref = filter_model(model, c.y[20:150], c.valid[20:150], c.initial_mean, 8, 72)[1]
        expected = sum(ref[s-20:e-20][c.score[s:e]].sum() for s, e in blocks)
        np.testing.assert_array_equal(losses[id(state)], [expected, expected])
        self.assertEqual(counts[id(state)], {'cpu_numpy': 2, 'cuda_philox': 0})

    def test_cpu_and_cuda_agree_for_exact_observed_case(self):
        c, state = self.corpus(), object()
        model = LatentModel((1., 1., .1), (Mode(0, 0, .2, 2.),))
        group = [(state, leg, model, 0, 150, 72, None, None, np.array([[30, 150]])) for leg in (0, 1)]
        cpu, _ = evaluate_batches([group], c, 8)
        gpu, counts = evaluate_batches([group], c, 8, cpu_threshold=0)
        np.testing.assert_allclose(cpu[id(state)], gpu[id(state)], atol=1e-10, rtol=1e-12)
        self.assertEqual(counts[id(state)], {'cpu_numpy': 0, 'cuda_philox': 2})

    def test_split_pair_is_never_assigned_different_generators(self):
        c, state = self.corpus(), object()
        model = LatentModel((1., 1., .1), ())
        pair = [(state, leg, model, 0, 150, 72, None, None, np.array([[30, 150]])) for leg in (0, 1)]
        others = [(object(), 0, model, 0, 150, 80+i, None, None, np.array([[30, 150]])) for i in range(9)]
        _, counts = evaluate_batches([[pair[0], *others], [pair[1]]], c, 8)
        self.assertEqual(counts[id(state)], {'cpu_numpy': 2, 'cuda_philox': 0})


if __name__ == '__main__':
    unittest.main()
