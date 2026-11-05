import json
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
from concurrent.futures import ThreadPoolExecutor
import unittest
import numpy as np
from hawkes_latent import LatentModel, filter_model, Diffusion
from hawkes_modulation import Anchor
from hawkes_research import Corpus, Mode
from hawkes_latent_minibatch import fit_joint
from hawkes_latent_gpu_fit import FitState, evaluate_trajectories, fit_joint_batch, backend_identity


class GPUFitTests(unittest.TestCase):
    def corpus(self):
        y = np.random.default_rng(3).poisson(3, (160, 2)).astype(float)
        valid = np.ones(160, bool)
        valid[45:50] = False
        return Corpus(y, valid, valid.copy(), 120, 160, np.array([3., 3.]),
                      {'train_scored_seconds': 115, 'validation_scored_seconds': 40})

    def model(self):
        return LatentModel((1., 1.5, .2), (Mode(0, 0, .1, 2.), Mode(1, 1, .2, 4.)),
                           (Anchor(0, 3.),), ((.02,), (.03,)))

    def settings(self):
        return dict(checkpoint_every=2, checkpoint_particles=8, window_seconds=16, batch_windows=2, minimum_context=1)

    def test_cpu_reference_is_bitwise_stable_under_threads(self):
        c = self.corpus()
        model = LatentModel((1., 1.5, .2), (Mode(0, 2, 2., 1.), Mode(2, 0, .03, 4.)),
                            diffusions=(Diffusion(2., (.2, -.1, .05)),))
        seeds = (1, 2, 3, 4)
        serial = [filter_model(model, c.y, c.valid, c.initial_mean, 16, seed) for seed in seeds]
        with ThreadPoolExecutor(max_workers=4) as pool:
            parallel = list(pool.map(lambda seed: filter_model(model, c.y, c.valid, c.initial_mean, 16, seed), seeds))
        for left, right in zip(serial, parallel):
            for a, b in zip(left, right):
                np.testing.assert_array_equal(a, b)

    def test_exact_observed_case_matches_cpu_optimizer_and_scores(self):
        c, model = self.corpus(), self.model()
        with TemporaryDirectory() as root:
            cpu, gpu = Path(root) / 'cpu', Path(root) / 'gpu'
            fit_joint(model, c, cpu, particles=8, iterations=2, **self.settings())
            fit_joint_batch([(model, gpu)], c, particles=8, iterations=2, cpu_workers=2, **self.settings())
            left, right = [json.loads((p / 'joint-progress.json').read_text()) for p in (cpu, gpu)]
            for key in ('raw', 'first_moment', 'second_moment'):
                np.testing.assert_allclose(left[key], right[key], atol=1e-8, rtol=1e-9)
            self.assertEqual(left['rng_state'], right['rng_state'])
            self.assertEqual(left['config'], right['config'])
            final = json.loads((gpu / 'fit-result.json').read_text())
            self.assertEqual(final['joint_parameters'], len(model.raw()))
            self.assertEqual(final['accelerator_transition']['first_accelerator_iteration'], 0)
            self.assertIsNone(final['accelerator_transition']['first_cuda_iteration'])
            self.assertFalse(final['convergence_established'])

    def test_interrupted_resume_preserves_every_optimizer_state(self):
        c, model = self.corpus(), self.model()
        with TemporaryDirectory() as root:
            whole, resumed, companion = [Path(root) / name for name in ('whole', 'resumed', 'companion')]
            fit_joint_batch([(model, whole)], c, particles=8, iterations=2, cpu_workers=2, **self.settings())
            state = FitState(model, c, resumed, 8, 2, **self.settings())
            state.history.append(state.checkpoint(-1, state.raw))
            trajectories = state.prepare()
            losses, _ = evaluate_trajectories(trajectories, c, 8)
            state.update(losses[id(state)])
            before = json.loads((resumed / 'joint-progress.json').read_text())
            restored = FitState(model, c, resumed, 8, 2, **self.settings())
            np.testing.assert_array_equal(restored.raw, before['raw'])
            np.testing.assert_array_equal(restored.first, before['first_moment'])
            np.testing.assert_array_equal(restored.second, before['second_moment'])
            self.assertEqual(restored.rng.bit_generator.state, before['rng_state'])
            fit_joint_batch([(model, resumed), (model, companion)], c, particles=8, iterations=2, cpu_workers=2, **self.settings())
            left, right = [json.loads((p / 'joint-progress.json').read_text()) for p in (whole, resumed)]
            for key in ('raw', 'first_moment', 'second_moment'):
                np.testing.assert_array_equal(left[key], right[key])
            self.assertEqual(left['rng_state'], right['rng_state'])
            self.assertEqual(right['next_iteration'], 2)

    def test_completed_async_initial_checkpoint_is_restored_to_history(self):
        c, model = self.corpus(), self.model()
        with TemporaryDirectory() as root:
            output = Path(root)
            state = FitState(model, c, output, 8, 2, **self.settings())
            state.checkpoint(-1, state.raw)
            trajectories = state.prepare()
            losses, _ = evaluate_trajectories(trajectories, c, 8)
            state.update(losses[id(state)])
            restored = FitState(model, c, output, 8, 2, **self.settings())
            self.assertEqual(restored.history[0]['iteration'], -1)
            self.assertTrue(restored.history[0]['full_data_checkpoint'])
            np.testing.assert_array_equal(restored.raw, state.raw)
            self.assertEqual(restored.rng.bit_generator.state, state.rng.bit_generator.state)

    def test_backend_transition_requires_exact_checkpoint_manifest(self):
        c, model = self.corpus(), self.model()
        with TemporaryDirectory() as root:
            output = Path(root)
            state = FitState(model, c, output, 8, 2, **self.settings())
            state.save()
            transition_path = output / 'accelerator-resume.json'
            transition = json.loads(transition_path.read_text())
            transition['backend'] = {'backend': 'previous tested numerical backend'}
            transition_path.write_text(json.dumps(transition))
            old_bytes = transition_path.read_bytes()
            with self.assertRaisesRegex(ValueError, 'without a checkpoint-bound'):
                FitState(model, c, output, 8, 2, **self.settings())
            manifest = {'target_backend': backend_identity(), 'next_iteration': 0,
                'progress_sha256': hashlib.sha256((output / 'joint-progress.json').read_bytes()).hexdigest(),
                'previous_transition_sha256': hashlib.sha256(old_bytes).hexdigest()}
            (output / 'backend-migration.json').write_text(json.dumps({**manifest, 'progress_sha256': 'wrong'}))
            with self.assertRaisesRegex(ValueError, 'does not bind'):
                FitState(model, c, output, 8, 2, **self.settings())
            (output / 'backend-migration.json').write_text(json.dumps(manifest))
            restored = FitState(model, c, output, 8, 2, **self.settings())
            self.assertEqual(restored.transition['numerical_backend_transitions'][0]['at_iteration'], 0)
            self.assertEqual(restored.transition['backend'], backend_identity())


if __name__ == '__main__':
    unittest.main()
