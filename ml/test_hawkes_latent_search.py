import unittest
import contextlib
import io
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from hawkes_latent import Diffusion, LatentModel, core_branching, decode, filter_model, physical_arrays, seed_latent
from hawkes_latent_search import edge_counts, lift_modulated, mirror_observed, modulate_latent, neighbors, search, seed_diffusion
from hawkes_modulation import Anchor, Modulated
from hawkes_research import Model, Mode


class LatentArchitectureTests(unittest.TestCase):
    def setUp(self):
        self.observed = Model((.5, .8), (Mode(0, 0, .3, 2.), Mode(1, 1, .4, 40.)))

    def test_mixture_additions_are_independent_for_all_nine_edges(self):
        model = seed_latent(self.observed)
        for name, candidate in neighbors(model, 'ML'):
            self.assertEqual(len(candidate.modes), len(model.modes) + 1)
            changed = np.array(edge_counts(candidate)) - edge_counts(model)
            self.assertEqual(changed.sum(), 1)
            self.assertEqual(np.count_nonzero(changed), 1)
            self.assertEqual(candidate.modes[:-1], model.modes)
            self.assertEqual(len(candidate.raw()), len(model.raw()) + 2)
            self.assertEqual(len(decode(candidate.raw(), candidate.layout()).modes), len(candidate.modes))
        self.assertEqual(len(list(neighbors(model, 'ML'))), 27)

    def test_modulation_and_diffusion_changes_keep_joint_layout_aligned(self):
        modulation = Modulated(self.observed, (Anchor(0, 200.), Anchor(1, 300.)), ((.1, .2), (.3, .4)))
        latent = lift_modulated(modulation)
        self.assertEqual(latent.beta[:2], modulation.beta)
        model = LatentModel(latent.background, latent.modes, latent.anchors, latent.beta, (Diffusion(300., (.1, -.1, .2)),))
        for name, candidate in list(neighbors(model, 'M2LZ')) + list(neighbors(model, 'M2LZ', additions=False)):
            self.assertEqual(len(candidate.beta), len(candidate.modes), name)
            self.assertTrue(all(len(row) == len(candidate.anchors) for row in candidate.beta), name)
            restored = decode(candidate.raw(), candidate.layout())
            self.assertEqual(len(restored.raw()), len(candidate.raw()), name)
        self.assertTrue(any('add_anchor_2' in name for name, _ in neighbors(model, 'M2LZ')))
        self.assertTrue(any('add_diffusion_imbalance' in name for name, _ in neighbors(model, 'M2LZ')))

    def test_nested_hierarchy_cannot_silently_include_extra_mechanisms(self):
        latent = lift_modulated(Modulated(self.observed, (Anchor(0, 200.),), ((.1,), (.2,))))
        with self.assertRaises(ValueError):
            list(neighbors(latent, 'ML'))
        diffusion = LatentModel(latent.background, latent.modes, latent.anchors, latent.beta, (Diffusion(300., (.1, .1, .1)),))
        with self.assertRaises(ValueError):
            list(neighbors(diffusion, 'M2L'))

    def test_zero_modulation_nests_ml_likelihood_and_forecasts_exactly(self):
        model = seed_latent(self.observed)
        nested = modulate_latent(model, (Anchor(0, 3.), Anchor(1, 35.)))
        y = np.random.default_rng(123).poisson(3, (100, 2)).astype(float)
        valid, mean = np.ones(100, dtype=bool), np.array([3., 3.])
        original = filter_model(model, y, valid, mean, 32, 123)
        changed = filter_model(nested, y, valid, mean, 32, 123)
        np.testing.assert_array_equal(original[0], changed[0])
        np.testing.assert_array_equal(original[1], changed[1])

    def test_regime_initialization_preserves_core_mean_and_keeps_parameters_free(self):
        model = modulate_latent(seed_latent(self.observed), (Anchor(0, 3.), Anchor(1, 35.)))
        mean = physical_arrays(model, np.array([3., 3.]))[8]
        extended = seed_diffusion(model, 900., (.8, .8, .3), np.array([3., 3.]), .85)
        np.testing.assert_allclose(np.array(model.background) + core_branching(model) @ mean,
                                   np.array(extended.background) + core_branching(extended) @ mean, atol=1e-12)
        self.assertEqual(len(extended.raw()), len(model.raw()) + 4)
        self.assertEqual(len(extended.diffusions), 1)

    def test_target_specific_diffusion_start_preserves_conditional_core_rates(self):
        model = seed_latent(self.observed)
        observed_mean = np.array([3., 4.])
        mean = physical_arrays(model, observed_mean)[8]
        extended = seed_diffusion(model, 1., (1., 0., 0.), observed_mean, (.5, 1., 1.))
        np.testing.assert_allclose(np.array(model.background) + core_branching(model) @ mean,
                                   np.array(extended.background) + core_branching(extended) @ mean, atol=1e-12)
        np.testing.assert_allclose(core_branching(extended)[1:], core_branching(model)[1:])

    def test_mirrored_initialization_exchanges_observations_and_is_involutive(self):
        model = seed_latent(self.observed)
        mirrored = mirror_observed(model)
        self.assertEqual(mirror_observed(mirrored), model)
        y = np.random.default_rng(123).poisson([3., 4.], (100, 2)).astype(float)
        valid, mean = np.ones(100, dtype=bool), np.array([3., 4.])
        original = filter_model(model, y, valid, mean, 32, 123)
        changed = filter_model(mirrored, y[:, ::-1].copy(), valid, mean[::-1].copy(), 32, 123)
        np.testing.assert_allclose(original[0], changed[0][:, ::-1], atol=1e-12)
        np.testing.assert_allclose(original[1], changed[1], atol=1e-12)

    def test_rare_hidden_channel_addition_can_start_with_material_observed_excitation(self):
        model = LatentModel((.4, .7, 1e-4), (Mode(0, 2, 100., 2.), Mode(2, 2, .2, 3.)), beta=((), ()))
        observed_mean = np.array([8., 7.])
        mean = physical_arrays(model, observed_mean)[8]
        additions = [(name, candidate) for name, candidate in neighbors(model, 'ML', observed_mean=observed_mean)
                     if candidate.modes[-1].source == 2 and candidate.modes[-1].target == 0]
        self.assertEqual(len(additions), 6)
        relative = [candidate.modes[-1].mass for name, candidate in additions if name.endswith('relative')]
        calibrated = [candidate.modes[-1].mass for name, candidate in additions if name.endswith('rate_calibrated')]
        self.assertEqual(len(relative), 3)
        self.assertEqual(len(calibrated), 3)
        self.assertTrue(all(mass > 1000. for mass in calibrated))
        np.testing.assert_allclose(np.array(calibrated) * mean[2], .05 * mean[0])

    def test_optimization_progress_alone_does_not_promote_added_modes(self):
        # Oracle objective: continuing the fit helps, but an added mode has no
        # explanatory value. Complexity must be compared after equal refits.
        model = LatentModel((.4, .7, .001), self.observed.modes, beta=((), ()))
        corpus = SimpleNamespace(initial_mean=np.array([8., 7.]), identity={'validation_scored_seconds': 10000})
        def oracle(seed, corpus, output, *args, **kwargs):
            candidate = LatentModel((seed.background[0] - .01, *seed.background[1:]), seed.modes, seed.anchors, seed.beta, seed.diffusions, seed.link)
            output.mkdir(parents=True, exist_ok=True)
            loss = candidate.background[0] + max(0, 2 - len(candidate.modes))
            (output / 'fit-result.json').write_text(json.dumps({'model': candidate.serialize(),
                'joint_parameters': len(candidate.raw()), 'validation_nll_without_factorial': loss,
                'spectral_radius': .4, 'convergence_established': True}))
            return candidate
        with tempfile.TemporaryDirectory() as temporary, patch('hawkes_latent_search.fit_joint', side_effect=oracle), contextlib.redirect_stdout(io.StringIO()):
            output = Path(temporary)
            selected = search([model], corpus, output, 'ML', rounds=2, iterations=1)
            result = json.loads((output / 'selected-ML.json').read_text())
        self.assertEqual(len(selected.modes), 2)
        self.assertTrue(result['fit']['operation'].startswith('joint_refit_control'))

    def test_batch_scheduling_preserves_greedy_controls_and_complete_neighborhoods(self):
        model = LatentModel((.4, .7, .001), self.observed.modes, beta=((), ()))
        corpus = SimpleNamespace(initial_mean=np.array([8., 7.]), identity={'validation_scored_seconds': 10000})
        batches = []
        def oracle(seed, corpus, output, *args, **kwargs):
            output.mkdir(parents=True, exist_ok=True)
            (output / 'fit-result.json').write_text(json.dumps({'model': seed.serialize(),
                'joint_parameters': len(seed.raw()), 'validation_nll_without_factorial': 1 + abs(len(seed.modes) - 2),
                'spectral_radius': .4, 'convergence_established': True}))
        def fit_many(jobs):
            batches.append(len(jobs))
            for seed, directory in jobs:
                oracle(seed, corpus, directory)
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
            serial, batch = Path(temporary) / 'serial', Path(temporary) / 'batch'
            with patch('hawkes_latent_search.fit_joint', side_effect=oracle):
                a = search([model], corpus, serial, 'ML', iterations=1, rounds=2)
            b = search([model], corpus, batch, 'ML', iterations=1, rounds=2, fit_many=fit_many)
            self.assertEqual(a, b)
            self.assertEqual((serial / 'model_selection.csv').read_text(), (batch / 'model_selection.csv').read_text())
            self.assertEqual(json.loads((serial / 'selected-ML.json').read_text()), json.loads((batch / 'selected-ML.json').read_text()))
        self.assertEqual(batches, [1, 48, 2])


if __name__ == '__main__':
    unittest.main()
