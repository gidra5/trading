import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from hawkes_final_test import LABELS, REQUIREMENTS, read_frozen_selection, reusable_forecasts, reuse_identical_latent_forecasts, run, sha256, verify_evaluation_plan
from hawkes_latent import LatentModel
from hawkes_modulation import Modulated
from hawkes_research import HORIZONS, Corpus, Model, Mode, _load_corpus, evaluate_forecasts, evaluation_region, forecast_origins, load_corpus, save_json


class FrozenEvaluationTests(unittest.TestCase):
    def test_frozen_plan_rejects_changed_budget_before_opening_test_data(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache, freeze = self.fixture(root)
            manifest = json.loads(freeze.read_text())
            plan = {'models': list(LABELS), 'evaluation_split': 'test',
                    'filter_particles': 512, 'filter_seed': 71391, 'future_paths': 2048,
                    'future_seed': 20261003, 'stride_seconds': 900,
                    'horizons_seconds': list(HORIZONS), 'warmup_seconds': 1,
                    'cache_sha256': manifest['cache_sha256'], 'test_evaluated_in_this_pipeline': False}
            path = root / 'evaluation-plan.json'
            save_json(path, plan)
            manifest.update(evaluation_plan=path.name, evaluation_plan_sha256=sha256(path))
            save_json(freeze, manifest)
            self.assertEqual(verify_evaluation_plan(manifest, freeze, 512, 2048, 900), path)
            with patch('hawkes_final_test._load_corpus', side_effect=AssertionError('Test data opened')):
                for particles, paths, stride in ((128, 2048, 900), (512, 512, 900), (512, 2048, 300)):
                    with self.assertRaisesRegex(ValueError, 'settings differ'):
                        run(freeze, cache, root / 'results', particles, paths, stride)
                save_json(path, dict(plan, future_paths=512))
                with self.assertRaisesRegex(ValueError, 'plan changed'):
                    run(freeze, cache, root / 'results')
            self.assertFalse((root / 'results').exists())

    def test_identical_model_reuse_preserves_numbers_and_rejects_changed_settings(self):
        model = LatentModel((.4, .7, .001), (Mode(0, 0, .3, 2.),))
        identity = {'toy': True}
        settings = {'model': model.serialize(), 'corpus': identity, 'split': 'test',
                    'test_evaluated': True, 'stride_seconds': 900,
                    'filter_particles': 512, 'filter_seed': 71391, 'future_paths': 2048}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, destination = root / 'ML', root / 'M2L'
            source.mkdir()
            save_json(source / 'forecast-protocol.json', settings)
            for name in ('test_forecasts.npz', 'test_predictive_loss.npz', 'test_latent_trace.npz',
                         'test_origin_latent_states.npz'):
                np.savez(source / name, numbers=np.array([1., 2., 3.]))
            save_json(source / 'test_one_step_metrics.json', {'total_mae': 3.5})
            save_json(source / 'test_latent_diagnostics.json', {'model': 'ML', 'filtered_mean': .2})
            (source / 'test_horizon_metrics.csv').write_text('model,horizon_seconds,total_mae\nML,3600,4.5\n')
            reuse_identical_latent_forecasts(source, destination, model, identity, 'M2L', 512, 2048, 900)
            self.assertTrue(reusable_forecasts(destination, model, identity, 'M2L', 512, 2048, 900))
            self.assertEqual(sha256(source / 'test_forecasts.npz'), sha256(destination / 'test_forecasts.npz'))
            self.assertEqual(json.loads((destination / 'test_latent_diagnostics.json').read_text()),
                             {'model': 'M2L', 'filtered_mean': .2})
            self.assertIn('M2L,3600,4.5', (destination / 'test_horizon_metrics.csv').read_text())
            self.assertEqual(json.loads((source / 'test_latent_diagnostics.json').read_text())['model'], 'ML')
            for changed, data, p in ((LatentModel((.5, .7, .001), model.modes), identity, 512),
                                     (model, {'toy': False}, 512), (model, identity, 128)):
                with self.assertRaisesRegex(ValueError, 'saved forecasts'):
                    reuse_identical_latent_forecasts(source, root / 'rejected', changed, data, 'M2L', p, 2048, 900)
            (source / 'test_latent_trace.npz').unlink()
            with self.assertRaisesRegex(ValueError, 'incomplete'):
                reuse_identical_latent_forecasts(source, root / 'incomplete', model, identity, 'M2L', 512, 2048, 900)

    def test_resume_rejects_foreign_forecasts_and_recomputes_incomplete_outputs(self):
        model = LatentModel((.4, .7, .001), (Mode(0, 0, .3, 2.),))
        identity = {'toy': True}
        settings = {'model': model.serialize(), 'corpus': identity, 'split': 'test',
                    'test_evaluated': True, 'stride_seconds': 900,
                    'filter_particles': 512, 'filter_seed': 71391, 'future_paths': 2048}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.assertFalse(reusable_forecasts(root, model, identity, 'ML', 512, 2048, 900))
            save_json(root / 'forecast-protocol.json', settings)
            self.assertFalse(reusable_forecasts(root, model, identity, 'ML', 512, 2048, 900))
            for name in ('test_forecasts.npz', 'test_one_step_metrics.json', 'test_horizon_metrics.csv', 'test_predictive_loss.npz',
                         'test_latent_diagnostics.json', 'test_latent_trace.npz', 'test_origin_latent_states.npz'):
                (root / name).touch()
            self.assertTrue(reusable_forecasts(root, model, identity, 'ML', 512, 2048, 900))
            for key, changed in [('model', LatentModel((.5, .7, .001), model.modes).serialize()),
                                 ('corpus', {'toy': False}), ('split', 'validation'),
                                 ('filter_particles', 128), ('filter_seed', 91337),
                                 ('future_paths', 512), ('stride_seconds', 300)]:
                with self.subTest(key=key):
                    save_json(root / 'forecast-protocol.json', dict(settings, **{key: changed}))
                    with self.assertRaisesRegex(ValueError, 'saved forecasts'):
                        reusable_forecasts(root, model, identity, 'ML', 512, 2048, 900)

    def fixture(self, directory):
        cache = directory / 'counts.npz'
        y = np.random.default_rng(123).poisson([2., 3.], (1000, 2))
        np.savez(cache, counts=y, valid=np.ones(1000, dtype=bool), identity=json.dumps({'toy': True}), audit='{}')
        model = Model((.4, .7), (Mode(0, 0, .3, 2.), Mode(1, 1, .4, 40.)))
        latent = LatentModel((.4, .7, .001), model.modes, beta=((), ()))
        models = {}
        for label in LABELS:
            value = model.serialize() if label == 'M0' else (Modulated(model, (), ((), ())).serialize() if label == 'M2' else latent.serialize())
            file, audit, csv = directory / f'{label}.json', directory / f'{label}-audit.json', directory / f'{label}.csv'
            save_json(file, {'model': value})
            csv.write_text('candidate,operation\ntoy,test_fixture\n')
            save_json(audit, {'ready_for_test': True, 'requirement_checks': dict.fromkeys(REQUIREMENTS, True),
                              'test_fixture': True, 'model_label': label, 'model_sha256': sha256(file),
                              'selection_csv_sha256': sha256(csv)})
            models[label] = {key: str(path.resolve()) for key, path in [('model', file), ('selection_audit', audit), ('selection_csv', csv)]}
            models[label].update({f'{key}_sha256': sha256(Path(models[label][key])) for key in ('model', 'selection_audit', 'selection_csv')})
        manifest = directory / 'freeze.json'
        save_json(manifest, {'ready_for_test': True, 'final_test_evaluated_in_this_pipeline_before_freeze': False,
                             'models': models, 'cache_sha256': sha256(cache), 'warmup_seconds': 1})
        return cache, manifest

    def test_unfinished_selection_is_rejected_before_test_data_loader(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            freeze = root / 'freeze.json'
            save_json(freeze, {'ready_for_test': False})
            with patch('hawkes_final_test._load_corpus', side_effect=AssertionError('Test data opened')):
                with self.assertRaisesRegex(ValueError, 'frozen'):
                    run(freeze, root / 'missing-cache.npz', root / 'results')

    def test_changed_parameters_or_unfinished_audit_cannot_reuse_freeze(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache, freeze = self.fixture(root)
            manifest, models = read_frozen_selection(freeze, cache)
            self.assertEqual(tuple(models), LABELS)
            audit = root / 'ML-audit.json'
            save_json(audit, {'ready_for_test': False})
            with self.assertRaisesRegex(ValueError, 'selection_audit changed'):
                read_frozen_selection(freeze, cache)
            manifest['models']['ML']['selection_audit_sha256'] = sha256(audit)
            save_json(freeze, manifest)
            with self.assertRaisesRegex(ValueError, 'unfinished'):
                read_frozen_selection(freeze, cache)

    def test_selection_arrays_exclude_test_and_means_remain_training_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            cache, _ = self.fixture(Path(temporary))
            selection = load_corpus(cache, 1)
            evaluation = _load_corpus(cache, 1, True)
            self.assertEqual(len(selection.y), 800)
            self.assertEqual(len(evaluation.y), 1000)
            np.testing.assert_array_equal(selection.initial_mean, evaluation.initial_mean)
            for array in (evaluation.y, evaluation.valid, evaluation.score, evaluation.initial_mean):
                self.assertFalse(array.flags.writeable)
            _, end, mask = evaluation_region(evaluation, 'test')
            self.assertEqual(end, 1000)
            self.assertFalse(mask[:800].any())
            self.assertTrue(mask[800:].all())
            with self.assertRaisesRegex(ValueError, 'unavailable'):
                evaluation_region(selection, 'test')
            origins = forecast_origins(evaluation, 'test', 20, 10)
            self.assertTrue((origins >= 800).all())
            self.assertTrue((origins + 10 <= 1000).all())

    def test_stale_review_is_rejected_even_when_its_own_manifest_hash_matches(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache, freeze = self.fixture(root)
            manifest = json.loads(freeze.read_text())
            path = root / 'ML-audit.json'
            original = json.loads(path.read_text())
            for field, stale in (('model_label', 'M2L'), ('model_sha256', '0' * 64),
                                 ('selection_csv_sha256', '0' * 64)):
                with self.subTest(field=field):
                    save_json(path, dict(original, **{field: stale}))
                    manifest['models']['ML']['selection_audit_sha256'] = sha256(path)
                    save_json(freeze, manifest)
                    with patch('hawkes_final_test._load_corpus', side_effect=AssertionError('Test data opened')):
                        with self.assertRaisesRegex(ValueError, 'audit does not match'):
                            run(freeze, cache, root / 'results')

    def test_test_open_loop_forecast_is_independent_of_future_test_counts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            y = np.random.default_rng(123).poisson(2., (20000, 2)).astype(float)
            valid = np.ones(len(y), dtype=bool)
            score = valid.copy()
            model = Model((.4, .7), (Mode(0, 0, .3, 2.), Mode(1, 1, .4, 40.)))
            mean = np.array([2., 2.])
            original = Corpus(y, valid, score, 12000, 16000, mean, {'toy': True})
            changed_y = y.copy()
            changed_y[16000:] = 1000.
            changed = Corpus(changed_y, valid, score, 12000, 16000, mean, {'toy': True})
            evaluate_forecasts(model, original, root / 'original', split='test')
            evaluate_forecasts(model, changed, root / 'changed', split='test')
            with np.load(root / 'original/test_forecasts.npz') as before, np.load(root / 'changed/test_forecasts.npz') as after:
                np.testing.assert_array_equal(before['origins'], [16000])
                np.testing.assert_array_equal(before['prediction'], after['prediction'])
                self.assertFalse(np.array_equal(before['actual'], after['actual']))
            with np.load(root / 'original/test_predictive_loss.npz') as trace:
                metric = json.loads((root / 'original/test_one_step_metrics.json').read_text())
                self.assertAlmostEqual(trace['marginal_nll'].mean(), metric['poisson_mean_log_score'], places=12)
                self.assertTrue((trace['seconds'] >= 16000).all())


if __name__ == '__main__':
    unittest.main()
