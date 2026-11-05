from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import json
import unittest
from hawkes_gpu_search import FitCoordinator
from hawkes_latent import LatentModel
from hawkes_latent_minibatch import fit_configuration
from hawkes_research import Corpus
import numpy as np


class CoordinatorTests(unittest.TestCase):
    def corpus(self):
        return Corpus(np.ones((10, 2)), np.ones(10, bool), np.ones(10, bool), 6, 10, np.ones(2), {'validation_scored_seconds': 4})

    def test_identical_jobs_are_batched_once_and_aliases_get_results(self):
        calls = []
        def fake(jobs, corpus, particles, iterations, workers):
            calls.append(jobs)
            for seed, output in jobs:
                output.mkdir(parents=True, exist_ok=True)
                config = fit_configuration(seed, corpus, particles, iterations, checkpoint_every=iterations, checkpoint_particles=max(32, 2 * particles))
                for name, value in [('joint-progress.json', {'config': config, 'next_iteration': iterations}),
                                    ('fit-result.json', {'model': seed.serialize()}), ('best-validation.json', {}), ('accelerator-resume.json', {})]:
                    (output / name).write_text(json.dumps(value))
        with TemporaryDirectory() as root, patch('hawkes_gpu_search.fit_joint_batch', side_effect=fake):
            paths = [Path(root) / p for p in ('a', 'b', 'c')]
            models = [LatentModel((1., 1., .1), ()), LatentModel((2., 1., .1), ())]
            coordinator = FitCoordinator(self.corpus(), Path(root) / 'cache', iterations=2)
            with ThreadPoolExecutor(2) as pool:
                futures = [pool.submit(coordinator.submit_many, [(models[0], paths[0]), (models[1], paths[1])]),
                           pool.submit(coordinator.submit_many, [(models[0], paths[2])])]
                for future in futures:
                    future.result(timeout=10)
            coordinator.close()
            self.assertEqual(sum(len(jobs) for jobs in calls), 2)
            self.assertEqual((paths[0] / 'fit-result.json').read_bytes(), (paths[2] / 'fit-result.json').read_bytes())

    def test_worker_failure_reaches_all_waiters(self):
        with TemporaryDirectory() as root, patch('hawkes_gpu_search.fit_joint_batch', side_effect=ValueError('failed kernel')):
            coordinator = FitCoordinator(self.corpus(), Path(root) / 'cache', iterations=2)
            with self.assertRaisesRegex(ValueError, 'failed kernel'):
                coordinator.submit_many([(LatentModel((1., 1., .1), ()), Path(root) / 'a')])
            with self.assertRaisesRegex(RuntimeError, 'worker failed'):
                coordinator.submit_many([(LatentModel((2., 1., .1), ()), Path(root) / 'b')])
            coordinator.close()


if __name__ == '__main__':
    unittest.main()
