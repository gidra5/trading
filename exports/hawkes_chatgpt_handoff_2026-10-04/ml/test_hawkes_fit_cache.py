from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import tempfile
import time
import unittest
from hawkes_fit_cache import configuration_key, shared_joint_fit


class SharedFitTests(unittest.TestCase):
    def test_identical_concurrent_requests_run_one_producer(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = {'iterations': 2, 'particles': 8, 'seed': [1, 2]}
            calls = []

            def request(name):
                directory = root / name

                def produce():
                    calls.append(name)
                    time.sleep(.05)
                    directory.mkdir()
                    (directory / 'joint-progress.json').write_text(json.dumps({'config': config, 'next_iteration': 2}))
                    (directory / 'best-validation.json').write_text('{}')
                    (directory / 'fit-result.json').write_text('{"model": "same joint fit"}')

                shared_joint_fit(directory, config, root / 'cache', produce)

            with ThreadPoolExecutor(max_workers=2) as executor:
                list(executor.map(request, ('ML', 'M2L')))
            self.assertEqual(len(calls), 1)
            self.assertEqual((root / 'ML' / 'fit-result.json').read_bytes(), (root / 'M2L' / 'fit-result.json').read_bytes())

    def test_different_settings_have_distinct_cache_keys(self):
        base = {'iterations': 2, 'particles': 8, 'seed_model': {'background': [1, 2]}, 'corpus': {'hash': 'a'}}
        for key, value in [('particles', 16), ('iterations', 3), ('corpus', {'hash': 'b'}), ('seed_model', {'background': [1, 3]})]:
            changed = dict(base, **{key: value})
            self.assertNotEqual(configuration_key(base), configuration_key(changed))

    def test_exception_releases_lock_for_a_later_producer(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = {'iterations': 2}
            with self.assertRaises(RuntimeError):
                shared_joint_fit(root / 'failed', config, root / 'cache', lambda: (_ for _ in ()).throw(RuntimeError('controlled failure')))
            directory = root / 'retry'

            def produce():
                directory.mkdir()
                (directory / 'joint-progress.json').write_text(json.dumps({'config': config, 'next_iteration': 2}))
                (directory / 'best-validation.json').write_text('{}')
                (directory / 'fit-result.json').write_text('{}')

            shared_joint_fit(directory, config, root / 'cache', produce)
            self.assertTrue((directory / 'fit-result.json').exists())


if __name__ == '__main__':
    unittest.main()
