import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from oracle_dense_worker_checkpoint import save_worker, restore_worker
import test_serve_oracle_dense as fixtures
from test_serve_oracle_dense import initialize, observation


class CheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.WorkerTests.setUpClass()

    def worker(self, directory):
        w = fixtures.WorkerTests().worker()
        w.state_directory = directory
        w.model_identity = 'a'*64
        return w

    def test_restore_matches_uninterrupted_account_field_and_front(self):
        with tempfile.TemporaryDirectory() as directory:
            w = self.worker(directory)
            w.process(initialize())
            w.process(dict(id='one', kind='tick', observation=observation(1)))
            saved = w.process(dict(id='save', kind='checkpoint'))
            restored = self.worker(directory)
            reply = restored.process(dict(id='restore', kind='restore'))
            self.assertEqual(reply['update'], 1)
            self.assertEqual(reply['account'], saved['account'])
            self.assertEqual(reply['observation'], observation(1))
            for time in (2, 3):
                a = w.process(dict(id='a', kind='tick', observation=observation(time)))
                b = restored.process(dict(id='b', kind='tick', observation=observation(time)))
                self.assertEqual(a['account'], b['account'])
                self.assertEqual(a['terminalWealth'], b['terminalWealth'])
                np.testing.assert_array_equal(w.controller.current.values, restored.controller.current.values)
                np.testing.assert_array_equal(w.controller.front['scores'], restored.controller.front['scores'])

    def test_failed_atomic_commit_keeps_previous_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            w = self.worker(directory)
            w.process(initialize())
            first = save_worker(w, directory)
            w.process(dict(id='one', kind='tick', observation=observation(1)))
            with patch('oracle_dense_worker_checkpoint.os.replace', side_effect=OSError('disk failure')):
                with self.assertRaises(OSError):
                    save_worker(w, directory)
            fresh = self.worker(directory)
            resumed = restore_worker(fresh, directory)
            self.assertEqual(resumed['snapshot'], first['snapshot'])
            self.assertEqual(resumed['update'], 0)
            fresh.controller = None  # Release Windows mmap handles before removing this fixture.

    def test_corrupt_array_or_wrong_model_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            w = self.worker(directory)
            w.process(initialize())
            saved = save_worker(w, directory)
            other = self.worker(directory)
            other.model_identity = 'b'*64
            with self.assertRaisesRegex(ValueError, 'model'):
                restore_worker(other, directory)
            path = Path(directory)/saved['snapshot']/'map'/'values.npy'
            with path.open('r+b') as f:
                f.seek(-1, 2)
                f.write(b'X')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                restore_worker(self.worker(directory), directory)

    def test_changed_configuration_and_failed_boundary_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            from dataclasses import replace
            w = self.worker(directory)
            w.process(initialize())
            save_worker(w, directory)
            other = self.worker(directory)
            other.settings = replace(other.settings, seed=24)
            with self.assertRaisesRegex(ValueError, 'configuration'):
                restore_worker(other, directory)
            w.failed = True
            with self.assertRaisesRegex(ValueError, 'successful active'):
                save_worker(w, directory)
            with self.assertRaisesRegex(ValueError, 'fresh worker'):
                restore_worker(w, directory)

    def test_pointer_cannot_escape_state_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, 'current.json').write_text(json.dumps(dict(snapshot='../outside', manifestSha256='')))
            with self.assertRaisesRegex(ValueError, 'identifier'):
                restore_worker(self.worker(directory), directory)

    def test_rotation_keeps_current_previous_and_open_restored_map(self):
        with tempfile.TemporaryDirectory() as directory:
            w = self.worker(directory)
            w.process(initialize())
            first = save_worker(w, directory)
            fresh = self.worker(directory)
            restore_worker(fresh, directory)
            second = save_worker(fresh, directory)
            third = save_worker(fresh, directory)
            self.assertTrue(Path(directory, first['snapshot']).exists())
            self.assertTrue(Path(directory, second['snapshot']).exists())
            self.assertTrue(Path(directory, third['snapshot']).exists())
            fresh.process(dict(id='one', kind='tick', observation=observation(1)))
            fourth = save_worker(fresh, directory)
            self.assertFalse(Path(directory, first['snapshot']).exists())
            self.assertFalse(Path(directory, second['snapshot']).exists())
            self.assertTrue(Path(directory, third['snapshot']).exists())
            self.assertTrue(Path(directory, fourth['snapshot']).exists())


if __name__ == '__main__':
    unittest.main()
