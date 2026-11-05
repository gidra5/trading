import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from hawkes_final_test import LABELS, sha256
from hawkes_freeze_selection import freeze
from hawkes_research import HORIZONS, save_json
import test_hawkes_final_test as final_test_fixtures


class FreezePublicationTests(unittest.TestCase):
    def fixture(self, root):
        cache, old = final_test_fixtures.FrozenEvaluationTests().fixture(root)
        manifest = json.loads(old.read_text())
        plan = root / 'plan.json'
        save_json(plan, {'models': list(LABELS), 'evaluation_split': 'test',
                        'filter_particles': 512, 'filter_seed': 71391,
                        'future_paths': 2048, 'future_seed': 20261003,
                        'stride_seconds': 900, 'horizons_seconds': list(HORIZONS),
                        'warmup_seconds': 1, 'cache_sha256': sha256(cache),
                        'test_evaluated_in_this_pipeline': False,
                        'historical_test_inspection': 'Toy fixture'})
        request = root / 'request.json'
        save_json(request, {'cache': cache.name, 'evaluation_plan': plan.name,
                            'models': {label: {key: Path(manifest['models'][label][key]).name
                                              for key in ('model', 'selection_audit', 'selection_csv')}
                                       for label in LABELS}})
        return request

    def test_valid_freeze_is_verified_idempotent_and_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            request = self.fixture(root)
            output = root / 'final' / 'freeze.json'
            with patch('hawkes_final_test._load_corpus', side_effect=AssertionError('Test data opened')):
                result = freeze(request, output)
                original = output.read_bytes()
                self.assertEqual(set(result['models']), set(LABELS))
                self.assertEqual(freeze(request, output), result)
                self.assertEqual(output.read_bytes(), original)
                changed = json.loads((root / 'plan.json').read_text())
                changed['historical_test_inspection'] = 'Changed disclosure'
                save_json(root / 'plan.json', changed)
                with self.assertRaisesRegex(ValueError, 'Existing freeze differs'):
                    freeze(request, output)
                self.assertEqual(output.read_bytes(), original)
            self.assertFalse(list(output.parent.glob('*.pending')))

    def test_unreviewed_selection_cannot_publish_a_freeze(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            request = self.fixture(root)
            audit = root / 'ML-audit.json'
            value = json.loads(audit.read_text())
            value['requirement_checks']['optimization_stopping_review'] = False
            save_json(audit, value)
            output = root / 'final' / 'freeze.json'
            with self.assertRaisesRegex(ValueError, 'unfinished'):
                freeze(request, output)
            self.assertFalse(output.exists())
            self.assertFalse(list(output.parent.glob('*.pending')))


if __name__ == '__main__':
    unittest.main()
