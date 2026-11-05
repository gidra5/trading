import csv
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from hawkes_latent import LatentModel
from hawkes_latent_ranking import paired_interval, shortlist


class RankingTests(unittest.TestCase):
    def test_paired_constant_difference_respects_score_and_gaps(self):
        difference = np.full(50, 1.25)
        difference[10:20] = np.nan
        mask = np.ones(50, dtype=bool)
        mask[10:20] = False
        self.assertEqual(paired_interval(difference, mask, width=7), [1.25, 1.25])
        mask[15] = True
        self.assertIsNone(paired_interval(difference, mask, width=7))

    def test_shortlist_retains_equal_budget_and_initial_controls(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = [dict(candidate='addition', operation='add_mode', selection_score=-3.),
                    dict(candidate='initial', operation='initial_0', selection_score=-1.),
                    dict(candidate='control', operation='joint_refit_control_additions', selection_score=-2.)]
            with (root / 'model_selection.csv').open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            model = LatentModel((1., 1., .1), ())
            for row in rows:
                path = root / 'candidates' / row['candidate'] / 'fit-result.json'
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps({'model': model.serialize()}))
            cohort, _ = shortlist(root, 1)
            self.assertEqual([r['candidate'] for r in cohort], ['addition', 'initial', 'control'])
            self.assertTrue(all(r['joint_parameters'] == 3 for r in cohort))
            self.assertTrue(all(len(r['source_sha256']) == 64 for r in cohort))


if __name__ == '__main__':
    unittest.main()
