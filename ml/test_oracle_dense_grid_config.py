import unittest
import json
from dataclasses import asdict
from oracle_account_grid import AccountGrid
from oracle_dense_grid_config import exposure_axis,future_candidate_count,replay_grid


class GridConfigTests(unittest.TestCase):
    def test_refinement_retains_original_targets_and_zero(self):
        for step,count in ((.25,21),(.125,41),(.01,501)):
            axis=exposure_axis(step)
            self.assertEqual(len(axis),count)
            self.assertIn(0.,axis)
            for x in AccountGrid.exposure_points:self.assertAlmostEqual(min(abs(x-y) for y in axis),0.)
        self.assertEqual(exposure_axis(),AccountGrid.exposure_points)

    def test_rejects_invalid_or_non_nested_axis(self):
        for step in (0.,-.1,float('nan'),float('inf'),.3,.3125,1e-8,1e-300):
            with self.assertRaises(ValueError):exposure_axis(step)

    def test_fine_grid_does_not_silently_drop_all_new_orders(self):
        grid=AccountGrid(exposure_points=exposure_axis(.125))
        self.assertEqual(future_candidate_count(grid),50)
        with self.assertRaises(ValueError):future_candidate_count(grid,16)
        self.assertEqual(future_candidate_count(AccountGrid()),16)

    def test_replay_refinement_and_resume_preserve_effective_grid(self):
        source=dict(grid=asdict(AccountGrid()),arguments=dict(candidates=16));before=json.dumps(source)
        grid,count=replay_grid(source)
        self.assertEqual(grid,AccountGrid());self.assertEqual(count,16)
        fine,count=replay_grid(source,exposure_step=.125)
        self.assertEqual(len(fine.exposure_points),41);self.assertEqual(count,50)
        parent=dict(grid=asdict(fine),arguments=dict(candidates=count))
        restored,budget=replay_grid(source,parent)
        self.assertEqual(restored,fine);self.assertEqual(budget,50)
        self.assertEqual(json.dumps(source),before)
        with self.assertRaises(ValueError):replay_grid(source,exposure_step=.125,candidates=16)

    def test_candidate_count_must_be_an_integer(self):
        for count in (1.5,float('nan'),float('inf'),True):
            with self.assertRaises(ValueError):future_candidate_count(AccountGrid(),count)


if __name__=='__main__':unittest.main()
