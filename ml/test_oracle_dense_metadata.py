import unittest
from unittest.mock import patch
import numpy as np
from oracle_account_grid import AccountGrid, CellErrors
from oracle_dense_metadata import prepared
from test_oracle_dense_controller import solved_map


class MetadataTests(unittest.TestCase):
    def test_all_cell_draws_and_canonical_indices_identical(self):
        m = solved_map(2, 2)
        v = prepared(m)
        a, b = np.random.default_rng(21), np.random.default_rng(21)
        for cell in range(len(m.a['coordinates'])):
            for _ in range(3):
                old, new = m.grid.sample_cell(cell, a), v.grid.sample_cell(cell, b)
                self.assertEqual(old, new)
                self.assertEqual(m.grid.canonical_cell(*old), v.grid.canonical_cell(*new))
        self.assertIs(v.a, m.a)
        self.assertIs(v.values, m.values)
        self.assertIs(v.errors, m.errors)
        self.assertIsInstance(m.grid, AccountGrid)

    def test_front_and_error_statistics_identical_without_grid_rebuild(self):
        m = solved_map(3, 1)
        state = np.array([100., 0., 100., 100., 0., 0., 0.])
        old = m.front(state, seed=43)
        with patch.object(AccountGrid, 'arrays', side_effect=AssertionError('Unexpected metadata rebuild')):
            new = prepared(m).front(state, seed=43)
        for key in old:
            np.testing.assert_array_equal(old[key], new[key])
        previous = m.observe(old, 100., np.zeros((2, 59)), np.array([100., 100.]), np.ones(59))
        errors = m.errors
        m.errors = CellErrors.empty(len(m.a['coordinates']))
        with patch.object(AccountGrid, 'arrays', side_effect=AssertionError('Unexpected metadata rebuild')):
            current = prepared(m).observe(new, 100., np.zeros((2, 59)), np.array([100., 100.]), np.ones(59))
        self.assertEqual(previous, current)
        for key in ('count', 'mean', 'm2'):
            np.testing.assert_array_equal(getattr(errors, key), getattr(m.errors, key))


if __name__ == '__main__':
    unittest.main()
