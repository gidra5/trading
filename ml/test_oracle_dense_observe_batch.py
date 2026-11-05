import unittest
import numpy as np
from numba import get_num_threads, set_num_threads
from gpu_oracle_bellman import Rules
from oracle_account_grid import CellErrors
from oracle_dense_metadata import prepared
from oracle_dense_observe_batch import observe_batched
from oracle_dense_value_map import DenseValueMap
from test_oracle_dense_controller import solved_map


class BatchedObservationTests(unittest.TestCase):
    def setUp(self):
        self.threads = get_num_threads()
        set_num_threads(2)

    def tearDown(self):
        set_num_threads(self.threads)

    def compare(self, m, front, price):
        args = (front, price, np.zeros((2, 59)), np.array([100., price]), np.ones(59))
        snapshots = []
        for threads in (0, 1, 2):
            m.errors = CellErrors.empty(len(m.a['coordinates']))
            if threads:
                set_num_threads(threads)
                result = observe_batched(prepared(m), *args)
            else:
                result = DenseValueMap.observe(prepared(m), *args)
            snapshots.append((result, m.errors))
        for result, errors in snapshots[1:]:
            self.assertEqual(result, snapshots[0][0])
            for key in ('count', 'mean', 'm2'):
                np.testing.assert_array_equal(getattr(errors, key), getattr(snapshots[0][1], key))

    def test_final_and_nonfinal_arrivals_fills_and_funding(self):
        for horizon in (2, 3):
            m = solved_map(horizon, 2, Rules(minimum=1., maximum=100.), funding=.999)
            for state in (np.array([100., 0., 100., 100., 0., 0., 0., 0., 0., 0.]),
                          np.array([150., -.5, 100., 100., 1., 99.99, 10., -1., 100.01, 10.])):
                front = prepared(m).front(state, seed=31)
                for price in (90., 100., 110., 1000.):
                    with self.subTest(horizon=horizon, price=price, asset=state[1]):
                        self.compare(m, front, price)

    def test_no_finite_proposals_and_final_map(self):
        m = solved_map(2)
        front = dict(scores=np.full(len(m.a['coordinates']), -np.inf),
                     states=np.zeros((len(m.a['coordinates']), 4)))
        self.compare(m, front, 100.)
        m = solved_map(1)
        self.compare(m, front, 100.)


if __name__ == '__main__':
    unittest.main()
