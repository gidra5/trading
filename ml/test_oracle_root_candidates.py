import unittest
import numpy as np
from numba import set_num_threads
from oracle_dense_metadata import prepared
from oracle_root_candidates import root_candidates
from test_oracle_dense_controller import solved_map


class CandidateTests(unittest.TestCase):
    def test_original_front_actions_are_covered_without_value_filtering(self):
        set_num_threads(2)
        m = solved_map(3,2)
        state = np.array([100.,0.,100.,100.,1.,99.99,5.,-1.,100.01,5.])
        front = prepared(m).front(state,seed=9)
        ids,states,slots = root_candidates(m,state,seed=9)
        self.assertEqual(len(np.unique(ids)),len(ids))
        self.assertTrue(np.all((ids>=0)&(ids<slots)))
        for expected in [front['state'],*front['states'][np.isfinite(front['scores'])]]:
            self.assertTrue(any(np.array_equal(expected,x) for x in states))
        before = states.copy()
        _,changed,_ = root_candidates(m,state,consumed=10000.,filled=True,seed=9)
        np.testing.assert_array_equal(before,states)
        self.assertTrue(np.all(changed[:,0]==state[0]))


if __name__ == '__main__': unittest.main()
