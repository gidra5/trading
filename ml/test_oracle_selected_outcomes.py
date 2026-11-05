import unittest
import numpy as np
import torch
from gpu_oracle_bellman import Rules
from oracle_book_mc import move_arrive, close_terminal, decide, valid_post
from oracle_selected_outcomes import SelectedOutcomes
from test_oracle_dense_controller import solved_map


@unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
class SelectedOutcomeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from numba import set_num_threads
        set_num_threads(2); torch.set_num_threads(2)

    def test_cpu_agreement_for_terminal_and_nonterminal_with_orders(self):
        m = solved_map(3,2,Rules(minimum=1.,maximum=100.),funding=.999)
        a,g = m.a,m.grid
        states = np.array([[100.,0.,100.,100.,0.,0.,0.,0.,0.,0.],
            [150.,-.5,100.,100.,1.,99.99,10.,-1.,100.01,10.],
            [-90.,1.9,100.,100.,0.,0.,0.,0.,0.,0.]])
        prices = np.tile([90.,100.,110.,1000.],(len(states),1))
        children = np.zeros(prices.shape,dtype=np.int64)
        factors = np.array([[.99,1.,1.01,1.]])
        candidates = np.arange(len(a['coordinates']))
        for remaining in (1,2,3):
            with self.subTest(remaining=remaining):
                child = m.values[1] if remaining>2 else np.full_like(m.values[1],-np.inf)
                evaluator = SelectedOutcomes(m,states,prices,children,child,factors,candidates,remaining)
                indices = np.tile([3,1,0,2],(len(states),1))
                counts = np.array([4,2,3])
                actual = evaluator.evaluate(indices,counts)
                for row in range(len(states)):
                    supported = valid_post(states[row],m.rules,False,False)
                    for price in prices[row]:
                        arrived,used,filled,okay = move_arrive(states[row],price,m.rules)
                        supported = supported and okay and np.isfinite(close_terminal(arrived,used,m.rules))
                    self.assertEqual(bool(evaluator.supported[row]),bool(supported))
                    for j in range(4):
                        if j>=counts[row]:
                            self.assertTrue(np.isnan(actual[row,j])); continue
                        arrived,used,filled,okay = move_arrive(states[row],prices[row,indices[row,j]],m.rules)
                        expected = -np.inf
                        if okay and valid_post(states[row],m.rules,False,False):
                            if remaining==1: expected=close_terminal(arrived,used,m.rules)
                            else:
                                expected=decide(arrived,used,filled,0,child,a['wealth'],a['exposures'],
                                    a['books'],a['book_counts'],a['code_offsets'],a['book_lookup'],a['size_edges'],
                                    g.price_bins,g.price_scale,m.rules,candidates,factors)[0]
                        self.assertEqual(np.isfinite(actual[row,j]),np.isfinite(expected))
                        if np.isfinite(expected): self.assertAlmostEqual(actual[row,j],expected,places=11)


if __name__ == '__main__': unittest.main()
