import unittest
import numpy as np
import torch
from gpu_oracle_bellman import Rules
from oracle_account_grid import AccountGrid
from oracle_book_mc import rule_vector
from oracle_dense_value_map import DenseValueMap
from oracle_incremental_table import IncrementalTable
from oracle_incremental_observation import observe
from oracle_local_market_refresh import candidate_schedule
from oracle_selected_outcomes import SelectedOutcomes
from test_oracle_book_gpu import market


@unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
class ObservedInnovationTests(unittest.TestCase):
    def test_finite_law_innovations_average_to_zero_and_do_not_modify_values(self):
        from numba import set_num_threads
        set_num_threads(2); torch.set_num_threads(2)
        grid = AccountGrid(max_orders=2,price_bins=2,size_bins=2,
            exposure_points=(-1.,0.,1.),wealth_points=(50.,100.,200.))
        cloud = market([[100.,100.],[99.,101.],[100.,100.],[100.,100.]])
        m = DenseValueMap(grid,cloud,rule_vector(Rules(),.999),None); m.sweep(7,6,4,backend='cuda')
        table = IncrementalTable(m,7,6)
        states = np.array([[100.,0.,100.,100.,0.,0.,0.,0.,0.,0.],
            [100.,0.,100.,100.,0.,0.,0.,0.,0.,0.],
            [150.,-.5,100.,100.,1.,99.99,10.,0.,0.,0.],
            [50.,.5,100.,100.,0.,0.,0.,0.,0.,0.]])
        prices = np.repeat(cloud.sampled_prices[0,:1],len(states),axis=0)
        children = np.repeat(cloud.destinations[0,:1],len(states),axis=0)
        evaluator = SelectedOutcomes(table.facade(),states,prices,children,table.field(1),cloud.factors(1),
            candidate_schedule(table.facade(),0,7,6),table.H)
        outcomes = evaluator.evaluate(np.tile(np.arange(table.B),(len(states),1)),np.full(len(states),table.B))
        front = dict(states=states,scores=outcomes.mean(1)); before = table.cpu_values()
        scores = front['scores'].copy(); means = []
        for edge in range(table.B):
            node = cloud.destinations[0,0,edge]
            result = observe(table,front,prices[0,edge],cloud.features[1,node],cloud.price_pairs[1,node],np.ones(59))
            means.append(result['meanResidual'])
            self.assertEqual(result['observations'],len(states))
            self.assertLess(result['distinctAccountCells'],len(states))
            self.assertEqual(result['terminalTimestamp'],3.)
        self.assertAlmostEqual(float(np.mean(means)),0.,places=12)
        self.assertEqual(int(table.errors.count.sum()),table.B*result['distinctAccountCells'])
        self.assertLessEqual(int(table.errors.count.max()),table.B)
        np.testing.assert_array_equal(table.cpu_values(),before)
        np.testing.assert_array_equal(front['scores'],scores)
        self.assertIsNone(table.root_controls)


if __name__=='__main__': unittest.main()
