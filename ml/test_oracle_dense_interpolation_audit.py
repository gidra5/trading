import unittest
import numpy as np
import torch
from audit_oracle_dense_interpolation import market, targets, census, terminal_score
from oracle_dense_value_map import DenseValueMap
from oracle_account_grid import AccountGrid
from oracle_book_mc import rule_vector
from gpu_oracle_bellman import Rules


class ExactAccountCensusTests(unittest.TestCase):
    def test_known_gain_with_terminal_close(self):
        grid=AccountGrid(max_orders=0,exposure_points=(0.,1.))
        a=grid.arrays();rules=rule_vector(Rules(maker=0.,taker=0.,maximum=100000.))
        cloud=market([100.,100.],[[110.,110.],[110.,110.]])
        # Starting from cash, future full investment earns exactly 10%.
        score=census(np.array([[1000.,0.,100.,100.]]),cloud,a,rules,grid.price_scale)
        self.assertAlmostEqual(score[0],np.log(1100.),places=12)

    def test_future_decision_cannot_see_its_next_outcome(self):
        grid=AccountGrid(max_orders=0,exposure_points=(-1.,0.,1.))
        a=grid.arrays();rules=rule_vector(Rules(maker=0.,taker=0.,maximum=100000.))
        cloud=market([100.,100.],[[90.,110.],[90.,110.]])
        score=census(np.array([[1000.,0.,100.,100.]]),cloud,a,rules,grid.price_scale)
        self.assertAlmostEqual(score[0],np.log(1000.),places=12)

    def test_committed_fill_consumes_terminal_capacity(self):
        rules=rule_vector(Rules(maker=0.,taker=0.,maximum=50.))
        post=np.array([1000.,0.,100.,100.,1.,99.,50.])
        # Buying consumes the entire per-tick cap, so closing cannot occur.
        self.assertTrue(np.isneginf(terminal_score(post,np.array([.98]),rules)))

    def test_targets_retain_exact_existing_order_or_cancel(self):
        grid=AccountGrid(max_orders=1,exposure_points=(-1.,0.,1.));a=grid.arrays()
        rules=rule_vector(Rules());state=np.array([1000.,0.,100.,100.,1.,99.5,70.])
        result=targets(state,0.,False,a['exposures'],a['books'],a['book_counts'],rules,grid.price_scale)
        self.assertTrue(any(np.array_equal(s,state) for s in result))
        self.assertTrue(any(np.array_equal(s,np.array([1000.,0.,100.,100.,0.,0.,0.])) for s in result))

    def test_penultimate_backup_uses_exact_terminal_actions(self):
        grid=AccountGrid(max_orders=1,wealth_points=(5000.,10000.,20000.));a=grid.arrays()
        rules=rule_vector(Rules())
        cloud=market([99.97,101.],[[99.83,99.83],[101.49,99.98]])
        state=np.array([10000.,0.,100.,100.,1.,99.99,5000.])
        expected=census(np.array([state]),cloud,a,rules,grid.price_scale)[0]-np.log(10000.)
        dense=DenseValueMap(grid,cloud,rules,None)
        dense.sweep(17,len(a['coordinates']),2,backend='cpu')
        # Book index 3 is the one-buy representative at z=.25, size=.5;
        # exposure index 3 is cash. No root/account interpolation is involved.
        self.assertAlmostEqual(dense.values[0,1,24,0],expected,places=12)
        if torch.cuda.is_available():
            gpu=DenseValueMap(grid,cloud,rules,None)
            gpu.sweep(17,len(a['coordinates']),2,backend='cuda')
            self.assertAlmostEqual(gpu.values[0,1,24,0],expected,places=12)


if __name__=='__main__':unittest.main()
