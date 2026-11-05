import unittest
import numpy as np
import torch
from numba import set_num_threads
from oracle_account_grid import AccountGrid, CellErrors
from oracle_book_mc import (rule_vector, move_arrive, close_terminal, execute_target,
    representative, lookup, backup_layer)
from gpu_oracle_bellman import Rules, TensorOracle


class DenseAccountTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        set_num_threads(2)
        torch.set_num_threads(2)

    def test_book_permutation_and_multiplicity(self):
        g=AccountGrid();a=g.arrays();self.assertEqual(len(a['books']),45)
        self.assertEqual(len(a['coordinates']),315)
        orders=((.2,.5),(.8,-.5))
        self.assertEqual(g.canonical_cell(.1,orders),g.canonical_cell(.1,orders[::-1]))
        self.assertNotEqual(g.canonical_cell(0,orders[:1]),g.canonical_cell(0,orders[:1]*2))
        rng=np.random.default_rng(12)
        for cell in range(len(a['coordinates'])):
            e,book=g.sample_cell(cell,rng)
            self.assertEqual(g.canonical_cell(e,book),cell)

    def test_single_book_matches_tensor_oracle(self):
        rules=Rules();r=rule_vector(rules,.99999);oracle=TensorOracle(rules)
        for exposure in [-2.,-.2,0.,.8,2.]:
            for side in [-1,0,1]:
                for price in [99.,100.,101.]:
                    s=np.array([10000*(1-exposure),100*exposure,100,100,side,100-side*.5,100.])
                    t=torch.tensor([[*s,0.,0.]],dtype=torch.float64)
                    expected,traffic,filled,valid=oracle.arrival(oracle.advance(t,torch.tensor([price]),.99999))
                    actual,c,f,v=move_arrive(s,price,r)
                    np.testing.assert_allclose(actual,expected.numpy()[0,:7],atol=1e-10)
                    self.assertEqual((c,f,v),(traffic.item(),filled.item(),valid.item()))
                    expected_close=oracle.terminal(oracle.advance(t,torch.tensor([price]),.99999)).item()
                    self.assertAlmostEqual(close_terminal(actual,c,r),np.log(expected_close) if expected_close>0 else -np.inf)

    def test_two_fills_share_cap_and_are_permutation_invariant(self):
        r=rule_vector(Rules(maximum=100.))
        s=np.array([1000.,0.,100.,100.,1.,99.5,60.,1.,99.,60.])
        x,c,filled,valid=move_arrive(s,98.,r)
        self.assertTrue(filled);self.assertFalse(valid);self.assertEqual(c,120.)
        reverse=np.r_[s[:4],s[7:10],s[4:7]]
        np.testing.assert_allclose(move_arrive(reverse,98.,r)[0],x)
        # Upward motion cannot fill a passive buy; both orders stay pending.
        np.testing.assert_allclose(move_arrive(s,101.,r)[0][4:],s[4:])

    def test_reserve_and_next_tick_activation(self):
        r=rule_vector(Rules());s=np.array([1000.,0.,100.,100.,0.,0.,0.,0.,0.,0.])
        book=np.array([[.5,1.5],[.5,-1.5]])
        placed,ok=execute_target(s,0.,False,0.,book,2,r,.0001)
        self.assertFalse(ok) # 1500 reserved against 1000 equity
        book[:,1]*=.2
        placed,ok=execute_target(s,0.,False,0.,book,2,r,.0001)
        self.assertTrue(ok);self.assertEqual(placed[1],0.)
        x,_,filled,ok=move_arrive(placed,99.9,r)
        self.assertTrue(ok and filled);self.assertEqual(x[4],0.);self.assertEqual(x[7],-1.)

    def test_liquidation_and_terminal_dust(self):
        r=rule_vector(Rules())
        s=np.array([-10000.,200.,100.,100.,1.,90.,100.])
        x,_,filled,ok=move_arrive(s,60.,r)
        self.assertFalse(ok or filled);self.assertTrue((x[4:]==0).all())
        s=np.array([1000.,.01,100.,100.,0.,0.,0.])
        self.assertEqual(close_terminal(s,0.,r),-np.inf)

    def test_lookup_retains_pending_book(self):
        g=AccountGrid();a=g.arrays();r=rule_vector(Rules())
        values=np.zeros((len(a['wealth']),len(a['coordinates']),1))
        values[:,len(a['exposures']):,:]=.03
        plain,_=representative(10000,0,a['books'][0],0,100,r,g.price_scale)
        book,_=representative(10000,0,a['books'][2],1,100,r,g.price_scale)
        args=(0,values,a['wealth'],a['exposures'],a['code_offsets'],a['book_lookup'],a['size_edges'],g.price_bins,g.price_scale)
        self.assertAlmostEqual(lookup(book,*args)-lookup(plain,*args),.03)

    def test_terminal_backup_averages_before_root_action(self):
        g=AccountGrid(max_orders=0,exposure_points=(-1.,0.,1.),wealth_points=(9000.,10000.,11000.))
        a=g.arrays();r=rule_vector(Rules(maker=0,taker=0,maximum=100000))
        v,var=backup_layer(np.array([100.]),np.array([[90.,110.]]),np.array([[0,1]]),np.array([[0,1]]),
            np.full(3,2),np.zeros((3,3,2)),a['wealth'],a['exposures'],a['books'],a['book_counts'],
            a['code_offsets'],a['book_lookup'],a['size_edges'],g.price_bins,g.price_scale,r,np.arange(3),True,np.zeros((2,2)))
        np.testing.assert_allclose(v[1,:,0],[.5*np.log(.99),0,.5*np.log(.99)],atol=1e-12)
        self.assertEqual(v[1,:,0].argmax(),1)
        self.assertGreater(var[1,0,0],0)

    def test_error_allocation_is_bounded_compute_not_probability(self):
        err=CellErrors.empty(20);err.observe(0,.2);err.observe(0,.4)
        self.assertAlmostEqual(err.mean[0],.3)
        n=err.allocate(np.zeros(20),np.zeros(20),8,4,9)
        self.assertEqual(n.sum(),80);self.assertTrue(((n>=2)&(n<=8)).all())


if __name__=='__main__':unittest.main()
