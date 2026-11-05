import unittest
from fractions import Fraction
import numpy as np
import torch
from numba import set_num_threads
from oracle_certified_value import ValuePlanes,SCALE,terminal_planes,close_planes,supporting_backup,plane_hulls,conditional_support
from gpu_oracle_bellman import Bellman,Menu,Rules
from oracle_order_value import order_backup,rational_order_backup
from oracle_plane_policy import closing_action,arrival_repairable
from gpu_oracle_bellman import TensorOracle


class CertifiedValueTests(unittest.TestCase):
    def setUp(self):
        set_num_threads(2);torch.set_num_threads(2)

    def test_integer_closure_covers_both_endpoints_and_fee_wedge(self):
        rng=np.random.default_rng(432)
        cn=rng.integers(SCALE//2,2*SCALE,(30,11),dtype=np.int64)
        dn=cn+rng.integers(-SCALE//20,SCALE//20,cn.shape,dtype=np.int64)
        den=np.full(cn.shape,3,dtype=np.int64)
        for buy,sell in [(5000,5000),(2000,5000),(5000,2000)]:
            got=close_planes(cn,dn,den,3,1,buy,sell)
            for index in np.ndindex(cn.shape):
                C,D=int(got.c[index]),int(got.d[index])
                self.assertLessEqual((buy-1)*D,buy*C)
                self.assertLessEqual((sell-1)*C,sell*D)
                for x in [Fraction(-3),Fraction(-5,7),Fraction(0),Fraction(13,8),Fraction(3)]:
                    want=Fraction(int(cn[index]),3)*(1-x)+Fraction(int(dn[index]),3)*x
                    self.assertLessEqual(want,C*(1-x)+D*x)

    def test_hull_support_matches_exhaustive_faces(self):
        rng=np.random.default_rng(434)
        c=rng.integers(SCALE,SCALE+100000,(9,67),dtype=np.int64)
        d=c+rng.integers(-10000,10000,c.shape,dtype=np.int64)
        pc=np.arange(10000,10009,dtype=np.int64);pp=np.array([10000,10003],dtype=np.int64)
        ds=np.array([[0,2,4,5],[1,3,7,8]],dtype=np.int32);x=np.linspace(-3,3,65)
        cn,dn=conditional_support(c,d,pc,pp,ds,x,*plane_hulls(c,d))
        for n in range(2):
            for j,e in enumerate(x):
                want=0.
                for b in range(4):
                    k=ds[n,b];want+=np.min(c[k]*(1-e)+d[k]*(pc[k]/pp[n])*e)
                got=(cn[n,j]*(1-e)+dn[n,j]*e)/pp[n]
                self.assertAlmostEqual(got,want,delta=3e-6)

    def test_one_step_continuous_upper_covers_finite_oracle(self):
        for prices in [[99.9,100.1],[99.,99.5],[100.4,101.]]:
            rules=Rules()
            solver=Bellman([torch.tensor([prices],dtype=torch.float64)],
                [torch.tensor([[.5,.5]],dtype=torch.float64)],[Menu(step=.01)],rules=rules)
            state=torch.tensor([[10000.,0,100,100,0,0,0,0,0]],dtype=torch.float64)
            exact=solver.solve(state)['value'].item()
            upper=supporting_backup(terminal_planes(2),np.rint(np.array(prices)*100).astype(np.int64),
                np.array([10000],np.int64),np.array([[0,1]],np.int32),points=65,leverage=2,fee_denominator=2000)
            self.assertLessEqual(exact,10000*upper.c.min()/SCALE+1e-8)

    def test_two_step_upper_covers_feedback_and_limit_placements(self):
        prices=[torch.tensor([[99.95,100.05]],dtype=torch.float64),
            torch.tensor([[99.,99.5],[100.5,101.]],dtype=torch.float64)]
        weights=[torch.full((1,2),.5,dtype=torch.float64),torch.full((2,2),.5,dtype=torch.float64)]
        solved=Bellman(prices,weights,[Menu(step=.25)]*2).solve(torch.tensor([[10000.,0,100,100,0,0,0,0,0]],dtype=torch.float64))
        child=supporting_backup(terminal_planes(4),np.array([9900,9950,10050,10100],np.int64),
            np.array([9995,10005],np.int64),np.array([[0,1],[2,3]],np.int32),points=65)
        root=supporting_backup(child,np.array([9995,10005],np.int64),np.array([10000],np.int64),np.array([[0,1]],np.int32),
            points=65,leverage=2,fee_denominator=2000)
        self.assertLessEqual(solved['value'].item(),10000*root.c.min()/SCALE+1e-8)

        tight=rational_order_backup(child,np.array([9995,10005],np.int64),np.array([10000],np.int64),
            np.array([[0,1]],np.int32),17,2,1)
        self.assertLessEqual(solved['value'].item(),10000*tight.c.min()/SCALE+1e-8)

    def test_unsupported_large_jump_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'guard'):
            supporting_backup(terminal_planes(2),np.array([1000,20000],np.int64),
                np.array([10000],np.int64),np.array([[0,1]],np.int32))

    def test_overflow_is_rejected_not_silently_certified(self):
        with self.assertRaisesRegex(OverflowError,'overflow'):
            supporting_backup(ValuePlanes(np.full((2,2),10**16,dtype=np.int64),np.full((2,2),10**16,dtype=np.int64)),
                np.array([9999,10001],np.int64),np.array([10000],np.int64),np.array([[0,1]],np.int32))

    def test_flat_market_cash_gap_is_only_conservative_rounding(self):
        value=terminal_planes(1)
        for _ in range(64):
            value=supporting_backup(value,np.array([10000]),np.array([10000]),np.array([[0,0]]),points=17)
        gap=value.c.min()/SCALE-1
        self.assertGreaterEqual(gap,0)
        self.assertLess(gap,1e-6)

    def test_predictable_order_upper_covers_actual_feedback(self):
        prices=[torch.tensor([[99.95,100.05]],dtype=torch.float64),
            torch.tensor([[99.5,99.6],[100.4,100.5]],dtype=torch.float64)]
        weights=[torch.full((1,2),.5,dtype=torch.float64),torch.full((2,2),.5,dtype=torch.float64)]
        menus=[Menu(step=.5,limit_fractions=(.25,1.),limit_offsets=(1,4,16),joint=True)]*2
        solved=Bellman(prices,weights,menus).solve(torch.tensor([[10000.,0,100,100,0,0,0,0,0]],dtype=torch.float64))
        child=order_backup(terminal_planes(4,fee_denominator=2000),np.array([9950,9960,10040,10050],np.int64),
            np.array([9995,10005],np.int64),np.array([[0,1],[2,3]],np.int32),points=17)
        root=order_backup(child,np.array([9995,10005],np.int64),np.array([10000],np.int64),np.array([[0,1]],np.int32),points=17,leverage=2)
        self.assertLessEqual(solved['value'].item(),10000*root.c.min()/SCALE+1e-8)

    def test_dust_exit_uses_two_valid_trades(self):
        oracle=TensorOracle()
        for quantity in [1e-5,-1e-5,.4,-.4]:
            state=torch.tensor([[10000.-quantity*60000,quantity,60000,60000,0,0,0,0,0]],dtype=torch.float64)
            for _ in range(8):
                state,amount,valid=closing_action(oracle,state)
                self.assertTrue(valid.item());self.assertTrue(oracle.size(amount).item())
                state=oracle.advance(state,60000.)
            self.assertEqual(state[0,1].item(),0)
            self.assertTrue(torch.isfinite(oracle.terminal(state)).item())

    def test_full_cap_fill_cannot_require_an_unavailable_repair(self):
        oracle=TensorOracle()
        # Buy consumes the complete per-tick capacity, and its fee pushes
        # the already fully invested account just past entry leverage two.
        s=torch.tensor([[0.,100.,99.99,100.,1.,99.99,10000.,0.,1.]],dtype=torch.float64)
        arrived,_,_,okay=oracle.arrival(s)
        valid=arrival_repairable(oracle,s,arrived,okay,oracle.descriptors(Menu(step=2.,limits=False),'cpu'))
        self.assertFalse(valid.item())
        _,_,closing_valid=closing_action(oracle,s)
        self.assertFalse(closing_valid.item())


if __name__=='__main__':unittest.main()
