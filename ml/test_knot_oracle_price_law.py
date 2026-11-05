import unittest
import numpy as np
from scipy.special import ndtr
from continuous_oracle_endpoint import Rules, continuous_decision
from knot_oracle_price_law import knot_cdf, knot_price_law, mixture_law, AnalyticGammaPriceLaw


class KnotOracleLawTests(unittest.TestCase):
    def setUp(self):
        self.knots=np.array([0.,.5,1.]);self.areas=np.array([.25,.5,.25])

    def law(self,p0=.3,sigma=.01,radius=8.):
        return knot_price_law(p0,self.areas,self.knots,self.areas,sigma,100.,radius=radius)

    def test_uniform_probability_coordinate_is_normal(self):
        u=np.linspace(0,1,101)
        np.testing.assert_allclose(knot_cdf(self.knots,np.ones(3),u),u,atol=1e-15)
        law=self.law();p=law.prices
        expected=.7*np.diff(ndtr(np.log(np.r_[p-.005,p[-1]+.005]/100)/.01))
        expected[np.argmin(abs(p-100))]+=.3
        np.testing.assert_allclose(np.diff(law.mass),expected,atol=3e-14)
        self.assertAlmostEqual(law.mass[-1]+sum(t.mass for t in law.tails),1,places=12)
        self.assertLess(abs(law.mean-100*(.3+.7*np.exp(.01**2/2))),.005+law.mean_error)

    def test_mixture_preserves_cost_and_moment_bounds(self):
        a,b=self.law(.1),self.law(.8,.02)
        m=mixture_law([a,b],[.4,.6]);rules=Rules()
        np.testing.assert_allclose(m.mean,.4*a.mean+.6*b.mean,atol=1e-10)
        self.assertFalse(m.can_close(.1,rules))
        for asset in [0.]:
            np.testing.assert_allclose(m.cost_bounds(asset,rules),.4*np.array(a.cost_bounds(asset,rules))+.6*np.array(b.cost_bounds(asset,rules)),atol=1e-10)

    def test_tail_refinement_and_continuous_solver(self):
        a,b=self.law(radius=7.),self.law(radius=8.)
        alo,ahi=a.value_bounds(0,100,Rules());blo,bhi=b.value_bounds(0,100,Rules())
        self.assertLessEqual(max(alo,blo),min(ahi,bhi)+1e-8)
        self.assertEqual(b.support_upper, np.inf)  # even when CDF-subtracted tail mass rounds to zero
        decision=continuous_decision(10000.,0.,100.,b,Rules())
        self.assertEqual(decision['marketQuote'],0.)
        self.assertLessEqual(decision['optimalityGap'],1e-4)

    def test_tiny_endpoint_hat_is_not_a_normal_tail(self):
        knots=np.array([0.,1e-35,.5,1.])
        areas=np.r_[np.diff(knots)[0]/2,(knots[2:]-knots[:-2])/2,np.diff(knots)[-1]/2]
        law=knot_price_law(.3,np.array([.001,.249,.5,.25]),knots,areas,.0001,100.)
        self.assertLess(law.mean_error,1e-7)
        self.assertLess(sum(t.mass for t in law.tails),1e-10)
        self.assertGreater(law.mean,99.)
        self.assertLess(law.mean,101.)

    def test_analytic_gamma_bounds_contain_dense_price_grid(self):
        from continuous_oracle_endpoint import log_hurdle_law
        packed=np.array([.2,-.3,.0001,.0002,.05,.1])
        fast=AnalyticGammaPriceLaw(packed,100.)
        dense=log_hurdle_law(packed,100.)
        self.assertLessEqual(abs(fast.mean-dense.mean),fast.mean_error+dense.mean_error+1e-10)
        self.assertEqual(fast.support_upper, np.inf)
        self.assertFalse(fast.can_close(.01, Rules()))
        for asset in [0.]:
            lo,hi=fast.cost_bounds(asset,Rules())
            dl,dh=dense.cost_bounds(asset,Rules())
            self.assertLessEqual(lo,dh+1e-10)
            self.assertGreaterEqual(hi,dl-1e-10)


if __name__=='__main__':unittest.main()
