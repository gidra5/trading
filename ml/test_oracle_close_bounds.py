import unittest
import numpy as np
from oracle_close_bounds import close_interval
from oracle_book_mc import move_arrive,close_terminal


RULES=np.array([.0002,.0005,2.,-2.,2.,-3.,3.,5.,10000.,.01,1.])


def exact(state,factors,rules):
    values=[]
    for factor in factors:
        price=max(1.,np.rint(state[2]*factor/rules[9]))*rules[9]
        arrived,used,_,valid=move_arrive(state,price,rules)
        if not valid:return -np.inf
        values.append(close_terminal(arrived,used,rules))
    return float(np.mean(values))


class CloseBoundTests(unittest.TestCase):
    def test_bounds_enclose_long_short_cash_with_funding(self):
        rng=np.random.default_rng(93);accepted_count=0
        for funding in (1.,.9999):
            rules=RULES.copy();rules[-1]=funding
            for exposure in (-.8,-.2,0.,.2,.8):
                state=np.array([10000*(1-exposure),10000*exposure/100.,100.,100.])
                for spread in (1e-6,1e-4,.01,.1):
                    factors=np.exp(rng.normal(0,spread,64))
                    accepted,lower,upper=close_interval(state,factors,rules)
                    actual=exact(state,factors,rules)
                    if accepted:
                        accepted_count+=1
                        self.assertTrue(np.isfinite(actual))
                        self.assertLessEqual(lower,actual);self.assertGreaterEqual(upper,actual)
                        self.assertLessEqual(upper-lower,1e-9)
        self.assertGreater(accepted_count,10)

    def test_possible_fill_declines_but_distant_orders_are_supported(self):
        factors=np.array([.9999,1.0001])
        state=np.array([10000.,0.,100.,100.,1.,99.99,1000.])
        self.assertFalse(close_interval(state,factors,RULES)[0])
        state[5]=90.
        accepted,lower,upper=close_interval(state,factors,RULES)
        self.assertTrue(accepted);self.assertLessEqual(lower,exact(state,factors,RULES));self.assertGreaterEqual(upper,np.log(10000.))

    def test_rejects_liquidation_uncloseable_dust_and_excess_close_flow(self):
        for state,factors in ((np.array([-15000.,250.,100.,100.]),np.array([.5,1.])),
            (np.array([9999.,.01,100.,100.]),np.array([1.,1.])),
            (np.array([-5000.,150.,100.,100.]),np.array([1.,1.]))):
            self.assertFalse(close_interval(state,factors,RULES)[0])

    def test_wide_interval_falls_back_instead_of_exceeding_tolerance(self):
        state=np.array([5000.,50.,100.,100.]);factors=np.array([.8,1.2])
        accepted,lower,upper=close_interval(state,factors,RULES,1e-9)
        self.assertFalse(accepted);self.assertGreater(upper-lower,1e-9)
        self.assertLessEqual(lower,exact(state,factors,RULES));self.assertGreaterEqual(upper,exact(state,factors,RULES))

    def test_random_carried_orders_and_execution_parameters(self):
        rng=np.random.default_rng(713);accepted_count=0
        for _ in range(1000):
            rules=RULES.copy();rules[0]=rng.uniform(0,.001);rules[1]=rng.uniform(rules[0],.003)
            rules[-1]=rng.uniform(.99,1.01);rules[7]=rng.uniform(0,10);rules[8]=rng.uniform(1000,20000)
            price=float(rng.choice([.1,100.,65000.]));wealth=rng.uniform(1000,20000);exposure=rng.uniform(-2.5,2.5)
            state=np.r_[wealth*(1-exposure),wealth*exposure/price,price,price,np.zeros(6)]
            for j in range(2):
                side=float(rng.choice([-1,0,1]));distance=int(rng.integers(1,20))
                limit=price-side*distance*rules[9]
                if side!=0 and limit>0:
                    state[4+3*j:7+3*j]=[side,limit,rng.uniform(10,1000)]
            factors=np.exp(rng.normal(0,float(rng.choice([1e-5,1e-3,.05])),8))
            accepted,lower,upper=close_interval(state,factors,rules)
            if accepted:
                accepted_count+=1;actual=exact(state,factors,rules)
                self.assertTrue(np.isfinite(actual));self.assertLessEqual(lower,actual);self.assertGreaterEqual(upper,actual)
        self.assertGreater(accepted_count,20)


if __name__=='__main__':unittest.main()
