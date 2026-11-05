import unittest

import numpy as np
from scipy.special import expit
from scipy.stats import gamma

from continuous_oracle_endpoint import (PriceLaw, Rules, close_cost,
    continuous_decision, execute, log_hurdle_law, terminal_close)


class ContinuousEndpointTests(unittest.TestCase):
    def test_full_law_mass_mean_and_tick_probabilities(self):
        packed = np.array([-.8, .4, .01, .02, 1.7, 2.1])
        law = log_hurdle_law(packed, 100., .01)
        p0, pu = expit(packed[:2])
        raw_mean = 100*(p0+(1-p0)*(pu*(1-.01/1.7)**-1.7+(1-pu)*(1+.02/2.1)**-2.1))
        self.assertAlmostEqual(law.mass[-1]+sum(t.mass for t in law.tails), 1, places=12)
        self.assertLess(abs(law.mean-raw_mean), .0050001)
        i = np.searchsorted(law.prices, 100)
        expected = p0+(1-p0)*(pu*gamma.cdf(np.log(100.005/100), 1.7, scale=.01/1.7)
            +(1-pu)*gamma.cdf(-np.log(99.995/100), 2.1, scale=.02/2.1))
        self.assertAlmostEqual(law.mass[i+1]-law.mass[i], expected, places=10)

    def test_full_close_fees_and_rejection_of_partial_close(self):
        law = PriceLaw([90., 110.], [.4, .6])
        rules = Rules(fee=.01, minimum=5, maximum=100)
        for asset in [0, .1, .5]:
            lo, hi = law.cost_bounds(asset, rules)
            expected = sum(w*close_cost(asset, p, rules) for p,w in [(90,.4),(110,.6)])
            self.assertAlmostEqual(lo, expected, places=10)
            self.assertAlmostEqual(hi, expected, places=10)
        for asset in [.001, 2.]:
            self.assertEqual(law.value_bounds(100, asset, rules), (-np.inf, -np.inf))
            with self.assertRaisesRegex(ValueError, 'terminal-exposure'):
                terminal_close(100, asset, 100, rules)
        q, a, flow = terminal_close(100, .5, 100, rules)
        self.assertEqual((q, a, flow), (149.5, 0., 49.5))
        with self.assertRaisesRegex(ValueError, 'terminal-exposure'):
            terminal_close(100, np.nextafter(0., 1.), .01, Rules(minimum=0))

    def test_zero_trade_is_identity_and_fee_free_optimum_is_continuous_endpoint(self):
        self.assertEqual(execute(50., .5, 100., 0., Rules()), (50., .5))
        for price, expected in [(120., -50.), (80., 50.)]:
            result = continuous_decision(50., .5, 100., PriceLaw([price], [1]), Rules(fee=0))
            self.assertAlmostEqual(result['marketQuote'], expected)
            self.assertLessEqual(result['optimalityGap'], 1e-4)

    def test_dust_loophole_cannot_beat_cash(self):
        rules = Rules(minimum=5, maximum=100)
        result = continuous_decision(100., 0., 100., PriceLaw([100.075], [1]), rules, tolerance=1e-7)
        self.assertEqual(result['marketQuote'], 0.)
        self.assertEqual(result['asset'], 0.)

    def test_fractional_position_respects_worst_outcome_closing_cap(self):
        rules = Rules(fee=0., minimum=5, maximum=100)
        result = continuous_decision(100., 0., 100., PriceLaw([120., 200.], [.5, .5]), rules)
        self.assertAlmostEqual(result['asset'], .5)
        self.assertAlmostEqual(result['marketQuote'], -50.)
        self.assertAlmostEqual(result['valueLower'], 130., places=6)

    def test_global_bound_dominates_independent_dense_search(self):
        rng = np.random.default_rng(72)
        rules = Rules(fee=.003, minimum=5, maximum=80)
        for _ in range(12):
            prices = np.sort(rng.uniform(50, 180, 9))
            weights = rng.dirichlet(np.ones(9))
            law = PriceLaw(prices, weights)
            result = continuous_decision(70, .3, 100, law, rules, tolerance=1e-5)
            trades = np.r_[np.linspace(-70, -5, 5001), 0, np.linspace(5, .997*.3*100, 5001)]
            q = 70+trades
            a = .3-trades*np.where(trades < 0, .997, 1/.997)/100
            flow = .997*a[:, None]*prices
            valid = ((flow >= 5) & (flow <= 80)).all(axis=1) | (a == 0)
            expected = np.where(valid, q+flow@weights, -np.inf)
            self.assertGreaterEqual(result['globalUpper']+1e-8, float(expected.max()))
            self.assertLessEqual(result['optimalityGap'], 1.00001e-5)

    def test_same_mean_different_support_changes_feasibility(self):
        rules = Rules(fee=.01, minimum=5, maximum=100)
        spread, point = PriceLaw([50, 250], [.5, .5]), PriceLaw([150], [1])
        self.assertEqual(spread.mean, point.mean)
        self.assertTrue(point.can_close(.5, rules))
        self.assertFalse(spread.can_close(.5, rules))

    def test_law_tails_not_dropped_and_integration_refines(self):
        packed = np.array([-1., .2, .02, .025, .5, 1.2])
        a = log_hurdle_law(packed, 100., .01, tail_probability=1e-7)
        b = log_hurdle_law(packed, 100., .01, tail_probability=1e-12)
        self.assertGreater(sum(t.mass for t in a.tails), 0)
        for law in (a,b):
            self.assertEqual(law.support_upper, np.inf)
            result = continuous_decision(100., 0., 100., law)
            self.assertEqual(result['asset'], 0.)
            self.assertEqual(result['marketQuote'], 0.)
            self.assertEqual(law.value_bounds(30, .01, Rules()), (-np.inf, -np.inf))
        with self.assertRaisesRegex(ValueError, 'No feasible flat-terminal'):
            continuous_decision(100., .001, 100., a)

    def test_rejects_leveraged_endpoint_account_and_infinite_mean(self):
        law = PriceLaw([100], [1])
        with self.assertRaisesRegex(ValueError, 'path law'):
            continuous_decision(-1, 1, 100, law)
        with self.assertRaisesRegex(ValueError, 'finite price expectation'):
            log_hurdle_law([0, 0, 2, 1, 1, 1], 100)


if __name__ == '__main__':
    unittest.main()
