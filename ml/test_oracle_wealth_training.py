import unittest
import numpy as np
import torch
from oracle_wealth_training import TrainingRules, rollout_torch, rollout_numpy


class OracleWealthTrainingTests(unittest.TestCase):
    def test_matches_independent_oracle_execution(self):
        rng = np.random.default_rng(78)
        static = rng.normal(0, .3, 32)
        prices = np.round(100*np.exp(np.r_[0., np.cumsum(rng.normal(0, .01, 32))]), 2)
        weights = np.array([.2, -.1, .3])
        a = rollout_numpy(static, weights, prices, trace=True)
        b = rollout_torch(torch.tensor(static[None]), torch.tensor(weights), torch.tensor(prices[None]), trace=True)
        np.testing.assert_allclose(a['trace'][:, 0], b['trades'].numpy()[0], atol=1e-8)
        self.assertAlmostEqual(a['wealth'], float(b['wealth'][0]), places=8)

    def test_no_intermediate_forced_close(self):
        r = rollout_numpy(np.zeros(20), np.zeros(3), np.full(21, 100.), trace=True)
        self.assertEqual(np.count_nonzero(r['trace'][:, 0]), 1)
        self.assertEqual(r['trades'], 2)
        self.assertAlmostEqual(r['wealth'], 10000-r['fees'], places=9)

    def test_below_minimum_can_remain_completely_inactive(self):
        r = rollout_torch(torch.full((1, 5), -12., dtype=torch.float64), torch.zeros(3).double(),
                          torch.full((1, 6), 100., dtype=torch.float64), trace=True)
        self.assertEqual(float(r['wealth'][0]), 10000.)
        self.assertEqual(float(abs(r['trades']).sum()), 0.)

    def test_future_price_does_not_change_first_trade(self):
        static = np.zeros(5)
        left = rollout_numpy(static, np.array([.2, .3, .4]), np.array([100., 99., 98., 99., 100., 101.]), trace=True)
        right = rollout_numpy(static, np.array([.2, .3, .4]), np.array([100., 120., 80., 90., 130., 100.]), trace=True)
        np.testing.assert_array_equal(left['trace'][0], right['trace'][0])

    def test_realized_wealth_gradient(self):
        static = np.array([-.4, .2, -.2, .5, -.1])
        prices = np.array([100., 101., 99., 103., 102., 104.])
        w = np.array([.2, -.1, .3])
        x = torch.tensor(static[None], requires_grad=True)
        wealth = rollout_torch(x, torch.tensor(w), torch.tensor(prices[None]))['wealth'].sum()
        wealth.backward()
        numerical = []
        for i in range(len(static)):
            up, down = static.copy(), static.copy()
            up[i] += 1e-5; down[i] -= 1e-5
            numerical.append((rollout_numpy(up, w, prices)['wealth']-rollout_numpy(down, w, prices)['wealth'])/2e-5)
        np.testing.assert_allclose(x.grad.numpy()[0], numerical, atol=1e-6, rtol=1e-6)

    def test_close_cap_preserves_residual(self):
        rules = TrainingRules(maximum=1000.)
        r = rollout_numpy(np.full(8, 3.), np.zeros(3), np.r_[np.full(8, 100.), 110.], rules, trace=True)
        self.assertLessEqual(np.max(abs(r['trace'][:, 0])), 1000.)
        self.assertEqual(r['close'], 1000.)
        self.assertGreater(r['asset'], 0.)

    def test_wealth_gradient_rewards_profitable_exposure(self):
        for endpoint, sign in [(110., 1), (90., -1)]:
            logit = torch.tensor(-3., dtype=torch.float64, requires_grad=True)
            prices = torch.tensor([[100., 100., 100., endpoint]], dtype=torch.float64)
            wealth = rollout_torch(logit.expand(1, 3), torch.zeros(3).double(), prices)['wealth'].sum()
            wealth.backward()
            self.assertGreater(sign*float(logit.grad), 0)
            self.assertGreater(sign*(float(wealth.detach())-10000), 0)


if __name__ == '__main__':
    unittest.main()
