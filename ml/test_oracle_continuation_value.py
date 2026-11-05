import unittest
import numpy as np
import torch
from oracle_continuation_value import ContinuationValue, bellman_backup, market_coordinates, timeline_inputs
from gpu_oracle_bellman import Rules, Menu, equity


class ContinuationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.value = ContinuationValue(torch.zeros(59), torch.ones(59), Rules(maker=0, taker=0, maximum=1000, minimum=0, lower=0, upper=1))
        self.state = torch.tensor([[100, 0, 100, 100, 0, 0, 0, 0, 0]], dtype=torch.float64)
        self.features = torch.zeros(1, 2, 59)
        self.prices = torch.tensor([[100, 100]], dtype=torch.float64)
        self.times = torch.tensor([[1700000000, 1700000001]], dtype=torch.float64)

    def children(self, prices):
        f = self.features[:, None].expand(-1, len(prices), -1, -1)
        p = torch.stack((torch.full((len(prices),), 100.), torch.tensor(prices)), -1).double()[None]
        t = (self.times+1)[:, None].expand(-1, len(prices), -1)
        return f, p, t

    def test_exact_terminal_and_nonanticipative_log_choice(self):
        decision = bellman_backup(self.value, self.state, *self.children([200., 50.]), torch.tensor([1]), Menu(step=.01, limits=False))
        self.assertAlmostEqual(decision['descriptor'][0, 0].item(), .5)
        self.assertAlmostEqual(decision['value'].item(), .5*__import__('math').log(1.125)-__import__('math').log(100), places=10)
        self.assertLess(decision['value'].item(), .5*(__import__('math').log(.02)+__import__('math').log(.01)))

    def test_terminal_ignores_network_and_requires_legal_close(self):
        v = ContinuationValue(torch.zeros(59), torch.ones(59))
        state = self.state.clone(); state[:, 1] = .001
        result = v(self.features, self.prices, self.times, state, torch.tensor([0]))
        self.assertTrue(torch.isneginf(result).all())
        state[:, 1] = 0
        before = v(self.features, self.prices, self.times, state, torch.tensor([0]))
        with torch.no_grad(): v.head[-1].bias.fill_(10000)
        after = v(self.features, self.prices, self.times, state, torch.tensor([0]))
        torch.testing.assert_close(before, after)

    def test_account_size_orders_and_horizon_remain_inputs(self):
        s = self.state.repeat(3, 1)
        s[1, 0] *= 2; s[2, 4:8] = torch.tensor([1, 99, 10, 0])
        coordinates = self.value.account_coordinates(s, torch.tensor([1, 1, 3600]))
        self.assertFalse(torch.equal(coordinates[0], coordinates[1]))
        self.assertFalse(torch.equal(coordinates[0], coordinates[2]))
        a = market_coordinates(self.features, self.prices, self.times, torch.zeros(59), torch.ones(59))
        b = market_coordinates(self.features, self.prices, self.times+3600, torch.zeros(59), torch.ones(59))
        self.assertFalse(torch.equal(a, b))

    def test_ruin_is_not_clipped_to_finite_log(self):
        dead = self.state.clone(); dead[:, :2] = 0
        score = self.value(self.features, self.prices, self.times, dead, torch.tensor([3600]))
        self.assertTrue(torch.isneginf(score).all())

    def test_action_chunking_preserves_the_decision(self):
        child = self.children([99., 101.])
        a = bellman_backup(self.value, self.state, *child, torch.tensor([10]), Menu(step=.01), chunk=16)
        b = bellman_backup(self.value, self.state, *child, torch.tensor([10]), Menu(step=.01), chunk=65536)
        torch.testing.assert_close(a['value'], b['value'], atol=1e-10, rtol=0)
        torch.testing.assert_close(a['state'], b['state'])

    def test_rejects_untrained_horizon(self):
        with self.assertRaisesRegex(ValueError, 'horizon'):
            self.value(self.features, self.prices, self.times, self.state, torch.tensor([3601]))

    def test_physical_seconds_do_not_inherit_the_archived_origin_stride(self):
        data = dict(origins=np.array([10, 14]), indices=np.array([0, 4]), timestamps=np.array([100000, 104000]),
            timeline=np.arange(20*59).reshape(20, 59), pricesWithPast=np.arange(99, 105))
        f, p, t = timeline_inputs(data, np.array([0, 1, 2]), 'cpu')
        self.assertEqual(f[1, -1, 0].item(), data['timeline'][11, 0])
        self.assertEqual(p[1, -1].item(), 101)
        self.assertEqual(t[1, -1].item(), 101)
        self.assertEqual(t[2, -1].item()-t[1, -1].item(), 1)

    def test_other_forecast_rows_cannot_change_an_earlier_backup(self):
        state = self.state.repeat(2, 1)
        f, p, t = [x.repeat(2, *([1]*(x.ndim-1))) for x in self.children([99., 101.])]
        a = bellman_backup(self.value, state, f, p, t, torch.tensor([10, 10]), Menu(step=.1))
        altered = f.clone(); altered[1] += 100
        prices = p.clone(); prices[1] *= 2
        b = bellman_backup(self.value, state, altered, prices, t, torch.tensor([10, 10]), Menu(step=.1))
        torch.testing.assert_close(a['scores'][0], b['scores'][0])
        single = bellman_backup(self.value, state[:1], f[:1], p[:1], t[:1], torch.tensor([10]), Menu(step=.1))
        torch.testing.assert_close(a['scores'][0], single['scores'][0])


if __name__ == '__main__': unittest.main()
