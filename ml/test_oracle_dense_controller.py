import unittest
from dataclasses import replace
from unittest.mock import patch
import numpy as np
import torch
from numba import set_num_threads
from gpu_oracle_bellman import Rules
from oracle_account_grid import AccountGrid
from oracle_book_mc import rule_vector
from oracle_dense_controller import DenseController, DenseControllerConfig
from oracle_dense_value_map import DenseValueMap, MarketCloud


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))
        self.register_buffer('feature_mean', torch.zeros(59))
        self.register_buffer('feature_std', torch.ones(59))


def flat_prediction(model, features, prices, times, horizon, branches, seed):
    return (np.repeat(features[:, -1, None, :], branches, axis=1),
            np.repeat(prices[:, -1, None], branches, axis=1))


def solved_map(horizon=3, orders=0, rules=Rules(), funding=1.):
    grid = AccountGrid(max_orders=orders, price_bins=1, size_bins=2,
                       exposure_points=(-1., 0., 1.), wealth_points=(50., 100., 200.))
    p = np.full((horizon+1, 1), 100.)
    cloud = MarketCloud(p, np.ones(horizon+1, dtype=np.int64),
        np.zeros((horizon, 1, 2), dtype=np.int64), np.ones((horizon+1, 1)),
        np.zeros((horizon+1, 1, 2, 59), dtype=np.float32),
        np.full((horizon+1, 1, 2), 100.), 0., np.full((horizon, 1, 2), 100.))
    current = DenseValueMap(grid, cloud, rule_vector(rules, funding), None)
    current.sweep(23, 6, 2, backend='cpu')
    return current


class ControllerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        set_num_threads(2)
        torch.set_num_threads(2)

    def controller(self, terminal=5):
        return DenseController(solved_map(), Model(), np.array([100., 0., 100., 100.]),
            DenseControllerConfig(terminal, 23, 6, backend='cpu'), predict=flat_prediction)

    def step(self, c, price=100.):
        time = c.current.cloud.origin_seconds
        return c.advance(np.zeros((2, 59)), np.array([c.state[2], price]), np.array([time, time+1]))

    def terminal_controller(self, state, rules=Rules(), funding=1.):
        current = solved_map(1, (len(state)-4)//3, rules, funding)
        front = dict(value=0., state=np.asarray(state, dtype=float))
        return DenseController(current, Model(), front['state'],
            DenseControllerConfig(1., 23, 6, backend='cpu'), front=front,
            episode_origin=-1., maximum_horizon=2, predict=flat_prediction)

    def test_rolling_then_shrinking_horizon_and_exact_flat_endpoint(self):
        c = self.controller()
        rows = [self.step(c) for _ in range(5)]
        self.assertEqual([r.get('horizon') for r in rows], [3, 3, 2, 1, None])
        self.assertEqual(c.phase, 'complete')
        self.assertEqual(c.terminal_wealth, 100.)
        self.assertTrue(all(not r['marketTrade'] for r in rows))
        with self.assertRaisesRegex(RuntimeError, 'complete'):
            self.step(c)

    def test_rejected_observation_does_not_advance_or_charge_account(self):
        c = self.controller()
        previous = c.state.copy()
        for f, p, t in ((np.zeros((2, 59)), [100., 100.], [0., 2.]),
                        (np.zeros((2, 59)), [101., 100.], [0., 1.]),
                        (np.ones((2, 59)), [100., 100.], [0., 1.]),
                        (np.zeros((2, 59)), [100., 100.001], [0., 1.])):
            with self.assertRaises(ValueError):
                c.advance(f, p, t)
            self.assertEqual(c.update, 0)
            self.assertEqual(c.phase, 'active')
            np.testing.assert_array_equal(c.state, previous)
        self.step(c)
        self.assertEqual(c.update, 1)

    def test_restore_preserves_global_seeds_state_and_values(self):
        whole, split = self.controller(), self.controller()
        for _ in range(2):
            self.step(whole)
            self.step(split)
        resumed = DenseController(split.current, split.model, split.state, split.config,
            front=split.front, episode_origin=0., maximum_horizon=3, predict=flat_prediction)
        for price in (100.01, 99.99, 100.02):
            a, b = self.step(whole, price), self.step(resumed, price)
            for key in ('state', 'marketTrade', 'filled', 'estimatedLogValue', 'terminalWealth'):
                self.assertEqual(a.get(key), b.get(key))
            np.testing.assert_array_equal(whole.current.values, resumed.current.values)

    def test_pending_fill_then_terminal_close_share_quote_cap(self):
        state = [100., 0., 100., 100., 1., 99., 10.]
        c = self.terminal_controller(state, Rules(maker=0., taker=0.))
        row = self.step(c, 98.)
        self.assertTrue(row['filled'])
        self.assertAlmostEqual(c.terminal_wealth, 90.+10.*98./99.)
        np.testing.assert_array_equal(c.state[4:], [0., 0., 0.])
        capped = self.terminal_controller(state, Rules(maker=0., taker=0., maximum=15.))
        with self.assertRaisesRegex(RuntimeError, 'terminal full close'):
            self.step(capped, 98.)
        self.assertGreater(capped.state[1], 0.)
        self.assertIsNone(capped.terminal_wealth)
        self.assertEqual(capped.phase, 'failed')

    def test_terminal_uses_funded_inventory_and_does_not_erase_dust(self):
        c = self.terminal_controller([90., .1, 100., 100.], Rules(maker=0., taker=0.), .99)
        self.step(c)
        self.assertAlmostEqual(c.terminal_wealth, 99.9)
        dust = self.terminal_controller([99., .01, 100., 100.])
        with self.assertRaisesRegex(RuntimeError, 'terminal full close'):
            self.step(dust)
        self.assertEqual(dust.state[1], .01)

    def test_planning_failure_cannot_reapply_arrival(self):
        c = self.controller()
        with patch('oracle_dense_controller.roll_market_cloud', side_effect=RuntimeError('forecast failed')):
            with self.assertRaisesRegex(RuntimeError, 'forecast failed'):
                self.step(c)
        self.assertEqual(c.phase, 'failed')
        self.assertEqual(c.update, 1)
        with self.assertRaisesRegex(RuntimeError, 'failed'):
            self.step(c)

    def test_rejects_invalid_episode_configuration(self):
        with self.assertRaises(ValueError):
            replace(DenseControllerConfig(5., 23, 6), repair_interval=0)
        with self.assertRaisesRegex(ValueError, 'timestamps'):
            self.controller(2)


if __name__ == '__main__':
    unittest.main()
