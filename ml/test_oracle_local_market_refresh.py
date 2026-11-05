import unittest
import numpy as np
from numba import set_num_threads
from gpu_oracle_bellman import Rules
from oracle_local_market_refresh import local_fields, root_decision
from test_oracle_dense_controller import Model, solved_map, flat_prediction


class LocalMarketRefreshTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        set_num_threads(2)

    def test_identical_histories_and_laws_give_identical_fields(self):
        m = solved_map(5, 0, Rules(maker=0., taker=0.))
        fields, factors, diagnostic = local_fields(m, Model(), np.zeros((2, 59)), np.array([100., 100.]),
            np.array([-1., 0.]), 0, 23, 3, 71, 2, predict=flat_prediction)
        for field in fields.values():
            np.testing.assert_array_equal(field, fields['archived'])
        self.assertEqual(diagnostic['coupledSampleRMSEBps'], 0.)
        np.testing.assert_array_equal(factors, [[1., 1.]])

    def test_omitted_predictive_feature_changes_continuation_with_same_random_draws(self):
        m = solved_map(5, 0, Rules(maker=0., taker=0.))
        calls = []
        def predict(model, features, prices, times, horizon, branches, seed):
            calls.append(seed)
            f = np.repeat(features[:, -1, None, :]+1., branches, axis=1)
            p = np.repeat(prices[:, -1, None]*(1.+.001*features[:, -1, :1]), branches, axis=1)
            return f, p
        fields, factors, diagnostic = local_fields(m, Model(), np.zeros((2, 59)), np.array([100., 100.]),
            np.array([-1., 0.]), 0, 23, 3, 71, 2, predict=predict)
        state = np.array([100., 0., 100., 100.])
        retained = root_decision(m, state, 0., False, fields['retained_resampled'], factors)
        fresh = root_decision(m, state, 0., False, fields['fresh_history'], factors)
        self.assertEqual(retained[1][1], 0.)
        # The first price move is flat with zero fees, so waiting to enter at
        # the next decision ties entering now. Its continuation still improves.
        self.assertEqual(fresh[1][1], 0.)
        self.assertGreater(fresh[0], retained[0])
        self.assertEqual(calls, [71, 72, 72])
        self.assertGreater(diagnostic['meanAbsoluteConditionalMeanErrorBps'], 0.)

    def test_retained_relative_moves_reanchor_without_price_level_jump(self):
        m = solved_map(5, 0, Rules(maker=0., taker=0.))
        def predict(model, features, prices, times, horizon, branches, seed):
            return (np.repeat(features[:, -1, None, :], branches, axis=1),
                    np.repeat(prices[:, -1, None]*1.01, branches, axis=1))
        # The observed price is half the old cloud's price. Identical relative
        # transition laws should still agree after reanchoring.
        fields, _, diagnostic = local_fields(m, Model(), np.zeros((2, 59)), np.array([50., 50.]),
            np.array([-1., 0.]), 0, 23, 3, 71, 2, predict=predict)
        np.testing.assert_array_equal(fields['fresh_history'], fields['retained_resampled'])
        self.assertEqual(diagnostic['coupledSampleRMSEBps'], 0.)


if __name__ == '__main__':
    unittest.main()
