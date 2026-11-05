import json
import unittest
from dataclasses import replace
from unittest.mock import patch
import numpy as np
from numba import set_num_threads
from gpu_oracle_bellman import Rules
from oracle_account_grid import AccountGrid
from nsf_scenario_graph import build_graph
from serve_oracle_dense import DenseWorker, WorkerSettings, process_line, public_diagnostics
from test_oracle_dense_controller import Model, flat_prediction


def flat_graph(*args, **kwargs):
    with patch('nsf_scenario_graph.next_observations', flat_prediction):
        return build_graph(*args, **kwargs)


def observation(t):
    return dict(features=[[0.]*59, [0.]*59], prices=[100., 100.], timestamps=[t-1., t])


def initialize():
    return dict(id='start', kind='initialize', observation=observation(0),
        account=dict(quote=100., asset=0., previousPrice=100., orders=[]),
        terminalSeconds=3., consumedQuote=0., filled=False)


class WorkerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        set_num_threads(2)

    def worker(self):
        s = WorkerSettings(AccountGrid(max_orders=1, price_bins=1, size_bins=2,
            exposure_points=(-1., 0., 1.), wealth_points=(50., 100., 200.)), Rules(),
            horizon=2, width=1, branches=2, candidates=6, seed=23, backend='cpu')
        return DenseWorker(Model(), s, graph_builder=flat_graph, predict=flat_prediction)

    def call(self, worker, request):
        result = process_line(worker, json.dumps(request))
        json.dumps(result, allow_nan=False)
        return result

    def test_complete_serial_episode_returns_executed_account(self):
        worker = self.worker()
        result = self.call(worker, initialize())
        self.assertEqual(result['phase'], 'active')
        self.assertEqual(result['account']['quote'], 100.)
        self.assertEqual(result['update'], 0)
        self.assertIsNone(result['terminalWealth'])
        for t in range(1, 4):
            result = self.call(worker, dict(id=str(t), kind='tick', observation=observation(t)))
            self.assertNotIn('error', result)
            self.assertEqual(result['update'], t)
        self.assertEqual(result['phase'], 'complete')
        self.assertEqual(result['terminalWealth'], 100.)
        self.assertEqual(result['account']['asset'], 0.)
        self.assertEqual(result['account']['orders'], [])

    def test_full_horizon_repair_diagnostics_fit_transport_without_changing_plan_record(self):
        updates = [dict(depth=j,rows=10000,outcomes=80000,partialRows=0,complete=False,
            supportRepair=True,promoted=100) for j in range(3600)]
        original = dict(table=dict(updates=updates,updatedLayers=3600,updatedRows=36000000,
            evaluatedOutcomes=288000000))
        public = public_diagnostics(original)
        self.assertLess(len(json.dumps(public)),65536)
        self.assertEqual(public['table']['updatedLayers'],3600)
        self.assertEqual(public['table']['omittedUpdateRecords'],3568)
        self.assertIs(public['table']['updates'][0],updates[0])
        self.assertIs(public['table']['updates'][-1],updates[-1])
        self.assertEqual(len(original['table']['updates']),3600)
        small = dict(table=dict(updates=updates[:4]))
        self.assertIs(public_diagnostics(small),small)

    def test_invalid_or_duplicate_request_never_reexecutes(self):
        w = self.worker()
        self.assertIn('error', self.call(w, dict(id='early', kind='tick', observation=observation(1))))
        self.assertNotIn('error', self.call(w, initialize()))
        self.assertIn('error', self.call(w, initialize()))
        self.assertIn('error', self.call(w, dict(id='skip', kind='tick', observation=observation(2))))
        self.assertEqual(w.controller.update, 0)
        self.assertNotIn('error', self.call(w, dict(id='one', kind='tick', observation=observation(1))))
        self.assertIn('error', self.call(w, dict(id='duplicate', kind='tick', observation=observation(1))))
        self.assertEqual(w.controller.update, 1)

    def test_input_rejects_future_fields_wrong_shape_and_nonfinite(self):
        w = self.worker()
        request = initialize()
        request['futurePrices'] = [100., 101.]
        self.assertIn('error', self.call(w, request))
        request = initialize()
        request['observation']['features'][0] = [0.]*58
        self.assertIn('error', self.call(w, request))
        self.assertIn('error', process_line(w, '{"id":"nan","kind":"tick","observation":NaN}'))
        self.assertIn('error', process_line(w, '{'))
        self.assertIsNone(w.controller)
        self.assertFalse(w.failed)

    def test_failed_forecast_stops_later_ticks(self):
        w = self.worker()
        self.call(w, initialize())
        with patch('oracle_dense_controller.roll_market_cloud', side_effect=RuntimeError('broken forecast')):
            failed = self.call(w, dict(id='one', kind='tick', observation=observation(1)))
        self.assertTrue(failed['error']['failed'])
        second = self.call(w, dict(id='two', kind='tick', observation=observation(2)))
        self.assertTrue(second['error']['failed'])
        self.assertEqual(w.controller.update, 1)

    def test_respects_initial_committed_fill_budget(self):
        w = self.worker()
        request = initialize()
        request['filled'] = True
        request['consumedQuote'] = w.settings.rules.maximum
        request['account'].update(quote=90., asset=.1)
        result = self.call(w, request)
        self.assertNotIn('error', result)
        # No remaining quote capacity at this decision, even though flat is preferred.
        self.assertEqual(result['account']['asset'], .1)
        self.assertEqual(result['account']['quote'], 90.)

    def test_rejects_settings_that_omit_exposures_or_order_candidates(self):
        w = self.worker()
        for kwargs in (dict(candidates=3), dict(branches=3), dict(horizon=1), dict(width=False)):
            with self.assertRaises(ValueError):
                replace(w.settings, **kwargs)


if __name__ == '__main__':
    unittest.main()
