import unittest
import tempfile
from dataclasses import replace
from unittest.mock import patch
import numpy as np
import torch
from gpu_oracle_bellman import Rules
from oracle_account_grid import AccountGrid
from oracle_book_mc import rule_vector, move_arrive, close_terminal, decide
from oracle_root_candidates import root_candidates
from oracle_local_market_refresh import candidate_schedule
from oracle_dense_value_map import DenseValueMap
from oracle_incremental_table import IncrementalTable, TableBudget
from oracle_incremental_checkpoint import save_table,load_table
from test_oracle_book_gpu import market


@unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
class IncrementalTableTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from numba import set_num_threads
        set_num_threads(2); torch.set_num_threads(2)

    def initial(self):
        g = AccountGrid(max_orders=2,price_bins=1,size_bins=2,
            exposure_points=(-1.,0.,1.),wealth_points=(100.,1000.))
        c = market([[100.,100.]]*7)
        m = DenseValueMap(g,c,rule_vector(Rules(maximum=1000.)),None)
        m.sweep(7,6,4,backend='cuda')
        return m

    def shift(self,c,tail=100.):
        prices = np.r_[c.prices[1:],np.full_like(c.prices[-1:],tail)]
        pairs = np.r_[c.price_pairs[1:],np.stack((c.prices[-1],prices[-1]),1)[None]]
        raw = np.r_[c.sampled_prices[1:],np.full_like(c.sampled_prices[-1:],tail)]
        return replace(c,prices=prices,price_pairs=pairs,sampled_prices=raw,
            features=np.r_[c.features[1:],c.features[-1:]],origin_seconds=c.origin_seconds+1)

    def assert_reference(self,m,table):
        expected = DenseValueMap(m.grid,table.cloud,m.rules,None)
        expected.sweep(7,6,4,backend='cuda')
        got = table.cpu_values()
        np.testing.assert_array_equal(np.isfinite(got),np.isfinite(expected.values))
        finite = np.isfinite(got)
        np.testing.assert_allclose(got[finite],expected.values[finite],atol=2e-11,rtol=0)

    def test_full_mode_matches_bellman_sweep_across_ring_wrap_and_shrink(self):
        m = self.initial(); before = m.values.copy()
        table = IncrementalTable(m,7,6,TableBudget(draws=4))
        pointer = table.values.data_ptr()
        for j in range(8):
            table.advance(self.shift(table.cloud,100.+.01*(j+1)),full=True)
            self.assertEqual(table.values.data_ptr(),pointer)
            self.assert_reference(m,table)
        while table.H>1:
            c = table.cloud
            shorter = replace(c,prices=c.prices[1:],counts=c.counts[1:],masses=c.masses[1:],
                features=c.features[1:],price_pairs=c.price_pairs[1:],destinations=c.destinations[1:],
                sampled_prices=c.sampled_prices[1:],origin_seconds=c.origin_seconds+1)
            table.advance(shorter,full=True)
            self.assert_reference(m,table)
        np.testing.assert_array_equal(m.values,before)

    def test_partial_schedule_is_bounded_and_preserves_exact_terminal_layers(self):
        m = self.initial()
        table = IncrementalTable(m,7,6,TableBudget(layers=2,rows=5,draws=2,cached_rows=10))
        report = table.advance(self.shift(table.cloud,100.02))
        for row in report['updates']:
            if not row['complete'] and not row['supportRepair']: self.assertLessEqual(row['rows'],5)
        self.assertLessEqual(len(table.controls),10)
        expected = DenseValueMap(m.grid,table.cloud,m.rules,None)
        expected.sweep(7,6,4,backend='cuda')
        for depth in (table.H-1,table.H-2):
            actual = table.field(depth).cpu().numpy()
            np.testing.assert_array_equal(np.isfinite(actual),np.isfinite(expected.values[depth]))
            finite = np.isfinite(actual)
            np.testing.assert_allclose(actual[finite],expected.values[depth][finite],atol=2e-11,rtol=0)

    def test_rejected_cloud_does_not_advance_and_numerical_failure_stops_updates(self):
        m = self.initial(); table = IncrementalTable(m,7,6)
        invalid = replace(m.cloud,origin_seconds=99.)
        with self.assertRaises(ValueError): table.advance(invalid)
        self.assertEqual(table.update,0)
        self.assertEqual(table.head,0)
        cloud = self.shift(table.cloud)
        with patch.object(table,'refresh_layer',side_effect=RuntimeError('kernel failed')):
            with self.assertRaisesRegex(RuntimeError,'kernel failed'): table.advance(cloud)
        self.assertTrue(table.failed)
        with self.assertRaisesRegex(RuntimeError,'restore'): table.advance(self.shift(table.cloud))

    def test_full_root_matches_independent_cpu_action_scores(self):
        m = self.initial()
        for H in (1,2,6):
            c = m.cloud
            c = replace(c,prices=c.prices[:H+1],counts=c.counts[:H+1],masses=c.masses[:H+1],
                features=c.features[:H+1],price_pairs=c.price_pairs[:H+1],
                destinations=c.destinations[:H],sampled_prices=c.sampled_prices[:H])
            source = DenseValueMap(m.grid,c,m.rules,None); source.sweep(7,6,4,backend='cuda')
            table = IncrementalTable(source,7,6)
            state = np.array([150.,-.5,100.,100.,1.,99.99,10.,-1.,100.01,10.])
            front,report = table.choose(state,reference_field=table.field(min(1,H-1)))
            ids,states,_ = root_candidates(table.facade(),state,seed=7)
            expected = []
            a,g = m.a,m.grid
            child = source.values[min(1,H-1)] if H!=2 else np.full_like(source.values[0],-np.inf)
            factors = c.factors(1) if H>1 else np.zeros((table.M,table.B))
            menu = candidate_schedule(table.facade(),0,7,6)
            for post in states:
                outcomes = []
                for j,price in enumerate(c.sampled_prices[0,0]):
                    arrived,used,filled,okay = move_arrive(post,price,m.rules)
                    value = -np.inf
                    if okay:
                        value = close_terminal(arrived,used,m.rules) if H==1 else decide(
                            arrived,used,filled,c.destinations[0,0,j],child,a['wealth'],a['exposures'],
                            a['books'],a['book_counts'],a['code_offsets'],a['book_lookup'],a['size_edges'],
                            g.price_bins,g.price_scale,m.rules,menu,factors)[0]
                    outcomes.append(value)
                expected.append(np.mean(outcomes))
            expected = np.asarray(expected); actual = front['scores'][ids]
            np.testing.assert_array_equal(np.isfinite(actual),np.isfinite(expected))
            finite = np.isfinite(expected)
            np.testing.assert_allclose(actual[finite],expected[finite],atol=2e-11,rtol=0)
            self.assertAlmostEqual(front['value'],expected.max(),places=11)
            self.assertLessEqual(report['reference']['regretBps'],1e-9)

    def test_node_specific_invalid_order_cells_remain_infeasible(self):
        g = AccountGrid(max_orders=2,price_bins=2,size_bins=2,
            exposure_points=(-1.,0.,1.),wealth_points=(10.,100.))
        c = market([[.01,.02],[.03,.01],[.01,.04],[.02,.01]])
        m = DenseValueMap(g,c,rule_vector(Rules(maximum=150.)),None)
        m.sweep(7,6,4,backend='cuda')
        table = IncrementalTable(m,7,6)
        for depth in range(table.H-1,-1,-1): table.refresh_layer(depth,full=True)
        self.assert_reference(m,table)

    def test_restored_sampler_produces_same_next_updates_and_decision(self):
        m = self.initial(); table = IncrementalTable(m,7,6,TableBudget(rows=5,draws=2))
        state = np.array([100.,0.,100.,100.,0.,0.,0.,0.,0.,0.])
        for _ in range(3):
            table.advance(self.shift(table.cloud))
            front,_ = table.choose(state,draws=2)
        with tempfile.TemporaryDirectory() as path:
            save_table(table,front,dict(model='test'),path)
            restored,_ = load_table(path,dict(model='test'))
            for _ in range(3):
                next_cloud = self.shift(table.cloud)
                table.advance(next_cloud); restored.advance(next_cloud)
                left,_ = table.choose(state,draws=2); right,_ = restored.choose(state,draws=2)
                np.testing.assert_array_equal(table.cpu_values(),restored.cpu_values())
                np.testing.assert_array_equal(table.visits,restored.visits)
                np.testing.assert_array_equal(table.root_controls,restored.root_controls)
                np.testing.assert_array_equal(left['scores'],right['scores'])
                np.testing.assert_array_equal(left['state'],right['state'])

    def test_constant_normalized_values_do_not_gain_account_size_sampling_noise(self):
        m = self.initial(); table = IncrementalTable(m,7,6,TableBudget(rows=512,draws=2))
        class ConstantGrowth:
            def __init__(self,map,states,prices,*args):
                self.growth = np.log(states[:,0]+states[:,1]*states[:,2])+.01
                self.supported = np.ones(len(states),dtype=bool)
            def evaluate(self,indices,counts):
                return np.where(np.arange(indices.shape[1])[None]<counts[:,None],self.growth[:,None],np.nan)
        with patch('oracle_incremental_table.SelectedOutcomes',ConstantGrowth):
            for update in range(1,20):
                table.update = update; table.refresh_layer(1)
                sampled = table.visits[table.slot(1)]>0
                actual = table.field(1).cpu().numpy()[sampled]
                np.testing.assert_allclose(actual,.01,atol=2e-14,rtol=0)
                self.assertLess(float(np.nanmax(table.sampling_variance[table.slot(1)])),1e-26)

    def test_newly_feasible_multitick_closes_propagate_despite_tiny_row_budget(self):
        g = AccountGrid(max_orders=0,exposure_points=(-1.5,-1.,0.,1.,1.5),wealth_points=(100.,200.))
        m = DenseValueMap(g,market([[100.,100.]]*7),rule_vector(Rules(maximum=100.)),None)
        m.sweep(7,6,4,backend='cuda')
        table = IncrementalTable(m,7,6,TableBudget(layers=2,rows=1,draws=2))
        promoted = 0
        for step in range(8):
            report = table.advance(self.shift(table.cloud))
            promoted += sum(row['promoted'] for row in report['updates'])
            expected = DenseValueMap(g,table.cloud,m.rules,None); expected.sweep(7,6,4,backend='cuda')
            np.testing.assert_array_equal(np.isfinite(table.cpu_values()[1:]),np.isfinite(expected.values[1:]))
        self.assertGreater(promoted,0)


if __name__=='__main__': unittest.main()
