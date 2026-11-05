import hashlib
import json
from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest
import numpy as np
from gpu_oracle_bellman import Rules
from oracle_account_grid import AccountGrid
from oracle_book_mc import rule_vector
from oracle_dense_value_map import DenseValueMap,MarketCloud
from oracle_dense_checkpoint import load_checkpoint,initial_archive
from audit_oracle_dense_ledger import audit
from report_oracle_dense_backtest import summarize


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.grid=AccountGrid(max_orders=1,price_bins=1,size_bins=2,exposure_points=(-1.,0.,1.),
            wealth_points=(5000.,10000.,20000.));self.rules=Rules()
        self.plan=dict(grid=asdict(self.grid),rules=asdict(self.rules),fundingMultiplier=1.,gamma=1,
            checkpointSha256='test-checkpoint',arguments=dict(seed=7,origin=0,horizon=3,branches=2,
            candidates=3,seconds=6,refresh_every=0,repair_tolerance=1e-5,repair_interval=8),
            sourceHashes={f'ml/{name}':hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                for name in ('oracle_book_mc.py','oracle_dense_rollout.py','nsf_scenario_graph.py','export_nsf_planning_trees.py')})
        (self.root/'plan.json').write_text(json.dumps(self.plan))
        prices=np.full((4,1),100.)
        def cloud(origin):return MarketCloud(prices,np.ones(4,dtype=np.int64),np.zeros((3,1,2),dtype=np.int64),
            np.ones((4,1)),np.zeros((4,1,2,59),dtype=np.float32),np.full((4,1,2),100.),origin,np.full((3,1,2),100.))
        initial=self.root/'initial';initial.mkdir();cloud(10.).save(initial)
        self.state=np.array([10000.,0.,100.,100.,0.,0.,0.]);np.savez(initial/'front.npz',state=self.state)
        current=DenseValueMap(self.grid,cloud(11.),rule_vector(self.rules),None)
        current.sweep(7,3,2,backend='cpu');current.export(self.root/'last-map')
        self.front=current.front(self.state,interior_samples=0,seed=7)
        np.savez(self.root/'last-map'/'front.npz',**self.front)
        self.row=dict(update=1,horizon=3,price=100.,originSeconds=11.,state=self.state.tolist(),tradeQuote=0.,
            markedEquity=10000.,exposure=0.,orders=0,filled=False,stepSeconds=.5,errorAllowance=0.,
            market=dict(rootExecutionPriceRMSEBps=0.,rootProjectionPriceRMSEBps=0.))
        self.result=dict(completed=False,failure=None,terminalWealth=None,observedUpdates=1,
            state=self.state.tolist(),elapsedSeconds=2.,initialSeconds=1.,marketTrades=0,fillTicks=0,
            comparisons=[],timeBudgetReached=True,markedEquity=10000.)
        self.write()

    def tearDown(self):self.temp.cleanup()

    def write(self):
        (self.root/'updates.jsonl').write_text(json.dumps(self.row)+'\n')
        (self.root/'results.json').write_text(json.dumps(self.result))

    def test_real_archive_restores_value_map_front_and_runtime(self):
        saved=load_checkpoint(self.root,self.plan,10.)
        self.assertEqual(saved.update,1);self.assertEqual(saved.current.shape[0],3)
        self.assertFalse(saved.current.writable);self.assertEqual(saved.elapsed_seconds,2.)
        self.assertEqual(saved.initial_seconds,1.);np.testing.assert_array_equal(saved.state,self.state)
        np.testing.assert_array_equal(saved.front['scores'],self.front['scores'])

    def test_pending_order_survives_checkpoint_boundary(self):
        state=self.state.copy();state[4:]=[1.,99.99,1000.]
        self.front['state']=state;np.savez(self.root/'last-map'/'front.npz',**self.front)
        self.row.update(state=state.tolist(),orders=1);self.result['state']=state.tolist();self.write()
        saved=load_checkpoint(self.root,self.plan,10.)
        np.testing.assert_array_equal(saved.state[4:],[1.,99.99,1000.])

    def test_rejects_changed_policy_or_forecast_contract(self):
        for key in ('seconds','seed','repair_tolerance'):
            expected=json.loads(json.dumps(self.plan));expected['arguments'][key]+=1
            with self.assertRaisesRegex(ValueError,f'changes {key}'):load_checkpoint(self.root,expected,10.)
        expected=json.loads(json.dumps(self.plan));expected['sourceHashes']['ml/oracle_dense_rollout.py']='different'
        with self.assertRaisesRegex(ValueError,'forecast generator'):load_checkpoint(self.root,expected,10.)

    def test_rejects_wrong_boundary_incomplete_log_and_changed_account(self):
        with self.assertRaisesRegex(ValueError,'time boundary'):load_checkpoint(self.root,self.plan,9.)
        (self.root/'updates.jsonl').write_text(json.dumps(self.row))
        with self.assertRaisesRegex(ValueError,'unflushed'):load_checkpoint(self.root,self.plan,10.)
        self.result['state'][0]=9999.;self.write()
        with self.assertRaisesRegex(ValueError,'account and chosen action'):load_checkpoint(self.root,self.plan,10.)

    def test_rejects_completed_or_failed_replays(self):
        self.result['completed']=True;self.write()
        with self.assertRaisesRegex(ValueError,'incomplete replay'):load_checkpoint(self.root,self.plan,10.)
        self.result.update(completed=False,failure=dict(reason='liquidated'));self.write()
        with self.assertRaisesRegex(ValueError,'incomplete replay'):load_checkpoint(self.root,self.plan,10.)

    def test_reports_and_ledger_use_shared_original_archive(self):
        resumed=self.root/'resumed';resumed.mkdir()
        plan={**self.plan,'initialArchive':str(self.root/'initial')}
        (resumed/'plan.json').write_text(json.dumps(plan))
        (resumed/'updates.jsonl').write_text(json.dumps(self.row)+'\n')
        (resumed/'results.json').write_text(json.dumps(self.result))
        self.assertEqual(initial_archive(resumed),self.root/'initial')
        self.assertEqual(audit(resumed)['checkedUpdates'],1)
        self.assertEqual(summarize(resumed)['observedMarketPriceReturn'],0.)


if __name__=='__main__':unittest.main()
