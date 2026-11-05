import tempfile
import unittest
import hashlib
import json
from dataclasses import asdict,replace
from pathlib import Path
import numpy as np
from numba import set_num_threads
from oracle_account_grid import AccountGrid
from oracle_book_mc import rule_vector
from oracle_book_mc import move_arrive,close_terminal
from oracle_dense_value_map import MarketCloud,DenseValueMap
from gpu_oracle_bellman import Rules


def cloud(prices,origin=0.):
    H=len(prices)-1;counts=np.ones(H+1,dtype=np.int64);p=np.array(prices,dtype=float)[:,None]
    pairs=np.stack((np.r_[p[0,0],p[:-1,0]],p[:,0]),axis=1)[:,None]
    return MarketCloud(p,counts,np.zeros((H,1,4),dtype=np.int64),np.ones((H+1,1)),
        np.zeros((H+1,1,2,59),dtype=np.float32),pairs,origin,np.repeat(p[1:,:,None],4,axis=2))


class DenseValueMapTests(unittest.TestCase):
    def setUp(self):
        set_num_threads(2)
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.grid=AccountGrid(max_orders=0,exposure_points=(-1.,0.,1.),wealth_points=(100.,1000.,10000.))
        self.rules=rule_vector(Rules(maker=0,taker=0,maximum=100000))

    def tearDown(self):
        # Windows requires all memmap references to be closed before deleting.
        import gc
        gc.collect();self.temp.cleanup()

    def test_deterministic_gain_terminal_flat_and_dense_shapes(self):
        m=DenseValueMap(self.grid,cloud([100,100,110]),self.rules,self.root/'map')
        m.sweep(1,3,4)
        s=np.array([1000.,0.,100.,100.]);f=m.front(s,interior_samples=0)
        self.assertAlmostEqual(f['value'],np.log(1100.),places=10)
        self.assertEqual(m.values.shape,(2,3,3,1))
        # A future entry at time one is available from initial cash.
        self.assertAlmostEqual(m.values[0,1,1,0],np.log(1.1),places=10)
        del m

    def test_observation_does_not_change_mean_or_probability(self):
        m=DenseValueMap(self.grid,cloud([100,100,110],1000.),self.rules,self.root/'old')
        m.sweep(1,3,4);s=np.array([1000.,0.,100.,100.]);front=m.front(s,interior_samples=0)
        before=m.values.copy();masses=m.cloud.masses.copy()
        report=m.observe(front,100.,np.zeros((2,59)),np.array([100.,100.]),np.ones(59))
        self.assertEqual(report['terminalTimestamp'],1002.)
        self.assertLessEqual(m.errors.count.max(),1)
        np.testing.assert_array_equal(before,m.values);np.testing.assert_array_equal(masses,m.cloud.masses)
        updated=DenseValueMap(self.grid,cloud([100,110,110],1001.),self.rules,self.root/'new',m.errors)
        updated.sweep(1,3,2,previous=m)
        self.assertEqual(updated.samples.min(),2);self.assertEqual(updated.samples.max(),2)
        np.testing.assert_array_equal(before,m.values)
        del updated,m

    def test_market_cloud_roundtrip_and_no_mutation(self):
        c=cloud([100,101,102],2000.);c.save(self.root)
        restored=MarketCloud.load(self.root/'market.npz')
        np.testing.assert_array_equal(restored.features,c.features)
        np.testing.assert_array_equal(restored.destinations,c.destinations)
        np.testing.assert_array_equal(restored.sampled_prices,c.sampled_prices)
        self.assertEqual(restored.origin_seconds,c.origin_seconds)

    def test_raw_price_overrides_representative_in_terminal_execution(self):
        c=cloud([100,110])
        c=replace(c,sampled_prices=np.full_like(c.sampled_prices,90.))
        m=DenseValueMap(self.grid,c,self.rules,self.root/'raw')
        m.sweep(1,3,4)
        # Representative goes up 10%, while every actual sampled edge falls
        # 10%. Terminal wealth must follow the latter.
        self.assertAlmostEqual(m.values[0,1,2,0],np.log(.9),places=12)
        self.assertAlmostEqual(m.lookup(np.array([0.,10.,100.,100.])),np.log(900.),places=12)
        del m

    def test_archive_without_raw_prices_is_rejected(self):
        path=self.root/'market.npz'
        np.savez(path,prices=np.array([[100.],[110.]]))
        with self.assertRaisesRegex(ValueError,'regenerate'):
            MarketCloud.load(path)

    def test_interpolation_cannot_hide_an_actual_next_arrival_liquidation(self):
        grid=AccountGrid(max_orders=0,exposure_points=(-2.5,0.,2.5),wealth_points=(100.,1000.))
        c=cloud([100,100,100]);raw=c.sampled_prices.copy();raw[0,0,1]=60.
        c=replace(c,sampled_prices=raw)
        # The first outcome cannot close within the quote cap. It must not
        # bypass the later outcome's actual liquidation check.
        m=DenseValueMap(grid,c,rule_vector(Rules(maximum=10)),self.root/'liquidation')
        m.values[:]=0.
        self.assertTrue(np.isneginf(m.lookup(np.array([-1500.,25.,100.,100.]))))
        del m

    def test_next_close_reanchors_at_unprojected_arrival(self):
        c=cloud([100,100,110]);raw=c.sampled_prices.copy();raw[0]=120.
        c=replace(c,sampled_prices=raw)
        m=DenseValueMap(self.grid,c,self.rules,self.root/'anchor')
        m.sweep(1,3,4)
        # Cash arrives at 120; entering long and closing on the next +10%
        # row uses 132, never the retained node's absolute price 110.
        self.assertAlmostEqual(m.values[0,1,1,0],np.log(1.1),places=12)
        np.testing.assert_allclose(m.query_prices(120.,1,0),132.)
        del m

    def test_completed_map_reloads_read_only_with_same_decision(self):
        rules=Rules(maker=0,taker=0,maximum=100000)
        kernel=Path(__file__).with_name('oracle_book_mc.py')
        (self.root/'plan.json').write_text(json.dumps(dict(grid=asdict(self.grid),rules=asdict(rules),
            fundingMultiplier=1.,sourceHashes={'ml/oracle_book_mc.py':hashlib.sha256(kernel.read_bytes()).hexdigest()})))
        m=DenseValueMap(self.grid,cloud([100,100,110]),self.rules,self.root/'saved')
        m.sweep(1,3,4);s=np.array([1000.,0.,100.,100.]);v=m.front(s,interior_samples=0)['value']
        restored=DenseValueMap.load(self.root/'saved')
        self.assertEqual(restored.front(s,interior_samples=0)['value'],v)
        self.assertFalse(restored.values.flags.writeable)
        with self.assertRaisesRegex(ValueError,'read-only'):restored.sweep(1)
        operator_path=self.root/'saved'/'operator.json'
        signature=json.loads(operator_path.read_text());signature['oracle_book_gpu.py']='changed'
        operator_path.write_text(json.dumps(signature))
        with self.assertRaisesRegex(ValueError,'different Bellman operator'):DenseValueMap.load(self.root/'saved')
        operator_path.unlink()
        with self.assertRaisesRegex(ValueError,'lacks a complete Bellman operator signature'):DenseValueMap.load(self.root/'saved')
        del restored,m

    def test_capped_root_proposal_uses_actual_wealth_and_exposure(self):
        rules=rule_vector(Rules(maximum=100,maker=.0002,taker=.0005))
        m=DenseValueMap(self.grid,cloud([100,100,110]),rules,self.root/'cap')
        m.sweep(1,3,4)
        f=m.front(np.array([1000.,0.,100.,100.]),interior_samples=0)
        # Every returned finite proposal has obeyed the 100 quote-flow cap,
        # even when its declared target was full investment.
        for state in f['states'][np.isfinite(f['scores'])]:
            self.assertLessEqual(abs(state[0]-1000),100+1e-9)
        del m

    def test_short_can_close_in_two_capped_decisions(self):
        rules=rule_vector(Rules())
        grid=AccountGrid(max_orders=0,exposure_points=(-2.,-1.,0.,1.,2.),wealth_points=(5000.,10000.,20000.))
        m=DenseValueMap(grid,cloud([100.,100.,100.]),rules,self.root/'short')
        m.sweep(1,5,4)
        s=np.array([20000.,-100.,100.,100.])
        best,post,_=m.decide(s,depth=1)
        self.assertTrue(np.isfinite(best))
        self.assertGreater(post[1],-100.)
        self.assertLessEqual(abs(post[0]-s[0]),10000.)
        del m

    def test_projected_prices_preserve_relative_moves(self):
        m=DenseValueMap(self.grid,cloud([100,101,102]),self.rules,self.root/'prices')
        np.testing.assert_allclose(m.query_prices(200.,0,0),202.)
        del m

    def test_last_decision_cannot_use_optimistic_interpolation_to_skip_close(self):
        m=DenseValueMap(self.grid,cloud([100,100,100]),self.rules,self.root/'last')
        m.sweep(1,3,4)
        # Poison the interpolator: last-step feasibility must still be exact.
        m.values[-1]=1000.
        s=np.array([2000.,-10.,100.,100.]);value,state,_=m.decide(s,depth=1)
        arrived,consumed,_,okay=move_arrive(state,100.,self.rules)
        self.assertTrue(okay and np.isfinite(close_terminal(arrived,consumed,self.rules)))
        self.assertAlmostEqual(value,np.log(1000.),places=10)
        del m


if __name__=='__main__':unittest.main()
