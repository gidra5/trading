import itertools
import tempfile
import hashlib
import json
import unittest
from dataclasses import replace,asdict
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from oracle_dense_reuse import paired_mean,project_market_nodes
from oracle_book_gpu import GPUBookSweep
from oracle_account_grid import AccountGrid
from oracle_book_mc import rule_vector
from gpu_oracle_bellman import Rules
from test_oracle_book_gpu import market
from test_oracle_dense_value_map import cloud
from oracle_dense_value_map import DenseValueMap


class ControlVariateTests(unittest.TestCase):
    def test_exhaustive_subsamples_recover_current_mean(self):
        # Controls need not be accurate, or have the same mean as new values.
        old=np.array([3.,-7.,10.,1.]);new=np.array([7.,-2.,9.,20.])
        estimates=[paired_mean(old,new,list(s)) for s in itertools.combinations(range(4),2)]
        self.assertAlmostEqual(np.mean(estimates),new.mean(),places=13)

    def test_common_shift_has_zero_correction_variance(self):
        old=np.array([-100.,-1.,20.,1000.]);new=old+.25
        estimates=[paired_mean(old,new,list(s)) for s in itertools.combinations(range(4),2)]
        self.assertEqual(np.var(estimates),0.)
        self.assertGreater(np.var([new[list(s)].mean() for s in itertools.combinations(range(4),2)]),0.)

    def test_nonfinite_control_requires_census(self):
        self.assertEqual(paired_mean([1.,-np.inf,2.],[3.,4.,5.],[0]),4.)

    def test_market_mapping_matches_time_and_keeps_source_immutable(self):
        old=market([[100,100],[90,110],[80,120]])
        new=replace(market([[110,90],[120,80],[125,75]]),origin_seconds=1.)
        # Match the complete observed price pair, not only its endpoint.
        new=replace(new,price_pairs=np.array([[[100,110],[100,90]],[[110,120],[90,80]],[[120,125],[80,75]]],float))
        before=old.price_pairs.copy();mapping=project_market_nodes(new,old,1,np.ones(59))
        np.testing.assert_array_equal(mapping,[[1,0],[-1,-1]])
        np.testing.assert_array_equal(old.price_pairs,before)
        with self.assertRaisesRegex(ValueError,'timestamps'):project_market_nodes(new,old,0)


@unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
class GPUReuseTests(unittest.TestCase):
    def test_shared_continuation_level_does_not_add_sampling_noise(self):
        g=AccountGrid(max_orders=0,exposure_points=(-1.,0.,1.),wealth_points=(100.,1000.,10000.))
        r=rule_vector(Rules(maker=0,taker=0,maximum=100000))
        c=market([[100]*4,[99,100,101,102],[99,100,101,102]])
        C=len(g.arrays()['coordinates']);catalog=np.arange(C);perm=np.tile(np.arange(4),(4,1))
        base=GPUBookSweep(g,c,r,paired=True);base.values[1].fill_(1.);base.step(0,perm,np.full(C,4),catalog)
        prior=SimpleNamespace(outcome_cache=base.cache.cpu().numpy(),cloud=c,shape=tuple(base.values.shape),
            outcome_representation='cash-centered-v1')
        # Change continuation by a different account-independent constant at
        # each child market state. Shared cash centering cancels this exactly.
        level=torch.tensor([2.,3.,4.,5.],device='cuda',dtype=torch.float64)
        truth=GPUBookSweep(g,c,r);truth.values[1].copy_(level.expand_as(truth.values[1]))
        truth.step(0,perm,np.full(C,4),catalog);expected=truth.values.cpu().numpy()[0,1]
        for selected in itertools.combinations(range(4),2):
            order=list(selected)+[j for j in range(4) if j not in selected]
            gpu=GPUBookSweep(g,c,r,previous=prior,shift=0,paired=True,feature_std=np.ones(59))
            gpu.values[1].copy_(level.expand_as(gpu.values[1]))
            gpu.step(0,np.tile(order,(4,1)),np.full(C,2),catalog)
            np.testing.assert_allclose(gpu.values.cpu().numpy()[0,1],expected,atol=3.e-14,rtol=0)

    def test_multistep_census_and_saved_cache_generation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            g=AccountGrid(max_orders=0,exposure_points=(-1.,0.,1.),wealth_points=(100.,1000.))
            rules=Rules(maker=0,taker=0,maximum=100000);r=rule_vector(rules)
            kernel=Path(__file__).with_name('oracle_book_mc.py')
            (root/'plan.json').write_text(json.dumps(dict(grid=asdict(g),rules=asdict(rules),fundingMultiplier=1.,
                sourceHashes={'ml/oracle_book_mc.py':hashlib.sha256(kernel.read_bytes()).hexdigest()})))
            old=DenseValueMap(g,cloud([100,100,105],1000),r,root/'old')
            old.sweep(11,3,4,estimator='paired');before=old.outcome_cache.copy()
            loaded=DenseValueMap.load(root/'old')
            self.assertFalse(loaded.outcome_cache.flags.writeable)
            new=DenseValueMap(g,cloud([100,105,110],1001),r,root/'new')
            new.sweep(11,3,4,previous=loaded,estimator='paired')
            reference=DenseValueMap(g,new.cloud,r,root/'reference');reference.sweep(11,3,4)
            np.testing.assert_allclose(new.values,reference.values,atol=2.e-14,rtol=0)
            np.testing.assert_array_equal(before,loaded.outcome_cache)
            self.assertEqual(new.reuse_generation,1)
            restored=DenseValueMap.load(root/'new');self.assertEqual(restored.reuse_generation,1)
            del old,loaded,new,reference,restored

    def test_paired_terminal_mean_cache_updates_and_census(self):
        g=AccountGrid(max_orders=0,exposure_points=(-1.,0.,1.),wealth_points=(100.,1000.))
        r=rule_vector(Rules(maker=0,taker=0,maximum=100000))
        old=market([[100]*4,[90,95,105,110]])
        new=market([[100]*4,[92,96,106,109]])
        a=g.arrays();C=len(a['coordinates']);catalog=np.arange(C,dtype=np.int64)
        full=np.full(C,4,dtype=np.int64);half=np.full(C,2,dtype=np.int64)
        identity=np.tile(np.arange(4),(4,1))
        base=GPUBookSweep(g,old,r,paired=True);base.step(0,identity,full,catalog)
        prior=SimpleNamespace(outcome_cache=base.cache.cpu().numpy(),cloud=old,shape=tuple(base.values.shape),outcome_representation='cash-centered-v1')
        before=prior.outcome_cache.copy()
        truth=GPUBookSweep(g,new,r);truth.step(0,identity,full,catalog);expected=truth.values.cpu().numpy()
        estimates=[]
        for selected in itertools.combinations(range(4),2):
            order=list(selected)+[j for j in range(4) if j not in selected]
            perm=np.tile(order,(4,1)).astype(np.int64)
            gpu=GPUBookSweep(g,new,r,previous=prior,shift=0,paired=True,feature_std=np.ones(59))
            gpu.step(0,perm,half,catalog)
            estimate=gpu.values.cpu().numpy();estimates.append(estimate)
            cache=gpu.cache.cpu().numpy()
            for j in order[2:]:np.testing.assert_array_equal(cache[...,j],before[...,j])
            # Selected canonical-edge cache entries carry freshly computed
            # terminal values, independently of permutation rank.
            for j in selected:
                expected_edge=np.empty((2,C,4))
                for wi,w in enumerate(a['wealth']):
                    for ei,e in enumerate(a['exposures']):
                        expected_edge[wi,ei]=np.log(w*(1-e)+w*e*new.prices[1,j]/100)-np.log(w)
                np.testing.assert_allclose(cache[0,...,j],expected_edge,atol=2.e-14,rtol=0)
            np.testing.assert_array_equal(before,prior.outcome_cache)
        np.testing.assert_allclose(np.mean(estimates,axis=0),expected,atol=2.e-14,rtol=0)
        census=GPUBookSweep(g,new,r,previous=prior,shift=0,paired=True,feature_std=np.ones(59))
        census.step(0,identity,full,catalog)
        np.testing.assert_allclose(census.values.cpu().numpy(),expected,atol=2.e-14,rtol=0)

    def test_nonfinite_old_row_forces_full_refresh(self):
        g=AccountGrid(max_orders=0,exposure_points=(-1.,0.,1.),wealth_points=(100.,1000.))
        r=rule_vector(Rules(maker=0,taker=0,maximum=100000))
        old=market([[100]*4,[80,90,110,180]]);new=market([[100]*4,[98,99,101,102]])
        C=len(g.arrays()['coordinates']);catalog=np.arange(C);perm=np.tile(np.arange(4),(4,1))
        base=GPUBookSweep(g,old,r,paired=True);base.step(0,perm,np.full(C,4),catalog)
        prior=SimpleNamespace(outcome_cache=base.cache.cpu().numpy(),cloud=old,shape=tuple(base.values.shape),outcome_representation='cash-centered-v1')
        gpu=GPUBookSweep(g,new,r,previous=prior,shift=0,paired=True,feature_std=np.ones(59))
        gpu.step(0,perm,np.full(C,2),catalog)
        counts=gpu.effective_counts.cpu().numpy()
        self.assertTrue(np.all(counts[:,:,0,:]==4))
        self.assertTrue(np.isfinite(gpu.values.cpu().numpy()[:,:,0,:]).all())

    def test_unsampled_new_liquidation_cannot_receive_finite_value(self):
        g=AccountGrid(max_orders=0,exposure_points=(-1.,0.,1.),wealth_points=(100.,1000.))
        r=rule_vector(Rules(maker=0,taker=0,maximum=100000))
        old=market([[100]*4,[98,99,101,102]]);new=market([[100]*4,[98,99,101,180]])
        C=len(g.arrays()['coordinates']);catalog=np.arange(C);perm=np.tile(np.arange(4),(4,1))
        base=GPUBookSweep(g,old,r,paired=True);base.step(0,perm,np.full(C,4),catalog)
        prior=SimpleNamespace(outcome_cache=base.cache.cpu().numpy(),cloud=old,shape=tuple(base.values.shape),
            outcome_representation='cash-centered-v1')
        gpu=GPUBookSweep(g,new,r,previous=prior,shift=0,paired=True,feature_std=np.ones(59))
        gpu.step(0,perm,np.full(C,2),catalog)
        self.assertTrue(np.all(gpu.effective_counts.cpu().numpy()[:,:,0,:]==4))
        self.assertTrue(np.isneginf(gpu.values.cpu().numpy()[:,:,0,:]).all())

    def test_partial_multistep_book_support_equals_census(self):
        g=AccountGrid(max_orders=2,price_bins=1,size_bins=2,
            exposure_points=(-2.,-1.,0.,1.,2.),wealth_points=(100.,1000.,10000.))
        r=rule_vector(Rules(maximum=1000),.999999)
        old=market([[100]*4,[99.8,99.9,100.1,100.2],[99.5,100,100.5,101],[99.8,100,100.2,101]])
        new=market([[100]*4,[70,100,110,160],[80,100,110,140],[90,95,105,110]])
        C=len(g.arrays()['coordinates']);catalog=np.arange(C);perm=np.tile(np.arange(4),(4,1))
        base=GPUBookSweep(g,old,r,paired=True)
        for h in range(2,-1,-1):base.step(h,perm,np.full(C,4),catalog)
        prior=SimpleNamespace(outcome_cache=base.cache.cpu().numpy(),cloud=old,shape=tuple(base.values.shape),
            outcome_representation='cash-centered-v1')
        approximate=GPUBookSweep(g,new,r,previous=prior,shift=0,paired=True,feature_std=np.ones(59))
        reference=GPUBookSweep(g,new,r)
        for h in range(2,-1,-1):
            approximate.step(h,perm,np.full(C,2),catalog);reference.step(h,perm,np.full(C,4),catalog)
        np.testing.assert_array_equal(np.isfinite(approximate.values.cpu().numpy()),np.isfinite(reference.values.cpu().numpy()))


if __name__=='__main__':unittest.main()
