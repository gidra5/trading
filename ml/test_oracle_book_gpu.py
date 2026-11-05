import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
import numpy as np
import torch
from numba import set_num_threads
from gpu_oracle_bellman import Rules
from oracle_account_grid import AccountGrid
from oracle_book_mc import rule_vector,backup_layer
from oracle_book_gpu import GPUBookSweep
from oracle_dense_value_map import MarketCloud


def market(prices,branches=4):
    p=np.asarray(prices,dtype=float);H,M=p.shape[0]-1,p.shape[1]
    dest=np.tile(np.arange(branches)%M,(H,M,1)).astype(np.int64)
    pairs=np.stack((np.r_[p[:1],p[:-1]],p),2)
    return MarketCloud(p,np.full(H+1,M,dtype=np.int64),dest,np.full((H+1,M),1/M),
        np.zeros((H+1,M,2,59),dtype=np.float32),pairs,0.,p[np.arange(1,H+1)[:,None,None],dest])


@unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
class GPUBookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        set_num_threads(2);torch.set_num_threads(2)

    def compare(self,g,c,r,budget=None):
        a=g.arrays();C=len(a['coordinates']);W=len(a['wealth']);M=c.prices.shape[1];B=c.destinations.shape[-1];H=len(c.counts)-1
        gpu=GPUBookSweep(g,c,r);child=np.zeros((W,C,M));rng=np.random.default_rng(27)
        expected=[];variances=[]
        for h in range(H-1,-1,-1):
            perm=np.array([rng.permutation(B) for _ in range(M)]);alloc=np.full(C,B if budget is None else budget)
            candidates=np.arange(C,dtype=np.int64)
            prices=c.factors(h+1) if h+1<H else np.zeros((M,B))
            continuation=np.full_like(child,-np.inf) if h==H-2 else child
            v,var=backup_layer(c.prices[h],c.sampled_prices[h],c.destinations[h],perm,alloc,continuation,
                a['wealth'],a['exposures'],a['books'],a['book_counts'],a['code_offsets'],a['book_lookup'],a['size_edges'],
                g.price_bins,g.price_scale,r,candidates,h==H-1,prices)
            gpu.step(h,perm,alloc,candidates);expected.append(v);variances.append(var);child=v
        gpu.synchronize();actual=gpu.values.cpu().numpy();actual_var=gpu.variance.cpu().numpy()
        expected=np.array(expected[::-1]);variances=np.array(variances[::-1])
        np.testing.assert_array_equal(np.isfinite(actual),np.isfinite(expected))
        mask=np.isfinite(expected)
        np.testing.assert_allclose(actual[mask],expected[mask],atol=2.e-11,rtol=0)
        np.testing.assert_allclose(actual_var[mask],variances[mask],atol=2.e-12,rtol=0)

    def test_cash_and_directional_actions(self):
        g=AccountGrid(max_orders=0,exposure_points=(-1.,0.,1.),wealth_points=(100.,1000.,10000.))
        c=market([[100,100],[90,110],[99,101]])
        self.compare(g,c,rule_vector(Rules(maker=0,taker=0,maximum=100000)))

    def test_two_orders_fees_caps_funding_and_liquidation(self):
        g=AccountGrid(max_orders=2,price_bins=1,size_bins=2,
            exposure_points=(-2.5,-2.,0.,1.,2.,2.5),wealth_points=(100.,1000.,10000.))
        c=market([[100,100],[99.99,100.01],[60,160],[99,101]])
        self.compare(g,c,rule_vector(Rules(maximum=1000),.99999))

    def test_three_orders_and_reduced_conditional_samples(self):
        g=AccountGrid(max_orders=3,price_bins=1,size_bins=2,
            exposure_points=(-1.,0.,1.),wealth_points=(100.,1000.))
        self.compare(g,market([[100,100],[99.98,100.02],[100,100]]),rule_vector(Rules(maximum=150)),2)

    def test_account_feasibility_changes_with_price_tick(self):
        g=AccountGrid(max_orders=2,price_bins=2,size_bins=2,
            exposure_points=(-1.,0.,1.),wealth_points=(10.,100.))
        # Buy limits can become nonpositive at the low-price node. A cell
        # feasible at another depth/node must remain in the GPU work list.
        self.compare(g,market([[.01,.02],[.03,.01],[.01,.04]]),rule_vector(Rules(maximum=150)))

    def test_unprojected_marks_drive_fills_liquidation_and_reanchored_fallback(self):
        g=AccountGrid(max_orders=2,price_bins=1,size_bins=2,
            exposure_points=(-2.5,-2.,0.,1.,2.,2.5),wealth_points=(100.,1000.,10000.))
        c=market([[100,100],[100,100],[100,100]])
        # Lookup nodes are all flat; real draws cross limits and maintenance
        # boundaries. The second row is reanchored to each actual arrival.
        raw=np.array([[[60.,99.98,100.02,160.]]*2,[[99.97,100.,100.01,100.05]]*2])
        self.compare(g,replace(c,sampled_prices=raw),rule_vector(Rules(maximum=1000),.99999))


if __name__=='__main__':unittest.main()
