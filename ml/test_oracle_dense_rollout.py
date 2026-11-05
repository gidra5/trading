import unittest
from unittest.mock import patch
import numpy as np
import torch
from oracle_dense_rollout import roll_market_cloud
from test_oracle_book_gpu import market
from nsf_scenario_graph import build_graph
from oracle_dense_value_map import MarketCloud


class FixedModel(torch.nn.Module):
    def __init__(self):
        super().__init__();self.dummy=torch.nn.Parameter(torch.zeros(1))
        self.register_buffer('feature_mean',torch.zeros(59));self.register_buffer('feature_std',torch.ones(59))


class RollingMarketTests(unittest.TestCase):
    def test_graph_compression_preserves_raw_execution_samples(self):
        def predict(model,features,prices,times,horizon,branches,seed):
            return np.repeat(features[:,-1,None,:],branches,axis=1),prices[:,-1,None]+np.array([-1.,1.])
        with patch('nsf_scenario_graph.next_observations',predict):
            graph=build_graph(FixedModel(),np.zeros((2,59)),np.array([100.,100.]),
                np.array([-1.,0.]),[1,2],width=1,samples=2)
        c=MarketCloud.from_graph(graph)
        np.testing.assert_array_equal(c.sampled_prices[0,0],[99.,101.])
        self.assertTrue(np.any(graph.prices[0].numpy()!=c.sampled_prices[0]))
        np.testing.assert_array_equal(c.sampled_prices[1,0],c.prices[1,0]+np.array([-1.,1.]))

    def test_new_root_and_tail_reuse_interior_and_recompute_probabilities(self):
        previous=market([[100,100],[99,101],[98,102],[97,103],[96,104]],branches=2)
        names=('prices','counts','features','price_pairs','destinations','masses','sampled_prices')
        originals={k:getattr(previous,k).copy() for k in names}
        for k in names:getattr(previous,k).flags.writeable=False
        calls=[]
        def predict(model,features,prices,times,horizon,branches,seed):
            calls.append((features.copy(),prices.copy(),times.copy(),seed))
            return np.repeat(features[:,-1,None,:],branches,axis=1),prices[:,-1,None]+np.array([-1.,1.])
        observed_features=np.zeros((2,59));observed_prices=np.array([100.,101.]);times=np.array([0.,1.])
        current,stats=roll_market_cloud(previous,FixedModel(),observed_features,observed_prices,times,23,predict=predict)
        self.assertEqual(len(calls),2)
        np.testing.assert_array_equal(calls[0][0][0],observed_features)
        np.testing.assert_array_equal(calls[0][1][0],observed_prices)
        np.testing.assert_array_equal(calls[0][2][0],times)
        np.testing.assert_array_equal(calls[1][0],previous.features[-1])
        np.testing.assert_array_equal(calls[1][2],[[3.,4.],[3.,4.]])
        self.assertEqual(calls[1][3],26)
        np.testing.assert_array_equal(current.destinations[1:3],previous.destinations[2:4])
        np.testing.assert_array_equal(current.sampled_prices[1:3],previous.sampled_prices[2:4])
        np.testing.assert_array_equal(current.sampled_prices[0,0],[100.,102.])
        self.assertEqual(stats['rootExecutionPriceRMSEBps'],0.)
        self.assertEqual(stats['rootExecutionMeanReturnShiftBps'],0.)
        np.testing.assert_array_equal(current.features[1:4],previous.features[2:5])
        np.testing.assert_array_equal(current.price_pairs[1:4],previous.price_pairs[2:5])
        np.testing.assert_allclose(current.masses.sum(1),1.,atol=1.e-14,rtol=0)
        for h in range(4):
            expected=np.bincount(current.destinations[h,:current.counts[h]].ravel(),
                weights=np.repeat(current.masses[h,:current.counts[h]]/2,2),minlength=2)
            np.testing.assert_array_equal(current.masses[h+1],expected)
        self.assertFalse(np.array_equal(current.masses[1],previous.masses[2]))
        self.assertGreater(stats['rootProjectionPriceRMSEBps'],0)
        self.assertEqual(stats['predictedParentRows'],3)
        self.assertEqual(stats['reusedConditionalRows'],4)
        for k in names:np.testing.assert_array_equal(getattr(previous,k),originals[k])

    def test_rejects_skipped_real_observations(self):
        previous=market([[100,100],[99,101],[98,102]],branches=2)
        with self.assertRaisesRegex(ValueError,'next observed second'):
            roll_market_cloud(previous,FixedModel(),np.zeros((2,59)),np.array([100.,101.]),np.array([1.,2.]),23)

    def test_fixed_terminal_roll_predicts_root_without_appending_tail(self):
        previous=market([[100,100],[99,101],[98,102]],branches=2)
        def predict(model,features,prices,times,horizon,branches,seed):
            return np.repeat(features[:,-1,None,:],branches,axis=1),prices[:,-1,None]+np.array([-1.,1.])
        current,stats=roll_market_cloud(previous,FixedModel(),np.zeros((2,59)),np.array([100.,101.]),
            np.array([0.,1.]),23,predict=predict,append_tail=False)
        self.assertEqual(len(current.counts)-1,1)
        self.assertEqual(stats['predictedParentRows'],1)
        np.testing.assert_array_equal(current.prices[-1],previous.prices[-1])
        np.testing.assert_array_equal(current.sampled_prices[0,0],[100.,102.])
        np.testing.assert_allclose(current.masses.sum(1),1.)


if __name__=='__main__':unittest.main()
