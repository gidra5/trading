import math
import unittest
import numpy as np
import torch
from numba import set_num_threads
from gpu_oracle_bellman import TensorOracle,Rules,Menu
from oracle_certified_value import SCALE
from oracle_plane_policy import plane_policy_action,nearest_observation_nodes


class PlanePolicyTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);set_num_threads(2)
        self.oracle=TensorOracle(Rules(maker=0,taker=0))
        self.state=torch.tensor([[10000.,0,100,100,0,0,0,0,0]],dtype=torch.float64)
        self.desc=self.oracle.descriptors(Menu(step=.01,limits=False),'cpu')

    def choose(self,prices,chunk=65536):
        p=torch.tensor([prices],dtype=torch.float64);n=np.arange(len(prices))[None,:]
        c=np.full((len(prices),1),SCALE,np.int64)
        return plane_policy_action(self.oracle,self.state,self.desc,p,n,c,c,9,chunk)

    def test_averages_before_action_maximum(self):
        got=self.choose([200.,50.])
        self.assertTrue(got['valid'].item())
        self.assertAlmostEqual(got['descriptor'][0,0].item(),.5)
        self.assertAlmostEqual(got['value'][0].item(),.5*math.log(1.125),places=10)

    def test_actual_predicted_price_is_used_without_snapping_to_value_node(self):
        got=self.choose([110.,110.])
        # Capacity is 10,000, so the greatest attainable entry is 1x.
        self.assertAlmostEqual(got['state'][0,1].item(),100.)
        self.assertAlmostEqual(got['value'][0].item(),math.log(1.1),places=10)

    def test_action_chunk_size_does_not_change_policy(self):
        a=self.choose([99.,101.],chunk=4);b=self.choose([99.,101.])
        torch.testing.assert_close(a['value'],b['value']);torch.testing.assert_close(a['state'],b['state'])

    def test_other_account_forecasts_do_not_affect_first_action(self):
        s=self.state.repeat(2,1);p=torch.tensor([[99.,101.],[99.,101.]],dtype=torch.float64)
        n=np.array([[0,1],[0,1]]);c=np.full((2,1),SCALE,np.int64)
        a=plane_policy_action(self.oracle,s,self.desc,p,n,c,c,9)
        p[1]=torch.tensor([90.,120.])
        b=plane_policy_action(self.oracle,s,self.desc,p,n,c,c,9)
        torch.testing.assert_close(a['scores'][0],b['scores'][0])

    def test_mapping_uses_joint_observable_features(self):
        centers=torch.zeros(2,2,59);centers[1,:,58]=10
        prices=torch.full((2,2),100.,dtype=torch.float64)
        nodes,error=nearest_observation_nodes(centers.flip(0),prices,centers,prices,torch.zeros(59),torch.ones(59),100.)
        np.testing.assert_array_equal(nodes,[1,0]);np.testing.assert_allclose(error,0,atol=1e-7)

    def test_input_arrays_remain_unchanged(self):
        saved=self.state.clone();self.choose([99.,101.]);torch.testing.assert_close(saved,self.state)


if __name__=='__main__':unittest.main()
