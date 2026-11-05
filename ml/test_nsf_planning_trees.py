import unittest
import numpy as np
import torch
from latent_neural_stochastic_flow import LatentNSF
from export_nsf_planning_trees import schedule, next_observations

class PlanningForecastTests(unittest.TestCase):
    def test_fixed_endpoint_and_nested_time_refinement(self):
        for remaining in [1,3,899,901,144533]:
            a,b=schedule(remaining,900,1),schedule(remaining,900,2)
            self.assertEqual(a[-1],remaining);self.assertEqual(b[-1],remaining)
            self.assertTrue(set(a)<=set(b))
            self.assertTrue(all(x>0 for x in np.diff([0]+a)))
            self.assertLessEqual(max(np.diff([0]+a)),86400)

    def test_other_realized_histories_do_not_change_current_tree_samples(self):
        torch.set_num_threads(2);torch.manual_seed(9)
        model=LatentNSF(torch.zeros(59),torch.ones(59),dimensions=4,width=8).eval()
        with torch.no_grad():
            model.posterior.weight.normal_(0,.01);model.decoder[-1].weight.normal_(0,.01)
        f=np.zeros((2,2,59),np.float32)
        p=np.array([[100,101],[110,111]],float);t=np.array([[1700000000,1700000001]]*2,float)
        h=np.array([900,900])
        before=next_observations(model,f,p,t,h,2,7)
        changed=f.copy();changed[1]+=100
        after=next_observations(model,changed,p,t,h,2,7)
        for x,y in zip(before,after):np.testing.assert_array_equal(x[0],y[0])
        self.assertTrue(np.any(before[0][1]!=after[0][1]))
        self.assertTrue(np.all(before[1]>0))
        np.testing.assert_allclose(before[1]/.01,np.round(before[1]/.01),atol=1e-8)

if __name__=='__main__':unittest.main()
