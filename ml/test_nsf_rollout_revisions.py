import unittest
import numpy as np
import torch
from latent_neural_stochastic_flow import LatentNSF
from export_nsf_planning_trees import next_observations
from audit_nsf_rollout_revisions import one_second


class RolloutRevisionTests(unittest.TestCase):
    def test_tensor_kernel_matches_existing_planner_sampler(self):
        torch.set_num_threads(2);torch.manual_seed(81)
        m=LatentNSF(torch.zeros(59),torch.ones(59),dimensions=4,width=8).eval()
        with torch.no_grad():
            for p in m.parameters():p.add_(torch.randn_like(p)*.001)
        f=torch.randn(3,2,59);p=torch.tensor([[100.,100.1],[101.,101.],[98.,99.]],dtype=torch.float64)
        t=torch.tensor([[1786320000.,1786320001.]],dtype=torch.float64).expand(3,-1)
        seed=54
        a,b=next_observations(m,f.numpy(),p.numpy(),t.numpy(),np.ones(3),2,seed)
        noise=torch.randn((3,1,2*m.dimensions+60),generator=torch.Generator().manual_seed(seed))[:,0]
        nf,np_,nt=one_second(m,f,p,t,noise)
        np.testing.assert_array_equal(nf[:,-1].numpy(),a[:,0])
        np.testing.assert_array_equal(np_[:,-1].numpy(),b[:,0])
        torch.testing.assert_close(nt[:,0],t[:,-1])
        torch.testing.assert_close(nt[:,1],t[:,-1]+1)


if __name__=='__main__':unittest.main()
