"""Run with TRITON_INTERPRET=1 for CPU-only kernel checks, or on CUDA."""
import os
import unittest
import numpy as np
import torch
import triton
import triton.language as tl
from oracle_gpu_action_compaction import compact_indices,compact_max


INTERPRET=os.environ.get('TRITON_INTERPRET')=='1'


@triton.jit
def pack_kernel(states, feasible, output, count, AC:tl.constexpr,D:tl.constexpr,HASH_MASK:tl.constexpr):
    lane=tl.arange(0,AC);coordinates=()
    for j in tl.static_range(D):coordinates+=(tl.load(states+lane*D+j),)
    indices,n=compact_indices(coordinates,tl.load(feasible+lane),AC,HASH_MASK)
    tl.store(output+lane,indices);tl.store(count,n)


@triton.jit
def example_score(states,node,args,constants:tl.constexpr):
    result=tl.full(states[0].shape,0.,tl.float64)
    for j in tl.static_range(len(states)):result+=states[j]*(j+1)
    return (result+node)*constants[0]


@triton.jit
def score_kernel(states,feasible,output,AC:tl.constexpr,D:tl.constexpr):
    lane=tl.arange(0,AC);coordinates=()
    for j in tl.static_range(D):coordinates+=(tl.load(states+lane*D+j),)
    value=compact_max(coordinates,tl.load(feasible+lane),3.,(),(2.,),AC,example_score)
    tl.store(output,value)


@unittest.skipUnless(INTERPRET or torch.cuda.is_available(),'CUDA or TRITON_INTERPRET=1 required')
class ActionCompactionTests(unittest.TestCase):
    def compare(self,states,feasible,hash_mask=0x7fffffffffffffff):
        device='cpu' if INTERPRET else 'cuda';A,D=states.shape
        coordinates=torch.tensor(states,device=device,dtype=torch.float64)
        mask=torch.tensor(feasible,device=device,dtype=torch.bool)
        output=torch.empty(A,device=device,dtype=torch.int32);count=torch.empty((),device=device,dtype=torch.int32)
        pack_kernel[(1,)](coordinates,mask,output,count,A,D,hash_mask,num_warps=4,enable_fp_fusion=False)
        indices=output.cpu().numpy();n=int(count.cpu())
        self.assertTrue(np.all(indices[n:]==A))
        self.assertTrue(np.all((indices[:n]>=0)&(indices[:n]<A)))
        self.assertTrue(np.all(feasible[indices[:n]]))
        expected={tuple(s) for s in states[feasible]};actual={tuple(states[i]) for i in indices[:n]}
        self.assertEqual(actual,expected)
        self.assertGreaterEqual(n,len(expected));self.assertLessEqual(n,int(feasible.sum()))
        result=torch.empty((),device=device,dtype=torch.float64)
        score_kernel[(1,)](coordinates,mask,result,A,D,num_warps=4,enable_fp_fusion=False)
        reference=float(np.max((states[feasible]@np.arange(1,D+1)+3.)*2.)) if feasible.any() else -np.inf
        np.testing.assert_allclose(float(result.cpu()),reference,atol=1e-12,rtol=0)
        return n

    def test_empty_and_single_feasible_candidate(self):
        states=np.arange(64*9,dtype=float).reshape(64,9)
        mask=np.zeros(64,dtype=bool);self.assertEqual(self.compare(states,mask),0)
        mask[-1]=True;self.assertEqual(self.compare(states,mask),1)

    def test_duplicates_across_all_coordinates_and_feasibility(self):
        rng=np.random.default_rng(918)
        for A in (4,8,32,64,256):
            states=rng.normal(size=(A,9));states[1::2]=states[::2]
            mask=rng.random(A)>.2
            n=self.compare(states,mask)
            self.assertEqual(n,len({tuple(s) for s in states[mask]}))

    def test_hash_collisions_do_not_remove_distinct_states(self):
        rng=np.random.default_rng(17);states=rng.normal(size=(64,9));states[17]=states[5]
        self.compare(states,np.ones(64,dtype=bool),hash_mask=0)

    def test_order_coordinates_and_signed_zero(self):
        states=np.tile(np.array([10000.,0.,100.,1.,99.,100.,-1.,101.,100.]),(32,1))
        states[1,4]=98.;states[2,5]=200.;states[3,7]=102.;states[4,8]=150.;states[5,1]=-0.
        mask=np.ones(32,dtype=bool)
        self.assertEqual(self.compare(states,mask),5)


if __name__=='__main__':unittest.main()
