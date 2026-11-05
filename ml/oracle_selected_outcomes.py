"""GPU evaluation of requested outcomes for arbitrary attainable post-states.

Only selected edges execute the expensive future-action maximization. A cheap
all-edge close witness protects finite-support feasibility before subsampling.
The same Oracle primitives and candidate lane order as the dense sweep are used.
"""
import numpy as np
import torch
import triton
import triton.language as tl
from oracle_book_gpu import _arrive, _close, _effective, _finite, _lookup, _post, _target, _valid


@triton.jit
def _support(states, prices, rules, supported, B:tl.constexpr, K:tl.constexpr):
    row = tl.program_id(0)
    s = (); r = ()
    for j in tl.static_range(3+3*K): s += (tl.load(states+row*(3+3*K)+j),)
    for j in tl.static_range(11): r += (tl.load(rules+j),)
    okay = _valid(s,r,False,False,K)
    for j in range(B):
        arrived, consumed, filled, valid = _arrive(s,tl.load(prices+row*B+j),r,K)
        okay &= valid & _finite(_close(arrived,consumed,r))
    tl.store(supported+row,okay)


@triton.jit
def _selected(states,prices,children,indices,budgets,values,wealth,exposures,books,counts,
              offsets,book_lookup,size_edges,rules,candidates,factors,output,
              W:tl.constexpr,E:tl.constexpr,C:tl.constexpr,M:tl.constexpr,K:tl.constexpr,B:tl.constexpr,
              PB:tl.constexpr,SB:tl.constexpr,PS:tl.constexpr,NC:tl.constexpr,AC:tl.constexpr,
              WBITS:tl.constexpr,EBITS:tl.constexpr,TERMINAL:tl.constexpr,LAST:tl.constexpr):
    task = tl.program_id(0); row = task//B; draw = task%B
    result = tl.full((),float('nan'),tl.float64)
    if draw < tl.load(budgets+row):
        result = tl.full((),-float('inf'),tl.float64)
        s = (); r = ()
        for j in tl.static_range(3+3*K): s += (tl.load(states+row*(3+3*K)+j),)
        for j in tl.static_range(11): r += (tl.load(rules+j),)
        if _valid(s,r,False,False,K):
            edge = tl.load(indices+task)
            child = tl.load(children+row*B+edge)
            arrived,consumed,filled,valid = _arrive(s,tl.load(prices+row*B+edge),r,K)
            if valid:
                if TERMINAL:
                    result = _close(arrived,consumed,r)
                else:
                    lane = tl.arange(0,AC); is_catalog = (lane>0)&(lane<=NC)
                    ci = tl.load(candidates+lane-1,mask=is_catalog,other=0)
                    book = ci//E; target_index = ci%E
                    retained = lane-NC-1; mask = retained//(E+1); retain_e = retained%(E+1)-1
                    occupied = tl.full((),0,tl.int32)
                    for j in tl.static_range(K): occupied |= tl.where(arrived[3+3*j]!=0,1<<j,0)
                    in_menu = is_catalog|((retained>=0)&(mask<(1<<K))&((mask&~occupied)==0))
                    target_index = tl.where(is_catalog,target_index,tl.maximum(retain_e,0))
                    target = tl.load(exposures+target_index)
                    target = tl.where((~is_catalog)&(retain_e<0),float('nan'),target)
                    book = tl.where(is_catalog,book,0); mask = tl.where(is_catalog,-1,mask)
                    out,feasible = _target(arrived,consumed,filled,target,book,mask,books,counts,r,PS,K)
                    val = _post(out,child,values,wealth,exposures,offsets,book_lookup,size_edges,r,factors,
                        W,E,C,M,WBITS,EBITS,PB,SB,PS,K,B,LAST)
                    val = tl.where(in_menu&feasible,val,-float('inf'))
                    closed = _close(arrived,consumed,r)
                    flat = (arrived[0]+_effective(arrived,r),tl.full((),0.,tl.float64),arrived[2])
                    for j in tl.static_range(K): flat += (tl.full((),0.,tl.float64),)*3
                    if LAST: cv = tl.full((),-float('inf'),tl.float64)
                    else:
                        cv = _lookup(flat,child,values,wealth,exposures,offsets,book_lookup,size_edges,
                            W,E,C,M,WBITS,EBITS,PB,SB,PS,K)
                    cash = tl.where(_finite(closed),tl.maximum(closed,cv),-float('inf'))
                    result = tl.maximum(cash,tl.max(val,0))
    tl.store(output+task,result)


class SelectedOutcomes:
    def __init__(self, map, states, prices, children, child_field, factors, candidates, remaining):
        a,g = map.a,map.grid
        states,prices,children = tuple(np.asarray(x) for x in (states,prices,children))
        if states.ndim != 2 or states.shape[1] != 4+3*g.max_orders:
            raise ValueError('Post-state rows must include previous price and all orders')
        self.rows,self.B = prices.shape
        if states.shape[0] != self.rows or children.shape != prices.shape or self.B < 2 or remaining < 1:
            raise ValueError('Invalid conditional row shapes or horizon')
        if np.any(~np.isfinite(states)) or np.any(states[:,2]<=0) or np.any(~np.isfinite(prices)|(prices<=0)):
            raise ValueError('Nonfinite post-states or invalid prices')
        W,C,M = child_field.shape; E = len(a['exposures'])
        if (W,C) != (len(a['wealth']),len(a['coordinates'])) or factors.shape != (M,self.B):
            raise ValueError('Continuation field differs from account/market shape')
        if not np.issubdtype(children.dtype,np.integer) or np.any((children<0)|(children>=M)):
            raise ValueError('Invalid child indices')
        candidates = np.asarray(candidates,dtype=np.int64)
        if np.any((candidates<0)|(candidates>=C)): raise ValueError('Invalid action candidates')
        def tensor(x):
            if isinstance(x,torch.Tensor):
                if x.device.type!='cuda' or x.dtype!=torch.float64 or not x.is_contiguous():
                    raise ValueError('Shared continuation must be contiguous CUDA float64')
                # The caller owns this workspace and must not update it during evaluate().
                return x
            return torch.tensor(np.ascontiguousarray(x),device='cuda')
        self.inputs = [tensor(x) for x in (np.c_[states[:,:3],states[:,4:]],prices,children)]
        self.tail = [tensor(x) for x in (child_field,a['wealth'],a['exposures'],a['books'],a['book_counts'],
            a['code_offsets'],a['book_lookup'],a['size_edges'],map.rules,candidates,factors)]
        self.constants = dict(W=W,E=E,C=C,M=M,K=g.max_orders,B=self.B,PB=g.price_bins,SB=g.size_bins,
            PS=g.price_scale,NC=len(candidates),AC=triton.next_power_of_2(1+len(candidates)+(1<<g.max_orders)*(E+1)),
            WBITS=W.bit_length(),EBITS=E.bit_length(),TERMINAL=remaining==1,LAST=remaining==2)
        supported = torch.empty(self.rows,device='cuda',dtype=torch.bool)
        _support[(self.rows,)](self.inputs[0],self.inputs[1],self.tail[8],supported,
            B=self.B,K=g.max_orders,num_warps=4,enable_fp_fusion=False)
        self.supported = supported.cpu().numpy()
        self.output = torch.empty((self.rows,self.B),device='cuda',dtype=torch.float64)

    def evaluate(self, indices, counts):
        indices,counts = np.asarray(indices),np.asarray(counts)
        if (indices.shape != (self.rows,self.B) or counts.shape != (self.rows,)
                or not np.issubdtype(indices.dtype,np.integer) or not np.issubdtype(counts.dtype,np.integer)
                or np.any((indices<0)|(indices>=self.B)) or np.any((counts<1)|(counts>self.B))):
            raise ValueError('Invalid selected edge plan')
        index = torch.tensor(indices,device='cuda'); count = torch.tensor(counts,device='cuda')
        _selected[(self.rows*self.B,)](*self.inputs,index,count,*self.tail,self.output,
            **self.constants,num_warps=4,enable_fp_fusion=False)
        return self.output.cpu().numpy().copy()
