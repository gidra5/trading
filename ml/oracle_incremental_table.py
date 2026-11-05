"""Resident dense values with bounded interior-cell Monte Carlo updates.

The time axis is a GPU ring. Forward/backward update fronts share the same
table; each local backup still averages outcomes after future maximization.
This is an empirical approximation. Only the full census/sweep mode reproduces
the complete finite Bellman operator; no global certificate is claimed for a
bounded asynchronous schedule.
"""
from collections import OrderedDict
from dataclasses import dataclass
from types import SimpleNamespace
import time
import numpy as np
import torch
from numba import njit
from oracle_book_gpu import active_account_indices
from oracle_book_mc import representative
from oracle_guided_sampling import sample_plan, corrected_estimate
from oracle_local_market_refresh import candidate_schedule
from oracle_selected_outcomes import SelectedOutcomes
from oracle_root_candidates import root_candidates
from oracle_root_aliases import exact_root_aliases
from oracle_account_grid import CellErrors


@njit(cache=True)
def representative_rows(row_ids,prices,wealth,exposures,books,counts,rules,price_scale,M):
    C = len(exposures)*len(books)
    states = np.zeros((len(row_ids),4+3*books.shape[1]))
    valid = np.zeros(len(row_ids),dtype=np.bool_)
    for j in range(len(row_ids)):
        row = row_ids[j]; node = row%M; cell = (row//M)%C; wi = row//(M*C)
        book,ei = cell//len(exposures),cell%len(exposures)
        states[j],valid[j] = representative(wealth[wi],exposures[ei],books[book],counts[book],
            prices[node],rules,price_scale)
    return states,valid


@dataclass(frozen=True)
class TableBudget:
    layers: int = 4
    rows: int = 512
    draws: int = 4
    exploration: float = .2
    cached_rows: int = 100000

    def __post_init__(self):
        for name in ('layers','rows','draws','cached_rows'):
            x = getattr(self,name)
            if not isinstance(x,int) or isinstance(x,bool) or x<1: raise ValueError('Invalid '+name)
        if not 0<self.exploration<=1: raise ValueError('Invalid exploration mixture')


class IncrementalTable:
    def __init__(self, initial, seed, candidates, budget=TableBudget()):
        from oracle_incremental_checkpoint import signatures
        self.source_identity = signatures()
        self.grid,self.a,self.rules = initial.grid,initial.a,initial.rules
        self.cloud = initial.cloud
        self.capacity,self.W,self.C,self.M = initial.shape
        self.H = self.capacity; self.head = 0; self.update = 0
        self.B = initial.cloud.destinations.shape[-1]
        if budget.draws>self.B: raise ValueError('Draw count exceeds conditional row size')
        self.budget,self.seed,self.candidates = budget,seed,candidates
        self.values = torch.tensor(np.asarray(initial.values),device='cuda',dtype=torch.float64)
        # Dense metadata, with a bounded cache of individual control outcomes.
        self.visits = np.zeros(initial.shape,dtype=np.uint32)
        self.sampling_variance = np.zeros(initial.shape,dtype=np.float32)
        self.controls = OrderedDict()
        self.root_controls = None
        # Observed errors share the relative account coordinates across time
        # and market samples; one actual tick counts at most once per cell.
        self.errors = CellErrors.empty(self.W*self.C)
        self.failed = False

    @property
    def shape(self): return (self.H,self.W,self.C,self.M)

    def slot(self, depth):
        if not 0<=depth<self.H: raise ValueError('Depth outside current horizon')
        return (self.head+depth)%self.capacity

    def field(self, depth): return self.values[self.slot(depth)]

    def facade(self):
        return SimpleNamespace(grid=self.grid,a=self.a,rules=self.rules,cloud=self.cloud,shape=self.shape)

    def schedule(self):
        # Two exact endpoint layers. Interior fronts rotate through the horizon;
        # depth one also receives frequent work because it feeds today's choice.
        ordinary = max(0,self.H-3)
        if not ordinary: return []
        forward = (self.budget.layers+1)//2; backward = self.budget.layers//2
        selected = {1}
        for j in range(forward): selected.add(1+((self.update-1)*forward+j)%ordinary)
        for j in range(backward): selected.add(self.H-3-((self.update-1)*backward+j)%ordinary)
        return sorted(selected,reverse=True)

    def refresh_layer(self, depth, full=False, repair_support=False):
        a,g,c = self.a,self.grid,self.cloud
        slot = self.slot(depth); n = int(c.counts[depth]); remaining = self.H-depth
        active = active_account_indices(a['wealth'],a['exposures'],a['books'],a['book_counts'],
            c.prices[depth,:n],self.rules,g.price_scale)
        row_ids = (active[:,None]*self.M+np.arange(n)[None]).ravel()
        complete = full or remaining<=2
        if repair_support:
            previous = self.values[slot].reshape(-1)[torch.tensor(row_ids,device='cuda')].cpu().numpy()
            row_ids = row_ids[~np.isfinite(previous)]
        if not complete and not repair_support and len(row_ids)>self.budget.rows:
            rng = np.random.default_rng(np.random.SeedSequence([self.seed,self.update,depth,0xCE11]))
            account = row_ids//self.M
            priority = abs(self.errors.mean[account])+np.sqrt(
                np.maximum(self.errors.m2[account],0)/np.maximum(self.errors.count[account]-1,1))
            weights = None
            if np.any(priority>0):
                # Positive exploration keeps every account row eligible.
                scaled = priority/priority.max()
                weights = self.budget.exploration/len(row_ids)+(1-self.budget.exploration)*scaled/scaled.sum()
            row_ids = np.sort(rng.choice(row_ids,self.budget.rows,replace=False,p=weights))
        if complete:
            self.values[slot].fill_(-float('inf'))
            self.values[slot,:,:,n:] = float('nan')
            self.sampling_variance[slot].fill(0.)
        if not len(row_ids): return dict(depth=depth,rows=0,outcomes=0,partialRows=0,complete=complete,
            supportRepair=repair_support,promoted=0)
        states,valid = representative_rows(row_ids,c.prices[depth],a['wealth'],a['exposures'],a['books'],
            a['book_counts'],self.rules,g.price_scale,self.M)
        invalid = row_ids[~valid]
        if len(invalid):
            self.values[slot].view(-1)[torch.tensor(invalid,device='cuda')] = -float('inf')
            self.sampling_variance[slot].reshape(-1)[invalid] = 0.
        row_ids,states = row_ids[valid],states[valid]
        if not len(row_ids): return dict(depth=depth,rows=0,outcomes=0,partialRows=0,complete=complete,
            supportRepair=repair_support,promoted=0)
        nodes = row_ids%self.M
        # One shared cash-reference row per market node, never an extra action.
        cash = np.zeros((n,states.shape[1])); cash[:,0] = a['wealth'][self.W//2]
        cash[:,2] = c.prices[depth,:n]; cash[:,3] = cash[:,2]
        states = np.r_[states,cash]; all_nodes = np.r_[nodes,np.arange(n)]
        child = self.field(depth+1) if remaining>1 else self.field(depth)
        factors = np.zeros((self.M,self.B))
        if remaining>1:
            raw = c.factors(depth+1); factors[:len(raw)] = raw
        evaluator = SelectedOutcomes(self.facade(),states,c.sampled_prices[depth,all_nodes],
            c.destinations[depth,all_nodes],child,factors,
            candidate_schedule(self.facade(),depth,self.seed,self.candidates),remaining)
        absolute = int(c.origin_seconds)+depth
        # An uncached row gets one full census. Invented zero controls followed
        # by partially filled vectors can create variance even on flat paths.
        controls = np.full((len(states),self.B),np.nan)
        keys = [(absolute,int(row)) for row in row_ids]
        if not complete:
            for i,key in enumerate(keys):
                if key in self.controls:
                    controls[i] = self.controls[key]; self.controls.move_to_end(key)
        supported = evaluator.supported.copy(); supported[-n:] = False
        plan = sample_plan(controls,supported,self.B if complete or repair_support else self.budget.draws,
            np.random.SeedSequence([self.seed,self.update,depth,0xD2A7]),self.budget.exploration)
        y = evaluator.evaluate(plan.indices,plan.counts)
        reference = y[-n:]
        selected_reference = np.take_along_axis(reference[all_nodes],plan.indices,axis=1)
        # Account wealth is known exactly. Remove its log offset before caching
        # partial controls, or unvisited edges manufacture account-size noise.
        offset = np.log((states[:,0]+states[:,1]*states[:,2])/cash[0,0])
        centered = y-selected_reference-offset[:,None]
        estimate,variance,updated = corrected_estimate(controls,centered,plan)
        estimate += reference.mean(1)[all_nodes]+offset
        result = estimate[:len(row_ids)]-np.log(a['wealth'][row_ids//(self.C*self.M)])
        device_ids = torch.tensor(row_ids,device='cuda')
        self.values[slot].view(-1)[device_ids] = torch.tensor(result,device='cuda')
        self.visits[slot].reshape(-1)[row_ids] += plan.counts[:len(row_ids)].astype(np.uint32)
        self.sampling_variance[slot].reshape(-1)[row_ids] = variance[:len(row_ids)]
        if not complete:
            for key,value in zip(keys,updated):
                if repair_support and not np.isfinite(value).all(): continue
                self.controls[key] = value.copy(); self.controls.move_to_end(key)
            while len(self.controls)>self.budget.cached_rows: self.controls.popitem(last=False)
        return dict(depth=depth,rows=len(row_ids),outcomes=int(plan.counts.sum()),
            partialRows=int(np.sum(plan.counts<self.B)),complete=complete,supportRepair=repair_support,
            promoted=int(np.isfinite(result).sum()) if repair_support else 0)

    def advance(self, cloud, full=False):
        if self.failed: raise RuntimeError('Incremental table failed; restore a complete checkpoint')
        H = len(cloud.counts)-1
        if (cloud.origin_seconds!=self.cloud.origin_seconds+1 or H not in (self.H,self.H-1)
                or cloud.prices.shape[1]!=self.M or cloud.destinations.shape[-1]!=self.B):
            raise ValueError('Expected aligned next-second cloud')
        # Reuse only unchanged interior conditional kernels.
        end = H if H==self.H else H+1
        for name in ('prices','counts','features','price_pairs'):
            if not np.array_equal(getattr(cloud,name)[1:end],getattr(self.cloud,name)[2:end+1]):
                raise ValueError('Changed interior market states')
        for name in ('destinations','sampled_prices'):
            if not np.array_equal(getattr(cloud,name)[1:end-1],getattr(self.cloud,name)[2:end]):
                raise ValueError('Changed interior conditional laws')
        started = time.perf_counter(); rolling = H==self.H
        self.head = (self.head+1)%self.capacity; self.H = H; self.cloud = cloud; self.update += 1
        if rolling:
            self.visits[self.slot(H-1)].fill(0); self.sampling_variance[self.slot(H-1)].fill(0.)
        expired = [key for key in self.controls if key[0]<cloud.origin_seconds]
        for key in expired: del self.controls[key]
        try:
            if full: records = [self.refresh_layer(depth,full=True) for depth in range(H-1,-1,-1)]
            else:
                endpoint = [H-1,H-2] if rolling and H>=2 else []
                records = [self.refresh_layer(depth) for depth in endpoint]
                if rolling:
                    # An extended deadline can make formerly impossible closes
                    # feasible. Propagate this discrete change until the next
                    # support mask is unchanged, using complete outcome rows.
                    for depth in range(H-3,0,-1):
                        repaired = self.refresh_layer(depth,repair_support=True)
                        records.append(repaired)
                        if repaired['promoted']==0: break
                records.extend(self.refresh_layer(depth) for depth in self.schedule())
            torch.cuda.synchronize()
        except Exception:
            self.failed = True
            raise
        return dict(seconds=time.perf_counter()-started,updates=records,
            updatedLayers=len(records),updatedRows=sum(x['rows'] for x in records),
            evaluatedOutcomes=sum(x['outcomes'] for x in records),cachedRows=len(self.controls),
            residentValueBytes=self.values.numel()*self.values.element_size(),
            scope='Bounded asynchronous finite-table updates; sampling variances are local, not a global error certificate')

    def cpu_values(self):
        return np.stack([self.field(h).cpu().numpy() for h in range(self.H)])

    def choose(self, arrived, consumed=0., filled=False, draws=None, reference_field=None):
        if self.failed: raise RuntimeError('Incremental table failed')
        started = time.perf_counter()
        draws = self.B if draws is None else draws
        ids,states,slots = root_candidates(self.facade(),arrived,consumed,filled,seed=self.seed+self.update)
        action_rows = len(states)
        cash = np.r_[self.a['wealth'][self.W//2],0.,arrived[2],arrived[2],np.zeros(3*self.grid.max_orders)]
        ids = np.r_[ids,slots]; states = np.r_[states,cash[None]]
        # One full root census shares its law, rules and child field across
        # descriptors. Partial plans can sample different edges per descriptor.
        if draws==self.B:
            first,inverse = exact_root_aliases(states)
        else:
            first = inverse = np.arange(len(states))
        evaluated_states = states[first]
        if self.root_controls is None: self.root_controls = np.full((slots+1,self.B),np.nan)
        factors = np.zeros((self.M,self.B))
        if self.H>1:
            raw = self.cloud.factors(1); factors[:len(raw)] = raw
        field = self.field(1) if self.H>1 else self.field(0)
        def make(child):
            return SelectedOutcomes(self.facade(),evaluated_states,np.repeat(self.cloud.sampled_prices[0,:1],len(first),axis=0),
                np.repeat(self.cloud.destinations[0,:1],len(first),axis=0),child,factors,
                candidate_schedule(self.facade(),0,self.seed,self.candidates),self.H)
        evaluator = make(field)
        controls = self.root_controls[ids]
        support = evaluator.supported[inverse].copy(); support[-1] = False
        plan = sample_plan(controls,support,draws,np.random.SeedSequence([self.seed,self.update,0xA071]),self.budget.exploration)
        y = evaluator.evaluate(plan.indices[first],plan.counts[first])[inverse]
        offset = np.log((states[:,0]+states[:,1]*states[:,2])/cash[0])
        centered = y-np.take_along_axis(np.broadcast_to(y[-1],y.shape),plan.indices,axis=1)-offset[:,None]
        value,variance,updated = corrected_estimate(controls,centered,plan)
        value += y[-1].mean()+offset
        self.root_controls[ids] = updated
        best = int(value[:action_rows].argmax())
        if not np.isfinite(value[best]): raise RuntimeError('No finite root action under sampled continuation')
        scores = np.full(slots,-np.inf); proposals = np.zeros((slots,states.shape[1]))
        scores[ids[:-1]] = value[:-1]; proposals[ids[:-1]] = states[:-1]
        front = dict(value=float(value[best]),state=states[best],choice=int(ids[best]),scores=scores,states=proposals)
        report = dict(seconds=time.perf_counter()-started,proposals=action_rows,
            uniqueProposals=int(len(np.unique(inverse[:action_rows]))),
            evaluatedOutcomes=int(plan.counts[first].sum()),logicalOutcomes=int(plan.counts.sum()),
            partialRows=int(np.sum(plan.counts<self.B)))
        if reference_field is not None:
            reference = make(reference_field).evaluate(np.tile(np.arange(self.B),(len(first),1)),
                np.full(len(first),self.B,dtype=np.int64))[inverse].mean(1)[:action_rows]
            if not np.isfinite(reference).any(): raise ValueError('No finite reference root action')
            optimum = int(reference.argmax())
            report['reference'] = dict(selectedFinite=bool(np.isfinite(reference[best])),
                sameAction=bool(np.array_equal(states[best],states[optimum])),
                regretBps=float(-np.expm1(reference[best]-reference[optimum])*10000),
                selectedValue=float(reference[best]),bestValue=float(reference[optimum]))
        return front,report
