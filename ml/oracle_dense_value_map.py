"""Dense account cells with shared sampled, conditional market-state clouds.

Values are E[log terminal wealth] - log current equity at the post-decision
phase. Account cells are dense; market nodes are sample representatives. The
first implementation uses bilinear wealth/exposure interpolation and constant
order-bin interpolation. Neither is a certified continuous Bellman solution.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from oracle_account_grid import AccountGrid, CellErrors
from oracle_book_mc import (backup_layer, decide, post_value, execute_target, move_arrive)


@dataclass(frozen=True)
class MarketCloud:
    prices: np.ndarray          # [time, node], last completed observed price
    counts: np.ndarray          # valid market nodes per time
    destinations: np.ndarray    # [time, parent, conditional sample]
    masses: np.ndarray         # display only; never used in conditional backups
    features: np.ndarray       # [time, node, two observations, 59]
    price_pairs: np.ndarray
    origin_seconds: float
    sampled_prices: np.ndarray  # [time, parent, sample]; execution is never projected

    def __post_init__(self):
        if self.sampled_prices.shape!=self.destinations.shape:
            raise ValueError('Sampled execution prices must match conditional edges')
        H,M,B=self.sampled_prices.shape
        if self.prices.shape!=(H+1,M) or self.counts.shape!=(H+1,):raise ValueError('Invalid market layer shapes')
        if np.any((self.counts<1)|(self.counts>M)):raise ValueError('Invalid node counts')
        valid=np.arange(M)[None,:]<self.counts[:-1,None]
        raw=self.sampled_prices[valid]
        if np.any(~np.isfinite(raw)|(raw<=0)):raise ValueError('Execution prices must be positive and finite')
        dest=self.destinations
        if np.any(valid[:,:,None]&((dest<0)|(dest>=self.counts[1:,None,None]))):
            raise ValueError('Invalid continuation destinations')

    @classmethod
    def from_graph(cls,graph):
        counts=np.array([len(x.prices) for x in graph.layers]);H=len(counts)-1
        M=int(counts.max());B=graph.destinations[0].shape[1]
        prices=np.zeros((H+1,M));dest=np.zeros((H,M,B),dtype=np.int64)
        sampled=np.zeros((H,M,B))
        masses=np.zeros((H+1,M));features=np.zeros((H+1,M,2,59),dtype=np.float32)
        pairs=np.zeros((H+1,M,2))
        for h,layer in enumerate(graph.layers):
            n=counts[h];prices[h,:n]=layer.prices[:,-1];masses[h,:n]=layer.mass.cpu().numpy()
            features[h,:n]=layer.features;pairs[h,:n]=layer.prices
            if h<H:
                weights=graph.weights[h].cpu().numpy()
                if not np.allclose(weights,1/B):raise ValueError('Require equal-weight sample rows')
                dest[h,:n]=graph.destinations[h].cpu().numpy()
                sampled[h,:n]=graph.sampled_prices[h].cpu().numpy()
        return cls(prices,counts,dest,masses,features,pairs,float(graph.layers[0].times[0,-1]),sampled)

    def save(self,directory):
        np.savez_compressed(Path(directory)/'market.npz',prices=self.prices,counts=self.counts,
            destinations=self.destinations,masses=self.masses,features=self.features,
            price_pairs=self.price_pairs,origin_seconds=self.origin_seconds,sampled_prices=self.sampled_prices)

    @classmethod
    def load(cls,path):
        with np.load(path) as data:
            if 'sampled_prices' not in data:raise ValueError('Market archive lacks raw sampled prices; regenerate it')
            return cls(data['prices'],data['counts'],data['destinations'],data['masses'],
                data['features'],data['price_pairs'],float(data['origin_seconds']),data['sampled_prices'])

    def factors(self,depth):
        # Reanchor the conditional relative moves at the queried account's
        # actual mark. The node remains a continuation-state approximation.
        n=self.counts[depth]
        return self.sampled_prices[depth,:n]/self.prices[depth,:n,None]

    def nearest(self,depth,features,prices,feature_std):
        n=self.counts[depth]
        df=(self.features[depth,:n]-features)/feature_std
        dp=np.log(self.price_pairs[depth,:n]/prices)*10000
        return int((np.mean(df**2,axis=(1,2))+np.sum(dp**2,axis=1)).argmin())


def finite_average(x,axis):
    mask=np.isfinite(x);n=mask.sum(axis=axis)
    return np.divide(np.where(mask,x,0).sum(axis=axis),n,out=np.full(n.shape,np.nan),where=n>0)


class DenseValueMap:
    def __init__(self,grid:AccountGrid,cloud:MarketCloud,rules,directory,errors=None):
        self.grid=grid;self.a=grid.arrays();self.cloud=cloud;self.rules=rules
        self.directory=Path(directory) if directory is not None else None
        if self.directory is not None:self.directory.mkdir(parents=True,exist_ok=False)
        H=len(cloud.counts)-1;W=len(self.a['wealth']);C=len(self.a['coordinates']);M=cloud.prices.shape[1]
        self.shape=(H,W,C,M);self.errors=errors if errors is not None else CellErrors.empty(C);self.writable=True
        if self.directory is None:
            self.values=np.empty(self.shape);self.variance=np.empty(self.shape)
        else:
            self.values=np.lib.format.open_memmap(self.directory/'values.npy',mode='w+',dtype='float64',shape=self.shape)
            self.variance=np.lib.format.open_memmap(self.directory/'variance.npy',mode='w+',dtype='float64',shape=self.shape)
        self.samples=np.zeros((H,C),dtype=np.int16)
        self.outcome_cache=None
        self.outcome_representation=None
        self.reuse_generation=0
        self.bellman_contract=None;self.bellman_residuals=None
        self.bellman_metadata=None;self.scheduled_accounts=None
        if self.directory is not None:
            self.cloud.save(self.directory)
            np.savez_compressed(self.directory/'account-grid.npz',**self.a)

    def _save_metadata(self,directory):
        # Bind archived values to the complete implemented Bellman operator,
        # including GPU/driver changes outside the CPU execution primitives.
        names=('oracle_book_mc.py','oracle_book_gpu.py','oracle_dense_value_map.py',
            'oracle_account_grid.py','oracle_dense_repair.py')
        signature={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in names}
        (directory/'operator.json').write_text(json.dumps(signature,indent=2))
        if self.outcome_cache is not None:
            (directory/'reuse.json').write_text(json.dumps(dict(generation=self.reuse_generation,
                representation=self.outcome_representation)))
        if self.bellman_metadata is not None:
            np.save(directory/'bellman-residuals.npy',self.bellman_residuals)
            (directory/'bellman.json').write_text(json.dumps(self.bellman_metadata))
        if self.scheduled_accounts is not None:np.save(directory/'scheduled-accounts.npy',self.scheduled_accounts)
        np.savez(directory/'errors.npz',count=self.errors.count,mean=self.errors.mean,m2=self.errors.m2)
        np.save(directory/'samples.npy',self.samples)

    def export(self,directory):
        """Archive an in-memory map explicitly, not on every observed second."""
        directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
        self.cloud.save(directory);np.savez_compressed(directory/'account-grid.npz',**self.a)
        np.save(directory/'values.npy',self.values);np.save(directory/'variance.npy',self.variance)
        if self.outcome_cache is not None:np.save(directory/'outcomes.npy',self.outcome_cache)
        self._save_metadata(directory)

    @classmethod
    def load(cls,directory):
        """Open a completed experiment field for decisions or refresh priorities.

        The archived grid and execution contract are checked before querying.
        Value arrays are mapped read-only; a refresh creates its own new map.
        """
        from gpu_oracle_bellman import Rules
        from oracle_book_mc import rule_vector
        directory=Path(directory);plan=json.loads((directory.parent/'plan.json').read_text())
        operator_file=directory/'operator.json'
        if not operator_file.exists():raise ValueError('Saved field lacks a complete Bellman operator signature; regenerate')
        signature=json.loads(operator_file.read_text())
        names=('oracle_book_mc.py','oracle_book_gpu.py','oracle_dense_value_map.py',
            'oracle_account_grid.py','oracle_dense_repair.py')
        expected={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in names}
        if signature!=expected:raise ValueError('Saved field has a different Bellman operator; regenerate')
        kernel=Path(__file__).with_name('oracle_book_mc.py')
        if hashlib.sha256(kernel.read_bytes()).hexdigest()!=plan['sourceHashes']['ml/oracle_book_mc.py']:
            raise ValueError('Saved field has a different execution/value kernel')
        if not (directory/'samples.npy').exists() or not (directory/'errors.npz').exists():
            raise ValueError('Value sweep is incomplete')
        grid_config=dict(plan['grid'])
        for key in ('exposure_points','wealth_points'):grid_config[key]=tuple(grid_config[key])
        result=cls.__new__(cls);result.directory=directory;result.grid=AccountGrid(**grid_config)
        result.a=result.grid.arrays();result.cloud=MarketCloud.load(directory/'market.npz')
        result.rules=rule_vector(Rules(**plan['rules']),plan['fundingMultiplier'])
        result.values=np.load(directory/'values.npy',mmap_mode='r')
        result.variance=np.load(directory/'variance.npy',mmap_mode='r')
        result.samples=np.load(directory/'samples.npy');result.shape=result.values.shape;result.writable=False
        outcome_file=directory/'outcomes.npy'
        result.outcome_cache=np.load(outcome_file,mmap_mode='r') if outcome_file.exists() else None
        result.outcome_representation=None
        result.reuse_generation=0
        result.bellman_contract=None;result.bellman_residuals=None
        result.bellman_metadata=None;result.scheduled_accounts=None
        if (directory/'scheduled-accounts.npy').exists():
            result.scheduled_accounts=np.load(directory/'scheduled-accounts.npy')
        if (directory/'bellman.json').exists():
            metadata=json.loads((directory/'bellman.json').read_text())
            result.bellman_metadata=metadata
            result.bellman_contract=metadata['contract']
            result.bellman_residuals=np.load(directory/'bellman-residuals.npy')
        if result.outcome_cache is not None:
            reuse=json.loads((directory/'reuse.json').read_text())
            result.reuse_generation=reuse['generation'];result.outcome_representation=reuse.get('representation','raw')
        expected=(len(result.cloud.counts)-1,len(result.a['wealth']),len(result.a['coordinates']),result.cloud.prices.shape[1])
        if result.shape!=expected or result.variance.shape!=expected:raise ValueError('Saved field shape differs from grid')
        if result.outcome_cache is not None and result.outcome_cache.shape!=expected+(result.cloud.destinations.shape[-1],):
            raise ValueError('Saved outcome-cache shape differs from grid')
        with np.load(directory/'errors.npz') as errors:
            result.errors=CellErrors(errors['count'],errors['mean'],errors['m2'])
        return result

    def lookup(self,state,depth=0,node=0):
        a=self.a;g=self.grid
        factors=self.cloud.factors(depth)
        field=self.values[depth] if depth+1<self.shape[0] else np.full_like(self.values[depth],-np.inf)
        return post_value(state,node,field,a['wealth'],a['exposures'],a['code_offsets'],
            a['book_lookup'],a['size_edges'],g.price_bins,g.price_scale,self.rules,factors)

    def query_prices(self,price,depth,node):
        # Nearest-cloud inference approximates relative moves. Using the saved
        # absolute child price with a different actual current price would
        # manufacture an immediate jump back to the cloud's price level.
        return np.maximum(1,np.rint(self.cloud.factors(depth)[node]*price/self.rules[9]))*self.rules[9]

    def decide(self,arrived,consumed=0.,filled=False,depth=0,node=0,candidates=None):
        a=self.a;g=self.grid
        if candidates is None:candidates=np.arange(len(a['coordinates']))
        factors=self.cloud.factors(depth)
        # At the final decision, evaluate actual candidate closing values on
        # all next-price samples. Interpolation cannot relax terminal validity.
        field=self.values[depth] if depth+1<self.shape[0] else np.full_like(self.values[depth],-np.inf)
        return decide(arrived,consumed,filled,node,field,a['wealth'],a['exposures'],a['books'],
            a['book_counts'],a['code_offsets'],a['book_lookup'],a['size_edges'],g.price_bins,g.price_scale,
            self.rules,candidates,factors)

    def sweep(self,seed,candidate_count=16,budget=None,previous=None,shift=1,on_progress=None,backend='auto',
              estimator='sampled',feature_std=None,repair_tolerance=1.e-6,repair_interval=8):
        if not self.writable:raise ValueError('Loaded field is read-only; refresh into a new map')
        H,W,C,M=self.shape;a=self.a;g=self.grid;B=self.cloud.destinations.shape[-1]
        if budget is None:budget=B
        if not 2<=budget<=B or candidate_count<1:
            raise ValueError('Require 2..B samples and a positive candidate count')
        if estimator not in ('sampled','paired','repair'):raise ValueError('Unknown estimator')
        repairing=estimator=='repair'
        if repairing and (budget!=B or not np.isfinite(repair_tolerance) or repair_tolerance<0 or repair_interval<1):
            raise ValueError('Repair needs a full conditional census and a nonnegative finite tolerance')
        if previous is not None and (previous.grid!=self.grid or not np.array_equal(previous.rules,self.rules)):
            raise ValueError('Cannot reuse priorities across different account contracts')
        if previous is self:raise ValueError('Refresh requires a separate destination map')
        self.bellman_contract=None;self.bellman_residuals=None
        self.bellman_metadata=None
        if self.directory is not None:
            for name in ('samples.npy','errors.npz','bellman.json','bellman-residuals.npy'):
                (self.directory/name).unlink(missing_ok=True)
        child=np.zeros((W,C,M));started=time.perf_counter();alloc_seconds=0.
        if backend not in ('auto','cpu','cuda'):raise ValueError('Unknown sweep backend')
        if backend=='auto':
            import torch
            backend='cuda' if torch.cuda.is_available() else 'cpu'
        if estimator=='paired' and backend!='cuda':raise ValueError('Paired outcome reuse requires the CUDA backend')
        if repairing and backend!='cuda':raise ValueError('Tail repair requires CUDA')
        old_values=None;repair_stop=None;repair_error=0.;repair_delta=0.;roundoff=0.;repair_checks=0
        if repairing and previous is not None:
            from oracle_dense_repair import validate_repair
            if shift!=1:raise ValueError('Tail repair requires a one-second shift')
            roundoff=validate_repair(self,previous,seed,candidate_count)
        gpu=None
        if backend=='cuda':
            from oracle_book_gpu import GPUBookSweep
            active_hint=None
            if repairing and previous is not None and previous.scheduled_accounts is not None:
                from oracle_book_gpu import active_account_indices
                # Interior parent marks are unchanged. Only the actual root
                # and the old terminal layer introduce new parent prices.
                extra_prices=np.unique(np.r_[self.cloud.prices[0,:self.cloud.counts[0]],
                    self.cloud.prices[-2,:self.cloud.counts[-2]]])
                extra=active_account_indices(a['wealth'],a['exposures'],a['books'],a['book_counts'],extra_prices,self.rules,g.price_scale)
                active_hint=np.union1d(previous.scheduled_accounts,extra)
            gpu=GPUBookSweep(g,self.cloud,self.rules,previous,shift,estimator=='paired',feature_std,active_hint)
            self.scheduled_accounts=gpu.active_indices
            if repairing and previous is not None:old_values=gpu.tensor(previous.values)
            if repairing and previous is not None and previous.shape[0]==H+1:
                residuals=np.r_[0.,previous.bellman_residuals[2:]]
                seams=np.flatnonzero(residuals)
                allowance=float(residuals.sum()+(int(seams[-1])+1)*roundoff) if len(seams) else 0.
                if allowance<=repair_tolerance:
                    gpu.values[1:].copy_(old_values[2:])
                    gpu.variance[1:].copy_(gpu.tensor(previous.variance[2:]))
                    self.samples[1:]=previous.samples[2:]
                    self.bellman_residuals=residuals
                    repair_stop=H;repair_error=allowance
        if estimator=='paired':
            self.outcome_cache=(np.empty(self.shape+(B,)) if self.directory is None else
                np.lib.format.open_memmap(self.directory/'outcomes.npy',mode='w+',dtype='float64',shape=self.shape+(B,)))
            self.outcome_representation='cash-centered-v1'
            self.reuse_generation=previous.reuse_generation+1 if previous is not None else 0
        for h in range(H-1,-1,-1):
            if repair_stop is not None and h>0:continue
            rng=np.random.default_rng(seed+int(self.cloud.origin_seconds)+h)
            # Candidate generation is independent of all current outcome draws.
            # The zero-order book is always searched; other cells rotate.
            E=len(a['exposures']);extra=max(0,min(C-E,candidate_count-E))
            candidates=np.r_[np.arange(E),rng.choice(np.arange(E,C),extra,replace=False)].astype(np.int64)
            n=int(self.cloud.counts[h]);child_n=int(self.cloud.counts[h+1])
            sample_rng=(np.random.default_rng(np.random.SeedSequence([seed,int(self.cloud.origin_seconds)+h,
                int(self.cloud.origin_seconds),self.reuse_generation,0xCACE])) if estimator=='paired' else rng)
            permutations=np.array([sample_rng.permutation(B) for _ in range(n)])
            if previous is None or h+shift>=previous.shape[0] or budget>=B:
                allocation=np.full(C,min(B,budget),dtype=np.int64)
            else:
                t=time.perf_counter();old=h+shift
                mean=finite_average(previous.values[old],(0,2))
                var=finite_average(previous.variance[old],(0,2))
                allocation=self.errors.allocate(mean,var,B,budget,seed+h)
                alloc_seconds+=time.perf_counter()-t
            self.samples[h]=allocation
            if gpu is not None:
                gpu.step(h,permutations,allocation,candidates)
                # Rolling the endpoint changes both the terminal-arrival
                # operator and the preceding exact-close decision operator.
                # Neither old layer may enter the copied prefix.
                if (old_values is not None and 2<=h<=H-3 and (H-h)%repair_interval==0):
                    import torch
                    from oracle_dense_repair import repair_bound,copy_residuals
                    x=gpu.values[h];y=old_values[h+1];finite=torch.isfinite(x)&torch.isfinite(y)
                    supported=torch.all(finite|(torch.isnan(x)&torch.isnan(y))|((x==y)&(~torch.isfinite(x))))
                    delta=float(torch.where(finite,(x-y).abs(),0.).max().item())
                    repair_checks+=1
                    allowance=repair_bound(previous.bellman_residuals,h,delta,roundoff)
                    if bool(supported.item()) and allowance<=repair_tolerance:
                        gpu.values[1:h].copy_(old_values[2:h+1])
                        gpu.variance[1:h].copy_(gpu.tensor(previous.variance[2:h+1]))
                        self.samples[1:h]=previous.samples[2:h+1]
                        self.bellman_residuals=copy_residuals(previous.bellman_residuals,h,delta,roundoff)[:H]
                        repair_stop=h;repair_error=allowance;repair_delta=delta
            else:
                next_prices=(self.cloud.factors(h+1)
                    if h+1<H else np.zeros((child_n,B)))
                # At the last future decision, every candidate must execute
                # its actual terminal close. A representative's terminal
                # value cannot substitute for that candidate's feasibility.
                continuation=np.full_like(child,-np.inf) if h==H-2 else child
                result,var=backup_layer(self.cloud.prices[h,:n],self.cloud.sampled_prices[h,:n],
                    self.cloud.destinations[h,:n],permutations,allocation,continuation[:,:,:child_n],
                    a['wealth'],a['exposures'],a['books'],a['book_counts'],a['code_offsets'],a['book_lookup'],
                    a['size_edges'],g.price_bins,g.price_scale,self.rules,candidates,h==H-1,next_prices)
                self.values[h]=np.nan;self.variance[h]=np.nan
                self.values[h,:,:,:n]=result;self.variance[h,:,:,:n]=var
                child=result
            if on_progress is not None and ((H-h)%60==0 or h==0):
                if gpu is not None:gpu.synchronize()
                on_progress(dict(depth=h,completed=H-h,seconds=time.perf_counter()-started))
        if gpu is not None:gpu.finish(self.values,self.variance,self.outcome_cache)
        if estimator in ('sampled','repair') and np.all(self.samples==B):
            from oracle_dense_repair import contract
            self.bellman_contract=contract(seed,candidate_count)
            if self.bellman_residuals is None:self.bellman_residuals=np.zeros(H)
            self.bellman_metadata=dict(contract=self.bellman_contract,
                rootErrorAllowance=repair_error,repairBoundary=repair_stop,boundaryDifference=repair_delta,
                roundoffPerLayer=roundoff,scope='Finite-grid Bellman reuse; roundoff allowance is not machine-verified',
                variance='Copied prefix dispersion is stale and serves only as a scheduling statistic')
        if self.directory is not None:
            self.values.flush();self.variance.flush()
            if self.outcome_cache is not None:self.outcome_cache.flush()
            self._save_metadata(self.directory)
        reuse_stats=gpu.reuse_stats() if gpu is not None else {}
        return dict(seconds=time.perf_counter()-started,allocationSeconds=alloc_seconds,backend=backend,estimator=estimator,
            **reuse_stats,
            repairedLayers=H-(repair_stop-1) if repair_stop is not None else H,
            reusedLayers=repair_stop-1 if repair_stop is not None else 0,
            rootErrorAllowance=repair_error,repairBoundary=repair_stop,repairChecks=repair_checks,
            scheduledAccountCells=gpu.active_count if gpu is not None else W*C,
            finiteCells=int(np.isfinite(self.values[0,:,:,0]).sum()),totalCells=W*C,
            valueStorageBytes=int(self.values.nbytes+self.variance.nbytes),
            meanConditionalSamples=float(self.samples.mean()))

    def front(self,arrived,consumed=0.,filled=False,interior_samples=1,seed=0):
        """Search actual attainable post-states, including samples inside cells.

        Exact R(s,a) determines costs, caps and the resulting wealth. Sampling
        an exposure target can land in another cell after the cap binds; lookup
        always uses that ACTUAL post-state. No pre-transition cell score is used.
        """
        if interior_samples<0:raise ValueError('Negative sample count')
        a=self.a;g=self.grid;C=len(a['coordinates']);E=len(a['exposures'])
        best,state,choice=self.decide(arrived,consumed,filled)
        scores=np.full(C,-np.inf);states=np.zeros((C,len(arrived)));rng=np.random.default_rng(seed)
        for c in range(C):
            b,ei=divmod(c,E)
            for sample in range(interior_samples+1):
                if sample==0:e=a['exposures'][ei];book=a['books'][b];count=a['book_counts'][b]
                else:
                    e,orders=g.sample_cell(c,rng);count=len(orders);book=np.zeros_like(a['books'][b])
                    if count:book[:count]=orders
                target,ok=execute_target(arrived,consumed,filled,e,book,count,self.rules,g.price_scale)
                if ok:
                    val=self.lookup(target)
                    if val>scores[c]:scores[c]=val;states[c]=target
                    if val>best:best=val;state=target;choice=c
        return dict(value=float(best),state=state,choice=int(choice),scores=scores,states=states)

    def observe(self,front,price,features,price_pair,feature_std):
        """Counterfactual one-step TD errors against the SAME old terminal date.

        Each old front proposal experiences the observed exogenous price. The
        continuation is the old depth-one field at its nearest market sample.
        These correlated, bootstrapped residuals guide compute only: they are
        neither independent evidence nor forecast-likelihood/SIR weights.
        """
        if self.shape[0]<2:return dict(observations=0)
        node=self.cloud.nearest(1,features,price_pair,feature_std);residuals=[];by_cell={}
        for c in np.flatnonzero(np.isfinite(front['scores'])):
            arrived,consumed,filled,ok=move_arrive(front['states'][c],price,self.rules)
            if not ok:continue
            continuation,_,_=self.decide(arrived,consumed,filled,depth=1,node=node)
            residual=continuation-front['scores'][c]
            if np.isfinite(residual):
                # A capped action may land outside its proposed target cell.
                # Credit the actual post-state, and count this observed tick
                # once per cell rather than once per duplicate proposal.
                state=front['states'][c];w=state[0]+state[1]*state[2];orders=[]
                for j in range(4,len(state),3):
                    if state[j]!=0:
                        x=abs(state[j+1]/state[2]-1)/self.grid.price_scale
                        orders.append((x/(1+x),state[j]*state[j+2]/w))
                cell=self.grid.canonical_cell(state[1]*state[2]/w,orders)
                by_cell.setdefault(cell,[]).append(residual);residuals.append(residual)
        for cell,rs in by_cell.items():self.errors.observe(cell,float(np.mean(rs)))
        return dict(observations=len(residuals),marketNode=node,
            distinctAccountCells=len(by_cell),
            meanResidual=float(np.mean(residuals)) if residuals else None,
            rmsResidual=float(np.sqrt(np.mean(np.square(residuals)))) if residuals else None,
            terminalTimestamp=self.cloud.origin_seconds+self.shape[0],
            interpretation='Correlated counterfactual TD residuals with nearest-cloud continuation; priority only')
