"""Measure a closing-utility shortcut on archived planner candidate states.

CPU-only eligibility, conformance and timing experiment. Does not change the
planner. Timing excludes compilation and is not a GPU speedup prediction.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from numba import njit
from oracle_book_mc import representative, move_arrive, execute_target, close_terminal
from oracle_close_bounds import close_interval
from oracle_dense_value_map import DenseValueMap


@njit(cache=True)
def exact_close(state, factors, rules):
    mean=0.
    for j in range(len(factors)):
        price=max(1.,np.rint(state[2]*factors[j]/rules[9]))*rules[9]
        arrived,used,_,okay=move_arrive(state,price,rules)
        if not okay:return -np.inf
        value=close_terminal(arrived,used,rules)
        if not np.isfinite(value):return -np.inf
        mean+=(value-mean)/(j+1)
    return mean


@njit(cache=True)
def evaluate(states, factors, rules, tolerance, shortcut):
    values=np.empty(len(states));accepted=np.zeros(len(states),np.bool_)
    lower=np.full(len(states),-np.inf);upper=np.full(len(states),np.inf)
    for i in range(len(states)):
        if shortcut:
            accepted[i],lower[i],upper[i]=close_interval(states[i],factors[i],rules,tolerance)
        values[i]=lower[i] if accepted[i] else exact_close(states[i],factors[i],rules)
    return values,accepted,lower,upper


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--map',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--samples',type=int,default=20000)
    parser.add_argument('--tolerance',type=float,default=1e-9)
    parser.add_argument('--seed',type=int,default=20260953)
    args=parser.parse_args()
    if args.samples<1 or not np.isfinite(args.tolerance) or args.tolerance<=0:
        raise ValueError('Positive sample count and finite positive tolerance required')
    args.output.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter();m=DenseValueMap.load(args.map)
    c=m.cloud;a=m.a;g=m.grid;r=m.rules;rng=np.random.default_rng(args.seed)
    states=[];factors=[];order_counts=[];attempts=0
    while len(states)<args.samples and attempts<args.samples*100:
        attempts+=1;h=int(rng.integers(0,len(c.sampled_prices)-1));node=int(rng.integers(c.counts[h]))
        wi=int(rng.integers(len(a['wealth'])));ei=int(rng.integers(len(a['exposures'])))
        book=int(rng.integers(len(a['books'])))
        state,okay=representative(a['wealth'][wi],a['exposures'][ei],a['books'][book],
            a['book_counts'][book],c.prices[h,node],r,g.price_scale)
        if not okay:continue
        draw=int(rng.integers(c.destinations.shape[2]));child=c.destinations[h,node,draw]
        arrived,used,filled,okay=move_arrive(state,c.sampled_prices[h,node,draw],r)
        if not okay:continue
        book=int(rng.integers(len(a['books'])));ei=int(rng.integers(-1,len(a['exposures'])))
        # Include retained pending orders as well as fresh catalog targets.
        mask=int(rng.integers(1<<g.max_orders)) if rng.random()<.5 else -1
        post,okay=execute_target(arrived,used,filled,np.nan if ei<0 else a['exposures'][ei],
            a['books'][book],a['book_counts'][book],r,g.price_scale,mask)
        if not okay:continue
        states.append(post);factors.append(c.factors(h+1)[child])
        order_counts.append(int(np.count_nonzero(post[4::3])))
    if len(states)!=args.samples:raise RuntimeError('Insufficient feasible candidate samples')
    states=np.asarray(states);factors=np.asarray(factors);order_counts=np.asarray(order_counts)
    # Warm both compiled branches before measuring; no CUDA work is launched.
    evaluate(states[:1],factors[:1],r,args.tolerance,False)
    evaluate(states[:1],factors[:1],r,args.tolerance,True)
    timings={};outputs={}
    for shortcut in (False,True):
        timings[str(shortcut)]=[]
        for _ in range(5):
            t=time.perf_counter();outputs[shortcut]=evaluate(states,factors,r,args.tolerance,shortcut)
            timings[str(shortcut)].append(time.perf_counter()-t)
    exact=outputs[False][0];approx,accepted,lower,upper=outputs[True]
    finite=np.isfinite(exact);violations=accepted&((~finite)|(exact<lower)|(exact>upper))
    if np.any(violations):raise AssertionError(f'{np.count_nonzero(violations)} enclosure violations')
    if not np.array_equal(finite,np.isfinite(approx)):raise AssertionError('Feasibility changed')
    rows=[]
    for k in range(g.max_orders+1):
        mask=order_counts==k;eligible=mask&accepted
        rows.append(dict(orderCount=k,count=int(mask.sum()),accepted=int(eligible.sum()),
            finiteExact=int((mask&finite).sum()),
            maxAcceptedError=float(np.max(exact[eligible]-approx[eligible])) if eligible.any() else None))
    result=dict(sourceMap=str(args.map),operator=json.loads((args.map/'operator.json').read_text()),
        sourceHashes={name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
            for name in ('audit_oracle_close_bounds.py','oracle_close_bounds.py')},
        samples=len(states),attempts=attempts,seed=args.seed,tolerance=args.tolerance,
        accepted=int(accepted.sum()),finiteExact=int(finite.sum()),enclosureViolations=int(violations.sum()),
        maxFiniteError=float(np.max(np.abs(exact[finite]-approx[finite]))),byOrderCount=rows,
        cpuBatchSeconds=timings,cpuMedianSpeedup=float(np.median(timings['False'])/np.median(timings['True'])),
        elapsedSeconds=time.perf_counter()-started,
        scope='Random feasible candidates from archived account representatives, arrival samples and catalog/retain actions. Eligibility is not weighted by actual GPU workload. CPU timing is not a GPU benchmark. Floating padding is not directed rounding. Planner remains unchanged.')
    (args.output/'results.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
