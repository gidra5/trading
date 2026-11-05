"""Two-step Bellman census with exact account states, for grid-error diagnosis.

The price law and action catalog are finite. Every future catalog action and
retain/cancel subset is evaluated, with account quantities carried exactly.
This removes account interpolation, not action discretization, and shares the
tested Oracle execution primitives. It is not a continuous optimum certificate.
"""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import time
import numpy as np
from numba import njit, set_num_threads
from oracle_account_grid import AccountGrid
from oracle_book_mc import execute_target, close_terminal, effective, move_arrive, rule_vector
from oracle_dense_value_map import DenseValueMap, MarketCloud
from gpu_oracle_bellman import Rules


@njit(cache=True)
def targets(arrived, consumed, filled, exposures, books, counts, rules, scale):
    """Same full catalog and retained-subset actions as the dense CPU planner."""
    K=(len(arrived)-4)//3
    out=np.empty((1+len(books)*len(exposures)+(1<<K)*(len(exposures)+1),len(arrived)))
    n=0
    if np.isfinite(close_terminal(arrived,consumed,rules)):
        flat=arrived.copy();flat[0]+=effective(arrived,rules);flat[1]=0.;flat[4:]=0.
        out[n]=flat;n+=1
    for b in range(len(books)):
        for e in exposures:
            state,ok=execute_target(arrived,consumed,filled,e,books[b],counts[b],rules,scale)
            if ok:out[n]=state;n+=int(ok)
    occupied=0
    for j in range(K):
        if arrived[4+3*j]!=0:occupied|=1<<j
    for mask in range(1<<K):
        if mask & ~occupied:continue
        for i in range(-1,len(exposures)):
            e=np.nan if i<0 else exposures[i]
            state,ok=execute_target(arrived,consumed,filled,e,books[0],0,rules,scale,mask)
            if ok:out[n]=state;n+=int(ok)
    return out[:n]


@njit(cache=True)
def terminal_score(post, factors, rules):
    result=0.
    for factor in factors:
        price=max(1.,np.rint(post[2]*factor/rules[9]))*rules[9]
        arrived,consumed,_,valid=move_arrive(post,price,rules)
        if not valid:return -np.inf
        value=close_terminal(arrived,consumed,rules)
        if not np.isfinite(value):return -np.inf
        result+=value/len(factors)
    return result


@njit(cache=True)
def two_step_score(post, first_factors, destinations, last_factors,
                   exposures, books, counts, rules, scale):
    result=0.
    for i in range(len(first_factors)):
        price=max(1.,np.rint(post[2]*first_factors[i]/rules[9]))*rules[9]
        arrived,consumed,filled,valid=move_arrive(post,price,rules)
        if not valid:return -np.inf
        best=-np.inf
        for target in targets(arrived,consumed,filled,exposures,books,counts,rules,scale):
            best=max(best,terminal_score(target,last_factors[destinations[i]],rules))
        if not np.isfinite(best):return -np.inf
        result+=best/len(first_factors)
    return result


def census(post_states, cloud, arrays, rules, scale):
    if len(cloud.counts)!=3:raise ValueError('This diagnostic requires exactly two transitions')
    return np.array([two_step_score(s,cloud.factors(0)[0],cloud.destinations[0,0],cloud.factors(1),
        arrays['exposures'],arrays['books'],arrays['book_counts'],rules,scale) for s in post_states])


def market(first, last):
    """Two equiprobable observations, followed by two conditional outcomes."""
    first=np.asarray(first,dtype=float);last=np.asarray(last,dtype=float)
    prices=np.array([[100.,100.],first,[100.,100.]])
    raw=np.stack((np.tile(first,(2,1)),last))
    pairs=np.stack((np.full_like(prices,100.),prices),axis=-1)
    return MarketCloud(prices,np.array([1,2,2]),np.tile(np.array([0,1]),(2,2,1)),
        np.array([[1.,0.],[.5,.5],[.5,.5]]),np.zeros((3,2,2,59),dtype=np.float32),pairs,0.,raw)


def compare(grid, cloud, rules, initial, reference_exposure_points=None):
    arrays=grid.arrays();started=time.perf_counter()
    dense=DenseValueMap(grid,cloud,rules,None)
    sweep=dense.sweep(17,len(arrays['coordinates']),2,backend='cpu')
    front=dense.front(initial,interior_samples=0)
    reference_grid=grid if reference_exposure_points is None else replace(grid,exposure_points=reference_exposure_points)
    reference_arrays=reference_grid.arrays()
    choices=targets(initial,0.,False,reference_arrays['exposures'],reference_arrays['books'],reference_arrays['book_counts'],rules,grid.price_scale)
    scores=census(choices,cloud,reference_arrays,rules,grid.price_scale)
    exact_ix=int(np.argmax(scores));exact=float(scores[exact_ix])
    chosen_exact=float(census(np.array([front['state']]),cloud,reference_arrays,rules,grid.price_scale)[0])
    chosen_feasible=bool(np.isfinite(chosen_exact))
    dense_scores=np.array([dense.lookup(s) for s in choices])
    common=np.isfinite(scores)&np.isfinite(dense_scores)
    return dict(grid=asdict(grid),referenceGrid=asdict(reference_grid),rootCandidates=len(choices),exactFiniteCatalogLogValue=exact,
        denseReportedLogValue=front['value'],denseChosenCensusFeasible=chosen_feasible,
        denseChosenExactContinuationLogValue=chosen_exact if chosen_feasible else None,
        rootDecisionRegretLog=exact-chosen_exact if chosen_feasible else None,
        rootDecisionCEGapBps=float(-np.expm1(chosen_exact-exact)*10000.) if chosen_feasible else None,
        maximumCommonScoreError=float(np.max(abs(scores[common]-dense_scores[common]))) if np.any(common) else None,
        finiteOnlyInCensus=int(np.sum(np.isfinite(scores)&~np.isfinite(dense_scores))),
        finiteOnlyInDense=int(np.sum(~np.isfinite(scores)&np.isfinite(dense_scores))),
        exactChosenState=choices[exact_ix].tolist(),denseChosenState=front['state'].tolist(),
        sweepSeconds=sweep['seconds'],seconds=time.perf_counter()-started)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--random-cases',type=int,default=0)
    parser.add_argument('--reference-exposure-count',type=int,help='Use a shared denser exact-account action catalog')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False);set_num_threads(1)
    rules=rule_vector(Rules())
    reference_points=None
    if args.reference_exposure_count is not None:
        if args.reference_exposure_count<21 or (args.reference_exposure_count-1)%20:
            raise ValueError('Shared reference axis must contain both tested axes: count = 20*n+1')
        reference_points=tuple(np.linspace(-2.5,2.5,args.reference_exposure_count))
    cases={'quiet':market([99.99,100.01],[[99.99,100.01],[99.99,100.01]]),
        'trend':market([100.15,100.3],[[100.30,100.45],[100.45,100.60]]),
        'reversal':market([99.7,100.3],[[99.99,100.01],[99.99,100.01]]),
        'asymmetric':market([99.8,100.05],[[99.95,100.15],[100.05,100.10]])}
    rng=np.random.default_rng(27)
    for i in range(args.random_cases):
        first=np.round(100*np.exp(rng.normal(0,.004,2)),2)
        last=np.round(first[:,None]*np.exp(rng.normal(0,.004,(2,2))),2)
        cases[f'random-{i}']=market(first,last)
    rows=[]
    for name,cloud in cases.items():
        for points in ((-2.5,-2.,-1.,0.,1.,2.,2.5),tuple(np.linspace(-2.5,2.5,21))):
            grid=AccountGrid(max_orders=1,exposure_points=points,wealth_points=(5000.,10000.,20000.))
            initial=np.array([10000.,0.,100.,100.,0.,0.,0.])
            row=dict(case=name,rawPrices=cloud.sampled_prices.tolist(),**compare(grid,cloud,rules,initial,reference_points));rows.append(row)
            (args.output/'results.json').write_text(json.dumps(dict(results=rows,
                scope='Two-step finite price law and full finite action catalog; exact account-state continuation. Shared execution primitives; not a continuous-action or learned-law certificate.'),indent=2))
            print(json.dumps({k:row[k] for k in ('case','rootCandidates','rootDecisionCEGapBps','maximumCommonScoreError','finiteOnlyInCensus','finiteOnlyInDense','seconds')}),flush=True)


if __name__=='__main__':main()
