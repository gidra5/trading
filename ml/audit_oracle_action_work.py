"""Count useful action lanes in the current dense GPU backup schedule.

Samples account/market/outcome work items uniformly before feasibility checks,
as in the dense workload rather than according to market reachability mass.
Does not execute GPU kernels or change the planner.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from oracle_dense_value_map import DenseValueMap
from oracle_book_mc import representative, move_arrive, execute_target


def action_work(arrived, used, filled, grid, arrays, rules, candidates):
    E=len(arrays['exposures']);K=grid.max_orders
    padded=1<<(1+len(candidates)+(1<<K)*(E+1)-1).bit_length()
    occupied=sum(1<<j for j in range(K) if arrived[4+3*j]!=0)
    feasible=0;states=set();in_menu=0
    for cell in candidates:
        book,ei=divmod(int(cell),E);in_menu+=1
        post,okay=execute_target(arrived,used,filled,arrays['exposures'][ei],arrays['books'][book],
            arrays['book_counts'][book],rules,grid.price_scale)
        if okay:feasible+=1;states.add(tuple(post))
    for mask in range(1<<K):
        if mask & ~occupied:continue
        for ei in range(-1,E):
            in_menu+=1
            post,okay=execute_target(arrived,used,filled,np.nan if ei<0 else arrays['exposures'][ei],
                arrays['books'][0],0,rules,grid.price_scale,mask)
            if okay:feasible+=1;states.add(tuple(post))
    return dict(padded=padded,inMenu=in_menu,feasible=feasible,uniqueFeasible=len(states),
        pendingOrders=int(np.count_nonzero(arrived[4::3])))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--map',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--samples',type=int,default=5000)
    parser.add_argument('--seed',type=int,default=20260954);args=parser.parse_args()
    if args.samples<1:raise ValueError('Positive samples required')
    args.output.mkdir(parents=True,exist_ok=False);started=time.perf_counter()
    m=DenseValueMap.load(args.map);g=m.grid;a=m.a;c=m.cloud;r=m.rules
    plan=json.loads((args.map.parent/'plan.json').read_text());settings=plan['arguments']
    E=len(a['exposures']);C=len(a['coordinates']);rng=np.random.default_rng(args.seed)
    rows=[];attempts=0
    while len(rows)<args.samples:
        attempts+=1
        if attempts>args.samples*100:raise RuntimeError('Too few valid arrival tasks')
        h=int(rng.integers(len(c.sampled_prices)-1));node=int(rng.integers(c.counts[h]))
        wi=int(rng.integers(len(a['wealth'])));cell=int(rng.integers(C));book,ei=divmod(cell,E)
        state,okay=representative(a['wealth'][wi],a['exposures'][ei],a['books'][book],
            a['book_counts'][book],c.prices[h,node],r,g.price_scale)
        if not okay:continue
        draw=int(rng.integers(c.destinations.shape[-1]))
        arrived,used,filled,okay=move_arrive(state,c.sampled_prices[h,node,draw],r)
        if not okay:continue
        schedule=np.random.default_rng(settings['seed']+int(c.origin_seconds)+h)
        extra=max(0,min(C-E,settings['candidates']-E))
        candidates=np.r_[np.arange(E),schedule.choice(np.arange(E,C),extra,replace=False)].astype(np.int64)
        rows.append(action_work(arrived,used,filled,g,a,r,candidates))
    totals={key:sum(row[key] for row in rows) for key in ('padded','inMenu','feasible','uniqueFeasible')}
    grouped=[]
    for k in range(g.max_orders+1):
        selected=[row for row in rows if row['pendingOrders']==k]
        grouped.append(dict(pendingOrders=k,arrivals=len(selected),
            **{key:float(np.mean([row[key] for row in selected])) if selected else None
               for key in totals}))
    result=dict(samples=len(rows),attempts=attempts,seed=args.seed,totals=totals,byPendingOrders=grouped,
        feasibleLaneFraction=totals['feasible']/totals['padded'],
        uniqueFeasibleLaneFraction=totals['uniqueFeasible']/totals['padded'],
        sourceMap=str(args.map),operator=json.loads((args.map/'operator.json').read_text()),
        sourceSha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),elapsedSeconds=time.perf_counter()-started,
        scope='Counts continuation-scoring lanes after a valid arrival, excluding the separate explicit cash candidate. Exactly equal post-account states are counted once in uniqueFeasible. No GPU speedup or profitability claim.')
    (args.output/'results.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
