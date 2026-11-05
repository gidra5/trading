"""Exact-account diagnostic with inventory and up to two carried limit orders."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from numba import set_num_threads
from audit_oracle_dense_interpolation import compare, market
from oracle_account_grid import AccountGrid
from oracle_book_mc import rule_vector, valid_post
from gpu_oracle_bellman import Rules


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False);set_num_threads(1)
    rows=json.loads(args.source.read_text())['results'];cases={}
    selected=('quiet','trend','reversal','asymmetric','random-4','random-45','random-46','random-52')
    for row in rows:
        if row['case'] in selected:cases[row['case']]=row['rawPrices']
    if len(cases)!=len(selected):raise ValueError('Missing benchmark laws')
    states={
        'buy':[10000.,0.,100.,100.,1.,99.99,5000.,0.,0.,0.],
        'sell':[10000.,0.,100.,100.,-1.,100.01,5000.,0.,0.,0.],
        'opposing':[10000.,0.,100.,100.,1.,99.99,2500.,-1.,100.01,2500.],
        'long-opposing':[5000.,50.,100.,100.,1.,99.99,2000.,-1.,100.01,2000.],
        'short-two-buys':[15000.,-50.,100.,100.,1.,99.99,1000.,1.,99.97,1000.],
    }
    rules=rule_vector(Rules());reference=tuple(np.linspace(-2.5,2.5,21));results=[];started=time.perf_counter()
    files=('audit_oracle_dense_carried.py','audit_oracle_dense_interpolation.py','oracle_account_grid.py',
        'oracle_book_mc.py','oracle_dense_value_map.py')
    (args.output/'plan.json').write_text(json.dumps(dict(source=str(args.source),
        sourceSha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
        sourceHashes={n:hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() for n in files},
        states=states,cases=selected,referenceExposurePoints=reference,
        scope='Two-step finite price/action law; same exact-state 21-exposure reference for both grids. Carried orders are cancellable before the first price move.'),indent=2))
    for n,exposures in ((7,(-2.5,-2.,-1.,0.,1.,2.,2.5)),(21,reference)):
        grid=AccountGrid(max_orders=2,exposure_points=exposures,wealth_points=(5000.,10000.,20000.))
        for case,raw in cases.items():
            raw=np.asarray(raw);cloud=market(raw[0,0],raw[1])
            for state_name,state in states.items():
                state=np.asarray(state)
                if not valid_post(state,rules,False,False):raise ValueError('Infeasible initial account')
                result=dict(case=case,initialStateName=state_name,initialState=state.tolist(),
                    **compare(grid,cloud,rules,state,reference))
                results.append(result)
                (args.output/'results.json').write_text(json.dumps(dict(results=results,elapsedSeconds=time.perf_counter()-started),indent=2))
        group=[r for r in results if len(r['grid']['exposure_points'])==n]
        gaps=[r['rootDecisionCEGapBps'] for r in group if r['rootDecisionCEGapBps'] is not None]
        print(json.dumps(dict(exposurePoints=n,cases=len(group),meanGapBps=float(np.mean(gaps)),
            maximumGapBps=max(gaps),infeasible=len(group)-len(gaps),elapsedSeconds=time.perf_counter()-started)),flush=True)


if __name__=='__main__':main()
