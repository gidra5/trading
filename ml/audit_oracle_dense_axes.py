"""Compare account-axis refinements against one fixed exact-account catalog."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from numba import set_num_threads
from audit_oracle_dense_interpolation import compare, market
from oracle_account_grid import AccountGrid
from oracle_book_mc import rule_vector
from gpu_oracle_bellman import Rules


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--variants',nargs='+',choices=('wealth-11-exposure-21','wealth-11-exposure-41','wealth-3-exposure-41'))
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    set_num_threads(1)
    source=json.loads(args.source.read_text());cases={}
    for row in source['results']:
        cases[row['case']]=row['rawPrices']
    dense_wealth=(5000.,9000.,9500.,9900.,9990.,10000.,10010.,10100.,10500.,11000.,20000.)
    grids={
        'wealth-11-exposure-21':AccountGrid(max_orders=1,exposure_points=tuple(np.linspace(-2.5,2.5,21)),wealth_points=dense_wealth),
        'wealth-11-exposure-41':AccountGrid(max_orders=1,exposure_points=tuple(np.linspace(-2.5,2.5,41)),wealth_points=dense_wealth),
        'wealth-3-exposure-41':AccountGrid(max_orders=1,exposure_points=tuple(np.linspace(-2.5,2.5,41)),wealth_points=(5000.,10000.,20000.)),
    }
    if args.variants:grids={n:grids[n] for n in args.variants}
    files=('audit_oracle_dense_axes.py','audit_oracle_dense_interpolation.py','oracle_account_grid.py',
        'oracle_book_mc.py','oracle_dense_value_map.py')
    plan=dict(source=str(args.source),sourceSha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
        sourceHashes={n:hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() for n in files},
        grids={n:asdict(g) for n,g in grids.items()},referenceExposurePoints=81,
        scope='Two-step synthetic finite laws, exact account-state reference. This does not certify the continuous optimum.')
    (args.output/'plan.json').write_text(json.dumps(plan,indent=2))
    rows=[];started=time.perf_counter();rules=rule_vector(Rules())
    for name,grid in grids.items():
        for case,raw in cases.items():
            raw=np.asarray(raw);cloud=market(raw[0,0],raw[1])
            row=dict(variant=name,case=case,**compare(grid,cloud,rules,
                np.array([10000.,0.,100.,100.,0.,0.,0.]),tuple(np.linspace(-2.5,2.5,81))))
            rows.append(row)
            (args.output/'results.json').write_text(json.dumps(dict(results=rows,elapsedSeconds=time.perf_counter()-started),indent=2))
        selected=[r for r in rows if r['variant']==name]
        gaps=[r['rootDecisionCEGapBps'] for r in selected if r['rootDecisionCEGapBps'] is not None]
        print(json.dumps(dict(variant=name,cases=len(selected),meanGapBps=float(np.mean(gaps)),
            maximumGapBps=max(gaps),infeasible=len(selected)-len(gaps),elapsedSeconds=time.perf_counter()-started)),flush=True)


if __name__=='__main__':main()
