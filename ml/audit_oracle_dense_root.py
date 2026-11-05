"""Refine only root exposure proposals against an unchanged archived field."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from oracle_dense_grid_config import exposure_axis
from oracle_dense_value_map import DenseValueMap
from oracle_book_mc import execute_target


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--map',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--step',type=float,default=.01)
    parser.add_argument('--cash',type=float,default=10000.);args=parser.parse_args()
    if not np.isfinite(args.cash) or args.cash<=0:raise ValueError('Positive starting cash required')
    args.output.mkdir(parents=True,exist_ok=False);started=time.perf_counter()
    m=DenseValueMap.load(args.map);a=m.a;r=m.rules;g=m.grid
    plan=json.loads((args.map.parent/'plan.json').read_text())
    state=np.r_[args.cash,0.,m.cloud.prices[0,0],m.cloud.prices[0,0],np.zeros(3*g.max_orders)]
    baseline=m.front(state,interior_samples=1,seed=plan['arguments']['seed'])
    best=baseline['value'];best_state=baseline['state'];rows=[]
    exposure=tuple(e for e in exposure_axis(args.step) if r[3]<=e<=r[4])
    for b in range(len(a['books'])):
        for e in exposure:
            post,ok=execute_target(state,0.,False,e,a['books'][b],a['book_counts'][b],r,g.price_scale)
            if not ok:continue
            value=m.lookup(post)
            if not np.isfinite(value):continue
            if value>best:best=value;best_state=post
            rows.append(dict(targetExposure=e,book=b,value=float(value),state=post.tolist()))
    result=dict(sourceMap=str(args.map),operator=json.loads((args.map/'operator.json').read_text()),
        auditSha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),startingState=state.tolist(),
        exposureStep=args.step,exposureTargets=len(exposure),orderBooks=len(a['books']),finiteProposals=len(rows),
        originalValue=baseline['value'],originalState=baseline['state'].tolist(),
        refinedValue=best,refinedState=best_state.tolist(),
        improvementCEBps=float(np.expm1(best-baseline['value'])*10000.),seconds=time.perf_counter()-started,
        scope='Cash-starting root search only, with unchanged archived continuation/account/market approximation. Original chosen action remains available. Not a continuous-optimum certificate.')
    (args.output/'results.json').write_text(json.dumps(result,indent=2))
    (args.output/'proposals.json').write_text(json.dumps(rows))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
