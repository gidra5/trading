"""Measure how far a new terminal layer changes an aligned full-census field."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from oracle_dense_value_map import DenseValueMap


def audit(previous,current):
    old=DenseValueMap.load(previous);new=DenseValueMap.load(current)
    if old.grid!=new.grid or not np.array_equal(old.rules,new.rules):raise ValueError('Different account contracts')
    a,b=old.cloud,new.cloud;H=new.shape[0]
    if old.shape!=new.shape or b.origin_seconds!=a.origin_seconds+1:raise ValueError('Need aligned one-second shift')
    for field in ('prices','counts','features','price_pairs'):
        if not np.array_equal(getattr(b,field)[1:H],getattr(a,field)[2:H+1]):raise ValueError('Interior market nodes changed')
    for field in ('destinations','sampled_prices'):
        if not np.array_equal(getattr(b,field)[1:H-1],getattr(a,field)[2:H]):raise ValueError('Interior conditional rows changed')
    for m in (old,new):
        if not np.all(m.samples==m.cloud.destinations.shape[-1]):raise ValueError('Full-census fields required')
    plans=[json.loads((Path(d).parent/'plan.json').read_text()) for d in (previous,current)]
    for key in ('seed','candidates'):
        if plans[0]['arguments'][key]!=plans[1]['arguments'][key]:raise ValueError('Candidate schedules differ')
    rows=[]
    for h in range(1,H-1):
        x=old.values[h+1];y=new.values[h];finite=np.isfinite(x)&np.isfinite(y)
        delta=x[finite]-y[finite]
        rows.append(dict(depth=h,unequal=int(np.count_nonzero(~((x==y)|(np.isnan(x)&np.isnan(y))))),
            support=int(np.count_nonzero(np.isfinite(x)!=np.isfinite(y))),
            maximumLogDifference=float(np.max(abs(delta))) if len(delta) else 0.))
    boundaries=[]
    for tolerance in (0.,1.e-10,1.e-8,1.e-7,1.e-6):
        eligible=[r for r in rows if r['support']==0 and r['maximumLogDifference']<=tolerance]
        if not eligible:continue
        row=eligible[-1];h=row['depth']
        prefix=rows[:h-1]
        boundaries.append(dict(tolerance=tolerance,boundary=h,tailLayers=H-h,
            boundaryDifference=row['maximumLogDifference'],
            maximumEarlierDifference=max((r['maximumLogDifference'] for r in prefix),default=0.),
            earlierSupportDisagreements=sum(r['support'] for r in prefix)))
    result=dict(previous=str(previous),current=str(current),unchangedLayers=sum(r['unequal']==0 for r in rows),
        alignedInteriorLayers=len(rows),boundaries=boundaries,rows=rows,
        sourceSha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        interpretation='Offline full-census comparison; candidate kernels align at absolute time. Supports a tail-repair design, but is not an implemented stopping rule or a repeated-update error certificate.')
    del old,new
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('previous',type=Path);p.add_argument('current',type=Path)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    report=audit(args.previous,args.current);args.output.write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='rows'}),flush=True)
