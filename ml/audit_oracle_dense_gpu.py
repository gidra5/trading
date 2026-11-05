"""Compare archived CPU/GPU fields under identical numerical contracts."""
import argparse
import json
from pathlib import Path
import numpy as np


def compare(reference,candidate):
    reference=Path(reference);candidate=Path(candidate)
    left=json.loads((reference/'plan.json').read_text());right=json.loads((candidate/'plan.json').read_text())
    for key in ('grid','rules','fundingMultiplier','gamma','checkpointSha256'):
        if left[key]!=right[key]:raise ValueError(f'Different contract: {key}')
    for key in ('horizon','width','branches','seed','candidates','origin'):
        if left['arguments'][key]!=right['arguments'][key]:raise ValueError(f'Different search: {key}')
    if left['sourceHashes']['ml/oracle_book_mc.py']!=right['sourceHashes']['ml/oracle_book_mc.py']:
        raise ValueError('Different reference execution kernels')
    rows=[]
    for directory in sorted(candidate.iterdir()):
        other=reference/directory.name
        if not directory.is_dir() or not (directory/'samples.npy').exists() or not (other/'samples.npy').exists():continue
        with np.load(directory/'market.npz') as a,np.load(other/'market.npz') as b:
            same_market=a.files==b.files and all(np.array_equal(a[k],b[k],equal_nan=True) for k in a.files)
        same_samples=np.array_equal(np.load(directory/'samples.npy'),np.load(other/'samples.npy'))
        row=dict(field=directory.name,sameMarket=same_market,sameSampleAllocation=same_samples)
        for name in ('values','variance'):
            a=np.load(directory/f'{name}.npy',mmap_mode='r');b=np.load(other/f'{name}.npy',mmap_mode='r')
            if a.shape!=b.shape:raise ValueError('Different field shapes')
            pairs=0;mask_errors=0;nonfinite_errors=0;max_error=0.;squared=0.
            for start in range(0,len(a),60):
                x=a[start:start+60];y=b[start:start+60];xf=np.isfinite(x);yf=np.isfinite(y);paired=xf&yf
                mask_errors+=int(np.count_nonzero(xf!=yf));pairs+=int(paired.sum())
                nonfinite_errors+=int(np.count_nonzero((~xf)&(~yf)&(~((x==y)|(np.isnan(x)&np.isnan(y))))))
                delta=x[paired]-y[paired]
                if len(delta):max_error=max(max_error,float(np.max(abs(delta))));squared+=float(np.dot(delta,delta))
            row[name]=dict(pairedFiniteEntries=pairs,finiteMaskDisagreements=mask_errors,
                nonfiniteKindDisagreements=nonfinite_errors,maximumAbsoluteDifference=max_error,
                rmsDifference=float(np.sqrt(squared/max(pairs,1))))
        rows.append(row)
    report=dict(reference=str(reference),candidate=str(candidate),fields=rows,
        interpretation='Backend equivalence requires the same market, sample allocation and search contract. It is not an optimality certificate.')
    (candidate/'cpu-gpu-comparison.json').write_text(json.dumps(report,indent=2))
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('reference',type=Path);p.add_argument('candidate',type=Path)
    args=p.parse_args();print(json.dumps(compare(args.reference,args.candidate),indent=2))
