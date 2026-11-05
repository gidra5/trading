"""Inspect saved control-cache locality without evaluating or changing the planner."""
import argparse
import json
from pathlib import Path
import numpy as np
from audit_oracle_incremental_checkpoint import digest,require


def summarize(directory):
    root=Path(directory)/'checkpoints'
    pointer=json.loads((root/'current.json').read_text());name=pointer['snapshot']
    require(isinstance(name,str) and len(name)==32 and all(c in '0123456789abcdef' for c in name),'Invalid snapshot name')
    snapshot=root/name
    require(digest(snapshot/'manifest.json')==pointer['manifestSha256'],'Manifest checksum differs')
    manifest=json.loads((snapshot/'manifest.json').read_text())
    for name in ('metadata.json','controls.npz'):
        require(digest(snapshot/name)==manifest[name],'Snapshot checksum differs: '+name)
    meta=json.loads((snapshot/'metadata.json').read_text())
    now=meta['context']['episodeOrigin']+meta['update']
    with np.load(snapshot/'controls.npz') as archive:
        keys=archive['keys'];controls=archive['values']
        depths=keys[:,0]-now
        unique,counts=np.unique(depths,return_counts=True)
        bins=(0,1,2,10,60,300,900,1800,3600,float('inf'))
        histogram=[dict(lower=a,upper=b if np.isfinite(b) else None,rows=int(np.count_nonzero((depths>=a)&(depths<b))))
            for a,b in zip(bins,bins[1:])]
        return dict(snapshot=pointer['snapshot'],update=meta['update'],cacheRows=len(keys),capacity=meta['budget']['cached_rows'],
            completeControlRows=int(np.count_nonzero(np.isfinite(controls).all(1))),
            expiredRows=int(np.count_nonzero(depths<0)),distinctDepths=len(unique),histogram=histogram,
            depthRows=[dict(depth=int(depth),rows=int(count)) for depth,count in zip(unique,counts)],
            scope='Saved finite-control availability by remaining seconds. Availability is not cache-hit rate, independence, or a value-error bound.')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('directory',type=Path);args=parser.parse_args()
    result=summarize(args.directory)
    (args.directory/'cache-summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='depthRows'},indent=2))
