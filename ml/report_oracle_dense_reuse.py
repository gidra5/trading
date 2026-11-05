"""Full-field and action-relative errors against each run's census reference."""
import argparse
import json
from pathlib import Path
import numpy as np
from oracle_dense_value_map import DenseValueMap


def report(directory):
    directory=Path(directory);results=json.loads((directory/'results.json').read_text());rows=[]
    for row in results['rows']:
        update=row['update']
        if not row.get('reference'):continue
        a=DenseValueMap.load(directory/f'update-{update}');r=DenseValueMap.load(directory/f'reference-{update}')
        pairs=0;coverage=0;squared=0.;maximum=0.
        for start in range(0,len(a.values),60):
            x=a.values[start:start+60];y=r.values[start:start+60];xf=np.isfinite(x);yf=np.isfinite(y);mask=xf&yf
            coverage+=int(np.count_nonzero(xf!=yf));delta=x[mask]-y[mask];pairs+=len(delta)
            if len(delta):maximum=max(maximum,float(np.max(abs(delta))));squared+=float(np.dot(delta,delta))
        with np.load(a.directory/'front.npz') as af,np.load(r.directory/'front.npz') as rf:
            states=np.concatenate((af['states'][np.isfinite(af['scores'])],rf['states'][np.isfinite(rf['scores'])]))
            av=np.array([a.lookup(s) for s in states]);rv=np.array([r.lookup(s) for s in states]);good=np.isfinite(av)&np.isfinite(rv)
            delta=av[good]-rv[good]
        rows.append(dict(update=update,seconds=row['seconds'],effectiveMeanConditionalSamples=row.get('effectiveMeanConditionalSamples'),
            cachedOutcomeFraction=row.get('cachedOutcomeFraction'),fieldFinitePairs=pairs,fieldCoverageDisagreements=coverage,
            maximumFieldLogErrorBps=maximum*10000,rmsFieldLogErrorBps=np.sqrt(squared/max(pairs,1))*10000,
            rootMedianLevelErrorBps=float(np.median(delta))*10000,
            maximumRootRelativeActionErrorBps=float(np.max(delta)-np.min(delta))*10000,
            selectedActionReferenceGapBps=row['reference']['referenceLogGap']*10000))
    result=dict(rows=rows,interpretation='Conditional finite-cloud/grid reference, not true-market or continuous-policy optimality. Relative action error is the range of paired score errors, removing a shared level shift.')
    (directory/'reuse-accuracy.json').write_text(json.dumps(result,indent=2));return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args();print(json.dumps(report(a.directory),indent=2))
