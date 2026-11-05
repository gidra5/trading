"""Extend the EXACT earlier one-hour prefix to a matched 48-hour finite law."""
from __future__ import annotations
import json
import hashlib
from pathlib import Path
import argparse
import numpy as np
import torch
from export_nsf_planning_trees import next_observations
from latent_neural_stochastic_flow import load_latent_nsf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', type=Path, default=Path('data/backtests/oracle-relative-48h-20260913-v3'))
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise ValueError('Choose a new output directory')
    old = json.loads((args.source/'plan.json').read_text())
    checkpoint = Path(old['checkpoint'])
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == old['checkpointSha256']
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('high')
    model, _ = load_latent_nsf(checkpoint, torch.device('cuda'))
    records = [r for r in json.loads((args.source/'forecasts.json').read_text()) if r['horizonSeconds'] == 3600]
    pending = []
    for row in records:
        # All prefix objects are newly parsed private builders. The source file
        # is never modified, nor are hypothetical observations taken from data.
        start = row['timestamp']/1000
        for first in row['tree']['children']:
            for second in first['children']:
                for leaf in second['children']:
                    f = np.array([json.loads(second['observationKey']), json.loads(leaf['observationKey'])], dtype=np.float32)
                    pending.append((leaf, f, np.array([second['price'],leaf['price']]), np.array([start+2250,start+3600])))
        row['stageOffsetsSeconds'] = [900,2250,3600,88200,172800]
        row['horizonSeconds'] = 172800
    for level in range(2):
        following = []
        for node, f, p, t in pending:
            nf, pp = next_observations(model,f[None],p[None],t[None],np.array([84600]),2,old['seed']+3+level)
            node['children'] = []
            for b in range(2):
                child = dict(probability='1/2',price=float(pp[0,b]),observationKey=json.dumps(nf[0,b].tolist(),separators=(',',':')),children=[])
                node['children'].append(child)
                if level == 0:
                    following.append((child,np.stack((f[-1],nf[0,b])),np.array([p[-1],pp[0,b]]),np.array([t[-1],t[-1]+84600])))
        pending = following
    sources = ['ml/export_oracle_shadow_tail.py','ml/export_nsf_planning_trees.py','ml/latent_neural_stochastic_flow.py',
               'packages/bot-algo/src/oracle-shadow-tail.ts','packages/bot-algo/src/oracle-crra-policy.ts',
               'packages/bot-algo/src/oracle-execution.ts','scripts/evaluate-oracle-shadow-tail.ts']
    plan = dict(source=str(args.source),sourceForecastSha256=hashlib.sha256((args.source/'forecasts.json').read_bytes()).hexdigest(),
                checkpoint=str(checkpoint),checkpointSha256=old['checkpointSha256'],seed=old['seed'],
                offsets=[900,2250,3600,88200,172800],branches=2,gamma=0,initialCash=10000,
                scope='one matched five-stage finite sampled law; no unmodeled per-second events or unsampled-law certificate',
                sourceHashes={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources})
    args.output.mkdir(parents=True)
    (args.output/'forecasts.json').write_text(json.dumps(records,allow_nan=False))
    (args.output/'plan.json').write_text(json.dumps(plan,indent=2))
    print(json.dumps(dict(output=str(args.output),roots=len(records),leavesPerRoot=32)),flush=True)


if __name__ == '__main__':
    main()
