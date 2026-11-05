"""Fresh matched NSF forecasts for the current three-stage prefix planner."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from export_nsf_planning_trees import next_observations, schedule
from latent_neural_stochastic_flow import load_latent_nsf
from train_latent_neural_stochastic_flow import latent_data
from target_horizon_forecast import HorizonRange


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--origins', default='0,300,900')
    args = ap.parse_args()
    if args.output.exists():
        raise ValueError('Choose a new experiment directory')
    origins = [int(x) for x in args.origins.split(',')]
    horizons = [3600, 21600, 86400, 172628, 172799, 172800]
    checkpoint = Path('data/backtests/oracle-latent-nsf-sampled-skips/latentSampledSkipFlow.pt')
    source_plan = json.loads((checkpoint.parent/'plan.json').read_text())
    selection = json.loads((checkpoint.parent/'selection.json').read_text())
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert digest == selection['models']['latentSampledSkipFlow']['checkpointSha256']
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('high')
    model, _ = load_latent_nsf(checkpoint, torch.device('cuda'))
    data = latent_data(Path(source_plan['nativeDirectory']), Path(source_plan['dataset']),
                       'test', HorizonRange(21600., 1))
    first = int(data['origins'][0]); start = float(data['timestamps'][0])/1000
    records, pending = [], []
    for idx in origins:
        if idx < 0 or idx >= len(data['prices']) - 1:
            raise ValueError('Origin outside observed test data')
        for horizon in horizons:
            offsets = schedule(horizon, 900, 1)
            row = dict(origin=idx, timestamp=int((start+idx)*1000), horizonSeconds=horizon,
                       price=float(data['prices'][idx]), stageOffsetsSeconds=offsets, tree={'children': []})
            records.append(row)
            pending.append((row['tree'], np.asarray(data['timeline'][first+idx-1:first+idx+1], dtype=np.float32),
                            data['pricesWithPast'][idx:idx+2], np.array([start+idx-1, start+idx]), offsets, start+idx))
    seed = 20260914
    level = 0
    while pending:
        # Evaluate each case separately with the same random-number stream so
        # different terminal horizons retain identical initial 900s samples.
        following = []
        for node, f, p, t, offsets, origin in pending:
            nf, pp = next_observations(model, f[None], p[None], t[None],
                                      np.array([origin+offsets[0]-t[-1]]), 2, seed+level)
            for b in range(2):
                child = dict(probability='1/2', price=float(pp[0,b]),
                             observationKey=json.dumps(nf[0,b].tolist(), separators=(',', ':')), children=[])
                node['children'].append(child)
                if len(offsets) > 1:
                    following.append((child, np.stack((f[-1], nf[0,b])), np.array([p[-1], pp[0,b]]),
                                      np.array([t[-1], origin+offsets[0]]), offsets[1:], origin))
        pending = following
        print(json.dumps(dict(stage=level, remainingNodes=len(pending))), flush=True)
        level += 1
    sources = ['ml/export_oracle_tail_bound.py', 'ml/export_nsf_planning_trees.py',
               'ml/latent_neural_stochastic_flow.py', 'packages/bot-algo/src/oracle-tail-bound.ts',
               'packages/bot-algo/src/oracle-crra-policy.ts', 'packages/bot-algo/src/oracle-execution.ts',
               'scripts/evaluate-oracle-tail-bound.ts']
    plan = dict(checkpoint=str(checkpoint), checkpointSha256=digest, seed=seed, origins=origins,
                horizonsSeconds=horizons, fullHorizonSeconds=172800, initialCash=10000,
                gamma=0, branches=2, planningLevel=1, maximumPrefixDecisions=3, decisionStrideSeconds=900,
                actualNumericalModel='current complete-replay finite tree/menu; hypothetical trading times remain coarse',
                tailCountConvention='remaining physical seconds, each allowed one capped decision',
                precisionScope='exact rational solve of quantized finite forecast; no learned-law/continuous-menu certificate',
                sourceHashes={p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources})
    args.output.mkdir(parents=True)
    (args.output/'forecasts.json').write_text(json.dumps(records, allow_nan=False))
    (args.output/'plan.json').write_text(json.dumps(plan, indent=2, allow_nan=False))
    print(json.dumps(dict(output=str(args.output), cases=len(records), checkpointSha256=digest)), flush=True)


if __name__ == '__main__':
    main()
