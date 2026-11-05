"""Measure speed and root projection error while rolling a retained market grid."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from latent_neural_stochastic_flow import load_latent_nsf
from train_latent_neural_stochastic_flow import latent_data
from target_horizon_forecast import HorizonRange
from oracle_continuation_value import timeline_inputs
from oracle_dense_value_map import MarketCloud
from oracle_dense_rollout import roll_market_cloud


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--updates',type=int,default=120);args=p.parse_args()
    if args.updates<1:raise ValueError('Need positive update count')
    args.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);torch.set_float32_matmul_precision('high')
    plan=json.loads((args.source/'plan.json').read_text());source=args.source/'update-0'/'market.npz'
    checkpoint=Path('data/backtests/oracle-latent-nsf-sampled-skips/latentSampledSkipFlow.pt')
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=plan['checkpointSha256']:raise ValueError('Different model checkpoint')
    training=json.loads((checkpoint.parent/'plan.json').read_text());model,_=load_latent_nsf(checkpoint,'cuda')
    data=latent_data(Path(training['nativeDirectory']),Path(training['dataset']),'test',HorizonRange(21600.,1))
    previous=MarketCloud.load(source);origin=plan['arguments']['origin']
    f,p,t=[x.cpu().numpy()[0] for x in timeline_inputs(data,np.array([origin]),'cpu')]
    if previous.origin_seconds!=t[-1] or not np.array_equal(previous.features[0,0],f) or not np.array_equal(previous.price_pairs[0,0],p):
        raise ValueError('Initial observation mismatch')
    rows=[];started=time.perf_counter()
    with (args.output/'updates.jsonl').open('w') as out:
        for update in range(1,args.updates+1):
            f,p,t=[x.cpu().numpy()[0] for x in timeline_inputs(data,np.array([origin+update]),'cpu')]
            previous,row=roll_market_cloud(previous,model,f,p,t,plan['arguments']['seed']+origin+update)
            row.update(update=update,originSeconds=float(t[-1]),observedPrice=float(p[-1]));rows.append(row)
            out.write(json.dumps(row)+'\n');out.flush()
            if update%20==0:print(json.dumps(row),flush=True)
    keys=('seconds','rootProjectionMeanReturnShiftBps','rootProjectionPriceRMSEBps','rootProjectionCoordinateRMSE',
        'rootExecutionPriceRMSEBps','rootExecutionMeanReturnShiftBps')
    summary={k:dict(mean=float(np.mean([x[k] for x in rows])),median=float(np.median([x[k] for x in rows])),
        maximumAbsolute=float(np.max(np.abs([x[k] for x in rows])))) for k in keys}
    result=dict(updates=args.updates,seconds=time.perf_counter()-started,summary=summary,
        rootPriceRMSEAbovePointOneBps=int(sum(x['rootProjectionPriceRMSEBps']>.1 for x in rows)),
        rootPriceRMSEAboveOneBps=int(sum(x['rootProjectionPriceRMSEBps']>1 for x in rows)),
        source=str(source),sourceSha256=hashlib.sha256(source.read_bytes()).hexdigest(),checkpointSha256=plan['checkpointSha256'],
        sourceHashes={f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in
            ('ml/oracle_dense_rollout.py','ml/audit_oracle_dense_rollout.py','ml/export_nsf_planning_trees.py','ml/nsf_scenario_graph.py')},
        interpretation='Forecast-grid reuse diagnostic using causal observed roots; thresholds are diagnostics, not Bellman error certificates')
    (args.output/'results.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':main()
