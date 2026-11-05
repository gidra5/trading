"""Causal NSF scenario trees spanning one fixed terminal date.

At every synthetic node the two-observation encoder is rerun on decoded market
observations. Hidden latent draws are discarded, never exposed to the policy.
"""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[key]='4'
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from latent_neural_stochastic_flow import load_latent_nsf
from train_latent_neural_stochastic_flow import latent_data
from target_horizon_forecast import HorizonRange
from audit_forecast_agreement import sha


def schedule(remaining, stride, level):
    first=min(stride,remaining)
    if first==remaining: return [remaining]
    return sorted(set([first]+[first+(remaining-first)*k//(2**level) for k in range(1,2**level+1)]))


@torch.inference_mode()
def next_observations(model, features, prices, timestamps, horizon, branches, seed):
    device=next(model.parameters()).device
    f=torch.as_tensor(features,dtype=torch.float32,device=device)
    p=torch.as_tensor(prices,dtype=torch.float64,device=device)
    t=torch.as_tensor(timestamps,dtype=torch.float64,device=device)
    h=torch.as_tensor(horizon,dtype=torch.float32,device=device)
    if not bool(((h>0)&(h<=86400)).all()): raise ValueError('NSF interval outside trained query range')
    obs=model.observations(f,(p/p[:,-1,None]).log())
    dt=torch.cat((torch.zeros_like(t[:,:1]),t[:,1:]-t[:,:-1]),1).float()
    mu,std=model.encode(obs,dt,t)
    rng=torch.Generator(device=device).manual_seed(seed)
    eps=torch.randn((len(f),branches//2,2*model.dimensions+60),generator=rng,device=device)
    eps=torch.cat((eps,-eps),1)
    initial=mu[:,-1,None]+std[:,-1,None]*eps[:,:,:model.dimensions]
    durations=(h/model.time_scale)[:,None].expand(-1,branches).reshape(-1)
    starts=(t[:,-1]/model.time_scale)[:,None].expand(-1,branches).reshape(-1)
    latent=model.transition.rsample(initial.reshape(-1,model.dimensions),obs.new_empty((len(starts),0)),durations,starts,
        eps[:,:,model.dimensions:2*model.dimensions].reshape(-1,model.dimensions))
    mean,scale=model.emission(latent)
    emitted=mean.reshape(len(f),branches,60)+scale.reshape(len(f),branches,60)*eps[:,:,-60:]
    future_price=p[:,-1,None]*torch.exp(emitted[:,:,0].double()*model.price_scale)
    future_price=(future_price/.01).round().clamp_min(1)*.01
    future_features=emitted[:,:,1:]*model.feature_std+model.feature_mean
    if not bool(torch.isfinite(future_price).all() & torch.isfinite(future_features).all()): raise ValueError('nonfinite NSF rollout')
    return future_features.cpu().numpy(),future_price.cpu().numpy()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output',type=Path,default=Path('data/backtests/oracle-nsf-planning'))
    ap.add_argument('--branches',type=int,default=2)
    ap.add_argument('--planning-level',type=int,default=1)
    ap.add_argument('--stride',type=int,default=900)
    ap.add_argument('--seed',type=int,default=20260914)
    args=ap.parse_args()
    if args.branches<2 or args.branches%2 or args.stride<1: raise ValueError('even branching and positive stride required')
    root=args.output; root.mkdir(parents=True,exist_ok=True)
    if (root/'forecasts.jsonl').exists(): raise ValueError('forecast experiment exists')
    checkpoint=Path('data/backtests/oracle-latent-nsf-sampled-skips/latentSampledSkipFlow.pt')
    source_plan=json.loads((checkpoint.parent/'plan.json').read_text())
    selection=json.loads((checkpoint.parent/'selection.json').read_text())
    assert sha(checkpoint)==selection['models']['latentSampledSkipFlow']['checkpointSha256']
    data=latent_data(Path(source_plan['nativeDirectory']),Path(source_plan['dataset']),'test',HorizonRange(21600.,1))
    torch.set_num_threads(4); torch.set_float32_matmul_precision('high')
    model,_=load_latent_nsf(checkpoint,torch.device('cuda'))
    prices=data['prices']; first=int(data['origins'][0]); count=len(prices)-1
    start_seconds=float(data['timestamps'][0])/1000
    records=[]; pending=[]
    for idx in range(0,count,args.stride):
        offsets=schedule(count-idx,args.stride,args.planning_level)
        row={'index':idx,'timestamp':int((start_seconds+idx)*1000),'price':float(prices[idx]),
             'terminalTimestamp':int((start_seconds+count)*1000),'stageOffsetsSeconds':offsets,'tree':{'children':[]}}
        records.append(row)
        f=np.asarray(data['timeline'][first+idx-1:first+idx+1],dtype=np.float32)
        p=data['pricesWithPast'][idx:idx+2]
        pending.append((row['tree'],f,p,np.array([start_seconds+idx-1,start_seconds+idx]),offsets,row['timestamp']/1000))
    level=0
    while pending:
        feats=np.stack([r[1] for r in pending]); pp=np.stack([r[2] for r in pending]); tt=np.stack([r[3] for r in pending])
        hh=np.array([r[5]+r[4][0]-r[3][-1] for r in pending])
        nf,np_=next_observations(model,feats,pp,tt,hh,args.branches,args.seed+level)
        following=[]
        for n,(node,f,p,t,offsets,origin) in enumerate(pending):
            for b in range(args.branches):
                child={'probability':f'1/{args.branches}','price':float(np_[n,b]),
                    'observationKey':json.dumps(nf[n,b].tolist(),separators=(',',':')),'children':[]}
                node['children'].append(child)
                if len(offsets)>1:
                    following.append((child,np.stack((f[-1],nf[n,b])),np.array([p[-1],np_[n,b]]),
                        np.array([t[-1],origin+offsets[0]]),offsets[1:],origin))
        print(json.dumps({'stage':level,'conditionalForecasts':len(pending),'nextNodes':len(following)}),flush=True)
        pending=following; level+=1
    (root/'forecasts.jsonl').write_text(''.join(json.dumps(r,separators=(',',':'),allow_nan=False)+'\n' for r in records))
    (root/'marks.json').write_text(json.dumps({'startTimestamp':int(start_seconds*1000),'prices':prices.tolist()},allow_nan=False))
    plan={'checkpoint':str(checkpoint),'checkpointSha256':sha(checkpoint),'gamma':0,'branches':args.branches,
        'planningLevel':args.planning_level,'decisionStrideSeconds':args.stride,'seed':args.seed,
        'terminalTimestamp':records[-1]['terminalTimestamp'],'decisions':len(records),'seconds':count,
        'law':'observable two-row kernel: encoder -> latent NSF -> full Gaussian observation decoder -> re-encode at child',
        'hiddenLatentsVisibleToPolicy':False,'futureRealizedFeaturesUsedForCurrentTree':False,
        'hypotheticalObservationCaveat':'Gaussian decoded features may violate identities in real production features; this is the learned model law',
        'timeApproximation':'future decision/mark grid spans the fixed endpoint; refining level adds intermediate decisions; primitive clock is one second',
        'sourceHashes':{str(Path(__file__)):sha(__file__),'ml/latent_neural_stochastic_flow.py':sha('ml/latent_neural_stochastic_flow.py')},
        'forecastSha256':sha(root/'forecasts.jsonl'),'marksSha256':sha(root/'marks.json')}
    (root/'forecast-plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps(plan),flush=True)

if __name__=='__main__': main()
