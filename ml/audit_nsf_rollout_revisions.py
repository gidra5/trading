"""Uncompressed 1s-kernel fixed-target revisions, coupled by future shocks.

The old forecast samples its first second; the new forecast starts at the
actual next observation. Both then advance to identical absolute timestamps.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from latent_neural_stochastic_flow import load_latent_nsf
from train_latent_neural_stochastic_flow import latent_data
from target_horizon_forecast import HorizonRange
from oracle_continuation_value import timeline_inputs
from forecast_revision_metrics import sample_metrics, summary


@torch.inference_mode()
def one_second(model, f, p, t, noise):
    obs=model.observations(f,(p/p[:,-1,None]).log())
    dt=torch.cat((torch.zeros_like(t[:,:1]),t[:,1:]-t[:,:-1]),1).float()
    mu,std=model.encode(obs,dt,t)
    d=model.dimensions
    initial=mu[:,-1]+std[:,-1]*noise[:,:d]
    latent=model.transition.rsample(initial,obs.new_empty((len(obs),0)),
        obs.new_full((len(obs),),1/model.time_scale),t[:,-1]/model.time_scale,noise[:,d:2*d])
    mean,scale=model.emission(latent)
    emitted=mean+scale*noise[:,-60:]
    price=(p[:,-1]*torch.exp(emitted[:,0].double()*model.price_scale)/.01).round().clamp_min(1)*.01
    features=emitted[:,1:]*model.feature_std+model.feature_mean
    return torch.stack((f[:,-1],features),1),torch.stack((p[:,-1],price),1),torch.stack((t[:,-1],t[:,-1]+1),1)


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--direct',type=Path,required=True)
    ap.add_argument('--paths',type=int,default=512)
    ap.add_argument('--routine',type=int,default=16)
    ap.add_argument('--stress',type=int,default=8)
    ap.add_argument('--horizons',type=int,nargs='+',default=[2,5,10,60,300,900,3600])
    ap.add_argument('--seed',type=int,default=20260935)
    args=ap.parse_args();args.output.mkdir()
    torch.set_num_threads(4);torch.set_float32_matmul_precision('high')
    device=torch.device('cuda')
    checkpoint=Path('data/backtests/oracle-latent-nsf-sampled-skips/latentSampledSkipFlow.pt')
    source=json.loads((checkpoint.parent/'plan.json').read_text())
    model,_=load_latent_nsf(checkpoint,device)
    data=latent_data(Path(source['nativeDirectory']),Path(source['dataset']),'test',HorizonRange(21600.,1))
    direct_plan=json.loads((args.direct/'plan.json').read_text())
    cases=np.load(args.direct/'cases.npz')
    indices=np.r_[np.linspace(0,direct_plan['routinePairs']-1,args.routine,dtype=int),
        np.arange(direct_plan['routinePairs'],direct_plan['routinePairs']+args.stress)]
    offsets=cases['offsets'][indices]
    repeated=np.repeat(offsets,args.paths)
    old=timeline_inputs(data,repeated,device)
    new=timeline_inputs(data,repeated+1,device)
    anchors=old[1][:,-1].reshape(len(offsets),args.paths)
    rng=torch.Generator(device=device).manual_seed(args.seed)
    started=time.perf_counter()
    n=len(repeated);d=2*model.dimensions+60
    old=one_second(model,*old,torch.randn((n,d),generator=rng,device=device))
    state=tuple(torch.cat((a,b),0) for a,b in zip(old,new))
    assert torch.equal(state[2][:n],state[2][n:])
    sources=[Path(__file__),Path('ml/forecast_revision_metrics.py'),Path('ml/latent_neural_stochastic_flow.py'),Path('ml/export_nsf_planning_trees.py')]
    plan=dict(paths=args.paths,seed=args.seed,routine=args.routine,stress=args.stress,horizons=args.horizons,
        checkpointSha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        offsets=offsets.tolist(),directCaseIndices=indices.tolist(),
        law='Same observable one-second NSF kernel as planner, no node compression; shared IID shocks at matched future timestamps',
        sourceHashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        valueFunctionEvaluated=False,fullPathReweightingClaim=False)
    (args.output/'source').mkdir()
    for p in sources:(args.output/'source'/p.name).write_bytes(p.read_bytes())
    (args.output/'plan.json').write_text(json.dumps(plan,indent=2))
    report={}
    for elapsed in range(2,max(args.horizons)+1):
        noise=torch.randn((n,d),generator=rng,device=device)
        state=one_second(model,*state,torch.cat((noise,noise),0))
        if elapsed in args.horizons:
            p=state[1][:,-1]
            x=torch.log(p[:n].reshape(len(offsets),args.paths)/anchors).cpu().numpy()
            y=torch.log(p[n:].reshape(len(offsets),args.paths)/anchors).cpu().numpy()
            metrics=sample_metrics(x,y)
            metrics['oldStdBps']=x.std(1)*1e4
            metrics['absoluteMeanLogPriceRevisionBps']=abs(y.mean(1)-x.mean(1))*1e4
            metrics['stdRatioNewOverOld']=y.std(1)/x.std(1)
            # Continuation can depend on the whole observable feature pair.
            # Compare its common-noise correction in training-standardized units.
            f=state[0].reshape(2,len(offsets),args.paths,2,59)/model.feature_std
            delta=f[1]-f[0]
            metrics['featurePairRmsTrainingStd']=delta.square().mean((1,2,3)).sqrt().cpu().numpy()
            metrics['featureDeltaVarianceOverFresh']=(delta.var(1,unbiased=True).mean((1,2))/
                f[1].var(1,unbiased=True).mean((1,2))).cpu().numpy()
            # Independent-law baseline destroys pairing but uses the same old
            # population: two disjoint halves are independent IID path sets.
            half=args.paths//2
            independent=sample_metrics(x[:,:half],x[:,half:2*half])
            metrics['independentHalfSampleW1Bps']=independent['w1Bps']
            np.savez_compressed(args.output/f'h{elapsed}.npz',oldLogPrices=x,newLogPrices=y,**metrics)
            report[str(elapsed)]={}
            for name,sl in [('routine',slice(0,args.routine)),('stress',slice(args.routine,None))]:
                report[str(elapsed)][name]={k:summary(v[sl]) for k,v in metrics.items()}
            print(json.dumps(dict(elapsed=elapsed,seconds=time.perf_counter()-started,
                w1=report[str(elapsed)]['routine']['w1Bps'],
                variance=report[str(elapsed)]['routine']['priceDeltaVarianceOverFresh'])),flush=True)
            (args.output/'report.json').write_text(json.dumps(dict(seconds=time.perf_counter()-started,results=report),indent=2))
        elif elapsed%300==0:print(json.dumps(dict(elapsed=elapsed,seconds=time.perf_counter()-started)),flush=True)


if __name__=='__main__':main()
