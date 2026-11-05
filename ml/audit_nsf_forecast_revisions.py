"""Measure fixed-timestamp NSF marginal changes after actual observations.

Direct horizon queries are distinct from the recursively re-encoded 1s law.
This runner examines the former, with shared and independent Sobol controls.
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
from forecast_revision_metrics import sample_metrics, summary, fixed_target_horizons


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@torch.inference_mode()
def compare(model, data, ids, horizon, components, draws, seed, device, independent=False):
    f0,p0,t0 = timeline_inputs(data, ids, device)
    f1,p1,t1 = timeline_inputs(data, ids+1, device)
    target = t0[:,-1].cpu().numpy()+horizon
    h0 = fixed_target_horizons(t0[:,-1].cpu().numpy(), target)
    h1 = fixed_target_horizons((t0 if independent else t1)[:,-1].cpu().numpy(), target)
    elapsed = torch.tensor([0.,1.],device=device).expand(len(ids),-1)
    a = model.distribution(f0,p0,elapsed,torch.as_tensor(h0,device=device),t0,components,seed)
    b = model.distribution(f0 if independent else f1,p0 if independent else p1,elapsed,
        torch.as_tensor(h1,device=device),t0 if independent else t1,components,seed+1 if independent else seed)
    anchor = p0[:,-1]
    x = torch.log(a.sample_prices(draws,seed+101,tick_size=None)/anchor[:,None]).cpu().numpy()
    y = torch.log(b.sample_prices(draws,seed+102 if independent else seed+101,tick_size=None)/anchor[:,None]).cpu().numpy()
    result = sample_metrics(x,y)
    mean0,mean1 = a.mean_price(), b.mean_price()
    result['meanPriceRevisionBps'] = ((mean1-mean0)/anchor*1e4).cpu().numpy()
    result['absoluteMeanPriceRevisionBps'] = abs(result['meanPriceRevisionBps'])
    std0 = (a.log_means.var(1,unbiased=False)+a.log_stds.square().mean(1)).sqrt()
    std1 = (b.log_means.var(1,unbiased=False)+b.log_stds.square().mean(1)).sqrt()
    result['oldStdBps'] = (std0*1e4).cpu().numpy()
    result['stdRatioNewOverOld'] = (std1/std0).cpu().numpy()
    result['actualOneSecondMoveBps'] = (torch.log(p1[:,-1]/anchor)*1e4).cpu().numpy()
    result['shapeW1Bps'] = sample_metrics(x-x.mean(1,keepdims=True),y-y.mean(1,keepdims=True))['w1Bps']
    for bp in [-10,0,10]:
        threshold = anchor*(1+bp/1e4)
        result[f'fixedThreshold{bp}CdfChangePercentagePoints'] = ((b.cdf(threshold)-a.cdf(threshold))*100).cpu().numpy()
    return result


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--components',type=int,default=512)
    ap.add_argument('--draws',type=int,default=4096)
    ap.add_argument('--windows',type=int,default=40)
    ap.add_argument('--window-seconds',type=int,default=60)
    ap.add_argument('--stress',type=int,default=32)
    ap.add_argument('--batch',type=int,default=32)
    ap.add_argument('--horizons',type=int,nargs='+',default=[2,5,10,60,300,900,3600])
    ap.add_argument('--refinement-cases',type=int,default=96)
    ap.add_argument('--seed',type=int,default=20260934)
    args=ap.parse_args();args.output.mkdir()
    torch.set_num_threads(4);torch.set_float32_matmul_precision('high')
    checkpoint=Path('data/backtests/oracle-latent-nsf-sampled-skips/latentSampledSkipFlow.pt')
    source=json.loads((checkpoint.parent/'plan.json').read_text())
    model,_=load_latent_nsf(checkpoint,torch.device('cuda'))
    data=latent_data(Path(source['nativeDirectory']),Path(source['dataset']),'test',HorizonRange(21600.,1))
    last=len(data['prices'])-max(args.horizons)-1
    rng=np.random.default_rng(args.seed)
    edges=np.linspace(0,last-args.window_seconds,args.windows+1,dtype=int)
    starts=np.array([rng.integers(a,b) for a,b in zip(edges[:-1],edges[1:])])
    routine=(starts[:,None]+np.arange(args.window_seconds)).ravel()
    moves=abs(np.diff(np.log(data['prices'][:last+2])))
    stress=np.argsort(moves)[-args.stress:][::-1].astype(int)
    ids=np.r_[routine,stress]
    refined=np.unique(np.r_[np.linspace(0,len(routine)-1,args.refinement_cases,dtype=int),np.arange(len(routine),len(ids))])
    np.savez_compressed(args.output/'cases.npz',offsets=ids,starts=starts,refinementIndices=refined,
        timestamps=float(data['timestamps'][0])/1000+ids)
    plan=dict(**vars(args),checkpoint=str(checkpoint),checkpointSha256=sha(checkpoint),
        split='test',routinePairs=len(routine),stressPairs=len(stress),
        sourceDataPlanSha256=sha(checkpoint.parent/'plan.json'),
        law='Direct requested-horizon marginal; two actual completed observation rows; same target timestamp',
        forecastTrainingChanged=False,policyAccuracyClaim=False,
        sourceHashes={str(p):sha(p) for p in [Path(__file__),Path('ml/forecast_revision_metrics.py'),
            Path('ml/latent_neural_stochastic_flow.py'),Path('ml/oracle_continuation_value.py')]})
    plan['output']=str(args.output)
    (args.output/'plan.json').write_text(json.dumps(plan,indent=2))
    (args.output/'source').mkdir()
    for p in plan['sourceHashes']:(args.output/'source'/Path(p).name).write_bytes(Path(p).read_bytes())
    started=time.perf_counter();report={}
    for horizon in args.horizons:
        for arm,selected,components,independent in [
            ('main',np.arange(len(ids)),args.components,False),
            ('refined',refined,args.components*4,False),
            ('noise',refined,args.components,True)]:
            rows={}
            for start in range(0,len(selected),args.batch):
                chosen=selected[start:start+args.batch]
                r=compare(model,data,ids[chosen],horizon,components,args.draws,args.seed,
                    torch.device('cuda'),independent)
                for k,v in r.items():rows.setdefault(k,[]).append(v)
                if start%512==0:print(json.dumps(dict(horizon=horizon,arm=arm,cases=start,seconds=time.perf_counter()-started)),flush=True)
            rows={k:np.concatenate(v) for k,v in rows.items()}
            np.savez_compressed(args.output/f'{arm}-h{horizon}.npz',caseIndices=selected,**rows)
            report[f'{arm}-h{horizon}']={}
            for group,mask in [('routine',selected<len(routine)),('stress',selected>=len(routine))]:
                report[f'{arm}-h{horizon}'][group]={k:summary(v[mask]) for k,v in rows.items()}
            print(json.dumps(dict(horizon=horizon,arm=arm,seconds=time.perf_counter()-started,
                w1=report[f'{arm}-h{horizon}']['routine']['w1Bps'])),flush=True)
            (args.output/'report.json').write_text(json.dumps(dict(seconds=time.perf_counter()-started,results=report),indent=2))


if __name__=='__main__':main()
