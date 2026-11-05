"""Independently audit maturity, adapter traces, predictions and export means."""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='4'
import argparse
import json
from pathlib import Path
import math
import numpy as np
import torch
from scipy.special import expit

from train_neural_sign_size_forecast import sha,write_json
from train_per_horizon_log_hurdle import load_combined,verified
from stochastic_price_forecast import load_timeline_split


def output_transform(raw,theta):
    # Independent vectorized implementation in physical/log-odds coordinates.
    result=raw.copy();result[:,:2]+=theta[:,:2]
    alpha=raw[:,4]
    result[:,2]=np.maximum(1e-12,np.minimum(.49*alpha,
        raw[:,2]*np.exp(theta[:,2])/(1+raw[:,2]/alpha*np.expm1(theta[:,2]))))
    result[:,3]=np.maximum(1e-12,raw[:,3]*np.exp(theta[:,3]))
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=Path('data/backtests/oracle-joint-calibration'))
    args=parser.parse_args();root=args.root
    def read(path):return json.loads(Path(path).read_text())
    evaluation=read(root/'evaluation.json');selection=read(root/'selection.json');plan=read(root/'plan.json')
    source=read(selection['baseSelectionFile']);baseline=read(evaluation['baselineFile'])
    if evaluation['testUsedForSelection'] or selection['testLoaded'] or not selection['allSettingsFrozen'] \
            or plan['hyperparameterSearch'] or not plan['settingsFrozenBeforeTest']:
        raise ValueError('unfrozen protocol')
    if sha(root/'plan.json')!=evaluation['planSha256'] or sha(root/'selection.json')!=evaluation['selectionSha256']:
        raise ValueError('plan or selection changed')
    for path,digest in evaluation['artifactHashes'].items():
        if sha(Path(path))!=digest:raise ValueError('artifact changed: '+path)
    torch.set_num_threads(4);torch.set_float32_matmul_precision('high')
    torch._dynamo.config.recompile_limit=64
    torch._dynamo.config.fail_on_recompile_limit_hit=True
    cache=Path('data/training/cache/torchinductor').resolve()
    os.environ['TORCHINDUCTOR_CACHE_DIR'],os.environ['TRITON_CACHE_DIR']=str(cache),str(cache/'triton')
    results=[];queries=0;step_count=0;inference_rows=0;max_metric_error=0.;max_logit_error=0.;max_mean_relative_error=0.
    for split in ('validation','test'):
        timeline=load_timeline_split(Path(source['plan']['dataset']),split)
        for row in (r for r in evaluation['results'] if r['split']==split):
            h=row['horizonSeconds'];entry=next(r for r in source['horizons'] if r['horizonSeconds']==h)
            old=next(r for r in baseline['results'] if r['split']==split and r['horizonSeconds']==h)
            if row['sampleHashes']!=old['sampleHashes']:raise ValueError('comparison populations differ')
            with np.load(old['models']['finetunedRaw']['savedStream']['parametersFile']) as z:raw=z['parameters']
            with np.load(row['marketManifest']['marketFile']) as z:prices=z['prices'];first=int(z['firstPriceOrigin'])
            ck=verified(entry['finetuned']);model=load_combined(entry['finetuned'],'cuda').requires_grad_(False)
            for expert in (model.zero,model.sign,model.size):
                expert.forward=torch.compile(expert.forward,dynamic=True,options={'triton.cudagraphs':False})
            for variant in ('output-only-calibrated','joint-calibrated'):
                metric=row['models'][variant];saved=metric['savedStream'];config=metric['config']
                if config!=selection['configs'][str(h)][variant]:raise ValueError('configuration changed')
                with np.load(saved['parametersFile']) as z:
                    predicted=z['parameters'];origins=z['origins'];stamps=z['timestamps'];accuracy=z['accuracyIndices']
                with np.load(saved['traceFile']) as z:trace={k:z[k] for k in z.files}
                updates=read(saved['updatesFile'])['events'];queries+=len(stamps);step_count+=len(updates)
                width=config['update_seconds']//config['cadence_seconds'];latest=-1;event_index=0
                for start in range(0,len(stamps),width):
                    end=min(start+width,len(stamps));now=int(stamps[start])
                    # Independently enumerate rather than call the live selector.
                    eligible=np.flatnonzero((np.arange(len(stamps))<start)&(stamps+h*1000<=now))
                    available=len(eligible)
                    window=eligible[stamps[eligible]+h*1000>=now-config['window_seconds']*1000]
                    if len(window):
                        edges=np.linspace(int(window[0]),int(window[-1])+1,min(config['batch_size'],len(window))+1,dtype=np.int64)
                        chosen=(edges[:-1]+edges[1:]-1)//2
                        weights=np.exp(-math.log(2)*(now-stamps[window]-h*1000)/(config['half_life_seconds']*1000))
                        weights*=min(1.,config['cadence_seconds']/h)
                        event=updates[event_index];event_index+=1
                        if event['timestamp']!=now or event['firstRepresentative']!=int(chosen[0]) \
                                or event['lastRepresentative']!=int(chosen[-1]) or event['representatives']!=len(chosen) \
                                or event['availableMaturedObservations']!=available:
                            raise ValueError('online schedule/representatives differ')
                        np.testing.assert_allclose(event['effectiveWeight'],weights.sum(),rtol=3e-14,atol=1e-14)
                        latest=max(latest,int(stamps[chosen[-1]]+h*1000))
                        if event['latestMaturityUsed']!=int(stamps[chosen[-1]]+h*1000):raise ValueError('wrong label maturity')
                    for key,value in (('maturedObservations',available),('latestMaturityUsed',latest),
                            ('lastUpdateTimestamp',-1 if not event_index else updates[event_index-1]['timestamp'])):
                        np.testing.assert_array_equal(trace[key][start:end],np.full(end-start,value))
                    for key in ('coefficients','inputLogGain','inputShift'):
                        np.testing.assert_array_equal(trace[key][start:end],np.repeat(trace[key][start:start+1],end-start,axis=0))
                if event_index!=len(updates):raise ValueError('extra optimizer event')
                cold=trace['lastUpdateTimestamp']<0
                np.testing.assert_array_equal(predicted[cold],raw[cold])
                np.testing.assert_array_equal(predicted[:,4:],raw[:,4:])
                if (np.abs(trace['inputLogGain'])>np.float32(math.log(2))).any() \
                        or (np.abs(trace['inputShift'])>1).any() \
                        or (np.abs(trace['coefficients'])>np.array([2,2,math.log(2),math.log(2)])).any():
                    raise ValueError('adapter bounds violated')
                if not config['input_enabled']:
                    np.testing.assert_array_equal(trace['inputLogGain'],np.zeros_like(trace['inputLogGain']))
                    np.testing.assert_array_equal(trace['inputShift'],np.zeros_like(trace['inputShift']))
                    np.testing.assert_allclose(predicted,output_transform(raw,trace['coefficients']),rtol=5e-15,atol=1e-15)
                else:
                    if not any(e['inputGradientNorm']>0 and e['outputGradientNorm']>0 for e in updates):
                        raise ValueError('both ends were not jointly trained')
                    # Independently reconstruct adapted inputs from the saved
                    # trace. Preserve live batch sizes and compiled numerics:
                    # high/TF32 matmul changes slightly with inference batch size.
                    # Cold forecasts are already checked exactly against cache.
                    mean=ck['inputMean'].cuda();std=ck['inputStd'].cuda()
                    for start in range(0,len(stamps),width):
                        end=min(start+width,len(stamps));idx=np.arange(start,end)
                        if cold[start]:continue
                        x=np.asarray(timeline.timeline[origins[idx,None]+np.array([-1,0])],dtype=np.float32)
                        x=torch.as_tensor(x,device='cuda')
                        gains=torch.as_tensor(trace['inputLogGain'][idx,None,:],device='cuda')
                        shifts=torch.as_tensor(trace['inputShift'][idx,None,:],device='cuda')
                        x=x+(x-mean)*torch.expm1(gains)+std*shifts
                        with torch.no_grad():p=model(x).cpu().numpy()[:,0]
                        expected=output_transform(p,trace['coefficients'][idx]);actual=predicted[idx]
                        le=float(np.max(np.abs(expected[:,:2]-actual[:,:2])))
                        me=float(np.max(np.abs(expected[:,2:4]-actual[:,2:4])/np.maximum(actual[:,2:4],1e-12)))
                        max_logit_error=max(max_logit_error,le);max_mean_relative_error=max(max_mean_relative_error,me)
                        if le>1e-3 or me>1e-3:raise ValueError(f'adapted replay differs: logits {le}, means {me}')
                        inference_rows+=len(idx)
                offset=origins[accuracy]-first;y=prices[offset+h]/prices[offset]-1;logy=np.log1p(y)
                p0=expit(predicted[accuracy,0]);pup=expit(predicted[accuracy,1])
                up,down=predicted[accuracy,2],predicted[accuracy,3]
                a,b=predicted[accuracy,4],predicted[accuracy,5]
                for coordinate,target,u,d in (('logReturns',logy,up,down),
                        ('simpleReturns',y,np.expm1(-a*np.log1p(-up/a)),-np.expm1(-b*np.log1p(down/b)))):
                    conditional=pup*u-(1-pup)*d;deployed=(1-p0)*conditional;active=target!=0
                    for key,pred,mask in (('allTickDeployedMseSkillVsZero',deployed,np.ones(len(y),dtype=bool)),
                            ('activeDeployedMseSkillVsZero',deployed,active),('activeConditionalMseSkillVsZero',conditional,active)):
                        skill=1-np.square(target[mask]-pred[mask]).mean()/np.square(target[mask]).mean()
                        error=abs(skill-metric[coordinate][key]);max_metric_error=max(max_metric_error,error)
                        if error>1e-11:raise ValueError('independent MSE mismatch')
                results.append({'split':split,'horizonSeconds':h,'variant':variant,'canonicalIssues':len(stamps),
                    'optimizerSteps':len(updates),'coldForecasts':int(cold.sum()),'allChecksPassed':True})
            del model;torch.cuda.empty_cache()
    result={'allChecksPassed':True,'canonicalQueriesChecked':queries,'optimizerStepsChecked':step_count,
        'adaptedNeuralPredictionsChecked':inference_rows,'maximumReplayedLogitError':max_logit_error,
        'maximumReplayedRelativeMagnitudeError':max_mean_relative_error,
        'maximumIndependentSkillError':max_metric_error,'streams':results,
        'scope':'Independent causal schedule/weights, immutable base provenance, bounds/cold identity, complete adapted forward replay with the live compiled batch schedule and log/simple scores; seven unit tests cover end-to-end gradients, future perturbation and restart.'}
    write_json(root/'calibration-diagnostics.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='streams'}))


if __name__=='__main__':main()
