"""Matched H1 full-law market-order backtest of frozen ensemble candidates."""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_key]='2'
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
import numpy as np
import torch
from continuous_oracle_endpoint import Rules, continuous_decision, execute, terminal_close, log_hurdle_law
from knot_oracle_price_law import knot_price_law, mixture_law, AnalyticGammaPriceLaw
from horizon_price_forecast import load_horizon_forecast
from stochastic_price_forecast import load_timeline_split, component_log_masses
from audit_forecast_agreement import sha
from backtest_continuous_oracle import action_encoding


def write(path, value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=Path('data/backtests/oracle-refitted-h1-backtest'))
    parser.add_argument('--limit',type=int)
    args=parser.parse_args();root=args.output;root.mkdir(parents=True,exist_ok=True)
    if (root/'comparison.json').exists(): raise ValueError('completed backtest exists')
    sources={}
    def checked(file,digest=None):
        file=Path(file);actual=sha(file)
        if digest is not None and actual!=digest: raise ValueError('changed source: '+str(file))
        sources[str(file)]=actual;return file
    for file in [__file__,'ml/knot_oracle_price_law.py','ml/continuous_oracle_endpoint.py']:
        checked(file)
    archive=Path('data/backtests/oracle-expanded-mixture')
    selection=json.loads(checked(archive/'refitted-selection.json').read_text())
    chosen=np.array(selection['objectives']['mse']['weights'])
    support=['currentSpecialists','direct59Refit','base17Refit']
    indices=[selection['names'].index(name) for name in support]
    if sum(chosen[i] for i in range(len(chosen)) if i not in indices)>1e-12:
        raise ValueError('mixture support changed; load its full laws explicitly')
    mixweights=chosen[indices];mixweights/=mixweights.sum()
    current=json.loads(checked('data/backtests/oracle-joint-calibration/evaluation.json').read_text())
    row=next(r for r in current['results'] if r['split']=='test' and r['horizonSeconds']==1)
    saved=row['models']['joint-calibrated']['savedStream']
    with np.load(checked(saved['parametersFile'],saved['parametersSha256'])) as f:
        ix=f['accuracyIndices'];packed=f['parameters'][ix];origins=f['origins'][ix];times=f['timestamps'][ix]
    if args.limit: packed,origins,times=packed[:args.limit],origins[:args.limit],times[:args.limit]
    count=len(origins)
    laws={}
    for family in ('direct59','base17'):
        file=checked(archive/family/'test-law.npz')
        with np.load(file) as f:
            np.testing.assert_array_equal(f['origins'][:count],origins)
            np.testing.assert_array_equal(f['timestamps'][:count],times)
            laws[family]={k:f[k][:count] if k in ('componentMasses','zeroProbability') else f[k]
                for k in ('componentMasses','zeroProbability','knots','areas','sigma')}
    market=row['marketManifest']
    with np.load(checked(market['marketFile'],market['marketSha256'])) as f:
        offsets=origins-int(f['firstPriceOrigin'])
        first=int(offsets[0]);last=int(offsets[-1]+1)
        prices=f['prices'][first:last+1].astype(float);offsets=offsets-first
    if np.any(np.diff(offsets)!=4) or np.any(prices<=0) or np.max(abs(prices/.01-np.rint(prices/.01)))>1e-6:
        raise ValueError('invalid decision cadence or price grid')
    tape={'origins':origins.tolist(),'timestamps':times.tolist(),'offsets':offsets.tolist(),'prices':prices.tolist()}
    write(root/'tape.json',tape);checked(root/'tape.json')
    knotfile=root/'joint-knot.npz'
    checkpoint=checked('data/training/runs/stochastic-crra-direct-horizons-gaussian-v2/h1/lambda1/best.pt')
    if not knotfile.exists():
        torch.set_num_threads(2);torch.set_float32_matmul_precision('high')
        device=torch.device('cuda');model,payload=load_horizon_forecast(checkpoint,device)
        source=load_timeline_split(Path(payload['plan']['dataset']),'test')
        features=source.features(origins);zero=[];masses=[]
        with torch.inference_mode():
            for start in range(0,count,2560):
                logits=model(torch.from_numpy(features[start:start+2560]).to(device))
                z=logits[:,0].sigmoid().cpu().numpy().astype(float)
                m=component_log_masses(logits[:,1:],model.areas).exp().cpu().numpy().astype(float)
                zero.append(z);masses.append(m/m.sum(axis=1,keepdims=True))
        np.savez_compressed(knotfile,zeroProbability=np.concatenate(zero),componentMasses=np.concatenate(masses),
            knots=np.asarray(model.basis['knots']),areas=np.asarray(model.basis['basisAreas']),sigma=np.array(model.sigma),origins=origins,
            checkpointSha256=np.array(sha(checkpoint)))
        del model,features;torch.cuda.empty_cache()
    with np.load(checked(knotfile)) as f:
        np.testing.assert_array_equal(f['origins'],origins)
        if str(f['checkpointSha256'])!=sha(checkpoint): raise ValueError('knot cache checkpoint mismatch')
        laws['jointKnot']={k:f[k] for k in ('componentMasses','zeroProbability','knots','areas','sigma')}
    variants=['jointKnot','currentSpecialists','equalPair','direct59','base17','expandedMixture']
    plan={'contract':'oracle-refitted-h1-roundtrip-full-law-flat-v2','terminalRequirement':'zero-asset-and-no-orders','gamma':0,'initialWealth':10000,
        'entryBand':[0,1],'maintenanceBand':[0,1],'fundingFactor':1,'orders':0,
        'rules':asdict(Rules()),'variants':variants,'feeScales':[1,0],
        'horizonSeconds':1,'forecastCadenceSeconds':4,'decisionCount':count,
        'execution':'one optimized entry/rebalance at issue; exact full close next second within order limits; idle until next forecast',
        'actionDomain':'continuous signed actual quote flow with minimum gap and aggregate cap; zero means no trade',
        'absoluteValueTolerance':1e-4,'mixtureSupport':support,'mixtureWeights':mixweights.tolist(),
        'noRefittingOrTestSelection':True,'historicalTestPreviouslyInspected':True,
        'scope':'unleveraged one-step CRRA0 endpoint optimizer with explicit closing contract; not full multi-period Bellman or leveraged path optimality',
        'projection':'full densities integrated into nearest positive price ticks; atom retained; tails bounded, not discarded',
        'previewLimit':args.limit}
    write(root/'plan.json',plan)
    sims={}
    for variant in variants:
        for fee in (1,0):
            directory=root/f'{variant}-fees{fee}';directory.mkdir(exist_ok=True)
            sims[variant,fee]={'q':10000.,'a':0.,'peak':10000.,'drawdown':0.,'trades':0,'turnover':0.,'fees':0.,
                'exposureSum':0.,'active':0,'fractional':0,'maxGap':0.,'maxEvaluations':0,
                'stream':(directory/'decisions.jsonl').open('w'),'directory':directory}
    def mark(sim,p):
        equity=sim['q']+sim['a']*p
        if equity<=0 or min(sim['q'],sim['a']) < -1e-8: raise ValueError('inadmissible account')
        sim['peak']=max(sim['peak'],equity);sim['drawdown']=max(sim['drawdown'],1-equity/sim['peak'])
        return equity
    started=time.monotonic()
    try:
        for i,offset in enumerate(offsets):
            price=prices[offset]
            full={name:knot_price_law(float(data['zeroProbability'][i]),data['componentMasses'][i],data['knots'],
                data['areas'],float(data['sigma']),price) for name,data in laws.items()}
            full['currentSpecialists']=AnalyticGammaPriceLaw(packed[i],price)
            full['equalPair']=mixture_law([full['jointKnot'],full['currentSpecialists']],[.5,.5])
            full['expandedMixture']=mixture_law([full['currentSpecialists'],full['direct59'],full['base17']],mixweights)
            for (variant,fee),sim in sims.items():
                rules=Rules(fee=.0005*fee);q,a=sim['q'],sim['a']
                decision=continuous_decision(q,a,price,full[variant],rules,tolerance=1e-4)
                d=decision['marketQuote'];action=action_encoding(d,q,a,price,rules)
                sim['q'],sim['a']=max(0.,decision['quote']),max(0.,decision['asset'])
                wealth=mark(sim,price);exposure=sim['a']*price/wealth
                sim['exposureSum']+=exposure;sim['active']+=exposure>1e-8;sim['fractional']+=1e-8<exposure<1-1e-8
                sim['maxGap']=max(sim['maxGap'],decision['optimalityGap']);sim['maxEvaluations']=max(sim['maxEvaluations'],decision['evaluations'])
                sim['trades']+=d!=0;sim['turnover']+=abs(d)
                sim['fees']+=rules.fee*abs(d)*(1 if d<0 else 1/(1-rules.fee))
                after=[sim['q'],sim['a']]
                # Future prices are first used here, after the action is fixed.
                nextprice=prices[offset+1];mark(sim,nextprice)
                closed_q, closed_a, close=terminal_close(sim['q'],sim['a'],nextprice,rules)
                close_action=action_encoding(close,sim['q'],sim['a'],nextprice,rules)
                sim['q'],sim['a']=closed_q,closed_a;mark(sim,nextprice)
                sim['trades']+=close!=0;sim['turnover']+=close;sim['fees']+=rules.fee/(1-rules.fee)*close
                nextoffset=int(offsets[i+1]) if i+1<count else int(offset+1)
                for p in prices[offset+2:nextoffset+1]: mark(sim,p)
                record={'index':i,'before':[q,a],'action':action,'trade':d,'after':after,
                    'closeAction':close_action,'closeTrade':close,'closed':[sim['q'],sim['a']],
                    'expectedPrice':full[variant].mean,'expectedPriceError':full[variant].mean_error,
                    'valueLower':decision['valueLower'],'globalUpper':decision['globalUpper'],'gap':decision['optimalityGap'],
                    'evaluations':decision['evaluations']}
                sim['stream'].write(json.dumps(record,separators=(',',':'),allow_nan=False)+'\n')
            if i%1000==0 or i+1==count:
                print(json.dumps({'decisions':i+1,'total':count,'seconds':time.monotonic()-started,
                    'mixturePaidWealth':sims['expandedMixture',1]['q']+sims['expandedMixture',1]['a']*prices[nextoffset],
                    'mixtureFreeWealth':sims['expandedMixture',0]['q']+sims['expandedMixture',0]['a']*prices[nextoffset]}),flush=True)
    finally:
        for sim in sims.values():sim['stream'].close()
    reports=[]
    for (variant,fee),sim in sims.items():
        terminal=sim['q']+sim['a']*prices[-1];file=sim['directory']/'decisions.jsonl'
        report={'variant':variant,'feeScale':fee,'gamma':0,'complete':True,'terminalRequirement':'zero-asset-and-no-orders','returnPercent':100*(terminal/10000-1),'terminalWealth':terminal,
            'maximumDrawdownPercent':100*sim['drawdown'],'markets':int(sim['trades']),'quoteTurnover':sim['turnover'],
            'feesQuoteEquivalent':sim['fees'],'averageEntryExposure':sim['exposureSum']/count,'nonzeroExposureDecisions':int(sim['active']),
            'fractionalExposureDecisions':int(sim['fractional']),'maximumValueGap':sim['maxGap'],'maximumEvaluations':sim['maxEvaluations'],
            'terminalQuote':sim['q'],'terminalResidualAsset':sim['a'],'decisionCount':count,
            'firstTimestamp':int(times[0]),'terminalTimestamp':int(times[-1]+1000),
            'decisionsFile':str(file),'decisionsSha256':sha(file),'rules':asdict(Rules(fee=.0005*fee))}
        write(sim['directory']/'report.json',report);reports.append(report)
    benchmarks={}
    for fee in (1,0):
        rules=Rules(fee=.0005*fee);q,a=execute(10000,0,prices[0],-10000,rules)
        wealth=q+a*prices;peaks=np.maximum.accumulate(np.r_[10000.,wealth])[1:]
        try:
            terminal, _, _=terminal_close(q,a,prices[-1],rules)
            benchmarks[str(fee)]={'feasible':True,'buyAndHoldReturnPercent':100*(terminal/10000-1),
                'buyAndHoldMaximumDrawdownPercent':100*max(float(np.max(1-wealth/peaks)),1-terminal/peaks[-1]),'cashReturnPercent':0.}
        except ValueError as error:
            benchmarks[str(fee)]={'feasible':False,'reason':str(error),'buyAndHoldReturnPercent':None,'cashReturnPercent':0.}
    for file,digest in sources.items():
        if sha(file)!=digest:raise ValueError('source changed during backtest: '+file)
    write(root/'comparison.json',{'plan':plan,'reports':reports,'benchmarks':benchmarks,'sourceHashes':sources,
        'seconds':time.monotonic()-started,'executionVerified':False,'formalContinuousOptimalityCertificate':False})


if __name__=='__main__':main()
