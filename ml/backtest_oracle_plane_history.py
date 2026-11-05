"""Historical one-second replay of the revised supporting-plane controller.

A graph is built from the episode's initial observed history only. Every live
decision uses fresh causal NSF one-second draws, with continuation functions
selected by the original observable-state graph metric. Historical next prices
are passed to execution only AFTER selection. Off-graph continuation reuse is
an explicitly measured approximation, not a transferred optimality bound.
"""
import argparse
from dataclasses import asdict
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
from numba import set_num_threads
from gpu_oracle_bellman import TensorOracle,Menu,equity
from latent_neural_stochastic_flow import load_latent_nsf
from train_latent_neural_stochastic_flow import latent_data
from target_horizon_forecast import HorizonRange
from oracle_continuation_value import timeline_inputs,conditional_children
from oracle_plane_policy import plane_policy_action,nearest_observation_nodes


def digest(path):
    with Path(path).open('rb') as source:return hashlib.file_digest(source,'sha256').hexdigest()


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--graph',type=Path,required=True);ap.add_argument('--value',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--samples',type=int,default=64)
    ap.add_argument('--forecast-batch',type=int,default=64);ap.add_argument('--seed',type=int,default=20260929)
    args=ap.parse_args();args.output.mkdir()
    if args.samples<2 or args.samples%2 or args.forecast_batch<1:raise ValueError('Invalid forecast batching')
    torch.set_num_threads(4);set_num_threads(4);torch.set_float32_matmul_precision('high');device='cuda'
    gp=json.loads((args.graph/'plan.json').read_text());construction=json.loads((args.graph/'construction.json').read_text())
    bound=json.loads((args.value/'report.json').read_text())
    graph_hash=digest(args.graph/'graph.npz')
    if graph_hash!=bound['graphSha256'] or graph_hash!=construction['graphSha256']:raise ValueError('Mismatched graph/value')
    if gp['split']!='test' or not gp['saveObservations']:raise ValueError('Require a test-origin graph with observable states')
    for name,expected in construction['observationHashes'].items():
        if digest(args.graph/name)!=expected:raise ValueError('Changed representative observations')
    checkpoint=Path('data/backtests/oracle-latent-nsf-sampled-skips/latentSampledSkipFlow.pt')
    if digest(checkpoint)!=gp['checkpointSha256']:raise ValueError('Changed forecast checkpoint')
    nsf,_=load_latent_nsf(checkpoint,torch.device(device))
    source=json.loads((checkpoint.parent/'plan.json').read_text())
    data=latent_data(Path(source['nativeDirectory']),Path(source['dataset']),'test',HorizonRange(21600.,1))
    first=int(gp['origin']);H=int(gp['horizon'])
    if first+H>=len(data['prices']):raise ValueError('Historical archive does not cover the predeclared episode')
    start=float(data['timestamps'][0])/1000+first
    if start!=gp['originTimestampSeconds'] or float(data['prices'][first])!=gp['originPrice']:
        raise ValueError('Graph was not built at this historical origin')
    market=np.load(args.graph/'graph.npz');counts=market['counts']
    rep_features=np.load(args.graph/'features.npy',mmap_mode='r')
    rep_prices=np.load(args.graph/'observation-price-ticks.npy',mmap_mode='r')
    C=np.load(args.value/'post-arrival-c.npy',mmap_mode='r');D=np.load(args.value/'post-arrival-d.npy',mmap_mode='r')
    if len(C)!=H+1 or C.shape!=D.shape:raise ValueError('Wrong continuation timeline')
    oracle=TensorOracle();menu=Menu(step=.01,limit_fractions=(.1,.25,.5,1.),limit_offsets=(1,10,100))
    desc=oracle.descriptors(menu,device)
    snapshots=args.output/'source';snapshots.mkdir();hashes={}
    for name in ('backtest_oracle_plane_history.py','oracle_plane_policy.py','oracle_certified_value.py',
                 'oracle_continuation_value.py','gpu_oracle_bellman.py','export_nsf_planning_trees.py',
                 'latent_neural_stochastic_flow.py','train_latent_neural_stochastic_flow.py'):
        p=Path(__file__).parent/name;(snapshots/name).write_bytes(p.read_bytes());hashes[name]=digest(p)
    utc=lambda t:datetime.fromtimestamp(t,timezone.utc).isoformat()
    plan=dict(resultKind='historical',arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        gamma=1,rules=asdict(oracle.r),menu=asdict(menu),fundingFactor=1,initialWealth=10000,
        startTimestampSeconds=start,endTimestampSeconds=start+H,startUtc=utc(start),endUtc=utc(start+H),
        horizonSeconds=H,archiveSeconds=H,decisionStrideSeconds=1,graphRefresh='once at the predeclared episode origin',
        horizonAtDecision='episode terminal minus current tick',
        immediateLaw='fresh conditional NSF draws from the two completed historical observation rows',
        continuation='same supporting-plane feedback as simulation, nearest observable child at the matching graph depth',
        graphSha256=graph_hash,observationHashes=construction['observationHashes'],checkpointSha256=digest(checkpoint),
        sourceHashes=hashes,actualPriceSource='immutable spot-btcusdt one-second close archive',
        forecastOriginBatchingIsRowLocal=True,futureObservedPricesUsedForSelection=False,
        continuousOptimumCertificate=None,graphBoundTransferredToHistory=False,
        episodeSelection='first complete test hour, fixed before evaluation; no parameter fitting on this replay')
    (args.output/'plan.json').write_text(json.dumps(plan,indent=2))
    marks=np.asarray(data['prices'][first:first+H+1]);np.save(args.output/'historical-prices.npy',marks)
    (args.output/'marks.json').write_text(json.dumps(dict(startTimestamp=int(start*1000),prices=marks.tolist())))
    plan['historicalPriceSha256']=digest(args.output/'historical-prices.npy')
    (args.output/'plan.json').write_text(json.dumps(plan,indent=2))
    trace=np.lib.format.open_memmap(args.output/'execution.npy',mode='w+',dtype=np.float64,shape=(H,1,3,9))
    state=torch.tensor([[10000.,0.,marks[0],marks[0],0,0,0,0,0]],device=device,dtype=torch.float64)
    started=time.perf_counter();forecasts=selection_seconds=0.;executed=markets=placements=cancels=fills=liquidations=0
    fee_market=fee_maker=0.;peak=10000.;drawdown=0.;failures=[];projection=[];latencies=[]
    with (args.output/'decisions.jsonl').open('w') as decisions:
        for block in range(0,H,args.forecast_batch):
            end=min(H,block+args.forecast_batch)
            f,p,t=timeline_inputs(data,first+np.arange(block,end),device)
            stamp=time.perf_counter()
            children=conditional_children(nsf,f,p,t,args.samples,args.seed+block)
            torch.cuda.synchronize();forecasts+=time.perf_counter()-stamp
            for local,h in enumerate(range(block,end)):
                stamp=time.perf_counter();remaining=H-h
                if float(t[local,-1])!=start+h or float(p[local,-1])!=float(state[0,2]):
                    raise ValueError('Forecast origin/account alignment failed')
                nf,np_,_=children
                cf=nf[local];cp=np_[local]
                rf=torch.tensor(np.asarray(rep_features[h+1,:counts[h+1]]),device=device)
                rp=torch.tensor(np.asarray(rep_prices[h+1,:counts[h+1]])/100,device=device)
                nodes,error=nearest_observation_nodes(cf,cp,rf,rp,nsf.feature_mean,nsf.feature_std,marks[0])
                price_error=np.abs(np.log(cp[:,-1].cpu().numpy()/(np.asarray(rep_prices[h+1,:counts[h+1]])[nodes,-1]/100)))*10000
                projection.append((float(error.mean()),float(error.max()),float(price_error.mean()),float(price_error.max())))
                result=plane_policy_action(oracle,state,desc,cp[None,:,-1],nodes[None,:],
                    np.asarray(C[h+1,:counts[h+1]]),np.asarray(D[h+1,:counts[h+1]]),remaining)
                if not result['valid'].item():
                    failures.append(dict(tick=h,reason='No feasible finite selected action on the actual account',state=state[0].tolist()));break
                selected=result['state'];amount=float(result['amount'][0])
                arrived,_,did_fill,arrival_ok=oracle.arrival(state)
                if not arrival_ok.item():raise ArithmeticError('Selected action has an invalid arrival')
                fills+=int(did_fill.item());markets+=amount!=0
                fee_maker+=float(torch.where(did_fill,state[:,6]*oracle.r.maker,0.).sum())
                fee_market+=abs(amount)*(oracle.r.taker if amount<0 else oracle.r.taker/(1-oracle.r.taker))
                descriptor=result['descriptor']
                if descriptor is None:cancels+=int(state[0,4]!=0);raw=None
                else:
                    raw=[None if math.isnan(x) else x for x in descriptor[0].tolist()]
                    placements+=raw[2]!=0;cancels+=raw[1]!=0
                # Historical price is used only after selecting and freezing this action.
                next_state=oracle.advance(selected,float(marks[h+1]))
                liquidations+=int((equity(selected)>0).item() and (equity(next_state)==0).item())
                trace[h,0,0]=state[0].cpu().numpy();trace[h,0,1]=selected[0].cpu().numpy();trace[h,0,2]=next_state[0].cpu().numpy()
                w=float(equity(next_state)[0]);peak=max(peak,w);drawdown=max(drawdown,1-w/peak)
                elapsed=time.perf_counter()-stamp;selection_seconds+=elapsed;latencies.append(elapsed);executed+=1
                decisions.write(json.dumps(dict(index=h,forecastOriginTimestampSeconds=float(t[local,-1]),remainingSeconds=remaining,
                    descriptor=raw,marketQuote=amount,limitFill=bool(did_fill.item()),
                    after=selected[0].tolist(),nextMarkedState=next_state[0].tolist(),
                    approximateActionLogValue=float(result['value'][0]),nextMarkedWealth=w,
                    projectionMeanPriceErrorBps=float(price_error.mean()),projectionMaximumPriceErrorBps=float(price_error.max())))+'\n')
                state=next_state
                if (h+1)%300==0:
                    decisions.flush();print(json.dumps(dict(decisions=executed,wealth=w,markets=markets,placements=placements,
                        fills=fills,seconds=time.perf_counter()-started)),flush=True)
            if failures:break
    complete=executed==H and not failures
    terminal=float(oracle.terminal(state)[0]) if complete else None
    if complete and (not math.isfinite(terminal) or float(state[0,1])!=0 or float(state[0,4])!=0):
        failures.append(dict(tick=H,reason='Final state is not a valid actual flat account'));complete=False;terminal=None
    trace.flush();np.save(args.output/'terminal-wealth.npy',np.array([terminal]) if complete else np.array([]))
    projection=np.array(projection);np.save(args.output/'projection-errors.npy',projection)
    report=dict(resultKind='historical',complete=complete,paths=1,decisions=executed,horizonSeconds=H,gamma=1,
        startUtc=utc(start),endUtc=utc(start+H),initialWealth=10000,terminalWealth=terminal,
        terminalReturnPercent=None if terminal is None else 100*(terminal/10000-1),
        terminalLogReturn=None if terminal is None or terminal<=0 else math.log(terminal/10000),
        maximumDrawdownPercent=100*drawdown,marketQuoteFees=fee_market,makerQuoteFees=fee_maker,
        totalFeesAtExecutionPrices=fee_market+fee_maker,markets=markets,placements=placements,cancellations=cancels,
        fills=fills,liquidations=liquidations,failures=failures,exposureGrid=menu.step,
        underlyingAssetReturnPercent=100*(float(marks[-1])/float(marks[0])-1),cashReturnPercent=0,
        seconds=time.perf_counter()-started,forecastingSeconds=forecasts,selectionSeconds=selection_seconds,
        meanDecisionSeconds=float(np.mean(latencies)) if latencies else None,
        projectionMeanPriceErrorBps=float(projection[:,2].mean()) if len(projection) else None,
        projectionMaximumPriceErrorBps=float(projection[:,3].max()) if len(projection) else None,
        source='historical spot BTCUSDT prices with declared Oracle fees and unit funding',
        scope='One predeclared test hour; fresh one-second forecasts and a reused origin continuation graph. No historical profitability or optimality guarantee beyond this measured interval.')
    (args.output/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)


if __name__=='__main__':main()
