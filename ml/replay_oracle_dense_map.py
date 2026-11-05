"""Realized one-hour replay of a FROZEN value map, with decreasing time to close.

New observations select a nearby saved market state. This is not a replay with
a refreshed NSF forecast and a fresh rolling one-hour map at every second.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from numba import set_num_threads
from latent_neural_stochastic_flow import load_latent_nsf
from train_latent_neural_stochastic_flow import latent_data
from target_horizon_forecast import HorizonRange
from oracle_continuation_value import timeline_inputs
from oracle_dense_value_map import DenseValueMap
from oracle_book_mc import move_arrive,close_terminal,effective


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--map',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    set_num_threads(2);started=time.perf_counter();m=DenseValueMap.load(args.map)
    plan=json.loads((args.map.parent/'plan.json').read_text());origin=plan['arguments']['origin']+int(args.map.name.split('-')[-1])
    checkpoint=Path('data/backtests/oracle-latent-nsf-sampled-skips/latentSampledSkipFlow.pt')
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=plan['checkpointSha256']:raise ValueError('Different forecast model')
    source=json.loads((checkpoint.parent/'plan.json').read_text())
    model,_=load_latent_nsf(checkpoint,'cpu');std=model.feature_std.detach().numpy();del model
    data=latent_data(Path(source['nativeDirectory']),Path(source['dataset']),'test',HorizonRange(21600.,1))
    H=m.shape[0];f,p,t=[x.numpy() for x in timeline_inputs(data,np.arange(origin,origin+H+1),'cpu')]
    if t[0,-1]!=m.cloud.origin_seconds:raise ValueError('Observed timeline is not the saved origin')
    load_seconds=time.perf_counter()-started;started=time.perf_counter()
    state=np.r_[10000.,0.,p[0,-1],p[0,-1],np.zeros(3*m.grid.max_orders)]
    trades=0;fills=0;peak=10000.;drawdown=0.;failure=None;rows=[]
    E=len(m.a['exposures']);C=len(m.a['coordinates'])
    with (args.output/'decisions.jsonl').open('w') as log:
        for h in range(H):
            consumed=0.;filled=False
            if h:
                state,consumed,filled,okay=move_arrive(state,p[h,-1],m.rules)
                if not okay:failure=dict(step=h,reason='Liquidation or invalid committed fills');break
            node=m.cloud.nearest(h,f[h],p[h],std);arrived=state
            if h==0:
                front=m.front(arrived,consumed,filled,1,plan['arguments']['seed']);state=front['state'];value=front['value']
            else:
                rng=np.random.default_rng(plan['arguments']['seed']+int(m.cloud.origin_seconds)+h)
                extra=max(0,min(C-E,plan['arguments']['candidates']-E))
                candidates=np.r_[np.arange(E),rng.choice(np.arange(E,C),extra,replace=False)].astype(np.int64)
                value,state,_=m.decide(arrived,consumed,filled,h,node,candidates)
            if not np.isfinite(value):failure=dict(step=h,reason='No finite continuation found');break
            trades+=int(state[1]!=arrived[1]);fills+=int(filled)
            equity=float(state[0]+state[1]*state[2]);peak=max(peak,equity);drawdown=max(drawdown,1-equity/peak)
            log.write(json.dumps(dict(step=h,price=float(p[h,-1]),equity=equity,exposure=float(state[1]*state[2]/equity),
                orders=int(np.count_nonzero(state[4::3])),marketNode=node,value=float(value),state=state.tolist()))+'\n')
            if (h+1)%600==0:print(json.dumps(dict(step=h+1,seconds=time.perf_counter()-started,equity=equity)),flush=True)
    terminal=None
    if failure is None:
        arrived,consumed,filled,okay=move_arrive(state,p[H,-1],m.rules)
        v=close_terminal(arrived,consumed,m.rules) if okay else -np.inf
        if np.isfinite(v):
            terminal=float(arrived[0]+effective(arrived,m.rules));state=arrived.copy();state[0]=terminal;state[1]=0.;state[4:]=0.
            fills+=int(filled);trades+=int(arrived[1]!=0);peak=max(peak,terminal);drawdown=max(drawdown,1-terminal/peak)
        else:failure=dict(step=H,reason='Terminal flat close is not feasible')
    result=dict(map=str(args.map),originSeconds=m.cloud.origin_seconds,horizonSeconds=H,
        forecastRefresh=False,marketStateUpdate='Nearest saved cloud representative from the currently observed feature/price pair',
        terminalWealth=terminal,returnPercent=(terminal/10000-1)*100 if terminal is not None else None,
        flatAtTerminal=terminal is not None,marketTradeDecisions=trades,ticksWithLimitFills=fills,
        maxDrawdownPercent=drawdown*100,failure=failure,seconds=time.perf_counter()-started,loadSeconds=load_seconds,
        sourceSha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        interpretation='Actual-price replay of the frozen approximate controller; not a refreshed rolling-map backtest')
    (args.output/'results.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':main()
