"""Prequential root-sampling audit on advancing observed account states.

Cached controls receive only selected current outcomes. The separate full-row
reference grades choices and never updates sampler state. The archived future
field stays fixed at its original terminal date.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from numba import set_num_threads
from latent_neural_stochastic_flow import load_latent_nsf
from train_latent_neural_stochastic_flow import latent_data
from target_horizon_forecast import HorizonRange
from oracle_continuation_value import timeline_inputs
from export_nsf_planning_trees import next_observations
from oracle_dense_rollout import coordinates
from oracle_dense_value_map import DenseValueMap
from oracle_book_mc import move_arrive
from oracle_root_candidates import root_candidates
from oracle_local_market_refresh import candidate_schedule
from oracle_selected_outcomes import SelectedOutcomes
from oracle_guided_sampling import sample_plan, corrected_estimate


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--map',type=Path,required=True)
    p.add_argument('--replay',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--start',type=int,default=1)
    p.add_argument('--steps',type=int,default=24)
    p.add_argument('--seeds',type=int,default=16)
    args = p.parse_args()
    if min(args.start,args.steps,args.seeds)<1: raise ValueError('Positive audit ranges required')
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2); set_num_threads(4)
    started = time.perf_counter()
    m = DenseValueMap.load(args.map)
    source = json.loads((args.map.parent/'plan.json').read_text()); settings = source['arguments']
    checkpoint = Path('data/backtests/oracle-latent-nsf-sampled-skips/latentSampledSkipFlow.pt')
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=source['checkpointSha256']:
        raise ValueError('Checkpoint differs from reference field')
    model,_ = load_latent_nsf(checkpoint,'cuda')
    training = json.loads((checkpoint.parent/'plan.json').read_text())
    data = latent_data(Path(training['nativeDirectory']),Path(training['dataset']),'test',HorizonRange(21600.,1))
    log = {row['update']:row for row in map(json.loads,(args.replay/'updates.jsonl').read_text().splitlines())}
    with np.load(args.map/'front.npz') as initial: initial_state = initial['state']
    caches = {}; trials = []; frames = []
    B = m.cloud.destinations.shape[-1]
    for offset in range(args.start,args.start+args.steps):
        if offset+2>=m.shape[0] or offset not in log or (offset>1 and offset-1 not in log):
            raise ValueError('Requested observations/continuation are unavailable')
        frame_start = time.perf_counter()
        f,prices,times = [x.numpy()[0] for x in timeline_inputs(data,np.array([settings['origin']+offset]),'cpu')]
        if prices[-1]!=log[offset]['price']: raise ValueError('Observation and replay price differ')
        before = initial_state if offset==1 else np.array(log[offset-1]['state'])
        arrived,used,filled,okay = move_arrive(before,float(prices[-1]),m.rules)
        if not okay: raise ValueError('Actual account arrival failed')
        ids,states,slots = root_candidates(m,arrived,used,filled,seed=settings['seed']+offset)
        action_count = len(states)
        reference_state = np.r_[10000.,0.,prices[-1],prices[-1],np.zeros(3*m.grid.max_orders)]
        ids = np.r_[ids,slots]; states = np.r_[states,reference_state[None]]
        future_f,future_p = next_observations(model,f[None],prices[None],times[None],np.ones(1),B,
            settings['seed']+settings['origin']+offset)
        raw_f = np.stack((np.repeat(f[-1,None],B,axis=0),future_f[0]),axis=1)
        raw_p = np.c_[np.full(B,prices[-1]),future_p[0]]
        n = m.cloud.counts[offset+1]
        x = coordinates(raw_f,raw_p,model,1.,prices[-1])
        y = coordinates(m.cloud.features[offset+1,:n],m.cloud.price_pairs[offset+1,:n],model,1.,prices[-1])
        children = (x[:,None]-y[None]).square().sum(2).argmin(1).cpu().numpy()
        evaluator = SelectedOutcomes(m,states,np.tile(future_p[0],(len(states),1)),
            np.tile(children,(len(states),1)),m.values[offset+1],m.cloud.factors(offset+1),
            candidate_schedule(m,offset,settings['seed'],settings['candidates']),m.shape[0]-offset)
        setup_seconds = time.perf_counter()-frame_start
        census = np.full(len(states),B,dtype=np.int64)
        identity = np.tile(np.arange(B),(len(states),1))
        t = time.perf_counter(); reference = evaluator.evaluate(identity,census)
        reference_seconds = time.perf_counter()-t
        truth = reference.mean(1); best = int(truth[:action_count].argmax()); optimum = truth[best]
        if not np.isfinite(optimum): raise ValueError('No finite reference action')
        support = evaluator.supported.copy(); support[-1] = False
        for method,exploration in (('uniform',1.),('two_tail',.2)):
            for seed in range(args.seeds):
                key = (method,seed)
                if key not in caches: caches[key] = np.full((slots+1,B),np.nan)
                cache = caches[key]; controls = cache[ids]
                t = time.perf_counter()
                plan = sample_plan(controls,support,4,2026091500+1000*seed+offset,exploration=exploration)
                outcomes = evaluator.evaluate(plan.indices,plan.counts)
                center = outcomes[-1].mean()
                centered = outcomes-np.take_along_axis(np.broadcast_to(outcomes[-1],outcomes.shape),plan.indices,axis=1)
                estimate,variance,updated = corrected_estimate(controls,centered,plan)
                estimate += center
                # Only sampled values and mandatory support censuses enter history.
                cache[ids] = updated
                seconds = time.perf_counter()-t
                finite = np.isfinite(truth[:action_count])
                if not np.array_equal(np.isfinite(estimate[:action_count]),finite):
                    raise AssertionError('Sampling changed the finite-law feasibility result')
                chosen = int(estimate[:action_count].argmax())
                difference = estimate[:action_count][finite]-truth[:action_count][finite]
                trials.append(dict(offset=offset,method=method,seed=seed,seconds=seconds,
                    meanDraws=float(plan.counts.mean()),sampledRows=int(np.sum(plan.counts<B)),
                    sameAction=bool(np.array_equal(states[chosen],states[best])),
                    regretBps=float(-np.expm1(truth[chosen]-optimum)*10000),
                    meanSquaredValueError=float(np.mean(difference**2)),
                    maximumValueErrorBps=float(abs(difference).max()*10000)))
        frames.append(dict(offset=offset,price=float(prices[-1]),actionProposals=action_count,
            setupSeconds=setup_seconds,censusSeconds=reference_seconds,referenceState=states[best].tolist(),
            referenceValue=float(optimum)))
        print(json.dumps(dict(offset=offset,setupSeconds=setup_seconds,censusSeconds=reference_seconds)),flush=True)
    summaries = []
    for method in ('uniform','two_tail'):
        rows = [x for x in trials if x['method']==method and x['offset']>args.start]
        if not rows: rows = [x for x in trials if x['method']==method]
        summaries.append(dict(method=method,trials=len(rows),
            medianUpdateSeconds=float(np.median([x['seconds'] for x in rows])),
            meanConditionalDraws=float(np.mean([x['meanDraws'] for x in rows])),
            actionAgreement=float(np.mean([x['sameAction'] for x in rows])),
            meanRegretBps=float(np.mean([x['regretBps'] for x in rows])),
            p95RegretBps=float(np.quantile([x['regretBps'] for x in rows],.95)),
            maximumRegretBps=max(x['regretBps'] for x in rows),
            rmsValueErrorBps=float(np.sqrt(np.mean([x['meanSquaredValueError'] for x in rows]))*10000)))
    result = dict(start=args.start,steps=args.steps,seeds=args.seeds,elapsedSeconds=time.perf_counter()-started,
        summaries=summaries,frames=frames,sourceMap=str(args.map),replay=str(args.replay),
        sourceHashes={name:hashlib.sha256(Path('ml',name).read_bytes()).hexdigest()
            for name in ('oracle_guided_sampling.py','oracle_selected_outcomes.py','oracle_root_candidates.py','audit_oracle_guided_sequence.py')},
        scope='Advancing observed states with causal fresh NSF roots and sampled-only cache updates. Complete generated root proposals, fixed archived distant field. Regret is against the eight-edge root reference on the SAME account, not a separate policy backtest or full optimality certificate.')
    (args.output/'results.json').write_text(json.dumps(result,indent=2))
    (args.output/'trials.json').write_text(json.dumps(trials))
    print(json.dumps(dict(summaries=summaries,elapsedSeconds=result['elapsedSeconds'])),flush=True)


if __name__=='__main__': main()
