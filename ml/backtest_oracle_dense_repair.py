"""Causal per-second replay with fresh NSF roots and checked value-tail repair.

Only two dense maps are retained in memory. The forecast horizon shortens at
the actual replay endpoint; terminal wealth requires an executed valid close.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from numba import get_num_threads, set_num_threads
from latent_neural_stochastic_flow import load_latent_nsf
from train_latent_neural_stochastic_flow import latent_data
from target_horizon_forecast import HorizonRange
from oracle_continuation_value import timeline_inputs
from oracle_dense_value_map import DenseValueMap,MarketCloud
from oracle_book_mc import rule_vector
from oracle_dense_controller import DenseController,DenseControllerConfig
from oracle_dense_metadata import prepared
from gpu_oracle_bellman import Rules
from oracle_dense_checkpoint import load_checkpoint
from oracle_dense_grid_config import replay_grid


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--market-source',type=Path,required=True)
    restart=ap.add_mutually_exclusive_group()
    restart.add_argument('--initial-map',type=Path,help='Reuse a verified complete full-census initial value map')
    restart.add_argument('--resume',type=Path,help='Continue a cleanly stopped replay in a new output directory')
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--seconds',type=int,help='Total original replay interval; defaults to 5400 or the resumed interval')
    ap.add_argument('--max-updates',type=int,help='Stop after this many new updates, without reporting terminal return')
    ap.add_argument('--reference-every',type=int,default=0)
    ap.add_argument('--reference-updates',type=int,nargs='*',default=[],help='Additional exact finite-field audit updates')
    ap.add_argument('--refresh-every',type=int,help='Periodically recompute the complete value field to reset repair residuals')
    ap.add_argument('--repair-tolerance',type=float)
    ap.add_argument('--repair-interval',type=int)
    ap.add_argument('--exposure-step',type=float,help='Refine the account and future-action exposure axis')
    ap.add_argument('--candidates',type=int,help='Future candidate budget; includes every zero-order exposure target')
    ap.add_argument('--max-wall-seconds',type=float,default=10800.)
    args=ap.parse_args()
    parent=json.loads((args.resume/'plan.json').read_text()) if args.resume else None
    for key,default in (('seconds',5400),('refresh_every',0),('repair_tolerance',1.e-6),('repair_interval',8)):
        if getattr(args,key) is None:setattr(args,key,parent['arguments'][key] if parent else default)
    source=json.loads((args.market_source/'plan.json').read_text());settings=source['arguments']
    H=settings['horizon']
    if (args.seconds<H or (args.max_updates is not None and args.max_updates<1) or args.reference_every<0
        or any(x<1 for x in args.reference_updates)
        or args.refresh_every<0 or not np.isfinite(args.max_wall_seconds) or args.max_wall_seconds<=0
        or not np.isfinite(args.repair_tolerance) or args.repair_tolerance<0 or args.repair_interval<1):
        raise ValueError('Replay must cover at least the initial horizon; use max-updates for a diagnostic')
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);torch.set_float32_matmul_precision('high')
    set_num_threads(min(4,get_num_threads()))
    checkpoint=Path('data/backtests/oracle-latent-nsf-sampled-skips/latentSampledSkipFlow.pt')
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=source['checkpointSha256']:raise ValueError('Checkpoint mismatch')
    training=json.loads((checkpoint.parent/'plan.json').read_text())
    model,_=load_latent_nsf(checkpoint,'cuda')
    data=latent_data(Path(training['nativeDirectory']),Path(training['dataset']),'test',HorizonRange(21600.,1))
    def observation(offset):
        return [x.cpu().numpy()[0] for x in timeline_inputs(data,np.array([settings['origin']+offset]),'cpu')]
    f,p,t=observation(0);initial=MarketCloud.load(args.market_source/'update-0'/'market.npz')
    if (initial.origin_seconds!=t[-1] or not np.array_equal(initial.features[0,0],f)
        or not np.array_equal(initial.price_pairs[0,0],p)):raise ValueError('Initial observation mismatch')
    observation(args.seconds)  # validate availability before starting a long run
    for name in ('ml/nsf_scenario_graph.py','ml/export_nsf_planning_trees.py'):
        if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=source['sourceHashes'][name]:raise ValueError('Initial market generator differs')
    grid,args.candidates=replay_grid(source,parent,args.exposure_step,args.candidates)
    settings={**settings,'candidates':args.candidates}
    rules=Rules(**source['rules']);r=rule_vector(rules,source['fundingMultiplier'])
    files=('ml/backtest_oracle_dense_repair.py','ml/oracle_dense_value_map.py','ml/oracle_dense_repair.py',
        'ml/oracle_dense_rollout.py','ml/oracle_book_gpu.py','ml/oracle_book_mc.py','ml/oracle_account_grid.py',
        'ml/nsf_scenario_graph.py','ml/export_nsf_planning_trees.py','ml/oracle_dense_checkpoint.py','ml/oracle_dense_grid_config.py',
        'ml/oracle_dense_controller.py','ml/oracle_dense_metadata.py','ml/oracle_dense_observe_batch.py')
    plan={**source,'arguments':{**settings,**{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}},
        'grid':asdict(grid),'rules':asdict(rules),'sourceHashes':{n:hashlib.sha256(Path(n).read_bytes()).hexdigest() for n in files},
        'terminal':'Fresh conditional forecasts each second; shorten horizon near endpoint; valid exact flat close and cancellation',
        'limitations':['finite account/action/market grids','conditional roundoff allowance is not machine-verified',
            'retained conditional market nodes; projection diagnostics do not certify learned-law accuracy']}
    resume=load_checkpoint(args.resume,plan,float(t[-1])) if args.resume else None
    if resume:
        f,p,t=observation(resume.update)
        saved=resume.current.cloud
        if (saved.origin_seconds!=t[-1] or not np.array_equal(saved.features[0,0],f)
            or not np.array_equal(saved.price_pairs[0,0],p)):
            raise ValueError('Saved checkpoint differs from the observed market history')
        plan.update(resumedFrom=str(resume.directory),initialArchive=str(resume.initial),
            resumedAtUpdate=resume.update,previousElapsedSeconds=resume.elapsed_seconds)
    (args.output/'plan.json').write_text(json.dumps(plan,indent=2));(args.output/'source').mkdir()
    for name in files:(args.output/'source'/Path(name).name).write_bytes(Path(name).read_bytes())
    started=time.perf_counter();initial_start=started
    if resume:
        current=resume.current;run=dict(seconds=0.,cachedInitialMap=str(resume.directory/'last-map'))
    elif args.initial_map is None:
        current=DenseValueMap(grid,initial,r,None)
        run=current.sweep(settings['seed'],settings['candidates'],settings['branches'],estimator='repair')
    else:
        from oracle_dense_repair import contract
        current=DenseValueMap.load(args.initial_map)
        if current.grid!=grid or not np.array_equal(current.rules,r):raise ValueError('Initial account contract mismatch')
        if (current.bellman_contract!=contract(settings['seed'],settings['candidates'])
            or current.bellman_residuals is None or np.any(current.bellman_residuals)
            or not np.all(current.samples==settings['branches'])):raise ValueError('Initial map must be a matching full census')
        for key in ('prices','counts','destinations','sampled_prices','features','price_pairs','origin_seconds'):
            if not np.array_equal(getattr(current.cloud,key),getattr(initial,key)):raise ValueError('Initial market map mismatch')
        run=dict(seconds=0.,cachedInitialMap=str(args.initial_map))
    if resume:
        state=resume.state;front=resume.front;initial_seconds=resume.initial_seconds
    else:
        state=np.r_[10000.,0.,p[-1],p[-1],np.zeros(3*grid.max_orders)]
        front=prepared(current).front(state,interior_samples=1,seed=settings['seed']);state=front['state']
        current.export(args.output/'initial');np.savez(args.output/'initial'/'front.npz',**front)
        initial_seconds=time.perf_counter()-initial_start
    previous_elapsed=resume.elapsed_seconds if resume else 0.
    start_update=resume.update if resume else 0
    controller=DenseController(current,model,state,DenseControllerConfig(
        float(initial.origin_seconds+args.seconds),settings['seed'],settings['candidates'],
        forecast_origin_index=settings['origin'],refresh_every=args.refresh_every,
        repair_tolerance=args.repair_tolerance,repair_interval=args.repair_interval),
        front=front,episode_origin=float(initial.origin_seconds),maximum_horizon=H)
    print(json.dumps(dict(phase='resume' if resume else 'initial',seconds=time.perf_counter()-initial_start,
        startUpdate=start_update,sweepSeconds=run['seconds'],
        cachedInitialMap=run.get('cachedInitialMap'))),flush=True)
    rows=list(resume.rows) if resume else [];comparisons=list(resume.comparisons) if resume else []
    trades=resume.market_trades if resume else int(state[1]!=0);fills=resume.fill_ticks if resume else 0
    terminal_wealth=None;failure=None;last_update=start_update;time_budget_reached=False
    max_updates=min(args.seconds,start_update+args.max_updates if args.max_updates is not None else args.seconds)
    with (args.output/'updates.jsonl').open('w') as log:
        for row in rows:log.write(json.dumps(row)+'\n')
        log.flush()
        if comparisons:(args.output/'comparisons.json').write_text(json.dumps(comparisons,indent=2))
        for update in range(start_update+1,max_updates+1):
            if time.perf_counter()-started>=args.max_wall_seconds:
                time_budget_reached=True;break
            f,p,t=observation(update)
            last_update=update
            old=current
            try:
                row=controller.advance(f,p,t)
            except RuntimeError as error:
                state=controller.state;fills+=int(controller.arrival_filled)
                failure=dict(update=update,reason=str(error),state=state.tolist());break
            current=controller.current;front=controller.front;state=controller.state
            fills+=int(row['filled']);trades+=int(row['marketTrade'])
            if controller.phase=='complete':
                terminal_wealth=controller.terminal_wealth
                rows.append(row);log.write(json.dumps(row)+'\n');log.flush();break
            cloud=current.cloud
            if (args.reference_every and update%args.reference_every==0) or update in args.reference_updates:
                reference=DenseValueMap(grid,cloud,r,None)
                reference_run=reference.sweep(settings['seed'],settings['candidates'],settings['branches'])
                mask=np.isfinite(current.values)&np.isfinite(reference.values)
                error=float(np.max(abs(current.values[mask]-reference.values[mask])))
                support=int(np.count_nonzero(np.isfinite(current.values)!=np.isfinite(reference.values)))
                root_error=float(np.max(abs(current.values[0][mask[0]]-reference.values[0][mask[0]])))
                comparison=dict(update=update,maximumFieldError=error,maximumRootError=root_error,
                    reportedAllowance=row['errorAllowance'],supportDisagreements=support,referenceSeconds=reference_run['seconds'])
                comparisons.append(comparison);print(json.dumps(dict(phase='reference',**comparison)),flush=True)
                (args.output/'comparisons.json').write_text(json.dumps(comparisons,indent=2))
                if support or error>row['errorAllowance']+1.e-12:raise AssertionError('Repair exceeded checked reference allowance')
                del reference
            rows.append(row);log.write(json.dumps(row)+'\n');log.flush()
            if update%10==0:print(json.dumps(dict(phase='replay',**row)),flush=True)
            (args.output/'progress.json').write_text(json.dumps(dict(update=update,elapsedSeconds=previous_elapsed+time.perf_counter()-started,
                markedEquity=row['markedEquity'],lastStepSeconds=row['stepSeconds'],failure=failure)))
            del old
    current.export(args.output/'last-map');np.savez(args.output/'last-map'/'front.npz',**front)
    active=[x for x in rows if 'sweepSeconds' in x]
    result=dict(completed=terminal_wealth is not None,plannedSeconds=args.seconds,observedUpdates=last_update,
        timeBudgetReached=time_budget_reached,
        failure=failure,terminalWealth=terminal_wealth,terminalReturn=(terminal_wealth/10000-1) if terminal_wealth is not None else None,
        markedEquity=float(state[0]+state[1]*state[2]),state=state.tolist(),marketTrades=trades,fillTicks=fills,
        elapsedSeconds=previous_elapsed+time.perf_counter()-started,initialSeconds=initial_seconds,comparisons=comparisons,
        resumedFrom=str(resume.directory) if resume else None,segmentStartUpdate=start_update,
        segmentSeconds=time.perf_counter()-started,
        medianStepSeconds=float(np.median([x['stepSeconds'] for x in active])) if active else None,
        maxStepSeconds=max((x['stepSeconds'] for x in active),default=None),
        interpretation='Actual causal observations; fresh root price laws each second. Terminal return exists only after valid flat completion. Accuracy checks compare the same finite market/action grid.')
    (args.output/'results.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':main()
