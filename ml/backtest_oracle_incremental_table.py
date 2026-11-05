"""Causal replay of the resident, incrementally sampled one-hour table."""
import argparse
from dataclasses import replace,asdict
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from numba import get_num_threads,set_num_threads
from latent_neural_stochastic_flow import load_latent_nsf
from train_latent_neural_stochastic_flow import latent_data
from target_horizon_forecast import HorizonRange
from oracle_continuation_value import timeline_inputs
from oracle_dense_value_map import DenseValueMap
from oracle_dense_controller import DenseControllerConfig
from oracle_incremental_table import TableBudget
from oracle_incremental_controller import IncrementalController
from oracle_incremental_reference import full_reference
from oracle_incremental_checkpoint import save_table


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--initial-map',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seconds',type=int,default=5400)
    p.add_argument('--planning-horizon',type=int,help='Shorter diagnostic horizon; solve its initial table afresh')
    p.add_argument('--max-updates',type=int)
    p.add_argument('--layers',type=int,default=4)
    p.add_argument('--rows',type=int,default=512)
    p.add_argument('--draws',type=int,default=4)
    p.add_argument('--root-draws',type=int,default=8)
    p.add_argument('--exploration',type=float,default=.2)
    p.add_argument('--reference-map',type=Path)
    p.add_argument('--reference-updates',type=int,nargs='*',default=[],
        help='Recompute the complete finite Bellman table at these observed updates for independent grading')
    p.add_argument('--max-wall-seconds',type=float,default=10800.)
    p.add_argument('--checkpoint-every',type=int,default=0,help='Persist the table and sampling state every N accepted observations')
    args = p.parse_args()
    if args.seconds<1 or (args.max_updates is not None and args.max_updates<1) or args.max_wall_seconds<=0:
        raise ValueError('Invalid replay interval')
    if any(x<1 or x>=args.seconds for x in args.reference_updates): raise ValueError('Invalid reference update')
    if args.checkpoint_every<0: raise ValueError('Invalid checkpoint interval')
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2); torch.set_float32_matmul_precision('high'); set_num_threads(min(4,get_num_threads()))
    initial = DenseValueMap.load(args.initial_map)
    source = json.loads((args.initial_map.parent/'plan.json').read_text()); settings = source['arguments']
    cold_sweep = None
    if args.planning_horizon is not None:
        H = args.planning_horizon
        if not 2<=H<=initial.shape[0]: raise ValueError('Diagnostic horizon outside saved cloud')
        c = initial.cloud
        cloud = replace(c,prices=c.prices[:H+1],counts=c.counts[:H+1],masses=c.masses[:H+1],
            features=c.features[:H+1],price_pairs=c.price_pairs[:H+1],destinations=c.destinations[:H],
            sampled_prices=c.sampled_prices[:H])
        initial = DenseValueMap(initial.grid,cloud,initial.rules,None)
        cold_sweep = initial.sweep(settings['seed'],settings['candidates'],cloud.destinations.shape[-1],backend='cuda')['seconds']
    if args.seconds<initial.shape[0]: raise ValueError('Replay must cover the initial planning horizon')
    checkpoint = Path('data/backtests/oracle-latent-nsf-sampled-skips/latentSampledSkipFlow.pt')
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=source['checkpointSha256']: raise ValueError('Checkpoint changed')
    model,_ = load_latent_nsf(checkpoint,'cuda')
    training = json.loads((checkpoint.parent/'plan.json').read_text())
    data = latent_data(Path(training['nativeDirectory']),Path(training['dataset']),'test',HorizonRange(21600.,1))
    def observation(step):
        return [x.numpy()[0] for x in timeline_inputs(data,np.array([settings['origin']+step]),'cpu')]
    f,prices,times = observation(0); observation(args.seconds)
    if (not np.array_equal(f,initial.cloud.features[0,0]) or not np.array_equal(prices,initial.cloud.price_pairs[0,0])
            or times[-1]!=initial.cloud.origin_seconds): raise ValueError('Initial observation mismatch')
    files = ('oracle_incremental_table.py','oracle_incremental_controller.py','oracle_incremental_observation.py',
        'oracle_root_aliases.py',
        'oracle_incremental_reference.py','oracle_incremental_checkpoint.py','oracle_dense_controller.py',
        'oracle_selected_outcomes.py','oracle_guided_sampling.py',
        'oracle_root_candidates.py','oracle_dense_metadata.py','oracle_dense_value_map.py',
        'oracle_dense_observe_batch.py','oracle_local_market_refresh.py',
        'oracle_dense_rollout.py','oracle_book_mc.py','oracle_book_gpu.py','oracle_account_grid.py',
        'latent_neural_stochastic_flow.py','export_nsf_planning_trees.py','backtest_oracle_incremental_table.py')
    inherited = {key:settings[key] for key in ('horizon','width','branches','candidates','origin','seed','backend')}
    inherited['horizon'] = initial.shape[0]
    plan = {**source,'initialPlannerArguments':settings,
        'arguments':{**inherited,**{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}},
        'initialArchive':'initial','sourceHashes':{'ml/'+name:hashlib.sha256(Path('ml',name).read_bytes()).hexdigest() for name in files},
        'estimator':'Resident dense field; exact endpoint layers plus rotating sampled interior rows prioritized by observed errors',
        'limitations':['No global Bellman error certificate for bounded updates','Finite NSF support, action menu and interpolation',
            'Raw outcome caches are control variates, not independent counts of current-law evidence']}
    (args.output/'plan.json').write_text(json.dumps(plan,indent=2)); (args.output/'source').mkdir()
    for name in files: (args.output/'source'/name).write_bytes(Path('ml',name).read_bytes())
    reference = DenseValueMap.load(args.reference_map) if args.reference_map else None
    def grade(table,update):
        if update in args.reference_updates:
            print(json.dumps(dict(phase='reference-start',update=update,horizon=table.H)),flush=True)
            def progress(row):
                print(json.dumps(dict(phase='reference-progress',update=update,**row)),flush=True)
            return full_reference(table,progress)
        if reference is not None and reference.cloud.origin_seconds==table.cloud.origin_seconds:
            for name in ('prices','counts','features','price_pairs','destinations','sampled_prices'):
                if not np.array_equal(getattr(reference.cloud,name),getattr(table.cloud,name)):
                    raise ValueError('Reference market cloud differs')
            return reference.values[min(1,table.H-1)],dict(scope='Saved finite reference field')
        return None,None
    started = time.perf_counter()
    terminal = initial.cloud.origin_seconds+args.seconds
    state = np.r_[10000.,0.,prices[-1],prices[-1],np.zeros(3*initial.grid.max_orders)]
    controller = IncrementalController(initial,model,state,
        DenseControllerConfig(terminal,settings['seed'],settings['candidates'],settings['origin'],backend='cuda'),
        budget=TableBudget(args.layers,args.rows,args.draws,args.exploration),root_draws=args.root_draws,reference=grade)
    state = controller.state
    checkpoint_context = dict(modelIdentity=source['checkpointSha256'],config=asdict(controller.config),
        budget=asdict(controller.current.budget),rootDraws=args.root_draws,episodeOrigin=controller.episode_origin,
        maximumHorizon=controller.maximum_horizon,sourceHashes=plan['sourceHashes'])
    (args.output/'initial').mkdir(); initial.cloud.save(args.output/'initial')
    np.savez(args.output/'initial'/'front.npz',**controller.front)
    initial_seconds = time.perf_counter()-started
    trades = int(state[0]!=10000.); fills = 0; failure = None
    last = 0; attempted = 0; times_used = []; comparisons = []; time_budget = False
    limit = min(args.seconds,args.max_updates or args.seconds)
    print(json.dumps(dict(phase='initial',seconds=initial_seconds,coldSweepSeconds=cold_sweep,root=controller.initial_root)),flush=True)
    with (args.output/'updates.jsonl').open('w') as log:
        for update in range(1,limit+1):
            if time.perf_counter()-started>=args.max_wall_seconds: time_budget=True; break
            attempted = update; f,prices,times = observation(update)
            try:
                row = controller.advance(f,prices,times)
            except Exception as exc:
                failure = type(exc).__name__+': '+str(exc); state = controller.state
                (args.output/'failure.json').write_text(json.dumps(dict(update=update,failure=failure,
                    arrivedState=state.tolist(),controllerPhase=controller.phase),indent=2))
                break
            last = update; state = controller.state
            trades += int(row['marketTrade']); fills += int(row['filled'])
            root = row.get('root',{})
            if 'reference' in root:
                comparison = dict(update=update,**root['reference'],table=row.get('reference'))
                comparisons.append(comparison)
                (args.output/'comparisons.json').write_text(json.dumps(comparisons,indent=2))
                print(json.dumps(dict(phase='reference-finished',**comparison)),flush=True)
            log.write(json.dumps(row)+'\n'); log.flush(); times_used.append(row['stepSeconds'])
            if args.checkpoint_every and update%args.checkpoint_every==0 and controller.phase=='active':
                checkpoint_start = time.perf_counter()
                try:
                    saved = save_table(controller.current,dict(controller.front,marketTrades=trades,fillTicks=fills),
                        checkpoint_context,args.output/'checkpoints')
                    print(json.dumps(dict(phase='checkpoint',seconds=time.perf_counter()-checkpoint_start,**saved)),flush=True)
                except Exception as exc:
                    failure = 'Checkpoint '+type(exc).__name__+': '+str(exc)
                    (args.output/'failure.json').write_text(json.dumps(dict(update=update,failure=failure),indent=2))
                    break
            (args.output/'progress.json').write_text(json.dumps(dict(update=update,elapsedSeconds=time.perf_counter()-started,
                markedEquity=float(state[0]+state[1]*state[2]),lastStepSeconds=row['stepSeconds'],failure=None)))
            if update%30==0 or update<=2:
                print(json.dumps(dict(update=update,markedEquity=row['markedEquity'],horizon=row.get('horizon'),
                    stepSeconds=row['stepSeconds'],elapsedSeconds=time.perf_counter()-started)),flush=True)
    terminal_wealth = controller.terminal_wealth
    complete = controller.phase=='complete'
    result = dict(completed=complete,plannedSeconds=args.seconds,observedUpdates=last,timeBudgetReached=time_budget,
        attemptedUpdate=attempted,failure=failure,terminalWealth=terminal_wealth,terminalReturn=terminal_wealth/10000.-1 if complete else None,
        markedEquity=float(state[0]+state[1]*state[2]),state=state.tolist(),marketTrades=trades,fillTicks=fills,
        elapsedSeconds=time.perf_counter()-started,initialSeconds=initial_seconds,comparisons=comparisons,
        coldSweepSeconds=cold_sweep,
        medianStepSeconds=float(np.median(times_used)) if times_used else None,
        maxStepSeconds=max(times_used) if times_used else None,
        interpretation='Resident incrementally sampled finite table. Terminal returns require actual flat completion. No global optimality certificate.')
    (args.output/'results.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result),flush=True)


if __name__=='__main__': main()
