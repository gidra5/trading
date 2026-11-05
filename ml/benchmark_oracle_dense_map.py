"""Reproducible learned-law dense account-map experiment, with observed updates."""
import argparse
from dataclasses import asdict
import hashlib
import json
import shutil
from pathlib import Path
import time
import numpy as np
import torch
from numba import set_num_threads
from latent_neural_stochastic_flow import load_latent_nsf
from train_latent_neural_stochastic_flow import latent_data
from target_horizon_forecast import HorizonRange
from oracle_continuation_value import timeline_inputs
from nsf_scenario_graph import build_graph
from gpu_oracle_bellman import Rules
from oracle_account_grid import AccountGrid, CellErrors
from oracle_book_mc import rule_vector, move_arrive, close_terminal
from oracle_dense_value_map import MarketCloud, DenseValueMap
from oracle_dense_rollout import roll_market_cloud
from oracle_dense_grid_config import exposure_axis, future_candidate_count


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--horizon',type=int,default=3600)
    ap.add_argument('--width',type=int,default=8)
    ap.add_argument('--branches',type=int,default=8)
    ap.add_argument('--budget',type=int,default=4)
    ap.add_argument('--candidates',type=int,help='Total future candidates; default is the exposure axis plus nine order proposals')
    ap.add_argument('--exposure-step',type=float,help='Nested exposure-axis spacing, e.g. 0.125 or 0.01')
    ap.add_argument('--updates',type=int,default=2)
    ap.add_argument('--origin',type=int,default=0)
    ap.add_argument('--price-bins',type=int,default=2)
    ap.add_argument('--size-bins',type=int,default=4)
    ap.add_argument('--max-orders',type=int,default=2)
    ap.add_argument('--threads',type=int,default=8)
    ap.add_argument('--backend',choices=['auto','cpu','cuda'],default='auto')
    ap.add_argument('--estimator',choices=['sampled','paired','repair'],default='sampled')
    ap.add_argument('--repair-tolerance',type=float,default=1.e-6)
    ap.add_argument('--repair-interval',type=int,default=8)
    ap.add_argument('--seed',type=int,default=20260940)
    ap.add_argument('--skip-reference',action='store_true')
    ap.add_argument('--market-cache',type=Path,help='Reuse frozen conditional clouds from an earlier identical forecast experiment')
    ap.add_argument('--roll-market',action='store_true',help='Refresh root and tail while retaining interior market kernels')
    ap.add_argument('--census-cache',type=Path,help='Reuse a verified complete initial map and full-sample references for the same finite law/search')
    ap.add_argument('--rebuild-initial',action='store_true',help='Compute a new initial map while using cached reference maps')
    args=ap.parse_args()
    if args.horizon<2 or args.budget<2 or args.budget>args.branches or args.updates<0:
        raise ValueError('Invalid horizon, update count or sample budget')
    if args.roll_market and args.market_cache is not None:raise ValueError('Choose generated rolling clouds or frozen cache')
    args.output.mkdir(parents=True,exist_ok=False)
    set_num_threads(args.threads);torch.set_num_threads(2);torch.set_float32_matmul_precision('high')
    checkpoint=Path('data/backtests/oracle-latent-nsf-sampled-skips/latentSampledSkipFlow.pt')
    source=json.loads((checkpoint.parent/'plan.json').read_text())
    model,_=load_latent_nsf(checkpoint,torch.device('cuda'))
    data=latent_data(Path(source['nativeDirectory']),Path(source['dataset']),'test',HorizonRange(21600.,1))
    grid=AccountGrid(max_orders=args.max_orders,price_bins=args.price_bins,size_bins=args.size_bins,
        exposure_points=exposure_axis(args.exposure_step),wealth_points=(5000.,10000.,20000.))
    args.candidates=future_candidate_count(grid,args.candidates)
    rules=Rules();r=rule_vector(rules);errors=CellErrors.empty(len(grid.arrays()['coordinates']))
    files=['ml/benchmark_oracle_dense_map.py','ml/oracle_dense_value_map.py','ml/oracle_book_mc.py','ml/oracle_book_gpu.py',
        'ml/oracle_account_grid.py','ml/oracle_dense_reuse.py','ml/nsf_scenario_graph.py','ml/export_nsf_planning_trees.py',
        'ml/oracle_dense_rollout.py','ml/oracle_dense_repair.py','ml/oracle_dense_grid_config.py']
    plan=dict(arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        grid=asdict(grid),rules=asdict(rules),fundingMultiplier=float(r[10]),gamma=1,
        checkpointSha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        sourceHashes={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in files},
        terminal='Exact valid flat close and cancellation at the endpoint',
        approximations=['finite conditional NSF sample cloud with market quantization',
            'bilinear wealth/exposure interpolation; constant order-bin interpolation',
            'finite sampled future candidates; continuous within-cell root proposals',
            'sampled conditional expectations; no uniform Bellman error certificate'],
        update='Shift prior statistics by one tick, record same-endpoint projected TD residuals, refresh all target means under the new conditional law',
        estimator=('Full cash-reference mean plus cached centered-row mean and a fresh sample of centered outcome differences; census for nonfinite controls'
            if args.estimator=='paired' else 'Fresh uniformly sampled conditional backup'),
        probabilityWeights='Equal within each conditional row; adaptive priorities allocate compute only')
    (args.output/'source').mkdir()
    for p in files:(args.output/'source'/Path(p).name).write_bytes(Path(p).read_bytes())
    (args.output/'plan.json').write_text(json.dumps(plan,indent=2))
    census_rows={}
    if args.census_cache is not None:
        census_plan=json.loads((args.census_cache/'plan.json').read_text())
        serialized_plan=json.loads((args.output/'plan.json').read_text())
        for key in ('grid','rules','fundingMultiplier','gamma','checkpointSha256'):
            if census_plan[key]!=serialized_plan[key]:raise ValueError(f'Census contract differs at {key}')
        for key in ('horizon','width','branches','origin','seed','candidates'):
            if census_plan['arguments'][key]!=plan['arguments'][key]:raise ValueError(f'Census search differs at {key}')
        for key in ('ml/oracle_book_mc.py','ml/nsf_scenario_graph.py','ml/export_nsf_planning_trees.py'):
            if census_plan['sourceHashes'][key]!=plan['sourceHashes'][key]:raise ValueError('Census kernel differs')
        census_rows={row['update']:row for row in json.loads((args.census_cache/'results.json').read_text())['rows']}
    def copy_census(name,cloud):
        source=args.census_cache/name;cached=DenseValueMap.load(source)
        for key in ('prices','counts','destinations','features','price_pairs','origin_seconds','sampled_prices'):
            if not np.array_equal(getattr(cached.cloud,key),getattr(cloud,key)):raise ValueError(f'Census cloud differs at {key}')
        if not np.all(cached.samples==args.branches):raise ValueError('Cached census does not use all outcomes')
        del cached
        shutil.copytree(source,args.output/name)
        return DenseValueMap.load(args.output/name)
    previous=None;front=None;state=None;rows=[]
    total_started=time.perf_counter()
    for update in range(args.updates+1):
        f,p,t=[x.cpu().numpy()[0] for x in timeline_inputs(data,np.array([args.origin+update]),'cpu')]
        observed=None;consumed=0.;filled=False
        if state is None:state=np.r_[10000.,0.,p[-1],p[-1],np.zeros(3*args.max_orders)]
        else:
            observed=previous.observe(front,float(p[-1]),f,p,model.feature_std.cpu().numpy())
            state,consumed,filled,ok=move_arrive(state,float(p[-1]),r)
            if not ok:raise RuntimeError('Actual replay liquidated or committed invalid aggregate fills')
        def progress(kind):
            def emit(row):
                if kind=='market' and row['depth']%300: return
                print(json.dumps(dict(update=update,phase=kind,**row)),flush=True)
            return emit
        started=time.perf_counter()
        if args.roll_market and previous is not None:
            cloud,compression=roll_market_cloud(previous.cloud,model,f,p,t,args.seed+args.origin+update)
            market_seconds=time.perf_counter()-started
        elif args.market_cache is None:
            graph=build_graph(model,f,p,t,list(range(1,args.horizon+1)),width=args.width,samples=args.branches,
                seed=args.seed+args.origin+update,on_layer=progress('market'))
            cloud=MarketCloud.from_graph(graph);market_seconds=time.perf_counter()-started
            compression=dict(maxMeanReturnShiftBps=max(x['worstParentMeanReturnShiftBps'] for x in graph.diagnostics),
                averagePriceRMSEBps=float(np.mean([x['weightedLogPriceRMSEBps'] for x in graph.diagnostics])))
            del graph
        else:
            cache_plan=json.loads((args.market_cache/'plan.json').read_text())
            for key in ('horizon','width','branches','origin','seed'):
                if cache_plan['arguments'][key]!=getattr(args,key):raise ValueError(f'Market cache differs at {key}')
            if cache_plan['checkpointSha256']!=plan['checkpointSha256']:raise ValueError('Checkpoint differs')
            for key in ('ml/nsf_scenario_graph.py','ml/export_nsf_planning_trees.py'):
                if cache_plan['sourceHashes'][key]!=plan['sourceHashes'][key]:raise ValueError('Market generator differs')
            cloud=MarketCloud.load(args.market_cache/f'update-{update}'/'market.npz')
            if cloud.origin_seconds!=t[-1] or not np.array_equal(cloud.features[0,0],f) or not np.array_equal(cloud.price_pairs[0,0],p):
                raise ValueError('Cached root observation differs')
            market_seconds=time.perf_counter()-started
            compression=dict(cachedFrom=str(args.market_cache),marketCacheSha256=hashlib.sha256(
                (args.market_cache/f'update-{update}'/'market.npz').read_bytes()).hexdigest())
        if update==0 and args.census_cache is not None and not args.rebuild_initial:
            started=time.perf_counter();current=copy_census('update-0',cloud);errors=current.errors
            if args.estimator=='paired' and current.outcome_representation!='cash-centered-v1':raise ValueError('Census lacks centered controls')
            keys=('allocationSeconds','backend','effectiveMeanConditionalSamples','cachedOutcomeFraction','outcomeStorageBytes',
                'scheduledAccountCells','finiteCells','totalCells','valueStorageBytes','meanConditionalSamples')
            run={key:census_rows[0][key] for key in keys if key in census_rows[0]}
            run.update(seconds=time.perf_counter()-started,estimator=args.estimator,cachedCensusFrom=str(args.census_cache/'update-0'),
                originalCensusSeconds=census_rows[0]['seconds'])
        else:
            current=DenseValueMap(grid,cloud,r,args.output/f'update-{update}',errors)
            run=current.sweep(args.seed,args.candidates,args.branches if update==0 else args.budget,
                previous=previous,on_progress=progress('bellman'),backend=args.backend,estimator=args.estimator,
                feature_std=model.feature_std.detach().cpu().numpy(),repair_tolerance=args.repair_tolerance,
                repair_interval=args.repair_interval)
        started=time.perf_counter();front=current.front(state,consumed,filled,interior_samples=1,seed=args.seed+update)
        front_seconds=time.perf_counter()-started
        np.savez(current.directory/'front.npz',**front)
        comparison=None
        if update>0 and not args.skip_reference:
            if args.census_cache is not None:
                started=time.perf_counter();reference=copy_census(f'reference-{update}',cloud)
                reference_run=dict(seconds=time.perf_counter()-started,cachedFrom=str(args.census_cache/f'reference-{update}'),
                    originalSeconds=census_rows[update]['reference']['seconds'])
            else:
                reference=DenseValueMap(grid,cloud,r,args.output/f'reference-{update}')
                reference_run=reference.sweep(args.seed,args.candidates,args.branches,on_progress=progress('reference'),backend=args.backend)
            truth=reference.front(state,consumed,filled,interior_samples=1,seed=args.seed+update)
            # This is a comparison to the SAME approximate grid/search with all
            # B conditional edges, not to an exact/continuous optimal policy.
            paired=np.isfinite(front['scores'])&np.isfinite(truth['scores'])
            # Evaluate the identical ACTUAL post-states under both fields.
            paired_states=np.concatenate((front['states'][paired],truth['states'][paired]))
            approx_scores=np.array([current.lookup(s) for s in paired_states])
            reference_scores=np.array([reference.lookup(s) for s in paired_states])
            finite=np.isfinite(approx_scores)&np.isfinite(reference_scores)
            delta=approx_scores[finite]-reference_scores[finite]
            selected_reference=reference.lookup(front['state'])
            comparison=dict(seconds=reference_run['seconds'],pairedRootPostStates=int(finite.sum()),
                censusCachedFrom=reference_run.get('cachedFrom'),originalCensusSeconds=reference_run.get('originalSeconds'),
                maximumPairedLogValueError=float(np.max(abs(delta))) if len(delta) else None,
                rmsPairedLogValueError=float(np.sqrt(np.mean(delta**2))) if len(delta) else None,
                referenceScoreOfSelected=float(selected_reference),referenceBest=float(truth['value']),
                referenceLogGap=float(truth['value']-selected_reference),
                samePostState=bool(np.allclose(truth['state'],front['state'],rtol=0,atol=1e-9)),
                finiteCoverageDisagreements=int(np.count_nonzero(np.isfinite(front['scores'])!=np.isfinite(truth['scores']))))
            np.savez(reference.directory/'front.npz',**truth)
            del reference
        equity=float(state[0]+state[1]*state[2]);state=front['state']
        row=dict(update=update,originSeconds=float(t[-1]),price=float(p[-1]),markedEquityBeforeDecision=equity,
            markedEquityAfterDecision=float(state[0]+state[1]*state[2]),
            exposure=float(state[1]*state[2]/(state[0]+state[1]*state[2])),
            activeOrders=int(np.count_nonzero(state[4::3])),estimatedLogValue=front['value'],
            estimatedCE=float(np.exp(front['value'])),marketSeconds=market_seconds,
            frontSeconds=front_seconds,**run,compression=compression,observed=observed,reference=comparison)
        rows.append(row);previous=current
        (args.output/'results.json').write_text(json.dumps(dict(rows=rows,seconds=time.perf_counter()-total_started),indent=2))
        print(json.dumps(row),flush=True)
    # Only diagnostics are reported for this short rolling replay. Its 1h
    # terminal date has not occurred; no marked wealth is called terminal return.
    print(json.dumps(dict(completed=True,seconds=time.perf_counter()-total_started)),flush=True)


if __name__=='__main__':main()
