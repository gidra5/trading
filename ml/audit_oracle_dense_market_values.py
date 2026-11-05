"""Compare retained-market and rebuilt-market values with all conditional draws."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from latent_neural_stochastic_flow import load_latent_nsf
from train_latent_neural_stochastic_flow import latent_data
from target_horizon_forecast import HorizonRange
from oracle_continuation_value import timeline_inputs
from nsf_scenario_graph import build_graph
from oracle_dense_rollout import roll_market_cloud
from oracle_dense_value_map import DenseValueMap,MarketCloud
from oracle_book_mc import rule_vector,move_arrive
from oracle_account_grid import AccountGrid
from gpu_oracle_bellman import Rules


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    original=json.loads((args.source/'plan.json').read_text());settings=original['arguments']
    checkpoint=Path('data/backtests/oracle-latent-nsf-sampled-skips/latentSampledSkipFlow.pt')
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=original['checkpointSha256']:raise ValueError('Checkpoint mismatch')
    training=json.loads((checkpoint.parent/'plan.json').read_text())
    torch.set_num_threads(2);torch.set_float32_matmul_precision('high')
    model,_=load_latent_nsf(checkpoint,'cuda')
    data=latent_data(Path(training['nativeDirectory']),Path(training['dataset']),'test',HorizonRange(21600.,1))
    old=MarketCloud.load(args.source/'update-0'/'market.npz')
    initial_f,initial_p,initial_t=[x.cpu().numpy()[0] for x in timeline_inputs(data,np.array([settings['origin']]),'cpu')]
    if (old.origin_seconds!=initial_t[-1] or not np.array_equal(old.features[0,0],initial_f)
        or not np.array_equal(old.price_pairs[0,0],initial_p)):raise ValueError('Origin mismatch')
    f,p,t=[x.cpu().numpy()[0] for x in timeline_inputs(data,np.array([settings['origin']+1]),'cpu')]
    rules=Rules(**original['rules']);r=rule_vector(rules,original['fundingMultiplier']);grid=AccountGrid(**original['grid'])
    with np.load(args.source/'update-0'/'front.npz') as front:previous_state=front['state']
    state,consumed,filled,valid=move_arrive(previous_state,float(p[-1]),r)
    if not valid:raise ValueError('Actual arrival is infeasible')
    files=('ml/audit_oracle_dense_market_values.py','ml/oracle_dense_rollout.py','ml/oracle_dense_value_map.py',
        'ml/oracle_book_gpu.py','ml/oracle_book_mc.py','ml/nsf_scenario_graph.py','ml/export_nsf_planning_trees.py')
    plan={**original,'arguments':{**settings,'origin':settings['origin']+1},'auditSource':str(args.source),
        'sourceHashes':{file:hashlib.sha256(Path(file).read_bytes()).hexdigest() for file in files}}
    (args.output/'plan.json').write_text(json.dumps(plan,indent=2));(args.output/'source').mkdir()
    for file in files:(args.output/'source'/Path(file).name).write_bytes(Path(file).read_bytes())
    seed=settings['seed']+settings['origin']+1
    retained,rolling=roll_market_cloud(old,model,f,p,t,seed)
    started=time.perf_counter()
    graph=build_graph(model,f,p,t,list(range(1,settings['horizon']+1)),width=settings['width'],
        samples=settings['branches'],seed=seed)
    fresh=MarketCloud.from_graph(graph);fresh_seconds=time.perf_counter()-started;del graph
    np.testing.assert_array_equal(retained.sampled_prices[0,0],fresh.sampled_prices[0,0])
    print(json.dumps(dict(phase='market',retainedSeconds=rolling['seconds'],freshSeconds=fresh_seconds)),flush=True)
    maps={};fronts={};timings={}
    for name,cloud in (('retained',retained),('fresh',fresh)):
        current=DenseValueMap(grid,cloud,r,args.output/name)
        def progress(row):
            if row['completed']%600==0:print(json.dumps(dict(phase=name,**row)),flush=True)
        timings[name]=current.sweep(settings['seed'],settings['candidates'],settings['branches'],
            backend='cuda',on_progress=progress)
        fronts[name]=current.front(state,consumed,filled,interior_samples=1,seed=settings['seed']+1)
        np.savez(current.directory/'front.npz',**fronts[name]);maps[name]=current
    states=np.concatenate([front['states'] for front in fronts.values()])
    scores={name:np.array([m.lookup(s) for s in states]) for name,m in maps.items()}
    finite=np.isfinite(scores['retained'])&np.isfinite(scores['fresh'])
    delta=scores['retained'][finite]-scores['fresh'][finite]
    chosen=maps['fresh'].lookup(fronts['retained']['state'])
    report=dict(rolling=rolling,freshMarketSeconds=fresh_seconds,sweeps=timings,
        sameRawRootPrices=True,pairedStates=int(finite.sum()),
        finiteSupportDisagreements=int(np.count_nonzero(np.isfinite(scores['retained'])!=np.isfinite(scores['fresh']))),
        maximumAbsoluteLogErrorBps=float(np.max(abs(delta))*10000) if len(delta) else None,
        actionRelativeErrorRangeBps=float(np.ptp(delta)*10000) if len(delta) else None,
        retainedChoiceFreshScore=float(chosen),bestFreshProposalScore=float(np.max(scores['fresh'])),
        proposalRegretBps=float((np.max(scores['fresh'])-chosen)*10000),
        rootValues={name:float(front['value']) for name,front in fronts.items()},
        states={name:front['state'].tolist() for name,front in fronts.items()},
        interpretation='One causal observed update, full 3600-second depth and all conditional draws; comparison of two finite market grids, not a bound against the underlying continuous law')
    (args.output/'results.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)


if __name__=='__main__':main()
