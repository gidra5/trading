"""Grade every recorded countdown decision using one full finite tail solve.

Fresh root laws use only each decision's observed history. The unchanged
interior kernels and terminal date permit reuse of the exact reference tail.
This grades the logged actions; it does not replace them or rerun the policy.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import time
import numpy as np
import torch
from numba import set_num_threads
from audit_oracle_incremental_checkpoint import digest,require
from audit_oracle_dense_ledger import audit as audit_ledger
from latent_neural_stochastic_flow import load_latent_nsf
from train_latent_neural_stochastic_flow import latent_data
from target_horizon_forecast import HorizonRange
from oracle_continuation_value import timeline_inputs
from oracle_account_grid import AccountGrid
from oracle_dense_value_map import DenseValueMap,MarketCloud
from oracle_dense_rollout import roll_market_cloud
from oracle_book_mc import move_arrive
from oracle_root_candidates import root_candidates
from oracle_selected_outcomes import SelectedOutcomes
from oracle_local_market_refresh import candidate_schedule
from oracle_dense_repair import contract
from oracle_dense_checkpoint import initial_archive


def unchanged_tail(anchor,cloud,offset):
    """Check the exact fields used by subsequent backups, excluding fresh root."""
    require(cloud.origin_seconds==anchor.origin_seconds+offset,'Tail timestamp changed')
    for name in ('prices','counts','features','price_pairs','destinations','sampled_prices'):
        require(np.array_equal(getattr(cloud,name)[1:],getattr(anchor,name)[offset+1:]),
            'Retained tail changed: '+name)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('directory',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--reference',type=Path,help='Reuse a completed matching full-census archive')
    args=parser.parse_args()
    root=args.directory;plan=json.loads((root/'plan.json').read_text())
    result=json.loads((root/'results.json').read_text())
    require(result['completed'],'A complete backtest is required')
    anchor_dir=root/'countdown-anchor';source=json.loads((anchor_dir/'source.json').read_text())
    for name,checksum in source['hashes'].items():require(digest(anchor_dir/name)==checksum,'Anchor checksum differs')
    meta=json.loads((anchor_dir/'metadata.json').read_text())
    # Execute the same frozen kernels and forecast implementation as the replay.
    for name,checksum in plan['sourceHashes'].items():require(digest(Path(name))==checksum,'Replay implementation changed: '+name)
    ledger=audit_ledger(root)
    start=meta['update'];end=plan['arguments']['seconds'];settings=plan['arguments']
    require(1<=start<end,'Anchor must follow an accepted nonterminal decision')
    require(start>=end-settings['horizon'] and meta['horizon']==end-start,'Anchor is not in the fixed-terminal phase')
    anchor=MarketCloud.load(anchor_dir/'market.npz');cloud=anchor
    grid_data=dict(meta['grid'])
    for name in ('wealth_points','exposure_points'):grid_data[name]=tuple(grid_data[name])
    grid=AccountGrid(**grid_data);rules=np.asarray(meta['rules']);B=anchor.destinations.shape[-1]
    rows=[json.loads(line) for line in (root/'updates.jsonl').read_text().splitlines() if line.strip()]
    require(len(rows)==end,'Incomplete execution log')
    comparisons={item['update']:item for item in result.get('comparisons',[])}
    checked_references=0
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);set_num_threads(2);torch.set_float32_matmul_precision('high')
    checkpoint=Path('data/backtests/oracle-latent-nsf-sampled-skips/latentSampledSkipFlow.pt')
    require(digest(checkpoint)==plan['checkpointSha256'],'Forecast model changed')
    model,_=load_latent_nsf(checkpoint,'cuda')
    training=json.loads((checkpoint.parent/'plan.json').read_text())
    data=latent_data(Path(training['nativeDirectory']),Path(training['dataset']),'test',HorizonRange(21600.,1))
    reference=DenseValueMap.load(args.reference) if args.reference else DenseValueMap(grid,anchor,rules,args.output/'reference')
    if args.reference:
        require(reference.grid==grid and np.array_equal(reference.rules,rules),'Reference account model differs')
        require(reference.bellman_contract==contract(settings['seed'],settings['candidates'])
            and np.all(reference.samples==B) and reference.bellman_residuals is not None
            and np.all(reference.bellman_residuals==0),'Reference is not a matching full census')
        require(reference.cloud.origin_seconds==anchor.origin_seconds,'Reference timestamp differs')
        for name in ('prices','counts','features','price_pairs','destinations','sampled_prices'):
            require(np.array_equal(getattr(reference.cloud,name),getattr(anchor,name)),'Reference cloud differs: '+name)
    (args.output/'plan.json').write_text(json.dumps(dict(plan,anchorUpdate=start,anchorSource=source,
        referenceArchive=str(args.reference) if args.reference else 'reference'),indent=2))
    (args.output/'audit-source.py').write_bytes(Path(__file__).read_bytes())
    began=time.perf_counter()
    sweep=dict(seconds=0.) if args.reference else reference.sweep(settings['seed'],settings['candidates'],B,backend='cuda',
        on_progress=lambda row:print(json.dumps(dict(phase='reference',**row)),flush=True))
    records=[]
    if start==1:
        with np.load(initial_archive(root,plan)/'front.npz') as initial:previous=initial['state'].copy()
    else:previous=np.asarray(rows[start-2]['state'])
    with (args.output/'decisions.jsonl').open('w') as output:
        for update in range(start,end):
            row=rows[update-1];offset=update-start
            f,p,t=[x.numpy()[0] for x in timeline_inputs(data,np.array([settings['origin']+update]),'cpu')]
            if offset:
                cloud,_=roll_market_cloud(cloud,model,f,p,t,settings['seed']+settings['origin']+update,append_tail=False)
            else:
                require(np.array_equal(f,cloud.features[0,0]) and np.array_equal(p,cloud.price_pairs[0,0]),'Anchor observation differs')
            unchanged_tail(anchor,cloud,offset)
            H=end-update
            require(row['horizon']==H and t[-1]==row['originSeconds'] and p[-1]==row['price'],'Logged decision time differs')
            arrived,consumed,filled,valid=move_arrive(previous,float(p[-1]),rules)
            require(valid,'Logged arrival is invalid')
            facade=SimpleNamespace(grid=grid,a=reference.a,rules=rules,cloud=cloud,shape=(H,*reference.shape[1:]))
            _,states,_=root_candidates(facade,arrived,consumed,filled,seed=settings['seed']+update)
            actual=np.asarray(row['state']);matches=np.flatnonzero(np.all(states==actual,axis=1))
            require(len(matches)>0,'Logged action is absent from reconstructed proposals')
            factors=np.zeros((reference.shape[-1],B))
            if H>1:
                value=cloud.factors(1);factors[:len(value)]=value
            field=reference.values[min(offset+1,reference.shape[0]-1)]
            evaluator=SelectedOutcomes(facade,states,np.repeat(cloud.sampled_prices[0,:1],len(states),axis=0),
                np.repeat(cloud.destinations[0,:1],len(states),axis=0),field,factors,
                candidate_schedule(facade,0,settings['seed'],settings['candidates']),H)
            values=evaluator.evaluate(np.tile(np.arange(B),(len(states),1)),np.full(len(states),B,dtype=np.int64)).mean(1)
            selected=float(values[matches[0]]);best=int(values.argmax())
            selected_finite=bool(np.isfinite(selected));best_finite=bool(np.isfinite(values[best]))
            if update in comparisons:
                point=comparisons[update]
                agrees=lambda x,y:x==y or abs(x-y)<1e-11
                require(agrees(selected,point['selectedValue']) and agrees(float(values[best]),point['bestValue']),
                    'Reused tail disagrees with independently recomputed point reference')
                checked_references+=1
            record=dict(update=update,horizon=H,proposals=len(states),selectedFinite=selected_finite,
                hasFiniteReferenceAction=best_finite,selectedReferenceValue=selected if selected_finite else None,
                bestReferenceValue=float(values[best]) if best_finite else None,
                regretBps=float(-np.expm1(selected-values[best])*10000) if best_finite else None,
                sameAction=bool(best_finite and np.array_equal(actual,states[best])),
                selectedValueErrorBps=float((row['estimatedLogValue']-selected)*10000) if selected_finite else None)
            records.append(record);output.write(json.dumps(record)+'\n');output.flush();previous=actual
            if update%100==0:print(json.dumps(dict(phase='grade',**record)),flush=True)
    regrets=np.array([r['regretBps'] for r in records if r['regretBps'] is not None])
    errors=np.array([r['selectedValueErrorBps'] for r in records if r['selectedValueErrorBps'] is not None])
    summary=dict(firstUpdate=start,lastUpdate=end-1,checkedDecisions=len(records),
        sameActionFraction=float(np.mean([r['sameAction'] for r in records])),
        nonfiniteSelectedDecisions=sum(not r['selectedFinite'] for r in records),
        noFiniteReferenceDecisions=sum(not r['hasFiniteReferenceAction'] for r in records),
        maximumRegretBps=float(regrets.max()) if len(regrets) else None,
        meanRegretBps=float(regrets.mean()) if len(regrets) else None,
        p95RegretBps=float(np.quantile(regrets,.95)) if len(regrets) else None,
        maximumAbsoluteSelectedValueErrorBps=float(np.max(abs(errors))) if len(errors) else None,referenceSeconds=sweep['seconds'],
        reusedReference=args.reference is not None,
        crossCheckedIndependentReferences=checked_references,
        elapsedSeconds=time.perf_counter()-began,executionLedgerChecked=ledger['terminalChecked'],
        scope='All reconstructed root proposals against a full finite-grid Bellman tail with identical retained kernels. No bound against continuous actions, uncompressed histories or the unbounded NSF law.')
    (args.output/'results.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
