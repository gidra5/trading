"""Four-sample root refinements against an eight-edge learned-law reference.

The archived continuation stays fixed. This measures conditional sampling and
action selection on attained front proposals, not global planning error.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from oracle_dense_value_map import DenseValueMap
from oracle_guided_sampling import sample_plan, corrected_estimate
from oracle_local_market_refresh import candidate_schedule
from oracle_selected_outcomes import SelectedOutcomes


def load_front(path):
    with np.load(path/'front.npz') as data:
        return {key:data[key] for key in data.files}


def evaluator(m,states,seed,candidates):
    B = m.cloud.destinations.shape[-1]
    count = len(states)
    return SelectedOutcomes(m,states,np.repeat(m.cloud.sampled_prices[0,:1],count,axis=0),
        np.repeat(m.cloud.destinations[0,:1],count,axis=0),m.values[1],m.cloud.factors(1),
        candidate_schedule(m,0,seed,candidates),m.shape[0])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--old-map',type=Path,required=True)
    p.add_argument('--new-map',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seeds',type=int,default=100)
    args = p.parse_args()
    if args.seeds < 1: raise ValueError('Positive seed count required')
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2)
    started = time.perf_counter()
    old,new = DenseValueMap.load(args.old_map),DenseValueMap.load(args.new_map)
    if old.grid != new.grid or not np.array_equal(old.rules,new.rules):
        raise ValueError('Account contracts differ')
    settings = json.loads((args.new_map.parent/'plan.json').read_text())['arguments']
    before,after = load_front(args.old_map),load_front(args.new_map)
    cells = np.flatnonzero(np.isfinite(after['scores']))
    prior = np.r_[before['states'][cells],before['state'][None]]
    states = np.r_[after['states'][cells],after['state'][None]]
    available = np.r_[np.isfinite(before['scores'][cells]),True]
    # A separate cash reference is not added to the feasible action menu.
    def cash(m): return np.r_[10000.,0.,m.cloud.prices[0,0],m.cloud.prices[0,0],np.zeros(3*m.grid.max_orders)]
    prior[~available] = cash(old)
    prior = np.r_[prior,cash(old)[None]]; states = np.r_[states,cash(new)[None]]
    available = np.r_[available,True]
    B = old.cloud.destinations.shape[-1]; R = len(states)
    indices = np.tile(np.arange(B),(R,1)); census = np.full(R,B,dtype=np.int64)
    a = evaluator(old,prior,settings['seed'],settings['candidates'])
    old_y = a.evaluate(indices,census); del a
    current = evaluator(new,states,settings['seed'],settings['candidates'])
    new_y = current.evaluate(indices,census)
    if not np.isfinite(old_y[-1]).all() or not np.isfinite(new_y[-1]).all():
        raise ValueError('Cash reference is nonfinite')
    controls = old_y-old_y[-1]
    controls[~available] = -np.inf
    support = current.supported.copy(); support[-1] = False  # Cash centering always uses a census.
    truth = new_y.mean(1); optimum = float(truth[:-1].max()); correct = int(truth[:-1].argmax())
    raw_timings = []
    for _ in range(10):
        t = time.perf_counter(); current.evaluate(indices,census); raw_timings.append(time.perf_counter()-t)
    records = []
    for mode,exploration in (('uniform',1.),('two_tail',.2)):
        for seed in range(args.seeds):
            cache = controls.copy()
            for generation in range(1,9):
                t = time.perf_counter()
                plan = sample_plan(cache,support,4,2026091400+seed*100+generation,exploration=exploration)
                values = current.evaluate(plan.indices,plan.counts)
                reference = values[-1].mean()
                centered = values-np.take_along_axis(np.broadcast_to(values[-1],values.shape),plan.indices,axis=1)
                estimate,variance,cache = corrected_estimate(cache,centered,plan)
                estimate += reference
                elapsed = time.perf_counter()-t
                finite = np.isfinite(truth[:-1])
                if not np.array_equal(np.isfinite(estimate[:-1]),finite):
                    raise AssertionError('Sampling changed feasible support')
                chosen = int(estimate[:-1].argmax())
                error = estimate[:-1][finite]-truth[:-1][finite]
                records.append(dict(method=mode,seed=seed,generation=generation,
                    seconds=elapsed,meanDraws=float(plan.counts.mean()),
                    actionAgreement=bool(np.array_equal(states[chosen],states[correct])),
                    regretBps=float(-np.expm1(truth[chosen]-optimum)*10000),
                    maximumValueErrorBps=float(np.max(abs(error))*10000),
                    meanValueSquaredError=float(np.mean(error**2)),
                    meanEstimatedVariance=float(np.mean(variance[:-1][finite])),
                    chosenAction=chosen))
    summaries = []
    for mode in ('uniform','two_tail'):
        for generation in (1,2,4,8):
            rows = [x for x in records if x['method']==mode and x['generation']==generation]
            summaries.append(dict(method=mode,generation=generation,
                medianSeconds=float(np.median([x['seconds'] for x in rows])),
                actionAgreement=float(np.mean([x['actionAgreement'] for x in rows])),
                meanRegretBps=float(np.mean([x['regretBps'] for x in rows])),
                maximumRegretBps=max(x['regretBps'] for x in rows),
                rmsValueErrorBps=float(np.sqrt(np.mean([x['meanValueSquaredError'] for x in rows]))*10000),
                maximumValueErrorBps=max(x['maximumValueErrorBps'] for x in rows)))
    result = dict(oldMap=str(args.old_map),newMap=str(args.new_map),seeds=args.seeds,
        candidateRows=R-1,conditionalEdges=B,partialRows=int((support & np.isfinite(controls).all(1)).sum()),
        referenceValue=optimum,referenceState=states[correct].tolist(),
        fullCensusMedianSeconds=float(np.median(raw_timings)),elapsedSeconds=time.perf_counter()-started,
        summaries=summaries,sourceHashes={name:hashlib.sha256(Path('ml',name).read_bytes()).hexdigest()
            for name in ('oracle_guided_sampling.py','oracle_selected_outcomes.py','benchmark_oracle_guided_root.py')},
        scope='Root proposals from saved actual decisions. Existing future menus and continuation field; eight-edge conditional law. Generations repeatedly refine the SAME target, not eight later observed seconds. No global optimality or full-NSF tail certificate.')
    (args.output/'results.json').write_text(json.dumps(result,indent=2))
    (args.output/'trials.json').write_text(json.dumps(records))
    print(json.dumps(result,indent=2))


if __name__ == '__main__': main()
