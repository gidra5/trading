"""Bounded one-second replay; every plan retains the archive's fixed terminal.

This measures feasibility and runtime of refinements, not a full backtest unless
--decisions covers the entire archive. Use the exact replay auditor afterwards.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from benchmark_gpu_oracle import synthetic_tree
from gpu_oracle_bellman import Bellman, Menu, equity
from latent_neural_stochastic_flow import load_latent_nsf
from train_latent_neural_stochastic_flow import latent_data
from target_horizon_forecast import HorizonRange


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--decisions', type=int, default=120)
    ap.add_argument('--branches', type=int, default=8)
    ap.add_argument('--chunk', type=int, default=131072)
    ap.add_argument('--future-step', type=float, default=1.)
    ap.add_argument('--near-seconds', type=int, default=2)
    ap.add_argument('--output', type=Path, default=Path('data/backtests/oracle-gpu-scaling/second-replay'))
    args = ap.parse_args()
    if args.decisions < 1 or args.branches < 2 or args.branches % 2 or args.near_seconds < 1:
        raise ValueError('Positive decisions and even branching required')
    root = args.output
    if (root/'plan.json').exists():
        raise ValueError('Replay already exists')
    root.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4); torch.set_float32_matmul_precision('high')
    device = torch.device('cuda')
    checkpoint = Path('data/backtests/oracle-latent-nsf-sampled-skips/latentSampledSkipFlow.pt')
    source_plan = json.loads((checkpoint.parent/'plan.json').read_text(encoding='utf8'))
    selection = json.loads((checkpoint.parent/'selection.json').read_text(encoding='utf8'))
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert digest == selection['models']['latentSampledSkipFlow']['checkpointSha256']
    data = latent_data(Path(source_plan['nativeDirectory']), Path(source_plan['dataset']), 'test', HorizonRange(21600., 1))
    model, _ = load_latent_nsf(checkpoint, device)
    first = int(data['origins'][0]); count = len(data['prices'])-1
    start = float(data['timestamps'][0])/1000
    limit = min(count, args.decisions)
    sources = ['ml/gpu_oracle_bellman.py', 'ml/benchmark_gpu_oracle.py', 'ml/backtest_gpu_nsf.py',
               'ml/export_nsf_planning_trees.py', 'ml/latent_neural_stochastic_flow.py']
    plan = dict(checkpointSha256=digest, decisionStrideSeconds=1, requestedDecisions=args.decisions,
                archiveSeconds=count, terminalTimestamp=int((start+count)*1000), gamma=0,
                rootBranches=args.branches, exposureStep=.01, futureExposureStep=args.future_step,
                nearSeconds=args.near_seconds,
                chunk=args.chunk,
                limitSlots=1, limitFractions=[.25], limitOffsetsTicks=[1], jointMarketPlacement=False,
                fixedTerminal=True, terminalRequirement="zero-asset-and-no-orders", fullFutureSecondGrid=False, exactNumericalCertificate=False,
                sourceHashes={s: hashlib.sha256(Path(s).read_bytes()).hexdigest() for s in sources})
    (root/'source').mkdir(exist_ok=True)
    for s in sources:
        (root/'source'/Path(s).name).write_bytes(Path(s).read_bytes())
    (root/'plan.json').write_text(json.dumps(plan, indent=2), encoding='utf8')
    (root/'marks.json').write_text(json.dumps(dict(startTimestamp=int(start*1000), prices=data['prices'][:limit+1].tolist())), encoding='utf8')
    state = torch.tensor([[10000., 0., data['prices'][0], data['prices'][0], 0., 0., 0., 0., 0.]], device=device, dtype=torch.float64)
    records = []
    started = time.perf_counter()
    for idx in range(limit):
        remaining = count-idx
        near = min(args.near_seconds, remaining)
        offsets = sorted(set(list(range(1, near+1))+[near+(remaining-near)//2, remaining]))
        offsets = [x for x in offsets if 0 < x <= remaining]
        branches = [args.branches]+[2]*(len(offsets)-1)
        features = np.asarray(data['timeline'][first+idx-1:first+idx+1], dtype=np.float32)
        pp = data['pricesWithPast'][idx:idx+2]
        times = np.array([start+idx-1, start+idx])
        tick = time.perf_counter()
        _, (prices, weights) = synthetic_tree(model, features, pp, times, offsets, branches, 20260915+idx*4, device)
        solver = Bellman(prices, weights, [Menu(step=.01)]+[Menu(step=args.future_step)]*(len(offsets)-1), chunk=args.chunk)
        result = solver.solve(state)
        torch.cuda.synchronize()
        descriptor = [None if np.isnan(x) else x for x in result['descriptor'][0].tolist()]
        state = solver.oracle.advance(result['state'], float(data['prices'][idx+1]))
        record = dict(index=idx, descriptor=descriptor, marketQuote=result['amount'].item(),
                      expectedTerminalWealth=result['value'].item(), after=result['state'][0].tolist(),
                      nextMarkedState=state[0].tolist(), nextMarkedWealth=equity(state).item(),
                      seconds=time.perf_counter()-tick, terminalEvaluations=result['work'])
        records.append(record)
        with (root/'decisions.jsonl').open('a', encoding='utf8') as f:
            f.write(json.dumps(record, allow_nan=False)+'\n')
        if idx % 10 == 0 or idx+1 == limit:
            print(json.dumps(dict(decisions=idx+1, total=limit, wealth=equity(state).item(), elapsedSeconds=time.perf_counter()-started)), flush=True)
    complete = limit == count
    marked = equity(state).item()
    reached_horizon = complete
    terminal = solver.oracle.terminal(state).item() if complete else None
    terminal_failure = None
    if terminal is not None and not np.isfinite(terminal):
        complete, terminal = False, None
        terminal_failure = 'No legal complete close at the terminal arrival'

    report = dict(complete=complete, reachedHorizon=reached_horizon, terminalFailure=terminal_failure,
                  terminalRequirement="zero-asset-and-no-orders", decisions=limit, elapsedSeconds=time.perf_counter()-started,
                  markedWealthBeforeNextArrival=marked, markedReturnPercent=100*(marked/10000-1), terminalWealth=terminal,
                  nonzeroMarketDecisions=sum(x['marketQuote'] != 0 for x in records),
                  placements=sum(x['descriptor'][2] != 0 for x in records),
                  meanDecisionSeconds=float(np.mean([x['seconds'] for x in records])),
                  p95DecisionSeconds=float(np.quantile([x['seconds'] for x in records], .95)),
                  terminalEvaluations=sum(x['terminalEvaluations'] for x in records))
    (root/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf8')
    print(json.dumps(report), flush=True)
    if terminal_failure:
        raise RuntimeError(terminal_failure)


if __name__ == '__main__':
    main()
