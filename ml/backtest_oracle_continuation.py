"""One-second replay with a rolling one-hour learned continuation value.

Forecast origins can be batched because the models are frozen and each row
uses its own completed observations. No forecast uses another row's future.
"""
import argparse
import hashlib
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
from gpu_oracle_bellman import Menu, equity
from latent_neural_stochastic_flow import load_latent_nsf
from train_latent_neural_stochastic_flow import latent_data
from target_horizon_forecast import HorizonRange
from oracle_continuation_value import load_value, timeline_inputs, conditional_children, bellman_backup


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--value', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--decisions', type=int)
    ap.add_argument('--samples', type=int, default=64)
    ap.add_argument('--forecast-batch', type=int, default=64)
    args = ap.parse_args()
    if args.output.exists(): raise ValueError('Experiment already exists')
    args.output.mkdir(parents=True)
    torch.set_num_threads(4); torch.set_float32_matmul_precision('high')
    device = torch.device('cuda')
    value, metadata = load_value(args.value, device)
    checkpoint = Path(metadata['checkpoint'])
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != metadata['checkpointSha256']:
        raise ValueError('NSF checkpoint changed since value training')
    nsf, _ = load_latent_nsf(checkpoint, device)
    source = json.loads((checkpoint.parent/'plan.json').read_text())
    data = latent_data(Path(source['nativeDirectory']), Path(source['dataset']), 'test', HorizonRange(21600., 1))
    total = len(data['prices'])-1
    count = min(total, args.decisions) if args.decisions else total
    start_time = int(data['timestamps'][0])
    sources = ['ml/backtest_oracle_continuation.py', 'ml/oracle_continuation_value.py', 'ml/gpu_oracle_bellman.py',
        'ml/export_nsf_planning_trees.py', 'ml/latent_neural_stochastic_flow.py']
    plan = dict(gamma=1, archiveSeconds=count, sourceArchiveSeconds=total,
        terminalTimestamp=start_time+1000*count, rollingHorizonSeconds=3600, decisionStrideSeconds=1,
        horizonAtDecision='min(3600, declared terminal tick minus current tick)', conditionalSamples=args.samples,
        forecastBatch=args.forecast_batch, valueCheckpoint=str(args.value), valueSha256=hashlib.sha256(args.value.read_bytes()).hexdigest(),
        trainingMetadata=metadata, sourceHashes={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources},
        terminalRequirement='zero-asset-and-no-orders', continuation='learned state/horizon value; one-second Bellman backup',
        latentSamplesVisibleToPolicy=False, futureOriginRowsVisibleToEarlierDecisions=False,
        uniformValueErrorCertificate=None, horizonTruncationCertificate='computed separately, not inferred from this replay')
    (args.output/'plan.json').write_text(json.dumps(plan, indent=2))
    (args.output/'marks.json').write_text(json.dumps(dict(startTimestamp=start_time, prices=data['prices'][:count+1].tolist())))
    state = torch.tensor([[10000, 0, data['prices'][0], data['prices'][0], 0, 0, 0, 0, 0]], dtype=torch.float64, device=device)
    menu = Menu(step=.01)
    started = time.perf_counter(); latencies = []; graph_seconds = 0.; failures = []
    markets = placements = cancellations = 0
    cached_cash_decisions = 0
    current = state[0].tolist()
    peak = 10000.; drawdown = 0.
    with (args.output/'decisions.jsonl').open('w') as trace:
        for block in range(0, count, args.forecast_batch):
            end = min(count, block+args.forecast_batch)
            f, p, t = timeline_inputs(data, np.arange(block, end), device)
            stamp = time.perf_counter()
            children = conditional_children(nsf, f, p, t, args.samples, 20260925+block)
            torch.cuda.synchronize(); graph_seconds += time.perf_counter()-stamp
            cash_cache = None
            # Evaluate counterfactual cash accounts in parallel. A cached row is
            # used ONLY if the actual causal account equals that cash account.
            # This is reuse of identical decisions, not skipping optimization.
            if current[1] == 0 and current[4] == 0 and current[0] > 0:
                cash_quote = current[0]
                candidates = torch.zeros((end-block, 9), device=device, dtype=torch.float64)
                candidates[:, 0] = cash_quote; candidates[:, 2:4] = p.flip(-1)
                candidates[:, 8] = torch.arange(block, end, device=device)
                horizons = torch.arange(count-block, count-end, -1, device=device).clamp_max(value.horizon)
                stamp = time.perf_counter()
                cached = bellman_backup(value, candidates, *children, horizons, menu)
                cash_cache = {key:cached[key].cpu().numpy() for key in ['value', 'state', 'amount', 'descriptor', 'valid']}
                cache_cost = (time.perf_counter()-stamp)/(end-block)
            for local, index in enumerate(range(block, end)):
                stamp = time.perf_counter()
                horizon = min(value.horizon, count-index)
                reuse = cash_cache is not None and current[0] == cash_quote and current[1] == 0 and current[4] == 0
                if reuse:
                    valid = bool(cash_cache['valid'][local]); after = cash_cache['state'][local].tolist()
                    raw_descriptor = cash_cache['descriptor'][local].tolist()
                    amount = float(cash_cache['amount'][local]); log_value = float(cash_cache['value'][local])
                    cached_cash_decisions += 1
                else:
                    state = torch.tensor([current], device=device, dtype=torch.float64)
                    h = torch.tensor([horizon], device=device)
                    result = bellman_backup(value, state, *(x[local:local+1] for x in children), h, menu)
                    valid = result['valid'].item(); after = result['state'][0].tolist()
                    raw_descriptor = result['descriptor'][0].tolist()
                    amount = result['amount'].item(); log_value = result['value'].item()
                if not valid:
                    failures.append(dict(index=index, reason='No finite sampled log continuation'))
                    break
                descriptor = [None if math.isnan(x) else x for x in raw_descriptor]
                if after[1] == 0 and after[4] == 0 and after[0] > 0:
                    current = list(after); current[3] = after[2]; current[2] = float(data['prices'][index+1]); current[8] += 1
                else:
                    state = value.oracle.advance(torch.tensor([after], device=device, dtype=torch.float64), float(data['prices'][index+1]))
                    current = state[0].tolist()
                w = current[0]+current[1]*current[2]; peak = max(peak, w); drawdown = max(drawdown, 1-w/peak)
                elapsed = time.perf_counter()-stamp+(cache_cost if reuse else 0); latencies.append(elapsed)
                markets += amount != 0; placements += descriptor[2] != 0; cancellations += descriptor[1] != 0
                row = dict(index=index, horizonSeconds=horizon, descriptor=descriptor, marketQuote=amount,
                    approximateLogValue=log_value, after=after, reusedCashDecision=reuse,
                    nextMarkedState=current, nextMarkedWealth=w, solveSeconds=elapsed)
                trace.write(json.dumps(row, allow_nan=False)+'\n')
                if index % 1000 == 0 or index+1 == count:
                    trace.flush()
                    print(json.dumps(dict(decisions=index+1, total=count, wealth=w, elapsedSeconds=time.perf_counter()-started,
                        markets=markets, placements=placements)), flush=True)
            if failures: break
    complete = len(latencies) == count
    state = torch.tensor([current], device=device, dtype=torch.float64)
    terminal = value.oracle.terminal(state).item() if complete else None
    if terminal is not None and not math.isfinite(terminal):
        failures.append(dict(index=count, reason='No valid complete terminal close')); complete = False; terminal = None
    report = dict(complete=complete, decisions=len(latencies), gamma=1, terminalWealth=terminal,
        terminalReturnPercent=None if terminal is None else (terminal/10000-1)*100,
        terminalLogReturn=None if terminal is None or terminal <= 0 else math.log(terminal/10000),
        maximumDrawdownPercent=drawdown*100, elapsedSeconds=time.perf_counter()-started,
        forecastingSeconds=graph_seconds, meanSolveSeconds=float(np.mean(latencies)) if latencies else None,
        p95SolveSeconds=float(np.quantile(latencies,.95)) if latencies else None,
        nonzeroMarketDecisions=markets, placements=placements, cancellations=cancellations, failures=failures,
        cachedCashDecisions=cached_cash_decisions,
        sampledApproximation=True, uniformOptimalityCertificate=None)
    (args.output/'report.json').write_text(json.dumps(report, indent=2)); print(json.dumps(report), flush=True)


if __name__ == '__main__': main()
