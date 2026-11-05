"""Differential verification and bounded NSF Bellman scaling measurements."""
from __future__ import annotations
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from gpu_oracle_bellman import Bellman, Menu


def tensor_tree(tree, device):
    levels, weights = [], []
    pending = [tree]
    while pending[0]['children']:
        sizes = {len(x['children']) for x in pending}
        if len(sizes) != 1:
            raise ValueError('Regular tree required')
        for node in pending:
            observations = [(x['price'], x.get('observationKey', '')) for x in node['children']]
            if len(set(observations)) != len(observations):
                raise ValueError('Merge indistinguishable observations before GPU optimization')
        levels.append(torch.tensor([[x['price'] for x in n['children']] for n in pending], dtype=torch.float64, device=device))
        from fractions import Fraction
        weights.append(torch.tensor([[float(Fraction(x['probability'])) for x in n['children']] for n in pending], dtype=torch.float64, device=device))
        pending = [x for n in pending for x in n['children']]
    return levels, weights


def audit(root, device):
    cases = json.loads((root/'reference.json').read_text(encoding='utf8'))
    results = []
    for c in cases:
        prices, weights = tensor_tree(c['tree'], device)
        solver = Bellman(prices, weights, [Menu(step=c['step'])]*len(prices), funding=[c['funding']]*len(prices))
        s = torch.tensor([c['state']], device=device, dtype=torch.float64)
        result = solver.solve(s)
        value = result['value'].item()
        if abs(value-c['expected']) > 1e-7:
            raise AssertionError((c['id'], value, c['expected']))
        descriptor = result['descriptor'][0].tolist()
        descriptor = [None if np.isnan(x) else x for x in descriptor]
        results.append(dict(id=c['id'], value=value, error=value-c['expected'], descriptor=descriptor, work=result['work']))
    report = dict(terminalRequirement="zero-asset-and-no-orders", device=str(device), cases=results, maximumValueError=max(abs(x['error']) for x in results))
    (root/'gpu-audit.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf8')
    print(json.dumps(dict(auditCases=len(results), maximumValueError=report['maximumValueError'])), flush=True)


def synthetic_tree(model, features, price, times, offsets, branching, seed, device):
    from export_nsf_planning_trees import next_observations
    if not offsets or len(offsets) != len(branching) or any(b < 2 or b % 2 for b in branching):
        raise ValueError('Positive even branching required at each depth')
    nodes = 1
    total = 0
    for b in branching:
        nodes *= b
        total += nodes
        if total > 100000:
            raise ValueError('Forecast tree exceeds 100,000-node generation budget; refusing exponential allocation')
    if any(h <= 0 or h > 86400 for h in np.diff([0]+offsets)):
        raise ValueError('Forecast gaps must lie in the NSF query range')
    f, p, t = features[None], price[None], times[None]
    tree = {'children': []}
    pending = [tree]
    for depth, (offset, branches) in enumerate(zip(offsets, branching)):
        h = np.full(len(f), times[-1]+offset)-t[:, -1]
        nf, pp = next_observations(model, f, p, t, h, branches, seed+depth)
        following = []
        for n, node in enumerate(pending):
            children = [dict(price=float(pp[n, b]), probability=f'1/{branches}',
                             observationKey=json.dumps(nf[n, b].tolist(), separators=(',', ':')), children=[])
                        for b in range(branches)]
            node['children'] = children
            following.extend(children)
        t = np.stack((np.repeat(t[:, -1], branches), np.full(len(f)*branches, times[-1]+offset)), axis=1)
        p = np.stack((np.repeat(p[:, -1], branches), pp.reshape(-1)), axis=1)
        f = np.stack((np.repeat(f[:, -1], branches, axis=0), nf.reshape(-1, 59)), axis=1)
        pending = following
    return tree, tensor_tree(tree, device)


def benchmark(root, device, chunk=131072):
    from latent_neural_stochastic_flow import load_latent_nsf
    from train_latent_neural_stochastic_flow import latent_data
    from target_horizon_forecast import HorizonRange
    checkpoint = Path('data/backtests/oracle-latent-nsf-sampled-skips/latentSampledSkipFlow.pt')
    plan = json.loads((checkpoint.parent/'plan.json').read_text(encoding='utf8'))
    selection = json.loads((checkpoint.parent/'selection.json').read_text(encoding='utf8'))
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert digest == selection['models']['latentSampledSkipFlow']['checkpointSha256']
    data = latent_data(Path(plan['nativeDirectory']), Path(plan['dataset']), 'test', HorizonRange(21600., 1))
    model, _ = load_latent_nsf(checkpoint, device)
    first = int(data['origins'][0])
    features = np.asarray(data['timeline'][first-1:first+1], dtype=np.float32)
    price = data['pricesWithPast'][:2]
    start = float(data['timestamps'][0])/1000
    times = np.array([start-1, start])
    remaining = len(data['prices'])-1
    specifications = [
        ('64-branches-one-second', [1], [64], [.01]),
        ('72-nodes-two-seconds', [1, 2], [8, 8], [.01, .01]),
        ('64-branches-two-seconds', [1, 2], [64, 2], [.01, .01]),
        ('64-per-node-two-seconds', [1, 2], [64, 64], [.01, .01]),
        ('fixed-terminal-fine-root', [1, 1+(remaining-1)//2, remaining], [8, 2, 2], [.01, 1., 1.]),
    ]
    reports = []
    for name, offsets, branches, steps in specifications:
        torch.cuda.reset_peak_memory_stats()
        start_time = time.perf_counter()
        tree, (prices, weights) = synthetic_tree(model, features, price, times, offsets, branches, 20260915, device)
        torch.cuda.synchronize()
        inference_seconds = time.perf_counter()-start_time
        solver = Bellman(prices, weights, [Menu(step=x) for x in steps], chunk=chunk, max_work=1_000_000_000)
        s = torch.tensor([[10000., 0., price[-1], price[-1], 0., 0., 0., 0., 0.]], device=device, dtype=torch.float64)
        tick = time.perf_counter()
        result = solver.solve(s)
        torch.cuda.synchronize()
        elapsed = time.perf_counter()-tick
        descriptor = [None if np.isnan(x) else x for x in result['descriptor'][0].tolist()]
        row = dict(name=name, stageOffsetsSeconds=offsets, branches=branches,
                   nonrootMarketNodes=sum(int(np.prod(branches[:i+1])) for i in range(len(branches))),
                   exposureSteps=steps, chunk=chunk, expectedWealth=result['value'].item(), descriptor=descriptor,
                   marketQuote=result['amount'].item(), terminalEvaluations=result['work'],
                   inferenceSeconds=inference_seconds, bellmanSeconds=elapsed,
                   peakAllocatedMiB=torch.cuda.max_memory_allocated()/2**20,
                   terminalIsFullBacktestEnd=offsets[-1] == remaining, terminalRequirement="zero-asset-and-no-orders")
        reports.append(row)
        (root/f'{name}-tree.json').write_text(json.dumps(tree), encoding='utf8')
        print(json.dumps(row, allow_nan=False), flush=True)
        (root/'benchmark.json').write_text(json.dumps(dict(gpu=torch.cuda.get_device_name(), checkpointSha256=digest,
            actualBacktestSeconds=remaining, results=reports, numericalCertificate=False), indent=2, allow_nan=False), encoding='utf8')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, default=Path('data/backtests/oracle-gpu-scaling'))
    ap.add_argument('--audit-only', action='store_true')
    ap.add_argument('--benchmark-only', action='store_true')
    ap.add_argument('--chunk', type=int, default=131072)
    args = ap.parse_args()
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('high')
    device = torch.device('cuda')
    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    if not args.benchmark_only:
        audit(root, device)
    if not args.audit_only:
        benchmark(root, device, args.chunk)


if __name__ == '__main__':
    main()
