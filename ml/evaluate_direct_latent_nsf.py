"""Matched held-out price-law and latency checks for direct NSF marginals."""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '4'
import argparse
from pathlib import Path
import json
import time
import numpy as np
import torch

from audit_forecast_agreement import sha
from direct_latent_nsf import load_direct_latent_nsf
from latent_neural_stochastic_flow import load_latent_nsf
from target_horizon_forecast import HorizonRange
from train_latent_neural_stochastic_flow import latent_data
from train_direct_latent_nsf import history
from train_target_horizon_forecast import write
from evaluate_neural_stochastic_flow import metrics


@torch.inference_mode()
def compare(model, parent, data, ids, horizon, device):
    accum = {name: {k: [] for k in ('draws', 'mean', 'zero', 'll', 'sign')} for name in ('direct', 'latentMixture')}
    actuals, differences = [], []
    grid = torch.linspace(-4, 4, 33, device=device, dtype=torch.float64)
    for start in range(0, len(ids), 32):
        ii = ids[start:start+32]
        f, p, dt, times = history(data, ii, device)
        actual = torch.as_tensor(data['prices'][data['indices'][ii]+horizon], device=device)
        actuals.append((actual/p[:, -1]).log().cpu().numpy())
        direct = model.distribution(f, p, dt, horizon, times)
        teacher = parent.distribution(f, p, dt, horizon, times, components=256, seed=101)
        for name, law in [('direct', direct), ('latentMixture', teacher)]:
            record = accum[name]
            record['draws'].append((law.sample_prices(512, seed=103)/p[:, -1:]).log().cpu().numpy())
            record['mean'].append(law.mean_log_return().cpu().numpy())
            record['zero'].append(law.tick_probability(law.current_price).cpu().numpy())
            record['ll'].append(law.tick_probability(actual).clamp_min(1e-300).log().cpu().numpy())
            down = law.cdf(law.current_price-.01)
            up = 1-law.cdf(law.current_price)
            record['sign'].append((up > down).cpu().numpy())
        queries = p[:, -1:]*torch.exp(grid[None]*model.head.sigma*np.sqrt(horizon))
        boundaries = (torch.floor(queries/.01+1e-8)+.5)*.01
        r = (boundaries/p[:, -1:]).log()
        cdf = torch.special.ndtr((r[:, :, None]-teacher.log_means[:, None])/teacher.log_stds[:, None]).mean(-1)
        differences.append((direct.cdf(queries)-cdf).cpu().numpy())
    y = np.concatenate(actuals)
    results, arrays = {}, {}
    for name, chunks in accum.items():
        record = {k: np.concatenate(v) for k, v in chunks.items()}
        score = metrics(record['draws'], y)
        mean = record['mean']
        score.update({'meanSkillPercent': float(100*(1-np.sum((mean-y)**2)/np.sum(y*y))),
            'tickNll': float(-record['ll'].mean()),
            'zeroBrier': float(np.mean((record['zero']-(y == 0))**2)),
            'signAccuracyNonzeroPercent': float(100*np.mean(record['sign'][y != 0] == (y[y != 0] > 0)))})
        results[name] = score
        arrays[name+'Mean'] = mean
        arrays[name+'Zero'] = record['zero']
    diff = np.concatenate(differences)
    results['projectionDifference'] = {'cdfRmsOn33Boundaries': float(np.sqrt(np.mean(diff**2))),
        'cdfMaximumOn33Boundaries': float(abs(diff).max()),
        'meanDifferenceRmsBps': float(np.sqrt(np.mean((arrays['directMean']-arrays['latentMixtureMean'])**2))*1e4),
        'scope': 'measured finite query discrepancy against 256-component sampled parent, not a uniform error bound'}
    return results, {**arrays, 'y': y, 'origins': data['origins'][ids], 'timestamps': data['timestamps'][ids]}


@torch.inference_mode()
def latency(model, parent, data, device):
    def sync():
        if device.type == 'cuda':
            torch.cuda.synchronize()
    result = {}
    for batch in (1, 32):
        f, p, dt, times = history(data, np.arange(batch), device)
        calls = {'direct': lambda: model.distribution(f, p, dt, 3600., times),
                 'latentMixture256': lambda: parent.distribution(f, p, dt, 3600., times, components=256)}
        timings = {k: [] for k in calls}
        for fn in calls.values():
            for _ in range(3):
                fn()
        for i in range(20):
            for name in (list(calls) if i % 2 else list(reversed(calls))):
                sync(); begin = time.perf_counter(); calls[name](); sync()
                timings[name].append((time.perf_counter()-begin)*1000)
        result[str(batch)] = {name: {'medianMs': float(np.median(v)), 'p95Ms': float(np.quantile(v, .95))}
                              for name, v in timings.items()}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('data/backtests/oracle-direct-latent-nsf-20260921'))
    parser.add_argument('--rows', type=int, default=1024)
    args = parser.parse_args()
    root = args.root
    if args.rows < 1:
        raise ValueError('positive evaluation row count required')
    if (root/'evaluation.json').exists():
        raise ValueError('completed evaluation exists')
    plan = json.loads((root/'plan.json').read_text())
    selection = json.loads((root/'selection.json').read_text())
    for path, expected in plan['sourceHashes'].items():
        if sha(path) != expected:
            raise ValueError(f'training source changed: {path}')
    if sha(root/'model.pt') != selection['checkpointSha256'] or sha(plan['parent']) != plan['parentSha256']:
        raise ValueError('selected checkpoint changed')
    torch.set_num_threads(4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, saved = load_direct_latent_nsf(root/'model.pt', device)
    if saved['planSha256'] != sha(root/'plan.json'):
        raise ValueError('direct training plan changed')
    parent, _ = load_latent_nsf(plan['parent'], device)
    if any(not torch.equal(v, model.nsf.state_dict()[k]) for k, v in parent.state_dict().items()):
        raise ValueError('latent transitions changed in direct checkpoint')
    begun = time.monotonic()
    data = latent_data(Path(plan['nativeDirectory']), Path(plan['dataset']), 'test', HorizonRange())
    ids = np.linspace(0, len(data['origins'])-1, min(args.rows, len(data['origins'])), dtype=int)
    results = {'checkpointSha256': selection['checkpointSha256'], 'split': 'test',
        'previouslyInspectedHistoricalTest': True, 'overlappingHorizonTargets': True,
        'testUsedForSelection': False, 'device': str(device), 'horizons': {},
        'sourceSha256': sha(__file__), 'latentDynamicsExactlyPreserved': True}
    for h in (1, 300, 900, 3600, 7200, 21600, 86400):
        scores, arrays = compare(model, parent, data, ids, h, device)
        results['horizons'][str(h)] = scores
        np.savez_compressed(root/f'evaluation-h{h}.npz', **arrays)
        print(json.dumps({'horizonSeconds': h, **scores}), flush=True)
    results['latency'] = latency(model, parent, data, device)
    f, p, dt, times = history(data, np.arange(3), device)
    h = torch.tensor([0., 1., 1234.5], device=device)
    law = model.distribution(f, p, dt, h, times)
    repeated = model.distribution(f, p, dt, h, times)
    torch.testing.assert_close(law.masses, repeated.masses, atol=0, rtol=0)
    torch.testing.assert_close(law.masses.sum(1), torch.ones(3, device=device, dtype=torch.float64), atol=1e-12, rtol=0)
    assert float(law.zero_probability[0]) == 1
    assert float(law.tick_probability(p[:, -1])[0]) == 1
    results['queryChecks'] = {'horizonsSeconds': h.cpu().tolist(), 'shape': list(law.masses.shape),
        'zeroProbabilities': law.zero_probability.cpu().tolist(), 'deterministicOutput': True,
        'zeroHorizonExact': True, 'massNormalization': True}
    results['seconds'] = time.monotonic()-begun
    write(root, 'evaluation.json', results)
    print(json.dumps({'latency': results['latency'], 'seconds': results['seconds']}), flush=True)


if __name__ == '__main__':
    main()
