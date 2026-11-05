"""Matched price-marginal and direct/two-step diagnostics for trained NSFs."""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '4'
from pathlib import Path
import json
import numpy as np
import torch
from audit_forecast_agreement import sha
from horizon_price_forecast import load_horizon_forecast
from neural_stochastic_flow import load_nsf
from target_horizon_forecast import HorizonRange
from train_target_horizon_forecast import targets, write
from train_neural_stochastic_flow import market_data


def metrics(draws, y):
    mean = draws.mean(1)
    ordered = np.sort(draws, axis=1); n = draws.shape[1]
    crps = np.mean(abs(draws-y[:, None]), axis=1) - ordered@((2*np.arange(1, n+1)-n-1)/n**2)
    pzero, ppositive = np.mean(draws == 0, axis=1), np.mean(draws > 0, axis=1)
    active = y != 0
    return {'examples': len(y), 'skillPercent': float(100*(1-np.sum((mean-y)**2)/np.sum(y*y))),
        'mseBpsSquared': float(np.mean((mean-y)**2)*1e8), 'crpsBps': float(np.mean(crps)*1e4),
        'zeroBrier': float(np.mean((pzero-(y == 0))**2)),
        'signAccuracyNonzeroPercent': float(100*np.mean((ppositive[active] > (1-pzero[active])/2) == (y[active] > 0))),
        'meanAbsolutePredictionBps': float(np.mean(abs(mean))*1e4)}


@torch.inference_mode()
def sample_nsf(model, data, ids, h, device, seed=13, samples=512):
    result = []
    for start in range(0, len(ids), 64):
        ii = ids[start:start+64]
        f = torch.from_numpy(data['features'][ii]).to(device)
        vol = torch.from_numpy(data['state'][ii, 1]).to(device)
        current = torch.from_numpy(data['prices'][data['indices'][ii]]).to(device)
        law = model.distribution(f, vol, current, h)
        prices = law.sample_prices(samples, seed)
        result.append(torch.log(prices/current[:, None]).cpu().numpy())
    return np.concatenate(result)


@torch.inference_mode()
def sample_parent(model, data, ids, h, device, samples=512):
    result = []
    uv = torch.quasirandom.SobolEngine(2, scramble=True, seed=13).draw(samples).numpy()
    for start in range(0, len(ids), 128):
        ii = ids[start:start+128]
        f = torch.from_numpy(data['features'][ii]).to(device)
        zero, masses = model.probabilities(f)
        from scipy.special import ndtri
        active = model.sigma*ndtri(model.quantile_units(masses, uv[:, 1].astype(float)))*np.sqrt(h)
        draws = np.where(uv[None, :, 0] < zero[:, None], 0., active)
        current = data['prices'][data['indices'][ii], None]
        prices = np.maximum(1, np.round(current*np.exp(draws)/.01))*.01
        result.append(np.log(prices/current))
    return np.concatenate(result)


@torch.inference_mode()
def consistency(model, data, device, h, samples=256):
    ids = np.linspace(0, len(data['state'])-1, 64, dtype=int)
    state = torch.from_numpy(data['state'][ids]).to(device)
    features = torch.from_numpy(data['features'][ids]).to(device)
    u = torch.quasirandom.SobolEngine(6, scramble=True, seed=21).draw(samples//2).to(device)
    eps = torch.special.ndtri(u.clamp(1e-7, 1-1e-7)); eps = torch.cat((eps, -eps), 0)
    x = state[:, None].expand(-1, samples, -1).reshape(-1, 2)
    f = features[:, None].expand(-1, samples, -1).reshape(len(ids)*samples, -1)
    noise = eps[None].expand(len(ids), -1, -1).reshape(-1, 6)
    d = torch.full((len(x),), h/model.time_scale, device=device); start = torch.zeros_like(d)
    direct = model.rsample(x, f, d, start, noise[:, :2]).reshape(len(ids), samples, 2)
    null = model.rsample(x, f, d, start, noise[:, 2:4]).reshape(len(ids), samples, 2)
    mid = model.rsample(x, f, d/2, start, noise[:, 2:4])
    two = model.rsample(mid, f, d/2, d/2, noise[:, 4:]).reshape(len(ids), samples, 2)
    def distance(a, b):
        # Float32 cdist/matmul can make the small biased MMD spuriously negative.
        a, b = a.double(), b.double()
        w1 = (a[:, :, 0].sort(1).values-b[:, :, 0].sort(1).values).abs().mean()*model.price_scale*1e4
        # Biased finite-sample MMD with a fixed RBF kernel on scaled 2D states.
        aa = torch.exp(-torch.cdist(a, a).square()/2).mean((1, 2))
        bb = torch.exp(-torch.cdist(b, b).square()/2).mean((1, 2))
        ab = torch.exp(-torch.cdist(a, b).square()/2).mean((1, 2))
        return {'priceWassersteinBps': float(w1), 'jointMmdSquared': float((aa+bb-2*ab).mean())}
    return {'directVsTwoStep': distance(direct, two), 'directVsDirectSamplingFloor': distance(direct, null)}


def main():
    root = Path('data/backtests/oracle-neural-stochastic-flow')
    if (root/'evaluation.json').exists():
        raise ValueError('completed evaluation exists')
    selection = json.loads((root/'selection.json').read_text())
    plan = json.loads((root/'plan.json').read_text())
    frozen = sha(root/'selection.json')
    for file, expected in plan['sourceHashes'].items():
        if sha(file) != expected:
            raise ValueError('training source changed')
    for name, record in selection['models'].items():
        if sha(root/f'{name}.pt') != record['checkpointSha256']:
            raise ValueError('checkpoint changed')
    torch.set_num_threads(4); torch.set_float32_matmul_precision('high')
    device = torch.device('cuda')
    native, dataset = Path(plan['nativeDirectory']), Path(plan['dataset'])
    if sha(native/'head.pt') != plan['h1ParentCheckpointSha256']:
        raise ValueError('existing H1 checkpoint changed')
    common = market_data(native, dataset, 'test', HorizonRange())
    h1 = market_data(native, dataset, 'test', HorizonRange(21600., 1))
    result = {'selectionSha256': frozen, 'sampleCount': 512, 'testStride': 8,
        'priceLawEvaluation': '512 antithetic scrambled Sobol draws, rounded to .01 price grid; marginal metrics, no joint-vs-marginal NLL comparison',
        'models': {}, 'consistency': {}, 'previousH1CheckpointUnchanged': True,
        'limitations': ['previously inspected test', 'overlapping long-horizon targets',
            'joint continuous observation likelihood has no explicit zero atom; price rounding creates tick masses',
            'fixed origin context during generated transitions', 'NSF approximation, not a proved exact Chapman-Kolmogorov law'],
        'sourceSha256': sha(__file__)}
    for name in ['parentDiffusiveExtension', *selection['models']]:
        model = load_horizon_forecast(native/'head.pt', device)[0] if name == 'parentDiffusiveExtension' else load_nsf(root/f'{name}.pt', device)[0]
        result['models'][name] = {}
        for h in [1, 300, 900, 7200, 21600, 86400]:
            data = h1 if h == 1 else common
            ids = np.arange(0, len(data['state']), 8)
            y = targets(data, h, ids)
            draws = sample_parent(model, data, ids, h, device) if name == 'parentDiffusiveExtension' else sample_nsf(model, data, ids, h, device)
            score = metrics(draws, y)
            np.savez_compressed(root/f'{name}-h{h}.npz', mean=draws.mean(1), y=y,
                zeroProbability=np.mean(draws == 0, axis=1), origins=data['origins'][ids], timestamps=data['timestamps'][ids])
            if name != 'parentDiffusiveExtension':
                audit_ids = ids[:256]
                repeated = sample_nsf(model, data, audit_ids, h, device, seed=37, samples=1024)
                difference = draws[:len(audit_ids)].mean(1)-repeated.mean(1)
                score['meanIntegrationDifferenceRmsBps'] = float(np.sqrt(np.mean(difference**2))*1e4)
            result['models'][name][str(h)] = score
            print(json.dumps({'model': name, 'horizonSeconds': h, **score}), flush=True)
        if name != 'parentDiffusiveExtension':
            result['consistency'][name] = {str(h): consistency(model, common, device, h) for h in [300, 7200, 21600]}
            # Exercise an off-grid duration, not merely the reported horizons.
            example = model.distribution(torch.from_numpy(common['features'][:2]).to(device),
                torch.from_numpy(common['state'][:2, 1]).to(device),
                torch.from_numpy(common['prices'][common['indices'][:2]]).to(device), 1234.5)
            values = example.sample_prices(64)
            if not bool(torch.isfinite(values).all()):
                raise ValueError('nonfinite arbitrary-horizon sample')
            result['models'][name]['arbitraryHorizonQuery'] = {'seconds': 1234.5, 'sampleShape': list(values.shape),
                'cdfAtCurrentPrice': example.cdf(example.current_price).cpu().tolist()}
    if sha(root/'selection.json') != frozen:
        raise ValueError('evaluation changed selection')
    write(root, 'evaluation.json', result)


if __name__ == '__main__':
    main()
