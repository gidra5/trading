"""Evaluate price marginals by integrating posterior, transition and decoder uncertainty."""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '4'
from pathlib import Path
import json
import argparse
import numpy as np
import torch
from audit_forecast_agreement import sha
from target_horizon_forecast import HorizonRange
from train_target_horizon_forecast import targets, write
from train_latent_neural_stochastic_flow import latent_data, sequence
from latent_neural_stochastic_flow import load_latent_nsf
from evaluate_neural_stochastic_flow import metrics


@torch.inference_mode()
def predict(model, data, ids, h, device, components=256, seed=13):
    draws, analytic_mean, zero_probability = [], [], []
    for start in range(0, len(ids), 64):
        ii = ids[start:start+64]
        f, price, elapsed, timestamps = sequence(data, ii, h)
        law = model.distribution(torch.from_numpy(f[:, :2]).to(device), torch.from_numpy(price[:, :2]).to(device),
            torch.from_numpy(elapsed[:, :2]).to(device), h, torch.from_numpy(timestamps[:, :2]).to(device), components, seed)
        values = torch.log(law.sample_prices(512)/law.current_price[:, None])
        draws.append(values.cpu().numpy())
        analytic_mean.append(law.mean_log_return().cpu().numpy())
        zero_probability.append(law.tick_probability(law.current_price).cpu().numpy())
    return np.concatenate(draws), np.concatenate(analytic_mean), np.concatenate(zero_probability)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('data/backtests/oracle-latent-nsf-sampled-skips'))
    root = parser.parse_args().root
    if (root/'evaluation.json').exists():
        raise ValueError('completed evaluation exists')
    selection = json.loads((root/'selection.json').read_text())
    plan = json.loads((root/'plan.json').read_text())
    for f, expected in plan['sourceHashes'].items():
        if sha(f) != expected:
            raise ValueError('source changed during experiment')
    frozen = sha(root/'selection.json')
    device = torch.device('cuda')
    torch.set_num_threads(4); torch.set_float32_matmul_precision('high')
    native, dataset = Path(plan['nativeDirectory']), Path(plan['dataset'])
    data = latent_data(native, dataset, 'test', HorizonRange())
    h1 = latent_data(native, dataset, 'test', HorizonRange(21600., 1))
    result = {'selectionSha256': frozen, 'models': {}, 'components': 256, 'observationSamples': 512,
        'testStride': 8, 'sourceSha256': sha(__file__),
        'notes': ['previously inspected historical test', 'long-horizon targets overlap',
            '256-component Gaussian mixture integrates current posterior and latent transition; decoder uncertainty retained',
            'current encoder sees only two past/current observations; no future endpoint or midpoint at inference',
            'no explicit zero atom; observation distribution rounded to .01 ticks']}
    for name, record in selection['models'].items():
        if sha(root/f'{name}.pt') != record['checkpointSha256']:
            raise ValueError('selected checkpoint changed')
        model, _ = load_latent_nsf(root/f'{name}.pt', device)
        result['models'][name] = {}
        for h in [1, 300, 900, 7200, 21600, 86400]:
            source = h1 if h == 1 else data
            ids = np.arange(0, len(source['origins']), 8)
            y = targets(source, h, ids)
            draws, mean, zero = predict(model, source, ids, h, device)
            score = metrics(draws, y)
            score['decoderIntegratedMeanSkillPercent'] = float(100*(1-np.sum((mean-y)**2)/np.sum(y*y)))
            score['decoderIntegratedZeroBrier'] = float(np.mean((zero-(y == 0))**2))
            _, alternate, _ = predict(model, source, ids[:128], h, device, components=512, seed=29)
            score['latentIntegrationMeanDifferenceRmsBps'] = float(np.sqrt(np.mean((mean[:128]-alternate)**2))*1e4)
            np.savez_compressed(root/f'{name}-h{h}.npz', mean=mean, sampledMean=draws.mean(1), y=y,
                zeroProbability=zero, origins=source['origins'][ids], timestamps=source['timestamps'][ids])
            result['models'][name][str(h)] = score
            print(json.dumps({'model': name, 'horizonSeconds': h, **score}), flush=True)
        f, p, elapsed, timestamps = sequence(data, np.array([0, 1]), 1234)
        law = model.distribution(torch.from_numpy(f[:, :2]).to(device), torch.from_numpy(p[:, :2]).to(device),
            torch.from_numpy(elapsed[:, :2]).to(device), 1234.5, torch.from_numpy(timestamps[:, :2]).to(device))
        result['models'][name]['arbitraryHorizonQuery'] = {'seconds': 1234.5, 'priceMixtureComponents': list(law.log_means.shape),
            'cdfAtCurrent': law.cdf(law.current_price).cpu().tolist(),
            'zeroMoveProbability': law.tick_probability(law.current_price).cpu().tolist()}
    if sha(root/'selection.json') != frozen or sha(native/'head.pt') != plan['existingParentSha256']:
        raise ValueError('selection or existing parent changed')
    write(root, 'evaluation.json', result)


if __name__ == '__main__':
    main()
