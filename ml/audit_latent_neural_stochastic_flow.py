"""Audit trained calendar-aware latent NSF checkpoints and causal price mixtures."""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '4'
import argparse
from pathlib import Path
import json
import numpy as np
import torch
from audit_forecast_agreement import sha
from latent_neural_stochastic_flow import load_latent_nsf, clock_features, sample_skip_pairs
from target_horizon_forecast import HorizonRange
from train_latent_neural_stochastic_flow import latent_data, sequence, sample_horizons
from train_target_horizon_forecast import write


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('data/backtests/oracle-latent-nsf-sampled-skips'))
    root = parser.parse_args().root
    if (root/'audit.json').exists():
        raise ValueError('completed audit exists')
    plan = json.loads((root/'plan.json').read_text())
    selection = json.loads((root/'selection.json').read_text())
    for file, expected in plan['sourceHashes'].items():
        if sha(file) != expected:
            raise ValueError('training source changed')
    device = torch.device('cuda')
    torch.set_num_threads(4); torch.set_float32_matmul_precision('high')
    data = latent_data(Path(plan['nativeDirectory']), Path(plan['dataset']), 'validation', HorizonRange())
    f, prices, elapsed, timestamps = sequence(data, np.arange(32), 7200)
    features, seconds, times = (torch.from_numpy(v).to(device) for v in (f, elapsed, timestamps))
    reports = {}
    for name, info in selection['models'].items():
        if sha(root/f'{name}.pt') != info['checkpointSha256']:
            raise ValueError('selected checkpoint changed')
        model, saved = load_latent_nsf(root/f'{name}.pt', device)
        assert model.dimensions == model.width == 256
        obs = model.observations(features, torch.from_numpy(np.log(prices/prices[:, 1, None])).to(device))
        with torch.inference_mode():
            old = model.encode(obs, seconds, times)
            changed = obs.clone(); changed[:, 2:] += 100
            new = model.encode(changed, seconds, times)
            assert torch.equal(old[0][:, :2], new[0][:, :2]) and torch.equal(old[1][:, :2], new[1][:, :2])
            args = (features[:, :2], torch.from_numpy(prices[:, :2]).to(device), seconds[:, :2], 1234.5)
            law = model.distribution(*args, times[:, :2])
            shifted = model.distribution(*args, times[:, :2]+21600)
            decoder_variance = law.log_stds.square().mean(1)
            latent_variance = law.log_means.var(1, unbiased=False)
            assert bool(torch.all(decoder_variance > 0) & torch.all(latent_variance > 0))
            prob = law.tick_probability(law.current_price)
            assert bool(torch.all(torch.isfinite(prob) & (prob >= 0) & (prob <= 1)))
            clock_changes_forecast = not torch.equal(law.log_means, shifted.log_means)
            if saved['epoch'] > 0:
                assert clock_changes_forecast
            reports[name] = {'selectedEpoch': saved['epoch'], 'causalPosterior': True, 'calendarChangesForecast': clock_changes_forecast,
                'mixtureShape': list(law.log_means.shape),
                'calendarShiftMeanDifferenceRmsBps': float(((law.mean_log_return()-shifted.mean_log_return()).square().mean()).sqrt()*1e4),
                'meanLatentVarianceLogReturn': float(latent_variance.mean()),
                'meanDecoderVarianceLogReturn': float(decoder_variance.mean())}
    skip_rng = torch.Generator().manual_seed(81)
    skips = {}
    for horizon in [1, 2, 3, 6, 7200, 86400]:
        _, _, gaps, clock = sequence(data, np.arange(32), horizon)
        dt = torch.from_numpy(gaps)
        batch, source, target = sample_skip_pairs(dt, uniforms=torch.rand(dt.shape, generator=skip_rng))
        active = dt > 0; active[:, 0] = True
        ranks = active.long().cumsum(1)
        gap = ranks[batch, target]-ranks[batch, source]
        assert bool(((gap >= 2) & (gap <= 4)).all())
        assert bool(active[batch, source].all() & active[batch, target].all())
        assert np.all(clock[batch, target] > clock[batch, source])
        assert (source*dt.shape[0]+batch).unique().numel() == len(batch)
        skips[str(horizon)] = {'pairs': len(batch), 'sourceIndices': source.unique().tolist(),
            'targetIndices': target.unique().tolist()}
    coverage = {}
    for name in selection['models']:
        epochs = len(json.loads((root/f'{name}-history.json').read_text()))-1
        h1 = 0; bins = np.zeros(4, dtype=int)
        for epoch in range(1, epochs+1):
            rng = np.random.default_rng(plan['seed']+epoch)
            rng.integers(0, selection['trainOrigins'], plan['sequencesPerEpoch'])
            horizons = sample_horizons(rng, plan['sequencesPerEpoch'])
            h1 += int((horizons == 1).sum())
            bins += np.histogram(horizons, [1, 21601, 43201, 64801, 86401])[0]
        coverage[name] = {'epochs': epochs, 'oneSecondExamples': h1, 'sixHourBinCounts': bins.tolist()}
    write(root, 'audit.json', {'sourceHashesVerified': True, 'checkpointHashesVerified': True,
        'models': reports, 'uniformSamplingCoverage': coverage, 'sampledSkipCoverage': skips, 'auditSourceSha256': sha(__file__)})
    print(json.dumps(reports), flush=True)


if __name__ == '__main__':
    main()
