"""Train shared full price laws with derivative-weighted target horizons."""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '4'
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
import numpy as np
import torch
from audit_forecast_agreement import sha, basis_cdf
from export_stochastic_price_forecast import load_close_prices
from horizon_price_forecast import fit_horizon_basis
from stochastic_price_forecast import load_timeline_split, component_log_masses
from target_horizon_forecast import CONTRACT, HorizonRange, TargetHorizonForecast, complete_origins, load_target_horizon_forecast


def write(root, name, value):
    (root/name).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def prepare(native_root, dataset, split, hrange):
    source = load_timeline_split(dataset, split)
    manifest = json.loads((dataset/'manifest.json').read_text())
    end = int(np.datetime64(manifest['splits'][split]['endExclusive'].replace('Z', ''), 'ms').astype('int64'))
    last = min(len(source.timeline)-1, (end-1-source.axis_start_ms)//1000)
    with np.load(native_root/f'{split}-native.npz') as f:
        original = f['origins']
        origins = complete_origins(original, last, hrange)
        n = len(origins)
        if n == 0:
            raise ValueError('no origins with complete maximum-horizon targets')
        np.testing.assert_array_equal(origins, original[:n])
        features = f['features'][:n]
        timestamps = f['timestamps'][:n]
        np.testing.assert_array_equal(timestamps, source.axis_start_ms+origins*1000)
    first, final = int(origins[0]), int(origins[-1]+hrange.maximum_seconds)
    prices = load_close_prices(source.axis_start_ms+np.arange(first, final+1)*1000,
        Path('data/market/immutable/refs/candles/spot-btcusdt/btcusdt/1s'))
    if not np.isfinite(prices).all() or np.any(prices <= 0):
        raise ValueError('invalid price history')
    np.testing.assert_allclose(prices*100, np.round(prices*100), atol=1e-7, rtol=0)
    return {'features': features, 'prices': prices, 'indices': origins-first,
        'origins': origins, 'timestamps': timestamps,
        'lastTargetTimestamp': int(source.axis_start_ms+final*1000), 'splitEndExclusive': end}


def targets(data, horizons, ids=None):
    ix = data['indices'] if ids is None else data['indices'][ids]
    if np.shape(horizons) not in ((), ix.shape) or np.any(horizons < 1):
        raise ValueError('invalid target horizons')
    return np.log1p(data['prices'][ix+horizons]/data['prices'][ix]-1)


def objective(model, features, h, y, normalization):
    logits = model(features, h)
    # Normalize MSE in the same sqrt(h) coordinate used for the law.
    error = (model.expectation_from_logits(logits, h).double()-y)/h.double().sqrt()
    return -model.log_prob_from_logits(logits, y, h).mean()+error.square().mean()/normalization


def metric(mean, y, ll, sign_probability):
    active = y != 0
    mse = np.mean((mean-y)**2)
    return {'examples': len(y), 'nonzeroExamples': int(active.sum()),
        'skillPercent': float(100*(1-mse/np.mean(y*y))),
        'mseBpsSquared': float(mse*1e8),
        'nonzeroSkillPercent': float(100*(1-np.mean((mean[active]-y[active])**2)/np.mean(y[active]**2))),
        'signAccuracyNonzeroPercent': float(100*np.mean((sign_probability[active] > .5) == (y[active] > 0))),
        'negativeLogLikelihood': float(-np.mean(ll)),
        'meanAbsolutePredictionBps': float(np.mean(abs(mean))*1e4),
        'meanAbsoluteRealizedBps': float(np.mean(abs(y))*1e4)}


@torch.inference_mode()
def evaluate(model, data, h, device):
    ys = targets(data, h)
    # Independent high-order float64 moments for final reporting.
    means = model.head._component_moments('log', 768)
    means_low = model.head._component_moments('log', 384)
    knots, areas = np.asarray(model.head.basis['knots']), np.asarray(model.head.basis['basisAreas'])
    positive = 1-basis_cdf(knots, areas, np.array([.5]))[:, 0]
    all_mean, all_ll, all_sign, all_zero = [], [], [], []
    max_error = 0.
    for start in range(0, len(ys), 2048):
        end = min(start+2048, len(ys))
        x = torch.from_numpy(data['features'][start:end, :model.feature_count]).to(device)
        hs = torch.full((len(x),), h, device=device)
        logits = model(x, hs)
        z = logits[:, 0].double().sigmoid().cpu().numpy()
        masses = component_log_masses(logits[:, 1:].double(), model.head.areas.double()).exp().cpu().numpy()
        masses /= masses.sum(axis=1, keepdims=True)
        mean = (1-z)*(masses@means)*np.sqrt(h)
        low = (1-z)*(masses@means_low)*np.sqrt(h)
        max_error = max(max_error, float(np.max(abs(mean-low))))
        ll = model.log_prob_from_logits(logits, torch.from_numpy(ys[start:end]).to(device), hs).cpu().numpy()
        all_mean.append(mean); all_ll.append(ll); all_sign.append(masses@positive); all_zero.append(z)
    mean, ll, sign, pzero = map(np.concatenate, (all_mean, all_ll, all_sign, all_zero))
    if max_error > 1e-8:
        raise ValueError('final log-moment quadrature has not converged')
    return {**metric(mean, ys, ll, sign), 'maximumQuadratureDifferenceLogReturn': max_error}, {
        'mean': mean, 'y': ys, 'logLikelihood': ll, 'conditionalPositiveProbability': sign, 'zeroProbability': pzero,
        'origins': data['origins'], 'timestamps': data['timestamps']}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('data/backtests/oracle-target-horizon'))
    parser.add_argument('--epochs', type=int, default=32)
    parser.add_argument('--scale-seconds', type=float, default=21600.)
    parser.add_argument('--maximum-seconds', type=int, default=86400)
    args = parser.parse_args()
    root = args.output
    if root.exists() and any(root.iterdir()):
        raise ValueError('refusing to overwrite an existing experiment')
    root.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4); torch.set_float32_matmul_precision('high')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    hrange = HorizonRange(args.scale_seconds, args.maximum_seconds)
    native_root = Path('data/backtests/oracle-expanded-mixture/direct59')
    native_plan = json.loads((native_root/'plan.json').read_text())
    dataset = Path(json.loads(Path(native_plan['nativePlan']).read_text())['datasetDir'])
    plan = {'contract': CONTRACT, 'horizonRange': asdict(hrange),
        'sampling': 'Pr(H=h)=[tanh(h/s)-tanh((h-1)/s)]/tanh(Hmax/s); independent horizon draws each epoch',
        'target': 'endpoint log(P[t+h]/P[t]); exact zero atom; invertible sqrt(h) coordinate',
        'forecast': 'shared width 128,128 GLU; full Gaussian-CDF triangular knot density',
        'nativeDirectory': str(native_root), 'nativePlanSha256': sha(native_root/'plan.json'),
        'dataset': str(dataset), 'epochs': args.epochs, 'patience': 8, 'minimumEpochs': 16,
        'samplesPerOriginPerEpoch': 2, 'batchSize': 4096, 'learningRate': .001, 'seed': 20260912,
        'loss': 'full raw-return mixed-law NLL + sqrt(h)-normalized mean MSE / train normalized second moment',
        'checkpointSelection': 'minimum loss on frozen derivative-sampled validation pairs; all validation used',
        'comparison': 'separately trained horizon-only law, same sampled targets and training objective',
        'onlineCalibration': 'none in this new architecture screen',
        'splitProtection': 'common origins have complete 24h target within their own split; no clipping of sampled horizons',
        'sourceHashes': {p: sha(p) for p in [__file__, 'ml/target_horizon_forecast.py',
            'ml/horizon_price_forecast.py', 'ml/stochastic_price_forecast.py']}}
    write(root, 'plan.json', plan)
    start_time = time.monotonic()
    train, val = (prepare(native_root, dataset, s, hrange) for s in ('train', 'validation'))
    if train['origins'][0] <= native_plan['nativeTrainingLastTargetOrigin']:
        raise ValueError('head training overlaps native fitting')
    rng = np.random.default_rng(20260912)
    fit_h = hrange.sample(rng, len(train['features']))
    scaled_y = targets(train, fit_h)/np.sqrt(fit_h)
    basis = fit_horizon_basis(scaled_y)
    normalization = float(np.mean(scaled_y**2))
    val_ids = np.tile(np.arange(len(val['features'])), 2)
    val_h = hrange.sample(np.random.default_rng(20260913), len(val_ids))
    val_y = targets(val, val_h, val_ids)
    np.savez_compressed(root/'validation-selection-pairs.npz', ids=val_ids, horizons=val_h, y=val_y)
    vx = torch.from_numpy(val['features'][val_ids]).to(device)
    vh, vy = torch.from_numpy(val_h).to(device), torch.from_numpy(val_y).to(device)
    tx = torch.from_numpy(train['features']).to(device)
    summaries = {}
    for split, data in [('train', train), ('validation', val)]:
        summaries[split] = {'origins': len(data['origins']), 'firstTimestamp': int(data['timestamps'][0]),
            'lastTimestamp': int(data['timestamps'][-1]), 'lastTargetTimestamp': data['lastTargetTimestamp'],
            'splitEndExclusive': data['splitEndExclusive'], 'cacheSha256': sha(native_root/f'{split}-native.npz')}
    selection = {'planSha256': sha(root/'plan.json'), 'splits': summaries, 'models': {}, 'testUsedForSelection': False}
    total_histogram = np.zeros(hrange.maximum_seconds, dtype=np.int64)
    for name, count in [('featuresAndHorizon', train['features'].shape[1]), ('horizonOnly', 0)]:
        torch.manual_seed(20260912)
        mu = torch.from_numpy(train['features'][:, :count].mean(axis=0, dtype=float).astype(np.float32))
        std = torch.from_numpy(np.maximum(train['features'][:, :count].std(axis=0, dtype=float), 1e-8).astype(np.float32))
        model = TargetHorizonForecast(mu, std, basis, hrange).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        best, best_epoch, history = float('inf'), -1, []
        for epoch in range(args.epochs+1):
            if epoch:
                erng = np.random.default_rng(20260912+epoch)
                ids = np.tile(np.arange(len(train['features'])), 2)
                erng.shuffle(ids)
                h = hrange.sample(erng, len(ids))
                if name == 'featuresAndHorizon':
                    total_histogram += np.bincount(h, minlength=hrange.maximum_seconds+1)[1:]
                y = targets(train, h, ids)
                model.train()
                for begin in range(0, len(ids), 4096):
                    sl = slice(begin, begin+4096)
                    ii = torch.from_numpy(ids[sl]).to(device)
                    hh, yy = torch.from_numpy(h[sl]).to(device), torch.from_numpy(y[sl]).to(device)
                    optimizer.zero_grad(set_to_none=True)
                    loss = objective(model, tx[ii, :count], hh, yy, normalization)
                    if not bool(torch.isfinite(loss)):
                        raise ValueError('nonfinite training loss')
                    loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 100); optimizer.step()
            model.eval()
            with torch.inference_mode():
                score = sum(float(objective(model, vx[i:i+4096, :count], vh[i:i+4096], vy[i:i+4096], normalization))*len(vy[i:i+4096])
                    for i in range(0, len(vy), 4096))/len(vy)
            row = {'epoch': epoch, 'validationLoss': score, 'seconds': time.monotonic()-start_time}
            history.append(row)
            if score < best:
                best, best_epoch = score, epoch
                torch.save({'contract': CONTRACT, 'model': model.state_dict(), 'featureMean': mu,
                    'featureStd': std, 'basis': basis, 'widths': list(model.widths), 'horizonRange': asdict(hrange),
                    'nativeDirectory': str(native_root), 'nativePlanSha256': plan['nativePlanSha256'],
                    'epoch': epoch, 'validationLoss': score, 'planSha256': selection['planSha256']}, root/f'{name}.pt')
            print(json.dumps({'model': name, **row, 'bestEpoch': best_epoch}), flush=True)
            write(root, name+'-history.json', history)
            if epoch >= 16 and epoch-best_epoch >= 8:
                break
        selection['models'][name] = {'epoch': best_epoch, 'validationLoss': best, 'checkpointSha256': sha(root/f'{name}.pt')}
    np.savez_compressed(root/'training-horizon-counts.npz', counts=total_histogram, probabilities=hrange.probabilities())
    selection['horizonSampling'] = {'totalPairs': int(total_histogram.sum()),
        'distinctHorizonsSeen': int(np.count_nonzero(total_histogram)),
        'upTo2hFraction': float(total_histogram[:7200].sum()/total_histogram.sum()),
        'beyond6hFraction': float(total_histogram[21600:].sum()/total_histogram.sum())}
    write(root, 'selection.json', selection)
    frozen_selection_hash = sha(root/'selection.json')
    # Only now read test prices, labels, features, and previously scored controls.
    test = prepare(native_root, dataset, 'test', hrange)
    result = {'selectionSha256': frozen_selection_hash, 'models': {}, 'test': {
        'origins': len(test['origins']), 'firstTimestamp': int(test['timestamps'][0]), 'lastTimestamp': int(test['timestamps'][-1]),
        'lastTargetTimestamp': test['lastTargetTimestamp'], 'splitEndExclusive': test['splitEndExclusive']},
        'caveats': ['test period was inspected in earlier experiments', 'overlapping horizon targets are not independent',
            'endpoint marginals do not specify a joint path law or hypothetical feature transitions',
            'no online calibration or policy backtest in this experiment']}
    for name in selection['models']:
        model, _ = load_target_horizon_forecast(root/f'{name}.pt', device)
        result['models'][name] = {}
        for h in [1, 5, 60, 300, 900, 3600, 7200, 14400, 21600, 43200, 86400]:
            if h > hrange.maximum_seconds:
                continue
            score, arrays = evaluate(model, test, h, device)
            result['models'][name][str(h)] = score
            np.savez_compressed(root/f'{name}-test-h{h}.npz', **arrays)
        print(json.dumps({'model': name, 'test': result['models'][name]}), flush=True)
    with np.load(native_root/'test-law.npz') as old:
        n = len(test['origins'])
        np.testing.assert_array_equal(old['origins'][:n], test['origins'])
        np.testing.assert_allclose(old['y'][:n], targets(test, 1), atol=0, rtol=0)
        y, mean, ll = old['y'][:n], old['mean'][:n], old['logLikelihood'][:n]
        result['matchedFrozenH1Control'] = {'examples': n, 'skillPercent': float(100*(1-np.sum((mean-y)**2)/np.sum(y*y))),
            'negativeLogLikelihood': float(-ll.mean())}
    if sha(root/'selection.json') != frozen_selection_hash:
        raise ValueError('selection changed during test scoring')
    result['seconds'] = time.monotonic()-start_time
    write(root, 'evaluation.json', result)


if __name__ == '__main__':
    main()
