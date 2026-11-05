"""Fit full price laws to freshly refitted historical feature forecasts.

The native model remains frozen. Head training uses the later training tail,
after the native model's last two-step target. Flat outcomes are restored.
"""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '4'
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.special import ndtr
import torch
from audit_forecast_agreement import sha, scores
from audit_current_forecast_mse import side_coefficients
from evaluate_stopped_structured_feature_process import model_from_state
from export_stochastic_price_forecast import load_close_prices
from horizon_price_forecast import HorizonPriceForecast, fit_horizon_basis, CONTRACT
from stochastic_price_forecast import load_timeline_split, component_log_masses
from trading_storage import load_torch_checkpoint


def price_targets(source, origins):
    first, last = int(origins[0]), int(origins[-1]+1)
    prices = load_close_prices(source.axis_start_ms+np.arange(first, last+1)*1000,
        Path('data/market/immutable/refs/candles/spot-btcusdt/btcusdt/1s'))
    ix = origins-first
    return np.log1p(prices[ix+1]/prices[ix]-1)


def forecast_law(model, features, y, device):
    _, up = side_coefficients(model, True, 1536)
    _, down = side_coefficients(model, False, 1536)
    _, lowup = side_coefficients(model, True, 768)
    _, lowdown = side_coefficients(model, False, 768)
    knots, areas = np.asarray(model.basis['knots']), np.asarray(model.basis['basisAreas'])
    output = {k: [] for k in ('mean', 'logLikelihood', 'zeroProbability', 'componentMasses')}
    for start in range(0, len(y), 2560):
        stop = min(start+2560, len(y)); target = y[start:stop]
        with torch.inference_mode():
            logits = model(torch.from_numpy(features[start:stop]).to(device))
            z = logits[:, 0].sigmoid().cpu().numpy().astype(float)
            masses = component_log_masses(logits[:, 1:], model.areas).exp().cpu().numpy().astype(float)
        masses /= masses.sum(axis=1, keepdims=True)
        mean = (1-z)*(masses@(up-down))
        np.testing.assert_allclose(mean, (1-z)*(masses@(lowup-lowdown)), atol=1e-12, rtol=0)
        unit = ndtr(target/model.sigma)
        interval = np.clip(np.searchsorted(knots, unit, side='right')-1, 0, len(knots)-2)
        fraction = (unit-knots[interval])/np.diff(knots)[interval]
        heights = masses/areas; ix = np.arange(len(target))
        density = heights[ix, interval]*(1-fraction)+heights[ix, interval+1]*fraction
        logdensity = np.log(density)-.5*(target/model.sigma)**2-np.log(model.sigma)-.5*np.log(2*np.pi)
        ll = np.where(target == 0, np.log(z), np.log1p(-z)+logdensity)
        for key, value in zip(output, (mean, ll, z, masses)):
            output[key].append(value)
    return {**{key: np.concatenate(values) for key, values in output.items()},
        'knots': knots, 'areas': areas, 'sigma': np.array(model.sigma)}


def evaluate_saved(root, device):
    plan = json.loads((root/'plan.json').read_text())
    selection = json.loads((root/'selection.json').read_text())
    if sha(root/'head.pt') != selection['headCheckpointSha256']:
        raise ValueError('frozen head changed')
    from horizon_price_forecast import load_horizon_forecast
    model, _ = load_horizon_forecast(root/'head.pt', device)
    result = {'plan': plan, 'selection': selection, 'splits': {},
        'inferenceDevice': str(device), 'inferenceBatchSize': 2560, 'matmulPrecision': 'high',
        'inferenceSourceSha256': sha(__file__), 'meanQuadrature': '1536-point per half-knot, checked against 768 at 1e-12 log-return tolerance'}
    for split in ('validation', 'test'):
        with np.load(root/f'{split}-native.npz') as data:
            law = forecast_law(model, data['features'], data['y'], device)
            np.savez_compressed(root/f'{split}-law.npz', **law, y=data['y'], origins=data['origins'], timestamps=data['timestamps'])
            result['splits'][split] = scores(law['mean'], data['y'], law['logLikelihood'])
    (root/'evaluation.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result['splits']), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidate', choices=['direct59', 'base17'], required=True)
    parser.add_argument('--evaluate-only', action='store_true')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    args = parser.parse_args()
    tag = 'oracle-ensemble-refit-'+args.candidate+'-20260912-v1'
    root = Path('data/backtests/oracle-expanded-mixture')/args.candidate
    root.mkdir(parents=True, exist_ok=True)
    if (root/'evaluation.json').exists():
        raise ValueError('finished head experiment already exists')
    device = torch.device(args.device)
    torch.set_num_threads(4); torch.set_float32_matmul_precision('high')
    if args.evaluate_only:
        evaluate_saved(root, device)
        return
    plan_file = Path('ml/training-plans')/(tag+'.json')
    plan = json.loads(plan_file.read_text()); run = Path(plan['runDir'])
    if json.loads((run/'state/status.json').read_text())['stage'] != 'complete':
        raise ValueError('native training must finish before head fitting')
    pointer = run/'checkpoints/selections/validation-mse.json'
    checkpoint = load_torch_checkpoint(pointer, map_location='cpu', weights_only=False)
    torch.manual_seed(20260912); np.random.seed(20260912)
    native = model_from_state(plan, checkpoint['model'], device)
    dataset = Path(plan['datasetDir'])
    manifest = json.loads((dataset/'manifest.json').read_text())
    active_origins = np.fromfile(dataset/manifest['files']['train']['origins'], dtype='<i4')
    last_native_target = int(active_origins[plan['subset']['examples']-1])+2
    head_plan = {'contract': 'frozen-historical-feature-knot-head-v1', 'candidate': args.candidate,
        'nativePlan': str(plan_file), 'nativePlanSha256': sha(plan_file),
        'nativeCheckpoint': str(pointer), 'nativeCheckpointPointer': json.loads(pointer.read_text()),
        'nativeSelectedEpochZeroBased': int(checkpoint['epoch']),
        'nativeTrainingLastTargetOrigin': last_native_target,
        'features': 'two predicted standardized native feature states flattened; no raw future labels or observed future states',
        'head': 'one width-128 normalized GLU; Gaussian-CDF triangular knot law plus exact zero atom',
        'headTraining': 'all outcomes in training tail strictly after native last target, cadence 4 seconds, at most 131072 rows',
        'epochs': 64, 'patience': 12, 'batchSize': 4096, 'learningRate': .001,
        'loss': 'NLL + mean LOG return MSE / training log-return second moment',
        'selection': 'minimum combined loss on first chronological validation half only',
        'onlineCalibration': 'none for new heads in this initial screen; archived calibrated models remain controls',
        'testUsedForSelection': False, 'datasetManifestSha256': sha(dataset/'manifest.json')}
    def write(name, value):
        (root/name).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    write('plan.json', head_plan)
    cached = {}
    for split in ('train', 'validation', 'test'):
        source = load_timeline_split(dataset, split)
        if split == 'train':
            origins = source.origins[source.origins > last_native_target][::4]
            if len(origins)>131072:
                origins = origins[np.linspace(0, len(origins)-1, 131072, dtype=int)]
            y = price_targets(source, origins)
            times = source.axis_start_ms+origins*1000
        else:
            with np.load(f'data/backtests/oracle-forecast-agreement/{split}-predictions.npz') as f:
                origins, y, times = f['origins'], f['y'], f['timestamps']
            np.testing.assert_array_equal(times, source.axis_start_ms+origins*1000)
            np.testing.assert_allclose(price_targets(source, origins), y, atol=0, rtol=0)
        if len(origins)==0:
            raise ValueError('no out-of-sample native forecasts for head training')
        outputs = []
        with torch.inference_mode():
            for start in range(0, len(origins), 2560):
                selected = origins[start:start+2560]
                inputs = np.asarray(source.timeline[selected[:, None]+np.array([-1, 0])], dtype=np.float32)
                forecast = native(torch.from_numpy(inputs).to(device))
                outputs.append(forecast.flatten(1).cpu().numpy())
        x = np.concatenate(outputs)
        if not np.isfinite(x).all(): raise ValueError('nonfinite native output')
        cached[split] = {'features': x, 'y': y, 'origins': origins, 'timestamps': times}
        np.savez_compressed(root/f'{split}-native.npz', **cached[split])
    del native; torch.cuda.empty_cache()
    train = cached['train']; tune = cached['validation']
    half = len(tune['y'])//2
    mu = torch.from_numpy(train['features'].mean(axis=0, dtype=float).astype(np.float32))
    std = torch.from_numpy(np.maximum(train['features'].std(axis=0, dtype=float), 1e-8).astype(np.float32))
    basis = fit_horizon_basis(train['y'])
    model = HorizonPriceForecast(mu, std, basis, widths=(128,)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    xt = torch.from_numpy(train['features']).to(device)
    yt = torch.from_numpy(train['y'].astype(np.float32)).to(device)
    xv = torch.from_numpy(tune['features'][:half]).to(device)
    yv = torch.from_numpy(tune['y'][:half].astype(np.float32)).to(device)
    normalization = float(np.mean(train['y']**2))
    def objective(logits, targets):
        return -model.log_prob_from_logits(logits, targets).mean()+(
            model.expectation_from_logits(logits)-targets).square().mean()/normalization
    best, best_epoch = float('inf'), -1
    history = []
    for epoch in range(65):
        if epoch:
            model.train()
            for ids in torch.randperm(len(yt), device=device).split(4096):
                optimizer.zero_grad(set_to_none=True)
                loss = objective(model(xt[ids]), yt[ids])
                if not bool(torch.isfinite(loss)): raise ValueError('nonfinite head loss')
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 100); optimizer.step()
        model.eval()
        with torch.inference_mode():
            value = sum(float(objective(model(xv[i:i+4096]), yv[i:i+4096]))*len(yv[i:i+4096])
                for i in range(0, len(yv), 4096))/len(yv)
        history.append({'epoch': epoch, 'validationSelectionLoss': value})
        if value < best:
            best, best_epoch = value, epoch
            torch.save({'contract': CONTRACT, 'model': model.state_dict(), 'featureMean': mu,
                'featureStd': std, 'basis': basis, 'widths': [128], 'plan': head_plan,
                'epoch': epoch, 'validationSelectionLoss': value}, root/'head.pt')
        print(json.dumps({'candidate': args.candidate, 'epoch': epoch, 'selectionLoss': value, 'bestEpoch': best_epoch}), flush=True)
        if epoch-best_epoch >= 12: break
    selected = torch.load(root/'head.pt', map_location=device, weights_only=False)
    model.load_state_dict(selected['model']); model.eval()
    selection = {'epoch': best_epoch, 'validationSelectionLoss': best, 'history': history,
        'headCheckpointSha256': sha(root/'head.pt'), 'frozenBeforeTestScoring': True,
        'nativeFitEnd': last_native_target, 'headFitFirstOrigin': int(train['origins'][0]),
        'headFitRows': len(yt), 'headSelectionRows': half}
    write('selection.json', selection)
    evaluate_saved(root, device)


if __name__ == '__main__':
    main()
