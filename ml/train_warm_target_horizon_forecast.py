"""Warm-start multi-horizon heads and audit exact parent preservation before fitting."""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '4'
import argparse
from dataclasses import asdict
from pathlib import Path
import json
import time
import numpy as np
import torch
from audit_forecast_agreement import sha
from horizon_price_forecast import load_horizon_forecast
from target_horizon_forecast import HorizonRange
from warm_target_horizon_forecast import CONTRACT, WarmTargetHorizonForecast, load_warm_forecast
from train_target_horizon_forecast import prepare, targets, objective, evaluate, write


@torch.inference_mode()
def equality_audit(model, features, device):
    error = 0.
    for start in range(0, len(features), 2560):
        x = torch.from_numpy(features[start:start+2560]).to(device)
        parent, extended = model.head(x), model(x, 1)
        if not torch.equal(parent, extended):
            raise ValueError('adding horizon input changed the parent H1 logits')
        error = max(error, float((parent-extended).abs().max()))
    return {'examples': len(features), 'maximumLogitDifference': error,
        'basisAndNormalizerUnchanged': True, 'h1ReturnScale': 1.0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('data/backtests/oracle-warm-target-horizon'))
    parser.add_argument('--epochs', type=int, default=48)
    args = parser.parse_args()
    root = args.output
    if root.exists() and any(root.iterdir()):
        raise ValueError('refusing to overwrite experiment')
    root.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4); torch.set_float32_matmul_precision('high')
    device, hrange = torch.device('cuda'), HorizonRange()
    start_time = time.monotonic()
    plan = {'contract': CONTRACT, 'parents': ['direct59', 'base17'], 'horizonRange': asdict(hrange),
        'initialization': 'copy every parent head weight, normalizer and knot parameter; add zero horizon weights at first GLU preactivation',
        'horizonInput': 'tanh(h/21600)-tanh(1/21600); h in integer seconds',
        'target': 'log(P[t+h]/P[t]) / sqrt(h), same parent basis; scale is exactly 1 at h=1',
        'sampling': 'same integrated tanh derivative as previous run; two draws per origin per epoch, h=1..86400',
        'arms': {'horizonWeightsOnly': 'parent frozen, new horizon weights only; H1 exactly preserved for all updates',
            'jointFinetune': 'parent head and new horizon weights train jointly; H1 preserved initially, can change later'},
        'nativeModels': 'frozen; original causal native feature caches reused',
        'loss': 'raw-return mixed-law NLL + scaled-return mean MSE / training scaled-return second moment',
        'epochs': args.epochs, 'minimumEpochs': 24, 'patience': 12, 'batchSize': 4096,
        'parentLearningRate': .0001, 'horizonLearningRate': .001, 'seed': 20260912,
        'selection': 'minimum fixed derivative-sampled validation loss for each arm; test opened after all selections freeze',
        'onlineCalibration': 'unchanged: none for these two checkpointed standalone refits; not a replacement of the calibrated ensemble',
        'sourceHashes': {f: sha(f) for f in [__file__, 'ml/warm_target_horizon_forecast.py',
            'ml/target_horizon_forecast.py', 'ml/train_target_horizon_forecast.py', 'ml/horizon_price_forecast.py']}}
    write(root, 'plan.json', plan)
    selection = {'planSha256': sha(root/'plan.json'), 'models': {}, 'parents': {}, 'testUsedForSelection': False}
    contexts = {}
    for parent in plan['parents']:
        native = Path('data/backtests/oracle-expanded-mixture')/parent
        nplan = json.loads((native/'plan.json').read_text())
        original_selection = json.loads((native/'selection.json').read_text())
        if sha(native/'head.pt') != original_selection['headCheckpointSha256']:
            raise ValueError('parent checkpoint changed')
        dataset = Path(json.loads(Path(nplan['nativePlan']).read_text())['datasetDir'])
        train, val = (prepare(native, dataset, split, hrange) for split in ('train', 'validation'))
        if train['origins'][0] <= nplan['nativeTrainingLastTargetOrigin']:
            raise ValueError('native/head training overlap')
        fit_h = hrange.sample(np.random.default_rng(20260912), len(train['features']))
        normalization = float(np.mean(targets(train, fit_h)**2/fit_h))
        vi = np.tile(np.arange(len(val['features'])), 2)
        vh = hrange.sample(np.random.default_rng(20260913), len(vi))
        vy = targets(val, vh, vi)
        tx = torch.from_numpy(train['features']).to(device)
        vx, vhh, vyy = torch.from_numpy(val['features'][vi]).to(device), torch.from_numpy(vh).to(device), torch.from_numpy(vy).to(device)
        contexts[parent] = (native, dataset)
        pmeta = {'headSha256': sha(native/'head.pt'), 'nativePlanSha256': sha(native/'plan.json'),
            'trainOrigins': len(train['features']), 'validationOrigins': len(val['features']),
            'trainCacheSha256': sha(native/'train-native.npz'), 'validationCacheSha256': sha(native/'validation-native.npz')}
        head, _ = load_horizon_forecast(native/'head.pt', device)
        initial = WarmTargetHorizonForecast(head).to(device).eval()
        for split in ('train', 'validation'):
            with np.load(native/f'{split}-native.npz') as f:
                pmeta[split+'InitialH1Audit'] = equality_audit(initial, f['features'], device)
        initial_payload = {'contract': CONTRACT, 'model': initial.state_dict(), 'featureMean': head.feature_mean,
            'featureStd': head.feature_std, 'basis': head.basis, 'widths': list(head.widths), 'horizonRange': asdict(hrange),
            'nativeDirectory': str(native), 'nativePlanSha256': sha(native/'plan.json'),
            'parentHeadSha256': pmeta['headSha256'], 'planSha256': selection['planSha256'], 'epoch': 0}
        torch.save(initial_payload, root/f'{parent}-initial.pt')
        pmeta['initialCheckpointSha256'] = sha(root/f'{parent}-initial.pt')
        selection['parents'][parent] = pmeta
        print(json.dumps({'parent': parent, 'initialH1Preserved': pmeta}), flush=True)
        for arm in plan['arms']:
            name = parent+'-'+arm
            model, payload = load_warm_forecast(root/f'{parent}-initial.pt', device)
            model.head.requires_grad_(arm == 'jointFinetune')
            groups = [{'params': [model.horizon_weight], 'lr': .001}]
            if arm == 'jointFinetune':
                groups.append({'params': list(model.head.parameters()), 'lr': .0001})
            optimizer = torch.optim.Adam(groups)
            best, best_epoch, history = float('inf'), -1, []
            histogram = np.zeros(hrange.maximum_seconds, dtype=np.int64)
            for epoch in range(args.epochs+1):
                train_sum = 0.
                if epoch:
                    rng = np.random.default_rng(20260912+epoch)
                    ids = np.tile(np.arange(len(train['features'])), 2); rng.shuffle(ids)
                    h = hrange.sample(rng, len(ids)); y = targets(train, h, ids)
                    histogram += np.bincount(h, minlength=hrange.maximum_seconds+1)[1:]
                    model.train()
                    for start in range(0, len(ids), 4096):
                        sl = slice(start, start+4096)
                        ii, hh, yy = (torch.from_numpy(a[sl]).to(device) for a in (ids, h, y))
                        optimizer.zero_grad(set_to_none=True)
                        loss = objective(model, tx[ii], hh, yy, normalization)
                        if not bool(torch.isfinite(loss)):
                            raise ValueError('nonfinite training objective')
                        train_sum += float(loss.detach())*len(ii)
                        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 100); optimizer.step()
                model.eval()
                with torch.inference_mode():
                    score = sum(float(objective(model, vx[i:i+4096], vhh[i:i+4096], vyy[i:i+4096], normalization))*len(vyy[i:i+4096])
                        for i in range(0, len(vyy), 4096))/len(vyy)
                row = {'epoch': epoch, 'validationLoss': score,
                    'onlineTrainingLoss': train_sum/len(ids) if epoch else None, 'seconds': time.monotonic()-start_time}
                history.append(row)
                if score < best:
                    best, best_epoch = score, epoch
                    torch.save({**payload, 'model': model.state_dict(), 'epoch': epoch, 'validationLoss': score,
                        'arm': arm}, root/f'{name}.pt')
                write(root, name+'-history.json', history)
                print(json.dumps({'model': name, **row, 'bestEpoch': best_epoch}), flush=True)
                if epoch >= 24 and epoch-best_epoch >= 12:
                    break
            torch.save({**payload, 'model': model.state_dict(), 'epoch': epoch, 'arm': arm,
                'diagnosticOnly': True}, root/f'{name}-last.pt')
            selected, _ = load_warm_forecast(root/f'{name}.pt', device)
            item = {'epoch': best_epoch, 'validationLoss': best, 'checkpointSha256': sha(root/f'{name}.pt'),
                'horizonWeightNorm': float(selected.horizon_weight.detach().norm()), 'trainingPairs': int(histogram.sum())}
            if arm == 'horizonWeightsOnly':
                for key, tensor in initial.head.state_dict().items():
                    if not torch.equal(tensor, selected.head.state_dict()[key]):
                        raise ValueError('frozen parent head changed')
                item['trainedH1Audit'] = equality_audit(selected, val['features'], device)
            selection['models'][name] = item
            np.savez_compressed(root/f'{name}-horizon-counts.npz', counts=histogram)
    write(root, 'selection.json', selection)
    frozen_hash = sha(root/'selection.json')
    result = {'selectionSha256': frozen_hash, 'models': {}, 'initialH1Audits': {},
        'note': 'Previously inspected historical test. Long-horizon observations overlap. No policy backtest.'}
    for parent, (native, dataset) in contexts.items():
        common = prepare(native, dataset, 'test', hrange)
        h1 = prepare(native, dataset, 'test', HorizonRange(hrange.scale_seconds, 1))
        initial, _ = load_warm_forecast(root/f'{parent}-initial.pt', device)
        result['initialH1Audits'][parent] = equality_audit(initial, h1['features'], device)
        for suffix in ['initial', *plan['arms']]:
            name = parent+'-'+suffix
            model, _ = load_warm_forecast(root/f'{name}.pt', device)
            result['models'][name] = {}
            for horizon in [1, 5, 60, 300, 900, 3600, 7200, 21600, 43200, 86400]:
                data = h1 if horizon == 1 else common
                metrics, arrays = evaluate(model, data, horizon, device)
                result['models'][name][str(horizon)] = metrics
                np.savez_compressed(root/f'{name}-test-h{horizon}.npz', **arrays)
            if suffix == 'horizonWeightsOnly':
                with torch.inference_mode():
                    for i in range(0, len(h1['features']), 2560):
                        xx = torch.from_numpy(h1['features'][i:i+2560]).to(device)
                        if not torch.equal(initial(xx, 1), model(xx, 1)):
                            raise ValueError('trained horizon-only weights changed H1')
            print(json.dumps({'model': name, 'test': result['models'][name]}), flush=True)
    if sha(root/'selection.json') != frozen_hash:
        raise ValueError('selection changed during test evaluation')
    result['seconds'] = time.monotonic()-start_time
    write(root, 'evaluation.json', result)


if __name__ == '__main__':
    main()
