"""Market experiment for the NeurIPS 2025 NSF transition architecture."""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '4'
import argparse
from pathlib import Path
import json
import time
import numpy as np
from scipy.signal import lfilter
import torch
from audit_forecast_agreement import sha
from stochastic_price_forecast import load_timeline_split
from target_horizon_forecast import HorizonRange
from train_target_horizon_forecast import prepare, targets, write
from neural_stochastic_flow import CONTRACT, NeuralStochasticFlow, Bridge, flow_loss, load_nsf


def market_data(native, dataset, split, hrange):
    data = prepare(native, dataset, split, hrange)
    source = load_timeline_split(dataset, split)
    r = np.asarray(source.timeline[:, 0], dtype=float)
    # Only completed returns enter this state. Initialization uses the first past return.
    decay = np.exp(-np.log(2)/300)
    variance, _ = lfilter([1-decay], [1, -decay], r*r, zi=[decay*max(r[0]**2, 1e-12)])
    vol = .5*np.log(np.maximum(variance, 1e-12)/1e-8)
    first = int(data['origins'][0])
    data['vol'] = vol[first:first+len(data['prices'])]
    data['state'] = np.column_stack((np.zeros(len(data['origins'])), data['vol'][data['indices']])).astype(np.float32)
    return data


def target_states(data, ids, horizon):
    return np.column_stack((targets(data, horizon, ids)/.01,
        data['vol'][data['indices'][ids]+horizon])).astype(np.float32)


@torch.inference_mode()
def validate(flow, data, ids, h, device):
    total = 0.
    for start in range(0, len(ids), 2048):
        ii, hh = ids[start:start+2048], h[start:start+2048]
        x, f, y = (torch.from_numpy(a).to(device) for a in (
            data['state'][ii], data['features'][ii], target_states(data, ii, hh)))
        d = torch.from_numpy(hh.astype(np.float32)).to(device)/flow.time_scale
        total += float(-flow.log_prob(y, x, f, d, torch.zeros_like(d)).sum())
    return total/len(ids)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('data/backtests/oracle-neural-stochastic-flow'))
    parser.add_argument('--epochs', type=int, default=32)
    args = parser.parse_args()
    root = args.output
    if root.exists() and any(root.iterdir()):
        raise ValueError('refusing to overwrite experiment')
    root.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4); torch.set_float32_matmul_precision('high')
    device, hrange = torch.device('cuda'), HorizonRange()
    native = Path('data/backtests/oracle-expanded-mixture/direct59')
    nplan = json.loads((native/'plan.json').read_text())
    dataset = Path(json.loads(Path(nplan['nativePlan']).read_text())['datasetDir'])
    plan = {'contract': CONTRACT,
        'paper': 'https://proceedings.neurips.cc/paper_files/paper/2025/file/18abbeef8cfe9203fdf9053c9c4fe191-Paper-Conference.pdf',
        'architecture': 'Eq5 Gaussian base + four alternating affine couplings Eq7; width64 two-layer SiLU conditioners; bounded scale and shift outputs',
        'state': '[log price displacement / .01, log(300s-half-life causal EWMA volatility / .0001)]',
        'context': '118 frozen direct59 native forecast features known at origin; held fixed across hypothetical transitions',
        'symmetries': 'autonomous conditional on fixed context, translation equivariant in log price',
        'time': 'seconds / 21600 in physical drift/diffusion/coupling factors; tanh(seconds/21600) as neural input',
        'queryRangeSeconds': [0, 86400], 'supportsNonIntegerQueryHorizons': True,
        'sampling': 'integer supervised horizon draws proportional to integrated tanh derivative on 1..86400',
        'loss': 'joint state negative log density plus lambda times both bridge variational KL bounds Eq8-10; no added MSE term',
        'arms': {'nsfNoFlowLoss': 0., 'nsfFlowLoss': .1},
        'bridge': 'conditional Gaussian Brownian interpolation base and four affine coupling layers; one inner update per main update',
        'flowBatchSize': 128, 'tripletSplit': 'independent uniform fraction in [0.1,0.9] of supervised horizon; continuous times',
        'epochs': args.epochs, 'pairsPerEpoch': 65536, 'batchSize': 2048, 'learningRate': .0003,
        'patience': 10, 'minimumEpochs': 16, 'seed': 20260912,
        'selection': 'minimum joint-state NLL on fixed horizon-sampled validation pairs for each arm',
        'observation': 'NSF has continuous joint density; price samples optionally rounded to .01 ticks. No extra learned zero atom in this first NSF screen.',
        'nativeDirectory': str(native), 'nativePlanSha256': sha(native/'plan.json'), 'dataset': str(dataset),
        'h1ParentCheckpointSha256': sha(native/'head.pt'), 'onlineCalibration': 'none',
        'sourceHashes': {f: sha(f) for f in [__file__, 'ml/neural_stochastic_flow.py', 'ml/train_target_horizon_forecast.py']}}
    write(root, 'plan.json', plan)
    train, val = (market_data(native, dataset, split, hrange) for split in ('train', 'validation'))
    if train['origins'][0] <= nplan['nativeTrainingLastTargetOrigin']:
        raise ValueError('native training overlap')
    mu = torch.from_numpy(train['features'].mean(0, dtype=float).astype(np.float32))
    std = torch.from_numpy(np.maximum(train['features'].std(0, dtype=float), 1e-8).astype(np.float32))
    vi = np.tile(np.arange(len(val['state'])), 2)
    vh = hrange.sample(np.random.default_rng(20260913), len(vi))
    np.savez_compressed(root/'validation-pairs.npz', ids=vi, horizons=vh)
    selection = {'planSha256': sha(root/'plan.json'), 'models': {},
        'trainOrigins': len(train['state']), 'validationOrigins': len(val['state']), 'testUsedForSelection': False}
    start_time = time.monotonic()
    for name, weight in plan['arms'].items():
        torch.manual_seed(20260912)
        flow, bridge = NeuralStochasticFlow(mu, std).to(device), Bridge(mu, std).to(device)
        optimizer = torch.optim.AdamW(flow.parameters(), lr=.0003, weight_decay=1e-4)
        bridge_optimizer = torch.optim.AdamW(bridge.parameters(), lr=.0003, weight_decay=1e-4)
        history, best, best_epoch = [], float('inf'), -1
        for epoch in range(args.epochs+1):
            nll_sum = reg_sum = 0.
            if epoch:
                rng = np.random.default_rng(20260912+epoch)
                ids = rng.integers(0, len(train['state']), plan['pairsPerEpoch'])
                h = hrange.sample(rng, len(ids))
                y = target_states(train, ids, h)
                fractions = rng.uniform(.1, .9, len(ids)).astype(np.float32)
                flow.train(); bridge.train()
                for start in range(0, len(ids), 2048):
                    sl = slice(start, start+2048); ii = ids[sl]
                    x, f, target = (torch.from_numpy(a).to(device) for a in (train['state'][ii], train['features'][ii], y[sl]))
                    d = torch.from_numpy(h[sl].astype(np.float32)).to(device)/flow.time_scale
                    s = torch.zeros_like(d)
                    small = min(plan['flowBatchSize'], len(x))
                    d1 = d[:small]*torch.from_numpy(fractions[sl][:small]).to(device)
                    d2 = d[:small]-d1
                    if weight:
                        flow.requires_grad_(False); bridge.requires_grad_(True)
                        bridge_optimizer.zero_grad(set_to_none=True)
                        inner = flow_loss(flow, bridge, x[:small], f[:small], d1, d2, s[:small])
                        if not bool(torch.isfinite(inner)):
                            raise ValueError('nonfinite bridge loss')
                        inner.backward(); torch.nn.utils.clip_grad_norm_(bridge.parameters(), 10); bridge_optimizer.step()
                        flow.requires_grad_(True); bridge.requires_grad_(False)
                    optimizer.zero_grad(set_to_none=True)
                    nll = -flow.log_prob(target, x, f, d, s).mean()
                    reg = flow_loss(flow, bridge, x[:small], f[:small], d1, d2, s[:small]) if weight else nll.new_zeros(())
                    loss = nll+weight*reg
                    if not bool(torch.isfinite(loss)):
                        raise ValueError('nonfinite main loss')
                    loss.backward(); torch.nn.utils.clip_grad_norm_(flow.parameters(), 10); optimizer.step()
                    nll_sum += float(nll.detach())*len(x); reg_sum += float(reg.detach())*len(x)
            flow.eval()
            score = validate(flow, val, vi, vh, device)
            row = {'epoch': epoch, 'validationJointNll': score,
                'trainingJointNll': nll_sum/plan['pairsPerEpoch'] if epoch else None,
                'trainingFlowBound': reg_sum/plan['pairsPerEpoch'] if epoch else None,
                'seconds': time.monotonic()-start_time}
            history.append(row)
            if score < best:
                best, best_epoch = score, epoch
                torch.save({'contract': CONTRACT, 'model': flow.state_dict(), 'bridge': bridge.state_dict(),
                    'featureMean': mu, 'featureStd': std, 'width': 64, 'timeScale': flow.time_scale, 'priceScale': flow.price_scale,
                    'epoch': epoch, 'validationJointNll': score, 'planSha256': selection['planSha256'], 'arm': name}, root/f'{name}.pt')
            write(root, name+'-history.json', history)
            print(json.dumps({'model': name, **row, 'bestEpoch': best_epoch}), flush=True)
            if epoch >= 16 and epoch-best_epoch >= 10:
                break
        selection['models'][name] = {'epoch': best_epoch, 'validationJointNll': best, 'checkpointSha256': sha(root/f'{name}.pt')}
    selection['seconds'] = time.monotonic()-start_time
    write(root, 'selection.json', selection)
    print(json.dumps(selection), flush=True)


if __name__ == '__main__':
    main()
