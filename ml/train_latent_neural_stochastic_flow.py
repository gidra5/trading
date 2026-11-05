"""Train causal latent NSF with reconstruction, transition KL, skip KL and flow loss."""
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
from target_horizon_forecast import HorizonRange
from train_target_horizon_forecast import prepare, write, targets
from stochastic_price_forecast import load_timeline_split
from export_stochastic_price_forecast import load_close_prices
from latent_neural_stochastic_flow import CONTRACT, LatentNSF, LatentBridge, flow_loss


def latent_data(native, dataset, split, hrange):
    data = prepare(native, dataset, split, hrange)
    source = load_timeline_split(dataset, split)
    first = int(data['origins'][0])
    previous = load_close_prices(np.array([source.axis_start_ms+(first-1)*1000]),
        Path('data/market/immutable/refs/candles/spot-btcusdt/btcusdt/1s'))
    data['pricesWithPast'] = np.r_[previous, data['prices']]
    data.pop('features')  # Native predictions are not model inputs.
    data['timeline'] = source.timeline
    return data


def sequence(data, ids, horizons):
    origins = data['origins'][ids]
    h = np.broadcast_to(horizons, origins.shape)
    future = h[:, None]*np.arange(1, 7)[None, :]//6
    offsets = np.column_stack((-np.ones(len(ids), dtype=int), np.zeros(len(ids), dtype=int), future))
    index = origins[:, None]+offsets
    features = np.asarray(data['timeline'][index], dtype=np.float32)
    price_index = data['indices'][ids, None]+1+offsets
    prices = data['pricesWithPast'][price_index]
    elapsed = np.column_stack((np.zeros(len(ids)), np.diff(offsets, axis=1))).astype(np.float32)
    timestamps = data['timestamps'][ids, None].astype(np.float64)/1000+offsets
    return features, prices, elapsed, timestamps


@torch.inference_mode()
def validation_score(model, data, ids, horizons, device):
    # A proper marginal score avoids selecting a good reconstruction but bad forecast.
    values = []
    for start in range(0, len(ids), 64):
        ii, h = ids[start:start+64], horizons[start:start+64]
        f, prices, elapsed, timestamps = sequence(data, ii, h)
        law = model.distribution(torch.from_numpy(f[:, :2]).to(device), torch.from_numpy(prices[:, :2]).to(device),
            torch.from_numpy(elapsed[:, :2]).to(device), torch.from_numpy(h.astype(np.float32)).to(device), timestamps_seconds=torch.from_numpy(timestamps[:, :2]).to(device), components=64, seed=11)
        draw = torch.log(law.sample_prices(128, seed=31)/law.current_price[:, None])
        actual = torch.from_numpy(np.log(prices[:, -1]/prices[:, 1])).to(device)
        ordered = draw.sort(1).values; n = draw.shape[1]
        w = (2*torch.arange(1, n+1, device=device, dtype=draw.dtype)-n-1)/n**2
        crps = (draw-actual[:, None]).abs().mean(1)-ordered@w
        values.append((crps/torch.from_numpy(h.astype(float)).to(device).sqrt()).cpu().numpy())
    return float(np.concatenate(values).mean()*1e4)


def sample_horizons(rng, count):
    return rng.integers(1, 86401, size=count)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('data/backtests/oracle-latent-nsf-sampled-skips'))
    parser.add_argument('--epochs', type=int, default=32)
    args = parser.parse_args()
    root = args.output
    if root.exists() and any(root.iterdir()):
        raise ValueError('refusing to overwrite experiment')
    root.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda')
    torch.set_num_threads(4); torch.set_float32_matmul_precision('high')
    native = Path('data/backtests/oracle-expanded-mixture/direct59')
    nplan = json.loads((native/'plan.json').read_text())
    dataset = Path(json.loads(Path(nplan['nativePlan']).read_text())['datasetDir'])
    plan = {'contract': CONTRACT,
        'paper': 'https://proceedings.neurips.cc/paper_files/paper/2025/file/18abbeef8cfe9203fdf9053c9c4fe191-Paper-Conference.pdf',
        'architecture': 'causal GRU256 Gaussian encoder; 256D latent NSF, four affine couplings; two-layer256 Gaussian observation decoder',
        'observations': '59 observed production features plus log price relative to the known origin price / .01',
        'history': 'two completed observed states, one second apart; train posterior additionally assimilates six future observations at floor(k*h/6), k=1..6',
        'latentInitialization': 'learnable price skip connection in posterior and decoder; remaining 255 coordinates learned; all posterior variances stochastic',
        'nativeModelUse': 'only its archived origins choose matched split pools; no frozen native context enters latent transitions',
        'time': 'physical d=seconds/21600 in NSF; neural [d,tanh(d),log1p(seconds)]; UTC daily/weekly sine/cosine at start and endpoint; float64 Unix timestamps; fractional queries [0,86400]',
        'sampling': 'uniform integer horizons 1..86400 inclusive; eight-observation windows [-1,0,floor(h/6),...,h]; duplicate times skipped causally',
        'arms': {'latentAdjacent': {'beta': .1, 'betaSkip': 0., 'flowWeight': 0.},
            'latentSampledSkipFlow': {'beta': .1, 'betaSkip': .1, 'flowWeight': .01}},
        'loss': 'time-averaged observation Gaussian NLL + beta adjacent posterior-to-prior KL + betaSkip mean-over-eligible-sources sampled skip KL + bidirectional flow bound',
        'bridge': '256D conditional flow with Brownian interpolation base; one inner bridge update per main update',
        'skipSampling': 'one independent uniform target per eligible source; 2..min(4,remaining) distinct observation intervals ahead; boundary-truncated source windows; averaged over eligible sources per sequence',
        'maximumSkipIntervals': 4, 'sequenceObservations': 8, 'dimensions': 256, 'width': 256, 'epochs': args.epochs, 'sequencesPerEpoch': 32768, 'batchSize': 1024, 'flowBatchSize': 64,
        'learningRate': .0003, 'patience': 10, 'minimumEpochs': 16, 'seed': 20260912,
        'selection': 'fixed 2048 validation origin/horizon pairs; 64 latent components, 128 observation draws; minimize mean CRPS(log return)/sqrt(h)',
        'prediction': 'sample current posterior, sample requested latent transition, integrate Gaussian price decoder as a mixture',
        'observationZero': 'no explicit zero atom; quantize Gaussian mixture price law to .01 ticks',
        'dataset': str(dataset), 'nativeDirectory': str(native), 'onlineCalibration': 'none',
        'existingParentSha256': sha(native/'head.pt'),
        'sourceHashes': {f: sha(f) for f in [__file__, 'ml/latent_neural_stochastic_flow.py', 'ml/neural_stochastic_flow.py', 'ml/train_target_horizon_forecast.py']}}
    write(root, 'plan.json', plan)
    train, val = (latent_data(native, dataset, split, HorizonRange()) for split in ('train', 'validation'))
    raw = np.asarray(train['timeline'][train['origins']], dtype=np.float32)
    mu = torch.from_numpy(raw.mean(0, dtype=float).astype(np.float32))
    std = torch.from_numpy(np.maximum(raw.std(0, dtype=float), 1e-8).astype(np.float32))
    vi = np.linspace(0, len(val['origins'])-1, 2048, dtype=int)
    vh = sample_horizons(np.random.default_rng(20260913), len(vi))
    np.savez_compressed(root/'validation-selection-pairs.npz', ids=vi, horizons=vh)
    selection = {'planSha256': sha(root/'plan.json'), 'models': {}, 'testUsedForSelection': False,
        'trainOrigins': len(train['origins']), 'validationOrigins': len(val['origins'])}
    begun = time.monotonic()
    for name, weights in plan['arms'].items():
        torch.manual_seed(20260912)
        model, bridge = LatentNSF(mu, std).to(device), LatentBridge().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=1e-4)
        bridge_optimizer = torch.optim.AdamW(bridge.parameters(), lr=.0003, weight_decay=1e-4)
        best, best_epoch, history = float('inf'), -1, []
        for epoch in range(args.epochs+1):
            totals = {key: 0. for key in ['reconstruction', 'adjacentKl', 'skipKl', 'flowBound']}
            if epoch:
                rng = np.random.default_rng(20260912+epoch)
                ids = rng.integers(0, len(train['origins']), plan['sequencesPerEpoch'])
                h = sample_horizons(rng, len(ids))
                f, price, elapsed, timestamps = sequence(train, ids, h)
                model.train()
                for start in range(0, len(ids), plan['batchSize']):
                    sl = slice(start, start+plan['batchSize'])
                    features, seconds = torch.from_numpy(f[sl]).to(device), torch.from_numpy(elapsed[sl]).to(device)
                    ret = torch.from_numpy(np.log(price[sl]/price[sl, 1, None])).to(device)
                    obs = model.observations(features, ret)
                    small = min(plan['flowBatchSize'], len(obs))
                    d = seconds[:small, 2:].sum(1)/model.time_scale
                    empty = obs.new_empty((small, 0))
                    times = torch.from_numpy(timestamps[sl]).to(device)
                    origin_time = times[:small, 1]/model.time_scale
                    if weights['flowWeight']:
                        with torch.no_grad():
                            qmu, qstd = model.encode(obs[:small, :2], seconds[:small, :2], times[:small, :2])
                            initial = qmu[:, -1]+qstd[:, -1]*torch.randn_like(qmu[:, -1])
                        model.transition.requires_grad_(False); bridge.requires_grad_(True)
                        bridge_optimizer.zero_grad(set_to_none=True)
                        inner = flow_loss(model.transition, bridge, initial, empty, d*.5, d*.5, origin_time)
                        if not torch.isfinite(inner):
                            raise ValueError('nonfinite latent bridge loss')
                        inner.backward(); torch.nn.utils.clip_grad_norm_(bridge.parameters(), 10); bridge_optimizer.step()
                        model.transition.requires_grad_(True); bridge.requires_grad_(False)
                    optimizer.zero_grad(set_to_none=True)
                    loss, terms, current = model.variational_loss(obs, seconds, times, weights['beta'], weights['betaSkip'], plan['maximumSkipIntervals'])
                    reg = flow_loss(model.transition, bridge, current[:small], empty, d*.5, d*.5, origin_time) if weights['flowWeight'] else loss.new_zeros(())
                    loss = loss+weights['flowWeight']*reg
                    if not bool(torch.isfinite(loss)):
                        raise ValueError('nonfinite latent objective')
                    loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 10); optimizer.step()
                    for key, value in {**terms, 'flowBound': reg}.items():
                        totals[key] += float(value.detach())*len(obs)/len(ids)
            model.eval()
            score = validation_score(model, val, vi, vh, device)
            row = {'epoch': epoch, 'validationScaledCrpsBps': score, 'trainingTerms': totals if epoch else None,
                'seconds': time.monotonic()-begun}
            history.append(row)
            if score < best:
                best, best_epoch = score, epoch
                torch.save({'contract': CONTRACT, 'model': model.state_dict(), 'bridge': bridge.state_dict(),
                    'featureMean': mu, 'featureStd': std, 'dimensions': model.dimensions, 'width': model.width, 'epoch': epoch,
                    'validationScaledCrpsBps': score, 'planSha256': selection['planSha256']}, root/f'{name}.pt')
            if epoch == 1:
                torch.save({'contract': CONTRACT, 'model': model.state_dict(), 'featureMean': mu, 'featureStd': std,
                    'dimensions': model.dimensions, 'width': model.width, 'epoch': epoch, 'diagnosticOnly': True,
                    'planSha256': selection['planSha256']}, root/f'{name}-epoch1-diagnostic.pt')
            write(root, name+'-history.json', history)
            print(json.dumps({'model': name, **row, 'bestEpoch': best_epoch}), flush=True)
            if epoch >= 16 and epoch-best_epoch >= 10:
                break
        selection['models'][name] = {'epoch': best_epoch, 'validationScaledCrpsBps': best, 'checkpointSha256': sha(root/f'{name}.pt')}
    selection['seconds'] = time.monotonic()-begun
    write(root, 'selection.json', selection)
    print(json.dumps(selection), flush=True)


if __name__ == '__main__':
    main()
