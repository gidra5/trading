"""Fit the direct knot marginal on a frozen, previously selected latent NSF.

Observed tick likelihood trains the new head. A finite CDF projection penalty
ties it to sampled marginals of the retained dynamics; it is not an exact
Chapman--Kolmogorov guarantee. No latent samples are needed at inference.
"""
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
from direct_latent_nsf import CONTRACT, DirectLatentNSF, load_direct_latent_nsf
from horizon_price_forecast import fit_horizon_basis
from latent_neural_stochastic_flow import load_latent_nsf
from target_horizon_forecast import HorizonRange
from train_latent_neural_stochastic_flow import latent_data
from train_target_horizon_forecast import targets, write

ANCHORS = np.array([1, 5, 15, 60, 300, 900, 3600, 7200, 21600, 86400])


def sample_horizons(rng, count):
    # Full-range coverage plus explicit short-horizon training. This is not
    # the old tanh-derivative sampling rule.
    h = rng.integers(1, 86401, count)
    use_anchor = rng.random(count) < .5
    h[use_anchor] = rng.choice(ANCHORS, use_anchor.sum())
    return h


def history(data, ids, device):
    """Read only t-1 and t; future features never enter this input path."""
    offsets = np.array([-1, 0])
    f = np.asarray(data['timeline'][data['origins'][ids, None]+offsets], dtype=np.float32)
    p = data['pricesWithPast'][data['indices'][ids, None]+1+offsets]
    dt = np.broadcast_to(np.array([0., 1.], dtype=np.float32), p.shape).copy()
    times = data['timestamps'][ids, None].astype(np.float64)/1000+offsets
    return tuple(torch.from_numpy(x).to(device) for x in (f, p, dt, times))


@torch.inference_mode()
def encode_data(nsf, data, device):
    means, scales = [], []
    for start in range(0, len(data['origins']), 1024):
        ids = np.arange(start, min(start+1024, len(data['origins'])))
        f, p, dt, times = history(data, ids, device)
        obs = nsf.observations(f, (p/p[:, -1:]).log())
        mu, std = nsf.encode(obs, dt, times)
        means.append(mu[:, -1].cpu()); scales.append(std[:, -1].cpu())
    # Tensors created outside inference mode can be saved for head backward.
    return torch.cat(means).cpu().numpy(), torch.cat(scales).cpu().numpy()


def posterior_tensors(encoded, data, device):
    return tuple(torch.as_tensor(x, device=device) for x in (*encoded,
        data['prices'][data['indices']], data['timestamps'].astype(np.float64)/1000))


def law_at(model, posterior, ids, horizons):
    mu, std, price, times = (x[ids] for x in posterior)
    return model.distribution_from_posterior(mu, std, price, horizons, times)


@torch.inference_mode()
def evaluate_nll(model, posterior, data, ids, horizons):
    ll = []
    for start in range(0, len(ids), 512):
        ii, h = ids[start:start+512], horizons[start:start+512]
        law = law_at(model, posterior, ii, h)
        actual = torch.as_tensor(data['prices'][data['indices'][ii]+h], device=law.current_price.device)
        ll.append(law.tick_probability(actual).clamp_min(1e-300).log().cpu().numpy())
    return float(-np.concatenate(ll).mean())


@torch.inference_mode()
def teacher_bank(nsf, data, ids, horizons, sigma, device):
    # Endpoints and half-tick boundaries are known numeric queries; all
    # predictions still use only the corresponding origin's observed history.
    grid = torch.linspace(-4, 4, 33, device=device, dtype=torch.float64)
    boundaries, cdfs = [], []
    for start in range(0, len(ids), 32):
        ii, h = ids[start:start+32], horizons[start:start+32]
        f, p, dt, times = history(data, ii, device)
        law = nsf.distribution(f, p, dt, torch.as_tensor(h, device=device), times, components=64, seed=73)
        r = grid[None]*sigma*torch.as_tensor(h, device=device).double().sqrt()[:, None]
        prices = law.current_price[:, None]*r.exp()
        prices = torch.cat((prices, law.current_price[:, None]), 1)
        b = ((prices/.01+1e-8).floor()+.5)*.01
        y = (b/law.current_price[:, None]).log()
        cdf = torch.special.ndtr((y[:, :, None]-law.log_means[:, None])/law.log_stds[:, None]).mean(-1)
        boundaries.append(prices.cpu()); cdfs.append(cdf.cpu())
    return torch.cat(boundaries).numpy(), torch.cat(cdfs).numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--parent', type=Path, default=Path('data/backtests/oracle-latent-nsf-sampled-skips/latentSampledSkipFlow.pt'))
    parser.add_argument('--output', type=Path, default=Path('data/backtests/oracle-direct-latent-nsf-20260921'))
    parser.add_argument('--epochs', type=int, default=32)
    parser.add_argument('--examples-per-epoch', type=int, default=32768)
    parser.add_argument('--teacher-examples', type=int, default=2048)
    parser.add_argument('--consistency-weight', type=float, default=1.)
    args = parser.parse_args()
    if args.epochs < 1 or args.examples_per_epoch < 1 or args.teacher_examples < 1 \
            or not np.isfinite(args.consistency_weight) or args.consistency_weight < 0:
        raise ValueError('positive training sizes and nonnegative consistency weight required')
    root = args.output
    if root.exists() and any(root.iterdir()):
        raise ValueError('refusing to overwrite an experiment')
    root.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4); torch.manual_seed(20260921)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    nsf, saved = load_latent_nsf(args.parent, device)
    parent_plan_file = args.parent.parent/'plan.json'
    if sha(parent_plan_file) != saved['planSha256']:
        raise ValueError('parent training plan changed')
    parent_plan = json.loads(parent_plan_file.read_text())
    for path, expected in parent_plan['sourceHashes'].items():
        if sha(path) != expected:
            raise ValueError(f'parent source changed: {path}')
    nsf.requires_grad_(False)
    native, dataset = Path(parent_plan['nativeDirectory']), Path(parent_plan['dataset'])
    plan = {'contract': CONTRACT, 'parent': str(args.parent), 'parentSha256': sha(args.parent),
        'parentPlanSha256': sha(parent_plan_file), 'dataset': str(dataset), 'nativeDirectory': str(native),
        'inputs': 'full frozen Gaussian latent posterior mean/std, horizon [d,tanh(d),log1p(seconds)], start/end daily/weekly clocks',
        'output': 'one zero logit and 256 requested Gaussian-coordinate triangular knot logits; deduplicated train-fitted knots',
        'scale': 'log return / sqrt(h); invertible coordinate, no Brownian independence assumption',
        'sampling': 'half uniform integer seconds 1..86400; half uniform anchor choice 1,5,15,60,300,900,3600,7200,21600,86400',
        'loss': 'observed .01-tick NLL + weight times mean squared CDF discrepancy from frozen NSF on 34 boundaries',
        'consistencyWeight': args.consistency_weight, 'teacherExamples': args.teacher_examples,
        'teacherLatentComponents': 64, 'teacherCdfBoundaries': 34,
        'consistencyScope': 'sampled marginal projection regularizer; no exact latent-marginal equality or flow consistency theorem',
        'latentFrozen': True, 'onlineCalibration': 'none', 'epochs': args.epochs,
        'examplesPerEpoch': args.examples_per_epoch, 'batchSize': 512, 'seed': 20260921,
        'learningRate': .001, 'selection': 'lowest observed tick NLL on fixed 4096 validation pairs; no test selection',
        'sourceHashes': {f: sha(f) for f in [__file__, 'ml/direct_latent_nsf.py',
            'ml/knot_price_distribution.py', 'ml/horizon_price_forecast.py', 'ml/stochastic_price_forecast.py',
            'ml/normalized_glu_next_return.py', 'ml/train_latent_neural_stochastic_flow.py']}}
    write(root, 'plan.json', plan)
    begun = time.monotonic()
    train, val = (latent_data(native, dataset, s, HorizonRange()) for s in ('train', 'validation'))
    encoded_train, encoded_val = (encode_data(nsf, d, device) for d in (train, val))
    posterior_train = posterior_tensors(encoded_train, train, device)
    posterior_val = posterior_tensors(encoded_val, val, device)
    rng = np.random.default_rng(20260921)
    basis_h = sample_horizons(rng, len(train['origins']))
    basis = fit_horizon_basis(targets(train, basis_h)/np.sqrt(basis_h))
    x = np.concatenate(encoded_train, 1)
    input_mean = torch.from_numpy(np.r_[x.mean(0, dtype=float), np.zeros(11)].astype(np.float32))
    input_std = torch.from_numpy(np.r_[np.maximum(x.std(0, dtype=float), 1e-6), np.ones(11)].astype(np.float32))
    model = DirectLatentNSF(nsf, basis, input_mean, input_std).to(device)
    teacher_ids = rng.integers(0, len(train['origins']), args.teacher_examples)
    teacher_h = sample_horizons(rng, len(teacher_ids))
    prices, cdfs = teacher_bank(nsf, train, teacher_ids, teacher_h, model.head.sigma, device)
    teacher_prices, teacher_cdfs = (torch.as_tensor(v, device=device) for v in (prices, cdfs))
    np.savez_compressed(root/'teacher-pairs.npz', ids=teacher_ids, horizons=teacher_h, prices=prices, cdfs=cdfs)
    vi = np.linspace(0, len(val['origins'])-1, 4096, dtype=int)
    vh = sample_horizons(np.random.default_rng(20260922), len(vi))
    np.savez_compressed(root/'validation-pairs.npz', ids=vi, horizons=vh)
    print(json.dumps({'phase': 'prepared', 'trainOrigins': len(x), 'knots': len(basis['knots']),
                      'seconds': time.monotonic()-begun}), flush=True)
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=.001, weight_decay=1e-4)
    best, best_epoch, rows = float('inf'), 0, []
    for epoch in range(args.epochs+1):
        nll_sum = cdf_sum = 0.
        if epoch:
            model.head.train()
            erng = np.random.default_rng(20260921+epoch)
            ids = erng.integers(0, len(x), args.examples_per_epoch)
            horizons = sample_horizons(erng, len(ids))
            for start in range(0, len(ids), 512):
                ii, h = ids[start:start+512], horizons[start:start+512]
                law = law_at(model, posterior_train, ii, h)
                actual = torch.as_tensor(train['prices'][train['indices'][ii]+h], device=device)
                nll = -law.tick_probability(actual).clamp_min(1e-300).log().mean()
                jj = erng.integers(0, len(teacher_ids), min(128, len(ii)))
                projected = law_at(model, posterior_train, teacher_ids[jj], teacher_h[jj])
                discrepancy = (projected.cdf(teacher_prices[jj])-teacher_cdfs[jj]).square().mean()
                loss = nll+args.consistency_weight*discrepancy
                if not bool(torch.isfinite(loss)):
                    raise ValueError('nonfinite direct-head loss')
                optimizer.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.head.parameters(), 10.)
                optimizer.step()
                nll_sum += float(nll.detach())*len(ii)/len(ids)
                cdf_sum += float(discrepancy.detach())*len(ii)/len(ids)
        model.eval()
        score = evaluate_nll(model, posterior_val, val, vi, vh)
        row = {'epoch': epoch, 'validationTickNll': score, 'trainingTickNll': nll_sum if epoch else None,
               'trainingProjectionCdfMse': cdf_sum if epoch else None, 'seconds': time.monotonic()-begun}
        rows.append(row)
        if score < best:
            best, best_epoch = score, epoch
            torch.save({**model.checkpoint(), 'epoch': epoch, 'validationTickNll': score,
                        'parentSha256': plan['parentSha256'], 'planSha256': sha(root/'plan.json')}, root/'model.pt')
        write(root, 'history.json', rows)
        print(json.dumps({**row, 'bestEpoch': best_epoch}), flush=True)
        if epoch >= 16 and epoch-best_epoch >= 10:
            break
    retained, _ = load_direct_latent_nsf(root/'model.pt', device)
    # Frozen dynamics must be literally preserved in the standalone artifact.
    if any(not torch.equal(value, retained.nsf.state_dict()[key]) for key, value in nsf.state_dict().items()):
        raise ValueError('direct checkpoint changed latent NSF dynamics')
    if sha(args.parent) != plan['parentSha256']:
        raise ValueError('parent checkpoint changed during training')
    write(root, 'selection.json', {'bestEpoch': best_epoch, 'validationTickNll': best,
        'checkpointSha256': sha(root/'model.pt'), 'parentSha256': plan['parentSha256'],
        'latentDynamicsExactlyPreserved': True, 'testUsedForSelection': False,
        'trainOrigins': len(x), 'validationOrigins': len(val['origins']), 'seconds': time.monotonic()-begun})


if __name__ == '__main__':
    main()
