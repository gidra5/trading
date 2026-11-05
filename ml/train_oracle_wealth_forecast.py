"""Forecast NLL/log-MSE plus realized Oracle episode wealth, gamma zero.

Research controller only: continuous long/cash, no limit orders, unit funding.
Compare forecast-only, frozen-forecast actor training, and joint training.
Checkpoints use validation first half; arms use second half; test loaded last.
"""
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
from scipy.special import ndtr
import torch
from audit_forecast_agreement import sha, scores
from export_stochastic_price_forecast import load_close_prices
from horizon_price_forecast import load_horizon_forecast, CONTRACT
from stochastic_price_forecast import component_log_masses
from oracle_wealth_training import DistributionController, TrainingRules, rollout_torch, rollout_numpy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('data/backtests/oracle-wealth-training'))
    parser.add_argument('--epochs', type=int, default=12)
    args = parser.parse_args()
    root = args.output; root.mkdir(parents=True, exist_ok=True)
    if (root/'evaluation.json').exists():
        raise ValueError('completed experiment exists')
    torch.set_num_threads(4); torch.set_float32_matmul_precision('high')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sources = {}
    def checked(file):
        file = Path(file); sources[str(file)] = sha(file); return file
    def write(file, obj):
        (root/file).write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')
    for file in [__file__, 'ml/oracle_wealth_training.py', 'ml/continuous_oracle_endpoint.py']:
        checked(file)
    archive = Path('data/backtests/oracle-expanded-mixture/direct59')
    checkpoint = checked(archive/'head.pt')
    base, payload = load_horizon_forecast(checkpoint, device)
    original = {k: v.detach().cpu().clone() for k, v in base.state_dict().items()}
    rules = TrainingRules()
    length, batch_size = 256, 64
    arms = [{'name': 'forecastOnly', 'weight': 0., 'freezeForecast': False},
            {'name': 'actorOnly', 'weight': 1., 'freezeForecast': True},
            *[{'name': 'joint'+str(w), 'weight': w, 'freezeForecast': False} for w in (.1, 1., 10.)]]
    plan = {'contract': 'oracle-realized-wealth-forecast-training-v1', 'gamma': 0,
        'forecast': 'direct59 frozen native feature predictor; fine-tune its full Gaussian-knot/zero-atom head',
        'controller': 'learned continuous long/cash target from full predicted masses, current exposure/equity and remaining horizon',
        'controllerOptimality': 'approximate learned controller; not the noncomputable Lean Bellman selector',
        'loss': 'NLL + log-return mean MSE/train second moment - lambda * mean(realized terminal wealth/initial - 1)*10000/10',
        'rewardScaleBps': 10., 'arms': arms, 'epochs': args.epochs, 'episodeDecisions': length,
        'batchEpisodes': batch_size, 'cadenceSeconds': 4, 'forecastHorizonSeconds': 1,
        'headLearningRate': .0001, 'controllerLearningRate': .01, 'seed': 20260912,
        'executionRules': asdict(rules), 'entryBand': [0, 1], 'maintenanceBand': [0, 1],
        'fundingFactor': 1, 'pendingOrders': 0,
        'episodeTiming': '256 issues four seconds apart; one final close one second after last issue; holdings retained between all issues',
        'gradients': 'exact piecewise autograd for declared account execution; no straight-through gradient at order-size gaps',
        'checkpointSelection': 'minimum own combined loss on first chronological validation half; include epoch0',
        'armSelection': 'maximum mean episode terminal wealth on second validation half; cash included, then lower forecast loss for ties',
        'historicalTestPreviouslyInspected': True, 'baseCheckpointUsedValidation': True,
        'testUsedForSelection': False, 'minimumActivityPenalty': False,
        'onlineCalibration': 'none added; strongest standalone refit is uncalibrated, as in previous evaluation',
        'fullSplitReplay': 'separate diagnostic: one cash start and one close at the end, no resets',
        'sourceCheckpoint': str(checkpoint), 'sourceCheckpointSha256': sources[str(checkpoint)]}
    write('plan.json', plan)
    def load(split):
        with np.load(checked(archive/f'{split}-native.npz')) as f:
            d = {k: f[k] for k in f.files}
        if np.any(np.diff(d['timestamps']) != 4000):
            raise ValueError('inconsistent forecast cadence')
        cache = root/f'{split}-prices.npz'
        if cache.exists():
            with np.load(cache) as f:
                np.testing.assert_array_equal(f['timestamps'], d['timestamps'])
                prices = f['prices']
        else:
            times = np.arange(d['timestamps'][0], d['timestamps'][-1]+1001, 1000)
            prices = load_close_prices(times, Path('data/market/immutable/refs/candles/spot-btcusdt/btcusdt/1s'))
            np.savez_compressed(cache, prices=prices, timestamps=d['timestamps'])
        checked(cache)
        if np.any(prices <= 0) or np.max(abs(prices/.01-np.rint(prices/.01))) > 1e-6:
            raise ValueError('invalid market price grid')
        offsets = np.arange(len(d['y']))*4
        np.testing.assert_allclose(np.log1p(prices[offsets+1]/prices[offsets]-1), d['y'], atol=0, rtol=0)
        d['price'], d['terminalPrice'] = prices[offsets], prices[offsets+1]
        d['tape'] = prices
        print(json.dumps({'event': 'loaded', 'split': split, 'rows': len(d['y'])}), flush=True)
        return d
    train, val = load('train'), load('validation')
    half = len(val['y'])//2
    if train['timestamps'][-1]+1000 >= val['timestamps'][0] or val['timestamps'][half-1]+1000 >= val['timestamps'][half]:
        raise ValueError('split target leakage')
    norm = float(np.mean(train['y']**2))
    def episodes(data, lo=0, hi=None):
        hi = len(data['y']) if hi is None else hi
        count = (hi-lo)//length
        ids = np.arange(lo, lo+count*length).reshape(count, length)
        return {'x': torch.from_numpy(data['features'][ids]).to(device),
                'y': torch.from_numpy(data['y'][ids].astype(np.float32)).to(device),
                'prices': torch.from_numpy(np.column_stack((data['price'][ids], data['terminalPrice'][ids[:, -1]]))).double().to(device),
                'ids': ids}
    training = episodes(train); selecting = episodes(val, 0, half); tuning = episodes(val, half)
    # Matched controller initialization for every arm and the frozen baseline.
    torch.manual_seed(20260912)
    actor = DistributionController(len(base.areas)).to(device)
    initial_actor = {k: v.detach().cpu().clone() for k, v in actor.state_dict().items()}
    def forward(model, controller, batch, ids=None):
        x, y, price = (batch[k] if ids is None else batch[k][ids] for k in ('x', 'y', 'prices'))
        b, n, features = x.shape
        logits = model(x.reshape(b*n, features))
        zero = torch.sigmoid(logits[:, 0])
        masses = component_log_masses(logits[:, 1:], model.areas).exp()
        static, weights = controller.coefficients(zero, masses)
        rollout = rollout_torch(static.reshape(b, n), weights, price, rules)
        nll = -model.log_prob_from_logits(logits, y.reshape(-1)).mean()
        mse = (model.expectation_from_logits(logits)-y.reshape(-1)).square().mean()/norm
        reward = (rollout['wealth']/rules.initial-1).mean()*10000/10
        return nll+mse, reward, rollout, nll, mse
    @torch.no_grad()
    def evaluate(model, controller, batch):
        count = len(batch['x']); totals = np.zeros(4); values = []
        for start in range(0, count, batch_size):
            ix = slice(start, min(start+batch_size, count))
            loss, reward, sim, nll, mse = forward(model, controller, batch, ix)
            weight = len(sim['wealth'])
            totals += np.array([float(loss), float(reward), float(nll), float(mse)])*weight/count
            values.extend(sim['wealth'].cpu().tolist())
        return {'forecastLoss': float(totals[0]), 'reward': float(totals[1]),
            'nll': float(totals[2]), 'normalizedLogMse': float(totals[3]),
            'meanEpisodeReturnBps': float(totals[1]*10), 'episodeWealth': values}
    # Use converged moment coefficients in both training and evaluation.
    for name, kind in [('component_means', 'log'), ('component_excess_means', 'excess')]:
        high, low = base._component_moments(kind, 768), base._component_moments(kind, 384)
        if np.max(abs(high-low)) > 2e-10:
            raise ValueError('moment quadrature did not converge')
        original[name] = torch.from_numpy(high).float()
    checkpoints = {}
    started = time.monotonic()
    for arm in [{'name': 'unchanged', 'weight': 0., 'freezeForecast': True}, *arms]:
        base.load_state_dict(original); base.requires_grad_(not arm['freezeForecast'])
        actor.load_state_dict(initial_actor)
        optimizer = torch.optim.Adam([{'params': actor.parameters(), 'lr': .01},
            {'params': [p for p in base.parameters() if p.requires_grad], 'lr': .0001}])
        best, best_epoch = float('inf'), -1
        history = []
        generator = torch.Generator(device=device).manual_seed(20260912)
        limit = 0 if arm['name'] == 'unchanged' else args.epochs
        for epoch in range(limit+1):
            if epoch:
                base.train(); actor.train()
                for ids in torch.randperm(len(training['x']), generator=generator, device=device).split(batch_size):
                    optimizer.zero_grad(set_to_none=True)
                    forecast, reward, _, _, _ = forward(base, actor, training, ids)
                    loss = forecast-arm['weight']*reward
                    if not loss.requires_grad:
                        raise ValueError('loss has no gradient')
                    loss.backward()
                    gradient = torch.nn.utils.clip_grad_norm_([*base.parameters(), *actor.parameters()], 10)
                    if not torch.isfinite(loss) or not torch.isfinite(gradient):
                        raise ValueError('nonfinite training loss/gradient')
                    optimizer.step()
            base.eval(); actor.eval()
            metric = evaluate(base, actor, selecting)
            objective = metric['forecastLoss']-arm['weight']*metric['reward']
            history.append({'epoch': epoch, 'objective': objective,
                **{k: v for k, v in metric.items() if k != 'episodeWealth'}})
            if objective < best:
                best, best_epoch = objective, epoch
                file = root/(arm['name']+'.pt')
                torch.save({'contract': 'oracle-wealth-forecast-controller-v1', 'headContract': CONTRACT,
                    'head': base.state_dict(), 'controller': actor.state_dict(), 'sourceCheckpoint': str(checkpoint),
                    'sourceCheckpointSha256': sources[str(checkpoint)], 'epoch': epoch, 'arm': arm}, file)
            print(json.dumps({'event': 'epoch', 'arm': arm['name'], 'epoch': epoch,
                'selectionObjective': objective, 'episodeReturnBps': metric['meanEpisodeReturnBps'],
                'bestEpoch': best_epoch, 'seconds': time.monotonic()-started}), flush=True)
            write(arm['name']+'-history.json', history)
        checkpoints[arm['name']] = {'file': str(root/(arm['name']+'.pt')), 'epoch': best_epoch,
                                   'objective': best, 'arm': arm}
    def restore(name):
        file = checked(checkpoints[name]['file'])
        saved = torch.load(file, map_location=device, weights_only=False)
        base.load_state_dict(saved['head']); actor.load_state_dict(saved['controller'])
        base.eval(); actor.eval()
    tune_metrics = {}
    for name in checkpoints:
        restore(name); tune_metrics[name] = evaluate(base, actor, tuning)
    choice = max(checkpoints, key=lambda k: (tune_metrics[k]['meanEpisodeReturnBps'], -tune_metrics[k]['forecastLoss']))
    if tune_metrics[choice]['meanEpisodeReturnBps'] <= 0:
        choice = 'cash'
    selection = {'chosen': choice, 'checkpoints': checkpoints,
        'tuning': tune_metrics, 'frozenBeforeTest': True,
        'cashMeanEpisodeReturnBps': 0., 'validationLastTargetTimestamp': int(val['timestamps'][-1]+1000)}
    write('selection.json', selection); frozen = sha(root/'selection.json')
    print(json.dumps({'event': 'selected', 'chosen': choice,
        'tuningBps': {k: v['meanEpisodeReturnBps'] for k, v in tune_metrics.items()}}), flush=True)
    test = load('test')
    if test['timestamps'][0] <= selection['validationLastTargetTimestamp'] or sha(root/'selection.json') != frozen:
        raise ValueError('test boundary/selection changed')
    testing = episodes(test)
    @torch.no_grad()
    def forecast_and_replay(data):
        means, ll, static_parts = [], [], []
        moment = base._component_moments('log', 1536)
        knots, areas = np.asarray(base.basis['knots']), np.asarray(base.basis['basisAreas'])
        for start in range(0, len(data['y']), 2560):
            x = torch.from_numpy(data['features'][start:start+2560]).to(device)
            logits = base(x)
            masses = component_log_masses(logits[:, 1:], base.areas).exp()
            static, state_weights = actor.coefficients(torch.sigmoid(logits[:, 0]), masses)
            static_parts.append(static.cpu().numpy())
            # Match the archived float64 density audit around narrow central knots.
            z = logits[:, 0].sigmoid().cpu().double().numpy()
            mass = masses.cpu().double().numpy()
            mass /= mass.sum(1, keepdims=True)
            target = data['y'][start:start+2560]
            unit = ndtr(target/base.sigma)
            index = np.clip(np.searchsorted(knots, unit, side='right')-1, 0, len(knots)-2)
            frac = (unit-knots[index])/np.diff(knots)[index]
            heights = mass/areas; ix = np.arange(len(target))
            density = heights[ix, index]*(1-frac)+heights[ix, index+1]*frac
            ld = np.log(density)-.5*(target/base.sigma)**2-np.log(base.sigma)-.5*np.log(2*np.pi)
            means.append((1-z)*(mass@moment))
            ll.append(np.where(target == 0, np.log(z), np.log1p(-z)+ld))
        static, weights = np.concatenate(static_parts), state_weights.cpu().numpy()
        prices = np.r_[data['price'], data['terminalPrice'][-1]]
        replay = rollout_numpy(static, weights, prices, rules, trace=True)
        trace = replay.pop('trace')
        # Audit all intermediate marks from the observed one-second tape.
        wealths = [rules.initial]
        for i, row in enumerate(trace):
            before_q, before_a = (rules.initial, 0.) if i == 0 else trace[i-1, 1:3]
            wealths.append(before_q+before_a*data['price'][i])
            end = min(4*i+4, len(data['tape']))
            wealths.extend(row[1]+row[2]*data['tape'][4*i:end])
        wealths.append(replay['wealth'])
        wealths = np.asarray(wealths)
        replay['maximumEveryTickDrawdownPercent'] = float(100*np.max(1-wealths/np.maximum.accumulate(wealths)))
        # Independently compare a complete initial episode against Torch execution.
        ts = torch.from_numpy(static[:length][None]).to(device)
        tp = torch.from_numpy(np.r_[data['price'][:length], data['terminalPrice'][length-1]][None]).to(device)
        exact = rollout_torch(ts, state_weights, tp, rules, trace=True)
        scalar = rollout_numpy(static[:length], weights, tp.cpu().numpy()[0], rules, trace=True)
        np.testing.assert_allclose(exact['trades'].cpu().numpy()[0], scalar['trace'][:, 0], atol=1e-8, rtol=1e-10)
        np.testing.assert_allclose(float(exact['wealth'][0]), scalar['wealth'], atol=1e-8, rtol=0)
        return scores(np.concatenate(means), data['y'], np.concatenate(ll)), replay, static, weights, trace
    results = {}
    for name in checkpoints:
        restore(name)
        episode_metrics = evaluate(base, actor, testing)
        forecast, replay, static, weights, trace = forecast_and_replay(test)
        results[name] = {'forecast': forecast, 'episodes': episode_metrics, 'fullTestReplay': replay,
                         'torchVsIndependentOracleEpisodeCheck': True}
        np.savez_compressed(root/(name+'-test-replay.npz'), staticLogits=static, stateWeights=weights,
            trace=trace, origins=test['origins'], timestamps=test['timestamps'])
        print(json.dumps({'event': 'test', 'arm': name, 'forecast': forecast,
            'episodeReturnBps': episode_metrics['meanEpisodeReturnBps'], 'fullReplay': replay}), flush=True)
    for file, digest in sources.items():
        if sha(file) != digest:
            raise ValueError('source changed during experiment: '+file)
    write('evaluation.json', {'plan': plan, 'selection': selection, 'test': results,
        'sourceHashes': sources, 'seconds': time.monotonic()-started,
        'testRows': len(test['y']), 'testEpisodes': len(testing['x']),
        'cashReturnPercent': 0., 'firstTestTimestamp': int(test['timestamps'][0]),
        'lastTestTimestamp': int(test['timestamps'][-1]+1000)})


if __name__ == '__main__':
    main()
