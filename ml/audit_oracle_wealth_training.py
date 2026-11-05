"""Final float64 forecast scoring and all-mark execution audit; no selection.

Matches the archived knot-law NLL evaluation, avoiding float32 CDF resolution
around the narrow central knots. Includes pre- and post-trade marks in drawdown.
"""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '4'
import json
from pathlib import Path
import numpy as np
from scipy.special import ndtr
import torch
from audit_forecast_agreement import sha, scores
from horizon_price_forecast import load_horizon_forecast
from stochastic_price_forecast import component_log_masses
from oracle_wealth_training import TrainingRules, rollout_numpy


def main():
    root = Path('data/backtests/oracle-wealth-training')
    file = root/'evaluation.json'
    result = json.loads(file.read_text())
    selection_hash = sha(root/'selection.json')
    if 'finalNumericalAudit' in result:
        raise ValueError('already audited')
    snapshot = root/'training-source.py'
    original_path = next(k for k in result['sourceHashes'] if Path(k).name == 'train_oracle_wealth_forecast.py')
    if sha(snapshot) != result['sourceHashes'][original_path]:
        raise ValueError('missing original training source snapshot')
    (root/'evaluation-before-float64-audit.json').write_text(file.read_text())
    torch.set_num_threads(4); torch.set_float32_matmul_precision('high')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, _ = load_horizon_forecast(Path(result['plan']['sourceCheckpoint']), device)
    high, low = model._component_moments('log', 1536), model._component_moments('log', 768)
    with np.load('data/backtests/oracle-expanded-mixture/direct59/test-native.npz') as f:
        features, y, timestamps = f['features'], f['y'], f['timestamps']
    with np.load(root/'test-prices.npz') as f:
        np.testing.assert_array_equal(f['timestamps'], timestamps)
        prices = f['prices']
    rules = TrainingRules(**result['plan']['executionRules'])
    knots, areas = np.asarray(model.basis['knots']), np.asarray(model.basis['basisAreas'])
    rows = {}
    for name, record in result['test'].items():
        saved = torch.load(result['selection']['checkpoints'][name]['file'], map_location=device, weights_only=False)
        model.load_state_dict(saved['head']); model.eval()
        means, likelihoods = [], []
        with torch.inference_mode():
            for start in range(0, len(y), 2560):
                target = y[start:start+2560]
                logits = model(torch.from_numpy(features[start:start+2560]).to(device))
                z = logits[:, 0].sigmoid().cpu().double().numpy()
                masses = component_log_masses(logits[:, 1:], model.areas).exp().cpu().double().numpy()
                masses /= masses.sum(1, keepdims=True)
                mean = (1-z)*(masses@high)
                np.testing.assert_allclose(mean, (1-z)*(masses@low), atol=1e-12, rtol=0)
                unit = ndtr(target/model.sigma)
                index = np.clip(np.searchsorted(knots, unit, side='right')-1, 0, len(knots)-2)
                frac = (unit-knots[index])/np.diff(knots)[index]
                height = masses/areas; ix = np.arange(len(target))
                density = height[ix, index]*(1-frac)+height[ix, index+1]*frac
                ld = np.log(density)-.5*(target/model.sigma)**2-np.log(model.sigma)-.5*np.log(2*np.pi)
                likelihoods.append(np.where(target == 0, np.log(z), np.log1p(-z)+ld))
                means.append(mean)
        mean, ll = np.concatenate(means), np.concatenate(likelihoods)
        if name in ('unchanged', 'actorOnly'):
            with np.load('data/backtests/oracle-expanded-mixture/direct59/test-law.npz') as f:
                np.testing.assert_allclose(mean, f['mean'], atol=1e-12, rtol=0)
                np.testing.assert_allclose(ll, f['logLikelihood'], atol=1e-9, rtol=0)
        record['forecastFloat32Diagnostic'] = record['forecast']
        record['forecast'] = scores(mean, y, ll)
        np.savez_compressed(root/(name+'-test-forecast.npz'), mean=mean, logLikelihood=ll, y=y, timestamps=timestamps)
        with np.load(root/(name+'-test-replay.npz')) as f:
            trace, static, weights = f['trace'], f['staticLogits'], f['stateWeights']
        replay = rollout_numpy(static, weights, np.r_[prices[np.arange(len(y))*4], prices[-1]], rules, trace=True)
        np.testing.assert_array_equal(replay['trace'], trace)
        np.testing.assert_allclose(replay['wealth'], record['fullTestReplay']['wealth'], atol=1e-9, rtol=0)
        if np.any((abs(trace[:, 0]) > 0) & (abs(trace[:, 0]) < rules.minimum)) or np.any(abs(trace[:, 0]) > rules.maximum+1e-9):
            raise ValueError('invalid actual order size')
        marks = [rules.initial]
        for i, row in enumerate(trace):
            q, a = (rules.initial, 0.) if i == 0 else trace[i-1, 1:3]
            marks.append(q+a*prices[4*i])
            marks.extend(row[1]+row[2]*prices[4*i:min(4*i+4, len(prices))])
        marks.append(replay['wealth']); marks = np.asarray(marks)
        dd = float(100*np.max(1-marks/np.maximum.accumulate(marks)))
        old_dd = record['fullTestReplay']['maximumEveryTickDrawdownPercent']
        record['fullTestReplay']['maximumEveryTickDrawdownPercent'] = dd
        rows[name] = {'forecastFloat64': record['forecast'], 'replayedTrades': replay['trades'],
            'wealthUnchanged': True, 'oldDrawdownPercent': old_dd, 'allMarkDrawdownPercent': dd}
    if sha(root/'selection.json') != selection_hash:
        raise ValueError('selection changed during diagnostic scoring')
    result['finalNumericalAudit'] = {'script': __file__, 'sha256': sha(__file__),
        'selectionUnchanged': True, 'trainingSourceSnapshot': str(snapshot),
        'forecastScoring': 'float64 Gaussian-CDF knot density and log moments, matching previous refit evaluation',
        'drawdown': 'all one-second marks, including both pre- and post-market execution', 'rows': rows}
    file.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'audited': list(rows), 'chosen': result['selection']['chosen'],
        'scores': {k: v['forecastFloat64'] for k, v in rows.items()}}))


if __name__ == '__main__':
    main()
