"""Frozen H1 distribution ensemble: chronological validation fit/select, then test.

Weights are on complete probability laws, including their zero atoms. Mean-MSE
and likelihood select separate pools; neither implies a trading profit gain.
"""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '4'
import json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logsumexp, ndtr
import torch
from audit_forecast_agreement import gamma_loglik, scores, block_ci, sha
from audit_current_forecast_mse import side_coefficients
from audit_hurdle_log_returns import log_means
from hurdle_sign_size_forecast import hurdle_nll_elements
from horizon_price_forecast import load_horizon_forecast
from stochastic_price_forecast import component_log_masses, load_timeline_split


def pool_loglik(ll, weights):
    w = np.asarray(weights)
    if np.any(w < 0) or not np.isclose(w.sum(), 1, atol=1e-12):
        raise ValueError('weights must form a probability simplex')
    with np.errstate(divide='ignore'):
        return logsumexp(ll+np.log(w), axis=1)


def fit_weights(means, ll, y, objective):
    n = means.shape[1]
    if objective == 'mse':
        x, target = means*1e4, y*1e4
        scale = np.mean(target**2)
        def loss(w):
            error = x@w-target
            return np.mean(error**2)/scale, 2*x.T@error/len(y)/scale
    elif objective == 'nll':
        offset = ll.max(axis=1)
        density = np.exp(ll-offset[:, None])
        def loss(w):
            mass = np.maximum(density@w, 1e-300)
            return -np.mean(np.log(mass)+offset), -(density/mass[:, None]).mean(axis=0)
    else:
        raise ValueError(objective)
    fitted = minimize(loss, np.full(n, 1/n), jac=True, method='SLSQP',
        bounds=[(0, 1)]*n, constraints=[{'type': 'eq', 'fun': lambda w: w.sum()-1,
        'jac': lambda w: np.ones_like(w)}], options={'maxiter': 1000, 'ftol': 1e-12})
    if not fitted.success:
        raise RuntimeError(fitted.message)
    w = np.maximum(fitted.x, 0); w /= w.sum()
    # Convex optimum must improve on every included vertex and equal pooling.
    if loss(w)[0] > min(loss(v)[0] for v in [*np.eye(n), np.full(n, 1/n)])+1e-9:
        raise ValueError('simplex optimizer failed the baseline check')
    return w


def simple_law(packed, y):
    up, down = log_means(packed, 128)
    low_up, low_down = log_means(packed, 64)
    p0, p = expit(packed[:, 0]), expit(packed[:, 1])
    mean = (1-p0)*(p*up+(1-p)*down)
    np.testing.assert_allclose(mean, (1-p0)*(p*low_up+(1-p)*low_down), atol=1e-12, rtol=0)
    # d(simple return)/d(log return)=exp(y); zero atom is unchanged.
    ll = -hurdle_nll_elements(torch.from_numpy(packed), torch.from_numpy(np.expm1(y))).numpy()+y
    return mean, ll


def main():
    root = Path('data/backtests/oracle-expanded-mixture')
    root.mkdir(parents=True, exist_ok=True)
    if (root/'evaluation.json').exists():
        raise ValueError('completed experiment already exists')
    sources = {}
    def checked(file, digest=None):
        file = Path(file); actual = sha(file)
        if digest is not None and actual != digest:
            raise ValueError('changed source: '+str(file))
        sources[str(file)] = actual
        return file
    def read(file):
        return json.loads(checked(file).read_text())
    def write(name, value):
        (root/name).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    checked(__file__)
    prior = read('data/backtests/oracle-forecast-agreement/audit.json')
    old_rules = read('data/backtests/oracle-forecast-agreement/selection.json')
    per = read('data/backtests/oracle-per-horizon-log/evaluation.json')
    shared = read('data/backtests/oracle-log-hurdle/evaluation.json')
    simple = read('data/training/runs/stochastic-crra-online-calibration-v1/result.json')
    simple_shared = read('data/training/runs/stochastic-crra-shared-hurdle-v1/result.json')
    plan = {'horizonSeconds': 1, 'target': 'expected log return, float64; zero atom retained',
        'protocol': 'Fit weights on first chronological validation half, select pool on second; refit full validation, freeze before test.',
        'objectives': ['mse', 'nll'], 'pools': 'base two; base two plus each candidate; all candidates; each singleton; equal base two',
        'historicalTestPreviouslyInspected': True, 'baseModelsUsedValidationForCheckpointSelection': True,
        'newTestBasedSelection': False, 'calibrations': 'frozen archived causal online streams, no new calibration fitting',
        'scope': 'H1 surviving compatible full laws only; removed checkpoints and other horizons excluded; asinh-logistic log tails excluded because price mean can diverge',
        'distribution': 'convex mixture of original full laws, not a mixture of representative outcomes'}
    write('plan.json', plan)
    torch.set_num_threads(4); torch.set_float32_matmul_precision('high')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = checked('data/training/runs/stochastic-crra-direct-horizons-gaussian-v2/h1/lambda0/best.pt')
    model, payload = load_horizon_forecast(checkpoint, device)
    dataset = Path(payload['plan']['dataset']); checked(dataset/'manifest.json')
    _, up = side_coefficients(model, True, 768)
    _, down = side_coefficients(model, False, 768)
    _, lowup = side_coefficients(model, True, 384)
    _, lowdown = side_coefficients(model, False, 384)
    moment = up-down
    np.testing.assert_allclose(moment, lowup-lowdown, atol=2e-10, rtol=0)
    knots, areas = np.asarray(model.basis['knots']), np.asarray(model.basis['basisAreas'])
    result = {'plan': plan, 'splits': {}}
    selection = None
    for split in ('validation', 'test'):
        with np.load(checked(f'data/backtests/oracle-forecast-agreement/{split}-predictions.npz')) as cache:
            base = {k: cache[k] for k in cache.files}
        y, origins = base['y'], base['origins']
        names = ['jointKnot', 'currentSpecialists']
        means, likelihoods = [base['jmean'], base['smean']], [base['jll'], base['sll']]
        def add(name, record, family):
            with np.load(checked(record['parametersFile'], record['parametersSha256'])) as f:
                ix = f['accuracyIndices']
                np.testing.assert_array_equal(f['origins'][ix], origins)
                np.testing.assert_array_equal(f['timestamps'][ix], base['timestamps'])
                packed = f['parameters'][ix].astype(float)
            if family == 'simple':
                mean, ll = simple_law(packed, y)
            else:
                p0, p = expit(packed[:, 0]), expit(packed[:, 1])
                mean = (1-p0)*(p*packed[:, 2]-(1-p)*packed[:, 3])
                ll = gamma_loglik(packed, y)
            names.append(name); means.append(mean); likelihoods.append(ll)
        for label, archive, models in [('perHLog', per, ['pretrainedRaw', 'pretrainedCalibrated', 'finetunedRaw', 'finetunedCalibrated']),
                                      ('sharedLog', shared, ['finetunedRaw', 'finetunedCalibrated'])]:
            row = next(r for r in archive['results'] if r['split'] == split and r['horizonSeconds'] == 1)
            for variant in models:
                add(label+'_'+variant, row['models'][variant]['savedStream'], 'log')
        for label, archive, models in [('perHSimple', simple, ['none', 'full']),
                                      ('sharedSimple', simple_shared, ['shared-raw', 'shared-online'])]:
            row = next(r for r in archive[split] if r['horizonSeconds'] == 1)
            for variant in models:
                add(label+'_'+variant, row['models'][variant]['savedStream'], 'simple')
        features = load_timeline_split(dataset, split).features(origins)
        knotmean, knotll = [], []
        for start in range(0, len(y), 2560):
            stop = min(start+2560, len(y)); target = y[start:stop]
            with torch.inference_mode():
                logits = model(torch.from_numpy(features[start:stop]).to(device))
                z = logits[:, 0].sigmoid().cpu().numpy().astype(float)
                masses = component_log_masses(logits[:, 1:], model.areas).exp().cpu().numpy().astype(float)
            masses /= masses.sum(axis=1, keepdims=True)
            np.testing.assert_allclose((1-z)*(masses@moment),
                (1-z)*(masses@(lowup-lowdown)), atol=1e-12, rtol=0)
            knotmean.append((1-z)*(masses@moment))
            unit = ndtr(target/model.sigma)
            interval = np.clip(np.searchsorted(knots, unit, side='right')-1, 0, len(knots)-2)
            frac = (unit-knots[interval])/(knots[interval+1]-knots[interval])
            heights = masses/areas; ix = np.arange(len(target))
            density = heights[ix, interval]*(1-frac)+heights[ix, interval+1]*frac
            ld = np.log(density)-.5*(target/model.sigma)**2-np.log(model.sigma)-.5*np.log(2*np.pi)
            knotll.append(np.where(target == 0, np.log(z), np.log1p(-z)+ld))
        names.append('likelihoodKnot'); means.append(np.concatenate(knotmean)); likelihoods.append(np.concatenate(knotll))
        x, ll = np.column_stack(means), np.column_stack(likelihoods)
        if not np.isfinite(x).all() or not np.isfinite(ll).all():
            raise ValueError('invalid forecast')
        n = len(names)
        predictions = {}
        if split == 'validation':
            half = len(y)//2
            pools = {'base': [0, 1], 'all': list(range(n))}
            pools.update({'add_'+names[i]: [0, 1, i] for i in range(2, n)})
            pools.update({'single_'+name: [i] for i, name in enumerate(names)})
            selection = {'names': names, 'objectives': {}, 'frozenBeforeTest': True,
                'fitLastTimestamp': int(base['timestamps'][half-1]), 'tuneFirstTimestamp': int(base['timestamps'][half])}
            for objective in ('mse', 'nll'):
                trials = {}
                for label, ids in pools.items():
                    local = fit_weights(x[:half, ids], ll[:half, ids], y[:half], objective) if len(ids)>1 else np.ones(1)
                    w = np.zeros(n); w[ids] = local
                    trials[label] = {'firstHalfWeights': w.tolist(), 'tuning': scores(x[half:]@w, y[half:], pool_loglik(ll[half:], w))}
                w = np.zeros(n); w[:2] = .5
                trials['equalBase'] = {'firstHalfWeights': w.tolist(), 'tuning': scores(x[half:]@w, y[half:], pool_loglik(ll[half:], w))}
                field = 'mseBpsSquared' if objective == 'mse' else 'negativeLogLikelihood'
                winner = min(trials, key=lambda k: trials[k]['tuning'][field])
                def refit(label):
                    w = np.zeros(n)
                    if label == 'equalBase': w[:2] = .5
                    else:
                        ids = pools[label]
                        w[ids] = fit_weights(x[:, ids], ll[:, ids], y, objective) if len(ids)>1 else 1.
                    return w.tolist()
                selection['objectives'][objective] = {'winner': winner, 'trials': trials,
                    'weights': refit(winner), 'allWeights': refit('all'), 'baseWeights': refit('base')}
            write('selection.json', selection); checked(root/'selection.json')
            eq = (x[:, 0]+x[:, 1])/2; error = eq-y
            result['validationDiversity'] = {'meanPredictionCorrelation': np.corrcoef(x.T).tolist(),
                'errorCorrelation': np.corrcoef((x-y[:, None]).T).tolist(),
                'marginalMseDerivativeVsEqualBase': (2*np.mean(error[:, None]*(x-eq[:, None]), axis=0)*1e8).tolist(),
                'interpretation': 'negative derivative means a small addition reduces validation MSE before refitting'}
        else:
            if names != selection['names']:
                raise ValueError('candidate order changed')
        for i, name in enumerate(names):
            predictions[name] = (x[:, i], ll[:, i])
        for objective, record in selection['objectives'].items():
            for label, key in [('selected', 'weights'), ('all', 'allWeights'), ('base', 'baseWeights')]:
                w = np.array(record[key]); predictions[objective+'_'+label] = (x@w, pool_loglik(ll, w))
        predictions['equalBase'] = ((x[:, 0]+x[:, 1])/2, pool_loglik(ll[:, :2], [.5, .5]))
        weight = np.where(x[:, 0]*x[:, 1] > 0, old_rules['agreementWeightJoint'], old_rules['disagreementWeightJoint'])
        from audit_forecast_agreement import mix_likelihood
        predictions['previousTwoRegime'] = (x[:, 1]+weight*(x[:, 0]-x[:, 1]), mix_likelihood(weight, ll[:, 0], ll[:, 1]))
        metrics = {name: scores(mean, y, loglike) for name, (mean, loglike) in predictions.items()}
        result['splits'][split] = metrics
        np.savez_compressed(root/f'{split}-predictions.npz', means=x, logLikelihoods=ll, y=y, origins=origins,
            timestamps=base['timestamps'], names=np.array(names), **{'mean_'+k: v[0] for k, v in predictions.items()})
        if split == 'test':
            result['selectedMseSkillGainIntervals'] = {label: block_ci(y, predictions['mse_selected'][0], predictions[label][0], base['timestamps'])
                for label in ['jointKnot', 'equalBase', 'previousTwoRegime']}
        print(json.dumps({'split': split, 'models': len(names), 'selected': {k: v['winner'] for k, v in selection['objectives'].items()},
            'scores': {k: metrics[k] for k in ['jointKnot', 'equalBase', 'mse_selected', 'nll_selected']}}), flush=True)
    result['sourceHashes'] = sources
    result['selection'] = selection
    write('evaluation.json', result)


if __name__ == '__main__':
    main()
