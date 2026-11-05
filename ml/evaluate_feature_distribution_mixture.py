"""Matched H1 experiment: pool sign/size laws separately, freeze before test."""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '4'
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from audit_forecast_agreement import sha, block_ci
from audit_current_forecast_mse import side_coefficients
from expanded_distribution_mixture import fit_weights, pool_loglik
from feature_distribution_mixture import (LAYOUTS, validate, classes, predict, loglik,
    fit_nll, fit_mse, full_prediction, metrics)

NAMES = ['jointKnot', 'currentSpecialists', 'direct59Refit', 'base17Refit']
POOLS = {'originalPair': [0, 1], 'retainedThree': [1, 2, 3], 'allFour': [0, 1, 2, 3]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('data/backtests/oracle-feature-mixture'))
    args = parser.parse_args()
    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    if (root/'evaluation.json').exists():
        raise ValueError('completed experiment already exists')
    hashes = {}
    def checked(file, digest=None):
        file = Path(file)
        actual = sha(file)
        if digest is not None and actual != digest:
            raise ValueError('changed source: '+str(file))
        hashes[str(file)] = actual
        return file
    def read(file):
        return json.loads(checked(file).read_text())
    def write(name, value):
        (root/name).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    for file in [__file__, 'ml/feature_distribution_mixture.py', 'ml/expanded_distribution_mixture.py',
                 'ml/audit_forecast_agreement.py', 'ml/audit_current_forecast_mse.py']:
        checked(file)
    old = read('data/backtests/oracle-forecast-agreement/audit.json')
    previous = read('data/backtests/oracle-expanded-mixture/refitted-evaluation.json')
    previous_selection = read('data/backtests/oracle-expanded-mixture/refitted-selection.json')
    plan = {'horizonSeconds': 1, 'target': 'log return, all ticks including exact zero',
        'models': NAMES, 'pools': POOLS, 'families': ['fullLaw', *LAYOUTS],
        'objectives': ['nll', 'mse'],
        'protocol': 'fit first chronological validation half; select pool per family/objective and best family on second half; refit chosen pools on full validation; freeze before test',
        'calibration': 'reuse frozen archived forecast streams with their existing causal calibration; no new calibration fitting',
        'magnitude': 'entire magnitude distribution conditional on direction; mixture weights shared across directions unless zeroSignSides',
        'zero': 'ternary includes zero; other variants give zero probability its own weights',
        'distribution': 'P(sign) times P(magnitude|sign); each factor is a normalized convex mixture of original factors',
        'mseOptimization': 'analytic gradient, SLSQP multistart at NLL weights, uniform weights, and each singleton; nonconvex, no global certificate',
        'nllOptimization': 'independent convex simplex fits for each factor using its proper likelihood',
        'historicalTestPreviouslyInspected': True, 'baseCheckpointsUsedValidation': True,
        'newTestBasedSelection': False, 'baseModelRetraining': False,
        'quadrature': 'refit half-knot moments order1536 vs768, require absolute mean error <=1e-12'}
    write('plan.json', plan)
    moments = {}
    def load(split):
        record = old['splits'][split]
        with np.load(checked(record['predictionsFile'], record['predictionsSha256'])) as f:
            data = {k: f[k] for k in ('origins', 'timestamps', 'y', 'jc', 'sc', 'jll', 'sll')}
        cs, likelihoods = [data['jc'], data['sc']], [data['jll'], data['sll']]
        expanded = Path('data/backtests/oracle-expanded-mixture')
        for family in ('direct59', 'base17'):
            file = expanded/family/f'{split}-law.npz'
            expected = previous['sourceHashes'].get(str(file).replace('\\', '/'))
            if expected is None:
                expected = previous['sourceHashes'].get(str(file))
            with np.load(checked(file, expected)) as f:
                for key in ('origins', 'timestamps', 'y'):
                    np.testing.assert_array_equal(f[key], data[key])
                key = (tuple(f['knots']), tuple(f['areas']), float(f['sigma']))
                if key not in moments:
                    model = SimpleNamespace(basis={'knots': f['knots'], 'basisAreas': f['areas']}, sigma=float(f['sigma']))
                    pu, up = side_coefficients(model, True, 1536)
                    pd, down = side_coefficients(model, False, 1536)
                    _, lowup = side_coefficients(model, True, 768)
                    _, lowdown = side_coefficients(model, False, 768)
                    moments[key] = (pu, pd, up, down, lowup, lowdown)
                pu, pd, up, down, lowup, lowdown = moments[key]
                masses = f['componentMasses']
                np.testing.assert_allclose(masses.sum(1), 1, atol=1e-12, rtol=0)
                z, p, pdown = f['zeroProbability'], masses@pu, masses@pd
                np.testing.assert_allclose(p+pdown, 1, atol=1e-12, rtol=0)
                c = np.column_stack((z, p, masses@up/p, masses@down/pdown))
                recomputed = (1-z)*(p*c[:, 2]-(1-p)*c[:, 3])
                np.testing.assert_allclose(recomputed, f['mean'], atol=1e-12, rtol=0)
                np.testing.assert_allclose(recomputed, (1-z)*(masses@(lowup-lowdown)), atol=1e-12, rtol=0)
                cs.append(c); likelihoods.append(f['logLikelihood'])
        data['c'], data['ll'] = np.stack(cs, axis=1), np.column_stack(likelihoods)
        validate(data['c'])
        for i in range(len(NAMES)):
            w = np.eye(len(NAMES))[i]
            p = full_prediction(data['c'], w)
            np.testing.assert_allclose(metrics(p, data['y'], data['ll'][:, i])['skillPercent'],
                previous['splits'][split][NAMES[i]]['skillPercent'], atol=1e-8, rtol=0)
            for layout in LAYOUTS:
                weights = np.tile(w, (len(LAYOUTS[layout]), 1))
                np.testing.assert_allclose(predict(data['c'], weights, layout)['mean'], p['mean'], atol=1e-16, rtol=0)
                np.testing.assert_allclose(loglik(data['c'], data['ll'], data['y'], weights, layout),
                    data['ll'][:, i], atol=1e-12, rtol=0)
        return data

    def fit(data, family, objective, ids, index):
        c, ll, y = data['c'][index][:, ids], data['ll'][index][:, ids], data['y'][index]
        if family == 'fullLaw':
            means = (1-c[..., 0])*(c[..., 1]*c[..., 2]-(1-c[..., 1])*c[..., 3])
            return fit_weights(means, ll, y, objective)
        return (fit_nll if objective == 'nll' else fit_mse)(c, ll, y, family)

    def evaluate(data, family, ids, weights, index=slice(None)):
        c, ll, y = data['c'][index][:, ids], data['ll'][index][:, ids], data['y'][index]
        if family == 'fullLaw':
            p, likelihood = full_prediction(c, weights), pool_loglik(ll, weights)
        else:
            p, likelihood = predict(c, weights, family), loglik(c, ll, y, weights, family)
        return metrics(p, y, likelihood), p, likelihood

    validation = load('validation')
    print(json.dumps({'event': 'validation_loaded', 'examples': len(validation['y'])}), flush=True)
    half = len(validation['y'])//2
    if validation['timestamps'][half-1]+1000 >= validation['timestamps'][half]:
        raise ValueError('fit targets must mature before tuning origins')
    selection = {'frozenBeforeTest': True, 'trials': {}, 'models': {}, 'bestFeatureFamily': {},
                 'bestOverall': {}, 'fitLastTimestamp': int(validation['timestamps'][half-1]),
                 'tuneFirstTimestamp': int(validation['timestamps'][half])}
    for objective in plan['objectives']:
        field = 'negativeLogLikelihood' if objective == 'nll' else 'mseBpsSquared'
        for family in plan['families']:
            trials = []
            for pool, ids in POOLS.items():
                w = fit(validation, family, objective, ids, slice(None, half))
                metric, _, _ = evaluate(validation, family, ids, w, slice(half, None))
                label = f'{family}_{objective}_{pool}'
                selection['trials'][label] = {'family': family, 'objective': objective,
                    'pool': pool, 'ids': ids, 'weights': w.tolist(), 'tuning': metric}
                trials.append(label)
                print(json.dumps({'event': 'trial', 'label': label, 'skill': metric['skillPercent'],
                                  'nll': metric['negativeLogLikelihood']}), flush=True)
            winner = min(trials, key=lambda k: selection['trials'][k]['tuning'][field])
            row = selection['trials'][winner]
            full_weights = fit(validation, family, objective, row['ids'], slice(None))
            selection['models'][family+'_'+objective] = {**row, 'weights': full_weights.tolist(), 'selectedTrial': winner}
        candidates = [name for name, row in selection['models'].items() if row['objective'] == objective]
        selection['bestOverall'][objective] = min(candidates, key=lambda k: selection['models'][k]['tuning'][field])
        selection['bestFeatureFamily'][objective] = min((k for k in candidates if not k.startswith('fullLaw')),
            key=lambda k: selection['models'][k]['tuning'][field])
    write('selection.json', selection)
    frozen = sha(root/'selection.json')
    result = {'plan': plan, 'selection': selection, 'splits': {}}
    for split in ('validation', 'test'):
        checked(root/'selection.json', frozen)
        data = validation if split == 'validation' else load(split)
        rows, forecasts = {}, {}
        for name, row in selection['models'].items():
            metric, p, likelihood = evaluate(data, row['family'], row['ids'], np.asarray(row['weights']))
            rows[name] = metric
            forecasts[name] = p
        for i, name in enumerate(NAMES):
            p = full_prediction(data['c'], np.eye(len(NAMES))[i])
            rows[name] = metrics(p, data['y'], data['ll'][:, i]); forecasts[name] = p
        for name, ids, weights in [('equalOriginalPair', [0, 1], np.array([.5, .5])),
                ('previousExpandedMse', list(range(4)), np.array([previous_selection['objectives']['mse']['weights'][
                    previous_selection['names'].index(n)] for n in NAMES]))]:
            np.testing.assert_allclose(weights.sum(), 1, atol=1e-10)
            weights = weights/weights.sum()
            metric, p, likelihood = evaluate(data, 'fullLaw', ids, weights)
            rows[name], forecasts[name] = metric, p
        np.testing.assert_allclose(rows['previousExpandedMse']['skillPercent'],
            previous['splits'][split]['mse_selected']['skillPercent'], atol=1e-8, rtol=0)
        result['splits'][split] = rows
        np.savez_compressed(root/f'{split}-predictions.npz', y=data['y'], origins=data['origins'],
            timestamps=data['timestamps'], sourceComponents=data['c'], sourceLogLikelihood=data['ll'],
            **{key+'_'+name: p[key] for name, p in forecasts.items() for key in p})
        if split == 'test':
            if validation['timestamps'][-1]+1000 >= data['timestamps'][0]:
                raise ValueError('validation targets overlap test')
            for objective, name in selection['bestFeatureFamily'].items():
                result.setdefault('pairedTestSkillGain', {})[name] = {
                    baseline: block_ci(data['y'], forecasts[name]['mean'], forecasts[baseline]['mean'], data['timestamps'])
                    for baseline in ['previousExpandedMse', 'direct59Refit', 'fullLaw_'+objective]}
        print(json.dumps({'event': 'scored', 'split': split,
            'bestFeatureFamily': selection['bestFeatureFamily'], 'scores': rows}), flush=True)
    for file, digest in hashes.items():
        if sha(file) != digest:
            raise ValueError('source changed during experiment: '+file)
    result['sourceHashes'] = hashes
    write('evaluation.json', result)


if __name__ == '__main__':
    main()
