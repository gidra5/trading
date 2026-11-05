"""Frozen validation-selected pooling of archived laws and fresh native refits."""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '4'
import json
from pathlib import Path
import numpy as np
from scipy.special import ndtr
from audit_forecast_agreement import scores, block_ci, mix_likelihood, sha, basis_cdf
from expanded_distribution_mixture import fit_weights, pool_loglik
from outcome_distribution_gate import RETURN_EDGES


def regime_weights(x, y):
    j, s = x[:, 0], x[:, 1]
    agreement = j*s > 0
    weights = []
    for mask in (agreement, ~agreement):
        delta = j[mask]-s[mask]
        weights.append(float(np.clip(np.dot(delta, y[mask]-s[mask])/max(np.dot(delta, delta), 1e-300), 0, 1)))
    return weights


def regime_predictions(x, ll, weights):
    weight = np.where(x[:, 0]*x[:, 1] > 0, *weights)
    return x[:, 1]+weight*(x[:, 0]-x[:, 1]), mix_likelihood(weight, ll[:, 0], ll[:, 1])


def main():
    root = Path('data/backtests/oracle-expanded-mixture')
    if (root/'refitted-evaluation.json').exists():
        raise ValueError('completed evaluation already exists')
    hashes = {__file__: sha(__file__)}
    def checked(file):
        hashes[str(file)] = sha(file)
        return file
    def write(file, value):
        (root/file).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    result = {'protocol': 'fit simplex weights on first validation half, select pool on second; refit full validation and freeze before test scoring',
        'scope': 'H1 full distributions; historical test previously inspected; native checkpoints and heads also selected on validation',
        'newRefits': ['direct59', 'base17'], 'splits': {}}
    write('refitted-plan.json', result)
    selected = None
    for split in ('validation', 'test'):
        with np.load(checked(root/f'{split}-predictions.npz')) as f:
            x, ll, y, names = f['means'], f['logLikelihoods'], f['y'], f['names'].tolist()
            origins, times = f['origins'], f['timestamps']
        with np.load(checked(Path('data/backtests/oracle-outcome-gate')/f'{split}-regions.npz')) as f:
            np.testing.assert_array_equal(origins, f['origins'])
            base_regions = {'jointKnot': f['jmass'], 'currentSpecialists': f['smass']}
        refit_diagnostics = {}
        for family in result['newRefits']:
            checked(root/family/'head.pt'); checked(root/family/'selection.json')
            with np.load(checked(root/family/f'{split}-law.npz')) as f:
                for key, expected in [('origins', origins), ('timestamps', times), ('y', y)]:
                    np.testing.assert_array_equal(f[key], expected)
                np.testing.assert_allclose(f['componentMasses'].sum(axis=1), 1, atol=1e-12)
                if np.any((f['zeroProbability']<0) | (f['zeroProbability']>1)):
                    raise ValueError('invalid atom')
                x = np.column_stack((x, f['mean'])); ll = np.column_stack((ll, f['logLikelihood']))
                names.append(family+'Refit')
                cdf = f['componentMasses']@basis_cdf(f['knots'], f['areas'], ndtr(RETURN_EDGES/float(f['sigma'])))
                z = f['zeroProbability']
                mass = np.column_stack((z, (1-z[:, None])*np.diff(cdf, axis=1)))
                np.testing.assert_allclose(mass.sum(axis=1), 1, atol=1e-12)
                pup = 1-(f['componentMasses']@basis_cdf(f['knots'], f['areas'], np.array([.5]))).ravel()
                active = y!=0
                refit_diagnostics[family] = {'zeroBrier': float(np.mean((z-(y==0))**2)),
                    'activeSignAccuracy': float(np.mean((pup[active]>=.5)==(y[active]>0))),
                    'binnedTotalVariationQuantilesP10P50P90': {name: np.quantile(.5*np.abs(mass-m).sum(axis=1), [.1,.5,.9]).tolist()
                        for name,m in base_regions.items()}, 'binnedDistanceIsLowerBound': True}
        n = len(names)
        if split == 'validation':
            half = len(y)//2
            pools = {'base': [0, 1], 'allArchived': list(range(n-2)), 'all': list(range(n)),
                'basePlusDirect59': [0, 1, n-2], 'basePlusBase17': [0, 1, n-1], 'basePlusBothRefits': [0, 1, n-2, n-1]}
            pools.update({'single_'+name: [i] for i, name in enumerate(names)})
            selected = {'names': names, 'objectives': {}, 'frozenBeforeTest': True,
                'regimeWeights': regime_weights(x, y), 'fitLastTimestamp': int(times[half-1]), 'tuneFirstTimestamp': int(times[half])}
            for objective in ('mse', 'nll'):
                trials = {}
                for label, ids in pools.items():
                    w = np.zeros(n)
                    w[ids] = fit_weights(x[:half, ids], ll[:half, ids], y[:half], objective) if len(ids)>1 else 1
                    trials[label] = {'weights': w.tolist(), 'tuning': scores(x[half:]@w, y[half:], pool_loglik(ll[half:], w))}
                eq = np.zeros(n); eq[:2] = .5
                trials['equalBase'] = {'weights': eq.tolist(), 'tuning': scores(x[half:]@eq, y[half:], pool_loglik(ll[half:], eq))}
                regime = regime_weights(x[:half], y[:half])
                mean, lik = regime_predictions(x[half:], ll[half:], regime)
                if objective == 'mse':
                    trials['twoRegime'] = {'weights': regime, 'tuning': scores(mean, y[half:], lik)}
                field = 'mseBpsSquared' if objective == 'mse' else 'negativeLogLikelihood'
                winner = min(trials, key=lambda label: trials[label]['tuning'][field])
                def refit(label):
                    if label == 'twoRegime': return selected['regimeWeights']
                    if label == 'equalBase': return eq.tolist()
                    ids = pools[label]; w = np.zeros(n)
                    w[ids] = fit_weights(x[:, ids], ll[:, ids], y, objective) if len(ids)>1 else 1
                    return w.tolist()
                selected['objectives'][objective] = {'winner': winner, 'trials': trials, 'weights': refit(winner),
                    'allWeights': refit('all'), 'bothRefitsWeights': refit('basePlusBothRefits')}
            write('refitted-selection.json', selected); checked(root/'refitted-selection.json')
            eqmean = (x[:, 0]+x[:, 1])/2
            result['validationDiversity'] = {'predictionCorrelation': np.corrcoef(x.T).tolist(),
                'errorCorrelation': np.corrcoef((x-y[:, None]).T).tolist(),
                'marginalMseDerivativeVsEqualBase': (2*np.mean((eqmean-y)[:, None]*(x-eqmean[:, None]), axis=0)*1e8).tolist()}
        elif names != selected['names']:
            raise ValueError('forecast columns changed')
        predictions = {name: (x[:, i], ll[:, i]) for i, name in enumerate(names)}
        for objective, record in selected['objectives'].items():
            for suffix, key in [('selected', 'weights'), ('all', 'allWeights'), ('bothRefits', 'bothRefitsWeights')]:
                weights = record[key]
                predictions[objective+'_'+suffix] = regime_predictions(x, ll, weights) if suffix=='selected' and record['winner']=='twoRegime' \
                    else (x@weights, pool_loglik(ll, weights))
        predictions['equalBase'] = ((x[:, 0]+x[:, 1])/2, pool_loglik(ll[:, :2], [.5, .5]))
        predictions['twoRegime'] = regime_predictions(x, ll, selected['regimeWeights'])
        result['splits'][split] = {name: scores(mean, y, lik) for name, (mean, lik) in predictions.items()}
        result.setdefault('newRefitDiagnostics', {})[split] = refit_diagnostics
        np.savez_compressed(root/f'{split}-refitted-predictions.npz', names=np.array(names), means=x, logLikelihoods=ll,
            y=y, origins=origins, timestamps=times, **{'mean_'+k: v[0] for k, v in predictions.items()})
        if split=='test':
            result['selectedMseSkillGainIntervals'] = {name: block_ci(y, predictions['mse_selected'][0], predictions[name][0], times)
                for name in ['jointKnot', 'equalBase', 'twoRegime']}
        print(json.dumps({'split': split, 'selected': {k: v['winner'] for k, v in selected['objectives'].items()},
            'scores': {k: result['splits'][split][k] for k in ['equalBase', 'direct59Refit', 'base17Refit', 'mse_selected', 'nll_selected']}}), flush=True)
    result['selection'] = selected; result['sourceHashes'] = hashes
    write('refitted-evaluation.json', result)


if __name__ == '__main__':
    main()
