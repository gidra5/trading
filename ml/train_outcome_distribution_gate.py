"""Validation-trained outcome-dependent distribution pool, frozen-model H=1 trial."""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '4'
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit, gammainc, gammaincc, ndtr, ndtri
import torch

from audit_forecast_agreement import basis_cdf, gamma_cdf, mix_likelihood, scores, block_ci
from horizon_price_forecast import load_horizon_forecast
from outcome_distribution_gate import (CONTRACT, RETURN_EDGES, GateData, context_features,
    fit_gate, outcome_regions, predict_gate)
from stochastic_price_forecast import component_log_masses, load_timeline_split


def sha(file):
    return hashlib.sha256(Path(file).read_bytes()).hexdigest()


def write(file, value):
    Path(file).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def knot_bin_moments(model, order):
    knots, areas = np.asarray(model.basis['knots']), np.asarray(model.basis['basisAreas'])
    edges = ndtr(RETURN_EDGES/model.sigma)
    nodes, weights = np.polynomial.legendre.leggauss(order)
    result = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        left, right = np.maximum(knots[:-1], lo), np.minimum(knots[1:], hi)
        half = np.maximum(right-left, 0)/2
        center = np.where(half > 0, (left+right)/2, .5)
        unit = center[:, None]+half[:, None]*nodes
        fraction = (unit-knots[:-1, None])/np.diff(knots)[:, None]
        value = model.sigma*ndtri(unit)
        integral = np.zeros(len(knots))
        integral[:-1] += np.sum(value*(1-fraction)*weights, axis=1)*half
        integral[1:] += np.sum(value*fraction*weights, axis=1)*half
        result.append(integral/areas)
    return np.asarray(result).T


def specialist_regions(packed):
    pzero, pup = expit(packed[:, 0]), expit(packed[:, 1])[:, None]
    cdf = gamma_cdf(packed, RETURN_EDGES)
    au, ad = packed[:, 4, None], packed[:, 5, None]
    up, down = packed[:, 2, None], packed[:, 3, None]
    cumulative = np.empty_like(cdf)
    negative = RETURN_EDGES < 0
    cumulative[:, negative] = -(1-pup)*down*gammaincc(ad+1, -RETURN_EDGES[None, negative]*ad/down)
    cumulative[:, ~negative] = -(1-pup)*down+pup*up*gammainc(au+1, RETURN_EDGES[None, ~negative]*au/up)
    mass = np.column_stack((pzero, (1-pzero[:, None])*np.diff(cdf, axis=1)))
    moment = np.column_stack((np.zeros(len(packed)), (1-pzero[:, None])*np.diff(cumulative, axis=1)))
    return np.maximum(mass, 0), moment


def fit_constant(data):
    fit = minimize_scalar(lambda w: -mix_likelihood(w, data.jll, data.sll).mean(),
        bounds=(0, 1), method='bounded', options={'xatol': 1e-10})
    if not fit.success:
        raise RuntimeError('constant-mixture fitting failed')
    return min([0., 1., float(fit.x)], key=lambda w: -mix_likelihood(w, data.jll, data.sll).mean())


def evaluate(gate, data, y):
    forecast = predict_gate(gate, data)
    np.testing.assert_allclose(forecast['regionMass'].sum(axis=1), 1, rtol=0, atol=5e-13)
    if np.any(forecast['regionMass'] < 0):
        raise ValueError('negative normalized probability')
    result = scores(forecast['mean'], y, forecast['logLikelihood'])
    result.update(zeroBrier=float(np.square(forecast['regionMass'][:, 0]-(y == 0)).mean()),
        normalizerQuantiles=np.quantile(forecast['normalizer'], [0, .1, .5, .9, 1]).tolist(),
        averageGateWeights=forecast['weights'].mean(axis=0).tolist(),
        maximumNormalizationError=float(np.max(abs(forecast['regionMass'].sum(axis=1)-1))))
    return result, forecast


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('data/backtests/oracle-outcome-gate'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output/'evaluation.json').exists():
        raise ValueError('completed experiment already exists')
    hashes = {}
    def checked(file, digest=None):
        actual = sha(file)
        if digest is not None and actual != digest:
            raise ValueError('changed source: '+str(file))
        hashes[str(file)] = actual
        return Path(file)
    old = json.loads(checked('data/backtests/oracle-forecast-agreement/audit.json').read_text())
    current = json.loads(checked('data/backtests/oracle-joint-calibration/evaluation.json').read_text())
    joint_record = json.loads(checked('data/backtests/oracle-online-calibration/joint-log-return-audit.json').read_text())
    checked(__file__); checked('ml/outcome_distribution_gate.py')
    plan = {'contract': CONTRACT, 'horizonSeconds': 1, 'primarySelectionMetric': 'normalized log-return NLL',
        'regionsBps': ['-infinity', -5, -1, -.2, -.05, -.01, -.003, 0, .003, .01, .05, .2, 1, 5, 'infinity'],
        'zeroAtomSeparate': True, 'gateFamilies': ['outcomeOnly', 'outcomeAndContext'],
        'ridges': [.001, .01, .1], 'prior': 'first-half-fitted constant likelihood mixture; coefficient and adjacent-continuous-region quadratic regularization',
        'contextFeatures': ['jointZero', 'jointUpGivenActive', 'specialistZero', 'specialistUpGivenActive',
            'logJointActiveMagnitude', 'logSpecialistActiveMagnitude', 'jointConditionalMeanBps', 'specialistConditionalMeanBps'],
        'protocol': 'fit first chronological validation half; select family/ridge on second half NLL; refit each family winner on full validation; freeze before test',
        'testUsedForSelection': False, 'historicalTestPreviouslyInspected': True,
        'frozen': 'both original forecast laws and their archived online calibrations',
        'loss': 'normalized mixed atom/density NLL only, no mean-MSE auxiliary',
        'normalizer': 'exact CDF region masses; both continuous tails and zero atom included in log Z gradient',
        'mean': 'gate-weighted regional first log moments divided by Z; joint Gauss-Legendre768 per intersected knot interval, Gamma analytic incomplete moments',
        'smoothing': 'piecewise constant outcome weights, neighboring coefficient differences penalized; base densities are preserved within regions'}
    write(args.output/'plan.json', plan); checked(args.output/'plan.json')
    torch.set_num_threads(4); torch.set_float32_matmul_precision('high')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, payload = load_horizon_forecast(checked(joint_record['checkpoint'], joint_record['checkpointSha256']), device)
    dataset = Path(payload['plan']['dataset'])
    checked(dataset/'manifest.json', joint_record['datasetManifestSha256'])
    masses_coeff = np.diff(basis_cdf(np.asarray(model.basis['knots']), np.asarray(model.basis['basisAreas']),
        ndtr(RETURN_EDGES/model.sigma)), axis=1)
    moments_coeff, low_moments_coeff = knot_bin_moments(model, 768), knot_bin_moments(model, 384)
    started = time.monotonic()
    output = {'plan': plan, 'splits': {}}
    selection = None
    for split in ('validation', 'test'):
        if split == 'test':
            checked(args.output/'selection.json', selection_hash)
        old_row = old['splits'][split]
        with np.load(checked(old_row['predictionsFile'], old_row['predictionsSha256'])) as archive:
            reference = {key: archive[key] for key in ('origins', 'timestamps', 'jc', 'sc', 'y', 'jll', 'sll', 'jmean', 'smean')}
        row = next(r for r in current['results'] if r['split'] == split and r['horizonSeconds'] == 1)
        saved = row['models']['joint-calibrated']['savedStream']
        with np.load(checked(saved['parametersFile'], saved['parametersSha256'])) as stream:
            ix = stream['accuracyIndices']; packed = stream['parameters'][ix]
            np.testing.assert_array_equal(stream['origins'][ix], reference['origins'])
        smass, smoment = specialist_regions(packed)
        timeline = load_timeline_split(dataset, split)
        features = timeline.features(reference['origins'])
        jmass, jmoment, low_jmoment = [], [], []
        with torch.inference_mode():
            for start in range(0, len(features), 2560):
                stop = min(start+2560, len(features))
                logits = model(torch.from_numpy(features[start:stop]).to(device))
                zero = logits[:, 0].sigmoid().cpu().numpy().astype(float)
                mass = component_log_masses(logits[:, 1:], model.areas).exp().cpu().numpy().astype(float)
                mass /= mass.sum(axis=1, keepdims=True)
                np.testing.assert_array_equal(zero, reference['jc'][start:stop, 0])
                jmass.append(np.column_stack((zero, (1-zero[:, None])*(mass@masses_coeff))))
                jmoment.append(np.column_stack((np.zeros(len(mass)), (1-zero[:, None])*(mass@moments_coeff))))
                low_jmoment.append(np.column_stack((np.zeros(len(mass)), (1-zero[:, None])*(mass@low_moments_coeff))))
        data = GateData(np.maximum(np.concatenate(jmass), 0), smass, np.concatenate(jmoment), smoment,
            reference['jll'], reference['sll'], outcome_regions(reference['y']), context_features(reference['jc'], reference['sc']))
        data.validate()
        np.testing.assert_allclose(data.jmoment.sum(axis=1), reference['jmean'], rtol=0, atol=1e-10)
        np.testing.assert_allclose(data.smoment.sum(axis=1), reference['smean'], rtol=0, atol=1e-16)
        low_moments = np.concatenate(low_jmoment)
        cache_file = args.output/f'{split}-regions.npz'
        np.savez_compressed(cache_file, **{key: getattr(data, key) for key in data.__dataclass_fields__},
            lowJmoment=low_moments, y=reference['y'], origins=reference['origins'], timestamps=reference['timestamps'])
        print(json.dumps({'event': 'prepared', 'split': split, 'seconds': time.monotonic()-started}), flush=True)
        if split == 'validation':
            half = len(data.region)//2
            if reference['timestamps'][half-1]+1000 > reference['timestamps'][half]:
                raise ValueError('validation target maturity overlaps tuning origin')
            train, tune = data.subset(slice(None, half)), data.subset(slice(half, None))
            initial = float(np.clip(fit_constant(train), .001, .999))
            trials = []
            for family in plan['gateFamilies']:
                for ridge in plan['ridges']:
                    gate = fit_gate(train, context=family == 'outcomeAndContext', ridge=ridge, initial_weight=initial)
                    train_score, _ = evaluate(gate, train, reference['y'][:half])
                    tune_score, _ = evaluate(gate, tune, reference['y'][half:])
                    trials.append({'family': family, 'ridge': ridge, 'gate': gate, 'train': train_score, 'tuning': tune_score})
                    print(json.dumps({'event': 'trial', 'family': family, 'ridge': ridge,
                        'tuning': tune_score, 'iterations': gate['iterations'], 'seconds': time.monotonic()-started}), flush=True)
            family_winners = {family: min((trial for trial in trials if trial['family'] == family),
                key=lambda trial: trial['tuning']['negativeLogLikelihood']) for family in plan['gateFamilies']}
            winner = min(trials, key=lambda trial: trial['tuning']['negativeLogLikelihood'])
            full_initial = float(np.clip(fit_constant(data), .001, .999))
            gates = {family: fit_gate(data, context=family == 'outcomeAndContext', ridge=trial['ridge'],
                initial_weight=full_initial) for family, trial in family_winners.items()}
            controls = {'joint': scores(reference['jmean'][half:], reference['y'][half:], reference['jll'][half:]),
                'specialists': scores(reference['smean'][half:], reference['y'][half:], reference['sll'][half:]),
                'constantLikelihoodMixture': scores(initial*reference['jmean'][half:]+(1-initial)*reference['smean'][half:],
                    reference['y'][half:], mix_likelihood(initial, reference['jll'][half:], reference['sll'][half:]))}
            candidate_nll = {name: value['negativeLogLikelihood'] for name, value in controls.items()}
            candidate_nll.update({family: trial['tuning']['negativeLogLikelihood'] for family, trial in family_winners.items()})
            selection = {'contract': CONTRACT, 'selectedGateFamily': winner['family'],
                'selectedIncludingControls': min(candidate_nll, key=candidate_nll.get), 'trials': trials,
                'tuningControls': controls, 'frozenGates': gates, 'constantWeight': full_initial,
                'firstHalfInitialWeight': initial, 'frozenBeforeTest': True, 'testUsedForSelection': False,
                'fitEndTimestamp': int(reference['timestamps'][half-1]), 'tuningFirstTimestamp': int(reference['timestamps'][half])}
            write(args.output/'selection.json', selection)
            selection_hash = sha(args.output/'selection.json'); checked(args.output/'selection.json')
        results = {}
        for family, gate in selection['frozenGates'].items():
            score, forecast = evaluate(gate, data, reference['y'])
            lower_order_mean = (data.smoment+forecast['weights']*(low_moments-data.smoment)).sum(axis=1)/forecast['normalizer']
            convergence = float(np.max(abs(forecast['mean']-lower_order_mean))*1e4)
            if convergence > 1e-6:
                raise ValueError('gate mean integration did not converge')
            score['maximum384vs768MeanDifferenceBps'] = convergence
            prediction_file = args.output/f'{split}-{family}.npz'
            np.savez_compressed(prediction_file, **forecast, origins=reference['origins'], timestamps=reference['timestamps'])
            score.update(predictionsFile=str(prediction_file), predictionsSha256=sha(prediction_file))
            if split == 'test':
                score['pairedMseSkillGainCI'] = {name: block_ci(reference['y'], forecast['mean'], baseline, reference['timestamps'])
                    for name, baseline in (('joint', reference['jmean']), ('specialists', reference['smean']))}
            results[family] = score
        output['splits'][split] = {'results': results, 'previousMixtures': old_row['rules'],
            'regionsFile': str(cache_file), 'regionsSha256': sha(cache_file)}
        print(json.dumps({'event': 'scored', 'split': split, 'results': results, 'seconds': time.monotonic()-started}), flush=True)
    for file, digest in hashes.items():
        if sha(file) != digest:
            raise ValueError('source changed during experiment: '+file)
    output.update(selectionFile=str(args.output/'selection.json'), selectionSha256=selection_hash,
        selectedGateFamily=selection['selectedGateFamily'], selectedIncludingControls=selection['selectedIncludingControls'],
        sourceHashes=hashes, seconds=time.monotonic()-started)
    write(args.output/'evaluation.json', output)


if __name__ == '__main__':
    main()
