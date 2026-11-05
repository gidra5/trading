"""Explain the matched one-second MSE gap without fitting or selecting models.

Compares the surviving joint knot density to the latest input/output calibrated
log specialists. Component swaps are diagnostics, not a deployable selection.
"""
from __future__ import annotations

import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = '4'

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.special import expit, ndtri
import torch

from audit_joint_log_returns import array_sha, score_means
from horizon_price_forecast import load_horizon_forecast
from stochastic_price_forecast import component_log_masses, load_timeline_split


def sha(file):
    return hashlib.sha256(Path(file).read_bytes()).hexdigest()


def side_coefficients(model, positive, order):
    knots = np.asarray(model.basis['knots'])
    areas = np.asarray(model.basis['basisAreas'])
    nodes, weights = np.polynomial.legendre.leggauss(order)
    left = np.maximum(knots[:-1], .5) if positive else knots[:-1]
    right = knots[1:] if positive else np.minimum(knots[1:], .5)
    half = np.maximum(right-left, 0)/2
    center = np.where(half > 0, (right+left)/2, .5)
    unit = center[:, None]+half[:, None]*nodes
    fraction = (unit-knots[:-1, None])/np.diff(knots)[:, None]
    result = []
    for value in (np.ones_like(unit), np.abs(model.sigma*ndtri(unit))):
        coefficients = np.zeros(len(knots))
        coefficients[:-1] += np.sum(value*(1-fraction)*weights, axis=1)*half
        coefficients[1:] += np.sum(value*fraction*weights, axis=1)*half
        result.append(coefficients/areas)
    return result


def components_mean(c):
    conditional = c[:, 1]*c[:, 2]-(1-c[:, 1])*c[:, 3]
    return conditional, (1-c[:, 0])*conditional


def score(c, y, constants):
    conditional, deployed = components_mean(c)
    active = y != 0
    baseline = np.square(y).sum()
    error = np.square(deployed-y)
    result = score_means(y, conditional, deployed)
    p = np.clip(c[:, 1], 1e-14, 1-1e-14)
    result.update({
        'zeroContributionSkillPoints': 100*np.square(deployed[~active]).sum()/baseline,
        'zeroOnlyMeanPredictionRmseBps': float(np.sqrt(np.square(deployed[~active]).mean())*1e4),
        'zeroOnlyConditionalPredictionRmseBps': float(np.sqrt(np.square(conditional[~active]).mean())*1e4),
        'zeroBrier': float(np.square(c[:, 0]-(~active)).mean()),
        'activeSignAccuracy': float(np.mean((p[active] >= .5) == (y[active] > 0))),
        'activeSignLogLoss': float(-np.where(y[active] > 0, np.log(p[active]), np.log1p(-p[active])).mean()),
        'positiveSizeSkill': 1-float(np.square(c[y > 0, 2]-y[y > 0]).sum()/np.square(constants['upMeanLog']-y[y > 0]).sum()),
        'negativeSizeSkill': 1-float(np.square(c[y < 0, 3]+y[y < 0]).sum()/np.square(constants['downMeanLog']+y[y < 0]).sum()),
    })
    np.testing.assert_allclose(result['allTickDeployedMseSkillVsZero']*100,
        result['activeDeployedMseSkillVsZero']*100-result['zeroContributionSkillPoints'], atol=1e-12)
    # Exact additive SSE attribution by realized return magnitude; descriptive,
    # not information provided to the forecasting model.
    groups = [ ('zero', ~active), ('nonzeroBelow1bp', active & (np.abs(y) < 1e-4)),
        ('1to5bp', (np.abs(y) >= 1e-4) & (np.abs(y) < 5e-4)), ('atLeast5bp', np.abs(y) >= 5e-4)]
    result['outcomeGroups'] = {label: {'examples': int(mask.sum()),
        'errorAsSkillPoints': float(100*error[mask].sum()/baseline)} for label, mask in groups}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('data/backtests/oracle-current-mse-audit'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    sources = {}

    def checked(file, digest=None):
        actual = sha(file)
        if digest is not None and actual != digest:
            raise ValueError('source hash changed: '+str(file))
        sources[str(file)] = actual
        return Path(file)

    old = json.loads(checked('data/backtests/oracle-online-calibration/joint-log-return-audit.json').read_text())
    current = json.loads(checked('data/backtests/oracle-joint-calibration/evaluation.json').read_text())
    checkpoint = checked(old['checkpoint'], old['checkpointSha256'])
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('high')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, payload = load_horizon_forecast(checkpoint, device)
    dataset = Path(payload['plan']['dataset'])
    checked(dataset/'manifest.json', old['datasetManifestSha256'])
    upmass, upmoment = side_coefficients(model, True, 768)
    downmass, downmoment = side_coefficients(model, False, 768)
    _, low_up = side_coefficients(model, True, 384)
    _, low_down = side_coefficients(model, False, 384)
    result = {'contract': 'current-log-specialists-vs-joint-knot-mse-audit-v1',
        'horizonSeconds': 1, 'changesTrainingOrSelection': False,
        'currentVariant': 'joint-calibrated', 'componentOrder': ['pZero', 'pUpGivenActive', 'upLogMean', 'downLogMean'],
        'sourceTrainingObjective': payload['plan']['trainingLoss'], 'splits': {}}
    for split in ('validation', 'test'):
        row = next(r for r in current['results'] if r['split'] == split and r['horizonSeconds'] == 1)
        saved = row['models']['joint-calibrated']['savedStream']
        with np.load(checked(saved['parametersFile'], saved['parametersSha256'])) as stream:
            ix = stream['accuracyIndices']
            packed = stream['parameters'][ix]
            origins = stream['origins'][ix]
            timestamps = stream['timestamps'][ix]
        manifest = row['marketManifest']
        with np.load(checked(manifest['marketFile'], manifest['marketSha256'])) as market:
            offsets = origins-int(market['firstPriceOrigin'])
            simple_y = market['prices'][offsets+1]/market['prices'][offsets]-1
            np.testing.assert_array_equal(origins, market['commonOrigins'])
        y = np.log1p(simple_y)
        sample_hashes = {'origins': array_sha(origins), 'timestamps': array_sha(timestamps),
            'simpleTargets': array_sha(simple_y), 'logTargets': array_sha(y)}
        if sample_hashes != row['sampleHashes']:
            raise ValueError('population mismatch')
        specialists = np.column_stack((expit(packed[:, 0]), expit(packed[:, 1]), packed[:, 2:4]))
        timeline = load_timeline_split(dataset, split)
        np.testing.assert_array_equal(timestamps, timeline.axis_start_ms+origins*1000)
        features = timeline.features(origins)
        zeros, mass_parts = [], []
        with torch.inference_mode():
            for start in range(0, len(origins), 2560):
                logits = model(torch.from_numpy(features[start:start+2560]).to(device))
                zeros.append(logits[:, 0].sigmoid().cpu().numpy().astype(float))
                mass_parts.append(component_log_masses(logits[:, 1:], model.areas).exp().cpu().numpy().astype(float))
        masses = np.concatenate(mass_parts)
        masses /= masses.sum(axis=1, keepdims=True)
        p_up, p_down = masses@upmass, masses@downmass
        np.testing.assert_allclose(p_up+p_down, 1., rtol=0, atol=3e-12)
        joint = np.column_stack((np.concatenate(zeros), p_up, (masses@upmoment)/p_up, (masses@downmoment)/p_down))
        conditional, _ = components_mean(joint)
        np.testing.assert_allclose(conditional, masses@model._component_moments('log', 768), atol=1e-16, rtol=0)
        convergence = float(np.max(np.abs(conditional-masses@(low_up-low_down)))*1e4)
        if convergence > 1e-6:
            raise ValueError('conditional moment quadrature did not converge')
        scores = {name: score(c, y, row['constants']) for name, c in [('jointKnot', joint), ('currentSpecialists', specialists)]}
        np.testing.assert_allclose(scores['currentSpecialists']['allTickDeployedMseSkillVsZero'],
            row['models']['joint-calibrated']['logReturns']['allTickDeployedMseSkillVsZero'], atol=2e-12, rtol=0)
        if split == 'test':
            if sample_hashes != old['sampleHashes']:
                raise ValueError('old joint audit population differs')
            np.testing.assert_allclose(scores['jointKnot']['allTickDeployedMseSkillVsZero'],
                old['logReturns']['allTickDeployedMseSkillVsZero'], atol=2e-12, rtol=0)
        swaps = {}
        for zero_from_joint, sign_from_joint, sizes_from_joint in itertools.product((False, True), repeat=3):
            c = np.column_stack(((joint if zero_from_joint else specialists)[:, 0],
                (joint if sign_from_joint else specialists)[:, 1],
                (joint if sizes_from_joint else specialists)[:, 2:4]))
            name = ''.join('J' if flag else 'S' for flag in (zero_from_joint, sign_from_joint, sizes_from_joint))
            swaps[name] = score(c, y, row['constants'])
        forecast_file = args.output/f'{split}-components.npz'
        np.savez_compressed(forecast_file, origins=origins, timestamps=timestamps, logTargets=y,
            jointKnot=joint, currentSpecialists=specialists)
        result['splits'][split] = {'sampleHashes': sample_hashes, 'scores': scores,
            'swapsZeroSignSizes': swaps, 'maximum384vs768ConditionalMeanDifferenceBps': convergence,
            'componentsFile': str(forecast_file), 'componentsSha256': sha(forecast_file)}
        print(json.dumps({'split': split, 'scores': scores,
            'swapSkillsPercent': {k: v['allTickDeployedMseSkillVsZero']*100 for k, v in swaps.items()}}), flush=True)
    checked(__file__)
    result['sourceHashes'] = sources
    for file, digest in sources.items():
        if sha(file) != digest:
            raise ValueError('source changed during audit')
    (args.output/'audit.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
