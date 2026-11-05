"""Distribution agreement and validation-selected combinations of frozen models.

H=1. Full laws are compared using exact mass on a common refining partition
(including a separate zero atom and both infinite tails). Binned TV/JS are
coarsened distances, not certified exact continuous-law distances. Mixtures
preserve complete laws; agreement abstention is a point-signal diagnostic only.
No test outcomes fit, choose, or adjust a combination rule.
"""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '4'
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit, gammainc, gammaincc, gammaln, ndtr, ndtri, xlogy
import torch

from horizon_price_forecast import load_horizon_forecast
from stochastic_price_forecast import component_log_masses, load_timeline_split


def sha(file):
    return hashlib.sha256(Path(file).read_bytes()).hexdigest()


def basis_cdf(knots, areas, units):
    """Exact integrated triangular basis, including boundary half hats."""
    result = np.zeros((len(knots), len(units)))
    for k, center in enumerate(knots):
        if k:
            width = center-knots[k-1]
            length = np.clip(units-knots[k-1], 0, width)
            result[k] += .5*length**2/width
        if k+1 < len(knots):
            width = knots[k+1]-center
            length = np.clip(units-center, 0, width)
            result[k] += length-.5*length**2/width
    return result/areas[:, None]


def gamma_cdf(packed, x):
    p = expit(packed[:, 1])[:, None]
    values = np.empty((len(packed), len(x)))
    negative = x < 0
    au, ad = packed[:, 4, None], packed[:, 5, None]
    tu, td = packed[:, 2, None]/au, packed[:, 3, None]/ad
    values[:, negative] = (1-p)*gammaincc(ad, -x[None, negative]/td)
    values[:, ~negative] = 1-p+p*gammainc(au, x[None, ~negative]/tu)
    return values


def law_distances(jzero, jcdf, szero, scdf):
    j = np.column_stack((jzero, (1-jzero[:, None])*np.diff(jcdf, axis=1)))
    s = np.column_stack((szero, (1-szero[:, None])*np.diff(scdf, axis=1)))
    if min(j.min(), s.min()) < -2e-12:
        raise ValueError('negative integrated mass')
    j, s = np.maximum(j, 0), np.maximum(s, 0)
    np.testing.assert_allclose(j.sum(axis=1), 1, atol=2e-12)
    np.testing.assert_allclose(s.sum(axis=1), 1, atol=2e-12)
    mid = (j+s)/2
    js = .5*np.sum(xlogy(j, np.divide(j, mid, out=np.ones_like(j), where=mid>0))
                  +xlogy(s, np.divide(s, mid, out=np.ones_like(s), where=mid>0)), axis=1)
    return .5*np.abs(j-s).sum(axis=1), js


def gamma_loglik(packed, y):
    positive = y > 0
    mean = np.where(positive, packed[:, 2], packed[:, 3])
    alpha = np.where(positive, packed[:, 4], packed[:, 5])
    rate = alpha/mean
    magnitude = np.where(y != 0, abs(y), 1.)
    density = alpha*np.log(rate)-gammaln(alpha)+(alpha-1)*np.log(magnitude)-rate*magnitude
    p0, pup = expit(packed[:, 0]), expit(packed[:, 1])
    return np.where(y == 0, np.log(p0), np.log1p(-p0)
        +np.where(positive, np.log(pup), np.log1p(-pup))+density)


def scores(mean, y, loglik=None):
    active = y != 0
    err = np.square(mean-y)
    out = {'skillPercent': float(100*(1-err.sum()/np.square(y).sum())),
        'mseBpsSquared': float(err.mean()*1e8),
        'nonzeroSkillPercent': float(100*(1-err[active].sum()/np.square(y[active]).sum())),
        'zeroPredictionRmseBps': float(np.sqrt(np.square(mean[~active]).mean())*1e4),
        'meanAbsolutePredictionBps': float(np.mean(abs(mean))*1e4)}
    if loglik is not None:
        out['negativeLogLikelihood'] = float(-loglik.mean())
    return out


def mix_likelihood(weight, jll, sll):
    weight = np.broadcast_to(weight, jll.shape)
    with np.errstate(divide='ignore'):
        return np.logaddexp(np.log(weight)+jll, np.log1p(-weight)+sll)


def diagnostics(data):
    j, s, y = data['jmean'], data['smean'], data['y']
    agree = j*s > 0
    jerr, serr = np.square(j-y), np.square(s-y)
    nll_adv = data['jll']-data['sll']
    advantage_bps = (abs(s-y)-abs(j-y))*1e4
    groups = {'all': np.ones(len(y), dtype=bool), 'sameMeanSign': agree,
        'oppositeMeanSign': ~agree, 'unchanged': y == 0, 'nonzero': y != 0,
        'TV_le_0.10': data['tv'] <= .10, 'TV_0.10_to_0.25': (data['tv'] > .10)&(data['tv'] <= .25),
        'TV_gt_0.25': data['tv'] > .25}
    output = {}
    for label, mask in groups.items():
        if not mask.any():
            output[label] = {'examples': 0}
            continue
        output[label] = {'examples': int(mask.sum()), 'fraction': float(mask.mean()),
            'jointLowerSquaredErrorFraction': float(np.mean(jerr[mask] < serr[mask])),
            'specialistLowerSquaredErrorFraction': float(np.mean(serr[mask] < jerr[mask])),
            'jointHigherLikelihoodFraction': float(np.mean(nll_adv[mask] > 0)),
            'sameMseAndNllWinnerFraction': float(np.mean((jerr[mask] < serr[mask]) == (nll_adv[mask] > 0))),
            'errorDifferenceLessThan0.01bpsFraction': float(np.mean(abs(advantage_bps[mask]) < .01)),
            'errorDifferenceAtLeast0.1bpsFraction': float(np.mean(abs(advantage_bps[mask]) >= .1)),
            'likelihoodRatioAtLeast2Fraction': float(np.mean(abs(nll_adv[mask]) >= np.log(2))),
            'jointMseBpsSquared': float(jerr[mask].mean()*1e8),
            'specialistMseBpsSquared': float(serr[mask].mean()*1e8)}
    jc, sc = data['jc'], data['sc']
    classes = lambda c: np.column_stack(((1-c[:, 0])*(1-c[:, 1]), c[:, 0], (1-c[:, 0])*c[:, 1])).argmax(axis=1)
    output['agreement'] = {
        'meanCorrelation': float(np.corrcoef(j, s)[0, 1]),
        'meanSignAgreement': float(agree.mean()),
        'activeSignProbabilityAgreement': float(np.mean((jc[:, 1] >= .5) == (sc[:, 1] >= .5))),
        'modalDownZeroUpAgreement': float(np.mean(classes(jc) == classes(sc))),
        'meanDifferenceBpsQuantiles': np.quantile(abs(j-s)*1e4, [.5, .9, .99]).tolist(),
        'binnedTVQuantiles': np.quantile(data['tv'], [.1, .5, .9, .99]).tolist(),
        'binnedJSNatsQuantiles': np.quantile(data['js'], [.1, .5, .9, .99]).tolist(),
        'maximumTVRefinementIncrease': float(np.max(data['tv']-data['coarse_tv'])),
        'TV010ClassificationChangesOnRefinement': int(np.sum((data['tv'] <= .1) != (data['coarse_tv'] <= .1))),
        'TV025ClassificationChangesOnRefinement': int(np.sum((data['tv'] <= .25) != (data['coarse_tv'] <= .25)))}
    return output


def gate_features(data):
    j, s = data['jc'], data['sc']
    ja, sa = j[:, 1]*j[:, 2]+(1-j[:, 1])*j[:, 3], s[:, 1]*s[:, 2]+(1-s[:, 1])*s[:, 3]
    return np.column_stack((j[:, :2], s[:, :2], np.log(j[:, 2:4]), np.log(s[:, 2:4]),
        data['jmean']*1e4, data['smean']*1e4, abs(data['jmean']-data['smean'])*1e4,
        (data['jmean']*data['smean'] > 0).astype(float), data['tv'], data['js'], ja*1e4, sa*1e4))


def fit_gate(x, j, s, y, ridge):
    center, scale = x.mean(axis=0), np.maximum(x.std(axis=0), 1e-8)
    design = np.column_stack((np.ones(len(x)), (x-center)/scale))
    norm = np.mean(y*y)
    def loss(beta):
        w = expit(design@beta)
        residual = s+w*(j-s)-y
        gradient = design.T@(2*residual*(j-s)*w*(1-w))/(len(y)*norm)
        gradient[1:] += ridge*beta[1:]
        return np.mean(residual**2)/norm+.5*ridge*np.dot(beta[1:], beta[1:]), gradient
    trained = minimize(loss, np.zeros(design.shape[1]), jac=True, method='L-BFGS-B',
        options={'maxiter': 1000, 'ftol': 1e-13, 'gtol': 1e-8})
    if not trained.success:
        raise ValueError('gate fitting failed: '+trained.message)
    return {'center': center.tolist(), 'scale': scale.tolist(), 'beta': trained.x.tolist(), 'ridge': ridge}


def apply_gate(gate, features):
    design = np.column_stack((np.ones(len(features)), (features-gate['center'])/gate['scale']))
    return expit(design@np.asarray(gate['beta']))


def block_ci(y, prediction, baseline, stamps):
    blocks = (stamps-stamps[0])//3_600_000
    totals = np.array([[np.square(y[blocks == k]).sum(),
        (np.square(baseline[blocks == k]-y[blocks == k])-np.square(prediction[blocks == k]-y[blocks == k])).sum()]
        for k in np.unique(blocks)])
    selected = np.random.default_rng(723).integers(0, len(totals), (3000, len(totals)))
    sums = totals[selected].sum(axis=1)
    return {'hourBlocks': len(totals), 'pairedSkillGainPercentile95': np.quantile(100*sums[:, 1]/sums[:, 0], [.025, .975]).tolist()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('data/backtests/oracle-forecast-agreement'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    hashes = {}
    def checked(file, expected=None):
        actual = sha(file)
        if expected is not None and expected != actual:
            raise ValueError('changed source: '+str(file))
        hashes[str(file)] = actual
        return Path(file)
    previous = json.loads(checked('data/backtests/oracle-current-mse-audit/audit.json').read_text())
    current = json.loads(checked('data/backtests/oracle-joint-calibration/evaluation.json').read_text())
    joint_record = json.loads(checked('data/backtests/oracle-online-calibration/joint-log-return-audit.json').read_text())
    checkpoint = checked(joint_record['checkpoint'], joint_record['checkpointSha256'])
    checked(__file__)
    protocol = {'horizonSeconds': 1, 'primaryMetric': 'all-tick log-return mean squared error',
        'baseModelsFrozen': True, 'combinationFitAndSelection': 'fit first validation half; select family/ridge on second half; refit all validation; freeze before test',
        'gateRidges': [.0001, .001, .01, .1, 1.], 'distributionPartition': 'training knots union Gaussian-CDF uniform1024, compared with nested uniform512; zero atom separate',
        'testUsedForSelection': False, 'historicalTestPreviouslyInspected': True,
        'agreementThresholds': {'TV': [.1, .25], 'absoluteErrorDifferenceBps': [.01, .1], 'likelihoodRatio': 2},
        'rules': ['joint', 'specialists', 'equalMixture', 'constantMseMixture', 'constantNllMixture',
            'agreementOnlyMean', 'disagreementOnlyMean', 'conservativeAgreementMean', 'twoRegimeMixture', 'forecastFeatureMixture'],
        'pointOnlyRules': ['agreementOnlyMean', 'disagreementOnlyMean', 'conservativeAgreementMean'],
        'selectionTieOrder': 'listed order; minimize validation MSE; do not change after test'}
    (args.output/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('high')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, payload = load_horizon_forecast(checkpoint, device)
    dataset = Path(payload['plan']['dataset'])
    checked(dataset/'manifest.json', joint_record['datasetManifestSha256'])
    knots, areas = np.asarray(model.basis['knots']), np.asarray(model.basis['basisAreas'])
    units = np.unique(np.r_[knots, np.linspace(0, 1, 1025)])
    coarse = np.searchsorted(units, np.unique(np.r_[knots, np.linspace(0, 1, 513)]))
    coefficients = basis_cdf(knots, areas, units)
    xaxis = model.sigma*ndtri(units)
    all_data = {}
    rules = None
    output = {'protocol': protocol, 'splits': {}}
    for split in ('validation', 'test'):
        record = previous['splits'][split]
        with np.load(checked(record['componentsFile'], record['componentsSha256'])) as cache:
            data = {k: cache[k] for k in ('origins', 'timestamps')}
            data.update(y=cache['logTargets'], jc=cache['jointKnot'], sc=cache['currentSpecialists'])
        cmean = lambda c: (1-c[:, 0])*(c[:, 1]*c[:, 2]-(1-c[:, 1])*c[:, 3])
        data['jmean'], data['smean'] = cmean(data['jc']), cmean(data['sc'])
        row = next(r for r in current['results'] if r['split'] == split and r['horizonSeconds'] == 1)
        saved = row['models']['joint-calibrated']['savedStream']
        with np.load(checked(saved['parametersFile'], saved['parametersSha256'])) as cache:
            indices = cache['accuracyIndices']
            packed = cache['parameters'][indices]
            np.testing.assert_array_equal(cache['origins'][indices], data['origins'])
        timeline = load_timeline_split(dataset, split)
        features = timeline.features(data['origins'])
        tv, js, coarse_tv, jll = [], [], [], []
        for start in range(0, len(features), 2560):
            stop = min(start+2560, len(features))
            with torch.inference_mode():
                logits = model(torch.from_numpy(features[start:stop]).to(device))
                zero = logits[:, 0].sigmoid().cpu().numpy().astype(float)
                masses = component_log_masses(logits[:, 1:], model.areas).exp().cpu().numpy().astype(float)
            masses /= masses.sum(axis=1, keepdims=True)
            np.testing.assert_array_equal(zero, data['jc'][start:stop, 0])
            jcdf = masses@coefficients
            scdf = gamma_cdf(packed[start:stop], xaxis)
            fine_tv, fine_js = law_distances(zero, jcdf, expit(packed[start:stop, 0]), scdf)
            ctv, _ = law_distances(zero, jcdf[:, coarse], expit(packed[start:stop, 0]), scdf[:, coarse])
            if np.min(fine_tv-ctv) < -1e-12:
                raise ValueError('nested partition must not decrease TV')
            tv.append(fine_tv); js.append(fine_js); coarse_tv.append(ctv)
            targets = data['y'][start:stop]
            unit = ndtr(targets/model.sigma)
            interval = np.clip(np.searchsorted(knots, unit, side='right')-1, 0, len(knots)-2)
            fraction = (unit-knots[interval])/(knots[interval+1]-knots[interval])
            heights = masses/areas
            ix = np.arange(len(targets))
            density = heights[ix, interval]*(1-fraction)+heights[ix, interval+1]*fraction
            log_density = np.log(density)-.5*(targets/model.sigma)**2-np.log(model.sigma)-.5*np.log(2*np.pi)
            jll.append(np.where(targets == 0, np.log(zero), np.log1p(-zero)+log_density))
            if start//2560 % 4 == 0:
                print(json.dumps({'event': 'distributionComparison', 'split': split, 'rows': stop,
                    'total': len(features)}), flush=True)
        data.update(tv=np.concatenate(tv), js=np.concatenate(js), coarse_tv=np.concatenate(coarse_tv),
            jll=np.concatenate(jll), sll=gamma_loglik(packed, data['y']))
        np.testing.assert_allclose(-data['sll'].mean(), row['models']['joint-calibrated']['logReturnNegativeLogLikelihood'], atol=1e-12)
        feats = gate_features(data)
        j, s, y = data['jmean'], data['smean'], data['y']
        agree = j*s > 0
        if split == 'validation':
            fit_weight = lambda mask: float(np.clip(np.dot(j[mask]-s[mask], y[mask]-s[mask])/
                max(1e-300, np.square(j[mask]-s[mask]).sum()), 0, 1))
            halfway = len(y)//2
            fit_rows = np.arange(len(y)) < halfway
            tune_rows = ~fit_rows
            fit_nll_weight = lambda mask: minimize_scalar(lambda w: -mix_likelihood(w,
                data['jll'][mask], data['sll'][mask]).mean(), bounds=(0, 1), method='bounded')
            weight_mse = fit_weight(np.ones(len(y), dtype=bool))
            fit_nll = fit_nll_weight(np.ones(len(y), dtype=bool))
            if not fit_nll.success:
                raise ValueError('constant likelihood mixture failed')
            weight_nll = min([0., 1., float(fit_nll.x)], key=lambda w: -mix_likelihood(w, data['jll'], data['sll']).mean())
            gate_trials = []
            for ridge in protocol['gateRidges']:
                gate = fit_gate(feats[:halfway], j[:halfway], s[:halfway], y[:halfway], ridge)
                w = apply_gate(gate, feats[halfway:])
                gate_trials.append({'ridge': ridge, 'firstHalfGate': gate,
                    'tuningMse': float(np.square(s[halfway:]+w*(j[halfway:]-s[halfway:])-y[halfway:]).mean())})
            ridge = min(gate_trials, key=lambda trial: trial['tuningMse'])['ridge']
            nll_first = fit_nll_weight(fit_rows)
            if not nll_first.success:
                raise ValueError('first-half likelihood mixture failed')
            nll_first_weight = min([0., 1., float(nll_first.x)], key=lambda w:
                -mix_likelihood(w, data['jll'][fit_rows], data['sll'][fit_rows]).mean())
            first_half_weights = {'joint': 1., 'specialists': 0., 'equalMixture': .5,
                'constantMseMixture': fit_weight(fit_rows), 'constantNllMixture': nll_first_weight,
                'twoRegimeMixture': np.where(agree, fit_weight(fit_rows & agree), fit_weight(fit_rows & ~agree)),
                'forecastFeatureMixture': apply_gate(min(gate_trials, key=lambda trial: trial['tuningMse'])['firstHalfGate'], feats)}
            tuning_predictions = {name: s+w*(j-s) for name, w in first_half_weights.items()}
            tuning_predictions.update(agreementOnlyMean=np.where(agree, (j+s)/2, 0.),
                disagreementOnlyMean=np.where(~agree, (j+s)/2, 0.),
                conservativeAgreementMean=np.where(agree, np.where(abs(j) < abs(s), j, s), 0.))
            tuning_scores = {name: scores(prediction[tune_rows], y[tune_rows]) for name, prediction in tuning_predictions.items()}
            rules = {'constantMseWeightJoint': weight_mse, 'constantNllWeightJoint': weight_nll,
                'agreementWeightJoint': fit_weight(agree), 'disagreementWeightJoint': fit_weight(~agree),
                'gateTrials': gate_trials, 'gate': fit_gate(feats, j, s, y, ridge),
                'gateFeatureDefinition': 'zero/sign probabilities; log conditional sizes; signed means; absolute mean gap; mean sign agreement; binned TV/JS; active absolute means',
                'gateTrainingHalfEndTimestamp': int(data['timestamps'][halfway-1]),
                'gateTuningHalfFirstTimestamp': int(data['timestamps'][halfway]),
                'heldOutTuningScores': tuning_scores}
        weights = {'joint': 1., 'specialists': 0., 'equalMixture': .5,
            'constantMseMixture': rules['constantMseWeightJoint'], 'constantNllMixture': rules['constantNllWeightJoint'],
            'twoRegimeMixture': np.where(agree, rules['agreementWeightJoint'], rules['disagreementWeightJoint']),
            'forecastFeatureMixture': apply_gate(rules['gate'], feats)}
        predictions = {name: s+w*(j-s) for name, w in weights.items()}
        predictions.update(agreementOnlyMean=np.where(agree, (j+s)/2, 0.),
            disagreementOnlyMean=np.where(~agree, (j+s)/2, 0.),
            conservativeAgreementMean=np.where(agree, np.where(abs(j) < abs(s), j, s), 0.))
        metrics = {name: scores(predictions[name], y, mix_likelihood(weights[name], data['jll'], data['sll'])
            if name in weights else None) for name in protocol['rules']}
        if split == 'validation':
            selected = min(protocol['rules'], key=lambda name: rules['heldOutTuningScores'][name]['mseBpsSquared'])
            selected_distribution = min(weights, key=lambda name: rules['heldOutTuningScores'][name]['mseBpsSquared'])
            rules.update(selectedRule=selected, selectedDistributionRule=selected_distribution,
                validationScores=metrics, frozenBeforeTest=True)
            (args.output/'selection.json').write_text(json.dumps(rules, indent=2)+'\n')
            checked(args.output/'selection.json')
        else:
            chosen = predictions[rules['selectedRule']]
            output['selectedTestUncertainty'] = {base: block_ci(y, chosen, predictions[base], data['timestamps'])
                for base in ('joint', 'specialists')}
        diagnostic = diagnostics(data)
        # Noncausal best-of-two is only an upper diagnostic, never a rule.
        hindsight = np.where(np.square(j-y) <= np.square(s-y), j, s)
        result = {'diagnostics': diagnostic, 'rules': metrics,
            'noncausalPerOutcomeChooser': scores(hindsight, y),
            'featureGateJointWeightQuantiles': np.quantile(weights['forecastFeatureMixture'], [.1, .5, .9]).tolist()}
        file = args.output/f'{split}-predictions.npz'
        np.savez_compressed(file, **data, **{f'rule_{key}': value for key, value in predictions.items()})
        result.update(predictionsFile=str(file), predictionsSha256=sha(file))
        output['splits'][split] = result
        print(json.dumps({'split': split, 'agreement': diagnostic['agreement'], 'rules': metrics,
            'selected': rules['selectedRule']}), flush=True)
        all_data[split] = data
    for file, digest in hashes.items():
        if sha(file) != digest:
            raise ValueError('source changed: '+file)
    output.update(selection=rules, sourceHashes=hashes)
    (args.output/'audit.json').write_text(json.dumps(output, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
