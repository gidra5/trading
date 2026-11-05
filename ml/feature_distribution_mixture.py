"""Separate simplex pools of sign probabilities and full conditional size laws.

Magnitude means are sufficient only for the mean calculation. Likelihood pools
the original conditional densities, not Gamma parameters or point outcomes.
Prediction weights never depend on the outcome being scored.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from expanded_distribution_mixture import fit_weights, pool_loglik

LAYOUTS = {'ternaryMagnitude': ('sign3', 'magnitude'),
           'zeroSignMagnitude': ('zero', 'sign', 'magnitude'),
           'zeroSignSides': ('zero', 'sign', 'upMagnitude', 'downMagnitude')}


def classes(c):
    z, p = c[..., 0], c[..., 1]
    return np.stack(((1-z)*(1-p), z, (1-z)*p), axis=-1)


def validate(c):
    if c.ndim != 3 or c.shape[-1] != 4 or not np.isfinite(c).all():
        raise ValueError('expected finite [sample, model, (zero, up, upMean, downMean)]')
    if np.any((c[..., :2] <= 0) | (c[..., :2] >= 1)) or np.any(c[..., 2:] <= 0):
        raise ValueError('source laws need interior sign masses and positive magnitude means')


def predict(c, weights, layout):
    """Return normalized sign masses and both conditional magnitude means."""
    w = np.asarray(weights)
    if w.shape != (len(LAYOUTS[layout]), c.shape[1]) or np.any(w < 0) or not np.allclose(w.sum(1), 1, atol=1e-10, rtol=0):
        raise ValueError('each feature needs its own probability simplex')
    if layout == 'ternaryMagnitude':
        cat = np.einsum('nmc,m->nc', classes(c), w[0])
        u, d = c[..., 2]@w[1], c[..., 3]@w[1]
    else:
        z, p = c[..., 0]@w[0], c[..., 1]@w[1]
        cat = np.column_stack(((1-z)*(1-p), z, (1-z)*p))
        u = c[..., 2]@w[2]
        d = c[..., 3]@w[2 if layout == 'zeroSignMagnitude' else 3]
    return {'classProbability': cat, 'upMean': u, 'downMean': d,
            'mean': cat[:, 2]*u-cat[:, 0]*d,
            'absoluteMean': cat[:, 2]*u+cat[:, 0]*d}


def conditional_loglik(c, ll, y):
    cat = classes(c)
    target = np.sign(y).astype(int)+1
    label_mass = cat[np.arange(len(y)), :, target]
    return np.where(y[:, None] != 0, ll-np.log(label_mass), 0.)


def loglik(c, ll, y, weights, layout):
    prediction = predict(c, weights, layout)
    target = np.sign(y).astype(int)+1
    result = np.log(prediction['classProbability'][np.arange(len(y)), target])
    cond = conditional_loglik(c, ll, y)
    mag = pool_loglik(cond, weights[1 if layout == 'ternaryMagnitude' else 2])
    if layout == 'zeroSignSides':
        mag = np.where(y > 0, mag, pool_loglik(cond, weights[3]))
    return result+np.where(y != 0, mag, 0.)


def fit_nll(c, ll, y, layout):
    target = np.sign(y).astype(int)+1
    active = y != 0
    fit = lambda logp: fit_weights(np.zeros_like(logp), logp, np.zeros(len(logp)), 'nll')
    if layout == 'ternaryMagnitude':
        weights = [fit(np.log(classes(c)[np.arange(len(y)), :, target]))]
    else:
        weights = [fit(np.log(np.where(active[:, None], 1-c[..., 0], c[..., 0]))),
                   fit(np.log(np.where(y[active, None] > 0, c[active, :, 1], 1-c[active, :, 1])))]
    cond = conditional_loglik(c, ll, y)
    masks = [y > 0, y < 0] if layout == 'zeroSignSides' else [active]
    weights.extend(fit(cond[mask]) for mask in masks)
    return np.asarray(weights)


def mse_objective(flat, c, y, layout):
    """Analytic derivative through the reconstructed expected log return."""
    w = flat.reshape(len(LAYOUTS[layout]), c.shape[1])
    # SLSQP may evaluate off the simplex while computing its search direction.
    up, down = c[..., 2]*1e4, c[..., 3]*1e4
    if layout == 'ternaryMagnitude':
        cat = classes(c)
        pos, neg = cat[..., 2]@w[0], cat[..., 0]@w[0]
        u, d = up@w[1], down@w[1]
        mean = pos*u-neg*d
        jac = [cat[..., 2]*u[:, None]-cat[..., 0]*d[:, None],
               pos[:, None]*up-neg[:, None]*down]
    else:
        z, p = c[..., 0]@w[0], c[..., 1]@w[1]
        u = up@w[2]
        d = down@w[2 if layout == 'zeroSignMagnitude' else 3]
        conditional = p*u-(1-p)*d
        mean = (1-z)*conditional
        jac = [-conditional[:, None]*c[..., 0], ((1-z)*(u+d))[:, None]*c[..., 1]]
        if layout == 'zeroSignMagnitude':
            jac.append((1-z)[:, None]*(p[:, None]*up-(1-p)[:, None]*down))
        else:
            jac.extend([((1-z)*p)[:, None]*up, -((1-z)*(1-p))[:, None]*down])
    target = y*1e4
    norm = np.mean(target**2)
    if norm <= 0:
        raise ValueError('MSE fit requires nonzero outcomes')
    error = mean-target
    gradient = np.stack([2*(error[:, None]*j).mean(axis=0)/norm for j in jac])
    return float(np.mean(error**2)/norm), gradient.ravel()


def fit_mse(c, ll, y, layout):
    """Small nonconvex fit: deterministic multistart, no global-optimum claim."""
    count, models = len(LAYOUTS[layout]), c.shape[1]
    starts = [fit_nll(c, ll, y, layout), np.full((count, models), 1/models)]
    starts.extend(np.tile(v, (count, 1)) for v in np.eye(models))
    equality = np.kron(np.eye(count), np.ones((1, models)))
    trials = []
    for start in starts:
        fitted = minimize(mse_objective, start.ravel(), args=(c, y, layout), jac=True,
            method='SLSQP', bounds=[(0, 1)]*(count*models),
            constraints=[{'type': 'eq', 'fun': lambda w: equality@w-1,
                          'jac': lambda w: equality}],
            options={'maxiter': 600, 'ftol': 1e-11})
        if not fitted.success:
            raise RuntimeError('feature mixture fit: '+fitted.message)
        w = np.maximum(fitted.x.reshape(count, models), 0)
        w /= w.sum(axis=1, keepdims=True)
        trials.append((mse_objective(w.ravel(), c, y, layout)[0], w))
    loss, w = min(trials, key=lambda item: item[0])
    if loss > min(mse_objective(s.ravel(), c, y, layout)[0] for s in starts)+1e-9:
        raise RuntimeError('fit worsened its starting baselines')
    return w


def full_prediction(c, weights):
    cat = classes(c)
    result = np.einsum('nmc,m->nc', cat, weights)
    upmoment = (cat[..., 2]*c[..., 2])@weights
    downmoment = (cat[..., 0]*c[..., 3])@weights
    return {'classProbability': result, 'upMean': upmoment/result[:, 2],
            'downMean': downmoment/result[:, 0], 'mean': upmoment-downmoment,
            'absoluteMean': upmoment+downmoment}


def metrics(prediction, y, ll):
    from audit_forecast_agreement import scores
    out = scores(prediction['mean'], y, ll)
    cat = prediction['classProbability']
    np.testing.assert_allclose(cat.sum(1), 1, atol=1e-10, rtol=0)
    target = np.sign(y).astype(int)+1
    active = y != 0
    p = cat[:, 2]/(cat[:, 0]+cat[:, 2])
    signll = np.log(cat[np.arange(len(y)), target])
    true_side_mean = np.where(y > 0, prediction['upMean'], prediction['downMean'])
    out.update(examples=len(y), nonzeroExamples=int(active.sum()),
        zeroBrier=float(np.mean((cat[:, 1]-(~active))**2)),
        ternaryBrier=float(np.mean(np.sum((cat-np.eye(3)[target])**2, axis=1))),
        ternaryNll=float(-signll.mean()),
        activeSignAccuracyPercent=float(100*np.mean((p[active] >= .5) == (y[active] > 0))),
        activeSignNll=float(-np.log(np.where(y[active] > 0, p[active], 1-p[active])).mean()),
        conditionalMagnitudeNll=float(-(ll-signll)[active].mean()),
        conditionalMagnitudeMseBpsSquared=float(np.mean((true_side_mean[active]-abs(y[active]))**2)*1e8),
        allTickMagnitudeMseBpsSquared=float(np.mean((prediction['absoluteMean']-abs(y))**2)*1e8))
    np.testing.assert_allclose(out['negativeLogLikelihood'],
        out['ternaryNll']+active.mean()*out['conditionalMagnitudeNll'], atol=1e-10)
    return out
