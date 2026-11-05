"""Normalized outcome-dependent pooling of two frozen atom/density forecasts.

Gate weights are constant within declared log-return regions. Original densities
remain intact within each region. Thus bin masses give the normalizer exactly;
signed bin first moments give the mean without representative scenarios. Both
depend on the weights and participate in the normalized likelihood objective.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


RETURN_EDGES = np.array([-np.inf, -5, -1, -.2, -.05, -.01, -.003, 0,
                         .003, .01, .05, .2, 1, 5, np.inf], dtype=np.float64)/1e4
RETURN_EDGES.flags.writeable = False
CONTRACT = 'normalized-outcome-region-distribution-gate-v1'


def outcome_regions(y):
    """Region zero is the discrete atom; continuous regions start at one."""
    if not np.isfinite(y).all():
        raise ValueError('outcomes must be finite')
    return np.where(y == 0, 0, 1+np.searchsorted(RETURN_EDGES[1:-1], y, side='right'))


@dataclass(frozen=True)
class GateData:
    jmass: np.ndarray
    smass: np.ndarray
    jmoment: np.ndarray
    smoment: np.ndarray
    jll: np.ndarray
    sll: np.ndarray
    region: np.ndarray
    features: np.ndarray

    def validate(self):
        n, bins = self.jmass.shape
        if bins != len(RETURN_EDGES) or self.smass.shape != (n, bins):
            raise ValueError('incorrect outcome-region mass shape')
        for value in (self.jmass, self.smass, self.jmoment, self.smoment, self.features, self.jll, self.sll):
            if len(value) != n or not np.isfinite(value).all():
                raise ValueError('inconsistent/nonfinite gate inputs')
        for mass in (self.jmass, self.smass):
            if np.any(mass < -1e-13) or not np.allclose(mass.sum(axis=1), 1, atol=2e-12, rtol=0):
                raise ValueError('source probabilities are not normalized')
        if self.region.shape != (n,) or np.any(self.region < 0) or np.any(self.region >= bins):
            raise ValueError('invalid target regions')

    def subset(self, index):
        return GateData(**{name: getattr(self, name)[index] for name in self.__dataclass_fields__})


def context_features(jc, sc):
    conditional = lambda c: c[:, 1]*c[:, 2]-(1-c[:, 1])*c[:, 3]
    absolute = lambda c: c[:, 1]*c[:, 2]+(1-c[:, 1])*c[:, 3]
    return np.column_stack((jc[:, :2], sc[:, :2], np.log(absolute(jc)), np.log(absolute(sc)),
        conditional(jc)*1e4, conditional(sc)*1e4))


def design_matrix(features, center, scale):
    return np.column_stack((np.ones(len(features)), (features-center)/scale))


def normalized_prediction(logits, data):
    weights = expit(logits)
    unnormalized = data.smass+weights*(data.jmass-data.smass)
    z = unnormalized.sum(axis=1)
    if np.any(z <= 0) or not np.isfinite(z).all():
        raise ValueError('normalizer must be positive and finite')
    row = np.arange(len(z))
    selected = logits[row, data.region]
    log_q = np.logaddexp(-np.logaddexp(0, -selected)+data.jll,
                        -np.logaddexp(0, selected)+data.sll)
    ll = log_q-np.log(z)
    mean = (data.smoment+weights*(data.jmoment-data.smoment)).sum(axis=1)/z
    return {'logLikelihood': ll, 'mean': mean, 'normalizer': z,
            'regionMass': unnormalized/z[:, None], 'weights': weights}


def objective(beta_flat, design, data, prior, ridge):
    """Analytic gradient includes log Z, the observed density, and priors."""
    beta = beta_flat.reshape(prior.shape)
    logits = design@beta
    weights = expit(logits)
    delta = data.jmass-data.smass
    z = (data.smass+weights*delta).sum(axis=1)
    n = len(z)
    row = np.arange(n)
    selected = logits[row, data.region]
    log_j = -np.logaddexp(0, -selected)+data.jll
    log_s = -np.logaddexp(0, selected)+data.sll
    log_q = np.logaddexp(log_j, log_s)
    nll = np.mean(np.log(z)-log_q)
    gradient_logits = delta/z[:, None]*weights*(1-weights)
    gradient_logits[row, data.region] += weights[row, data.region]-np.exp(log_j-log_q)
    gradient = design.T@gradient_logits/n
    change = beta-prior
    # Adjacent continuous regions are regularized; atom has its own coefficient.
    differences = np.diff(beta[:, 1:], axis=1)
    penalty = .5*ridge*(np.mean(change**2)+np.mean(differences**2))
    gradient += ridge*change/change.size
    smooth = ridge*differences/differences.size
    gradient[:, 1:-1] -= smooth
    gradient[:, 2:] += smooth
    return float(nll+penalty), gradient.ravel()


def fit_gate(data, *, context, ridge, initial_weight):
    data.validate()
    if not 0 < initial_weight < 1 or ridge <= 0:
        raise ValueError('positive regularization and interior initialization required')
    features = data.features if context else data.features[:, :0]
    center, scale = features.mean(axis=0), np.maximum(features.std(axis=0), 1e-8)
    design = design_matrix(features, center, scale)
    prior = np.zeros((design.shape[1], data.jmass.shape[1]))
    prior[0] = np.log(initial_weight/(1-initial_weight))
    result = minimize(objective, prior.ravel(), args=(design, data, prior, ridge), jac=True,
        method='L-BFGS-B', options={'maxiter': 1200, 'ftol': 1e-12, 'gtol': 1e-7, 'maxls': 40})
    if not result.success:
        raise RuntimeError('outcome gate optimization failed: '+result.message)
    return {'contract': CONTRACT, 'contextEnabled': context, 'ridge': ridge,
        'initialWeight': initial_weight, 'center': center.tolist(), 'scale': scale.tolist(),
        'coefficients': result.x.reshape(prior.shape).tolist(), 'iterations': int(result.nit),
        'finalObjective': float(result.fun), 'gradientInfinityNorm': float(np.max(abs(result.jac)))}


def predict_gate(gate, data):
    if gate['contract'] != CONTRACT:
        raise ValueError('unsupported outcome gate')
    features = data.features if gate['contextEnabled'] else data.features[:, :0]
    design = design_matrix(features, np.asarray(gate['center']), np.asarray(gate['scale']))
    return normalized_prediction(design@np.asarray(gate['coefficients']), data)
