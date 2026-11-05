"""Causal online calibration of a signed-Gamma law for LOG returns.

The state clock, forgetting, ridge prior, and mode settings are shared with the
simple-return experiment. The magnitude likelihood is different: both signed
log magnitudes are Gamma. UP mean/shape odds are adjusted to keep E[exp(X)]
finite; DOWN mean scales without imposing a simple-return bound on log returns.
"""
from __future__ import annotations

import math

import numpy as np

from online_hurdle_calibration import (COEFFICIENT_BOUNDS, MODES, CalibrationConfig,
    OnlineHurdleCalibrator, sigmoid)
from log_hurdle_sign_size_forecast import MINIMUM_MEAN, MAXIMUM_UP_MEAN_RATIO

CONTRACT = "causal-online-log-hurdle-calibration-v1"


def validate_raw(raw: np.ndarray) -> None:
    if raw.shape != (6,) or not all(math.isfinite(float(x)) for x in raw):
        raise ValueError("raw log forecast must be finite packed [zeroLogit,signLogit,meanLogUp,meanLogDown,alphaUp,alphaDown]")
    if not 0 < raw[4] <= 100 or not 0 < raw[5] <= 100 \
            or not MINIMUM_MEAN <= raw[2] <= MAXIMUM_UP_MEAN_RATIO*raw[4] or raw[3] < MINIMUM_MEAN:
        raise ValueError("raw log mean or shape violates positivity or finite UP exponential-moment support")


def _up_logit(mean: float, alpha: float) -> float:
    return math.log(mean)-math.log(alpha-mean)


def transform_raw(raw: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    """Return a changed private forecast; identity and untouched heads are exact."""
    result = raw.astype(np.float64,copy=True)
    result[:2] += coefficients[:2]
    if coefficients[2] != 0:
        alpha = float(raw[4])
        mean = alpha*sigmoid(_up_logit(float(raw[2]),alpha)+float(coefficients[2]))
        result[2] = min(MAXIMUM_UP_MEAN_RATIO*alpha,max(MINIMUM_MEAN,mean))
    if coefficients[3] != 0:
        result[3] = max(MINIMUM_MEAN,float(raw[3])*math.exp(float(coefficients[3])))
    return result


def likelihood_gradient_fisher(raw: np.ndarray, coefficients: np.ndarray,
                               realized_return: float, mode: str) -> tuple[np.ndarray,np.ndarray]:
    """NLL gradient and expected Fisher; realized_return is a LOG return.

For a Gamma(alpha,mean), dNLL/dlog(mean)=alpha*(1-y/mean). UP logit
parameterization contributes dlog(mean)/dtheta=1-mean/alpha, and Fisher
therefore scales by its square. No observed-Hessian substitution is used.
"""
    adjusted = transform_raw(raw,coefficients)
    gradient,information = np.zeros(4),np.zeros(4)
    if mode == "none":
        return gradient,information
    p0 = sigmoid(float(adjusted[0]))
    gradient[0],information[0] = p0-float(realized_return == 0),p0*(1-p0)
    if mode == "zero" or realized_return == 0:
        return gradient,information
    pup = sigmoid(float(adjusted[1]))
    gradient[1],information[1] = pup-float(realized_return > 0),pup*(1-pup)
    if mode == "probabilities":
        return gradient,information
    if realized_return > 0:
        alpha,mean = float(adjusted[4]),float(adjusted[2])
        unbounded_logit = _up_logit(float(raw[2]),alpha)+float(coefficients[2])
        # Compare in the same logit coordinates so exact boundary values do not
        # accidentally fall outside after a sigmoid round trip.
        floor_logit = _up_logit(MINIMUM_MEAN,alpha)
        cap_logit = _up_logit(MAXIMUM_UP_MEAN_RATIO*alpha,alpha)
        if floor_logit <= unbounded_logit <= cap_logit:
            derivative = 1-mean/alpha
            gradient[2] = alpha*(1-realized_return/mean)*derivative
            information[2] = alpha*derivative*derivative
    elif float(raw[3])*math.exp(float(coefficients[3])) >= MINIMUM_MEAN:
        alpha,mean = float(adjusted[5]),float(adjusted[3])
        gradient[3],information[3] = alpha*(1-abs(realized_return)/mean),alpha
    return gradient,information


class OnlineLogHurdleCalibrator(OnlineHurdleCalibrator):
    """Use the original event-anchored state machinery with log-law updates.

Only update/apply depend on the return law. Inherited timestamp validation,
read-only queries, wall-time decay, coefficient bounds, and snapshots retain
the previously tested causal behavior without modifying the original module.
"""

    def update(self, origin_timestamp_ms: int, raw: np.ndarray, realized_return: float) -> None:
        origin = self._timestamp(origin_timestamp_ms)
        if self.last_origin_ms is not None and origin <= self.last_origin_ms:
            raise ValueError("each calibration origin must be processed once in increasing order")
        validate_raw(raw)
        y = float(realized_return)
        if not math.isfinite(y):
            raise ValueError("realized LOG return must be finite")
        maturity = origin+1000*self.config.horizon_seconds
        decay = self._decay_at(maturity)
        theta = self.coefficients_at(maturity)
        gradient,fisher = likelihood_gradient_fisher(raw,theta,y,self.config.mode)
        if not np.isfinite(gradient).all() or not np.isfinite(fisher).all() or (fisher < 0).any():
            raise ValueError("nonfinite log likelihood calibration update")
        weight = self.config.observation_weight
        self.information = decay*self.information+weight*fisher
        self.linear = decay*self.linear+weight*(fisher*theta-gradient)
        self.last_origin_ms,self.last_maturity_ms = origin,maturity
        self.observations += 1
        self.component_updates += fisher > 0
        unconstrained = self.linear/(self.information+self.config.ridge_precision)
        self.boundary_updates += np.abs(unconstrained) > COEFFICIENT_BOUNDS

    def apply(self, timestamp_ms: int, raw: np.ndarray) -> np.ndarray:
        validate_raw(raw)
        return transform_raw(raw,self.coefficients_at(timestamp_ms))
