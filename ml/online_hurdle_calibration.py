"""Causal, discounted online likelihood calibration of frozen hurdle forecasts.

Only four coefficients adapt: two probability-logit intercepts, an upward
log-mean scale, and a downward log-odds-mean scale. Each matured observation
adds a local Fisher quadratic of its likelihood. This is an online
approximation, not an assertion of exact batch-likelihood optimization.
Queries never mutate state or add calibration observations.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral

import numpy as np
from scipy.special import digamma, polygamma

CONTRACT = "causal-online-hurdle-calibration-v1"
MINIMUM_MEAN = 1e-12
MAXIMUM_DOWN_MEAN = 1-1e-12
MODES = ("none", "zero", "probabilities", "full")
COEFFICIENT_BOUNDS = np.array([2., 2., math.log(2.), math.log(2.)])
COEFFICIENT_BOUNDS.setflags(write=False)


@dataclass(frozen=True)
class CalibrationConfig:
    horizon_seconds: int
    mode: str
    half_life_seconds: float = 14400.
    ridge_precision: float = 8.
    cadence_seconds: int = 4

    def __post_init__(self):
        if not isinstance(self.horizon_seconds, Integral) or isinstance(self.horizon_seconds, bool) or self.horizon_seconds <= 0:
            raise ValueError("positive integer horizon required")
        if not isinstance(self.cadence_seconds, Integral) or isinstance(self.cadence_seconds, bool) or self.cadence_seconds <= 0:
            raise ValueError("positive integer calibration cadence required")
        if self.mode not in MODES:
            raise ValueError("unsupported calibration mode")
        if not math.isfinite(self.half_life_seconds) or self.half_life_seconds <= 0 \
                or not math.isfinite(self.ridge_precision) or self.ridge_precision <= 0:
            raise ValueError("finite positive half-life and ridge precision required")

    @property
    def observation_weight(self) -> float:
        return min(1., self.cadence_seconds/self.horizon_seconds)


def sigmoid(x: float) -> float:
    if x >= 0:
        return 1/(1+math.exp(-x))
    value = math.exp(x)
    return value/(1+value)


def validate_raw(raw: np.ndarray) -> None:
    if raw.shape != (6,) or not all(math.isfinite(float(x)) for x in raw):
        raise ValueError("raw forecast must be finite packed [zeroLogit,signLogit,up,down,alphaUp,alphaDown]")
    if raw[2] < MINIMUM_MEAN or not MINIMUM_MEAN <= raw[3] <= MAXIMUM_DOWN_MEAN \
            or not 0 < raw[4] <= 100 or not 0 < raw[5] <= 100:
        raise ValueError("raw mean or shape lies outside the supported physical range")


def transform_raw(raw: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    """Construct changed parameters without modifying the frozen base forecast."""
    result = raw.astype(np.float64, copy=True)
    result[:2] += coefficients[:2]
    if coefficients[2] != 0:
        result[2] = max(MINIMUM_MEAN, float(raw[2])*math.exp(float(coefficients[2])))
    if coefficients[3] != 0:
        odds_scaled = sigmoid(math.log(float(raw[3]))-math.log1p(-float(raw[3]))+float(coefficients[3]))
        result[3] = min(MAXIMUM_DOWN_MEAN, max(MINIMUM_MEAN, odds_scaled))
    return result


def beta_log_odds_fisher(alpha: float, beta: float) -> tuple[float, float]:
    """Return beta*(psi(beta+alpha)-psi(beta)) and its log-odds Fisher.

Large-beta expressions use differentiated log-Gamma-ratio expansions, avoiding
cancellation in digamma and trigamma differences. The model bounds alpha by
100. Tests compare values/gradients with independent high-precision arithmetic.
"""
    if (beta > 1e4 and alpha/beta < 1e-3) or (beta > 1000 and alpha <= 2):
        inv = 1/beta
        a = alpha
        score = (a-a*(a-1)*inv/2+a*(2*a*a-3*a+1)*inv*inv/6
                 -a*a*(a-1)*(a-1)*inv**3/4)
        fisher = (a-a*(a-1)*inv+a*(2*a*a-3*a+1)*inv*inv/2
                  -a*a*(a-1)*(a-1)*inv**3)
    else:
        score = beta*float(digamma(beta+alpha)-digamma(beta))
        fisher = beta*beta*float(polygamma(1, beta)-polygamma(1, beta+alpha))
    if not math.isfinite(score) or not math.isfinite(fisher) or fisher <= 0:
        raise ValueError("invalid Beta calibration Fisher information")
    return score, fisher


def likelihood_gradient_fisher(raw: np.ndarray, coefficients: np.ndarray,
                               realized_return: float, mode: str) -> tuple[np.ndarray, np.ndarray]:
    """Per-observation NLL gradient and expected information for enabled heads."""
    adjusted = transform_raw(raw, coefficients)
    gradient, information = np.zeros(4), np.zeros(4)
    if mode == "none":
        return gradient, information
    p0 = sigmoid(float(adjusted[0]))
    gradient[0], information[0] = p0-float(realized_return == 0), p0*(1-p0)
    if mode == "zero" or realized_return == 0:
        return gradient, information
    pup = sigmoid(float(adjusted[1]))
    gradient[1], information[1] = pup-float(realized_return > 0), pup*(1-pup)
    if mode == "probabilities":
        return gradient, information
    if realized_return > 0:
        # Below the numerical mean floor, the represented likelihood is locally
        # constant in this coefficient. At equality use the interior derivative.
        if float(raw[2])*math.exp(float(coefficients[2])) >= MINIMUM_MEAN:
            alpha, mean = float(adjusted[4]), float(adjusted[2])
            gradient[2], information[2] = alpha*(1-realized_return/mean), alpha
    else:
        down = float(adjusted[3])
        unbounded_logit = math.log(float(raw[3]))-math.log1p(-float(raw[3]))+float(coefficients[3])
        floor_logit = math.log(MINIMUM_MEAN)-math.log1p(-MINIMUM_MEAN)
        cap_logit = math.log(MAXIMUM_DOWN_MEAN)-math.log1p(-MAXIMUM_DOWN_MEAN)
        if floor_logit <= unbounded_logit <= cap_logit:
            alpha = float(adjusted[5])
            beta = alpha*(1-down)/down
            score, fisher = beta_log_odds_fisher(alpha, beta)
            gradient[3] = score+beta*math.log1p(realized_return)
            information[3] = fisher
    return gradient, information


class OnlineHurdleCalibrator:
    """Private mutable sufficient statistics; forecast inputs/configs stay fixed.

`update` timestamps a label at its actual origin+horizon maturity. The caller
must only submit labels available by the current query. `apply` rejects a query
earlier than an incorporated maturity, detecting premature updates. Queries
age the event-anchored state without mutating it, so query frequency cannot
change subsequent forecasts.
"""

    def __init__(self, config: CalibrationConfig):
        self.config = config
        self.information = np.zeros(4)
        self.linear = np.zeros(4)
        self.last_origin_ms: int | None = None
        self.last_maturity_ms: int | None = None
        self.observations = 0
        self.component_updates = np.zeros(4, dtype=np.int64)
        self.boundary_updates = np.zeros(4, dtype=np.int64)

    @staticmethod
    def _timestamp(value: int) -> int:
        if not isinstance(value, Integral) or isinstance(value, bool) or value < 0:
            raise ValueError("nonnegative integer timestamp required")
        return int(value)

    def _decay_at(self, timestamp_ms: int) -> float:
        t = self._timestamp(timestamp_ms)
        if self.last_maturity_ms is None:
            return 1.
        if t < self.last_maturity_ms:
            raise ValueError("query precedes a label already incorporated; future-information leak")
        return math.exp(-math.log(2)*(t-self.last_maturity_ms)/(1000*self.config.half_life_seconds))

    def coefficients_at(self, timestamp_ms: int) -> np.ndarray:
        decay = self._decay_at(timestamp_ms)
        return np.clip(decay*self.linear/(decay*self.information+self.config.ridge_precision),
                       -COEFFICIENT_BOUNDS, COEFFICIENT_BOUNDS)

    def update(self, origin_timestamp_ms: int, raw: np.ndarray, realized_return: float) -> None:
        origin = self._timestamp(origin_timestamp_ms)
        if self.last_origin_ms is not None and origin <= self.last_origin_ms:
            raise ValueError("each calibration origin must be processed once in increasing order")
        validate_raw(raw)
        y = float(realized_return)
        if not math.isfinite(y) or y <= -1:
            raise ValueError("realized simple return must be finite and greater than -1")
        maturity = origin+1000*self.config.horizon_seconds
        decay = self._decay_at(maturity)
        theta = self.coefficients_at(maturity)
        gradient, fisher = likelihood_gradient_fisher(raw, theta, y, self.config.mode)
        if not np.isfinite(gradient).all() or not np.isfinite(fisher).all() or (fisher < 0).any():
            raise ValueError("nonfinite likelihood calibration update")
        weight = self.config.observation_weight
        self.information = decay*self.information+weight*fisher
        self.linear = decay*self.linear+weight*(fisher*theta-gradient)
        self.last_origin_ms, self.last_maturity_ms = origin, maturity
        self.observations += 1
        self.component_updates += (fisher > 0)
        unconstrained = self.linear/(self.information+self.config.ridge_precision)
        self.boundary_updates += (np.abs(unconstrained) > COEFFICIENT_BOUNDS)

    def apply(self, timestamp_ms: int, raw: np.ndarray) -> np.ndarray:
        validate_raw(raw)
        return transform_raw(raw, self.coefficients_at(timestamp_ms))

    def snapshot(self, timestamp_ms: int) -> dict:
        decay = self._decay_at(timestamp_ms)
        return {"timestamp": int(timestamp_ms), "lastMaturedOrigin": self.last_origin_ms,
                "latestMaturityUsed": self.last_maturity_ms, "maturedObservations": self.observations,
                "coefficients": self.coefficients_at(timestamp_ms).tolist(),
                "discountedInformation": (decay*self.information).tolist(),
                "componentUpdates": self.component_updates.tolist(), "boundaryUpdates": self.boundary_updates.tolist(),
                "observationWeight": self.config.observation_weight, "mode": self.config.mode}
