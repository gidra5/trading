"""Two-tail importance proposals for finite conditional Bellman rows.

Controls and guides are fixed before drawing new indices. Nonfinite controls
or missing support witnesses force a census. Reused controls are known finite
vectors, not additional independent observations of the current target.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class SamplePlan:
    indices: np.ndarray
    counts: np.ndarray
    weights: np.ndarray
    proposal: np.ndarray


def proposals(control, exploration=.2, guide=None):
    control = np.asarray(control, dtype=float)
    if control.ndim != 2 or control.shape[1] < 2 or not 0 < exploration <= 1:
        raise ValueError('Need conditional rows and positive exploration at most one')
    finite = np.isfinite(control).all(1)
    safe = np.where(finite[:, None], control, 0.)
    deviation = abs(safe-safe.mean(1, keepdims=True)) if guide is None else np.asarray(guide, dtype=float)
    if deviation.shape != control.shape or np.any(~np.isfinite(deviation) | (deviation < 0)):
        raise ValueError('Guide must be a finite nonnegative row array')
    total = deviation.sum(1, keepdims=True)
    B = control.shape[1]
    tail = np.divide(deviation, total, out=np.full_like(deviation, 1/B), where=total > 0)
    return exploration/B+(1-exploration)*tail


def sample_plan(control, supported, draws, seed, exploration=.2, guide=None):
    control = np.asarray(control, dtype=float)
    q = proposals(control, exploration, guide)
    rows, B = q.shape
    supported = np.asarray(supported, dtype=bool)
    if supported.shape != (rows,) or not isinstance(draws, int) or not 1 <= draws <= B:
        raise ValueError('Invalid support witnesses or draw count')
    partial = supported & np.isfinite(control).all(1) & (draws < B)
    counts = np.where(partial, draws, B).astype(np.int64)
    indices = np.tile(np.arange(B), (rows, 1))
    weights = np.ones_like(q)
    # Shared random numbers help compare actions; each row still has law q.
    u = np.random.default_rng(seed).random(draws)
    cdf = np.cumsum(q, axis=1)
    cdf[:, -1] = 1.
    rows_selected = np.flatnonzero(partial)
    for j in range(draws):
        selected = np.sum(cdf[rows_selected] <= u[j],axis=1)
        indices[rows_selected, j] = selected
        weights[rows_selected, j] = 1/(B*q[rows_selected, selected])
    return SamplePlan(indices, counts, weights, q)


def corrected_estimate(control, evaluated, plan):
    control = np.asarray(control, dtype=float)
    evaluated = np.asarray(evaluated, dtype=float)
    if control.shape != evaluated.shape or control.shape != plan.indices.shape:
        raise ValueError('Outcome and plan shapes differ')
    rows, B = control.shape
    used = np.arange(B)[None,:] < plan.counts[:,None]
    finite = np.all(~used | np.isfinite(evaluated),axis=1)
    safe_y = np.where(used & np.isfinite(evaluated),evaluated,0.)
    safe_control = np.where(np.isfinite(control),control,0.)
    selected = np.take_along_axis(safe_control,plan.indices,axis=1)
    correction = np.where(used,plan.weights*(safe_y-selected),0.)
    delta = correction.sum(1)/plan.counts
    partial = plan.counts < B
    estimate = np.where(partial,safe_control.mean(1)+delta,safe_y.mean(1))
    estimate = np.where(finite,estimate,-np.inf)
    variance = np.sum(np.where(used,(correction-delta[:,None])**2,0.),axis=1)
    variance /= np.maximum(plan.counts-1,1)*plan.counts
    variance = np.where(~partial,0.,variance)
    variance = np.where(finite & ((plan.counts>1)|~partial),variance,np.nan)
    refreshed = control.copy()
    rr,jj = np.nonzero(used)
    refreshed[rr,plan.indices[rr,jj]] = evaluated[rr,jj]
    return estimate, variance, refreshed
