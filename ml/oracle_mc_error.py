"""Numerical uniform error of log plane envelopes, modulo market-only offsets.

On each interval between either envelope's breakpoints, their ratio is a
fractional linear function. Its extrema occur at interval endpoints. Thus the
union of hull breakpoints suffices; no exposure grid is used for this check.
These floating-point diagnostics are not Lean/exact-rational certificates.
"""
import numpy as np
from numba import njit
from oracle_certified_value import plane_hulls


@njit(cache=True)
def _interval_errors(c0, d0, c1, d1, ids0, cuts0, sizes0, ids1, cuts1, sizes1, limit):
    absolute = np.zeros(len(c0)); centered = np.zeros(len(c0))
    for n in range(len(c0)):
        cash_shift = np.log(c1[n].min()/c0[n].min())
        points = np.empty(sizes0[n]+sizes1[n]+2)
        size = 2; points[0] = -limit; points[1] = limit
        for j in range(1, sizes0[n]):
            x = cuts0[n, j]
            if -limit < x < limit:
                points[size] = x; size += 1
        for j in range(1, sizes1[n]):
            x = cuts1[n, j]
            if -limit < x < limit:
                points[size] = x; size += 1
        for k in range(size):
            x = points[k]; f0 = 0; f1 = 0
            while f0+1 < sizes0[n] and cuts0[n, f0+1] <= x:
                f0 += 1
            while f1+1 < sizes1[n] and cuts1[n, f1+1] <= x:
                f1 += 1
            j0 = ids0[n, f0]; j1 = ids1[n, f1]
            v0 = c0[n, j0]+x*(d0[n, j0]-c0[n, j0])
            v1 = c1[n, j1]+x*(d1[n, j1]-c1[n, j1])
            if v0 <= 0 or v1 <= 0:
                absolute[n] = np.inf; centered[n] = np.inf
            else:
                delta = np.log(v1/v0)
                absolute[n] = max(absolute[n], abs(delta))
                centered[n] = max(centered[n], abs(delta-cash_shift))
    return absolute, centered


def log_envelope_errors(reference, estimate, limit=3.):
    if limit <= 0 or not np.isfinite(limit) or len(reference.c) != len(estimate.c):
        raise ValueError('Invalid common account interval')
    return _interval_errors(reference.c, reference.d, estimate.c, estimate.d,
        *plane_hulls(reference.c, reference.d), *plane_hulls(estimate.c, estimate.d), limit)


def action_loss_bound_percent(centered_log_error):
    """2*epsilon argmax bound expressed as relative CE loss.

    Requires identical feasible actions and exogenous market transitions, and
    every evaluated continuation exposure to lie in the checked interval.
    """
    if np.any(np.asarray(centered_log_error) < 0):
        raise ValueError('An error bound cannot be negative')
    return 100*(-np.expm1(-2*np.asarray(centered_log_error)))
