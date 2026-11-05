"""Checked reuse of an unchanged Bellman prefix after repairing its new tail.

Residuals belong to individual time layers, so earlier repair seams are not
forgotten on the next update. The bound concerns this finite numerical model.
It does not bound market quantization, interpolation or action discretization.
"""
import numpy as np


def contract(seed,candidates):
    return dict(seed=int(seed),candidates=int(candidates),schedule='absolute-time-full-census-terminal-close-v2')


def validate_repair(current,previous,seed,candidates):
    H=current.shape[0];a=previous.cloud;b=current.cloud
    shrinking=previous.shape[0]==H+1
    if (previous.shape[1:]!=current.shape[1:] or previous.shape[0] not in (H,H+1)
        or b.origin_seconds!=a.origin_seconds+1):
        raise ValueError('Repair requires a one-second shift with the same field shape')
    if previous.bellman_contract!=contract(seed,candidates) or previous.bellman_residuals is None:
        raise ValueError('Repair requires matching saved full-census schedules and residual accounting')
    if previous.grid!=current.grid or not np.array_equal(previous.rules,current.rules):
        raise ValueError('Repair account contracts differ')
    end=H+1 if shrinking else H
    for key in ('prices','counts','features','price_pairs'):
        if not np.array_equal(getattr(b,key)[1:end],getattr(a,key)[2:end+1]):
            raise ValueError('Repair requires unchanged interior market nodes')
    for key in ('destinations','sampled_prices'):
        if not np.array_equal(getattr(b,key)[1:end-1],getattr(a,key)[2:end]):
            raise ValueError('Repair requires unchanged interior conditional rows')
    residuals=previous.bellman_residuals
    if residuals.shape!=(previous.shape[0],) or np.any(~np.isfinite(residuals)|(residuals<0)):
        raise ValueError('Invalid saved Bellman residuals')
    # Double-precision allowance for comparisons of affine interpolation and
    # sequential conditional means. Execution/log terms are fixed constants
    # in these comparisons. 1500 covers the log range of finite float64 wealth
    # and its normalization; the field's actual magnitude is added explicitly.
    finite=previous.values[np.isfinite(previous.values)]
    magnitude=float(np.max(abs(finite))) if len(finite) else 0.
    eta=32*b.destinations.shape[-1]*np.finfo(np.float64).eps*(1500.+magnitude)
    return eta


def repair_bound(previous_residuals,boundary,difference,roundoff_per_layer):
    """Root error allowance if layers 1..boundary-1 are copied from old 2..boundary.

    The new boundary and its complete tail have already been recomputed.
    The root is always recomputed. Same finite support is checked separately.
    """
    return float(np.sum(previous_residuals[2:boundary+1])+difference+(boundary+1)*roundoff_per_layer)


def copy_residuals(previous_residuals,boundary,difference,roundoff_per_layer):
    result=np.zeros_like(previous_residuals)
    result[1:boundary]=previous_residuals[2:boundary+1]
    result[boundary-1]+=difference+roundoff_per_layer
    return result
