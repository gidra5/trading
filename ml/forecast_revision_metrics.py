"""Fixed-target forecast comparisons and common-noise MC diagnostics.

All samples in a pair must use the same price anchor and target timestamp.
Variance ratios describe a scalar expectation correction with an already
accurate old expectation, not Bellman or trajectory-reuse guarantees.
"""
import numpy as np


def summary(values):
    x = np.asarray(values, dtype=np.float64)
    finite = x[np.isfinite(x)]
    if not len(finite):
        return dict(count=len(x), finite=0)
    return dict(count=len(x), finite=len(finite), mean=float(finite.mean()),
        median=float(np.median(finite)), p90=float(np.quantile(finite, .9)),
        p95=float(np.quantile(finite, .95)), p99=float(np.quantile(finite, .99)),
        maximum=float(finite.max()))


def sample_metrics(old_log, new_log):
    """Rows are paired draws of log(target price / fixed old spot)."""
    x, y = np.asarray(old_log), np.asarray(new_log)
    if x.shape != y.shape or x.ndim != 2 or x.shape[1] < 2:
        raise ValueError('Expected matching [case,draw] arrays')
    sx, sy = np.sort(x, axis=1), np.sort(y, axis=1)
    w1 = np.mean(abs(sy-sx), axis=1)
    ks = np.empty(len(x))
    for i in range(len(x)):
        # Evaluate both empirical CDFs at every jump, including ties.
        kx = np.searchsorted(sx[i], sx[i], side='right')/x.shape[1]
        ky = np.searchsorted(sy[i], sy[i], side='right')/x.shape[1]
        ks[i] = max(abs(kx-np.searchsorted(sy[i], sx[i], side='right')/x.shape[1]).max(),
            abs(ky-np.searchsorted(sx[i], sy[i], side='right')/x.shape[1]).max())
    result = dict(w1Bps=w1*1e4, w1OverOldStd=w1/x.std(1).clip(1e-30),
        empiricalKsPercentagePoints=ks*100)
    functions = dict(log_price=lambda z:z, price=lambda z:np.exp(z),
        upside_10bps=lambda z:np.maximum(np.expm1(z)-.001, 0),
        above_10bps=lambda z:(z>np.log1p(.001)).astype(float))
    for name, f in functions.items():
        a, b = f(x), f(y)
        va, vb = a.var(1, ddof=1), b.var(1, ddof=1)
        vd = (b-a).var(1, ddof=1)
        result[name+'DeltaVarianceOverFresh'] = np.divide(vd, vb,
            out=np.full_like(vd, np.nan), where=vb>1e-30)
        result[name+'DeltaVarianceOverIndependent'] = np.divide(vd, va+vb,
            out=np.full_like(vd, np.nan), where=va+vb>1e-30)
    return result


def fixed_target_horizons(origin_times, target_times):
    h = np.asarray(target_times)-np.asarray(origin_times)
    if np.any(h <= 0):
        raise ValueError('Target must still be in the future')
    return h
