"""Optional next-close utility enclosure; not yet used by the live planner.

For positive X in [lo,hi], Taylor's curvature bounds and E[X-mu]=0 give
 log(mu)-Var(X)/(2 lo^2) <= E[log X] <= log(mu)-Var(X)/(2 hi^2).
Float64 padding below is an operation-scale allowance, not directed rounding.
"""
import numpy as np
from numba import njit
from oracle_book_mc import move_arrive,effective,allowed_size


@njit(cache=True)
def close_interval(state, factors, rules, tolerance=1e-9):
    """Enclose next-tick closing utility when no order can fill in the row.

    Endpoint survival suffices here: cash and inventory remain constant during
    marking, effective exposure is monotone, and close flow has fixed sign.
    A fill at either extreme disables this shortcut. General fills use the
    existing full calculation. Returns (accepted, lower, upper).
    """
    prices=np.maximum(1.,np.rint(state[2]*factors/rules[9]))*rules[9]
    low_price=np.min(prices);high_price=np.max(prices)
    low,low_used,low_filled,low_ok=move_arrive(state,low_price,rules)
    high,high_used,high_filled,high_ok=move_arrive(state,high_price,rules)
    if not low_ok or not high_ok or low_filled or high_filled:
        return False,-np.inf,np.inf
    low_flow=effective(low,rules);high_flow=effective(high,rules)
    if (not allowed_size(low_flow,rules) or not allowed_size(high_flow,rules)
        or low_used+abs(low_flow)>rules[8]+1e-8 or high_used+abs(high_flow)>rules[8]+1e-8):
        return False,-np.inf,np.inf
    endpoint1=low[0]+low_flow;endpoint2=high[0]+high_flow
    minimum=min(endpoint1,endpoint2);maximum=max(endpoint1,endpoint2)
    if minimum<=0 or not np.isfinite(maximum):return False,-np.inf,np.inf
    coefficient=state[1]*rules[10]*((1-rules[1]) if state[1]>=0 else 1/(1-rules[1]))
    mean_price=np.mean(prices);mean=state[0]+coefficient*mean_price
    variance=coefficient**2*np.mean((prices-mean_price)**2)
    center=np.log(mean)
    padding=64*np.finfo(np.float64).eps*(1+abs(center)+(abs(state[0])+abs(coefficient*mean_price))/minimum)
    lower=center-.5*variance/minimum**2-padding
    upper=center-.5*variance/maximum**2+padding
    accepted=np.isfinite(lower) and np.isfinite(upper) and upper>=lower and upper-lower<=tolerance
    return accepted,lower,upper
