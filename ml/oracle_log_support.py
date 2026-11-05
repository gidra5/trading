"""One-arrival liquidation witnesses for the unbounded NSF price support.

This checks existing inventory before pending fills, matching Oracle's phase
ordering. Absence of a witness is not a proof that the pending book or future
terminal close is feasible throughout the distribution's support.
"""
import numpy as np
from oracle_book_mc import move_arrive


def liquidation_witness(state, rules, minimum_price=.01):
    q, asset, price = state[:3]
    asset *= rules[10]
    tick = rules[9]
    if not (q+asset*price > 0 and minimum_price > 0 and rules[5] < 0 and rules[6] > 1):
        raise ValueError('Require a positive-equity account and maintenance bounds surrounding [0,1]')
    if asset < 0:
        leverage = -rules[5]
        threshold = leverage*q*(1-rules[1])/((1+leverage)*(-asset))
        witness = max(minimum_price, (np.ceil(max(price, threshold)/tick)+2)*tick)
        direction = 'upper'
    elif asset > 0 and q < 0:
        leverage = rules[6]
        threshold = -leverage*q/((leverage-1)*(1-rules[1])*asset)
        witness = max(minimum_price, (np.floor(threshold/tick)-2)*tick)
        direction = 'lower'
    else:
        return None
    if not np.isfinite(witness):
        raise ValueError('Liquidation witness exceeds finite arithmetic')
    arrived, _, _, valid = move_arrive(np.asarray(state, dtype=float), float(witness), rules)
    if valid:
        return None  # E.g. the maintenance threshold is below the model's minimum price.
    if arrived[0] != 0 or arrived[1] != 0:
        raise AssertionError('Expected pre-fill liquidation, not an invalid order-size transition')
    return dict(direction=direction, maintenanceThreshold=float(threshold), witnessPrice=float(witness),
        currentPrice=float(price), witnessRelativeMove=float(witness/price-1), liquidatedEquity=0.)
