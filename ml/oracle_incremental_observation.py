"""Observed counterfactual innovations for computation priorities only.

All proposals see the same actual next price and the SAME old terminal date.
The continuation field and future-action menu are those of the preceding root
evaluation. This avoids treating a changed forecast or extended horizon as an
observed reward error. Market-state projection still makes this approximate.
"""
import numpy as np
from oracle_dense_metadata import prepared
from oracle_local_market_refresh import candidate_schedule
from oracle_selected_outcomes import SelectedOutcomes


def observe(table,front,price,features,price_pair,feature_std):
    if table.H<2: return dict(observations=0)
    selected = np.flatnonzero(np.isfinite(front['scores']))
    if not len(selected): return dict(observations=0)
    states = front['states'][selected]
    node = table.cloud.nearest(1,features,price_pair,feature_std)
    prices = np.full((len(states),table.B),float(price))
    children = np.full(prices.shape,node,dtype=np.int64)
    factors = np.zeros((table.M,table.B)); raw = table.cloud.factors(1); factors[:len(raw)] = raw
    evaluator = SelectedOutcomes(table.facade(),states,prices,children,table.field(1),factors,
        candidate_schedule(table.facade(),0,table.seed,table.candidates),table.H)
    # Each row evaluates exactly one observed outcome, not eight copies of it.
    actual = evaluator.evaluate(np.zeros(prices.shape,dtype=np.int64),np.ones(len(states),dtype=np.int64))[:,0]
    residual = actual-front['scores'][selected]
    finite = np.isfinite(residual); groups = {}; grid = prepared(table.facade()).grid
    for state,error in zip(states[finite],residual[finite]):
        wealth = state[0]+state[1]*state[2]; orders = []
        for j in range(4,len(state),3):
            if state[j]!=0:
                distance = abs(state[j+1]/state[2]-1)/grid.price_scale
                orders.append((distance/(1+distance),state[j]*state[j+2]/wealth))
        try:
            cell = grid.canonical_cell(state[1]*state[2]/wealth,orders)
        except ValueError:
            continue  # A valid held account can lie outside the interpolation domain.
        wi = int(np.argmin(abs(np.log(table.a['wealth']/wealth))))
        groups.setdefault(wi*table.C+cell,[]).append(float(error))
    for cell,errors in groups.items(): table.errors.observe(cell,float(np.mean(errors)))
    return dict(observations=int(finite.sum()),nonfiniteOutcomes=int((~np.isfinite(actual)).sum()),
        distinctAccountCells=len(groups),marketNode=node,
        meanResidual=float(residual[finite].mean()) if finite.any() else None,
        rmsResidual=float(np.sqrt(np.mean(np.square(residual[finite])))) if finite.any() else None,
        terminalTimestamp=float(table.cloud.origin_seconds+table.H),
        interpretation='Correlated observed-price innovations against the old field/menu; compute priority only, not probability weights or a global value correction')
