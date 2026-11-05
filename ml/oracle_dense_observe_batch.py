"""Parallel counterfactual arrivals with deterministic error accumulation.

Each proposal evaluates the existing execution and decision kernels. Only
independent proposals run concurrently; per-cell means and running statistics
are updated in the original proposal order. These errors allocate computation,
not market probabilities or rewards.
"""
import numpy as np
from numba import njit, prange
from oracle_book_mc import decide, move_arrive


@njit(cache=True, parallel=True)
def _residuals(states, scores, indices, price, node, field, wealth, exposures,
               books, counts, offsets, book_lookup, size_edges, price_bins,
               price_scale, rules, candidates, factors):
    result = np.full(len(indices), np.nan)
    for i in prange(len(indices)):
        c = indices[i]
        arrived, consumed, filled, okay = move_arrive(states[c], price, rules)
        if okay:
            continuation, _, _ = decide(arrived, consumed, filled, node, field,
                wealth, exposures, books, counts, offsets, book_lookup,
                size_edges, price_bins, price_scale, rules, candidates, factors)
            result[i] = continuation - scores[c]
    return result


def observe_batched(map, front, price, features, price_pair, feature_std):
    if map.shape[0] < 2:
        return dict(observations=0)
    node = map.cloud.nearest(1, features, price_pair, feature_std)
    a, g = map.a, map.grid
    indices = np.flatnonzero(np.isfinite(front['scores']))
    # The final decision must use actual closing values, as in map.decide.
    field = map.values[1] if map.shape[0] > 2 else np.full_like(map.values[1], -np.inf)
    batch = _residuals(front['states'], front['scores'], indices, price, node,
        field, a['wealth'], a['exposures'], a['books'], a['book_counts'],
        a['code_offsets'], a['book_lookup'], a['size_edges'], g.price_bins,
        g.price_scale, map.rules, np.arange(len(a['coordinates'])), map.cloud.factors(1))
    residuals, by_cell = [], {}
    for c, residual in zip(indices, batch):
        if not np.isfinite(residual):
            continue
        state = front['states'][c]
        w = state[0] + state[1] * state[2]
        orders = []
        for j in range(4, len(state), 3):
            if state[j] != 0:
                x = abs(state[j+1] / state[2] - 1) / g.price_scale
                orders.append((x / (1+x), state[j] * state[j+2] / w))
        cell = g.canonical_cell(state[1] * state[2] / w, orders)
        by_cell.setdefault(cell, []).append(residual)
        residuals.append(residual)
    for cell, rs in by_cell.items():
        map.errors.observe(cell, float(np.mean(rs)))
    return dict(observations=len(residuals), marketNode=node,
        distinctAccountCells=len(by_cell),
        meanResidual=float(np.mean(residuals)) if residuals else None,
        rmsResidual=float(np.sqrt(np.mean(np.square(residuals)))) if residuals else None,
        terminalTimestamp=map.cloud.origin_seconds+map.shape[0],
        interpretation='Correlated counterfactual TD residuals with nearest-cloud continuation; priority only')
