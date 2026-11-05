"""Attainable root actions before evaluating their stochastic values.

Fixed descriptor IDs allow cached control vectors to follow changing account
states without assuming that a descriptor always executes the same trade.
"""
import numpy as np
from oracle_book_mc import execute_target, close_terminal, effective
from oracle_dense_metadata import prepared


def root_candidates(map, arrived, consumed=0., filled=False, interior_samples=1, seed=0):
    if interior_samples < 0: raise ValueError('Negative interior sample count')
    a,g = map.a,prepared(map).grid
    E,C,K = len(a['exposures']),len(a['coordinates']),g.max_orders
    rng = np.random.default_rng(seed)
    states,ids = [],[]
    def append(identifier,state,okay):
        if okay: ids.append(identifier); states.append(state)
    for cell in range(C):
        book,ei = divmod(cell,E)
        for sample in range(interior_samples+1):
            if sample == 0:
                exposure,orders,count = a['exposures'][ei],a['books'][book],a['book_counts'][book]
            else:
                exposure,coordinates = g.sample_cell(cell,rng)
                count = len(coordinates); orders = np.zeros_like(a['books'][book])
                if count: orders[:count] = coordinates
            state,okay = execute_target(arrived,consumed,filled,exposure,orders,count,map.rules,g.price_scale)
            append(cell*(interior_samples+1)+sample,state,okay)
    base = C*(interior_samples+1)
    occupied = sum(1<<j for j in range(K) if arrived[4+3*j]!=0)
    for mask in range(1<<K):
        if mask & ~occupied: continue
        for ei in range(-1,E):
            state,okay = execute_target(arrived,consumed,filled,np.nan if ei<0 else a['exposures'][ei],
                a['books'][0],0,map.rules,g.price_scale,mask)
            append(base+mask*(E+1)+ei+1,state,okay)
    closing = base+(1<<K)*(E+1)
    if np.isfinite(close_terminal(arrived,consumed,map.rules)):
        state = arrived.copy(); state[0] += effective(arrived,map.rules); state[1] = 0.; state[4:] = 0.
        append(closing,state,True)
    if not states: raise RuntimeError('No attainable root proposals')
    return np.asarray(ids,dtype=np.int64),np.asarray(states),closing+1
