"""Exact aliases within one root query's common law/rules/continuation context.

Only byte-identical float64 post-states are merged. This does not identify
reciprocal problem contexts or geometrically nearby account cells. Callers
must scatter the evaluated outcomes back before applying descriptor-specific
controls or tie breaking. The helper does not mutate the states.
"""
import numpy as np


def exact_root_aliases(states):
    states=np.asarray(states)
    if (states.ndim!=2 or states.dtype!=np.float64 or not len(states)
            or not states.shape[1] or not np.isfinite(states).all()):
        raise ValueError('Require nonempty finite float64 post-state rows')
    # Keep signed zeros distinct as well: the optimization introduces no
    # floating-point identity assumptions beyond identical input bytes.
    keys=np.ascontiguousarray(states).view(np.uint64)
    _,first,inverse=np.unique(keys,axis=0,return_index=True,return_inverse=True)
    order=np.argsort(first)
    rank=np.empty(len(order),dtype=np.int64); rank[order]=np.arange(len(order))
    return first[order],rank[inverse]
