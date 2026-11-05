"""Reuse unchanged finite-graph continuation planes and propagate exact changes.

This schedules the existing Oracle supporting-plane backup. It does not turn
that upper-bound construction into an exact Bellman solver or certify NSF
state projection. Market rows are immutable; shared rows need no comparison.
"""
from dataclasses import dataclass
import time
import numpy as np
from oracle_certified_value import ValuePlanes, terminal_planes
from oracle_order_value import rational_order_backup


@dataclass(frozen=True)
class PlaneGraph:
    prices: tuple[np.ndarray, ...]
    destinations: tuple[np.ndarray, ...]
    points: int = 65
    gross_numerator: int = 2017
    gross_denominator: int = 1000

    def validate(self):
        if len(self.prices) != len(self.destinations)+1 or not self.destinations:
            raise ValueError('A positive finite horizon is required')
        if self.points < 2 or not 2*self.gross_denominator <= self.gross_numerator <= 3*self.gross_denominator:
            raise ValueError('Invalid backup parameters')
        for h, p in enumerate(self.prices):
            if p.ndim != 1 or len(p) < 1 or p.dtype.kind not in 'iu' or np.any(p <= 1):
                raise ValueError('Require positive integer cent prices with a passive buy tick')
            if h == len(self.destinations):
                continue
            ds = self.destinations[h]
            if ds.ndim != 2 or len(ds) != len(p) or ds.shape[1] < 1 or ds.dtype.kind not in 'iu':
                raise ValueError('Invalid conditional rows')
            if np.any(ds < 0) or np.any(ds >= len(self.prices[h+1])):
                raise ValueError('Invalid child index')

    def backup(self, h, child, rows):
        return rational_order_backup(child, self.prices[h+1], self.prices[h][rows],
            self.destinations[h][rows], self.points,
            2 if h == 0 else self.gross_numerator,
            1 if h == 0 else self.gross_denominator)

    def terminal(self):
        base = terminal_planes(len(self.prices[-1]), fee_denominator=2000)
        columns = np.arange(6*self.points+2) % 2
        return ValuePlanes(base.c[:, columns], base.d[:, columns])


@dataclass(frozen=True)
class PlaneCache:
    graph: PlaneGraph
    values: tuple[ValuePlanes, ...]


def changed_rows(old, new):
    if old is new:
        return np.zeros(len(new), dtype=bool)
    if old.shape != new.shape:
        return np.ones(len(new), dtype=bool)
    unequal = old != new
    return unequal if unequal.ndim == 1 else unequal.any(axis=tuple(range(1, unequal.ndim)))


def update_planes(previous, graph, on_layer=None):
    """Recompute exactly those rows whose inputs may have changed.

    Changed prices invalidate both their own backup and incoming transitions.
    Changed destinations invalidate their parents. A changed terminal is
    propagated too, including when an additional second extends the horizon.
    Stop propagation only on bit-identical coefficients, never a small delta.
    Account states remain functions of cash/inventory, not a merged account grid.
    """
    started = time.perf_counter()
    graph.validate()
    old = previous.graph
    if len(previous.values) != len(old.prices):
        raise ValueError('Cache and original graph have different horizons')
    H = len(graph.destinations)
    values = [None]*(H+1)
    values[H] = graph.terminal()
    same_parameters = (graph.points, graph.gross_numerator, graph.gross_denominator) == (
        old.points, old.gross_numerator, old.gross_denominator)
    changed = np.ones(len(graph.prices[H]), dtype=bool)
    if H < len(previous.values):
        prior = previous.values[H]
        changed = changed_rows(prior.c, values[H].c) | changed_rows(prior.d, values[H].d)
        if not changed.any():
            values[H] = prior
    records = []
    for h in range(H-1, -1, -1):
        tick = time.perf_counter()
        ds = graph.destinations[h]
        reusable = same_parameters and h < len(old.destinations) and (
            previous.values[h].c.shape == (len(ds), 6*graph.points+2))
        if reusable:
            current_price_changed = changed_rows(old.prices[h], graph.prices[h])
            child_price_changed = changed_rows(old.prices[h+1], graph.prices[h+1])
            dirty = current_price_changed | changed_rows(old.destinations[h], ds)
            dirty |= (changed | child_price_changed)[ds].any(1)
        else:
            dirty = np.ones(len(ds), dtype=bool)
        rows = np.flatnonzero(dirty)
        changed = np.zeros(len(ds), dtype=bool)
        if len(rows):
            updated = graph.backup(h, values[h+1], rows)
            if reusable:
                prior = previous.values[h]
                changed[rows] = (prior.c[rows] != updated.c).any(1) | (prior.d[rows] != updated.d).any(1)
                if changed.any():
                    c = prior.c.copy(); d = prior.d.copy()
                    c[rows] = updated.c; d[rows] = updated.d
                    values[h] = ValuePlanes(c, d)
                else:
                    values[h] = prior
            else:
                values[h] = updated
                changed[:] = True
        else:
            values[h] = previous.values[h]
        record = dict(depth=h, nodes=len(ds), recomputed=int(len(rows)), changed=int(changed.sum()),
            seconds=time.perf_counter()-tick)
        records.append(record)
        if on_layer is not None:
            on_layer(record)
    return PlaneCache(graph, tuple(values)), dict(seconds=time.perf_counter()-started,
        totalRows=sum(map(len, graph.destinations)), recomputedRows=sum(r['recomputed'] for r in records),
        changedRows=sum(r['changed'] for r in records), layers=records)
