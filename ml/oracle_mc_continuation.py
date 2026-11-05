"""Monte Carlo continuation updates on a cached finite market law.

Plain sampled backups are the default. Optional paired corrections use the
same children for old/new backups and center at the scalar cash-value control
variate. Nonlinear support selection means neither construction is claimed an
unbiased Bellman-value estimator. Error is measured against the full backup;
a census recovers that backup. Estimates are not certified upper bounds for
the original, unsampled law.
"""
from dataclasses import dataclass
import time
import numpy as np
from oracle_certified_value import ValuePlanes
from oracle_incremental_value import PlaneCache, PlaneGraph
from oracle_order_value import rational_order_backup


@dataclass(frozen=True)
class MonteCarloPlan:
    samples: int
    seed: int


def sampled_destinations(destinations, samples, seed, depth):
    """Nested uniform samples without replacement within each conditional row.

    Random streams are keyed by depth. All sample counts share the same
    permutation prefix. Sample IDs do not become policy observations.
    """
    branches = destinations.shape[1]
    if not 1 <= samples <= branches:
        raise ValueError('Sample count must be within the finite conditional support')
    if samples == branches:
        return destinations
    rng = np.random.default_rng(np.random.SeedSequence([seed, depth]))
    order = np.argsort(rng.random(destinations.shape), axis=1)[:, :samples]
    return np.take_along_axis(destinations, order, axis=1)


def sampled_backup(graph, depth, child, destinations):
    return rational_order_backup(child, graph.prices[depth+1], graph.prices[depth],
        destinations, graph.points, 2 if depth == 0 else graph.gross_numerator,
        1 if depth == 0 else graph.gross_denominator)


def correction(old_full, new_sampled, old_sampled):
    c = old_full.c + (new_sampled.c-old_sampled.c)
    d = old_full.d + (new_sampled.d-old_sampled.d)
    target = old_full.c.min(1) + new_sampled.c.min(1)-old_sampled.c.min(1)
    if np.any(c <= 0) or np.any(d <= 0) or np.any(target <= 0):
        raise ArithmeticError('Nonpositive Monte Carlo continuation coefficient')
    # min(c_old + delta_c) is generally NOT min(c_old) + delta_min(c).
    # Preserve the paired scalar estimate at cash instead of introducing a
    # spurious level shift whenever the minimizing support face changes.
    # Scale both coordinates together, retaining the estimated exposure shape.
    scale = target/c.min(1)
    c = np.rint(c*scale[:, None]).astype(np.int64)
    d = np.rint(d*scale[:, None]).astype(np.int64)
    return ValuePlanes(c, d)


def estimate_extension(cache: PlaneCache, extended: PlaneGraph, plan: MonteCarloPlan,
                       *, paired=False, on_layer=None):
    """Update an appended horizon with fixed, reused earlier market rows.

    Existing price/transition arrays must be shared immutable references. This
    prevents accidentally treating a changed forecast law as an unchanged
    control variate. The new tail is evaluated in full; earlier corrections
    are sampled. Only one live layer is needed, apart from the existing cache.
    The callback can retain selected rows for independent action checks.
    """
    old = cache.graph
    if len(extended.destinations) <= len(old.destinations):
        raise ValueError('Expected an appended horizon')
    if (old.points, old.gross_numerator, old.gross_denominator) != (
            extended.points, extended.gross_numerator, extended.gross_denominator):
        raise ValueError('Changed backup parameters require another baseline')
    if any(a is not b for a, b in zip(old.prices, extended.prices)) or any(
            a is not b for a, b in zip(old.destinations, extended.destinations)):
        raise ValueError('This estimator requires the unchanged finite prefix')
    if not 1 <= plan.samples <= min(ds.shape[1] for ds in old.destinations):
        raise ValueError('Invalid sample count')
    started = time.perf_counter(); sampling_seconds = baseline_seconds = update_seconds = 0.
    value = extended.terminal(); B = old.destinations[0].shape[1]
    boundary = len(old.destinations); records = []; sampled_edges = 0
    for h in range(len(extended.destinations)-1, -1, -1):
        tick = time.perf_counter()
        ds = (sampled_destinations(extended.destinations[h], plan.samples, plan.seed, h)
              if h < boundary else extended.destinations[h])
        sampling_seconds += time.perf_counter()-tick
        tick = time.perf_counter()
        updated = sampled_backup(extended, h, value, ds)
        update_seconds += time.perf_counter()-tick
        if paired and h < boundary and plan.samples < B:
            tick = time.perf_counter()
            baseline = sampled_backup(old, h, cache.values[h+1], ds)
            baseline_seconds += time.perf_counter()-tick
            value = correction(cache.values[h], updated, baseline)
        else:
            value = updated
        sampled_edges += ds.size
        # The estimator is not allowed to quietly repair its output into a
        # different algorithm. Record violations for the independent evaluator.
        wedge_violations = int(((1999*value.d > 2000*value.c) | (1999*value.c > 2000*value.d)).sum())
        record = dict(depth=h, conditionalSamples=ds.shape[1], wedgeViolations=wedge_violations,
            elapsedSeconds=time.perf_counter()-started)
        records.append(record)
        if on_layer is not None:
            on_layer(h, value, record)
    elapsed = time.perf_counter()-started
    return value, dict(samplesPerNode=plan.samples, seed=plan.seed, paired=paired,
        census=plan.samples == B, sampledEdges=sampled_edges,
        seconds=elapsed, samplingSeconds=sampling_seconds,
        oldSampledBackupSeconds=baseline_seconds, newSampledBackupSeconds=update_seconds,
        potentialWarmUpdateSeconds=elapsed-baseline_seconds,
        warmTimingIsEstimate=paired and plan.samples < B, totalWedgeViolations=sum(r['wedgeViolations'] for r in records),
        layers=records, certifiedUpper=False, unbiasedClaim=False)
