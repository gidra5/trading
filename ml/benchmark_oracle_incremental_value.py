"""Measure exact cache updates against a complete backward recomputation.

Append one genuine NSF second to the saved graph, preserving all existing
nodes and conditional rows. This isolates horizon-update cost; it is not a
historical backtest or a test of adapting graph coverage to a new observation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from numba import set_num_threads
from oracle_certified_value import ValuePlanes, SCALE
from oracle_incremental_value import PlaneGraph, PlaneCache, update_planes
from latent_neural_stochastic_flow import load_latent_nsf
from export_nsf_planning_trees import next_observations
from nsf_scenario_graph import representatives


def digest(path):
    with Path(path).open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


@torch.inference_mode()
def extend_market(model, graph, plan, counts, features, price_pairs):
    started = time.perf_counter()
    H = len(graph.destinations)
    n = len(graph.prices[-1]); samples = plan['samples']
    # The same unconditional representative-fitting masses as a full build.
    mass = np.ones(1)
    for h, ds in enumerate(graph.destinations):
        mass = np.bincount(ds.reshape(-1), weights=np.repeat(mass/samples, samples),
            minlength=len(graph.prices[h+1]))
    f = np.asarray(features[H, :n]); p = np.asarray(price_pairs[H, :n])/100
    origin = plan['originTimestampSeconds']
    t = np.broadcast_to(np.array([origin+H-1, origin+H]), (n, 2))
    future_f, future_p = next_observations(model, f, p, t, np.ones(n), samples, plan['seed']+H)
    pairs_f = np.stack((np.repeat(f[:, -1], samples, axis=0), future_f.reshape(-1, 59)), axis=1)
    pairs_p = np.stack((np.repeat(p[:, -1], samples), future_p.reshape(-1)), axis=1)
    device = next(model.parameters()).device
    standardized = (torch.as_tensor(pairs_f, device=device)-model.feature_mean)/model.feature_std
    coordinates = torch.cat((standardized.double().reshape(len(pairs_f), 118)/np.sqrt(118),
        torch.as_tensor(np.log(pairs_p/plan['originPrice'])/.0001, device=device)), dim=1)
    weights = torch.as_tensor(np.repeat(mass/samples, samples), device=device)
    selected, assignment = representatives(coordinates, weights, plan['width'])
    chosen = selected.cpu().numpy()
    price = np.rint(pairs_p[chosen, -1]*100).astype(np.int64)
    ds = assignment.reshape(n, samples).cpu().numpy().astype(np.int32)
    extended_prices = graph.prices+(price,)
    low = min(int(p.min()) for p in extended_prices); high = max(int(p.max()) for p in extended_prices)
    denominator = 3*low-2*high
    if denominator <= 0:
        raise ValueError('The extended price envelope has no solvent leverage bound')
    required = (2*high*graph.gross_denominator+denominator-1)//denominator
    extended = PlaneGraph(extended_prices, graph.destinations+(ds,), points=graph.points,
        gross_numerator=max(graph.gross_numerator, required), gross_denominator=graph.gross_denominator)
    return extended, dict(seconds=time.perf_counter()-started, newNodes=len(price),
        generatedOutcomes=n*samples, unchangedExistingMarketRows=sum(map(len, graph.destinations)),
        leverageBoundChanged=extended.gross_numerator != graph.gross_numerator), pairs_f[chosen], pairs_p[chosen]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--graph', type=Path, required=True)
    ap.add_argument('--value', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args(); args.output.mkdir()
    torch.set_num_threads(4); set_num_threads(4); torch.set_float32_matmul_precision('high')
    gp = json.loads((args.graph/'plan.json').read_text())
    vr = json.loads((args.value/'report.json').read_text())
    if digest(args.graph/'graph.npz') != vr['graphSha256']:
        raise ValueError('Cache belongs to another market graph')
    if not vr['reachableLeverageRefinement'] or not vr['predictableOrderCoverage']:
        raise ValueError('Require the rational order-aware cache')
    for name in ('oracle_order_value.py', 'oracle_certified_value.py'):
        if digest(Path(__file__).parent/name) != vr['sourceHashes'][name]:
            raise ValueError('Cached backup implementation changed')
    data = np.load(args.graph/'graph.npz'); counts = data['counts']
    pp = data['price_ticks']; dd = data['destinations']
    graph = PlaneGraph(tuple(pp[h, :n] for h, n in enumerate(counts)),
        tuple(dd[h, :n] for h, n in enumerate(counts[:-1])), points=vr['supportingPoints'],
        gross_numerator=vr['reachableLeverageNumerator'], gross_denominator=vr['reachableLeverageDenominator'])
    cc = np.load(args.value/'post-arrival-c.npy', mmap_mode='r')
    dc = np.load(args.value/'post-arrival-d.npy', mmap_mode='r')
    cache = PlaneCache(graph, tuple(ValuePlanes(cc[h, :n], dc[h, :n]) for h, n in enumerate(counts)))
    # Warm the numerical kernels before timing cache updates.
    warm = graph.backup(0, cache.values[1], np.array([0]))
    np.testing.assert_array_equal(warm.c, cache.values[0].c)
    np.testing.assert_array_equal(warm.d, cache.values[0].d)
    snapshots = args.output/'source'; snapshots.mkdir()
    names = ('benchmark_oracle_incremental_value.py', 'oracle_incremental_value.py',
        'oracle_order_value.py', 'oracle_certified_value.py', 'nsf_scenario_graph.py',
        'export_nsf_planning_trees.py', 'latent_neural_stochastic_flow.py')
    hashes = {}
    for name in names:
        path = Path(__file__).parent/name
        (snapshots/name).write_bytes(path.read_bytes()); hashes[name] = digest(path)
    checkpoint = Path('data/backtests/oracle-latent-nsf-sampled-skips/latentSampledSkipFlow.pt')
    if digest(checkpoint) != gp['checkpointSha256']:
        raise ValueError('Forecast checkpoint changed')
    plan = dict(graphSha256=vr['graphSha256'], checkpointSha256=gp['checkpointSha256'],
        sourceHashes=hashes, arguments={k: str(v) for k, v in vars(args).items()},
        scope='Exact cache scheduling for existing plane algorithm; one new NSF second, no historical trading or state-projection certificate')
    (args.output/'plan.json').write_text(json.dumps(plan, indent=2))
    _, unchanged = update_planes(cache, graph)
    print(json.dumps(dict(case='unchanged', **{k: v for k, v in unchanged.items() if k != 'layers'})), flush=True)
    # A localized change of one empirical probability row, not a new history.
    root_ds = graph.destinations[0].copy()
    root_ds[0, 0] = (root_ds[0, 0]+1) % len(graph.prices[1])
    localized = PlaneGraph(graph.prices, (root_ds,)+graph.destinations[1:],
        graph.points, graph.gross_numerator, graph.gross_denominator)
    local_cache, local_stats = update_planes(cache, localized)
    expected = localized.backup(0, cache.values[1], np.array([0]))
    np.testing.assert_array_equal(local_cache.values[0].c, expected.c)
    np.testing.assert_array_equal(local_cache.values[0].d, expected.d)
    local_stats['allUnaffectedLayersShared'] = all(a is b for a, b in zip(local_cache.values[1:], cache.values[1:]))
    print(json.dumps(dict(case='localized_root_law', **{k: v for k, v in local_stats.items() if k != 'layers'})), flush=True)
    model, _ = load_latent_nsf(checkpoint, torch.device('cuda'))
    extended, extension, features, prices = extend_market(model, graph, gp, counts,
        np.load(args.graph/'features.npy', mmap_mode='r'),
        np.load(args.graph/'observation-price-ticks.npy', mmap_mode='r'))
    np.savez_compressed(args.output/'extension.npz', features=features, price_pairs=prices,
        price_ticks=extended.prices[-1], destinations=extended.destinations[-1])
    print(json.dumps(dict(case='market_extension', **extension)), flush=True)
    start = time.perf_counter()
    def progress(record):
        if record['depth'] % 300 == 0:
            print(json.dumps(dict(case='incremental', depth=record['depth'],
                changed=record['changed'], recomputed=record['recomputed'], seconds=time.perf_counter()-start)), flush=True)
    updated, stats = update_planes(cache, extended, progress)
    (args.output/'incremental.json').write_text(json.dumps(stats, indent=2))
    # Full independent sweep, comparing ALL coefficients at ALL states.
    start = time.perf_counter(); current = extended.terminal(); mismatches = []
    for h in range(len(extended.destinations)-1, -1, -1):
        current = extended.backup(h, current, np.arange(len(extended.prices[h])))
        if not (np.array_equal(current.c, updated.values[h].c) and np.array_equal(current.d, updated.values[h].d)):
            mismatches.append(h)
        if h % 300 == 0:
            print(json.dumps(dict(case='full_reference', depth=h, seconds=time.perf_counter()-start)), flush=True)
    full_seconds = time.perf_counter()-start
    report = dict(plan=plan, oldHorizonSeconds=len(graph.destinations),
        newHorizonSeconds=len(extended.destinations), width=gp['width'], samples=gp['samples'],
        unchanged={k: v for k, v in unchanged.items() if k != 'layers'},
        localized={k: v for k, v in local_stats.items() if k != 'layers'}, marketExtension=extension,
        incremental={k: v for k, v in stats.items() if k != 'layers'},
        fullBackwardSecondsIncludingComparison=full_seconds,
        bitIdenticalAtEveryState=not mismatches, mismatchedDepths=mismatches,
        backwardSpeedup=full_seconds/stats['seconds'],
        oldRootCashUpperRatio=float(cache.values[0].c.min()/SCALE),
        newRootCashUpperRatio=float(updated.values[0].c.min()/SCALE),
        historicalBacktestPerformed=False, rollingRootAdaptationTested=False,
        claim='Incremental and full plane results match on this extended finite law; this does not prove planning optimality')
    (args.output/'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)
    if mismatches:
        raise ArithmeticError('Incremental cache differs from the full reference')


if __name__ == '__main__':
    main()
