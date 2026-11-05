"""Sample-count study for paired Monte Carlo updates of the one-hour cache."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from numba import set_num_threads
from oracle_certified_value import ValuePlanes, plane_hulls
from oracle_incremental_value import PlaneGraph, PlaneCache
from oracle_mc_continuation import MonteCarloPlan, estimate_extension
from oracle_plane_policy import plane_policy_action, account_values
from gpu_oracle_bellman import TensorOracle, Menu


def digest(path):
    with Path(path).open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def templates(price, depth):
    """Declared account cases, including feasible pre-existing passive orders."""
    rows = []
    for e, side, wealth in [(0, 0, 10000), (.5, 0, 10000), (-.5, 0, 10000),
            (1.9, 0, 10000), (-1.9, 0, 10000), (0, 1, 10000),
            (.5, -1, 10000), (-.5, 1, 10000), (0, 0, 2500), (0, 0, 20000)]:
        rows.append([(1-e)*wealth, e*wealth/price, price, price,
            side, (round(price/.01)-side*10)*.01 if side else 0,
            .25*wealth if side else 0, depth-1 if side else 0, depth])
    return rows


@torch.inference_mode()
def evaluate(reference, candidate, cases, oracle, descriptors, total_horizon):
    errors = []; agreement = []; losses = []; decisions = []
    started = time.perf_counter()
    for h, item in cases.items():
        old = reference[h+1]; new = candidate[h+1]
        prices = item['childPrices']; nodes = item['children']
        state = item['state']
        ref = item['referencePolicy']
        got = plane_policy_action(oracle, state, descriptors, prices, nodes,
            new.c, new.d, total_horizon-h)
        if not got['valid'].all():
            return dict(complete=False, reason='No finite selected action for an evaluated account', depth=h)
        choice = got['scores'].argmax(1)
        correct = ref['scores'].argmax(1)
        ref_selected = ref['scores'][torch.arange(len(state), device=state.device), choice]
        loss = (-torch.expm1(ref_selected-ref['value'])).clamp_min(0).cpu().numpy()*100
        losses.extend(loss.tolist()); agreement.extend((choice == correct).cpu().tolist())
        decisions.extend(dict(depth=h, case=i, referenceAction=int(correct[i]),
            estimatedAction=int(choice[i]), referenceCeLossPercent=float(loss[i])) for i in range(len(state)))
        # Estimate error over every market node at this child depth and 41
        # account exposures. Account wealth cancels in this specific plane law.
        n = len(new.c); ee = np.linspace(-2, 2, 41)
        probe = np.zeros((n*len(ee), 9)); probe[:, 0] = np.tile(10000*(1-ee), n)
        probe[:, 1] = np.tile(10000*ee, n); probe[:, 2] = 1
        ii = np.repeat(np.arange(n), len(ee))
        truth = account_values(old.c, old.d, *plane_hulls(old.c, old.d), probe, ii)
        estimated = account_values(new.c, new.d, *plane_hulls(new.c, new.d), probe, ii)
        if not np.isfinite(estimated).all():
            return dict(complete=False, reason='Nonfinite estimated continuation on evaluated account domain', depth=h)
        errors.extend((100*np.abs(np.expm1(estimated-truth))).tolist())
    return dict(complete=True, cases=len(losses), actionAgreementPercent=100*float(np.mean(agreement)),
        maximumReferenceCeLossPercent=float(np.max(losses)), meanReferenceCeLossPercent=float(np.mean(losses)),
        p95ReferenceCeLossPercent=float(np.quantile(losses, .95)),
        maximumContinuationRelativeErrorPercent=float(np.max(errors)),
        p95ContinuationRelativeErrorPercent=float(np.quantile(errors, .95)),
        meanContinuationRelativeErrorPercent=float(np.mean(errors)),
        selectionSeconds=time.perf_counter()-started, decisions=decisions)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--graph', type=Path, required=True); ap.add_argument('--value', type=Path, required=True)
    ap.add_argument('--extension', type=Path, required=True); ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--samples', type=int, nargs='+', default=[1, 4, 16])
    ap.add_argument('--seeds', type=int, nargs='+', default=[20260930, 20260931, 20260932])
    ap.add_argument('--tolerance-percent', type=float, default=.1)
    ap.add_argument('--estimator', choices=('plain', 'paired'), default='plain')
    ap.add_argument('--include-comparison', action='store_true')
    ap.add_argument('--reference', type=Path)
    args = ap.parse_args(); args.output.mkdir()
    torch.set_num_threads(4); set_num_threads(4); torch.set_float32_matmul_precision('high')
    gp = json.loads((args.graph/'plan.json').read_text()); vr = json.loads((args.value/'report.json').read_text())
    ep = json.loads((args.extension/'plan.json').read_text())
    if digest(args.graph/'graph.npz') != vr['graphSha256'] or ep['graphSha256'] != vr['graphSha256']:
        raise ValueError('Mismatched graph, baseline or extension')
    for name in ('oracle_order_value.py', 'oracle_certified_value.py'):
        if digest(Path(__file__).parent/name) != vr['sourceHashes'][name]:
            raise ValueError('Cached backup implementation changed')
    data = np.load(args.graph/'graph.npz'); counts = data['counts']; pp = data['price_ticks']; ds = data['destinations']
    base = PlaneGraph(tuple(pp[h, :n] for h, n in enumerate(counts)),
        tuple(ds[h, :n] for h, n in enumerate(counts[:-1])), vr['supportingPoints'],
        vr['reachableLeverageNumerator'], vr['reachableLeverageDenominator'])
    cc = np.load(args.value/'post-arrival-c.npy', mmap_mode='r'); dc = np.load(args.value/'post-arrival-d.npy', mmap_mode='r')
    cache = PlaneCache(base, tuple(ValuePlanes(cc[h, :n], dc[h, :n]) for h, n in enumerate(counts)))
    addition = np.load(args.extension/'extension.npz')
    graph = PlaneGraph(base.prices+(addition['price_ticks'],), base.destinations+(addition['destinations'],),
        base.points, base.gross_numerator, base.gross_denominator)
    H = len(graph.destinations); B = ds.shape[-1]
    # Predeclared before any Monte Carlo run: four depths, low/central/high
    # representative prices, and the complete same Oracle action menu.
    depths = [0, 600, 1800, 3000]
    if H <= depths[-1]+8:
        raise ValueError('This benchmark is specified for the one-hour graph')
    selected = set(h+1 for h in depths)
    menu = Menu(step=.01, limit_fractions=(.1, .25, .5, 1.), limit_offsets=(1, 10, 100))
    oracle = TensorOracle(); descriptors = oracle.descriptors(menu, 'cuda')
    snapshots = args.output/'source'; snapshots.mkdir(); hashes = {}
    for name in ('benchmark_oracle_mc_continuation.py', 'oracle_mc_continuation.py', 'oracle_incremental_value.py',
            'oracle_plane_policy.py', 'oracle_order_value.py', 'oracle_certified_value.py', 'gpu_oracle_bellman.py'):
        path = Path(__file__).parent/name; (snapshots/name).write_bytes(path.read_bytes()); hashes[name] = digest(path)
    plan = dict(arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        graphSha256=vr['graphSha256'], extensionSha256=digest(args.extension/'extension.npz'), sourceHashes=hashes,
        menu=asdict(menu), gamma=1, oldHorizonSeconds=H-1, newHorizonSeconds=H,
        source='Frozen learned NSF finite graph, with Oracle funding factor 1 and declared fees',
        depths=depths, accountCaseCountPerNode=10, reference='Full 64-outcome plane update and its greedy feedback scores',
        accuracyScope='Sampling error relative to this approximate controller, not true optimality under NSF',
        tolerancePercent=args.tolerance_percent, meanTakenBeforeActionMaximum=True,
        withoutReplacement=True, selectionUsesAll64Outcomes=True)
    (args.output/'plan.json').write_text(json.dumps(plan, indent=2))
    warm = base.backup(0, cache.values[1], np.array([0]))
    np.testing.assert_array_equal(warm.c, cache.values[0].c)
    reference = {}
    def retain(h, value, record):
        if h in selected:
            reference[h] = value
        if h % 600 == 0:
            print(json.dumps(dict(stage='reference', **record)), flush=True)
    if args.reference is None:
        root, ref_stats = estimate_extension(cache, graph, MonteCarloPlan(B, 0), on_layer=retain)
        expected_root = json.loads((args.extension/'report.json').read_text())['newRootCashUpperRatio']
        if float(root.c.min()/2**30) != expected_root:
            raise ArithmeticError('Full reference differs from the independent prior sweep')
    else:
        prior = json.loads((args.reference/'plan.json').read_text())
        if any(prior[key] != plan[key] for key in ('graphSha256', 'extensionSha256', 'depths', 'newHorizonSeconds')):
            raise ValueError('Reference belongs to a different comparison')
        for h in selected:
            saved = np.load(args.reference/f'reference-{h}.npz')
            reference[h] = ValuePlanes(saved['c'], saved['d'])
        ref_stats = json.loads((args.reference/'reference.json').read_text())
        print(json.dumps(dict(stage='reference_reused', path=str(args.reference))), flush=True)
    for h, value in reference.items():
        np.savez_compressed(args.output/f'reference-{h}.npz', c=value.c, d=value.d)
    (args.output/'reference.json').write_text(json.dumps(ref_stats, indent=2))
    cases = {}
    for h in depths:
        ordered = np.argsort(graph.prices[h]); pick = np.unique(ordered[np.rint(np.linspace(0, len(ordered)-1, 3)).astype(int)])
        nodes = np.repeat(pick, 10); state = torch.tensor(np.concatenate([
            templates(graph.prices[h][n]/100, h) for n in pick]), dtype=torch.float64, device='cuda')
        children = graph.destinations[h][nodes]
        prices = torch.tensor(graph.prices[h+1][children]/100, dtype=torch.float64, device='cuda')
        policy = plane_policy_action(oracle, state, descriptors, prices, children,
            reference[h+1].c, reference[h+1].d, H-h)
        if not policy['valid'].all():
            raise ArithmeticError('Reference is invalid for a predeclared account case')
        cases[h] = dict(state=state, children=children, childPrices=prices, referencePolicy=policy)
    np.savez_compressed(args.output/'account-cases.npz', **{f'state_{h}': row['state'].cpu().numpy() for h, row in cases.items()})
    zero = evaluate(reference, {h: cache.values[h] for h in selected}, cases, oracle, descriptors, H)
    results = []
    paired = args.estimator == 'paired'
    specifications = [(paired, n, seed) for n in args.samples for seed in args.seeds]
    if args.include_comparison:
        specifications += [(not paired, min(args.samples), args.seeds[0])]
    for paired, n, seed in specifications:
        key = f'{"paired" if paired else "naive"}-{n}-{seed}'
        saved = {}; prefix = args.output/key; prefix.mkdir()
        def observe(h, value, record):
            if h in selected:
                saved[h] = value
            if h % 600 == 0:
                print(json.dumps(dict(stage=key, **record)), flush=True)
        try:
            _, stats = estimate_extension(cache, graph, MonteCarloPlan(n, seed), paired=paired, on_layer=observe)
            metrics = evaluate(reference, saved, cases, oracle, descriptors, H)
            for h, value in saved.items():
                np.savez_compressed(prefix/f'continuation-{h}.npz', c=value.c, d=value.d)
            (prefix/'timing.json').write_text(json.dumps(stats, indent=2))
            (prefix/'accuracy.json').write_text(json.dumps(metrics, indent=2))
            compact = {k: v for k, v in metrics.items() if k != 'decisions'}
            result = dict(samples=n, seed=seed, paired=paired, **compact,
                seconds=stats['seconds'], potentialWarmUpdateSeconds=stats['potentialWarmUpdateSeconds'],
                oldSampledBackupSeconds=stats['oldSampledBackupSeconds'], totalWedgeViolations=stats['totalWedgeViolations'])
            result['passesMeasuredActionTolerance'] = metrics['complete'] and metrics['maximumReferenceCeLossPercent'] <= args.tolerance_percent
        except (ArithmeticError, ValueError, OverflowError) as error:
            result = dict(samples=n, seed=seed, paired=paired, complete=False,
                passesMeasuredActionTolerance=False, reason=str(error))
            (prefix/'failure.json').write_text(json.dumps(result, indent=2))
        results.append(result)
        (args.output/'partial-results.json').write_text(json.dumps(results, indent=2))
        print(json.dumps(dict(stage='trial_complete', **result)), flush=True)
    groups = []
    for n in args.samples:
        rows = [r for r in results if r['paired'] == (args.estimator == 'paired') and r['samples'] == n]
        good = [r for r in rows if r['complete']]
        groups.append(dict(samples=n, estimator=args.estimator, trials=len(rows), completed=len(good),
            allTrialsPass=all(r['passesMeasuredActionTolerance'] for r in rows),
            worstReferenceCeLossPercent=max((r['maximumReferenceCeLossPercent'] for r in good), default=None),
            minimumActionAgreementPercent=min((r['actionAgreementPercent'] for r in good), default=None),
            maximumContinuationRelativeErrorPercent=max((r['maximumContinuationRelativeErrorPercent'] for r in good), default=None),
            meanSeconds=float(np.mean([r['seconds'] for r in good])) if good else None))
    passed = [r['samples'] for r in groups if r['allTrialsPass']]
    report = dict(plan=plan, referenceSeconds=ref_stats['seconds'], noUpdate=zero,
        groups=groups, trials=results, smallestTestedPassingCount=min(passed) if passed else None,
        sampleCountIsPerNode=True, independentSeeds=len(args.seeds),
        highProbabilityGuarantee=False, fullNsfOptimalityCertified=False, historicalBacktestPerformed=False)
    (args.output/'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k not in ('plan', 'noUpdate', 'trials')}), flush=True)


if __name__ == '__main__':
    main()
