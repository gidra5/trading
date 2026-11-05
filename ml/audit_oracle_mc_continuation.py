"""Check continuous exposure envelopes for every saved Monte Carlo trial."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from oracle_certified_value import ValuePlanes
from oracle_mc_error import log_envelope_errors, action_loss_bound_percent
from gpu_oracle_bellman import TensorOracle, Menu, equity
from oracle_plane_policy import arrival_repairable


def read_planes(path):
    z = np.load(path)
    return ValuePlanes(z['c'], z['d'])


@torch.inference_mode()
def check_exposure_domain(run, plan):
    """Validate the bound's interval premise for all feasible benchmark actions."""
    torch.set_num_threads(4)
    graph = np.load(Path(plan['arguments']['graph'])/'graph.npz')
    cases = np.load(run/'account-cases.npz')
    p = graph['price_ticks']; counts = graph['counts']; ds = graph['destinations']
    oracle = TensorOracle(); desc = oracle.descriptors(Menu(**plan['menu']), 'cuda')
    repair = oracle.descriptors(Menu(step=2., limits=False), 'cuda')
    low = np.inf; high = -np.inf; checked = 0
    for h in plan['depths']:
        state = torch.tensor(cases[f'state_{h}'], dtype=torch.float64, device='cuda')
        ordered = np.argsort(p[h, :counts[h]])
        pick = np.unique(ordered[np.rint(np.linspace(0, len(ordered)-1, 3)).astype(int)])
        nodes = np.repeat(pick, plan['accountCaseCountPerNode'])
        prices = torch.tensor(p[h+1, ds[h, nodes]]/100, dtype=torch.float64, device='cuda')
        out, valid, _ = oracle.actions(state, desc)
        row, action = valid.nonzero(as_tuple=True)
        for begin in range(0, len(row), 128):
            r = row[begin:begin+128]; a = action[begin:begin+128]
            moved = oracle.advance(out[r, a, None].expand(-1, prices.shape[1], -1), prices[r])
            arrived, _, _, okay = oracle.arrival(moved)
            okay = arrival_repairable(oracle, moved, arrived, okay, repair)
            wealth = equity(arrived)
            admitted = (okay & (wealth > 0)).all(1)
            if admitted.any():
                exposure = arrived[..., 1]*arrived[..., 2]/torch.where(wealth > 0, wealth, 1)
                selected = exposure[admitted]
                low = min(low, float(selected.min())); high = max(high, float(selected.max()))
                checked += selected.numel()
    return dict(checkedOutcomes=checked, minimumExposure=low, maximumExposure=high,
        allInsideInterval=bool(low >= -3 and high <= 3))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--run', type=Path, required=True)
    args = ap.parse_args(); plan = json.loads((args.run/'plan.json').read_text())
    value_plan = json.loads((Path(plan['arguments']['value'])/'report.json').read_text())
    reachable = value_plan['reachableLeverageNumerator']/value_plan['reachableLeverageDenominator']
    depths = [h+1 for h in plan['depths']]
    reference = {h: read_planes(args.run/f'reference-{h}.npz') for h in depths}
    trials = []
    for path in sorted(args.run.glob('*/accuracy.json')):
        stats = json.loads(path.read_text())
        if not stats['complete']:
            continue
        trials.append((path.parent.name, {h: read_planes(path.parent/f'continuation-{h}.npz') for h in depths}))
    baseline_c = np.load(Path(plan['arguments']['value'])/'post-arrival-c.npy', mmap_mode='r')
    baseline_d = np.load(Path(plan['arguments']['value'])/'post-arrival-d.npy', mmap_mode='r')
    trials.append(('unchanged_cache', {h: ValuePlanes(baseline_c[h, :len(reference[h].c)],
        baseline_d[h, :len(reference[h].c)]) for h in depths}))
    records = []
    for name, candidates in trials:
        values = []; absolute = []; detailed = []; maintenance_values = []
        for h in depths:
            candidate = candidates[h]
            a, e = log_envelope_errors(reference[h], candidate, reachable)
            _, broad = log_envelope_errors(reference[h], candidate, 3.)
            absolute.extend(a.tolist()); values.extend(e.tolist())
            maintenance_values.extend(broad.tolist())
            detailed.append(dict(depth=h, maximumCenteredLogError=float(e.max()),
                conditionalCeLossBoundPercent=float(action_loss_bound_percent(e.max()))))
        epsilon = max(values)
        row = dict(trial=name, exposureInterval=[-reachable, reachable],
            maximumCenteredLogError=epsilon,
            maximumCenteredRelativeErrorPercent=100*float(np.expm1(epsilon)),
            maximumAbsoluteLogError=max(absolute),
            conditionalCeLossBoundPercent=float(action_loss_bound_percent(epsilon)),
            belowPointOnePercent=bool(action_loss_bound_percent(epsilon) <= .1),
            maintenanceIntervalCeLossBoundPercent=float(action_loss_bound_percent(max(maintenance_values))),
            depths=detailed)
        records.append(row)
    domain = check_exposure_domain(args.run, plan)
    domain['allInsideReachableInterval'] = bool(domain['minimumExposure'] >= -reachable and domain['maximumExposure'] <= reachable)
    report = dict(exposureGridUsed=False, arithmetic='float64 hull intersections and log ratios',
        benchmarkExposureDomainCheck=domain,
        exposureInterval=[-reachable, reachable], maintenanceInterval=[-3, 3],
        scope='All represented market nodes at the four declared child depths; existing graph-wide reachable leverage bound and separate maintenance-interval comparison',
        boundAssumptions=['Same feasible actions for estimated and reference scoring',
            'Exogenous market transition probabilities shared by every action',
            'All scored continuation exposures lie inside the checked interval'],
        referenceIsTrueOptimal=False, highProbabilitySamplingGuarantee=False, records=records)
    (args.run/'centered-error-audit.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
