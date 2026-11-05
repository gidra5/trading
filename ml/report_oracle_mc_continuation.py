"""Combine completed sample-count trials and their continuous-error audits."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', type=Path, nargs='+', required=True)
    ap.add_argument('--plain-runs', type=Path, nargs='*', default=[])
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args(); args.output.mkdir()
    rows = []; reports = []; provenance = []
    for run in args.runs:
        report = json.loads((run/'report.json').read_text())
        audit = json.loads((run/'centered-error-audit.json').read_text())
        if reports:
            first_plan = reports[0][0]['plan']
            if any(report['plan'][key] != first_plan[key] for key in
                    ('graphSha256', 'extensionSha256', 'depths', 'menu', 'tolerancePercent')):
                raise ValueError('Cannot combine studies with different forecast/account comparisons')
            if report['plan']['sourceHashes']['oracle_mc_continuation.py'] != first_plan['sourceHashes']['oracle_mc_continuation.py']:
                raise ValueError('Cannot combine different Monte Carlo estimators')
        if not audit['benchmarkExposureDomainCheck']['allInsideReachableInterval']:
            raise ArithmeticError('The actual action outcomes exceed the compared interval')
        lookup = {r['trial']: r for r in audit['records']}
        reports.append((report, audit))
        for trial in report['trials']:
            if not trial['paired']:
                continue
            if not trial['complete']:
                raise ValueError(f'Failed trial must not be omitted from an accuracy recommendation: {run}')
            key = f"paired-{trial['samples']}-{trial['seed']}"
            rows.append(dict(**trial, continuousAudit=lookup[key]))
        provenance.append(dict(run=str(run), hashes={name: hashlib.sha256((run/name).read_bytes()).hexdigest()
            for name in ('report.json', 'centered-error-audit.json', 'plan.json')}))
    if len({(r['samples'], r['seed']) for r in rows}) != len(rows):
        raise ValueError('Duplicate sample-count/seed results')
    tolerance = reports[0][0]['plan']['tolerancePercent']
    groups = []
    for samples in sorted({r['samples'] for r in rows}):
        selected = [r for r in rows if r['samples'] == samples]
        groups.append(dict(samplesPerNode=samples, seeds=len(selected),
            maximumAbsoluteValueErrorPercent=100*float(np.expm1(max(r['continuousAudit']['maximumAbsoluteLogError'] for r in selected))),
            maximumProbedAbsoluteValueErrorPercent=max(r['maximumContinuationRelativeErrorPercent'] for r in selected),
            maximumObservedActionLossPercent=max(r['maximumReferenceCeLossPercent'] for r in selected),
            maximumContinuousActionLossBoundPercent=max(r['continuousAudit']['conditionalCeLossBoundPercent'] for r in selected),
            maximumMaintenanceIntervalLossBoundPercent=max(r['continuousAudit']['maintenanceIntervalCeLossBoundPercent'] for r in selected),
            minimumActionAgreementPercent=min(r['actionAgreementPercent'] for r in selected),
            maximumActionAgreementPercent=max(r['actionAgreementPercent'] for r in selected),
            meanColdSeconds=float(np.mean([r['seconds'] for r in selected])),
            meanPotentialWarmSeconds=float(np.mean([r['potentialWarmUpdateSeconds'] for r in selected])),
            absoluteValueTargetPassed=all(100*np.expm1(r['continuousAudit']['maximumAbsoluteLogError']) <= tolerance for r in selected),
            continuousActionTargetPassed=all(r['continuousAudit']['conditionalCeLossBoundPercent'] <= tolerance for r in selected)))
    first, audit = reports[0]
    cached = next(r for r in audit['records'] if r['trial'] == 'unchanged_cache')
    passed_actions = [g['samplesPerNode'] for g in groups if g['continuousActionTargetPassed']]
    passed_values = [g['samplesPerNode'] for g in groups if g['absoluteValueTargetPassed']]
    plain_rows = []
    for run in args.plain_runs:
        report = json.loads((run/'report.json').read_text())
        detail = json.loads((run/'centered-error-audit.json').read_text())
        if any(report['plan'][key] != first['plan'][key] for key in
                ('graphSha256', 'extensionSha256', 'depths', 'menu', 'tolerancePercent')):
            raise ValueError('Plain comparator uses a different forecast/account comparison')
        if report['plan']['sourceHashes']['oracle_mc_continuation.py'] != first['plan']['sourceHashes']['oracle_mc_continuation.py']:
            raise ValueError('Plain comparator uses a different estimator implementation')
        if not detail['benchmarkExposureDomainCheck']['allInsideReachableInterval']:
            raise ArithmeticError('Plain comparator exposure premise failed')
        lookup = {r['trial']: r for r in detail['records']}
        for trial in report['trials']:
            if trial['paired']:
                continue
            if not trial['complete']:
                raise ValueError('A failed plain trial cannot be omitted')
            plain_rows.append(dict(**trial, continuousAudit=lookup[f"naive-{trial['samples']}-{trial['seed']}"]))
        provenance.append(dict(run=str(run), hashes={name: hashlib.sha256((run/name).read_bytes()).hexdigest()
            for name in ('report.json', 'centered-error-audit.json', 'plan.json')}))
    if len({(r['samples'], r['seed']) for r in plain_rows}) != len(plain_rows):
        raise ValueError('Duplicate plain-comparator trials')
    plain_groups = []
    for n in sorted({r['samples'] for r in plain_rows}):
        selected = [r for r in plain_rows if r['samples'] == n]
        plain_groups.append(dict(samplesPerNode=n, seeds=len(selected),
            maximumAbsoluteValueErrorPercent=100*float(np.expm1(max(r['continuousAudit']['maximumAbsoluteLogError'] for r in selected))),
            maximumObservedActionLossPercent=max(r['maximumReferenceCeLossPercent'] for r in selected),
            maximumContinuousActionLossBoundPercent=max(r['continuousAudit']['conditionalCeLossBoundPercent'] for r in selected),
            minimumActionAgreementPercent=min(r['actionAgreementPercent'] for r in selected),
            maximumActionAgreementPercent=max(r['actionAgreementPercent'] for r in selected),
            meanColdSeconds=float(np.mean([r['seconds'] for r in selected])),
            allTrialsPass=all(100*np.expm1(r['continuousAudit']['maximumAbsoluteLogError']) <= tolerance and
                r['continuousAudit']['conditionalCeLossBoundPercent'] <= tolerance for r in selected)))
    summary = dict(provenance=provenance, groups=groups,
        plainGroups=plain_groups,
        smallestTestedActionCount=min(passed_actions) if passed_actions else None,
        smallestTestedValueCount=min(passed_values) if passed_values else None,
        tolerancePercent=tolerance, referenceSeconds=first['referenceSeconds'],
        noUpdate=dict(metrics={k: v for k, v in first['noUpdate'].items() if k != 'decisions'}, continuousAudit=cached),
        naiveTrials=[r for report, _ in reports for r in report['trials'] if not r['paired']],
        accuracyScope='One-second extension of one frozen finite NSF graph; four audited child depths; not a full NSF optimality certificate',
        sourceSha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2))
    table = ['| Estimator | Draws / node | Worst value error | Worst action loss | Continuous action-loss bound | Action agreement | Cold update |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for kind, g in [('Paired', g) for g in groups]+[('Plain', g) for g in plain_groups]:
        table.append(f"| {kind} | {g['samplesPerNode']} | {g['maximumAbsoluteValueErrorPercent']:.6f}% | "
            f"{g['maximumObservedActionLossPercent']:.6f}% | {g['maximumContinuousActionLossBoundPercent']:.6f}% | "
            f"{g['minimumActionAgreementPercent']:.0f}–{g['maximumActionAgreementPercent']:.0f}% | {g['meanColdSeconds']:.1f} s |")
    (args.output/'table.md').write_text('\n'.join(table)+'\n', encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 5.2), layout='constrained')
    x = [g['samplesPerNode'] for g in groups]
    for field, label, color in [
            ('maximumAbsoluteValueErrorPercent', 'Value error', '#7047a3'),
            ('maximumContinuousActionLossBoundPercent', 'Action-loss bound over reachable exposures', '#ce751a'),
            ('maximumObservedActionLossPercent', 'Action loss in 100 account cases', '#287db0')]:
        ax.plot(x, [g[field] for g in groups], marker='o', label=label, color=color, linewidth=2)
    ax.axhline(tolerance, color='#b1383b', linestyle='--', label=f'{tolerance:g}% target')
    ax.axhline(cached['conditionalCeLossBoundPercent'], color='#555555', linestyle=':',
        label='Unchanged-cache action-loss bound')
    if plain_groups:
        ax.scatter([g['samplesPerNode'] for g in plain_groups], [g['maximumContinuousActionLossBoundPercent'] for g in plain_groups],
            marker='D', s=55, facecolor='white', edgecolor='#288847', linewidth=2,
            label='Plain Monte Carlo action-loss bound', zorder=5)
    ax.set_xscale('log', base=2); ax.set_yscale('log')
    ax.set_xticks(x, [str(n) for n in x]); ax.set_xlabel('Monte Carlo draws per conditional node')
    ax.set_ylabel('Error or CE loss (%)'); ax.grid(True, which='both', alpha=.18)
    ax.set_title('One-second continuation update: worst result across three seeds')
    ax.legend(fontsize=8, loc='upper right')
    fig.savefig(args.output/'accuracy.png', dpi=170)
    fig.savefig(args.output/'accuracy.svg')
    plt.close(fig)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
