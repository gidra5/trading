"""Summarize complete or running dense-policy replays without fabricating returns."""
import argparse
import json
from pathlib import Path
import numpy as np
from oracle_dense_checkpoint import initial_archive


def distribution(values):
    values=np.asarray(values,dtype=float)
    if not len(values):return None
    return dict(count=len(values),mean=float(values.mean()),median=float(np.median(values)),
        p95=float(np.quantile(values,.95)),maximum=float(values.max()))


def sampling_work(decisions):
    groups={}
    for decision in decisions:
        for row in decision.get('table',{}).get('updates',[]):
            kind='supportRepair' if row.get('supportRepair') else 'complete' if row.get('complete') else 'ordinary'
            group=groups.setdefault(kind,dict(layerRefreshes=0,accountRowRefreshes=0,partialAccountRowRefreshes=0,evaluatedOutcomesIncludingCashReferences=0))
            group['layerRefreshes']+=1;group['accountRowRefreshes']+=row['rows']
            group['partialAccountRowRefreshes']+=row['partialRows']
            group['evaluatedOutcomesIncludingCashReferences']+=row['outcomes']
    for group in groups.values():
        group['partialAccountRowFraction']=(group['partialAccountRowRefreshes']/group['accountRowRefreshes']
            if group['accountRowRefreshes'] else None)
    return groups or None


def summarize(directory):
    directory=Path(directory);plan=json.loads((directory/'plan.json').read_text())
    result_path=directory/'results.json'
    result=json.loads(result_path.read_text()) if result_path.exists() else None
    interruption_path=directory/'interruption.json'
    interruption=json.loads(interruption_path.read_text()) if interruption_path.exists() else None
    progress_path=directory/'progress.json'
    progress=json.loads(progress_path.read_text()) if progress_path.exists() else {}
    path=directory/'updates.jsonl';text=path.read_text() if path.exists() else ''
    # A live writer may not have flushed the final line yet.
    lines=text.splitlines() if text.endswith('\n') else text.splitlines()[:-1]
    rows=[json.loads(line) for line in lines if line.strip()]
    decisions=[r for r in rows if 'horizon' in r]
    updates=int(result['observedUpdates']) if result else int(progress.get('update',rows[-1]['update'] if rows else 0))
    wealth=[10000.]+[r['markedEquity'] for r in rows]
    if result and result['markedEquity']!=wealth[-1]:wealth.append(result['markedEquity'])
    wealth=np.asarray(wealth);peak=np.maximum.accumulate(wealth)
    completed=bool(result and result['completed'])
    flat=bool(completed and result['state'][1]==0 and not np.any(np.asarray(result['state'][4::3])))
    if completed and not flat:raise ValueError('Completed replay does not end flat')
    if completed and (result['terminalWealth']!=result['markedEquity'] or result['terminalWealth']!=result['state'][0]):
        raise ValueError('Terminal wealth and cash disagree')
    status=('completed' if completed else 'failed' if result and result.get('failure') else
        'time_budget_reached' if result and result.get('timeBudgetReached') else 'stopped_early' if result else
        'interrupted' if interruption else 'running_or_interrupted')
    H=plan['arguments']['horizon']
    full=[r for r in decisions if r.get('fullRefresh',False)]
    graded=[r for r in decisions if r.get('reference') is not None]
    repair=[r for r in decisions if r['horizon']==H and not r.get('fullRefresh',False) and r.get('reference') is None]
    countdown=[r for r in decisions if r['horizon']<H and r.get('reference') is None]
    comparisons_path=directory/'comparisons.json'
    comparisons=result.get('comparisons',[]) if result else (json.loads(comparisons_path.read_text()) if comparisons_path.exists() else [])
    price_return=None
    initial_front=initial_archive(directory,plan)/'front.npz'
    if initial_front.exists() and (result or decisions):
        with np.load(initial_front) as initial:start_price=float(initial['state'][2])
        end_price=float(result['state'][2]) if result else float(decisions[-1]['price'])
        price_return=end_price/start_price-1
    summary=dict(status=status,interruption=interruption,completed=completed,plannedSeconds=plan['arguments']['seconds'],
        observedSeconds=updates,terminalFlat=flat if completed else None,
        terminalWealth=result['terminalWealth'] if completed else None,
        terminalReturn=result['terminalReturn'] if completed else None,
        observedMarketPriceReturn=price_return,
        latestMarkedEquity=float(wealth[-1]),maximumObservedDrawdown=float(np.max(1-wealth/peak)),
        exposedDecisionFraction=float(np.mean([abs(r['exposure'])>1.e-12 for r in decisions])) if decisions else None,
        pendingOrderDecisionFraction=float(np.mean([r['orders']>0 for r in decisions])) if decisions else None,
        maximumActiveOrders=max((r['orders'] for r in decisions),default=0),
        recordedMarketTradeDecisions=sum(r.get('tradeQuote',0.)!=0 for r in decisions),
        recordedFillTicks=sum(r.get('filled',False) for r in decisions),
        totalMarketTrades=result.get('marketTrades') if result else None,
        elapsedSeconds=result.get('elapsedSeconds') if result else progress.get('elapsedSeconds'),
        runtime=dict(rollingRepair=distribution([r['stepSeconds'] for r in repair]),
            fullRefresh=distribution([r['stepSeconds'] for r in full]),
            independentlyGradedDecisions=distribution([r['stepSeconds'] for r in graded]),
            terminalCountdown=distribution([r['stepSeconds'] for r in countdown])),
        maximumValueErrorAllowance=(max(r['errorAllowance'] for r in decisions)
            if decisions and all(r.get('errorAllowance') is not None for r in decisions) else None),
        maximumExecutionProjectionRMSEBps=max((r['market']['rootExecutionPriceRMSEBps'] for r in decisions),default=0.),
        continuationProjectionRMSEBps=distribution([r['market']['rootProjectionPriceRMSEBps'] for r in decisions]),
        referenceChecks=comparisons,
        incrementalSamplingWork=sampling_work(decisions),
        scope='Realized replay metrics. A null error allowance means no bound was supplied. Available allowances concern each finite-grid value query, not episode regret or accuracy against the continuous market law. No terminal return is reported for an incomplete run.')
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);args=p.parse_args()
    summary=summarize(args.directory)
    (args.directory/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
