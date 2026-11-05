"""Predefined cash and half-exposure baselines under the replay's Oracle rules."""
import argparse
import json
from pathlib import Path
import numpy as np
from oracle_book_mc import execute_target,move_arrive,close_terminal,effective,rule_vector
from gpu_oracle_bellman import Rules
from oracle_dense_checkpoint import initial_archive
from audit_oracle_dense_ledger import audit


def hold_baseline(prices,rules,exposure,initial_wealth=10000.):
    prices=np.asarray(prices,dtype=float)
    if len(prices)<2 or np.any(~np.isfinite(prices)|(prices<=0)):raise ValueError('Need positive observed prices')
    initial=np.array([initial_wealth,0.,prices[0],prices[0]])
    state,okay=execute_target(initial,0.,False,exposure,np.zeros((0,2)),0,rules,1.)
    if not okay:return dict(targetExposure=exposure,completed=False,failure='Entry infeasible',terminalWealth=None,terminalReturn=None)
    equity=float(state[0]+state[1]*state[2]);entry_fee=initial_wealth-equity
    peak=max(initial_wealth,equity);drawdown=1-equity/peak
    for tick,price in enumerate(prices[1:],1):
        state,used,_,okay=move_arrive(state,price,rules)
        if not okay:return dict(targetExposure=exposure,completed=False,failure=f'Invalid arrival at tick {tick}',
            terminalWealth=None,terminalReturn=None)
        equity=float(state[0]+state[1]*state[2]);peak=max(peak,equity);drawdown=max(drawdown,1-equity/peak)
    if not np.isfinite(close_terminal(state,used,rules)):
        return dict(targetExposure=exposure,completed=False,failure='Terminal full close infeasible',terminalWealth=None,terminalReturn=None)
    terminal=float(state[0]+effective(state,rules));closing_fee=equity-terminal
    drawdown=max(drawdown,1-terminal/peak)
    return dict(targetExposure=exposure,completed=True,failure=None,terminalWealth=terminal,
        terminalReturn=terminal/initial_wealth-1,entryFeeQuote=float(entry_fee),closingFeeQuote=float(closing_fee),
        maximumDrawdown=float(drawdown),marketTrades=0 if exposure==0 else 2)


def summarize(directory):
    directory=Path(directory);plan=json.loads((directory/'plan.json').read_text())
    result=json.loads((directory/'results.json').read_text())
    if not result['completed']:raise ValueError('Baselines require a completed observed interval')
    checked=audit(directory)
    rows=[json.loads(line) for line in (directory/'updates.jsonl').read_text().splitlines() if line.strip()]
    with np.load(initial_archive(directory,plan)/'front.npz') as front:initial_price=float(front['state'][2])
    prices=np.array([initial_price]+[row['price'] if 'horizon' in row else result['state'][2] for row in rows])
    rules=rule_vector(Rules(**plan['rules']),plan['fundingMultiplier'])
    baselines={name:hold_baseline(prices,rules,exposure) for name,exposure in
        (('cash',0.),('long_half',.5),('short_half',-.5))}
    return dict(observedSeconds=len(prices)-1,sourcePriceStart=float(prices[0]),sourcePriceEnd=float(prices[-1]),
        learnedPolicyTerminalWealth=result['terminalWealth'],learnedPolicyTerminalReturn=result['terminalReturn'],
        baselines=baselines,executionLedgerChecked=checked['terminalChecked'],
        scope='Fixed entry exposure, hold funded quantity without resizing, valid full close at the same final tick. Same fees, funding, liquidation and quote caps as the learned-policy replay. These predefined baselines are not hindsight-optimized policies; an infeasible close is not reported as a valid return.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);args=p.parse_args();result=summarize(args.directory)
    (args.directory/'baselines.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
