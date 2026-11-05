"""Compare a split GPU replay with an uninterrupted replay of the same ticks."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from oracle_dense_value_map import DenseValueMap
from audit_oracle_dense_ledger import audit


def main():
    p=argparse.ArgumentParser();p.add_argument('--market-source',type=Path,required=True)
    p.add_argument('--initial-map',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    runner=Path(__file__).with_name('backtest_oracle_dense_repair.py')
    shared=[sys.executable,str(runner),'--market-source',str(args.market_source),'--max-wall-seconds','1200']
    first=args.output/'part';resumed=args.output/'resumed';whole=args.output/'whole'
    for name,directory,options in (
        ('part',first,['--initial-map',str(args.initial_map),'--seconds','5400','--max-updates','2','--repair-tolerance','1e-5','--refresh-every','60']),
        ('resumed',resumed,['--resume',str(first),'--max-updates','2']),
        ('whole',whole,['--initial-map',str(args.initial_map),'--seconds','5400','--max-updates','4','--repair-tolerance','1e-5','--refresh-every','60'])):
        print(json.dumps(dict(phase=name)),flush=True)
        log_path=args.output/f'{name}.log'
        with log_path.open('w') as log:
            result=subprocess.run(shared+['--output',str(directory)]+options,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            print(log_path.read_text()[-6000:],flush=True);raise RuntimeError(f'{name} replay failed')
    lhs=json.loads((resumed/'results.json').read_text());rhs=json.loads((whole/'results.json').read_text())
    if lhs['observedUpdates']!=4 or rhs['observedUpdates']!=4:raise AssertionError('Wrong resumed decision interval')
    for key in ('completed','failure','terminalWealth','terminalReturn','state','marketTrades','fillTicks'):
        if lhs[key]!=rhs[key]:raise AssertionError(f'Different realized replay result: {key}')
    prior=json.loads((first/'results.json').read_text())
    if abs(lhs['elapsedSeconds']-lhs['segmentSeconds']-prior['elapsedSeconds'])>1e-3:
        raise AssertionError('Earlier runtime was discarded')
    lrows=[json.loads(x) for x in (resumed/'updates.jsonl').read_text().splitlines()]
    rrows=[json.loads(x) for x in (whole/'updates.jsonl').read_text().splitlines()]
    keys=('update','originSeconds','price','horizon','state','exposure','orders','filled','tradeQuote','estimatedLogValue')
    for left,right in zip(lrows,rrows,strict=True):
        for key in keys:
            if left[key]!=right[key]:raise AssertionError(f'Decision {left["update"]} differs: {key}')
    left=DenseValueMap.load(resumed/'last-map');right=DenseValueMap.load(whole/'last-map')
    for key in ('prices','counts','destinations','sampled_prices','features','price_pairs','masses','origin_seconds'):
        np.testing.assert_array_equal(getattr(left.cloud,key),getattr(right.cloud,key))
    finite=np.isfinite(left.values);np.testing.assert_array_equal(finite,np.isfinite(right.values))
    error=float(np.max(abs(left.values[finite]-right.values[finite])))
    if error>1e-11:raise AssertionError('Resumed value field differs')
    np.testing.assert_array_equal(left.samples,right.samples)
    np.testing.assert_array_equal(left.bellman_residuals,right.bellman_residuals)
    for key in ('count','mean','m2'):np.testing.assert_array_equal(getattr(left.errors,key),getattr(right.errors,key))
    with np.load(resumed/'last-map'/'front.npz') as a,np.load(whole/'last-map'/'front.npz') as b:
        for key in a.files:np.testing.assert_array_equal(a[key],b[key])
    result=dict(passed=True,updates=4,splitAfter=2,maximumValueFieldError=error,
        resumedLedger=audit(resumed),uninterruptedLedger=audit(whole),
        cumulativeResumedSeconds=lhs['elapsedSeconds'],uninterruptedSeconds=rhs['elapsedSeconds'],
        sourceHashes={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in
            ('check_oracle_dense_resume.py','backtest_oracle_dense_repair.py','oracle_dense_checkpoint.py')},
        scope='GPU restart equivalence over four observed ticks with full one-hour value fields. Does not replace terminal-completion or long-window evaluation.')
    (args.output/'results.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':main()
