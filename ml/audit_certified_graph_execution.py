"""Recheck recorded graph decisions against current Oracle execution rules.

This float64 conformance audit does not certify an expected policy value.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from gpu_oracle_bellman import TensorOracle,trade,equity


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True)
    args=ap.parse_args();report=json.loads((args.run/'report.json').read_text())
    if not report['complete']:raise ValueError('Cannot audit an incomplete run as complete')
    trace=np.load(args.run/'execution.npy',mmap_mode='r')
    if not np.array_equal(trace[1:,:,0],trace[:-1,:,2]):raise ArithmeticError('Discontinuous account trace')
    if not ((trace[0,:,0,0]==10000)&(trace[0,:,0,1]==0)&(trace[0,:,0,4]==0)&(trace[0,:,0,8]==0)).all():
        raise ArithmeticError('Wrong initial account')
    plan=json.loads((args.run/'plan.json').read_text())
    graph_path=Path(plan['arguments']['graph'])/'graph.npz'
    if hashlib.sha256(graph_path.read_bytes()).hexdigest()!=plan['graphSha256']:raise ArithmeticError('Changed forecast graph')
    historical=plan.get('resultKind')=='historical'
    if historical:
        path=args.run/'historical-prices.npy'
        if hashlib.sha256(path.read_bytes()).hexdigest()!=plan['historicalPriceSha256']:
            raise ArithmeticError('Changed historical marks')
        marks=np.load(path)
        if not np.array_equal(trace[:,:,0,2],marks[:-1,None]) or not np.array_equal(trace[:,:,2,2],marks[1:,None]):
            raise ArithmeticError('Execution does not follow actual historical marks')
        with (args.run/'decisions.jsonl').open() as source:
            rows=[json.loads(line) for line in source]
        if len(rows)!=len(trace) or any(row['index']!=h or row['forecastOriginTimestampSeconds']!=plan['startTimestampSeconds']+h for h,row in enumerate(rows)):
            raise ArithmeticError('Misaligned historical forecast origin')
    else:
        graph=np.load(graph_path);destinations=graph['destinations'];price_ticks=graph['price_ticks']
        nodes=np.zeros(report['paths'],np.int64)
        rng=np.random.default_rng(report['independentPolicyEvaluationSeed'])
        for h in range(len(trace)):
            ds=destinations[h,nodes]
            nodes=ds[np.arange(len(nodes)),rng.integers(0,ds.shape[1],len(nodes))]
            if not np.array_equal(trace[h,:,2,2],price_ticks[h+1,nodes]/100):
                raise ArithmeticError('Trace does not follow declared independent graph draws')
    x=trace.reshape(-1,3,9)
    oracle=TensorOracle();r=oracle.r;torch.set_num_threads(4);checked=0;largest_error=0.
    for start in range(0,len(x),8192):
        source=torch.tensor(np.asarray(x[start:start+8192]),device='cuda')
        before,after,moved=source.unbind(1)
        arrived,consumed,filled,arrival_ok=oracle.arrival(before)
        amount=after[:,0]-arrived[:,0]
        expected=trade(arrived,amount,r.taker)
        # Algebraic full closes deliberately avoid subtraction dust.
        fee_error=(after[:,:4]-expected[:,:4]).abs()
        if not (fee_error<=1e-9*(1+expected[:,:4].abs())).all():raise ArithmeticError('Wrong market fee/account update')
        largest_error=max(largest_error,float(fee_error.max()))
        nonzero=amount!=0;w=equity(after);e=after[:,1]*after[:,2]/torch.where(w!=0,w,1.)
        size_ok=~nonzero|((amount.abs()>=r.minimum-1e-8)&oracle.le(amount.abs(),r.maximum))
        valid=arrival_ok&size_ok&oracle.le(consumed+amount.abs(),r.maximum)&(w>0)
        valid&=~(nonzero|filled)|(oracle.le(-e,-r.lower)&oracle.le(e,r.upper))
        pending=after[:,4]!=0
        reserve=torch.where(pending,after[:,6]/r.reserve_leverage,0.)
        valid&=oracle.le(reserve,w)&(~pending|oracle.le(reserve+(after[:,1]*after[:,2]).abs()/r.reserve_leverage,w))
        kept=pending&(after[:,4:8]==arrived[:,4:8]).all(1)
        new=pending&~kept
        flow=torch.where(after[:,4]>0,-after[:,6],(1-r.maker)*after[:,6])
        passive=torch.where(after[:,4]>0,after[:,5]<after[:,2],after[:,5]>after[:,2])
        valid&=~new|((after[:,7]==before[:,8])&passive&(after[:,5]>0)&oracle.size(flow))
        grid=(after[:,5]/r.tick-(after[:,5]/r.tick).round()).abs()<=1e-7
        valid&=~pending|grid
        valid&=(after[:,8]==before[:,8])&(after[:,3]==before[:,3])
        if not valid.all():raise ArithmeticError(f'Invalid actual rebalance near flattened decision {start}')
        advanced=oracle.advance(after,moved[:,2])
        if not torch.allclose(advanced,moved,rtol=1e-12,atol=1e-10):raise ArithmeticError('Wrong funding/mark/liquidation transition')
        checked+=len(before)
    last=np.load(args.run/'execution.npy',mmap_mode='r')[-1,:,2]
    state=torch.tensor(np.asarray(last),device='cuda')
    terminal=oracle.terminal(state).cpu().numpy()
    if not ((last[:,1]==0)&(last[:,4]==0)).all():raise ArithmeticError('Nonflat actual ending')
    if not np.allclose(terminal,np.load(args.run/'terminal-wealth.npy'),rtol=1e-12,atol=1e-9):raise ArithmeticError('Wrong terminal score')
    result=dict(passed=True,decisionsChecked=checked,paths=report['paths'],largestAbsoluteFeeArithmeticError=largest_error,
        pricesChecked='historical archive' if historical else 'independent graph draws',
        scope='Float64 conformance with current Oracle; no formal extraction or expectation certificate.',
        oracleSourceSha256=hashlib.sha256((Path(__file__).parent/'gpu_oracle_bellman.py').read_bytes()).hexdigest())
    (args.run/'execution-audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))


if __name__=='__main__':main()
