"""Compare the declared graph with direct NSF draws and report accuracy gaps.

Probability propagation is exact up to float64 arithmetic on the saved finite
graph. Direct-path statistics and policy scores remain Monte Carlo estimates.
This report never upgrades them to a uniform or full-law accuracy certificate.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--graph',type=Path,required=True)
    ap.add_argument('--paths',type=Path,required=True)
    ap.add_argument('--upper',type=Path,required=True)
    ap.add_argument('--policy',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();args.output.mkdir()
    gp=json.loads((args.graph/'plan.json').read_text())
    pp=json.loads((args.paths/'plan.json').read_text())
    for key in ('checkpointSha256','origin','split','horizon'):
        if gp[key]!=pp[key]:raise ValueError(f'Incomparable graph and direct paths: {key}')
    upper=json.loads((args.upper/'report.json').read_text())
    policy=json.loads((args.policy/'report.json').read_text())
    graph_hash=hashlib.sha256((args.graph/'graph.npz').read_bytes()).hexdigest()
    if upper['graphSha256']!=graph_hash:raise ValueError('Upper bound uses another graph')
    policy_plan=json.loads((args.policy/'plan.json').read_text())
    if policy_plan['graphSha256']!=graph_hash:raise ValueError('Policy uses another graph')
    graph=np.load(args.graph/'graph.npz');prices=graph['price_ticks'];counts=graph['counts'];dest=graph['destinations']
    paths=np.load(args.paths/'prices.npy');H=len(counts)-1;B=dest.shape[-1]
    mass=np.ones(1);rows=[]
    for h in range(H):
        ds=dest[h,:counts[h]]
        mass=np.bincount(ds.ravel(),weights=np.repeat(mass/B,B),minlength=int(counts[h+1]))
        if abs(float(mass.sum())-1)>1e-11:raise ArithmeticError('Probability mass lost')
        if h+1 in (60,300,900,1800,H):
            r=prices[h+1,:counts[h+1]]/prices[0,0]-1
            mean=float(mass@r);std=float(np.sqrt(mass@(r-mean)**2))
            direct=paths[:,h+1]/paths[:,0]-1
            rows.append(dict(seconds=h+1,graphMeanBps=mean*10000,graphStdBps=std*10000,
                directSampleMeanBps=float(direct.mean()*10000),directSampleStdBps=float(direct.std(ddof=1)*10000),
                directMeanStandardErrorBps=float(direct.std(ddof=1)/np.sqrt(len(direct))*10000),
                graphToDirectStdRatio=std/float(direct.std(ddof=1))))
    ratio=upper['meanWealthUpperNumerator']/upper['meanWealthUpperDenominator']/10000
    measured_gap=100*(ratio/np.exp(policy['meanLogReturn'])-1) if policy['complete'] else None
    report=dict(graphSha256=graph_hash,horizonSeconds=H,gamma=1,directSampleCount=len(paths),
        marginalComparison=rows,graphCashCeRegretUpperPercent=upper['ceRelativeRegretUpperPercent'],
        evaluatedPolicyCeReturnPercent=policy['ceReturnPercent'],
        upperMinusEmpiricalPolicyCeRelativePercent=measured_gap,
        policyLogStandardError=policy['standardErrorLog'],
        empiricalPolicyIsCertifiedLower=False,fullNsfPlanningAccuracyCertified=False,
        status='0.1% full-NSF planning accuracy is not established.',
        unresolved=['The chosen policy value is estimated, not a proved lower expectation.',
                    'The finite graph changes the recursively sampled NSF law.',
                    'The Python order-aware upper verifier is not fully refined into Lean.',
                    'One forecast origin does not cover every account and backtest decision.'])
    (args.output/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
