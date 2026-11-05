"""Compute continuous-action account bounds for a frozen every-second law."""
import argparse
from dataclasses import asdict
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from numba import set_num_threads
from oracle_certified_value import ValuePlanes,terminal_planes,supporting_backup,upper_cash_ratio,SCALE
from oracle_order_value import order_backup,rational_order_backup
from oracle_affine_value import terminal_affine,affine_backup
from oracle_full_support_value import full_support_backup
from gpu_oracle_bellman import Rules


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--graph',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--points',type=int,default=33)
    ap.add_argument('--save-values',action='store_true')
    modes=ap.add_mutually_exclusive_group()
    modes.add_argument('--directional',action='store_true')
    modes.add_argument('--order-aware',action='store_true')
    modes.add_argument('--affine',action='store_true')
    modes.add_argument('--full-support-class',action='store_true')
    ap.add_argument('--reachable-leverage',action='store_true')
    args=ap.parse_args();args.output.mkdir()
    source_dir=Path(__file__).resolve().parent
    snapshot=args.output/'source';snapshot.mkdir()
    source_hashes={}
    for name in ('certify_oracle_graph.py','oracle_certified_value.py','oracle_order_value.py',
                 'oracle_affine_value.py','oracle_full_support_value.py','gpu_oracle_bellman.py'):
        content=(source_dir/name).read_bytes();(snapshot/name).write_bytes(content)
        source_hashes[name]=hashlib.sha256(content).hexdigest()
    graph_hash=hashlib.sha256((args.graph/'graph.npz').read_bytes()).hexdigest()
    (args.output/'plan.json').write_text(json.dumps(dict(arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        sourceHashes=source_hashes,graphSha256=graph_hash,rules=asdict(Rules()),gamma=1,
        initialAccount=dict(quote=10000,asset=0,pendingOrders=[]),fundingFactor=1,
        terminal='Zero asset and empty book after valid size-limited closing actions.'),indent=2))
    torch.set_num_threads(4);set_num_threads(4)
    data=np.load(args.graph/'graph.npz');counts=data['counts'];p=data['price_ticks'];dest=data['destinations']
    H=len(counts)-1
    if counts[0]!=1 or len(p)!=H+1 or len(dest)!=H or H<1:
        raise ValueError('Invalid finite graph dimensions')
    if np.any(counts<1) or np.any(counts>p.shape[1]) or dest.shape[-1]<1:
        raise ValueError('Invalid finite graph node or outcome counts')
    for h in range(H):
        if np.any(dest[h,:counts[h]]<0) or np.any(dest[h,:counts[h]]>=counts[h+1]):
            raise ValueError('Invalid finite graph edge')
    if any(np.any(p[h,:counts[h]]<=0) for h in range(H+1)):
        raise ValueError('Nonpositive graph price')
    positive=p[p>0];price_low=int(positive.min());price_high=int(positive.max())
    gross_denominator=1000
    drift_denominator=3*price_low-2*price_high
    if drift_denominator<=0:raise ValueError('Price envelope does not provide a solvent reachable leverage bound')
    gross_numerator=(2*price_high*gross_denominator+drift_denominator-1)//drift_denominator
    if args.reachable_leverage and not args.order_aware:raise ValueError('Reachability refinement uses post-arrival order bounds')
    fee_pairs=[(5000,2000),(2000,2000),(2000,5000)]
    def stack(values):
        return ValuePlanes(np.stack([v.c for v in values],axis=1).reshape(-1,values[0].c.shape[1]),
            np.stack([v.d for v in values],axis=1).reshape(-1,values[0].d.shape[1]))
    if args.order_aware and args.directional:raise ValueError('Choose one bound construction')
    value=terminal_planes(int(counts[-1]),fee_denominator=2000) if args.order_aware else (stack([terminal_planes(int(counts[-1]),fee_denominator=b,sell_fee_denominator=s) for b,s in fee_pairs]) if args.directional else terminal_planes(int(counts[-1])))
    if args.affine:
        if args.save_values:raise ValueError('Affine replay storage is not yet supported')
        value=terminal_affine(int(counts[-1]))
    if args.full_support_class:
        if args.save_values:raise ValueError('Full-support diagnostic does not export a deployable policy')
        value=terminal_planes(int(counts[-1]),fee_denominator=2000)
    if args.save_values:
        if not (args.directional or args.order_aware):raise ValueError('Saved arrival values require execution coverage')
        face_count=6*args.points+(2 if args.reachable_leverage else 0) if args.order_aware else args.points+2
        saved_c=np.lib.format.open_memmap(args.output/'post-arrival-c.npy',mode='w+',dtype=np.int64,shape=(H+1,p.shape[1],face_count))
        saved_d=np.lib.format.open_memmap(args.output/'post-arrival-d.npy',mode='w+',dtype=np.int64,shape=saved_c.shape)
        terminal=terminal_planes(int(counts[-1]),fee_denominator=2000)
        for j in range(face_count):
            saved_c[H,:counts[-1],j]=terminal.c[:,j%2];saved_d[H,:counts[-1],j]=terminal.d[:,j%2]
    start=time.perf_counter()
    history=[]
    for h in range(H-1,-1,-1):
        # Root starts cash with an empty book; only taker market execution can
        # change its balances immediately. Its actual entry leverage is 2.
        child_p=p[h+1,:counts[h+1]];parent_p=p[h,:counts[h]];ds=dest[h,:counts[h]]
        if args.full_support_class:
            value=full_support_backup(value,child_p,parent_p,ds,points=args.points)
        elif args.affine:
            value=affine_backup(value,child_p,parent_p,ds,points=args.points,leverage=2 if h==0 else 3)
        elif args.order_aware:
            value=rational_order_backup(value,child_p,parent_p,ds,args.points,2 if h==0 else gross_numerator,1 if h==0 else gross_denominator) if args.reachable_leverage else order_backup(value,child_p,parent_p,ds,points=args.points,leverage=2 if h==0 else 3)
            if args.save_values:
                saved_c[h,:counts[h]]=value.c;saved_d[h,:counts[h]]=value.d
        elif args.directional:
            ds=3*ds+np.sign(child_p[ds]-parent_p[:,None]).astype(np.int32)+1
            child_p=np.repeat(child_p,3)
            variants=[supporting_backup(value,child_p,parent_p,ds,points=args.points,
                leverage=2 if h==0 else 3,fee_denominator=b,sell_fee_denominator=s)
                for b,s in ([(2000,2000)] if h==0 else fee_pairs)]
            if args.save_values:
                post=variants[0] if h==0 else variants[1]
                saved_c[h,:counts[h]]=post.c;saved_d[h,:counts[h]]=post.d
            value=variants[0] if h==0 else stack(variants)
        else:
            value=supporting_backup(value,child_p,parent_p,ds,
                points=args.points,leverage=2 if h==0 else 3,fee_numerator=1,fee_denominator=2000 if h==0 else 5000)
        if h%300==0:
            effective_c=value.c+value.beta if args.affine else value.c
            record=dict(depth=h,remaining=H-h,cashUpperRatio=float(effective_c.min(1).max()/SCALE),seconds=time.perf_counter()-start)
            history.append(record);print(json.dumps(record),flush=True)
    numerator,denominator=(int((value.c+value.beta)[0].min()),SCALE) if args.affine else upper_cash_ratio(value)
    if args.save_values:saved_c.flush();saved_d.flush()
    exported=dict(c=value.c,d=value.d)
    if args.affine:exported.update(beta=value.beta,theta=value.theta)
    np.savez_compressed(args.output/'root-planes.npz',**exported)
    report=dict(horizonSeconds=H,supportingPoints=args.points,fullSupportComparisonClass=args.full_support_class,affineQuoteCapCoverage=args.affine,directionalFillCoverage=args.directional,predictableOrderCoverage=args.order_aware or args.affine or args.full_support_class,
        reachableLeverageRefinement=args.reachable_leverage,reachableLeverageNumerator=gross_numerator,reachableLeverageDenominator=gross_denominator,
        graphPriceTicksRange=[price_low,price_high],forecastNodes=int(counts.sum()),
        forecastEdges=int(counts[:-1].sum()*dest.shape[-1]),
        graphSha256=graph_hash,sourceHashes=source_hashes,
        cashPolicyWealth=10000,meanWealthUpperNumerator=numerator*10000,meanWealthUpperDenominator=denominator,
        ceRelativeRegretUpperNumerator=numerator-denominator,ceRelativeRegretUpperDenominator=denominator,
        ceRelativeRegretUpperPercent=100*(numerator/denominator-1),
        belowPointOnePercent=(numerator-denominator)*1000<denominator,
        accountControlScope=('Only finite-log-utility policies extendable to NSF unbounded-upward and tick-floor support; this does NOT bound unrestricted bounded-graph policies.' if args.full_support_class else 'All feasible continuous Oracle trades and limit orders; no account/action grid. Original fees, unit funding, zero-account liquidation, flat ending.'),
        marketLawScope='Only the frozen finite conditional graph; NSF sample/quantization error is separate and not certified.',
        uniformNeuralResidualUsed=False,integerInequalitiesChecked=True,
        leanScope='Supporting-plane domination, conditional coefficient backup, actual rebalance and liquidation lemmas, policy induction and Jensen. The Python verifier and predictable-order/reachability refinements have not been fully refined into Lean.',
        fullNsfPlanningAccuracyCertified=False,
        seconds=time.perf_counter()-start,history=history)
    (args.output/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)


if __name__=='__main__':main()
