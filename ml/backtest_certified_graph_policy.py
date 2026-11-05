"""Evaluate feasible Oracle feedback using the certified continuation planes.

This Monte Carlo policy evaluation is an empirical lower estimate, not an
exact expected-value certificate. The independent upper retains its scope.
"""
import argparse
import hashlib
from dataclasses import asdict
import json
import time
from pathlib import Path
import numpy as np
import torch
from numba import set_num_threads
from oracle_plane_policy import plane_policy_action
from gpu_oracle_bellman import TensorOracle,Menu


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--graph',type=Path,required=True)
    ap.add_argument('--value',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--paths',type=int,default=64)
    ap.add_argument('--step',type=float,default=.1)
    ap.add_argument('--limit-fractions',type=float,nargs='+',default=[.25])
    ap.add_argument('--limit-offsets',type=int,nargs='+',default=[1])
    ap.add_argument('--seed',type=int,default=20260928)
    args=ap.parse_args();args.output.mkdir()
    menu=Menu(step=args.step,limit_fractions=tuple(args.limit_fractions),limit_offsets=tuple(args.limit_offsets))
    source_dir=Path(__file__).resolve().parent
    snapshots=args.output/'source';snapshots.mkdir()
    source_hashes={}
    for name in ('backtest_certified_graph_policy.py','oracle_plane_policy.py','oracle_certified_value.py','gpu_oracle_bellman.py'):
        content=(source_dir/name).read_bytes();(snapshots/name).write_bytes(content)
        source_hashes[name]=hashlib.sha256(content).hexdigest()
    (args.output/'plan.json').write_text(json.dumps(dict(arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        sourceHashes=source_hashes,menu=asdict(menu),gamma=1,
        graphSha256=hashlib.sha256((args.graph/'graph.npz').read_bytes()).hexdigest(),
        valueReport=json.loads((args.value/'report.json').read_text())),indent=2))
    torch.set_num_threads(4);set_num_threads(4);device='cuda';oracle=TensorOracle()
    graph=np.load(args.graph/'graph.npz');counts=graph['counts'];p=graph['price_ticks'];dest=graph['destinations'];H=len(counts)-1;B=dest.shape[-1]
    C=np.load(args.value/'post-arrival-c.npy',mmap_mode='r');D=np.load(args.value/'post-arrival-d.npy',mmap_mode='r')
    state=torch.zeros((args.paths,9),dtype=torch.float64,device=device);state[:,0]=10000;state[:,2:4]=p[0,0]/100
    nodes=np.zeros(args.paths,np.int64);rng=np.random.default_rng(args.seed);desc=oracle.descriptors(menu,device)
    trace=np.lib.format.open_memmap(args.output/'execution.npy',mode='w+',dtype=np.float64,shape=(H,args.paths,3,9))
    started=time.perf_counter();markets=orders=cancels=0;failures=[];record=[]
    for h in range(H):
        ds=dest[h,nodes];future_price=torch.tensor(p[h+1,ds]/100,device=device)
        result=plane_policy_action(oracle,state,desc,future_price,ds,
            np.asarray(C[h+1,:counts[h+1]]),np.asarray(D[h+1,:counts[h+1]]),H-h)
        if not result['valid'].all():
            failures.append(dict(tick=h,accounts=int((~result['valid']).sum()),reason='No feasible selected continuation'));break
        selected=result['state'];markets+=int((result['amount']!=0).sum())
        if result['descriptor'] is None:
            cancels+=int((state[:,4]!=0).sum())
        else:
            orders+=int((result['descriptor'][:,2]!=0).sum());cancels+=int((result['descriptor'][:,1]!=0).sum())
        trace[h,:,0]=state.cpu().numpy();trace[h,:,1]=selected.cpu().numpy()
        edge=rng.integers(0,B,args.paths);nodes=ds[np.arange(args.paths),edge]
        state=oracle.advance(selected,torch.tensor(p[h+1,nodes]/100,device=device))
        trace[h,:,2]=state.cpu().numpy()
        if (h+1)%300==0:
            q=state[:,0]+state[:,1]*state[:,2]
            r=dict(ticks=h+1,meanMarkedWealth=float(q.mean()),markets=markets,orders=orders,seconds=time.perf_counter()-started)
            record.append(r);print(json.dumps(r),flush=True)
    terminal=oracle.terminal(state).cpu().numpy() if not failures else np.array([])
    complete=not failures and np.isfinite(terminal).all() and (terminal>0).all()
    logs=np.log(terminal/10000) if complete else np.array([])
    report=dict(complete=bool(complete),paths=args.paths,horizonSeconds=H,exposureGrid=args.step,
        meanTerminalWealth=float(terminal.mean()) if complete else None,
        meanLogReturn=float(logs.mean()) if complete else None,
        ceReturnPercent=float(100*np.expm1(logs.mean())) if complete else None,
        standardErrorLog=float(logs.std(ddof=1)/np.sqrt(len(logs))) if complete else None,
        minTerminalWealth=float(terminal.min()) if complete else None,maxTerminalWealth=float(terminal.max()) if complete else None,
        markets=markets,orders=orders,cancellations=cancels,failures=failures,
        independentPolicyEvaluationSeed=args.seed,exactLowerCertificate=False,
        scope='Monte Carlo evaluation on the frozen finite graph, not a realized-market backtest or a full NSF law certificate.',
        menu=asdict(menu),seconds=time.perf_counter()-started,progress=record)
    trace.flush()
    np.save(args.output/'terminal-wealth.npy',terminal)
    (args.output/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)


if __name__=='__main__':main()
