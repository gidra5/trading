"""Profile a warmed finite-grid GPU sweep, separating kernels from transfers."""
import argparse
from dataclasses import fields,replace,asdict
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from oracle_account_grid import AccountGrid
from oracle_book_mc import rule_vector
from oracle_book_gpu import GPUBookSweep
from oracle_dense_value_map import MarketCloud
from gpu_oracle_bellman import Rules
from oracle_dense_grid_config import exposure_axis, future_candidate_count


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--depth',type=int,default=60)
    p.add_argument('--exposure-step',type=float)
    p.add_argument('--candidates',type=int)
    args=p.parse_args()
    plan=json.loads((args.source/'plan.json').read_text())
    cloud=MarketCloud.load(args.source/'update-0'/'market.npz')
    H=args.depth
    if not 2<=H<len(cloud.counts):raise ValueError('Depth outside saved horizon')
    data={f.name:getattr(cloud,f.name) for f in fields(MarketCloud)}
    for key in ('prices','counts','masses','features','price_pairs'):data[key]=data[key][:H+1]
    for key in ('destinations','sampled_prices'):data[key]=data[key][:H]
    cloud=MarketCloud(**data)
    config={**plan['grid'],'exposure_points':tuple(plan['grid']['exposure_points']),
        'wealth_points':tuple(plan['grid']['wealth_points'])}
    g=AccountGrid(**config);r=rule_vector(Rules(**plan['rules']),plan['fundingMultiplier'])
    if args.exposure_step is not None:g=replace(g,exposure_points=exposure_axis(args.exposure_step))
    candidates_count=future_candidate_count(g,args.candidates if args.candidates is not None else
        (None if args.exposure_step is not None else plan['arguments']['candidates']))
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    gpu=GPUBookSweep(g,cloud,r)
    C=len(g.arrays()['coordinates']);B=cloud.destinations.shape[-1]
    schedules=[]
    for h in range(H):
        rng=np.random.default_rng(plan['arguments']['seed']+int(cloud.origin_seconds)+h)
        E=len(g.exposure_points);extra=max(0,min(C-E,candidates_count-E))
        candidates=np.r_[np.arange(E),rng.choice(np.arange(E,C),extra,replace=False)].astype(np.int64)
        permutations=np.array([rng.permutation(B) for _ in range(cloud.counts[h])])
        schedules.append((permutations,np.full(C,B),candidates))
    def sweep():
        for h in range(H-1,-1,-1):gpu.step(h,*schedules[h])
        gpu.synchronize()
    sweep()  # exclude compilation from both timing and trace
    timings=[]
    for _ in range(3):
        started=time.perf_counter();sweep();timings.append(time.perf_counter()-started)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                          torch.profiler.ProfilerActivity.CUDA]) as prof:
        sweep()
    events=[dict(name=e.key,count=e.count,cpuSelfUs=e.self_cpu_time_total,
                 deviceSelfUs=e.self_device_time_total) for e in prof.key_averages()]
    events.sort(key=lambda e:e['deviceSelfUs'],reverse=True)
    result=dict(depth=H,grid=asdict(g),candidates=candidates_count,
        peakAllocatedBytes=torch.cuda.max_memory_allocated(),unprofiledSeconds=timings,events=events,
        source=str(args.source),device=torch.cuda.get_device_name(),
        sourceHashes={name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in
            ('ml/profile_oracle_dense_gpu.py','ml/oracle_book_gpu.py','ml/oracle_dense_grid_config.py')},
        interpretation='Warmed full-outcome sweep of the first saved layers; profiling adds overhead; no forecasting or disk export')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(depth=H,unprofiledSeconds=timings,topEvents=events[:8])),flush=True)


if __name__=='__main__':main()
