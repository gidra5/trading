"""Freeze independent one-second NSF paths for planning accuracy diagnostics."""
import argparse
import json
import hashlib
import time
from pathlib import Path
import numpy as np
import torch
from latent_neural_stochastic_flow import load_latent_nsf
from train_latent_neural_stochastic_flow import latent_data
from target_horizon_forecast import HorizonRange
from oracle_continuation_value import timeline_inputs, conditional_children


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--paths', type=int, default=64)
    ap.add_argument('--horizon', type=int, default=3600)
    ap.add_argument('--origin', type=int, default=0)
    ap.add_argument('--split', default='validation', choices=['train', 'validation', 'test'])
    ap.add_argument('--seed', type=int, default=20260926)
    args = ap.parse_args()
    args.output.mkdir()
    torch.set_num_threads(4); torch.set_float32_matmul_precision('high')
    device = torch.device('cuda')
    checkpoint = Path('data/backtests/oracle-latent-nsf-sampled-skips/latentSampledSkipFlow.pt')
    source = json.loads((checkpoint.parent/'plan.json').read_text())
    model, _ = load_latent_nsf(checkpoint, device)
    data = latent_data(Path(source['nativeDirectory']), Path(source['dataset']), args.split, HorizonRange(21600., 1))
    f,p,t = timeline_inputs(data, np.full(args.paths,args.origin), device)
    prices = np.empty((args.paths,args.horizon+1)); prices[:,0] = p[:,-1].cpu()
    started = time.perf_counter()
    for i in range(args.horizon):
        # Separate parent random draws; keep the first child from each pair.
        children = conditional_children(model, f,p,t,2,args.seed+i)
        f,p,t = [v[:,0] for v in children]
        prices[:,i+1] = p[:,-1].cpu()
        if (i+1)%600 == 0:
            print(json.dumps(dict(ticks=i+1,seconds=time.perf_counter()-started,
                terminalReturnMean=float((prices[:,i+1]/prices[:,0]-1).mean()))),flush=True)
    np.save(args.output/'prices.npy', prices)
    plan = dict(horizon=args.horizon,paths=args.paths,origin=args.origin,split=args.split,seed=args.seed,
        initialPrice=float(prices[0,0]),checkpoint=str(checkpoint),checkpointSha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        law='Independent recursively re-encoded one-second NSF draws; not a finite tree policy or a full-law certificate',
        seconds=time.perf_counter()-started)
    (args.output/'plan.json').write_text(json.dumps(plan,indent=2))
    print(json.dumps(plan),flush=True)


if __name__=='__main__':main()
