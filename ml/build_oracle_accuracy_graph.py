"""Freeze the complete every-second finite forecast law before certification."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from latent_neural_stochastic_flow import load_latent_nsf
from train_latent_neural_stochastic_flow import latent_data
from target_horizon_forecast import HorizonRange
from oracle_continuation_value import timeline_inputs
from nsf_scenario_graph import build_graph


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--horizon',type=int,default=3600)
    ap.add_argument('--width',type=int,default=256)
    ap.add_argument('--samples',type=int,default=64)
    ap.add_argument('--split',default='validation',choices=['validation','test'])
    ap.add_argument('--origin',type=int,default=0)
    ap.add_argument('--seed',type=int,default=20260927)
    ap.add_argument('--save-observations',action='store_true')
    args=ap.parse_args();args.output.mkdir()
    torch.set_num_threads(4);torch.set_float32_matmul_precision('high')
    checkpoint=Path('data/backtests/oracle-latent-nsf-sampled-skips/latentSampledSkipFlow.pt')
    source=json.loads((checkpoint.parent/'plan.json').read_text())
    model,_=load_latent_nsf(checkpoint,torch.device('cuda'))
    data=latent_data(Path(source['nativeDirectory']),Path(source['dataset']),args.split,HorizonRange(21600.,1))
    f,p,t=[x.cpu().numpy()[0] for x in timeline_inputs(data,np.array([args.origin]),'cpu')]
    plan=dict(horizon=args.horizon,width=args.width,samples=args.samples,origin=args.origin,split=args.split,seed=args.seed,
        checkpointSha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        scope='Frozen finite conditional forecast graph; every second retained. Not a certificate of NSF quantization or sampling error.',
        originTimestampSeconds=float(t[-1]),originPrice=float(p[-1]),saveObservations=args.save_observations,
        sourceHashes={name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in ['ml/build_oracle_accuracy_graph.py','ml/nsf_scenario_graph.py','ml/export_nsf_planning_trees.py','ml/latent_neural_stochastic_flow.py']})
    snapshots=args.output/'source';snapshots.mkdir()
    for name in plan['sourceHashes']:(snapshots/Path(name).name).write_bytes(Path(name).read_bytes())
    (args.output/'plan.json').write_text(json.dumps(plan,indent=2))
    started=time.perf_counter()
    def progress(row):
        if row['depth']%300==0:print(json.dumps(dict(depth=row['depth'],seconds=time.perf_counter()-started)),flush=True)
    graph=build_graph(model,f,p,t,list(range(1,args.horizon+1)),width=args.width,samples=args.samples,seed=args.seed,on_layer=progress)
    # Integer cent prices and integer sample counts admit independent rational
    # checking without storing the neural feature arrays in the certificate.
    counts=np.array([len(x.prices) for x in graph.layers],dtype=np.int32)
    prices=np.zeros((args.horizon+1,args.width),dtype=np.int64)
    dest=np.zeros((args.horizon,args.width,args.samples),dtype=np.int32)
    for h,layer in enumerate(graph.layers):prices[h,:counts[h]]=np.rint(layer.prices[:,-1]*100).astype(np.int64)
    for h,d in enumerate(graph.destinations):dest[h,:counts[h]]=d.cpu().numpy()
    np.savez_compressed(args.output/'graph.npz',counts=counts,price_ticks=prices,destinations=dest)
    observation_hashes={}
    if args.save_observations:
        features=np.lib.format.open_memmap(args.output/'features.npy',mode='w+',dtype=np.float32,shape=(args.horizon+1,args.width,2,59))
        pair_ticks=np.lib.format.open_memmap(args.output/'observation-price-ticks.npy',mode='w+',dtype=np.int64,shape=(args.horizon+1,args.width,2))
        for h,layer in enumerate(graph.layers):
            features[h,:counts[h]]=layer.features
            pair_ticks[h,:counts[h]]=np.rint(layer.prices*100).astype(np.int64)
        features.flush();pair_ticks.flush()
        observation_hashes={name:hashlib.sha256((args.output/name).read_bytes()).hexdigest() for name in ['features.npy','observation-price-ticks.npy']}
    result=dict(seconds=time.perf_counter()-started,nodes=int(counts.sum()),edges=int(counts[:-1].sum()*args.samples),
        initialPrice=float(p[-1]),graphSha256=hashlib.sha256((args.output/'graph.npz').read_bytes()).hexdigest(),observationHashes=observation_hashes,diagnostics=graph.diagnostics)
    (args.output/'construction.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='diagnostics'}),flush=True)


if __name__=='__main__':main()
