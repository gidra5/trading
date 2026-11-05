"""Compare market-node matching by its error in the model's next-price law.

CPU-only diagnostic. Uses common random numbers across candidate projections,
not realized future prices. It neither changes the running map nor its policy.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from latent_neural_stochastic_flow import load_latent_nsf
from train_latent_neural_stochastic_flow import latent_data
from target_horizon_forecast import HorizonRange
from oracle_continuation_value import timeline_inputs
from oracle_dense_value_map import MarketCloud
from export_nsf_planning_trees import next_observations


def input_coordinates(features,prices,mean,std,anchor=None):
    f=((features-mean)/std).reshape(len(features),-1)/np.sqrt(np.prod(features.shape[1:]))
    # A common anchor reproduces the current absolute-price metric. A separate
    # last price per observation pair matches the NSF encoder's price inputs.
    anchor=prices[:,-1,None] if anchor is None else anchor
    return np.concatenate((f,np.log(prices/anchor)/.0001),axis=1)


def nearest(left,right):
    return np.argmin(np.sum((left[:,None]-right[None])**2,axis=2),axis=1)


@torch.inference_mode()
def posterior_coordinates(model,features,prices,times):
    f=torch.as_tensor(features,dtype=torch.float32);p=torch.as_tensor(prices,dtype=torch.float64)
    t=torch.as_tensor(times,dtype=torch.float64)
    obs=model.observations(f,(p/p[:,-1,None]).log())
    dt=torch.cat((torch.zeros_like(t[:,:1]),t[:,1:]-t[:,:-1]),1).float()
    mean,std=model.encode(obs,dt,t)
    # Euclidean distance of (mean,std) is W2 for diagonal Gaussian posteriors.
    return torch.cat((mean[:,-1],std[:,-1]),1).numpy()


def law_errors(actual,candidate,anchors):
    a=np.log(actual/anchors[:,None]);b=np.log(candidate/anchors[:,None])
    return dict(meanAbsoluteConditionalMeanErrorBps=float(np.mean(abs(a.mean(1)-b.mean(1)))*10000),
        empiricalW2Bps=float(np.sqrt(np.mean((np.sort(a,axis=1)-np.sort(b,axis=1))**2))*10000),
        coupledSampleRMSEBps=float(np.sqrt(np.mean((a-b)**2))*10000))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source-map',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--offsets',type=int,nargs='+',default=[1,30,120,300,600,1200,1800])
    parser.add_argument('--branches',type=int,default=64)
    parser.add_argument('--replicates',type=int,default=1)
    parser.add_argument('--horizons',type=float,nargs='+',default=[1.,60.])
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    if args.branches<2 or args.branches%2 or args.replicates<1 or any(not 0<h<=86400 for h in args.horizons):
        raise ValueError('Require even branching, positive replicate count and supported positive horizons')
    torch.set_num_threads(1);started=time.perf_counter()
    source=json.loads((args.source_map.parent/'plan.json').read_text())
    cloud=MarketCloud.load(args.source_map/'market.npz')
    checkpoint=Path('data/backtests/oracle-latent-nsf-sampled-skips/latentSampledSkipFlow.pt')
    require_hash=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if require_hash!=source['checkpointSha256']:raise ValueError('Checkpoint mismatch')
    model,_=load_latent_nsf(checkpoint,'cpu')
    training=json.loads((checkpoint.parent/'plan.json').read_text())
    data=latent_data(Path(training['nativeDirectory']),Path(training['dataset']),'test',HorizonRange(21600.,1))
    mean=model.feature_mean.numpy();std=model.feature_std.numpy();rows=[]
    for offset,replicate in ((o,k) for o in args.offsets for k in range(args.replicates)):
        if not 0<offset<len(cloud.counts)-1:raise ValueError('Offset must remain inside the original cloud')
        f,p,t=[x.numpy()[0] for x in timeline_inputs(data,np.array([source['arguments']['origin']+offset]),'cpu')]
        if t[-1]!=cloud.origin_seconds+offset:raise ValueError('Observed timestamp differs from saved cloud')
        B=8;root_seed=source['arguments']['seed']+source['arguments']['origin']+offset+100000*replicate
        future_f,future_p=next_observations(model,f[None],p[None],t[None],np.ones(1),B,root_seed)
        raw_f=np.stack((np.repeat(f[-1][None],B,axis=0),future_f[0]),1)
        raw_p=np.stack((np.full(B,p[-1]),future_p[0]),1)
        n=int(cloud.counts[offset+1]);ret_f=cloud.features[offset+1,:n];ret_p=cloud.price_pairs[offset+1,:n]
        raw_t=np.tile([t[-1],t[-1]+1],(B,1));ret_t=np.tile(raw_t[0],(n,1))
        indices={
            'absolute':nearest(input_coordinates(raw_f,raw_p,mean,std,p[-1]),input_coordinates(ret_f,ret_p,mean,std,p[-1])),
            'relative':nearest(input_coordinates(raw_f,raw_p,mean,std),input_coordinates(ret_f,ret_p,mean,std)),
            'posterior':nearest(posterior_coordinates(model,raw_f,raw_p,raw_t),posterior_coordinates(model,ret_f,ret_p,ret_t)),
        }
        for horizon in args.horizons:
            seed=31000+offset+int(horizon)+100000*replicate
            _,actual=next_observations(model,raw_f,raw_p,raw_t,np.full(B,horizon),args.branches,seed)
            for name,ix in indices.items():
                # Reanchor every selected history at the actual sampled mark,
                # as the dense account evaluator does for relative moves.
                selected_p=ret_p[ix]*raw_p[:,-1,None]/ret_p[ix,-1,None]
                _,candidate=next_observations(model,ret_f[ix],selected_p,raw_t,np.full(B,horizon),args.branches,seed)
                rows.append(dict(offset=offset,replicate=replicate,horizonSeconds=horizon,metric=name,selectedNodes=ix.tolist(),
                    changedFromAbsolute=int(np.sum(ix!=indices['absolute'])),
                    **law_errors(actual,candidate,raw_p[:,-1])))
        (args.output/'results.json').write_text(json.dumps(dict(rows=rows,elapsedSeconds=time.perf_counter()-started,
            checkpointSha256=require_hash,auditSha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
            scope='CPU common-random-number comparison of conditional NSF price laws at sampled hypothetical histories. Not forecast accuracy against realized prices, value error, or a GPU replay reproduction.'),indent=2))
        print(json.dumps(dict(offset=offset,replicate=replicate,elapsedSeconds=time.perf_counter()-started)),flush=True)


if __name__=='__main__':main()
