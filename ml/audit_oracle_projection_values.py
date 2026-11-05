"""Compare root decisions after rematching market nodes, keeping a fixed tail.

Recomputes one complete CPU Bellman layer with fresh NSF root samples. The
archived continuation remains fixed at its original terminal date. This is a
projection sensitivity experiment, not a reference to the true optimal policy.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from numba import set_num_threads
from audit_oracle_market_projection import input_coordinates,nearest,posterior_coordinates
from latent_neural_stochastic_flow import load_latent_nsf
from train_latent_neural_stochastic_flow import latent_data
from target_horizon_forecast import HorizonRange
from oracle_continuation_value import timeline_inputs
from oracle_dense_value_map import DenseValueMap
from oracle_book_mc import backup_layer,decide,post_value
from export_nsf_planning_trees import next_observations


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--map',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--offsets',type=int,nargs='+',default=[1,30,120,300,600,1200,1800])
    parser.add_argument('--replicates',type=int,default=2);args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1);set_num_threads(1)
    started=time.perf_counter();m=DenseValueMap.load(args.map);g=m.grid;a=m.a;r=m.rules;c=m.cloud
    if g.max_orders<2:raise ValueError('Carried-state comparison requires at least two order slots')
    source=json.loads((args.map.parent/'plan.json').read_text())
    checkpoint=Path('data/backtests/oracle-latent-nsf-sampled-skips/latentSampledSkipFlow.pt')
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=source['checkpointSha256']:raise ValueError('Checkpoint mismatch')
    model,_=load_latent_nsf(checkpoint,'cpu');training=json.loads((checkpoint.parent/'plan.json').read_text())
    data=latent_data(Path(training['nativeDirectory']),Path(training['dataset']),'test',HorizonRange(21600.,1))
    mean=model.feature_mean.numpy();std=model.feature_std.numpy();E=len(a['exposures']);C=len(a['coordinates']);B=8
    if args.replicates<1:raise ValueError('Positive replicate count required')
    results=[]
    for offset,replicate in ((o,k) for o in args.offsets for k in range(args.replicates)):
        if not 0<offset<m.shape[0]-1:raise ValueError('Need at least two remaining transitions')
        f,p,t=[x.numpy()[0] for x in timeline_inputs(data,np.array([source['arguments']['origin']+offset]),'cpu')]
        if t[-1]!=c.origin_seconds+offset:raise ValueError('Incorrect observed timestamp')
        seed=source['arguments']['seed']+source['arguments']['origin']+offset+100000*replicate
        future_f,future_p=next_observations(model,f[None],p[None],t[None],np.ones(1),B,seed)
        raw_f=np.stack((np.repeat(f[-1][None],B,axis=0),future_f[0]),1)
        raw_p=np.stack((np.full(B,p[-1]),future_p[0]),1)
        n=c.counts[offset+1];ret_f=c.features[offset+1,:n];ret_p=c.price_pairs[offset+1,:n]
        raw_t=np.tile([t[-1],t[-1]+1],(B,1));ret_t=np.tile(raw_t[0],(n,1))
        mappings={
            'absolute':nearest(input_coordinates(raw_f,raw_p,mean,std,p[-1]),input_coordinates(ret_f,ret_p,mean,std,p[-1])),
            'relative':nearest(input_coordinates(raw_f,raw_p,mean,std),input_coordinates(ret_f,ret_p,mean,std)),
            'posterior':nearest(posterior_coordinates(model,raw_f,raw_p,raw_t),posterior_coordinates(model,ret_f,ret_p,ret_t)),
        }
        rng=np.random.default_rng(source['arguments']['seed']+int(c.origin_seconds)+offset)
        extra=max(0,min(C-E,source['arguments']['candidates']-E))
        candidates=np.r_[np.arange(E),rng.choice(np.arange(E,C),extra,replace=False)].astype(np.int64)
        child=m.values[offset+1] if offset<m.shape[0]-2 else np.full_like(m.values[offset+1],-np.inf)
        fields={}
        for metric,indices in mappings.items():
            fields[metric],_=backup_layer(np.array([p[-1]]),future_p,indices[None],np.arange(B)[None],
                np.full(C,B),child,a['wealth'],a['exposures'],a['books'],a['book_counts'],a['code_offsets'],
                a['book_lookup'],a['size_edges'],g.price_bins,g.price_scale,r,candidates,False,c.factors(offset+1))
        states={};tick=r[9];buy=(round(p[-1]/tick)-3)*tick;sell=(round(p[-1]/tick)+3)*tick
        for name,exposure,orders in (('cash',0.,()),('long-buy',.5,((1.,buy,1000.),)),
            ('short-sell',-.5,((-1.,sell,1000.),)),('two-orders',0.,((1.,buy,2000.),(-1.,sell,2000.)))):
            state=np.r_[10000*(1-exposure),10000*exposure/p[-1],p[-1],p[-1],np.zeros(3*g.max_orders)]
            for j,order in enumerate(orders):state[4+3*j:7+3*j]=order
            states[name]=state
        factors=future_p/p[-1];all_candidates=np.arange(C,dtype=np.int64)
        for name,state in states.items():
            choices={}
            for metric,field in fields.items():
                value,post,_=decide(state,0.,False,0,field,a['wealth'],a['exposures'],a['books'],a['book_counts'],
                    a['code_offsets'],a['book_lookup'],a['size_edges'],g.price_bins,g.price_scale,r,all_candidates,factors)
                choices[metric]=(value,post)
            for metric,(value,post) in choices.items():
                absolute_value=post_value(post,0,fields['absolute'],a['wealth'],a['exposures'],a['code_offsets'],
                    a['book_lookup'],a['size_edges'],g.price_bins,g.price_scale,r,factors)
                results.append(dict(offset=offset,replicate=replicate,remainingSeconds=m.shape[0]-offset,
                    initialState=name,metric=metric,value=float(value),state=post.tolist(),
                    sameActionAsAbsolute=bool(np.array_equal(post,choices['absolute'][1])),
                    valueShiftLog=float(value-choices['absolute'][0]),
                    selectedActionGapUnderAbsoluteBps=float(-np.expm1(absolute_value-choices['absolute'][0])*10000.)
                        if np.isfinite(absolute_value) else None))
        (args.output/'results.json').write_text(json.dumps(dict(rows=results,elapsedSeconds=time.perf_counter()-started,
            sourceMap=str(args.map),operator=json.loads((args.map/'operator.json').read_text()),
            auditSha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            scope='Fresh CPU root samples and one recomputed Bellman layer. Fixed archived tail and finite menus; original terminal date, not a rolling one-hour horizon. Cross-scoring against the absolute metric is a sensitivity measure, not true policy regret.'),indent=2))
        print(json.dumps(dict(offset=offset,replicate=replicate,elapsedSeconds=time.perf_counter()-started)),flush=True)


if __name__=='__main__':main()
