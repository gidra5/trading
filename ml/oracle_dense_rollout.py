"""Roll a finite observable-market grid without rebuilding its fixed kernels.

For a fixed model, the conditional law at an unchanged observation pair and
absolute timestamp does not change when another real observation arrives.
The root row is freshly predicted; the interior is shifted; one tail layer
is appended. All marginal masses are recomputed from the new root. Projection
onto the retained grid remains an approximation, measured explicitly here.
"""
import time
import numpy as np
import torch
from export_nsf_planning_trees import next_observations
from nsf_scenario_graph import representatives
from oracle_dense_value_map import MarketCloud


def coordinates(features,prices,model,price_resolution_bps,price_anchor):
    device=next(model.parameters()).device
    f=torch.as_tensor(features,dtype=torch.float64,device=device)
    p=torch.as_tensor(prices,dtype=torch.float64,device=device)
    standard=((f-model.feature_mean)/model.feature_std).flatten(1)/np.sqrt(f.shape[1]*f.shape[2])
    return torch.cat((standard,(p/price_anchor).log()/(price_resolution_bps/10000)),1)


@torch.inference_mode()
def roll_market_cloud(previous,model,features,prices,timestamps,seed,price_resolution_bps=1.,predict=next_observations,append_tail=True):
    started=time.perf_counter();oldH=len(previous.counts)-1;H=oldH if append_tail else oldH-1
    M=previous.prices.shape[1];B=previous.destinations.shape[-1]
    if oldH<2 or B<2 or B%2:raise ValueError('Need at least two one-second layers and even branching')
    f=np.asarray(features);p=np.asarray(prices);t=np.asarray(timestamps)
    if f.shape!=(2,59) or p.shape!=(2,) or t.shape!=(2,) or not np.isfinite(f).all() or np.any(~np.isfinite(p)|(p<=0)):
        raise ValueError('Invalid observed market pair')
    if t[-1]!=previous.origin_seconds+1 or t[-1]-t[0]!=1:raise ValueError('Expected the next observed second')
    if not np.isfinite(price_resolution_bps) or price_resolution_bps<=0:raise ValueError('Invalid projection scale')
    device=next(model.parameters()).device
    # These are changed-value builders. The source cloud is never mutated.
    price=np.zeros_like(previous.prices[:H+1]);counts=np.zeros_like(previous.counts[:H+1]);dest=np.zeros_like(previous.destinations[:H])
    sampled=np.zeros_like(previous.sampled_prices[:H])
    feat=np.zeros_like(previous.features[:H+1]);pairs=np.zeros_like(previous.price_pairs[:H+1]);mass=np.zeros_like(previous.masses[:H+1])
    price[0,0]=p[-1];counts[0]=1;feat[0,0]=f;pairs[0,0]=p;mass[0,0]=1.
    retained_end=H if append_tail else H+1
    for output,source in ((price,previous.prices),(counts,previous.counts),(feat,previous.features),(pairs,previous.price_pairs)):
        output[1:retained_end]=source[2:retained_end+1]
    dest[1:retained_end-1]=previous.destinations[2:retained_end]
    sampled[1:retained_end-1]=previous.sampled_prices[2:retained_end]
    future_f,future_p=predict(model,f[None],p[None],t[None],np.ones(1),B,seed)
    sampled[0,0]=future_p[0]
    raw_f=np.stack((np.repeat(f[-1][None],B,axis=0),future_f[0]),1)
    raw_p=np.stack((np.full(B,p[-1]),future_p[0]),1)
    raw_coordinates=coordinates(raw_f,raw_p,model,price_resolution_bps,p[-1])
    retained=coordinates(feat[1,:counts[1]],pairs[1,:counts[1]],model,price_resolution_bps,p[-1])
    # Direct differences avoid cancellation in cdist's quadratic expansion for
    # large absolute log-price coordinates.
    distance=(raw_coordinates[:,None]-retained[None]).square().sum(2)
    root_indices=distance.argmin(1).cpu().numpy();dest[0,0]=root_indices
    projected=price[1,root_indices]
    root_shift=float(np.mean((projected-future_p[0])/p[-1])*10000)
    root_rmse=float(np.sqrt(np.mean(np.log(projected/future_p[0])**2))*10000)
    for h in range(retained_end-1):
        n=counts[h]
        mass[h+1]=np.bincount(dest[h,:n].ravel(),weights=np.repeat(mass[h,:n]/B,B),minlength=M)
    if not append_tail:
        cloud=MarketCloud(price,counts,dest,mass,feat,pairs,float(t[-1]),sampled)
        return cloud,dict(seconds=time.perf_counter()-started,predictedParentRows=1,
            reusedConditionalRows=int(counts[1:H].sum()),rootProjectionMeanReturnShiftBps=root_shift,
            rootProjectionPriceRMSEBps=root_rmse,rootExecutionPriceRMSEBps=0.,rootExecutionMeanReturnShiftBps=0.,
            rootProjectionCoordinateRMSE=float(distance.min(1).values.mean().sqrt()),
            maximumMassNormalizationError=float(np.max(abs(mass.sum(1)-1))),
            approximation='Fresh root execution law with retained interior kernels and unchanged terminal date')
    n=counts[H-1]
    tail_times=np.tile([t[-1]+H-2,t[-1]+H-1],(n,1))
    tail_f,tail_p=predict(model,feat[H-1,:n],pairs[H-1,:n],tail_times,np.ones(n),B,seed+H-1)
    generated_f=np.stack((np.repeat(feat[H-1,:n,-1],B,axis=0),tail_f.reshape(-1,59)),1)
    generated_p=np.stack((np.repeat(price[H-1,:n],B),tail_p.reshape(-1)),1)
    points=coordinates(generated_f,generated_p,model,price_resolution_bps,p[-1])
    # Actual reachability masses can be zero after a changed root row. A tiny
    # fitting weight keeps every sample assignable; decision probabilities
    # and reported masses continue to use the original normalized weights.
    actual_mass=np.repeat(mass[H-1,:n]/B,B)
    fitting=torch.as_tensor(np.maximum(actual_mass,1.e-30),device=device)
    selected,assignment=representatives(points,fitting,M)
    chosen=selected.cpu().numpy();assigned=assignment.cpu().numpy();k=len(chosen)
    counts[H]=k;feat[H,:k]=generated_f[chosen];pairs[H,:k]=generated_p[chosen];price[H,:k]=generated_p[chosen,-1]
    dest[H-1,:n]=assigned.reshape(n,B)
    sampled[H-1,:n]=tail_p
    mass[H]=np.bincount(assigned,weights=actual_mass,minlength=M)
    cloud=MarketCloud(price,counts,dest,mass,feat,pairs,float(t[-1]),sampled)
    tail_projected=price[H,assigned].reshape(n,B)
    diagnostics=dict(seconds=time.perf_counter()-started,predictedParentRows=1+int(n),
        reusedConditionalRows=int(counts[1:H-1].sum()),
        rootProjectionMeanReturnShiftBps=root_shift,rootProjectionPriceRMSEBps=root_rmse,
        rootExecutionPriceRMSEBps=float(np.sqrt(np.mean(np.log(sampled[0,0]/future_p[0])**2))*10000),
        rootExecutionMeanReturnShiftBps=float(np.mean((sampled[0,0]-future_p[0])/p[-1])*10000),
        rootProjectionCoordinateRMSE=float(distance.min(1).values.mean().sqrt()),
        tailProjectionPriceRMSEBps=float(np.sqrt(np.sum(actual_mass*np.log(tail_projected.ravel()/tail_p.ravel())**2))*10000),
        maximumMassNormalizationError=float(np.max(abs(mass.sum(1)-1))),
        approximation='Raw sampled execution marks; continuation lookup uses retained observable states and reanchored conditional moves; fresh root and tail')
    return cloud,diagnostics
