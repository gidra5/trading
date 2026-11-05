"""Shared feedback policy for graph simulations and causal historical replay.

Continuation planes are approximate outside their original graph state domain.
Every action is averaged over its children before the action maximum is taken.
"""
import numpy as np
import torch
from numba import njit,prange
from oracle_certified_value import plane_hulls,SCALE
from gpu_oracle_bellman import Menu,trade,equity


def closing_action(oracle,state):
    arrived,consumed,filled,okay=oracle.arrival(state)
    effective=oracle.effective(arrived);absolute=effective.abs();available=oracle.r.maximum-consumed
    amount=effective.sign()*torch.minimum(absolute,available)
    dust=(absolute>0)&(absolute<oracle.r.minimum)
    amount=torch.where(dust,-effective.sign()*(2*oracle.r.minimum),amount)
    residual=absolute-amount.abs()
    small_residual=(residual>0)&(residual<2*oracle.r.minimum)&~dust
    amount=torch.where(small_residual,effective.sign()*(absolute-2*oracle.r.minimum),amount)
    out=trade(arrived,amount,oracle.r.taker)
    complete=amount==effective
    out[:,1]=torch.where(complete,0.,out[:,1]);out[:,4:8]=0
    w=equity(out);e=out[:,1]*out[:,2]/w
    okay &= oracle.size(amount)&oracle.le(consumed+amount.abs(),oracle.r.maximum)&(w>0)
    okay &= ~((amount!=0)|filled)|(oracle.le(-e,-oracle.r.lower)&oracle.le(e,oracle.r.upper))
    return out,amount,okay


def arrival_repairable(oracle,moved,arrived,okay,repair_descriptors):
    w=equity(arrived);e=arrived[...,1]*arrived[...,2]/torch.where(w!=0,w,1.)
    check=~(oracle.le(-e,-oracle.r.lower)&oracle.le(e,oracle.r.upper))
    if check.any():
        _,valid,_=oracle.actions(moved[check],repair_descriptors)
        okay=okay.clone();okay[check]&=valid.any(1)
    return okay


@njit(cache=True,parallel=True)
def account_values(c,d,ids,breaks,sizes,states,nodes):
    result=np.empty(len(states))
    for i in prange(len(states)):
        q=states[i,0];v=states[i,1]*states[i,2];w=q+v;n=nodes[i]
        if w<=0:result[i]=-np.inf;continue
        e=v/w;lo=0;hi=sizes[n]
        while lo+1<hi:
            middle=(lo+hi)//2
            if e>=breaks[n,middle]:lo=middle
            else:hi=middle
        face=ids[n,lo];upper=(c[n,face]*q+d[n,face]*v)/SCALE
        result[i]=np.log(upper/10000) if upper>0 else -np.inf
    return result


@torch.inference_mode()
def plane_policy_action(oracle,state,descriptors,future_prices,child_nodes,c,d,remaining,chunk=65536):
    """Use the SAME conditional price row for all actions of each account.

    child_nodes only select continuation functions. Execution uses the supplied
    future_prices unchanged, including when these are fresh off-graph draws.
    """
    if remaining<=8:
        out,amount,valid=closing_action(oracle,state)
        return dict(state=out,amount=amount,valid=valid,value=torch.where(valid,0.,-torch.inf),
            descriptor=None,scores=None)
    if future_prices.shape!=child_nodes.shape or len(future_prices)!=len(state):
        raise ValueError('Conditional price/node rows must match accounts')
    out,valid,amount=oracle.actions(state,descriptors)
    row,action=valid.nonzero(as_tuple=True)
    scores=torch.full(valid.shape,-torch.inf,device=state.device,dtype=torch.float64)
    B=future_prices.shape[1];hull=plane_hulls(c,d)
    repair=oracle.descriptors(Menu(step=2.,limits=False),state.device)
    batch=max(1,chunk//B)
    for begin in range(0,len(row),batch):
        r=row[begin:begin+batch];a=action[begin:begin+batch]
        moved=oracle.advance(out[r,a,None,:].expand(-1,B,-1),future_prices[r])
        arrived,_,_,okay=oracle.arrival(moved)
        okay=arrival_repairable(oracle,moved,arrived,okay,repair)
        nodes=child_nodes[r.cpu().numpy()].reshape(-1)
        values=account_values(c,d,*hull,arrived.cpu().numpy().reshape(-1,9),nodes).reshape(-1,B)
        score=torch.tensor(values,device=state.device)
        scores[r,a]=torch.where(okay,score,-torch.inf).mean(1)
    best,choice=scores.max(1);ix=torch.arange(len(state),device=state.device)
    return dict(state=out[ix,choice],amount=amount[ix,choice],valid=torch.isfinite(best),value=best,
        descriptor=descriptors[choice],scores=scores)


def observation_coordinates(features,price_pairs,mean,std,origin_price):
    """The graph's original joint metric, with frozen training normalization."""
    normalized=((features-mean)/std).reshape(*features.shape[:-2],118).double()/np.sqrt(118)
    log_prices=torch.log(price_pairs.double()/origin_price)/.0001
    return torch.cat((normalized,log_prices),-1)


def nearest_observation_nodes(features,prices,representative_features,representative_prices,mean,std,origin_price):
    query=observation_coordinates(features,prices,mean,std,origin_price)
    centers=observation_coordinates(representative_features,representative_prices,mean,std,origin_price)
    distance=torch.cdist(query,centers)
    error,nodes=distance.min(-1)
    return nodes.cpu().numpy(),error.cpu().numpy()
