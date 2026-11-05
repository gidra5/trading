"""Finite-row control variates for refreshed dense Bellman maps.

The old outcome vector is a fixed control variate, not a probability law. Its
exact row mean is known even if its components were refreshed on different
observations. Uniformly sampling differences gives a corrected current mean.
This local identity does not remove nonlinear/backward approximation error.
"""
import numpy as np


def paired_mean(old,new,indices):
    """Reference formula; nonfinite controls require a full current census."""
    old=np.asarray(old);new=np.asarray(new);indices=np.asarray(indices)
    if old.shape!=new.shape or old.ndim!=1:raise ValueError('Expected equal vectors')
    if not len(indices) or len(np.unique(indices))!=len(indices) or np.any((indices<0)|(indices>=len(old))):
        raise ValueError('Expected distinct valid sample indices')
    if len(indices)==len(old) or not np.isfinite(old).all():return float(new.mean())
    return float(old.mean()+(new[indices]-old[indices]).mean())


def project_market_nodes(current,previous,shift,feature_std=None):
    """Map each new parent to a previous parent at the same absolute second.

    This chooses the control variate only; it never changes current forecast
    probabilities. Price and observed feature pairs describe the parent.
    """
    if shift<0 or current.origin_seconds!=previous.origin_seconds+shift:
        raise ValueError('Reuse must align absolute timestamps')
    H=len(current.counts)-1;M=current.prices.shape[1]
    result=np.full((H,M),-1,dtype=np.int64)
    if feature_std is None:
        valid=np.arange(previous.prices.shape[1])[None,:]<previous.counts[:,None]
        feature_std=np.maximum(np.std(previous.features[valid],axis=(0,1)),1.e-6)
    scale=np.asarray(feature_std)
    if scale.shape!=(current.features.shape[-1],) or np.any(~np.isfinite(scale)|(scale<=0)):
        raise ValueError('Invalid causal feature scale')
    for h in range(H):
        old=h+shift
        if old>=len(previous.counts)-1:continue
        n=current.counts[h];m=previous.counts[old]
        df=(current.features[h,:n,None]-previous.features[old,None,:m])/scale
        dp=np.log(current.price_pairs[h,:n,None]/previous.price_pairs[old,None,:m])*10000
        distance=np.mean(df**2,axis=(2,3))+np.sum(dp**2,axis=2)
        result[h,:n]=np.argmin(distance,axis=1)
    return result
