"""Integer-checked supporting planes for Oracle continuation wealth.

Each row represents an upper c*quote+d*asset*price, with coefficients / SCALE.
Forecasts may be sampled/quantized, but every account and continuous control
is covered under the declared finite law. The neural value is not a premise.
"""
from dataclasses import dataclass
import numpy as np
import torch
from numba import njit, prange

SCALE=2**30


@njit(cache=True)
def plane_hulls(c,d):
    nodes,faces=c.shape
    ids=np.zeros((nodes,faces),np.int64);breaks=np.full((nodes,faces),np.inf)
    sizes=np.zeros(nodes,np.int64)
    for n in range(nodes):
        slope=d[n]-c[n]
        order=np.argsort(-slope)
        size=0
        for candidate in order:
            skip=False
            while size:
                last=ids[n,size-1]
                if slope[last]==slope[candidate]:
                    if c[n,last]<=c[n,candidate]:skip=True;break
                    size-=1;continue
                crossing=(float(c[n,candidate])-float(c[n,last]))/(float(slope[last])-float(slope[candidate]))
                if crossing<=breaks[n,size-1]:size-=1
                else:break
            if skip:continue
            ids[n,size]=candidate
            breaks[n,size]=-np.inf if size==0 else crossing
            size+=1
        sizes[n]=size
    return ids,breaks,sizes


@njit(cache=True,parallel=True)
def conditional_support(c,d,child_ticks,parent_ticks,destinations,slopes,ids,breaks,sizes):
    parents,branches=destinations.shape;points=len(slopes)
    cn=np.zeros((parents,points),np.int64);dn=np.zeros((parents,points),np.int64)
    for n in prange(parents):
        for branch in range(branches):
            child=destinations[n,branch];price=child_ticks[child];ratio=price/parent_ticks[n]
            index=0
            for j in range(points):
                x=slopes[j];moved=x*ratio/(1+x*(ratio-1))
                while index+1<sizes[child] and moved>=breaks[child,index+1]:index+=1
                face=ids[child,index]
                cn[n,j]+=c[child,face]
                dn[n,j]+=d[child,face]*price
        for j in range(points):cn[n,j]*=parent_ticks[n]
    return cn,dn


@dataclass(frozen=True)
class ValuePlanes:
    c: np.ndarray
    d: np.ndarray


def terminal_planes(nodes, fee_numerator=1, fee_denominator=5000, sell_fee_denominator=None):
    """Maker spread covers arrival fills as well as a taker terminal close."""
    a,b=fee_denominator-fee_numerator,fee_denominator
    sb=sell_fee_denominator or b;sa=sb-fee_numerator
    c=np.full((nodes,2),SCALE,dtype=np.int64)
    # Round INSIDE the no-gain wedge. Both planes individually upper-bound
    # terminal flat cash; exact liquidation value need not equal this upper.
    d=np.tile([(sa*SCALE+sb-1)//sb,b*SCALE//a],(nodes,1)).astype(np.int64)
    return ValuePlanes(c,d)


def close_planes(cnum,dnum,denominator,leverage=3,fee_numerator=1,fee_denominator=5000,sell_fee_denominator=None):
    """Dominate each input plane on the post-rebalance leverage cone, then
    put it inside the fee wedge so ALL actual rebalances cannot increase it.
    Float proposals affect tightness only; returned inequalities are integer
    checked at both endpoints, and hence throughout the cone.
    """
    L=leverage;a=fee_denominator-fee_numerator;b=fee_denominator
    if not isinstance(L,(int,np.integer)) or L<1:raise ValueError('Certificate leverage must be a positive integer')
    sb=sell_fee_denominator or b;sa=sb-fee_numerator
    C=cnum/denominator;D=dnum/denominator;retained=a/b;fee=1-retained
    sell_retained=sa/sb;sell_fee=1-sell_retained
    buy=retained*D>C;sell=D<sell_retained*C
    buy_scale=(C+L*(D-C))/(retained+fee*L)
    sell_scale=(C-L*(D-C))/(1+sell_fee*L)
    proposal_c=np.where(buy,retained*buy_scale,np.where(sell,sell_scale,C))
    proposal_d=np.where(buy,buy_scale,np.where(sell,sell_retained*sell_scale,D))
    if not np.isfinite(proposal_c).all() or not (proposal_c>0).all():
        raise ValueError('Nonpositive/nonfinite supporting plane')
    # All products below must be exact int64, including the signed endpoints.
    max_int=np.iinfo(np.int64).max
    magnitude=max(float(np.max(np.abs(cnum))),float(np.max(np.abs(dnum))))
    if magnitude*(1+2*L)>max_int/2:
        raise OverflowError('Certificate arithmetic needs a larger integer type')
    for padding in (4,16,64,256):
        c=np.ceil(proposal_c).astype(np.int64)+padding
        d=np.rint(proposal_d).astype(np.int64)+padding
        d=np.maximum((sa*c+sb-1)//sb,np.minimum(d,b*c//a))
        if float(np.max(np.maximum(c,d)))*float(np.max(denominator))*(1+2*L)>max_int/2:
            raise OverflowError('Certificate endpoint product overflow')
        lower=((1+L)*c-L*d)*denominator >= (1+L)*cnum-L*dnum
        upper=((1-L)*c+L*d)*denominator >= (1-L)*cnum+L*dnum
        valid=lower & upper & (a*d<=b*c) & (sa*c<=sb*d) & (c>0) & (d>0)
        if valid.all():return ValuePlanes(c,d)
    raise ArithmeticError('Unable to certify proposed continuation planes')


def wedge_mixtures(cn,dn,den,L,fee_numerator,fee_denominator,sell_fee_denominator=None):
    """Convex combinations of neighboring support planes can meet the fee
    wedge without paying the loose leverage-endpoint relaxation. Integer
    mixture weights and Python integer endpoint checks cover rounding.
    """
    a=fee_denominator-fee_numerator;b=fee_denominator;W=65536
    sb=sell_fee_denominator or b;sa=sb-fee_numerator
    out_c=np.empty((len(cn),2),np.int64);out_d=np.empty_like(out_c)
    for n in range(len(cn)):
        C=cn[n].astype(np.float64);D=dn[n].astype(np.float64)
        for j,derivative in enumerate((a*D-b*C,sb*D-sa*C)):
            crossing=np.flatnonzero((derivative[:-1]>0)&(derivative[1:]<=0))
            if len(crossing):
                i=int(crossing[0]);fraction=-derivative[i+1]/(derivative[i]-derivative[i+1])
                w=max(0,min(W,round(fraction*W)))
                cnum=w*int(cn[n,i])+(W-w)*int(cn[n,i+1])
                dnum=w*int(dn[n,i])+(W-w)*int(dn[n,i+1])
                denominator=int(den[n,0])*W
            else:
                i=int(np.argmin(np.abs(derivative)))
                cnum=int(cn[n,i]);dnum=int(dn[n,i]);denominator=int(den[n,0])
            c=(cnum+denominator-1)//denominator+L+1
            d=(dnum+denominator-1)//denominator+L+1
            if ((1+L)*c-L*d)*denominator<(1+L)*cnum-L*dnum or ((1-L)*c+L*d)*denominator<(1-L)*cnum+L*dnum:
                raise ArithmeticError('Convex support rounding check failed')
            out_c[n,j]=c;out_d[n,j]=d
    return close_planes(out_c,out_d,np.ones_like(out_c),L,fee_numerator,fee_denominator,sell_fee_denominator)


@torch.inference_mode()
def supporting_backup(child,child_ticks,parent_ticks,destinations,points=33,
                      leverage=3,fee_numerator=1,fee_denominator=5000,sell_fee_denominator=None,device='cuda',chunk=8):
    """No account grid is imposed. Points choose valid supporting planes;
    increasing their count improves tightness, not theorem coverage.
    """
    if points<2 or leverage<1:raise ValueError('Invalid supporting-point menu')
    B=destinations.shape[1];L=leverage
    if B<=0 or np.any(parent_ticks<=0) or np.any(child_ticks<=0):raise ValueError('Invalid finite price law')
    if np.any(destinations<0) or np.any(destinations>=len(child_ticks)):raise ValueError('Invalid destination')
    prices=child_ticks[destinations]
    # Every child plane has its shadow ratio inside the maker spread.
    # A fixed leverage guard suffices for all parents and all child planes.
    a=1999;b=2000
    if np.any(L*np.abs(a*prices-b*parent_ticks[:,None])>b*parent_ticks[:,None]) or np.any(
        L*np.abs(b*prices-a*parent_ticks[:,None])>a*parent_ticks[:,None]):
        raise ValueError('Liquidation shadow guard fails; no certificate returned')
    if max(int(child.c.max()),int(child.d.max()))*int(prices.max())*B*(1+2*L)>np.iinfo(np.int64).max//2:
        raise OverflowError('Conditional coefficient products would overflow')
    if np.any(child.c<=0) or np.any(child.d<=0) or np.any(1999*child.d>2000*child.c) or np.any(1999*child.c>2000*child.d):
        raise ValueError('Child shadow ratios exceed the verified liquidation guard')
    slopes=np.linspace(-L,L,points)
    # A convex hull makes queries linear in points+faces, rather than their
    # product. Floating intersections select faces only; coefficient sums and
    # all value-domination checks below remain exact integers.
    ids,breaks,sizes=plane_hulls(child.c,child.d)
    cn,dn=conditional_support(child.c,child.d,child_ticks,parent_ticks,destinations,slopes,ids,breaks,sizes)
    den=np.broadcast_to(B*parent_ticks[:,None],cn.shape)
    direct=close_planes(cn,dn,den,L,fee_numerator,fee_denominator,sell_fee_denominator)
    mixture=wedge_mixtures(cn,dn,den,L,fee_numerator,fee_denominator,sell_fee_denominator)
    return ValuePlanes(np.concatenate((direct.c,mixture.c),1),np.concatenate((direct.d,mixture.d),1))


def upper_cash_ratio(planes):
    return int(planes.c[0].min()),SCALE


def hull_wedge_mixtures(c,d):
    """Additional global supports mixing adjacent faces of the lower hull.

    The hull only proposes which pair to mix. The mixture weights are exact
    nonnegative integers and the rounded closure is verified independently.
    """
    ids,_,sizes=plane_hulls(c,d)
    # Pad with the last hull face so wedge_mixtures sees an ordered row.
    ordered_c=np.empty_like(c);ordered_d=np.empty_like(d)
    for n in range(len(c)):
        ix=ids[n,np.minimum(np.arange(c.shape[1]),sizes[n]-1)]
        ordered_c[n]=c[n,ix];ordered_d[n]=d[n,ix]
    return wedge_mixtures(ordered_c,ordered_d,np.ones((len(c),1),np.int64),3,1,2000)


def common_price_certificate(price_ticks,fee_numerator=1,fee_denominator=5000):
    """Independent simple exact check for all paths staying in one fee band."""
    a=fee_denominator-fee_numerator;b=fee_denominator
    low=int(np.min(price_ticks));high=int(np.max(price_ticks))
    return a*high<=b*low
