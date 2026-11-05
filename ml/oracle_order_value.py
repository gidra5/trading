"""Post-arrival continuation bounds with predictable limit prices.

Unlike the maker-spread relaxation, a new order chooses its one fixed limit
before the random move. The reservation budget is shared with inventory.
"""
import numpy as np
from numba import njit,prange
from oracle_certified_value import ValuePlanes,plane_hulls,close_planes,SCALE,hull_wedge_mixtures


@njit(cache=True,parallel=True)
def order_support(c,d,pc,pp,ds,points,ids,breaks,sizes,constant=None,capacity_slope=None):
    rows,B=ds.shape;G=len(points);cn=np.empty((rows,3*G),np.int64);dn=np.empty_like(cn);option=np.empty_like(cn);bn=np.zeros_like(cn);kb=np.empty_like(cn);ks=np.empty_like(cn)
    for n in prange(rows):
        order=np.argsort(pc[ds[n]])
        for pattern in range(3):
            for g in range(G):
                x=points[g];N=max(0.,2-abs(x)) if pattern else 0.
                chosen_c=np.empty(B,np.int64);chosen_d=np.empty(B,np.int64);chosen_t=np.zeros(B,np.int64)
                total_c=0;total_dp=0;total_beta=0
                for edge in range(B):
                    child=ds[n,edge];price=pc[child];ratio=price/pp[n]
                    q=1-x;v=x*ratio
                    if pattern==1 and price<pp[n]:q-=N;v+=.9998*N*price/(pp[n]-1)
                    if pattern==2 and price>pp[n]:q+=.9998*N;v-=N*price/(pp[n]+1)
                    e=v/(q+v) if q+v>0 else 0.
                    lo=0;hi=sizes[child]
                    while lo+1<hi:
                        middle=(lo+hi)//2
                        if e>=breaks[child,middle]:lo=middle
                        else:hi=middle
                    face=ids[child,lo];C=c[child,face];D=d[child,face]
                    chosen_c[edge]=C;chosen_d[edge]=D;total_c+=C;total_dp+=D*price
                    if constant is not None:total_beta+=constant[child,face]
                    if capacity_slope is not None:chosen_t[edge]=capacity_slope[child,face]
                j=pattern*G+g;cn[n,j]=total_c*pp[n];dn[n,j]=total_dp;bn[n,j]=total_beta
                # Every optimal threshold on a fixed fill set is one of the
                # outcome grid prices. All equal-price outcomes fill together.
                prefix_c=0;prefix_dp=0;best=0
                for k in range(B):
                    edge=order[k];price=pc[ds[n,edge]]
                    prefix_c+=chosen_c[edge]+chosen_t[edge]
                    maker_d=(4999*chosen_d[edge]+4999)//5000
                    prefix_dp+=maker_d*price
                    if price<pp[n] and (k+1==B or pc[ds[n,order[k+1]]]!=price):
                        num=prefix_dp-price*prefix_c;den=B*price
                        if num>0:best=max(best,(num+den-1)//den)
                suffix_c=0;suffix_dp=0
                kb[n,j]=best;best_sell=0
                for k in range(B-1,-1,-1):
                    edge=order[k];price=pc[ds[n,edge]]
                    maker_c=(4999*(chosen_c[edge]-chosen_t[edge])+4999)//5000
                    suffix_c+=maker_c;suffix_dp+=chosen_d[edge]*price
                    if price>pp[n] and (k==0 or pc[ds[n,order[k-1]]]!=price):
                        num=price*suffix_c-suffix_dp;den=B*price
                        if num>0:best_sell=max(best_sell,(num+den-1)//den)
                option[n,j]=max(best,best_sell);ks[n,j]=best_sell
    return cn,dn,option,bn,kb,ks


def order_backup(child,child_ticks,parent_ticks,destinations,points=33,leverage=3):
    B=destinations.shape[1];L=leverage;prices=child_ticks[destinations]
    if np.any(parent_ticks<=1) or np.any(child_ticks<=0):raise ValueError('Positive grid prices with a passive buy tick required')
    # With a resting order, the shared reserve gives raw leverage <=2.
    # A one-percent move stays strictly inside effective maintenance [-3,3]
    # at the declared taker fee, so liquidation cannot erase a promised fill.
    if np.any(100*np.abs(prices-parent_ticks[:,None])>parent_ticks[:,None]):
        raise ValueError('Order survival guard fails; no certificate returned')
    if max(int(child.c.max()),int(child.d.max()))*int(prices.max())*B*(1+2*L)>np.iinfo(np.int64).max//2:
        raise OverflowError('Order certificate coefficient overflow')
    cn,dn,K,_,_,_=order_support(child.c,child.d,child_ticks,parent_ticks,destinations,np.linspace(-L,L,points),*plane_hulls(child.c,child.d))
    denominator=B*parent_ticks[:,None]
    C=(cn+denominator-1)//denominator+L+1;D=(dn+denominator-1)//denominator+L+1
    if not np.all(((1+L)*C-L*D)*denominator >= (1+L)*cn-L*dn) or not np.all(((1-L)*C+L*D)*denominator >= (1-L)*cn+L*dn):
        raise ArithmeticError('Order support rounding check failed')
    # N <= max(2W-|asset*price|,0) on the feasible order domain. On |e|<=L,
    # 2W-(2/L)|asset*price| is a concave upper envelope, including N=0
    # outside the order-reservation cone. L=2 at the cash root.
    slope=(2*K)//L
    C=C+2*K;D=D+2*K
    raw_c=np.concatenate((C,C),1);raw_d=np.concatenate((D-slope,D+slope),1)
    return close_planes(raw_c,raw_d,np.ones_like(raw_c),L,fee_denominator=2000)


def rational_order_backup(child,child_ticks,parent_ticks,destinations,points,gross_numerator,gross_denominator):
    """Same support argument with a tighter graph-wide reachable leverage.
    Raw rational supports are first rounded on [-3,3]; subsequent checks use
    integer numerators of the tighter leverage endpoints.
    """
    n=int(gross_numerator);k=int(gross_denominator);L=n/k;B=destinations.shape[1]
    if not (2<=L<=3):raise ValueError('Expected a reachable leverage bound in [2,3]')
    prices=child_ticks[destinations]
    if np.any(100*np.abs(prices-parent_ticks[:,None])>parent_ticks[:,None]):raise ValueError('Order survival guard')
    if max(int(child.c.max()),int(child.d.max()))*int(prices.max())*B*7>np.iinfo(np.int64).max//2:raise OverflowError('Coefficient overflow')
    cn,dn,K,_,_,_=order_support(child.c,child.d,child_ticks,parent_ticks,destinations,np.linspace(-L,L,points),*plane_hulls(child.c,child.d))
    den=B*parent_ticks[:,None];C=(cn+den-1)//den+4;D=(dn+den-1)//den+4
    if not np.all((4*C-3*D)*den>=4*cn-3*dn) or not np.all((-2*C+3*D)*den>=-2*cn+3*dn):raise ArithmeticError('Raw support rounding')
    slope=(2*K*k)//n;C=C+2*K;D=D+2*K
    C=np.concatenate((C,C),1);D=np.concatenate((D-slope,D+slope),1)
    b=.9995;f=.0005;buy=b*D>C;sell=D<b*C
    up=(C+L*(D-C))/(b+f*L);down=(C-L*(D-C))/(1+f*L)
    pc=np.where(buy,b*up,np.where(sell,down,C));pd=np.where(buy,up,np.where(sell,b*down,D))
    c=np.ceil(pc).astype(np.int64)+8;d=np.rint(pd).astype(np.int64)+8
    d=np.maximum((1999*c+1999)//2000,np.minimum(d,2000*c//1999))
    if not np.all((k+n)*c-n*d>=(k+n)*C-n*D) or not np.all((k-n)*c+n*d>=(k-n)*C+n*D):raise ArithmeticError('Rational leverage closure')
    if not np.all(1999*d<=2000*c) or not np.all(1999*c<=2000*d):raise ArithmeticError('Taker cone')
    mixed=hull_wedge_mixtures(C,D)
    return ValuePlanes(np.concatenate((c,mixed.c),1),np.concatenate((d,mixed.d),1))
