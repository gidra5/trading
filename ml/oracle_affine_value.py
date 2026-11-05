"""Affine supporting bounds retain the absolute per-decision quote cap.

Values are (c*q+d*a*p)/SCALE + beta*10000/SCALE. Every plane is an
independently valid upper; the account query grid only selects supports.
"""
from dataclasses import dataclass
import numpy as np
from oracle_certified_value import plane_hulls,close_planes,terminal_planes
from oracle_order_value import order_support


@dataclass(frozen=True)
class AffinePlanes:
    c:np.ndarray
    d:np.ndarray
    beta:np.ndarray
    theta:np.ndarray


def terminal_affine(nodes):
    v=terminal_planes(nodes,fee_denominator=2000)
    return AffinePlanes(v.c,v.d,np.zeros_like(v.c),np.zeros_like(v.c))


def affine_backup(child,child_ticks,parent_ticks,destinations,points=17,leverage=3):
    B=destinations.shape[1];L=leverage;prices=child_ticks[destinations]
    if np.any(100*np.abs(prices-parent_ticks[:,None])>parent_ticks[:,None]):raise ValueError('Price guard')
    # With c,d positive, these inequalities imply a moved shadow ratio in
    # [.99^2,1.01^2], hence nonnegative shadow wealth for raw leverage <=3.
    if np.any(child.c<=0) or np.any(child.d<=0) or np.any(child.theta<0) or np.any(child.beta<child.theta) or np.any(100*np.abs(child.d-child.c)>child.c):raise ValueError('Affine liquidation guard')
    if max(int(child.c.max()),int(child.d.max()))*int(prices.max())*B*7>np.iinfo(np.int64).max//2:raise OverflowError('Coefficient overflow')
    # The hull query at unit wealth is a support-selection heuristic only.
    # The selected affine plane remains valid at every wealth and account.
    cn,dn,K,bn,_,_=order_support(child.c,child.d,child_ticks,parent_ticks,destinations,
        np.linspace(-L,L,points),*plane_hulls(child.c+child.beta,child.d+child.beta),child.beta,child.theta)
    den=B*parent_ticks[:,None];C=(cn+den-1)//den+4;D=(dn+den-1)//den+4
    if not np.all((4*C-3*D)*den>=4*cn-3*dn) or not np.all((-2*C+3*D)*den>=-2*cn+3*dn):raise ArithmeticError('Affine support rounding')
    beta=(bn+B-1)//B
    # Gross sell notional is capped by M/(1-maker), buys by M. M/W0=1.
    capped_order_beta=beta+(5000*K+4998)//4999
    closed=close_planes(C,D,np.ones_like(C),L,fee_denominator=2000)
    # A second valid bound keeps raw coefficients and pays the actual maximum
    # possible market gain per quote dollar instead of relaxing order size.
    buy_gain=np.maximum(0,(1999*D-2000*C+1999)//2000)
    sell_gain=np.maximum(0,(2000*C-1999*D+1998)//1999)
    market_gain=np.maximum(buy_gain,sell_gain)
    slope=(2*K)//L
    hc=np.concatenate((C+2*K,C+2*K),1);hd=np.concatenate((D+2*K-slope,D+2*K+slope),1)
    reserved=close_planes(hc,hd,np.ones_like(hc),L,fee_denominator=2000)
    return AffinePlanes(np.concatenate((closed.c,C,reserved.c),1),np.concatenate((closed.d,D,reserved.d),1),
        np.concatenate((capped_order_beta,capped_order_beta+market_gain,beta,beta),1),
        np.concatenate((np.zeros_like(C),market_gain,np.zeros_like(hc)),1))
