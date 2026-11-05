"""Diagnostic bounds for the finite-log-utility class under NSF full support.

This is a DIFFERENT comparison class from unconstrained policies under a
bounded finite graph. It must not silently certify that bounded-price model.
Unbounded upward tails prohibit held shorts; the minimum positive tick and
maintenance imply only a microscopic amount of long borrowing at BTC prices.
"""
import numpy as np
from oracle_certified_value import ValuePlanes,plane_hulls
from oracle_order_value import order_support


def full_support_backup(child,child_ticks,parent_ticks,destinations,points=33):
    B=destinations.shape[1];eps_den=1000000;n=eps_den+1;k=eps_den;L=n/k
    prices=child_ticks[destinations]
    if min(int(child_ticks.min()),int(parent_ticks.min()))<4*eps_den:
        raise ValueError('The microscopic borrowing envelope requires prices >= 40000 on the .01 grid')
    if np.any(100*np.abs(prices-parent_ticks[:,None])>parent_ticks[:,None]):raise ValueError('Price guard')
    if max(int(child.c.max()),int(child.d.max()))*int(prices.max())*B*7>np.iinfo(np.int64).max//2:raise OverflowError('Coefficient overflow')
    cn,dn,_,_,kb,ks=order_support(child.c,child.d,child_ticks,parent_ticks,destinations,np.linspace(0,L,points),*plane_hulls(child.c,child.d))
    den=B*parent_ticks[:,None];C=(cn+den-1)//den+4;D=(dn+den-1)//den+4
    if not np.all((4*C-3*D)*den>=4*cn-3*dn) or not np.all((-2*C+3*D)*den>=-2*cn+3*dn):raise ArithmeticError('Support rounding')
    # Every fillable buy has N <= q + 3*eps*W from solvency on a tick-floor
    # outcome. Every sell must be covered by already-held asset quantity.
    # max gross sell N <= asset * largest fill price; positive coupon only.
    margin=(3*kb+eps_den-1)//eps_den
    sell=(ks*prices.max(1)[:,None]+parent_ticks[:,None]-1)//parent_ticks[:,None]
    C=C+kb+margin;D=D+sell+margin
    b=.9995;f=.0005
    buy=b*D>C;sale=D<b*C
    top=(C+L*(D-C))/(b+f*L)
    pc=np.where(buy,b*top,C);pd=np.where(buy,top,np.where(sale,b*C,D))
    c=np.ceil(pc).astype(np.int64)+8;d=np.rint(pd).astype(np.int64)+8
    d=np.maximum((1999*c+1999)//2000,np.minimum(d,2000*c//1999))
    if not np.all(c>=C) or not np.all((k-n)*c+n*d>=(k-n)*C+n*D):raise ArithmeticError('Full-support leverage closure')
    return ValuePlanes(c,d)
