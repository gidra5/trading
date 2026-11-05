"""Account pending-order execution and approximate dense-cell Bellman backups.

All functions operate at a declared phase. Post-decision orders are eligible
at the next tick; age is unnecessary after that phase under Oracle's one-tick
activation rule. Arrivals happen before new actions. Prices are tick-rounded.
No unfilled order is dropped from the lookup state. Value interpolation and
finite candidate/outcome samples are approximations, not optimality proofs.
"""
import numpy as np
from numba import njit,prange


def rule_vector(rules, funding=1.):
    from gpu_oracle_bellman import TensorOracle
    TensorOracle(rules)  # shared public contract validation
    if not np.isfinite(funding) or not 0<funding<=1:
        raise ValueError('Funding multiplier must be in (0,1]')
    return np.array([rules.maker,rules.taker,rules.reserve_leverage,rules.lower,rules.upper,
        rules.maintenance_lower,rules.maintenance_upper,rules.minimum,rules.maximum,rules.tick,funding])

# q,asset,p,previous; followed by max_orders (side,limit,notional) triples.


@njit(cache=True)
def allowed_size(x,r):
    return x==0 or (abs(x)>=r[7] and abs(x)<=r[8]+1e-8)


@njit(cache=True)
def effective(s,r):
    v=s[1]*s[2]
    return (1-r[1])*v if v>=0 else v/(1-r[1])


@njit(cache=True)
def move_arrive(s,price,r):
    x=s.copy();x[3]=s[2];x[2]=price;x[1]*=r[10]
    raw=x[0]+x[1]*price;eff=effective(x,r);w=x[0]+eff
    alive=raw>0 and w>0 and r[5]-1e-10<=eff/w<=r[6]+1e-10
    if not alive:
        x[0]=0.;x[1]=0.;x[4:]=0.
        return x,0.,False,False
    consumed=0.;filled=False;valid=True;K=(len(x)-4)//3
    for j in range(K):
        i=4+3*j;side,L,N=x[i:i+3]
        pt=round(price/r[9]);prev=round(s[2]/r[9]);lt=round(L/r[9])
        hit=(side>0 and pt<prev and pt<=lt<=prev) or (side<0 and prev<pt and prev<=lt<=pt)
        if hit:
            dq=-N if side>0 else (1-r[0])*N
            x[0]+=dq
            x[1]+=((1-r[0])*N/L if side>0 else -N/L)
            consumed+=abs(dq);valid=valid and allowed_size(dq,r);filled=True
            x[i:i+3]=0.
    valid=valid and consumed<=r[8]+1e-8
    return x,consumed,filled,valid


@njit(cache=True)
def close_terminal(arrived,consumed,r):
    flow=effective(arrived,r)
    if (not allowed_size(flow,r) or consumed+abs(flow)>r[8]+1e-8
            or (arrived[1]!=0 and flow==0)):
        return -np.inf
    w=arrived[0]+flow
    return np.log(w) if w>0 else -np.inf


@njit(cache=True)
def valid_post(s,r,entry,filled):
    w=s[0]+s[1]*s[2]
    if w<=0:return False
    e=s[1]*s[2]/w
    if (entry or filled) and not r[3]-1e-9<=e<=r[4]+1e-9:return False
    reserve=0.;orders=0
    for i in range(4,len(s),3):
        if s[i]!=0:
            orders+=1;reserve+=s[i+2]/r[2]
            dq=-s[i+2] if s[i]>0 else (1-r[0])*s[i+2]
            if not allowed_size(dq,r) or s[i+1]<=0:return False
    if orders and (reserve+abs(s[1]*s[2])/r[2]>w+1e-8):return False
    return True


@njit(cache=True)
def execute_target(arrived,consumed,filled,e_target,book,book_count,r,price_scale,keep_mask=-1):
    out=arrived.copy();w=out[0]+out[1]*out[2]
    if w<=0:return out,False
    e=out[1]*out[2]/w;amount=0.;request=0.
    if np.isfinite(e_target):
        if not r[3]<=e_target<=r[4]:return out,False
        den=(1-r[1]+r[1]*e_target) if e<e_target else (1-r[1]*e_target)
        if den<=0:return out,False
        request=(-w*(e_target-e)/den) if e<e_target else ((1-r[1])*w*(e-e_target)/den)
        amount=np.sign(request)*min(abs(request),max(0.,r[8]-consumed))
        if not allowed_size(amount,r):return out,False
        out[0]+=amount
        out[1]-=(1-r[1])*amount/out[2] if amount<=0 else amount/((1-r[1])*out[2])
        if e_target==0 and abs(request)<=r[8]-consumed and allowed_size(request,r):out[1]=0.
    if keep_mask<0:
        out[4:]=0.;new_w=out[0]+out[1]*out[2]
        for j in range(book_count):
            z,frac=book[j];side=1. if frac>0 else -1.
            distance=z/(1-z)*price_scale
            ticks=max(1.,np.rint(out[2]*distance/r[9]))
            L=(round(out[2]/r[9])-side*ticks)*r[9]
            out[4+3*j]=side;out[5+3*j]=L;out[6+3*j]=abs(frac)*new_w
    else:
        for j in range((len(out)-4)//3):
            if (keep_mask & (1<<j))==0:out[4+3*j:7+3*j]=0.
    return out,valid_post(out,r,amount!=0,filled)


@njit(cache=True)
def representative(w,e,book,count,p,r,price_scale):
    s=np.zeros(4+3*len(book));s[0]=w*(1-e);s[1]=w*e/p;s[2]=p;s[3]=p
    for j in range(count):
        z,n=book[j];side=1. if n>0 else -1.
        ticks=max(1.,np.rint(p*z/(1-z)*price_scale/r[9]))
        s[4+3*j]=side;s[5+3*j]=(round(p/r[9])-side*ticks)*r[9];s[6+3*j]=abs(n)*w
    return s,valid_post(s,r,False,False)


@njit(cache=True)
def bracket(axis,x):
    if x<axis[0]-1e-9 or x>axis[-1]+1e-9:return -1,0.
    i=min(max(np.searchsorted(axis,x,side='right')-1,0),len(axis)-2)
    return i,min(max((x-axis[i])/(axis[i+1]-axis[i]),0.),1.)


@njit(cache=True)
def lookup(s,node,values,wealth,exposures,offsets,book_lookup,size_edges,price_bins,price_scale):
    w=s[0]+s[1]*s[2]
    if w<=0:return -np.inf
    e=s[1]*s[2]/w;wi,wa=bracket(wealth,w);ei,ea=bracket(exposures,e)
    if wi<0 or ei<0:return -np.inf
    ids=np.empty((len(s)-4)//3,dtype=np.int64);K=0;S=len(size_edges)-1;T=S*price_bins
    for i in range(4,len(s),3):
        if s[i]!=0:
            n=s[i]*s[i+2]/w
            if n<size_edges[0] or n>size_edges[-1]:return -np.inf
            x=abs(s[i+1]/s[2]-1)/price_scale;z=x/(1+x)
            pi=min(int(z*price_bins),price_bins-1)
            si=min(max(np.searchsorted(size_edges,n,side='right')-1,0),S-1)
            ids[K]=pi*S+si;K+=1
    sorted_ids=np.sort(ids[:K]);code=offsets[K]
    for i in range(K):code+=sorted_ids[i]*T**i
    book=book_lookup[code]
    if book<0:return -np.inf
    norm=0.
    for dw in range(2):
        for de in range(2):
            weight=(wa if dw else 1-wa)*(ea if de else 1-ea)
            if weight>1e-12:
                v=values[wi+dw,book*len(exposures)+ei+de,node]
                if not np.isfinite(v):return -np.inf
                norm+=weight*v
    return np.log(w)+norm


@njit(cache=True)
def post_value(s,node,values,wealth,exposures,offsets,book_lookup,size_edges,price_bins,price_scale,r,next_factors):
    """Grid estimate plus an explicitly executable next-arrival closing policy.

    This avoids interpreting a missing interpolation neighbor as proof that an
    actual residual cannot be closed. Pending orders still fill BEFORE closing.
    The fallback is checked against every edge of the finite conditional row.
    """
    estimated=lookup(s,node,values,wealth,exposures,offsets,book_lookup,size_edges,price_bins,price_scale)
    mean=0.;closable=True
    for j in range(next_factors.shape[1]):
        arrived,consumed,_,okay=move_arrive(s,max(1.,np.rint(s[2]*next_factors[node,j]/r[9]))*r[9],r)
        if not okay:return -np.inf
        v=close_terminal(arrived,consumed,r)
        if not np.isfinite(v):closable=False
        else:mean+=(v-mean)/(j+1)
    return max(estimated,mean) if closable else estimated


@njit(cache=True)
def decide(arrived,consumed,filled,node,values,wealth,exposures,books,counts,offsets,book_lookup,
           size_edges,price_bins,price_scale,r,candidates,next_factors):
    best=-np.inf;chosen=arrived.copy();choice=-1
    # Explicit cancellation and valid full closing are an attainable cash policy.
    closed=close_terminal(arrived,consumed,r)
    if np.isfinite(closed):
        flat=arrived.copy();flat[0]=arrived[0]+effective(arrived,r);flat[1]=0.;flat[4:]=0.
        cv=lookup(flat,node,values,wealth,exposures,offsets,book_lookup,size_edges,price_bins,price_scale)
        # Cash forever remains attainable outside the represented wealth range.
        best=max(closed,cv);chosen=flat;choice=-2
    for ci in candidates:
        book,ei=ci//len(exposures),ci%len(exposures)
        target,okay=execute_target(arrived,consumed,filled,exposures[ei],books[book],counts[book],r,price_scale)
        if okay:
            val=post_value(target,node,values,wealth,exposures,offsets,book_lookup,size_edges,price_bins,price_scale,r,next_factors)
            if val>best:best=val;chosen=target;choice=ci
    # Retain every subset of the existing book, with no trade, a close request,
    # or each entry target. These books keep their actual prices and notionals.
    K=(len(arrived)-4)//3
    occupied=0
    for j in range(K):
        if arrived[4+3*j]!=0:occupied|=1<<j
    for mask in range(1<<K):
        if mask & ~occupied:continue
        for ei in range(-1,len(exposures)):
            e=np.nan if ei<0 else exposures[ei]
            target,okay=execute_target(arrived,consumed,filled,e,books[0],0,r,price_scale,mask)
            if okay:
                val=post_value(target,node,values,wealth,exposures,offsets,book_lookup,size_edges,price_bins,price_scale,r,next_factors)
                if val>best:best=val;chosen=target;choice=-3
    return best,chosen,choice


@njit(cache=True,parallel=True)
def backup_layer(parent_prices,sampled_prices,destinations,permutations,allocation,child_values,
                 wealth,exposures,books,counts,offsets,book_lookup,size_edges,price_bins,price_scale,r,candidates,terminal,next_factors):
    W=len(wealth);C=len(books)*len(exposures);M=len(parent_prices)
    result=np.full((W,C,M),-np.inf);variance=np.full((W,C,M),np.nan)
    for task in prange(W*C*M):
        node=task%M;cell=(task//M)%C;wi=task//(M*C);book,ei=cell//len(exposures),cell%len(exposures)
        state,valid=representative(wealth[wi],exposures[ei],books[book],counts[book],parent_prices[node],r,price_scale)
        if not valid:continue
        mean=0.;m2=0.;okay=True;N=allocation[cell]
        for draw in range(N):
            edge=permutations[node,draw];child=destinations[node,edge]
            arrived,consumed,filled,valid=move_arrive(state,sampled_prices[node,edge],r)
            if not valid:okay=False;break
            if terminal:v=close_terminal(arrived,consumed,r)
            else:v,_,_=decide(arrived,consumed,filled,child,child_values,wealth,exposures,books,counts,
                offsets,book_lookup,size_edges,price_bins,price_scale,r,candidates,next_factors)
            if not np.isfinite(v):okay=False;break
            v-=np.log(wealth[wi]);delta=v-mean;mean+=delta/(draw+1);m2+=delta*(v-mean)
        if okay:result[wi,cell,node]=mean;variance[wi,cell,node]=m2/max(N-1,1)
    return result,variance
