"""GPU implementation of the dense account Bellman sweep.

One Triton program evaluates the action menu for one account/market/outcome
point. A second kernel averages outcomes, AFTER maximizing the future action.
The numerical grid and finite conditional law are unchanged. Values remain on
the device through the full backward sweep; only completed fields are exported.
"""
import numpy as np
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice as lib
from numba import njit
from oracle_book_mc import representative


@njit(cache=True)
def active_account_indices(wealth,exposures,books,counts,prices,rules,price_scale):
    """Schedule only representatives feasible at some price in this cloud.

    This uses the reference arithmetic, including tick rounding. It does not
    remove cells from the dense field or prune feasible candidate actions.
    """
    C=len(books)*len(exposures);active=[]
    for wi in range(len(wealth)):
        for cell in range(C):
            book=cell//len(exposures);ei=cell%len(exposures)
            for p in prices:
                _,valid=representative(wealth[wi],exposures[ei],books[book],counts[book],p,rules,price_scale)
                if valid:
                    active.append(wi*C+cell)
                    break
    return np.asarray(active,dtype=np.int64)


@triton.jit
def _finite(x):
    return (x==x)&(tl.abs(x)!=float('inf'))


@triton.jit
def _size(x,r):
    return (x==0)|((tl.abs(x)>=r[7])&(tl.abs(x)<=r[8]+1.e-8))


@triton.jit
def _effective(s,r):
    v=s[1]*s[2]
    return tl.where(v>=0,(1-r[1])*v,v/(1-r[1]))


@triton.jit
def _valid(s,r,entry,filled,K:tl.constexpr):
    w=s[0]+s[1]*s[2];e=s[1]*s[2]/w
    okay=(w>0)&(((entry|filled)==0)|((e>=r[3]-1.e-9)&(e<=r[4]+1.e-9)))
    reserve=tl.full((),0.,tl.float64);orders=tl.full((),0,tl.int32)
    for j in tl.static_range(K):
        side=s[3+3*j];L=s[4+3*j];n=s[5+3*j];active=side!=0
        reserve+=tl.where(active,n/r[2],0.);orders+=active.to(tl.int32)
        dq=tl.where(side>0,-n,(1-r[0])*n)
        okay&=(~active)|(_size(dq,r)&(L>0))
    return okay&((orders==0)|(reserve+tl.abs(s[1]*s[2])/r[2]<=w+1.e-8))


@triton.jit
def _arrive(s,p,r,K:tl.constexpr):
    q=s[0];a=s[1]*r[10];eff=tl.where(a*p>=0,(1-r[1])*a*p,a*p/(1-r[1]));w=q+eff
    okay=(q+a*p>0)&(w>0)&(eff/w>=r[5]-1.e-10)&(eff/w<=r[6]+1.e-10)
    consumed=tl.full((),0.,tl.float64);filled=tl.full((),False,tl.int1);orders=()
    pt=lib.nearbyint(p/r[9]);prev=lib.nearbyint(s[2]/r[9])
    for j in tl.static_range(K):
        side=s[3+3*j];L=s[4+3*j];n=s[5+3*j];lt=lib.nearbyint(L/r[9])
        hit=((side>0)&(pt<prev)&(pt<=lt)&(lt<=prev))|((side<0)&(prev<pt)&(prev<=lt)&(lt<=pt))
        dq=tl.where(side>0,-n,(1-r[0])*n)
        q+=tl.where(hit,dq,0.);a+=tl.where(hit,tl.where(side>0,(1-r[0])*n/tl.where(L>0,L,1.),-n/tl.where(L>0,L,1.)),0.)
        consumed+=tl.where(hit,tl.abs(dq),0.);filled|=hit;okay&=(~hit)|_size(dq,r)
        orders+=(tl.where(hit,0.,side),tl.where(hit,0.,L),tl.where(hit,0.,n))
    return (q,a,p)+orders,consumed,filled,okay&(consumed<=r[8]+1.e-8)


@triton.jit
def _close(s,consumed,r):
    flow=_effective(s,r);w=s[0]+flow
    okay=_size(flow,r)&(consumed+tl.abs(flow)<=r[8]+1.e-8)&((s[1]==0)|(flow!=0))&(w>0)
    return tl.where(okay,lib.log(w),-float('inf'))


@triton.jit
def _target(s,consumed,filled,target,book,mask,books,counts,r,PS:tl.constexpr,K:tl.constexpr):
    w=s[0]+s[1]*s[2];e=s[1]*s[2]/w;trade=_finite(target)
    t=tl.where(trade,target,0.)
    den=tl.where(e<t,1-r[1]+r[1]*t,1-r[1]*t)
    request=tl.where(e<t,-w*(t-e)/den,(1-r[1])*w*(e-t)/den)
    request=tl.where(trade,request,0.)
    amount=tl.where(request>0,1.,tl.where(request<0,-1.,0.))*tl.minimum(tl.abs(request),tl.maximum(0.,r[8]-consumed))
    okay=(w>0)&((~trade)|((t>=r[3])&(t<=r[4])&(den>0)))&_size(amount,r)
    q=s[0]+amount;a=s[1]-tl.where(amount<=0,(1-r[1])*amount/s[2],amount/((1-r[1])*s[2]))
    a=tl.where(trade&(t==0)&(tl.abs(request)<=r[8]-consumed)&_size(request,r),0.,a)
    new_w=q+a*s[2];orders=();count=tl.load(counts+book)
    for j in tl.static_range(K):
        z=tl.load(books+(book*K+j)*2);frac=tl.load(books+(book*K+j)*2+1)
        side=tl.where(frac>0,1.,-1.);distance=z/(1-z)*PS
        ticks=tl.maximum(1.,lib.nearbyint(s[2]*distance/r[9]))
        L=(lib.nearbyint(s[2]/r[9])-side*ticks)*r[9]
        place=(mask<0)&(j<count);keep=(mask>=0)&((mask&(1<<j))!=0)
        orders+=(tl.where(place,side,tl.where(keep,s[3+3*j],0.)),
                 tl.where(place,L,tl.where(keep,s[4+3*j],0.)),
                 tl.where(place,tl.abs(frac)*new_w,tl.where(keep,s[5+3*j],0.)))
    out=(q,a,s[2])+orders
    return out,okay&_valid(out,r,amount!=0,filled,K)


@triton.jit
def _bracket(axis,x,N:tl.constexpr,BITS:tl.constexpr):
    lo=tl.full(x.shape,0,tl.int32);hi=tl.full(x.shape,N,tl.int32)
    for _ in tl.static_range(BITS):
        mid=(lo+hi)//2;y=tl.load(axis+mid,mask=mid<N,other=float('inf'))
        right=(mid<N)&(y<=x);lo=tl.where(right,mid+1,lo);hi=tl.where(right,hi,mid)
    i=tl.minimum(tl.maximum(lo-1,0),N-2)
    left=tl.load(axis+i);right=tl.load(axis+i+1)
    alpha=tl.minimum(tl.maximum((x-left)/(right-left),0.),1.)
    okay=(x>=tl.load(axis)-1.e-9)&(x<=tl.load(axis+N-1)+1.e-9)
    return i,alpha,okay


@triton.jit
def _lookup(s,node,values,wealth,exposures,offsets,book_lookup,size_edges,
            W:tl.constexpr,E:tl.constexpr,C:tl.constexpr,M:tl.constexpr,WBITS:tl.constexpr,EBITS:tl.constexpr,
            PB:tl.constexpr,SB:tl.constexpr,PS:tl.constexpr,K:tl.constexpr):
    w=s[0]+s[1]*s[2];e=s[1]*s[2]/w
    wi,wa,wok=_bracket(wealth,w,W,WBITS);ei,ea,eok=_bracket(exposures,e,E,EBITS)
    okay=(w>0)&wok&eok;count=tl.full(w.shape,0,tl.int32);ids=();T:tl.constexpr=PB*SB
    for j in tl.static_range(K):
        side=s[3+3*j];L=s[4+3*j];n=side*s[5+3*j]/w;active=side!=0
        okay&=(~active)|((n>=tl.load(size_edges))&(n<=tl.load(size_edges+SB)))
        x=tl.abs(L/s[2]-1)/PS;z=x/(1+x);pi=tl.minimum((z*PB).to(tl.int32),PB-1)
        si=tl.full(n.shape,-1,tl.int32)
        for k in tl.static_range(SB+1):si+=(n>=tl.load(size_edges+k)).to(tl.int32)
        si=tl.minimum(tl.maximum(si,0),SB-1)
        ids+=(tl.where(active,pi*SB+si,T),);count+=active.to(tl.int32)
    # Sorting network on at most three slots, preserving order multiplicity.
    if K>=2:ids=(tl.minimum(ids[0],ids[1]),tl.maximum(ids[0],ids[1]))+ids[2:]
    if K==3:
        ids=(ids[0],tl.minimum(ids[1],ids[2]),tl.maximum(ids[1],ids[2]))
        ids=(tl.minimum(ids[0],ids[1]),tl.maximum(ids[0],ids[1]),ids[2])
    code=tl.load(offsets+count)
    for j in tl.static_range(K):code+=tl.where(j<count,ids[j]*T**j,0)
    book=tl.load(book_lookup+code);okay&=book>=0;book=tl.maximum(book,0)
    norm=tl.full(w.shape,0.,tl.float64)
    for dw in tl.static_range(2):
        for de in tl.static_range(2):
            weight=(wa if dw else 1-wa)*(ea if de else 1-ea);used=weight>1.e-12
            v=tl.load(values+((wi+dw)*C+book*E+ei+de)*M+node,mask=okay&used,other=0.)
            okay&=(~used)|_finite(v);norm+=tl.where(used,weight*v,0.)
    return tl.where(okay,lib.log(w)+norm,-float('inf'))


@triton.jit
def _post(s,node,values,wealth,exposures,offsets,book_lookup,size_edges,r,next_factors,
          W:tl.constexpr,E:tl.constexpr,C:tl.constexpr,M:tl.constexpr,WBITS:tl.constexpr,EBITS:tl.constexpr,
          PB:tl.constexpr,SB:tl.constexpr,PS:tl.constexpr,K:tl.constexpr,B:tl.constexpr,LAST_DECISION:tl.constexpr):
    if LAST_DECISION:
        estimate=tl.full(s[0].shape,-float('inf'),tl.float64)
    else:
        estimate=_lookup(s,node,values,wealth,exposures,offsets,book_lookup,size_edges,W,E,C,M,WBITS,EBITS,PB,SB,PS,K)
    mean=tl.full(estimate.shape,0.,tl.float64);okay=tl.full(estimate.shape,True,tl.int1)
    arrivals_ok=tl.full(estimate.shape,True,tl.int1)
    for j in range(B):
        p=tl.maximum(1.,lib.nearbyint(s[2]*tl.load(next_factors+node*B+j)/r[9]))*r[9]
        arrived,consumed,filled,valid=_arrive(s,p,r,K);v=_close(arrived,consumed,r)
        arrivals_ok&=valid
        okay&=valid&_finite(v);mean+=(tl.where(_finite(v),v,0.)-mean)/(j+1)
    return tl.where(arrivals_ok,tl.where(okay,tl.maximum(estimate,mean),estimate),-float('inf'))


@triton.jit(do_not_specialize=['depth'])
def _outcomes(prices,sampled_prices,node_counts,destinations,perms,allocation,values,wealth,exposures,books,counts,
              offsets,book_lookup,size_edges,rules,candidates,next_factors,outcomes,depth,
              active_accounts,
              W:tl.constexpr,E:tl.constexpr,C:tl.constexpr,M:tl.constexpr,K:tl.constexpr,B:tl.constexpr,
              PB:tl.constexpr,SB:tl.constexpr,PS:tl.constexpr,NC:tl.constexpr,AC:tl.constexpr,
              WBITS:tl.constexpr,EBITS:tl.constexpr,TERMINAL:tl.constexpr,LAST_DECISION:tl.constexpr,PER_POINT:tl.constexpr):
    index=tl.program_id(0);draw=index%B;task=index//B;node=task%M
    account=tl.load(active_accounts+task//M);cell=account%C;wi=account//C
    r=()
    for j in tl.static_range(11):r+=(tl.load(rules+j),)
    result=tl.full((),-float('inf'),tl.float64)
    budget_index=account*M+node if PER_POINT else cell
    if (node<tl.load(node_counts+depth))&(draw<tl.load(allocation+budget_index)):
        b=cell//E;ei=cell%E;w=tl.load(wealth+wi);e=tl.load(exposures+ei);p=tl.load(prices+depth*M+node)
        orders=();count=tl.load(counts+b)
        for j in tl.static_range(K):
            z=tl.load(books+(b*K+j)*2);n=tl.load(books+(b*K+j)*2+1);side=tl.where(n>0,1.,-1.)
            ticks=tl.maximum(1.,lib.nearbyint(p*z/(1-z)*PS/r[9]));L=(lib.nearbyint(p/r[9])-side*ticks)*r[9]
            orders+=(tl.where(j<count,side,0.),tl.where(j<count,L,0.),tl.where(j<count,tl.abs(n)*w,0.))
        s=(w*(1-e),w*e/p,p)+orders
        if _valid(s,r,False,False,K):
            edge=tl.load(perms+node*B+draw);child=tl.load(destinations+(depth*M+node)*B+edge)
            arrived,consumed,filled,valid=_arrive(s,tl.load(sampled_prices+(depth*M+node)*B+edge),r,K)
            if valid:
                if TERMINAL:
                    result=_close(arrived,consumed,r)-lib.log(w)
                else:
                    lane=tl.arange(0,AC);is_catalog=(lane>0)&(lane<=NC)
                    ci=tl.load(candidates+lane-1,mask=is_catalog,other=0);book=ci//E;target_index=ci%E
                    retained=lane-NC-1;mask=retained//(E+1);retain_e=retained%(E+1)-1
                    occupied=tl.full((),0,tl.int32)
                    for j in tl.static_range(K):occupied|=tl.where(arrived[3+3*j]!=0,1<<j,0)
                    in_menu=is_catalog|((retained>=0)&(mask<(1<<K))&((mask&~occupied)==0))
                    target_index=tl.where(is_catalog,target_index,tl.maximum(retain_e,0))
                    target=tl.load(exposures+target_index)
                    target=tl.where((~is_catalog)&(retain_e<0),float('nan'),target)
                    book=tl.where(is_catalog,book,0);mask=tl.where(is_catalog,-1,mask)
                    out,feasible=_target(arrived,consumed,filled,target,book,mask,books,counts,r,PS,K)
                    child_values=values+(depth+1)*W*C*M
                    val=_post(out,child,child_values,wealth,exposures,offsets,book_lookup,size_edges,r,next_factors,
                              W,E,C,M,WBITS,EBITS,PB,SB,PS,K,B,LAST_DECISION)
                    val=tl.where(in_menu&feasible,val,-float('inf'))
                    closed=_close(arrived,consumed,r);flat=(arrived[0]+_effective(arrived,r),tl.full((),0.,tl.float64),arrived[2])
                    for j in tl.static_range(K):flat+=(tl.full((),0.,tl.float64),tl.full((),0.,tl.float64),tl.full((),0.,tl.float64))
                    if LAST_DECISION:
                        cv=tl.full((),-float('inf'),tl.float64)
                    else:
                        cv=_lookup(flat,child,child_values,wealth,exposures,offsets,book_lookup,size_edges,W,E,C,M,WBITS,EBITS,PB,SB,PS,K)
                    cash=tl.where(_finite(closed),tl.maximum(closed,cv),-float('inf'))
                    result=tl.maximum(cash,tl.max(val,0))-lib.log(w)
    tl.store(outcomes+(account*M+node)*B+draw,result)


@triton.jit(do_not_specialize=['depth'])
def _average(outcomes,allocation,node_counts,values,variance,depth,W:tl.constexpr,C:tl.constexpr,M:tl.constexpr,B:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);active=i<W*C*M;node=i%M;cell=(i//M)%C
    n=tl.load(allocation+cell,mask=active,other=2)
    mean=tl.full((BLOCK,),0.,tl.float64);m2=tl.full((BLOCK,),0.,tl.float64);okay=tl.full((BLOCK,),True,tl.int1)
    for j in range(B):
        used=j<n;v=tl.load(outcomes+i*B+j,mask=active&used,other=0.)
        okay&=(~used)|_finite(v);v=tl.where(_finite(v),v,0.)
        delta=v-mean;updated=mean+delta/(j+1);m2+=tl.where(used,delta*(v-updated),0.);mean=tl.where(used,updated,mean)
    parent=node<tl.load(node_counts+depth)
    value=tl.where(parent,tl.where(okay,mean,-float('inf')),float('nan'))
    var=tl.where(parent&okay,m2/tl.maximum(n-1,1),float('nan'))
    tl.store(values+depth*W*C*M+i,value,mask=active);tl.store(variance+depth*W*C*M+i,var,mask=active)


@triton.jit(do_not_specialize=['depth','shift'])
def _reuse_plan(previous,mapping,requested,baseline,base_mean,effective,depth,shift,
                prices,sampled_prices,node_counts,wealth,exposures,books,book_counts,rules,
                W:tl.constexpr,C:tl.constexpr,M:tl.constexpr,B:tl.constexpr,OLD_M:tl.constexpr,OLD_H:tl.constexpr,
                E:tl.constexpr,K:tl.constexpr,PS:tl.constexpr,REFERENCE:tl.constexpr,HAS_PREVIOUS:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);active=i<W*C*M;node=i%M;account=i//M;cell=account%C
    old_node=tl.load(mapping+depth*M+node,mask=active,other=-1)
    available=active&(old_node>=0)&(depth+shift<OLD_H)
    okay=available;mean=tl.full((BLOCK,),0.,tl.float64)
    for j in range(B):
        old=tl.full((BLOCK,),-float('inf'),tl.float64)
        if HAS_PREVIOUS:
            old=tl.load(previous+(((depth+shift)*W*C+account)*OLD_M+old_node)*B+j,
                mask=available,other=-float('inf'))
        okay&=_finite(old);mean+=(tl.where(_finite(old),old,0.)-mean)/(j+1)
        tl.store(baseline+i*B+j,old,mask=active)
    # Partial value estimates may not miss an infeasible conditional outcome.
    # Closing after every possible arrival is a cheap, explicit feasibility
    # witness. Without that witness, evaluate the complete action backup.
    parent=active&(node<tl.load(node_counts+depth));w=tl.load(wealth+account//C,mask=active,other=1.)
    e=tl.load(exposures+cell%E,mask=active,other=0.);p=tl.load(prices+depth*M+node,mask=parent,other=1.)
    count=tl.load(book_counts+cell//E,mask=active,other=0);r=();orders=()
    for j in tl.static_range(11):r+=(tl.load(rules+j),)
    for j in tl.static_range(K):
        z=tl.load(books+((cell//E)*K+j)*2,mask=active,other=0.)
        frac=tl.load(books+((cell//E)*K+j)*2+1,mask=active,other=0.);side=tl.where(frac>0,1.,-1.)
        ticks=tl.maximum(1.,lib.nearbyint(p*z/(1-z)*PS/r[9]));L=(lib.nearbyint(p/r[9])-side*ticks)*r[9]
        orders+=(tl.where(j<count,side,0.),tl.where(j<count,L,0.),tl.where(j<count,tl.abs(frac)*w,0.))
    s=(w*(1-e),w*e/p,p)+orders
    okay&=parent&_valid(s,r,False,False,K)
    for j in range(B):
        target_price=tl.load(sampled_prices+(depth*M+node)*B+j,mask=parent,other=1.)
        arrived,consumed,filled,valid=_arrive(s,target_price,r,K)
        okay&=valid&_finite(_close(arrived,consumed,r))
    n=tl.load(requested+cell,mask=active,other=B)
    tl.store(effective+i,tl.where(okay&(account!=REFERENCE),n,B),mask=active)
    tl.store(base_mean+i,mean,mask=active)


@triton.jit(do_not_specialize=['depth'])
def _paired_average(outcomes,effective,node_counts,values,variance,baseline,base_mean,cache,perms,depth,
                    W:tl.constexpr,C:tl.constexpr,M:tl.constexpr,B:tl.constexpr,REFERENCE:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);active=i<W*C*M;node=i%M
    n=tl.load(effective+i,mask=active,other=B);partial=n<B
    mean=tl.full((BLOCK,),0.,tl.float64);m2=tl.full((BLOCK,),0.,tl.float64);okay=tl.full((BLOCK,),True,tl.int1)
    reference_mean=tl.full((BLOCK,),0.,tl.float64)
    parent=node<tl.load(node_counts+depth)
    # Copy every old component BEFORE replacing the sampled canonical edges.
    for j in range(B):
        old=tl.load(baseline+i*B+j,mask=active,other=0.)
        tl.store(cache+(depth*W*C*M+i)*B+j,tl.where(parent,old,float('nan')),mask=active)
    for j in range(B):
        used=j<n;edge=tl.load(perms+node*B+j,mask=active,other=0)
        v=tl.load(outcomes+i*B+j,mask=active&used,other=0.)
        reference=tl.load(outcomes+(REFERENCE*M+node)*B+j,mask=active,other=0.)
        reference_mean+=(reference-reference_mean)/(j+1)
        old=tl.load(baseline+i*B+edge,mask=active&partial,other=0.)
        okay&=(~used)|_finite(v)
        d=tl.where(partial,(v-reference)-old,v);d=tl.where(_finite(d),d,0.)
        delta=d-mean;updated=mean+delta/(j+1);m2+=tl.where(used,delta*(d-updated),0.);mean=tl.where(used,updated,mean)
        tl.store(cache+(depth*W*C*M+i)*B+edge,v-reference,mask=active&parent&used)
    estimate=mean+tl.where(partial,tl.load(base_mean+i,mask=active,other=0.)+reference_mean,0.)
    value=tl.where(parent,tl.where(okay,estimate,-float('inf')),float('nan'))
    var=tl.where(parent&okay,m2/tl.maximum(n-1,1),float('nan'))
    tl.store(values+depth*W*C*M+i,value,mask=active);tl.store(variance+depth*W*C*M+i,var,mask=active)


class GPUBookSweep:
    def __init__(self,grid,cloud,rules,previous=None,shift=1,paired=False,feature_std=None,active_hint=None):
        if not torch.cuda.is_available():raise RuntimeError('A CUDA GPU is required')
        a=grid.arrays();self.grid=grid;self.cloud=cloud;self.device=torch.device('cuda')
        self.W=len(a['wealth']);self.E=len(a['exposures']);self.C=len(a['coordinates']);self.M=cloud.prices.shape[1]
        self.B=cloud.destinations.shape[-1];self.H=len(cloud.counts)-1
        self.arrays={k:self.tensor(a[k]) for k in ('wealth','exposures','books','book_counts','code_offsets','book_lookup','size_edges')}
        self.prices=self.tensor(cloud.prices);self.counts=self.tensor(cloud.counts);self.dest=self.tensor(cloud.destinations)
        self.sampled_prices=self.tensor(cloud.sampled_prices)
        factors=np.zeros_like(cloud.sampled_prices)
        np.divide(cloud.sampled_prices,cloud.prices[:-1,:,None],out=factors,where=cloud.prices[:-1,:,None]>0)
        self.factors=self.tensor(factors)
        self.terminal_factors=self.tensor(np.zeros((self.M,self.B)))
        self.rules=self.tensor(rules)
        self.values=torch.empty((self.H,self.W,self.C,self.M),device=self.device,dtype=torch.float64)
        self.variance=torch.empty_like(self.values)
        # Unscheduled outcomes stay -inf; averaging still writes EVERY cell.
        self.outcomes=torch.full((self.W*self.C*self.M*self.B,),-float('inf'),device=self.device,dtype=torch.float64)
        if active_hint is None:
            market_prices=np.unique(cloud.prices[:-1][np.arange(self.M)[None,:]<cloud.counts[:-1,None]])
            active=active_account_indices(a['wealth'],a['exposures'],a['books'],a['book_counts'],market_prices,rules,grid.price_scale)
        else:
            active=np.asarray(active_hint,dtype=np.int64)
            if np.any((active<0)|(active>=self.W*self.C)):raise ValueError('Invalid account work indices')
        self.active_indices=active
        self.active_accounts=self.tensor(active);self.active_count=len(active)
        self.paired=paired;self.shift=shift
        if paired:
            from oracle_dense_reuse import project_market_nodes
            self.cache=torch.empty((self.H,self.W,self.C,self.M,self.B),device=self.device,dtype=torch.float64)
            self.baseline=torch.empty_like(self.outcomes)
            self.baseline_mean=torch.empty((self.W*self.C*self.M,),device=self.device,dtype=torch.float64)
            self.effective_counts=torch.empty((self.H,self.W,self.C,self.M),device=self.device,dtype=torch.int32)
            self.previous_cache=None;self.previous_M=self.M;self.previous_H=0
            self.reference_account=(self.W//2)*self.C+int(np.flatnonzero(a['exposures']==0)[0])
            self.mapping=self.tensor(np.full((self.H,self.M),-1,dtype=np.int64))
            if previous is not None:
                if previous.outcome_representation!='cash-centered-v1':raise ValueError('Expected cash-centered outcome cache')
                if previous.outcome_cache is None:raise ValueError('Previous map has no outcome cache')
                if previous.outcome_cache.shape[1:3]!=(self.W,self.C) or previous.outcome_cache.shape[-1]!=self.B:
                    raise ValueError('Previous outcome grid differs')
                self.previous_M=previous.shape[-1];self.previous_H=previous.shape[0]
                self.previous_cache=self.tensor(previous.outcome_cache)
                self.mapping=self.tensor(project_market_nodes(cloud,previous.cloud,shift,feature_std))

    def tensor(self,x):
        # Device storage is a private builder, including for read-only cache
        # mappings. Never create a writable tensor alias of the host input.
        return torch.tensor(np.ascontiguousarray(x),device=self.device)

    def step(self,depth,permutations,allocation,candidates):
        a=self.arrays;g=self.grid;B=self.B
        next_factors=self.factors[depth+1] if depth+1<self.H else self.terminal_factors
        # Root has fewer nodes than the common device stride.
        padded=np.zeros((self.M,B),dtype=np.int64);padded[:len(permutations)]=permutations
        perms=self.tensor(padded);budget=self.tensor(allocation)
        if self.paired:
            previous=self.previous_cache if self.previous_cache is not None else self.baseline
            effective=self.effective_counts[depth]
            _reuse_plan[(triton.cdiv(self.W*self.C*self.M,128),)](previous,self.mapping,budget,self.baseline,
                self.baseline_mean,effective,depth,self.shift,self.prices,self.sampled_prices,self.counts,
                a['wealth'],a['exposures'],a['books'],a['book_counts'],self.rules,
                self.W,self.C,self.M,B,self.previous_M,self.previous_H,self.E,g.max_orders,g.price_scale,
                self.reference_account,self.previous_cache is not None,128,num_warps=4,enable_fp_fusion=False)
            budget=effective
        args=(self.prices,self.sampled_prices,self.counts,self.dest,perms,budget,self.values,
            a['wealth'],a['exposures'],a['books'],a['book_counts'],a['code_offsets'],a['book_lookup'],a['size_edges'],
            self.rules,self.tensor(candidates),next_factors,self.outcomes,depth,self.active_accounts)
        action_count=1+len(candidates)+(1<<g.max_orders)*(self.E+1)
        _outcomes[(self.active_count*self.M*B,)](*args,self.W,self.E,self.C,self.M,g.max_orders,B,
            g.price_bins,g.size_bins,g.price_scale,len(candidates),triton.next_power_of_2(action_count),
            self.W.bit_length(),self.E.bit_length(),depth==self.H-1,depth==self.H-2,self.paired,num_warps=4,enable_fp_fusion=False)
        if self.paired:
            _paired_average[(triton.cdiv(self.W*self.C*self.M,128),)](self.outcomes,budget,self.counts,self.values,
                self.variance,self.baseline,self.baseline_mean,self.cache,perms,depth,
                self.W,self.C,self.M,B,self.reference_account,128,num_warps=4,enable_fp_fusion=False)
        else:
            _average[(triton.cdiv(self.W*self.C*self.M,128),)](self.outcomes,budget,self.counts,self.values,self.variance,
                depth,self.W,self.C,self.M,B,128,num_warps=4,enable_fp_fusion=False)

    def synchronize(self):
        torch.cuda.synchronize(self.device)

    def finish(self,values,variance,outcome_cache=None):
        np.copyto(values,self.values.cpu().numpy());np.copyto(variance,self.variance.cpu().numpy())
        if self.paired:
            if outcome_cache is None:raise ValueError('Missing outcome-cache destination')
            np.copyto(outcome_cache,self.cache.cpu().numpy())

    def reuse_stats(self):
        if not self.paired:return {}
        counts=self.effective_counts.reshape(self.H,self.W*self.C,self.M)[:,self.active_accounts]
        valid=torch.arange(self.M,device=self.device)[None,None,:]<self.counts[:-1,None,None]
        denominator=self.active_count*int(self.cloud.counts[:-1].sum())
        total=int(torch.where(valid,counts,0).sum().item())
        return dict(effectiveMeanConditionalSamples=total/denominator,
            cachedOutcomeFraction=1-total/(denominator*self.B),outcomeStorageBytes=self.cache.numel()*8)
