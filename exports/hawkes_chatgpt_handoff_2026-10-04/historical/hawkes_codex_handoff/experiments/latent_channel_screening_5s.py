#!/usr/bin/env python3
"""Reference residual-proxy latent-channel screen.

WARNING: this does NOT fit a true latent Hawkes channel. It constructs an
observation-driven pseudo-event process from common positive B/S residuals.
Use only as a numerical reference/sanity check.
"""
import argparse
import numpy as np, pandas as pd
from scipy.signal import lfilter
from scipy.optimize import nnls
from scipy.special import xlogy


def run(cache, output):
    Y=np.load(cache)["Y"].astype(float)
    n=len(Y); ntr=int(.60*n); nv=int(.20*n); iva=np.arange(ntr,ntr+nv); ite=np.arange(ntr+nv,n)
    taus=np.array([5,10,20,40,80,160,320,640,1280,2560,5120,10240,20480,43200.])
    X=[]
    for tau in taus:
        r=np.exp(-5/tau)
        for j in range(2):
            z=lfilter([1.],[1.,-r],Y[:,j]); X.append(np.r_[0,z[:-1]]*(1-r))
    X=np.asarray(X).T
    def ploss(y,p): p=np.maximum(p,1e-7); return float(np.mean(p-xlogy(y,p)))
    def fit(F,y): c,_=nnls(np.c_[np.ones(len(F)),F],y,maxiter=3000); return c
    def pred(F,c): return np.maximum(c[0]+F@c[1:],1e-7)
    sels=[]
    for out in range(2):
        cur=[]; best=np.inf
        for _ in range(12):
            cand=None
            for q in range(X.shape[1]):
                if q in cur: continue
                c=fit(X[:ntr,cur+[q]],Y[:ntr,out]); s=ploss(Y[iva,out],pred(X[iva][:,cur+[q]],c))
                if cand is None or s<cand[0]: cand=(s,q)
            if cur and best-cand[0]<2e-4: break
            best,q=cand; cur.append(q)
        sels.append(cur)
    basecols=sorted(set(sels[0]+sels[1])); F0=X[:,basecols]
    C0=[fit(F0[:ntr],Y[:ntr,o]) for o in range(2)]; P0=np.column_stack([pred(F0,c) for c in C0]); R=Y-P0
    latent_taus=np.array([10,40,160,640,2560,10240,43200.])
    def ltrace(NL,tau):
        r=np.exp(-5/tau); z=lfilter([1.],[1.,-r],NL); return np.r_[0,z[:-1]]*(1-r)
    def fitL(K):
        NL=np.maximum(np.minimum(R[:,0],R[:,1]),0); NL=np.maximum(NL-np.quantile(NL[:ntr],.70),0)
        if NL[:ntr].mean()>0: NL/=NL[:ntr].mean()
        chosen=[]
        for _ in range(K):
            best=None
            for tau in latent_taus:
                if tau in chosen: continue
                FL=np.column_stack([ltrace(NL,t) for t in chosen+[tau]]); F=np.c_[F0,FL]
                cs=[fit(F[:ntr],Y[:ntr,o]) for o in range(2)]; pv=np.column_stack([pred(F[iva],c) for c in cs])
                s=sum(ploss(Y[iva,o],pv[:,o]) for o in range(2))
                if best is None or s<best[0]: best=(s,tau)
            chosen.append(best[1])
        for _ in range(6):
            FL=np.column_stack([ltrace(NL,t) for t in chosen]); F=np.c_[F0,FL]
            cs=[fit(F[:ntr],Y[:ntr,o]) for o in range(2)]; p=np.column_stack([pred(F,c) for c in cs]); rr=Y-p
            score=np.maximum(0,.5*(rr[:,0]+rr[:,1])); thr=np.quantile(score[:ntr],.75); new=np.maximum(score-thr,0)
            if chosen: new += .15*np.maximum(sum(ltrace(NL,t) for t in chosen)/len(chosen),0)
            if new[:ntr].mean()>0: new/=new[:ntr].mean()
            NL=.5*NL+.5*new
        FL=np.column_stack([ltrace(NL,t) for t in chosen]); F=np.c_[F0,FL]; cs=[fit(F[:ntr],Y[:ntr,o]) for o in range(2)]
        return chosen,NL,F,cs
    rows=[]
    pt=np.column_stack([pred(F0[ite],c) for c in C0]); pv=np.column_stack([pred(F0[iva],c) for c in C0])
    rows.append(["2ch",[],sum(ploss(Y[iva,o],pv[:,o]) for o in range(2)),sum(ploss(Y[ite,o],pt[:,o]) for o in range(2)),np.mean(np.abs(Y[ite].sum(1)-pt.sum(1))),np.nan,np.nan])
    for K in [1,2,3]:
        chosen,NL,F,cs=fitL(K); pv=np.column_stack([pred(F[iva],c) for c in cs]); pt=np.column_stack([pred(F[ite],c) for c in cs])
        rows.append([f"latent-L K={K}",chosen,sum(ploss(Y[iva,o],pv[:,o]) for o in range(2)),sum(ploss(Y[ite,o],pt[:,o]) for o in range(2)),np.mean(np.abs(Y[ite].sum(1)-pt.sum(1))),np.mean(NL[ite]>.1),np.corrcoef(NL[ite][:-1],NL[ite][1:])[0,1]])
    df=pd.DataFrame(rows,columns=["model","latent_taus_s","val_BS_poisson_loss","test_BS_poisson_loss","test_total_MAE_5s","latent_active_fraction","latent_lag1_corr"])
    df.to_csv(output,index=False); print(df.to_string(index=False))

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("cache"); ap.add_argument("output"); a=ap.parse_args(); run(a.cache,a.output)
