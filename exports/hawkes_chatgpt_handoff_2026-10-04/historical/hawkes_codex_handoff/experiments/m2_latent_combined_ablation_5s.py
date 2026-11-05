#!/usr/bin/env python3
"""Reference M0 vs M2 vs residual-proxy-L vs M2+proxy-L screen.

WARNING: L is observation-driven here, not a true generative latent channel.
"""
import argparse
import numpy as np, pandas as pd
from scipy.signal import lfilter
from scipy.optimize import nnls
from scipy.special import xlogy


def run(cache, output):
    Y=np.load(cache)["Y"].astype(float)
    n=len(Y); ntr=int(.60*n); nv=int(.20*n); va=slice(ntr,ntr+nv); te=slice(ntr+nv,n)
    taus=np.array([5,10,20,40,80,160,320,640,1280,2560,5120,10240,20480,43200.])
    X=[]
    for tau in taus:
        r=np.exp(-5/tau)
        for j in range(2):
            z=lfilter([1.],[1.,-r],Y[:,j]); X.append(np.r_[0,z[:-1]]*(1-r))
    X=np.asarray(X).T
    def loss(y,p): p=np.maximum(p,1e-7); return float(np.mean(p-xlogy(y,p)))
    def fit(F,y): c,_=nnls(np.c_[np.ones(len(F)),F],y,maxiter=3000); return c
    def pred(F,c): return np.maximum(c[0]+F@c[1:],1e-7)
    sels=[]
    for o in range(2):
        cur=[]; best=np.inf
        for _ in range(12):
            cand=None
            for q in range(X.shape[1]):
                if q in cur: continue
                c=fit(X[:ntr,cur+[q]],Y[:ntr,o]); s=loss(Y[va,o],pred(X[va,cur+[q]],c))
                if cand is None or s<cand[0]: cand=(s,q)
            if cur and best-cand[0]<2e-4: break
            best,q=cand; cur.append(q)
        sels.append(cur)
    basecols=sorted(set(sels[0]+sels[1])); F0=X[:,basecols]
    mu=X[:ntr].mean(0); sd=X[:ntr].std(0)+1e-8; Z=(X-mu)/sd; Zb=Z[:,basecols]
    prods=[]
    for a,qa in enumerate(basecols):
        for b,qb in enumerate(basecols):
            if taus[qb//2]>=4*taus[qa//2]: prods.append(Zb[:,a]*Zb[:,b])
    G=np.asarray(prods).T; FM2=np.c_[F0,np.maximum(G,0),np.maximum(-G,0)]
    def fitL(Fobs,tauL=10.,iters=8):
        cs=[fit(Fobs[:ntr],Y[:ntr,o]) for o in range(2)]; P=np.column_stack([pred(Fobs,c) for c in cs])
        score=np.maximum(0,.5*((Y-P)[:,0]+(Y-P)[:,1])); NL=np.maximum(score-np.quantile(score[:ntr],.75),0)
        if NL[:ntr].mean()>0: NL/=NL[:ntr].mean()
        r=np.exp(-5/tauL)
        for _ in range(iters):
            z=lfilter([1.],[1.,-r],NL); z=np.r_[0,z[:-1]]*(1-r); F=np.c_[Fobs,z]
            cs=[fit(F[:ntr],Y[:ntr,o]) for o in range(2)]; P=np.column_stack([pred(F,c) for c in cs]); rr=Y-P
            s=np.maximum(0,.5*(rr[:,0]+rr[:,1])); new=np.maximum(s-np.quantile(s[:ntr],.75),0)+.15*np.maximum(z,0)
            if new[:ntr].mean()>0:new/=new[:ntr].mean()
            NL=.5*NL+.5*new
        z=lfilter([1.],[1.,-r],NL); z=np.r_[0,z[:-1]]*(1-r); F=np.c_[Fobs,z]; cs=[fit(F[:ntr],Y[:ntr,o]) for o in range(2)]
        return F,cs,NL
    def ev(name,F,cs,NL=None):
        pv=np.column_stack([pred(F[va],c) for c in cs]); pt=np.column_stack([pred(F[te],c) for c in cs]); yt=Y[te]
        return dict(model=name,n_features=F.shape[1],val_loss=sum(loss(Y[va,o],pv[:,o]) for o in range(2)),test_loss=sum(loss(yt[:,o],pt[:,o]) for o in range(2)),test_total_MAE=float(np.mean(np.abs(yt.sum(1)-pt.sum(1)))),buy_MAE=float(np.mean(np.abs(yt[:,0]-pt[:,0]))),sell_MAE=float(np.mean(np.abs(yt[:,1]-pt[:,1]))),latent_active=(float(np.mean(NL[te]>.1)) if NL is not None else np.nan))
    C0=[fit(F0[:ntr],Y[:ntr,o]) for o in range(2)]; CM2=[fit(FM2[:ntr],Y[:ntr,o]) for o in range(2)]
    rows=[ev("M0",F0,C0),ev("M2",FM2,CM2)]
    FL,CL,NL=fitL(F0); rows.append(ev("M0+L",FL,CL,NL))
    FML,CML,NLM=fitL(FM2); rows.append(ev("M2+L",FML,CML,NLM))
    df=pd.DataFrame(rows); base=df.loc[df.model=="M0"].iloc[0]; df["MAE_gain_vs_M0_pct"]=100*(base.test_total_MAE-df.test_total_MAE)/base.test_total_MAE; df["test_loss_gain_vs_M0"]=base.test_loss-df.test_loss
    df.to_csv(output,index=False); print(df.to_string(index=False))

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("cache"); ap.add_argument("output"); a=ap.parse_args(); run(a.cache,a.output)
