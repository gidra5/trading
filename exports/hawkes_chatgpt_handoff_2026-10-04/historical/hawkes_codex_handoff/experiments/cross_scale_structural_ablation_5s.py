#!/usr/bin/env python3
"""Reference reproduction of the latest 5s M0/M2/M3 screening.

This is a feature-based structural screen, not the final continuous-time model.
It uses a finite timescale bank and NNLS Poisson-mean fitting.
"""
import argparse, csv
import numpy as np
import pandas as pd
from scipy.signal import lfilter
from scipy.optimize import nnls
from scipy.special import xlogy


def run(cache, output):
    d=np.load(cache); Y=d["Y"].astype(float)
    n=len(Y); ntr=int(.60*n); nv=int(.20*n); va=slice(ntr,ntr+nv); te=slice(ntr+nv,n)
    taus=np.array([5,10,20,40,80,160,320,640,1280,2560,5120,10240,20480,43200.],float)
    X=[]
    for tau in taus:
        rho=np.exp(-5/tau)
        for j in range(2):
            z=lfilter([1.0],[1.0,-rho],Y[:,j])
            X.append(np.r_[0,z[:-1]]*(1-rho))
    X=np.asarray(X).T
    def ploss(y,p):
        p=np.maximum(p,1e-6); return float(np.mean(p-xlogy(y,p)))
    def fit(F,y):
        c,_=nnls(np.c_[np.ones(len(F)),F],y,maxiter=2000); return c
    def pred(F,c): return np.maximum(c[0]+F@c[1:],1e-6)
    selected=[[],[]]
    for out in range(2):
        cur=[]; best=np.inf
        for _ in range(12):
            bt=None
            for q in range(X.shape[1]):
                if q in cur: continue
                cols=cur+[q]; c=fit(X[:ntr,cols],Y[:ntr,out]); s=ploss(Y[va,out],pred(X[va,cols],c))
                if bt is None or s<bt[0]: bt=(s,q)
            if cur and best-bt[0]<2e-4: break
            best,q=bt; cur.append(q)
        selected[out]=cur
    basecols=sorted(set(selected[0]+selected[1])); Fb=X[:,basecols]
    mu=X[:ntr].mean(0); sd=X[:ntr].std(0)+1e-8; Z=(X-mu)/sd; Zb=Z[:,basecols]
    prod2=[]
    for a,qa in enumerate(basecols):
        ta=taus[qa//2]
        for b,qb in enumerate(basecols):
            tb=taus[qb//2]
            if tb>=4*ta: prod2.append(Zb[:,a]*Zb[:,b])
    F2=np.c_[Fb,np.asarray(prod2).T] if prod2 else Fb
    prod3=[]
    for a,qa in enumerate(basecols):
        for b in range(a,len(basecols)):
            qb=basecols[b]; ratio=max(taus[qa//2],taus[qb//2])/min(taus[qa//2],taus[qb//2])
            if ratio<4: prod3.append(Zb[:,a]*Zb[:,b])
    F3=np.c_[F2,np.asarray(prod3).T] if prod3 else F2
    def signed_expand(base,full):
        if full.shape[1]==base.shape[1]: return base
        G=full[:,base.shape[1]:]; return np.c_[base,np.maximum(G,0),np.maximum(-G,0)]
    models={"M0":Fb,"M2":signed_expand(Fb,F2),"M3":signed_expand(Fb,F3)}
    rows=[]
    for mn,F in models.items():
        for out,name in enumerate(["B","S"]):
            c=fit(F[:ntr],Y[:ntr,out]); pv=pred(F[va],c); pt=pred(F[te],c)
            rows.append([mn,name,F.shape[1],ploss(Y[va,out],pv),ploss(Y[te,out],pt),float(np.mean(np.abs(Y[te,out]-pt)))])
        pp=[]
        for out in range(2):
            c=fit(F[:ntr],Y[:ntr,out]); pp.append(pred(F[te],c))
        rows.append([mn,"TOTAL",F.shape[1],np.nan,np.nan,float(np.mean(np.abs(Y[te].sum(1)-sum(pp))))])
    df=pd.DataFrame(rows,columns=["model","output","n_features","val_poisson_loss","test_poisson_loss","test_MAE_5s"])
    df.to_csv(output,index=False); print(df.to_string(index=False))

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("cache"); ap.add_argument("output"); a=ap.parse_args(); run(a.cache,a.output)
