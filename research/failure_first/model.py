"""Failure decomposition is a modeling hypothesis, not independent risk evidence."""
from __future__ import annotations
from dataclasses import dataclass, asdict
import hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from threadpoolctl import threadpool_limits

CHANNELS = ('unresolved', 'severe', 'market', 'giveback', 'residual')
METHODS = ('binary', 'failure_sum', 'failure_veto', 'failure_veto_path', 'naive_channels')

@dataclass(frozen=True)
class Config:
    horizons: tuple[int,...] = (30,60,90,126,180,252,504,756)
    thresholds: tuple[float,...] = (.70,.80,.85,.90,.95,.975)
    train_window: int = 2520
    calibration_window: int = 252
    train_every: int = 20
    minimum_training_dates: int = 24
    minimum_calibration_dates: int = 26
    half_life: int = 756
    n_estimators: int = 80
    seed: int = 20260909
    threads: int = 2
    def __post_init__(self):
        if not self.horizons or any(type(x) is not int or x<30 for x in self.horizons):
            raise ValueError('horizons must be integer sessions >=30')
        if tuple(sorted(set(self.horizons))) != self.horizons:
            raise ValueError('horizons must be unique and sorted')
        if not self.thresholds or any(not 0<x<1 for x in self.thresholds):
            raise ValueError('invalid confidence threshold')
        if min(self.train_window,self.calibration_window,self.train_every,
               self.minimum_training_dates,self.minimum_calibration_dates,
               self.half_life,self.n_estimators,self.threads)<=0:
            raise ValueError('invalid size parameter')

def fingerprint(cfg):
    return hashlib.sha256(json.dumps(asdict(cfg),sort_keys=True).encode()).hexdigest()


def date_weights(origins, asof, half_life=756):
    """Balance stocks within each date, but do NOT cancel recency across dates."""
    i=np.asarray(origins,int)
    _,ix,cnt=np.unique(i,return_inverse=True,return_counts=True)
    w=np.exp2(-(asof-i)/half_life)/cnt[ix]
    return w/w.mean()


def partition_failures(ret, benchmark_ret, peak, resolved, matured):
    """0=success, 1..5 exhaustive disjoint price-observable failure pathways.

    Market/price co-occurrence is NOT a causal attribution. Missing is not bankruptcy.
    Pending outcomes use -1 and must never enter supervised fitting.
    """
    r,b,p=np.broadcast_arrays(np.asarray(ret,float),np.asarray(benchmark_ret,float),np.asarray(peak,float))
    resolved=np.asarray(resolved,bool); matured=np.asarray(matured,bool)
    if r.shape!=resolved.shape or r.shape!=matured.shape:
        raise ValueError('label shapes differ')
    if np.any(resolved & (~matured | ~np.isfinite(r))):
        raise ValueError('invalid resolved endpoint')
    c=np.full(r.shape,-1,np.int8)
    c[matured]=5
    c[matured & ~resolved]=1
    success=matured & resolved & (r>0)
    c[success]=0
    lose=matured & resolved & ~success
    severe=lose & (r<=-.20)
    c[severe]=2
    market=lose & ~severe & np.isfinite(b) & (b<=0)
    c[market]=3
    giveback=lose & ~severe & ~market & np.isfinite(p) & (p>=.10)
    c[giveback]=4
    return c


def label_panel(f,p,market,h):
    """Label AFTER candidates are fixed; preserve immature and unavailable endpoints."""
    if type(h) is not int or h<30: raise ValueError('invalid horizon')
    i=f.i.to_numpy(int); j=p.columns.get_indexer(f.ticker)
    if (j<0).any() or (i<0).any() or (i>=len(p)).any(): raise ValueError('bad row coordinates')
    a=p.to_numpy(float); entry=a[i,j]
    if not np.allclose(entry,f.reference_price,equal_nan=False): raise ValueError('reference mismatch')
    exit_i=i+h; mature=exit_i<len(p)
    end=np.full(len(f),np.nan); end[mature]=a[exit_i[mature],j[mature]]
    resolved=mature & np.isfinite(end) & (end>0)
    ret=end/entry-1
    rev=p.iloc[::-1]
    high=rev.rolling(h,min_periods=1).max().iloc[::-1].shift(-1).to_numpy()[i,j]/entry-1
    low=rev.rolling(h,min_periods=1).min().iloc[::-1].shift(-1).to_numpy()[i,j]/entry-1
    count=rev.notna().rolling(h,min_periods=1).sum().iloc[::-1].shift(-1).to_numpy()[i,j]
    complete=mature & (count==h)
    bm=market.to_numpy(float); bret=np.full(len(f),np.nan)
    bret[mature]=bm[exit_i[mature]]/bm[i[mature]]-1
    cls=partition_failures(ret,bret,np.where(complete,high,np.nan),resolved,mature)
    # Entry at NEXT close with original locked endpoint: a stricter implementability check.
    ni=i+1; next_entry=np.full(len(f),np.nan); good=ni<len(p)
    next_entry[good]=a[ni[good],j[good]]
    next_res=resolved & np.isfinite(next_entry) & (next_entry>0)
    nextr=end/next_entry-1
    # Missing next entry is conservative failure, not assumed tradable.
    return pd.DataFrame({'row_id':f.row_id.to_numpy(),'i':i,'horizon':h,'exit_i':exit_i,
        'matured':mature,'resolved':resolved,'return':ret,'failure_class':cls,
        'success':np.where(mature,(resolved & (ret>0)).astype(float),np.nan),
        'path_complete':complete,'path_failure':np.where(mature,(~complete | (low<=-.20)).astype(float),np.nan),
        'min_return':np.where(complete,low,np.nan),'max_return':np.where(complete,high,np.nan),
        'benchmark_return':bret,'next_entry_resolved':next_res,'next_entry_return':nextr,
        'next_entry_success':np.where(mature,(next_res & (nextr>0)).astype(float),np.nan)})


def split_masks(f, labels, asof, h, cfg):
    """Full horizon purge between train, calibration and outer test."""
    start=asof-h-cfg.calibration_window
    end=labels.exit_i.to_numpy(int); i=f.i.to_numpy(int)
    cal=(i>=start)&(i<asof)&(end<asof)&labels.matured.to_numpy(bool)
    train=(end<start)&(i>=start-h-cfg.train_window)&((i-252)%cfg.train_every==0)&labels.matured.to_numpy(bool)
    return train,cal,start

class ConstantRisk:
    def __init__(self,p): self.p=float(p)
    def predict_proba(self,x):
        p=np.full(len(x),self.p); return np.column_stack([1-p,p])


def fit_head(x,y,origins,asof,cfg):
    y=np.asarray(y,int)
    if not np.isin(y,[0,1]).all(): raise ValueError('invalid supervised label')
    w=date_weights(origins,asof,cfg.half_life)
    dates=len(np.unique(origins))
    if min(y.sum(),len(y)-y.sum())<10:
        # Half an observed date of prior mass on each class; never declare unseen risk zero.
        rate=float(w@y/w.sum())
        return ConstantRisk((rate*dates+.5)/(dates+1))
    model=LGBMClassifier(n_estimators=cfg.n_estimators, num_leaves=7,max_depth=3,
        learning_rate=.05,min_child_samples=150,reg_lambda=20,max_bin=63,
        n_jobs=cfg.threads,random_state=cfg.seed,verbosity=-1,deterministic=True,
        force_col_wise=True)
    model.fit(x,y,sample_weight=w)
    return model

class Calibrator:
    """Out-of-training Platt calibrator; no probabilities from current test labels."""
    def fit(self,p,y,origins):
        p=np.asarray(p,float); y=np.asarray(y,int)
        self.constant=None; self.model=None
        if len(np.unique(y))<2:
            n=len(np.unique(origins)); self.constant=(float(y.mean())*n+.5)/(n+1)
        else:
            self.model=LogisticRegression(C=1.,solver='lbfgs',max_iter=300)
            self.model.fit(logit(np.clip(p,1e-6,1-1e-6)).reshape(-1,1),y,
                           sample_weight=date_weights(origins,0,half_life=10**9))
        return self
    def predict(self,p):
        if self.constant is not None: return np.full(len(p),self.constant)
        return self.model.predict_proba(logit(np.clip(p,1e-6,1-1e-6)).reshape(-1,1))[:,1]


def aggregate_risk(channel_risks):
    q=np.asarray(channel_risks,float)
    if q.ndim!=2 or q.shape[1]!=len(CHANNELS) or not np.isfinite(q).all() or ((q<0)|(q>1)).any():
        raise ValueError('invalid failure channel probabilities')
    return np.minimum(q.sum(axis=1),1),q.max(axis=1)


def fit_fold(f,labels,cols,asof,h,cfg,test_mask, shuffle=False):
    tr,ca,start=split_masks(f,labels,asof,h,cfg)
    audit={'asof_i':int(asof),'horizon':int(h),'calibration_start_i':int(start),
      'train_rows':int(tr.sum()),'calibration_rows':int(ca.sum()),
      'train_dates':int(f.loc[tr,'i'].nunique()),'calibration_dates':int(f.loc[ca,'i'].nunique()),
      'train_max_exit_i':int(labels.loc[tr,'exit_i'].max()) if tr.any() else None,
      'calibration_max_exit_i':int(labels.loc[ca,'exit_i'].max()) if ca.any() else None,
      'null_shuffled_training':bool(shuffle)}
    if audit['train_dates']<cfg.minimum_training_dates or audit['calibration_dates']<cfg.minimum_calibration_dates:
        audit['status']='insufficient_past_data'; return pd.DataFrame(),audit
    assert audit['train_max_exit_i']<start and audit['calibration_max_exit_i']<asof
    X=f[cols].to_numpy(np.float32)
    xt,xc,xe=X[tr],X[ca],X[test_mask]
    it,ic=f.loc[tr,'i'].to_numpy(),f.loc[ca,'i'].to_numpy()
    yt=labels.loc[tr,'failure_class'].to_numpy(int).copy(); yc=labels.loc[ca,'failure_class'].to_numpy(int)
    pt=labels.loc[tr,'path_failure'].to_numpy(int).copy(); pc=labels.loc[ca,'path_failure'].to_numpy(int)
    if shuffle:
        # Joint permutation preserves the joint failure/path label distribution.
        perm=np.random.default_rng(cfg.seed+asof+h).permutation(len(yt)); yt=yt[perm]; pt=pt[perm]
    predict=lambda m,x:m.predict_proba(x)[:,1]
    rawc=[]; rawe=[]
    with threadpool_limits(limits=cfg.threads):
        for k in range(1,6):
            head=fit_head(xt,(yt==k).astype(int),it,asof,cfg)
            rawc.append(predict(head,xc)); rawe.append(predict(head,xe))
        head=fit_head(xt,(yt>0).astype(int),it,asof,cfg)
        bc,be=predict(head,xc),predict(head,xe)
        head=fit_head(xt,pt,it,asof,cfg)
        ac,ae=predict(head,xc),predict(head,xe)
        sc,_=aggregate_risk(np.column_stack(rawc)); se,naive=aggregate_risk(np.column_stack(rawe))
        binary_cal=Calibrator().fit(bc,yc>0,ic); sum_cal=Calibrator().fit(sc,yc>0,ic)
        path_cal=Calibrator().fit(ac,pc,ic)
        binary=binary_cal.predict(be); summed=sum_cal.predict(se); path=path_cal.predict(ae)
    out=f.loc[test_mask,['row_id','i','date','ticker','vol63_rank','vol63']].copy()
    out['horizon']=h; out['exit_i']=out.i+h; out['fit_i']=asof
    out['binary']=binary; out['failure_sum']=summed
    out['failure_veto']=np.maximum(binary,summed)
    out['failure_veto_path']=np.maximum(out.failure_veto,path)
    out['naive_channels']=naive
    for name,q in zip(CHANNELS,rawe): out['channel_'+name]=q
    out['raw_sum']=se; out['raw_binary']=be; out['path_risk']=path
    audit['status']='fitted'
    audit['train_class_counts']={str(k):int((yt==k).sum()) for k in range(6)}
    audit['calibration_class_counts']={str(k):int((yc==k).sum()) for k in range(6)}
    return out.reset_index(drop=True),audit
