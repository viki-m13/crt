"""Eight frozen bearish mechanisms; missing prices are not successful declines."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from threadpoolctl import threadpool_limits
from research.failure_first.model import (Config as BaseConfig, Calibrator,
    date_weights, fit_head, label_panel, split_masks)

METHODS = ('direct','competing','recent','rebound','persistence',
           'residual_cdf','consensus','squeeze_veto')

@dataclass(frozen=True)
class Config(BaseConfig):
    thresholds: tuple[float,...] = (.60,.70,.80,.85,.90,.95,.975)
    seed: int = 20260910


def outcome_classes(ret, benchmark, resolved, matured):
    """0 observed nondecline; 1 severe; 2 systemic; 3 other decline; 4 unknown.

    These are disjoint observable groups, not verified causes. Pending = -1.
    """
    r=np.asarray(ret,float); b=np.asarray(benchmark,float)
    known=np.asarray(resolved,bool); done=np.asarray(matured,bool)
    if any(x.shape!=r.shape for x in (b,known,done)):
        raise ValueError('inconsistent outcome shapes')
    if np.any(known & (~done | ~np.isfinite(r))):
        raise ValueError('invalid resolved outcome')
    cls=np.full(r.shape,-1,dtype=np.int8)
    cls[done]=4
    cls[done & known & (r>=0)]=0
    decline=done & known & (r<0)
    cls[decline]=3
    cls[decline & np.isfinite(b) & (b<0)]=2
    cls[decline & (r<=-.20)]=1
    return cls


def labels_for_horizon(f, prices, market, h):
    y=label_panel(f,prices,market,h).copy()
    r=y['return'].to_numpy(); known=y.resolved.to_numpy(); done=y.matured.to_numpy()
    y['class']=outcome_classes(r,y.benchmark_return,known,done)
    down=known & (r<0)
    y['down']=np.where(done,down.astype(float),np.nan)
    y['up']=np.where(done,(known & (r>0)).astype(float),np.nan)
    y['flat']=done & known & (r==0)
    y['unknown']=done & ~known
    a=prices.to_numpy(); i=f.i.to_numpy(int); j=prices.columns.get_indexer(f.ticker)
    persistent=down.copy()
    for step in (h//2,(3*h)//4):
        valid=i+step<len(prices); value=np.full(len(f),np.nan)
        value[valid]=a[i[valid]+step,j[valid]]
        persistent &= np.isfinite(value) & (value<f.reference_price.to_numpy())
    y['persistent_down']=np.where(done,persistent.astype(float),np.nan)
    y['squeeze_proxy']=np.where(done,(~y.path_complete | (y.max_return>=.20)).astype(float),np.nan)
    y['next_close_down']=np.where(done,(y.next_entry_resolved & (y.next_entry_return<0)).astype(float),np.nan)
    # Drop ambiguous legacy buy-label names, so they cannot be inverted by accident.
    return y.drop(columns=['success','failure_class','path_failure','next_entry_success'])


def classifier(cfg, **extra):
    return LGBMClassifier(n_estimators=cfg.n_estimators,num_leaves=7,max_depth=3,
        learning_rate=.05,min_child_samples=150,reg_lambda=20,max_bin=63,
        n_jobs=cfg.threads,random_state=cfg.seed,verbosity=-1,deterministic=True,
        force_col_wise=True,**extra)


class CompetingOutcomes:
    """Normalized multiclass model; unobserved outcomes never enter decline sum."""
    def fit(self,x,y,origins,asof,cfg):
        y=np.asarray(y,int)
        if not np.isin(y,np.arange(5)).all(): raise ValueError('invalid outcome class')
        w=date_weights(origins,asof,cfg.half_life)
        self.n_dates=len(np.unique(origins))
        self.prior=(np.bincount(y,weights=w,minlength=5)+.1)/(w.sum()+.5)
        self.model=None
        if len(np.unique(y))>1:
            self.model=classifier(cfg,objective='multiclass')
            self.model.fit(x,y,sample_weight=w)
        return self
    def predict(self,x):
        p=np.tile(self.prior,(len(x),1))
        if self.model is not None:
            p[:]=0
            p[:,self.model.classes_.astype(int)]=self.model.predict_proba(x)
            # A prior date avoids an unjustified zero for absent/rare classes.
            p=(p*self.n_dates+self.prior)/(self.n_dates+1)
        if not np.allclose(p.sum(axis=1),1) or (p<0).any(): raise ValueError('not normalized')
        return p


def strict_weighted_cdf(values,weights,queries):
    """Weighted P(residual < query), not <=: exact equality is not a decline."""
    v=np.asarray(values,float);w=np.asarray(weights,float);q=np.asarray(queries,float)
    if v.ndim!=1 or len(v)!=len(w) or not len(v) or not np.isfinite(v).all():
        raise ValueError('invalid CDF samples')
    if not np.isfinite(w).all() or (w<=0).any() or not np.isfinite(q).all():
        raise ValueError('invalid CDF weights/queries')
    order=np.argsort(v,kind='stable');v=v[order];w=w[order]
    total=np.r_[0,np.cumsum(w)]
    return total[np.searchsorted(v,q,side='left')]/total[-1]


def combine(direct,recent,competing):
    a=np.column_stack([direct,recent,competing])
    if not np.isfinite(a).all() or ((a<0)|(a>1)).any(): raise ValueError('invalid probabilities')
    # A selection rule, NOT proof of independence or improved calibration.
    return a.min(axis=1)


def fit_fold(f,y,cols,asof,h,cfg,test_mask,shuffle=False):
    tr,ca,start=split_masks(f,y,asof,h,cfg)
    audit={'asof_i':int(asof),'horizon':int(h),'calibration_start_i':int(start),
      'train_rows':int(tr.sum()),'calibration_rows':int(ca.sum()),
      'train_dates':int(f.loc[tr,'i'].nunique()),'calibration_dates':int(f.loc[ca,'i'].nunique()),
      'train_max_exit_i':int(y.loc[tr,'exit_i'].max()) if tr.any() else None,
      'calibration_max_exit_i':int(y.loc[ca,'exit_i'].max()) if ca.any() else None,
      'null_shuffled_training':bool(shuffle)}
    if audit['train_dates']<cfg.minimum_training_dates or audit['calibration_dates']<cfg.minimum_calibration_dates:
        audit['status']='insufficient_history'; return pd.DataFrame(),audit
    assert audit['train_max_exit_i']<start and audit['calibration_max_exit_i']<asof
    X=f[cols].to_numpy(np.float32);xt,xc,xe=X[tr],X[ca],X[test_mask]
    it=f.loc[tr,'i'].to_numpy();ic=f.loc[ca,'i'].to_numpy()
    yt=y.loc[tr].reset_index(drop=True).copy();yc=y.loc[ca].reset_index(drop=True)
    if shuffle:
        perm=np.random.default_rng(cfg.seed+asof+h).permutation(len(yt))
        # Permute all targets jointly; do not move the date or feature rows.
        yt=yt.iloc[perm].reset_index(drop=True)
    prob=lambda model,x:model.predict_proba(x)[:,1]
    def binary(target,mask=None):
        mask=np.ones(len(xt),bool) if mask is None else mask
        model=fit_head(xt[mask],yt.loc[mask,target].astype(int).to_numpy(),it[mask],asof,cfg)
        cal=Calibrator().fit(prob(model,xc),yc[target].astype(int),ic)
        return cal.predict(prob(model,xe)),model,cal
    with threadpool_limits(limits=cfg.threads):
        direct,_,_=binary('down')
        rec=it>=start-h-cfg.train_window//2
        if len(np.unique(it[rec]))<cfg.minimum_training_dates: rec[:]=True
        recent,_,_=binary('down',rec)
        model=CompetingOutcomes().fit(xt,yt['class'].to_numpy(int),it,asof,cfg)
        cc=model.predict(xc);ce=model.predict(xe)
        cal=Calibrator().fit(cc[:,1:4].sum(axis=1),yc.down.astype(int),ic)
        competing=cal.predict(ce[:,1:4].sum(axis=1))
        persistent,_,_=binary('persistent_down')
        squeeze,_,_=binary('squeeze_proxy')
        obs,_,_=binary('resolved')
        # Median + held-out residual distribution; conditional on observed endpoint.
        known=yt.resolved.to_numpy(bool)
        scale=f.loc[tr,'vol63'].to_numpy()*np.sqrt(h)
        z=np.log1p(yt['return'].to_numpy())/scale
        loc=LGBMRegressor(objective='quantile',alpha=.5,n_estimators=cfg.n_estimators,
            num_leaves=7,max_depth=3,learning_rate=.05,min_child_samples=150,
            reg_lambda=20,max_bin=63,n_jobs=cfg.threads,random_state=cfg.seed,
            verbosity=-1,deterministic=True,force_col_wise=True)
        loc.fit(xt[known],z[known],sample_weight=date_weights(it[known],asof,cfg.half_life))
        zc=np.log1p(yc['return'].to_numpy())/(f.loc[ca,'vol63'].to_numpy()*np.sqrt(h))
        kc=yc.resolved.to_numpy(bool)
        residual=zc[kc]-loc.predict(xc[kc])
        wc=date_weights(ic[kc],asof,cfg.half_life)
        # F(z<0 | endpoint observed)*P(endpoint observed); no independent-event multiplication.
        cdf=strict_weighted_cdf(residual,wc,-loc.predict(xe))*obs
    out=f.loc[test_mask,['row_id','i','date','ticker','vol63','vol63_rank','rel63_rank',
        'reference_price','regime','ma200','rel63','r21','lower_high63']].copy()
    out['horizon']=h;out['exit_i']=out.i+h;out['fit_i']=asof
    out['direct']=direct;out['recent']=recent;out['competing']=competing
    out['rebound']=direct;out['persistence']=persistent;out['residual_cdf']=cdf
    out['consensus']=combine(direct,recent,competing);out['squeeze_veto']=out.consensus
    out['gate_rebound']=(out.ma200<0)&(out.rel63<0)&(out.r21>0)&(out.lower_high63<0)
    out['gate_squeeze_veto']=squeeze<=.10
    out['squeeze_probability']=squeeze;out['observation_probability']=obs
    for k,name in enumerate(['nondecline','severe','systemic','other_down','unresolved']):
        out['class_'+name]=ce[:,k]
    audit['status']='fitted'
    audit['class_counts']={str(k):int((yt['class']==k).sum()) for k in range(5)}
    audit['recent_train_dates']=int(len(np.unique(it[rec])))
    return out.reset_index(drop=True),audit
