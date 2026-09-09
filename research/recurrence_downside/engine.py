"""Price-state recurrence with per-date support and cross-era bearish agreement.

Raw neighborhood estimates and minima of scores are NOT certified probabilities.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from threadpoolctl import threadpool_limits
from research.failure_first.model import Config as BaseConfig, label_panel, split_masks, fit_head, Calibrator

METHODS = ('direct','recurrence','regime_recurrence','cross_era','consensus',
           'confirmed_recovery','robust_consensus')
DISTANCE_COLS = ('r5','r21','r63','r126','r252','vol63','vol_ratio','dd','ma50',
                'ma200','rel21_rank','rel63_rank','rel126_rank','semivol_ratio',
                'trend_accel','breadth','market21','market63','market200','market_vol63')

@dataclass(frozen=True)
class Config(BaseConfig):
    thresholds: tuple[float,...] = (.60,.70,.80,.85,.90,.95,.975)
    neighbors: int = 128
    search_neighbors: int = 512
    min_neighbors: int = 32
    min_dates: int = 12
    candidate_k: int = 5
    def __post_init__(self):
        super().__post_init__()
        if not 0 < self.min_dates <= self.min_neighbors <= self.neighbors <= self.search_neighbors:
            raise ValueError('Invalid recurrence support')
        if self.candidate_k < 1: raise ValueError('candidate_k must be positive')


def recovery_state(f):
    """Frozen ex-ante failed-recovery hypothesis; no labels in the predicate."""
    return (f.ma200 < 0) & (f.rel63 < 0) & (f.r21 > 0) & (f.r5 < 0)


def candidate_rows(f, k=5):
    """Union of three top-k lists chosen without future labels."""
    if k < 1: raise ValueError('k must be positive')
    if f.duplicated(['i','ticker']).any(): raise ValueError('duplicate feature coordinates')
    weak=f.sort_values(['i','rel63','ticker']).groupby('i',sort=False).head(k)
    accel=f.sort_values(['i','trend_accel','ticker']).groupby('i',sort=False).head(k)
    failed=f.loc[recovery_state(f)].sort_values(['i','rel63','ticker']).groupby('i',sort=False).head(k)
    return pd.concat([weak,accel,failed]).drop_duplicates('row_id').sort_values(['i','ticker']).copy()


def downside_labels(f,p,market,h):
    y=label_panel(f,p,market,h)
    # Never complement the buy label: flat and unobserved are not bearish wins.
    y['success']=np.where(y.matured, (y.resolved & (y['return']<0)).astype(float),np.nan)
    y['next_entry_success']=np.where(y.matured,
        (y.next_entry_resolved & (y.next_entry_return<0)).astype(float),np.nan)
    y['outcome_kind']=np.select([~y.matured,~y.resolved,y['return']<0,y['return']==0],
                               ['pending','unobserved','decline','flat'],default='rise')
    return y


def within_date_permutation(origins,seed):
    """Stock-selection null: preserve each date's market-wide outcome counts."""
    i=np.asarray(origins); perm=np.arange(len(i)); rng=np.random.default_rng(seed)
    for d in np.unique(i):
        ix=np.flatnonzero(i==d); perm[ix]=rng.permutation(ix)
    return perm


def state_normalization(x):
    x=np.asarray(x,float)
    if x.ndim!=2 or len(x)==0 or not np.isfinite(x).all(): raise ValueError('invalid reference states')
    med=np.median(x,axis=0); q=np.quantile(x,[.25,.75],axis=0)
    return med,np.maximum(q[1]-q[0],1e-6)


def transform(x,med,scale):
    return np.clip((np.asarray(x,float)-med)/scale,-8,8)


def recurrence_estimate(outcomes,origins,asof,cfg):
    y=np.asarray(outcomes,float); i=np.asarray(origins,int)
    if len(y)!=len(i) or not np.isfinite(y).all() or not np.isin(y,[0,1]).all():
        raise ValueError('recurrence needs confirmed matured 0/1 outcomes')
    if len(y)<cfg.min_neighbors or len(np.unique(i))<cfg.min_dates:
        return np.nan, len(np.unique(i)),0.
    _,ix,count=np.unique(i,return_inverse=True,return_counts=True)
    w=np.exp2(-(asof-i)/cfg.half_life)/count[ix]
    date_w=np.bincount(ix,weights=w)
    neff=float(date_w.sum()**2/(date_w@date_w))
    w=w/w.sum()*neff
    # Two pseudo-dates centered at 50%; never turn unobserved failures into zero risk.
    return float((w@y+1)/(neff+2)),len(count),neff


def neighbor_scores(neighbors,ref,f,labels,query,asof,h,cfg):
    """Filter by label maturity before reading labels; rows arrive in distance order."""
    ref_i=ref.i.to_numpy(int); ref_ids=ref.row_id.to_numpy(int)
    valid=(neighbors>=0)&(neighbors<len(ref))
    nn=np.asarray(neighbors,int)[valid]
    nn=nn[ref_i[nn]+h < asof]
    ids=ref_ids[nn]
    if len(ids) and not labels.loc[ids,'matured'].all(): raise ValueError('unmatured neighbor')
    all_ys=labels.success.to_numpy(float)
    old=ref_i[nn] < asof-1260
    regime=(ref.market200.to_numpy()[nn]>0)==bool(query['market200']>0)
    def score(mask):
        chosen=nn[mask][:cfg.neighbors]
        return recurrence_estimate(all_ys[ref_ids[chosen]],ref_i[chosen],asof,cfg)
    p_all,n_all,e_all=score(np.ones(len(nn),bool))
    p_reg,n_reg,e_reg=score(regime)
    p_old,n_old,_=score(regime&old); p_recent,n_recent,_=score(regime&~old)
    cross=float(min(p_reg,p_old,p_recent)) if np.isfinite([p_reg,p_old,p_recent]).all() else np.nan
    return {'recurrence':p_all,'regime_recurrence':p_reg,'cross_era':cross,
            'neighbor_dates':n_all,'effective_neighbor_dates':e_all,
            'regime_dates':n_reg,'old_era_dates':n_old,'recent_era_dates':n_recent}


def reference_neighbors(f,candidates,asof,cfg):
    mask=(f.i>=asof-cfg.train_window)&(f.i<asof-30)&((f.i-252)%cfg.train_every==0)
    ref=f.loc[mask].reset_index(drop=True)
    if len(ref)<cfg.min_neighbors: return ref,None,{}
    med,scale=state_normalization(ref[list(DISTANCE_COLS)].to_numpy())
    x=transform(ref[list(DISTANCE_COLS)],med,scale)
    q=transform(candidates[list(DISTANCE_COLS)],med,scale)
    tree=cKDTree(x)
    dist,ix=tree.query(q,k=min(cfg.search_neighbors,len(ref)),workers=cfg.threads)
    if ix.ndim==1: ix=ix[:,None]
    # cKDTree's equal-distance ties resolved deterministically by original reference index.
    order=np.lexsort((ix,dist),axis=1); ix=np.take_along_axis(ix,order,axis=1)
    return ref,ix,{'reference_rows':len(ref),'reference_dates':int(ref.i.nunique()),
                  'reference_max_i':int(ref.i.max()),'normalization_median':med.tolist(),
                  'normalization_iqr':scale.tolist()}


def direct_predictions(f,y,cols,asof,h,cfg,test_mask,shuffle=False):
    tr,ca,start=split_masks(f,y,asof,h,cfg)
    audit={'fit_i':asof,'horizon':h,'train_rows':int(tr.sum()),'calibration_rows':int(ca.sum()),
           'train_dates':int(f.loc[tr,'i'].nunique()),'calibration_dates':int(f.loc[ca,'i'].nunique()),
           'train_max_exit_i':int(y.loc[tr,'exit_i'].max()) if tr.any() else None,
           'calibration_start_i':int(start),
           'calibration_max_exit_i':int(y.loc[ca,'exit_i'].max()) if ca.any() else None}
    if audit['train_dates']<cfg.minimum_training_dates or audit['calibration_dates']<cfg.minimum_calibration_dates:
        audit['status']='unsupported'; return np.full(int(test_mask.sum()),np.nan),audit
    assert audit['train_max_exit_i'] < start and audit['calibration_max_exit_i'] < asof
    x=f[cols].to_numpy(np.float32); yt=y.loc[tr,'success'].to_numpy(int)
    it=f.loc[tr,'i'].to_numpy(); ic=f.loc[ca,'i'].to_numpy()
    if shuffle:yt=yt[within_date_permutation(it,cfg.seed+asof+h)]
    with threadpool_limits(limits=cfg.threads):
        head=fit_head(x[tr],yt,it,asof,cfg)
        pcal=head.predict_proba(x[ca])[:,1]
        cal=Calibrator().fit(pcal,y.loc[ca,'success'].to_numpy(int),ic)
        pe=cal.predict(head.predict_proba(x[test_mask])[:,1])
    audit['status']='fitted';return pe,audit


def annual_predictions(f,y_by_h,cols,asof,until,cfg,shuffle=False):
    if not np.array_equal(f.row_id.to_numpy(),np.arange(len(f))): raise ValueError('row_id must align to range index')
    c=candidate_rows(f,cfg.candidate_k)
    c=c.loc[(c.i>=asof)&(c.i<until)].copy()
    if c.empty:return pd.DataFrame(),[]
    test_mask=f.row_id.isin(c.row_id).to_numpy()
    # f and c both sort by (i,ticker), so direct and recurrence query rows are aligned.
    assert np.array_equal(f.loc[test_mask,'row_id'],c.row_id)
    ref,nn,ref_meta=reference_neighbors(f,c,asof,cfg)
    chunks=[];audits=[]
    for h,y in y_by_h.items():
        direct,audit=direct_predictions(f,y,cols,asof,h,cfg,test_mask,shuffle)
        out=c[['row_id','i','date','ticker','vol63','vol63_rank','rel63_rank','reference_price','market200']].copy()
        out['horizon']=h;out['exit_i']=out.i+h;out['fit_i']=asof
        use_y=y
        if shuffle:
            use_y=y.copy()
            known=(use_y.exit_i<asof)&use_y.matured
            idx=np.flatnonzero(known)
            perm=within_date_permutation(f.loc[known,'i'].to_numpy(),cfg.seed+asof+h)
            use_y.loc[idx,'success']=y.loc[idx[perm],'success'].to_numpy()
        ns=[]
        for j,q in enumerate(c.to_dict('records')):
            ns.append(neighbor_scores(nn[j],ref,f,use_y,q,asof,h,cfg) if nn is not None else
                      {'recurrence':np.nan,'regime_recurrence':np.nan,'cross_era':np.nan})
        ndf=pd.DataFrame(ns,index=out.index)
        for k in ndf:out[k]=ndf[k]
        out['direct']=direct
        out['consensus']=np.minimum(out.direct,out.regime_recurrence)
        out['confirmed_recovery']=out.consensus.where(recovery_state(c))
        out['robust_consensus']=np.minimum(out.direct,out.cross_era)
        audit.update(ref_meta);audit['predictions']=len(out);audit['null_within_date']=shuffle
        chunks.append(out.reset_index(drop=True));audits.append(audit)
    return pd.concat(chunks,ignore_index=True),audits
