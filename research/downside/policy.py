"""Evaluate the whole bearish stock/threshold/horizon policy causally."""
from __future__ import annotations
import math
import numpy as np
import pandas as pd
from research.failure_first.evaluate import clean_json,cp_lower,nonoverlap,block_bootstrap
from .model import METHODS


def choose(today,method,threshold,busy):
    if method not in METHODS or not 0<threshold<1:raise ValueError('bad policy')
    if today.empty:return None
    if today.i.nunique()!=1 or today.duplicated(['ticker','horizon']).any():
        raise ValueError('expected unique forecasts at one time')
    if (today.fit_i>today.i).any() or not (today.exit_i==today.i+today.horizon).all():
        raise ValueError('noncausal fit or invalid locked endpoint')
    if (today.horizon<30).any():raise ValueError('horizon below 30')
    p=today[method]
    if not np.isfinite(p).all() or not p.between(0,1).all():raise ValueError('bad probability')
    good=p.ge(threshold)
    if method in ('rebound','squeeze_veto'):good &= today['gate_'+method].astype(bool)
    pool=today.loc[good].copy()
    if pool.empty:return None
    pool=pool.loc[[int(i)>busy.get(str(s),-1) for i,s in zip(pool.i,pool.ticker)]]
    if pool.empty:return None
    pool=pool.sort_values(['ticker','horizon']).drop_duplicates('ticker')
    r=pool.sort_values([method,'horizon','ticker'],ascending=[False,True,True]).iloc[0]
    # Whitelisted fields: no target/outcome can be consumed or propagated by selection.
    names=['row_id','i','date','ticker','horizon','exit_i','fit_i','reference_price',
        'vol63','vol63_rank','rel63_rank','regime']
    answer={k:r[k] for k in names}
    answer['estimated_down']=float(r[method]);return answer


def evidence_score(history,asof,look,total):
    if not history:return 0.
    past=pd.DataFrame(history)
    past=past.loc[past.exit_i<asof]
    past=nonoverlap(past)
    if len(past)<20:return 0.
    alpha=.05/(total*look*(look+1))
    return cp_lower(int(past.down.sum()),len(past),alpha)


def replay(predictions,outcomes,dates,cfg):
    keys=['row_id','horizon']
    if predictions.duplicated(keys).any() or outcomes.duplicated(keys).any():raise ValueError('duplicate keys')
    matches=predictions[keys+['i','exit_i']].merge(outcomes[keys+['i','exit_i']],on=keys,
        suffixes=('_p','_y'),how='left',validate='one_to_one')
    if not ((matches.i_p==matches.i_y)&(matches.exit_i_p==matches.exit_i_y)).all():
        raise ValueError('prediction/label misalignment')
    ix=outcomes.set_index(keys);bydate={i:g for i,g in predictions.groupby('i',sort=True)}
    specs=[(m,t,f'{m}@{t:g}') for m in METHODS for t in cfg.thresholds]
    busy={n:{} for _,_,n in specs};past={n:[] for _,_,n in specs};adaptive_busy={}
    picks=[];decisions=[]
    for look,i in enumerate(dates,1):
        today=bydate.get(i,pd.DataFrame());qualified=[]
        for method,t,name in specs:
            bound=evidence_score(past[name],i,look,len(specs))
            if bound>.95:qualified.append((bound,method,t,name))
            pick=choose(today,method,t,busy[name])
            decisions.append(dict(policy=name,i=int(i),issued=pick is not None,available=not today.empty))
            if pick is None:continue
            busy[name][pick['ticker']]=int(pick['exit_i'])
            # Outcomes attached only after the decision; history reads require maturity.
            y=ix.loc[(pick['row_id'],pick['horizon'])].to_dict()
            row={**pick,**y,'policy':name,'method':method,'threshold':t}
            picks.append(row);past[name].append(row)
        chosen=None
        for bound,method,t,name in sorted(qualified,key=lambda x:(-x[0],x[3])):
            chosen=choose(today,method,t,adaptive_busy)
            if chosen is not None:break
        decisions.append(dict(policy='adaptive95',i=int(i),issued=chosen is not None,available=not today.empty))
        if chosen is not None:
            adaptive_busy[chosen['ticker']]=int(chosen['exit_i'])
            y=ix.loc[(chosen['row_id'],chosen['horizon'])].to_dict()
            picks.append({**chosen,**y,'policy':'adaptive95','method':method,'threshold':t,
                          'evidence_lower_diagnostic':bound})
    return pd.DataFrame(picks),pd.DataFrame(decisions)


def controls(picks,y,f):
    if picks.empty:return picks
    meta=f[['row_id','vol63_rank','rel63_rank','ticker']].copy()
    meta['vol_bin']=np.minimum((meta.vol63_rank*5).astype(int),4)
    meta['rel_bin']=np.minimum((meta.rel63_rank*5).astype(int),4)
    d=y.merge(meta,on='row_id',validate='many_to_one')
    random=d.groupby(['i','horizon']).down.mean().to_dict()
    peer=d.groupby(['i','horizon','vol_bin','rel_bin']).down.mean().to_dict()
    weak=(d.sort_values(['i','horizon','rel63_rank','ticker'])
        .drop_duplicates(['i','horizon']).set_index(['i','horizon']).down.to_dict())
    p=picks.copy()
    p['random_expected']=[random[(r.i,r.horizon)] for r in p.itertuples()]
    p['peer_expected']=[peer[(r.i,r.horizon,min(int(r.vol63_rank*5),4),min(int(r.rel63_rank*5),4))]
                       for r in p.itertuples()]
    p['weakest_momentum_down']=[weak[(r.i,r.horizon)] for r in p.itertuples()]
    p['benchmark_down']=np.where(p.matured,(p.benchmark_return<0).astype(float),np.nan)
    return p


def overlap_components(g):
    if g.empty:return 0
    n=0;end=-1
    for row in g.sort_values(['i','exit_i']).itertuples():
        if row.i>end:n+=1
        end=max(end,int(row.exit_i))
    return n


def metrics(picks,decisions,cutoff):
    out=[]
    for name,ds in decisions.groupby('policy',sort=True):
        g=picks.loc[picks.policy==name].copy() if len(picks) else pd.DataFrame()
        d=g.loc[g.exit_i<=cutoff].copy() if len(g) else pd.DataFrame()
        n=len(d);wins=int(d.down.sum()) if n else 0;unknown=int(d.unknown.sum()) if n else 0
        independent=nonoverlap(d);longest=run=0
        for v in ds.issued:
            run=0 if v else run+1;longest=max(longest,run)
        r=dict(policy=name,issued=len(g),matured=n,pending=len(g)-n,wins=wins,
            failed_verified_predictions=n-wins,unknown=unknown,
            precision=wins/n if n else None,precision_if_all_unknown_down=(wins+unknown)/n if n else None,
            resolved_precision=wins/(n-unknown) if n>unknown else None,
            active_fraction=float(ds.issued.mean()),decision_dates=len(ds),
            longest_no_pick_decisions=longest,model_available_dates=int(ds.available.sum()),
            unique_tickers=int(g.ticker.nunique()) if len(g) else 0,nonoverlap_n=len(independent),
            overlap_components=overlap_components(d),production_certified=False)
        if n:
            r.update(nonoverlap_precision=float(independent.down.mean()) if len(independent) else None,
                nonoverlap_iid_lower95=cp_lower(int(independent.down.sum()),len(independent)) if len(independent) else None,
                naive_iid_lower95=cp_lower(wins,n),random_expected=float(d.random_expected.mean()),
                peer_expected=float(d.peer_expected.mean()),weakest_momentum=float(d.weakest_momentum_down.mean()),
                benchmark_down=float(d.benchmark_down.mean()),next_close_precision=float(d.next_close_down.mean()),
                margin20bp_precision=float((d['return'].fillna(0)<-.002).mean()),
                margin50bp_precision=float((d['return'].fillna(0)<-.005).mean()),
                median_observed_return=float(d['return'].median()),largest_observed_rise=float(d['return'].max()),
                largest_complete_path_rise=float(d.max_return.max()),
                observed_up=int(d.up.sum()),observed_flat=int(d.flat.sum()),
                estimated_down_mean=float(d.estimated_down.mean()),unique_years=int(pd.to_datetime(d.date).dt.year.nunique()),
                horizons={str(k):int(v) for k,v in d.horizon.value_counts().sort_index().items()})
            d['success']=d.down;d['excess']=d.down-d.random_expected;d['peer_excess']=d.down-d.peer_expected
            r['precision_block_ci']=block_bootstrap(d,ds.i.to_numpy(),reps=500)
            r['excess_block_ci']=block_bootstrap(d,ds.i.to_numpy(),value_col='excess',reps=500)
            r['peer_excess_block_ci']=block_bootstrap(d,ds.i.to_numpy(),value_col='peer_excess',reps=500)
            r['by_year']={str(year):{'n':len(t),'wins':int(t.down.sum()),'precision':float(t.down.mean()),
                'random':float(t.random_expected.mean()),'peer':float(t.peer_expected.mean())}
                for year,t in d.groupby(pd.to_datetime(d.date).dt.year)}
            q=d.loc[d.date>='2024-01-01']
            r['since2024']={'n':len(q),'wins':int(q.down.sum()),'precision':float(q.down.mean()) if len(q) else None}
        out.append(r)
    return clean_json(out)


def calibration(preds,y):
    d=preds.merge(y[['row_id','horizon','matured','down','persistent_down']],on=['row_id','horizon'],validate='one_to_one')
    d=d.loc[d.matured];out=[]
    for h,q in d.groupby('horizon'):
        for method in METHODS:
            p=q[method].to_numpy();truth=q.down.to_numpy();bins=[]
            for lo,hi in zip([0,.5,.6,.7,.8,.85,.9,.95,.975],[.5,.6,.7,.8,.85,.9,.95,.975,1.00001]):
                m=(p>=lo)&(p<hi)
                bins.append(dict(lower=lo,upper=min(hi,1),n=int(m.sum()),predicted=float(p[m].mean()) if m.any() else None,
                                 observed=float(truth[m].mean()) if m.any() else None))
            out.append(dict(horizon=int(h),method=method,n=len(q),base_rate=float(truth.mean()),
                 brier=float(np.mean((p-truth)**2)),max_estimate=float(p.max()),bins=bins,
                 note='persistence targets a stricter joint event; gating arms share underlying estimates'))
    return clean_json(out)
