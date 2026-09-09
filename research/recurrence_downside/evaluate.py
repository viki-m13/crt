"""Complete adaptive bearish policies, conservative missing outcomes, paired controls."""
from __future__ import annotations
import math
import numpy as np
import pandas as pd
from research.failure_first.evaluate import clean_json,nonoverlap,cp_lower,block_bootstrap
from .engine import METHODS,candidate_rows


def choose(today,method,threshold,busy):
    if method not in METHODS or not 0<threshold<1:raise ValueError('invalid policy')
    if today.empty:return None
    if today.i.nunique()!=1 or today.duplicated(['ticker','horizon']).any():raise ValueError('bad decision frame')
    if (today.fit_i>today.i).any() or (today.horizon<30).any() or (today.exit_i!=today.i+today.horizon).any():
        raise ValueError('future fit or invalid deadline')
    scores=today[method]
    if ((scores.dropna()<0)|(scores.dropna()>1)).any():raise ValueError('invalid score')
    pool=today.loc[np.isfinite(scores)&(scores>threshold)].copy()
    pool=pool.loc[[i>busy.get(s,-1) for i,s in zip(pool.i,pool.ticker)]]
    if pool.empty:return None
    pool=pool.sort_values(['ticker','horizon']).drop_duplicates('ticker')
    r=pool.sort_values([method,'horizon','ticker'],ascending=[False,True,True]).iloc[0].to_dict()
    r['estimated_success']=float(r[method]);return r


def evidence_choice(histories,asof,look):
    """Conservative diagnostic under IID-like assumptions; never a live certificate."""
    best=None;alpha=.05/(max(1,len(histories))*look*(look+1))
    for policy,hist in histories.items():
        if len(hist)<20:continue
        past=pd.DataFrame([r for r in hist if r['exit_i']<asof])
        if past.empty:continue
        independent=nonoverlap(past)
        if len(independent)<20:continue
        lower=cp_lower(int(independent.success.sum()),len(independent),alpha)
        if lower>.95:
            key=(lower,len(independent),policy)
            if best is None or key>best[0]:best=(key,policy)
    return best[1] if best else None


def policy_replay(pred,outcomes,dates,cfg):
    keys=['row_id','horizon']
    if pred.duplicated(keys).any() or outcomes.duplicated(keys).any():raise ValueError('duplicate forecast keys')
    indexed=outcomes.set_index(keys)
    specs=[(m,t,f'{m}@{t:g}') for m in METHODS for t in cfg.thresholds]
    busy={s:{} for _,_,s in specs};hist={s:[] for _,_,s in specs};busy_strict={}
    groups={int(i):g for i,g in pred.groupby('i',sort=True)}
    issued=[];decisions=[]
    for look,i in enumerate(dates,1):
        today=groups.get(int(i),pd.DataFrame())
        for method,t,name in specs:
            r=choose(today,method,t,busy[name])
            decisions.append({'i':int(i),'policy':name,'issued':r is not None,'model_available':not today.empty})
            if r is not None:
                # Expose outcomes only after the immutable decision has been taken.
                result=indexed.loc[(r['row_id'],r['horizon'])].to_dict()
                if int(result['i'])!=i or int(result['exit_i'])!=int(r['exit_i']):raise ValueError('label clock mismatch')
                row={**r,**result,'method':method,'threshold':t,'policy':name}
                busy[name][r['ticker']]=int(r['exit_i']);issued.append(row);hist[name].append(row)
        selected=evidence_choice(hist,int(i),look)
        r=None
        if selected:
            method,t=selected.split('@');r=choose(today,method,float(t),busy_strict)
        decisions.append({'i':int(i),'policy':'adaptive95','issued':r is not None,'model_available':not today.empty})
        if r is not None:
            row={**r,**indexed.loc[(r['row_id'],r['horizon'])].to_dict(),
                 'method':method,'threshold':float(t),'policy':'adaptive95','source_policy':selected}
            busy_strict[r['ticker']]=int(r['exit_i']);issued.append(row)
    columns=list(dict.fromkeys([*pred.columns,*outcomes.columns,'method','threshold','policy','estimated_success']))
    output=pd.DataFrame(issued) if issued else pd.DataFrame(columns=columns)
    return output,pd.DataFrame(decisions)


def controls(picks,outcomes,f,cfg):
    if picks.empty:return picks
    meta=f[['row_id','ticker','vol63_rank','rel63_rank']]
    d=outcomes[['row_id','i','horizon','success']].merge(meta,on='row_id',validate='many_to_one')
    ids=set(candidate_rows(f,cfg.candidate_k).row_id)
    d['vol_bin']=np.minimum((d.vol63_rank.astype(np.float64)*10).astype(int),9)
    d['rel_bin']=np.minimum((d.rel63_rank.astype(np.float64)*5).astype(int),4)
    all_=d.groupby(['i','horizon']).success.mean().to_dict()
    cand=d[d.row_id.isin(ids)].groupby(['i','horizon']).success.mean().to_dict()
    vol=d.groupby(['i','horizon','vol_bin']).success.mean().to_dict()
    peer=d.groupby(['i','horizon','vol_bin','rel_bin']).success.mean().to_dict()
    out=picks.copy()
    out['random_expected']=[all_[(r.i,r.horizon)] for r in out.itertuples()]
    out['candidate_expected']=[cand[(r.i,r.horizon)] for r in out.itertuples()]
    out['volmatched_expected']=[vol[(r.i,r.horizon,min(int(r.vol63_rank*10),9))] for r in out.itertuples()]
    out['peer_expected']=[peer[(r.i,r.horizon,min(int(r.vol63_rank*10),9),min(int(r.rel63_rank*5),4))] for r in out.itertuples()]
    return out


def episodes(g):
    if g.empty:return 0
    end=-1; n=0
    for r in g.sort_values(['i','exit_i']).itertuples():
        if r.i>end:n+=1
        end=max(end,int(r.exit_i))
    return n


def summary(picks,decisions):
    report=[]
    for name,ds in decisions.groupby('policy',sort=True):
        g=picks[picks.policy==name].copy() if not picks.empty else pd.DataFrame()
        done=g[g.matured].copy() if len(g) else pd.DataFrame()
        n=len(done);wins=int(done.success.sum()) if n else 0;ind=nonoverlap(done)
        gap=longest=0
        for b in ds.sort_values('i').issued:
            gap=0 if b else gap+1;longest=max(longest,gap)
        row={'policy':name,'issued':len(g),'matured':n,'wins':wins,'losses':n-wins,
             'pending':len(g)-n,'precision':wins/n if n else None,
             'active_date_fraction':float(ds.issued.mean()),'decision_dates':len(ds),
             'longest_no_pick_decisions':longest,'nonoverlap_n':len(ind),'overlap_episodes':episodes(done),
             'production_certified':False,'random_expected':None,'candidate_expected':None}
        if n:
            unresolved=int((~done.resolved).sum())
            row.update(unresolved=unresolved,flat=int((done.outcome_kind=='flat').sum()),
                       observed_only_precision=float(done.loc[done.resolved,'success'].mean()),
                       missing_outcome_bounds=[wins/n,(wins+unresolved)/n],
                       unique_tickers=int(done.ticker.nunique()),unique_years=int(pd.to_datetime(done.date).dt.year.nunique()),
                       random_expected=float(done.random_expected.mean()),candidate_expected=float(done.candidate_expected.mean()),
                       volmatched_expected=float(done.volmatched_expected.mean()),peer_expected=float(done.peer_expected.mean()),
                       next_close_precision=float(done.next_entry_success.mean()),
                       median_endpoint_price_change=float(done['return'].median()),
                       mean_endpoint_price_change=float(done['return'].mean()),
                       largest_endpoint_rise=float(done['return'].max()),
                       largest_interim_rise=float(done.max_return.max()),
                       incomplete_paths=int((~done.path_complete).sum()),
                       nonoverlap_precision=float(ind.success.mean()) if len(ind) else None,
                       nonoverlap_iid_lower95=cp_lower(int(ind.success.sum()),len(ind)) if len(ind) else None,
                       mean_score=float(done.estimated_success.mean()),
                       horizons={str(int(k)):int(v) for k,v in done.horizon.value_counts().items()})
            for bps in (20,50,500,1000):
                row[f'margin_{bps}bp_precision']=float((done.resolved&(done['return'] < -bps/10000)).mean())
            dates=ds.i.to_numpy(int)
            done['excess']=done.success-done.random_expected
            done['candidate_excess']=done.success-done.candidate_expected
            row['block_precision']=block_bootstrap(done,dates,reps=1000)
            row['block_excess']=block_bootstrap(done,dates,'excess',reps=1000)
            row['block_candidate_excess']=block_bootstrap(done,dates,'candidate_excess',reps=1000)
            row['years']={str(year):{'n':len(part),'wins':int(part.success.sum()),
                        'precision':float(part.success.mean()),'random':float(part.random_expected.mean())}
                        for year,part in done.groupby(pd.to_datetime(done.date).dt.year)}
            late=done[done.date>='2024-01-01']
            row['since2024']={'n':len(late),'wins':int(late.success.sum()),
                             'precision':float(late.success.mean()) if len(late) else None}
        report.append(row)
    return report
