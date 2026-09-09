"""Evaluate a complete stock/threshold/horizon policy, never cherry-picked buckets."""
from __future__ import annotations
import argparse,hashlib,json,math
from pathlib import Path
from functools import lru_cache
import numpy as np
import pandas as pd
from scipy.stats import beta
from .model import Config,METHODS


def clean_json(x):
    if isinstance(x,dict): return {str(k):clean_json(v) for k,v in x.items()}
    if isinstance(x,(list,tuple,np.ndarray)): return [clean_json(v) for v in x]
    if isinstance(x,(np.bool_,)): return bool(x)
    if isinstance(x,(np.integer,)): return int(x)
    if isinstance(x,(float,np.floating)): return float(x) if np.isfinite(x) else None
    return x


def select(today, method, threshold, busy):
    """This function sees estimates and current metadata, NEVER any future outcome."""
    if not 0<threshold<1: raise ValueError('invalid threshold')
    if method not in METHODS: raise ValueError('unknown method')
    if today.empty: return None
    if today.i.nunique()!=1: raise ValueError('expected one issue timestamp')
    if today.duplicated(['ticker','horizon']).any(): raise ValueError('duplicate predictions')
    if not (today.fit_i<=today.i).all(): raise ValueError('model from future')
    if not (today.exit_i==today.i+today.horizon).all() or (today.horizon<30).any():
        raise ValueError('invalid locked deadline')
    if not np.isfinite(today[method]).all() or not today[method].between(0,1).all():
        raise ValueError('invalid risk estimates')
    pool=today.loc[today[method] < 1-threshold].copy()
    if pool.empty: return None
    pool=pool.loc[[int(i)>busy.get(str(s),-1) for i,s in zip(pool.i,pool.ticker)]]
    if pool.empty: return None
    # Shortest horizon BEFORE comparing stock risk. Deterministic ordering.
    pool=pool.sort_values(['ticker','horizon',method]).drop_duplicates('ticker')
    pick=pool.sort_values([method,'horizon','ticker']).iloc[0].to_dict()
    pick['estimated_success']=1-pick[method]
    return pick


def nonoverlap(rows,embargo=5):
    if rows.empty: return rows.copy()
    if embargo<0: raise ValueError('negative embargo')
    ordered=rows.sort_values(['i','ticker','horizon']); keep=[]; end=-1
    for ix,r in ordered.iterrows():
        if r.i>end: keep.append(ix);end=int(r.exit_i)+embargo
    return ordered.loc[keep].copy()


def cp_lower(wins,n,alpha=.05):
    if not isinstance(wins,(int,np.integer)) or not 0<=wins<=n or not 0<alpha<1:
        raise ValueError('invalid binomial counts')
    return float(beta.ppf(alpha,wins,n-wins+1)) if wins else 0.


def historical_screen(past,asof,look):
    """Conditional diagnostic, not a distribution-free certificate for stock returns."""
    if past.empty: return False
    done=past.loc[(past.exit_i<asof)&(past.i<asof)].copy()
    done=nonoverlap(done)
    if len(done)<20: return False
    # 5 methods x 6 thresholds, plus repeated-look union accounting.
    alpha=.05/(30*max(1,look)*(max(1,look)+1))
    return cp_lower(int(done.success.sum()),len(done),alpha)>.95


def joint_policy(risks, outcomes, decision_dates, cfg):
    if risks.duplicated(['row_id','horizon']).any(): raise ValueError('duplicate risk keys')
    if outcomes.duplicated(['row_id','horizon']).any(): raise ValueError('duplicate outcome keys')
    alignment=risks[['row_id','horizon','i','exit_i']].merge(
        outcomes[['row_id','horizon','i','exit_i']],on=['row_id','horizon'],
        how='left',suffixes=('_risk','_outcome'),validate='one_to_one')
    if not ((alignment.i_risk==alignment.i_outcome)&(alignment.exit_i_risk==alignment.exit_i_outcome)).all():
        raise ValueError('prediction/outcome timestamps do not align')
    del alignment
    indexed=outcomes.set_index(['row_id','horizon'])
    by_issue=risks.sort_values(['i','ticker','horizon']).set_index('i',drop=False)
    available=set(risks.i)
    specs=[(m,t) for m in METHODS for t in cfg.thresholds]
    busy={f'{m}@{t:g}':{} for m,t in specs}; strict_busy={}
    histories={name:[] for name in busy}; output=[]; decisions=[]
    for look,i in enumerate(decision_dates,1):
        today=by_issue.loc[[i]].reset_index(drop=True) if i in available else pd.DataFrame()
        for method,t in specs:
            name=f'{method}@{t:g}'
            pick=select(today,method,t,busy[name])
            decisions.append({'policy':name,'i':i,'issued':pick is not None,'model_available':not today.empty})
            if pick is None: continue
            busy[name][pick['ticker']]=int(pick['exit_i'])
            # Join AFTER selection. No label column can enter selection.
            result=indexed.loc[(pick['row_id'],pick['horizon'])].to_dict()
            row={**pick,**result,'policy':name,'method':method,'threshold':t}
            output.append(row); histories[name].append(row)
        # Selective evidence policy uses only earlier matured virtual .95 decisions.
        past=pd.DataFrame(histories['failure_veto@0.95'])
        allow=historical_screen(past,i,look)
        pick=select(today,'failure_veto',.95,strict_busy) if allow else None
        decisions.append({'policy':'evidence_95','i':i,'issued':pick is not None,'model_available':not today.empty})
        if pick is not None:
            strict_busy[pick['ticker']]=int(pick['exit_i'])
            result=indexed.loc[(pick['row_id'],pick['horizon'])].to_dict()
            output.append({**pick,**result,'policy':'evidence_95','method':'failure_veto','threshold':.95})
    return pd.DataFrame(output),pd.DataFrame(decisions)


def add_controls(picks,outcomes,f):
    """Exact expected random hit rate from same-date eligible stocks, no seed lottery."""
    if picks.empty:return picks
    meta=f[['row_id','ticker','vol63','vol63_rank']]
    o=outcomes[['row_id','i','horizon','success']].merge(meta,on='row_id',how='left',validate='many_to_one')
    o['vol_bin']=np.minimum((o.vol63_rank*10).astype(int),9)
    overall=o.groupby(['i','horizon']).success.mean().to_dict()
    vols=o.groupby(['i','horizon','vol_bin']).success.mean().to_dict()
    low=o.sort_values(['i','horizon','vol63','ticker']).drop_duplicates(['i','horizon'])
    low=low.set_index(['i','horizon']).success.to_dict()
    out=picks.copy()
    out['random_expected']=[overall[(int(r.i),int(r.horizon))] for r in out.itertuples()]
    out['volmatched_expected']=[vols[(int(r.i),int(r.horizon),min(int(r.vol63_rank*10),9))] for r in out.itertuples()]
    out['lowvol_success']=[low[(int(r.i),int(r.horizon))] for r in out.itertuples()]
    return out


def block_bootstrap(done,dates,value_col='success',reps=2000,seed=20260909):
    """Moving issue-time blocks of at least the maximum horizon. Conditional diagnostic.

    Empty decision dates remain zero-count dates. Resample complete time blocks;
    never declare thousands of correlated stock/horizon rows independent.
    """
    if done.empty or len(dates)<2: return None
    dates=np.asarray(dates,int); index={int(x):k for k,x in enumerate(dates)}
    series=np.zeros(len(dates)); count=np.zeros(len(dates))
    for r in done.itertuples():
        k=index[int(r.i)]; series[k]+=float(getattr(r,value_col));count[k]+=1
    stride=int(np.median(np.diff(dates)))
    length=max(1,int(math.ceil(done.horizon.max()/stride)))
    length=min(length,len(dates))
    number=int(math.ceil(len(dates)/length))
    rng=np.random.default_rng(seed);values=[]
    for _ in range(reps):
        starts=rng.integers(0,len(dates)-length+1,size=number)
        loc=np.concatenate([np.arange(s,s+length) for s in starts])[:len(dates)]
        n=count[loc].sum()
        if n: values.append(series[loc].sum()/n)
    if not values:return None
    return {'ci95':np.quantile(values,[.025,.975]).tolist(),'resamples':len(values),
            'block_length_decisions':length,'rough_time_blocks':len(dates)/length,
            'interpretation':'conditional moving-block diagnostic, not certified future reliability'}


def summarize(picks,decisions,cutoff):
    summaries=[]
    for policy,ds in decisions.groupby('policy',sort=True):
        g=picks.loc[picks.policy==policy] if not picks.empty else pd.DataFrame()
        done=g.loc[g.exit_i<=cutoff].copy() if len(g) else pd.DataFrame()
        n=len(done); wins=int(done.success.sum()) if n else 0
        ind=nonoverlap(done); longest=run=0
        for emitted in ds.issued:
            run=0 if emitted else run+1;longest=max(longest,run)
        d={'policy':policy,'issued':len(g),'matured':n,'pending':len(g)-n,'wins':wins,'losses':n-wins,
           'unresolved':int((~done.resolved).sum()) if n else 0,'precision':wins/n if n else None,
           'observed_only_precision':float(done.loc[done.resolved,'success'].mean()) if n else None,
           'decision_dates':len(ds),'model_available_dates':int(ds.model_available.sum()),
           'active_date_fraction':float(ds.issued.mean()),'longest_no_pick_decisions':longest,
           'unique_tickers':int(g.ticker.nunique()) if len(g) else 0,
           'nonoverlap_n':len(ind),'nonoverlap_precision':float(ind.success.mean()) if len(ind) else None,
           'nonoverlap_iid_lower95':cp_lower(int(ind.success.sum()),len(ind)) if len(ind) else None,
           'production_certified':False}
        if n:
            for key,col in [('random_expected_precision','random_expected'),('volmatched_expected_precision','volmatched_expected'),
                            ('lowvol_precision','lowvol_success'),('next_close_entry_precision','next_entry_success')]:
                d[key]=float(done[col].mean())
            d['matched_excess']=d['precision']-d['random_expected_precision']
            d['margin_20bp_precision']=float((done['return'].fillna(-1)>.002).mean())
            d['margin_50bp_precision']=float((done['return'].fillna(-1)>.005).mean())
            d['median_observed_return']=float(done['return'].median())
            d['mean_return_missing_minus100']=float(done['return'].fillna(-1).mean())
            d['worst_observed_return']=float(done['return'].min())
            d['worst_complete_path_return']=float(done.min_return.min())
            d['median_complete_path_min_return']=float(done.min_return.median())
            d['predicted_success_mean']=float(done.estimated_success.mean())
            d['horizon_distribution']={str(k):int(v) for k,v in done.horizon.value_counts().items()}
            d['most_common_ticker_fraction']=float(done.ticker.value_counts(normalize=True).max())
            done['excess']=done.success-done.random_expected
            dates=ds.i.to_numpy(int)
            d['precision_block_bootstrap']=block_bootstrap(done,dates)
            d['excess_block_bootstrap']=block_bootstrap(done,dates,'excess')
            d['eras']={str(k):{'matured':len(s),'wins':int(s.success.sum()),'precision':float(s.success.mean()),
                               'random_expected':float(s.random_expected.mean())}
                     for k,s in done.groupby(pd.to_datetime(done.date).dt.year//3*3)}
            late=done.loc[done.date>='2024-01-01']
            d['since2024']={'matured':len(late),'wins':int(late.success.sum()),
                            'precision':float(late.success.mean()) if len(late) else None}
        summaries.append(d)
    return summaries


def reliability(risks,outcomes):
    keys=['row_id','horizon']; d=risks.merge(outcomes[keys+['success','matured']],on=keys,validate='one_to_one')
    d=d.loc[d.matured]; rows=[]
    for h,part in d.groupby('horizon'):
        y=1-part.success.to_numpy(float)
        for m in METHODS:
            p=part[m].to_numpy(float)
            row={'horizon':int(h),'method':m,'n':len(y),'brier':float(np.mean((p-y)**2)),
                'log_loss':float(-np.mean(y*np.log(np.clip(p,1e-6,1))+(1-y)*np.log(np.clip(1-p,1e-6,1)))),
                'mean_estimated_success':float((1-p).mean()),'realized_base_rate':float(1-y.mean()),
                'max_estimated_success':float(1-p.min()),'naive_not_probability':m=='naive_channels'}
            bands=[]
            for lo,hi in zip([0,.50,.70,.80,.85,.90,.95,.975],[.50,.70,.80,.85,.90,.95,.975,1.00001]):
                mask=((1-p)>=lo)&((1-p)<hi)
                bands.append({'from':lo,'to':min(hi,1),'n':int(mask.sum()),
                              'predicted':float((1-p)[mask].mean()) if mask.any() else None,
                              'realized':float(1-y[mask].mean()) if mask.any() else None})
            row['bins']=bands;rows.append(row)
    return rows


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--directory',type=Path,required=True)
    args=ap.parse_args(); root=args.directory; cfg=Config()
    meta=json.loads((root/'metadata.json').read_text());universe=meta['inputs']['universe']
    f=pd.read_parquet(root/'features.parquet',columns=['row_id','i','date','ticker','vol63','vol63_rank'])
    first='2013-01-01' if universe=='sp500' else '2018-01-01'
    audit=json.loads((root/'fit_audit.json').read_text())
    if any(a['null_shuffled_training'] for a in audit): first=str(min(a['year'] for a in audit))+'-01-01'
    first_i=int(f.loc[f.date>=first,'i'].min())
    rcols=['row_id','i','date','ticker','vol63_rank','vol63','horizon','exit_i','fit_i',*METHODS]
    risks_list=[];out_list=[];calibration=[]
    for h in cfg.horizons:
        r=pd.read_parquet(root/f'risks_{h}.parquet')
        y=pd.read_parquet(root/f'outcomes_{h}.parquet')
        y=y.loc[y.i>=first_i].copy()
        if not r.empty:
            r=r[rcols].copy()
            calibration.extend(reliability(r,y))
            risks_list.append(r)
        out_list.append(y)
    risks=pd.concat(risks_list,ignore_index=True);outcomes=pd.concat(out_list,ignore_index=True)
    del risks_list,out_list,r,y
    (root/'reliability.json').write_text(json.dumps(clean_json(calibration),indent=2,allow_nan=False))
    dates=f.loc[f.date>=first,'i'].drop_duplicates().sort_values().tolist()
    picks,ds=joint_policy(risks,outcomes,dates,cfg)
    picks=add_controls(picks,outcomes,f)
    picks.to_csv(root/'picks.csv',index=False);ds.to_csv(root/'decisions.csv',index=False)
    # Cutoff is the full available exchange calendar, not the last scheduled issue date.
    import exchange_calendars as xc
    # All i are the loader's calendar; label 'matured' carries the authoritative cutoff.
    cutoff=int(outcomes.loc[outcomes.matured,'exit_i'].max())
    report=summarize(picks,ds,cutoff)
    (root/'summary.json').write_text(json.dumps(clean_json(report),indent=2,allow_nan=False))
    flat=pd.DataFrame([{k:v for k,v in d.items() if not isinstance(v,(dict,list))} for d in report])
    flat.to_csv(root/'summary.csv',index=False)
    print(flat[['policy','issued','matured','wins','precision','random_expected_precision','nonoverlap_n']].to_string(index=False))

if __name__=='__main__':main()
