"""Exploratory selection by rarity among past failures, not a probability cutoff.

Inspired by Jin & Candes (arXiv:2210.01408). This financial adaptation does NOT
inherit their FDR guarantee: time dependence, changing models and top-one
postselection violate assumptions needed for that claim. Measure it explicitly.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from .core import metrics,write_json


def bh_mask(values,alpha=.05):
    values=np.asarray(values,dtype=float)
    if not 0<alpha<1 or not np.isfinite(values).all():raise ValueError('Invalid selection inputs')
    m=len(values)
    if not m:return np.zeros(0,dtype=bool)
    order=np.argsort(values,kind='stable');ok=values[order]<=alpha*np.arange(1,m+1)/m
    if not ok.any():return np.zeros(m,dtype=bool)
    cutoff=values[order[np.flatnonzero(ok)[-1]]]
    return values<=cutoff


def failure_ranks(history,now,score='p_model',regime=False):
    out=now.copy();out['failure_rank']=1.
    if now.empty:return out
    history=history.loc[history.exit_i.lt(int(now.i.min()))&history.matured]
    for h,group in now.groupby('horizon'):
        for state in [False,True] if regime else [None]:
            current=group if state is None else group.loc[group.market_ma_gap.gt(0).eq(state)]
            past=history.loc[history.horizon.eq(h)]
            if state is not None:past=past.loc[past.market_ma_gap.gt(0).eq(state)]
            if len(past)<1000 or past.i.nunique()<24:continue
            negative=np.sort(past.loc[~(past.resolved.fillna(False)&past.up.fillna(False)),score].to_numpy())
            larger=len(negative)-np.searchsorted(negative,current[score],side='left')
            out.loc[current.index,'failure_rank']=(1+larger)/(1+len(past))
    return out


def simulate(raw,score='p_model',regime=False):
    selections=[];audit=[];open_until={}
    for i,now in raw.groupby('i',sort=True):
        history=raw.loc[raw.exit_i.lt(i)&raw.i.ge(i-10*252)&raw.matured]
        ranked=failure_ranks(history,now,score,regime)
        # Include EVERY stock/horizon comparison in the batch denominator.
        ranked['batch_pass']=bh_mask(ranked.failure_rank.to_numpy())
        eligible=ranked.loc[ranked.batch_pass&ranked.agreement].copy()
        eligible=eligible.loc[[i>open_until.get(r.ticker,-1) for r in eligible.itertuples()]]
        if len(eligible):
            eligible=eligible.sort_values(['ticker','horizon'],kind='stable').drop_duplicates('ticker')
            pick=eligible.sort_values(['failure_rank','horizon','ticker'],kind='stable').iloc[0].to_dict()
            selections.append(pick);open_until[pick['ticker']]=int(pick['exit_i'])
        audit.append(dict(i=int(i),comparisons=len(now),batch_selected=int(ranked.batch_pass.sum()),
                          eligible_stocks=len(eligible),issued=int(len(eligible)>0)))
    out=pd.DataFrame(selections,columns=list(raw.columns)+['failure_rank','batch_pass'])
    out['date']=pd.to_datetime(out['date'])
    return out,audit


def run(raw_path,out):
    raw=pd.read_parquet(raw_path);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    report=dict(status='EXPLORATORY_NOT_A_95_PERCENT_GUARANTEE',policies={})
    cutoff=int(raw.loc[raw.matured,'exit_i'].max())
    for score in ['p_model','q05']:
        for regime in [False,True]:
            name=f'{score}_'+('regime' if regime else 'pooled')
            picked,audit=simulate(raw,score,regime)
            picked.to_csv(out/(name+'_picks.csv'),index=False)
            for start in [2013,2019]:
                dates=raw.loc[raw.date.dt.year.ge(start),'i'].unique()
                key=name+'_'+str(start)
                report['policies'][key]=metrics(picked.loc[picked.date.dt.year.ge(start)],dates,cutoff)
            write_json(out/(name+'_audit.json'),audit)
    write_json(out/'failure_rank_report.json',report)
    print(report,flush=True)
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--raw',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args();run(a.raw,a.out)
