"""Experiment B: response asymmetry, dissemination and reserved state clocks."""
import numpy as np
import pandas as pd
from .signals import proposals as base_proposals

NEW_METHODS=('ipd_balanced60','absorption60','breadth_release30','rotation60','convexity60','barbell60')


def enrich(f,p,market):
    f=f.copy();r=np.log(p).diff();rm=np.log(market).diff();up=rm.clip(lower=0);dn=rm.clip(upper=0)
    bu=r.rolling(126).cov(up).div(up.rolling(126).var(),axis=0)
    bd=r.rolling(126).cov(dn).div(dn.rolling(126).var(),axis=0)
    i=f.i.to_numpy(int);j=p.columns.get_indexer(f.ticker)
    f['absorption_gap']=(bu-bd).to_numpy()[i,j]
    f['absorption_rank']=f.groupby('i').absorption_gap.rank(pct=True)
    f['rank_change']=f.groupby('i').e21.rank(pct=True)-f.groupby('i').e63.rank(pct=True)
    bs=f.groupby('i').breadth.first();old=bs.reindex(bs.index-21,method='ffill').to_numpy()
    change=pd.Series(bs.to_numpy()-old,index=bs.index)
    f['breadth_change']=f.i.map(change)
    return f


def proposals(f,method,hmap=None,held=()):
    if method not in NEW_METHODS:return base_proposals(f,method,hmap,held)
    parity=((int(f.i.iloc[0])-252)//5)%2
    if method=='ipd_balanced60':return base_proposals(f,'information60' if parity==0 else 'transient60',hmap,held)
    if method=='barbell60':return proposals(f,'absorption60' if parity==0 else 'convexity60',hmap,held)
    g=f.loc[~f.ticker.isin(held)].copy();g['h']=60
    if method=='absorption60':
        mask=(g.market21<0)&(g.e21>0)&(g.dd>-.30)&(g.absorption_rank>.75)
        score=g.absorption_gap+g.e21/g.rv
    elif method=='breadth_release30':
        mask=(g.breadth_change>.1)&(g.market21>-.05)&(g.market200<0)&(g.r63<0)&(g.e5>0)
        score=g.e5/g.rv-g.z21;g['h']=30
    elif method=='rotation60':
        mask=(g.breadth>.45)&(g.breadth_change>0)&(g.e21>0)&(g.rank_change>.25)&(g.vol_rank<.6)
        score=g.rank_change+g.eff_rank
    elif method=='convexity60':
        mask=(g.e63>0)&(g.e21>g.e63/3)&(g.e5>0)&(g.e5<g.e21/2)&(g.eff_rank>.5)&(g.volratio<1)
        score=(g.e21-g.e63/3)/g.rv
    else:raise ValueError(method)
    g['score']=score;g=g[mask&np.isfinite(score)]
    return g.sort_values(['score','ticker'],ascending=[False,True])
