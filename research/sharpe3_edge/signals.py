"""IPD state identification before outcomes; scores are not probabilities."""
import numpy as np
import pandas as pd
from .data import clean

HORIZONS=(30,60,90,126,180,252)
METHODS=('ipd_adaptive','ipd_fixed60','information60','transient60','ipd_selective','ipd_volume60',
         'momentum60','reversal60','lowvol_momentum60','equal60','inverted60',
         'random0','random1','random2','random3','random4')


def features(prices,member,market,volume=None,stride=5):
    p=clean(prices)
    if not p.index.equals(member.index) or not p.columns.equals(member.columns) or not p.index.equals(market.index):
        raise ValueError('Misaligned stock/member/market data')
    if member.isna().any().any() or not member.isin([0,1]).all().all(): raise ValueError('Invalid membership')
    if stride<1: raise ValueError('Invalid stride')
    lp=np.log(p); lm=np.log(market); r=lp.diff(); rm=lm.diff()
    beta=r.rolling(126).cov(rm).div(rm.rolling(126).var(),axis=0)
    eps=r.sub(beta.shift(1).mul(rm,axis=0))
    v=eps.rolling(63).std(); vol=r.rolling(63).std()
    fs={f'e{k}':eps.rolling(k).sum() for k in (5,21,63)}
    fs.update(vol=vol,beta=beta,rv=v,volratio=eps.rolling(21).std()/v,
        eff=eps.rolling(21).sum().abs()/eps.abs().rolling(21).sum(),
        ma200=lp-lp.rolling(200).mean(),dd=lp-lp.rolling(252).max(),
        r21=lp.diff(21),r63=lp.diff(63),r126=lp.diff(126))
    fs['z21']=fs['e21']/(v*np.sqrt(21)); fs['z63']=fs['e63']/(v*np.sqrt(63))
    if volume is not None:
        if not volume.index.equals(p.index) or not volume.columns.equals(p.columns):raise ValueError('Volume alignment')
        vr=volume.where(volume>0).rolling(5).mean()/volume.where(volume>0).rolling(63).median()
    else: vr=None
    common=pd.DataFrame({'market21':lm.diff(21),'market63':lm.diff(63),'market200':lm-lm.rolling(200).mean()})
    rows=[]; coverage=[]
    for i in range(252,len(p),stride):
        symbols=p.columns[member.iloc[i].to_numpy(bool)]
        f=pd.DataFrame({k:a.iloc[i].reindex(symbols) for k,a in fs.items()})
        f['reference']=p.iloc[i].reindex(symbols)
        f=f.replace([np.inf,-np.inf],np.nan).dropna()
        f=f[(f.vol>1e-5)&(f.rv>1e-5)]
        coverage.append(dict(i=i,date=str(p.index[i].date()),represented_members=len(symbols),eligible=len(f)))
        if f.empty or common.iloc[i].isna().any():continue
        f['eff_rank']=f.eff.rank(pct=True); f['vol_rank']=f.vol.rank(pct=True)
        agree=((np.sign(f.e5)==np.sign(f.e21)).astype(float)+(np.sign(f.e63)==np.sign(f.e21)).astype(float))/2
        expand=(f.volratio-1).clip(0,1)
        f['information']=.4*f.eff_rank+.3*agree+.3*(1-expand)
        f['transient']=.4*(1-f.eff_rank)+.3*(1-agree)+.3*expand
        f['separation']=f.information-f.transient
        f['strength']=f.separation.abs()*f.z21.abs().clip(upper=3)
        f['breadth']=float(f.ma200.gt(0).mean())
        for k,val in common.iloc[i].items():f[k]=float(val)
        continuation=(f.z21>.5)&(f.e63>0)&(f.separation>.2)
        reversal=(f.z21<-.5)&(f.e5>0)&(f.separation<-.2)&((f.market21>0)|(f.breadth>.5))
        f['family']=np.where(continuation,'information',np.where(reversal,'transient','none'))
        f['volume_ratio']=vr.iloc[i].reindex(f.index) if vr is not None else np.nan
        f['volume_rank']=f.volume_ratio.rank(pct=True)
        f['i']=i; f['date']=str(p.index[i].date()); f['ticker']=f.index
        rows.append(f.reset_index(drop=True))
    return pd.concat(rows,ignore_index=True) if rows else pd.DataFrame(),pd.DataFrame(coverage)


def labels(f,p,market,h,lag=1):
    """Matured complete-path returns for payoff estimation; unavailable path = -100%."""
    if h<30 or lag<1:raise ValueError('Invalid horizon/lag')
    a=p.to_numpy(float); i=f.i.to_numpy(int); j=p.columns.get_indexer(f.ticker)
    if (j<0).any():raise ValueError('Unknown ticker')
    entry_i=i+lag; exit_i=entry_i+h; matured=exit_i<len(p)
    entry=np.full(len(f),np.nan); end=entry.copy()
    available=entry_i<len(p); entry[available]=a[entry_i[available],j[available]]
    end[matured]=a[exit_i[matured],j[matured]]
    # Cumulative missing counts make future completeness explicit and cheap.
    missing=np.vstack([np.zeros((1,a.shape[1]),int),np.cumsum(~np.isfinite(a)|(a<=0),axis=0)])
    complete=np.zeros(len(f),bool)
    complete[matured]=(missing[exit_i[matured]+1,j[matured]]-missing[entry_i[matured],j[matured]])==0
    ret=np.full(len(f),np.nan); ret[matured]=-1
    ret[matured&complete]=end[matured&complete]/entry[matured&complete]-1
    bm=market.to_numpy(float); br=np.full(len(f),np.nan)
    br[matured]=bm[exit_i[matured]]/bm[entry_i[matured]]-1
    out=pd.DataFrame(dict(i=i,h=h,entry_i=entry_i,exit_i=exit_i,matured=matured,complete=complete,ret=ret,market_return=br))
    out['pool_return']=out.groupby('i').ret.transform('mean')
    out['excess']=out.ret-out.pool_return
    return out


def fit_horizons(f,labs,asof,window=2520):
    """Fit only fully matured, costed state-vs-pool payoffs. No test-year outcomes."""
    rows=[]
    for h,y in labs.items():
        visible=(y.exit_i<asof)&y.matured&(f.i>=asof-window)
        for family in ('information','transient'):
            mask=visible&f.family.eq(family)
            q=pd.DataFrame({'i':f.loc[mask,'i'],'value':((y.loc[mask,'excess']-.005)*252/h).clip(-2,2)})
            d=q.groupby('i').value.mean()
            if d.empty:continue
            b=d.groupby(d.index//h).mean(); n=len(b)
            w=np.exp2(-(asof-d.index.to_numpy())/756); mean=float(np.average(d,weights=w))
            shrink=n/(n+8); mu=mean*shrink
            se=float(b.std(ddof=1)/np.sqrt(n)) if n>1 else float('inf')
            buckets=np.minimum(2,((d.index-(asof-window))*3//window)).astype(int)
            era=d.groupby(buckets).mean()
            rows.append(dict(asof=asof,family=family,h=h,dates=len(d),blocks=n,mean=mean,mu=mu,se=se,
                lower=mu-se,all_eras_positive=len(era)==3 and bool((era>0).all()),
                max_exit_i=int(y.loc[mask,'exit_i'].max())))
    return pd.DataFrame(rows)


def horizon_map(stats,strict=False):
    if stats.empty:return {}
    eligible=stats[(stats.dates>=24)&(stats.blocks>=(12 if strict else 8))&(stats.lower>0)]
    if strict:eligible=eligible[eligible.all_eras_positive]
    return {family:int(g.sort_values(['lower','h'],ascending=[False,True]).iloc[0].h)
            for family,g in eligible.groupby('family')}


def proposals(f,method,hmap=None,held=()):
    """Known-at-signal ranking. No outcome columns or price arrays accepted here."""
    g=f.loc[~f.ticker.isin(held)].copy()
    if g.empty:return g.assign(h=pd.Series(dtype=int),score=pd.Series(dtype=float))
    g['h']=60
    if method.startswith('ipd') or method in ('information60','transient60','inverted60'):
        if method=='inverted60':g=g[g.family.eq('none')]; g['score']=-g.strength
        else:
            g=g[g.family.ne('none')]; g['score']=g.strength
            if method=='information60':g=g[g.family.eq('information')]
            if method=='transient60':g=g[g.family.eq('transient')]
            if method in ('ipd_adaptive','ipd_selective'):
                g['h']=g.family.map(hmap or {}); g=g[g.h.notna()]
            if method=='ipd_volume60':g=g[g.volume_rank>=.5]
    elif method=='momentum60':g['score']=g.z63
    elif method=='reversal60':g['score']=-g.z21
    elif method=='lowvol_momentum60':
        g=g[(g.e63>0)&(g.ma200>0)&(g.vol_rank<.5)];g['score']=g.z63
    elif method=='equal60':g['score']=0.;g['all_members']=True
    elif method.startswith('random'):
        rng=np.random.default_rng(20260910+int(method[-1])+int(f.i.iloc[0])*100)
        g=g.sort_values('ticker');g['score']=rng.random(len(g))
    else:raise ValueError('Unknown method '+method)
    g['h']=g.h.astype(int)
    return g.sort_values(['score','ticker'],ascending=[False,True])
