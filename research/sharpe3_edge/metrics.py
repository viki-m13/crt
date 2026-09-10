"""Calendar-complete daily NAV statistics; cash/no-exposure days are retained."""
import numpy as np
import pandas as pd


def sharpe(r):
    r=np.asarray(r,float); sd=r.std(ddof=1)
    return float(np.sqrt(252)*r.mean()/sd) if len(r)>1 and sd>1e-12 else None


def hac_sharpe(r,lags=21):
    r=np.asarray(r,float);x=r-r.mean();n=len(x)
    if n<2 or x.var()<1e-24:return None
    v=float(x@x/n)
    for k in range(1,min(lags,n-1)+1):v+=2*(1-k/(lags+1))*float(x[k:]@x[:-k]/n)
    return float(np.sqrt(252)*r.mean()/np.sqrt(v)) if v>0 else None


def describe(d,market):
    r=d['return'].to_numpy(float); dates=pd.to_datetime(d.date)
    nav=np.cumprod(1+r); peak=np.maximum.accumulate(np.r_[1.,nav])[1:]
    years=max(len(r)/252,1/252); calendar=max((dates.iloc[-1]-dates.iloc[0]).days/365.25,1/252)
    ann=(1.03)**(1/252)-1; downside=np.sqrt(np.mean(np.minimum(r,0)**2))
    bm=market.pct_change().reindex(pd.DatetimeIndex(dates)).fillna(0).to_numpy()
    beta=float(np.cov(r,bm)[0,1]/bm.var(ddof=1)) if bm.var()>0 else None
    annual=pd.Series(r,index=dates).groupby(dates.dt.year.to_numpy()).apply(lambda x:float(np.prod(1+x)-1)).to_dict()
    return dict(days=len(r),years=calendar,cagr=float(nav[-1]**(1/calendar)-1),arithmetic_return=float(r.mean()*252),
        volatility=float(r.std(ddof=1)*np.sqrt(252)),sharpe_zero=sharpe(r),sharpe_cash_hurdle3=sharpe(r-ann),
        sharpe_hac21=hac_sharpe(r,21),max_drawdown=float((nav/peak-1).min()),
        sortino=float(r.mean()*np.sqrt(252)/downside) if downside>0 else None,beta=beta,
        annual_returns={str(k):v for k,v in annual.items()},worst_year=min(annual.values()),
        average_exposure=float(d.exposure.mean()),max_gross=float(d.gross.max()),
        active_days=int((d.positions>0).sum()),costs=float(d.costs.iloc[-1]-d.costs.iloc[0]),
        borrow=float(d.borrow.iloc[-1]-d.borrow.iloc[0]),margin_breaches=int(d.margin_breach.sum()),
        terminal_multiple=float(nav[-1]))


def bootstrap(r,bm=None,block=126,reps=500,seed=192):
    """Circular moving-block diagnostic, not correction for repeated research selection."""
    r=np.asarray(r,float);n=len(r);rng=np.random.default_rng(seed);out=[];diff=[]
    for _ in range(reps):
        starts=rng.integers(0,n,int(np.ceil(n/block)));ids=((starts[:,None]+np.arange(block))%n).ravel()[:n]
        s=sharpe(r[ids]);out.append(np.nan if s is None else s)
        if bm is not None:
            b=sharpe(np.asarray(bm)[ids]);diff.append(s-b if s is not None and b is not None else np.nan)
    finite=np.asarray(out)[np.isfinite(out)]
    res={'block_sessions':block,'repetitions':reps,'sharpe_ci95':np.quantile(finite,[.025,.975]).tolist() if len(finite) else None,
         'qualification':'conditional same-history diagnostic, not multiplicity-adjusted or prospective'}
    if bm is not None:
        finite=np.asarray(diff)[np.isfinite(diff)];res['paired_sharpe_difference_ci95']=np.quantile(finite,[.025,.975]).tolist() if len(finite) else None
    return res
