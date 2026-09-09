"""Shared long/short P&L evaluator. Money, not IC."""
import numpy as np, pandas as pd
def ls_pnl(df, sig, tgt, spread='spread', nq=5, minn=15):
    d = df[['day',sig,tgt,spread]].dropna().copy()
    d = d[d.groupby('day')[sig].transform('size') >= minn]
    if len(d) < 100: return None
    d['q'] = d.groupby('day')[sig].transform(
        lambda s: pd.qcut(s.rank(method='first'), nq, labels=False, duplicates='drop'))
    lo = d[d.q==0].groupby('day').agg(r=(tgt,'mean'), c=(spread,'mean'))
    hi = d[d.q==nq-1].groupby('day').agg(r=(tgt,'mean'), c=(spread,'mean'))
    j = hi.join(lo, lsuffix='_h', rsuffix='_l', how='inner')
    if len(j) < 30: return None
    gross = j.r_h - j.r_l
    cost  = j.c_h + j.c_l
    net   = gross - cost
    def t(x): return x.mean()/(x.std(ddof=1)/np.sqrt(len(x)))
    return dict(gross=gross.mean()*1e4, t_gross=t(gross), cost=cost.mean()*1e4,
                net=net.mean()*1e4, t_net=t(net),
                ec=gross.mean()/cost.mean() if cost.mean()>0 else np.nan, ndays=len(j))
def show(tag, tr, va):
    f=lambda r: ("   n/a" if r is None else
        f"gross{r['gross']:+7.2f}(t{r['t_gross']:+5.2f}) cost{r['cost']:6.2f} NET{r['net']:+7.2f}(t{r['t_net']:+5.2f}) e/c{r['ec']:+6.2f}")
    print(f"{tag:5s} TRAIN {f(tr)}")
    print(f"{'':5s} VALID {f(va)}")
