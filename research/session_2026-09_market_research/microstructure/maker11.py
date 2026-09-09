"""PREREG11: maker fills simulated against the ACTUAL 1-min high/low path.
Adverse selection is measured, not assumed."""
import numpy as np, pandas as pd, glob
from pathlib import Path
W=5; MAKER=0.0002; TAKER=0.0005
DELTAS=[5e-4,10e-4,20e-4]; HOLDS=[15,30]
RNG=np.random.default_rng(2718)
rows=[]
for f in sorted(glob.glob('/tmp/venues/c1m_*.parquet')):
    sym=Path(f).stem[4:]
    d=pd.read_parquet(f,columns=['t','o','h','l','c','v','tbv'])
    d['t']=pd.to_datetime(d.t,unit='ms',utc=True); d=d[d.t<'2026-01-01']
    c=d.c.to_numpy(float); hi=d.h.to_numpy(float); lo=d.l.to_numpy(float)
    v=d.v.to_numpy(float); tb=d.tbv.to_numpy(float); ts=d.t.to_numpy()
    ok=np.isfinite(c)&(c>0)&np.isfinite(hi)&np.isfinite(lo)&np.isfinite(v)
    c,hi,lo,v,tb,ts=c[ok],hi[ok],lo[ok],v[ok],tb[ok],ts[ok]
    sf=2*tb-v
    vbar=pd.Series(v).shift(1).rolling(100,min_periods=50).mean().to_numpy()
    idx=np.arange(240,len(c)-100,5)
    F=np.array([sf[i-4:i+1].sum() for i in idx])/np.maximum(vbar[idx],1e-9)
    keep=np.isfinite(F); idx=idx[keep]; F=F[keep]
    day=pd.to_datetime(ts[idx]).normalize()
    fr=pd.Series(F).groupby(pd.Series(day)).rank(pct=True).to_numpy()
    for name,cond in (('flow',fr>=0.8),('rand',RNG.random(len(idx))<0.2)):
        ii=idx[cond]; dd=day[cond]
        if len(ii)<1000: continue
        for delta in DELTAS:
            L=c[ii]*(1+delta)                      # limit SELL above market
            fillmin=np.full(len(ii),-1)
            for w in range(1,W+1):
                m=(fillmin<0)&(hi[ii+w]>L)          # strict: traded THROUGH the level
                fillmin[m]=w
            got=fillmin>0
            for k in HOLDS:
                j=ii[got]+fillmin[got]
                jj=np.minimum(j+k,len(c)-1)
                pnl=(L[got]-c[jj])/L[got]-MAKER-TAKER      # short at L, cover at close
                unc=(c[ii[got]]*(1+delta)-c[np.minimum(ii[got]+k,len(c)-1)])/L[got]  # same events, no fill filter
                rows.append(pd.DataFrame(dict(sym=sym,kind=name,delta=delta*1e4,k=k,
                    day=dd[got],pnl=pnl,fillrate=got.mean(),
                    fwd_filled=(L[got]-c[jj])/L[got])))
            # unconditional forward move for the SAME event set (adverse-selection ref)
            for k in HOLDS:
                jj=np.minimum(ii+k,len(c)-1)
                rows.append(pd.DataFrame(dict(sym=sym,kind=name+'_ALL',delta=delta*1e4,k=k,
                    day=dd,pnl=np.nan,fillrate=got.mean(),
                    fwd_filled=(c[ii]*(1+delta)-c[jj])/(c[ii]*(1+delta)))))
    print(f"{sym} done",flush=True)
R=pd.concat(rows,ignore_index=True); R.to_parquet('/tmp/venues/maker11.parquet',index=False)
print(f"\n{len(R):,} rows")
