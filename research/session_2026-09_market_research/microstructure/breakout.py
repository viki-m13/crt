"""PREREG14: the ASE inversion. Breakout continuation on US equities, 15-min bars,
zero commission (Alpaca), spread charged as a full half-spread each way."""
import numpy as np, pandas as pd
from pathlib import Path
D=Path('/tmp/databranch/data/equity_1m_alpaca')
SP=pd.read_parquet('/tmp/timbre/panel4.parquet').groupby('sym').spread.median()  # Corwin-Schultz
DELTAS=[10e-4,25e-4,50e-4]; HOLDS=[1,2,4]
RNG=np.random.default_rng(1414)
rows=[]
for f in sorted(D.glob('*.parquet')):
    sym=f.stem; hs=float(SP.get(sym,np.nan))/2.0
    if not np.isfinite(hs): continue
    d=pd.read_parquet(f,columns=['t','o','h','l','c'])
    d['t']=pd.to_datetime(d.t,utc=True).dt.tz_convert('America/New_York')
    d=d[(d.t.dt.time>=pd.Timestamp('09:30').time())&(d.t.dt.time<=pd.Timestamp('16:00').time())]
    d['day']=d.t.dt.normalize().dt.tz_localize(None); d=d[d.day<'2024-01-01']
    g=d.set_index('t').resample('15min').agg(o=('o','first'),h=('h','max'),l=('l','min'),c=('c','last')).dropna()
    g['day']=g.index.normalize().tz_localize(None)
    # keep bars within a single session; drop cross-day transitions
    c=g.c.to_numpy(float); hi=g.h.to_numpy(float); lo=g.l.to_numpy(float); dy=g.day.to_numpy()
    n=len(c)
    if n<500: continue
    for delta in DELTAS:
        for k in HOLDS:
            i=np.arange(1,n-k-1)
            same=(dy[i]==dy[i-1])&(dy[i+k]==dy[i])          # no overnight gaps
            up=(hi[i]>c[i-1]*(1+delta))&same
            dn=(lo[i]<c[i-1]*(1-delta))&same
            # LONG: entry at the break level, exit at close k bars later
            eL=c[i-1]*(1+delta); xL=c[i+k]
            pL=(xL[up]-eL[up])/eL[up]-2*hs
            eS=c[i-1]*(1-delta); xS=c[i+k]
            pS=(eS[dn]-xS[dn])/eS[dn]-2*hs
            # CONTROL B5: random entries, same count, same holding, same charge
            nrand=len(pL)+len(pS)
            if nrand>10:
                ridx=RNG.choice(i[same],size=min(nrand,same.sum()),replace=False)
                side=RNG.choice([-1,1],size=len(ridx))
                pr=side*(c[ridx+k]-c[ridx])/c[ridx]-2*hs
                rows.append(pd.DataFrame(dict(sym=sym,kind='random',delta=delta*1e4,k=k,
                    day=dy[ridx],pnl=pr)))
            if len(pL): rows.append(pd.DataFrame(dict(sym=sym,kind='breakout',delta=delta*1e4,k=k,day=dy[i][up],pnl=pL)))
            if len(pS): rows.append(pd.DataFrame(dict(sym=sym,kind='breakout',delta=delta*1e4,k=k,day=dy[i][dn],pnl=pS)))
    print(f'{sym} ok',flush=True)
R=pd.concat(rows,ignore_index=True); R.to_parquet('/tmp/ase/breakout.parquet',index=False)
print(f"\n{len(R):,} trades")
