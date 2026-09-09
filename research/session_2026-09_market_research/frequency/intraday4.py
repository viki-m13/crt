"""PREREG4 I8/I9/I10: volume-clock sampling, slow-vs-fast band NET OF COST, time-of-day."""
import numpy as np, pandas as pd
from pathlib import Path
from scipy.signal import lfilter
D=Path('/tmp/databranch/data/equity_1m_alpaca')
SYMS=sorted(p.stem for p in D.glob('*.parquet'))
SPREAD=pd.read_parquet('/tmp/timbre/panel4.parquet').groupby('sym').spread.median()

def ema(x,n):
    a=2.0/(n+1.0); return lfilter([a],[1.0,-(1.0-a)],x,zi=[(1.0-a)*x[0]])[0]

rows_ic=[]; rows_pnl=[]; rows_tod=[]
for sym in SYMS:
    hs=float(SPREAD.get(sym,np.nan))/2.0          # half-spread, fraction of price
    d=pd.read_parquet(D/f'{sym}.parquet',columns=['t','c','v'])
    d['t']=pd.to_datetime(d.t,utc=True).dt.tz_convert('America/New_York')
    d=d[(d.t.dt.time>=pd.Timestamp('09:30').time())&(d.t.dt.time<=pd.Timestamp('16:00').time())]
    d['day']=d.t.dt.normalize().dt.tz_localize(None)
    d=d[d.day<pd.Timestamp('2024-01-01')]
    d['mn']=d.t.dt.hour*60+d.t.dt.minute
    for day,g in d.groupby('day',sort=True):
        g=g.sort_values('mn'); c=g.c.to_numpy(float); v=g.v.to_numpy(float); mn=g.mn.to_numpy()
        if len(c)<300 or not np.all(np.isfinite(c)) or c.min()<=0: continue
        lc=np.log(c)
        # ---- wall-clock 5-min bars
        idx5=np.arange(4,len(c),5)
        if len(idx5)<40: continue
        c5=lc[idx5]; m5=mn[idx5]
        r5=np.diff(c5)
        # ---- volume bars, SAME COUNT as wall bars
        cv=np.cumsum(v); tot=cv[-1]
        if tot<=0: continue
        edges=np.searchsorted(cv, tot*np.arange(1,len(idx5)+1)/len(idx5))
        edges=np.clip(edges,0,len(c)-1); edges=np.unique(edges)
        if len(edges)<40: continue
        cvb=lc[edges]; mvb=mn[edges]
        # signals (causal): oscillator on cumulative log price
        def osc(x,f,s): 
            z=x-x[0]; return ema(z,f)-ema(z,s)
        sig_wall=osc(c5,3,12)
        sig_vol =osc(cvb,3,12)
        sig_slow=osc(c5,12,48)
        # target: forward 60 minutes in WALL time from each wall bar
        def fwd60(m_arr, base_ok):
            out=np.full(len(m_arr),np.nan)
            for i,mm in enumerate(m_arr):
                j=np.searchsorted(mn, mm+60)
                if j<len(lc) and mm+60<=mn[-1]: out[i]=lc[j]-lc[np.searchsorted(mn,mm)]
            return out
        f_wall=fwd60(m5,None)
        f_vol =fwd60(mvb,None)
        def ic(s,f):
            ok=np.isfinite(s)&np.isfinite(f)
            if ok.sum()<20 or np.std(s[ok])==0 or np.std(f[ok])==0: return np.nan
            return float(np.corrcoef(s[ok],f[ok])[0,1])
        rows_ic.append(dict(sym=sym,day=day,ic_wall=ic(sig_wall,f_wall),
                            ic_vol=ic(sig_vol,f_vol),ic_slow=ic(sig_slow,f_wall)))
        # ---- I9 net-of-cost sim on 5-min bars, both bands
        for lab,s in (('fast',sig_wall),('slow',sig_slow)):
            pos=-np.sign(s[:-1])                       # trade next bar
            if len(pos)<10: continue
            gross=float((pos*r5).sum())
            turn=float(np.abs(np.diff(np.concatenate(([0.0],pos)))).sum())
            cost=turn*hs if np.isfinite(hs) else np.nan
            rows_pnl.append(dict(sym=sym,day=day,band=lab,gross=gross,cost=cost,net=gross-cost,turn=turn))
        # ---- I10 time-of-day IC of the fast signal
        for lo,hiq,lab in ((570,660,'0930-1100'),(660,780,'1100-1300'),(780,900,'1300-1500'),(900,960,'1500-1600')):
            m=(m5[:-1]>=lo)&(m5[:-1]<hiq)
            if m.sum()<8: continue
            s=sig_wall[:-1][m]; f=f_wall[:-1][m]
            ok=np.isfinite(s)&np.isfinite(f)
            if ok.sum()<8 or np.std(s[ok])==0 or np.std(f[ok])==0: continue
            rows_tod.append(dict(sym=sym,day=day,bucket=lab,ic=float(np.corrcoef(s[ok],f[ok])[0,1])))
    print(f'{sym} ok',flush=True)

for nm,rs in (('ic',rows_ic),('pnl',rows_pnl),('tod',rows_tod)):
    pd.DataFrame(rs).to_parquet(f'/tmp/timbre/i4_{nm}.parquet',index=False)
print('saved', len(rows_ic), len(rows_pnl), len(rows_tod))
