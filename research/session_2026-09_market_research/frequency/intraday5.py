"""Extra intraday features needed by batch 2: last-5m return, volume-time last-30m,
first-30m return, and the volume-bar reversion trade (I13)."""
import numpy as np, pandas as pd
from pathlib import Path
from scipy.signal import lfilter
D=Path('/tmp/databranch/data/equity_1m_alpaca')
SP=pd.read_parquet('/tmp/timbre/panel4.parquet').groupby('sym').spread.median()
SYMS=sorted(p.stem for p in D.glob('*.parquet'))
def ema(x,n):
    a=2.0/(n+1.0); return lfilter([a],[1.0,-(1.0-a)],x,zi=[(1.0-a)*x[0]])[0]
feat=[]; vpnl=[]
for sym in SYMS:
    hs=float(SP.get(sym,np.nan))/2.0
    d=pd.read_parquet(D/f'{sym}.parquet',columns=['t','c','v'])
    d['t']=pd.to_datetime(d.t,utc=True).dt.tz_convert('America/New_York')
    d=d[(d.t.dt.time>=pd.Timestamp('09:30').time())&(d.t.dt.time<=pd.Timestamp('16:00').time())]
    d['day']=d.t.dt.normalize().dt.tz_localize(None); d=d[d.day<'2024-01-01']
    d['mn']=d.t.dt.hour*60+d.t.dt.minute
    for day,g in d.groupby('day',sort=True):
        g=g.sort_values('mn'); c=g.c.to_numpy(float); v=g.v.to_numpy(float); mn=g.mn.to_numpy()
        if len(c)<300 or c.min()<=0: continue
        lc=np.log(c); cv=np.cumsum(v); tot=cv[-1]
        if tot<=0: continue
        # volume-time "last 30 minutes" = the final X% of the day's volume,
        # where X = the average volume share of the last 30 wall minutes
        share=float((cv[-1]-cv[max(0,len(cv)-31)])/tot)
        j=int(np.searchsorted(cv, tot*(1.0-share)))
        j=min(max(j,0),len(lc)-2)
        feat.append(dict(sym=sym,day=day,
            ret_last5=float(lc[-1]-lc[-6]) if len(lc)>6 else np.nan,
            ret_first30=float(lc[min(30,len(lc)-1)]-lc[0]),
            ret_lastvol30=float(lc[-1]-lc[j]),
            vol_share_last30=share))
        # ---- I13: volume-bar reversion, non-overlapping, horizon-matched
        nbar=len(np.arange(4,len(c),5))
        if nbar<40: continue
        edges=np.unique(np.clip(np.searchsorted(cv, tot*np.arange(1,nbar+1)/nbar),0,len(c)-1))
        if len(edges)<40: continue
        cvb=lc[edges]; mvb=mn[edges]
        z=cvb-cvb[0]; sig=ema(z,3)-ema(z,12)
        pnl=[]; n=0
        for i in range(0,len(mvb),12):
            mm=mvb[i]
            if mm+60>mn[-1] or not np.isfinite(sig[i]): continue
            r=lc[np.searchsorted(mn,mm+60)]-lc[np.searchsorted(mn,mm)]
            pos=-np.sign(sig[i])
            if pos==0: continue
            pnl.append(pos*r-2*hs); n+=1
        if n: vpnl.append(dict(sym=sym,day=day,net=float(np.sum(pnl)),
                               gross=float(np.sum(pnl))+n*2*hs,cost=n*2*hs,ntr=n))
    print(f'{sym} ok',flush=True)
pd.DataFrame(feat).to_parquet('/tmp/timbre/feat5.parquet',index=False)
pd.DataFrame(vpnl).to_parquet('/tmp/timbre/i13_pnl.parquet',index=False)
print('saved',len(feat),len(vpnl))
