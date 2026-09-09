"""THE decisive control: is the cross-asset signal doing anything the follower's OWN
oscillator doesn't already do? If SELF >= CROSS, there is no lead-lag content."""
import numpy as np, pandas as pd
from pathlib import Path
D=Path('/tmp/databranch/data/equity_1m_alpaca')
def load(s):
    d=pd.read_parquet(D/f'{s}.parquet',columns=['t','c'])
    d['t']=pd.to_datetime(d.t,utc=True).dt.tz_convert('America/New_York')
    d=d[(d.t.dt.time>=pd.Timestamp('09:30').time())&(d.t.dt.time<=pd.Timestamp('16:00').time())]
    d['day']=d.t.dt.date; d['m']=d.t.dt.hour*60+d.t.dt.minute
    return d[['day','m','c']]
def ema(x,n):
    a=2.0/(n+1.0); o=np.empty_like(x); o[0]=x[0]
    for i in range(1,len(x)): o[i]=a*x[i]+(1-a)*o[i-1]
    return o
cache={}
def get(s):
    if s not in cache: cache[s]=load(s)
    return cache[s]
rows=[]
for lead,foll in [('NVDA','SMH'),('AMD','SMH'),('GLD','SLV')]:
    M=get(lead).merge(get(foll),on=['day','m'],suffixes=('_l','_f'))
    for day,g in M.groupby('day'):
        g=g.sort_values('m')
        if len(g)<200: continue
        rl=np.diff(np.log(g.c_l.to_numpy(float))); rf=np.diff(np.log(g.c_f.to_numpy(float)))
        if not(np.all(np.isfinite(rl))and np.all(np.isfinite(rf))): continue
        sig_cross=(lambda c: ema(c,3)-ema(c,12))(np.cumsum(rl))[:-1]
        sig_self =(lambda c: ema(c,3)-ema(c,12))(np.cumsum(rf))[:-1]
        for k in (30,60,120):
            if len(rf)<k+2: continue
            fut=np.array([rf[i+1:i+1+k].sum() if i+1+k<=len(rf) else np.nan for i in range(len(sig_cross))])
            ok=np.isfinite(fut)
            if ok.sum()<50: continue
            rows.append(dict(pair=f'{lead}->{foll}',day=pd.Timestamp(day),k=k,
                ic_cross=float(np.corrcoef(sig_cross[ok],fut[ok])[0,1]),
                ic_self =float(np.corrcoef(sig_self[ok], fut[ok])[0,1])))
    print(f'{lead}->{foll} done',flush=True)
R=pd.DataFrame(rows); R.to_parquet('/tmp/timbre/control.parquet',index=False)
print()
print('CONTROL: cross-asset signal vs the follower OWN oscillator (same target)')
print(f'{"pair":12s} {"k":>4s} {"IC cross":>10s} {"IC SELF":>10s} {"cross-self":>11s}  verdict')
for p in R.pair.unique():
    for k in (30,60,120):
        g=R[(R.pair==p)&(R.k==k)]
        c,s=g.ic_cross.mean(),g.ic_self.mean()
        d=c-s; td=d/(( g.ic_cross-g.ic_self).std()/np.sqrt(len(g)))
        v='cross ADDS' if abs(c)>abs(s) and abs(td)>2 else ('SELF explains it' if abs(s)>=abs(c) else 'no diff')
        print(f'{p:12s} {k:4d} {c:+10.4f} {s:+10.4f} {d:+8.4f}(t{td:+5.1f})  {v}')
