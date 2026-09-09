"""Step 2 (PRE-REGISTERED): strictly causal implementation + real costs.
Band via CAUSAL EMA difference. Signal at bar t uses only bars <= t. Trade t+1."""
import numpy as np, pandas as pd
from pathlib import Path
D=Path('/tmp/databranch/data/equity_1m_alpaca')

def load(sym):
    d=pd.read_parquet(D/f'{sym}.parquet',columns=['t','o','h','l','c'])
    d['t']=pd.to_datetime(d.t,utc=True).dt.tz_convert('America/New_York')
    d=d[(d.t.dt.time>=pd.Timestamp('09:30').time())&(d.t.dt.time<=pd.Timestamp('16:00').time())]
    d['day']=d.t.dt.date; d['m']=d.t.dt.hour*60+d.t.dt.minute
    return d[['day','m','o','h','l','c']]

def ema(x,n):
    a=2.0/(n+1.0); out=np.empty_like(x); out[0]=x[0]
    for i in range(1,len(x)): out[i]=a*x[i]+(1-a)*out[i-1]
    return out

def corwin_schultz(h,l):
    """2-day high-low effective spread estimator, in fraction of price."""
    h=np.asarray(h,float); l=np.asarray(l,float)
    if len(h)<2: return np.nan
    b=(np.log(h[1:]/l[1:]))**2+(np.log(h[:-1]/l[:-1]))**2
    hh=np.maximum(h[1:],h[:-1]); ll=np.minimum(l[1:],l[:-1])
    g=(np.log(hh/ll))**2
    k=1/(np.sqrt(2)-1) if False else 3-2*np.sqrt(2)
    a=(np.sqrt(2*b)-np.sqrt(b))/k - np.sqrt(g/k)
    s=2*(np.exp(a)-1)/(1+np.exp(a))
    s=s[np.isfinite(s)]
    return float(np.median(s[s>0])) if len(s[s>0]) else np.nan

PAIRS=[('NVDA','SMH'),('AMD','SMH'),('GLD','SLV')]   # leader, follower  (from step 1)
cache={}
def get(s):
    if s not in cache: cache[s]=load(s)
    return cache[s]

rows=[]
for lead,foll in PAIRS:
    L,F=get(lead),get(foll)
    M=L.merge(F,on=['day','m'],suffixes=('_l','_f'))
    for day,g in M.groupby('day'):
        g=g.sort_values('m')
        if len(g)<200: continue
        cl=g.c_l.to_numpy(float); cf=g.c_f.to_numpy(float)
        rl=np.diff(np.log(cl)); rf=np.diff(np.log(cf))
        if not(np.all(np.isfinite(rl))and np.all(np.isfinite(rf))): continue
        # CAUSAL band-pass on the LEADER's cumulative return: EMA(3) - EMA(12) ~ 2-20min band
        c=np.cumsum(rl)
        sig=ema(c,3)-ema(c,12)
        s=sig[:-1]                       # signal known at end of bar t
        for k in (1,5,15,30,60,120):
            if len(rf)<k+2: continue
            fut=np.array([rf[i+1:i+1+k].sum() if i+1+k<=len(rf) else np.nan for i in range(len(s))])
            ok=np.isfinite(fut)&np.isfinite(s)
            if ok.sum()<50: continue
            rows.append(dict(pair=f'{lead}->{foll}',day=pd.Timestamp(day),k=k,
                ic=float(np.corrcoef(s[ok],fut[ok])[0,1]),
                sd_fut=float(np.nanstd(fut[ok])),n=int(ok.sum())))
    sp=corwin_schultz(get(foll).h.to_numpy(),get(foll).l.to_numpy())
    print(f'{lead}->{foll}: effective spread (Corwin-Schultz) {sp*1e4:.1f} bp',flush=True)
R=pd.DataFrame(rows); R.to_parquet('/tmp/timbre/causal.parquet',index=False)
print(f'\n{len(R):,} rows')
