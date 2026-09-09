"""Step 1: does band-resolved lead-lag exist, and does its SIGN FLIP across bands?
Non-causal bandpass here -- this establishes the phenomenon only. Trading test is causal."""
import numpy as np, pandas as pd
from pathlib import Path
D=Path('/tmp/databranch/data/equity_1m_alpaca')

BANDS={'FAST':(2,6),'MID':(6,20),'SLOW':(20,90)}
PAIRS=[('SMH','NVDA','basket'),('SMH','AMD','basket'),('QQQ','AAPL','basket'),
       ('QQQ','MSFT','basket'),('QQQ','NVDA','basket'),('SPY','AAPL','basket'),
       ('GLD','SLV','related'),('GLD','TLT','control'),('GLD','HYG','control'),
       ('TLT','XLE','control')]

def load(sym):
    d=pd.read_parquet(D/f'{sym}.parquet',columns=['t','c'])
    d['t']=pd.to_datetime(d.t,utc=True).dt.tz_convert('America/New_York')
    d=d[(d.t.dt.time>=pd.Timestamp('09:30').time())&(d.t.dt.time<=pd.Timestamp('16:00').time())]
    d['day']=d.t.dt.date
    d['m']=d.t.dt.hour*60+d.t.dt.minute
    return d[['day','m','c']]

def bandpass(r, lo_min, hi_min):
    """zero FFT coeffs outside [1/hi_min, 1/lo_min] cycles/min"""
    n=len(r); R=np.fft.rfft(r); f=np.fft.rfftfreq(n, d=1.0)   # cycles per minute
    keep=(f>=1.0/hi_min)&(f<=1.0/lo_min)
    R2=np.where(keep,R,0)
    return np.fft.irfft(R2,n=n)

cache={}
def get(sym):
    if sym not in cache: cache[sym]=load(sym)
    return cache[sym]

rows=[]
for a,b,kind in PAIRS:
    A,B=get(a),get(b)
    M=A.merge(B,on=['day','m'],suffixes=('_a','_b'))
    for day,g in M.groupby('day'):
        g=g.sort_values('m')
        if len(g)<200: continue
        ra=np.diff(np.log(g.c_a.to_numpy(float))); rb=np.diff(np.log(g.c_b.to_numpy(float)))
        if not (np.all(np.isfinite(ra)) and np.all(np.isfinite(rb))): continue
        for bn,(lo,hi) in BANDS.items():
            fa=bandpass(ra,lo,hi); fb=bandpass(rb,lo,hi)
            if fa.std()<1e-12 or fb.std()<1e-12: continue
            # does A at t predict B at t+1 ?  and B at t predict A at t+1 ?
            rows.append(dict(pair=f'{a}-{b}',kind=kind,band=bn,day=pd.Timestamp(day),
                a_leads=float(np.corrcoef(fa[:-1],fb[1:])[0,1]),
                b_leads=float(np.corrcoef(fb[:-1],fa[1:])[0,1]),
                contemp=float(np.corrcoef(fa,fb)[0,1])))
    print(f'{a}-{b} done ({kind})',flush=True)
R=pd.DataFrame(rows); R.to_parquet('/tmp/timbre/xspec.parquet',index=False)
print(f'\n{len(R):,} pair-day-band rows')
