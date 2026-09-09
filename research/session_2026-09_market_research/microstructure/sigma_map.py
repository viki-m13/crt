"""sigma (entropy production per step) on a COMMON scale across venues.
Same estimator as PREREG8: ordinal-pattern KL(fwd||rev), de-biased with 5
phase-randomized surrogates. Returns normalized per-day/per-block so sigma is
scale-free. Computed at a common 5-minute bar for cross-venue comparison."""
import numpy as np, pandas as pd, glob
from pathlib import Path
RNG=np.random.default_rng(1234); M=3
def codes(r,m=M):
    W=np.lib.stride_tricks.sliding_window_view(r,m)
    return (np.argsort(W,axis=1,kind='stable')*(m**np.arange(m))).sum(1)
def sig_kl(r,m=M):
    cf=codes(r,m); cr=codes(r[::-1],m); nb=m**m
    pf=np.bincount(cf,minlength=nb).astype(float); pr=np.bincount(cr,minlength=nb).astype(float)
    if pf.sum()<500: return np.nan
    pf/=pf.sum(); pr/=pr.sum(); k=(pf>0)&(pr>0)
    return float((pf[k]*np.log(pf[k]/pr[k])).sum())
def phase_rand(r,rng):
    n=len(r); R=np.fft.rfft(r); ph=rng.uniform(0,2*np.pi,len(R)); ph[0]=0
    if n%2==0: ph[-1]=0
    return np.fft.irfft(np.abs(R)*np.exp(1j*ph),n=n)
def sigma_of(r):
    r=r[np.isfinite(r)]
    if len(r)<5000 or r.std()==0: return np.nan
    r=r/r.std()
    s=sig_kl(r); surr=np.nanmean([sig_kl(phase_rand(r,RNG)) for _ in range(5)])
    return max(0.0, s-surr)
rows=[]
# --- crypto 1m -> 5m blocks
for f in sorted(glob.glob('/tmp/venues/c1m_*.parquet')):
    d=pd.read_parquet(f,columns=['t','c'])
    c=d.c.to_numpy(float); c=c[::5]                      # 5-min sampling
    r=np.diff(np.log(c[c>0]))
    rows.append(dict(venue='crypto-perp',sym=Path(f).stem.replace('c1m_',''),n=len(r),sigma=sigma_of(r)))
    print(rows[-1],flush=True)
# --- FX 5m
for f in sorted(glob.glob('/tmp/venues/fx_fx2_*.parquet')):
    d=pd.read_parquet(f,columns=['close'])
    c=d.close.to_numpy(float); c=c[c>0]
    r=np.diff(np.log(c))
    r=r[r!=0]                                            # dukascopy pads flat ticks
    rows.append(dict(venue='fx',sym=Path(f).stem.replace('fx_fx2_',''),n=len(r),sigma=sigma_of(r)))
    print(rows[-1],flush=True)
# --- equities 5m (recompute at 5m for comparability)
D=Path('/tmp/databranch/data/equity_1m_alpaca')
for f in sorted(D.glob('*.parquet')):
    d=pd.read_parquet(f,columns=['t','c'])
    d['t']=pd.to_datetime(d.t,utc=True).dt.tz_convert('America/New_York')
    d=d[(d.t.dt.time>=pd.Timestamp('09:30').time())&(d.t.dt.time<=pd.Timestamp('16:00').time())]
    d['day']=d.t.dt.normalize()
    R=[]
    for _,g in d.groupby('day',sort=True):
        c=g.c.to_numpy(float)[::5]
        if len(c)<40 or c.min()<=0: continue
        x=np.diff(np.log(c))
        if x.std()>0: R.append(x/x.std())
    if len(R)<200: continue
    r=np.concatenate(R)
    rows.append(dict(venue='equity',sym=f.stem,n=len(r),sigma=sigma_of(r)))
    print(rows[-1],flush=True)
T=pd.DataFrame(rows); T.to_parquet('/tmp/venues/sigma_map.parquet',index=False)
print("\n=== SIGMA MAP (entropy production per 5-min step, scale-free) ===")
print(T.groupby('venue').sigma.agg(['count','median','mean','min','max']).round(5).to_string())
print()
print(T.sort_values('sigma',ascending=False).head(25).round(5).to_string(index=False))
