"""ATI: Arrow-of-Time Index from ordinal patterns, with the phase-randomized
surrogate null (identical power spectrum, time-reversible by construction)."""
import numpy as np, pandas as pd
from pathlib import Path
D=Path('/tmp/databranch/data/equity_1m_alpaca')
SYMS=sorted(p.stem for p in D.glob('*.parquet'))
RNG=np.random.default_rng(606)
NSURR=20; M=3

def pat_hist(r, m=M):
    """histogram of ordinal patterns of length m, as probabilities over m! bins"""
    n=len(r)-m+1
    if n<20: return None
    W=np.lib.stride_tricks.sliding_window_view(r,m)          # (n,m)
    order=np.argsort(W,axis=1,kind='stable')                 # permutation per row
    # encode permutation as base-m integer, then map to dense 0..m!-1
    code=(order*(m**np.arange(m))).sum(1)
    h=np.bincount(code,minlength=m**m).astype(float)
    h=h[h.sum()>0 or slice(None)] if False else h
    return h/max(h.sum(),1)

def ati_raw(r):
    pf=pat_hist(r); pr=pat_hist(r[::-1])
    if pf is None or pr is None: return np.nan
    return 0.5*np.abs(pf-pr).sum()

def phase_randomize(r, rng):
    n=len(r); R=np.fft.rfft(r)
    ph=rng.uniform(0,2*np.pi,len(R)); ph[0]=0
    if n%2==0: ph[-1]=0
    return np.fft.irfft(np.abs(R)*np.exp(1j*ph), n=n)

rows=[]
for sym in SYMS:
    d=pd.read_parquet(D/f'{sym}.parquet',columns=['t','c'])
    d['t']=pd.to_datetime(d.t,utc=True).dt.tz_convert('America/New_York')
    d=d[(d.t.dt.time>=pd.Timestamp('09:30').time())&(d.t.dt.time<=pd.Timestamp('15:55').time())]
    d['day']=d.t.dt.normalize().dt.tz_localize(None); d=d[d.day<'2024-01-01']
    for day,g in d.groupby('day',sort=True):
        c=g.c.to_numpy(float)
        if len(c)<300 or c.min()<=0: continue
        r=np.diff(np.log(c))
        if not np.all(np.isfinite(r)) or np.std(r)==0: continue
        a=ati_raw(r)
        if not np.isfinite(a): continue
        s=np.array([ati_raw(phase_randomize(r,RNG)) for _ in range(NSURR)])
        s=s[np.isfinite(s)]
        if len(s)<10: continue
        mu,sd=s.mean(),s.std(ddof=1)
        rows.append(dict(sym=sym,day=day,ati_raw=a,surr_mu=mu,surr_sd=sd,
                         ati=a-mu, z_ati=(a-mu)/sd if sd>0 else np.nan, n=len(r)))
    print(f'{sym} ok ({len(rows):,})',flush=True)
A=pd.DataFrame(rows); A.to_parquet('/tmp/timbre/ati.parquet',index=False)
print(f"\n{len(A):,} stock-days")
print("=== N0 GO/NO-GO: real ATI vs phase-randomized surrogate (same power spectrum) ===")
print(f"  ATI_raw mean            {A.ati_raw.mean():.5f}")
print(f"  surrogate mean          {A.surr_mu.mean():.5f}")
print(f"  ATI (excess) mean       {A.ati.mean():+.5f}")
print(f"  mean z_ATI              {A.z_ati.mean():+.3f}   <-- must exceed +3")
dm=A.groupby('day').z_ati.mean()
print(f"  day-clustered t on z    {dm.mean()/(dm.std(ddof=1)/np.sqrt(len(dm))):+.2f}  ({len(dm)} days)")
print(f"  share of stock-days with z>2: {(A.z_ati>2).mean():.1%}")
print(f"  VERDICT: {'PASS - the tape is time-irreversible' if A.z_ati.mean()>3 else 'FAIL - time-reversible, idea dead'}")
