"""PREREG8: thermodynamic ceiling on alpha.
sigma from ordinal-pattern KL (fwd vs reversed), de-biased by phase-randomized
surrogates. Bound = sqrt(sigma/2). Matched empirical ceiling = pattern oracle."""
import numpy as np, pandas as pd
from pathlib import Path
D=Path('/tmp/databranch/data/equity_1m_alpaca')
SYMS=sorted(p.stem for p in D.glob('*.parquet'))
RNG=np.random.default_rng(808); M=3; NSURR=20

def codes(r,m=M):
    W=np.lib.stride_tricks.sliding_window_view(r,m)
    o=np.argsort(W,axis=1,kind='stable')
    return (o*(m**np.arange(m))).sum(1)

def sigma_kl(r,m=M):
    """coarse-grained entropy production per step: KL(P_fwd || P_rev), nats"""
    cf=codes(r,m); cr=codes(r[::-1],m)
    nb=m**m
    pf=np.bincount(cf,minlength=nb).astype(float); pr=np.bincount(cr,minlength=nb).astype(float)
    if pf.sum()<50: return np.nan
    pf/=pf.sum(); pr/=pr.sum()
    k=(pf>0)&(pr>0)
    return float((pf[k]*np.log(pf[k]/pr[k])).sum())

def phase_rand(r,rng):
    n=len(r); R=np.fft.rfft(r); ph=rng.uniform(0,2*np.pi,len(R)); ph[0]=0
    if n%2==0: ph[-1]=0
    return np.fft.irfft(np.abs(R)*np.exp(1j*ph),n=n)

def pattern_oracle(r,m=M):
    """best strategy in the SAME info set: position = sign(in-sample mean next
    return | current pattern). Look-ahead by construction -> an upper bound."""
    c=codes(r,m); nxt=r[m:]; c=c[:len(nxt)]
    if len(nxt)<200: return np.nan
    mu=np.zeros(m**m)
    for k in np.unique(c):
        s=nxt[c==k]
        if len(s)>=20: mu[k]=s.mean()
    pos=np.sign(mu[c])
    pnl=pos*nxt
    return float(pnl.mean()/pnl.std()) if pnl.std()>0 else np.nan

rows=[]
for sym in SYMS:
    d=pd.read_parquet(D/f'{sym}.parquet',columns=['t','c'])
    d['t']=pd.to_datetime(d.t,utc=True).dt.tz_convert('America/New_York')
    d=d[(d.t.dt.time>=pd.Timestamp('09:30').time())&(d.t.dt.time<=pd.Timestamp('16:00').time())]
    d['day']=d.t.dt.normalize().dt.tz_localize(None); d=d[d.day<'2024-01-01']
    R=[]
    for day,g in d.groupby('day',sort=True):
        c=g.c.to_numpy(float)
        if len(c)<300 or c.min()<=0: continue
        r=np.diff(np.log(c))
        if np.all(np.isfinite(r)) and r.std()>0: R.append(r/r.std())   # normalize per day
    if len(R)<200: continue
    r=np.concatenate(R)
    s_real=sigma_kl(r)
    surr=np.array([sigma_kl(phase_rand(r,RNG)) for _ in range(5)])
    sig=max(0.0, s_real-np.nanmean(surr))
    orc=pattern_oracle(r)
    # T4: same oracle on a surrogate (must collapse)
    orc_s=np.nanmean([pattern_oracle(phase_rand(r,RNG)) for _ in range(3)])
    rows.append(dict(sym=sym,n=len(r),sigma_raw=s_real,sigma_surr=float(np.nanmean(surr)),
                     sigma=sig,bound=np.sqrt(sig/2),oracle=orc,oracle_surr=orc_s))
    print(f"{sym} sigma={sig:.5f} bound={np.sqrt(sig/2):.4f} oracle={orc:.4f} orc_surr={orc_s:.4f}",flush=True)
T=pd.DataFrame(rows); T.to_parquet('/tmp/timbre/tur.parquet',index=False)
print()
from scipy import stats
print("T4 SURROGATE CONTROL (must run first)")
print(f"  real oracle Sharpe/step   mean {T.oracle.mean():.4f}")
print(f"  surrogate oracle          mean {T.oracle_surr.mean():.4f}")
print(f"  ratio surr/real           {T.oracle_surr.mean()/T.oracle.mean():.3f}  (must be well below 1)")
print(f"  VERDICT: {'PASS' if T.oracle_surr.mean() < 0.5*T.oracle.mean() else 'FAIL - oracle is noise'}")
print()
T['ratio']=T.oracle/T.bound
print("T1 VIOLATION TEST  (oracle must be <= bound for ALL 20)")
v=(T.oracle>T.bound).sum()
print(f"  violations: {v}/20   {'PASS' if v==0 else 'FAIL - TUR bound broken'}")
print(f"T2 TIGHTNESS  median ratio oracle/bound = {T.ratio.median():.3f}  (useful if >0.05)")
rho=stats.spearmanr(T.bound,T.oracle).statistic
print(f"T3 PREDICTIVE ORDERING  spearman(bound, oracle) = {rho:+.3f}  (need >+0.5)")
print()
print(T[['sym','sigma','bound','oracle','ratio']].sort_values('bound',ascending=False).round(4).to_string(index=False))
