import numpy as np, pandas as pd
from pathlib import Path
from scipy import stats
D=Path('/tmp/databranch/data/equity_1m_alpaca')
T8=pd.read_parquet('/tmp/timbre/tur.parquet').set_index('sym')
RNG=np.random.default_rng(909); M=3
def codes(r,m=M):
    W=np.lib.stride_tricks.sliding_window_view(r,m)
    return (np.argsort(W,axis=1,kind='stable')*(m**np.arange(m))).sum(1)
def oracle_oos(r,m=M):
    """fit mu_pi on first half, EVALUATE on second half"""
    c=codes(r,m); nxt=r[m:]; c=c[:len(nxt)]
    h=len(nxt)//2
    if h<500: return np.nan
    mu=np.zeros(m**m)
    for k in np.unique(c[:h]):
        s=nxt[:h][c[:h]==k]
        if len(s)>=50: mu[k]=s.mean()
    pos=np.sign(mu[c[h:]]); pnl=pos*nxt[h:]
    return float(pnl.mean()/pnl.std()) if pnl.std()>0 else np.nan
def phase_rand(r,rng):
    n=len(r); R=np.fft.rfft(r); ph=rng.uniform(0,2*np.pi,len(R)); ph[0]=0
    if n%2==0: ph[-1]=0
    return np.fft.irfft(np.abs(R)*np.exp(1j*ph),n=n)
rows=[]
for sym in T8.index:
    d=pd.read_parquet(D/f'{sym}.parquet',columns=['t','c'])
    d['t']=pd.to_datetime(d.t,utc=True).dt.tz_convert('America/New_York')
    d=d[(d.t.dt.time>=pd.Timestamp('09:30').time())&(d.t.dt.time<=pd.Timestamp('16:00').time())]
    d['day']=d.t.dt.normalize().dt.tz_localize(None); d=d[d.day<'2024-01-01']
    R=[]
    for day,g in d.groupby('day',sort=True):
        c=g.c.to_numpy(float)
        if len(c)<300 or c.min()<=0: continue
        r=np.diff(np.log(c))
        if np.all(np.isfinite(r)) and r.std()>0: R.append(r/r.std())
    if len(R)<200: continue
    r=np.concatenate(R)
    o=oracle_oos(r)
    o_sh=np.nanmean([oracle_oos(RNG.permutation(r)) for _ in range(3)])
    o_ph=np.nanmean([oracle_oos(phase_rand(r,RNG)) for _ in range(3)])
    rows.append(dict(sym=sym,bound=T8.loc[sym,'bound'],sigma=T8.loc[sym,'sigma'],
                     oos=o,shuf=o_sh,phase=o_ph,nonbounce=o-o_ph))
    print(f"{sym} oos={o:+.4f} shuffle={o_sh:+.4f} phaserand={o_ph:+.4f}",flush=True)
T=pd.DataFrame(rows); T.to_parquet('/tmp/timbre/tur2.parquet',index=False)
print()
print("T4' IID-SHUFFLE NULL (must run first)")
print(f"  real OOS oracle  mean {T.oos.mean():.5f}")
print(f"  IID-shuffle      mean {T.shuf.mean():.5f}   ratio {abs(T.shuf.mean()/T.oos.mean()):.3f}")
ok4=abs(T.shuf.mean())<0.25*abs(T.oos.mean())
print(f"  VERDICT: {'PASS' if ok4 else 'FAIL - achieved side still noise, v9 void'}")
if ok4:
    T['ratio']=T.oos/T.bound
    print(f"\nT1' violations (oos > bound): {(T.oos>T.bound).sum()}/{len(T)}")
    print(f"T2' median oos/bound = {T.ratio.median():.3f}")
    r1=stats.spearmanr(T.bound,T.oos).statistic
    r2=stats.spearmanr(T.bound,T.nonbounce).statistic
    print(f"T3' spearman(bound, oos)        = {r1:+.3f}")
    print(f"    spearman(bound, non-bounce) = {r2:+.3f}   <-- the real test")
print()
print(T.sort_values('bound',ascending=False).round(4).to_string(index=False))
