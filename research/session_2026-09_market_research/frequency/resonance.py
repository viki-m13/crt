"""PREREG3 H1: is the intraday reversion oscillator better when TUNED to the
stock-day's own dominant period tau* than at the conventional fixed 3/12 band?
Strictly causal: tau* from the MORNING half only, evaluated on the AFTERNOON half.
TEST (2024+) is filtered out and never touched."""
import numpy as np, pandas as pd
from pathlib import Path
from scipy.signal import lfilter

D = Path('/tmp/databranch/data/equity_1m_alpaca')
SYMS = sorted(p.stem for p in D.glob('*.parquet'))
RNG = np.random.default_rng(20260904)

def ema(x, n):
    a = 2.0/(n+1.0)
    return lfilter([a], [1.0, -(1.0-a)], x, zi=[(1.0-a)*x[0]])[0]

def osc(c, fast, slow):
    return ema(c, fast) - ema(c, slow)

def tau_star(r):
    """Dominant period (minutes) of detrended 1-min returns, restricted 4..60 min."""
    n = len(r)
    if n < 60: return np.nan
    x = (r - r.mean()) * np.hanning(n)
    P = np.abs(np.fft.rfft(x))**2
    f = np.fft.rfftfreq(n, d=1.0)          # cycles per minute
    with np.errstate(divide='ignore'):
        per = 1.0/f
    m = np.isfinite(per) & (per >= 4.0) & (per <= 60.0)
    if not m.any(): return np.nan
    idx = np.flatnonzero(m)[np.argmax(P[m])]
    return float(per[idx])

rows = []
for sym in SYMS:
    d = pd.read_parquet(D/f'{sym}.parquet', columns=['t','c'])
    d['t'] = pd.to_datetime(d.t, utc=True).dt.tz_convert('America/New_York')
    d = d[(d.t.dt.time >= pd.Timestamp('09:30').time()) & (d.t.dt.time <= pd.Timestamp('16:00').time())]
    d['day'] = d.t.dt.normalize().dt.tz_localize(None)
    d = d[d.day < pd.Timestamp('2024-01-01')]          # TEST LOCKED
    d['mn'] = d.t.dt.hour*60 + d.t.dt.minute
    for day, g in d.groupby('day', sort=False):
        g = g.sort_values('mn')
        c = g.c.to_numpy(float); mn = g.mn.to_numpy()
        if len(c) < 300: continue
        r = np.diff(np.log(c))
        if not np.all(np.isfinite(r)): continue
        mm = mn[1:]
        morn = r[mm <= 750]                 # <=12:30
        aft  = r[mm >  750]
        if len(morn) < 100 or len(aft) < 150: continue
        t_star = tau_star(morn)
        if not np.isfinite(t_star): continue
        t_rand = float(RNG.uniform(4.0, 60.0))          # control R1
        C = np.cumsum(aft)
        s_fix  = osc(C, 3.0, 12.0)[:-1]
        s_tun  = osc(C, max(2.0, t_star/4.0), t_star)[:-1]
        s_rnd  = osc(C, max(2.0, t_rand/4.0), t_rand)[:-1]
        Cf = np.concatenate(([0.0], C))                 # Cf[i] = sum of first i returns
        rv = float(np.std(r))
        for k in (15, 30, 60):
            i = np.arange(len(s_fix))
            hi = i + 1 + k
            ok = hi <= len(aft)
            if ok.sum() < 50: continue
            fut = Cf[hi[ok]] - Cf[i[ok] + 1]
            def ic(s):
                sv = s[ok]
                if np.std(sv) == 0 or np.std(fut) == 0: return np.nan
                return float(np.corrcoef(sv, fut)[0,1])
            rows.append(dict(sym=sym, day=day, k=k, tau=t_star, rv=rv,
                             ic_fix=ic(s_fix), ic_tun=ic(s_tun), ic_rnd=ic(s_rnd)))
    print(f'{sym} done ({len(rows):,} rows)', flush=True)

R = pd.DataFrame(rows).dropna()
R.to_parquet('/tmp/timbre/resonance.parquet', index=False)
print(f'\n{len(R):,} stock-day-k rows, {R.day.nunique():,} days, {R.sym.nunique()} symbols')
print('tau* distribution:', np.round(R.tau.describe()[['mean','25%','50%','75%']].values, 1))
