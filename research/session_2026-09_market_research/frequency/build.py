"""Build per-stock-day spectral timbre features from Alpaca 1-min bars. PIT: only
bars <= 15:55 ET feed the signal; the trade happens at 16:00 close."""
import numpy as np, pandas as pd
from pathlib import Path

D = Path('/tmp/databranch/data/equity_1m_alpaca')
OUT = Path('/tmp/timbre'); OUT.mkdir(exist_ok=True)

def spectral(r):
    """r: 1-min log returns for one session. Returns timbre features."""
    n = len(r)
    if n < 60: return None
    r = r - r.mean()
    w = np.hanning(n)
    R = np.fft.rfft(r * w)
    P = (np.abs(R) ** 2)[1:]                 # drop DC
    if len(P) < 10 or not np.all(np.isfinite(P)): return None
    P = np.maximum(P, 1e-300)
    tot = P.sum()
    if tot <= 0: return None
    gm = np.exp(np.log(P).mean()); am = P.mean()
    flat = gm / am                            # Wiener entropy, scale-free
    f = np.arange(1, len(P) + 1) / (2.0 * len(P))   # normalised to Nyquist
    cent = float((f * P).sum() / tot)
    cum = np.cumsum(P) / tot
    roll = float(f[np.searchsorted(cum, 0.85)]) if cum[-1] >= 0.85 else float(f[-1])
    return float(flat), cent, roll

rows = []
for p in sorted(D.glob('*.parquet')):
    sym = p.stem
    d = pd.read_parquet(p, columns=['t','o','h','l','c','v','n'])
    d['t'] = pd.to_datetime(d.t, utc=True).dt.tz_convert('America/New_York')
    d = d[(d.t.dt.time >= pd.Timestamp('09:30').time()) &
          (d.t.dt.time <= pd.Timestamp('16:00').time())]
    d['day'] = d.t.dt.date
    for day, g in d.groupby('day'):
        g = g.sort_values('t')
        sig = g[g.t.dt.time <= pd.Timestamp('15:55').time()]
        if len(sig) < 120: continue
        c = sig.c.to_numpy(float)
        r = np.diff(np.log(c))
        r = r[np.isfinite(r)]
        if len(r) < 60: continue
        s = spectral(r)
        if s is None: continue
        flat, cent, roll = s
        close_1600 = float(g.c.iloc[-1])
        rows.append(dict(sym=sym, day=pd.Timestamp(day),
            flat=flat, cent=cent, roll=roll,
            dayret=float(np.log(close_1600 / sig.o.iloc[0])),
            rv=float(r.std() * np.sqrt(len(r))),
            close=close_1600, vol=float(g.v.sum()), ntr=float(g.n.sum()),
            rng=float((g.h.max() - g.l.min()) / close_1600), nbar=len(sig)))
    print(f'{sym}: {len([x for x in rows if x["sym"]==sym])} sessions', flush=True)

T = pd.DataFrame(rows).sort_values(['day','sym'])
# overnight = next open / this close, per symbol
T['next_open'] = T.groupby('sym').close.shift(-1) * np.nan
opens = {}
for p in sorted(D.glob('*.parquet')):
    sym = p.stem
    d = pd.read_parquet(p, columns=['t','o'])
    d['t'] = pd.to_datetime(d.t, utc=True).dt.tz_convert('America/New_York')
    d = d[(d.t.dt.time >= pd.Timestamp('09:30').time())]
    d['day'] = d.t.dt.date
    f = d.groupby('day').o.first()
    opens[sym] = pd.Series(f.values, index=pd.to_datetime(f.index))
T['nxt_open'] = [opens[r.sym].shift(-1).get(r.day, np.nan) for r in T.itertuples()]
T['overnight'] = np.log(T.nxt_open / T.close)
T = T[np.isfinite(T.overnight)]
T.to_parquet(OUT/'timbre.parquet', index=False)
print(f'\nBUILT {len(T):,} stock-days, {T.sym.nunique()} symbols, {T.day.min().date()} -> {T.day.max().date()}')
