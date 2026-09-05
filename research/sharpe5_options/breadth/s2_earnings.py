"""Per-EVENT structures: are earnings vol-crush bets independent across names?"""
import numpy as np, pandas as pd, glob, os, re
from common import *

files = sorted(glob.glob(f"{BH}/n_*.json"))
px = {}
for f in files:
    s = re.match(r"n_(.+)\.json", os.path.basename(f)).group(1)
    try:
        c = close(f, adjust=True)
    except Exception: continue
    if len(c) < 2500: continue
    px[s] = c
spy = close(f"{SP}/etf_SPY.json", adjust=True)
R = pd.DataFrame(px).apply(np.log).diff()
Rm = np.log(spy).diff().reindex(R.index)
R = R.loc['2005-01-01':]; Rm = Rm.loc['2005-01-01':]
R = R.dropna(axis=1, thresh=int(0.9*len(R)))
print(f"universe: {R.shape[1]} names, {R.shape[0]} sessions, {R.index[0].date()}..{R.index[-1].date()}")

# --- idiosyncratic returns: per-name annual beta to SPY (past-only would matter for a
#     trading rule; here we only need a variance decomposition, so full-year beta is fine)
E = pd.DataFrame(index=R.index, columns=R.columns, dtype=float)
BETA = {}
for yr, idx in R.groupby(R.index.year).groups.items():
    rm = Rm.loc[idx]
    for c in R.columns:
        y = R.loc[idx, c]
        m = y.notna() & rm.notna()
        if m.sum() < 100: continue
        b = np.cov(y[m], rm[m])[0,1] / np.var(rm[m], ddof=1)
        E.loc[idx, c] = y - b*rm
        BETA[(yr,c)] = b
E = E.astype(float)

print("\n"+"="*78); print("STUDY 2A - COMMON FACTOR IN IDIOSYNCRATIC VOLATILITY (Herskovic CIV)"); print("="*78)
# monthly idio vol per name -> log changes -> correlation
IV = E.resample('ME').std(ddof=1) * np.sqrt(21)
IV = IV[IV.count(axis=1) > 40]
LD = np.log(IV).diff().dropna(how='all')
LD = LD.dropna(axis=1, thresh=int(0.8*len(LD)))
C = LD.corr(min_periods=60)
rbar = mean_offdiag(C.values); N = C.shape[0]
print(f"names={N}  months={len(LD)}")
print(f"mean pairwise corr of monthly log idio-vol CHANGES = {rbar:+.3f}")
print(f"N_eff(equi) = {eff_n_equi(N, rbar):.1f}   (Herskovic et al. 2016 report ~0.2-0.3)")
Z = LD.fillna(0.0)
w = np.linalg.eigvalsh(np.corrcoef(Z.T.values))[::-1]
print(f"PC1 share of variance = {w[0]/w.sum():.1%}   PC1-3 = {w[:3].sum()/w.sum():.1%}")

print("\n"+"="*78); print("STUDY 2B - DETECTED EARNINGS EVENTS: cross-name dependence"); print("="*78)
# detect earnings day = max |idio ret| within each fiscal-quarter window per name.
# Accepted limitation: no true announcement dates locally (see report).
tr_sig = E.rolling(63, min_periods=40).std(ddof=1).shift(1)   # trailing, past-only
ev = []
q = pd.PeriodIndex(E.index, freq='Q')
for c in E.columns:
    e = E[c]
    for per in pd.unique(q):
        m = (q == per) & e.notna() & tr_sig[c].notna()
        if m.sum() < 40: continue
        sl = e[m]; d = sl.abs().idxmax()
        s0 = tr_sig.loc[d, c]
        if not np.isfinite(s0) or s0 <= 0: continue
        ev.append(dict(name=c, per=per, date=d, idio=e.loc[d], z=e.loc[d]/s0,
                       mkt=Rm.loc[d], tot=R.loc[d, c]))
EV = pd.DataFrame(ev)
print(f"detected events: {len(EV)}  names={EV.name.nunique()}  quarters={EV.per.nunique()}")
print(f"median |z| at event = {EV.z.abs().median():.2f}  (a real earnings move is ~3-5 trailing sigma)")

# short-straddle P&L proxy: sell an implied move of k*sigma, pay |realized idio move|
# P&L (in trailing-sigma units) = k - |z| ; correlation structure is invariant to k.
EV['pl'] = -EV.z.abs()
piv = EV.pivot_table(index='per', columns='name', values='pl')
piv = piv.dropna(axis=1, thresh=int(0.85*len(piv)))
Cq = piv.corr(min_periods=25)
rq = mean_offdiag(Cq.values); Nq = Cq.shape[0]
print(f"\nquarterly panel: {Nq} names x {len(piv)} quarters")
print(f"mean pairwise corr of per-name quarterly event P&L = {rq:+.3f}")
print(f"N_eff(equi) = {eff_n_equi(Nq, rq):.1f} of {Nq} nominal")

# ICC via one-way random effects on the pooled event panel (season = calendar quarter)
g = EV.groupby('per')['pl']
k = g.count(); gm = EV.pl.mean()
n0 = (k.sum() - (k**2).sum()/k.sum()) / (len(k)-1)
msb = (k*(g.mean()-gm)**2).sum()/(len(k)-1)
msw = EV.groupby('per')['pl'].apply(lambda s: ((s-s.mean())**2).sum()).sum()/(k.sum()-len(k))
icc = (msb-msw)/(msb+(n0-1)*msw)
print(f"\nICC (season random effect) = {icc:+.4f}   [avg names per season n0={n0:.0f}]")
for Ncal in (100, 500):
    print(f"  breadth from {Ncal} names in one season: N_eff = {eff_n_equi(Ncal, max(icc,0)):.1f}"
          f"  -> per year (x4) = {4*eff_n_equi(Ncal, max(icc,0)):.0f}")

print("\n--- variance decomposition of the event-day move")
tv = EV.tot.var(); iv_ = EV.idio.var(); mv = (EV.tot-EV.idio).var()
print(f"total var {tv:.6f} = idio {iv_:.6f} ({iv_/tv:.1%}) + market-beta {mv:.6f} ({mv/tv:.1%})")
print("-> on the DAY, an earnings move is overwhelmingly idiosyncratic. The dependence")
print("   that matters is in the PRICING level, measured in 2A/2C, not the realized move.")

print("\n"+"="*78); print("STUDY 2C - are event days spread out, and does season vol drive them?"); print("="*78)
EV['ym'] = EV.date.dt.to_period('M')
den = EV.groupby('ym').size()
print(f"events per month: median {den.median():.0f}, p90 {den.quantile(.9):.0f}, max {den.max()}")
# season-level mean z vs season-level market vol (the common pricing factor)
vix = close(f"{SP}/vix_daily.json", adjust=False)
sm = EV.groupby('per').agg(mz=('z', lambda s: s.abs().mean()), d=('date','mean'))
vq = vix.resample('QE').mean(); vq.index = vq.index.to_period('Q')
sm['vix'] = vq.reindex(sm.index)
sm = sm.dropna()
c1 = np.corrcoef(sm.mz, sm.vix)[0,1]
print(f"corr(season mean |z|, season mean VIX) = {c1:+.3f}  over {len(sm)} quarters")
print(f"season mean |z| : std {sm.mz.std():.3f} vs within-season std {EV.groupby('per').z.apply(lambda s: s.abs().std()).mean():.3f}")
EV.to_csv("out_s2_events.csv", index=False)
