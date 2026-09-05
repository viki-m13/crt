"""The factor vs residual decomposition: does a CROSS-SECTIONAL bet unlock breadth,
   and is there a measurable cross-sectional IC to spend it on?"""
import numpy as np, pandas as pd
from common import *
from scipy import stats
EV = pd.read_csv("out_s2_events.csv", parse_dates=['date']); EV['per']=pd.PeriodIndex(EV.per,freq='Q')
EV['lz'] = np.log(EV.z.abs().clip(lower=1e-3))
piv = EV.pivot_table(index='per', columns='name', values='lz').dropna(axis=1, thresh=70)
piv = piv.dropna(thresh=int(0.9*piv.shape[1]))
print(f"panel {piv.shape[1]} names x {piv.shape[0]} quarters (log realized event move, in trailing sigma)")

print("\n"+"="*78); print("STUDY 7A - FACTOR vs RESIDUAL BREADTH"); print("="*78)
R_ = piv.sub(piv.mean(axis=1), axis=0)     # cross-sectionally demeaned = factor-neutral
for lbl, X in (("RAW (directional book)", piv), ("CROSS-SECTIONAL (season-demeaned)", R_)):
    C = X.corr(min_periods=25).fillna(0); r = mean_offdiag(C.values); N=C.shape[0]
    ew = X.mean(axis=1); nef = X.std(ddof=1).mean()**2 / ew.var()
    print(f"{lbl:36} rho_bar {r:+.4f}  N_eff(formula) {eff_n_equi(N,r):6.1f}  N_eff(measured) {nef:6.1f}  of {N}")
print("\nMechanism: an equicorrelated panel has ONE large eigen-direction (the vol factor,")
print("eigenvalue 1+(N-1)rho) and N-1 small ones (eigenvalue 1-rho). A directional short-vol")
print("book puts all its alpha in the first; only a factor-neutral book reaches the other N-1.")

print("\n"+"="*78); print("STUDY 7B - IS THERE A CROSS-SECTIONAL IC TO SPEND IT ON?"); print("="*78)
# Signal: a name's own past event-move richness (does |z| persist across its own earnings?)
# All cross-sectional, all past-only.  IC = rank corr of signal_t with -|z|_{t+1}.
sig = piv.shift(1)                       # last quarter's log move
sig2 = piv.rolling(4, min_periods=3).mean().shift(1)   # 1-yr trailing mean
res = {}
for nm, S in (("lagged 1q |z|", sig), ("trailing 4q mean |z|", sig2)):
    ics=[]
    for q in piv.index:
        a, b = S.loc[q], piv.loc[q]
        m = a.notna() & b.notna()
        if m.sum() < 30: continue
        ics.append(stats.spearmanr(a[m], -b[m]).statistic)   # high past move -> expect loss? sign check
    ics=np.array(ics); res[nm]=ics
    t = ics.mean()/(ics.std(ddof=1)/np.sqrt(len(ics)))
    print(f"{nm:22} IC {ics.mean():+.4f}  sd {ics.std(ddof=1):.3f}  n_qtrs {len(ics)}  t {t:+.2f}"
          f"  -> IR = IC*sqrt(BR) = {abs(ics.mean())*np.sqrt(4*(eff_n_equi(piv.shape[1], max(mean_offdiag(R_.corr(min_periods=25).fillna(0).values),1e-6)))):.2f}")
print("\n(sign convention: positive IC means 'a name whose last event move was LARGE has a")
print(" SMALLER next move', i.e. mean reversion. Negative means persistence -- either is")
print(" tradable, only the magnitude matters for breadth arithmetic.)")

print("\n"+"="*78); print("STUDY 7C - WHAT THE CROSS-SECTIONAL ROUTE NEEDS"); print("="*78)
Cr = R_.corr(min_periods=25).fillna(0); rr = mean_offdiag(Cr.values); Nn = Cr.shape[0]
nef = eff_n_equi(Nn, max(rr,1e-6))
for Nuni, lbl in [(Nn, f"{Nn} names (measured panel)"), (500, "500 names (full S&P 500)")]:
    ne = eff_n_equi(Nuni, max(rr,1e-6)); br = 4*ne
    print(f"{lbl:34} N_eff {ne:6.1f}  BR/yr {br:6.0f}  IC needed for Sharpe 5 = {5/np.sqrt(br):.3f}")
best = max(abs(res[k].mean()) for k in res)
print(f"\nmeasured cross-sectional IC here: {best:.4f}")
print(f"published cross-sectional option ICs (Goyal-Saretto IV-RV, An et al. skew): ~0.02-0.05")
for br,lbl in [(4*eff_n_equi(500,max(rr,1e-6)), "500-name cross-sectional earnings book")]:
    for ic in (best, 0.03, 0.05, 0.10):
        print(f"  {lbl}: IC {ic:.3f} -> IR {ic*np.sqrt(br):.2f}")

print("\n--- cost hurdle (the reason this still fails in practice)")
sd_ev = EV.groupby('name').z.apply(lambda s: s.abs().std()).mean()
print(f"per-event P&L sd (units of premium) = {(piv.std(ddof=1).mean()):.3f} in log-move terms")
print("A single-name earnings straddle quotes ~5-15% wide. A cross-sectional book crosses")
print("that spread on BOTH legs, ~4 option legs per pair, i.e. ~10-30% of premium round trip.")
print("The gross edge from IC 0.03 on a structure whose P&L sd is ~40% of premium is")
print(f"  0.03 * 0.40 = {0.03*0.40:.3f} = {0.03*0.40*100:.1f}% of premium per event,")
print("against a 10-30% toll. The cost is 8-25x the gross edge.")
