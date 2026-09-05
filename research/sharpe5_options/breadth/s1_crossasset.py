"""Cross-asset variance risk premium: are the vol premia independent?"""
import numpy as np, pandas as pd, sys
from common import *
np.set_printoptions(suppress=True)

# (name, IV index path, IV scale->annualized decimal vol, underlying path, kind)
SPECS = [
 ("EQ_SPX",  f"{SP}/vix_daily.json",   0.01, f"{SP}/etf_SPY.json",  "price"),
 ("EQ_NDX",  f"{BH}/IDX_VXN.json",     0.01, f"{SP}/stk_QQQ.json",  "price"),
 ("OIL",     f"{BH}/IDX_OVX.json",     0.01, f"{BH}/USO.json",      "price"),
 ("GOLD",    f"{BH}/IDX_GVZ.json",     0.01, f"{BH}/GLD.json",      "price"),
 ("FX_EUR",  f"{BH}/IDX_EVZ.json",     0.01, f"{BH}/FXE.json",      "price"),
 ("RATES",   f"{BH}/IDX_MOVE.json",    1.0,  f"{SP}/trend_TNX.json","yield"),
]

iv, rv_src, kind = {}, {}, {}
for nm, ivp, sc, up, kd in SPECS:
    iv[nm] = close(ivp, adjust=False) * sc
    if kd == "price":
        px = close(up, adjust=True)
        rv_src[nm] = np.log(px).diff()          # log returns
    else:
        y = close(up, adjust=False)             # ^TNX = yield in pct
        rv_src[nm] = y.diff() * 100.0           # daily change in basis points
    kind[nm] = kd

names = list(iv.keys())
# monthly, NON-OVERLAPPING: observe IV on last session of month m, realize over month m+1
rows = []
for nm in names:
    s_iv, s_r = iv[nm].dropna(), rv_src[nm].dropna()
    idxm = s_r.index.to_period('M')
    for per, grp in s_r.groupby(idxm):
        if len(grp) < 15: continue
        prev_end = s_iv.loc[:grp.index[0] - pd.Timedelta(days=1)]
        if len(prev_end) == 0: continue
        gap = (grp.index[0] - prev_end.index[-1]).days
        if gap > 7: continue
        iv0 = prev_end.iloc[-1]
        rv = grp.std(ddof=1) * np.sqrt(252)      # annualized, same units as IV index
        if not np.isfinite(iv0) or not np.isfinite(rv) or iv0 <= 0: continue
        rows.append(dict(asset=nm, per=per, iv=iv0, rv=rv,
                         vrp_var=iv0**2 - rv**2, vrp_vol=iv0 - rv, n=len(grp)))
P = pd.DataFrame(rows)
# normalize each asset's P&L to unit std so correlation/breadth is scale-free
piv_var = P.pivot(index='per', columns='asset', values='vrp_var')[names]
piv_vol = P.pivot(index='per', columns='asset', values='vrp_vol')[names]

print("="*78); print("STUDY 1 - CROSS-ASSET VARIANCE RISK PREMIUM (monthly, non-overlapping)"); print("="*78)
print("\nCoverage & standalone economics (short-vol carry, per asset):")
print(f"{'asset':8} {'months':>7} {'start':>9} {'end':>9} {'meanIV':>7} {'meanRV':>7} {'VRPvol':>7} {'t':>6} {'Sharpe_ann':>10}")
for nm in names:
    s = P[P.asset == nm]
    x = s.vrp_var.values
    xv = s.vrp_vol.values
    t = x.mean()/ (x.std(ddof=1)/np.sqrt(len(x)))
    sh = x.mean()/x.std(ddof=1)*np.sqrt(12)
    print(f"{nm:8} {len(s):7d} {str(s.per.iloc[0]):>9} {str(s.per.iloc[-1]):>9} "
          f"{s.iv.mean():7.3f} {s.rv.mean():7.3f} {xv.mean():7.3f} {t:6.2f} {sh:10.2f}")

for label, piv in (("VARIANCE units (IV^2 - RV^2)", piv_var), ("VOL units (IV - RV)", piv_vol)):
    D = piv.dropna()
    C = D.corr()
    print(f"\n--- {label} | common window {D.index[0]}..{D.index[-1]}  n={len(D)} months")
    print(C.round(3).to_string())
    rbar = mean_offdiag(C.values)
    N = len(names)
    print(f"mean pairwise corr           = {rbar:+.3f}")
    print(f"N nominal                    = {N}")
    print(f"N_eff (participation ratio)  = {eff_n_pr(C.values):.2f}")
    print(f"N_eff (equicorrelated)       = {eff_n_equi(N, rbar):.2f}")
    # drop the redundant second equity series
    sub = [n for n in names if n != 'EQ_NDX']
    Cs = D[sub].corr()
    print(f"[ex-NDX, 5 distinct classes] mean corr = {mean_offdiag(Cs.values):+.3f}, "
          f"N_eff(PR) = {eff_n_pr(Cs.values):.2f}, N_eff(equi) = {eff_n_equi(len(sub), mean_offdiag(Cs.values)):.2f}")

# equal-risk basket Sharpe vs components (the Addendum-16 test)
D = piv_var.dropna()
Z = (D - D.mean()) / D.std(ddof=1)
bask = Z.mean(axis=1)
print("\n--- Diversification test: equal-risk basket of the 6 short-vol carries")
print(f"{'asset':8} {'Sharpe_ann':>10}")
for nm in names:
    x = D[nm].values; print(f"{nm:8} {x.mean()/x.std(ddof=1)*np.sqrt(12):10.2f}")
# basket in raw risk-normalized units: use the standardized-sum but with real means
W = D / D.std(ddof=1)          # unit-vol weights
bk = W.mean(axis=1)
print(f"{'BASKET':8} {bk.mean()/bk.std(ddof=1)*np.sqrt(12):10.2f}   <- realized")
# what it would be if independent
ind = np.mean([D[n].mean()/D[n].std(ddof=1) for n in names]) * np.sqrt(12) * np.sqrt(len(names))
print(f"{'(if indep)':8} {ind:10.2f}   <- sqrt(N) benchmark")
D.to_csv("out_s1_vrp_var.csv"); piv_vol.to_csv("out_s1_vrp_vol.csv")
