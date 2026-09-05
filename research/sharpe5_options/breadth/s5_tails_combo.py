"""Tail-conditional correlation, and the best achievable combined book."""
import numpy as np, pandas as pd
from common import *

D = pd.read_csv("out_s1_vrp_var.csv", index_col=0)
names = list(D.columns)
Z = (D - D.mean())/D.std(ddof=1)
print("="*78); print("STUDY 5A - DOES CROSS-ASSET DIVERSIFICATION SURVIVE THE TAIL?"); print("="*78)
agg = Z.mean(axis=1)
for lbl, mask in [("all months", np.ones(len(Z),bool)),
                  ("worst tercile of the aggregate", agg <= agg.quantile(1/3)),
                  ("worst decile of the aggregate", agg <= agg.quantile(0.10))]:
    C = Z[mask].corr()
    print(f"{lbl:32} n={mask.sum():4d}  mean pairwise corr = {mean_offdiag(C.values):+.3f}"
          f"   N_eff(PR) = {eff_n_pr(C.values):.2f}")
# co-exceedance: when equity short-vol has its worst 10% month, how do others do?
bad = Z.EQ_SPX <= Z.EQ_SPX.quantile(0.10)
print("\nIn the worst decile of EQUITY short-vol months, mean z-score of each asset:")
for n in names: print(f"  {n:8} {Z.loc[bad,n].mean():+.2f} sigma   (loss rate {(Z.loc[bad,n]<0).mean():.0%})")

print("\n"+"="*78); print("STUDY 5B - BEST HONEST COMBINED BOOK FROM MEASURED CORRELATIONS"); print("="*78)
# take the measured standalone Sharpes and the measured correlation matrix,
# ask what an optimally-weighted book of these sleeves could reach.
mu = D.mean().values * 12
sg = D.std(ddof=1).values * np.sqrt(12)
S  = mu/sg
C  = D.corr().values
Cov = np.outer(sg,sg)*C
w = np.linalg.solve(Cov, mu); w = w/np.abs(w).sum()
port_S = (w@mu)/np.sqrt(w@Cov@w)
print("standalone annual Sharpes: " + "  ".join(f"{n}={s:.2f}" for n,s in zip(names,S)))
print(f"equal-risk book Sharpe        = {(Z.mean(axis=1).mean()/Z.mean(axis=1).std(ddof=1))*np.sqrt(12):.2f}")
print(f"IN-SAMPLE mean-variance optimal Sharpe = {port_S:.2f}   (upper bound, uses future means)")
print(f"if the 6 sleeves were INDEPENDENT with these Sharpes: {np.sqrt((S**2).sum()):.2f}")
print(f"gap attributable purely to correlation: {np.sqrt((S**2).sum()) - port_S:.2f} Sharpe points")

print("\n"+"="*78); print("STUDY 5C - REQUIRED IC AT EACH MEASURED BREADTH (target Sharpe 5)"); print("="*78)
cands = [
 ("Single underlying, monthly expiries (measured)",            11.4),
 ("Single underlying, weekly expiries (assume indep.)",        52.0),
 ("Cross-asset vol premia, 5 classes x 12 (measured N_eff)",   12*1.87),
 ("Cross-asset, generous PR breadth 2.85 x 12",                12*2.85),
 ("Earnings, 500 names/qtr, rho=0.168 (idio-vol CIV)",         4*eff_n_equi(500,0.168)),
 ("Earnings, 500 names/qtr, rho=0.072 (measured event P&L)",   4*eff_n_equi(500,0.072)),
 ("Earnings, 500 names/qtr, rho=0.045 (measured ICC, floor)",  4*eff_n_equi(500,0.045)),
 ("Earnings, hypothetical rho=0.01",                           4*eff_n_equi(500,0.01)),
 ("38-name EOD option book (prior REPORT.md)",                 12*11.7),
 ("Market making, 1,000 indep. fills/day x 252",               252*1000.0),
]
print(f"{'structure':58} {'BR/yr':>9} {'IC needed':>10} {'x best clean IC (0.033)':>24}")
for lbl, br in cands:
    ic = required_ic(5.0, br)
    print(f"{lbl:58} {br:9.0f} {ic:10.3f} {ic/0.033:>21.0f}x" + ("  IMPOSSIBLE(>1)" if ic>1 else ""))
