"""Per-EXPIRY serial independence, and term-structure RV orthogonality."""
import numpy as np, pandas as pd
from common import *

vix  = close(f"{SP}/vix_daily.json",   adjust=False)
v9   = close(f"{SP}/vix9d_daily.json", adjust=False)
v3m  = close(f"{SP}/vix3m_daily.json", adjust=False)
vvix = close(f"{SP}/vvix_daily.json",  adjust=False)
spy  = close(f"{SP}/etf_SPY.json",     adjust=True)
r    = np.log(spy).diff().dropna()

print("="*78); print("STUDY 3 - PER-EXPIRY BETS ON ONE UNDERLYING: is each expiry a new bet?"); print("="*78)
# non-overlapping monthly short-variance P&L on SPX
rows=[]
for per, g in r.groupby(r.index.to_period('M')):
    if len(g) < 15: continue
    pv = vix.loc[:g.index[0]-pd.Timedelta(days=1)]
    if len(pv)==0 or (g.index[0]-pv.index[-1]).days>7: continue
    iv0 = pv.iloc[-1]/100.0
    rv  = g.std(ddof=1)*np.sqrt(252)
    rows.append(dict(per=per, pl=iv0**2-rv**2, iv=iv0, rv=rv))
M = pd.DataFrame(rows).set_index('per')
x = M.pl.values
print(f"non-overlapping monthly bets: n={len(x)}  {M.index[0]}..{M.index[-1]}")
print(f"mean {x.mean():+.5f}  sd {x.std(ddof=1):.5f}  Sharpe_ann {x.mean()/x.std(ddof=1)*np.sqrt(12):.2f}")
ac=[np.corrcoef(x[:-k],x[k:])[0,1] for k in range(1,13)]
print("autocorr of the monthly P&L, lags 1..12:")
print("  " + "  ".join(f"{a:+.3f}" for a in ac))
vr = 1 + 2*sum((1-k/12)*ac[k-1] for k in range(1,12))
print(f"variance-ratio inflation factor 1+2*sum(rho) (Bartlett, 12 lags) = {vr:.3f}")
print(f"=> effective independent bets per 12 nominal monthly expiries = {12/max(vr,1e-6):.2f}")

# tail clustering: where the money is actually lost
q = np.quantile(x, 0.10)
bad = M.index[x <= q]
bt = np.array([p.ordinal for p in bad])
gaps = np.diff(np.sort(bt))
epi = 1 + (gaps > 3).sum()          # >3 months apart = separate episode
print(f"\nworst-decile months: {len(bad)}; independent loss EPISODES (>3mo apart): {epi}")
print(f"share of total downside variance in those months: "
      f"{(np.minimum(x,0)**2)[x<=q].sum()/ (np.minimum(x,0)**2).sum():.1%}")
print(f"=> {len(x)/12:.0f} years of monthly expiries deliver ~{epi} independent tail events "
      f"({epi/(len(x)/12):.2f}/yr)")

print("\n"+"="*78); print("STUDY 4 - TERM-STRUCTURE / CALENDAR RV: a different bet, or the same one?"); print("="*78)
df = pd.concat(dict(v9=v9, vix=vix, v3m=v3m, vvix=vvix), axis=1).dropna()
print(f"VIX9D/VIX/VIX3M common window {df.index[0].date()}..{df.index[-1].date()} n={len(df)}")
# daily P&L proxies (vega-neutral-ish, in vol points):
# OUTRIGHT short vol   : -d(VIX)
# CALENDAR (short front/long back, slope RV): -d(VIX9D - VIX3M)
# BUTTERFLY (curvature) : -d(VIX9D - 2*VIX + VIX3M)
d = df.diff().dropna()
P = pd.DataFrame({
    'outright' : -d.vix,
    'calendar' : -(d.v9 - d.v3m),
    'butterfly': -(d.v9 - 2*d.vix + d.v3m),
    'volofvol' : -d.vvix,
})
W = P.resample('W-FRI').sum().dropna()
C = W.corr()
print("\nweekly correlation of RV structure P&L proxies:")
print(C.round(3).to_string())
print(f"\ncorr(calendar, outright)  = {C.loc['calendar','outright']:+.3f}")
print(f"corr(butterfly, outright) = {C.loc['butterfly','outright']:+.3f}")
# how much of the calendar bet is just the outright bet?
b = np.polyfit(W.outright, W.calendar, 1)[0]
resid = W.calendar - b*W.outright
print(f"beta(calendar on outright) = {b:+.3f}; R^2 = {1-resid.var()/W.calendar.var():.3f}")
print(f"=> residual (genuinely new) share of calendar variance = {resid.var()/W.calendar.var():.1%}")
sub=['outright','calendar','butterfly']
Cs=C.loc[sub,sub]
print(f"N_eff(PR) over {{outright, calendar, butterfly}} = {eff_n_pr(Cs.values):.2f} of 3")
M.to_csv("out_s3_monthly.csv"); W.to_csv("out_s4_rv.csv")
