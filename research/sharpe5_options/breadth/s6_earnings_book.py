"""The Addendum-16 basket test applied to an earnings vol-crush book.

P&L per event, per unit of premium sold:  pl = 1 - |z|/k
where |z| = realized idio move / trailing idio sigma, and k = the implied event move
the market charges, in the same units.  Note that book_Sharpe / single_name_Sharpe
= sd_single / sd_book, which is INVARIANT to k -- so the breadth conclusion does not
depend on the calibration.  k is fixed only so the levels are readable, by imposing
the published (small) earnings-straddle variance premium of ~+3% of premium.
"""
import numpy as np, pandas as pd
from common import *
EV = pd.read_csv("out_s2_events.csv", parse_dates=['date'])
EV['per'] = pd.PeriodIndex(EV.per, freq='Q')
vix = close(f"{SP}/vix_daily.json", adjust=False)
m = EV.z.abs().mean(); K = m/0.97
EV['pl'] = 1 - EV.z.abs()/K
print(f"mean|z| = {m:.3f}; implied-move multiple k set to {K:.3f} -> book mean "
      f"{EV.pl.mean():+.4f} of premium per event (imposed, not measured)")

print("\n"+"="*78); print("STUDY 6 - DOES BREADTH ACTUALLY BUY SHARPE IN AN EARNINGS BOOK?"); print("="*78)
sn, sds = [], []
for n, g in EV.groupby('name'):
    if len(g) < 40: continue
    sn.append(g.pl.mean()/g.pl.std(ddof=1)*np.sqrt(4)); sds.append(g.pl.std(ddof=1))
sn, sds = np.array(sn), np.array(sds)
print(f"single-name Sharpe: median {np.median(sn):+.2f}  mean {sn.mean():+.2f}  (n={len(sn)} names, 4 events/yr)")

rows=[]
for Nsub in (1,3,5,10,20,40,60,119):
    rng = np.random.default_rng(7); shs=[]; sd_b=[]
    reps = 300 if Nsub < 119 else 1
    for _ in range(reps):
        cols = rng.choice(EV.name.unique(), size=min(Nsub, EV.name.nunique()), replace=False)
        b = EV[EV.name.isin(cols)].groupby('per').pl.mean()
        b = b[b.index.isin(EV.groupby('per').size()[lambda s: s>20].index)]
        if len(b) < 40: continue
        shs.append(b.mean()/b.std(ddof=1)*np.sqrt(4)); sd_b.append(b.std(ddof=1))
    ideal = sn.mean()*np.sqrt(Nsub)
    rows.append((Nsub, np.mean(shs), ideal, np.mean(sd_b)))
    print(f"  N={Nsub:4d} -> book Sharpe {np.mean(shs):+6.2f}   sqrt(N) ideal {ideal:+6.2f}"
          f"   capture {np.mean(shs)/ideal:5.1%}")
sd1 = rows[0][3]
print(f"\nEFFECTIVE BREADTH from vol reduction (k-invariant):")
for Nsub, sh, ideal, sd in rows:
    print(f"  N={Nsub:4d} nominal -> N_eff = (sd_1/sd_N)^2 = {(sd1/sd)**2:6.1f}")

B = EV.groupby('per').pl.mean()
print(f"\nfull 119-name book Sharpe {B.mean()/B.std(ddof=1)*np.sqrt(4):+.2f} over {len(B)} quarters "
      f"({len(B)/4:.0f} years)")
q = B.quantile(0.10); bad = B[B<=q]
ords = np.sort([p.ordinal for p in bad.index]); gaps=np.diff(ords)
print(f"worst-decile quarters {len(bad)}, independent episodes (>2 qtrs apart): {1+(gaps>2).sum()}")
print("worst 6:", ", ".join(f"{p}({v:+.2f})" for p,v in B.nsmallest(6).items()))
vq = vix.resample('QE').mean(); vq.index = vq.index.to_period('Q')
al = pd.concat([B.rename('pl'), vq.rename('vix')], axis=1).dropna()
print(f"corr(book P&L, VIX level) {al.pl.corr(al.vix):+.3f}   corr(book P&L, dVIX) {al.pl.corr(al.vix.diff()):+.3f}")

piv = EV.pivot_table(index='per', columns='name', values='pl').dropna(axis=1, thresh=70)
agg = piv.mean(axis=1)
print("\ntail-conditional cross-name correlation:")
for lbl, msk in [("all quarters", agg.notna()), ("worst tercile", agg<=agg.quantile(1/3)),
                 ("worst quintile", agg<=agg.quantile(0.20))]:
    C = piv[msk].corr(min_periods=8); r = mean_offdiag(C.values)
    print(f"  {lbl:16} n={msk.sum():3d}  mean corr {r:+.3f}  N_eff(equi,119) {eff_n_equi(119,max(r,1e-6)):5.1f}")
