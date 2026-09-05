"""Is the cross-sectional IC real, or an artifact of the trailing-sigma normalizer?"""
import numpy as np, pandas as pd, glob, os, re
from common import *
from scipy import stats
EV = pd.read_csv("out_s2_events.csv", parse_dates=['date']); EV['per']=pd.PeriodIndex(EV.per,freq='Q')

def panel_ic(piv, k=4, lbl=""):
    sig = piv.rolling(k, min_periods=3).mean().shift(1)
    ics=[]
    for q in piv.index:
        a,b = sig.loc[q], piv.loc[q]; m=a.notna()&b.notna()
        if m.sum()<30: continue
        ics.append(stats.spearmanr(a[m], b[m]).statistic)
    ics=np.array(ics)
    t=ics.mean()/(ics.std(ddof=1)/np.sqrt(len(ics)))
    print(f"  {lbl:44} IC {ics.mean():+.4f}  t {t:+6.2f}  n {len(ics)}")
    return ics.mean()

print("="*78); print("STUDY 8 - PLACEBO / CONTROL TESTS ON THE CROSS-SECTIONAL IC"); print("="*78)
piv = EV.pivot_table(index='per', columns='name', values='z').abs().apply(np.log)
piv = piv.dropna(axis=1, thresh=70).dropna(thresh=100)
print("\n(1) the headline: trailing-4q mean of log|z| predicting next log|z|")
a = panel_ic(piv, 4, "EVENT days (max-idio day per quarter)")

# PLACEBO: rebuild the identical statistic on RANDOM non-event days.
files = sorted(glob.glob(f"{BH}/n_*.json")); px={}
for f in files:
    s=re.match(r"n_(.+)\.json",os.path.basename(f)).group(1)
    try: c=close(f)
    except Exception: continue
    if len(c)>=2500: px[s]=c
spy=close(f"{SP}/etf_SPY.json")
R=pd.DataFrame(px).apply(np.log).diff().loc['2005-01-01':]
Rm=np.log(spy).diff().reindex(R.index)
E=pd.DataFrame(index=R.index,columns=R.columns,dtype=float)
for yr,idx in R.groupby(R.index.year).groups.items():
    rm=Rm.loc[idx]
    for c in R.columns:
        y=R.loc[idx,c]; m=y.notna()&rm.notna()
        if m.sum()<100: continue
        E.loc[idx,c]=y-np.cov(y[m],rm[m])[0,1]/np.var(rm[m],ddof=1)*rm
E=E.astype(float)
tr=E.rolling(63,min_periods=40).std(ddof=1).shift(1)
Zall=(E/tr).abs()
q=pd.PeriodIndex(E.index,freq='Q')
rng=np.random.default_rng(3)
print("\n(2) PLACEBO: same construction on a RANDOM day in each quarter (no earnings)")
for rep in range(3):
    rows=[]
    for per in pd.unique(q):
        sl=Zall[q==per]
        if len(sl)<40: continue
        d=sl.index[rng.integers(10,len(sl)-1)]
        rows.append(sl.loc[d].rename(per))
    P2=pd.DataFrame(rows); P2=np.log(P2.replace(0,np.nan)).dropna(axis=1,thresh=70).dropna(thresh=80)
    panel_ic(P2, 4, f"RANDOM day, draw {rep+1}")

print("\n(3) PLACEBO: max-idio day EXCLUDING the top-1 day (2nd largest move in quarter)")
rows=[]
for c in E.columns:
    e=E[c]
    for per in pd.unique(q):
        m=(q==per)&e.notna()&tr[c].notna()
        if m.sum()<40: continue
        sl=e[m].abs()/tr.loc[m,c]
        srt=sl.sort_values(ascending=False)
        if len(srt)<2: continue
        rows.append(dict(name=c,per=per,z=srt.iloc[1]))
P3=pd.DataFrame(rows).pivot_table(index='per',columns='name',values='z')
P3=np.log(P3).dropna(axis=1,thresh=70).dropna(thresh=80)
panel_ic(P3, 4, "2nd-largest idio day per quarter")

print("\n(4) CONTROL: does the signal survive after removing the trailing-sigma LEVEL?")
lvl = np.log(tr).resample('QE').last(); lvl.index=lvl.index.to_period('Q')
lvl = lvl.reindex(index=piv.index, columns=piv.columns)
resid = piv.copy()
for qq in piv.index:
    y,x = piv.loc[qq], lvl.loc[qq]; m=y.notna()&x.notna()
    if m.sum()<30: continue
    b=np.polyfit(x[m],y[m],1); resid.loc[qq,m]=y[m]-np.polyval(b,x[m])
panel_ic(resid, 4, "log|z| orthogonalized to trailing idio-vol level")

print("\nREAD: if (2)/(3) show a similar IC, the effect is a property of the normalizer")
print("(mean reversion in a 63-day rolling sigma estimate), not of earnings information.")
