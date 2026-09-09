"""PREREG4 evaluation: 9 panel ideas (I1-I7, I11, I12) with day-clustered t,
shuffle nulls, and Holm-Bonferroni across the 12-idea family."""
import numpy as np, pandas as pd
from scipy import stats
P = pd.read_parquet('/tmp/timbre/panel4.parquet')
P['split'] = np.where(P.day < '2022-01-01','TRAIN','VALID')
RNG = np.random.default_rng(4444)

def xs_ic(df, sig, tgt):
    """per-day cross-sectional Spearman IC; returns (mean, t, ndays, series)"""
    d = df[[ 'day', sig, tgt ]].dropna()
    def f(g):
        if len(g) < 8: return np.nan
        return stats.spearmanr(g[sig], g[tgt]).statistic
    ic = d.groupby('day').apply(f, include_groups=False).dropna()
    if len(ic) < 30: return np.nan, np.nan, len(ic), ic
    return ic.mean(), ic.mean()/(ic.std(ddof=1)/np.sqrt(len(ic))), len(ic), ic

def shuffle_null(df, sig, tgt, ndraw=200):
    d = df[['day',sig,tgt]].dropna().copy()
    ts=[]
    for _ in range(ndraw):
        d['_s'] = d.groupby('day')[sig].transform(lambda s: RNG.permutation(s.values))
        ic = d.groupby('day').apply(lambda g: stats.spearmanr(g['_s'],g[tgt]).statistic if len(g)>=8 else np.nan, include_groups=False).dropna()
        ts.append(abs(ic.mean()/(ic.std(ddof=1)/np.sqrt(len(ic)))))
    return float(np.percentile(ts,99))

P['s_I1'] = -P.ret_last30
P['s_I2'] = -P.vwap_dev
P['s_I3'] = P.semi_asym
P['s_I4'] = P.eff*np.sign(P.dayret)
P['s_I11']= P.skew1m
P['s_I12']= np.sign(P.dayret)/P.rng_z

IDEAS = [('I1','s_I1','T_on','overnight reversal of last-30m move'),
         ('I2','s_I2','T_on','close-vs-VWAP dislocation'),
         ('I3','s_I3','T_cc','semivariance asymmetry'),
         ('I4','s_I4','T_cc','efficiency-ratio trend continuation'),
         ('I11','s_I11','T_on','1-min return skewness'),
         ('I12','s_I12','T_cc','range-compression breakout')]

res={}
print(f"{'id':4s} {'target':6s} {'TRAIN IC':>9s} {'t':>7s} | {'VALID IC':>9s} {'t':>7s} {'ndays':>6s}  desc")
for i,(k,sig,tgt,desc) in enumerate(IDEAS):
    m1,t1,_,_ = xs_ic(P[P.split=='TRAIN'], sig, tgt)
    m2,t2,n2,_ = xs_ic(P[P.split=='VALID'], sig, tgt)
    res[k]=dict(t_valid=t2,t_train=t1,ic_valid=m2,ic_train=m1,tgt=tgt,desc=desc)
    print(f"{k:4s} {tgt:6s} {m1:+9.4f} {t1:+7.2f} | {m2:+9.4f} {t2:+7.2f} {n2:6d}  {desc}")

# --- conditional ideas ---
print()
def tercile_ic(df, cond, sig, tgt, label):
    d=df[['day',cond,sig,tgt]].dropna().copy()
    d=d[d.groupby('day')[cond].transform('size')>=9]
    d['tc']=d.groupby('day')[cond].transform(lambda s: pd.qcut(s.rank(method='first'),3,labels=[0,1,2],duplicates='drop'))
    out=[]
    for t_ in (0,2):
        g=d[d.tc==t_]
        ic=g.groupby('day').apply(lambda x: stats.spearmanr(x[sig],x[tgt]).statistic if len(x)>=4 else np.nan, include_groups=False).dropna()
        out.append((ic.mean(), ic.mean()/(ic.std(ddof=1)/np.sqrt(len(ic))), len(ic)))
    (lo,tlo,nlo),(hiv,thi,nhi)=out
    print(f"  {label:38s} low={lo:+.4f}(t{tlo:+5.1f})  high={hiv:+.4f}(t{thi:+5.1f})  spread={hiv-lo:+.4f}")
    return hiv-lo

P['s_rev']=-P.dayret
print("I5  Amihud illiquidity as amplifier of I1 (target T_on), VALID:")
V=P[P.split=='VALID']; TR=P[P.split=='TRAIN']
i5_v=tercile_ic(V,'amihud','s_I1','T_on','VALID amihud tercile')
i5_t=tercile_ic(TR,'amihud','s_I1','T_on','TRAIN amihud tercile')
print("I6  Trade-count surprise gating reversal (-dayret -> T_cc):")
i6_v=tercile_ic(V,'ntr_z','s_rev','T_cc','VALID ntr_z tercile')
i6_t=tercile_ic(TR,'ntr_z','s_rev','T_cc','TRAIN ntr_z tercile')

print("I7  Cross-sectional dispersion regime gate on I1:")
disp=P.groupby('day').dayret.std().rename('disp')
Pd=P.join(disp,on='day')
for sp in ('TRAIN','VALID'):
    d=Pd[Pd.split==sp]
    med=d.disp.median()
    for lab,sub in (('lowdisp',d[d.disp<=med]),('highdisp',d[d.disp>med])):
        m,t,n,_=xs_ic(sub,'s_I1','T_on')
        print(f"  {sp} {lab:9s} IC={m:+.4f} t={t:+6.2f} ndays={n}")

import json
json.dump({k:{kk:(None if (isinstance(vv,float) and not np.isfinite(vv)) else vv) for kk,vv in v.items()} for k,v in res.items()},
          open('/tmp/timbre/eval4_panel.json','w'), indent=1, default=str)
print("\nraw VALID |t| for Holm later:", {k:round(abs(v['t_valid']),2) for k,v in res.items()})
