#!/usr/bin/env python3
"""Study 23: de-risk the ladder, then add an orthogonal sleeve.

Two remaining questions for the deployed strategy:
  1. 2022 cost -15.9%. Can an option-implied gate size rungs down before the
     damage, using only past data? Earlier work here found binary sit-out gates
     always lose (they surrender more return than risk) but CONTINUOUS sizing
     on implied vol worked. Test sizing, not switching.
  2. The skew->stock long-short sleeve has ~0 correlation to a short-put book.
     Adding an uncorrelated stream raises Sharpe without needing more edge.
"""
import math, os
import numpy as np, pandas as pd
import study20_optimize as S20

DEV_END = pd.Timestamp("2024-12-31")
HERE = os.path.dirname(os.path.abspath(__file__))

def main():
    panel = S20.load_spy()
    d = S20.build(0.05, 0.05, 45, 75, panel=panel)      # the improved config
    f = pd.read_parquet(os.path.join(HERE,"cache","features.parquet"))
    f["date"]=pd.to_datetime(f.date)
    spy=f[f.act_symbol=="SPY"].sort_values("date")
    iv=spy.set_index("date").iv_front
    ivz=(iv-iv.rolling(150,min_periods=40).mean())/iv.rolling(150,min_periods=40).std()
    ivmed=iv.expanding(min_periods=40).median()

    print("="*84); print("STUDY 23a — rung sizing gates (60d/5%/5%)"); print("="*84)
    base=S20.ladder(d)
    print(f"  {'no gate (baseline)':<42} Sharpe={base['sharpe']:+.2f} "
          f"CAGR={base['cagr']:+.1%} maxDD={base['dd']:+.1%}")
    def mk(fn):
        def sz(r):
            try: return fn(r.date)
            except Exception: return 1.0
        return sz
    gates=[("half size when IV z>1", lambda dt: 0.5 if (dt in ivz.index and ivz.get(dt,0)>1) else 1.0),
           ("size 1/IV (vol targeting)", lambda dt: float(np.clip(ivmed.get(dt,np.nan)/iv.get(dt,np.nan),0.3,2.0)) if dt in iv.index and np.isfinite(iv.get(dt,np.nan)) else 1.0),
           ("skip when IV z>1.5", lambda dt: 0.0 if (dt in ivz.index and ivz.get(dt,0)>1.5) else 1.0),
           ("double when IV z<-0.5", lambda dt: 1.5 if (dt in ivz.index and ivz.get(dt,0)<-0.5) else 1.0)]
    for nm,fn in gates:
        r=S20.ladder(d,size_fn=mk(fn))
        if r: print(f"  {nm:<42} Sharpe={r['sharpe']:+.2f} CAGR={r['cagr']:+.1%} maxDD={r['dd']:+.1%}")

    print("\n"+"="*84); print("STUDY 23b — add the orthogonal skew sleeve"); print("="*84)
    eq=base["eq"]; lad=eq.pct_change().dropna()
    p=os.path.join(HERE,"cache","optsig_full.parquet")
    if os.path.exists(p):
        o=pd.read_parquet(p); o["date"]=pd.to_datetime(o.date)
        o=o.dropna(subset=["skew25","fwd8"]); rows=[]
        for dt,g in o.groupby("date"):
            if len(g)<20: continue
            lo,hi=g.skew25.quantile(.2),g.skew25.quantile(.8)
            L,S=g[g.skew25>=hi].fwd8,g[g.skew25<=lo].fwd8
            if len(L)<3 or len(S)<3: continue
            rows.append((dt,0.5*(L.mean()-S.mean())-2/1e4))
        b=(pd.Series(dict(rows)).sort_index())/8.0
        R=pd.DataFrame({"ladder":lad,"skew":b}).dropna()
        print(f"  correlation ladder vs skew sleeve: {R.corr().iloc[0,1]:+.3f}  (n={len(R)})")
        ppy=len(R)/((R.index[-1]-R.index[0]).days/365.25)
        def sh(x): return x.mean()/x.std()*math.sqrt(ppy) if x.std()>0 else np.nan
        print(f"  ladder alone      Sharpe={sh(R.ladder):+.2f}")
        print(f"  skew sleeve alone Sharpe={sh(R.skew):+.2f}")
        for w in (0.2,0.3,0.5):
            c=(1-w)*R.ladder+w*R.skew
            print(f"  ladder {1-w:.0%} + skew {w:.0%}   Sharpe={sh(c):+.2f}")
        dev=R[R.index<=DEV_END]; ww=(1/dev.std())/(1/dev.std()).sum()
        comb=(R*ww).sum(axis=1)
        print(f"  inverse-vol (dev-fit) weights {dict(ww.round(3))}: Sharpe={sh(comb):+.2f}")
        for lab,seg in (("dev",comb[comb.index<=DEV_END]),("holdout",comb[comb.index>DEV_END])):
            if len(seg)>20: print(f"    {lab:<8} Sharpe={sh(seg):+.2f}")

if __name__=="__main__":
    main()
