"""Analog finder + HONEST walk-forward validation.
For each date t: match the trailing W-day normalized path against every prior
window (strictly before t, no overlap with the future), take the K nearest,
average their forward H-day paths -> projection. Then compare to what happened."""
import numpy as np, pandas as pd, glob
from pathlib import Path
D='/tmp/databranch/data/daily_multiasset/'
def ld(s):
    d=pd.read_parquet(D+f'{s}.parquet'); d['d']=pd.to_datetime(d['d'])
    return d.set_index('d')['adj'].sort_index()
SYMS=['SPY','QQQ','IWM','GLD','TLT','XLE','XLF','XLK','EEM','AAPL','MSFT','AMZN','NVDA','SLV','HYG','USO','VNQ','XLV','XLP','XLU']
def analog_eval(sym,W=120,H=63,K=20,minhist=750):
    p=ld(sym).dropna()
    lp=np.log(p.values); idx=p.index; n=len(lp)
    rows=[]
    for t in range(minhist,n-H):
        cur=lp[t-W+1:t+1]-lp[t-W+1]                      # normalized trailing path
        # candidate windows END strictly before t-H so their FUTURE is also pre-t (no leakage)
        ends=np.arange(W, t-H)
        if len(ends)<50: continue
        M=np.lib.stride_tricks.sliding_window_view(lp,W)  # (n-W+1, W)
        cand=M[ends-W+1]                                   # windows ending at each `ends`
        cand=cand-cand[:,[0]]
        dist=np.sqrt(((cand-cur)**2).mean(axis=1))
        best=ends[np.argsort(dist)[:K]]
        fwd=np.array([lp[e+H]-lp[e] for e in best])        # analog forward returns
        proj=fwd.mean(); spread=fwd.std()
        actual=lp[t+H]-lp[t]
        rows.append(dict(sym=sym,t=idx[t],proj=proj,spread=spread,actual=actual,
                         dbest=float(dist.min()),n_pos=int((fwd>0).sum())))
    return pd.DataFrame(rows)
allr=[]
for s in SYMS:
    try:
        r=analog_eval(s)
        if len(r)>200: allr.append(r); print(f"{s}: {len(r)} evaluations",flush=True)
    except Exception as e: print(f"{s}: {e}",flush=True)
A=pd.concat(allr,ignore_index=True)
A.to_parquet('/tmp/analog/val.parquet')
print(f"\nTOTAL {len(A):,} out-of-sample analog projections, {A.sym.nunique()} tickers")
