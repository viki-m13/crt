"""Public pinned archive adapter. No later-history filtering of candidates."""
from pathlib import Path
import numpy as np
import pandas as pd
from .archive import load_archive,features


def make_features(p,m,market,stride=5):
    f,coverage=features(p,m,market,stride=stride)
    if f.empty: return f,coverage,[]
    # Exclude last unplanned partial-grid scan row from historical tests.
    f=f.loc[(f.i-252)%stride==0].copy()
    i=f.i.to_numpy(int); j=p.columns.get_indexer(f.ticker)
    lp=np.log(p); r=lp.diff(); rm=np.log(market).diff()
    down=r.clip(upper=0).pow(2).rolling(63).mean().pow(.5)
    up=r.clip(lower=0).pow(2).rolling(63).mean().pow(.5)
    extras={'semivol_ratio':down/(up+1e-8),'skew126':r.rolling(126).skew(),
        'worst63':r.rolling(63).min(),'best21':r.rolling(21).max(),
        'down_frequency':(r<0).astype(float).where(r.notna()).rolling(63).mean(),
        'dd_change21':(lp-lp.rolling(252,min_periods=240).max()).diff(21),
        'trend_accel':lp.diff(21)-lp.diff(63)/3}
    for name,v in extras.items(): f[name]=v.to_numpy()[i,j]
    f['market_vol63']=rm.rolling(63).std().to_numpy()[i]
    f['market_dd']= (np.log(market)-np.log(market).rolling(252,min_periods=240).max()).to_numpy()[i]
    exclusions={'i','date','ticker','regime','reference_price','row_id'}
    cols=[c for c in f.columns if c not in exclusions]
    good=np.isfinite(f[cols].to_numpy(float)).all(axis=1)
    f=f.loc[good].sort_values(['i','ticker']).reset_index(drop=True)
    f['row_id']=np.arange(len(f))
    for c in cols: f[c]=f[c].astype(np.float32)
    counts=f.groupby('i').size()
    coverage['ffv_eligible']=coverage.i.map(counts).fillna(0).astype(int)
    return f,coverage,cols
