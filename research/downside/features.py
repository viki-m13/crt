"""Causal bearish recovery geometry on the fixed historical eligible universe."""
import numpy as np
import pandas as pd
from research.failure_first.data import make_features as prior_features
from research.failure_first.archive import load_archive


def make_features(p,m,market,stride=5):
    f,coverage,cols=prior_features(p,m,market,stride)
    if f.empty:return f,coverage,cols
    lp=np.log(p);r=lp.diff();rm=np.log(market).diff()
    high=lp.rolling(63).max();low=lp.rolling(63).min()
    fresh=lp.ge(lp.rolling(126).max())
    # Peak age since most recent trailing-126-session high, only past information.
    index=np.arange(len(p),dtype=float)[:,None]
    peaks=pd.DataFrame(np.where(fresh,index,np.nan),index=p.index,columns=p.columns).ffill()
    age=pd.DataFrame(index-peaks.to_numpy(),index=p.index,columns=p.columns)
    below=lp.lt(lp.rolling(50).mean()).where(lp.notna())
    neg=r.lt(0).astype(float).where(r.notna())
    vol21=r.rolling(21).std()
    positive_market=rm>0;negative_market=rm<0
    up_response=r.where(positive_market,axis=0).rolling(63,min_periods=10).mean()
    dn_response=r.where(negative_market,axis=0).rolling(63,min_periods=10).mean()
    extras={
        'lower_high63':high-high.shift(63),
        'lower_low63':low-low.shift(63),
        'rebound_fraction':(lp-low)/(high-low+1e-8),
        'drawdown_recovery21':lp-lp.shift(21).rolling(63).min(),
        'peak_age126':age.clip(upper=252),
        'below50_fraction63':below.astype(float).rolling(63).mean(),
        'negative_autocorr21':neg.rolling(21).corr(neg.shift(1)),
        'return_autocorr63':r.rolling(63).corr(r.shift(1)),
        'asymmetric_response':up_response+dn_response,
        'market_up_response':up_response,
        'market_down_response':dn_response,
        'rebound_efficiency21':lp.diff(21)/(r.abs().rolling(21).sum()+1e-8),
        'vol_acceleration':vol21/vol21.shift(21)-1,
        'shock_recovery':lp.diff(5)/(r.abs().rolling(63).max()+1e-8),
        'lower_ma_slope21':lp.rolling(50).mean().diff(21),
    }
    i=f.i.to_numpy(int);j=p.columns.get_indexer(f.ticker)
    for name,v in extras.items():
        # Degenerate autocorrelation/never-new-high are marked by fixed causal fill values.
        default=252. if name=='peak_age126' else 0.
        values=v.to_numpy()[i,j]
        f[name]=np.where(np.isfinite(values),values,default).astype(np.float32)
        cols.append(name)
    for col in ('rebound_fraction','lower_high63','peak_age126'):
        name=col+'_rank';f[name]=f.groupby('i')[col].rank(pct=True).astype(np.float32);cols.append(name)
    # Explicit flags describe regularization imputations; no future completeness filter.
    f=f.sort_values(['i','ticker']).reset_index(drop=True);f['row_id']=np.arange(len(f))
    return f,coverage,cols
