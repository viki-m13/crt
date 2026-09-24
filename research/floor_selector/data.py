"""Past-only panel adapter. Existing archives are research proxies, not live quotes."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import exchange_calendars as xc

FEATURES = ['r5','r21','r63','r126','r252','vol21','vol63','vol252',
            'vol_ratio','drawdown','ma_gap','beta','relative63','relative252',
            'up_fraction','rank63','rank_acceleration','recovery','efficiency',
            'breadth','dispersion','market63','market252','market_vol',
            'market_drawdown','market_ma_gap','log_horizon']


def sessions(start, end):
    return xc.get_calendar('XNYS', start=pd.Timestamp(start),
                           end=pd.Timestamp(end)).sessions.tz_localize(None)


def clean_prices(p):
    p = p.copy()
    p.index = pd.DatetimeIndex(p.index).tz_localize(None)
    if p.index.has_duplicates or not p.index.is_monotonic_increasing:
        raise ValueError('Price dates must be unique and increasing')
    p.columns = p.columns.astype(str).str.replace('.', '-', regex=False)
    if p.columns.has_duplicates:
        raise ValueError('Duplicate symbols after normalization')
    return p.astype(float).where(lambda v: np.isfinite(v) & (v > 0))


def read(path):
    return pq.read_table(path).to_pandas()


def load(root: Path, universe: str):
    manifest = json.loads((root/'manifest.json').read_text())
    hashes = {}
    for item in manifest:
        p = root/item['file']
        if p.parent.resolve() != root.resolve():
            raise ValueError('Manifest path traversal')
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        if digest != item['sha256']:
            raise ValueError('Input hash mismatch: '+p.name)
        hashes[p.name] = digest
    broad = clean_prices(read(root/'prices_extended_pit.parquet'))
    if universe == 'sp500':
        mem = read(root/'sp500_membership_monthly.parquet')
        mem['ticker'] = mem.ticker.str.replace('.', '-', regex=False)
        current = mem.loc[mem['asof'].eq(mem['asof'].max()), 'ticker'].tolist()
        p = broad
    elif universe == 'ndx':
        p = clean_prices(read(root/'n100_panel_close.parquet'))
        mem = read(root/'n100_panel_member__bonds.parquet').astype(bool)
        mem.index = pd.DatetimeIndex(mem.index).tz_localize(None)
        mem.columns = mem.columns.str.replace('.', '-', regex=False)
        current = mem.columns[mem.iloc[-1]].tolist()
    else:
        raise ValueError('Unknown universe')
    # This latest-members diagnostic sets an administrative FEED cutoff only.
    # It is never used to filter the historical stock-selection universe.
    ends = pd.Series({s:p[s].last_valid_index() for s in current if s in p})
    counts = ends.dropna().value_counts()
    if counts.empty or counts.iloc[0] / len(ends) < .8:
        raise ValueError('No common administrative feed cutoff; obtain provider metadata')
    cutoff = min(counts.index[0], broad.SPY.last_valid_index())
    idx = sessions('2001-01-01', cutoff)
    p = p.reindex(idx)
    market = broad.SPY.reindex(idx)
    requests = []
    if universe == 'sp500':
        for d, part in mem.groupby('asof', sort=True):
            # Membership snapshot must already be known, not inferred from a
            # future weekend/month-end snapshot at the prior Friday close.
            i = int(idx.searchsorted(pd.Timestamp(d), side='right'))
            if i < len(idx):
                requests += [(i,s) for s in sorted(set(part.ticker))]
    else:
        mem = mem.reindex(index=idx, columns=p.columns, fill_value=False)
        for d in pd.date_range(idx[0], idx[-1], freq='ME'):
            i = int(idx.searchsorted(d, side='right'))
            if i < len(idx) and i > 0:
                requests += [(i,s) for s in mem.columns[mem.iloc[i-1]]]
    req = pd.DataFrame(requests, columns=['i','ticker']).drop_duplicates()
    p = p.loc[:,p.columns.isin(req.ticker)]
    meta = dict(universe=universe, price_basis='adjustment_not_verified',
                data_quality_certified=False, feed_end=str(idx[-1].date()),
                source_hashes=hashes,
                limitations=['Incomplete historical membership/failed-stock coverage',
                  'Mixed adjusted closes: not verified price-only appreciation',
                  'Unknown terminal settlements: unresolved outcomes retained',
                  'Reused historical research sample, not a virgin holdout'])
    return p, req, market, meta


def features(prices, requests, market):
    """Extraction never inspects later prices or later member completeness."""
    prices = clean_prices(prices)
    if not market.index.equals(prices.index):
        raise ValueError('Stock and market calendars differ')
    lp = np.log(prices); r = lp.diff(); lm = np.log(market); mr = lm.diff()
    v21 = r.rolling(21).std(); v63 = r.rolling(63).std()
    v252 = r.rolling(252).std()
    ma = lp-lp.rolling(200).mean()
    dd = lp-lp.rolling(252).max()
    fields = {**{f'r{k}':lp.diff(k) for k in (5,21,63,126,252)},
       'vol21':v21, 'vol63':v63, 'vol252':v252, 'vol_ratio':v21/v252,
       'drawdown':dd,'ma_gap':ma,
       'beta':r.rolling(126).cov(mr).div(mr.rolling(126).var(),axis=0),
       'relative63':lp.diff(63).sub(lm.diff(63),axis=0),
       'relative252':lp.diff(252).sub(lm.diff(252),axis=0),
       'up_fraction':r.gt(0).where(r.notna()).rolling(63).mean(),
       'recovery':lp-lp.rolling(126).min(),
       'efficiency':lp.diff(63)/r.abs().rolling(63).sum()}
    common = pd.DataFrame(dict(market63=lm.diff(63), market252=lm.diff(252),
       market_vol=mr.rolling(63).std(), market_drawdown=lm-lm.rolling(252).max(),
       market_ma_gap=lm-lm.rolling(200).mean()))
    rows=[]; coverage=[]
    for i, grp in requests.groupby('i',sort=True):
        names=sorted(set(grp.ticker)); present=[s for s in names if s in prices]
        if not present:
            continue
        f=pd.DataFrame({k:v.iloc[i].reindex(present) for k,v in fields.items()})
        f['rank63']=f.r63.rank(pct=True)
        old=(lp.iloc[max(0,i-21)]-lp.iloc[max(0,i-84)]).reindex(present).rank(pct=True)
        f['rank_acceleration']=f.rank63-old
        f['breadth']=f.ma_gap.dropna().gt(0).mean()
        f['dispersion']=f.r21.std()
        for k in common:
            f[k]=common.iloc[i][k]
        f['reference']=prices.iloc[i].reindex(present)
        f['i']=int(i); f['date']=prices.index[i]; f['ticker']=f.index
        valid=np.isfinite(f[FEATURES[:-1]]).all(axis=1)&f.vol63.gt(1e-5)
        coverage.append(dict(date=str(prices.index[i].date()),requested=len(names),
                             represented=len(present),eligible=int(valid.sum())))
        rows.append(f.loc[valid].reset_index(drop=True))
    if not rows or not any(len(r) for r in rows):
        raise ValueError('Insufficient past history')
    return pd.concat(rows,ignore_index=True), pd.DataFrame(coverage)


def targets(f, p, horizons):
    if not horizons or any(type(h) is not int or h<30 for h in horizons):
        raise ValueError('Horizons must be integer trading-session counts >=30')
    if len(set(horizons))!=len(horizons):
        raise ValueError('Duplicate horizons')
    pos=p.columns.get_indexer(f.ticker)
    if (pos<0).any():
        raise ValueError('Unmapped stock')
    rows=[]; arr=p.to_numpy()
    for h in sorted(horizons):
        d=f.copy(); d['horizon']=h; d['log_horizon']=np.log(h/30)
        d['exit_i']=d.i+h; mature=d.exit_i.lt(len(p)).to_numpy()
        end=np.full(len(d),np.nan)
        end[mature]=arr[d.exit_i.to_numpy()[mature],pos[mature]]
        d['matured']=mature; d['resolved']=mature&np.isfinite(end)&(end>0)
        d['return']=end/d.reference-1
        d['up']=np.where(d.resolved,d['return'].gt(0),False)
        d['scale']=d.vol63*np.sqrt(h)
        # Training proxy only: missing matured endpoints get an explicitly
        # pessimistic near-total-loss log return, never zero or an omitted row.
        lr=np.log(np.maximum(end/d.reference.to_numpy(),1e-6))
        d['z']=np.where(d.resolved,lr,np.log(1e-6))/d.scale
        d.loc[~d.matured,'z']=np.nan
        rows.append(d)
    return pd.concat(rows,ignore_index=True)
