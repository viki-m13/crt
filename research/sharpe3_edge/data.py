"""Pinned, incomplete historical-member archives: no production data claims."""
from pathlib import Path
import hashlib, json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import exchange_calendars as xc


def clean(frame):
    p=frame.copy(); p.index=pd.DatetimeIndex(p.index).tz_localize(None)
    if p.empty or p.index.has_duplicates or not p.index.is_monotonic_increasing or p.columns.has_duplicates:
        raise ValueError('Require nonempty unique ordered session index and ticker columns')
    return p.astype(float).where(lambda a: np.isfinite(a)&(a>0))


def load(root, universe):
    root=Path(root).resolve(); manifest=json.loads((root/'manifest.json').read_text())
    for item in manifest:
        path=(root/item['file']).resolve()
        if path.parent!=root or hashlib.sha256(path.read_bytes()).hexdigest()!=item['sha256']:
            raise ValueError('Unsafe or changed input: '+item['file'])
    read=lambda n:pq.read_table(root/n).to_pandas()
    broad=clean(read('prices_extended_pit.parquet'))
    if universe=='sp500':
        raw=broad; m=read('sp500_membership_monthly.parquet').copy()
        m['ticker']=m.ticker.str.replace('.','-',regex=False); m['asof']=pd.to_datetime(m['asof'])
        latest=m.loc[m['asof'].eq(m['asof'].max()),'ticker'].tolist()
    elif universe=='ndx':
        raw=clean(read('n100_panel_close.parquet')); m=read('n100_panel_member__bonds.parquet').fillna(False).astype(bool)
        latest=m.columns[m.iloc[-1]].tolist()
    else: raise ValueError('universe must be sp500 or ndx')
    ends=pd.Series({s:raw[s].last_valid_index() for s in latest if s in raw})
    counts=ends.dropna().value_counts()
    if counts.empty or counts.iloc[0]/len(ends)<.8: raise ValueError('Provider cutoff required')
    end=min(pd.Timestamp(counts.index[0]),broad.SPY.last_valid_index())
    idx=xc.get_calendar('XNYS',start=max(raw.index[0],pd.Timestamp('2001-01-01')),end=end).sessions.tz_localize(None)
    p=raw.reindex(idx); market=broad.SPY.reindex(idx)
    if universe=='sp500':
        p=p[sorted(set(m.ticker)&set(p.columns))]
        member=pd.DataFrame(False,index=idx,columns=p.columns)
        groups=list(m.groupby('asof',sort=True))
        for j,(day,frame) in enumerate(groups):
            lo=idx.searchsorted(day); hi=idx.searchsorted(groups[j+1][0]) if j+1<len(groups) else len(idx)
            member.loc[idx[lo:hi],p.columns.intersection(frame.ticker)]=True
        volume=None
    else:
        member=m.reindex(index=idx,columns=p.columns,fill_value=False)
        volume=read('n100_panel_volume.parquet').reindex(index=idx,columns=p.columns)
    if market.isna().any(): raise ValueError('Missing benchmark price')
    notes=dict(universe=universe,cutoff=str(idx[-1].date()),calendar='XNYS',calendar_version=xc.__version__,
       price_kind='adjusted/mixed, not corporate-action certified',input_manifest=manifest,
       limitations=['Incomplete historical constituent prices and monthly S&P membership',
                    'NDX membership starts 2015 and is incomplete',
                    'Unknown terminal corporate actions; pessimistic missing writeoffs',
                    'Reused historical sample, NOT an untouched research holdout',
                    'No observed executable fills, spread, market impact or historical borrow'],production_eligible=False)
    return p,member,market,volume,notes
