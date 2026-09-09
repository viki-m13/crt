"""Pinned public inputs; as-of eligibility, actual sessions, no future fill."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import exchange_calendars as xc

def read(p):
    return pq.read_table(p).to_pandas()

def load(root, universe='sp500'):
    root=Path(root)
    manifest=json.loads((root/'manifest.json').read_text())
    for item in manifest:
        p=root/item['file']
        if p.parent.resolve()!=root.resolve(): raise ValueError('unsafe manifest')
        if hashlib.sha256(p.read_bytes()).hexdigest()!=item['sha256']: raise ValueError('input checksum')
    broad=read(root/'prices_extended_pit.parquet')
    broad.index=pd.DatetimeIndex(broad.index).tz_localize(None)
    # A source-level diagnosed cutoff, not per-security future eligibility.
    cutoff=pd.Timestamp('2026-03-20' if universe in ('sp500','structural') else '2026-05-07')
    dates=xc.get_calendar('XNYS',start='2001-01-01',end=cutoff).sessions.tz_localize(None)
    if universe=='sp500':
        snap=read(root/'sp500_membership_monthly.parquet')
        def resolve(s):
            return s if s in broad else s.replace('.','-')
        snap['ticker']=snap.ticker.map(resolve)
        cols=sorted(set(snap.ticker)&set(broad.columns))
        px=broad.reindex(index=dates,columns=cols).astype(float)
        membership=np.zeros(px.shape,bool); last=np.zeros(len(cols),bool)
        lookup={s:i for i,s in enumerate(cols)}; events={}
        missing=[]
        for day,grp in snap.groupby('asof'):
            i=dates.searchsorted(pd.Timestamp(day),side='left')
            if i>=len(dates): continue
            v=np.zeros(len(cols),bool)
            for s in grp.ticker:
                if s in lookup: v[lookup[s]]=True
            events[i]=v
            missing.append({'asof':str(day),'requested':len(grp),'represented':int(v.sum())})
        for i in range(len(dates)):
            if i in events: last=events[i]
            membership[i]=last
    elif universe=='ndx':
        px=read(root/'n100_panel_close.parquet').reindex(dates).astype(float)
        membership=read(root/'n100_panel_member__bonds.parquet').reindex(index=dates,columns=px.columns,fill_value=False).to_numpy(bool)
        missing=[]
    elif universe=='structural':
        px=broad.reindex(dates).astype(float); membership=np.isfinite(px.to_numpy());missing=[]
    else: raise ValueError(universe)
    if px.index.has_duplicates or px.columns.has_duplicates: raise ValueError('duplicate prices')
    px=px.where(np.isfinite(px)&(px>0))
    return px,membership,{'universe':universe,'cutoff':str(dates[-1].date()),'source_sha256':hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest(),'coverage':missing,'limitations':['Adjusted/mixed research prices; corporate actions not independently certified.','Incomplete historical constituent coverage; not a full point-in-time security master.','Fixed structural pair list has selection/coverage limitations.','No historical borrow availability, lending quotes or fill/volume-capacity validation.','Existing history was previously researched, not a virgin holdout.']}

def french_rf(root,dates):
    import zipfile,re
    z=zipfile.ZipFile(Path(root)/'french_daily.zip')
    text=z.read(z.namelist()[0]).decode('utf-8-sig')
    rows=[]
    for line in text.splitlines():
        if re.match(r'^\d{8},',line):
            a=line.split(',');rows.append((pd.Timestamp(a[0]),float(a[4])/100))
    rf=pd.Series(dict(rows)).reindex(dates)
    if rf.isna().any():raise ValueError('missing actual RF dates')
    return rf

def load_bonds(root):
    root=Path(root)
    for name,h in json.loads((root/'export_hashes.json').read_text()).items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=h:raise ValueError('independent input hash')
    px=read(root/'bonds_prices.parquet');px.index=pd.DatetimeIndex(px.index).tz_localize(None)
    # Punctuation alias only, not a security-identity inference.
    if 'BRK-B' in px and 'BRK.B' not in px:px=px.rename(columns={'BRK-B':'BRK.B'})
    dates=xc.get_calendar('XNYS',start='2001-01-01',end='2026-03-20').sessions.tz_localize(None)
    px=px.reindex(dates).astype(float);px=px.where(np.isfinite(px)&(px>0))
    snap=read(root/'sp500_membership.parquet');lookup={s:i for i,s in enumerate(px)}
    events={};coverage=[]
    for day,grp in snap.groupby('asof'):
        i=dates.searchsorted(pd.Timestamp(day),'left')
        if i>=len(dates):continue
        v=np.zeros(len(px.columns),bool)
        for s in grp.ticker:
            k=s if s in lookup else s.replace('.','-')
            if k in lookup:v[lookup[k]]=True
        events[i]=v;coverage.append({'asof':str(day),'requested':len(grp),'represented':int(v.sum())})
    m=np.zeros(px.shape,bool);last=np.zeros(len(px.columns),bool)
    for i in range(len(px)):
        if i in events:last=events[i]
        m[i]=last
    return px,m,french_rf(root,dates),coverage
