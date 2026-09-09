"""Point-in-time input adapter and past-only, cross-section-normalized states."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import exchange_calendars as xc
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def sessions(start, end):
    return xc.get_calendar('XNYS', start=start, end=end).sessions.tz_localize(None)


def checked_prices(frame):
    p = frame.copy()
    p.index = pd.DatetimeIndex(p.index).tz_localize(None)
    if not p.index.is_monotonic_increasing or p.index.has_duplicates or p.columns.has_duplicates:
        raise ValueError('Require unique increasing dates and unique ticker columns')
    if len(p) == 0:
        raise ValueError('Empty price history')
    return p.astype(float).where(lambda x: np.isfinite(x) & (x > 0))


def load_archive(root: Path, universe: str):
    """Load the pinned CRT archive, preserving its known research limitations."""
    manifest = json.loads((root / 'manifest.json').read_text())
    for item in manifest:
        path = (root / item['file']).resolve()
        if path.parent != root.resolve():
            raise ValueError('Unsafe manifest path')
        if hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
            raise ValueError(f'Input digest mismatch: {path.name}')
    read = lambda name: pq.read_table(root / name).to_pandas()
    broad = checked_prices(read('prices_extended_pit.parquet'))
    if universe == 'sp500':
        m = read('sp500_membership_monthly.parquet').copy()
        m['ticker'] = m.ticker.str.replace('.', '-', regex=False)
        m['asof'] = pd.to_datetime(m["asof"])
        current = m.loc[m["asof"].eq(m["asof"].max()), 'ticker'].tolist()
        raw = broad
    elif universe == 'ndx':
        raw = checked_prices(read('n100_panel_close.parquet'))
        m = read('n100_panel_member__bonds.parquet').fillna(False).astype(bool)
        current = list(m.columns[m.iloc[-1]])
    else:
        raise ValueError('Universe must be sp500 or ndx')
    ends = pd.Series({s: raw[s].last_valid_index() for s in current if s in raw})
    counts = ends.dropna().value_counts()
    if counts.empty or counts.iloc[0] / len(ends) < .8:
        raise ValueError('Ambiguous archive cutoff; need provider-certified cutoff')
    cutoff = min(pd.Timestamp(counts.index[0]), broad['SPY'].last_valid_index())
    start = max(raw.index[0], pd.Timestamp('2001-01-01'))
    idx = sessions(start, cutoff)
    p = raw.reindex(idx)
    market = broad['SPY'].reindex(idx)
    if universe == 'sp500':
        tickers = sorted(set(m.ticker) & set(p.columns))
        p = p[tickers]
        membership = pd.DataFrame(False, index=idx, columns=tickers)
        grouped = list(m.groupby('asof', sort=True))
        for j, (day, frame) in enumerate(grouped):
            lo = idx.searchsorted(day, side='left')
            hi = idx.searchsorted(grouped[j+1][0], side='left') if j+1 < len(grouped) else len(idx)
            cols = p.columns.intersection(frame.ticker)
            membership.loc[idx[lo:hi], cols] = True
    else:
        # No backward filling of a later constituent matrix.
        membership = m.reindex(index=idx, columns=p.columns, fill_value=False)
    notes = {
        'universe': universe, 'price_cutoff': str(idx[-1].date()),
        'price_kind': 'adjusted_or_mixed_unverified', 'production_eligible': False,
        'calendar': 'XNYS', 'calendar_version': xc.__version__,
        'manifest_sha256': hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest(),
        'source_files': manifest,
        'limitations': [
            'Adjusted/mixed prices: NOT a validated price-only return target.',
            'Incomplete historical constituent prices; terminal corporate actions unverified.',
            'Previously researched data; chronological tests are not virgin holdouts.',
            'S&P membership is monthly; NDX membership is incomplete and starts in 2015.',
            'No executable fill/slippage model; outputs are price forecasts, not trading returns.']}
    return p, membership, market, notes


def features(prices, membership, market, stride=5):
    """Past-only states on a fixed grid; missing future prices NEVER filter a row.

    Rank normalization uses only the members eligible at the current date.
    Extra nonmember columns cannot affect features or eligibility.
    """
    p = checked_prices(prices)
    if not p.index.equals(membership.index) or not p.index.equals(market.index):
        raise ValueError('Prices, benchmark and membership must share a session calendar')
    if not p.columns.equals(membership.columns) or membership.isna().any().any():
        raise ValueError('Membership must have all ticker columns and no missing flags')
    if not membership.isin([True, False]).all().all():
        raise ValueError('Membership values must be booleans or 0/1')
    if stride < 1:
        raise ValueError('stride must be positive')
    lp = np.log(p)
    r = lp.diff()
    lm = np.log(market)
    rm = lm.diff()
    beta = r.rolling(126, min_periods=126).cov(rm).div(rm.rolling(126).var(), axis=0)
    fs = {f'r{k}': lp.diff(k) for k in (5,21,63,126,252)}
    fs.update({
        'vol21': r.rolling(21).std(), 'vol63': r.rolling(63).std(),
        'vol252': r.rolling(252, min_periods=240).std(),
        'dd': lp - lp.rolling(252, min_periods=240).max(),
        'ma50': lp - lp.rolling(50).mean(),
        'ma200': lp - lp.rolling(200).mean(),
        'eff': lp.diff(63).abs() / r.abs().rolling(63).sum(),
        'worst21': r.rolling(21).min(), 'jump21': r.abs().rolling(21).max(),
        'beta': beta})
    for k in (21,63,126):
        fs[f'rel{k}'] = fs[f'r{k}'].sub(beta.mul(lm.diff(k), axis=0))
    common = pd.DataFrame({'market21': lm.diff(21), 'market63': lm.diff(63),
                           'market200': lm - lm.rolling(200).mean()}, index=p.index)
    rows, coverage = [], []
    origins = list(range(252, len(p), stride))
    # Latest-date snapshot is usable by scan, never an extra backtest decision.
    if len(p)-1 >= 252 and (not origins or origins[-1] != len(p)-1):
        origins.append(len(p)-1)
    for i in origins:
        symbols = p.columns[membership.iloc[i].to_numpy(dtype=bool)]
        f = pd.DataFrame({k: v.iloc[i].reindex(symbols) for k,v in fs.items()})
        f['reference_price'] = p.iloc[i].reindex(symbols)
        f = f.replace([np.inf,-np.inf],np.nan).dropna()
        f = f.loc[(f.vol63 > 1e-5) & (f.jump21 < .40)]
        coverage.append({'i':i, 'date':str(p.index[i].date()), 'members_in_price_panel':len(symbols),
                         'eligible':len(f), 'scheduled':(i-252)%stride == 0})
        if f.empty or common.iloc[i].isna().any():
            continue
        for k in ('vol63','rel21','rel63','rel126','eff'):
            f[k+'_rank'] = f[k].rank(pct=True, method='average')
        f['vol_ratio'] = f.vol21 / f.vol63
        f['breadth'] = float(f.ma50.gt(0).mean())
        for k,v in common.iloc[i].items():
            f[k] = float(v)
        f['i'] = i
        f['date'] = str(p.index[i].date())
        f['regime'] = 'bull' if common.iloc[i]['market200'] > 0 else 'stress'
        f['ticker'] = f.index
        rows.append(f.reset_index(drop=True))
    return (pd.concat(rows,ignore_index=True) if rows else pd.DataFrame()), pd.DataFrame(coverage)


def specialist_candidates(frame):
    """Six fixed economic hypotheses. This function NEVER receives future labels."""
    rows = []
    for i,f in frame.groupby('i',sort=True):
        bull = f.market200 > 0
        stable = (f.ma200 > 0) & (f.r126 > 0)
        safety = (f.worst21 > -.15) & (f.beta.between(-.5,2))
        definitions = {
            'quiet_leader': (stable & bull & (f.rel63>0) & (f.vol_ratio<.9) & (f.dd>-.15),
                             f.rel63_rank + f.eff_rank - f.vol63_rank),
            'resilient_pullback': (stable & (f.market21<0) & (f.rel21>0) & (f.r5>0),
                                   f.rel21_rank + f.eff_rank - f.vol63_rank),
            'recovery_turn': ((f.dd.between(-.55,-.10)) & (f.r21>0) & (f.ma50>0) & (f.market21>0),
                              (f.r21-f.r63/3)/f.vol63 + f.eff_rank),
            'rank_rotation': ((f.rel21_rank-f.rel126_rank>.25) & (f.ma50>0) & (f.vol63_rank<.75),
                              f.rel21_rank-f.rel126_rank + f.eff_rank - f.vol63_rank),
            'defensive_drift': (stable & bull & (f.r252>.08) & (f.vol63_rank<.4) & (f.eff_rank>.5),
                                f.r252/(f.vol63*np.sqrt(252)) + f.eff_rank - f.vol63_rank),
            'shock_absorption': (stable & (f.r21<0) & (f.r5>0) & (f.vol_ratio<1.5),
                                 f.r5/f.vol63 + f.rel126_rank - f.vol63_rank)}
        # Baselines are selected at the same date before seeing outcomes.
        ordered = f.sort_values('ticker')
        seed = int(hashlib.sha256(str(f.iloc[0]['date']).encode()).hexdigest()[:8],16)
        control = ordered.iloc[seed % len(ordered)].ticker
        lowvol = f.sort_values(['vol63','ticker']).iloc[0].ticker
        for family,(mask,score) in definitions.items():
            pool = f.loc[mask & safety].copy()
            if pool.empty:
                continue
            pool['strength'] = score.loc[pool.index]
            chosen = pool.sort_values(['strength','ticker'],ascending=[False,True]).iloc[0].to_dict()
            chosen.update(family=family, random_control=control, lowvol_control=lowvol)
            rows.append(chosen)
    return pd.DataFrame(rows)
