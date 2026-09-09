"""Survivorship-bias-free daily price panel, assembled once and cached.

Any claim of the form "this setup is higher N days later X% of the time" is
worthless on a universe of companies that still exist. The stocks that went to
zero are exactly the ones that would have been "higher X days later" and were
not, and a panel of today's listed names has quietly deleted them.

This builds the panel from the Tiingo shards in the bonds repo together with
its point-in-time universe file, which carries a `startDate` and `endDate`
per ticker — 16,026 stocks including the dead ones. Prices outside a ticker's
listed window are masked out, so a name cannot contribute observations before
it listed or after it stopped trading.

    from data import load_panel
    px, meta = load_panel()          # DataFrame (dates x tickers), float32

WHAT THIS STILL CANNOT DO: when a stock disappears mid-horizon, the panel
knows the last traded price but not why it stopped — an acquisition at a
premium and a bankruptcy both just end. Callers must decide how to score
those, and `delisting_mask` marks them so the decision is explicit rather
than accidental. `research/uptrend/baserates.py` reports the answer both
ways and the gap between them.
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

BONDS = "/home/user/bonds"
SHARDS = os.path.join(BONDS, "dca/research/data/tiingo/prices")
UNIVERSE = os.path.join(BONDS, "dca/research/data/tiingo/tiingo_universe_pit.parquet")
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".panel.pkl")

MIN_PRICE = 5.0          # sub-$5 names are a different market: wide spreads,
                         # and a "50% gain" that no one could have captured
MIN_BARS = 400           # enough history to compute the features we need


def _universe(verbose: bool = True) -> pd.DataFrame:
    """Point-in-time listing windows, one per ticker.

    606 of 15,402 stock tickers carry more than one listing window, because
    symbols get reused: AAC was American Addiction Centers until 2019 and Ares
    Acquisition Corp from 2021. The panel has one column per symbol, so those
    two companies' prices are concatenated into one series and the seam is a
    fabricated return of arbitrary size.

    We keep each ticker's LONGEST window and mask the rest away. That throws
    out some real history, but the alternative — trusting a column that
    silently switches company — puts fake jumps into exactly the tail of the
    return distribution this study is trying to measure.
    """
    u = pd.read_parquet(UNIVERSE)
    u = u[u.assetType == "Stock"].copy()
    u["startDate"] = pd.to_datetime(u.startDate)
    u["endDate"] = pd.to_datetime(u.endDate)
    u = u.dropna(subset=["ticker", "startDate"])
    u["endDate"] = u.endDate.fillna(u.startDate.max())
    reused = u.ticker.duplicated(keep=False).sum()
    u["_span"] = (u.endDate - u.startDate).dt.days
    u = (u.sort_values("_span", ascending=False)
           .drop_duplicates("ticker", keep="first")
           .drop(columns="_span"))
    if verbose and reused:
        print(f"  {reused:,} rows across reused tickers -> longest window kept")
    return u


def build(verbose: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    u = _universe(verbose)
    keep = set(u.ticker)
    frames = []
    for f in sorted(glob.glob(os.path.join(SHARDS, "ac_*.parquet"))):
        d = pd.read_parquet(f)
        cols = [c for c in d.columns if c in keep]
        if cols:
            frames.append(d[cols])
        if verbose:
            print(f"  {os.path.basename(f)}: {len(cols):,} stock columns",
                  flush=True)
    px = pd.concat(frames, axis=1, sort=False)
    px = px.loc[:, ~px.columns.duplicated()]
    px.index = pd.to_datetime(px.index)
    px = px.sort_index()

    # --- the point-in-time mask -------------------------------------------
    # Outside [startDate, endDate] a ticker did not exist to be bought. Some
    # shards carry values there anyway (backfilled or reused symbols); those
    # are removed rather than trusted.
    if verbose:
        print("  applying the point-in-time listing windows…", flush=True)
    span = u.set_index("ticker").reindex(px.columns)
    starts = span.startDate.to_numpy()
    ends = span.endDate.fillna(px.index[-1]).to_numpy()
    starts = np.where(pd.isna(starts), np.datetime64("2100-01-01"), starts)
    idx = px.index.to_numpy()[:, None]
    live = (idx >= starts[None, :]) & (idx <= ends[None, :])
    arr = px.to_numpy(dtype=np.float32)
    arr[~live] = np.nan
    px = pd.DataFrame(arr, index=px.index, columns=px.columns)

    # --- drop what cannot carry a usable observation ----------------------
    enough = px.notna().sum() >= MIN_BARS
    px = px.loc[:, enough]
    if verbose:
        print(f"  panel: {px.shape[0]:,} bars x {px.shape[1]:,} tickers "
              f"({px.index[0].date()} -> {px.index[-1].date()})")
    meta = span.loc[px.columns].reset_index()
    return px, meta


def load_panel(verbose: bool = True):
    if os.path.exists(CACHE):
        if verbose:
            print("using cached panel")
        d = pd.read_pickle(CACHE)
        return d["px"], d["meta"]
    px, meta = build(verbose)
    pd.to_pickle({"px": px, "meta": meta}, CACHE)
    return px, meta


def last_bar_index(px: pd.DataFrame) -> np.ndarray:
    """Row index of each ticker's final traded bar — where its data ends."""
    ok = px.notna().to_numpy()
    n = ok.shape[0]
    return np.where(ok.any(axis=0), n - 1 - ok[::-1].argmax(axis=0), -1)


def delisting_mask(px: pd.DataFrame, horizon: int) -> np.ndarray:
    """True where a ticker stops trading within `horizon` bars of that row.

    These are the observations whose outcome is genuinely unknown: the stock
    went away. Scoring them as "not higher" and dropping them entirely give
    different answers, and the difference is the size of the survivorship
    problem — which is why this is returned rather than resolved here.
    """
    ok = px.notna().to_numpy()
    last = last_bar_index(px)
    rows = np.arange(ok.shape[0])[:, None]
    return ok & (rows + horizon > last[None, :]) & (last[None, :] >= 0)


if __name__ == "__main__":
    px, meta = load_panel()
    print(f"\n{px.shape[0]:,} bars x {px.shape[1]:,} tickers")
    print(f"first {px.index[0].date()}  last {px.index[-1].date()}")
    print(f"live tickers on the last bar: {px.iloc[-1].notna().sum():,}")
    alive_2000 = px.loc[:"2000-12-31"].iloc[-1].notna().sum()
    print(f"live tickers on 2000-12-29:  {alive_2000:,}")
    print(f"tickers whose data ends before 2026: "
          f"{(meta.endDate < '2026-01-01').sum():,}  (the dead ones)")
