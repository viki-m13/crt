"""Extended-history bt_core: same API as bt_core.load_market but sources data
from the parquet files produced by regen_scores_ext.py.

Parquet schema (max/research/data/bt_ext.parquet):
  long-form: ticker, date, price, final  (date = pd.Timestamp)

Benchmark prices come from the raw AdjClose parquet.
"""
from __future__ import annotations

import math
import os
from datetime import datetime, timedelta
from typing import Iterable

import numpy as np
import pandas as pd

from bt_core import MarketData, DCA_MONTHLY, TRADING_DAYS_YR  # noqa: F401 reuse dataclass

DATA_DIR = "/home/user/crt/max/research/data"


def load_market_ext(bt_path: str = None, raw_dir: str = None,
                    bench_tickers: Iterable[str] = ("SPY",)) -> MarketData:
    bt_path = bt_path or os.path.join(DATA_DIR, "bt_ext.parquet")
    raw_dir = raw_dir or os.path.join(DATA_DIR, "raw")

    df = pd.read_parquet(bt_path)
    df["date"] = pd.to_datetime(df["date"])
    df["date_str"] = df["date"].dt.strftime("%Y-%m-%d")

    # All dates = union of dates across tickers (only keep non-stale tickers)
    max_end = df["date_str"].max()
    stale_cutoff = (pd.to_datetime(max_end) - pd.Timedelta(days=365)).strftime("%Y-%m-%d")
    last_per_tk = df.groupby("ticker")["date_str"].max()
    live_tks = set(last_per_tk[last_per_tk >= stale_cutoff].index)
    df_live = df[df["ticker"].isin(live_tks)]

    all_dates = sorted(df_live["date_str"].unique())
    date_idx = {dd: i for i, dd in enumerate(all_dates)}
    n = len(all_dates)

    has_wash = "wash" in df_live.columns
    has_raw = "final_raw" in df_live.columns
    has_q = "quality" in df_live.columns

    prices: dict = {}
    finals: dict = {}
    washes: dict = {} if has_wash else None
    finals_raw: dict = {} if has_raw else None
    qualities: dict = {} if has_q else None
    for tk, grp in df_live.groupby("ticker"):
        p = np.full(n, np.nan)
        f = np.full(n, np.nan)
        w = np.full(n, np.nan) if has_wash else None
        fr = np.full(n, np.nan) if has_raw else None
        dd_vals = grp["date_str"].values
        pv_vals = grp["price"].values
        fv_vals = grp["final"].values
        wv_vals = grp["wash"].values if has_wash else None
        frv_vals = grp["final_raw"].values if has_raw else None
        for idx_row in range(len(dd_vals)):
            dd = dd_vals[idx_row]
            if dd in date_idx:
                ix = date_idx[dd]
                p[ix] = pv_vals[idx_row]
                f[ix] = fv_vals[idx_row]
                if has_wash:
                    w[ix] = wv_vals[idx_row]
                if has_raw:
                    fr[ix] = frv_vals[idx_row]
        prices[tk] = p
        finals[tk] = f
        if has_wash:
            washes[tk] = w
        if has_raw:
            finals_raw[tk] = fr
        if has_q:
            # quality is today-snapshot; take first non-null value for this ticker
            q_ser = grp["quality"].dropna()
            qualities[tk] = float(q_ser.iloc[0]) if len(q_ser) else None

    # Bench filled from raw AdjClose
    ac = pd.read_parquet(os.path.join(raw_dir, "AdjClose.parquet"))
    ac.index = pd.to_datetime(ac.index, utc=True).tz_localize(None).strftime("%Y-%m-%d")

    bench_filled = {}
    for tk in bench_tickers:
        if tk not in ac.columns:
            continue
        s = ac[tk].dropna()
        # Seed with the latest value on or before all_dates[0]
        seed = 0.0
        if all_dates:
            before = s[s.index <= all_dates[0]]
            if len(before):
                seed = float(before.iloc[-1])
        arr = np.zeros(n)
        last = seed
        for i, dd in enumerate(all_dates):
            v = s.get(dd, np.nan)
            if isinstance(v, float) and math.isfinite(v) and v > 0:
                last = v
            arr[i] = last
        bench_filled[tk] = arr

    # month-first index
    month_first_idx = []
    prev_ym = ""
    for i, dd in enumerate(all_dates):
        ym = dd[:7]
        if ym != prev_ym:
            month_first_idx.append(i)
            prev_ym = ym

    stocks = sorted([tk for tk in prices.keys() if tk not in set(bench_tickers)])

    return MarketData(
        all_dates=all_dates, date_idx=date_idx, prices=prices, finals=finals,
        month_first_idx=month_first_idx, bench_filled=bench_filled,
        stocks=stocks, items_by_ticker={},
        washes=washes, finals_raw=finals_raw, qualities=qualities,
    )


def load_and_prep_ext():
    from bt_core import first_valid_month_idx
    md = load_market_ext()
    start_m = first_valid_month_idx(md, md.stocks, min_tickers=3)
    return md, start_m
