#!/usr/bin/env python3
"""Turn the EDGAR 8-K/2.02 filings into a clean earnings-event table.

  1. dedupe non-earnings Item-2.02 filings (TSLA delivery numbers, ABBV segment
     updates) by requiring a 10-Q/10-K within [-3, +12] days of the 8-K;
  2. convert acceptanceDateTime (UTC) -> America/New_York;
  3. map to the market-moving session per the pre-registered convention;
  4. attach the realized event-session move from adjusted daily bars.

Writes events.parquet
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
BARS = "/home/user/bonds/data/intraday_daily"
EXTRA = os.path.join(HERE, "extra_bars")
OUT = os.path.join(HERE, "events.parquet")


def load_bars() -> dict[str, pd.DataFrame]:
    out = {}
    src = [(BARS, f) for f in sorted(os.listdir(BARS))]
    if os.path.isdir(EXTRA):
        src += [(EXTRA, f) for f in sorted(os.listdir(EXTRA))]
    for root, f in src:
        if not f.endswith(".csv"):
            continue
        sym = f[:-4]
        d = pd.read_csv(os.path.join(root, f))
        if "open" not in d.columns:
            d["open"] = d["high"] = d["low"] = d["close"]
        d["date"] = pd.to_datetime(d["ts"]).dt.tz_localize(None).dt.normalize()
        d = d[["date", "open", "high", "low", "close"]].dropna().sort_values("date")
        d = d.drop_duplicates("date").reset_index(drop=True)
        out[sym] = d
    return out


def main():
    ek = pd.read_parquet(os.path.join(HERE, "earnings_8k.parquet"))
    per = pd.read_parquet(os.path.join(HERE, "periodic.parquet"))

    n0 = len(ek)
    # ---- 1. keep only 8-Ks accompanied by a periodic report --------------
    keep = []
    for sym, g in ek.groupby("symbol"):
        pd_dates = per.loc[per.symbol == sym, "filingDate"].values.astype("datetime64[D]")
        for _, r in g.iterrows():
            d = np.datetime64(r.filingDate, "D")
            gap = (pd_dates - d).astype(int)
            keep.append(bool(((gap >= -3) & (gap <= 12)).any()))
    ek = ek[keep].copy()
    print(f"Item-2.02 8-Ks: {n0} -> {len(ek)} after requiring a 10-Q/10-K within [-3,+12]d")

    # ---- 2. UTC -> ET ----------------------------------------------------
    acc = ek["acceptance"].dt.tz_localize("UTC").dt.tz_convert("America/New_York")
    ek["acc_et"] = acc.dt.tz_localize(None)
    ek["acc_hour_et"] = acc.dt.hour + acc.dt.minute / 60.0

    # ---- 3. event session ------------------------------------------------
    bars = load_bars()
    ek = ek[ek.symbol.isin(bars)].copy()
    print(f"symbols with adjusted daily bars: {ek.symbol.nunique()}")

    rows = []
    for sym, g in ek.groupby("symbol"):
        b = bars[sym]
        dts = b["date"].values
        cl = b["close"].values
        for _, r in g.iterrows():
            d = np.datetime64(r.acc_et.normalize(), "ns")
            after_close = r.acc_hour_et >= 16.0
            i = int(np.searchsorted(dts, d, "left"))
            if after_close:
                # first session strictly after the acceptance day
                while i < len(dts) and dts[i] <= d:
                    i += 1
            else:
                # the acceptance day itself if it trades, else the next session
                while i < len(dts) and dts[i] < d:
                    i += 1
            if i <= 0 or i >= len(dts):
                continue
            rows.append(dict(
                symbol=sym, cik=r.cik, filingDate=r.filingDate,
                acc_et=r.acc_et, acc_hour_et=r.acc_hour_et,
                after_close=after_close,
                event_session=pd.Timestamp(dts[i]),
                prev_session=pd.Timestamp(dts[i - 1]),
                px_prev=cl[i - 1], px_event=cl[i],
                r_event=float(np.log(cl[i] / cl[i - 1])),
            ))
    ev = pd.DataFrame(rows).sort_values(["symbol", "event_session"]).reset_index(drop=True)

    # ---- 4. dedupe residual clusters (keep first of any <45d pair) -------
    ev["gap"] = ev.groupby("symbol").event_session.diff().dt.days
    ev = ev[(ev.gap.isna()) | (ev.gap >= 45)].drop(columns="gap").reset_index(drop=True)

    # trailing sigma (past-only): 63-session close-to-close std ending the day
    # BEFORE the previous session, so the event day itself never enters.
    sig = []
    for sym, g in ev.groupby("symbol"):
        b = bars[sym]
        lr = np.log(b["close"].values[1:] / b["close"].values[:-1])
        bd = b["date"].values[1:]
        for _, r in g.iterrows():
            j = int(np.searchsorted(bd, np.datetime64(r.prev_session, "ns"), "left"))
            w = lr[max(0, j - 63):j]
            sig.append(float(np.std(w, ddof=1)) if len(w) >= 30 else np.nan)
    ev["sigma_tr"] = sig
    ev["z"] = ev.r_event.abs() / ev.sigma_tr
    ev["abs_r"] = ev.r_event.abs()

    # past-only trailing statistics, computed on the FULL event history so that
    # the warm-up requirement does not throw away chain-covered events
    ev = ev.sort_values(["symbol", "event_session"]).reset_index(drop=True)
    g = ev.groupby("symbol")
    ev["prior_logz4"] = g.z.transform(lambda s: np.log(s.clip(lower=1e-4)).shift(1).rolling(4).mean())
    ev["prior_absr4"] = g.abs_r.transform(lambda s: s.shift(1).rolling(4).mean())
    ev["nprior"] = g.cumcount()

    ev.to_parquet(OUT, index=False)
    w = ev[ev.event_session >= "2019-01-01"]
    print(f"\nwrote {OUT}")
    print(f"  events total {len(ev)}  |  2019+ {len(w)}  symbols {w.symbol.nunique()}")
    print(f"  range {ev.event_session.min().date()} .. {ev.event_session.max().date()}")
    print(f"  after-close share {ev.after_close.mean():.1%}")
    print(f"  median |z| {ev.z.median():.2f}   mean |z| {ev.z.mean():.2f}")
    print(f"  median |r_event| {ev.abs_r.median():.3%}")


if __name__ == "__main__":
    main()
