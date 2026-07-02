"""Systematic conditional-alpha scan for credit-spread selection.

Emits one row per candidate trade (unconditional entries every 3rd
session, both sides, k * sigma * sqrt(h) strikes, conservative fills)
with a broad library of CAUSAL features known at entry, plus the trade
outcome. The analysis step then measures, feature by feature, whether
conditioning moves net expectancy — on the 2008-2018 design window —
and validates surviving features once on 2019-2026.

Features (close-only, causal at entry):
  vr_10_60      sigma10 / sigma60      (vol expanding vs calming)
  vr_60_252     sigma60 / sigma252     (medium vs long vol regime)
  trend200      close / SMA200
  trend50       close / SMA50
  dd252         drawdown from 252d high
  up252         rally off 252d low
  rsi14         Wilder RSI
  ret5, ret21   trailing returns
  gap_age       sessions since last |1d move| > 4 sigma (earnings proxy);
                earnings recur ~63 sessions apart, so gap_age in
                [50, 70] predicts an event inside a 14-session window
  breadth       market % above SMA200 (panel-wide, causal)
  d_breadth10   10-session change in breadth
  sigma60       realized vol level
  dow           day of week (0=Mon)

Run:   CS_DATA_DIR=$PWD/cache_full python3 feature_scan.py        # emit rows
       python3 feature_scan.py analyze                            # decile scan
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

from common import covered_options_expiry, list_tickers, load_series, _rolling_mean
from pricing import COMMISSION_PER_SHARE, bs_call, bs_put, expected_fill_credit, iv_at_strike
from tech_spreads import rolling_std, rsi, sigma60 as sig60_fn, spread_entry

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "feature_rows.npz")
K_SIGMA = 1.5
HORIZON = 14
STRIDE = 3
START = np.datetime64("2008-01-02")
DESIGN_END = np.datetime64("2018-12-31")
WIDTH_PCT = 0.05

FEATURES = ["vr_10_60", "vr_60_252", "trend200", "trend50", "dd252", "up252",
            "rsi14", "ret5", "ret21", "gap_age", "breadth", "d_breadth10",
            "sigma60", "dow"]


def emit() -> int:
    tickers = list_tickers()
    limit = os.environ.get("CS_LIMIT")
    if limit:
        tickers = tickers[: int(limit)]
    with open(os.path.join(HERE, "results", "optionable.json")) as fh:
        optionable = json.load(fh)["optionable"]
    br = json.load(open(os.path.join(HERE, "results", "breadth_full.json")))
    bmap = dict(zip(br["dates"], br["pct_above_sma200"]))

    cols: dict[str, list] = {k: [] for k in
                             ["date", "ticker", "expiry", "side", "win", "pnl", "risk"] + FEATURES}
    expiry_cache: dict = {}
    t0 = time.time()
    for ti, t in enumerate(tickers, 1):
        ts = load_series(t)
        if ts is None or not optionable.get(t, False):
            continue
        close, dates = ts.close, ts.dates
        if (dates[-1] - dates[0]).astype(int) < 3652:
            continue
        n = len(close)
        lr = np.concatenate(([0.0], np.diff(np.log(close))))
        s10 = rolling_std(lr, 10) * np.sqrt(252)
        s60 = rolling_std(lr, 60) * np.sqrt(252)
        s252 = rolling_std(lr, 252) * np.sqrt(252)
        sma200 = _rolling_mean(close, 200)
        sma50 = _rolling_mean(close, 50)
        hi252 = np.full(n, np.nan)
        lo252 = np.full(n, np.nan)
        for i in range(251, n):
            w = close[i - 251:i + 1]
            hi252[i] = w.max()
            lo252[i] = w.min()
        r14 = rsi(close, 14)
        ret5 = np.full(n, np.nan)
        ret5[5:] = close[5:] / close[:-5] - 1
        ret21 = np.full(n, np.nan)
        ret21[21:] = close[21:] / close[:-21] - 1
        # earnings proxy: sessions since last 1-day move > 4 x sigma60
        big = np.abs(lr) > 4.0 * (s60 / np.sqrt(252))
        gap_age = np.full(n, 999.0)
        last = -1
        for i in range(n):
            if big[i]:
                last = i
            if last >= 0:
                gap_age[i] = i - last
        dow = np.array([np.datetime64(d, "D").astype("datetime64[D]").item().weekday()
                        for d in dates])

        i0 = int(np.searchsorted(dates, START))
        for j in range(i0, n, STRIDE):
            if not np.isfinite(s60[j]) or s60[j] <= 0:
                continue
            dkey = (str(dates[j]), HORIZON)
            if dkey not in expiry_cache:
                expiry_cache[dkey] = covered_options_expiry(str(dates[j]), HORIZON)
            snap = expiry_cache[dkey]
            if snap is None:
                continue
            exp_iso, _k, cal_days, _s = snap
            exp_d = np.datetime64(exp_iso)
            if dates[-1] < exp_d:
                continue
            ke = int(np.searchsorted(dates, exp_d, side="right")) - 1
            b = K_SIGMA * (s60[j] / np.sqrt(252)) * np.sqrt(HORIZON)
            for side in ("put", "call"):
                ent = spread_entry(side, close[j], b, cal_days, s60[j])
                if ent is None:
                    continue
                Ks, _Kl, net = ent
                width = close[j] * WIDTH_PCT
                S_T = close[ke]
                intr = (min(max(Ks - S_T, 0), width) if side == "put"
                        else min(max(S_T - Ks, 0), width))
                pnl = (net - intr) * 100.0
                cols["date"].append(str(dates[j]))
                cols["ticker"].append(t)
                cols["expiry"].append(exp_iso)
                cols["side"].append(side)
                cols["win"].append(intr == 0)
                cols["pnl"].append(pnl)
                cols["risk"].append((width - net) * 100.0)
                cols["vr_10_60"].append(s10[j] / s60[j] if s60[j] > 0 else np.nan)
                cols["vr_60_252"].append(s60[j] / s252[j] if np.isfinite(s252[j]) and s252[j] > 0 else np.nan)
                cols["trend200"].append(close[j] / sma200[j] if np.isfinite(sma200[j]) else np.nan)
                cols["trend50"].append(close[j] / sma50[j] if np.isfinite(sma50[j]) else np.nan)
                cols["dd252"].append(1 - close[j] / hi252[j] if np.isfinite(hi252[j]) else np.nan)
                cols["up252"].append(close[j] / lo252[j] - 1 if np.isfinite(lo252[j]) else np.nan)
                cols["rsi14"].append(r14[j])
                cols["ret5"].append(ret5[j])
                cols["ret21"].append(ret21[j])
                cols["gap_age"].append(gap_age[j])
                cols["breadth"].append(bmap.get(str(dates[j]), np.nan))
                db = (bmap.get(str(dates[j]), np.nan)
                      - bmap.get(str(dates[max(j - 10, 0)]), np.nan))
                cols["d_breadth10"].append(db)
                cols["sigma60"].append(s60[j])
                cols["dow"].append(dow[j])
        if ti % 200 == 0:
            print(f"  {ti}/{len(tickers)} rows={len(cols['date'])} "
                  f"{time.time()-t0:.0f}s", flush=True)
    np.savez_compressed(OUT, **{k: np.asarray(v) for k, v in cols.items()})
    print(f"wrote {OUT}: {len(cols['date'])} rows ({time.time()-t0:.0f}s)")
    return 0


def analyze() -> int:
    d = np.load(OUT, allow_pickle=False)
    dates = d["date"]
    design = dates.astype("datetime64[D]") <= DESIGN_END
    pnl, risk = d["pnl"], d["risk"]
    side = d["side"]
    print(f"rows: {len(dates)}  design: {int(design.sum())}  "
          f"validation: {int((~design).sum())}")

    def ror(m):
        return pnl[m].sum() / risk[m].sum() * 100 if risk[m].sum() else np.nan

    for s in ("put", "call"):
        base = side == s
        print(f"\n=== {s.upper()} side — design-window decile scan "
              f"(baseline ror {ror(base & design):.2f}%) ===")
        print(f"{'feature':<12} {'D1 ror':>8} {'D10 ror':>8} {'spread':>8} "
              f"| validation D1/D10")
        for f in FEATURES:
            x = d[f]
            ok = base & np.isfinite(x)
            des = ok & design
            if des.sum() < 5000:
                continue
            q = np.nanpercentile(x[des], [10, 90])
            lo_m, hi_m = des & (x <= q[0]), des & (x >= q[1])
            lo_v = ok & ~design & (x <= q[0])
            hi_v = ok & ~design & (x >= q[1])
            print(f"{f:<12} {ror(lo_m):>7.2f}% {ror(hi_m):>7.2f}% "
                  f"{ror(hi_m)-ror(lo_m):>7.2f}% | "
                  f"{ror(lo_v):>6.2f}% / {ror(hi_v):>6.2f}%")
    return 0


if __name__ == "__main__":
    sys.exit(analyze() if (len(sys.argv) > 1 and sys.argv[1] == "analyze") else emit())
