"""Pure stock-outcome scan at fixed sigma-distances — no options
pricing anywhere, per the 'assume the credit, grade on the stock'
framing.

For weekly entry candidates (every 5th session, both directions), and
strike distances d = c * sigma60_daily * sqrt(14) for c in CS, record
whether the stock closed on the safe side of the strike at the snapped
expiry (win), together with the causal feature library from
feature_scan. The analysis maps the achievable accuracy frontier:

  - unconditional no-breach rate per distance c
  - vol-crush-gated rate (frozen thresholds from §12)
  - gradient-boosted composite of ALL features (fit on 2008-2018 only,
    validated once on 2019-2026, top-decile confidence slice)

and therefore answers directly: at the distance where a spread pays
~1/3 of width (c ~ 0.4-0.6), what accuracy can ANY observable
conditioning reach — and at what distance does 95% accuracy first
appear, with the fair credit that distance actually commands.

Run:  CS_DATA_DIR=$PWD/cache_full python3 sigma_distance_scan.py
      python3 sigma_distance_scan.py analyze
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

from common import covered_options_expiry, list_tickers, load_series, _rolling_mean
from tech_spreads import rolling_std, rsi

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "sigma_distance_rows.npz")
CS = [0.4, 0.6, 0.8, 1.0, 1.2]
HORIZON = 14
STRIDE = 5                     # weekly cadence
START = np.datetime64("2008-01-02")
DESIGN_END = np.datetime64("2018-12-31")

FEATURES = ["vr_10_60", "vr_60_252", "trend200", "trend50", "dd252", "up252",
            "rsi14", "ret5", "ret21", "gap_age", "breadth", "d_breadth10",
            "sigma60"]


def emit() -> int:
    tickers = list_tickers()
    limit = os.environ.get("CS_LIMIT")
    if limit:
        tickers = tickers[: int(limit)]
    with open(os.path.join(HERE, "results", "optionable.json")) as fh:
        optionable = json.load(fh)["optionable"]
    br = json.load(open(os.path.join(HERE, "results", "breadth_full.json")))
    bmap = dict(zip(br["dates"], br["pct_above_sma200"]))

    win_cols = [f"win_c{c}" for c in CS]
    cols: dict[str, list] = {k: [] for k in
                             ["date", "ticker", "side", "expiry", "spot",
                              "s_exp", "cal_days"] + win_cols + FEATURES}
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
        big = np.abs(lr) > 4.0 * (s60 / np.sqrt(252))
        gap_age = np.full(n, 999.0)
        last = -1
        for i in range(n):
            if big[i]:
                last = i
            if last >= 0:
                gap_age[i] = i - last

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
            exp_iso, _kind, cal_days, _sess = snap
            exp_d = np.datetime64(exp_iso)
            if dates[-1] < exp_d:
                continue
            ke = int(np.searchsorted(dates, exp_d, side="right")) - 1
            move = close[ke] / close[j] - 1.0
            sig_h = (s60[j] / np.sqrt(252)) * np.sqrt(HORIZON)
            for side, sgn in (("put", -1.0), ("call", 1.0)):
                cols["date"].append(str(dates[j]))
                cols["ticker"].append(t)
                cols["side"].append(side)
                cols["expiry"].append(exp_iso)
                cols["spot"].append(close[j])
                cols["s_exp"].append(close[ke])
                cols["cal_days"].append(cal_days)
                for c in CS:
                    b = c * sig_h
                    ok = move >= -b if side == "put" else move <= b
                    cols[f"win_c{c}"].append(ok)
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
                cols["d_breadth10"].append(
                    bmap.get(str(dates[j]), np.nan)
                    - bmap.get(str(dates[max(j - 10, 0)]), np.nan))
                cols["sigma60"].append(s60[j])
        if ti % 200 == 0:
            print(f"  {ti}/{len(tickers)} rows={len(cols['date'])} "
                  f"{time.time()-t0:.0f}s", flush=True)
    np.savez_compressed(OUT, **{k: np.asarray(v) for k, v in cols.items()})
    print(f"wrote {OUT}: {len(cols['date'])} rows ({time.time()-t0:.0f}s)")
    return 0


def analyze() -> int:
    d = np.load(OUT, allow_pickle=False)
    dates = d["date"].astype("datetime64[D]")
    design = dates <= DESIGN_END
    side = d["side"]
    X = np.column_stack([d[f] for f in FEATURES])
    finite = np.isfinite(X).all(axis=1)
    vr, vc = d["vr_60_252"], d["vr_10_60"]
    crush = finite & (vr >= 1.183) & (vc <= 0.858)   # frozen §12 gate

    from sklearn.ensemble import HistGradientBoostingClassifier

    print(f"rows: {len(dates)}  (weekly cadence, both sides, h={HORIZON})")
    print(f"{'c(sigma)':>8} {'uncond val':>11} {'vol-crush val':>13} "
          f"{'GBM top-10% val':>15} {'n(top10%)':>10}")
    for c in CS:
        y = d[f"win_c{c}"].astype(int)
        res = {}
        for s in ("put", "call"):
            b = (side == s) & finite
            tr = b & design
            va = b & ~design
            res[s] = (100 * y[va].mean(),
                      100 * y[va & crush].mean() if (va & crush).sum() else np.nan)
            # GBM composite: fit design, score validation, take the most
            # confident decile
            clf = HistGradientBoostingClassifier(max_iter=150, max_depth=4,
                                                 random_state=0)
            clf.fit(X[tr], y[tr])
            p = clf.predict_proba(X[va])[:, 1]
            thr = np.percentile(p, 90)
            top = p >= thr
            res[s] += (100 * y[va][top].mean(), int(top.sum()))
        # average the two sides for the table (they're near-symmetric)
        u = (res["put"][0] + res["call"][0]) / 2
        g = (res["put"][1] + res["call"][1]) / 2
        m = (res["put"][2] + res["call"][2]) / 2
        n10 = res["put"][3] + res["call"][3]
        print(f"{c:>8.1f} {u:>10.2f}% {g:>12.2f}% {m:>14.2f}% {n10:>10d}")
    return 0


if __name__ == "__main__":
    sys.exit(analyze() if (len(sys.argv) > 1 and sys.argv[1] == "analyze")
             else emit())
