"""Generate the Conviction Pick's currently-OPEN positions (pending).

The backtest picks are all resolved; a genuine "pending" pick is one the
frozen rule published recently whose expiry hasn't arrived yet. This
scans the last LOOKBACK sessions of the fresh panel, applies the exact
frozen conviction rule (same model, threshold, high-liquidity filter,
one best per ISO week), and keeps every qualifying pick whose snapped
expiry is still in the future — i.e. open. Each is reality-verified
against the live chain and written into signals.json as
``conviction_open`` so live_log.py logs it as a pending Conviction Pick
that resolves at its real expiry close.

Run inside the daily scan (after tier2.py). Honors CS_DATA_DIR.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import sys

import numpy as np

from common import covered_options_expiry, list_tickers, load_series, _rolling_mean
from tech_spreads import rolling_std, rsi
from tier2 import (CONV_ENGINE, C_SIGMA, WIDTH_PCT, HORIZON, FEATURES,
                   MIN_HISTORY_CAL_DAYS, RESULTS, _load_model)

LOOKBACK = int(os.environ.get("CS_OPEN_LOOKBACK", "20"))


def _feature_matrix(close, breadth_now, breadth_10, idxs):
    """Features (FEATURES order) at each index in idxs. NaN row if short."""
    n = len(close)
    lr = np.concatenate(([0.0], np.diff(np.log(close))))
    s10 = rolling_std(lr, 10) * np.sqrt(252)
    s60 = rolling_std(lr, 60) * np.sqrt(252)
    s252 = rolling_std(lr, 252) * np.sqrt(252)
    sma200 = _rolling_mean(close, 200)
    sma50 = _rolling_mean(close, 50)
    r14 = rsi(close, 14)
    big = np.abs(lr) > 4.0 * (s60 / np.sqrt(252))
    out = []
    for j in idxs:
        if j < 252 or not np.isfinite(s60[j]) or s60[j] <= 0:
            out.append(None)
            continue
        hi = close[j - 251:j + 1].max()
        lo = close[j - 251:j + 1].min()
        gap = 999.0
        bidx = np.where(big[:j + 1])[0]
        if len(bidx):
            gap = float(j - bidx[-1])
        f = {
            "vr_10_60": s10[j] / s60[j], "vr_60_252": s60[j] / s252[j]
            if np.isfinite(s252[j]) and s252[j] > 0 else np.nan,
            "trend200": close[j] / sma200[j] if np.isfinite(sma200[j]) else np.nan,
            "trend50": close[j] / sma50[j] if np.isfinite(sma50[j]) else np.nan,
            "dd252": 1 - close[j] / hi, "up252": close[j] / lo - 1,
            "rsi14": r14[j], "ret5": close[j] / close[j - 5] - 1,
            "ret21": close[j] / close[j - 21] - 1, "gap_age": gap,
            "breadth": breadth_now, "d_breadth10": breadth_now - breadth_10,
            "sigma60": s60[j],
        }
        out.append(f if all(np.isfinite(v) for v in f.values()) else None)
    return out


def _iso_week(dstr: str) -> str:
    y, w, _ = _dt.date.fromisoformat(dstr).isocalendar()
    return f"{y}-W{w:02d}"


def main() -> int:
    clf, meta = _load_model()
    conv_thr = meta.get("conviction_threshold", meta["threshold"])
    elite_thr = meta.get("elite_threshold", 0.9662)
    min_adv = meta.get("conviction_min_adv_usd", 250e6)
    with open(os.path.join(RESULTS, "optionable.json")) as fh:
        optionable = json.load(fh)["optionable"]
    try:
        adv = json.load(open(os.path.join(RESULTS, "adv.json")))["adv_usd"]
    except Exception:  # noqa: BLE001
        adv = {}

    tickers = list_tickers()
    limit = os.environ.get("CS_LIMIT")
    if limit:
        tickers = tickers[: int(limit)]

    # breadth today (same as tier2)
    series = {}
    above = above10 = total = total10 = 0
    for t in tickers:
        ts = load_series(t)
        if ts is None or len(ts.close) < 300:
            continue
        sma = _rolling_mean(ts.close, 200)
        if np.isfinite(sma[-1]):
            total += 1
            above += ts.close[-1] >= sma[-1]
        if len(ts.close) > 210 and np.isfinite(sma[-11]):
            total10 += 1
            above10 += ts.close[-11] >= sma[-11]
        series[t] = ts
    breadth_now = above / max(total, 1)
    breadth_10 = above10 / max(total10, 1)
    panel_last = max(str(ts.dates[-1]) for ts in series.values())

    # collect qualifying candidates over the last LOOKBACK sessions
    cands = []   # (date, ticker, proba, spot, sigma60)
    for t, ts in series.items():
        if not optionable.get(t, False) or adv.get(t, 0.0) < min_adv:
            continue
        if (ts.dates[-1] - ts.dates[0]).astype(int) < MIN_HISTORY_CAL_DAYS:
            continue
        close = ts.close
        idxs = list(range(max(len(close) - LOOKBACK, 252), len(close)))
        feats = _feature_matrix(close, breadth_now, breadth_10, idxs)
        for j, f in zip(idxs, feats):
            if f is None:
                continue
            X = np.array([[f[k] for k in FEATURES]])
            proba = float(clf.predict_proba(X)[0, 1])
            if proba >= conv_thr:
                cands.append((str(ts.dates[j]), t, proba, float(close[j]),
                              f["sigma60"]))

    # one best per ISO week; keep only those whose expiry is still open
    byweek = {}
    for c in cands:
        wk = _iso_week(c[0])
        if wk not in byweek or c[2] > byweek[wk][2]:
            byweek[wk] = c

    from reality import ChainCache, verify_rung
    from dataclasses import asdict
    cache = ChainCache()
    open_picks = []
    for _wk, (date, t, proba, spot, sig) in sorted(byweek.items()):
        snap = covered_options_expiry(date, HORIZON)
        if snap is None:
            continue
        exp_iso = snap[0]
        if exp_iso <= panel_last:
            continue  # already resolved — not pending
        b = C_SIGMA * (sig / np.sqrt(252)) * np.sqrt(HORIZON)
        k_short = spot * (1 - b)
        k_long = k_short - spot * WIDTH_PCT
        if k_long <= 0:
            continue
        rs = verify_rung(cache, t, "put", spot, date, HORIZON,
                         k_short, k_long, None, min_net=0.05)
        if rs is None:
            continue
        pick = {
            "engine": CONV_ENGINE, "ticker": t, "side": "put",
            "today_close": spot, "end_date": date, "horizon": HORIZON,
            "gbm_confidence": proba, "elite": bool(proba >= elite_thr),
            "strike": rs.short_strike, "expiry_date": rs.expiry,
            "buffer_pct": rs.real_buffer_pct, "variant": "gbm",
            "real": asdict(rs), "status": "pending",
            "ladder": [{"engine": CONV_ENGINE, "horizon": HORIZON,
                        "expiry_date": rs.expiry, "strike": rs.short_strike,
                        "buffer_pct": rs.real_buffer_pct, "variant": "gbm"}],
        }
        open_picks.append(pick)

    sig_path = os.path.join(RESULTS, "signals.json")
    with open(sig_path) as fh:
        blob = json.load(fh)
    blob["conviction_open"] = open_picks
    blob.setdefault("summary", {})["conviction_open"] = {
        "n_open": len(open_picks), "panel_last": panel_last,
        "lookback_sessions": LOOKBACK,
        "tickers": [p["ticker"] for p in open_picks],
    }
    with open(sig_path, "w") as fh:
        json.dump(blob, fh, indent=2)
    print(f"conviction_open: {len(open_picks)} open/pending picks "
          f"(panel_last={panel_last}): {[p['ticker'] for p in open_picks]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
