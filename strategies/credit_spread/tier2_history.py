"""Emit the Tier 2 (Vol-Alpha) VALIDATED backtest trade history for the
site — the actual 2019-2026 validation trades, on the same $250M
high-liquidity subset production trades, so users can browse Tier 2's
record while its live log accrues.

Reproduces the frozen Tier 2 selection (committed model + threshold),
deduped to independent trades (one per ticker/expiry), priced with the
conservative model, filtered to the production ADV floor, and written
to spreads/docs/data/tier2_history.json:

    { "as_of", "floor_adv_usd", "summary": {...}, "by_year": [...],
      "trades": [ {date, ticker, side, short_strike, long_strike,
                   expiry, credit, ror, outcome, pnl}, ... ] }

Run:  python3 tier2_history.py
"""
from __future__ import annotations

import json
import os

import numpy as np

from pricing import (COMMISSION_PER_SHARE, bs_put, expected_fill_credit,
                     iv_at_strike)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(REPO, "spreads", "docs", "data", "tier2_history.json")
DESIGN_END = np.datetime64("2018-12-31")
MIN_ADV = float(os.environ.get("CS_MIN_ADV_USD", str(250e6)))


def main() -> int:
    import joblib
    dz = np.load(os.path.join(HERE, "results", "sigma_distance_rows.npz"),
                 allow_pickle=False)
    d = {k: dz[k] for k in dz.files}
    meta = json.load(open(os.path.join(HERE, "results", "tier2_meta.json")))
    clf = joblib.load(os.path.join(HERE, "results", "tier2_model.joblib"))
    adv = json.load(open(os.path.join(HERE, "results", "adv.json")))["adv_usd"]
    FE, thr, C, WP = meta["features"], meta["threshold"], meta["c_sigma"], meta["width_pct"]

    dates = d["date"].astype("datetime64[D]")
    X = np.column_stack([d[f] for f in FE])
    finite = np.isfinite(X).all(axis=1)
    m = (d["side"] == "put") & finite & (dates > DESIGN_END)
    p = clf.predict_proba(np.nan_to_num(X, nan=0.0))[:, 1]
    sel = np.where(m & (p >= thr))[0]

    spot, s_exp = d["spot"].astype(float), d["s_exp"].astype(float)
    cal, sig = d["cal_days"].astype(float), d["sigma60"].astype(float)

    seen: set = set()
    trades = []
    for i in sel[np.argsort(d["date"][sel])]:
        t = str(d["ticker"][i])
        exp = str(d["expiry"][i])
        key = (t, exp)
        if key in seen:
            continue
        if adv.get(t, 0.0) < MIN_ADV:
            continue
        b = C * (sig[i] / np.sqrt(252)) * np.sqrt(14)
        Ks = spot[i] * (1 - b)
        Kl = Ks - spot[i] * WP
        if Kl <= 0:
            continue
        T = max(cal[i], 1) / 365.0
        ivs = iv_at_strike(spot[i], Ks, T, sig[i] * 1.3, "put")
        ivl = iv_at_strike(spot[i], Kl, T, sig[i] * 1.3, "put")
        mid = max(bs_put(spot[i], Ks, T, ivs) - bs_put(spot[i], Kl, T, ivl), 0)
        fill, _ = expected_fill_credit(mid, T)
        net = fill - COMMISSION_PER_SHARE
        if net < 0.05:
            continue
        seen.add(key)
        width = spot[i] * WP
        intr = min(max(Ks - s_exp[i], 0), width)
        pnl = (net - intr) * 100.0
        risk = (width - net) * 100.0
        trades.append({
            "date": str(d["date"][i]), "ticker": t, "side": "put",
            "short_strike": round(float(Ks), 2), "long_strike": round(float(Kl), 2),
            "expiry": exp, "credit": round(net * 100.0, 0),
            "ror": round(net / max(width - net, 0.01) * 100.0, 1),
            "close_at_expiry": round(float(s_exp[i]), 2),
            "outcome": "win" if intr == 0 else "loss",
            "pnl": round(pnl, 0),
        })

    # aggregates
    yrs: dict = {}
    for tr in trades:
        y = tr["date"][:4]
        a = yrs.setdefault(y, {"trades": 0, "losses": 0, "pnl": 0.0, "risk": 0.0})
        a["trades"] += 1
        a["losses"] += tr["outcome"] == "loss"
        a["pnl"] += tr["pnl"]
    # per-year ROR needs risk; recompute cheaply from credit+? use pnl/… keep simple: report pnl and win rate
    by_year = []
    for y in sorted(yrs):
        a = yrs[y]
        by_year.append({
            "year": y, "trades": a["trades"], "losses": a["losses"],
            "win_rate": round(100 * (a["trades"] - a["losses"]) / a["trades"], 2),
            "pnl": round(a["pnl"], 0),
        })
    n = len(trades)
    losses = sum(t["outcome"] == "loss" for t in trades)
    tot_pnl = sum(t["pnl"] for t in trades)
    summary = {
        "trades": n, "losses": losses,
        "win_rate": round(100 * (n - losses) / n, 2) if n else None,
        "pnl": round(tot_pnl, 0),
        "worst_trade": round(min((t["pnl"] for t in trades), default=0), 0),
        "avg_ror": round(float(np.mean([t["ror"] for t in trades])), 1) if n else None,
    }
    trades.sort(key=lambda t: (t["date"], t["ticker"]), reverse=True)
    import time
    blob = {
        "as_of": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "engine": "t2-volalpha-gbm",
        "floor_adv_usd": MIN_ADV,
        "window": "2019-2026 validation (untouched)",
        "summary": summary, "by_year": by_year, "trades": trades,
    }
    with open(OUT, "w") as fh:
        json.dump(blob, fh, indent=1)
    print(f"wrote {OUT}: {n} trades, {losses} losses, "
          f"{summary['win_rate']}% win, ${tot_pnl:.0f} P&L")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
