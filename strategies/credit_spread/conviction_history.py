"""Emit the Conviction Pick validated backtest history.

The Conviction Pick is the single product the site now leads with: at
most ONE trade per week — the highest-model-confidence put spread among
high-liquidity (≥$250M/day) names, published only when it clears the
frozen conviction threshold (design 97th percentile). This reproduces
that selection over the untouched 2019-2026 validation window and
writes spreads/docs/data/conviction_history.json.

Run:  python3 conviction_history.py
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import time

import numpy as np

from pricing import (COMMISSION_PER_SHARE, bs_put, expected_fill_credit,
                     iv_at_strike)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(REPO, "spreads", "docs", "data", "conviction_history.json")
DESIGN_END = np.datetime64("2018-12-31")


def _week(dstr: str) -> str:
    y, w, _ = _dt.date.fromisoformat(dstr).isocalendar()
    return f"{y}-W{w:02d}"


def main() -> int:
    import joblib
    dz = np.load(os.path.join(HERE, "results", "sigma_distance_rows.npz"),
                 allow_pickle=False)
    d = {k: dz[k] for k in dz.files}
    meta = json.load(open(os.path.join(HERE, "results", "tier2_meta.json")))
    clf = joblib.load(os.path.join(HERE, "results", "tier2_model.joblib"))
    adv = json.load(open(os.path.join(HERE, "results", "adv.json")))["adv_usd"]
    FE, C, WP = meta["features"], meta["c_sigma"], meta["width_pct"]
    thr = meta["conviction_threshold"]
    elite_thr = meta.get("elite_threshold", 0.9662)
    min_adv = meta.get("conviction_min_adv_usd", 250e6)

    dates = d["date"].astype("datetime64[D]")
    X = np.column_stack([d[f] for f in FE])
    finite = np.isfinite(X).all(axis=1)
    advarr = np.array([adv.get(t, 0.0) for t in d["ticker"]])
    p = clf.predict_proba(np.nan_to_num(X, nan=0.0))[:, 1]
    base = ((dates > DESIGN_END) & finite & (d["side"] == "put")
            & (advarr >= min_adv) & (p >= thr))
    idx = np.where(base)[0]

    # one pick per ISO week: highest model confidence
    byweek: dict = {}
    for i in idx:
        k = _week(str(d["date"][i]))
        if k not in byweek or p[i] > p[byweek[k]]:
            byweek[k] = i

    spot, s_exp = d["spot"].astype(float), d["s_exp"].astype(float)
    cal, sig = d["cal_days"].astype(float), d["sigma60"].astype(float)
    trades = []
    for i in sorted(byweek.values(), key=lambda i: str(d["date"][i])):
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
        width = spot[i] * WP
        intr = min(max(Ks - s_exp[i], 0), width)
        trades.append({
            "date": str(d["date"][i]), "ticker": str(d["ticker"][i]), "side": "put",
            "spot": round(float(spot[i]), 2),
            "short_strike": round(float(Ks), 2), "long_strike": round(float(Kl), 2),
            "expiry": str(d["expiry"][i]), "credit": round(net * 100.0, 0),
            "ror": round(net / max(width - net, 0.01) * 100.0, 1),
            "confidence": round(float(p[i]) * 100.0, 1),
            "elite": bool(p[i] >= elite_thr),
            "adv_usd": round(float(advarr[i]), 0),
            "close_at_expiry": round(float(s_exp[i]), 2),
            "outcome": "win" if intr == 0 else "loss",
            "pnl": round((net - intr) * 100.0, 0),
        })

    yrs: dict = {}
    for tr in trades:
        a = yrs.setdefault(tr["date"][:4], {"trades": 0, "losses": 0, "pnl": 0.0})
        a["trades"] += 1
        a["losses"] += tr["outcome"] == "loss"
        a["pnl"] += tr["pnl"]
    by_year = [{"year": y, "trades": a["trades"], "losses": a["losses"],
                "win_rate": round(100 * (a["trades"] - a["losses"]) / a["trades"], 1),
                "pnl": round(a["pnl"], 0)} for y, a in sorted(yrs.items())]
    n = len(trades)
    losses = sum(t["outcome"] == "loss" for t in trades)
    el = [t for t in trades if t["elite"]]
    el_losses = sum(t["outcome"] == "loss" for t in el)
    summary = {
        "trades": n, "losses": losses,
        "win_rate": round(100 * (n - losses) / n, 2) if n else None,
        "avg_ror": round(float(np.mean([t["ror"] for t in trades])), 1) if n else None,
        "pnl": round(sum(t["pnl"] for t in trades), 0),
        "worst_trade": round(min((t["pnl"] for t in trades), default=0), 0),
        "per_year": round(n / 7.5, 1),
        "elite": {
            "trades": len(el), "losses": el_losses,
            "win_rate": round(100 * (len(el) - el_losses) / len(el), 2) if el else None,
            "avg_ror": round(float(np.mean([t["ror"] for t in el])), 1) if el else None,
            "per_year": round(len(el) / 7.5, 1),
            "threshold_confidence": round(elite_thr * 100, 1),
        },
    }
    trades.sort(key=lambda t: (t["date"], t["ticker"]), reverse=True)
    blob = {
        "as_of": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "threshold": thr, "min_adv_usd": min_adv,
        "window": "2019-2026 validation (untouched)",
        "summary": summary, "by_year": by_year, "trades": trades,
    }
    with open(OUT, "w") as fh:
        json.dump(blob, fh, indent=1)
    print(f"wrote {OUT}: {n} weekly picks, {losses} losses, "
          f"{summary['win_rate']}% win, {summary['avg_ror']}% avg ROR, "
          f"${summary['pnl']:.0f} P&L, {summary['per_year']}/yr")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
