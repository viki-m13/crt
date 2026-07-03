"""How do the two tiers perform once the liquidity filter is applied?

The reality-layer liquidity gate has three parts; only the underlying
ADV floor can be applied HISTORICALLY (we have current ADV per name in
results/adv.json; historical option OI / bid-ask spreads don't exist).
So this measures the honest, applicable question: restrict each tier's
untouched-2019-2026 validation trade set to names that clear the ADV
floor, and report accuracy + ROR. The OI and short-leg-spread gates
only ever remove MORE trades (they never add), and act on the contract
at publication time, not on the historical outcome distribution.

Tier 1 (Sigma-Clear) is rebuilt from the deep replay; Tier 2 (Vol-Alpha)
from the sigma-distance rows + the frozen model. Both dedup to
independent trades and price with the same conservative model used
throughout (VALIDATION.md).

Run:  python3 liquidity_impact.py
"""
from __future__ import annotations

import glob
import json
import os
import re

import numpy as np

from replay_analysis import HERE, attach_pricing, load_rows
from research import CAP_LONG, CAP_SHORT, HIST_CLEAR, K_SIGMA
from pricing import (COMMISSION_PER_SHARE, MIN_TRADEABLE_FILL, bs_put,
                     expected_fill_credit, iv_at_strike)

DESIGN_END = np.datetime64("2018-12-31")
ADV = json.load(open(os.path.join(HERE, "results", "adv.json")))["adv_usd"]
FLOORS = [0.0, 50e6, 100e6, 250e6]


def ticker_starts() -> dict[str, str]:
    starts = {}
    for p in glob.glob(os.path.join(HERE, "cache_full", "*.json")):
        t = os.path.basename(p)[:-5]
        head = open(p).read(200)
        m = re.search(r'"dates": \["(\d{4}-\d{2}-\d{2})"', head)
        if m:
            starts[t] = m.group(1)
    return starts


def report(name, tickers, wins, pnl, risk):
    print(f"\n=== {name} — validation 2019-2026, by underlying ADV floor ===")
    print(f"{'ADV floor':>12} {'trades':>7} {'names':>6} {'accuracy':>9} "
          f"{'ROR/trade':>10} {'P&L $':>10}")
    adv_arr = np.array([ADV.get(t, 0.0) for t in tickers])
    for f in FLOORS:
        m = adv_arr >= f if f > 0 else np.ones(len(tickers), bool)
        if not m.sum():
            continue
        acc = 100 * wins[m].mean()
        ror = pnl[m].sum() / risk[m].sum() * 100
        lab = "none" if f == 0 else f"${f/1e6:.0f}M/day"
        print(f"{lab:>12} {int(m.sum()):>7} {len(set(tickers[m])):>6} "
              f"{acc:>8.2f}% {ror:>9.2f}% {pnl[m].sum():>10.0f}")


def tier1():
    R = load_rows(os.path.join(HERE, "results", "replay_rows_full.csv.gz"))
    res = R["win"] >= 0
    is_put = R["side"] == "put"
    h = R["horizon"]
    sig_d = R["sigma60"] / np.sqrt(252)
    starts = ticker_starts()
    opt = json.load(open(os.path.join(HERE, "results", "optionable.json")))["optionable"]
    optionable = np.array([opt.get(t, False) for t in R["ticker"]])
    start_arr = np.array([starts.get(t, "2026-01-01") for t in R["ticker"]])
    years10 = (R["date"].astype("datetime64[D]")
               - start_arr.astype("datetime64[D]")).astype(int) >= 3652
    caps = np.where(h >= 42, CAP_LONG, CAP_SHORT)
    b = K_SIGMA * sig_d * np.sqrt(h) + 0.01
    okb = np.isfinite(b) & (b <= caps) & (b >= HIST_CLEAR * R["hist_max"])
    Rt = dict(R)
    Rt["buffer"] = b
    attach_pricing(Rt, iv_mult=1.30)
    pub = optionable & years10 & okb & (Rt["net"] >= MIN_TRADEABLE_FILL) & res
    S, width, net = R["spot"], Rt["width"], Rt["net"]
    Ks = np.where(is_put, S * (1 - b), S * (1 + b))
    S_T = R["close_at_expiry"]
    intr = np.where(is_put, np.minimum(np.maximum(Ks - S_T, 0), width),
                    np.minimum(np.maximum(S_T - Ks, 0), width))
    pnl = (net - intr) * 100.0
    val = R["date"] > "2018-12-31"
    seen, keep = set(), []
    for i in np.where(pub & val)[0][np.argsort(R["date"][pub & val])]:
        k = (R["ticker"][i], R["side"][i], int(h[i]), R["expiry_date"][i])
        if k in seen:
            continue
        seen.add(k)
        keep.append(i)
    keep = np.array(keep)
    report("TIER 1 (Sigma-Clear)", R["ticker"][keep], (intr[keep] == 0),
           pnl[keep], ((width - net) * 100.0)[keep])


def tier2():
    import joblib
    dz = np.load(os.path.join(HERE, "results", "sigma_distance_rows.npz"),
                 allow_pickle=False)
    d = {k: dz[k] for k in dz.files}
    meta = json.load(open(os.path.join(HERE, "results", "tier2_meta.json")))
    clf = joblib.load(os.path.join(HERE, "results", "tier2_model.joblib"))
    FE, thr, C, WP = meta["features"], meta["threshold"], meta["c_sigma"], meta["width_pct"]
    dates = d["date"].astype("datetime64[D]")
    X = np.column_stack([d[f] for f in FE])
    finite = np.isfinite(X).all(axis=1)
    m = (d["side"] == "put") & finite & (dates > DESIGN_END)
    p = clf.predict_proba(np.nan_to_num(X, nan=0.0))[:, 1]
    sel = np.where(m & (p >= thr))[0]
    seen, keep = set(), []
    for i in sel[np.argsort(d["date"][sel])]:
        k = (d["ticker"][i], d["expiry"][i])
        if k in seen:
            continue
        seen.add(k)
        keep.append(i)
    spot, s_exp = d["spot"].astype(float), d["s_exp"].astype(float)
    cal, sig = d["cal_days"].astype(float), d["sigma60"].astype(float)
    tk, wins, pnl, risk = [], [], [], []
    for i in keep:
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
        tk.append(d["ticker"][i])
        wins.append(intr == 0)
        pnl.append((net - intr) * 100)
        risk.append((width - net) * 100)
    report("TIER 2 (Vol-Alpha)", np.array(tk), np.array(wins),
           np.array(pnl), np.array(risk))


if __name__ == "__main__":
    tier1()
    tier2()
