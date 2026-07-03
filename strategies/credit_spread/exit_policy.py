"""Learned, state-dependent exit policy for Tier 2 trades.

Fixed exit triggers (strike-touch stops, fixed take-profits) always
LOWERED both accuracy and P&L (VALIDATION.md §7): they cannot separate
threats from noise. This experiment tests the adaptive version: a
second GBM that, on each day of an open Tier-2 position, scores
whether exiting now (at next-day modeled cost, with slippage and
closing commissions) beats holding to expiry — trained ONLY on
design-window (2008–2018) trade-days, thresholded on design, evaluated
once on the untouched validation Tier-2 slice.

Daily position-state features:
    frac_time        sessions elapsed / total sessions
    cushion_sig      (S_j - K_short)/(S_j·σ_entry_d·√sessions_left)
    pnl_frac         fraction of max profit captured at today's mark
    ret_norm         (S_j/S_entry - 1) / (σ_entry_d·√elapsed)
    vr_10_60_now     current short/medium vol ratio
    rsi_now          current RSI(14)
    ret5_now         current 5-session return
    sig_entry        σ60 at entry

Run:  CS_DATA_DIR=$PWD/cache_full python3 exit_policy.py
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

from common import load_series
from pricing import COMMISSION_PER_SHARE, bs_put, expected_fill_credit, iv_at_strike
from tech_spreads import rolling_std, rsi

HERE = os.path.dirname(os.path.abspath(__file__))
ROWS = os.path.join(HERE, "results", "sigma_distance_rows.npz")
C_SIGMA = 0.6
WIDTH_PCT = 0.025
DESIGN_END = np.datetime64("2018-12-31")

STATE_FE = ["frac_time", "cushion_sig", "pnl_frac", "ret_norm",
            "vr_10_60_now", "rsi_now", "ret5_now", "sig_entry"]


def entry_net(spot, sig, cal_days, ivm=1.30):
    b = C_SIGMA * (sig / np.sqrt(252)) * np.sqrt(14)
    Ks = spot * (1 - b)
    Kl = Ks - spot * WIDTH_PCT
    if Kl <= 0:
        return None
    T = max(cal_days, 1) / 365.0
    atm = sig * ivm
    ivs = iv_at_strike(spot, Ks, T, atm, "put")
    ivl = iv_at_strike(spot, Kl, T, atm, "put")
    mid = max(bs_put(spot, Ks, T, ivs) - bs_put(spot, Kl, T, ivl), 0)
    fill, _ = expected_fill_credit(mid, T)
    return fill - COMMISSION_PER_SHARE, Ks, Kl


def mark(S, Ks, Kl, T_rem, sig):
    if T_rem <= 0:
        return min(max(Ks - S, 0.0), Ks - Kl)
    atm = sig * 1.30
    ivs = iv_at_strike(S, Ks, T_rem, atm, "put")
    ivl = iv_at_strike(S, Kl, T_rem, atm, "put")
    return max(bs_put(S, Ks, T_rem, ivs) - bs_put(S, Kl, T_rem, ivl), 0.0)


def build_trade_days(sel_idx, d, series_cache):
    """For each selected trade, emit per-day state rows + trade refs."""
    rows = {k: [] for k in STATE_FE}
    refs = []          # (trade_id, day_offset, exit_pnl, spot_day_idx)
    trades = []        # per-trade: hold_pnl, risk, entry fields
    for tid, i in enumerate(sel_idx):
        t = d["ticker"][i]
        if t not in series_cache:
            ts = load_series(t)
            if ts is None:
                series_cache[t] = None
            else:
                lr = np.concatenate(([0.0], np.diff(np.log(ts.close))))
                series_cache[t] = (ts.dates, ts.close,
                                   rolling_std(lr, 10) * np.sqrt(252),
                                   rolling_std(lr, 60) * np.sqrt(252),
                                   rsi(ts.close, 14))
        sc = series_cache[t]
        if sc is None:
            continue
        dates, close, s10a, s60a, r14a = sc
        d0 = np.datetime64(d["date"][i])
        de = np.datetime64(d["expiry"][i])
        j0 = int(np.searchsorted(dates, d0))
        ke = int(np.searchsorted(dates, de, side="right")) - 1
        if ke <= j0 + 1 or dates[j0] != d0:
            continue
        spot = float(close[j0])
        sig = float(d["sigma60"][i])
        cal = float(d["cal_days"][i])
        ent = entry_net(spot, sig, cal)
        if ent is None or ent[0] < 0.05:
            continue
        net, Ks, Kl = ent
        width = spot * WIDTH_PCT
        S_T = float(close[ke])
        intr = min(max(Ks - S_T, 0), width)
        hold_pnl = (net - intr) * 100.0
        trades.append({"hold": hold_pnl, "risk": (width - net) * 100.0,
                       "win_hold": intr == 0, "date": d["date"][i],
                       "n_days": ke - j0})
        sig_d = sig / np.sqrt(252)
        total = ke - j0
        for j in range(j0 + 1, ke):        # decision days (exit fills j+1)
            S = float(close[j])
            elapsed = j - j0
            left = ke - j
            T_rem = max(cal * left / total, 1) / 365.0
            m_now = mark(S, Ks, Kl, T_rem, sig)
            # exit executes at NEXT session close with slippage+commission
            jx = min(j + 1, ke)
            Sx = float(close[jx])
            T_x = max(cal * (ke - jx) / total, 0.5) / 365.0
            cost = mark(Sx, Ks, Kl, T_x, sig)
            debit = min(cost + max(0.05, 0.10 * cost) / 2.0
                        + COMMISSION_PER_SHARE, width)
            exit_pnl = (net - debit) * 100.0
            rows["frac_time"].append(elapsed / total)
            rows["cushion_sig"].append((S - Ks) / max(S * sig_d * np.sqrt(left), 1e-9))
            rows["pnl_frac"].append((net - m_now) / net)
            rows["ret_norm"].append((S / spot - 1) / max(sig_d * np.sqrt(elapsed), 1e-9))
            rows["vr_10_60_now"].append(s10a[j] / s60a[j] if s60a[j] > 0 else np.nan)
            rows["rsi_now"].append(r14a[j])
            rows["ret5_now"].append(float(close[j] / close[max(j - 5, 0)] - 1))
            rows["sig_entry"].append(sig)
            refs.append((len(trades) - 1, elapsed, exit_pnl))
    X = np.column_stack([np.asarray(rows[k], float) for k in STATE_FE])
    refs = np.array(refs, dtype=float)
    return X, refs, trades


def policy_eval(X, refs, trades, clf, tau):
    """Sequential: exit at first decision-day with proba >= tau."""
    proba = clf.predict_proba(np.nan_to_num(X, nan=0.0))[:, 1]
    n_tr = len(trades)
    pnl = np.array([t["hold"] for t in trades])
    exited = np.zeros(n_tr, bool)
    order = np.lexsort((refs[:, 1], refs[:, 0]))   # by trade, then day
    for r in order:
        tid = int(refs[r, 0])
        if exited[tid]:
            continue
        if proba[r] >= tau:
            pnl[tid] = refs[r, 2]
            exited[tid] = True
    risk = np.array([t["risk"] for t in trades])
    wins = pnl > 0
    return (pnl.sum() / risk.sum() * 100, 100 * wins.mean(),
            float(pnl.min()), int(exited.sum()), pnl)


def main() -> int:
    from sklearn.ensemble import HistGradientBoostingClassifier
    import joblib
    dz = np.load(ROWS, allow_pickle=False)
    # materialize: NpzFile decompresses on every key ACCESS; element loops
    # over d[k][i] would re-decompress whole arrays each iteration
    d = {k: dz[k] for k in dz.files}
    dates = d["date"].astype("datetime64[D]")
    design = dates <= DESIGN_END
    side = d["side"]
    FE = json.load(open(os.path.join(HERE, "results", "tier2_meta.json")))["features"]
    Xe = np.column_stack([d[f] for f in FE])
    finite = np.isfinite(Xe).all(axis=1)
    y = d[f"win_c{C_SIGMA}"].astype(int)
    m = (side == "put") & finite

    entry_clf = joblib.load(os.path.join(HERE, "results", "tier2_model.joblib"))
    thr = json.load(open(os.path.join(HERE, "results", "tier2_meta.json")))["threshold"]
    p = entry_clf.predict_proba(Xe)[:, 1]

    # widen the design slice (d97-equivalent) for exit-model training data
    tr_mask = m & design
    p_tr = p[tr_mask]
    order = np.argsort(-p_tr)
    cum = np.cumsum(y[tr_mask][order]) / np.arange(1, order.size + 1)
    thr97 = p_tr[order][np.where(cum >= 0.97)[0].max()]

    # dedup selected trades by (ticker, expiry)
    def dedup(idx):
        seen, keep = set(), []
        for i in idx[np.argsort(d["date"][idx])]:
            k = (d["ticker"][i], d["expiry"][i])
            if k in seen:
                continue
            seen.add(k)
            keep.append(i)
        return np.array(keep)

    sel_design = dedup(np.where(tr_mask & (p >= thr97))[0])
    sel_val = dedup(np.where(m & ~design & (p >= thr))[0])
    print(f"design trades (d97, train exit model): {len(sel_design)}  "
          f"validation trades (d99 tier2): {len(sel_val)}")

    cache: dict = {}
    t0 = time.time()
    Xd, refs_d, trades_d = build_trade_days(sel_design, d, cache)
    Xv, refs_v, trades_v = build_trade_days(sel_val, d, cache)
    print(f"trade-days: design {len(refs_d)}, validation {len(refs_v)} "
          f"({time.time()-t0:.0f}s)")

    # label: exiting now beats holding
    yd = (refs_d[:, 2] > np.array([trades_d[int(t)]["hold"]
                                   for t in refs_d[:, 0]])).astype(int)
    clf = HistGradientBoostingClassifier(max_iter=200, max_depth=4,
                                         random_state=0)
    clf.fit(np.nan_to_num(Xd, nan=0.0), yd)

    hold_v = np.array([t["hold"] for t in trades_v])
    risk_v = np.array([t["risk"] for t in trades_v])
    print(f"\nHOLD baseline (validation): ror {hold_v.sum()/risk_v.sum()*100:.2f}%  "
          f"acc {100*(hold_v>0).mean():.2f}%  worst ${hold_v.min():.0f}")
    print(f"{'tau':>5} {'design ror/acc':>16} | {'VALIDATION ror/acc':>18} "
          f"{'worst$':>7} {'exited%':>8}")
    best = None
    hold_d = np.array([t["hold"] for t in trades_d])
    risk_d = np.array([t["risk"] for t in trades_d])
    acc_hold_d = 100 * (hold_d > 0).mean()
    for tau in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
        rd = policy_eval(Xd, refs_d, trades_d, clf, tau)
        rv = policy_eval(Xv, refs_v, trades_v, clf, tau)
        print(f"{tau:>5.2f} {rd[0]:>7.2f}%/{rd[1]:>6.2f}% | "
              f"{rv[0]:>8.2f}%/{rv[1]:>6.2f}% {rv[2]:>7.0f} "
              f"{100*rv[3]/len(trades_v):>7.1f}%")
        # freeze on design: max design ror subject to design acc >= hold acc
        if rd[1] >= acc_hold_d and (best is None or rd[0] > best[0]):
            best = (rd[0], tau)
    if best:
        print(f"\nfrozen tau (design-selected): {best[1]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
