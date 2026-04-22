#!/usr/bin/env python3
"""Step 46 — ATR-scaled SL with a −15% floor (tail-cap).

step45 found ATR-scaled SL (k=5) delivers +13.51% CAGR / 77% WR, up from
fixed 10/-15/252d's +11.13% / 70.5% — but MDD jumps from 48% to 69%
because very-high-vol names (SMCI, MARA, COIN) get -20%+ stops that let
losers drift.

Hypothesis: cap the ATR-scaled SL at −15% max loss. Keep the volatility
scaling benefit for low-vol names (KO, PG get tighter stops, improving
Calmar per-trade) while preserving the tail cap for high-vol names.

Formula: sl_frac = min(k × ATR14_pct, SL_CAP).

Also tests ATR-scaled TP (wider TP for volatile stocks since 10% is
small relative to their daily noise) with fixed SL.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from step43_tp_multislot import (
    load_data, compute_rank_signal, month_first_indices, spy_regime_200dma,
    spy_dca_baseline, compute_metrics,
)
from step45_dynamic_exits import compute_atr14, _walk

DCA_MONTHLY = 1000.0
TRADING_DAYS_YR = 252
RANK_LOOKBACK = 252


def simulate_capped_atr(
    close, high, low, rank_signal, atr_pct,
    *,
    tp_pct=10.0,
    tp_atr_k=None,        # if set, TP = entry * (1 + k * ATR14%) capped at TP_MAX
    tp_max=30.0,
    tp_floor=5.0,
    sl_atr_k=5.0,
    sl_cap=15.0,          # max SL loss in pct
    sl_floor=5.0,         # min SL loss in pct
    ts_bars=252,
):
    dates = close.index
    n = len(dates)
    month_idx = month_first_indices(dates)
    tickers = [t for t in close.columns if t != "SPY"]
    close_arr = close[tickers].to_numpy()
    high_arr = high[tickers].to_numpy()
    low_arr = low[tickers].to_numpy()
    rank_arr = rank_signal[tickers].to_numpy()
    atr_arr = atr_pct[tickers].to_numpy()
    n_tk = len(tickers)
    price_valid = (
        pd.DataFrame(close_arr).notna()
        .rolling(RANK_LOOKBACK, min_periods=RANK_LOOKBACK).sum().to_numpy()
    )

    cash = 0.0
    equity = np.zeros(n)
    positions = []
    open_pos = None
    total_invested = 0.0

    for mi, di in enumerate(month_idx):
        total_invested += DCA_MONTHLY
        cash += DCA_MONTHLY
        entry_idx = di + 1
        if entry_idx >= n:
            break

        if open_pos is None:
            scores = rank_arr[di]
            valid_hist = price_valid[di]
            best_tk = -1
            best_s = -np.inf
            for ti in range(n_tk):
                s = scores[ti]
                px_rank = close_arr[di, ti]
                if not (np.isfinite(s) and np.isfinite(px_rank) and px_rank > 0 and valid_hist[ti] >= RANK_LOOKBACK):
                    continue
                if s > best_s:
                    best_s = s
                    best_tk = ti
            if best_tk >= 0:
                entry_px = close_arr[entry_idx, best_tk]
                atr_d = atr_arr[di, best_tk]
                if np.isfinite(entry_px) and entry_px > 0 and np.isfinite(atr_d) and atr_d > 0:
                    # SL: scaled and capped
                    sl_frac = max(sl_floor / 100.0, min(sl_atr_k * atr_d, sl_cap / 100.0))
                    sl_px = entry_px * (1.0 - sl_frac)
                    # TP: optionally scaled
                    if tp_atr_k is not None:
                        tp_frac = max(tp_floor / 100.0, min(tp_atr_k * atr_d, tp_max / 100.0))
                    else:
                        tp_frac = tp_pct / 100.0
                    tp_px = entry_px * (1.0 + tp_frac)

                    deploy = cash
                    cash = 0.0
                    shares = deploy / entry_px
                    open_pos = {
                        "tk_idx": best_tk,
                        "entry_idx": entry_idx,
                        "entry_px": entry_px,
                        "tp_px": tp_px,
                        "sl_px": sl_px,
                        "shares": shares,
                        "stop_idx": entry_idx + ts_bars,
                        "cost": deploy,
                        "hwm": entry_px,
                        "tp_frac": tp_frac,
                        "sl_frac": sl_frac,
                    }

        next_di = month_idx[mi + 1] if (mi + 1) < len(month_idx) else n
        # Reuse step45's _walk (simple TP/SL/time-stop, no trailing here)
        cash, open_pos = _walk(
            equity, di, next_di, open_pos, close_arr, high_arr, low_arr,
            cash, positions, 1.0 + tp_pct / 100.0,
            None, None, None, n,  # no trail, no BE, no TP decay
        )

    if month_idx and open_pos is not None:
        cash, open_pos = _walk(
            equity, month_idx[-1], n, open_pos, close_arr, high_arr, low_arr,
            cash, positions, 1.0 + tp_pct / 100.0,
            None, None, None, n,
        )

    for i in range(n):
        if equity[i] == 0 and i > 0:
            equity[i] = equity[i - 1]

    return {"equity": equity, "positions": positions,
            "total_invested": total_invested, "dates": dates}


def main():
    print("Loading data...", flush=True)
    close, high, low, finals = load_data()
    rank_signal = compute_rank_signal(finals)
    atr_pct = compute_atr14(close, high, low)
    spy = spy_dca_baseline(close)

    results = {"spy_dca": spy}
    print(f"\nSPY DCA: CAGR {spy['cagr']*100:.2f}%  MDD {spy['maxdd']*100:.1f}%", flush=True)

    # Control: Fixed 10/-15/252 from step41
    ctrl = simulate_capped_atr(close, high, low, rank_signal, atr_pct,
                                sl_atr_k=1000.0, sl_cap=15.0, sl_floor=15.0)  # force fixed -15%
    m = compute_metrics({**ctrl, "positions": ctrl["positions"]})
    # Fix positions key for compute_metrics
    def _m(r):
        r2 = {**r}
        # ensure exit_reason present
        for p in r2["positions"]:
            if "exit_reason" not in p:
                p["exit_reason"] = p.get("reason") if p.get("reason") != "sl_or_trail" else "sl"
        return compute_metrics(r2)

    m = _m(ctrl)
    results["ctrl_fixed_10_15"] = m
    print(f"\n[CTRL] CAGR {m['cagr']*100:>6.2f}%  MDD {m['maxdd']*100:>5.1f}%  Cal {m['calmar']:>5.3f}  WR {m['win_rate']*100:>5.1f}%  N {m['n_trades']}", flush=True)

    # --- Variant A: ATR-scaled SL with -15% cap ---
    print("\n[A] ATR-scaled SL, TP fixed 10%, capped at SL_cap, floor 5%...", flush=True)
    for k_sl in [3, 4, 5, 6, 7, 8, 10]:
        for sl_cap in [12, 15, 18, 20, 25]:
            r = simulate_capped_atr(close, high, low, rank_signal, atr_pct,
                                     sl_atr_k=k_sl, sl_cap=sl_cap, sl_floor=5)
            m = _m(r)
            key = f"A_atrSL_k{k_sl}_cap{sl_cap}"
            results[key] = m
            if m['cagr'] > 0.10:  # print only interesting ones
                print(f"  k={k_sl}  cap={sl_cap}  CAGR {m['cagr']*100:>6.2f}%  MDD {m['maxdd']*100:>5.1f}%  Cal {m['calmar']:>5.3f}  WR {m['win_rate']*100:>5.1f}%  N {m['n_trades']}", flush=True)

    # --- Variant B: ATR-scaled TP (no SL scaling) ---
    print("\n[B] ATR-scaled TP (SL fixed -15%, TS 252)...", flush=True)
    for k_tp in [3, 4, 5, 6, 7, 8, 10]:
        r = simulate_capped_atr(close, high, low, rank_signal, atr_pct,
                                 tp_atr_k=k_tp, tp_max=30, tp_floor=5,
                                 sl_atr_k=1000.0, sl_cap=15, sl_floor=15)
        m = _m(r)
        key = f"B_atrTP_k{k_tp}"
        results[key] = m
        print(f"  k={k_tp}  CAGR {m['cagr']*100:>6.2f}%  MDD {m['maxdd']*100:>5.1f}%  Cal {m['calmar']:>5.3f}  WR {m['win_rate']*100:>5.1f}%  N {m['n_trades']}", flush=True)

    # --- Variant C: ATR-scaled BOTH TP and SL (with caps) ---
    print("\n[C] ATR-scaled TP AND SL (both capped)...", flush=True)
    for k_tp, k_sl in [(5, 5), (5, 7), (7, 5), (7, 7), (5, 4), (4, 5), (6, 6), (10, 5), (10, 7)]:
        r = simulate_capped_atr(close, high, low, rank_signal, atr_pct,
                                 tp_atr_k=k_tp, tp_max=25, tp_floor=5,
                                 sl_atr_k=k_sl, sl_cap=15, sl_floor=5)
        m = _m(r)
        key = f"C_atrBoth_tp{k_tp}_sl{k_sl}"
        results[key] = m
        print(f"  k_tp={k_tp} k_sl={k_sl}  CAGR {m['cagr']*100:>6.2f}%  MDD {m['maxdd']*100:>5.1f}%  Cal {m['calmar']:>5.3f}  WR {m['win_rate']*100:>5.1f}%  N {m['n_trades']}", flush=True)

    # --- Final top rankings ---
    print("\n=== Top 10 by CAGR (beats SPY) ===", flush=True)
    items = [(k, v) for k, v in results.items() if v and v.get("cagr", 0) > spy["cagr"]]
    items.sort(key=lambda kv: kv[1].get("cagr", 0), reverse=True)
    for k, v in items[:12]:
        print(f"  {k:<28} CAGR {v['cagr']*100:>6.2f}%  MDD {v['maxdd']*100:>5.1f}%  Shp {v['sharpe']:>4.2f}  Cal {v['calmar']:>5.3f}  WR {v.get('win_rate',0)*100:>5.1f}%  N {v.get('n_trades','-')}", flush=True)

    print("\n=== Top 10 by Calmar (beats SPY) ===", flush=True)
    items.sort(key=lambda kv: kv[1].get("calmar", 0), reverse=True)
    for k, v in items[:12]:
        print(f"  {k:<28} CAGR {v['cagr']*100:>6.2f}%  MDD {v['maxdd']*100:>5.1f}%  Shp {v['sharpe']:>4.2f}  Cal {v['calmar']:>5.3f}  WR {v.get('win_rate',0)*100:>5.1f}%  N {v.get('n_trades','-')}", flush=True)

    out_path = Path("/home/user/crt/max/research/step46_results.json")
    out_path.write_text(json.dumps(results, indent=2, default=float))
    print(f"\nWrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
