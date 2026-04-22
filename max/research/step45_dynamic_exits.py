#!/usr/bin/env python3
"""Step 45 — Dynamic exits: ATR-scaled SL, trailing, breakeven, time-decay, regime.

Tests whether dynamic (per-trade) exit rules beat the fixed TP10/SL15/TS252
winner. All variants use the same ranker (CAP5+SMA12M top-1 monthly) and
the same entry (next-day close). What changes is how we exit.

The base simulator is in step43_tp_multislot.simulate_multislot (already
has SL + trailing support). This script extends it for additional dynamic
exits that step43 doesn't model: ATR-scaled SL distance, breakeven ratchet,
TP time-decay, regime-aware time-stop.

Output: `max/research/step45_results.json` + `step45_summary.md`.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from step43_tp_multislot import (  # reuse framework pieces
    load_data, compute_rank_signal, month_first_indices, spy_regime_200dma,
    spy_dca_baseline, compute_metrics,
)

DCA_MONTHLY = 1000.0
TRADING_DAYS_YR = 252
RANK_LOOKBACK = 252


def compute_atr14(close: pd.DataFrame, high: pd.DataFrame, low: pd.DataFrame) -> pd.DataFrame:
    """Wilder ATR14 as % of close (same shape as close): atr_pct[i, tk]."""
    prev_close = close.shift(1)
    tr = pd.DataFrame(
        np.maximum.reduce([
            (high - low).to_numpy(),
            (high - prev_close).abs().to_numpy(),
            (low - prev_close).abs().to_numpy(),
        ]),
        index=close.index, columns=close.columns,
    )
    # Wilder ATR = EMA with alpha = 1/14
    atr = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    atr_pct = atr / close
    return atr_pct


def simulate_dynamic(
    close, high, low, rank_signal, atr_pct, spy_above,
    *,
    tp_pct=10.0,
    sl_mode="fixed",          # "fixed" | "atr" | "trail" | "breakeven"
    sl_pct=15.0,              # fixed SL (used when sl_mode="fixed")
    sl_atr_k=3.0,             # for sl_mode="atr": SL = entry * (1 - k * ATR14%)
    trail_pct=None,           # active trailing stop from HWM (fires if price falls trail_pct from HWM)
    breakeven_trigger=None,   # once price >= entry*(1+breakeven_trigger), move SL to entry
    tp_decay=None,            # list of (bar_threshold, tp_pct) pairs; TP drops at each threshold
    ts_mode="fixed",          # "fixed" | "regime"
    ts_bars=252,
    ts_bars_bear=378,         # when SPY < 200dma, use this time_stop instead
    ts_bars_bull=252,
):
    """One-slot TP simulation with configurable dynamic exits."""
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

    tp_mult_init = 1.0 + tp_pct / 100.0
    sl_fixed_mult = 1.0 - sl_pct / 100.0

    for mi, di in enumerate(month_idx):
        total_invested += DCA_MONTHLY
        cash += DCA_MONTHLY
        entry_idx = di + 1
        if entry_idx >= n:
            break

        if open_pos is None:
            # Rank on di
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
                if np.isfinite(entry_px) and entry_px > 0:
                    # Regime-aware time-stop uses SPY > 200dma at rank_date
                    if ts_mode == "regime" and spy_above is not None:
                        ts_use = ts_bars_bull if spy_above[di] else ts_bars_bear
                    else:
                        ts_use = ts_bars

                    # Initial SL
                    if sl_mode == "atr":
                        atr_d = atr_arr[di, best_tk]
                        if np.isfinite(atr_d) and atr_d > 0:
                            init_sl = entry_px * (1.0 - sl_atr_k * atr_d)
                        else:
                            init_sl = entry_px * sl_fixed_mult
                    else:
                        init_sl = entry_px * sl_fixed_mult

                    deploy = cash
                    cash = 0.0
                    shares = deploy / entry_px

                    open_pos = {
                        "tk_idx": best_tk,
                        "entry_idx": entry_idx,
                        "entry_px": entry_px,
                        "tp_px": entry_px * tp_mult_init,
                        "sl_px": init_sl,
                        "shares": shares,
                        "stop_idx": entry_idx + ts_use,
                        "cost": deploy,
                        "hwm": entry_px,
                        "tp_tier": 0,  # for TP decay
                    }

        # Walk to next month
        next_di = month_idx[mi + 1] if (mi + 1) < len(month_idx) else n
        cash, open_pos = _walk(
            equity, di, next_di, open_pos, close_arr, high_arr, low_arr,
            cash, positions, tp_mult_init, trail_pct, breakeven_trigger, tp_decay, n,
        )

    if month_idx and open_pos is not None:
        cash, open_pos = _walk(
            equity, month_idx[-1], n, open_pos, close_arr, high_arr, low_arr,
            cash, positions, tp_mult_init, trail_pct, breakeven_trigger, tp_decay, n,
        )

    for i in range(n):
        if equity[i] == 0 and i > 0:
            equity[i] = equity[i - 1]

    return {
        "equity": equity, "positions": positions, "total_invested": total_invested,
        "open_pos": open_pos, "dates": dates,
    }


def _walk(equity, start_i, end_i, open_pos, close_arr, high_arr, low_arr,
          cash, positions, tp_mult_init, trail_pct, breakeven_trigger, tp_decay, n):
    for d in range(start_i, end_i):
        if open_pos is not None and d > open_pos["entry_idx"]:
            tk = open_pos["tk_idx"]
            hi = high_arr[d, tk]
            lo = low_arr[d, tk]

            # Update HWM
            if np.isfinite(hi) and hi > open_pos["hwm"]:
                open_pos["hwm"] = hi

            # Breakeven ratchet
            if breakeven_trigger is not None and open_pos["hwm"] >= open_pos["entry_px"] * (1.0 + breakeven_trigger):
                if open_pos["sl_px"] < open_pos["entry_px"]:
                    open_pos["sl_px"] = open_pos["entry_px"]

            # Trailing stop (active trail)
            if trail_pct is not None:
                trail_sl = open_pos["hwm"] * (1.0 - trail_pct / 100.0)
                if trail_sl > open_pos["sl_px"]:
                    open_pos["sl_px"] = trail_sl

            # TP decay: based on bars since entry
            if tp_decay is not None:
                bars_held = d - open_pos["entry_idx"]
                # tp_decay is list of (bar_threshold, tp_pct). Apply the last triggered one.
                cur_tp_pct = None
                for thresh, tp_new in tp_decay:
                    if bars_held >= thresh:
                        cur_tp_pct = tp_new
                if cur_tp_pct is not None:
                    open_pos["tp_px"] = open_pos["entry_px"] * (1.0 + cur_tp_pct / 100.0)

            # Exit checks: SL first (worst), then TP, then time-stop
            exit_px = None
            reason = None
            if np.isfinite(lo) and lo <= open_pos["sl_px"]:
                exit_px = open_pos["sl_px"]
                reason = "sl_or_trail"
            elif np.isfinite(hi) and hi >= open_pos["tp_px"]:
                exit_px = open_pos["tp_px"]
                reason = "tp"
            elif d >= open_pos["stop_idx"]:
                px = close_arr[d, tk]
                if not (np.isfinite(px) and px > 0):
                    for back in range(d, open_pos["entry_idx"], -1):
                        p2 = close_arr[back, tk]
                        if np.isfinite(p2) and p2 > 0:
                            px = p2
                            break
                    else:
                        px = open_pos["entry_px"]
                exit_px = px
                reason = "time"

            if exit_px is not None:
                cash += open_pos["shares"] * exit_px
                ret = exit_px / open_pos["entry_px"] - 1.0
                positions.append({
                    "tk": int(tk), "entry_idx": int(open_pos["entry_idx"]),
                    "exit_idx": int(d), "entry_px": float(open_pos["entry_px"]),
                    "exit_px": float(exit_px), "cost": float(open_pos["cost"]),
                    "proceeds": float(open_pos["shares"] * exit_px),
                    "days_held": int(d - open_pos["entry_idx"]),
                    "hit_tp": reason == "tp", "reason": reason,
                    "exit_reason": reason if reason != "sl_or_trail" else "sl",
                    "ret": float(ret),
                })
                open_pos = None

        eq = cash
        if open_pos is not None:
            tk = open_pos["tk_idx"]
            if d < open_pos["entry_idx"]:
                eq += open_pos["cost"]
            else:
                px = close_arr[d, tk]
                if not np.isfinite(px):
                    for back in range(d, open_pos["entry_idx"] - 1, -1):
                        p2 = close_arr[back, tk]
                        if np.isfinite(p2) and p2 > 0:
                            px = p2
                            break
                eq += open_pos["shares"] * px if np.isfinite(px) else open_pos["cost"]
        equity[d] = eq

    return cash, open_pos


def metrics_from_result(result):
    return compute_metrics({
        "equity": result["equity"],
        "total_invested": result["total_invested"],
        "positions": result["positions"],
        "dates": result["dates"],
    })


def main():
    print("Loading data...", flush=True)
    close, high, low, finals = load_data()
    rank_signal = compute_rank_signal(finals)
    atr_pct = compute_atr14(close, high, low)
    spy_above = spy_regime_200dma(close)
    print(f"  shape: {close.shape}  ATR shape: {atr_pct.shape}", flush=True)

    results = {}

    # SPY baseline
    spy = spy_dca_baseline(close)
    results["spy_dca"] = spy
    print(f"\nSPY DCA: CAGR {spy['cagr']*100:.2f}%  MDD {spy['maxdd']*100:.1f}%  Sharpe {spy['sharpe']:.2f}", flush=True)

    # Control: step41 winner (fixed TP10/SL15/TS252)
    print("\n[CTRL] fixed TP10/SL15/TS252...", flush=True)
    ctrl = simulate_dynamic(close, high, low, rank_signal, atr_pct, spy_above)
    m = metrics_from_result(ctrl)
    results["ctrl_fixed_10_15_252"] = m
    print(f"   CAGR {m['cagr']*100:.2f}%  MDD {m['maxdd']*100:.1f}%  Shp {m['sharpe']:.2f}  Cal {m['calmar']:.3f}  WR {m['win_rate']*100:.1f}%  N {m['n_trades']}", flush=True)

    # --- Variant A: ATR-scaled SL ---
    print("\n[A] ATR-scaled SL (TP10, TS252)...", flush=True)
    for k in [1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0]:
        r = simulate_dynamic(close, high, low, rank_signal, atr_pct, spy_above,
                             sl_mode="atr", sl_atr_k=k)
        m = metrics_from_result(r)
        key = f"A_atr_sl_k{k}"
        results[key] = m
        print(f"  k={k:>4}  CAGR {m['cagr']*100:>6.2f}%  MDD {m['maxdd']*100:>5.1f}%  Shp {m['sharpe']:>4.2f}  Cal {m['calmar']:>5.3f}  WR {m['win_rate']*100:>5.1f}%  N {m['n_trades']}", flush=True)

    # --- Variant B: Trailing stop (active from entry) ---
    print("\n[B] Active trailing stop (TP10, TS252, replaces -15% SL)...", flush=True)
    for t in [8, 10, 12, 15, 18, 22]:
        r = simulate_dynamic(close, high, low, rank_signal, atr_pct, spy_above,
                             sl_pct=t, trail_pct=t)  # trail is the effective SL
        m = metrics_from_result(r)
        key = f"B_trail_{t}"
        results[key] = m
        print(f"  trail={t:>3}%  CAGR {m['cagr']*100:>6.2f}%  MDD {m['maxdd']*100:>5.1f}%  Shp {m['sharpe']:>4.2f}  Cal {m['calmar']:>5.3f}  WR {m['win_rate']*100:>5.1f}%  N {m['n_trades']}", flush=True)

    # --- Variant C: Breakeven ratchet ---
    print("\n[C] Breakeven ratchet (TP10/SL15/TS252, advance SL to entry at threshold)...", flush=True)
    for bt in [0.02, 0.03, 0.04, 0.05, 0.06, 0.08]:
        r = simulate_dynamic(close, high, low, rank_signal, atr_pct, spy_above,
                             breakeven_trigger=bt)
        m = metrics_from_result(r)
        key = f"C_breakeven_{int(bt*100)}"
        results[key] = m
        print(f"  BE@+{int(bt*100)}%  CAGR {m['cagr']*100:>6.2f}%  MDD {m['maxdd']*100:>5.1f}%  Shp {m['sharpe']:>4.2f}  Cal {m['calmar']:>5.3f}  WR {m['win_rate']*100:>5.1f}%  N {m['n_trades']}", flush=True)

    # --- Variant D: TP time-decay ---
    print("\n[D] TP time-decay (start 10%, reduce over time)...", flush=True)
    decay_schedules = {
        "D_decay_soft":   [(63, 8.0), (126, 6.0), (189, 4.0)],     # slow
        "D_decay_mod":    [(42, 8.0), (84, 6.0), (126, 4.0), (168, 2.0)],
        "D_decay_aggr":   [(21, 8.0), (63, 5.0), (126, 2.0), (189, 0.0)],
        "D_decay_stair":  [(63, 7.0), (126, 5.0), (189, 3.0)],
    }
    for name, sched in decay_schedules.items():
        r = simulate_dynamic(close, high, low, rank_signal, atr_pct, spy_above,
                             tp_decay=sched)
        m = metrics_from_result(r)
        results[name] = m
        print(f"  {name:<18} CAGR {m['cagr']*100:>6.2f}%  MDD {m['maxdd']*100:>5.1f}%  Shp {m['sharpe']:>4.2f}  Cal {m['calmar']:>5.3f}  WR {m['win_rate']*100:>5.1f}%  N {m['n_trades']}", flush=True)

    # --- Variant E: Regime-aware time-stop ---
    print("\n[E] Regime-aware time-stop (SPY200 bull=X, bear=Y)...", flush=True)
    for ts_bull, ts_bear in [(126, 378), (126, 504), (189, 378), (252, 504), (378, 126)]:
        r = simulate_dynamic(close, high, low, rank_signal, atr_pct, spy_above,
                             ts_mode="regime", ts_bars_bull=ts_bull, ts_bars_bear=ts_bear)
        m = metrics_from_result(r)
        key = f"E_regime_ts_{ts_bull}_{ts_bear}"
        results[key] = m
        print(f"  bull{ts_bull}/bear{ts_bear}  CAGR {m['cagr']*100:>6.2f}%  MDD {m['maxdd']*100:>5.1f}%  Shp {m['sharpe']:>4.2f}  Cal {m['calmar']:>5.3f}  WR {m['win_rate']*100:>5.1f}%  N {m['n_trades']}", flush=True)

    # --- Variant F: Combined "best-of" stacks ---
    print("\n[F] Combined dynamic stacks...", flush=True)

    # F1: ATR SL + breakeven ratchet
    r = simulate_dynamic(close, high, low, rank_signal, atr_pct, spy_above,
                         sl_mode="atr", sl_atr_k=5.0, breakeven_trigger=0.05)
    m = metrics_from_result(r)
    results["F1_atr5_plus_BE5"] = m
    print(f"  ATR5+BE5  CAGR {m['cagr']*100:>6.2f}%  MDD {m['maxdd']*100:>5.1f}%  Cal {m['calmar']:>5.3f}  WR {m['win_rate']*100:>5.1f}%", flush=True)

    # F2: Trail 15 + regime ts
    r = simulate_dynamic(close, high, low, rank_signal, atr_pct, spy_above,
                         trail_pct=15, sl_pct=15,
                         ts_mode="regime", ts_bars_bull=189, ts_bars_bear=378)
    m = metrics_from_result(r)
    results["F2_trail15_regime"] = m
    print(f"  trail15+regimeTS  CAGR {m['cagr']*100:>6.2f}%  MDD {m['maxdd']*100:>5.1f}%  Cal {m['calmar']:>5.3f}  WR {m['win_rate']*100:>5.1f}%", flush=True)

    # F3: ATR SL + regime TS
    r = simulate_dynamic(close, high, low, rank_signal, atr_pct, spy_above,
                         sl_mode="atr", sl_atr_k=5.0,
                         ts_mode="regime", ts_bars_bull=189, ts_bars_bear=378)
    m = metrics_from_result(r)
    results["F3_atr5_regime"] = m
    print(f"  ATR5+regimeTS  CAGR {m['cagr']*100:>6.2f}%  MDD {m['maxdd']*100:>5.1f}%  Cal {m['calmar']:>5.3f}  WR {m['win_rate']*100:>5.1f}%", flush=True)

    # Sort output
    print("\n=== Top 10 by CAGR ===", flush=True)
    items = [(k, v) for k, v in results.items() if v and v.get("cagr", 0) > 0]
    items.sort(key=lambda kv: kv[1].get("cagr", 0), reverse=True)
    for k, v in items[:12]:
        print(f"  {k:<30} CAGR {v['cagr']*100:>6.2f}%  MDD {v['maxdd']*100:>5.1f}%  Shp {v['sharpe']:>4.2f}  Cal {v['calmar']:>5.3f}  WR {v.get('win_rate',0)*100:>5.1f}%  N {v.get('n_trades','-')}", flush=True)

    print("\n=== Top 10 by Calmar ===", flush=True)
    items.sort(key=lambda kv: kv[1].get("calmar", 0), reverse=True)
    for k, v in items[:12]:
        print(f"  {k:<30} CAGR {v['cagr']*100:>6.2f}%  MDD {v['maxdd']*100:>5.1f}%  Shp {v['sharpe']:>4.2f}  Cal {v['calmar']:>5.3f}  WR {v.get('win_rate',0)*100:>5.1f}%  N {v.get('n_trades','-')}", flush=True)

    out_path = Path("/home/user/crt/max/research/step45_results.json")
    out_path.write_text(json.dumps(results, indent=2, default=float))
    print(f"\nWrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
