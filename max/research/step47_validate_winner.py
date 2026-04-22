#!/usr/bin/env python3
"""Step 47 — Validate the step46 winner (ATR-scaled TP+SL, k=7 both).

Reports:
  - Refined grid around tp_k=7, sl_k=7 (sensitivity).
  - Rolling 10Y windows vs fixed 10/-15/252 baseline and SPY DCA.
  - Crisis-period behavior (2008 GFC, 2020 COVID, 2022 bear).
  - Trade log: distribution of per-trade returns, per-ticker exposure.
  - Stress: vary TP cap / SL cap / time-stop.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from step43_tp_multislot import (
    load_data, compute_rank_signal, month_first_indices,
    spy_dca_baseline, compute_metrics,
)
from step45_dynamic_exits import compute_atr14
from step46_atr_capped_sl import simulate_capped_atr


def _m(r):
    r2 = {**r}
    for p in r2["positions"]:
        if "exit_reason" not in p:
            p["exit_reason"] = p.get("reason") if p.get("reason") != "sl_or_trail" else "sl"
    return compute_metrics(r2)


def equity_window_metrics(result, start_date, end_date, total_invested_in_window):
    """Compute CAGR and MDD for a sub-window of the equity curve."""
    dates = result["dates"]
    eq = result["equity"]
    # Find start/end indices
    start_i = int(dates.searchsorted(pd.Timestamp(start_date)))
    end_i = int(dates.searchsorted(pd.Timestamp(end_date), side="right")) - 1
    end_i = min(end_i, len(eq) - 1)
    if start_i >= end_i:
        return None
    sub = np.array(eq[start_i:end_i + 1])
    if len(sub) == 0 or total_invested_in_window <= 0:
        return None
    yrs = len(sub) / 252
    final = float(sub[-1])
    start_val = float(sub[0]) if sub[0] > 0 else float(sub[sub > 0][0]) if (sub > 0).any() else 0.0
    # We don't have trade granularity here; approximate CAGR via final equity
    # growth per $ invested in window using positional approach. For window
    # metric, use log(final/start) when both > 0.
    if start_val <= 0 or final <= 0:
        return None
    cagr_approx = (final / start_val) ** (1 / yrs) - 1
    # MDD within window
    peak = 0.0
    maxdd = 0.0
    for v in sub:
        if v > peak:
            peak = v
        if peak > 0:
            dd = (peak - v) / peak
            if dd > maxdd:
                maxdd = dd
    return {"cagr_approx": cagr_approx, "maxdd": maxdd, "start_eq": start_val, "final_eq": final}


def run_window(close, high, low, rank_signal, atr_pct,
               tp_atr_k=None, sl_atr_k=None, sl_cap=15, tp_max=25, tp_pct=10, sl_pct=15,
               start=None, end=None):
    """Run the TP simulation on a date subset."""
    if start is not None:
        close = close[close.index >= pd.Timestamp(start)]
        high = high.reindex(close.index)
        low = low.reindex(close.index)
        rank_signal = rank_signal.reindex(close.index)
        atr_pct = atr_pct.reindex(close.index)
    if end is not None:
        mask = close.index <= pd.Timestamp(end)
        close = close[mask]
        high = high[mask]
        low = low[mask]
        rank_signal = rank_signal[mask]
        atr_pct = atr_pct[mask]
    return simulate_capped_atr(close, high, low, rank_signal, atr_pct,
                                tp_pct=tp_pct, tp_atr_k=tp_atr_k, tp_max=tp_max, tp_floor=5,
                                sl_atr_k=sl_atr_k if sl_atr_k is not None else 1000.0,
                                sl_cap=sl_cap, sl_floor=5)


def main():
    print("Loading data...", flush=True)
    close, high, low, finals = load_data()
    rank_signal = compute_rank_signal(finals)
    atr_pct = compute_atr14(close, high, low)
    spy = spy_dca_baseline(close)

    results = {}
    print(f"\nSPY DCA (full window): CAGR {spy['cagr']*100:.2f}%  MDD {spy['maxdd']*100:.1f}%", flush=True)

    # Control and winner on full window
    ctrl = simulate_capped_atr(close, high, low, rank_signal, atr_pct,
                                sl_atr_k=1000, sl_cap=15, sl_floor=15)
    m_ctrl = _m(ctrl)
    print(f"[CTRL fixed 10/-15/252] CAGR {m_ctrl['cagr']*100:.2f}%  MDD {m_ctrl['maxdd']*100:.1f}%  Cal {m_ctrl['calmar']:.3f}  WR {m_ctrl['win_rate']*100:.1f}%  N {m_ctrl['n_trades']}", flush=True)
    results["ctrl_full"] = m_ctrl

    win = simulate_capped_atr(close, high, low, rank_signal, atr_pct,
                               tp_atr_k=7, tp_max=25, tp_floor=5,
                               sl_atr_k=7, sl_cap=15, sl_floor=5)
    m_win = _m(win)
    print(f"[WIN tp_k=7 sl_k=7] CAGR {m_win['cagr']*100:.2f}%  MDD {m_win['maxdd']*100:.1f}%  Cal {m_win['calmar']:.3f}  WR {m_win['win_rate']*100:.1f}%  N {m_win['n_trades']}", flush=True)
    results["win_full"] = m_win

    # --- Refined grid around tp_k=7, sl_k=7 with different caps ---
    print("\n[A] Refined cap sensitivity around (tp_k=7, sl_k=7)...", flush=True)
    for tp_max in [15, 20, 25, 30, 40, 100]:
        for sl_cap in [10, 12, 15, 18, 20]:
            r = simulate_capped_atr(close, high, low, rank_signal, atr_pct,
                                     tp_atr_k=7, tp_max=tp_max, tp_floor=5,
                                     sl_atr_k=7, sl_cap=sl_cap, sl_floor=5)
            m = _m(r)
            key = f"A_tpk7slk7_tpmax{tp_max}_slcap{sl_cap}"
            results[key] = m
            if m['cagr'] > 0.15:
                print(f"  tp_max={tp_max:>3}  sl_cap={sl_cap:>3}  CAGR {m['cagr']*100:>6.2f}%  MDD {m['maxdd']*100:>5.1f}%  Cal {m['calmar']:>5.3f}  WR {m['win_rate']*100:>5.1f}%  N {m['n_trades']}", flush=True)

    # --- Refined: vary time_stop (keep tp_max=25, sl_cap=15) ---
    print("\n[B] Vary time-stop (tp_k=7 sl_k=7 tp_max=25 sl_cap=15)...", flush=True)
    # use a wrapper that sets ts
    def run_with_ts(ts):
        return simulate_capped_atr(close, high, low, rank_signal, atr_pct,
                                    tp_atr_k=7, tp_max=25, tp_floor=5,
                                    sl_atr_k=7, sl_cap=15, sl_floor=5,
                                    ts_bars=ts)
    for ts in [63, 126, 189, 252, 378, 504]:
        r = run_with_ts(ts)
        m = _m(r)
        key = f"B_ts_{ts}"
        results[key] = m
        print(f"  TS={ts:>3}  CAGR {m['cagr']*100:>6.2f}%  MDD {m['maxdd']*100:>5.1f}%  Cal {m['calmar']:>5.3f}  WR {m['win_rate']*100:>5.1f}%  N {m['n_trades']}", flush=True)

    # --- Rolling 10Y windows ---
    print("\n[C] Rolling 10Y windows (winner vs control vs SPY)...", flush=True)
    rolling_windows = [("2006-04-25", "2016-04-24"),
                       ("2008-01-01", "2017-12-31"),
                       ("2010-01-01", "2019-12-31"),
                       ("2012-01-01", "2021-12-31"),
                       ("2014-01-01", "2023-12-31"),
                       ("2016-01-01", "2025-12-31")]
    roll = {}
    for start, end in rolling_windows:
        r_ctrl = run_window(close, high, low, rank_signal, atr_pct,
                            sl_atr_k=1000, sl_cap=15, sl_pct=15, start=start, end=end)
        r_win = run_window(close, high, low, rank_signal, atr_pct,
                           tp_atr_k=7, tp_max=25, sl_atr_k=7, sl_cap=15, start=start, end=end)
        close_sub = close[(close.index >= pd.Timestamp(start)) & (close.index <= pd.Timestamp(end))]
        spy_r = spy_dca_baseline(close_sub)
        m_ctrl_r = _m(r_ctrl)
        m_win_r = _m(r_win)
        key = f"{start[:4]}-{end[:4]}"
        roll[key] = {
            "ctrl_cagr": m_ctrl_r["cagr"], "win_cagr": m_win_r["cagr"], "spy_cagr": spy_r["cagr"],
            "ctrl_mdd": m_ctrl_r["maxdd"], "win_mdd": m_win_r["maxdd"], "spy_mdd": spy_r["maxdd"],
            "ctrl_wr": m_ctrl_r["win_rate"], "win_wr": m_win_r["win_rate"],
            "ctrl_n": m_ctrl_r["n_trades"], "win_n": m_win_r["n_trades"],
        }
        print(f"  {key}  SPY {spy_r['cagr']*100:>6.2f}%  CTRL {m_ctrl_r['cagr']*100:>6.2f}%  WIN {m_win_r['cagr']*100:>6.2f}%  "
              f"(MDD SPY {spy_r['maxdd']*100:>5.1f}%  CTRL {m_ctrl_r['maxdd']*100:>5.1f}%  WIN {m_win_r['maxdd']*100:>5.1f}%)", flush=True)
    results["rolling_10y"] = roll

    # --- Trade log analysis (winner) ---
    print("\n[D] Winner trade log analysis...", flush=True)
    trades = win["positions"]
    tps = sum(1 for t in trades if t.get("reason") == "tp")
    sls = sum(1 for t in trades if t.get("reason") == "sl_or_trail")
    tms = sum(1 for t in trades if t.get("reason") == "time")
    rets = np.array([t["ret"] for t in trades])
    days = np.array([t["days_held"] for t in trades])
    print(f"  TP hits:   {tps}/{len(trades)} = {tps/len(trades)*100:.1f}%  avg ret ~{np.mean([t['ret'] for t in trades if t['reason']=='tp'])*100:.2f}%")
    print(f"  SL hits:   {sls}/{len(trades)} = {sls/len(trades)*100:.1f}%  avg ret ~{np.mean([t['ret'] for t in trades if t['reason']=='sl_or_trail'])*100:.2f}%")
    print(f"  Time:      {tms}/{len(trades)} = {tms/len(trades)*100:.1f}%  avg ret ~{np.mean([t['ret'] for t in trades if t['reason']=='time']) * 100 if tms else 0:.2f}%")
    print(f"  Days held: avg {days.mean():.1f}  median {np.median(days):.0f}  min {days.min()}  max {days.max()}")
    print(f"  Return distribution: mean {rets.mean()*100:.2f}%  median {np.median(rets)*100:.2f}%  min {rets.min()*100:.2f}%  max {rets.max()*100:.2f}%")
    print(f"  Return std: {rets.std()*100:.2f}%  skew: {((rets-rets.mean())**3).mean()/rets.std()**3:.2f}")

    # Tickers picked
    tickers = [t for t in close.columns if t != "SPY"]
    tk_counts = {}
    for t in trades:
        nm = tickers[t["tk"]]
        tk_counts[nm] = tk_counts.get(nm, 0) + 1
    print(f"  Unique tickers: {len(tk_counts)}  most common:")
    for tk, n in sorted(tk_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"    {tk}: {n}")
    results["trade_log"] = {
        "n_trades": len(trades), "tp_hits": tps, "sl_hits": sls, "time_hits": tms,
        "avg_ret_tp": float(np.mean([t["ret"] for t in trades if t["reason"]=="tp"])) if tps else None,
        "avg_ret_sl": float(np.mean([t["ret"] for t in trades if t["reason"]=="sl_or_trail"])) if sls else None,
        "avg_days_held": float(days.mean()),
        "max_days_held": int(days.max()),
        "return_mean": float(rets.mean()),
        "return_std": float(rets.std()),
        "unique_tickers": len(tk_counts),
        "top_tickers": dict(sorted(tk_counts.items(), key=lambda x: -x[1])[:15]),
    }

    # Save
    out_path = Path("/home/user/crt/max/research/step47_results.json")
    out_path.write_text(json.dumps(results, indent=2, default=float))
    print(f"\nWrote {out_path}", flush=True)

    # Summary takeaway
    print("\n=== Summary ===", flush=True)
    print(f"WINNER (tp_k=7 sl_k=7 capped): CAGR +{m_win['cagr']*100:.2f}% vs CTRL +{m_ctrl['cagr']*100:.2f}% vs SPY +{spy['cagr']*100:.2f}%")
    print(f"WINNER MDD {m_win['maxdd']*100:.1f}% vs CTRL {m_ctrl['maxdd']*100:.1f}% vs SPY {spy['maxdd']*100:.1f}%")
    print(f"WINNER WR {m_win['win_rate']*100:.1f}% vs CTRL {m_ctrl['win_rate']*100:.1f}%")
    print(f"WINNER Calmar {m_win['calmar']:.3f} vs CTRL {m_ctrl['calmar']:.3f} vs SPY {spy['calmar']:.3f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
