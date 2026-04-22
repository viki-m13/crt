#!/usr/bin/env python3
"""Step 44 — Apply stop-loss / trailing stop to step41's single-slot winner.

Goal: preserve the +11.52% CAGR and 87.9% win rate of TP=10%/252d
(step41 winner) while reducing 78.8% MDD.

Also sweeps a refined grid around the winner (TP ∈ {7, 8, 10, 12, 15}%
× time_stop ∈ {63, 126, 189, 252, 378, 504}) to check sensitivity.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from step43_tp_multislot import (  # reuse framework
    load_data, compute_rank_signal, simulate_multislot,
    compute_metrics, spy_dca_baseline,
)


def main():
    print("Loading data...", flush=True)
    close, high, low, finals = load_data()
    rank_signal = compute_rank_signal(finals)

    results = {}
    spy = spy_dca_baseline(close)
    results["spy_dca"] = spy
    print(f"SPY DCA: CAGR {spy['cagr']*100:.2f}%  MDD {spy['maxdd']*100:.1f}%  Sharpe {spy['sharpe']:.2f}", flush=True)

    # --- Single-slot winner + stop-loss ---
    print("\n[A] Single-slot TP10/252d + stop-loss sweep...", flush=True)
    for sl in [15, 20, 25, 30, 40, 50]:
        r = simulate_multislot(close, high, low, rank_signal, tp_pct=10.0, time_stop_bars=252, n_slots=1, stop_loss_pct=sl)
        m = compute_metrics(r)
        results[f"slot1_tp10_ts252_sl{sl}"] = m
        print(f"  SL{sl}%: CAGR {m['cagr']*100:>6.2f}%  MDD {m['maxdd']*100:>5.1f}%  Shp {m['sharpe']:>4.2f}  Calmar {m['calmar']:>5.3f}  WR {m['win_rate']*100:>5.1f}%  SL-hits {m['sl_hits']}  N {m['n_trades']}", flush=True)

    # --- Single-slot + trailing stop ---
    print("\n[B] Single-slot TP10/252d + trailing stop sweep...", flush=True)
    for ts in [15, 20, 25, 30, 40, 50]:
        r = simulate_multislot(close, high, low, rank_signal, tp_pct=10.0, time_stop_bars=252, n_slots=1, trail_stop_pct=ts)
        m = compute_metrics(r)
        results[f"slot1_tp10_ts252_trail{ts}"] = m
        print(f"  trail{ts}%: CAGR {m['cagr']*100:>6.2f}%  MDD {m['maxdd']*100:>5.1f}%  Shp {m['sharpe']:>4.2f}  Calmar {m['calmar']:>5.3f}  WR {m['win_rate']*100:>5.1f}%  trail-hits {m['trail_hits']}", flush=True)

    # --- Refined TP × time_stop grid (single-slot, no stop) ---
    print("\n[C] Refined TP × time_stop grid (single slot, no stop)...", flush=True)
    for tp in [7, 8, 10, 12, 15]:
        for ts in [63, 126, 189, 252, 378, 504]:
            r = simulate_multislot(close, high, low, rank_signal, tp_pct=float(tp), time_stop_bars=ts, n_slots=1)
            m = compute_metrics(r)
            key = f"slot1_tp{tp}_ts{ts}"
            results[key] = m
            print(f"  TP{tp:>2}/TS{ts:>3}: CAGR {m['cagr']*100:>6.2f}%  MDD {m['maxdd']*100:>5.1f}%  Calmar {m['calmar']:>5.3f}  WR {m['win_rate']*100:>5.1f}%  N {m['n_trades']}", flush=True)

    # --- Single slot + SL and time-stop scaled together ---
    print("\n[D] Combined: SL25 + shorter time stops...", flush=True)
    for ts in [63, 126, 189, 252]:
        r = simulate_multislot(close, high, low, rank_signal, tp_pct=10.0, time_stop_bars=ts, n_slots=1, stop_loss_pct=25)
        m = compute_metrics(r)
        key = f"slot1_tp10_ts{ts}_sl25"
        results[key] = m
        print(f"  TS{ts:>3}+SL25: CAGR {m['cagr']*100:>6.2f}%  MDD {m['maxdd']*100:>5.1f}%  Calmar {m['calmar']:>5.3f}  WR {m['win_rate']*100:>5.1f}%  SL-hits {m['sl_hits']}", flush=True)

    # Sorted output
    print("\n=== Sorted by (Calmar, WR) — aiming for high WR AND low MDD ===", flush=True)
    items = [(k, v) for k, v in results.items() if v and v.get("cagr", 0) > spy["cagr"]]
    items.sort(key=lambda kv: kv[1].get("calmar", 0), reverse=True)
    for k, v in items[:15]:
        print(f"  {k:<32} CAGR {v['cagr']*100:>6.2f}% MDD {v['maxdd']*100:>5.1f}% Shp {v['sharpe']:>4.2f} Calmar {v['calmar']:>5.3f} WR {v.get('win_rate',0)*100:>5.1f}% N {v.get('n_trades','-')}", flush=True)

    print("\n=== Sorted by CAGR (beats SPY) ===", flush=True)
    items.sort(key=lambda kv: kv[1].get("cagr", 0), reverse=True)
    for k, v in items[:15]:
        print(f"  {k:<32} CAGR {v['cagr']*100:>6.2f}% MDD {v['maxdd']*100:>5.1f}% Shp {v['sharpe']:>4.2f} Calmar {v['calmar']:>5.3f} WR {v.get('win_rate',0)*100:>5.1f}% N {v.get('n_trades','-')}", flush=True)

    out_path = Path("/home/user/crt/max/research/step44_results.json")
    out_path.write_text(json.dumps(results, indent=2, default=float))
    print(f"\nWrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
