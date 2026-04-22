#!/usr/bin/env python3
"""Step 43 — Multi-slot take-profit: concurrent positions, regime gates, stops.

step41 found TP=10% × 252d produces +11.52% CAGR with 87.9% win rate, but
with 78.8% MDD because it's single-pick — during 2008-09 and 2022 the one
open position stays underwater for a year. This script tests the same
winner config with extensions that target MDD reduction without killing
CAGR or win-rate:

  1. Multi-slot: open a new position EVERY month in parallel; allow up to
     N concurrent positions. Each slot runs independently TP/time-stop.
     Capital split: $1000/month fills into ONE new slot (rotating or
     top-ranked-not-already-held).
  2. Regime gate (SPY > 200dma): skip new entries when SPY below its
     200-day SMA. Cash accumulates and deploys on the next in-regime
     month. Variant: partial gate (reduce position size when off-regime).
  3. Hard stop-loss: exit at entry × (1 - L) if Low[t] <= stop; sweep L.
  4. Trailing stop: track high-water-mark, exit if Low[t] <= HWM × (1 - T).

Base config is step41's winner: CAP5+SMA12M top-1 per month, TP=10%,
time_stop=252 bars, fill at D+1 close, $1000/mo.

Window: 2006-04-25 → 2026-04-22 (matching step41).
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DCA_MONTHLY = 1000.0
TRADING_DAYS_YR = 252
WINDOW_START = pd.Timestamp("2006-04-25")
WINDOW_END = pd.Timestamp("2026-04-22")
RANK_LOOKBACK = 252

DATA_DIR = Path("/home/user/crt/max/research/data")
RAW = DATA_DIR / "raw"

# Default TP config (step41 winner)
TP_PCT = 10.0
TIME_STOP = 252


def load_data():
    close = pd.read_parquet(RAW / "Close.parquet")
    high = pd.read_parquet(RAW / "High.parquet")
    low = pd.read_parquet(RAW / "Low.parquet")
    bt = pd.read_parquet(DATA_DIR / "bt_ext.parquet")
    close = close[(close.index >= WINDOW_START) & (close.index <= WINDOW_END)]
    high = high[(high.index >= WINDOW_START) & (high.index <= WINDOW_END)]
    low = low[(low.index >= WINDOW_START) & (low.index <= WINDOW_END)]
    bt = bt[(bt.date >= WINDOW_START) & (bt.date <= WINDOW_END)].copy()
    finals = bt.pivot(index="date", columns="ticker", values="final")
    finals = finals.reindex(close.index).ffill()
    return close, high, low, finals


def month_first_indices(idx: pd.DatetimeIndex) -> list[int]:
    out = []
    prev_ym = None
    for i, dd in enumerate(idx):
        ym = (dd.year, dd.month)
        if ym != prev_ym:
            out.append(i)
            prev_ym = ym
    return out


def compute_rank_signal(finals: pd.DataFrame) -> pd.DataFrame:
    return finals.rolling(window=RANK_LOOKBACK, min_periods=RANK_LOOKBACK).mean()


def spy_regime_200dma(close: pd.DataFrame) -> np.ndarray:
    """Returns boolean array: True on days where SPY close > 200-day SMA."""
    spy = close["SPY"].ffill()
    sma = spy.rolling(200, min_periods=200).mean()
    above = (spy > sma).to_numpy()
    return above


def simulate_multislot(
    close: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    rank_signal: pd.DataFrame,
    tp_pct: float = TP_PCT,
    time_stop_bars: int = TIME_STOP,
    n_slots: int = 1,                      # max concurrent positions
    stop_loss_pct: float | None = None,   # hard stop in pct (e.g. 15 for -15%)
    trail_stop_pct: float | None = None,  # trail stop in pct (e.g. 12)
    regime_gate: str | None = None,       # 'spy200', None
    regime_downsize: float = 0.0,         # off-regime position size multiplier (0 = skip)
) -> dict:
    dates = close.index
    n = len(dates)
    month_idx = month_first_indices(dates)
    tickers = [t for t in close.columns if t != "SPY"]
    close_arr = close[tickers].to_numpy()
    high_arr = high[tickers].to_numpy()
    low_arr = low[tickers].to_numpy()
    rank_arr = rank_signal[tickers].to_numpy()
    n_tk = len(tickers)

    price_valid = (
        pd.DataFrame(close_arr).notna().rolling(RANK_LOOKBACK, min_periods=RANK_LOOKBACK).sum().to_numpy()
    )

    spy_above = spy_regime_200dma(close) if regime_gate == "spy200" else None

    cash = 0.0
    equity = np.zeros(n)
    positions = []          # closed trades
    open_positions = []     # list of dicts: tk_idx, entry_idx, entry_px, tp_px, sl_px, shares, stop_idx, hwm, tk
    total_invested = 0.0

    tp_mult = 1.0 + tp_pct / 100.0
    sl_mult = 1.0 - (stop_loss_pct / 100.0) if stop_loss_pct is not None else None
    trail_mult = 1.0 - (trail_stop_pct / 100.0) if trail_stop_pct is not None else None

    for mi, di in enumerate(month_idx):
        total_invested += DCA_MONTHLY
        cash += DCA_MONTHLY
        entry_idx = di + 1
        if entry_idx >= n:
            break

        # Regime check on entry_idx
        in_regime = True
        if regime_gate == "spy200" and spy_above is not None:
            in_regime = bool(spy_above[di])  # check on signal day (D)
        size_mult = 1.0 if in_regime else regime_downsize

        if len(open_positions) < n_slots and size_mult > 0:
            # Need to open new position: rank excluding already-held tickers
            held = {p["tk_idx"] for p in open_positions}
            scores = rank_arr[di]
            pv_row = close_arr[di]
            valid_hist = price_valid[di]
            best_tk, best_score = -1, -np.inf
            for ti in range(n_tk):
                if ti in held:
                    continue
                s, px, vh = scores[ti], pv_row[ti], valid_hist[ti]
                if not (np.isfinite(s) and np.isfinite(px) and px > 0 and vh >= RANK_LOOKBACK):
                    continue
                if s > best_score:
                    best_score = s
                    best_tk = ti
            if best_tk >= 0:
                entry_px = close_arr[entry_idx, best_tk]
                if np.isfinite(entry_px) and entry_px > 0:
                    # Size: deploy cash / remaining slot count (or full DCA if n_slots==1)
                    remaining_slots = n_slots - len(open_positions)
                    deploy = (cash / remaining_slots) * size_mult
                    cash -= deploy
                    shares = deploy / entry_px
                    sl_px = entry_px * sl_mult if sl_mult is not None else None
                    open_positions.append({
                        "tk_idx": best_tk,
                        "tk": tickers[best_tk],
                        "entry_idx": entry_idx,
                        "entry_px": entry_px,
                        "tp_px": entry_px * tp_mult,
                        "sl_px": sl_px,
                        "shares": shares,
                        "stop_idx": entry_idx + time_stop_bars,
                        "hwm": entry_px,
                        "cost": deploy,
                    })

        next_di = month_idx[mi + 1] if (mi + 1) < len(month_idx) else n
        cash = _run_segment(
            equity, di, next_di, open_positions, close_arr, high_arr, low_arr,
            cash, positions, trail_mult, n,
        )

    if month_idx:
        cash = _run_segment(
            equity, month_idx[-1], n, open_positions, close_arr, high_arr, low_arr,
            cash, positions, trail_mult, n, final=True,
        )
    for i in range(n):
        if equity[i] == 0 and i > 0:
            equity[i] = equity[i - 1]

    return {
        "equity": equity,
        "positions": positions,
        "open_positions": open_positions,
        "total_invested": total_invested,
        "dates": dates,
    }


def _run_segment(equity, start_i, end_i, open_positions, close_arr, high_arr, low_arr,
                 cash, positions, trail_mult, n, final=False):
    for d in range(start_i, end_i):
        # Process each open position: check SL -> TP -> trail -> time stop
        still_open = []
        for pos in open_positions:
            tk = pos["tk_idx"]
            if d <= pos["entry_idx"]:
                still_open.append(pos)
                continue
            exited = False
            lo = low_arr[d, tk]
            hi = high_arr[d, tk]

            # Stop loss (check first: worst case)
            if pos["sl_px"] is not None and np.isfinite(lo) and lo <= pos["sl_px"]:
                exit_px = pos["sl_px"]  # conservative: assume fill at stop
                cash += pos["shares"] * exit_px
                positions.append({**_close_trade(pos, d, exit_px, reason="sl")})
                exited = True

            # Trailing stop
            if not exited and trail_mult is not None:
                # Update HWM using intraday high (conservative: HWM set before trail trigger this bar)
                prev_hwm = pos["hwm"]
                if np.isfinite(hi) and hi > prev_hwm:
                    pos["hwm"] = hi
                trail_px = pos["hwm"] * trail_mult
                if np.isfinite(lo) and lo <= trail_px:
                    # Use min(open, trail_px) - simplest is trail_px
                    cash += pos["shares"] * trail_px
                    positions.append({**_close_trade(pos, d, trail_px, reason="trail")})
                    exited = True

            # Take profit
            if not exited and np.isfinite(hi) and hi >= pos["tp_px"]:
                cash += pos["shares"] * pos["tp_px"]
                positions.append({**_close_trade(pos, d, pos["tp_px"], reason="tp")})
                exited = True

            # Time stop
            if not exited and d >= pos["stop_idx"]:
                px = close_arr[d, tk]
                if not (np.isfinite(px) and px > 0):
                    for back in range(d, pos["entry_idx"], -1):
                        p2 = close_arr[back, tk]
                        if np.isfinite(p2) and p2 > 0:
                            px = p2
                            break
                    else:
                        px = pos["entry_px"]
                cash += pos["shares"] * px
                positions.append({**_close_trade(pos, d, px, reason="time")})
                exited = True

            if not exited:
                still_open.append(pos)
        open_positions[:] = still_open

        # Mark-to-market equity
        eq = cash
        for pos in open_positions:
            tk = pos["tk_idx"]
            if d < pos["entry_idx"]:
                eq += pos["cost"]
            else:
                px = close_arr[d, tk]
                if not np.isfinite(px):
                    for back in range(d, pos["entry_idx"] - 1, -1):
                        p2 = close_arr[back, tk]
                        if np.isfinite(p2) and p2 > 0:
                            px = p2
                            break
                eq += pos["shares"] * px if np.isfinite(px) else pos["cost"]
        equity[d] = eq
    return cash


def _close_trade(pos, exit_idx, exit_px, reason):
    return {
        "tk": pos["tk"],
        "entry_idx": pos["entry_idx"],
        "exit_idx": exit_idx,
        "entry_px": pos["entry_px"],
        "exit_px": exit_px,
        "cost": pos["cost"],
        "proceeds": pos["shares"] * exit_px,
        "days_held": exit_idx - pos["entry_idx"],
        "hit_tp": reason == "tp",
        "exit_reason": reason,
    }


def compute_metrics(result: dict) -> dict:
    equity = result["equity"]
    total_invested = result["total_invested"]
    positions = result["positions"]
    if total_invested <= 0 or len(equity) == 0:
        return {}
    final = float(equity[-1])
    start_i = next((i for i, v in enumerate(equity) if v > 0), 0)
    eq_slice = equity[start_i:]
    yrs = len(eq_slice) / TRADING_DAYS_YR
    cagr = (final / total_invested) ** (1 / yrs) - 1 if yrs > 0 and final > 0 else -1.0

    ret = np.zeros(len(eq_slice))
    for i in range(1, len(eq_slice)):
        if eq_slice[i - 1] > 0:
            ret[i] = eq_slice[i] / eq_slice[i - 1] - 1
    std = ret.std()
    avg = ret.mean()
    sharpe = (avg / std) * math.sqrt(TRADING_DAYS_YR) if std > 0 else 0.0

    peak = 0.0
    maxdd = 0.0
    for v in eq_slice:
        if v > peak:
            peak = v
        if peak > 0:
            dd = (peak - v) / peak
            if dd > maxdd:
                maxdd = dd
    calmar = cagr / maxdd if maxdd > 0 else 0.0

    n_trades = len(positions)
    tp_hits = sum(1 for p in positions if p["exit_reason"] == "tp")
    sl_hits = sum(1 for p in positions if p["exit_reason"] == "sl")
    trail_hits = sum(1 for p in positions if p["exit_reason"] == "trail")
    time_hits = sum(1 for p in positions if p["exit_reason"] == "time")
    # "Win rate" = hit_tp. "Gross win rate" = final trade return > 0
    win_rate = tp_hits / n_trades if n_trades > 0 else 0.0
    trade_rets = [p["proceeds"] / p["cost"] - 1 for p in positions if p["cost"] > 0]
    gross_win_rate = sum(1 for r in trade_rets if r > 0) / n_trades if n_trades > 0 else 0.0
    avg_ret = float(np.mean(trade_rets)) if trade_rets else 0.0
    avg_days = float(np.mean([p["days_held"] for p in positions])) if positions else 0.0
    avg_winner = float(np.mean([r for r in trade_rets if r > 0])) if trade_rets and any(r > 0 for r in trade_rets) else 0.0
    avg_loser = float(np.mean([r for r in trade_rets if r < 0])) if trade_rets and any(r < 0 for r in trade_rets) else 0.0
    return {
        "cagr": cagr, "maxdd": maxdd, "sharpe": sharpe, "calmar": calmar,
        "win_rate": win_rate, "gross_win_rate": gross_win_rate,
        "avg_trade_ret": avg_ret, "avg_days_held": avg_days, "n_trades": n_trades,
        "tp_hits": tp_hits, "sl_hits": sl_hits, "trail_hits": trail_hits, "time_hits": time_hits,
        "avg_winner": avg_winner, "avg_loser": avg_loser,
        "final_equity": final, "total_invested": total_invested,
    }


def spy_dca_baseline(close: pd.DataFrame) -> dict:
    dates = close.index
    month_idx = month_first_indices(dates)
    spy = close["SPY"].ffill().to_numpy()
    n = len(dates)
    equity = np.zeros(n)
    shares_owned = 0.0
    total_invested = 0.0
    next_mi = 0
    for d in range(n):
        while next_mi < len(month_idx) and month_idx[next_mi] + 1 == d:
            px = spy[d]
            if np.isfinite(px) and px > 0:
                shares_owned += DCA_MONTHLY / px
                total_invested += DCA_MONTHLY
            next_mi += 1
        if np.isfinite(spy[d]) and spy[d] > 0:
            equity[d] = shares_owned * spy[d]
        else:
            equity[d] = equity[d - 1] if d > 0 else 0.0
    return compute_metrics({"equity": equity, "total_invested": total_invested, "positions": [], "dates": dates})


def main():
    print("Loading data...", flush=True)
    close, high, low, finals = load_data()
    rank_signal = compute_rank_signal(finals)
    print(f"  shape: {close.shape}  date range: {close.index[0].date()} -> {close.index[-1].date()}", flush=True)

    results = {}

    # --- SPY DCA baseline ---
    print("\n[1] SPY DCA baseline...", flush=True)
    m = spy_dca_baseline(close)
    results["spy_dca"] = m
    print(f"   CAGR {m['cagr']*100:.2f}%  MDD {m['maxdd']*100:.1f}%  Sharpe {m['sharpe']:.2f}", flush=True)

    # --- Step41 winner re-run (control: single-slot, 10%/252) ---
    print("\n[2] step41 winner re-run (control, 1 slot, 10%/252d)...", flush=True)
    r = simulate_multislot(close, high, low, rank_signal, n_slots=1)
    m = compute_metrics(r)
    results["ctrl_1slot"] = m
    print(f"   CAGR {m['cagr']*100:.2f}%  MDD {m['maxdd']*100:.1f}%  WR {m['win_rate']*100:.1f}%  N {m['n_trades']}", flush=True)

    # --- Multi-slot sweep ---
    print("\n[3] Multi-slot sweep...", flush=True)
    for n_slots in [2, 3, 5, 10]:
        r = simulate_multislot(close, high, low, rank_signal, n_slots=n_slots)
        m = compute_metrics(r)
        results[f"slots_{n_slots}"] = m
        print(f"   {n_slots}-slot: CAGR {m['cagr']*100:.2f}%  MDD {m['maxdd']*100:.1f}%  Sharpe {m['sharpe']:.2f}  WR {m['win_rate']*100:.1f}%  Calmar {m['calmar']:.2f}  N {m['n_trades']}", flush=True)

    # --- Regime gate (SPY > 200dma) ---
    print("\n[4] Regime gate SPY > 200dma...", flush=True)
    for ns in [1, 3, 5]:
        r = simulate_multislot(close, high, low, rank_signal, n_slots=ns, regime_gate="spy200")
        m = compute_metrics(r)
        results[f"spy200_slots{ns}"] = m
        print(f"   {ns}-slot + SPY200: CAGR {m['cagr']*100:.2f}%  MDD {m['maxdd']*100:.1f}%  Sharpe {m['sharpe']:.2f}  WR {m['win_rate']*100:.1f}%  N {m['n_trades']}", flush=True)

    # --- Stop-loss variants ---
    print("\n[5] Hard stop-loss (3-slot base)...", flush=True)
    for sl in [10, 15, 20, 25]:
        r = simulate_multislot(close, high, low, rank_signal, n_slots=3, stop_loss_pct=sl)
        m = compute_metrics(r)
        results[f"sl{sl}_slots3"] = m
        print(f"   SL{sl}%: CAGR {m['cagr']*100:.2f}%  MDD {m['maxdd']*100:.1f}%  Sharpe {m['sharpe']:.2f}  WR {m['win_rate']*100:.1f}%  SL-hits {m['sl_hits']}", flush=True)

    # --- Trailing stop variants ---
    print("\n[6] Trailing stop (3-slot base)...", flush=True)
    for ts in [10, 15, 20, 25]:
        r = simulate_multislot(close, high, low, rank_signal, n_slots=3, trail_stop_pct=ts)
        m = compute_metrics(r)
        results[f"trail{ts}_slots3"] = m
        print(f"   trail{ts}%: CAGR {m['cagr']*100:.2f}%  MDD {m['maxdd']*100:.1f}%  Sharpe {m['sharpe']:.2f}  WR {m['win_rate']*100:.1f}%  trail-hits {m['trail_hits']}", flush=True)

    # --- Combo: 3-slot + SPY200 + SL 15% ---
    print("\n[7] Best combo candidates (3-slot + SPY200 + stop)...", flush=True)
    r = simulate_multislot(close, high, low, rank_signal, n_slots=3, regime_gate="spy200", stop_loss_pct=15)
    m = compute_metrics(r)
    results["combo_3slot_spy200_sl15"] = m
    print(f"   3-slot + SPY200 + SL15: CAGR {m['cagr']*100:.2f}%  MDD {m['maxdd']*100:.1f}%  Sharpe {m['sharpe']:.2f}  WR {m['win_rate']*100:.1f}%  Calmar {m['calmar']:.2f}", flush=True)

    r = simulate_multislot(close, high, low, rank_signal, n_slots=5, regime_gate="spy200", stop_loss_pct=20)
    m = compute_metrics(r)
    results["combo_5slot_spy200_sl20"] = m
    print(f"   5-slot + SPY200 + SL20: CAGR {m['cagr']*100:.2f}%  MDD {m['maxdd']*100:.1f}%  Sharpe {m['sharpe']:.2f}  WR {m['win_rate']*100:.1f}%  Calmar {m['calmar']:.2f}", flush=True)

    r = simulate_multislot(close, high, low, rank_signal, n_slots=3, trail_stop_pct=15)
    m = compute_metrics(r)
    results["combo_3slot_trail15"] = m
    print(f"   3-slot + trail15: CAGR {m['cagr']*100:.2f}%  MDD {m['maxdd']*100:.1f}%  Sharpe {m['sharpe']:.2f}  WR {m['win_rate']*100:.1f}%  Calmar {m['calmar']:.2f}", flush=True)

    out_path = Path("/home/user/crt/max/research/step43_results.json")
    out_path.write_text(json.dumps(results, indent=2, default=float))
    print(f"\nWrote {out_path}", flush=True)

    # Sorted summary
    print("\n=== Sorted by Calmar ===", flush=True)
    items = [(k, v) for k, v in results.items() if v and v.get("cagr", 0) > 0]
    items.sort(key=lambda kv: kv[1].get("calmar", 0), reverse=True)
    for k, v in items[:10]:
        print(f"  {k:<30} CAGR {v['cagr']*100:>6.2f}% MDD {v['maxdd']*100:>5.1f}% Shp {v['sharpe']:>4.2f} Calmar {v['calmar']:>5.3f} WR {v.get('win_rate',0)*100:>5.1f}%  N {v.get('n_trades','-')}", flush=True)

    print("\n=== Sorted by CAGR ===", flush=True)
    items = [(k, v) for k, v in results.items() if v and v.get("cagr", 0) > 0]
    items.sort(key=lambda kv: kv[1].get("cagr", 0), reverse=True)
    for k, v in items[:10]:
        print(f"  {k:<30} CAGR {v['cagr']*100:>6.2f}% MDD {v['maxdd']*100:>5.1f}% Shp {v['sharpe']:>4.2f} Calmar {v['calmar']:>5.3f} WR {v.get('win_rate',0)*100:>5.1f}%  N {v.get('n_trades','-')}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
