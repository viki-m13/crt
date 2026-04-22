#!/usr/bin/env python3
"""Step 41 — Fixed-percentage take-profit grid search using CAP5+SMA12M ranking.

Strategy:
  - On the first trading day of each month (call it D), rank tickers by
    CAP5+SMA12M = trailing 252-bar mean of `final` (conviction score from
    bt_ext.parquet). SPY is excluded. Tickers need >=252 bars of price history
    AND >=252 bars of finite `final` history to be rankable.
  - Entry fill: pick-top-1; fill at the close of day D+1 (one-bar delay,
    matching production CAP5 conventions).
  - Exit: take-profit = entry_price * (1 + TP%). Fill at the TP price on the
    first subsequent day where High[t] >= TP target. If no TP hit within
    `time_stop_bars` trading days (measured from D+1), exit at close of the
    last bar in the window.
  - Cash sits idle between trades; $1000/mo contributions accumulate into cash
    when a trade is open and deploy at the next monthly entry.

Window: 2006-04-25 through 2026-04-22.

Grid:
  TP% in {2, 3, 5, 7, 10, 15, 20, 30}
  time_stop_bars in {10, 21, 42, 63, 126, 252, 504}

Outputs:
  - step41_results.json: full grid
  - Console: top-10 by CAGR and by win_rate*Sharpe (feeds the summary MD)
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

DCA_MONTHLY = 1000.0
TRADING_DAYS_YR = 252
WINDOW_START = pd.Timestamp("2006-04-25")
WINDOW_END = pd.Timestamp("2026-04-22")
RANK_LOOKBACK = 252  # CAP5+SMA12M => 12 months ~= 252 trading days

DATA_DIR = Path("/home/user/crt/max/research/data")
RAW = DATA_DIR / "raw"


def load_data():
    close = pd.read_parquet(RAW / "Close.parquet")
    high = pd.read_parquet(RAW / "High.parquet")
    bt = pd.read_parquet(DATA_DIR / "bt_ext.parquet")
    # Restrict to window
    close = close[(close.index >= WINDOW_START) & (close.index <= WINDOW_END)]
    high = high[(high.index >= WINDOW_START) & (high.index <= WINDOW_END)]
    bt = bt[(bt.date >= WINDOW_START) & (bt.date <= WINDOW_END)].copy()

    # Pivot final column to wide (date x ticker), align with close index
    finals = bt.pivot(index="date", columns="ticker", values="final")
    # Align indexes — use close's trading day index as master
    finals = finals.reindex(close.index)
    # Forward-fill finals within each ticker: `final` is updated ~weekly in
    # bt_ext, so between updates the last-known score carries (point-in-time
    # as of that update). This is how the production ranker behaves.
    finals = finals.ffill()
    return close, high, finals


def compute_rank_signal(finals: pd.DataFrame) -> pd.DataFrame:
    """CAP5+SMA12M — trailing 252-bar mean of `final` (per ticker)."""
    return finals.rolling(window=RANK_LOOKBACK, min_periods=RANK_LOOKBACK).mean()


def month_first_indices(idx: pd.DatetimeIndex) -> list[int]:
    out = []
    prev_ym = None
    for i, dd in enumerate(idx):
        ym = (dd.year, dd.month)
        if ym != prev_ym:
            out.append(i)
            prev_ym = ym
    return out


def simulate(
    close: pd.DataFrame,
    high: pd.DataFrame,
    rank_signal: pd.DataFrame,
    tp_pct: float,
    time_stop_bars: int,
) -> dict:
    dates = close.index
    n = len(dates)
    month_idx = month_first_indices(dates)
    tickers = [t for t in close.columns if t != "SPY"]

    # All arrays: rows = dates, cols = tickers
    close_arr = close[tickers].to_numpy()
    high_arr = high[tickers].to_numpy()
    rank_arr = rank_signal[tickers].to_numpy()
    n_tk = len(tickers)

    # Pre-compute price history validity: to rank a ticker on day d, it needs
    # at least RANK_LOOKBACK bars of non-nan close history ending at d.
    price_valid_count = (
        pd.DataFrame(close_arr).notna().rolling(RANK_LOOKBACK, min_periods=RANK_LOOKBACK).sum().to_numpy()
    )

    cash = 0.0
    equity = np.zeros(n)
    positions = []  # closed trades
    open_pos = None  # {tk_idx, entry_idx, entry_px, tp_px, shares, stop_idx}
    total_invested = 0.0
    pending_deposits = 0.0  # contributions accumulated while trade open

    tp_mult = 1.0 + tp_pct / 100.0

    for mi, di in enumerate(month_idx):
        # Monthly contribution
        monthly = DCA_MONTHLY
        pending_deposits += monthly
        total_invested += monthly

        entry_idx = di + 1  # one-bar delay: fill at D+1 close
        if entry_idx >= n:
            break

        # If no open position, try to open one. If one is open, just accrue.
        if open_pos is None:
            # Rank candidates by rank_arr at day di (close of day D, known
            # before entry at D+1)
            scores = rank_arr[di]  # shape (n_tk,)
            pv_row = close_arr[di]
            valid_hist = price_valid_count[di]
            best_tk = -1
            best_score = -np.inf
            for ti in range(n_tk):
                s = scores[ti]
                px = pv_row[ti]
                vh = valid_hist[ti]
                if not (np.isfinite(s) and np.isfinite(px) and px > 0 and vh >= RANK_LOOKBACK):
                    continue
                if s > best_score:
                    best_score = s
                    best_tk = ti
            if best_tk < 0:
                # Nothing rankable — cash holds, advance segment w/o open pos
                next_di = month_idx[mi + 1] if (mi + 1) < len(month_idx) else n
                cash, pending_deposits, open_pos = _run_segment(
                    equity, di, next_di, open_pos, close_arr, high_arr,
                    cash, pending_deposits, positions, tp_mult, n,
                )
                continue
            entry_px = close_arr[entry_idx, best_tk]
            if not (np.isfinite(entry_px) and entry_px > 0):
                next_di = month_idx[mi + 1] if (mi + 1) < len(month_idx) else n
                cash, pending_deposits, open_pos = _run_segment(
                    equity, di, next_di, open_pos, close_arr, high_arr,
                    cash, pending_deposits, positions, tp_mult, n,
                )
                continue

            deploy = cash + pending_deposits
            pending_deposits = 0.0
            cash = 0.0
            shares = deploy / entry_px
            tp_px = entry_px * tp_mult
            stop_idx = entry_idx + time_stop_bars
            open_pos = {
                "tk_idx": best_tk,
                "tk": tickers[best_tk],
                "entry_idx": entry_idx,
                "entry_px": entry_px,
                "tp_px": tp_px,
                "shares": shares,
                "stop_idx": stop_idx,
                "cost": deploy,
            }

        # Fill equity up to next month's start (or end of series)
        next_di = month_idx[mi + 1] if (mi + 1) < len(month_idx) else n
        cash, pending_deposits, open_pos = _run_segment(
            equity, di, next_di, open_pos, close_arr, high_arr,
            cash, pending_deposits, positions, tp_mult, n,
        )

    # After final month, keep stepping to end of series to liquidate/mark
    if month_idx:
        last_di = month_idx[-1]
        if open_pos is not None or last_di < n - 1:
            cash, pending_deposits, open_pos = _run_segment(
                equity, last_di, n, open_pos, close_arr, high_arr,
                cash, pending_deposits, positions, tp_mult, n,
                final_liquidate=True,
            )

    # Fill any remaining equity slots
    for i in range(n):
        if equity[i] == 0 and i > 0:
            equity[i] = equity[i - 1]

    return {
        "equity": equity,
        "positions": positions,
        "total_invested": total_invested,
        "open_pos": open_pos,
        "cash_final": cash,
        "dates": dates,
    }


def _run_segment(
    equity,
    start_i,
    end_i,
    open_pos,
    close_arr,
    high_arr,
    cash,
    pending_deposits,
    positions,
    tp_mult,
    n,
    final_liquidate=False,
):
    """Walk from start_i up to end_i-1, updating equity daily and handling TP/time-stop."""
    for d in range(start_i, end_i):
        if open_pos is not None:
            tk_idx = open_pos["tk_idx"]
            # Check TP first (intraday high). Only check on days AFTER entry
            # (we entered at entry_idx close — intrabar high already happened).
            if d > open_pos["entry_idx"]:
                hi = high_arr[d, tk_idx]
                tp_target = open_pos["tp_px"]
                if np.isfinite(hi) and hi >= tp_target:
                    proceeds = open_pos["shares"] * tp_target
                    cash += proceeds
                    positions.append({
                        "tk": open_pos["tk"],
                        "entry_idx": open_pos["entry_idx"],
                        "exit_idx": d,
                        "entry_px": open_pos["entry_px"],
                        "exit_px": tp_target,
                        "cost": open_pos["cost"],
                        "proceeds": proceeds,
                        "days_held": d - open_pos["entry_idx"],
                        "hit_tp": True,
                    })
                    open_pos = None
                elif d >= open_pos["stop_idx"]:
                    # Time stop: close at close price of this bar
                    px = close_arr[d, tk_idx]
                    if not (np.isfinite(px) and px > 0):
                        # fallback: walk back to last valid close
                        for back in range(d, open_pos["entry_idx"], -1):
                            p2 = close_arr[back, tk_idx]
                            if np.isfinite(p2) and p2 > 0:
                                px = p2
                                break
                        else:
                            px = open_pos["entry_px"]
                    proceeds = open_pos["shares"] * px
                    cash += proceeds
                    positions.append({
                        "tk": open_pos["tk"],
                        "entry_idx": open_pos["entry_idx"],
                        "exit_idx": d,
                        "entry_px": open_pos["entry_px"],
                        "exit_px": px,
                        "cost": open_pos["cost"],
                        "proceeds": proceeds,
                        "days_held": d - open_pos["entry_idx"],
                        "hit_tp": False,
                    })
                    open_pos = None

        # Mark equity
        eq_val = cash + pending_deposits
        if open_pos is not None and d >= open_pos["entry_idx"]:
            tk_idx = open_pos["tk_idx"]
            px = close_arr[d, tk_idx]
            if not np.isfinite(px):
                # carry-forward last valid
                for back in range(d, open_pos["entry_idx"] - 1, -1):
                    p2 = close_arr[back, tk_idx]
                    if np.isfinite(p2) and p2 > 0:
                        px = p2
                        break
            if np.isfinite(px):
                eq_val += open_pos["shares"] * px
            else:
                # Nothing valid; keep cash as equity (was deployed but price missing)
                eq_val += open_pos["cost"]
        elif open_pos is not None and d < open_pos["entry_idx"]:
            # Position created for entry at a later day — the funds are still
            # "in cash" from an accounting standpoint until entry fills.
            eq_val += open_pos["cost"]
        equity[d] = eq_val

    if final_liquidate and open_pos is not None:
        # Final mark-out at end of window — leave open but equity already
        # reflects mark-to-market. Treat as "still open" trade.
        pass

    return cash, pending_deposits, open_pos


def compute_metrics(result: dict) -> dict:
    equity = result["equity"]
    dates = result["dates"]
    total_invested = result["total_invested"]
    positions = result["positions"]

    if total_invested <= 0 or len(equity) == 0:
        return {}

    # Include any still-open position as "alive" in equity final (already marked)
    final = float(equity[-1])

    # Find the first non-zero equity index (strategy start)
    start_i = 0
    for i, v in enumerate(equity):
        if v > 0:
            start_i = i
            break

    eq_slice = equity[start_i:]
    dates_slice = dates[start_i:]
    yrs = len(eq_slice) / TRADING_DAYS_YR

    # CAGR from invested to final (DCA)
    cagr = (final / total_invested) ** (1 / yrs) - 1 if yrs > 0 and final > 0 else -1.0

    # Daily returns of equity (only after it's positive)
    # Avoid first-step jumps from deposit-on-day-1
    # Use log-like daily pct change
    ret = np.zeros(len(eq_slice))
    for i in range(1, len(eq_slice)):
        if eq_slice[i - 1] > 0:
            ret[i] = eq_slice[i] / eq_slice[i - 1] - 1
    # Sharpe (annualized) from daily returns - naive, no risk-free rate
    std = ret.std()
    avg = ret.mean()
    sharpe = (avg / std) * math.sqrt(TRADING_DAYS_YR) if std > 0 else 0.0

    # MaxDD
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

    # Trade stats
    n_trades = len(positions)
    wins = sum(1 for p in positions if p["hit_tp"])
    win_rate = wins / n_trades if n_trades > 0 else 0.0
    trade_rets = [p["proceeds"] / p["cost"] - 1 for p in positions if p["cost"] > 0]
    avg_ret = float(np.mean(trade_rets)) if trade_rets else 0.0
    avg_days = float(np.mean([p["days_held"] for p in positions])) if positions else 0.0

    return {
        "cagr": cagr,
        "maxdd": maxdd,
        "sharpe": sharpe,
        "calmar": calmar,
        "win_rate": win_rate,
        "avg_trade_ret": avg_ret,
        "avg_days_held": avg_days,
        "n_trades": n_trades,
        "final_equity": final,
        "total_invested": total_invested,
    }


def spy_dca_baseline(close: pd.DataFrame) -> dict:
    """Monthly DCA into SPY for comparison."""
    dates = close.index
    month_idx = month_first_indices(dates)
    # Forward-fill SPY to handle missing final-day prices
    spy = close["SPY"].ffill().to_numpy()
    n = len(dates)
    equity = np.zeros(n)
    positions = []
    total_invested = 0.0
    for mi, di in enumerate(month_idx):
        entry_idx = di + 1
        if entry_idx >= n:
            break
        px = spy[entry_idx]
        if not (np.isfinite(px) and px > 0):
            continue
        total_invested += DCA_MONTHLY
        positions.append({"entry_idx": entry_idx, "shares": DCA_MONTHLY / px})
    for d in range(n):
        val = 0.0
        for p in positions:
            if d >= p["entry_idx"]:
                px = spy[d]
                if np.isfinite(px):
                    val += p["shares"] * px
        equity[d] = val
    result = {"equity": equity, "dates": dates, "total_invested": total_invested, "positions": []}
    yrs = n / TRADING_DAYS_YR
    final = equity[-1]
    cagr = (final / total_invested) ** (1 / yrs) - 1 if total_invested > 0 and yrs > 0 else 0.0
    peak, maxdd = 0.0, 0.0
    for v in equity:
        if v > peak:
            peak = v
        if peak > 0:
            dd = (peak - v) / peak
            if dd > maxdd:
                maxdd = dd
    ret = np.zeros(n)
    for i in range(1, n):
        if equity[i - 1] > 0:
            ret[i] = equity[i] / equity[i - 1] - 1
    std = ret.std()
    avg = ret.mean()
    sharpe = (avg / std) * math.sqrt(TRADING_DAYS_YR) if std > 0 else 0.0
    return {
        "cagr": cagr, "maxdd": maxdd, "sharpe": sharpe,
        "final_equity": float(final), "total_invested": total_invested,
    }


def main():
    print("Loading data...")
    close, high, finals = load_data()
    print(f"  close: {close.shape}, window {close.index[0].date()} .. {close.index[-1].date()}")
    print("Computing CAP5+SMA12M ranking signal (trailing 252-bar mean of final)...")
    rank_sig = compute_rank_signal(finals)

    print("Running SPY DCA baseline...")
    spy_base = spy_dca_baseline(close)
    print(f"  SPY DCA CAGR: {spy_base['cagr']*100:.2f}%  MDD: {spy_base['maxdd']*100:.2f}%  Sharpe: {spy_base['sharpe']:.2f}")

    tps = [2, 3, 5, 7, 10, 15, 20, 30]
    stops = [10, 21, 42, 63, 126, 252, 504]

    all_results = []
    for tp in tps:
        for ts in stops:
            print(f"Running TP={tp}% time_stop={ts} bars ...", end=" ", flush=True)
            res = simulate(close, high, rank_sig, tp, ts)
            m = compute_metrics(res)
            m["tp_pct"] = tp
            m["time_stop_bars"] = ts
            all_results.append(m)
            print(f"CAGR {m['cagr']*100:+.2f}%  WR {m['win_rate']*100:.1f}%  N {m['n_trades']}  AvgD {m['avg_days_held']:.0f}")

    out = {
        "spy_dca_baseline": spy_base,
        "grid": all_results,
        "methodology": {
            "rank": "CAP5+SMA12M = trailing 252-bar mean of `final` from bt_ext.parquet",
            "entry": "close of D+1 (one-bar delay after monthly rank date D)",
            "exit": "first day after entry where high >= entry*(1+tp); else close of bar at entry+time_stop",
            "cash_rule": "cash idle between trades; $1000/mo contribution pools to cash when trade open",
            "universe": "128 tickers minus SPY; tickers need >=252 bars history to be rankable",
            "window": f"{WINDOW_START.date()} to {WINDOW_END.date()}",
        },
    }

    out_path = Path("/home/user/crt/max/research/step41_results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nWrote {out_path}")

    # Print top-10 by CAGR
    srted = sorted(all_results, key=lambda r: r["cagr"], reverse=True)
    print("\nTop 10 by CAGR:")
    print(f"{'TP':>5} {'TS':>5} {'CAGR':>8} {'MDD':>8} {'Sharpe':>7} {'WR':>7} {'AvgRet':>8} {'AvgD':>6} {'N':>5}")
    for r in srted[:10]:
        print(f"{r['tp_pct']:>5} {r['time_stop_bars']:>5} {r['cagr']*100:>7.2f}% {r['maxdd']*100:>7.2f}% {r['sharpe']:>7.2f} {r['win_rate']*100:>6.1f}% {r['avg_trade_ret']*100:>+7.2f}% {r['avg_days_held']:>6.0f} {r['n_trades']:>5}")

    # Top by win_rate * sharpe (balanced)
    bal = sorted(all_results, key=lambda r: r["win_rate"] * r["sharpe"], reverse=True)
    print("\nTop 10 by win_rate * Sharpe:")
    print(f"{'TP':>5} {'TS':>5} {'CAGR':>8} {'MDD':>8} {'Sharpe':>7} {'WR':>7} {'AvgRet':>8} {'AvgD':>6} {'N':>5}")
    for r in bal[:10]:
        print(f"{r['tp_pct']:>5} {r['time_stop_bars']:>5} {r['cagr']*100:>7.2f}% {r['maxdd']*100:>7.2f}% {r['sharpe']:>7.2f} {r['win_rate']*100:>6.1f}% {r['avg_trade_ret']*100:>+7.2f}% {r['avg_days_held']:>6.0f} {r['n_trades']:>5}")


if __name__ == "__main__":
    main()
