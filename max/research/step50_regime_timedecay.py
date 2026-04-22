#!/usr/bin/env python3
"""Step 50 — Time-decay and regime-adaptive exits on step41 winner.

Base config (control):
  CAP5+SMA12M monthly top-1, single-slot, TP=10%, SL=-15%, time_stop=252d,
  $1000/mo DCA, fill at D+1 close, cash idle between trades.

Variants:
  A) TP time-decay: TP shrinks as the trade ages. Several decay schedules.
  B) Regime-aware time-stop (SPY 200dma at entry): bull=126d, bear=378d.
     Regime evaluated at signal-day D (no look-ahead).
  C) Regime-aware TP: SPY>200dma => TP=10%, else TP=15% (and reverse).
  D) Volatility-regime time-stop (SPY 21d realized vol annualized, computed
     at day D): vol>25% => bear TS=378d; else TS=126d.
  E) Combined: best of A/B/C/D.

Exits:
  - SL fixed at -15% (intraday low <= entry*(1-0.15)) -- checked first.
  - TP: intraday high >= current_tp (for A, current_tp changes with age).
  - Time-stop: close of bar at entry+stop_bars.
  - Priority per bar: SL -> TP -> time-stop. (SL-first is conservative.)
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


def spy_above_200dma(close: pd.DataFrame) -> np.ndarray:
    spy = close["SPY"].ffill()
    sma = spy.rolling(200, min_periods=200).mean()
    return (spy > sma).to_numpy()


def spy_realized_vol(close: pd.DataFrame, window: int = 21) -> np.ndarray:
    """21-day annualized realized vol of SPY log returns (as of end of each bar)."""
    spy = close["SPY"].ffill()
    logret = np.log(spy / spy.shift(1))
    vol = logret.rolling(window, min_periods=window).std() * math.sqrt(TRADING_DAYS_YR)
    return vol.to_numpy()


# ---------- Core simulation (single-slot with SL, regime-aware TP and TS,
#            and age-based TP time-decay) ----------

def simulate(
    close: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    rank_signal: pd.DataFrame,
    tp_schedule: list[tuple[int, float]] | None = None,  # [(age_bars, tp_pct), ...] sorted ascending
    tp_pct_bull: float | None = None,  # regime-aware TP (C)
    tp_pct_bear: float | None = None,
    ts_bull: int | None = None,        # regime-aware TS (B/D)
    ts_bear: int | None = None,
    regime_type: str = "200dma",       # '200dma' or 'vol25'
    sl_pct: float = 15.0,
    default_tp_pct: float = 10.0,
    default_ts: int = 252,
) -> dict:
    """
    tp_schedule: if provided, age-based TP decay. e.g. [(0, 10), (63, 5), (126, 2), (189, 0)]
                 applies TP = entry*(1+x/100) where x is the value for the
                 largest age_bars <= current_age.
    tp_pct_bull/tp_pct_bear: if both provided, regime-aware TP overrides default_tp_pct.
    ts_bull/ts_bear: if both provided, regime-aware time-stop overrides default_ts.
    regime_type: '200dma' uses SPY>200dma; 'vol25' uses 21d vol > 25%.
    """
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

    if regime_type == "200dma":
        regime_bull = spy_above_200dma(close)
    elif regime_type == "vol25":
        vol = spy_realized_vol(close, 21)
        # bull = low-vol regime (vol <= 25%)
        regime_bull = (vol <= 0.25)
        # treat NaNs as bull to avoid bias toward longer stop on early days
        regime_bull = np.where(np.isfinite(vol), regime_bull, True)
    else:
        regime_bull = np.ones(n, dtype=bool)

    sl_mult = 1.0 - (sl_pct / 100.0) if sl_pct is not None else None

    # Pre-sort tp_schedule just in case
    if tp_schedule is not None:
        tp_schedule = sorted(tp_schedule, key=lambda x: x[0])

    cash = 0.0
    equity = np.zeros(n)
    positions = []
    open_pos = None  # single slot
    total_invested = 0.0

    def _get_tp_mult_for_age(age: int, fallback_mult: float) -> float:
        if tp_schedule is None:
            return fallback_mult
        cur = tp_schedule[0][1]
        for age_thresh, tp in tp_schedule:
            if age >= age_thresh:
                cur = tp
            else:
                break
        return 1.0 + cur / 100.0

    for mi, di in enumerate(month_idx):
        total_invested += DCA_MONTHLY
        cash += DCA_MONTHLY
        entry_idx = di + 1
        if entry_idx >= n:
            break

        # Open new position (only if no open position: single-slot)
        if open_pos is None:
            bull_at_entry = bool(regime_bull[di])
            scores = rank_arr[di]
            pv_row = close_arr[di]
            valid_hist = price_valid[di]
            best_tk, best_score = -1, -np.inf
            for ti in range(n_tk):
                s, px, vh = scores[ti], pv_row[ti], valid_hist[ti]
                if not (np.isfinite(s) and np.isfinite(px) and px > 0 and vh >= RANK_LOOKBACK):
                    continue
                if s > best_score:
                    best_score = s
                    best_tk = ti
            if best_tk >= 0:
                entry_px = close_arr[entry_idx, best_tk]
                if np.isfinite(entry_px) and entry_px > 0:
                    # Decide TP and TS based on regime at entry (signal day D)
                    if tp_pct_bull is not None and tp_pct_bear is not None:
                        tp_base = tp_pct_bull if bull_at_entry else tp_pct_bear
                    else:
                        tp_base = default_tp_pct
                    if ts_bull is not None and ts_bear is not None:
                        ts_bars = ts_bull if bull_at_entry else ts_bear
                    else:
                        ts_bars = default_ts

                    deploy = cash
                    cash = 0.0
                    shares = deploy / entry_px
                    # Initial TP — if tp_schedule present, first entry is at age 0
                    init_tp_mult = 1.0 + tp_base / 100.0
                    open_pos = {
                        "tk_idx": best_tk,
                        "tk": tickers[best_tk],
                        "entry_idx": entry_idx,
                        "entry_px": entry_px,
                        "tp_base_pct": tp_base,  # regime-based floor if no decay
                        "init_tp_mult": init_tp_mult,
                        "sl_px": entry_px * sl_mult if sl_mult is not None else None,
                        "shares": shares,
                        "stop_idx": entry_idx + ts_bars,
                        "cost": deploy,
                        "bull_at_entry": bull_at_entry,
                        "ts_bars": ts_bars,
                    }

        next_di = month_idx[mi + 1] if (mi + 1) < len(month_idx) else n
        cash, open_pos = _run_segment(
            equity, di, next_di, open_pos, close_arr, high_arr, low_arr,
            cash, positions, tp_schedule, regime_bull,
        )

    if month_idx:
        cash, open_pos = _run_segment(
            equity, month_idx[-1], n, open_pos, close_arr, high_arr, low_arr,
            cash, positions, tp_schedule, regime_bull, final=True,
        )
    for i in range(n):
        if equity[i] == 0 and i > 0:
            equity[i] = equity[i - 1]

    return {
        "equity": equity,
        "positions": positions,
        "open_pos": open_pos,
        "total_invested": total_invested,
        "dates": dates,
    }


def _get_tp_mult_for_age(tp_schedule, age, fallback_mult):
    if tp_schedule is None:
        return fallback_mult
    cur = tp_schedule[0][1]
    for age_thresh, tp in tp_schedule:
        if age >= age_thresh:
            cur = tp
        else:
            break
    return 1.0 + cur / 100.0


def _run_segment(equity, start_i, end_i, open_pos, close_arr, high_arr, low_arr,
                 cash, positions, tp_schedule, regime_bull, final=False):
    for d in range(start_i, end_i):
        if open_pos is not None and d > open_pos["entry_idx"]:
            tk = open_pos["tk_idx"]
            lo = low_arr[d, tk]
            hi = high_arr[d, tk]
            exited = False

            # Stop loss first (conservative)
            if open_pos["sl_px"] is not None and np.isfinite(lo) and lo <= open_pos["sl_px"]:
                exit_px = open_pos["sl_px"]
                cash += open_pos["shares"] * exit_px
                positions.append(_close_trade(open_pos, d, exit_px, "sl"))
                open_pos = None
                exited = True

            # Take profit with possible age-based decay
            if not exited and open_pos is not None:
                age = d - open_pos["entry_idx"]
                tp_mult = _get_tp_mult_for_age(tp_schedule, age, open_pos["init_tp_mult"])
                tp_px = open_pos["entry_px"] * tp_mult
                # If decay has made TP <= 1 (breakeven), fill at close (conservative exit at close)
                # but still prefer intraday high-based fill if high >= tp_px
                if np.isfinite(hi) and hi >= tp_px:
                    # For TP <= entry, exit at close to avoid over-optimism
                    if tp_mult <= 1.0:
                        # breakeven/exit: use close if open >= tp, else fill at close
                        close_px = close_arr[d, tk]
                        exit_px = close_px if np.isfinite(close_px) and close_px > 0 else tp_px
                        cash += open_pos["shares"] * exit_px
                        positions.append(_close_trade(open_pos, d, exit_px, "tp_decay"))
                    else:
                        cash += open_pos["shares"] * tp_px
                        positions.append(_close_trade(open_pos, d, tp_px, "tp"))
                    open_pos = None
                    exited = True

            # Time stop
            if not exited and open_pos is not None and d >= open_pos["stop_idx"]:
                px = close_arr[d, tk]
                if not (np.isfinite(px) and px > 0):
                    for back in range(d, open_pos["entry_idx"], -1):
                        p2 = close_arr[back, tk]
                        if np.isfinite(p2) and p2 > 0:
                            px = p2
                            break
                    else:
                        px = open_pos["entry_px"]
                cash += open_pos["shares"] * px
                positions.append(_close_trade(open_pos, d, px, "time"))
                open_pos = None

        # Mark equity
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
        "exit_reason": reason,
        "bull_at_entry": pos.get("bull_at_entry", None),
    }


def compute_metrics(result: dict, dates=None) -> dict:
    equity = result["equity"]
    total_invested = result["total_invested"]
    positions = result["positions"]
    if dates is None:
        dates = result.get("dates")
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
    tp_hits = sum(1 for p in positions if p["exit_reason"] in ("tp", "tp_decay"))
    sl_hits = sum(1 for p in positions if p["exit_reason"] == "sl")
    time_hits = sum(1 for p in positions if p["exit_reason"] == "time")
    win_rate = tp_hits / n_trades if n_trades > 0 else 0.0
    trade_rets = [p["proceeds"] / p["cost"] - 1 for p in positions if p["cost"] > 0]
    gross_win_rate = sum(1 for r in trade_rets if r > 0) / n_trades if n_trades > 0 else 0.0
    avg_ret = float(np.mean(trade_rets)) if trade_rets else 0.0
    avg_days = float(np.mean([p["days_held"] for p in positions])) if positions else 0.0

    return {
        "cagr": cagr, "maxdd": maxdd, "sharpe": sharpe, "calmar": calmar,
        "win_rate": win_rate, "gross_win_rate": gross_win_rate,
        "avg_trade_ret": avg_ret, "avg_days_held": avg_days, "n_trades": n_trades,
        "tp_hits": tp_hits, "sl_hits": sl_hits, "time_hits": time_hits,
        "final_equity": final, "total_invested": total_invested,
    }


def crisis_stats(result, close):
    """Compute equity drawdown in 2008 (GFC), 2020 (COVID), 2022 (bear)."""
    equity = result["equity"]
    dates = close.index
    def window_mdd(start, end):
        mask = (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
        eq = np.asarray(equity)[mask]
        if len(eq) == 0:
            return None
        peak, mdd = 0.0, 0.0
        for v in eq:
            if v > peak:
                peak = v
            if peak > 0:
                d = (peak - v) / peak
                if d > mdd:
                    mdd = d
        start_v = eq[0] if len(eq) > 0 else 0
        end_v = eq[-1] if len(eq) > 0 else 0
        period_ret = (end_v / start_v - 1) if start_v > 0 else None
        return {"window_mdd": mdd, "period_ret": period_ret, "start_eq": float(start_v), "end_eq": float(end_v)}
    return {
        "gfc_2008_09": window_mdd("2007-10-01", "2009-06-30"),
        "covid_2020": window_mdd("2020-02-01", "2020-12-31"),
        "bear_2022": window_mdd("2022-01-01", "2022-12-31"),
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
    return {"equity": equity, "total_invested": total_invested, "positions": [], "dates": dates}


def run_variant(close, high, low, rank_signal, label, **kwargs):
    r = simulate(close, high, low, rank_signal, **kwargs)
    m = compute_metrics(r, dates=close.index)
    m["label"] = label
    m["config"] = {k: v for k, v in kwargs.items() if k != "tp_schedule" or v is None}
    if kwargs.get("tp_schedule") is not None:
        m["config"]["tp_schedule"] = kwargs["tp_schedule"]
    return r, m


def main():
    print("Loading data...", flush=True)
    close, high, low, finals = load_data()
    rank_signal = compute_rank_signal(finals)
    print(f"  shape: {close.shape}  range: {close.index[0].date()} -> {close.index[-1].date()}", flush=True)

    results = {}

    # ----- SPY DCA baseline -----
    print("\n[baseline] SPY DCA...", flush=True)
    spy_r = spy_dca_baseline(close)
    spy_m = compute_metrics(spy_r, dates=close.index)
    spy_m["crisis"] = crisis_stats(spy_r, close)
    results["spy_dca"] = spy_m
    print(f"   CAGR {spy_m['cagr']*100:.2f}%  MDD {spy_m['maxdd']*100:.1f}%  Sharpe {spy_m['sharpe']:.2f}", flush=True)

    # ----- Control: step41 winner (TP10/SL15/TS252 fixed single-slot) -----
    print("\n[control] TP10/SL15/TS252 fixed, single slot...", flush=True)
    ctrl_r, ctrl_m = run_variant(close, high, low, rank_signal, "control_tp10_sl15_ts252",
                                 tp_schedule=None, default_tp_pct=10.0, default_ts=252, sl_pct=15.0)
    ctrl_m["crisis"] = crisis_stats(ctrl_r, close)
    results["control"] = ctrl_m
    print(f"   CAGR {ctrl_m['cagr']*100:.2f}%  MDD {ctrl_m['maxdd']*100:.1f}%  Calmar {ctrl_m['calmar']:.3f}  WR {ctrl_m['win_rate']*100:.1f}%  N {ctrl_m['n_trades']}", flush=True)

    # =========================================================
    # Variant A — TP time-decay
    # =========================================================
    print("\n[A] TP time-decay schedules...", flush=True)
    # Gentle, medium, aggressive decays; stepped vs linear-ish
    schedules = {
        "A_step_10_5_2_0":       [(0, 10), (63, 5),  (126, 2),  (189, 0)],
        "A_step_10_7_5_2":       [(0, 10), (63, 7),  (126, 5),  (189, 2)],
        "A_step_10_5":           [(0, 10), (126, 5)],
        "A_step_12_8_5_2":       [(0, 12), (63, 8),  (126, 5),  (189, 2)],
        "A_aggressive_10_3":     [(0, 10), (63, 3)],
        "A_gentle_10_8_6_4":     [(0, 10), (63, 8),  (126, 6),  (189, 4)],
        "A_bigger_15_10_5_0":    [(0, 15), (63, 10), (126, 5),  (189, 0)],
        "A_monthly_10_7_5_3_1":  [(0, 10), (42, 7),  (84, 5),   (126, 3),  (168, 1)],
    }
    for label, sched in schedules.items():
        r, m = run_variant(close, high, low, rank_signal, label,
                           tp_schedule=sched, default_ts=252, sl_pct=15.0)
        m["crisis"] = crisis_stats(r, close)
        results[label] = m
        print(f"   {label:<28} CAGR {m['cagr']*100:>6.2f}%  MDD {m['maxdd']*100:>5.1f}%  Calmar {m['calmar']:>5.3f}  WR {m['win_rate']*100:>5.1f}%  AvgD {m['avg_days_held']:>5.0f}  N {m['n_trades']}", flush=True)

    # =========================================================
    # Variant B — Regime-aware time-stop (SPY 200dma)
    # =========================================================
    print("\n[B] Regime-aware time-stop (SPY 200dma)...", flush=True)
    ts_variants = {
        "B_spy200_bull126_bear378":  (126, 378),
        "B_spy200_bull126_bear504":  (126, 504),
        "B_spy200_bull189_bear504":  (189, 504),
        "B_spy200_bull252_bear504":  (252, 504),
        "B_spy200_bull63_bear378":   (63,  378),
        "B_spy200_reverse_252_126":  (252, 126),  # reverse: shorter in bear (does it hurt?)
    }
    for label, (tbull, tbear) in ts_variants.items():
        r, m = run_variant(close, high, low, rank_signal, label,
                           ts_bull=tbull, ts_bear=tbear, regime_type="200dma",
                           default_tp_pct=10.0, sl_pct=15.0)
        m["crisis"] = crisis_stats(r, close)
        results[label] = m
        print(f"   {label:<30} CAGR {m['cagr']*100:>6.2f}%  MDD {m['maxdd']*100:>5.1f}%  Calmar {m['calmar']:>5.3f}  WR {m['win_rate']*100:>5.1f}%  AvgD {m['avg_days_held']:>5.0f}", flush=True)

    # =========================================================
    # Variant C — Regime-aware TP (SPY 200dma)
    # =========================================================
    print("\n[C] Regime-aware TP (SPY 200dma)...", flush=True)
    tp_variants = {
        "C_spy200_bull10_bear15": (10.0, 15.0),
        "C_spy200_bull10_bear20": (10.0, 20.0),
        "C_spy200_bull15_bear10": (15.0, 10.0),   # reverse
        "C_spy200_bull8_bear15":  (8.0,  15.0),
        "C_spy200_bull12_bear20": (12.0, 20.0),
    }
    for label, (tpb, tpbr) in tp_variants.items():
        r, m = run_variant(close, high, low, rank_signal, label,
                           tp_pct_bull=tpb, tp_pct_bear=tpbr, regime_type="200dma",
                           default_ts=252, sl_pct=15.0)
        m["crisis"] = crisis_stats(r, close)
        results[label] = m
        print(f"   {label:<30} CAGR {m['cagr']*100:>6.2f}%  MDD {m['maxdd']*100:>5.1f}%  Calmar {m['calmar']:>5.3f}  WR {m['win_rate']*100:>5.1f}%  AvgD {m['avg_days_held']:>5.0f}", flush=True)

    # =========================================================
    # Variant D — Volatility-regime time-stop
    # =========================================================
    print("\n[D] Volatility-regime time-stop (21d annualized vol > 25%)...", flush=True)
    dvol_variants = {
        "D_vol25_bull126_bear378": (126, 378),
        "D_vol25_bull126_bear504": (126, 504),
        "D_vol25_bull252_bear504": (252, 504),
        "D_vol25_bull63_bear378":  (63,  378),
    }
    for label, (tbull, tbear) in dvol_variants.items():
        r, m = run_variant(close, high, low, rank_signal, label,
                           ts_bull=tbull, ts_bear=tbear, regime_type="vol25",
                           default_tp_pct=10.0, sl_pct=15.0)
        m["crisis"] = crisis_stats(r, close)
        results[label] = m
        print(f"   {label:<30} CAGR {m['cagr']*100:>6.2f}%  MDD {m['maxdd']*100:>5.1f}%  Calmar {m['calmar']:>5.3f}  WR {m['win_rate']*100:>5.1f}%  AvgD {m['avg_days_held']:>5.0f}", flush=True)

    # =========================================================
    # Variant E — Combined best
    # =========================================================
    print("\n[E] Combined best variants...", flush=True)
    # Pick best A by Calmar, best B, best C
    def _best(prefix):
        cands = {k: v for k, v in results.items() if k.startswith(prefix) and v}
        if not cands:
            return None
        return max(cands.items(), key=lambda kv: kv[1].get("calmar", 0))

    bestA = _best("A_")
    bestB = _best("B_spy200")
    bestC = _best("C_spy200")
    print(f"   best A: {bestA[0] if bestA else None}")
    print(f"   best B: {bestB[0] if bestB else None}")
    print(f"   best C: {bestC[0] if bestC else None}")

    # Combo 1: A-best TP decay + B-best regime TS
    if bestA and bestB:
        bestA_sched = schedules[bestA[0]]
        bull_ts, bear_ts = ts_variants[bestB[0]]
        label = "E_combo_Abest_decay_Bbest_regime_ts"
        r, m = run_variant(close, high, low, rank_signal, label,
                           tp_schedule=bestA_sched, ts_bull=bull_ts, ts_bear=bear_ts,
                           regime_type="200dma", sl_pct=15.0)
        m["crisis"] = crisis_stats(r, close)
        results[label] = m
        print(f"   {label:<40} CAGR {m['cagr']*100:>6.2f}%  MDD {m['maxdd']*100:>5.1f}%  Calmar {m['calmar']:>5.3f}  WR {m['win_rate']*100:>5.1f}%", flush=True)

    # Combo 2: regime TP + regime TS
    if bestB and bestC:
        bull_ts, bear_ts = ts_variants[bestB[0]]
        tpb, tpbr = tp_variants[bestC[0]]
        label = "E_combo_Cbest_tp_Bbest_ts"
        r, m = run_variant(close, high, low, rank_signal, label,
                           tp_pct_bull=tpb, tp_pct_bear=tpbr,
                           ts_bull=bull_ts, ts_bear=bear_ts,
                           regime_type="200dma", sl_pct=15.0)
        m["crisis"] = crisis_stats(r, close)
        results[label] = m
        print(f"   {label:<40} CAGR {m['cagr']*100:>6.2f}%  MDD {m['maxdd']*100:>5.1f}%  Calmar {m['calmar']:>5.3f}  WR {m['win_rate']*100:>5.1f}%", flush=True)

    # Combo 3: full stack A + B + SL15
    if bestA and bestB:
        label = "E_combo_Aplus_Bplus"
        bestA_sched = schedules[bestA[0]]
        bull_ts, bear_ts = ts_variants[bestB[0]]
        r, m = run_variant(close, high, low, rank_signal, label,
                           tp_schedule=bestA_sched,
                           ts_bull=bull_ts, ts_bear=bear_ts,
                           regime_type="200dma", sl_pct=15.0)
        m["crisis"] = crisis_stats(r, close)
        results[label] = m
        print(f"   {label:<40} CAGR {m['cagr']*100:>6.2f}%  MDD {m['maxdd']*100:>5.1f}%  Calmar {m['calmar']:>5.3f}  WR {m['win_rate']*100:>5.1f}%", flush=True)

    # Save
    out_path = Path("/home/user/crt/max/research/step50_results.json")
    # Drop non-serializable config nesting issues
    out_path.write_text(json.dumps(results, indent=2, default=float))
    print(f"\nWrote {out_path}", flush=True)

    # Summary sort
    print("\n=== All variants sorted by CAGR (ex-baseline) ===", flush=True)
    items = [(k, v) for k, v in results.items() if k != "spy_dca" and v and v.get("cagr", 0) > -0.5]
    items.sort(key=lambda kv: kv[1].get("cagr", 0), reverse=True)
    for k, v in items[:20]:
        print(f"  {k:<40} CAGR {v['cagr']*100:>6.2f}%  MDD {v['maxdd']*100:>5.1f}%  Calmar {v['calmar']:>5.3f}  WR {v.get('win_rate',0)*100:>5.1f}%  N {v.get('n_trades','-')}", flush=True)

    print("\n=== Sorted by Calmar ===", flush=True)
    items.sort(key=lambda kv: kv[1].get("calmar", 0), reverse=True)
    for k, v in items[:20]:
        print(f"  {k:<40} CAGR {v['cagr']*100:>6.2f}%  MDD {v['maxdd']*100:>5.1f}%  Calmar {v['calmar']:>5.3f}  WR {v.get('win_rate',0)*100:>5.1f}%", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
