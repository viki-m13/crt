#!/usr/bin/env python3
"""Step 42: Volatility-adaptive take-profit grid search.

Strategy skeleton
-----------------
- Entry ranker: CAP5+SMA12M = trailing 252-bar mean of `final` from bt_ext.
- Monthly top-1 pick (SPY excluded); enter at next-day close.
- Take-profit exit at first day the High >= TP (fill at TP price).
- Else exit at close of last bar in time_stop window.
- Monthly $1000 DCA; cash idle between trades.
- No look-ahead: ATR/sigma/quantile computed strictly before the rank date.
- Skip tickers with <252 bars of history on rank date.

TP formulas
-----------
1. ATR-based:    TP = entry * (1 + k * ATR14 / entry),        k in {1,1.5,2,2.5,3,4,5}
2. Sigma-based:  TP = entry * (1 + k * sigma_60d_scaled),     k in {0.5,1,1.5,2,2.5}
                 (sigma_60d is 60-bar std of log returns; scaled as sqrt(time_stop))
3. Quantile:     TP = entry * (1 + Q% of ticker's historical n-bar forward returns,
                 where n = time_stop_bars and window = last 3Y at rank date)
                 Q in {50,60,70,80}

Pair each TP formula with time_stop_bars in {21, 42, 63, 126, 252}.

Outputs
-------
- max/research/step42_results.json: all combos + summary stats
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"

START_DATE = pd.Timestamp("2006-04-25")
END_DATE = pd.Timestamp("2026-04-22")
DCA_MONTHLY = 1000.0
TRADING_DAYS_YR = 252
SMOOTH_WINDOW = 252  # CAP5+SMA12M: trailing-252 mean of `final`
MIN_HISTORY = 252

TIME_STOPS = [21, 42, 63, 126, 252]
ATR_KS = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
SIGMA_KS = [0.5, 1.0, 1.5, 2.0, 2.5]
QUANTILE_QS = [50, 60, 70, 80]


# ----------------------------- Data loading ----------------------------- #

def load_data():
    close = pd.read_parquet(RAW_DIR / "Close.parquet")
    high = pd.read_parquet(RAW_DIR / "High.parquet")
    low = pd.read_parquet(RAW_DIR / "Low.parquet")
    bt = pd.read_parquet(DATA_DIR / "bt_ext.parquet")

    # Trim to backtest window
    mask = (close.index >= START_DATE) & (close.index <= END_DATE)
    close = close.loc[mask]
    high = high.reindex(close.index)
    low = low.reindex(close.index)
    # Drop any trailing date where all prices are NaN (e.g. partial end-of-data
    # row) so the final equity mark-to-market isn't NaN-starved.
    valid = close.notna().any(axis=1)
    last_valid_pos = int(np.where(valid)[0][-1])
    close = close.iloc[: last_valid_pos + 1]
    high = high.iloc[: last_valid_pos + 1]
    low = low.iloc[: last_valid_pos + 1]

    # Build smoothed final (CAP5+SMA12M) as a wide DataFrame reindexed to close.
    # bt_ext already holds per-ticker point-in-time final.
    pivot = bt.pivot(index="date", columns="ticker", values="final")
    pivot.index = pd.to_datetime(pivot.index)
    pivot = pivot.reindex(close.index).ffill(limit=5)
    # Trailing 252-day mean (CAP5 is "final" already; SMA12M is the 252-bar mean)
    final_smooth = pivot.rolling(window=SMOOTH_WINDOW, min_periods=SMOOTH_WINDOW).mean()

    return close, high, low, final_smooth


def monthly_first_indices(dates: pd.DatetimeIndex) -> list[int]:
    months = pd.Series(dates).dt.to_period("M").values
    idxs = []
    prev = None
    for i, m in enumerate(months):
        if m != prev:
            idxs.append(i)
            prev = m
    return idxs


# ----------------------------- Volatility ----------------------------- #

def wilder_atr(high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame,
               period: int = 14) -> pd.DataFrame:
    """Wilder ATR per ticker, wide DataFrame (float64)."""
    prev_close = close.shift(1)
    a = (high - low).values
    b = (high - prev_close).abs().values
    c = (low - prev_close).abs().values
    with np.errstate(invalid="ignore"):
        stack = np.stack([a, b, c], axis=0)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            tr_np = np.nanmax(stack, axis=0)
    tr = pd.DataFrame(tr_np, index=close.index, columns=close.columns)
    # Wilder's smoothing: EMA with alpha = 1/period (equivalent to SMMA)
    atr = tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    return atr


def realized_log_sigma(close: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """Rolling std of log-returns over `window` bars (per-bar, unannualized)."""
    lr = np.log(close).diff()
    return lr.rolling(window=window, min_periods=window).std()


# ----------------------------- Simulator ----------------------------- #

@dataclass
class Trade:
    ticker: str
    buy_date: pd.Timestamp
    sell_date: pd.Timestamp
    buy_price: float
    tp_price: float
    sell_price: float
    ret: float
    days_held: int
    hit_tp: bool


def compute_tp(formula: str, k_or_q, entry: float, ctx: dict) -> float:
    if formula == "atr":
        atr = ctx["atr14"]
        if not math.isfinite(atr) or atr <= 0:
            return math.nan
        return entry * (1.0 + k_or_q * atr / entry)
    if formula == "sigma":
        sigma60 = ctx["sigma60"]
        ts = ctx["time_stop"]
        if not math.isfinite(sigma60) or sigma60 <= 0:
            return math.nan
        # Scale per-bar log-vol to the time-stop window:
        # expected total move ~ sigma60 * sqrt(time_stop)
        move = sigma60 * math.sqrt(ts)
        return entry * (1.0 + k_or_q * move)
    if formula == "quantile":
        fwd_rets = ctx["fwd_rets"]  # numpy array of historical forward returns
        if fwd_rets is None or len(fwd_rets) < 30:
            return math.nan
        q = float(np.quantile(fwd_rets, k_or_q / 100.0))
        # quantile of historical forward returns is a relative move
        return entry * (1.0 + q)
    raise ValueError(formula)


def precompute_fwd_returns(close: pd.DataFrame, ts: int) -> dict:
    """For each ticker, return an array of forward ts-bar returns aligned to close.index."""
    fwd = close.shift(-ts) / close - 1.0
    return {tk: fwd[tk].values for tk in close.columns}


def run_combo(close, high, low, final_smooth,
              atr14, sigma60, fwd_rets_cache,
              formula: str, param, time_stop: int,
              excluded=("SPY",)) -> dict:
    """Execute one (formula, param, time_stop) combo. Returns metrics dict."""
    dates = close.index
    n = len(dates)
    mf_idxs = monthly_first_indices(dates)

    tickers = [c for c in close.columns if c not in excluded]

    # Precompute numpy arrays per ticker for speed.
    # close_np: raw prices (for trade entry/exit decisions).
    # close_mtm: ffilled prices (for MTM valuation only) so a trailing NaN bar
    # doesn't zero out equity.
    close_np = {tk: close[tk].values for tk in tickers}
    close_mtm = {tk: close[tk].ffill().values for tk in tickers}
    high_np = {tk: high[tk].values for tk in tickers}
    fs_np = {tk: final_smooth[tk].values for tk in tickers}
    atr_np = {tk: atr14[tk].values for tk in tickers}
    sig_np = {tk: sigma60[tk].values for tk in tickers}
    fwd_np = fwd_rets_cache  # keyed by ticker, aligned forward returns for this ts

    equity = np.zeros(n)
    cash = 0.0
    total_invested = 0.0
    open_positions = []  # list of dicts
    trades: list[Trade] = []

    for m_idx, di in enumerate(mf_idxs):
        # Rank: choose top-1 eligible ticker by final_smooth[di]
        best_tk, best_score = None, -math.inf
        for tk in tickers:
            # History check
            if di < MIN_HISTORY:
                continue
            fv = fs_np[tk][di]
            pv = close_np[tk][di]
            if not (math.isfinite(fv) and math.isfinite(pv) and pv > 0 and fv > 0):
                continue
            # Need ATR/sigma present too (no look-ahead, computed up to di)
            a = atr_np[tk][di]
            s = sig_np[tk][di]
            if formula == "atr" and not (math.isfinite(a) and a > 0):
                continue
            if formula == "sigma" and not (math.isfinite(s) and s > 0):
                continue
            if formula == "quantile":
                # Need 3Y of forward returns strictly before di, sampled at index <= di - ts
                lo = max(0, di - 3 * TRADING_DAYS_YR)
                fr = fwd_np[tk][lo:di - time_stop + 1] if di - time_stop + 1 > lo else None
                if fr is None:
                    continue
                fr = fr[np.isfinite(fr)]
                if len(fr) < 30:
                    continue
            if fv > best_score:
                best_score = fv
                best_tk = tk

        entry_idx = di + 1
        if best_tk is None or entry_idx >= n:
            continue

        entry = close_np[best_tk][entry_idx]
        if not (math.isfinite(entry) and entry > 0):
            continue

        # Build TP
        ctx = {
            "atr14": atr_np[best_tk][di],
            "sigma60": sig_np[best_tk][di],
            "time_stop": time_stop,
        }
        if formula == "quantile":
            lo = max(0, di - 3 * TRADING_DAYS_YR)
            fr = fwd_np[best_tk][lo:di - time_stop + 1]
            fr = fr[np.isfinite(fr)]
            ctx["fwd_rets"] = fr
        tp = compute_tp(formula, param, entry, ctx)
        if not (math.isfinite(tp) and tp > entry):
            # Skip if TP computation fails or is non-positive above entry
            continue

        total_invested += DCA_MONTHLY
        shares = DCA_MONTHLY / entry

        # Walk forward to find exit
        last_idx = min(entry_idx + time_stop - 1, n - 1)
        hit_tp = False
        sell_price = math.nan
        sell_idx = last_idx
        for d in range(entry_idx, last_idx + 1):
            hi = high_np[best_tk][d]
            if math.isfinite(hi) and hi >= tp:
                sell_price = tp
                sell_idx = d
                hit_tp = True
                break
        if not hit_tp:
            # Close on last bar; fall back through the window if that close is NaN
            for d in range(last_idx, entry_idx - 1, -1):
                px = close_np[best_tk][d]
                if math.isfinite(px) and px > 0:
                    sell_price = px
                    sell_idx = d
                    break

        if not math.isfinite(sell_price) or sell_price <= 0:
            # Couldn't resolve an exit; undo the buy to avoid polluting stats
            total_invested -= DCA_MONTHLY
            continue

        cash_proceeds = shares * sell_price
        ret = sell_price / entry - 1.0
        trades.append(Trade(
            ticker=best_tk,
            buy_date=dates[entry_idx],
            sell_date=dates[sell_idx],
            buy_price=entry,
            tp_price=tp,
            sell_price=sell_price,
            ret=ret,
            days_held=sell_idx - entry_idx,
            hit_tp=hit_tp,
        ))

        # For equity curve we reconstruct per-day holding value
        open_positions.append({
            "tk": best_tk,
            "buy_idx": entry_idx,
            "sell_idx": sell_idx,
            "shares": shares,
            "sell_price": sell_price,
            "cost": DCA_MONTHLY,
        })

    # Build equity curve: invested cash is withdrawn from "external DCA pool".
    # Convention: equity = cash from closed trades + MTM of open positions + idle cash.
    # Idle cash = cumulative DCA - cost of currently-open positions.
    # Simpler: track everything day-by-day.
    monthly_di_set = set(mf_idxs)
    # Determine per-day: DCA in, buy out (same day as buy_idx == di + 1), sell in.
    # Easier to iterate positions.
    # cumulative invested (by day)
    inv_by_day = np.zeros(n)
    cum = 0.0
    # mark monthly in-flows ONLY if a trade was opened that month (because
    # cash "sits idle"-- but here we treat the $1000 as only invested if a
    # position was opened, matching total_invested above).
    # Actually the spec says: "monthly $1000 DCA regardless of trade status",
    # i.e. the cash arrives regardless. If no trade is opened, it sits as idle cash.
    # Re-interpret: total_invested should be every month's DCA; idle cash
    # accumulates between trades.
    # We'll rebuild total_invested:
    total_invested_all = DCA_MONTHLY * len(mf_idxs)

    # Build equity day by day with "idle cash" semantics
    # Cash schedule: +DCA_MONTHLY on each month-first day; -cost on buy_idx of each trade;
    # +proceeds on sell_idx of each trade.
    cash_flow = np.zeros(n)
    for di in mf_idxs:
        cash_flow[di] += DCA_MONTHLY
    trade_lookup = {}
    for pos in open_positions:
        cash_flow[pos["buy_idx"]] -= pos["cost"]
        cash_flow[pos["sell_idx"]] += pos["shares"] * pos["sell_price"]
    cash_running = np.cumsum(cash_flow)

    # Mark-to-market value of open positions each day (uses ffilled price).
    mtm = np.zeros(n)
    for pos in open_positions:
        for d in range(pos["buy_idx"], pos["sell_idx"]):
            px = close_mtm[pos["tk"]][d]
            if math.isfinite(px) and px > 0:
                mtm[d] += pos["shares"] * px
    equity = cash_running + mtm

    # Metrics
    return summarize(equity, total_invested_all, trades, mf_idxs, dates, n)


def summarize(equity, total_invested, trades, mf_idxs, dates, n) -> dict:
    if total_invested <= 0 or n == 0:
        return {"cagr": 0, "mdd": 0, "sharpe": 0, "calmar": 0,
                "win_rate": 0, "avg_ret": 0, "avg_days": 0, "n_trades": 0,
                "sigma_ticker_wr": 0, "final": 0, "invested": 0,
                "per_ticker_wr": {}}
    final = float(equity[-1])
    yrs = n / TRADING_DAYS_YR
    cagr = (final / total_invested) ** (1 / yrs) - 1 if yrs > 0 and final > 0 else 0.0

    # Monthly returns for Sharpe
    monthly_rets = []
    for i in range(1, len(mf_idxs)):
        a = equity[mf_idxs[i - 1]]
        b = equity[mf_idxs[i]]
        if a > 0:
            monthly_rets.append(b / a - 1.0)
    arr = np.array(monthly_rets)
    avg = arr.mean() if arr.size else 0.0
    std = arr.std() if arr.size else 0.0
    sharpe = (avg / std) * math.sqrt(12) if std > 0 else 0.0

    # Max DD
    peak = 0.0
    mdd = 0.0
    for v in equity:
        if v > peak:
            peak = v
        if peak > 0:
            dd = (peak - v) / peak
            if dd > mdd:
                mdd = dd
    calmar = cagr / mdd if mdd > 0 else 0.0

    if trades:
        win_rate = sum(1 for t in trades if t.hit_tp) / len(trades)
        avg_ret = float(np.mean([t.ret for t in trades]))
        avg_days = float(np.mean([t.days_held for t in trades]))
    else:
        win_rate = avg_ret = avg_days = 0.0

    # Per-ticker win rates
    per_tk = {}
    for t in trades:
        per_tk.setdefault(t.ticker, []).append(1 if t.hit_tp else 0)
    per_ticker_wr = {tk: (sum(v) / len(v)) for tk, v in per_tk.items()}
    if per_ticker_wr:
        # Weight ignores sample-size; spec calls for simple sigma over tickers.
        sigma_tk_wr = float(np.std(list(per_ticker_wr.values())))
    else:
        sigma_tk_wr = 0.0

    return {
        "cagr": cagr,
        "mdd": mdd,
        "sharpe": sharpe,
        "calmar": calmar,
        "win_rate": win_rate,
        "avg_ret": avg_ret,
        "avg_days": avg_days,
        "n_trades": len(trades),
        "sigma_ticker_wr": sigma_tk_wr,
        "final": final,
        "invested": total_invested,
        "per_ticker_wr": per_ticker_wr,
        "per_ticker_n": {tk: len(v) for tk, v in per_tk.items()},
    }


def run_fixed_pct(close, high, low, final_smooth, pct: float, time_stop: int,
                  excluded=("SPY",)) -> dict:
    """Fixed-% TP reference: TP = entry * (1 + pct)."""
    dates = close.index
    n = len(dates)
    mf_idxs = monthly_first_indices(dates)
    tickers = [c for c in close.columns if c not in excluded]

    close_np = {tk: close[tk].values for tk in tickers}
    close_mtm = {tk: close[tk].ffill().values for tk in tickers}
    high_np = {tk: high[tk].values for tk in tickers}
    fs_np = {tk: final_smooth[tk].values for tk in tickers}

    trades = []
    open_positions = []

    for di in mf_idxs:
        best_tk, best_score = None, -math.inf
        for tk in tickers:
            if di < MIN_HISTORY:
                continue
            fv = fs_np[tk][di]
            pv = close_np[tk][di]
            if not (math.isfinite(fv) and math.isfinite(pv) and pv > 0 and fv > 0):
                continue
            if fv > best_score:
                best_score = fv
                best_tk = tk
        entry_idx = di + 1
        if best_tk is None or entry_idx >= n:
            continue
        entry = close_np[best_tk][entry_idx]
        if not (math.isfinite(entry) and entry > 0):
            continue
        tp = entry * (1.0 + pct)
        last_idx = min(entry_idx + time_stop - 1, n - 1)
        hit = False
        sell_price = math.nan
        sell_idx = last_idx
        for d in range(entry_idx, last_idx + 1):
            hi = high_np[best_tk][d]
            if math.isfinite(hi) and hi >= tp:
                sell_price = tp
                sell_idx = d
                hit = True
                break
        if not hit:
            for d in range(last_idx, entry_idx - 1, -1):
                px = close_np[best_tk][d]
                if math.isfinite(px) and px > 0:
                    sell_price = px
                    sell_idx = d
                    break
        if not math.isfinite(sell_price) or sell_price <= 0:
            continue
        shares = DCA_MONTHLY / entry
        ret = sell_price / entry - 1.0
        trades.append(Trade(
            ticker=best_tk, buy_date=dates[entry_idx], sell_date=dates[sell_idx],
            buy_price=entry, tp_price=tp, sell_price=sell_price, ret=ret,
            days_held=sell_idx - entry_idx, hit_tp=hit,
        ))
        open_positions.append({
            "tk": best_tk, "buy_idx": entry_idx, "sell_idx": sell_idx,
            "shares": shares, "sell_price": sell_price, "cost": DCA_MONTHLY,
        })

    total_invested_all = DCA_MONTHLY * len(mf_idxs)
    cash_flow = np.zeros(n)
    for di in mf_idxs:
        cash_flow[di] += DCA_MONTHLY
    for pos in open_positions:
        cash_flow[pos["buy_idx"]] -= pos["cost"]
        cash_flow[pos["sell_idx"]] += pos["shares"] * pos["sell_price"]
    cash_running = np.cumsum(cash_flow)
    mtm = np.zeros(n)
    for pos in open_positions:
        for d in range(pos["buy_idx"], pos["sell_idx"]):
            px = close_np[pos["tk"]][d]
            if math.isfinite(px) and px > 0:
                mtm[d] += pos["shares"] * px
    equity = cash_running + mtm
    return summarize(equity, total_invested_all, trades, mf_idxs, dates, n)


# ----------------------------- Benchmark: SPY DCA ----------------------------- #

def spy_dca(close: pd.DataFrame) -> dict:
    dates = close.index
    n = len(dates)
    mf_idxs = monthly_first_indices(dates)
    spy_ffill = close["SPY"].ffill().values
    # Each month: $1000 into SPY at next close, held to end.
    total_invested = 0.0
    equity = np.zeros(n)
    buys = []  # (buy_idx, shares)
    for di in mf_idxs:
        ei = di + 1
        if ei >= n:
            continue
        px = spy_ffill[ei]
        if not (math.isfinite(px) and px > 0):
            continue
        total_invested += DCA_MONTHLY
        s = DCA_MONTHLY / px
        buys.append((ei, s))

    cum_shares = np.zeros(n)
    for ei, s in buys:
        cum_shares[ei:] += s
    for d in range(n):
        px = spy_ffill[d]
        if math.isfinite(px):
            equity[d] = cum_shares[d] * px
    return summarize(equity, total_invested, [], mf_idxs, dates, n)


# ----------------------------- Driver ----------------------------- #

def main():
    print("[load] reading parquet data...")
    close, high, low, final_smooth = load_data()
    print(f"[load] close shape {close.shape}, dates {close.index[0].date()}..{close.index[-1].date()}")

    print("[prep] computing ATR14 and sigma60...")
    atr14 = wilder_atr(high, low, close, period=14)
    sigma60 = realized_log_sigma(close, window=60)

    print("[prep] caching forward-return arrays per time_stop...")
    fwd_cache = {ts: precompute_fwd_returns(close, ts) for ts in TIME_STOPS}

    # SPY DCA benchmark
    print("[bench] SPY DCA...")
    spy = spy_dca(close)
    print(f"  SPY DCA  CAGR={spy['cagr']*100:.2f}%  MDD={spy['mdd']*100:.2f}%  Sharpe={spy['sharpe']:.2f}")

    # Fixed-% reference line (agent #2 is the real test; we include 5% x 60d to anchor)
    print("[ref] fixed 5% x 60d...")
    fixed_ref = run_fixed_pct(close, high, low, final_smooth, pct=0.05, time_stop=60)
    print(f"  Fixed 5% x60  CAGR={fixed_ref['cagr']*100:.2f}%  "
          f"WR={fixed_ref['win_rate']*100:.1f}%  N={fixed_ref['n_trades']}")
    # Also 63 since that's one of our time_stops
    fixed_ref_63 = run_fixed_pct(close, high, low, final_smooth, pct=0.05, time_stop=63)

    results = {"spy_dca": spy, "fixed_5pct_60d": fixed_ref, "fixed_5pct_63d": fixed_ref_63,
               "atr": {}, "sigma": {}, "quantile": {}}

    # ATR
    print("[grid] ATR formula...")
    for k in ATR_KS:
        for ts in TIME_STOPS:
            fwd = fwd_cache[ts]
            r = run_combo(close, high, low, final_smooth, atr14, sigma60, fwd,
                          formula="atr", param=k, time_stop=ts)
            key = f"k{k}_ts{ts}"
            results["atr"][key] = r
            print(f"  ATR k={k:<4} ts={ts:<3}  CAGR={r['cagr']*100:6.2f}%  "
                  f"WR={r['win_rate']*100:5.1f}%  N={r['n_trades']:3d}  "
                  f"sigTkWR={r['sigma_ticker_wr']:.3f}")

    # Sigma
    print("[grid] sigma formula...")
    for k in SIGMA_KS:
        for ts in TIME_STOPS:
            fwd = fwd_cache[ts]
            r = run_combo(close, high, low, final_smooth, atr14, sigma60, fwd,
                          formula="sigma", param=k, time_stop=ts)
            key = f"k{k}_ts{ts}"
            results["sigma"][key] = r
            print(f"  SIG k={k:<4} ts={ts:<3}  CAGR={r['cagr']*100:6.2f}%  "
                  f"WR={r['win_rate']*100:5.1f}%  N={r['n_trades']:3d}  "
                  f"sigTkWR={r['sigma_ticker_wr']:.3f}")

    # Quantile
    print("[grid] quantile formula...")
    for q in QUANTILE_QS:
        for ts in TIME_STOPS:
            fwd = fwd_cache[ts]
            r = run_combo(close, high, low, final_smooth, atr14, sigma60, fwd,
                          formula="quantile", param=q, time_stop=ts)
            key = f"q{q}_ts{ts}"
            results["quantile"][key] = r
            print(f"  Q   q={q:<4} ts={ts:<3}  CAGR={r['cagr']*100:6.2f}%  "
                  f"WR={r['win_rate']*100:5.1f}%  N={r['n_trades']:3d}  "
                  f"sigTkWR={r['sigma_ticker_wr']:.3f}")

    # Save
    out_path = ROOT / "step42_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, default=_json_default, indent=2)
    print(f"[save] wrote {out_path}")


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Not serializable: {type(o)}")


if __name__ == "__main__":
    main()
