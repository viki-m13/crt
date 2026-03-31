#!/usr/bin/env python3
"""
VALET: Vol-Adaptive Leveraged ETF Timing
==========================================
SIMPLEST POSSIBLE VERSION:
  - ONE leveraged ETF (TQQQ or SPXL)
  - ONE cash instrument (SHY)
  - Binary decision: leverage or cash

RULES (dead simple, check once per week):
  1. Is SPY above its 200-day SMA? (trend filter)
  2. Is 42-day realized vol < 18%? (vol filter — slow entry)
  3. Is 5-day realized vol < 25%? (fast exit on vol spike)

  ALL YES → 100% TQQQ (or SPXL)
  ANY NO  → 100% SHY (cash)

That's it. No sector picking, no multi-factor, no inverse ETFs.
Leverage ONLY when it's mathematically safe (low vol + uptrend).

Vol-target the portfolio to 12% annualized.
Weekly rebalance, next-day open, 10bps slippage.
"""

import os, sys, datetime, math
import numpy as np
import pandas as pd
import yfinance as yf

ALL_ETFS = [
    "SPY", "QQQ", "TQQQ", "SPXL", "SQQQ", "SPXS", "SHY",
    "SSO", "QLD", "SDS", "SH",
    # Sector leveraged for experimentation
    "TECL", "FAS", "SOXL", "ERX", "CURE", "TNA",
    "TECS", "FAZ", "SOXS", "ERY", "TZA",
    # Havens
    "TLT", "GLD", "IEF", "UGL", "TMF",
    # Sectors
    "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC",
]
BENCHMARK = "SPY"
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# The ONE leveraged ETF to use
LEVERAGE_ETF = "TQQQ"  # 3x Nasdaq-100
CASH_ETF = "SHY"

# Vol thresholds
VOL_SLOW_THRESH = 0.18   # 42-day vol must be below this to enter
VOL_FAST_THRESH = 0.25   # 5-day vol must be below this (fast exit)
VOL_SLOW_WINDOW = 42
VOL_FAST_WINDOW = 5

TARGET_VOL = 0.12
VOL_SCALE_LOOKBACK = 21
VOL_FLOOR = 0.15
VOL_CAP = 2.0
SLIPPAGE_BPS = 10

TRAIN_START, TRAIN_END = "2012-01-01", "2019-12-31"
VALID_START, VALID_END = "2020-04-01", "2022-12-31"
TEST_START, TEST_END = "2023-04-01", "2026-03-28"


def download_etfs():
    os.makedirs(DATA_DIR, exist_ok=True)
    results = {}
    today = datetime.date.today().isoformat()
    for ticker in ALL_ETFS:
        cache_path = os.path.join(DATA_DIR, f"{ticker}.csv")
        if os.path.exists(cache_path):
            try:
                df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
                if len(df) > 100:
                    results[ticker] = df
                    continue
            except Exception:
                pass
        try:
            df = yf.download(ticker, start="2008-01-01", end=today, progress=False, auto_adjust=True)
            if df is None or len(df) < 100:
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.to_csv(cache_path)
            results[ticker] = df
        except Exception:
            pass
    return results


def run_strategy(data, leverage_etf, start, end):
    """
    Simple binary: leverage or cash.
    Check weekly. Execute at next-day open.
    """
    spy = data[BENCHMARK]
    spy_close = spy["Close"]
    spy_ret = spy_close.pct_change()
    spy_sma200 = spy_close.rolling(200).mean()
    spy_vol_fast = spy_ret.rolling(VOL_FAST_WINDOW, min_periods=3).std() * np.sqrt(252)
    spy_vol_slow = spy_ret.rolling(VOL_SLOW_WINDOW, min_periods=21).std() * np.sqrt(252)

    dates = spy.loc[start:end].index
    slip = SLIPPAGE_BPS / 10000

    daily_rets, raw_rets = [], []
    in_leverage = False
    last_week = None
    trades = 0

    for date in dates:
        idx = spy.index.get_loc(date)
        if idx < 252:
            daily_rets.append(0.0)
            raw_rets.append(0.0)
            continue

        week = date.isocalendar()[1]
        rebalance = (last_week is not None and week != last_week)
        last_week = week

        dr = 0.0

        if rebalance:
            sma = spy_sma200.loc[date] if date in spy_sma200.index else None
            vf = spy_vol_fast.loc[date] if date in spy_vol_fast.index else 0.20
            vs = spy_vol_slow.loc[date] if date in spy_vol_slow.index else 0.20
            if pd.isna(vf): vf = 0.20
            if pd.isna(vs): vs = 0.20

            trend_ok = sma is not None and not pd.isna(sma) and spy_close.loc[date] > sma
            vol_ok = vs < VOL_SLOW_THRESH and vf < VOL_FAST_THRESH

            should_leverage = trend_ok and vol_ok

            if should_leverage != in_leverage:
                # Switching: exit old, enter new
                old_etf = leverage_etf if in_leverage else CASH_ETF
                new_etf = leverage_etf if should_leverage else CASH_ETF

                # Exit old position (prev_close to today_open)
                df_old = data.get(old_etf)
                if df_old is not None and date in df_old.index:
                    si = df_old.index.get_loc(date)
                    if si > 0:
                        prev_c = df_old.iloc[si - 1]["Close"]
                        today_o = df_old.loc[date, "Open"] if "Open" in df_old.columns else prev_c
                        dr += (today_o * (1 - slip) / prev_c - 1)

                # Enter new position (today_open to today_close)
                df_new = data.get(new_etf)
                if df_new is not None and date in df_new.index:
                    today_o = df_new.loc[date, "Open"] if "Open" in df_new.columns else df_new.loc[date, "Close"]
                    buy = today_o * (1 + slip)
                    today_c = df_new.loc[date, "Close"]
                    if buy > 0:
                        dr += (today_c / buy - 1)

                in_leverage = should_leverage
                trades += 1
            else:
                # No switch: hold current position
                etf = leverage_etf if in_leverage else CASH_ETF
                df_e = data.get(etf)
                if df_e is not None and date in df_e.index:
                    si = df_e.index.get_loc(date)
                    if si > 0:
                        dr = df_e.iloc[si]["Close"] / df_e.iloc[si - 1]["Close"] - 1
        else:
            # Non-rebalance day: hold
            etf = leverage_etf if in_leverage else CASH_ETF
            df_e = data.get(etf)
            if df_e is not None and date in df_e.index:
                si = df_e.index.get_loc(date)
                if si > 0:
                    dr = df_e.iloc[si]["Close"] / df_e.iloc[si - 1]["Close"] - 1

        raw_rets.append(dr)

        # Vol-targeting
        if len(raw_rets) >= VOL_SCALE_LOOKBACK:
            realized_vol = np.std(raw_rets[-VOL_SCALE_LOOKBACK:]) * np.sqrt(252)
            if realized_vol > 0.01:
                exposure = np.clip(TARGET_VOL / realized_vol, VOL_FLOOR, VOL_CAP)
            else:
                exposure = VOL_CAP
            daily_rets.append(dr * exposure)
        else:
            daily_rets.append(dr)

    return pd.Series(daily_rets, index=dates), trades


def calc_metrics(rets, rf=0.02):
    if len(rets) == 0 or rets.std() == 0:
        return {"sharpe": 0, "cagr": 0, "max_dd": 0, "sortino": 0, "ann_vol": 0, "calmar": 0}
    excess = rets - rf / 252
    n_years = len(rets) / 252
    sharpe = excess.mean() / excess.std() * np.sqrt(252)
    cum = (1 + rets).cumprod()
    total = cum.iloc[-1] - 1
    cagr = (1 + total) ** (1 / n_years) - 1 if n_years >= 1 else total
    mdd = ((cum - cum.cummax()) / cum.cummax()).min()
    ds = excess[excess < 0]
    sortino = excess.mean() / ds.std() * np.sqrt(252) if len(ds) > 0 and ds.std() > 0 else 0
    ann_vol = rets.std() * np.sqrt(252)
    calmar = cagr / abs(mdd) if mdd != 0 else 0
    return {
        "sharpe": round(float(sharpe), 3), "cagr": round(float(cagr), 4),
        "max_dd": round(float(mdd), 4), "sortino": round(float(sortino), 3),
        "ann_vol": round(float(ann_vol), 4), "calmar": round(float(calmar), 3),
    }


if __name__ == "__main__":
    print("=" * 60)
    print("VALET: Vol-Adaptive Leveraged ETF Timing")
    print("=" * 60)
    data = download_etfs()
    print(f"Loaded {len(data)} ETFs")

    # Test with different leveraged ETFs
    for lev_etf in ["TQQQ", "SPXL", "SSO", "QLD"]:
        if lev_etf not in data:
            continue
        print(f"\n{'#'*60}")
        print(f"  TESTING WITH: {lev_etf}")
        print(f"{'#'*60}")

        for name, s, e in [("TRAIN", TRAIN_START, TRAIN_END), ("VALID", VALID_START, VALID_END),
                            ("TEST", TEST_START, TEST_END), ("FULL", "2012-01-01", TEST_END)]:
            rets, trades = run_strategy(data, lev_etf, s, e)
            met = calc_metrics(rets)
            spy = calc_metrics(data[BENCHMARK].loc[s:e, "Close"].pct_change().dropna())

            print(f"\n  {name}: Sharpe {met['sharpe']:>6.3f} (SPY {spy['sharpe']:.3f}) | "
                  f"CAGR {met['cagr']:>6.1%} | MaxDD {met['max_dd']:>7.1%} | "
                  f"Vol {met['ann_vol']:>5.1%} | Trades {trades}")

        # Walk-forward for this ETF
        print(f"\n  Walk-forward with {lev_etf}:")
        sharpes = []
        for year in range(2012, 2026):
            s, e = f"{year}-01-01", f"{year}-12-31"
            try:
                rets, _ = run_strategy(data, lev_etf, s, e)
                met = calc_metrics(rets)
                spy = calc_metrics(data[BENCHMARK].loc[s:e, "Close"].pct_change().dropna())
                beat = "✓" if met["sharpe"] > spy["sharpe"] else " "
                print(f"    {year}: Sharpe {met['sharpe']:>6.3f} vs SPY {spy['sharpe']:>6.3f} {beat} | "
                      f"CAGR {met['cagr']:>6.1%} | MaxDD {met['max_dd']:>7.1%}")
                sharpes.append(met["sharpe"])
            except Exception:
                pass
        if sharpes:
            pos = sum(1 for s in sharpes if s > 0)
            print(f"    Avg: {np.mean(sharpes):.3f} | Pos: {pos}/{len(sharpes)} | "
                  f"Min: {min(sharpes):.3f} | Max: {max(sharpes):.3f}")

    # Current state
    latest = data[BENCHMARK]["Close"].index[-1]
    spy_ret = data[BENCHMARK]["Close"].pct_change()
    vf = spy_ret.rolling(VOL_FAST_WINDOW).std().iloc[-1] * np.sqrt(252)
    vs = spy_ret.rolling(VOL_SLOW_WINDOW).std().iloc[-1] * np.sqrt(252)
    sma = data[BENCHMARK]["Close"].rolling(200).mean().iloc[-1]
    spy_now = data[BENCHMARK]["Close"].iloc[-1]

    print(f"\n{'='*60}")
    print(f"CURRENT STATE — {latest.date()}")
    print(f"  SPY: {spy_now:.2f} vs SMA200: {sma:.2f} → {'ABOVE ✓' if spy_now > sma else 'BELOW ✗'}")
    print(f"  Fast vol (5d):  {vf:.1%} → {'OK ✓' if vf < VOL_FAST_THRESH else 'HIGH ✗'}")
    print(f"  Slow vol (42d): {vs:.1%} → {'OK ✓' if vs < VOL_SLOW_THRESH else 'HIGH ✗'}")
    leverage_on = spy_now > sma and vf < VOL_FAST_THRESH and vs < VOL_SLOW_THRESH
    print(f"  POSITION: {'TQQQ (3x leverage)' if leverage_on else 'SHY (cash)'}")
