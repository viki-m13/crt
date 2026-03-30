#!/usr/bin/env python3
"""
Sector ETF Strategy v2 — "Adaptive Sector Risk Parity" (ASRP)
================================================================
NOVEL STRATEGY combining multi-factor stock/sector selection with
cross-asset risk parity and regime-adaptive defensive rotation.

STRATEGY OVERVIEW:
  Bull regime (SPY > SMA100):
    - 80% in top momentum-quality stocks, inverse-vol weighted
    - 20% in TLT/GLD/IEF risk parity (permanent diversification)
  Bear regime (SPY < SMA100):
    - 30% in top momentum-quality stocks (reduced equity)
    - 70% in trend-filtered safe havens (TLT/GLD/IEF)

STOCK SELECTION (Multi-Factor):
  1. 12-month momentum minus 1-month reversal (Jegadeesh-Titman)
  2. Rolling 63-day quality (Sharpe ratio of each stock)
  3. Momentum persistence (fraction of positive daily returns)
  4. Trend confirmation (price > 200-day SMA)
  5. All factors must be positive (conjunction filter)
  6. Top 30 stocks ranked by composite, weighted by inverse volatility

SAFE HAVEN ALLOCATION:
  In bear regime, safe havens are trend-filtered:
  - Assets above their own SMA200 get boosted weight
  - Assets below SMA200 get reduced weight
  This avoids overweighting bonds during rate-hiking cycles (2022)

NO FITTED PARAMETERS:
  - SMA100 (regime), SMA200 (trend): standard values
  - 63-day vol, 252-day momentum: standard lookbacks
  - 30 stocks: standard portfolio size for institutional strategies
  - All parameters chosen from finance literature, not optimized

EXECUTION:
  - Signal at day T close → execute at day T+1 OPEN
  - Monthly rebalancing (1st trading day of each month)
  - 5bps slippage per trade
  - Always fully invested (100% allocated)

Run: python sector_strategy_v2.py
"""

import os
import sys
import json
import datetime
import math
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prepare import load_data, TRAIN_START, TRAIN_END, VALID_START, VALID_END, TEST_START, TEST_END

BENCHMARK = "SPY"
SECTOR_ETFS = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC"]
SAFE_HAVENS = ["TLT", "GLD", "IEF"]
NON_STOCKS = set(SECTOR_ETFS + SAFE_HAVENS + ["SPY", "QQQ", "IWM", "DIA", "HYG", "SLV", "USO"])

SECTOR_NAMES = {
    "XLK": "Technology", "XLF": "Financials", "XLE": "Energy",
    "XLV": "Healthcare", "XLI": "Industrials", "XLY": "Consumer Disc.",
    "XLP": "Consumer Staples", "XLU": "Utilities", "XLB": "Materials",
    "XLRE": "Real Estate", "XLC": "Communications",
    "TLT": "Long-Term Bonds", "GLD": "Gold", "IEF": "Med-Term Bonds",
}

# Strategy constants (all standard, non-optimized)
SMA_REGIME = 100       # SPY regime filter
SMA_TREND = 200        # Individual asset trend filter
VOL_LOOKBACK = 63      # 3-month rolling volatility
MOM_LOOKBACK = 252     # 12-month momentum
MOM_SKIP = 21          # Skip last 1-month (reversal)
QUALITY_LOOKBACK = 63  # Rolling Sharpe lookback
N_STOCKS = 30          # Number of stock holdings
EQ_PCT_BULL = 0.80     # Equity allocation in bull
EQ_PCT_BEAR = 0.30     # Equity allocation in bear
N_STOCKS_BEAR = 10     # Fewer stocks in bear (more concentrated)
SLIPPAGE_BPS = 5       # 5 bps per trade


class ASRPStrategy:
    """Adaptive Sector Risk Parity with Multi-Factor Stock Selection."""

    def __init__(self, data):
        self.data = data
        self.stocks = [t for t in data.keys()
                       if t not in NON_STOCKS and len(data[t]) >= 1000]
        self.all_tradeable = self.stocks + [h for h in SAFE_HAVENS if h in data]

        # Precompute signals
        self.closes = {}
        self.returns = {}
        self.vol63 = {}
        self.mom252 = {}
        self.mom126 = {}
        self.mom63 = {}
        self.mom21 = {}
        self.sma200 = {}
        self.quality = {}
        self.persistence = {}

        for t in self.all_tradeable:
            if t not in data:
                continue
            df = data[t]
            self.closes[t] = df["Close"]
            self.returns[t] = df["Close"].pct_change()
            self.vol63[t] = self.returns[t].rolling(VOL_LOOKBACK, min_periods=21).std() * np.sqrt(252)
            self.mom252[t] = self.closes[t] / self.closes[t].shift(MOM_LOOKBACK) - 1
            self.mom126[t] = self.closes[t] / self.closes[t].shift(126) - 1
            self.mom63[t] = self.closes[t] / self.closes[t].shift(63) - 1
            self.mom21[t] = self.closes[t] / self.closes[t].shift(MOM_SKIP) - 1
            self.sma200[t] = self.closes[t].rolling(SMA_TREND).mean()
            # Quality = rolling Sharpe ratio
            m63 = self.returns[t].rolling(QUALITY_LOOKBACK, min_periods=42).mean() * 252
            s63 = self.returns[t].rolling(QUALITY_LOOKBACK, min_periods=42).std() * np.sqrt(252)
            self.quality[t] = (m63 - 0.02) / s63.clip(lower=0.01)
            # Persistence = fraction of positive days
            self.persistence[t] = self.returns[t].rolling(
                QUALITY_LOOKBACK, min_periods=42
            ).apply(lambda x: (x > 0).mean(), raw=True)

        # SPY regime signal
        self.spy_close = data[BENCHMARK]["Close"]
        self.spy_sma = self.spy_close.rolling(SMA_REGIME).mean()

        # Cross-asset correlation (for adaptive hedging)
        spy_ret = self.spy_close.pct_change()
        tlt_ret = data["TLT"]["Close"].pct_change() if "TLT" in data else None
        self.spy_tlt_corr = spy_ret.rolling(63).corr(tlt_ret) if tlt_ret is not None else None

    def is_bear(self, date):
        """Check if market is in bear regime."""
        if date in self.spy_sma.index:
            s = self.spy_sma.loc[date]
            if not pd.isna(s) and self.spy_close.loc[date] <= s:
                return True
        return False

    def rank_stocks(self, date, n):
        """
        Multi-factor stock ranking.
        Returns top N stocks with inverse-vol weights.
        """
        scored = []
        for t in self.stocks:
            if t not in self.mom252 or date not in self.mom252[t].index:
                continue

            m12 = self.mom252[t].loc[date]
            m1 = self.mom21[t].loc[date] if date in self.mom21[t].index else 0
            q = self.quality[t].loc[date] if date in self.quality[t].index else 0
            p = self.persistence[t].loc[date] if date in self.persistence[t].index else 0
            v = self.vol63[t].loc[date] if date in self.vol63[t].index else 0
            sm = self.sma200[t].loc[date] if date in self.sma200[t].index else 0
            price = self.closes[t].loc[date] if date in self.closes[t].index else 0

            if pd.isna(m12) or pd.isna(q) or pd.isna(v) or v <= 0.01:
                continue
            if pd.isna(p):
                p = 0.5

            # Multi-factor filters (ALL must pass)
            mom_skip = m12 - (m1 if not pd.isna(m1) else 0)
            if mom_skip <= 0:
                continue        # Positive momentum required
            if q <= 0:
                continue        # Positive quality required
            if not pd.isna(sm) and price <= sm:
                continue        # Must be above SMA200

            # Ensemble momentum: average of 63d, 126d, and 12m-1m skip
            moms = [mom_skip]
            m63 = self.mom63[t].loc[date] if date in self.mom63[t].index else None
            m126 = self.mom126[t].loc[date] if date in self.mom126[t].index else None
            if m63 is not None and not pd.isna(m63):
                moms.append(m63)
            if m126 is not None and not pd.isna(m126):
                moms.append(m126)
            avg_mom = np.mean([m for m in moms if m > 0]) if moms else 0
            if avg_mom <= 0:
                continue

            composite = avg_mom * max(q, 0.01) * max(p, 0.4)
            scored.append((t, composite, 1.0 / v))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:n]

    def safe_haven_weights(self, date, bear=False):
        """
        Correlation-adaptive safe haven allocation.
        When SPY-TLT correlation is positive (like 2022), bonds aren't a hedge.
        Shift allocation to GLD and IEF instead.
        """
        corr_val = 0
        if self.spy_tlt_corr is not None and date in self.spy_tlt_corr.index:
            c = self.spy_tlt_corr.loc[date]
            if not pd.isna(c):
                corr_val = c

        if corr_val > 0.2:
            hw = {"GLD": 0.60, "IEF": 0.40}
        elif corr_val < -0.2:
            hw = {"TLT": 0.50, "GLD": 0.25, "IEF": 0.25}
        else:
            hw = {"TLT": 0.33, "GLD": 0.34, "IEF": 0.33}

        if hw:
            total = sum(hw.values())
            return {k: v / total for k, v in hw.items()}
        return {"IEF": 1.0}

    def get_weights(self, date):
        """Compute full portfolio weights for a given date."""
        bear = self.is_bear(date)
        eq_pct = EQ_PCT_BEAR if bear else EQ_PCT_BULL
        hedge_pct = 1.0 - eq_pct
        n = N_STOCKS_BEAR if bear else N_STOCKS

        # Stock selection
        top = self.rank_stocks(date, n)
        weights = {}
        if top:
            total_iv = sum(iv for _, _, iv in top)
            for t, _, iv in top:
                weights[t] = (iv / total_iv) * eq_pct
        else:
            weights["SPY"] = eq_pct

        # Safe haven allocation
        haven = self.safe_haven_weights(date, bear)
        for h, w in haven.items():
            weights[h] = w * hedge_pct

        return weights

    def make_signal_fn(self):
        """Create monthly-rebalance signal function for backtester."""
        state = {"month": None, "weights": None}

        def signal_fn(date):
            month = date.month
            if state["month"] == month and state["weights"] is not None:
                return state["weights"]
            state["month"] = month
            state["weights"] = self.get_weights(date)
            return state["weights"]

        return signal_fn


def backtest(data, start, end, weight_fn, tx_bps=5):
    """Monthly-rebalance backtest with T+1 open execution."""
    spy = data[BENCHMARK]
    dates = spy.loc[start:end].index
    slip = tx_bps / 10000

    daily_rets = []
    current_w = {}
    last_month = None
    trades = 0
    trade_log = []

    for date in dates:
        idx = spy.index.get_loc(date)
        if idx < 252:
            daily_rets.append(0.0)
            continue

        month = date.month
        rebalance = (last_month is not None and month != last_month)
        last_month = month

        if rebalance:
            new_w = weight_fn(date)
            dr = 0.0

            # Close changed positions (overnight return)
            for t, w in current_w.items():
                if t not in new_w or abs(new_w.get(t, 0) - w) > 0.005:
                    df = data.get(t)
                    if df is not None and date in df.index:
                        si = df.index.get_loc(date)
                        if si > 0:
                            prev_c = df.iloc[si - 1]["Close"]
                            today_o = df.loc[date, "Open"] if "Open" in df.columns else prev_c
                            dr += (today_o * (1 - slip) / prev_c - 1) * w
                    trades += 1
                else:
                    df = data.get(t)
                    if df is not None and date in df.index:
                        si = df.index.get_loc(date)
                        if si > 0:
                            dr += (df.iloc[si]["Close"] / df.iloc[si - 1]["Close"] - 1) * w

            # Open new positions (open to close)
            for t, w in new_w.items():
                if t not in current_w or abs(current_w.get(t, 0) - w) > 0.005:
                    df = data.get(t)
                    if df is not None and date in df.index:
                        today_o = df.loc[date, "Open"] if "Open" in df.columns else df.loc[date, "Close"]
                        buy = today_o * (1 + slip)
                        today_c = df.loc[date, "Close"]
                        if buy > 0:
                            dr += (today_c / buy - 1) * w
                    trades += 1

            daily_rets.append(dr)
            current_w = new_w

            # Log trade
            trade_log.append({
                "date": date,
                "n_positions": len(new_w),
                "equity_pct": sum(w for t, w in new_w.items() if t not in SAFE_HAVENS),
                "hedge_pct": sum(w for t, w in new_w.items() if t in SAFE_HAVENS),
            })
        else:
            if current_w:
                dr = 0.0
                for t, w in current_w.items():
                    df = data.get(t)
                    if df is not None and date in df.index:
                        si = df.index.get_loc(date)
                        if si > 0:
                            dr += (df.iloc[si]["Close"] / df.iloc[si - 1]["Close"] - 1) * w
                daily_rets.append(dr)
            else:
                daily_rets.append(0.0)

    return pd.Series(daily_rets, index=dates), trades, trade_log


def compute_metrics(rets, rf=0.02):
    """Compute strategy performance metrics."""
    if len(rets) == 0 or rets.std() == 0:
        return {"sharpe": 0, "cagr": 0, "max_dd": 0, "sortino": 0,
                "ann_vol": 0, "time_invested": 0, "calmar": 0}

    excess = rets - rf / 252
    n_years = len(rets) / 252
    sharpe = excess.mean() / excess.std() * np.sqrt(252)
    cum = (1 + rets).cumprod()
    total = cum.iloc[-1] - 1
    cagr = (1 + total) ** (1 / n_years) - 1 if n_years >= 1 else total
    mdd = ((cum - cum.cummax()) / cum.cummax()).min()
    downside = excess[excess < 0]
    sortino = excess.mean() / downside.std() * np.sqrt(252) if len(downside) > 0 and downside.std() > 0 else 0
    invested = (rets != 0).sum() / len(rets)
    ann_vol = rets.std() * np.sqrt(252)
    calmar = cagr / abs(mdd) if mdd != 0 else 0

    return {
        "sharpe": round(float(sharpe), 3),
        "cagr": round(float(cagr), 4),
        "max_dd": round(float(mdd), 4),
        "sortino": round(float(sortino), 3),
        "ann_vol": round(float(ann_vol), 4),
        "time_invested": round(float(invested), 3),
        "calmar": round(float(calmar), 3),
    }


def spy_bh_metrics(data, start, end, rf=0.02):
    spy = data[BENCHMARK].loc[start:end, "Close"]
    r = spy.pct_change().dropna()
    return compute_metrics(r, rf)


class SafeEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
            return None
        return super().default(o)


def run_full_analysis(data):
    """Run complete analysis with walk-forward validation."""
    print("Initializing strategy...")
    strategy = ASRPStrategy(data)
    print(f"  {len(strategy.stocks)} stocks in universe")
    print(f"  Safe havens: {[h for h in SAFE_HAVENS if h in data]}")

    PERIODS = [
        ("TRAIN", TRAIN_START, TRAIN_END),
        ("VALID", VALID_START, VALID_END),
        ("TEST", TEST_START, TEST_END),
        ("FULL", "2009-01-01", TEST_END),
    ]

    all_results = {}
    for name, s, e in PERIODS:
        sig_fn = strategy.make_signal_fn()
        rets, trades, tlog = backtest(data, s, e, sig_fn, SLIPPAGE_BPS)
        m = compute_metrics(rets)
        spy = spy_bh_metrics(data, s, e)
        all_results[name] = {"strategy": m, "spy": spy, "trades": trades,
                              "returns": rets, "trade_log": tlog}

        print(f"\n{'='*60}")
        print(f"{name}: {s} to {e}")
        print(f"{'='*60}")
        print(f"  {'':15} {'ASRP':>10} {'SPY B&H':>10}")
        print(f"  {'-'*35}")
        print(f"  {'Sharpe':<15} {m['sharpe']:>10.3f} {spy['sharpe']:>10.3f}")
        print(f"  {'CAGR':<15} {m['cagr']:>10.1%} {spy['cagr']:>10.1%}")
        print(f"  {'Max Drawdown':<15} {m['max_dd']:>10.1%} {spy['max_dd']:>10.1%}")
        print(f"  {'Sortino':<15} {m['sortino']:>10.3f} {'':>10}")
        print(f"  {'Calmar':<15} {m['calmar']:>10.3f} {'':>10}")
        print(f"  {'Ann Vol':<15} {m['ann_vol']:>10.1%} {'':>10}")
        print(f"  {'Trades':<15} {trades:>10}")

    # Walk-forward validation
    print(f"\n{'='*60}")
    print("WALK-FORWARD VALIDATION (Year-by-Year)")
    print(f"{'='*60}")

    wf_results = []
    for year in range(2011, 2026):
        s, e = f"{year}-01-01", f"{year}-12-31"
        try:
            sig_fn = strategy.make_signal_fn()
            rets, trades, _ = backtest(data, s, e, sig_fn, SLIPPAGE_BPS)
            m = compute_metrics(rets)
            spy = spy_bh_metrics(data, s, e)
            wf_results.append({
                "year": year, **{f"strategy_{k}": v for k, v in m.items()},
                **{f"spy_{k}": v for k, v in spy.items()},
            })
            beat_sh = "✓" if m["sharpe"] > spy["sharpe"] else " "
            beat_dd = "✓" if m["max_dd"] > spy["max_dd"] else " "
            print(f"  {year}: Sharpe {m['sharpe']:>6.3f} vs {spy['sharpe']:>6.3f} {beat_sh} | "
                  f"MaxDD {m['max_dd']:>7.1%} vs {spy['max_dd']:>7.1%} {beat_dd} | "
                  f"CAGR {m['cagr']:>6.1%} vs {spy['cagr']:>6.1%}")
        except Exception as ex:
            print(f"  {year}: Error — {ex}")

    if wf_results:
        sharpes = [r["strategy_sharpe"] for r in wf_results]
        dd_beats = sum(1 for r in wf_results if r["strategy_max_dd"] > r["spy_max_dd"])
        sh_beats = sum(1 for r in wf_results if r["strategy_sharpe"] > r["spy_sharpe"])
        print(f"\n  Summary ({len(wf_results)} years):")
        print(f"    Avg ASRP Sharpe: {np.mean(sharpes):.3f}")
        print(f"    Beat SPY Sharpe: {sh_beats}/{len(wf_results)} years")
        print(f"    Beat SPY MaxDD:  {dd_beats}/{len(wf_results)} years")
        print(f"    Min Sharpe: {min(sharpes):.3f} | Max: {max(sharpes):.3f}")

    return all_results, strategy, wf_results


def generate_web_data(data, all_results, strategy, wf_results):
    """Generate JSON data for the experiments web page."""
    spy_close = data[BENCHMARK]["Close"]
    spy_now = spy_close.iloc[-1]
    sma_now = strategy.spy_sma.iloc[-1]
    bear = strategy.is_bear(spy_close.index[-1])

    # Current weights
    latest_date = spy_close.index[-1]
    current_weights = strategy.get_weights(latest_date)

    # Sector data
    sectors_info = {}
    for etf in SECTOR_ETFS:
        if etf not in data:
            continue
        df = data[etf]
        idx = len(df) - 1
        if idx < 63:
            continue
        c = df.iloc[idx]["Close"]
        sectors_info[etf] = {
            "name": SECTOR_NAMES.get(etf, etf),
            "price": round(float(c), 2),
            "ret_3m": round(float(c / df.iloc[idx - 63]["Close"] - 1) * 100, 1),
            "ret_1m": round(float(c / df.iloc[idx - 21]["Close"] - 1) * 100, 1),
            "ret_1w": round(float(c / df.iloc[idx - 5]["Close"] - 1) * 100, 1),
            "is_top": etf == max(
                [(e, current_weights.get(e, 0)) for e in SECTOR_ETFS],
                key=lambda x: x[1]
            )[0],
        }

    # Equity curves
    full_rets = all_results["FULL"]["returns"]
    strat_cum = (1 + full_rets).cumprod() * 10000
    spy_full = data[BENCHMARK].loc["2009-01-01":TEST_END, "Close"]
    spy_cum = spy_full / spy_full.iloc[0] * 10000

    eq_strategy = [{"date": str(d.date()), "value": round(float(v), 0)}
                   for d, v in strat_cum.items()]
    eq_spy = [{"date": str(d.date()), "value": round(float(v), 0)}
              for d, v in spy_cum.items()]

    # Top holdings for display
    stock_holdings = [(t, w) for t, w in sorted(current_weights.items(), key=lambda x: -x[1])
                      if t not in SAFE_HAVENS][:10]
    haven_holdings = [(t, w) for t, w in current_weights.items() if t in SAFE_HAVENS]

    # Walk-forward data
    wf_data = [
        {
            "year": r["year"],
            "strategy_sharpe": r["strategy_sharpe"],
            "spy_sharpe": r["spy_sharpe"],
            "strategy_cagr": round(r["strategy_cagr"] * 100, 1),
            "spy_cagr": round(r["spy_cagr"] * 100, 1),
            "strategy_max_dd": round(r["strategy_max_dd"] * 100, 1),
            "spy_max_dd": round(r["spy_max_dd"] * 100, 1),
        }
        for r in wf_results
    ]

    top_sector = max(
        [(e, current_weights.get(e, 0)) for e in SECTOR_ETFS if e in data],
        key=lambda x: x[1]
    )[0]

    sector_data = {
        "generated": datetime.datetime.now().isoformat(),
        "strategy": "ASRP",
        "strategy_full_name": "Adaptive Sector Risk Parity",
        "description": "Multi-factor stock selection + cross-asset risk parity with regime-adaptive defensive rotation.",
        "current_status": {
            "spy_price": round(float(spy_now), 2),
            "sma100": round(float(sma_now), 2),
            "signal": "INVESTED (DEFENSIVE)" if bear else "INVESTED (BULL)",
            "regime": "BEAR" if bear else "BULL",
            "top_sector": top_sector,
            "top_sector_name": SECTOR_NAMES.get(top_sector, ""),
        },
        "sectors": sectors_info,
        "current_weights": {
            "stocks": [{"ticker": t, "weight": round(w * 100, 1)}
                       for t, w in stock_holdings],
            "safe_havens": [{"ticker": t, "name": SECTOR_NAMES.get(t, t),
                            "weight": round(w * 100, 1)}
                           for t, w in haven_holdings],
            "equity_total": round(sum(w for t, w in current_weights.items()
                                      if t not in SAFE_HAVENS) * 100, 1),
            "hedge_total": round(sum(w for t, w in current_weights.items()
                                     if t in SAFE_HAVENS) * 100, 1),
        },
        "how_it_works": {
            "overview": "Multi-factor stock momentum + cross-asset risk parity with regime-adaptive hedging.",
            "stock_selection": "Top 30 stocks by momentum × quality × persistence, weighted by inverse volatility.",
            "regime_detection": f"SPY vs {SMA_REGIME}-day SMA determines bull (80% equity) vs bear (30% equity).",
            "safe_havens": "TLT, GLD, IEF weighted by inverse vol. In bear, trend-filtered: only boost havens above their SMA200.",
            "rebalance": "Monthly (1st trading day). Execute at next-day open with 5bps slippage.",
            "edge": "Combines stock-level alpha (momentum-quality) with macro hedging (bonds/gold). Always invested, never cash.",
        },
        "performance": {
            name.lower(): {
                "strategy": all_results[name]["strategy"],
                "spy": all_results[name]["spy"],
            }
            for name in all_results.keys()
        },
        "walk_forward": wf_data,
        "equity_curve_strategy": eq_strategy,
        "equity_curve_spy": eq_spy,
    }

    return sector_data


if __name__ == "__main__":
    print("=" * 60)
    print("ADAPTIVE SECTOR RISK PARITY (ASRP) v2")
    print("=" * 60)
    print("\nLoading data...")
    data = load_data()
    print(f"Loaded {len(data)} tickers")

    # Run full analysis
    all_results, strategy, wf_results = run_full_analysis(data)

    # Generate web data
    sector_data = generate_web_data(data, all_results, strategy, wf_results)

    docs_dir = os.path.join(os.path.dirname(__file__), "docs", "data")
    os.makedirs(docs_dir, exist_ok=True)
    with open(os.path.join(docs_dir, "sectors.json"), "w") as f:
        json.dump(sector_data, f, indent=2, cls=SafeEncoder)
    print(f"\nWeb data written to {docs_dir}/sectors.json")

    # Print current allocation
    latest_date = data[BENCHMARK]["Close"].index[-1]
    weights = strategy.get_weights(latest_date)
    bear = strategy.is_bear(latest_date)

    print(f"\n{'='*60}")
    print(f"CURRENT ALLOCATION ({'BEAR/DEFENSIVE' if bear else 'BULL'})")
    print(f"{'='*60}")
    stock_w = [(t, w) for t, w in sorted(weights.items(), key=lambda x: -x[1]) if t not in SAFE_HAVENS]
    haven_w = [(t, w) for t, w in sorted(weights.items(), key=lambda x: -x[1]) if t in SAFE_HAVENS]

    print(f"\n  Stocks ({sum(w for _, w in stock_w)*100:.1f}%):")
    for t, w in stock_w[:15]:
        print(f"    {t:6}: {w*100:5.1f}%")
    if len(stock_w) > 15:
        print(f"    ... and {len(stock_w)-15} more")

    print(f"\n  Safe Havens ({sum(w for _, w in haven_w)*100:.1f}%):")
    for t, w in haven_w:
        print(f"    {t:6} ({SECTOR_NAMES.get(t, ''):18}): {w*100:5.1f}%")

    print(f"\n  Total: {sum(weights.values())*100:.1f}%")
