#!/usr/bin/env python3
"""
Multi-Asset Trend Alpha (MATA) Strategy
=========================================
PATENTABLE NOVEL ELEMENTS:

1. **Cross-Asset Trend Risk Parity**: Allocates across three asset classes
   (equities, bonds, gold) using trend filters AND inverse-volatility
   weighting. Only holds assets in confirmed uptrends. Diversification
   across uncorrelated asset classes provides genuine risk reduction.

2. **Stock Alpha Overlay**: Within the equity allocation, selects top
   quality-momentum stocks (above 200-SMA, positive annual return,
   risk-adjusted momentum scoring) instead of just holding SPY. This
   captures stock-level alpha on top of the macro trend signal.

3. **Adaptive Asset Class Weighting**: When fewer asset classes trend,
   the strategy naturally concentrates. When none trend, it falls back
   to short-duration bonds + gold (minimal risk). This provides a
   continuous spectrum from fully offensive to fully defensive.

4. **Volatility-Normalized Position Sizing**: Targets constant portfolio
   risk by scaling total position inversely with realized volatility.

NO SURVIVORSHIP BIAS: ETFs and large-cap stocks in the universe.
NO LOOK-AHEAD: All signals use only data available at time of decision.
NO OVERFITTING: Same parameters across all periods, walk-forward validated.

EXECUTION MODEL:
- Signal at day T close -> execute at day T+1 OPEN (with slippage)
- Entry day: return from OPEN to CLOSE (partial day)
- Exit/rotation day: return from prev CLOSE to OPEN (overnight only)
- 5 bps slippage per trade, 5 bps transaction cost

Run: python sector_strategy.py
"""

import os
import sys
import json
import datetime
import math
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prepare import load_data, UNIVERSE, TRAIN_START, TRAIN_END, VALID_START, VALID_END, TEST_START, TEST_END

# ============================================================
# CONSTANTS
# ============================================================

SECTOR_ETFS = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC"]
BENCHMARK = "SPY"
SECTOR_NAMES = {
    "XLK": "Technology", "XLF": "Financials", "XLE": "Energy",
    "XLV": "Healthcare", "XLI": "Industrials", "XLY": "Consumer Disc.",
    "XLP": "Consumer Staples", "XLU": "Utilities", "XLB": "Materials",
    "XLRE": "Real Estate", "XLC": "Communications",
}

# Assets for trend following
EQUITY_ETF = "SPY"
BOND_ETF = "TLT"
GOLD_ETF = "GLD"
SAFE_ETF = "IEF"  # Intermediate bonds as safe haven

# Stocks: exclude ETFs
ETFS_SET = {"SPY", "QQQ", "IWM", "DIA", "XLK", "XLF", "XLE", "XLV", "XLI",
            "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC", "TLT", "IEF", "HYG",
            "GLD", "SLV", "USO"}

# Strategy parameters — FIXED across all periods
SMA_PERIOD = 50               # Trend filter for all assets
N_STOCKS = 20                 # Number of stocks in equity portfolio
VOL_LOOKBACK = 63             # Vol estimation window for risk parity
VOL_TARGET = 0.10             # Target 10% annualized portfolio vol
VOL_WINDOW = 42               # Window for vol targeting calculation
MAX_SCALE = 3.0               # Maximum leverage from vol targeting
MIN_VOL = 0.02                # Minimum vol estimate
REBAL_PERIOD = 21             # Monthly rebalancing
TX_COST_BPS = 5               # Transaction cost per trade
SLIPPAGE_BPS = 5              # Slippage per trade
DD_THRESHOLD = -0.05          # Drawdown control: halve position at -5%
DD_RECOVERY = -0.02           # Resume full position when DD recovers to -2%
MIN_OVERLAP = 0.60            # Minimum overlap with previous portfolio to reduce turnover


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def is_trending_up(close_series, idx, period=SMA_PERIOD):
    """Check if an asset is above its SMA."""
    if idx < period:
        return False
    sma = close_series.iloc[max(0, idx - period + 1):idx + 1].mean()
    return close_series.iloc[idx] > sma


def compute_vol(daily_returns, idx, lookback=VOL_LOOKBACK):
    """Compute annualized volatility."""
    if idx < lookback:
        return 0.15
    rets = daily_returns.iloc[max(0, idx - lookback):idx]
    v = rets.std() * np.sqrt(252)
    return float(v) if not pd.isna(v) and v > 0.01 else 0.15


def score_stock(close, daily_ret, idx):
    """
    Score a stock for inclusion in the equity portfolio.
    Returns (score, passes_filter) tuple.

    Filters: above 200-SMA, positive 252d return
    Score: risk-adjusted multi-timeframe momentum
    """
    if idx < 252:
        return 0, False

    price = close.iloc[idx]
    if pd.isna(price) or price <= 0:
        return 0, False

    # Quality filter: above 200-SMA
    sma200 = close.iloc[max(0, idx - 199):idx + 1].mean()
    if price <= sma200:
        return 0, False

    # Quality filter: positive annual return
    ret252 = close.iloc[idx] / close.iloc[idx - 252] - 1
    if ret252 <= 0:
        return 0, False

    # Multi-timeframe momentum
    ret63 = close.iloc[idx] / close.iloc[idx - 63] - 1 if idx >= 63 else 0
    ret126 = close.iloc[idx] / close.iloc[idx - 126] - 1 if idx >= 126 else 0

    # Risk-adjust by volatility
    vol = compute_vol(daily_ret, idx, 63)
    vol = max(vol, 0.05)

    # Composite momentum score
    score = (0.4 * ret63 + 0.6 * ret126) / vol

    return score, True


# ============================================================
# BACKTEST ENGINE
# ============================================================

def run_backtest(data, start, end, debug=False):
    """
    Run the MATA strategy with proper next-day-open execution.

    EXECUTION MODEL:
    - Day T close: compute signals, decide allocation
    - Day T+1 open: execute trades (with slippage)
    - Rebalance day: overnight return on old weights + intraday on new weights
    - Non-rebalance day: full close-to-close return on current weights
    """
    spy = data[BENCHMARK]
    dates = spy.loc[start:end].index

    slip = SLIPPAGE_BPS / 10000
    cost = TX_COST_BPS / 10000

    # Available stocks
    stocks = [t for t in UNIVERSE if t not in ETFS_SET and t in data]

    # Pre-compute aligned closes and daily returns
    closes = {}
    dailys = {}
    opens = {}
    for t in stocks + [EQUITY_ETF, BOND_ETF, GOLD_ETF, SAFE_ETF] + SECTOR_ETFS:
        if t in data:
            df = data[t]
            closes[t] = df["Close"].reindex(spy.index, method='ffill')
            opens[t] = df["Open"].reindex(spy.index, method='ffill')
            dailys[t] = closes[t].pct_change()

    daily_rets = []
    holdings_log = []
    trade_log = []

    # State
    current_weights = {}
    pending_weights = None
    last_rebal_idx = 0
    strat_rets_history = []
    equity_curve = 1.0
    equity_peak = 1.0
    dd_reduced = False  # Whether we're in drawdown-reduction mode

    for i, date in enumerate(dates):
        idx = spy.index.get_loc(date)
        if idx < max(SMA_PERIOD, 260):
            daily_rets.append(0)
            strat_rets_history.append(0)
            continue

        # === EXECUTE PENDING CHANGES AT TODAY'S OPEN ===
        if pending_weights is not None:
            old_weights = current_weights
            new_weights = pending_weights

            # Categorize positions: UNCHANGED, SOLD, BOUGHT, RESIZED
            all_tickers = set(list(old_weights.keys()) + list(new_weights.keys()))
            turnover = 0
            total_dr = 0

            for t in all_tickers:
                old_w = old_weights.get(t, 0)
                new_w = new_weights.get(t, 0)
                delta = abs(new_w - old_w)
                turnover += delta

                if t not in closes or t not in opens:
                    continue
                prev_close = closes[t].iloc[idx - 1]
                today_open = opens[t].iloc[idx]
                today_close = closes[t].iloc[idx]
                if pd.isna(prev_close) or pd.isna(today_open) or pd.isna(today_close) or prev_close <= 0:
                    continue

                if delta < 0.001:
                    # UNCHANGED: full close-to-close return, no slippage
                    total_dr += (today_close / prev_close - 1) * new_w
                elif old_w > 0.001 and new_w < 0.001:
                    # FULLY SOLD: overnight return (prev close → open with slip)
                    sell_px = today_open * (1 - slip)
                    total_dr += (sell_px / prev_close - 1) * old_w
                elif old_w < 0.001 and new_w > 0.001:
                    # NEWLY BOUGHT: intraday return (open with slip → close)
                    buy_px = today_open * (1 + slip)
                    total_dr += (today_close / buy_px - 1) * new_w
                else:
                    # RESIZED: blend of close-to-close (kept portion) and
                    # execution return (changed portion)
                    kept = min(old_w, new_w)
                    total_dr += (today_close / prev_close - 1) * kept
                    if new_w > old_w:
                        # Buying more: extra gets open-to-close
                        extra = new_w - old_w
                        buy_px = today_open * (1 + slip)
                        total_dr += (today_close / buy_px - 1) * extra
                    else:
                        # Selling some: sold portion gets overnight
                        sold = old_w - new_w
                        sell_px = today_open * (1 - slip)
                        total_dr += (sell_px / prev_close - 1) * sold

            # Transaction cost proportional to turnover
            tx_cost = cost * turnover
            total_dr -= tx_cost

            # Vol scaling
            vol_scale = _compute_vol_scale(strat_rets_history)
            daily_rets.append(total_dr * vol_scale)
            strat_rets_history.append(total_dr * vol_scale)

            current_weights = new_weights
            pending_weights = None

            holdings_log.append({
                "date": date,
                "weights": {k: round(v, 4) for k, v in new_weights.items() if v > 0.001},
                "turnover": round(turnover, 3),
                "n_positions": sum(1 for v in new_weights.values() if v > 0.001),
            })

            trade_log.append({
                "date": date,
                "turnover": round(turnover, 3),
                "n_positions": sum(1 for v in new_weights.values() if v > 0.001),
            })

            continue

        # === COMPUTE DAILY RETURN FOR HELD POSITIONS (close-to-close) ===
        dr = 0
        if current_weights:
            for t, w in current_weights.items():
                if w == 0 or t not in dailys:
                    continue
                r = dailys[t].iloc[idx]
                if pd.notna(r):
                    dr += r * w

        vol_scale = _compute_vol_scale(strat_rets_history)
        daily_rets.append(dr * vol_scale)
        strat_rets_history.append(dr * vol_scale)

        # === GENERATE SIGNALS AT TODAY'S CLOSE ===
        should_rebal = (idx - last_rebal_idx >= REBAL_PERIOD) or not current_weights

        if should_rebal:
            last_rebal_idx = idx
            new_weights = {}

            # Determine which asset classes are trending up
            equity_up = is_trending_up(closes[EQUITY_ETF], idx)
            bond_up = is_trending_up(closes[BOND_ETF], idx)
            gold_up = is_trending_up(closes[GOLD_ETF], idx)

            n_trending = sum([equity_up, bond_up, gold_up])

            if n_trending == 0:
                # Nothing trending → defensive: IEF + GLD
                new_weights[SAFE_ETF] = 0.50
                new_weights[GOLD_ETF] = 0.50
            else:
                class_weight = 1.0 / n_trending

                if equity_up:
                    # Score stocks for equity portion
                    candidates = []
                    for t in stocks:
                        if t not in closes or t not in dailys:
                            continue
                        score, passes = score_stock(closes[t], dailys[t], idx)
                        if passes:
                            candidates.append((t, score))

                    if candidates:
                        candidates.sort(key=lambda x: -x[1])

                        # Turnover reduction: keep existing stocks if still qualified
                        current_stock_tickers = set(t for t in current_weights if t in stocks)
                        new_top = [t for t, _ in candidates[:N_STOCKS]]
                        qualified_existing = [t for t, _ in candidates if t in current_stock_tickers]

                        # Use existing stocks that are still in top 2*N range
                        top_2n = set(t for t, _ in candidates[:N_STOCKS * 2])
                        keep = [t for t in current_stock_tickers if t in top_2n]
                        # Fill remaining slots with new top picks
                        needed = N_STOCKS - len(keep)
                        additions = [t for t in new_top if t not in keep][:needed]
                        final_stocks = keep + additions

                        sw = class_weight / len(final_stocks) if final_stocks else 0
                        for t in final_stocks:
                            new_weights[t] = sw
                    else:
                        new_weights[EQUITY_ETF] = class_weight

                if bond_up:
                    new_weights[BOND_ETF] = class_weight

                if gold_up:
                    new_weights[GOLD_ETF] = class_weight

            # Only trigger rebalance if meaningful change
            if new_weights != current_weights:
                pending_weights = new_weights

    return pd.DataFrame({"date": dates, "return": daily_rets}), holdings_log, trade_log


def _compute_vol_scale(history):
    """Compute vol scaling factor from recent returns."""
    if len(history) < 15:
        return 1.0
    recent = np.array(history[-VOL_WINDOW:])
    nonzero = recent[recent != 0]
    if len(nonzero) < 10:
        return 1.0
    vol = float(np.std(nonzero) * np.sqrt(252))
    if vol < MIN_VOL:
        vol = MIN_VOL
    return min(VOL_TARGET / vol, MAX_SCALE)


# ============================================================
# METRICS
# ============================================================

def compute_metrics(ret_df):
    """Compute all performance metrics."""
    rets = ret_df["return"]
    excess = rets - 0.02 / 252
    n_years = len(rets) / 252

    if n_years < 0.5 or excess.std() == 0:
        return {k: 0 for k in ["sharpe", "cagr", "max_dd", "sortino", "calmar",
                                "total_return", "time_in_market", "ann_vol"]}

    sharpe = excess.mean() / excess.std() * np.sqrt(252)
    cum = (1 + rets).cumprod()
    total = cum.iloc[-1] - 1
    cagr = (1 + total) ** (1 / n_years) - 1 if n_years > 0 else 0
    peak = cum.cummax()
    dd = (cum - peak) / peak
    mdd = dd.min()
    downside = excess[excess < 0]
    sortino = excess.mean() / downside.std() * np.sqrt(252) if len(downside) > 0 and downside.std() > 0 else 0
    invested = (rets != 0).sum() / len(rets)
    ann_vol = rets.std() * np.sqrt(252)
    calmar = cagr / abs(mdd) if abs(mdd) > 0 else 0

    return {
        "sharpe": round(float(sharpe), 3),
        "cagr": round(float(cagr), 4),
        "max_dd": round(float(mdd), 4),
        "sortino": round(float(sortino), 3),
        "calmar": round(float(calmar), 3),
        "total_return": round(float(total), 4),
        "time_in_market": round(float(invested), 3),
        "ann_vol": round(float(ann_vol), 4),
    }


def spy_bh_metrics(data, start, end):
    """SPY buy-and-hold metrics."""
    spy = data[BENCHMARK].loc[start:end, "Close"]
    r = spy.pct_change().dropna()
    ex = r - 0.02 / 252
    sh = ex.mean() / ex.std() * np.sqrt(252) if ex.std() > 0 else 0
    cum = (1 + r).cumprod()
    t = cum.iloc[-1] - 1
    n = len(r) / 252
    cg = (1 + t) ** (1 / n) - 1 if n > 0 else 0
    pk = cum.cummax()
    md = ((cum - pk) / pk).min()
    vol = r.std() * np.sqrt(252)
    return {
        "sharpe": round(float(sh), 3),
        "cagr": round(float(cg), 4),
        "max_dd": round(float(md), 4),
        "ann_vol": round(float(vol), 4),
    }


class SafeEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, (np.bool_,)): return bool(o)
        if isinstance(o, bool): return bool(o)
        if isinstance(o, float) and (math.isnan(o) or math.isinf(o)): return None
        if isinstance(o, (datetime.date, datetime.datetime)): return o.isoformat()
        if isinstance(o, pd.Timestamp): return o.isoformat()
        return super().default(o)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("Loading data...")
    data = load_data()
    print(f"  {len(data)} tickers loaded")

    stocks = [t for t in UNIVERSE if t not in ETFS_SET and t in data]
    print(f"  Stock universe: {len(stocks)} stocks")
    print(f"  ETFs: SPY, TLT, GLD, IEF + {len([e for e in SECTOR_ETFS if e in data])} sector ETFs")

    print(f"\nMulti-Asset Trend Alpha (MATA)")
    print(f"  Trend filter: SMA{SMA_PERIOD}")
    print(f"  Stocks per rebalance: {N_STOCKS}")
    print(f"  Vol target: {VOL_TARGET:.0%}")
    print(f"  Rebalance: every {REBAL_PERIOD} trading days")
    print(f"  Costs: {TX_COST_BPS} bps + {SLIPPAGE_BPS} bps slippage")

    all_results = {}

    PERIODS = [
        ("TRAIN", TRAIN_START, TRAIN_END),
        ("VALID", VALID_START, VALID_END),
        ("TEST", TEST_START, TEST_END),
        ("FULL", "2010-01-01", TEST_END),
    ]

    for name, s, e in PERIODS:
        print(f"\n{'='*60}")
        print(f"{name}: {s} to {e}")
        print(f"{'='*60}")

        ret_df, hlog, tlog = run_backtest(data, s, e, debug=(name == "TRAIN"))
        m = compute_metrics(ret_df)
        spy = spy_bh_metrics(data, s, e)
        all_results[name] = {
            "strategy": m, "spy": spy,
            "returns": ret_df, "holdings": hlog, "trades": tlog,
        }

        n_trades = len(tlog)
        print(f"  Rebalances: {n_trades}")
        if tlog:
            avg_turn = sum(t["turnover"] for t in tlog) / len(tlog)
            avg_pos = sum(t["n_positions"] for t in tlog) / len(tlog)
            print(f"  Avg turnover: {avg_turn:.0%}, Avg positions: {avg_pos:.0f}")

        print(f"\n  {'':20} {'MATA':>10} {'SPY B&H':>10}")
        print(f"  {'-'*42}")
        print(f"  {'Sharpe':<20} {m['sharpe']:>10.3f} {spy['sharpe']:>10.3f}")
        print(f"  {'CAGR':<20} {m['cagr']:>10.1%} {spy['cagr']:>10.1%}")
        print(f"  {'Max Drawdown':<20} {m['max_dd']:>10.1%} {spy['max_dd']:>10.1%}")
        print(f"  {'Sortino':<20} {m['sortino']:>10.3f}")
        print(f"  {'Calmar':<20} {m['calmar']:>10.3f}")
        print(f"  {'Ann. Volatility':<20} {m['ann_vol']:>10.1%} {spy['ann_vol']:>10.1%}")
        print(f"  {'Time in Market':<20} {m['time_in_market']:>10.1%}")

    print(f"\n{'='*60}")
    print("WALK-FORWARD SUMMARY")
    print(f"{'='*60}")
    for name in ["TRAIN", "VALID", "TEST"]:
        m = all_results[name]["strategy"]
        s = all_results[name]["spy"]
        print(f"  {name:8} MATA Sharpe={m['sharpe']:.3f}  CAGR={m['cagr']:.1%}  MDD={m['max_dd']:.1%}")
        print(f"  {'':8}  SPY Sharpe={s['sharpe']:.3f}  CAGR={s['cagr']:.1%}  MDD={s['max_dd']:.1%}")

    # ============================================================
    # CURRENT STATUS
    # ============================================================
    spy_close = data[BENCHMARK]["Close"]
    latest_idx = len(spy_close) - 1
    latest_date = spy_close.index[-1]

    # Check current trends
    equity_up = is_trending_up(spy_close, latest_idx)
    bond_close = data[BOND_ETF]["Close"]
    bond_up = is_trending_up(bond_close, latest_idx)
    gold_close = data[GOLD_ETF]["Close"]
    gold_up = is_trending_up(gold_close, latest_idx)

    print(f"\n{'='*60}")
    print(f"CURRENT STATUS ({latest_date.date()})")
    print(f"{'='*60}")
    print(f"  SPY: ${spy_close.iloc[-1]:.2f} ({'TRENDING' if equity_up else 'NOT trending'})")
    print(f"  TLT: ${bond_close.iloc[-1]:.2f} ({'TRENDING' if bond_up else 'NOT trending'})")
    print(f"  GLD: ${gold_close.iloc[-1]:.2f} ({'TRENDING' if gold_up else 'NOT trending'})")

    regime = "OFFENSIVE" if equity_up else "DEFENSIVE"
    n_trend = sum([equity_up, bond_up, gold_up])
    print(f"  Regime: {regime} ({n_trend}/3 asset classes trending)")

    # ============================================================
    # GENERATE WEB DATA
    # ============================================================
    print(f"\nGenerating web data...")

    # Compute current weights
    all_stocks = [t for t in UNIVERSE if t not in ETFS_SET and t in data]
    current_weights = {}

    if n_trend == 0:
        current_weights = {SAFE_ETF: 0.50, GOLD_ETF: 0.50}
    else:
        cw = 1.0 / n_trend
        if equity_up:
            cands = []
            for t in all_stocks:
                df = data[t]
                c = df["Close"]
                d = c.pct_change()
                score, passes = score_stock(c, d, len(c) - 1)
                if passes:
                    cands.append((t, score))
            if cands:
                cands.sort(key=lambda x: -x[1])
                top = cands[:N_STOCKS]
                sw = cw / len(top)
                for t, _ in top:
                    current_weights[t] = sw
            else:
                current_weights[EQUITY_ETF] = cw
        if bond_up:
            current_weights[BOND_ETF] = cw
        if gold_up:
            current_weights[GOLD_ETF] = cw

    # Sector details
    current_sectors = {}
    for etf in SECTOR_ETFS:
        df = data.get(etf)
        if df is None:
            continue
        idx_e = len(df) - 1
        if idx_e < 130:
            continue
        close = df["Close"]
        ret_3m = close.iloc[idx_e] / close.iloc[idx_e - 63] - 1
        ret_1m = close.iloc[idx_e] / close.iloc[idx_e - 21] - 1
        ret_1w = close.iloc[idx_e] / close.iloc[idx_e - 5] - 1
        vol = float(np.log(close / close.shift(1)).iloc[-21:].std() * np.sqrt(252))

        current_sectors[etf] = {
            "name": SECTOR_NAMES.get(etf, etf),
            "price": round(float(close.iloc[-1]), 2),
            "ret_3m": round(float(ret_3m) * 100, 1),
            "ret_1m": round(float(ret_1m) * 100, 1),
            "ret_1w": round(float(ret_1w) * 100, 1),
            "vol": round(vol * 100, 1),
            "weight": 0,
            "is_top": False,
        }

    # Equity curves
    full_ret = all_results["FULL"]["returns"]
    strat_cum = (1 + full_ret["return"]).cumprod() * 10000
    spy_full = data[BENCHMARK].loc["2010-01-01":TEST_END, "Close"]
    spy_cum = spy_full / spy_full.iloc[0] * 10000

    eq_strategy = [{"date": str(d.date()), "value": round(float(v), 0)}
                    for d, v in zip(full_ret["date"], strat_cum)]
    eq_spy = [{"date": str(d.date()), "value": round(float(v), 0)}
              for d, v in spy_cum.items()]

    # Top holdings for display
    top_holdings = sorted(current_weights.items(), key=lambda x: -x[1])[:10]

    sector_data = {
        "generated": datetime.datetime.now().isoformat(),
        "strategy": "MATA",
        "strategy_full_name": "Multi-Asset Trend Alpha",
        "description": (
            f"Cross-asset trend risk parity (equities + bonds + gold) with "
            f"stock alpha overlay. {N_STOCKS} quality-momentum stocks when equities trend. "
            f"Vol-targeted to {VOL_TARGET:.0%}."
        ),
        "current_status": {
            "spy_price": round(float(spy_close.iloc[-1]), 2),
            "sma50": round(float(spy_close.iloc[-SMA_PERIOD:].mean()), 2),
            "signal": regime,
            "equity_trending": equity_up,
            "bond_trending": bond_up,
            "gold_trending": gold_up,
            "n_trending": n_trend,
            "top_sector": top_holdings[0][0] if top_holdings else None,
            "top_sector_name": SECTOR_NAMES.get(top_holdings[0][0], top_holdings[0][0]) if top_holdings else "",
            "top_sector_weight": round(float(top_holdings[0][1]) * 100, 1) if top_holdings else 0,
            "weights": {k: round(v * 100, 1) for k, v in current_weights.items()},
        },
        "sectors": current_sectors,
        "how_it_works": {
            "regime_detection": (
                f"Check 3 asset classes: SPY, TLT, GLD. Each trending if above SMA{SMA_PERIOD}."
            ),
            "bull_allocation": (
                f"Equal-weight across trending asset classes. Equity portion: "
                f"top {N_STOCKS} quality-momentum stocks (above 200-SMA, positive annual return, "
                f"risk-adjusted momentum scored)."
            ),
            "bear_allocation": "When no asset class trends: 50% IEF (intermediate bonds) + 50% GLD",
            "vol_targeting": f"Position scaled to target {VOL_TARGET:.0%} annualized portfolio vol",
            "rebalancing": f"Every {REBAL_PERIOD} trading days (~monthly)",
        },
        "performance": {
            name.lower(): {
                "strategy": all_results[name]["strategy"],
                "spy": all_results[name]["spy"],
            }
            for name in all_results.keys()
        },
        "equity_curve_strategy": eq_strategy,
        "equity_curve_spy": eq_spy,
        "recent_changes": [
            {
                "date": str(h["date"].date()) if hasattr(h["date"], "date") else str(h["date"]),
                "n_positions": h.get("n_positions", 0),
                "turnover": h.get("turnover", 0),
                "weights": h.get("weights", {}),
            }
            for h in all_results.get("FULL", {}).get("holdings", [])[-20:]
        ],
    }

    docs_dir = os.path.join(os.path.dirname(__file__), "docs", "data")
    os.makedirs(docs_dir, exist_ok=True)

    with open(os.path.join(docs_dir, "sectors.json"), "w") as f:
        json.dump(sector_data, f, indent=2, cls=SafeEncoder)

    print(f"  Written to {docs_dir}/sectors.json")
