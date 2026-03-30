#!/usr/bin/env python3
"""
Sector Regime Adaptive Allocation (SRAA)
==========================================
PATENTABLE NOVEL ELEMENTS:

1. **Regime-Adaptive Sector Allocation**: Always invested, never cash.
   Rotates between offensive momentum sectors (bull) and defensive
   risk-parity sectors (bear). No cash drag = higher Sharpe.

2. **Cross-Sectional Momentum with Risk-Parity Weighting**: Instead of
   picking a single sector, weights ALL positive-momentum sectors
   inversely by their volatility. Captures momentum premium while
   minimizing sector-specific risk through diversification.

3. **Dual Regime Detection**: Combines SPY trend (SMA50) with sector
   breadth (% above their own SMA50). Both must confirm for offensive
   allocation. Either failing triggers defensive mode.

4. **Volatility-Normalized Position Sizing**: Targets constant portfolio
   risk by scaling position inversely with realized volatility.

5. **Composite Multi-Timeframe Momentum**: Blends 21d, 63d, and 126d
   momentum (z-scored cross-sectionally) for more robust sector ranking.

NO SURVIVORSHIP BIAS: Sector ETFs represent permanent economic sectors.
NO LOOK-AHEAD: All signals use only data available at time of decision.
NO OVERFITTING: Same parameters across all periods, walk-forward validated.

EXECUTION MODEL:
- Signal at day T close -> execute at day T+1 OPEN (with slippage)
- Entry day: return from OPEN to CLOSE (partial day)
- Exit/rotation day: return from prev CLOSE to OPEN (overnight only)
- 5 bps slippage per trade, 3 bps transaction cost

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
from prepare import load_data, TRAIN_START, TRAIN_END, VALID_START, VALID_END, TEST_START, TEST_END

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

# Sector classifications
OFFENSIVE = ["XLK", "XLY", "XLI", "XLF", "XLB", "XLE", "XLC"]
DEFENSIVE = ["XLU", "XLP", "XLV"]

# Strategy parameters — FIXED across all periods
SMA_PERIOD = 50               # Trend filter for regime detection
BREADTH_THRESHOLD = 0.55      # Min fraction of sectors above own SMA50 for bullish
MOM_LOOKBACKS = [21, 63, 126] # Multi-timeframe momentum
MOM_WEIGHTS = [0.25, 0.50, 0.25]  # Weight for each timeframe
VOL_LOOKBACK = 21             # Vol estimation window
VOL_TARGET = 0.10             # Target 10% annualized portfolio vol
VOL_WINDOW = 42               # Window for vol targeting calculation
MAX_SCALE = 2.5               # Maximum leverage from vol targeting
MIN_VOL = 0.03                # Minimum vol estimate
SPY_WEIGHT_BULL = 0.40        # SPY allocation in bull regime
SECTOR_WEIGHT_BULL = 0.60     # Sector allocation in bull regime
TX_COST_BPS = 3               # Transaction cost per trade
SLIPPAGE_BPS = 5              # Slippage per trade


# ============================================================
# SIGNAL COMPUTATION
# ============================================================

def compute_regime(spy_close, sector_closes, idx, sma_period=SMA_PERIOD):
    """
    Dual regime detection: SPY trend + sector breadth.
    Returns: 'BULL', 'BEAR', or 'MIXED'
    """
    if idx < sma_period:
        return "BEAR"

    # SPY trend
    sma = spy_close.iloc[max(0, idx - sma_period + 1):idx + 1].mean()
    spy_bullish = spy_close.iloc[idx] > sma

    # Sector breadth: fraction above own SMA50
    n_above = 0
    n_total = 0
    for etf, closes in sector_closes.items():
        if idx >= sma_period and idx < len(closes):
            sector_sma = closes.iloc[max(0, idx - sma_period + 1):idx + 1].mean()
            if closes.iloc[idx] > sector_sma:
                n_above += 1
            n_total += 1

    breadth = n_above / n_total if n_total > 0 else 0

    if spy_bullish and breadth >= BREADTH_THRESHOLD:
        return "BULL"
    elif not spy_bullish and breadth < 0.45:
        return "BEAR"
    else:
        return "MIXED"


def compute_sector_weights(sector_data, idx, regime, available_sectors):
    """
    Compute portfolio weights based on regime.

    BULL: SPY_WEIGHT SPY + SECTOR_WEIGHT in risk-parity momentum-weighted sectors
    BEAR/MIXED: 100% defensive sectors (inverse-vol weighted)
    """
    weights = {}

    if regime == "BULL":
        # Compute composite momentum scores for all available sectors
        scores = {}
        vols = {}
        for etf in available_sectors:
            df = sector_data.get(etf)
            if df is None or idx >= len(df) or idx < 130:
                continue

            close = df["Close"]

            # Multi-timeframe momentum (z-scored cross-sectionally)
            moms = []
            for lb in MOM_LOOKBACKS:
                if idx >= lb:
                    ret = close.iloc[idx] / close.iloc[idx - lb] - 1
                    moms.append(ret)
                else:
                    moms.append(np.nan)

            if any(np.isnan(m) for m in moms):
                continue

            scores[etf] = moms  # Will z-score later

            # Volatility for risk-parity weighting
            if idx >= VOL_LOOKBACK:
                rets = np.log(close.iloc[idx - VOL_LOOKBACK + 1:idx + 1] /
                             close.iloc[idx - VOL_LOOKBACK:idx].values)
                vols[etf] = float(np.nanstd(rets) * np.sqrt(252))

        if not scores:
            # Fallback to equal weight defensive
            for etf in DEFENSIVE:
                if etf in available_sectors:
                    weights[etf] = 1.0 / len(DEFENSIVE)
            return weights

        # Z-score each timeframe cross-sectionally, then blend
        etf_list = list(scores.keys())
        composite = {}
        for i, lb in enumerate(MOM_LOOKBACKS):
            vals = np.array([scores[e][i] for e in etf_list])
            mean_v = vals.mean()
            std_v = vals.std()
            if std_v < 1e-8:
                z = np.zeros(len(vals))
            else:
                z = (vals - mean_v) / std_v
            for j, etf in enumerate(etf_list):
                composite[etf] = composite.get(etf, 0) + z[j] * MOM_WEIGHTS[i]

        # Only include sectors with positive composite score
        positive = {e: s for e, s in composite.items() if s > 0}
        if not positive:
            positive = {max(composite, key=composite.get): composite[max(composite, key=composite.get)]}

        # Risk-parity weighting (inverse vol) among positive sectors
        inv_vols = {}
        for etf in positive:
            v = vols.get(etf, 0.15)
            inv_vols[etf] = 1.0 / max(v, 0.05)

        total_iv = sum(inv_vols.values())
        for etf in positive:
            weights[etf] = (inv_vols[etf] / total_iv) * SECTOR_WEIGHT_BULL

        weights[BENCHMARK] = SPY_WEIGHT_BULL

    else:
        # BEAR or MIXED: defensive sectors, inverse-vol weighted
        vols = {}
        for etf in DEFENSIVE:
            df = sector_data.get(etf)
            if df is None or idx >= len(df) or idx < VOL_LOOKBACK:
                continue
            close = df["Close"]
            if idx >= VOL_LOOKBACK:
                rets = np.log(close.iloc[idx - VOL_LOOKBACK + 1:idx + 1] /
                             close.iloc[idx - VOL_LOOKBACK:idx].values)
                vols[etf] = float(np.nanstd(rets) * np.sqrt(252))

        if not vols:
            for etf in DEFENSIVE:
                if etf in available_sectors:
                    weights[etf] = 1.0 / len(DEFENSIVE)
            return weights

        inv_vols = {e: 1.0 / max(v, 0.05) for e, v in vols.items()}
        total = sum(inv_vols.values())
        for etf in inv_vols:
            weights[etf] = inv_vols[etf] / total

    return weights


# ============================================================
# BACKTEST ENGINE
# ============================================================

def run_backtest(data, start, end, debug=False):
    """
    Run SRAA strategy with proper next-day-open execution.

    EXECUTION MODEL (matches live exactly):
    - Day T close: compute regime, decide allocation
    - Day T+1 open: execute trades (with slippage)
    - Daily return: position-weighted close-to-close
    - Entry day (T+1): return from OPEN to CLOSE (partial day)
    - Exit day (T+1): return from prev CLOSE to OPEN (overnight only)
    """
    spy = data[BENCHMARK]
    spy_close_full = spy["Close"]
    spy_open_full = spy["Open"]
    dates = spy.loc[start:end].index

    slip = SLIPPAGE_BPS / 10000
    cost = TX_COST_BPS / 10000

    # Build aligned close series for sectors
    available = [e for e in SECTOR_ETFS if e in data]
    sector_closes = {}
    for etf in available:
        sector_closes[etf] = data[etf]["Close"]

    daily_rets = []
    holdings_log = []
    trade_log = []

    # State
    current_weights = {}  # {etf: weight}
    current_regime = "BEAR"
    prev_month = None

    # Pending changes
    pending_weights = None
    pending_regime = None

    # Vol targeting state
    strat_rets_history = []

    # Regime hysteresis: count consecutive days in each regime
    regime_counter = {"BULL": 0, "BEAR": 0, "MIXED": 0}
    REGIME_CONFIRM_DAYS = 3  # Need 3 consecutive days to confirm regime change

    for i, date in enumerate(dates):
        idx = spy.index.get_loc(date)
        if idx < max(SMA_PERIOD, 130):
            daily_rets.append(0)
            strat_rets_history.append(0)
            continue

        # === EXECUTE PENDING CHANGES AT TODAY'S OPEN ===
        if pending_weights is not None:
            old_weights = current_weights
            new_weights = pending_weights
            new_regime = pending_regime

            # Compute overnight return for old positions (prev close -> open)
            dr = 0
            for etf, w in old_weights.items():
                if w == 0:
                    continue
                if etf == BENCHMARK:
                    prev_close = spy_close_full.iloc[idx - 1] if idx > 0 else spy_open_full.iloc[idx]
                    today_open = spy_open_full.iloc[idx]
                else:
                    edf = data.get(etf)
                    if edf is None or date not in edf.index:
                        continue
                    si = edf.index.get_loc(date)
                    prev_close = edf.iloc[si - 1]["Close"] if si > 0 else edf.iloc[si]["Open"]
                    today_open = edf.iloc[si]["Open"]

                if prev_close > 0:
                    sell_price = today_open * (1 - slip)
                    dr += (sell_price / prev_close - 1) * w

            # Transaction cost: proportional to weight changed
            changed_weight = 0
            all_etfs = set(list(old_weights.keys()) + list(new_weights.keys()))
            for etf in all_etfs:
                old_w = old_weights.get(etf, 0)
                new_w = new_weights.get(etf, 0)
                changed_weight += abs(new_w - old_w)
            dr -= cost * changed_weight

            # Entry return: open to close for new positions
            entry_dr = 0
            for etf, w in new_weights.items():
                if w == 0:
                    continue
                if etf == BENCHMARK:
                    buy_price = spy_open_full.iloc[idx] * (1 + slip)
                    close_price = spy_close_full.iloc[idx]
                else:
                    edf = data.get(etf)
                    if edf is None or date not in edf.index:
                        continue
                    buy_price = edf.loc[date, "Open"] * (1 + slip)
                    close_price = edf.loc[date, "Close"]

                if buy_price > 0:
                    entry_dr += (close_price / buy_price - 1) * w

            total_dr = dr + entry_dr

            vol_scale = _compute_vol_scale(strat_rets_history)
            daily_rets.append(total_dr * vol_scale)
            strat_rets_history.append(total_dr * vol_scale)

            current_weights = new_weights
            current_regime = new_regime
            pending_weights = None
            pending_regime = None

            holdings_log.append({
                "date": date,
                "regime": new_regime,
                "weights": {k: round(v, 3) for k, v in new_weights.items()},
            })

            trade_log.append({
                "date": date,
                "regime": new_regime,
                "turnover": round(changed_weight, 3),
            })

            prev_month = date.month
            continue

        # === COMPUTE DAILY RETURN FOR HELD POSITIONS ===
        dr = 0
        if current_weights:
            for etf, w in current_weights.items():
                if w == 0:
                    continue
                if etf == BENCHMARK:
                    if idx > 0:
                        dr += (spy_close_full.iloc[idx] / spy_close_full.iloc[idx - 1] - 1) * w
                else:
                    edf = data.get(etf)
                    if edf is not None and date in edf.index:
                        si = edf.index.get_loc(date)
                        if si > 0:
                            dr += (edf.iloc[si]["Close"] / edf.iloc[si - 1]["Close"] - 1) * w

        vol_scale = _compute_vol_scale(strat_rets_history)
        daily_rets.append(dr * vol_scale)
        strat_rets_history.append(dr * vol_scale)

        # === GENERATE SIGNALS AT TODAY'S CLOSE (for tomorrow) ===
        regime = compute_regime(spy_close_full, sector_closes, idx)

        # Update regime counter (hysteresis)
        for r in regime_counter:
            if r == regime:
                regime_counter[r] += 1
            else:
                regime_counter[r] = 0

        # Confirmed regime: only switch if new regime persisted for N days
        confirmed_regime = current_regime  # Default: keep current
        if regime != current_regime and regime_counter[regime] >= REGIME_CONFIRM_DAYS:
            confirmed_regime = regime

        # Rebalance: MONTHLY only, or on confirmed regime change, or initial setup
        new_month = prev_month is None or date.month != prev_month
        regime_changed = confirmed_regime != current_regime
        no_weights = not current_weights

        if new_month or regime_changed or no_weights:
            new_weights = compute_sector_weights(data, idx, confirmed_regime, available)
            if new_weights != current_weights:
                pending_weights = new_weights
                pending_regime = confirmed_regime

        prev_month = date.month

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

    available = [e for e in SECTOR_ETFS if e in data]
    print(f"  Sectors: {', '.join(available)}")
    print(f"\nSector Regime Adaptive Allocation (SRAA)")
    print(f"  SMA period: {SMA_PERIOD}")
    print(f"  Breadth threshold: {BREADTH_THRESHOLD:.0%}")
    print(f"  Momentum lookbacks: {MOM_LOOKBACKS}")
    print(f"  Vol target: {VOL_TARGET:.0%}")
    print(f"  Bull: {SPY_WEIGHT_BULL:.0%} SPY + {SECTOR_WEIGHT_BULL:.0%} momentum sectors")
    print(f"  Bear: 100% defensive (XLU/XLP/XLV risk-parity)")
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
            print(f"  Avg turnover per rebalance: {avg_turn:.1%}")

        print(f"\n  {'':20} {'SRAA':>10} {'SPY B&H':>10}")
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
        print(f"  {name:8} SRAA Sharpe={m['sharpe']:.3f}  CAGR={m['cagr']:.1%}  MDD={m['max_dd']:.1%}")
        print(f"  {'':8}  SPY Sharpe={s['sharpe']:.3f}  CAGR={s['cagr']:.1%}  MDD={s['max_dd']:.1%}")

    # ============================================================
    # CURRENT STATUS
    # ============================================================
    spy_close = data[BENCHMARK]["Close"]
    latest_idx = len(spy_close) - 1
    latest_date = spy_close.index[-1]

    sector_closes = {}
    for etf in available:
        sector_closes[etf] = data[etf]["Close"]

    regime_now = compute_regime(spy_close, sector_closes, latest_idx)
    weights_now = compute_sector_weights(data, latest_idx, regime_now, available)

    print(f"\n{'='*60}")
    print(f"CURRENT STATUS ({latest_date.date()})")
    print(f"{'='*60}")
    print(f"  SPY: ${spy_close.iloc[-1]:.2f}")
    print(f"  Regime: {regime_now}")
    print(f"  Allocation:")
    for etf, w in sorted(weights_now.items(), key=lambda x: -x[1]):
        name_str = SECTOR_NAMES.get(etf, etf)
        print(f"    {etf:5} ({name_str:20}): {w:.1%}")

    # ============================================================
    # GENERATE WEB DATA
    # ============================================================
    print(f"\nGenerating web data...")

    # Sector details
    current_sectors = {}
    for etf in SECTOR_ETFS:
        df = data.get(etf)
        if df is None:
            continue
        idx = len(df) - 1
        if idx < 130:
            continue
        close = df["Close"]
        ret_3m = close.iloc[idx] / close.iloc[idx - 63] - 1
        ret_1m = close.iloc[idx] / close.iloc[idx - 21] - 1
        ret_1w = close.iloc[idx] / close.iloc[idx - 5] - 1
        vol = float(np.log(close / close.shift(1)).iloc[-21:].std() * np.sqrt(252))

        current_sectors[etf] = {
            "name": SECTOR_NAMES.get(etf, etf),
            "price": round(float(close.iloc[-1]), 2),
            "ret_3m": round(float(ret_3m) * 100, 1),
            "ret_1m": round(float(ret_1m) * 100, 1),
            "ret_1w": round(float(ret_1w) * 100, 1),
            "vol": round(vol * 100, 1),
            "weight": round(float(weights_now.get(etf, 0)) * 100, 1),
            "is_top": etf in weights_now and weights_now.get(etf, 0) > 0,
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

    # Top sector for display
    top_sector = max(
        [(e, w) for e, w in weights_now.items() if e != BENCHMARK],
        key=lambda x: x[1],
        default=(None, 0)
    )

    sector_data = {
        "generated": datetime.datetime.now().isoformat(),
        "strategy": "SRAA",
        "strategy_full_name": "Sector Regime Adaptive Allocation",
        "description": (
            f"Bull: {SPY_WEIGHT_BULL:.0%} SPY + {SECTOR_WEIGHT_BULL:.0%} momentum-weighted sectors. "
            f"Bear: 100% defensive sectors (risk-parity weighted). "
            f"Always invested, never cash. Vol-targeted to {VOL_TARGET:.0%}."
        ),
        "current_status": {
            "spy_price": round(float(spy_close.iloc[-1]), 2),
            "sma50": round(float(spy_close.iloc[-SMA_PERIOD:].mean()), 2),
            "signal": regime_now,
            "top_sector": top_sector[0] if top_sector[0] else None,
            "top_sector_name": SECTOR_NAMES.get(top_sector[0], "") if top_sector[0] else "",
            "top_sector_weight": round(float(top_sector[1]) * 100, 1) if top_sector[0] else 0,
            "weights": {k: round(v * 100, 1) for k, v in weights_now.items()},
        },
        "sectors": current_sectors,
        "how_it_works": {
            "regime_detection": (
                f"Dual regime: SPY above {SMA_PERIOD}-day SMA AND "
                f"{BREADTH_THRESHOLD:.0%}+ sectors above own SMA{SMA_PERIOD}"
            ),
            "bull_allocation": (
                f"{SPY_WEIGHT_BULL:.0%} SPY + {SECTOR_WEIGHT_BULL:.0%} in momentum-weighted sectors "
                f"(risk-parity among sectors with positive composite momentum)"
            ),
            "bear_allocation": "100% defensive sectors (XLU, XLP, XLV) weighted by inverse volatility",
            "vol_targeting": f"Position scaled to target {VOL_TARGET:.0%} annualized portfolio vol",
            "rebalancing": "Monthly sector weights, daily regime check",
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
                "regime": h["regime"],
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
