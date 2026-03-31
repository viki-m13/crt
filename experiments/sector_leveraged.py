#!/usr/bin/env python3
"""
PRISM-V: Parallel Risk-Isolated Strategy Multiplexer with Volatility-targeting
================================================================================
PATENTABLE: Multi-sleeve ensemble of uncorrelated ETF strategies.
Each sleeve runs independently, targeting different market dynamics.
Combined via equal-risk allocation + portfolio-level vol-targeting.

THEORETICAL BASIS:
  If N strategies each have Sharpe S and pairwise correlation ρ:
  Combined Sharpe ≈ S × √(N / (1 + (N-1)×ρ))

  With 4 sleeves, Sharpe 0.8 each, correlation 0.2:
  Combined ≈ 0.8 × √(4 / (1 + 3×0.2)) = 0.8 × √(2.5) = 1.26

  With vol-targeting bonus (~+0.2) and better per-sleeve Sharpe:
  Target range: 1.5-2.5

FOUR SLEEVES:

  Sleeve 1: EQUITY MOMENTUM (TQQQ/SHY)
    - Long TQQQ when SPY > SMA200 AND vol < 18%
    - Otherwise SHY (cash)
    - Captures equity trends with leverage when safe

  Sleeve 2: BOND TREND (TMF/TBT/SHY)
    - Long TMF (3x long bonds) when TLT > SMA50
    - Long TBT (short bonds) when TLT < SMA50
    - Otherwise SHY
    - Profits from rate trends in BOTH directions

  Sleeve 3: GOLD TREND (UGL/SHY)
    - Long UGL (2x gold) when GLD > SMA50 AND GLD momentum > 0
    - Otherwise SHY
    - Flight-to-safety alpha, inflation hedge

  Sleeve 4: SECTOR MEAN REVERSION
    - Buy the worst-performing sector ETF of the last month
    - Sell after 1 week (or at next rebalance)
    - Profits from short-term overreaction
    - ANTICORRELATED with momentum sleeves (profits in chop)

Each sleeve: independently vol-targeted to 10%.
Combined: equal-weight sleeves → portfolio vol-targeted to 10%.

EXECUTION: Weekly rebalance, next-day open.
Manually: check 4 simple conditions, place 4-8 orders.
"""

import os, sys, datetime, math
import numpy as np
import pandas as pd
import yfinance as yf

ALL_ETFS = [
    "SPY", "QQQ", "TQQQ", "SPXL", "SHY", "SSO", "QLD",
    "TLT", "TMF", "TBT", "IEF", "UBT", "TMV",
    "GLD", "UGL", "SLV",
    "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC",
    "SQQQ", "SPXS", "SH", "SDS",
    "HYG",
]
BENCHMARK = "SPY"
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
SECTORS = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC"]

SLEEVE_VOL_TARGET = 0.15
PORTFOLIO_VOL_TARGET = 0.12
VOL_LOOKBACK = 21
VOL_FLOOR = 0.15
VOL_CAP = 2.5
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


def get_ret(data, ticker, date):
    """Get close-to-close return for a ticker on a date."""
    df = data.get(ticker)
    if df is None or date not in df.index:
        return 0.0
    si = df.index.get_loc(date)
    if si < 1:
        return 0.0
    return df.iloc[si]["Close"] / df.iloc[si - 1]["Close"] - 1


def get_switch_ret(data, old_ticker, new_ticker, date, slip):
    """Return on a switch day: exit old at open, enter new at open."""
    dr = 0.0
    # Exit: prev_close to today_open
    df_old = data.get(old_ticker)
    if df_old is not None and date in df_old.index:
        si = df_old.index.get_loc(date)
        if si > 0:
            prev_c = df_old.iloc[si - 1]["Close"]
            today_o = df_old.loc[date, "Open"] if "Open" in df_old.columns else prev_c
            dr += today_o * (1 - slip) / prev_c - 1

    # Enter: today_open to today_close
    df_new = data.get(new_ticker)
    if df_new is not None and date in df_new.index:
        today_o = df_new.loc[date, "Open"] if "Open" in df_new.columns else df_new.loc[date, "Close"]
        today_c = df_new.loc[date, "Close"]
        buy = today_o * (1 + slip)
        if buy > 0:
            dr += today_c / buy - 1

    return dr


class MultiSleeveStrategy:
    def __init__(self, data):
        self.data = data

        # Precompute signals
        spy_c = data["SPY"]["Close"]
        spy_r = spy_c.pct_change()
        self.spy_sma200 = spy_c.rolling(200).mean()
        self.spy_vol_slow = spy_r.rolling(42, min_periods=21).std() * np.sqrt(252)
        self.spy_vol_fast = spy_r.rolling(5, min_periods=3).std() * np.sqrt(252)

        tlt_c = data["TLT"]["Close"]
        self.tlt_sma50 = tlt_c.rolling(50).mean()
        self.tlt_mom21 = tlt_c / tlt_c.shift(21) - 1
        self.tlt_close = tlt_c

        gld_c = data["GLD"]["Close"]
        self.gld_sma50 = gld_c.rolling(50).mean()
        self.gld_mom21 = gld_c / gld_c.shift(21) - 1
        self.gld_close = gld_c

        # Sector 21-day returns for mean reversion
        self.sector_mom21 = {}
        for s in SECTORS:
            if s in data:
                self.sector_mom21[s] = data[s]["Close"] / data[s]["Close"].shift(21) - 1

    def sleeve1_equity(self, date):
        """TQQQ when SPY uptrend + low vol, else SHY."""
        sma = self.spy_sma200.loc[date] if date in self.spy_sma200.index else None
        vs = self.spy_vol_slow.loc[date] if date in self.spy_vol_slow.index else 0.20
        vf = self.spy_vol_fast.loc[date] if date in self.spy_vol_fast.index else 0.20
        if pd.isna(vs): vs = 0.20
        if pd.isna(vf): vf = 0.20

        spy_above = sma is not None and not pd.isna(sma) and self.data["SPY"]["Close"].loc[date] > sma
        vol_ok = vs < 0.18 and vf < 0.25

        if spy_above and vol_ok:
            return "TQQQ"
        return "SHY"

    def sleeve2_bonds(self, date):
        """TMF when TLT trending up, TBT when trending down, SHY when flat.
        Enhanced: use TLT price vs SMA + momentum direction."""
        sma = self.tlt_sma50.loc[date] if date in self.tlt_sma50.index else None
        mom = self.tlt_mom21.loc[date] if date in self.tlt_mom21.index else 0
        price = self.tlt_close.loc[date] if date in self.tlt_close.index else 0
        if pd.isna(mom): mom = 0

        if sma is not None and not pd.isna(sma):
            if price > sma and mom > 0:
                return "TMF"  # 3x long bonds — rates falling
            elif price < sma and mom < 0:
                return "TBT"  # Short bonds — rates rising
        return "SHY"

    def sleeve3_gold(self, date):
        """UGL when GLD trending up, else SHY."""
        sma = self.gld_sma50.loc[date] if date in self.gld_sma50.index else None
        mom = self.gld_mom21.loc[date] if date in self.gld_mom21.index else 0
        price = self.gld_close.loc[date] if date in self.gld_close.index else 0
        if pd.isna(mom): mom = 0

        if sma is not None and not pd.isna(sma) and price > sma and mom > 0:
            return "UGL"  # 2x gold
        return "SHY"

    def sleeve4_credit(self, date):
        """Credit momentum sleeve: SSO when credit expanding, SHY when contracting.
        HYG (high yield) is the canary in the coal mine — it leads equities."""
        if "HYG" not in self.data:
            return "SHY"
        hyg = self.data["HYG"]["Close"]
        if date not in hyg.index:
            return "SHY"
        si = hyg.index.get_loc(date)
        if si < 50:
            return "SHY"

        # HYG above its SMA50 AND positive 21-day momentum = credit healthy
        sma50 = hyg.iloc[si-49:si+1].mean()
        mom21 = hyg.iloc[si] / hyg.iloc[si-21] - 1 if si >= 21 else 0

        if hyg.iloc[si] > sma50 and mom21 > 0:
            return "SSO"  # 2x SPY — credit expanding, risk-on
        return "SHY"


def run_ensemble(data, start, end):
    """Run 4 sleeves independently, combine with equal weight."""
    strategy = MultiSleeveStrategy(data)
    spy = data[BENCHMARK]
    dates = spy.loc[start:end].index
    slip = SLIPPAGE_BPS / 10000

    # State per sleeve
    n_sleeves = 4
    sleeve_fns = [strategy.sleeve1_equity, strategy.sleeve2_bonds,
                  strategy.sleeve3_gold, strategy.sleeve4_credit]
    current = [None] * n_sleeves  # Current holding per sleeve
    raw_sleeve_rets = [[] for _ in range(n_sleeves)]
    combined_rets = []
    raw_combined = []
    last_week = None
    trades = 0

    for date in dates:
        idx = spy.index.get_loc(date)
        if idx < 252:
            combined_rets.append(0.0)
            raw_combined.append(0.0)
            for i in range(n_sleeves):
                raw_sleeve_rets[i].append(0.0)
            continue

        week = date.isocalendar()[1]
        rebalance = (last_week is not None and week != last_week)
        last_week = week

        sleeve_dr = [0.0] * n_sleeves

        for i in range(n_sleeves):
            target = sleeve_fns[i](date)

            if rebalance and target != current[i]:
                # Switch
                if current[i] is not None and current[i] != target:
                    sleeve_dr[i] = get_switch_ret(data, current[i], target, date, slip)
                    trades += 1
                elif current[i] is None:
                    # First entry
                    df_new = data.get(target)
                    if df_new is not None and date in df_new.index:
                        si = df_new.index.get_loc(date)
                        if si > 0:
                            sleeve_dr[i] = df_new.iloc[si]["Close"] / df_new.iloc[si - 1]["Close"] - 1
                    trades += 1
                current[i] = target
            else:
                # Hold
                if current[i] is not None:
                    sleeve_dr[i] = get_ret(data, current[i], date)
                if rebalance:
                    current[i] = target

            raw_sleeve_rets[i].append(sleeve_dr[i])

        # Per-sleeve vol-targeting, then combine
        scaled_dr = [0.0] * n_sleeves
        for i in range(n_sleeves):
            if len(raw_sleeve_rets[i]) >= VOL_LOOKBACK:
                rv = np.std(raw_sleeve_rets[i][-VOL_LOOKBACK:]) * np.sqrt(252)
                if rv > 0.005:
                    exp = np.clip(SLEEVE_VOL_TARGET / rv, VOL_FLOOR, VOL_CAP)
                else:
                    exp = VOL_CAP
                scaled_dr[i] = sleeve_dr[i] * exp
            else:
                scaled_dr[i] = sleeve_dr[i]

        # Equal-weight combination
        raw_port = sum(scaled_dr) / n_sleeves
        raw_combined.append(raw_port)

        # Portfolio-level vol-targeting
        if len(raw_combined) >= VOL_LOOKBACK:
            port_vol = np.std(raw_combined[-VOL_LOOKBACK:]) * np.sqrt(252)
            if port_vol > 0.005:
                port_exp = np.clip(PORTFOLIO_VOL_TARGET / port_vol, VOL_FLOOR, VOL_CAP)
            else:
                port_exp = VOL_CAP
            combined_rets.append(raw_port * port_exp)
        else:
            combined_rets.append(raw_port)

    return pd.Series(combined_rets, index=dates), trades, strategy, raw_sleeve_rets


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
    print("PRISM-V: Multi-Sleeve Ensemble Strategy")
    print("=" * 60)
    data = download_etfs()
    print(f"Loaded {len(data)} ETFs\n")

    for name, s, e in [("TRAIN", TRAIN_START, TRAIN_END), ("VALID", VALID_START, VALID_END),
                        ("TEST", TEST_START, TEST_END), ("FULL", "2012-01-01", TEST_END)]:
        rets, trades, strat, sleeve_rets = run_ensemble(data, s, e)
        met = calc_metrics(rets)
        spy = calc_metrics(data[BENCHMARK].loc[s:e, "Close"].pct_change().dropna())

        # Per-sleeve metrics
        sleeve_names = ["Equity(TQQQ)", "Bonds(TMF/TBT)", "Gold(UGL)", "Credit(SSO/SHY)"]
        sleeve_metrics = []
        spy_dates = data[BENCHMARK].loc[s:e].index
        for i in range(4):
            sr = pd.Series(sleeve_rets[i], index=spy_dates[:len(sleeve_rets[i])])
            sm = calc_metrics(sr)
            sleeve_metrics.append(sm)

        print(f"{'='*60}")
        print(f"{name}: {s} to {e}")
        print(f"{'='*60}")
        print(f"  {'':20} {'ENSEMBLE':>10} {'SPY':>10}")
        print(f"  {'-'*40}")
        for k in ["sharpe", "cagr", "max_dd", "sortino", "ann_vol", "calmar"]:
            label = k.replace("_", " ").title()
            if k in ("cagr", "max_dd", "ann_vol"):
                print(f"  {label:<20} {met[k]:>10.1%} {spy[k]:>10.1%}")
            else:
                print(f"  {label:<20} {met[k]:>10.3f} {spy[k]:>10.3f}")
        print(f"  {'Trades':<20} {trades:>10}")

        # Sleeve breakdown
        print(f"\n  Sleeve Sharpes:")
        for i, sn in enumerate(sleeve_names):
            print(f"    {sn:20} {sleeve_metrics[i]['sharpe']:>6.3f}  CAGR {sleeve_metrics[i]['cagr']:>6.1%}")
        print()

    # Walk-forward
    print(f"{'='*60}")
    print("WALK-FORWARD")
    print(f"{'='*60}")
    sharpes = []
    for year in range(2012, 2026):
        s, e = f"{year}-01-01", f"{year}-12-31"
        try:
            rets, _, _, _ = run_ensemble(data, s, e)
            met = calc_metrics(rets)
            spy = calc_metrics(data[BENCHMARK].loc[s:e, "Close"].pct_change().dropna())
            beat = "✓" if met["sharpe"] > spy["sharpe"] else " "
            print(f"  {year}: Sharpe {met['sharpe']:>6.3f} vs SPY {spy['sharpe']:>6.3f} {beat} | "
                  f"CAGR {met['cagr']:>6.1%} | MaxDD {met['max_dd']:>7.1%} | Vol {met['ann_vol']:>5.1%}")
            sharpes.append(met["sharpe"])
        except Exception as ex:
            print(f"  {year}: Error — {ex}")
    if sharpes:
        pos = sum(1 for s in sharpes if s > 0)
        print(f"\n  Avg Sharpe: {np.mean(sharpes):.3f} | Positive: {pos}/{len(sharpes)} | "
              f"Min: {min(sharpes):.3f} | Max: {max(sharpes):.3f}")

    # Current positions
    latest = data[BENCHMARK]["Close"].index[-1]
    strat = MultiSleeveStrategy(data)
    print(f"\n{'='*60}")
    print(f"CURRENT POSITIONS — {latest.date()}")
    print(f"{'='*60}")
    print(f"  Sleeve 1 (Equity):     {strat.sleeve1_equity(latest)}")
    print(f"  Sleeve 2 (Bonds):      {strat.sleeve2_bonds(latest)}")
    print(f"  Sleeve 3 (Gold):       {strat.sleeve3_gold(latest)}")
    print(f"  Sleeve 4 (Credit):     {strat.sleeve4_credit(latest)}")
