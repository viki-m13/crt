#!/usr/bin/env python3
"""
Adaptive Leverage Sector Rotation (ALSR)
==========================================
Key insight: 3x ETFs compound beautifully in STRONG trends but decay
in chop. So match leverage level to trend strength:

  STRONG BULL (SPY > SMA50 AND SPY > SMA200 AND mom_63d > 5%):
    → Top 2 leveraged ETFs (3x/2x) by momentum
  MILD BULL (SPY > SMA200 but not strong):
    → Top 3 regular sector ETFs by momentum
  BEAR (SPY < SMA200):
    → SHY (cash) + defensive ETFs (GLD, TLT if trending up)

Weekly rebalance, next-day open, 10bps slippage for leveraged, 5bps regular.
"""

import os, sys, json, datetime, math
import numpy as np
import pandas as pd
import yfinance as yf

REGULAR = [
    "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC",
    "SPY", "QQQ", "IWM", "DIA",
]

LEVERAGED_LONG = [
    "SPXL", "UPRO", "SSO", "TQQQ", "QLD",
    "TECL", "FAS", "ERX", "CURE", "SOXL",
    "TNA", "UDOW", "DRN", "DPST", "NAIL",
    "DUSL", "WANT", "RETL", "LABU", "MIDU", "WEBL",
]

LEVERAGED_SHORT = [
    "SPXS", "SDS", "SH", "SQQQ", "QID", "PSQ",
    "TECS", "FAZ", "ERY", "SOXS", "TZA", "SDOW",
    "DRV", "LABD", "DRIP", "WEBS",
]

BONDS_HAVENS = [
    "TLT", "IEF", "SHY", "TMF", "UBT", "TBT", "TBF", "TMV",
    "GLD", "UGL", "SLV",
]

ALL_ETFS = list(set(REGULAR + LEVERAGED_LONG + LEVERAGED_SHORT + BONDS_HAVENS))
BENCHMARK = "SPY"
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# Periods
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
            print(f"  Downloading {ticker}...")
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


class ALSRStrategy:
    def __init__(self, data):
        self.data = data
        self.tickers = [t for t in ALL_ETFS if t in data and len(data[t]) >= 300]

        self.closes, self.returns, self.mom63, self.vol21, self.sma50, self.sma200 = {}, {}, {}, {}, {}, {}
        for t in self.tickers:
            df = data[t]
            c = df["Close"]
            self.closes[t] = c
            self.returns[t] = c.pct_change()
            self.mom63[t] = c / c.shift(63) - 1
            self.vol21[t] = self.returns[t].rolling(21, min_periods=10).std() * np.sqrt(252)
            self.sma50[t] = c.rolling(50).mean()
            self.sma200[t] = c.rolling(200).mean()

        self.spy_close = data[BENCHMARK]["Close"]
        self.spy_sma50 = self.spy_close.rolling(50).mean()
        self.spy_sma200 = self.spy_close.rolling(200).mean()
        self.spy_mom63 = self.spy_close / self.spy_close.shift(63) - 1
        print(f"  Universe: {len(self.tickers)} ETFs")

    def regime(self, date):
        """3-tier regime: 'strong_bull', 'mild_bull', 'bear'."""
        if date not in self.spy_sma200.index:
            return "mild_bull"
        sma200 = self.spy_sma200.loc[date]
        sma50 = self.spy_sma50.loc[date]
        spy = self.spy_close.loc[date]
        mom = self.spy_mom63.loc[date] if date in self.spy_mom63.index else 0

        if pd.isna(sma200) or pd.isna(sma50):
            return "mild_bull"

        if spy < sma200:
            return "bear"
        if spy > sma50 and spy > sma200 and not pd.isna(mom) and mom > 0.05:
            return "strong_bull"
        return "mild_bull"

    def rank_etfs(self, date, pool):
        """Rank ETFs by risk-adjusted momentum, filtered by SMA50 trend."""
        scored = []
        for t in pool:
            if t not in self.tickers:
                continue
            if date not in self.mom63[t].index:
                continue
            m = self.mom63[t].loc[date]
            v = self.vol21[t].loc[date] if date in self.vol21[t].index else None
            sma = self.sma50[t].loc[date] if date in self.sma50[t].index else None
            p = self.closes[t].loc[date] if date in self.closes[t].index else None

            if pd.isna(m) or v is None or pd.isna(v) or v < 0.01:
                continue
            if m <= 0:
                continue
            if sma is not None and p is not None and not pd.isna(sma) and p < sma:
                continue

            scored.append((t, m / v, v))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def get_weights(self, date):
        r = self.regime(date)

        if r == "strong_bull":
            # Leveraged: top 2 by RAM
            candidates = self.rank_etfs(date, LEVERAGED_LONG)
            if not candidates:
                candidates = self.rank_etfs(date, REGULAR)
            if not candidates:
                return {"SPY": 1.0}
            top = candidates[:2]
            total_iv = sum(1.0 / v for _, _, v in top)
            return {t: (1.0 / v) / total_iv for t, _, v in top}

        elif r == "mild_bull":
            # Regular sectors: top 3
            candidates = self.rank_etfs(date, REGULAR)
            if not candidates:
                return {"SPY": 1.0}
            top = candidates[:3]
            total_iv = sum(1.0 / v for _, _, v in top)
            return {t: (1.0 / v) / total_iv for t, _, v in top}

        else:  # bear
            weights = {}
            # Check if GLD or TLT are trending up
            for h in ["GLD", "TLT", "IEF"]:
                if h in self.tickers and date in self.mom63.get(h, pd.Series()).index:
                    m = self.mom63[h].loc[date]
                    sma = self.sma50[h].loc[date] if date in self.sma50.get(h, pd.Series()).index else None
                    p = self.closes[h].loc[date] if date in self.closes.get(h, pd.Series()).index else None
                    if not pd.isna(m) and m > 0 and sma is not None and p is not None and p > sma:
                        weights[h] = 0.30
                        break

            # Check if any inverse ETF is trending (momentum > 0, above SMA)
            inv = self.rank_etfs(date, LEVERAGED_SHORT)
            if inv:
                t, _, _ = inv[0]
                weights[t] = 0.20

            # Rest to cash
            used = sum(weights.values())
            if used < 1.0:
                weights["SHY"] = 1.0 - used
            return weights

    def make_signal_fn(self):
        state = {"week": None, "weights": None}
        def signal_fn(date):
            week = date.isocalendar()[1]
            if state["week"] == week and state["weights"] is not None:
                return state["weights"]
            state["week"] = week
            state["weights"] = self.get_weights(date)
            return state["weights"]
        return signal_fn


def backtest(data, start, end, weight_fn):
    spy = data[BENCHMARK]
    dates = spy.loc[start:end].index

    daily_rets, raw_rets = [], []
    current_w = {}
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
            new_w = weight_fn(date)
            # Figure out slippage per asset
            for t, w in current_w.items():
                if t not in new_w or abs(new_w.get(t, 0) - w) > 0.005:
                    slip = 0.0010 if t in set(LEVERAGED_LONG + LEVERAGED_SHORT) else 0.0005
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

            for t, w in new_w.items():
                if t not in current_w or abs(current_w.get(t, 0) - w) > 0.005:
                    slip = 0.0010 if t in set(LEVERAGED_LONG + LEVERAGED_SHORT) else 0.0005
                    df = data.get(t)
                    if df is not None and date in df.index:
                        today_o = df.loc[date, "Open"] if "Open" in df.columns else df.loc[date, "Close"]
                        buy = today_o * (1 + slip)
                        today_c = df.loc[date, "Close"]
                        if buy > 0:
                            dr += (today_c / buy - 1) * w
                    trades += 1
            current_w = new_w
        elif current_w:
            for t, w in current_w.items():
                df = data.get(t)
                if df is not None and date in df.index:
                    si = df.index.get_loc(date)
                    if si > 0:
                        dr += (df.iloc[si]["Close"] / df.iloc[si - 1]["Close"] - 1) * w

        raw_rets.append(dr)

        # Vol-targeting: 15% target
        if len(raw_rets) >= 21:
            realized_vol = np.std(raw_rets[-21:]) * np.sqrt(252)
            if realized_vol > 0.01:
                exposure = np.clip(0.15 / realized_vol, 0.25, 1.50)
            else:
                exposure = 1.50
            daily_rets.append(dr * exposure)
        else:
            daily_rets.append(dr)

    return pd.Series(daily_rets, index=dates), trades


def m(rets, rf=0.02):
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
    print("ADAPTIVE LEVERAGE SECTOR ROTATION (ALSR)")
    print("=" * 60)
    data = download_etfs()
    print(f"Loaded {len(data)} ETFs")
    strategy = ALSRStrategy(data)

    for name, s, e in [("TRAIN", TRAIN_START, TRAIN_END), ("VALID", VALID_START, VALID_END),
                        ("TEST", TEST_START, TEST_END), ("FULL", "2012-01-01", TEST_END)]:
        sig_fn = strategy.make_signal_fn()
        rets, trades = backtest(data, s, e, sig_fn)
        met = m(rets)
        spy = m(data[BENCHMARK].loc[s:e, "Close"].pct_change().dropna())

        print(f"\n{'='*60}")
        print(f"{name}: {s} to {e}")
        print(f"{'='*60}")
        print(f"  {'':15} {'ALSR':>10} {'SPY':>10}")
        print(f"  {'-'*35}")
        for k in ["sharpe", "cagr", "max_dd", "sortino", "ann_vol", "calmar"]:
            label = k.replace("_", " ").title()
            if k in ("cagr", "max_dd", "ann_vol"):
                print(f"  {label:<15} {met[k]:>10.1%} {spy[k]:>10.1%}")
            else:
                print(f"  {label:<15} {met[k]:>10.3f} {spy[k]:>10.3f}")
        print(f"  {'Trades':<15} {trades:>10}")

    print(f"\n{'='*60}")
    print("WALK-FORWARD")
    print(f"{'='*60}")
    for year in range(2012, 2026):
        s, e = f"{year}-01-01", f"{year}-12-31"
        try:
            sig_fn = strategy.make_signal_fn()
            rets, _ = backtest(data, s, e, sig_fn)
            met = m(rets)
            spy = m(data[BENCHMARK].loc[s:e, "Close"].pct_change().dropna())
            beat = "✓" if met["sharpe"] > spy["sharpe"] else " "
            print(f"  {year}: Sharpe {met['sharpe']:>6.3f} vs SPY {spy['sharpe']:>6.3f} {beat} | "
                  f"CAGR {met['cagr']:>6.1%} | MaxDD {met['max_dd']:>7.1%}")
        except Exception as ex:
            print(f"  {year}: Error — {ex}")

    latest = data[BENCHMARK]["Close"].index[-1]
    w = strategy.get_weights(latest)
    r = strategy.regime(latest)
    print(f"\nCURRENT: {r.upper()} — {latest.date()}")
    for t, wt in sorted(w.items(), key=lambda x: -x[1]):
        tag = ""
        if t in LEVERAGED_LONG: tag = " [LEV]"
        elif t in LEVERAGED_SHORT: tag = " [INV]"
        elif t in BONDS_HAVENS: tag = " [HAVEN]"
        print(f"  {t:6} {wt*100:5.1f}%{tag}")
