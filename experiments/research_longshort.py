#!/usr/bin/env python3
"""
LONG/SHORT SECTOR MOMENTUM — Path to 3+ Sharpe

The key insight: market-neutral long/short has VERY LOW volatility (3-8%).
If sector momentum spread is ~10-15% annually, Sharpe = 10%/5% = 2.0+

Strategy:
- Long top N sectors by momentum
- Short bottom N sectors by momentum
- Equal dollar long/short = market neutral
- Portfolio vol: ~3-8% (vs ~15% for long-only)

This is fundamentally different from long-only.
"""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prepare import load_data, TRAIN_START, TRAIN_END, VALID_START, VALID_END, TEST_START, TEST_END

BENCHMARK = "SPY"
SECTOR_ETFS = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC"]
SAFE_HAVENS = ["TLT", "GLD", "IEF"]

def compute_metrics(rets, rf=0.02):
    if len(rets) == 0 or rets.std() == 0:
        return {"sharpe": 0, "cagr": 0, "max_dd": 0, "sortino": 0, "ann_vol": 0}
    excess = rets - rf / 252
    n_years = len(rets) / 252
    sharpe = excess.mean() / excess.std() * np.sqrt(252)
    cum = (1 + rets).cumprod()
    total = cum.iloc[-1] - 1
    cagr = (1 + total) ** (1 / n_years) - 1 if n_years >= 1 else total
    mdd = ((cum - cum.cummax()) / cum.cummax()).min()
    downside = excess[excess < 0]
    sortino = excess.mean() / downside.std() * np.sqrt(252) if len(downside) > 0 and downside.std() > 0 else 0
    return {"sharpe": round(float(sharpe), 3), "cagr": round(float(cagr), 4),
            "max_dd": round(float(mdd), 4), "sortino": round(float(sortino), 3),
            "ann_vol": round(float(rets.std() * np.sqrt(252)), 4)}


def run_tests(data):
    available = [e for e in SECTOR_ETFS if e in data]
    print(f"Available sectors: {len(available)}")

    # Precompute
    closes = {e: data[e]["Close"] for e in available}
    rets = {e: data[e]["Close"].pct_change() for e in available}
    vol63 = {e: rets[e].rolling(63, min_periods=21).std() * np.sqrt(252) for e in available}

    spy_close = data[BENCHMARK]["Close"]
    spy_ret = spy_close.pct_change()
    spy_sma100 = spy_close.rolling(100).mean()

    # ================================================================
    # Approach 1: Pure L/S sector momentum (daily rebalance for analysis)
    # Long top 3, Short bottom 3, by 63d momentum
    # ================================================================
    print("\n" + "="*60)
    print("LS1: Long top 3 / Short bottom 3 (63d momentum)")
    print("     Rebalance monthly, equal weight L/S")
    print("="*60)

    for mom_lookback in [21, 42, 63, 126]:
        mom = {e: closes[e] / closes[e].shift(mom_lookback) - 1 for e in available}

        for n_long, n_short in [(3,3), (4,4), (5,5)]:
            # Monthly rebalance backtest
            all_daily_rets = []
            last_month = None
            long_pos = []
            short_pos = []

            for date in spy_close.loc[TRAIN_START:TEST_END].index:
                idx = spy_close.index.get_loc(date)
                if idx < max(252, mom_lookback + 10):
                    all_daily_rets.append(0.0)
                    continue

                month = date.month
                rebalance = (last_month is not None and month != last_month)
                last_month = month

                if rebalance:
                    # Rank sectors
                    scored = []
                    for e in available:
                        if date in mom[e].index:
                            m = mom[e].loc[date]
                            if not pd.isna(m):
                                scored.append((e, m))
                    scored.sort(key=lambda x: x[1], reverse=True)

                    if len(scored) >= n_long + n_short:
                        long_pos = [e for e, _ in scored[:n_long]]
                        short_pos = [e for e, _ in scored[-n_short:]]

                # Daily return
                dr = 0.0
                w_long = 0.5 / max(len(long_pos), 1)  # 50% long
                w_short = 0.5 / max(len(short_pos), 1)  # 50% short

                for e in long_pos:
                    if e in rets and date in rets[e].index:
                        r = rets[e].loc[date]
                        if not pd.isna(r):
                            dr += r * w_long
                for e in short_pos:
                    if e in rets and date in rets[e].index:
                        r = rets[e].loc[date]
                        if not pd.isna(r):
                            dr -= r * w_short  # Short: profit when goes down

                all_daily_rets.append(dr)

            daily_series = pd.Series(all_daily_rets, index=spy_close.loc[TRAIN_START:TEST_END].index)

            # Split into periods
            for pname, s, e in [("TRAIN", TRAIN_START, TRAIN_END), ("VALID", VALID_START, VALID_END), ("TEST", TEST_START, TEST_END)]:
                period_rets = daily_series.loc[s:e]
                m = compute_metrics(period_rets)
                if pname == "TRAIN":
                    print(f"  Mom={mom_lookback:3d}d L{n_long}/S{n_short}: Train Sh={m['sharpe']:6.3f} Vol={m['ann_vol']:.1%} CAGR={m['cagr']:.1%}", end="")
                elif pname == "VALID":
                    print(f" | Val Sh={m['sharpe']:6.3f}", end="")
                else:
                    print(f" | Test Sh={m['sharpe']:6.3f} Vol={m['ann_vol']:.1%}")

    # ================================================================
    # Approach 2: L/S with hedge overlay
    # Long top, Short bottom, PLUS safe haven when overall market weak
    # ================================================================
    print("\n" + "="*60)
    print("LS2: Long/Short Sectors + Bond/Gold overlay in bear")
    print("="*60)

    mom63 = {e: closes[e] / closes[e].shift(63) - 1 for e in available}

    for ls_pct in [0.3, 0.4, 0.5]:
        hedge_pct = 1.0 - 2 * ls_pct  # Remaining to bonds/gold

        all_daily_rets = []
        last_month = None
        long_pos = []; short_pos = []
        bear = False

        for date in spy_close.loc[TRAIN_START:TEST_END].index:
            idx = spy_close.index.get_loc(date)
            if idx < 252:
                all_daily_rets.append(0.0); continue

            month = date.month
            rebalance = (last_month is not None and month != last_month)
            last_month = month

            if rebalance:
                # Check regime
                if date in spy_sma100.index:
                    s = spy_sma100.loc[date]
                    bear = not pd.isna(s) and spy_close.loc[date] <= s

                scored = []
                for e in available:
                    if date in mom63[e].index:
                        m = mom63[e].loc[date]
                        if not pd.isna(m): scored.append((e, m))
                scored.sort(key=lambda x: x[1], reverse=True)
                if len(scored) >= 6:
                    long_pos = [e for e, _ in scored[:3]]
                    short_pos = [e for e, _ in scored[-3:]]

            dr = 0.0
            actual_ls = ls_pct
            actual_hedge = hedge_pct
            if bear:
                actual_ls = ls_pct * 0.5  # Reduce L/S in bear
                actual_hedge = 1.0 - 2 * actual_ls

            w_long = actual_ls / max(len(long_pos), 1)
            w_short = actual_ls / max(len(short_pos), 1)

            for e in long_pos:
                if e in rets and date in rets[e].index:
                    r = rets[e].loc[date]
                    if not pd.isna(r): dr += r * w_long
            for e in short_pos:
                if e in rets and date in rets[e].index:
                    r = rets[e].loc[date]
                    if not pd.isna(r): dr -= r * w_short

            # Hedge return
            for h in SAFE_HAVENS:
                if h in data and date in data[h].index:
                    si = data[h].index.get_loc(date)
                    if si > 0:
                        hr = data[h].iloc[si]["Close"] / data[h].iloc[si-1]["Close"] - 1
                        dr += hr * (actual_hedge / len(SAFE_HAVENS))

            all_daily_rets.append(dr)

        daily_series = pd.Series(all_daily_rets, index=spy_close.loc[TRAIN_START:TEST_END].index)
        for pname, s, e in [("TRAIN", TRAIN_START, TRAIN_END), ("VALID", VALID_START, VALID_END), ("TEST", TEST_START, TEST_END)]:
            period_rets = daily_series.loc[s:e]
            m = compute_metrics(period_rets)
            if pname == "TRAIN":
                print(f"  LS={ls_pct:.0%} Hedge={hedge_pct:.0%}: Train Sh={m['sharpe']:6.3f} Vol={m['ann_vol']:.1%} CAGR={m['cagr']:.1%}", end="")
            elif pname == "VALID":
                print(f" | Val Sh={m['sharpe']:6.3f}", end="")
            else:
                print(f" | Test Sh={m['sharpe']:6.3f} Vol={m['ann_vol']:.1%} CAGR={m['cagr']:.1%}")

    # ================================================================
    # Approach 3: L/S stocks (not just sectors — wider universe)
    # ================================================================
    print("\n" + "="*60)
    print("LS3: Long/Short STOCKS (98-stock universe)")
    print("="*60)

    NON_STOCKS = set(SECTOR_ETFS + SAFE_HAVENS + ["SPY","QQQ","IWM","DIA","HYG","SLV","USO"])
    stocks = [t for t in data.keys() if t not in NON_STOCKS and len(data[t]) >= 1000]
    print(f"  {len(stocks)} stocks")

    stock_closes = {t: data[t]["Close"] for t in stocks}
    stock_rets = {t: data[t]["Close"].pct_change() for t in stocks}
    stock_mom = {t: stock_closes[t] / stock_closes[t].shift(252) - 1 for t in stocks}
    stock_mom_skip = {}
    for t in stocks:
        m12 = stock_closes[t] / stock_closes[t].shift(252) - 1
        m1 = stock_closes[t] / stock_closes[t].shift(21) - 1
        stock_mom_skip[t] = m12 - m1
    stock_vol = {t: stock_rets[t].rolling(63, min_periods=21).std() * np.sqrt(252) for t in stocks}
    stock_sma200 = {t: stock_closes[t].rolling(200).mean() for t in stocks}

    for n_long, n_short in [(10, 10), (15, 15), (20, 20)]:
        all_daily_rets = []
        last_month = None
        long_pos = []; short_pos = []

        for date in spy_close.loc[TRAIN_START:TEST_END].index:
            idx = spy_close.index.get_loc(date)
            if idx < 300:
                all_daily_rets.append(0.0); continue

            month = date.month
            rebalance = (last_month is not None and month != last_month)
            last_month = month

            if rebalance:
                scored = []
                for t in stocks:
                    if t not in stock_mom_skip or date not in stock_mom_skip[t].index:
                        continue
                    ms = stock_mom_skip[t].loc[date]
                    v = stock_vol[t].loc[date] if date in stock_vol[t].index else 0
                    if pd.isna(ms) or pd.isna(v) or v <= 0:
                        continue
                    scored.append((t, ms, v))

                scored.sort(key=lambda x: x[1], reverse=True)
                if len(scored) >= n_long + n_short:
                    # Long: top momentum, weight by inverse vol
                    long_picks = scored[:n_long]
                    short_picks = scored[-n_short:]

                    # Inverse vol weights
                    long_iv = sum(1/v for _, _, v in long_picks)
                    short_iv = sum(1/v for _, _, v in short_picks)

                    long_pos = [(t, (1/v)/long_iv) for t, _, v in long_picks]
                    short_pos = [(t, (1/v)/short_iv) for t, _, v in short_picks]

            dr = 0.0
            for t, w in long_pos:
                if t in stock_rets and date in stock_rets[t].index:
                    r = stock_rets[t].loc[date]
                    if not pd.isna(r): dr += r * w * 0.5
            for t, w in short_pos:
                if t in stock_rets and date in stock_rets[t].index:
                    r = stock_rets[t].loc[date]
                    if not pd.isna(r): dr -= r * w * 0.5

            all_daily_rets.append(dr)

        daily_series = pd.Series(all_daily_rets, index=spy_close.loc[TRAIN_START:TEST_END].index)
        for pname, s, e in [("TRAIN", TRAIN_START, TRAIN_END), ("VALID", VALID_START, VALID_END), ("TEST", TEST_START, TEST_END)]:
            period_rets = daily_series.loc[s:e]
            m = compute_metrics(period_rets)
            if pname == "TRAIN":
                print(f"  L{n_long}/S{n_short}: Train Sh={m['sharpe']:6.3f} Vol={m['ann_vol']:.1%} CAGR={m['cagr']:.1%} DD={m['max_dd']:.1%}", end="")
            elif pname == "VALID":
                print(f" | Val={m['sharpe']:6.3f}", end="")
            else:
                print(f" | Test={m['sharpe']:6.3f} Vol={m['ann_vol']:.1%}")

    # ================================================================
    # Approach 4: COMBINED — L/S alpha + long-only beta hedge
    # Long best stocks, short worst, plus bonds/gold for carry
    # ================================================================
    print("\n" + "="*60)
    print("LS4: L/S Stocks + Bond/Gold carry (the full combo)")
    print("="*60)

    for n_ls in [10, 15, 20]:
        for hedge_frac in [0.0, 0.2, 0.4]:
            ls_frac = (1.0 - hedge_frac) / 2  # each side

            all_daily_rets = []
            last_month = None
            long_pos = []; short_pos = []; bear = False

            for date in spy_close.loc[TRAIN_START:TEST_END].index:
                idx = spy_close.index.get_loc(date)
                if idx < 300:
                    all_daily_rets.append(0.0); continue

                month = date.month
                rebalance = (last_month is not None and month != last_month)
                last_month = month

                if rebalance:
                    if date in spy_sma100.index:
                        s = spy_sma100.loc[date]
                        bear = not pd.isna(s) and spy_close.loc[date] <= s

                    scored = []
                    for t in stocks:
                        if t not in stock_mom_skip or date not in stock_mom_skip[t].index: continue
                        ms = stock_mom_skip[t].loc[date]
                        v = stock_vol[t].loc[date] if date in stock_vol[t].index else 0
                        if pd.isna(ms) or pd.isna(v) or v <= 0: continue
                        scored.append((t, ms, v))
                    scored.sort(key=lambda x: x[1], reverse=True)

                    if len(scored) >= 2*n_ls:
                        lp = scored[:n_ls]; sp = scored[-n_ls:]
                        liv = sum(1/v for _, _, v in lp)
                        siv = sum(1/v for _, _, v in sp)
                        long_pos = [(t, (1/v)/liv) for t, _, v in lp]
                        short_pos = [(t, (1/v)/siv) for t, _, v in sp]

                dr = 0.0
                eff_ls = ls_frac * (0.5 if bear else 1.0)
                eff_hedge = 1.0 - 2*eff_ls

                for t, w in long_pos:
                    if t in stock_rets and date in stock_rets[t].index:
                        r = stock_rets[t].loc[date]
                        if not pd.isna(r): dr += r * w * eff_ls
                for t, w in short_pos:
                    if t in stock_rets and date in stock_rets[t].index:
                        r = stock_rets[t].loc[date]
                        if not pd.isna(r): dr -= r * w * eff_ls

                if eff_hedge > 0:
                    for h in SAFE_HAVENS:
                        if h in data and date in data[h].index:
                            si = data[h].index.get_loc(date)
                            if si > 0:
                                hr = data[h].iloc[si]["Close"] / data[h].iloc[si-1]["Close"] - 1
                                dr += hr * (eff_hedge / len(SAFE_HAVENS))

                all_daily_rets.append(dr)

            daily_series = pd.Series(all_daily_rets, index=spy_close.loc[TRAIN_START:TEST_END].index)
            res = []
            for pname, s, e in [("TRAIN", TRAIN_START, TRAIN_END), ("VALID", VALID_START, VALID_END), ("TEST", TEST_START, TEST_END)]:
                res.append(compute_metrics(daily_series.loc[s:e]))
            print(f"  L/S{n_ls} Hedge={hedge_frac:.0%} Regime: Train={res[0]['sharpe']:.3f}/{res[0]['ann_vol']:.1%} Val={res[1]['sharpe']:.3f} Test={res[2]['sharpe']:.3f}/{res[2]['ann_vol']:.1%} DD={res[0]['max_dd']:.1%}/{res[2]['max_dd']:.1%}")


if __name__ == "__main__":
    print("Loading data...")
    data = load_data()
    print(f"Loaded {len(data)} tickers")
    run_tests(data)
