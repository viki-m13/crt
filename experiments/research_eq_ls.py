#!/usr/bin/env python3
"""
Equities-only: market-neutral L/S with multi-factor + long overlay.
"""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prepare import load_data, TRAIN_START, TRAIN_END, VALID_START, VALID_END, TEST_START, TEST_END

BENCHMARK = "SPY"
SECTOR_ETFS = ["XLK","XLF","XLE","XLV","XLI","XLY","XLP","XLU","XLB","XLRE","XLC"]
NON_STOCKS = set(SECTOR_ETFS + ["TLT","GLD","IEF","SPY","QQQ","IWM","DIA","HYG","SLV","USO"])

def compute_metrics(rets, rf=0.02):
    if len(rets) == 0 or rets.std() == 0:
        return {"sharpe": 0, "cagr": 0, "max_dd": 0, "ann_vol": 0}
    excess = rets - rf / 252
    n_years = len(rets) / 252
    sharpe = excess.mean() / excess.std() * np.sqrt(252)
    cum = (1 + rets).cumprod()
    total = cum.iloc[-1] - 1
    cagr = (1 + total) ** (1 / n_years) - 1 if n_years >= 1 else total
    mdd = ((cum - cum.cummax()) / cum.cummax()).min()
    return {"sharpe": round(float(sharpe), 3), "cagr": round(float(cagr), 4),
            "max_dd": round(float(mdd), 4), "ann_vol": round(float(rets.std() * np.sqrt(252)), 4)}

def run(data):
    stocks = [t for t in data.keys() if t not in NON_STOCKS and len(data[t]) >= 1000]
    print(f"{len(stocks)} stocks")

    closes = {}; ret_d = {}; vol63 = {}
    for t in stocks:
        if t not in data: continue
        closes[t] = data[t]["Close"]; ret_d[t] = data[t]["Close"].pct_change()
        vol63[t] = ret_d[t].rolling(63, min_periods=21).std() * np.sqrt(252)

    spy_close = data[BENCHMARK]["Close"]
    spy_sma100 = spy_close.rolling(100).mean()
    sma200 = {t: closes[t].rolling(200).mean() for t in stocks}

    stock_mom63 = {t: closes[t] / closes[t].shift(63) - 1 for t in stocks}
    stock_mom126 = {t: closes[t] / closes[t].shift(126) - 1 for t in stocks}
    stock_mom252 = {t: closes[t] / closes[t].shift(252) - 1 for t in stocks}
    stock_mom21 = {t: closes[t] / closes[t].shift(21) - 1 for t in stocks}
    quality = {}
    for t in stocks:
        m63 = ret_d[t].rolling(63, min_periods=42).mean() * 252
        s63 = ret_d[t].rolling(63, min_periods=42).std() * np.sqrt(252)
        quality[t] = (m63 - 0.02) / s63.clip(lower=0.01)

    # ================================================================
    # Test different L/S + long-only combinations
    # ================================================================
    print("\n" + "="*60)
    print("EQUITIES-ONLY STRATEGY GRID")
    print("  L/S: market-neutral momentum spread")
    print("  Long: regime-gated long-only momentum")
    print("="*60)

    for ls_pct in [0.0, 0.25, 0.50]:
        for lo_pct in [0.0, 0.50, 0.75, 1.0]:
            if ls_pct == 0 and lo_pct == 0: continue
            if ls_pct * 2 + lo_pct > 1.5: continue  # Reasonable leverage limit

            for n in [10, 20]:
                all_rets = []
                last_month = None
                ls_l = []; ls_s = []; lo = {}

                for date in spy_close.loc[TRAIN_START:TEST_END].index:
                    idx = spy_close.index.get_loc(date)
                    if idx < 300: all_rets.append(0.0); continue

                    month = date.month
                    rebal = (last_month is not None and month != last_month)
                    last_month = month

                    if rebal:
                        # Score all stocks
                        scored = []
                        for t in stocks:
                            if date not in stock_mom63[t].index: continue
                            m3 = stock_mom63[t].loc[date]
                            m6 = stock_mom126[t].loc[date] if date in stock_mom126[t].index else None
                            m12 = stock_mom252[t].loc[date] if date in stock_mom252[t].index else None
                            m1 = stock_mom21[t].loc[date] if date in stock_mom21[t].index else 0
                            v = vol63[t].loc[date] if date in vol63[t].index else 0
                            q = quality[t].loc[date] if date in quality[t].index else 0
                            if pd.isna(m3) or pd.isna(v) or v <= 0.01: continue
                            moms = [m3]
                            if m6 is not None and not pd.isna(m6): moms.append(m6)
                            if m12 is not None and not pd.isna(m12) and not pd.isna(m1):
                                moms.append(m12 - m1)
                            avg = np.mean(moms)
                            qw = max(abs(q), 0.01) * np.sign(q) if not pd.isna(q) and q != 0 else 0.01
                            score = avg * qw
                            scored.append((t, score, v, q))

                        scored.sort(key=lambda x: x[1], reverse=True)

                        # L/S picks
                        if ls_pct > 0 and len(scored) >= 2*n:
                            lp = scored[:n]; sp = scored[-n:]
                            l_iv = sum(1/v for _, _, v, _ in lp)
                            s_iv = sum(1/v for _, _, v, _ in sp)
                            ls_l = [(t, (1/v)/l_iv) for t, _, v, _ in lp]
                            ls_s = [(t, (1/v)/s_iv) for t, _, v, _ in sp]
                        else:
                            ls_l = []; ls_s = []

                        # Long-only (regime-gated)
                        lo = {}
                        if lo_pct > 0:
                            bear = False
                            if date in spy_sma100.index:
                                s = spy_sma100.loc[date]
                                bear = not pd.isna(s) and spy_close.loc[date] <= s
                            if not bear:
                                top_long = [(t, sc, v) for t, sc, v, q in scored if sc > 0 and (not pd.isna(q)) and q > 0]
                                sm_t = [t for t, _, _, _ in scored if t in sma200 and date in sma200[t].index and
                                        date in closes[t].index and closes[t].loc[date] > sma200[t].loc[date]]
                                top_long = [(t,s,v) for t,s,v in top_long if t in sm_t][:n]
                                if top_long:
                                    ti = sum(1/v for _, _, v in top_long)
                                    lo = {t: (1/v)/ti for t, _, v in top_long}

                    dr = 0.0
                    for t, w in ls_l:
                        if t in ret_d and date in ret_d[t].index:
                            r = ret_d[t].loc[date]
                            if not pd.isna(r): dr += r * w * ls_pct
                    for t, w in ls_s:
                        if t in ret_d and date in ret_d[t].index:
                            r = ret_d[t].loc[date]
                            if not pd.isna(r): dr -= r * w * ls_pct
                    for t, w in lo.items():
                        if t in ret_d and date in ret_d[t].index:
                            r = ret_d[t].loc[date]
                            if not pd.isna(r): dr += r * w * lo_pct
                    all_rets.append(dr)

                rets = pd.Series(all_rets, index=spy_close.loc[TRAIN_START:TEST_END].index)
                res = []
                for _, s, e in [("T", TRAIN_START, TRAIN_END), ("V", VALID_START, VALID_END), ("S", TEST_START, TEST_END)]:
                    res.append(compute_metrics(rets.loc[s:e]))
                label = f"LS={ls_pct:.0%}x2 LO={lo_pct:.0%} N={n}"
                print(f"  {label:25s}: Tr={res[0]['sharpe']:6.3f}/{res[0]['ann_vol']:.1%}/{res[0]['cagr']:.1%} "
                      f"Va={res[1]['sharpe']:6.3f} Te={res[2]['sharpe']:6.3f}/{res[2]['ann_vol']:.1%}/{res[2]['cagr']:.1%}")


if __name__ == "__main__":
    print("Loading data...")
    data = load_data()
    print(f"Loaded {len(data)} tickers")
    run(data)
