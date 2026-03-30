#!/usr/bin/env python3
"""
FINAL PUSH: Maximum Sharpe through multi-frequency alpha stacking.

Layer 1: Stock momentum-quality (monthly rebal, 60% weight)
Layer 2: Short-term sector reversal (weekly, 15% weight)
Layer 3: Cross-asset trend (monthly, safe haven allocation)
Layer 4: Vol-scaling overlay (daily, scales total exposure)

Each layer targets different alpha source:
- L1: Cross-sectional momentum (months-horizon)
- L2: Mean reversion (week-horizon)
- L3: Risk management (regime-dependent)
- L4: Volatility timing (reduces in crisis)
"""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prepare import load_data, TRAIN_START, TRAIN_END, VALID_START, VALID_END, TEST_START, TEST_END

BENCHMARK = "SPY"
SAFE_HAVENS = ["TLT", "GLD", "IEF"]
SECTOR_ETFS = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB"]
NON_STOCKS = set(SECTOR_ETFS + ["XLRE", "XLC"] + SAFE_HAVENS + ["SPY","QQQ","IWM","DIA","HYG","SLV","USO"])

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


def run(data):
    stocks = [t for t in data.keys() if t not in NON_STOCKS and len(data[t]) >= 1000]
    print(f"{len(stocks)} stocks, {len([e for e in SECTOR_ETFS if e in data])} sectors")

    closes = {}; ret_d = {}; vol63 = {}
    for t in stocks + SECTOR_ETFS + SAFE_HAVENS:
        if t not in data: continue
        closes[t] = data[t]["Close"]; ret_d[t] = data[t]["Close"].pct_change()
        vol63[t] = ret_d[t].rolling(63, min_periods=21).std() * np.sqrt(252)

    spy_close = data[BENCHMARK]["Close"]
    spy_ret = spy_close.pct_change()
    spy_sma100 = spy_close.rolling(100).mean()
    spy_vol21 = spy_ret.rolling(21).std() * np.sqrt(252)

    tlt_ret = data["TLT"]["Close"].pct_change()
    spy_tlt_corr = spy_ret.rolling(63).corr(tlt_ret)

    # Precompute stock features
    mom_skip = {}; quality_s = {}; sma200 = {}
    for t in stocks:
        c = closes[t]; r = ret_d[t]
        mom_skip[t] = (c / c.shift(252) - 1) - (c / c.shift(21) - 1)
        m63 = r.rolling(63, min_periods=42).mean() * 252
        s63 = r.rolling(63, min_periods=42).std() * np.sqrt(252)
        quality_s[t] = (m63 - 0.02) / s63.clip(lower=0.01)
        sma200[t] = c.rolling(200).mean()

    # Additional momentum lookbacks for ensemble
    mom63 = {t: closes[t] / closes[t].shift(63) - 1 for t in stocks if t in closes}
    mom126 = {t: closes[t] / closes[t].shift(126) - 1 for t in stocks if t in closes}

    # Sector 5d returns (for reversal signal)
    sec_mom5 = {e: data[e]["Close"] / data[e]["Close"].shift(5) - 1 for e in SECTOR_ETFS if e in data}
    sec_mom63 = {e: data[e]["Close"] / data[e]["Close"].shift(63) - 1 for e in SECTOR_ETFS if e in data}

    # ================================================================
    # Multi-layer strategy
    # ================================================================

    # State
    stock_weights = {}      # Monthly
    sector_tilt = {}        # Weekly
    hedge_weights = {}      # Monthly

    last_month = None
    last_week = None

    daily_rets = []

    for date in spy_close.loc[TRAIN_START:TEST_END].index:
        idx = spy_close.index.get_loc(date)
        if idx < 300:
            daily_rets.append(0.0); continue

        month = date.month
        week = date.isocalendar()[1]
        monthly_rebal = (last_month is not None and month != last_month)
        weekly_rebal = (last_week is not None and week != last_week and date.weekday() <= 1)
        last_month = month
        last_week = week

        # === REGIME ===
        bear = False
        if date in spy_sma100.index:
            s = spy_sma100.loc[date]
            bear = not pd.isna(s) and spy_close.loc[date] <= s

        # === LAYER 1: Monthly stock momentum (60% of portfolio) ===
        if monthly_rebal:
            eq_pct = 0.25 if bear else 0.65
            scored = []
            for t in stocks:
                if t not in mom_skip or date not in mom_skip[t].index: continue
                ms = mom_skip[t].loc[date]
                q = quality_s[t].loc[date] if date in quality_s[t].index else 0
                v = vol63[t].loc[date] if date in vol63[t].index else 0
                sm = sma200[t].loc[date] if date in sma200[t].index else 0
                price = closes[t].loc[date] if date in closes[t].index else 0
                if pd.isna(ms) or pd.isna(v) or v <= 0.01 or pd.isna(q): continue
                if ms <= 0 or q <= 0: continue
                if not pd.isna(sm) and price <= sm: continue
                # Ensemble momentum
                moms = [ms]
                if t in mom63 and date in mom63[t].index:
                    m = mom63[t].loc[date]
                    if not pd.isna(m): moms.append(m)
                if t in mom126 and date in mom126[t].index:
                    m = mom126[t].loc[date]
                    if not pd.isna(m): moms.append(m)
                avg_mom = np.mean([m for m in moms if m > 0]) if moms else 0
                if avg_mom <= 0: continue
                composite = avg_mom * max(q, 0.01)
                scored.append((t, composite, 1.0/v))

            scored.sort(key=lambda x: x[1], reverse=True)
            top = scored[:20 if not bear else 8]
            stock_weights = {}
            if top:
                ti = sum(iv for _, _, iv in top)
                for t, _, iv in top: stock_weights[t] = (iv/ti) * eq_pct

            # === LAYER 3: Hedge allocation ===
            hedge_pct = 1.0 - eq_pct - 0.15  # Reserve 15% for sector tilt
            corr_val = spy_tlt_corr.loc[date] if date in spy_tlt_corr.index else 0
            if pd.isna(corr_val): corr_val = 0

            hedge_weights = {}
            if corr_val > 0.2:
                hedge_weights["GLD"] = hedge_pct * 0.60
                hedge_weights["IEF"] = hedge_pct * 0.40
            elif corr_val < -0.2:
                hedge_weights["TLT"] = hedge_pct * 0.50
                hedge_weights["GLD"] = hedge_pct * 0.25
                hedge_weights["IEF"] = hedge_pct * 0.25
            else:
                hedge_weights["TLT"] = hedge_pct * 0.33
                hedge_weights["GLD"] = hedge_pct * 0.34
                hedge_weights["IEF"] = hedge_pct * 0.33

        # === LAYER 2: Weekly sector reversal tilt (15% of portfolio) ===
        if weekly_rebal or monthly_rebal:
            tilt_pct = 0.05 if bear else 0.15
            # Buy sectors that dipped most in last 5 days (mean reversion)
            # among sectors with positive 63d momentum (uptrend)
            sec_scored = []
            for e in SECTOR_ETFS:
                if e not in sec_mom5 or e not in sec_mom63: continue
                if date not in sec_mom5[e].index or date not in sec_mom63[e].index: continue
                m5 = sec_mom5[e].loc[date]
                m63 = sec_mom63[e].loc[date]
                if pd.isna(m5) or pd.isna(m63): continue
                if m63 <= 0: continue  # Only sectors in uptrend
                # Score: negative 5d return = dip = buy opportunity
                score = -m5  # More negative = bigger dip = higher score
                if score > 0:  # Must have actually dipped
                    sec_scored.append((e, score))

            sec_scored.sort(key=lambda x: x[1], reverse=True)
            top_sec = sec_scored[:3] if sec_scored else []

            sector_tilt = {}
            if top_sec:
                w = tilt_pct / len(top_sec)
                for e, _ in top_sec:
                    sector_tilt[e] = w
            else:
                # No dips: spread across all uptrending sectors
                up_secs = [e for e in SECTOR_ETFS if e in sec_mom63 and date in sec_mom63[e].index
                          and not pd.isna(sec_mom63[e].loc[date]) and sec_mom63[e].loc[date] > 0]
                if up_secs:
                    w = tilt_pct / len(up_secs)
                    for e in up_secs: sector_tilt[e] = w
                else:
                    # All sectors down: add to hedge
                    for h in hedge_weights:
                        hedge_weights[h] = hedge_weights.get(h, 0) + tilt_pct / len(hedge_weights)

        # === LAYER 4: Vol scaling (daily) ===
        vol_scale = 1.0
        if date in spy_vol21.index:
            v = spy_vol21.loc[date]
            if not pd.isna(v) and v > 0:
                # Target 10% portfolio vol; scale down when market vol is high
                vol_scale = min(0.10 / v, 1.0)
                # Only scale down significantly if vol is very high
                if vol_scale > 0.8:
                    vol_scale = 1.0  # Don't scale if vol is reasonable

        # === COMBINE ALL LAYERS ===
        combined = {}
        for t, w in stock_weights.items():
            combined[t] = combined.get(t, 0) + w * vol_scale
        for e, w in sector_tilt.items():
            combined[e] = combined.get(e, 0) + w * vol_scale
        for h, w in hedge_weights.items():
            combined[h] = combined.get(h, 0) + w

        # If vol_scale reduced equity, add remainder to hedge
        eq_reduced = sum(stock_weights.values()) * (1 - vol_scale) + sum(sector_tilt.values()) * (1 - vol_scale)
        if eq_reduced > 0.01:
            for h in hedge_weights:
                combined[h] = combined.get(h, 0) + eq_reduced * (hedge_weights[h] / max(sum(hedge_weights.values()), 0.01))

        # Daily return
        dr = 0.0
        for t, w in combined.items():
            df = data.get(t)
            if df is not None and date in df.index:
                si = df.index.get_loc(date)
                if si > 0:
                    dr += (df.iloc[si]["Close"] / df.iloc[si-1]["Close"] - 1) * w
        daily_rets.append(dr)

    rets = pd.Series(daily_rets, index=spy_close.loc[TRAIN_START:TEST_END].index)

    print("\n" + "="*60)
    print("MULTI-LAYER STRATEGY RESULTS")
    print("="*60)
    for name, s, e in [("TRAIN", TRAIN_START, TRAIN_END), ("VALID", VALID_START, VALID_END), ("TEST", TEST_START, TEST_END)]:
        r = rets.loc[s:e]
        m = compute_metrics(r)
        sm = compute_metrics(data[BENCHMARK].loc[s:e, "Close"].pct_change().dropna())
        print(f"  {name}: Sharpe={m['sharpe']:.3f} CAGR={m['cagr']:.1%} MaxDD={m['max_dd']:.1%} Vol={m['ann_vol']:.1%} Sortino={m['sortino']:.3f}")
        print(f"        SPY: Sharpe={sm['sharpe']:.3f} CAGR={sm['cagr']:.1%}")

    # Full period
    m = compute_metrics(rets)
    sm = compute_metrics(data[BENCHMARK].loc[TRAIN_START:TEST_END, "Close"].pct_change().dropna())
    print(f"  FULL: Sharpe={m['sharpe']:.3f} CAGR={m['cagr']:.1%} MaxDD={m['max_dd']:.1%} Vol={m['ann_vol']:.1%}")
    print(f"        SPY: Sharpe={sm['sharpe']:.3f} CAGR={sm['cagr']:.1%}")

    # Walk-forward
    print(f"\n  Walk-forward:")
    sharpes = []
    for year in range(2011, 2026):
        s, e = f"{year}-01-01", f"{year}-12-31"
        r = rets.loc[s:e]
        if len(r) < 100: continue
        m = compute_metrics(r)
        sm = compute_metrics(data[BENCHMARK].loc[s:e, "Close"].pct_change().dropna())
        sharpes.append(m["sharpe"])
        beat = "✓" if m["sharpe"] > sm["sharpe"] else " "
        print(f"    {year}: Sh={m['sharpe']:6.3f} vs SPY {sm['sharpe']:6.3f} {beat} | DD={m['max_dd']:6.1%} vs {sm['max_dd']:6.1%} | CAGR={m['cagr']:5.1%}")
    if sharpes:
        print(f"    Avg={np.mean(sharpes):.3f} Min={min(sharpes):.3f} Max={max(sharpes):.3f}")
        # Years above 2.0 and 3.0
        above2 = sum(1 for s in sharpes if s >= 2.0)
        above3 = sum(1 for s in sharpes if s >= 3.0)
        print(f"    Years with Sharpe >= 2.0: {above2}/{len(sharpes)}")
        print(f"    Years with Sharpe >= 3.0: {above3}/{len(sharpes)}")


if __name__ == "__main__":
    print("Loading data...")
    data = load_data()
    print(f"Loaded {len(data)} tickers")
    run(data)
