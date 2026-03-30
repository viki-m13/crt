#!/usr/bin/env python3
"""
ML-based stock ranking for higher alpha.
Use walk-forward: train on expanding window, predict next month OOS.
Features: multiple momentum lookbacks, quality, vol, persistence.
Model: simple rank regression (no overfitting risk with regularization).
"""
import os, sys, numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prepare import load_data, TRAIN_START, TRAIN_END, VALID_START, VALID_END, TEST_START, TEST_END

BENCHMARK = "SPY"
SAFE_HAVENS = ["TLT", "GLD", "IEF"]
NON_STOCKS = set(["XLK","XLF","XLE","XLV","XLI","XLY","XLP","XLU","XLB","XLRE","XLC",
                   "TLT","GLD","IEF","SPY","QQQ","IWM","DIA","HYG","SLV","USO"])

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
    stocks = [t for t in data.keys() if t not in NON_STOCKS and len(data[t]) >= 1000]
    print(f"Stock universe: {len(stocks)} stocks")

    # Compute features for all stocks at all dates
    print("Computing features...")
    closes = {}; ret_d = {}
    for t in stocks + SAFE_HAVENS:
        if t not in data: continue
        df = data[t]; closes[t] = df["Close"]; ret_d[t] = df["Close"].pct_change()

    spy_close = data[BENCHMARK]["Close"]
    spy_ret = spy_close.pct_change()
    spy_sma100 = spy_close.rolling(100).mean()

    tlt_ret = data["TLT"]["Close"].pct_change()
    spy_tlt_corr = spy_ret.rolling(63).corr(tlt_ret)

    # Feature computation per stock per date
    def get_features(t, date):
        """Compute feature vector for stock t at date."""
        if t not in closes or date not in closes[t].index:
            return None
        c = closes[t]
        r = ret_d[t]
        idx = c.index.get_loc(date)
        if idx < 260: return None

        feats = {}
        # Momentum at multiple lookbacks
        for lb in [21, 42, 63, 126, 252]:
            if idx >= lb:
                feats[f'mom_{lb}d'] = float(c.iloc[idx] / c.iloc[idx-lb] - 1)
            else:
                return None

        # 12m - 1m skip momentum
        feats['mom_skip'] = feats['mom_252d'] - feats['mom_21d']

        # Volatility
        for lb in [21, 63]:
            vol = r.iloc[max(0,idx-lb):idx+1].std() * np.sqrt(252)
            feats[f'vol_{lb}d'] = float(vol) if not np.isnan(vol) else 0

        # Vol ratio (short/long)
        if feats['vol_63d'] > 0:
            feats['vol_ratio'] = feats['vol_21d'] / feats['vol_63d']
        else:
            feats['vol_ratio'] = 1.0

        # Quality (rolling Sharpe 63d)
        mean_r = r.iloc[max(0,idx-63):idx+1].mean() * 252
        std_r = r.iloc[max(0,idx-63):idx+1].std() * np.sqrt(252)
        feats['quality'] = float((mean_r - 0.02) / max(std_r, 0.01))

        # Persistence (frac positive days over 63d)
        recent = r.iloc[max(0,idx-63):idx+1]
        feats['persistence'] = float((recent > 0).mean())

        # Drawdown from 252d high
        high252 = c.iloc[max(0,idx-252):idx+1].max()
        feats['drawdown'] = float(c.iloc[idx] / high252 - 1)

        # Relative to SMA200
        sma = c.iloc[max(0,idx-200):idx+1].mean()
        feats['rel_sma200'] = float(c.iloc[idx] / sma - 1) if sma > 0 else 0

        # Market-relative momentum (stock mom - SPY mom)
        spy_mom = float(spy_close.iloc[idx] / spy_close.iloc[max(0,idx-63)] - 1)
        feats['rel_mom_63d'] = feats['mom_63d'] - spy_mom

        return feats

    def get_forward_return(t, date, horizon=21):
        """Get forward return for stock t from date."""
        if t not in closes or date not in closes[t].index:
            return None
        c = closes[t]
        idx = c.index.get_loc(date)
        if idx + horizon >= len(c):
            return None
        return float(c.iloc[idx+horizon] / c.iloc[idx] - 1)

    # Build training data: monthly snapshots
    print("Building training data...")
    spy_dates = spy_close.index
    monthly_dates = []
    last_month = None
    for d in spy_dates:
        if d.month != last_month:
            monthly_dates.append(d)
            last_month = d.month

    feature_names = None
    all_rows = []
    for date in monthly_dates:
        for t in stocks:
            feats = get_features(t, date)
            if feats is None: continue
            fwd = get_forward_return(t, date, 21)
            if fwd is None: continue
            if feature_names is None:
                feature_names = sorted(feats.keys())
            row = [date, t] + [feats[f] for f in feature_names] + [fwd]
            all_rows.append(row)

    cols = ['date', 'ticker'] + feature_names + ['fwd_ret']
    df_all = pd.DataFrame(all_rows, columns=cols)
    print(f"  {len(df_all)} stock-month observations, {len(feature_names)} features")

    # ================================================================
    # Walk-forward ML prediction
    # Train on expanding window, predict next month
    # ================================================================
    print("\n" + "="*60)
    print("ML WALK-FORWARD: Ridge regression for stock ranking")
    print("="*60)

    # Get rebalance dates in each period
    rebal_dates = sorted(df_all['date'].unique())

    def corr_hedge_weights(date, hedge_pct):
        corr_val = 0
        if date in spy_tlt_corr.index:
            c = spy_tlt_corr.loc[date]
            if not pd.isna(c): corr_val = c
        w = {}
        if corr_val > 0.2:
            w["GLD"] = hedge_pct * 0.60; w["IEF"] = hedge_pct * 0.40
        elif corr_val < -0.2:
            w["TLT"] = hedge_pct * 0.50; w["GLD"] = hedge_pct * 0.25; w["IEF"] = hedge_pct * 0.25
        else:
            w["TLT"] = hedge_pct * 0.33; w["GLD"] = hedge_pct * 0.34; w["IEF"] = hedge_pct * 0.33
        return w

    for alpha_val in [1.0, 10.0, 100.0]:
        print(f"\n  Ridge alpha={alpha_val}:")

        # Walk-forward predictions
        predictions = {}  # date -> [(ticker, predicted_return, inv_vol)]
        min_train_months = 36  # Need at least 3 years of training data

        for i, pred_date in enumerate(rebal_dates):
            if i < min_train_months: continue

            # Training data: all months before pred_date
            train_data = df_all[df_all['date'] < pred_date]
            if len(train_data) < 100: continue

            # Prediction data: stocks at pred_date
            pred_data = df_all[df_all['date'] == pred_date]
            if len(pred_data) < 10: continue

            X_train = train_data[feature_names].values
            y_train = train_data['fwd_ret'].values

            X_pred = pred_data[feature_names].values
            tickers_pred = pred_data['ticker'].values

            # Clean NaN/Inf
            mask = np.isfinite(X_train).all(axis=1) & np.isfinite(y_train)
            X_train = X_train[mask]
            y_train = y_train[mask]

            mask_pred = np.isfinite(X_pred).all(axis=1)
            X_pred = X_pred[mask_pred]
            tickers_pred = tickers_pred[mask_pred]

            if len(X_train) < 50 or len(X_pred) < 5: continue

            # Standardize features
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_pred_s = scaler.transform(X_pred)

            # Fit model
            model = Ridge(alpha=alpha_val)
            model.fit(X_train_s, y_train)

            # Predict
            y_pred = model.predict(X_pred_s)

            # Get inverse vol for weighting
            preds = []
            for j, t in enumerate(tickers_pred):
                vol = 1.0
                if t in ret_d and pred_date in ret_d[t].index:
                    si = ret_d[t].index.get_loc(pred_date)
                    v = ret_d[t].iloc[max(0,si-63):si+1].std() * np.sqrt(252)
                    if not np.isnan(v) and v > 0: vol = v
                preds.append((t, y_pred[j], 1.0/vol))

            predictions[pred_date] = preds

        # Backtest with ML predictions
        daily_rets = []
        current_w = {}
        last_month = None

        for date in spy_close.loc[TRAIN_START:TEST_END].index:
            idx_spy = spy_close.index.get_loc(date)
            if idx_spy < 300:
                daily_rets.append(0.0); continue

            month = date.month
            rebalance = (last_month is not None and month != last_month)
            last_month = month

            if rebalance:
                # Find nearest prediction date
                pred_date = None
                for d in sorted(predictions.keys()):
                    if d <= date:
                        pred_date = d
                if pred_date is None:
                    daily_rets.append(0.0); continue

                preds = predictions.get(pred_date, [])

                # Regime
                bear = False
                if date in spy_sma100.index:
                    s = spy_sma100.loc[date]
                    bear = not pd.isna(s) and spy_close.loc[date] <= s

                eq_pct = 0.30 if bear else 0.80
                hedge_pct = 1.0 - eq_pct
                n_stocks = 10 if bear else 20

                # Rank by ML prediction, take top N with positive prediction
                preds.sort(key=lambda x: x[1], reverse=True)
                top = [(t, p, iv) for t, p, iv in preds if p > 0][:n_stocks]

                weights = {}
                if top:
                    ti = sum(iv for _, _, iv in top)
                    for t, _, iv in top:
                        weights[t] = (iv/ti) * eq_pct
                else:
                    weights["SPY"] = eq_pct

                hw = corr_hedge_weights(date, hedge_pct)
                for k, v in hw.items():
                    weights[k] = weights.get(k, 0) + v

                # Apply with close-to-close (MOC) + tx costs
                dr = 0.0
                for t, w in weights.items():
                    df = data.get(t)
                    if df is not None and date in df.index:
                        si = df.index.get_loc(date)
                        if si > 0:
                            dr += (df.iloc[si]["Close"] / df.iloc[si-1]["Close"] - 1) * w
                # TX costs on changes
                for t in set(list(current_w.keys()) + list(weights.keys())):
                    old_w = current_w.get(t, 0)
                    new_w = weights.get(t, 0)
                    if abs(old_w - new_w) > 0.005:
                        dr -= abs(old_w - new_w) * 0.0003  # 3bps

                daily_rets.append(dr)
                current_w = weights
            else:
                if current_w:
                    dr = 0.0
                    for t, w in current_w.items():
                        df = data.get(t)
                        if df is not None and date in df.index:
                            si = df.index.get_loc(date)
                            if si > 0:
                                dr += (df.iloc[si]["Close"] / df.iloc[si-1]["Close"] - 1) * w
                    daily_rets.append(dr)
                else:
                    daily_rets.append(0.0)

        rets = pd.Series(daily_rets, index=spy_close.loc[TRAIN_START:TEST_END].index)

        for name, s, e in [("TRAIN", TRAIN_START, TRAIN_END), ("VALID", VALID_START, VALID_END), ("TEST", TEST_START, TEST_END)]:
            r = rets.loc[s:e]
            m = compute_metrics(r)
            spy_r = data[BENCHMARK].loc[s:e, "Close"].pct_change().dropna()
            sm = compute_metrics(spy_r)
            print(f"    {name}: Sharpe={m['sharpe']:.3f} CAGR={m['cagr']:.1%} MaxDD={m['max_dd']:.1%} Vol={m['ann_vol']:.1%} | SPY Sh={sm['sharpe']:.3f}")

    # ================================================================
    # Also test: NON-ML ensemble ranker (our best hand-crafted)
    # for comparison
    # ================================================================
    print("\n" + "="*60)
    print("COMPARISON: Hand-crafted ensemble (no ML)")
    print("="*60)

    # Same as CH2 from research_corr_hedge.py
    daily_rets_hc = []
    current_w = {}
    last_month = None

    for date in spy_close.loc[TRAIN_START:TEST_END].index:
        idx_spy = spy_close.index.get_loc(date)
        if idx_spy < 300:
            daily_rets_hc.append(0.0); continue

        month = date.month
        rebalance = (last_month is not None and month != last_month)
        last_month = month

        if rebalance:
            bear = False
            if date in spy_sma100.index:
                s = spy_sma100.loc[date]
                bear = not pd.isna(s) and spy_close.loc[date] <= s

            eq_pct = 0.30 if bear else 0.80
            hedge_pct = 1.0 - eq_pct

            scored = []
            for t in stocks:
                feats = get_features(t, date)
                if feats is None: continue
                if feats['mom_skip'] <= 0 or feats['quality'] <= 0: continue
                if feats['rel_sma200'] < 0: continue
                moms = [feats[f'mom_{lb}d'] for lb in [63, 126] if f'mom_{lb}d' in feats]
                moms.append(feats['mom_skip'])
                avg_mom = np.mean([m for m in moms if m > 0]) if moms else 0
                if avg_mom <= 0: continue
                composite = avg_mom * max(feats['quality'], 0.01)
                scored.append((t, composite, 1.0/max(feats['vol_63d'], 0.01)))

            scored.sort(key=lambda x: x[1], reverse=True)
            top = scored[:20 if not bear else 10]

            weights = {}
            if top:
                ti = sum(iv for _, _, iv in top)
                for t, _, iv in top: weights[t] = (iv/ti) * eq_pct
            else:
                weights["SPY"] = eq_pct

            hw = corr_hedge_weights(date, hedge_pct)
            for k, v in hw.items(): weights[k] = weights.get(k, 0) + v

            dr = 0.0
            for t, w in weights.items():
                df = data.get(t)
                if df is not None and date in df.index:
                    si = df.index.get_loc(date)
                    if si > 0: dr += (df.iloc[si]["Close"] / df.iloc[si-1]["Close"] - 1) * w
            for t in set(list(current_w.keys()) + list(weights.keys())):
                if abs(current_w.get(t, 0) - weights.get(t, 0)) > 0.005:
                    dr -= abs(current_w.get(t, 0) - weights.get(t, 0)) * 0.0003
            daily_rets_hc.append(dr)
            current_w = weights
        else:
            if current_w:
                dr = 0.0
                for t, w in current_w.items():
                    df = data.get(t)
                    if df is not None and date in df.index:
                        si = df.index.get_loc(date)
                        if si > 0: dr += (df.iloc[si]["Close"] / df.iloc[si-1]["Close"] - 1) * w
                daily_rets_hc.append(dr)
            else:
                daily_rets_hc.append(0.0)

    rets_hc = pd.Series(daily_rets_hc, index=spy_close.loc[TRAIN_START:TEST_END].index)
    for name, s, e in [("TRAIN", TRAIN_START, TRAIN_END), ("VALID", VALID_START, VALID_END), ("TEST", TEST_START, TEST_END)]:
        r = rets_hc.loc[s:e]
        m = compute_metrics(r)
        print(f"  {name}: Sharpe={m['sharpe']:.3f} CAGR={m['cagr']:.1%} MaxDD={m['max_dd']:.1%} Vol={m['ann_vol']:.1%}")

    # Full period
    r = rets_hc
    m = compute_metrics(r)
    print(f"  FULL: Sharpe={m['sharpe']:.3f} CAGR={m['cagr']:.1%} MaxDD={m['max_dd']:.1%}")


if __name__ == "__main__":
    print("Loading data...")
    data = load_data()
    print(f"Loaded {len(data)} tickers")
    run_tests(data)
