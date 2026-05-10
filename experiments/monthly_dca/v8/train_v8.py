"""
v8 ML re-training with stronger model + ensemble.

Reads features panel built in panel_v8.parquet, trains a walk-forward
HistGradientBoostingRegressor (and optionally LightGBM if installed) to
predict cross-sectional rank of forward returns at horizons (3, 6, 12),
producing per-(asof, ticker) prediction rows compatible with the v6 engine.

Leakage discipline: train cutoff = test_month - embargo_months.
- 12m fwd target → embargo = 13 months
- 6m fwd target → embargo = 7 months
- 3m fwd target → embargo = 4 months
"""
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

OUT = Path('experiments/monthly_dca/v8')
PANEL = OUT / 'panel_v8.parquet'

EXCLUDE = {"SPY","QQQ","IWM","VTI","RSP","DIA","BTC-USD","ETH-USD",
           "TQQQ","SQQQ","UPRO","SPXL","SPXS","TZA","TNA","SOXL","SOXS",
           "FAS","FAZ","TMF","TMV","UGL","GLL","BOIL","KOLD"}

# Horizons → embargo (months)
EMBARGO = {1: 2, 3: 4, 6: 7, 12: 13}

MODEL_KW = dict(max_iter=400, learning_rate=0.04, max_depth=6,
                min_samples_leaf=300, l2_regularization=1.0)


def main(seeds=(0, 1, 2), retrain_freq='Q', horizons=(3, 6, 12)):
    print('=== Loading panel ===')
    big = pd.read_parquet(PANEL)
    big['asof'] = pd.to_datetime(big['asof'])
    big = big[~big['ticker'].isin(EXCLUDE)].copy()
    print(f'panel: {big.shape}')

    feature_cols_raw = [c for c in big.columns
                        if c not in ('asof', 'ticker')
                        and not c.startswith('fwd_')
                        and not c.startswith('rank_target_')
                        and c != 'price']

    print(f'features: {len(feature_cols_raw)}')

    # Cross-section rank features (already independent of regime)
    print('cross-section ranking features...')
    t0 = time.time()
    for c in feature_cols_raw:
        big[c + '_xs'] = big.groupby('asof')[c].transform(
            lambda x: (x.rank(pct=True) - 0.5) * 2
        )
    print(f'  {time.time()-t0:.1f}s')
    xs_features = [c + '_xs' for c in feature_cols_raw]

    # Compute rank targets
    target_cols = {}
    for h in horizons:
        col = f'rank_target_{h}m'
        big[col] = big.groupby('asof')[f'fwd_{h}m_ret'].rank(pct=True)
        target_cols[h] = col

    # Determine retrain dates
    months = sorted(big['asof'].unique())
    months = [pd.Timestamp(m) for m in months if m >= pd.Timestamp('2003-01-01')]
    retrain_dates = []
    last_q = None
    for m in months:
        q = (m.year, (m.month - 1) // 3)  # quarter index
        if retrain_freq == 'Q':
            if last_q != q:
                retrain_dates.append(m)
                last_q = q
        elif retrain_freq == 'Y':
            if last_q is None or last_q[0] != m.year:
                retrain_dates.append(m)
                last_q = (m.year, 0)
        elif retrain_freq == 'M':
            retrain_dates.append(m)
    retrain_set = set(retrain_dates)
    print(f'retrain dates: {len(retrain_set)}')

    # Train per (horizon, seed) walk-forward
    big_sorted = big.sort_values('asof').reset_index(drop=True)
    big_sorted['asof'] = pd.to_datetime(big_sorted['asof'])

    all_preds = []
    last_models = {h: {} for h in horizons}  # {h: {seed: model}}

    for tm in months:
        # If retrain date, refit per (horizon, seed)
        if tm in retrain_set:
            for h in horizons:
                cutoff = tm - pd.DateOffset(months=EMBARGO[h])
                train = big_sorted[big_sorted['asof'] < cutoff]
                m = train[target_cols[h]].notna()
                if m.sum() < 10000:
                    continue
                Xt = train.loc[m, xs_features].values
                yt = train.loc[m, target_cols[h]].values
                for seed in seeds:
                    kw = dict(MODEL_KW)
                    kw['random_state'] = int(seed)
                    mdl = HistGradientBoostingRegressor(**kw)
                    mdl.fit(Xt, yt)
                    last_models[h][seed] = mdl
            print(f'  retrain at {tm.date()} (train rows={len(train)})')
        # Predict for tm
        test = big_sorted[big_sorted['asof'] == tm]
        if len(test) == 0:
            continue
        Xtest = test[xs_features].values
        per_h_preds = {}
        for h in horizons:
            mdls = last_models.get(h, {})
            if not mdls:
                continue
            preds = np.mean([m.predict(Xtest) for m in mdls.values()], axis=0)
            per_h_preds[h] = preds
        if not per_h_preds:
            continue
        pred_avg = np.mean(list(per_h_preds.values()), axis=0)
        rows = test[['asof', 'ticker']].copy()
        rows['fwd_1m_ret'] = test['fwd_1m_ret'].values
        rows['pred'] = pred_avg
        for h, p in per_h_preds.items():
            rows[f'pred_{h}m'] = p
        all_preds.append(rows)

    out = pd.concat(all_preds, axis=0, ignore_index=True)
    out.to_parquet(OUT / 'ml_preds_v8.parquet')
    print(f'saved {OUT / "ml_preds_v8.parquet"} ({out.shape})')

    # Quick metric: cross-section IC by month
    sub = out.dropna(subset=['pred', 'fwd_1m_ret'])
    ic = sub.groupby('asof').apply(lambda g: g[['pred','fwd_1m_ret']].rank().corr().iloc[0,1])
    print(f'mean IC: {ic.mean():.4f}, median: {ic.median():.4f}')


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--retrain', default='Q', choices=['Q', 'Y', 'M'])
    ap.add_argument('--seeds', default='0,1,2')
    ap.add_argument('--horizons', default='3,6,12')
    args = ap.parse_args()
    seeds = tuple(int(x) for x in args.seeds.split(','))
    horizons = tuple(int(x) for x in args.horizons.split(','))
    main(seeds=seeds, retrain_freq=args.retrain, horizons=horizons)
