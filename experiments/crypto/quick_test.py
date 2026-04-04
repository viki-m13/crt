#!/usr/bin/env python3
"""Quick focused test of best dip-buying configs on all periods."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from train_hf import *

print("Loading data...")
data, _ = load_or_download()
print(f"  {len(data)} tickers")

print("Computing features...")
features_dict = compute_all_enhanced(data)
feature_cols = get_feature_cols(features_dict)
print(f"  {len(feature_cols)} features")

# Quick training (minimal)
print("Training models...")
cfg = HFConfig()
purge = str((pd.Timestamp(TRAIN_END) - pd.Timedelta(days=30)).date())
X_tr, y_tr, _ = create_dataset(data, features_dict, TRAIN_START, purge,
                                hold_days=7, feature_cols=feature_cols, sample_every=3)
X_va, y_va, _ = create_dataset(data, features_dict, VALID_START, VALID_END,
                                hold_days=7, feature_cols=feature_cols, sample_every=3)
scaler = StandardScaler()
X_tr_s = np.nan_to_num(scaler.fit_transform(X_tr), nan=0, posinf=3, neginf=-3)
X_va_s = np.nan_to_num(scaler.transform(X_va), nan=0, posinf=3, neginf=-3)

tab_model, device = train_tabtransformer(X_tr_s, y_tr, X_va_s, y_va, cfg)
xgb_model = train_xgboost(X_tr, y_tr, X_va, y_va, cfg)
lgb_model = train_lgbm(X_tr, y_tr, X_va, y_va, cfg)
models = {"tab": (tab_model, device), "xgb": xgb_model, "lgb": lgb_model}

# Test best configs
print("\n" + "="*70)
print("TESTING BEST CONFIGS ON ALL PERIODS")
print("="*70)

configs = [
    {"tp": 0.06, "sl": -0.20, "hold": 21, "name": "TP6/SL20/H21"},
    {"tp": 0.08, "sl": -0.20, "hold": 21, "name": "TP8/SL20/H21"},
    {"tp": 0.10, "sl": -0.20, "hold": 30, "name": "TP10/SL20/H30"},
    {"tp": 0.06, "sl": -0.15, "hold": 21, "name": "TP6/SL15/H21"},
    {"tp": 0.08, "sl": -0.15, "hold": 21, "name": "TP8/SL15/H21"},
    {"tp": 0.10, "sl": -0.15, "hold": 30, "name": "TP10/SL15/H30"},
    {"tp": 0.12, "sl": -0.20, "hold": 30, "name": "TP12/SL20/H30"},
]

for c in configs:
    cfg2 = HFConfig()
    cfg2.take_profit = c["tp"]; cfg2.stop_loss = c["sl"]; cfg2.max_hold = c["hold"]

    results = {}
    for pname, start, end in [("Train", TRAIN_START, TRAIN_END),
                               ("Valid", VALID_START, VALID_END),
                               ("Test", TEST_START, TEST_END)]:
        tr, dr = run_weekly_backtest(data, features_dict, models, scaler,
                                      feature_cols, start, end, cfg2)
        m = evaluate_strategy(tr, dr, "")
        results[pname] = m

    t = results["Train"]; v = results["Valid"]; te = results["Test"]
    print(f"\n{c['name']}:")
    print(f"  Train:  Sharpe={t.get('sharpe',0):.3f} WR={t.get('win_rate',0):.2%} N={t.get('n_trades',0)}")
    print(f"  Valid:  Sharpe={v.get('sharpe',0):.3f} WR={v.get('win_rate',0):.2%} N={v.get('n_trades',0)}")
    print(f"  Test:   Sharpe={te.get('sharpe',0):.3f} WR={te.get('win_rate',0):.2%} N={te.get('n_trades',0)}")
