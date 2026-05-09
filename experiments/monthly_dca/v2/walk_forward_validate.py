"""
Walk-forward validation: split the full window into TRAIN/TEST blocks and
re-evaluate the v2 strategy on each TEST block independently.

This protects against the criticism that the model "saw the future" because
training is restricted to data strictly older than the test block start.

Splits:
  A1  TRAIN 2003-2010  TEST 2011-2018
  A2  TRAIN 2003-2014  TEST 2015-2021
  A3  TRAIN 2003-2017  TEST 2018-2024
  R1  TRAIN 2003-2007  TEST 2008-2010 (GFC)
  R2  TRAIN 2005-2010  TEST 2011-2013
  R3  TRAIN 2008-2013  TEST 2014-2016
  R4  TRAIN 2011-2016  TEST 2017-2019
  R5  TRAIN 2014-2019  TEST 2020-2022 (COVID + 2022 bear)
  R6  TRAIN 2017-2022  TEST 2023-2024 (AI rally)
  STRICT TRAIN 2003-2020 TEST 2021-2024

For each split we:
  1. Re-fit the model on TRAIN
  2. Predict on TEST
  3. Apply regime gate + conviction weighting (frozen choice)
  4. Compute CAGR, Sharpe, MaxDD, edge vs SPY DCA on TEST

Output:
  cache/v2/wf_v2_train.csv, wf_v2_test.csv, wf_v2_aggregate.csv

Run: python3 -m experiments.monthly_dca.v2.walk_forward_validate
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from experiments.monthly_dca.v2.ml_strategy import (
    OUT, build_strategy_outputs, simulate_strategy, cagr,
    EXCLUDE,
)


SPLITS = [
    ("A1",     "2003-01-01", "2010-12-31", "2011-01-01", "2018-12-31"),
    ("A2",     "2003-01-01", "2014-12-31", "2015-01-01", "2021-12-31"),
    ("A3",     "2003-01-01", "2017-12-31", "2018-01-01", "2024-12-31"),
    ("R1",     "2003-01-01", "2007-12-31", "2008-01-01", "2010-12-31"),
    ("R2",     "2005-01-01", "2010-12-31", "2011-01-01", "2013-12-31"),
    ("R3",     "2008-01-01", "2013-12-31", "2014-01-01", "2016-12-31"),
    ("R4",     "2011-01-01", "2016-12-31", "2017-01-01", "2019-12-31"),
    ("R5",     "2014-01-01", "2019-12-31", "2020-01-01", "2022-12-31"),
    ("R6",     "2017-01-01", "2022-12-31", "2023-01-01", "2024-12-31"),
    ("STRICT", "2003-01-01", "2020-12-31", "2021-01-01", "2024-12-31"),
]


def fit_predict_block(big: pd.DataFrame, train_start: str, train_end: str,
                       test_start: str, test_end: str,
                       target_horizons=(1, 3, 6),
                       embargo_months: int = 7,
                       model_kwargs=None) -> pd.DataFrame:
    """Fit on training block, predict on TEST."""
    if model_kwargs is None:
        model_kwargs = dict(
            max_iter=300, learning_rate=0.04, max_depth=6,
            min_samples_leaf=300, l2_regularization=1.0
        )
    big = big.reset_index()
    big["asof"] = pd.to_datetime(big["asof"])
    big = big[~big["ticker"].isin(EXCLUDE)].copy()

    # Compute targets
    target_cols = [f"rank_target_{h}m" for h in target_horizons]
    fwd_cols = [f"fwd_{h}m_ret" for h in target_horizons]
    for h, tc, fc in zip(target_horizons, target_cols, fwd_cols):
        big[tc] = big.groupby("asof")[fc].rank(pct=True)

    feature_cols_raw = [c for c in big.columns
                        if c not in ("asof", "ticker") and not c.startswith("fwd_")
                        and not c.startswith("rank_target_")]
    # Cross-sectional ranking
    for c in feature_cols_raw:
        big[c + "_xs"] = big.groupby("asof")[c].transform(lambda x: (x.rank(pct=True) - 0.5) * 2)
    xs_features = [c + "_xs" for c in feature_cols_raw]

    train_start_ts = pd.Timestamp(train_start)
    train_end_ts = pd.Timestamp(train_end)
    test_start_ts = pd.Timestamp(test_start)
    test_end_ts = pd.Timestamp(test_end)

    # Hard cutoff: training data must be older than test_start - embargo
    train_cutoff = test_start_ts - pd.DateOffset(months=embargo_months)
    train_subset = big[(big["asof"] >= train_start_ts) & (big["asof"] < train_cutoff) &
                       (big["asof"] <= train_end_ts)]
    if len(train_subset) < 30000:
        return pd.DataFrame()
    models = {}
    for h, tc in zip(target_horizons, target_cols):
        m = train_subset[tc].notna()
        if m.sum() < 10000:
            continue
        Xt = train_subset.loc[m, xs_features].values
        yt = train_subset.loc[m, tc].values
        mdl = HistGradientBoostingRegressor(**model_kwargs)
        mdl.fit(Xt, yt)
        models[h] = mdl
    test_subset = big[(big["asof"] >= test_start_ts) & (big["asof"] <= test_end_ts)]
    if test_subset.empty:
        return pd.DataFrame()
    Xtest = test_subset[xs_features].values
    per_horizon = {h: models[h].predict(Xtest) for h in target_horizons if h in models}
    pred_avg = np.mean(list(per_horizon.values()), axis=0)
    out = test_subset[["asof", "ticker", "fwd_1m_ret"]].assign(pred=pred_avg)
    for h, p in per_horizon.items():
        out[f"pred_{h}m"] = p
    return out


def spy_dca(monthly_returns: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Compute SPY DCA equity curve over the test block (deposit at start, hold)."""
    sub = monthly_returns.loc[start:end, "SPY"].dropna()
    eq = (1 + sub).cumprod()
    return pd.DataFrame({"date": eq.index, "ret_m": sub.values, "equity": eq.values})


def main():
    big = pd.read_parquet(OUT / "panel_cross_section_v3.parquet")
    monthly_returns = pd.read_parquet(OUT / "monthly_returns_clean.parquet")

    # Strategy params (frozen — winner of fast_sweep):
    K_NORMAL = 15
    K_RECOVERY = 7
    K_BULL = 7
    USE_CONV = False           # equal-weight beats conviction-weight on this dataset
    CASH_CRASH = True
    REGIME_MODE = "tight"

    rows_train, rows_test = [], []
    for split in SPLITS:
        name, ts, te, vs, ve = split
        print(f"\n=== Split {name}: TRAIN {ts}..{te} TEST {vs}..{ve} ===")
        # Fit on TRAIN, predict on TEST
        preds_test = fit_predict_block(big, ts, te, vs, ve)
        if preds_test.empty:
            print(f"  Empty test predictions, skipping.")
            continue

        # Also predict on TRAIN (in-sample sanity)
        preds_train = fit_predict_block(big, ts, te, ts, te)  # in-sample fit
        for label, preds_block in (("test", preds_test), ("train", preds_train)):
            if preds_block.empty:
                continue
            outs = build_strategy_outputs(
                preds_block, big,
                top_k_normal=K_NORMAL, top_k_recovery=K_RECOVERY, top_k_bull=K_BULL,
                use_conviction_weighting=USE_CONV, cash_in_crash=CASH_CRASH,
                regime_mode=REGIME_MODE,
            )
            eq = simulate_strategy(outs, monthly_returns, cost_bps=10.0, starting_cash=1.0)
            if eq.empty:
                continue
            c = cagr(eq) * 100
            roll_max = eq["equity"].cummax()
            dd = (eq["equity"] / roll_max - 1).min() * 100
            sharpe = eq["ret_m"].mean() / eq["ret_m"].std() * np.sqrt(12) if eq["ret_m"].std() > 0 else 0
            # SPY DCA benchmark over the same period
            ts_block = (vs if label == "test" else ts)
            te_block = (ve if label == "test" else te)
            spy_eq = spy_dca(monthly_returns, ts_block, te_block)
            spy_cagr = ((spy_eq["equity"].iloc[-1]) ** (12.0 / max(len(spy_eq), 1)) - 1) * 100
            edge = c - spy_cagr
            row = {
                "split": name, "label": label,
                "ts": ts_block, "te": te_block,
                "CAGR_pct": round(c, 2),
                "MaxDD_pct": round(dd, 2),
                "Sharpe": round(sharpe, 3),
                "SPY_CAGR_pct": round(spy_cagr, 2),
                "Edge_pp": round(edge, 2),
                "Final_equity": round(eq["equity"].iloc[-1], 2),
                "N_months": len(eq),
            }
            print(f"  {label:5s} CAGR={c:.2f}% Sharpe={sharpe:.2f} MaxDD={dd:.1f}% Edge_vs_SPY={edge:+.2f}pp ({len(eq)} months)")
            (rows_train if label == "train" else rows_test).append(row)

    train_df = pd.DataFrame(rows_train)
    test_df = pd.DataFrame(rows_test)
    train_df.to_csv(OUT / "wf_v2_train.csv", index=False)
    test_df.to_csv(OUT / "wf_v2_test.csv", index=False)

    # Aggregate
    if not test_df.empty:
        agg = pd.Series({
            "n_splits": len(test_df),
            "mean_CAGR": test_df["CAGR_pct"].mean(),
            "median_CAGR": test_df["CAGR_pct"].median(),
            "min_CAGR": test_df["CAGR_pct"].min(),
            "max_CAGR": test_df["CAGR_pct"].max(),
            "mean_edge_pp": test_df["Edge_pp"].mean(),
            "median_edge_pp": test_df["Edge_pp"].median(),
            "mean_Sharpe": test_df["Sharpe"].mean(),
            "min_Sharpe": test_df["Sharpe"].min(),
            "mean_MaxDD": test_df["MaxDD_pct"].mean(),
            "n_positive_splits": int((test_df["CAGR_pct"] > 0).sum()),
            "n_beats_spy": int((test_df["Edge_pp"] > 0).sum()),
        })
        print("\n=== Aggregate (TEST) ===")
        print(agg.to_string())
        agg.to_csv(OUT / "wf_v2_aggregate.csv")


if __name__ == "__main__":
    main()
