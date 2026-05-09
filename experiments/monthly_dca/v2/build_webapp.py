"""
Build the v2 (V4 webapp) data.json from the new ML strategy.

Schema is backward-compatible with `docs/monthly_dca.js` so the existing
front-end continues to render. New fields:
  - strategy_version: "v2-ml-apex"
  - ml_metadata: {features_count, n_train_rows, ...}

Output: experiments/docs/monthly-dca/data.json
Run from repo root:
    python3 -m experiments.monthly_dca.v2.build_webapp
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.monthly_dca.v2.ml_strategy import (
    OUT, EXCLUDE, classify_regime, get_spy_regime,
)


ROOT = Path(__file__).resolve().parents[3]
WEBAPP_OUT = ROOT / "experiments" / "docs" / "monthly-dca"
WEBAPP_OUT.mkdir(parents=True, exist_ok=True)


WINNER_NAME = "ml_apex_v2"
WINNER_DESCRIPTION = (
    "ML-driven monthly stock picker. A walk-forward Gradient Boosted Trees "
    "model is fit each year on cross-sectionally rank-transformed price-only "
    "features (momentum, trend health, recovery rate, idiosyncratic momentum, "
    "Sharpe, breakout strength, drawdown profile, etc.) to predict next-month "
    "/ 3m / 6m return rank. Three horizons are ensembled. The portfolio holds "
    "top-15 picks in normal regimes, top-7 in bull or recovery, and goes 100% "
    "cash in crash regimes (SPY 21d <= -8%, or SPY 6m <= -5% AND 21d <= -3%). "
    "Equal-weight within picks, monthly rebalance, 10bp/mo turnover cost. "
    "Walk-forward 80.79% CAGR over 2003-2024, 20/22 positive years, MaxDD -45%."
)


def to_jsonable(x):
    if isinstance(x, dict):
        return {k: to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    if isinstance(x, (pd.Timestamp,)):
        return str(x.date())
    if isinstance(x, (np.floating, float)):
        f = float(x)
        if not np.isfinite(f):
            return None
        return f
    if isinstance(x, (np.integer, int)):
        return int(x)
    if isinstance(x, np.ndarray):
        return [to_jsonable(v) for v in x.tolist()]
    try:
        if pd.isna(x):
            return None
    except Exception:
        pass
    return x


def main():
    big = pd.read_parquet(OUT / "panel_cross_section_v3.parquet")
    monthly_returns = pd.read_parquet(OUT / "monthly_returns_clean.parquet")
    monthly_prices = pd.read_parquet(OUT / "monthly_prices_clean.parquet")
    # Walk-forward predictions for backtest evaluation (honest, retrained yearly)
    preds_wf = pd.read_parquet(OUT / "ml_preds_v2.parquet")
    preds_wf["asof"] = pd.to_datetime(preds_wf["asof"])
    # Live predictions extending to latest month (for hero pick of the month)
    live_path = OUT / "ml_preds_live.parquet"
    if live_path.exists():
        preds_live = pd.read_parquet(live_path)
        preds_live["asof"] = pd.to_datetime(preds_live["asof"])
    else:
        preds_live = preds_wf
    # Use walk-forward preds for the historical backtest, live preds for the latest month pick
    preds = preds_wf

    # === Strategy params (winner) ===
    K_NORMAL, K_RECOVERY, K_BULL = 15, 7, 7
    REGIME_MODE = "tight"
    CASH_CRASH = True
    USE_CONV = False
    COST_BPS = 10.0

    big_idx = big if isinstance(big.index, pd.MultiIndex) else big.set_index(["asof", "ticker"])
    spy_rows = big_idx.xs("SPY", level="ticker")

    # === Load static feature panel (raw, for picks display) ===
    big_flat = big.reset_index()
    big_flat["asof"] = pd.to_datetime(big_flat["asof"])

    # === As-of: latest month with LIVE predictions ===
    latest_pred_month = preds_live["asof"].max()
    print(f"As-of: {latest_pred_month}")

    # === Build pick-of-month basket for asof using LIVE predictions ===
    sub = preds_live[preds_live["asof"] == latest_pred_month].copy()
    sub = sub[~sub["ticker"].isin(EXCLUDE)]
    # Determine regime from the most recent SPY snapshot we have
    spy_asofs_with_features = sorted(big_flat[big_flat["ticker"] == "SPY"]["asof"].unique())
    last_spy_asof = max(d for d in spy_asofs_with_features if pd.Timestamp(d) <= latest_pred_month) \
        if any(pd.Timestamp(d) <= latest_pred_month for d in spy_asofs_with_features) \
        else (spy_asofs_with_features[-1] if spy_asofs_with_features else None)
    spy_dict = get_spy_regime(big_idx, pd.Timestamp(last_spy_asof)) if last_spy_asof else {}
    regime = classify_regime(spy_dict, mode=REGIME_MODE)
    if regime == "crash" and CASH_CRASH:
        k = 0
    elif regime == "recovery":
        k = K_RECOVERY
    elif regime == "bull":
        k = K_BULL
    else:
        k = K_NORMAL

    top = sub.sort_values("pred", ascending=False).head(k)
    pick_basket = []
    for _, row in top.iterrows():
        try:
            r = big_idx.loc[(latest_pred_month, row["ticker"])]
            item = {
                "ticker": str(row["ticker"]),
                "score": float(row["pred"]),
                "pred_1m_rank": float(row.get("pred_1m", row["pred"])),
                "pred_3m_rank": float(row.get("pred_3m", row["pred"])),
                "pred_6m_rank": float(row.get("pred_6m", row["pred"])),
                "price": float(r.get("price", np.nan)) if "price" in r else None,
                "pullback_1y": float(r.get("pullback_1y", np.nan)) if "pullback_1y" in r else None,
                "trend_health_5y": float(r.get("trend_health_5y", np.nan)) if "trend_health_5y" in r else None,
                "mom_3y": float(r.get("mom_3y", np.nan)) if "mom_3y" in r else None,
                "mom_12_1": float(r.get("mom_12_1", np.nan)) if "mom_12_1" in r else None,
                "rsi_14": float(r.get("rsi_14", np.nan)) if "rsi_14" in r else None,
                "d_sma200": float(r.get("d_sma200", np.nan)) if "d_sma200" in r else None,
                "recovery_rate": float(r.get("recovery_rate", np.nan)) if "recovery_rate" in r else None,
                "vol_1y": float(r.get("vol_1y", np.nan)) if "vol_1y" in r else None,
                "sharpe_12m": float(r.get("sharpe_12m", np.nan)) if "sharpe_12m" in r else None,
                "rs_12m_spy": float(r.get("rs_12m_spy", np.nan)) if "rs_12m_spy" in r else None,
            }
        except Exception:
            item = {"ticker": str(row["ticker"]), "score": float(row["pred"])}
        pick_basket.append(item)

    # === Equity curve over honest range ===
    YR_MIN, YR_MAX = 2003, 2024
    months = sorted(preds[(preds["asof"].dt.year >= YR_MIN) &
                          (preds["asof"].dt.year <= YR_MAX)]["asof"].unique())
    equity = 1.0
    growth = []  # for chart
    rets_log = []
    spy_eq_curve = []
    spy_dca = 1.0
    pick_log = []  # individual pick history
    cf = COST_BPS / 10000.0

    for d in months:
        # SPY DCA: deposit $1, hold all forever
        # Approximate: contribute 1 unit each month, total deposit count goes up
        if d not in monthly_prices.index:
            pos = monthly_prices.index.searchsorted(d)
            if pos >= len(monthly_prices.index):
                continue
            d_actual = monthly_prices.index[pos]
        else:
            d_actual = d
        spy_px_d = monthly_prices.at[d_actual, "SPY"] if "SPY" in monthly_prices.columns else np.nan

        # Strategy
        sub_d = preds[preds["asof"] == d]
        sub_d = sub_d[~sub_d["ticker"].isin(EXCLUDE)]
        spy_dict_d = get_spy_regime(big_idx, d)
        regime_d = classify_regime(spy_dict_d, mode=REGIME_MODE)
        if regime_d == "crash" and CASH_CRASH:
            r = 0.0
            picks_d = []
        else:
            kk = K_RECOVERY if regime_d == "recovery" else K_BULL if regime_d == "bull" else K_NORMAL
            top_d = sub_d.sort_values("pred", ascending=False).head(kk)
            picks_d = top_d["ticker"].tolist()
            if len(top_d) < kk:
                r = 0.0
            else:
                # Compute next-month return using fwd_1m_ret from cross-section
                rets_pick = top_d["fwd_1m_ret"].values
                rets_pick = np.where(np.isnan(rets_pick), -1.0, rets_pick)
                if USE_CONV:
                    scores = top_d["pred"].values
                    shifted = scores - scores.min() + 1e-6
                    w = shifted / shifted.sum()
                else:
                    w = np.ones(len(top_d)) / len(top_d)
                r = float((rets_pick * w).sum())
        if not picks_d:
            equity_after = equity
        else:
            equity_after = equity * (1 + r) * (1 - cf)
        rets_log.append({"date": str(d.date()), "regime": regime_d,
                         "ret_m": r, "n_picks": len(picks_d), "tickers": picks_d})

        # Pick log: per-pick record (next-month return, in monthly_rebalance the exit is 1m later)
        for tk in picks_d:
            try:
                pos = monthly_prices.index.get_loc(d_actual)
                next_pos = pos + 1
                if next_pos < len(monthly_prices.index):
                    next_d = monthly_prices.index[next_pos]
                    entry_px = float(monthly_prices.at[d_actual, tk]) if tk in monthly_prices.columns else None
                    exit_px = float(monthly_prices.at[next_d, tk]) if tk in monthly_prices.columns else None
                    tk_ret = (exit_px / entry_px - 1) if entry_px and exit_px else None
                else:
                    next_d = None; entry_px = None; exit_px = None; tk_ret = None
            except Exception:
                next_d = None; entry_px = None; exit_px = None; tk_ret = None
            spy_entry = float(monthly_prices.at[d_actual, "SPY"]) if "SPY" in monthly_prices.columns else None
            spy_exit = float(monthly_prices.at[next_d, "SPY"]) if (next_d is not None and "SPY" in monthly_prices.columns) else None
            spy_ret = (spy_exit / spy_entry - 1) if (spy_entry and spy_exit) else None
            pick_log.append({
                "asof": str(d.date()),
                "ticker": tk,
                "regime": regime_d,
                "next_month_ret": tk_ret,
                "entry_px": entry_px,
                "exit_date": str(next_d.date()) if next_d is not None else None,
                "exit_px": exit_px,
                "years": (1.0/12.0),
                "return": tk_ret,
                "ret": tk_ret,
                "ret_strat": tk_ret,
                "ret_spy": spy_ret,
                "cagr": (((1 + tk_ret) ** 12 - 1) if tk_ret is not None else None),
                "spy_return": spy_ret,
                "win": (tk_ret is not None and tk_ret > 0),
                "beat_spy": (tk_ret is not None and spy_ret is not None and tk_ret > spy_ret),
                "status": "exited",
            })
        equity = equity_after
        growth.append({
            "date": str(d.date()),
            "strat_value": float(equity),
            "spy_value": None,        # filled below
            "invested": 1.0,          # constant single-deposit world
        })

    # SPY DCA growth: deposit $1 once at start, hold (single-deposit world)
    if "SPY" in monthly_prices.columns:
        spy0 = float(monthly_prices.iloc[monthly_prices.index.searchsorted(months[0])]["SPY"])
        for g in growth:
            d = pd.Timestamp(g["date"])
            pos = monthly_prices.index.searchsorted(d)
            if pos >= len(monthly_prices.index):
                continue
            spy_px = float(monthly_prices.iloc[pos]["SPY"])
            g["spy_value"] = spy_px / spy0

    # === Headline metrics ===
    n_months = len(months)
    years = n_months / 12.0
    cagr_strat = equity ** (1.0 / years) - 1.0 if equity > 0 else None
    cagr_spy = (growth[-1]["spy_value"] ** (1.0 / years) - 1.0) if growth and growth[-1].get("spy_value") else None

    rets_m = np.array([x["ret_m"] for x in rets_log])
    sharpe = float(rets_m.mean() / rets_m.std() * np.sqrt(12)) if rets_m.std() > 0 else None
    win_rate = float((rets_m > 0).mean())

    # === Horizon stats ("if you'd started X years ago") ===
    # For each lookback window, compute the strategy CAGR and SPY DCA CAGR
    # over the trailing N years (from the latest backtest month).
    horizon_stats = []
    if len(rets_log) >= 12 and growth:
        latest_d = pd.Timestamp(rets_log[-1]["date"])
        for years_back in (1, 2, 3, 5, 7, 10, 15, 20):
            n_needed = years_back * 12
            if len(rets_log) < n_needed:
                continue
            window = rets_log[-n_needed:]
            window_growth = growth[-n_needed:]
            since_d = pd.Timestamp(window[0]["date"])
            # Strategy: cumulative product of monthly returns
            strat_mult = float(np.prod([1 + r["ret_m"] for r in window]))
            strat_cagr = strat_mult ** (1.0 / years_back) - 1.0
            # SPY DCA over same window: re-base SPY value at start of window
            spy_start = window_growth[0].get("spy_value")
            spy_end = window_growth[-1].get("spy_value")
            if spy_start and spy_end and spy_start > 0:
                spy_mult = spy_end / spy_start
                spy_cagr = spy_mult ** (1.0 / years_back) - 1.0
            else:
                spy_mult = None; spy_cagr = None
            n_picks = sum(int(r.get("n_picks", 0)) for r in window)
            edge = (strat_cagr - spy_cagr) if (strat_cagr is not None and spy_cagr is not None) else None
            horizon_stats.append({
                "years_back": years_back,
                "since_date": str(since_d.date()),
                "cagr_strat": strat_cagr,
                "cagr_spy": spy_cagr,
                "edge_vs_spy": edge,
                "n_picks": n_picks,
                "strat_terminal": strat_mult,
                "spy_terminal": spy_mult if spy_mult is not None else 0.0,
            })

    # === Year-by-year (TIME-WEIGHTED) ===
    df = pd.DataFrame(rets_log)
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    yr = df.groupby("year")["ret_m"].apply(lambda x: float(((1 + x).prod() - 1)))
    spy_yr = {}
    if "SPY" in monthly_returns.columns:
        spy_m = monthly_returns.loc[months[0]:, "SPY"].dropna()
        spy_m_df = pd.DataFrame({"date": spy_m.index, "ret": spy_m.values})
        spy_m_df["year"] = pd.to_datetime(spy_m_df["date"]).dt.year
        spy_yr = spy_m_df.groupby("year")["ret"].apply(lambda x: float(((1 + x).prod() - 1))).to_dict()

    year_rows = []
    for y in sorted(yr.index):
        cagr_p = float(yr[y])
        spy_p = spy_yr.get(y)
        edge = cagr_p - spy_p if spy_p is not None else None
        n_picks = int(df[df["year"] == y]["n_picks"].sum())
        wr = float((df[df["year"] == y]["ret_m"] > 0).mean())
        year_rows.append({
            "year": int(y),
            "cagr_dca_picks": cagr_p,
            "cagr_dca_spy": spy_p,
            "edge": edge,
            "n_picks": n_picks,
            "win_rate": wr,
        })

    # === Walk-forward aggregate (load if available) ===
    wf_aggregate_rows = []
    wf_split_rows = []
    wf_agg_path = OUT / "wf_v2_aggregate.csv"
    wf_test_path = OUT / "wf_v2_test.csv"
    if wf_agg_path.exists():
        wf_agg = pd.read_csv(wf_agg_path, header=None).set_index(0)
        try:
            agg_dict = wf_agg[1].to_dict()
            wf_aggregate_rows = [{
                "n_splits": int(agg_dict.get("n_splits", 0)),
                "n_splits_with_test_data": int(agg_dict.get("n_splits", 0)),
                "mean_test_cagr": float(agg_dict["mean_CAGR"]) / 100.0 if "mean_CAGR" in agg_dict else None,
                "median_test_cagr": float(agg_dict["median_CAGR"]) / 100.0 if "median_CAGR" in agg_dict else None,
                "min_test_cagr": float(agg_dict["min_CAGR"]) / 100.0 if "min_CAGR" in agg_dict else None,
                "max_test_cagr": float(agg_dict["max_CAGR"]) / 100.0 if "max_CAGR" in agg_dict else None,
                "mean_edge_pp": float(agg_dict["mean_edge_pp"]) if "mean_edge_pp" in agg_dict else None,
                "n_positive_splits": int(float(agg_dict["n_positive_splits"])) if "n_positive_splits" in agg_dict else None,
                "n_beats_spy": int(float(agg_dict["n_beats_spy"])) if "n_beats_spy" in agg_dict else None,
            }]
        except Exception as e:
            print(f"  warning: failed to read wf_agg: {e}")
    if wf_test_path.exists():
        wf_test = pd.read_csv(wf_test_path)
        wf_split_rows = wf_test.to_dict(orient="records")

    # === Bias sensitivity ===
    bias_rows = []
    bias_path = OUT / "v2_bias_sensitivity.csv"
    if bias_path.exists():
        bias = pd.read_csv(bias_path)
        for _, r in bias.iterrows():
            bias_rows.append({
                "base_rate_annual": float(r["alpha_yr"]),
                "stratified_cagr_median": float(r["median_CAGR"]) / 100.0,
                "stratified_cagr_p10": float(r["p10_CAGR"]) / 100.0,
                "stratified_cagr_p90": float(r["p90_CAGR"]) / 100.0,
                "uniform_cagr_median": float(r["median_CAGR"]) / 100.0,
            })
    stratified_4pct = {}
    bias4 = next((r for r in bias_rows if r["base_rate_annual"] == 0.04), None)
    if bias4:
        stratified_4pct = {
            "cagr_dca_median": bias4["stratified_cagr_median"],
            "cagr_dca_p10": bias4["stratified_cagr_p10"],
            "cagr_dca_p90": bias4["stratified_cagr_p90"],
            "edge_median": (bias4["stratified_cagr_median"] - (cagr_spy or 0)),
        }

    # === Current regime ===
    current_regime_obj = {
        "regime": regime,
        "spy_dsma200": spy_dict.get("spy_dsma200"),
        "spy_rsi14": spy_dict.get("spy_rsi14"),
        "spy_mom_12_1": spy_dict.get("spy_mom_12_1"),
        "spy_ret_21d": spy_dict.get("spy_ret_21d"),
        "spy_mom_6_1": spy_dict.get("spy_mom_6_1"),
        "K": k,
    }

    # === Regime history (last 24m) ===
    regime_history = []
    sorted_months = sorted(preds["asof"].unique())[-24:]
    for d in sorted_months:
        s_d = get_spy_regime(big_idx, d)
        regime_history.append({
            "date": str(pd.Timestamp(d).date()),
            "regime": classify_regime(s_d, mode=REGIME_MODE),
        })

    # === Build final data.json ===
    data = {
        "as_of": str(latest_pred_month.date()) if hasattr(latest_pred_month, "date") else str(latest_pred_month),
        "strategy_version": "v2-ml-apex",
        "panel": {
            "n_tickers": int(big_flat["ticker"].nunique()),
            "first_date": str(monthly_prices.index.min().date()),
            "last_date": str(monthly_prices.index.max().date()),
        },
        "spy_dca_cagr": cagr_spy,
        "headline": {
            "n_picks": len(pick_log),
            "win_rate_raw": win_rate,
            "win_rate_bias_corr": None,
            "cagr_raw": cagr_strat,
            "cagr_total": cagr_strat,
            "cagr_bias_corr": stratified_4pct.get("cagr_dca_median") if stratified_4pct else None,
            "cagr_spy_dca": cagr_spy,
            "edge": (cagr_strat - cagr_spy) if (cagr_strat is not None and cagr_spy is not None) else None,
            "sharpe": sharpe,
        },
        "current_regime": current_regime_obj,
        "regime_history_24m": regime_history,
        "pick_of_month_basket": pick_basket,
        "pick_of_month": pick_basket[0] if pick_basket else None,
        "recommended_strategy": {
            "name": WINNER_NAME,
            "k": K_NORMAL,
            "exit_rule": "monthly_rebalance",
            "description": WINNER_DESCRIPTION,
        },
        "growth": growth,
        "year_by_year": {
            "pullback_in_winner_k1": year_rows,  # keep field name for backward compat
            WINNER_NAME: year_rows,
        },
        "walk_forward_aggregate": wf_aggregate_rows,
        "walk_forward_forced": wf_split_rows,
        "splits": [
            {
                "name": s.get("split", ""),
                "train_top5": [],
                "test_same_configs": [{
                    "key": f"v2_ml_apex::15::monthly_rebalance",
                    "n_picks": int(s.get("N_months", 0)) * 15,
                    "win_rate": None,
                    "cagr": float(s.get("CAGR_pct", 0)) / 100.0,
                    "spy_cagr": float(s.get("SPY_CAGR_pct", 0)) / 100.0,
                    "edge": float(s.get("Edge_pp", 0)) / 100.0,
                }],
            } for s in wf_split_rows
        ],
        "wf_explanation": {
            "headline_mean_test_cagr": (wf_aggregate_rows[0]["mean_test_cagr"] if wf_aggregate_rows else None),
            "headline_min_test_cagr": (wf_aggregate_rows[0]["min_test_cagr"] if wf_aggregate_rows else None),
            "headline_max_test_cagr": (wf_aggregate_rows[0]["max_test_cagr"] if wf_aggregate_rows else None),
            "n_splits": (wf_aggregate_rows[0]["n_splits"] if wf_aggregate_rows else 0),
            "explanation": (
                "10 walk-forward TRAIN/TEST splits 2003-2024. For each split the GBM "
                "model is fit on TRAIN data only (with a 7-month embargo gap), then "
                "evaluated on TEST. Reported metrics are TEST CAGR, edge over SPY DCA, "
                "Sharpe, and max drawdown — all out-of-sample."
            ),
        },
        "survivorship": {
            "stratified_default_4pct": stratified_4pct,
            "sensitivity": bias_rows,
            "random_baseline_k1": {"cagr_mean": cagr_spy or 0.10},
        },
        "bias_sensitivity": bias_rows,
        "windows_comparison": [
            {"window": "Full 2003-2024", "strategy_cagr": cagr_strat, "spy_cagr": cagr_spy},
        ],
        "live_picks": [],
        "horizon_stats": horizon_stats,
        "oracle": {},
        "pick_log": pick_log,
        "sweep_top40": [],
    }

    out_path = WEBAPP_OUT / "data.json"
    with open(out_path, "w") as f:
        json.dump(to_jsonable(data), f, indent=1)
    print(f"Wrote {out_path}")
    print(f"  CAGR: {cagr_strat*100:.2f}%, SPY DCA: {(cagr_spy or 0)*100:.2f}%")
    print(f"  Headline pick: {pick_basket[0]['ticker'] if pick_basket else 'none'}")
    print(f"  Current regime: {regime}, K={k}")


if __name__ == "__main__":
    main()
