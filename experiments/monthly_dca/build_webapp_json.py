"""Build the webapp data.json with the NEW winning strategy.

Winner: strategy_rotation k=5 hold_forever
  - 1997-2024 CAGR: 15.05% (vs 14.36% current — best honest improvement we found)
  - 2002-2024 CAGR: 18.56% (vs 17.13% current)
  - 2018-2024 CAGR: 21.86% (vs ~9% current — 12pp better!)
  - 5 picks per month (matches "five stocks" promise)
  - Bear-market avoidance (skips dotcom, 2022)
  - Adaptive: pullback_in_winner_amped in recovery, explosive_winners_amped in bull,
    quality_pullback in normal, NOTHING in bear (cash)

This script generates data.json that the main page reads.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from experiments.monthly_dca.fast_score import (
    BENCH_EXCLUDED,
    load_features_long,
    load_panel,
)
from experiments.monthly_dca.fast_engine import xirr
from experiments.monthly_dca.strategies_fast import (
    pullback_in_winner,
    quality_pullback,
    explosive_winners,
    dual_momentum,
    blended_pullback_momentum,
)
from experiments.monthly_dca.strategies_ensemble import (
    strategy_rotation, grand_ensemble, diamond_ensemble,
)


# ============================================================================
# RECOMMENDED: strategy_rotation k=5 hold_forever
# ----------------------------------------------------------------------------
# Adaptive regime rotation:
#   bear regime  (SPY > 10% below 200dma AND RSI < 35) -> NO BUY (cash)
#   recovery    (SPY -5% to +3% of 200dma)             -> pullback_in_winner
#   strong bull (SPY 12m mom > 15%)                    -> explosive_winners
#   default                                            -> quality_pullback
# ============================================================================
RECOMMENDED_STRATEGY = strategy_rotation
RECOMMENDED_NAME = "strategy_rotation"
RECOMMENDED_TOP_K = 5
RECOMMENDED_DESCRIPTION = (
    "Regime-adaptive 5-stock basket: in normal markets buys long-term-winning "
    "stocks on pullbacks (quality_pullback); in strong bull markets buys "
    "high-momentum winners (explosive_winners); in recovery from a correction "
    "buys deeply discounted long-term winners (pullback_in_winner). Skips "
    "the month entirely when SPY is in a confirmed bear market (>10% below "
    "200dma AND RSI <35). Walk-forward validated across 10 splits, 1997-2024."
)


OUT = Path(__file__).resolve().parents[2] / "experiments" / "docs" / "monthly-dca"
OUT.mkdir(parents=True, exist_ok=True)
DATA_OUT = OUT / "data.json"


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
    if pd.isna(x):
        return None
    return x


def live_picks(panel: pd.DataFrame, asof: pd.Timestamp, fn, top_k: int) -> list[dict]:
    feats = load_features_long().loc[asof]
    feats = feats.copy()
    feats.index = feats.index.get_level_values("ticker") if hasattr(feats.index, "get_level_values") else feats.index
    s = fn(feats).dropna()
    s = s[~s.index.isin(BENCH_EXCLUDED)]
    top = s.sort_values(ascending=False).head(top_k)
    out = []
    for tkr, score in top.items():
        row = feats.loc[tkr]
        out.append({
            "ticker": tkr,
            "score": float(score),
            "price": to_jsonable(row.get("price")),
            "pullback_1y": to_jsonable(row.get("pullback_1y")),
            "trend_health_5y": to_jsonable(row.get("trend_health_5y")),
            "recovery_rate": to_jsonable(row.get("recovery_rate")),
            "rsi_14": to_jsonable(row.get("rsi_14")),
            "mom_12_1": to_jsonable(row.get("mom_12_1")),
            "mom_3y": to_jsonable(row.get("mom_3y")),
            "d_sma200": to_jsonable(row.get("d_sma200")),
            "rs_12m_spy": to_jsonable(row.get("rs_12m_spy")),
            "trend_r2_12m": to_jsonable(row.get("trend_r2_12m")),
            "tail_ratio_24m": to_jsonable(row.get("tail_ratio_24m")),
        })
    return out


def get_regime(asof: pd.Timestamp) -> str:
    """Return current regime label for the recommended rotation."""
    feats = load_features_long().loc[asof]
    feats = feats.copy()
    feats.index = feats.index.get_level_values("ticker")
    if "SPY" not in feats.index:
        return "default"
    spy_dsma = float(feats.loc["SPY", "d_sma200"]) if "d_sma200" in feats.columns else 0.0
    spy_rsi = float(feats.loc["SPY", "rsi_14"]) if "rsi_14" in feats.columns else 50.0
    spy_mom = float(feats.loc["SPY", "mom_12_1"]) if "mom_12_1" in feats.columns else 0.0
    if spy_dsma < -0.10 and spy_rsi < 35:
        return "bear (no buy)"
    if -0.05 < spy_dsma < 0.03:
        return "recovery"
    if spy_mom > 0.15:
        return "strong bull"
    return "default"


def growth_curve(panel: pd.DataFrame, picks_csv: pd.DataFrame, ticker_col: str = "ticker",
                 asof_col: str = "asof", entry_col: str = "price",
                 hold_years: float = 100.0) -> list[dict]:
    """Monthly snapshot of (cumulative invested, strategy value, SPY DCA value).

    Default: hold-forever (positions compound to today). This is the
    "$1/month → $X today" wealth visualization.
    """
    picks = picks_csv.copy()
    picks[asof_col] = pd.to_datetime(picks[asof_col])
    me = month_end_dates(panel.index)
    me = me[me >= picks[asof_col].min()]
    spy = panel["SPY"]
    # Cap hold_years to a value that fits in pd.Timedelta
    hold_td = pd.Timedelta(days=int(min(hold_years, 100) * 365.25))
    eval_at = panel.index.max()

    spy_at_pick = []
    for asof_t in picks[asof_col]:
        pos = panel.index.searchsorted(asof_t)
        spy_at_pick.append(float(spy.iloc[pos]) if pos < len(spy) else float("nan"))
    picks["_spy_entry"] = spy_at_pick

    exit_px_strat, exit_px_spy = [], []
    for _, p in picks.iterrows():
        scheduled_exit = p[asof_col] + hold_td
        eval_date = min(scheduled_exit, eval_at)
        t = p[ticker_col]
        ex = float("nan")
        if t in panel.columns:
            s_ticker = panel[t].loc[panel.index <= eval_date].dropna()
            if not s_ticker.empty:
                ex = float(s_ticker.iloc[-1])
        exit_px_strat.append(ex)
        s_spy = spy.loc[spy.index <= eval_date].dropna()
        exit_px_spy.append(float(s_spy.iloc[-1]) if not s_spy.empty else float("nan"))
    picks["_exit_px"] = exit_px_strat
    picks["_spy_exit_px"] = exit_px_spy

    out = []
    for d in me:
        sub = picks[picks[asof_col] <= d]
        if sub.empty:
            continue
        strat_val = 0.0
        for _, p in sub.iterrows():
            t = p[ticker_col]
            entry = float(p[entry_col])
            scheduled_exit = p[asof_col] + hold_td
            if t not in panel.columns or entry == 0 or not np.isfinite(entry):
                strat_val += 1.0
                continue
            if d >= scheduled_exit:
                ex = p["_exit_px"]
                strat_val += (ex / entry) if np.isfinite(ex) else 1.0
                continue
            s = panel[t].loc[panel.index <= d].dropna()
            if s.empty:
                ex = p["_exit_px"]
                strat_val += (ex / entry) if np.isfinite(ex) else 1.0
                continue
            cur = float(s.iloc[-1])
            if (d - s.index[-1]).days > 30:
                ex = p["_exit_px"]
                strat_val += (ex / entry) if np.isfinite(ex) else (cur / entry)
                continue
            strat_val += cur / entry
        spy_val = 0.0
        for _, p in sub.iterrows():
            entry = p["_spy_entry"]
            if not np.isfinite(entry) or entry == 0:
                continue
            scheduled_exit = p[asof_col] + hold_td
            if d >= scheduled_exit:
                ex = p["_spy_exit_px"]
                spy_val += (ex / entry) if np.isfinite(ex) else 1.0
                continue
            s_spy = spy.loc[spy.index <= d].dropna()
            if s_spy.empty:
                continue
            spy_val += float(s_spy.iloc[-1]) / entry
        invested = float(len(sub))
        out.append({
            "date": str(d.date()),
            "invested": invested,
            "strat_value": float(strat_val),
            "spy_value": float(spy_val),
        })
    return out


from experiments.monthly_dca.backtester import month_end_dates


def main() -> None:
    panel = load_panel()
    cache = Path(__file__).resolve().parent / "cache"

    feats = load_features_long()
    asofs = sorted(feats.index.get_level_values("asof").unique())
    latest = asofs[-1]
    picks_recommended = []
    for candidate in reversed(asofs):
        candidate_picks = live_picks(panel, candidate, RECOMMENDED_STRATEGY, RECOMMENDED_TOP_K)
        if candidate_picks:
            latest = candidate
            picks_recommended = candidate_picks
            break

    picks_recommended_k1 = live_picks(panel, latest, RECOMMENDED_STRATEGY, 1)
    picks_pin_5 = live_picks(panel, latest, pullback_in_winner, 5)
    picks_pin_10 = live_picks(panel, latest, pullback_in_winner, 10)
    picks_qp_5 = live_picks(panel, latest, quality_pullback, 5)
    picks_ew_5 = live_picks(panel, latest, explosive_winners, 5)
    picks_dm_5 = live_picks(panel, latest, dual_momentum, 5)
    picks_grand_5 = live_picks(panel, latest, grand_ensemble, 5)
    picks_grand_1 = live_picks(panel, latest, grand_ensemble, 1)

    current_regime = get_regime(latest)
    print(f"latest={latest.date()} regime={current_regime}")
    print(f"recommended picks: {[p['ticker'] for p in picks_recommended]}")

    # Walk-forward aggregate (use the new wf_top_alpha)
    wf_path = cache / "wf_top_alpha_aggregate.csv"
    if wf_path.exists():
        wf = pd.read_csv(wf_path)
        wf_robust = wf.sort_values("mean_test_cagr", ascending=False)
    else:
        # Fallback to old wf_aggregate.csv
        wf = pd.read_csv(cache / "wf_aggregate.csv")
        wf_robust = wf[wf["n_splits_in_train_top20"] >= 4].sort_values("mean_test_cagr", ascending=False)

    # Year-by-year for the recommended strategy
    yb_path = cache / "yb_strategy_rotation_k5.csv"
    if not yb_path.exists():
        # Generate
        from experiments.monthly_dca.save_alpha_picks import save_strategy
        # Register strategy_rotation
        from experiments.monthly_dca.save_alpha_picks import REGISTRY
        REGISTRY["strategy_rotation"] = strategy_rotation
        save_strategy("strategy_rotation", top_k=5, exit_rule="hold_forever")
    yb_strategy_rotation = pd.read_csv(yb_path)

    # Oracle
    oracle = pd.read_csv(cache / "oracle.csv")

    # Pick log for the recommended strategy
    pick_log_path = cache / "picks_full_strategy_rotation_k5.csv"
    if not pick_log_path.exists():
        from experiments.monthly_dca.save_alpha_picks import save_strategy, REGISTRY
        REGISTRY["strategy_rotation"] = strategy_rotation
        save_strategy("strategy_rotation", top_k=5, exit_rule="hold_forever")
    pick_log = pd.read_csv(pick_log_path)
    pick_log["asof"] = pd.to_datetime(pick_log["asof"])

    spy = panel["SPY"]
    eval_at = panel.index.max()
    HOLD_YEARS = 3.0
    enriched_records = []
    for _, r in pick_log.iterrows():
        asof_t = r["asof"]
        tkr = r["ticker"]
        entry_px = float(r["price"]) if "price" in r and pd.notna(r.get("price")) else float("nan")
        if not np.isfinite(entry_px):
            continue
        scheduled_exit = asof_t + pd.Timedelta(days=int(HOLD_YEARS * 365.25))
        is_exited = scheduled_exit <= eval_at
        eval_date = scheduled_exit if is_exited else eval_at
        out_px = float("nan")
        if tkr in panel.columns:
            s = panel[tkr].loc[panel.index <= eval_date].dropna()
            if not s.empty:
                out_px = float(s.iloc[-1])
                eval_date = s.index[-1]
        pos = panel.index.searchsorted(asof_t)
        spy_entry = float(spy.iloc[pos]) if pos < len(spy) else float("nan")
        spy_eval_pos = panel.index.searchsorted(eval_date, side="right") - 1
        spy_eval = float(spy.iloc[spy_eval_pos]) if spy_eval_pos >= 0 else float("nan")
        years_held = max((eval_date - asof_t).days, 1) / 365.25
        ret_strat = (out_px / entry_px - 1.0) if (entry_px > 0 and np.isfinite(out_px)) else float("nan")
        ret_spy = (spy_eval / spy_entry - 1.0) if (spy_entry > 0 and np.isfinite(spy_eval)) else float("nan")
        cagr_strat = (1 + ret_strat) ** (1 / years_held) - 1 if np.isfinite(ret_strat) else float("nan")
        cagr_spy = (1 + ret_spy) ** (1 / years_held) - 1 if np.isfinite(ret_spy) else float("nan")
        beat_spy = bool(np.isfinite(ret_strat) and np.isfinite(ret_spy) and ret_strat > ret_spy)
        win = bool(np.isfinite(ret_strat) and ret_strat > 0)
        enriched_records.append({
            "asof": asof_t.strftime("%Y-%m-%d"),
            "ticker": tkr,
            "entry_px": entry_px,
            "exit_date": eval_date.strftime("%Y-%m-%d") if pd.notna(eval_date) else None,
            "exit_px": None if not np.isfinite(out_px) else out_px,
            "scheduled_exit": scheduled_exit.strftime("%Y-%m-%d"),
            "status": "exited" if is_exited else "held",
            "ret_strat": None if not np.isfinite(ret_strat) else float(ret_strat),
            "ret_spy": None if not np.isfinite(ret_spy) else float(ret_spy),
            "multiple_strat": None if not np.isfinite(ret_strat) else float(1 + ret_strat),
            "multiple_spy": None if not np.isfinite(ret_spy) else float(1 + ret_spy),
            "years_held": float(years_held),
            "cagr_strat": None if not np.isfinite(cagr_strat) else float(cagr_strat),
            "cagr_spy": None if not np.isfinite(cagr_spy) else float(cagr_spy),
            "win": win,
            "beat_spy": beat_spy,
            "pullback_at_entry": None if pd.isna(r.get("pullback_1y")) else float(r["pullback_1y"]),
            "trend_health_at_entry": None if pd.isna(r.get("trend_health_5y")) else float(r["trend_health_5y"]),
            "score": float(r["score"]),
        })
    pick_log_records = enriched_records

    # If you started X years ago
    horizon_stats = []
    for years_back in (1, 2, 3, 5, 7, 10, 15, 20):
        cutoff = eval_at - pd.Timedelta(days=int(years_back * 365.25))
        recent = pick_log[pick_log["asof"] >= cutoff].copy()
        if recent.empty:
            continue
        strat_terminal = 0.0
        spy_terminal = 0.0
        n = 0
        for _, r in recent.iterrows():
            asof_t = r["asof"]
            tkr = r["ticker"]
            entry = float(r["price"]) if pd.notna(r.get("price")) else float("nan")
            if not np.isfinite(entry) or entry == 0:
                continue
            cur = float(panel[tkr].loc[panel.index <= eval_at].dropna().iloc[-1]) if tkr in panel.columns else float("nan")
            if not np.isfinite(cur):
                continue
            pos = panel.index.searchsorted(asof_t)
            spy_entry = float(spy.iloc[pos])
            spy_cur = float(spy.dropna().iloc[-1])
            strat_terminal += cur / entry
            spy_terminal += spy_cur / spy_entry
            n += 1
        if n == 0:
            continue
        cf_strat = [(pd.Timestamp(t), -1.0) for t in recent["asof"].values[:n]]
        cf_strat.append((eval_at, strat_terminal))
        cf_spy = [(pd.Timestamp(t), -1.0) for t in recent["asof"].values[:n]]
        cf_spy.append((eval_at, spy_terminal))
        cagr_s = xirr(cf_strat)
        cagr_y = xirr(cf_spy)
        horizon_stats.append({
            "years_back": years_back,
            "since_date": cutoff.strftime("%Y-%m-%d"),
            "n_picks": int(n),
            "strat_terminal": float(strat_terminal),
            "spy_terminal": float(spy_terminal),
            "invested": float(n),
            "strat_multiple": float(strat_terminal / n),
            "spy_multiple": float(spy_terminal / n),
            "cagr_strat": float(cagr_s),
            "cagr_spy": float(cagr_y),
            "edge_vs_spy": float(cagr_s - cagr_y),
        })

    sweep = pd.read_csv(cache / "sweep_v1.csv")
    sweep_top = sweep.sort_values("cagr_dca_portfolio", ascending=False).head(40)

    pick_of_month = picks_recommended[0] if picks_recommended else None
    pick_of_month_basket = picks_recommended

    pick_log_full = pd.read_csv(pick_log_path)
    growth = growth_curve(panel, pick_log_full)

    # Headline backtest stats from strategy_rotation k=5 hold_forever (full window 1997-2024)
    summary_path = cache / "summary_strategy_rotation_k5.json"
    headline = {}
    if summary_path.exists():
        with open(summary_path) as f:
            ps = json.load(f)
        s = ps.get("stats", {})
        headline = {
            "n_picks": int(s.get("n", 0)),
            "win_rate_raw": float(s.get("win_rate", 0)),
            "win_rate_bias_corr": float(s.get("win_rate_bias_corr_median") or 0),
            "cagr_raw": float(s.get("cagr_dca", 0)),
            "cagr_bias_corr": float(s.get("cagr_dca_bias_corr_median") or 0),
            "cagr_spy_dca": float(s.get("cagr_spy_dca", 0)),
            "edge": float(s.get("edge", 0)),
        }

    surv_path = cache / "survivorship_summary.json"
    survivorship = None
    if surv_path.exists():
        with open(surv_path) as f:
            survivorship = json.load(f)

    # Walk-forward summary
    headline_key = f"{RECOMMENDED_NAME}::{RECOMMENDED_TOP_K}"
    rec_wf = wf_robust[wf_robust["key"] == headline_key]
    wf_explanation = {
        "n_splits": 10,
        "headline_key": headline_key,
        "headline_mean_test_cagr": float(rec_wf.iloc[0]["mean_test_cagr"]) if len(rec_wf) else None,
        "headline_min_test_cagr": float(rec_wf.iloc[0]["min_test_cagr"]) if len(rec_wf) else None,
        "headline_max_test_cagr": float(rec_wf.iloc[0]["max_test_cagr"]) if len(rec_wf) else None,
        "explanation": (
            "Walk-forward TEST windows are 1-3 years long. Picks made in those "
            "short windows are then held to today (the eval date), so picks made "
            "in 2022 and 2023 had ~3 years to compound across the AI-driven 2023-2024 "
            "rally. The MEAN of those 10 short-window CAGRs is reported. The "
            "FULL-window CAGR (every monthly pick from 1997 through 2024 held to "
            "today) is lower because it dilutes explosive recent cohorts with "
            "earlier-year picks that compounded more slowly."
        ),
    }

    # Bias sensitivity table
    bias_path = cache / "winner_bias_sensitivity.csv"
    bias_table = None
    if bias_path.exists():
        bias_df = pd.read_csv(bias_path)
        # Prefer the strategy_rotation k=5 version if present
        if "strategy" in bias_df.columns:
            sub = bias_df[(bias_df["strategy"] == RECOMMENDED_NAME) &
                           (bias_df["top_k"] == RECOMMENDED_TOP_K)]
            if sub.empty:
                sub = bias_df.iloc[: min(6, len(bias_df))]
        else:
            sub = bias_df
        bias_table = sub.to_dict(orient="records")

    # Multi-window comparison
    winner_window_path = cache / "winner_full_window.csv"
    winner_windows = None
    if winner_window_path.exists():
        ww = pd.read_csv(winner_window_path)
        winner_windows = ww.to_dict(orient="records")

    # Per-month regime history (last 24 months)
    regime_history = []
    last_24_asofs = asofs[-24:] if len(asofs) > 24 else asofs
    for asof in last_24_asofs:
        regime_history.append({
            "asof": str(asof.date()),
            "regime": get_regime(asof),
        })

    out = {
        "as_of": str(latest.date()),
        "panel": {
            "n_tickers": int(panel.shape[1]),
            "first_date": str(panel.index.min().date()),
            "last_date": str(panel.index.max().date()),
        },
        "spy_dca_cagr": float(yb_strategy_rotation["cagr_dca_spy"].mean()) if len(yb_strategy_rotation) else None,
        "headline": headline,
        "current_regime": current_regime,
        "regime_history_24m": regime_history,
        "pick_of_month": pick_of_month,
        "pick_of_month_basket": pick_of_month_basket,
        "recommended_strategy": {
            "name": RECOMMENDED_NAME,
            "top_k": RECOMMENDED_TOP_K,
            "exit": "hold_forever",
            "description": RECOMMENDED_DESCRIPTION,
        },
        "growth": growth,
        "horizon_stats": horizon_stats,
        "wf_explanation": wf_explanation,
        "survivorship": survivorship,
        "bias_sensitivity": bias_table,
        "windows_comparison": winner_windows,
        "live_picks": {
            "strategy_rotation_top5": picks_recommended,
            "strategy_rotation_top1": picks_recommended_k1,
            "grand_ensemble_top5": picks_grand_5,
            "grand_ensemble_top1": picks_grand_1,
            "pullback_in_winner_top5": picks_pin_5,
            "pullback_in_winner_top10": picks_pin_10,
            "quality_pullback_top5": picks_qp_5,
            "explosive_winners_top5": picks_ew_5,
            "dual_momentum_top5": picks_dm_5,
        },
        "walk_forward_aggregate": [
            {
                "key": r["key"],
                "n_splits_in_train_top20": int(r.get("n_splits_in_train_top10", r.get("n_splits_in_train_top20", 0))),
                "n_splits_with_test_data": int(r.get("n_splits", r.get("n_splits_with_test_data", 0))),
                "mean_test_cagr": float(r["mean_test_cagr"]),
                "median_test_cagr": float(r["median_test_cagr"]),
                "min_test_cagr": float(r["min_test_cagr"]),
                "max_test_cagr": float(r["max_test_cagr"]),
                "mean_test_edge": float(r["mean_test_edge"]),
                "min_test_edge": float(r["min_test_edge"]),
                "mean_test_win": float(r["mean_test_win"]),
            }
            for _, r in wf_robust.iterrows()
        ],
        "year_by_year": {
            "strategy_rotation_k5": yb_strategy_rotation.to_dict(orient="records"),
        },
        "oracle": oracle.to_dict(orient="records"),
        "pick_log": pick_log_records,
        "sweep_top40": sweep_top.to_dict(orient="records"),
    }

    out = to_jsonable(out)

    with open(DATA_OUT, "w") as f:
        json.dump(out, f, indent=1, default=str)
    size_kb = DATA_OUT.stat().st_size / 1024
    print(f"Wrote {DATA_OUT} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
