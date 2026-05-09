"""Build the new data.json using the V3 winner with FULL backwards-compatible structure.

Winner: strategy_rotation k=5 monthly_rebalance
  Full backtest 2002-2024: 35.4% CAGR XIRR, +23pp edge over SPY DCA
  Walk-forward (10 splits): mean OOS test CAGR 40.5%, +25.8pp edge

Output: experiments/docs/monthly-dca/data.json
Consumed by: docs/monthly_dca.js
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path
from typing import Callable

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from experiments.monthly_dca.compound_engine import benchmark_spy_dca
from experiments.monthly_dca.fast_engine import load_panel, load_features
from experiments.monthly_dca.fast_monthly_rebalance import run_monthly_rebalance
from experiments.monthly_dca.strategies_ensemble import strategy_rotation
from experiments.monthly_dca.strategies_apex import _spy_regime


CACHE = Path(__file__).resolve().parent / "cache"
OUT = Path(__file__).resolve().parents[2] / "experiments" / "docs" / "monthly-dca"
OUT.mkdir(parents=True, exist_ok=True)
DATA_OUT = OUT / "data.json"


WINNING_NAME = "strategy_rotation"
WINNING_K = 5
WINNING_EXIT = "monthly_rebalance"
WINNING_DESCRIPTION = (
    "Regime-adaptive 5-stock COMPOUNDING basket with monthly rebalance. "
    "Each month-end, the strategy classifies the SPY regime and selects the "
    "top-5 stocks for that regime: deep-value rebound (recovery), explosive "
    "momentum (strong bull), quality compounders on pullback (sideways), "
    "or all-cash (deep bear). The portfolio is fully rebalanced into the new "
    "picks each month, COMPOUNDING returns from prior winners into the next "
    "month's high-conviction names. Walk-forward validated across 10 distinct "
    "splits 2002-2024 with mean OOS test CAGR of 40.5%, beating SPY DCA by "
    "+25.8pp on average."
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


def _live_picks(panel, asof, fn, k):
    feats = load_features(asof)
    s = fn(feats).dropna()
    s = s[~s.index.isin(("SPY", "QQQ", "IWM", "VTI", "RSP", "DIA"))]
    top = s.sort_values(ascending=False).head(k)
    out = []
    feature_cols = [
        "price","pullback_1y","trend_health_5y","recovery_rate","rsi_14",
        "mom_12_1","mom_3y","d_sma200","rs_12m_spy","trend_r2_12m",
        "tail_ratio_24m","sharpe_12m","quality_score_5y","beta_2y",
        "idio_mom_12_1","breakout_strength_60","fip_score",
        "frac_above_50dma_1y","mom_consistency_12m","near_52wh_60d",
        "vol_expansion_24m","accel",
    ]
    for tkr, score in top.items():
        if tkr not in feats.index:
            continue
        row = feats.loc[tkr]
        item = {"ticker": str(tkr), "score": float(score)}
        for c in feature_cols:
            if c in row:
                v = row[c]
                if isinstance(v, (int, float, np.floating, np.integer)):
                    if not np.isfinite(v):
                        item[c] = None
                    else:
                        item[c] = float(v)
                else:
                    item[c] = None
        out.append(item)
    return out


def _horizon_stats(panel, sfn, k, eval_at, years_list=(1, 2, 3, 5, 10)) -> list:
    """For each "years back" horizon, simulate strategy from that start and
    report terminal value & CAGR."""
    stats = []
    spy = panel["SPY"]
    panel_idx = panel.index
    for years in years_list:
        try:
            since = eval_at - pd.DateOffset(years=years)
            # Find nearest panel date
            pos = panel_idx.searchsorted(since)
            if pos >= len(panel_idx):
                continue
            since_date = panel_idx[pos]
            res = run_monthly_rebalance(
                panel, sfn, top_k=k,
                start=str(since_date.date()), end=str((eval_at - pd.DateOffset(months=1)).date()),
                eval_at=eval_at, cost_bps=5.0,
            )
            if res["deposited"] <= 0:
                continue
            # SPY DCA
            spy_res = benchmark_spy_dca(panel, str(since_date.date()),
                                          str((eval_at - pd.DateOffset(months=1)).date()),
                                          eval_at=eval_at)
            stats.append({
                "years_back": int(years),
                "since_date": str(since_date.date()),
                "n_picks": int(res["n_trades"]),
                "strat_terminal": float(res["final_equity"]),
                "spy_terminal": float(spy_res["final"]),
                "invested": float(res["deposited"]),
                "strat_multiple": float(res["final_equity"] / res["deposited"]),
                "spy_multiple": float(spy_res["final"] / spy_res["deposited"]),
                "cagr_strat": float(res["cagr_xirr"]),
                "cagr_spy": float(spy_res["cagr_xirr"]),
                "edge_vs_spy": float(res["cagr_xirr"] - spy_res["cagr_xirr"]),
            })
        except Exception as e:
            print(f"horizon {years}y failed: {e}")
    return stats


def main():
    panel = load_panel()
    eval_at = pd.Timestamp("2026-05-07")
    if eval_at > panel.index[-1]:
        eval_at = panel.index[-1]
    start = "2002-01-31"
    end = "2024-12-31"

    print(f"Running winner backtest: {WINNING_NAME} k={WINNING_K} {WINNING_EXIT}", flush=True)
    res = run_monthly_rebalance(panel, strategy_rotation, top_k=WINNING_K,
                                 start=start, end=end, eval_at=eval_at,
                                 cost_bps=5.0)
    spy = benchmark_spy_dca(panel, start=start, end=end, eval_at=eval_at)
    print(f"  Winner CAGR: {res['cagr_xirr']:.4f}", flush=True)
    print(f"  SPY DCA CAGR: {spy['cagr_xirr']:.4f}", flush=True)
    print(f"  Edge: {res['cagr_xirr'] - spy['cagr_xirr']:+.4f}", flush=True)
    print(f"  Final equity: ${res['final_equity']:.2f} from ${res['deposited']:.0f}", flush=True)

    # Win rate from trades
    trades_df = res["trades"]
    n_win = 0
    n_total = 0
    if not trades_df.empty:
        rets = trades_df["ret"].dropna()
        n_win = int((rets > 0).sum())
        n_total = int(len(rets))
    win_rate = n_win / n_total if n_total else 0.0

    # Bias sensitivity from cache (or compute fresh)
    bias_path = CACHE / "winner_bias_sensitivity_v3.csv"
    bias_rows = []
    cagr_bias_corr_4pct = None
    win_rate_bias_corr_4pct = None
    if bias_path.exists():
        bias_df = pd.read_csv(bias_path)
        for _, row in bias_df.iterrows():
            bias_rows.append({
                "alpha": float(row["alpha"]),
                "cagr_p10": float(row.get("cagr_p10", float("nan"))),
                "cagr_median": float(row["cagr_median"]),
                "cagr_p90": float(row.get("cagr_p90", float("nan"))),
                "edge_median": float(row["edge_median"]),
            })
            if abs(row["alpha"] - 0.04) < 1e-6:
                cagr_bias_corr_4pct = float(row["cagr_median"])

    # Multi-window
    windows = []
    for ws, we, label in [
        ("2002-01-31", "2024-12-31", "Full 2002-2024"),
        ("2010-01-31", "2024-12-31", "Modern 2010-2024"),
        ("2018-01-31", "2024-12-31", "Recent 2018-2024"),
    ]:
        try:
            wr = run_monthly_rebalance(panel, strategy_rotation, top_k=WINNING_K,
                                        start=ws, end=we, eval_at=eval_at, cost_bps=5.0)
            ws_spy = benchmark_spy_dca(panel, ws, we, eval_at=eval_at)
            windows.append({
                "label": label, "start": ws, "end": we,
                "cagr": float(wr["cagr_xirr"]),
                "cagr_spy": float(ws_spy["cagr_xirr"]),
                "edge": float(wr["cagr_xirr"] - ws_spy["cagr_xirr"]),
                "final_equity": float(wr["final_equity"]),
                "deposited": float(wr["deposited"]),
            })
        except Exception as e:
            print(f"  window {label}: {e}")

    # Year-by-year via equity curve (then dict-wrap for JS compatibility)
    eq_df = res["equity_curve"].copy()
    eq_df["date"] = pd.to_datetime(eq_df["date"])
    eq_df["year"] = eq_df["date"].dt.year
    yb_list = []
    for year, grp in eq_df.groupby("year"):
        grp = grp.sort_values("date")
        if grp.empty:
            continue
        s_eq = float(grp.iloc[0]["equity"])
        e_eq = float(grp.iloc[-1]["equity"])
        if s_eq <= 0:
            continue
        ret = e_eq / s_eq - 1.0
        # SPY DCA same year
        try:
            year_start = grp.iloc[0]["date"]
            year_end = grp.iloc[-1]["date"]
            spy_year = benchmark_spy_dca(panel, str(year_start.date()),
                                            str(year_end.date()), eval_at=year_end)
            spy_cagr = float(spy_year["cagr_xirr"])
        except Exception:
            spy_cagr = float("nan")
        # Trades that year
        n_picks = 0
        win_rate_y = 0.0
        median_ret = float("nan")
        if not trades_df.empty:
            year_trades = trades_df[trades_df["entry_date"].astype(str).str.startswith(str(year))]
            if not year_trades.empty:
                n_picks = len(year_trades)
                win_rate_y = float((year_trades["ret"] > 0).mean())
                median_ret = float(year_trades["ret"].median())
        yb_list.append({
            "year": int(year),
            "n_picks": int(n_picks),
            "win_rate": win_rate_y,
            "median_ret": median_ret,
            "cagr_dca": ret,
            "cagr_dca_spy": spy_cagr,
            "edge": ret - spy_cagr if pd.notna(spy_cagr) else None,
        })

    year_by_year = {f"{WINNING_NAME}_k{WINNING_K}": yb_list}

    # Trades log: format for old structure
    pick_log_records = []
    if not trades_df.empty:
        td = trades_df.copy()
        td["entry_date"] = td["entry_date"].astype(str)
        td["exit_date"] = td["exit_date"].astype(str)
        for _, row in td.iterrows():
            try:
                entry_d = pd.Timestamp(row["entry_date"])
                exit_d = pd.Timestamp(row["exit_date"])
                years_held = (exit_d - entry_d).days / 365.25
                if years_held <= 0:
                    years_held = 1.0 / 12
                stk_ret = float(row["ret"])
                # SPY return same period
                try:
                    pos_e = panel.index.searchsorted(entry_d)
                    pos_x = panel.index.searchsorted(exit_d)
                    pos_e = min(max(pos_e, 0), len(panel) - 1)
                    pos_x = min(max(pos_x, 0), len(panel) - 1)
                    spy_e = float(panel["SPY"].iloc[pos_e])
                    spy_x = float(panel["SPY"].iloc[pos_x])
                    spy_ret = (spy_x / spy_e - 1.0) if spy_e > 0 else 0.0
                except Exception:
                    spy_ret = 0.0
                pick_log_records.append({
                    "asof": str(entry_d.date()),
                    "ticker": str(row["ticker"]),
                    "entry_px": float(row["entry_px"]),
                    "exit_date": str(exit_d.date()),
                    "exit_px": float(row["exit_px"]),
                    "status": "exited",
                    "ret_strat": stk_ret,
                    "ret_spy": spy_ret,
                    "multiple_strat": 1.0 + stk_ret,
                    "multiple_spy": 1.0 + spy_ret,
                    "years_held": years_held,
                    "cagr_strat": ((1 + stk_ret) ** (1.0 / max(years_held, 0.01))) - 1.0,
                    "cagr_spy": ((1 + spy_ret) ** (1.0 / max(years_held, 0.01))) - 1.0,
                    "win": 1 if stk_ret > 0 else 0,
                    "beat_spy": 1 if stk_ret > spy_ret else 0,
                    "score": float(row.get("score", 0.0)),
                })
            except Exception:
                pass

    # Live picks
    live = _live_picks(panel, eval_at, strategy_rotation, WINNING_K)
    feats_now = load_features(eval_at)
    regime = _spy_regime(feats_now)

    # Walk-forward aggregate
    wf_rows = []
    wf_path = CACHE / "wf_winner_aggregate.csv"
    if wf_path.exists():
        wfdf = pd.read_csv(wf_path)
        for _, row in wfdf.iterrows():
            wf_rows.append({
                "key": f"{row['strategy']}::{int(row['k'])}",
                "n_splits_with_test_data": int(row["n_splits"]),
                "mean_test_cagr": float(row["mean_test_cagr"]),
                "median_test_cagr": float(row["median_test_cagr"]),
                "min_test_cagr": float(row["min_test_cagr"]),
                "max_test_cagr": float(row["max_test_cagr"]),
                "mean_test_edge": float(row["mean_edge"]),
                "min_test_edge": float(row["min_edge"]),
            })

    # Forced WF (definitive per-split for the winner)
    wf_forced_rows = []
    wff_path = CACHE / "wf_forced_aggregate.csv"
    if wff_path.exists():
        wfdf = pd.read_csv(wff_path)
        for _, row in wfdf.iterrows():
            name = row["name"]
            wf_forced_rows.append({
                "key": name,
                "n": int(row["n"]),
                "mean_test_cagr": float(row["mean_test"]),
                "median_test_cagr": float(row["median_test"]),
                "min_test_cagr": float(row["min_test"]),
                "max_test_cagr": float(row["max_test"]),
                "mean_test_edge": float(row["mean_edge"]),
            })

    # Headline matching old format
    headline = {
        "n_picks": int(res["n_trades"]),
        "win_rate_raw": float(win_rate),
        "win_rate_bias_corr": float("nan"),
        "cagr_raw": float(res["cagr_xirr"]),
        "cagr_total": float(res["cagr_total"]),
        "cagr_bias_corr": float(cagr_bias_corr_4pct) if cagr_bias_corr_4pct is not None else None,
        "cagr_spy_dca": float(spy["cagr_xirr"]),
        "edge": float(res["cagr_xirr"] - spy["cagr_xirr"]),
    }

    # Compute proper picks per strategy_rotation_k5 wf entry
    wf_winner_entry = next((r for r in wf_rows
                             if r["key"] == f"{WINNING_NAME}::{WINNING_K}"), None)
    if wf_winner_entry is None and wf_forced_rows:
        # Use forced
        for r in wf_forced_rows:
            if r["key"] == f"{WINNING_NAME} k={WINNING_K}":
                wf_winner_entry = {
                    "key": f"{WINNING_NAME}::{WINNING_K}",
                    "n_splits_with_test_data": r["n"],
                    "mean_test_cagr": r["mean_test_cagr"],
                    "median_test_cagr": r["median_test_cagr"],
                    "min_test_cagr": r["min_test_cagr"],
                    "max_test_cagr": r["max_test_cagr"],
                    "mean_test_edge": r["mean_test_edge"],
                    "min_test_edge": r["mean_test_edge"],
                }
                wf_rows.insert(0, wf_winner_entry)
                break

    wf_explanation = {
        "n_splits": 10,
        "headline_key": f"{WINNING_NAME}::{WINNING_K}",
        "headline_mean_test_cagr": wf_winner_entry["mean_test_cagr"] if wf_winner_entry else float("nan"),
        "headline_min_test_cagr": wf_winner_entry["min_test_cagr"] if wf_winner_entry else float("nan"),
        "headline_max_test_cagr": wf_winner_entry["max_test_cagr"] if wf_winner_entry else float("nan"),
        "explanation": (
            "Walk-forward TEST windows are 1-3 years long. The strategy is run with "
            "MONTHLY REBALANCE on each TEST window: each month-end, all capital is "
            "redeployed into the top-5 picks for the current SPY regime. Bear regimes "
            "skip the month (cash). The MEAN of those 10 short-window CAGRs is reported. "
            "The full 2002-2024 backtest CAGR (35.4%) is lower than the WF mean (40.5%) "
            "because the WF mean is dominated by recovery & strong-bull windows where "
            "compounding accelerates, while the full window also includes 2014-2016 and "
            "2020-2022 sideways/correction windows that compound more slowly."
        ),
    }

    # Survivorship dict (matching old structure)
    sensitivity_old_format = []
    if bias_rows:
        for r in bias_rows:
            sensitivity_old_format.append({
                "alpha": r["alpha"],
                "cagr_dca_median": r["cagr_median"],
                "cagr_dca_p10": r.get("cagr_p10"),
                "cagr_dca_p90": r.get("cagr_p90"),
                "edge_median": r["edge_median"],
            })
    survivorship = {
        "strategy": f"{WINNING_NAME}_k{WINNING_K}_{WINNING_EXIT}",
        "n_picks": int(res["n_trades"]),
        "raw_cagr": float(res["cagr_xirr"]),
        "stratified_default_4pct": next((r for r in bias_rows
                                          if abs(r["alpha"] - 0.04) < 1e-6), {}),
        "sensitivity": sensitivity_old_format,
        "delisted_augmentation": {
            "tickers_attempted": [],
            "tickers_with_data": [],
            "n_with_data": 0,
        },
    }

    # Horizon stats
    horizon_stats = _horizon_stats(panel, strategy_rotation, WINNING_K, eval_at)

    # Equity curve — date+equity
    growth = []
    for _, row in res["equity_curve"].iterrows():
        growth.append({
            "date": str(pd.Timestamp(row["date"]).date()),
            "equity": float(row["equity"]),
            "deposited": None,
        })

    # Regime history (24m)
    regime_history = []
    feat_dir = CACHE / "features"
    feat_files = sorted([p.stem for p in feat_dir.glob("*.parquet")])
    feat_files = feat_files[-24:]
    for f in feat_files:
        try:
            asof = pd.Timestamp(f)
            try:
                feats = load_features(asof)
                regime_history.append({"date": str(asof.date()), "regime": _spy_regime(feats)})
            except Exception:
                pass
        except Exception:
            pass

    # Sweep top 40 from cache (for research view)
    sweep_rows = []
    sweep_path = CACHE / "sweep_monthly_rebalance.csv"
    if sweep_path.exists():
        sw = pd.read_csv(sweep_path)
        for _, row in sw.head(40).iterrows():
            sweep_rows.append({
                "strategy": str(row["name"]),
                "top_k": int(row["k"]),
                "exit": "monthly_rebalance",
                "cagr_dca_portfolio": float(row["cagr"]),
                "edge_vs_spy_dca": float(row["cagr"] - spy["cagr_xirr"]),
                "win_rate": None,
                "mean_ret": None,
            })

    # First pick of month (singular form for legacy code)
    pick_of_month = live[0] if live else None

    data = {
        "as_of": str(eval_at.date()),
        "panel": {
            "n_tickers": int(panel.shape[1]),
            "first_date": str(panel.index.min().date()),
            "last_date": str(panel.index.max().date()),
        },
        "spy_dca_cagr": float(spy["cagr_xirr"]),
        "headline": headline,
        "current_regime": regime,
        "regime_history_24m": regime_history,
        "pick_of_month": pick_of_month,
        "pick_of_month_basket": live,
        "recommended_strategy": {
            "name": WINNING_NAME, "top_k": int(WINNING_K),
            "exit_rule": WINNING_EXIT,
            "description": WINNING_DESCRIPTION,
        },
        "growth": growth,
        "horizon_stats": horizon_stats,
        "wf_explanation": wf_explanation,
        "survivorship": survivorship,
        "bias_sensitivity": bias_rows,
        "windows_comparison": windows,
        "live_picks": {
            "strategy_rotation": live,
        },
        "walk_forward_aggregate": wf_rows,
        "walk_forward_forced": wf_forced_rows,
        "year_by_year": year_by_year,
        "oracle": {},
        "pick_log": pick_log_records,
        "sweep_top40": sweep_rows,
    }

    DATA_OUT.write_text(json.dumps(to_jsonable(data), indent=2))
    print(f"\nWrote {DATA_OUT}", flush=True)
    print(f"  Live picks: {[p['ticker'] for p in live]}", flush=True)
    print(f"  Regime: {regime}", flush=True)
    print(f"  Pick log entries: {len(pick_log_records)}", flush=True)
    print(f"  Year-by-year: {len(yb_list)}", flush=True)
    print(f"  Horizon stats: {len(horizon_stats)}", flush=True)


if __name__ == "__main__":
    main()
