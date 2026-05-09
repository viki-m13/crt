"""Build the next-gen data.json for the main page.

Uses the winning strategy from the APEX sweep + walk-forward validation.

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

from experiments.monthly_dca.compound_engine import (
    ExitSpec, Strategy as CompStrategy,
    benchmark_spy_dca, run_compound,
)
from experiments.monthly_dca.fast_engine import load_panel, load_features
from experiments.monthly_dca.strategies_apex import (
    apex_reloaded, apex_turbocharged, apex_balanced, apex_hybrid,
    _spy_regime,
)
from experiments.monthly_dca.strategies_v3 import dyn_conc_score, dyn_conc_k


CACHE = Path(__file__).resolve().parent / "cache"
OUT = Path(__file__).resolve().parents[2] / "experiments" / "docs" / "monthly-dca"
OUT.mkdir(parents=True, exist_ok=True)
DATA_OUT = OUT / "data.json"


# Will be set by main() based on sweep results
WINNING_STRATEGY: Callable | None = None
WINNING_NAME = ""
WINNING_K = 5
WINNING_EXIT_NAME = "monthly_rebalance"
WINNING_EXIT: ExitSpec | None = None
WINNING_DESCRIPTION = ""


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
    s = s[~s.index.isin(("SPY","QQQ","IWM","VTI","RSP","DIA"))]
    top = s.sort_values(ascending=False).head(k)
    out = []
    feature_cols = ["price","pullback_1y","trend_health_5y","recovery_rate","rsi_14",
                    "mom_12_1","mom_3y","d_sma200","rs_12m_spy","trend_r2_12m",
                    "tail_ratio_24m","sharpe_12m","quality_score_5y","beta_2y",
                    "idio_mom_12_1","breakout_strength_60","fip_score",
                    "frac_above_50dma_1y","mom_consistency_12m","near_52wh_60d",
                    "vol_expansion_24m","accel"]
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


def _regime_label_at(asof, panel):
    try:
        feats = load_features(asof)
        return _spy_regime(feats)
    except Exception:
        return "default"


def build_main(panel, eval_at, start, end):
    """Run the winning strategy on the full window, return picks + curve."""
    global WINNING_STRATEGY, WINNING_K, WINNING_EXIT
    res = run_compound(
        panel, CompStrategy(WINNING_NAME, WINNING_STRATEGY, top_k=WINNING_K),
        WINNING_EXIT, start=start, end=end,
        eval_at=pd.Timestamp(eval_at), cost_bps=5.0,
    )
    return res


def build_data_json(force_strategy=None, force_k=None, force_exit=None,
                     force_name=None, force_desc=None):
    global WINNING_STRATEGY, WINNING_K, WINNING_EXIT, WINNING_NAME
    global WINNING_EXIT_NAME, WINNING_DESCRIPTION
    if force_strategy is not None:
        WINNING_STRATEGY = force_strategy
    if force_k is not None:
        WINNING_K = force_k
    if force_exit is not None:
        WINNING_EXIT = force_exit
        WINNING_EXIT_NAME = force_exit.name
    if force_name is not None:
        WINNING_NAME = force_name
    if force_desc is not None:
        WINNING_DESCRIPTION = force_desc

    panel = load_panel()
    eval_at = pd.Timestamp("2026-05-07")
    if eval_at > panel.index[-1]:
        eval_at = panel.index[-1]

    start = "2002-01-31"
    end = "2024-12-31"

    res = build_main(panel, eval_at, start, end)

    # SPY DCA benchmark
    spy = benchmark_spy_dca(panel, start=start, end=end, eval_at=eval_at)

    # Equity curve
    eq_df = res.equity_curve.copy()
    eq_df["date"] = eq_df["date"].astype(str)
    growth = eq_df.to_dict(orient="records")

    # Trade log
    trades_df = res.trades.copy()
    trades_records = []
    if not trades_df.empty:
        for col in ("entry_date", "exit_date"):
            if col in trades_df.columns:
                trades_df[col] = trades_df[col].astype(str)
        trades_records = trades_df.to_dict(orient="records")

    # Year-by-year CAGR using equity_curve
    year_records = []
    if not eq_df.empty:
        eq = res.equity_curve.copy()
        eq["year"] = pd.to_datetime(eq["date"]).dt.year
        for year, grp in eq.groupby("year"):
            start_eq = float(grp.iloc[0]["equity"])
            end_eq = float(grp.iloc[-1]["equity"])
            if start_eq <= 0:
                continue
            year_ret = end_eq / start_eq - 1.0
            year_records.append({"year": int(year), "year_return": year_ret,
                                  "start_equity": start_eq, "end_equity": end_eq,
                                  "n_trades": int((trades_df["entry_date"].astype(str).str.startswith(str(year))).sum() if not trades_df.empty else 0)})

    # Live picks for current month
    live = _live_picks(panel, eval_at, WINNING_STRATEGY, WINNING_K)
    regime = _regime_label_at(eval_at, panel)

    # Bias sensitivity
    bias_rows = []
    for alpha in [0.0, 0.04, 0.08, 0.12, 0.16, 0.20]:
        cagrs, edges = [], []
        for seed in range(10):
            r = run_compound(
                panel, CompStrategy(WINNING_NAME, WINNING_STRATEGY, top_k=WINNING_K),
                WINNING_EXIT, start=start, end=end, eval_at=eval_at,
                delist_alpha=alpha, delist_seed=seed, cost_bps=5.0,
            )
            cagrs.append(r.cagr_money_weighted)
            edges.append(r.cagr_money_weighted - spy["cagr_xirr"])
        bias_rows.append({
            "alpha": alpha, "cagr_p10": float(np.percentile(cagrs, 10)),
            "cagr_median": float(np.median(cagrs)),
            "cagr_p90": float(np.percentile(cagrs, 90)),
            "edge_median": float(np.median(edges)),
        })

    # Multi-window comparison
    windows = []
    for win_start, win_end, win_label in [
        ("2002-01-31", "2024-12-31", "Full 2002-2024"),
        ("2010-01-31", "2024-12-31", "Modern 2010-2024"),
        ("2018-01-31", "2024-12-31", "Recent 2018-2024"),
    ]:
        try:
            wr = run_compound(
                panel, CompStrategy(WINNING_NAME, WINNING_STRATEGY, top_k=WINNING_K),
                WINNING_EXIT, start=win_start, end=win_end, eval_at=eval_at, cost_bps=5.0,
            )
            wspy = benchmark_spy_dca(panel, start=win_start, end=win_end, eval_at=eval_at)
            windows.append({
                "label": win_label, "start": win_start, "end": win_end,
                "cagr": float(wr.cagr_money_weighted),
                "cagr_spy": float(wspy["cagr_xirr"]),
                "edge": float(wr.cagr_money_weighted - wspy["cagr_xirr"]),
                "final_equity": float(wr.final_equity),
                "deposited": float(wr.total_deposited),
            })
        except Exception as e:
            print(f"  window {win_label}: {e}")

    # Read existing walk-forward aggregate if present
    wf_records = []
    wf_path = CACHE / "wf_apex_aggregate.csv"
    if wf_path.exists():
        wf_df = pd.read_csv(wf_path)
        # Find our winning combo's row
        m = wf_df[(wf_df["strategy"] == WINNING_NAME)
                  & (wf_df["k"] == WINNING_K)
                  & (wf_df["exit"] == WINNING_EXIT_NAME)]
        wf_records = wf_df.to_dict(orient="records")

    data = {
        "as_of": str(eval_at.date()),
        "panel": {
            "n_tickers": int(panel.shape[1]),
            "first_date": str(panel.index.min().date()),
            "last_date": str(panel.index.max().date()),
        },
        "current_regime": regime,
        "spy_dca_cagr": float(spy["cagr_xirr"]),
        "headline": {
            "cagr_raw": float(res.cagr_money_weighted),
            "cagr_spy_dca": float(spy["cagr_xirr"]),
            "edge_vs_spy": float(res.cagr_money_weighted - spy["cagr_xirr"]),
            "final_equity": float(res.final_equity),
            "total_deposited": float(res.total_deposited),
            "n_trades": int(res.n_trades),
        },
        "recommended_strategy": {
            "name": WINNING_NAME, "top_k": int(WINNING_K),
            "exit_rule": WINNING_EXIT_NAME,
            "description": WINNING_DESCRIPTION,
        },
        "pick_of_month_basket": live,
        "growth": growth,
        "year_by_year": year_records,
        "walk_forward_aggregate": wf_records,
        "bias_sensitivity": bias_rows,
        "windows_comparison": windows,
        "trades": trades_records[-200:] if len(trades_records) > 200 else trades_records,
    }

    DATA_OUT.write_text(json.dumps(to_jsonable(data), indent=2))
    print(f"\nWrote {DATA_OUT}", flush=True)
    print(f"  CAGR: {res.cagr_money_weighted:.4f}", flush=True)
    print(f"  Edge: {res.cagr_money_weighted - spy['cagr_xirr']:+.4f}", flush=True)
    print(f"  Live picks: {[p['ticker'] for p in live]}", flush=True)
    return data


def main():
    """Main entry: pick winner from sweep CSV, build data.json."""
    sweep_path = CACHE / "sweep_apex_focused_full.csv"
    if not sweep_path.exists():
        # Fallback: use apex_balanced k=5 monthly_rebalance
        global WINNING_STRATEGY, WINNING_NAME, WINNING_K, WINNING_EXIT, WINNING_EXIT_NAME
        WINNING_STRATEGY = apex_balanced
        WINNING_NAME = "apex_balanced"
        WINNING_K = 5
        WINNING_EXIT = ExitSpec("monthly_rebalance", monthly_rebalance=True)
        WINNING_EXIT_NAME = "monthly_rebalance"
        WINNING_DESCRIPTION = (
            "APEX BALANCED — multi-leg compounding portfolio with monthly "
            "rebalance. Combines momentum, quality and asymmetry legs with "
            "regime-conditional weights and hard delisting filters."
        )
        return build_data_json()

    df = pd.read_csv(sweep_path)
    # Pick winner by highest CAGR XIRR
    winner = df.nlargest(1, "cagr_xirr").iloc[0]
    name = winner["strategy"]
    k = int(winner["k"])
    exit_name = winner["exit"]
    print(f"Winner from sweep: {name} k={k} exit={exit_name} CAGR={winner['cagr_xirr']:.4f}")
    raise SystemExit("Use build_data_json() with explicit args.")


if __name__ == "__main__":
    main()
