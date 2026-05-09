"""Run all alpha strategies, sweep top-K and exit rules, save results."""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)

from experiments.monthly_dca.fast_score import (
    BENCH_EXCLUDED,
    load_features_long,
    load_fwd,
    load_panel,
)
from experiments.monthly_dca.fast_engine import xirr
from experiments.monthly_dca.strategies_alpha import all_alpha_strategies
from experiments.monthly_dca.strategies_alpha2 import all_alpha2_strategies
from experiments.monthly_dca.strategies_fast import (
    pullback_in_winner, blended_pullback_momentum, quality_pullback,
    explosive_winners, dual_momentum,
)


CACHE = Path("experiments/monthly_dca/cache")


def evaluate_strategy_fast(score_fn, top_k: int, name: str,
                           start: str = "2002-01-01", end: str = "2024-12-31",
                           rules: list[str] = None,
                           feats_long: pd.DataFrame = None,
                           fwd: pd.DataFrame = None,
                           panel: pd.DataFrame = None,
                           delist_iters: int = 100,
                           delist_prob_annual: float = 0.04) -> list[dict]:
    if rules is None:
        rules = ["hold_forever", "fixed_3y", "fixed_5y"]

    if feats_long is None:
        feats_long = load_features_long()
    if fwd is None:
        fwd = load_fwd()
    if panel is None:
        panel = load_panel()

    feats = feats_long.loc[(feats_long.index.get_level_values("asof") >= pd.Timestamp(start)) &
                           (feats_long.index.get_level_values("asof") <= pd.Timestamp(end))]
    fwd_w = fwd.loc[(fwd.index.get_level_values("asof") >= pd.Timestamp(start)) &
                    (fwd.index.get_level_values("asof") <= pd.Timestamp(end))]

    # Score per (asof, ticker) by grouping per asof
    scores_chunks = []
    for asof, sub in feats.groupby(level="asof"):
        df_asof = sub.copy()
        df_asof.index = df_asof.index.get_level_values("ticker")
        try:
            s = score_fn(df_asof)
        except Exception as e:
            continue
        s = s.dropna()
        # Universe filter
        bad = [t for t in BENCH_EXCLUDED if t in s.index]
        s = s.drop(bad, errors="ignore")
        if s.empty:
            continue
        s = s.reset_index()
        s.columns = ["ticker", "score"]
        s["asof"] = asof
        scores_chunks.append(s)
    if not scores_chunks:
        return []
    df = pd.concat(scores_chunks, ignore_index=True)
    df = df.sort_values(["asof", "score"], ascending=[True, False])
    picks = df.groupby("asof", group_keys=False).head(top_k).reset_index(drop=True)
    if picks.empty:
        return []

    fwd_reset = fwd_w.reset_index()
    merged = picks.merge(fwd_reset, on=["asof", "ticker"], how="left")
    eval_at = panel.index.max()

    rng = np.random.default_rng(42)
    out_rows = []
    for rule in rules:
        rc = f"ret__{rule}"
        if rc not in merged.columns:
            continue
        f = merged[rc].to_numpy(dtype=float)
        days_col = f"days__{rule}"
        d = merged[days_col].to_numpy(dtype=float) if days_col in merged.columns else np.full(len(f), np.nan)
        valid = np.isfinite(f)
        if not valid.any():
            continue
        fv = f[valid]
        dv = d[valid]
        ah = pd.to_datetime(merged["asof"].to_numpy()[valid])

        # SPY same window
        spy = panel["SPY"].dropna()
        bv = []
        for asof_t in ah:
            pos = spy.index.searchsorted(asof_t)
            if pos >= len(spy):
                bv.append(np.nan); continue
            arr = spy.iloc[pos:].to_numpy(dtype=float)
            mask = np.isfinite(arr)
            if mask.any():
                bv.append(arr[mask][-1] / arr[0] - 1.0)
            else:
                bv.append(np.nan)
        bv = np.asarray(bv, dtype=float)

        win = float((fv > 0).mean())
        beat = float((fv > bv).mean())

        # Bias-corrected
        if delist_iters > 0:
            days_to_eval = np.asarray([(eval_at - t).days for t in ah], dtype=float)
            years_to_eval = np.maximum(days_to_eval, 1.0) / 365.25
            p_del = 1.0 - (1.0 - delist_prob_annual) ** years_to_eval
            wins, means = [], []
            for it in range(delist_iters):
                u = rng.random(len(fv))
                fv_mc = np.where(u < p_del, -1.0, fv)
                wins.append(float((fv_mc > 0).mean()))
                means.append(float(fv_mc.mean()))
            win_corr = float(np.median(wins))
            mean_corr = float(np.median(means))
        else:
            win_corr = float("nan"); mean_corr = float("nan")

        # CAGR DCA portfolio
        cashflows = [(pd.Timestamp(t), -1.0) for t in ah]
        cashflows.append((eval_at, float(np.sum(1 + fv))))
        cagr_dca = xirr(cashflows)
        cashflows_spy = [(pd.Timestamp(t), -1.0) for t in ah]
        cashflows_spy.append((eval_at, float(np.sum(1 + bv[np.isfinite(bv)]))))
        cagr_spy = xirr(cashflows_spy)

        # Per-pick CAGR median
        years_held = np.maximum(dv, 1) / 252.0
        per_pick_cagr = (1 + fv) ** (1.0 / years_held) - 1.0
        per_pick_cagr_median = float(np.nanmedian(per_pick_cagr))

        # Mean & p10
        mean_ret = float(np.nanmean(fv))
        p10 = float(np.nanpercentile(fv, 10))
        p90 = float(np.nanpercentile(fv, 90))
        median_ret = float(np.nanmedian(fv))

        # Worst-year CAGR
        df_pick_yr = pd.DataFrame({"asof": ah, "ret": fv, "spy": bv})
        df_pick_yr["year"] = df_pick_yr["asof"].dt.year
        yearly = df_pick_yr.groupby("year").agg(
            n=("ret", "size"),
            mean_ret=("ret", "mean"),
            spy_ret=("spy", "mean"),
        ).reset_index()
        # Per-year CAGR (geometric over hold)
        worst_year_edge = float(np.min((yearly["mean_ret"] - yearly["spy_ret"]).values))
        years_below_spy = int((yearly["mean_ret"] < yearly["spy_ret"]).sum())

        out_rows.append({
            "strategy": name, "exit": rule, "top_k": top_k,
            "n_picks": int(valid.sum()),
            "win_rate": win, "win_rate_bias_corr": win_corr,
            "beat_spy_rate": beat,
            "median_ret": median_ret, "mean_ret": mean_ret,
            "p10_ret": p10, "p90_ret": p90,
            "per_pick_cagr_median": per_pick_cagr_median,
            "cagr_dca_portfolio": cagr_dca,
            "cagr_spy_dca": cagr_spy,
            "edge_vs_spy_dca": cagr_dca - cagr_spy,
            "mean_ret_bias_corr": mean_corr,
            "worst_year_edge": worst_year_edge,
            "years_below_spy": years_below_spy,
            "n_years": int(yearly["year"].nunique()),
        })
    return out_rows


def main():
    print("Loading data...", flush=True)
    feats = load_features_long()
    fwd = load_fwd()
    panel = load_panel()
    eval_at = panel.index.max()
    print(f"  features: {feats.shape}")
    print(f"  fwd: {fwd.shape}")
    print(f"  eval_at: {eval_at.date()}")

    all_rows: list[dict] = []
    rules = ["hold_forever", "fixed_3y", "fixed_5y"]
    top_ks = [1, 2, 3, 5, 10]

    # Evaluate baselines for comparison
    baselines = [
        ("blended_pullback_momentum", blended_pullback_momentum),
        ("pullback_in_winner", pullback_in_winner),
        ("quality_pullback", quality_pullback),
        ("explosive_winners", explosive_winners),
        ("dual_momentum", dual_momentum),
    ]
    print("\n=== Baselines ===")
    for name, fn in baselines:
        for k in top_ks:
            print(f"  {name} k={k}", flush=True)
            rows = evaluate_strategy_fast(fn, k, name, rules=rules,
                                           feats_long=feats, fwd=fwd, panel=panel)
            all_rows.extend(rows)

    # Evaluate alpha strategies
    print("\n=== Alpha strategies ===")
    for strat in all_alpha_strategies(top_k=5):  # top_k will be overridden in loop
        for k in top_ks:
            print(f"  {strat.name} k={k}", flush=True)
            rows = evaluate_strategy_fast(strat.score_fn, k, strat.name, rules=rules,
                                           feats_long=feats, fwd=fwd, panel=panel)
            all_rows.extend(rows)

    print("\n=== Alpha2 strategies (ensembles) ===")
    for strat in all_alpha2_strategies(top_k=5):
        for k in top_ks:
            print(f"  {strat.name} k={k}", flush=True)
            rows = evaluate_strategy_fast(strat.score_fn, k, strat.name, rules=rules,
                                           feats_long=feats, fwd=fwd, panel=panel)
            all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    df = df.sort_values("cagr_dca_portfolio", ascending=False)
    df.to_csv(CACHE / "sweep_alpha.csv", index=False)
    print(f"\nWrote {CACHE / 'sweep_alpha.csv'}: {df.shape}")
    print()
    print(df.head(40).to_string(index=False))


if __name__ == "__main__":
    main()
