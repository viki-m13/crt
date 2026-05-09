"""Save full picks log + summary stats for the best alpha strategy.

Run after a strategy is selected. Reads the strategy from a registry mapping
name -> score function.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from experiments.monthly_dca.fast_score import (
    BENCH_EXCLUDED, load_features_long, load_fwd, load_panel,
)
from experiments.monthly_dca.fast_engine import xirr
from experiments.monthly_dca.strategies_alpha import (
    nova_star, smooth_trend_compounder, persistent_winner, multibagger_engine,
    consensus_top_decile, alpha_intersect, asymmetric_recovery_plus, nova_star_deep,
    rs_beast, rank_intersect, fallen_angel_recovery, vol_contraction_breakout,
    institutional_accumulation, nova_regime, nova_star_pro, clean_compounder,
)
from experiments.monthly_dca.strategies_alpha2 import (
    alpha_omega, alpha_omega_deep, alpha_omega_momentum, ultra_nova,
    nova_dual, nova_star_prime, nova_regime_x, nova_sharpe, the_bagger,
    multibagger_max, apex, nova_tier1,
)


REGISTRY = {
    "nova_star": nova_star,
    "smooth_trend_compounder": smooth_trend_compounder,
    "persistent_winner": persistent_winner,
    "multibagger_engine": multibagger_engine,
    "consensus_top_decile": consensus_top_decile,
    "alpha_intersect": alpha_intersect,
    "asymmetric_recovery_plus": asymmetric_recovery_plus,
    "nova_star_deep": nova_star_deep,
    "rs_beast": rs_beast,
    "rank_intersect": rank_intersect,
    "fallen_angel_recovery": fallen_angel_recovery,
    "vol_contraction_breakout": vol_contraction_breakout,
    "institutional_accumulation": institutional_accumulation,
    "nova_regime": nova_regime,
    "nova_star_pro": nova_star_pro,
    "clean_compounder": clean_compounder,
    "alpha_omega": alpha_omega,
    "alpha_omega_deep": alpha_omega_deep,
    "alpha_omega_momentum": alpha_omega_momentum,
    "ultra_nova": ultra_nova,
    "nova_dual": nova_dual,
    "nova_star_prime": nova_star_prime,
    "nova_regime_x": nova_regime_x,
    "nova_sharpe": nova_sharpe,
    "the_bagger": the_bagger,
    "multibagger_max": multibagger_max,
    "apex": apex,
    "nova_tier1": nova_tier1,
}


def picks_for(score_fn, top_k: int, start: str = "1997-01-01", end: str = "2024-12-31"):
    feats = load_features_long()
    feats = feats.loc[(feats.index.get_level_values("asof") >= pd.Timestamp(start)) &
                      (feats.index.get_level_values("asof") <= pd.Timestamp(end))]
    chunks = []
    for asof, sub in feats.groupby(level="asof"):
        df_asof = sub.copy()
        df_asof.index = df_asof.index.get_level_values("ticker")
        try:
            s = score_fn(df_asof).dropna()
        except Exception:
            continue
        bad = [t for t in BENCH_EXCLUDED if t in s.index]
        s = s.drop(bad, errors="ignore")
        if s.empty:
            continue
        s = s.reset_index()
        s.columns = ["ticker", "score"]
        s["asof"] = asof
        chunks.append(s)
    if not chunks:
        return pd.DataFrame(columns=["asof", "ticker", "score"])
    df = pd.concat(chunks, ignore_index=True)
    df = df.sort_values(["asof", "score"], ascending=[True, False])
    return df.groupby("asof", group_keys=False).head(top_k).reset_index(drop=True)


def merge_with_features_and_fwd(picks: pd.DataFrame) -> pd.DataFrame:
    feats = load_features_long().reset_index()
    fwd = load_fwd().reset_index()
    out = picks.merge(feats, on=["asof", "ticker"], how="left")
    out = out.merge(fwd[[c for c in fwd.columns if c == "asof" or c == "ticker"
                          or c.startswith("ret__") or c.startswith("days__")]],
                     on=["asof", "ticker"], how="left")
    return out


def cagr_dca_for(picks_with_ret: pd.DataFrame, ret_col: str, eval_at, panel,
                  delist_iters: int = 200, delist_p: float = 0.04) -> dict:
    df = picks_with_ret.dropna(subset=[ret_col])
    if df.empty:
        return {}
    fv = df[ret_col].to_numpy(dtype=float)
    ah = pd.to_datetime(df["asof"].to_numpy())
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
    cashflows = [(pd.Timestamp(t), -1.0) for t in ah]
    cashflows.append((eval_at, float(np.sum(1 + fv))))
    cagr = xirr(cashflows)
    cashflows_spy = [(pd.Timestamp(t), -1.0) for t in ah]
    cashflows_spy.append((eval_at, float(np.sum(1 + bv[np.isfinite(bv)]))))
    cagr_spy = xirr(cashflows_spy)

    rng = np.random.default_rng(0)
    if delist_iters > 0:
        days_to_eval = np.asarray([(eval_at - t).days for t in ah], dtype=float)
        years_to_eval = np.maximum(days_to_eval, 1.0) / 365.25
        p_del = 1.0 - (1.0 - delist_p) ** years_to_eval
        cagrs = []
        wins = []
        for it in range(delist_iters):
            u = rng.random(len(fv))
            fv_mc = np.where(u < p_del, -1.0, fv)
            cf = [(pd.Timestamp(t), -1.0) for t in ah]
            cf.append((eval_at, float(np.sum(1 + fv_mc))))
            try:
                cagrs.append(xirr(cf))
            except Exception:
                pass
            wins.append(float((fv_mc > 0).mean()))
        cagr_bc = float(np.median([c for c in cagrs if np.isfinite(c)])) if cagrs else float("nan")
        win_bc = float(np.median(wins))
    else:
        cagr_bc = float("nan")
        win_bc = float("nan")

    return {
        "n": int(len(fv)),
        "win_rate": float((fv > 0).mean()),
        "win_rate_bias_corr_median": win_bc,
        "median_ret": float(np.nanmedian(fv)),
        "mean_ret": float(np.nanmean(fv)),
        "cagr_dca": float(cagr),
        "cagr_dca_bias_corr_median": cagr_bc,
        "cagr_spy_dca": float(cagr_spy),
        "edge": float(cagr - cagr_spy),
    }


def per_year_breakdown(picks_with_ret: pd.DataFrame, ret_col: str, panel, eval_at):
    df = picks_with_ret.dropna(subset=[ret_col]).copy()
    df["year"] = pd.to_datetime(df["asof"]).dt.year
    spy = panel["SPY"].dropna()
    rows = []
    for year, g in df.groupby("year"):
        n = len(g)
        ah = pd.to_datetime(g["asof"].to_numpy())
        fv = g[ret_col].to_numpy(dtype=float)
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
        cashflows = [(pd.Timestamp(t), -1.0) for t in ah]
        cashflows.append((eval_at, float(np.sum(1 + fv))))
        cagr = xirr(cashflows)
        cashflows_spy = [(pd.Timestamp(t), -1.0) for t in ah]
        cashflows_spy.append((eval_at, float(np.sum(1 + bv[np.isfinite(bv)]))))
        cagr_spy = xirr(cashflows_spy)
        rows.append({
            "year": int(year),
            "n_picks": int(n),
            "win_rate": float((fv > 0).mean()),
            "median_ret": float(np.nanmedian(fv)),
            "cagr_dca": float(cagr),
            "cagr_dca_spy": float(cagr_spy),
            "edge": float(cagr - cagr_spy),
        })
    return pd.DataFrame(rows)


def save_strategy(name: str, top_k: int = 5, exit_rule: str = "fixed_3y",
                  start: str = "1997-01-01", end: str = "2024-12-31"):
    if name not in REGISTRY:
        raise ValueError(f"unknown strategy: {name}; pick from {list(REGISTRY)}")
    fn = REGISTRY[name]
    panel = load_panel()
    eval_at = panel.index.max()
    print(f"=== Saving {name} (top_k={top_k}, exit={exit_rule}) ===")

    picks = picks_for(fn, top_k=top_k, start=start, end=end)
    print(f"  total picks: {len(picks)} unique tickers: {picks['ticker'].nunique()}")
    if picks.empty:
        return

    out = merge_with_features_and_fwd(picks)
    out_path = Path(f"experiments/monthly_dca/cache/picks_full_{name}_k{top_k}.csv")
    out.to_csv(out_path, index=False)
    print(f"  wrote {out_path}")

    # Summary stats
    ret_col = f"ret__{exit_rule}"
    stats = cagr_dca_for(out, ret_col, eval_at, panel, delist_iters=200)
    print(f"  stats: {stats}")

    # Year-by-year
    yb = per_year_breakdown(out, ret_col, panel, eval_at)
    yb.rename(columns={"cagr_dca_spy": "cagr_dca_spy"}, inplace=True)
    yb.to_csv(Path(f"experiments/monthly_dca/cache/yb_{name}_k{top_k}.csv"), index=False)
    print(f"  wrote yb_{name}_k{top_k}.csv")

    summary = {
        "strategy": name,
        "top_k": top_k,
        "exit_rule": exit_rule,
        "eval_at": str(eval_at.date()),
        "stats": {k: (None if pd.isna(v) else float(v)) for k, v in stats.items()},
        "year_by_year": yb.to_dict(orient="records"),
        "n_picks": int(len(picks)),
        "first_pick_date": str(pd.to_datetime(picks["asof"]).min().date()),
        "last_pick_date": str(pd.to_datetime(picks["asof"]).max().date()),
        "unique_tickers": int(picks["ticker"].nunique()),
        "ticker_frequency": picks["ticker"].value_counts().head(20).to_dict(),
    }
    with open(Path(f"experiments/monthly_dca/cache/summary_{name}_k{top_k}.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  wrote summary_{name}_k{top_k}.json")


def main():
    if len(sys.argv) < 2:
        print("Usage: save_alpha_picks.py STRATEGY_NAME [top_k] [exit_rule]")
        print(f"Available: {list(REGISTRY.keys())}")
        sys.exit(1)
    name = sys.argv[1]
    top_k = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    exit_rule = sys.argv[3] if len(sys.argv) > 3 else "fixed_3y"
    save_strategy(name, top_k, exit_rule)


if __name__ == "__main__":
    main()
