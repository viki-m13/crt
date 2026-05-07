"""Year-by-year breakdown, bias-corrected CAGR, ensemble strategies, OOS walk-forward."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from experiments.monthly_dca.fast_score import (
    BENCH_EXCLUDED,
    load_features_long,
    load_fwd,
    load_panel,
)
from experiments.monthly_dca.fast_engine import xirr
from experiments.monthly_dca.strategies_fast import (
    pullback_in_winner,
    quality_pullback,
    dual_momentum,
    explosive_winners,
    proprietary_v6,
    proprietary_v8,
    winner_only,
    min_dd_compounders,
    proprietary_v3,
    proprietary_v4,
    proprietary_v5,
    proprietary_v7,
    low_vol_trend,
)


# ---------------------------------------------------------------------------
def picks_for(score_fn, top_k: int = 5, start: str | None = None,
              end: str | None = None) -> pd.DataFrame:
    feats = load_features_long()
    if start:
        feats = feats.loc[feats.index.get_level_values("asof") >= pd.Timestamp(start)]
    if end:
        feats = feats.loc[feats.index.get_level_values("asof") <= pd.Timestamp(end)]
    scores = score_fn(feats).dropna()
    df = scores.reset_index()
    df.columns = ["asof", "ticker", "score"]
    df = df[~df["ticker"].isin(BENCH_EXCLUDED)]
    df = df.sort_values(["asof", "score"], ascending=[True, False])
    return df.groupby("asof", group_keys=False).head(top_k).reset_index(drop=True)


def merge_fwd(picks: pd.DataFrame) -> pd.DataFrame:
    fwd = load_fwd().reset_index()
    return picks.merge(fwd, on=["asof", "ticker"], how="left")


def cagr_dca(merged: pd.DataFrame, ret_col: str, eval_at: pd.Timestamp,
             panel: pd.DataFrame, delist_iters: int = 0,
             delist_prob_annual: float = 0.04) -> dict:
    f = merged[ret_col].to_numpy(dtype=float)
    valid = np.isfinite(f)
    fv = f[valid]
    ah = pd.to_datetime(merged["asof"].to_numpy()[valid])
    if len(fv) == 0:
        return {}
    spy = panel["SPY"]
    bv = []
    for asof_t in ah:
        pos = panel.index.searchsorted(asof_t)
        if pos >= len(panel.index):
            bv.append(np.nan); continue
        arr = spy.iloc[pos:].to_numpy(dtype=float)
        mask = np.isfinite(arr)
        bv.append(arr[mask][-1] / arr[0] - 1.0 if mask.any() else np.nan)
    bv = np.asarray(bv, dtype=float)

    cf = [(pd.Timestamp(t), -1.0) for t in ah]
    cf.append((eval_at, float(np.sum(1 + fv))))
    c = xirr(cf)
    cf_spy = [(pd.Timestamp(t), -1.0) for t in ah]
    cf_spy.append((eval_at, float(np.sum(1 + bv[np.isfinite(bv)]))))
    c_spy = xirr(cf_spy)

    out = {
        "n": int(len(fv)),
        "win_rate": float((fv > 0).mean()),
        "beat_spy_rate": float((fv > bv).mean()),
        "median_ret": float(np.nanmedian(fv)),
        "mean_ret": float(np.nanmean(fv)),
        "cagr_dca": c,
        "cagr_spy_dca": c_spy,
        "edge": c - c_spy,
    }

    if delist_iters > 0:
        rng = np.random.default_rng(0)
        days_to_eval = np.asarray([(eval_at - t).days for t in ah], dtype=float)
        years_to_eval = np.maximum(days_to_eval, 1.0) / 365.25
        p_del = 1.0 - (1.0 - delist_prob_annual) ** years_to_eval
        cagrs = []
        wins = []
        for it in range(delist_iters):
            u = rng.random(len(fv))
            fv_mc = np.where(u < p_del, -1.0, fv)
            cf_mc = [(pd.Timestamp(t), -1.0) for t in ah]
            cf_mc.append((eval_at, float(np.sum(1 + fv_mc))))
            try:
                cagrs.append(xirr(cf_mc))
            except Exception:
                cagrs.append(np.nan)
            wins.append(float((fv_mc > 0).mean()))
        out["cagr_dca_bias_corr_median"] = float(np.nanmedian(cagrs))
        out["cagr_dca_bias_corr_p10"] = float(np.nanpercentile(cagrs, 10))
        out["cagr_dca_bias_corr_p90"] = float(np.nanpercentile(cagrs, 90))
        out["win_rate_bias_corr_median"] = float(np.median(wins))

    return out


def per_year_breakdown(merged: pd.DataFrame, ret_col: str, panel: pd.DataFrame,
                       eval_at: pd.Timestamp) -> pd.DataFrame:
    rows = []
    asof = pd.to_datetime(merged["asof"])
    for y in sorted(asof.dt.year.unique()):
        sub = merged[asof.dt.year == y].copy()
        f = sub[ret_col].to_numpy(dtype=float)
        valid = np.isfinite(f)
        fv = f[valid]
        if len(fv) == 0:
            continue
        ah = pd.to_datetime(sub["asof"].to_numpy()[valid])
        spy = panel["SPY"]
        bv = []
        for asof_t in ah:
            pos = panel.index.searchsorted(asof_t)
            arr = spy.iloc[pos:].to_numpy(dtype=float)
            mask = np.isfinite(arr)
            bv.append(arr[mask][-1] / arr[0] - 1.0 if mask.any() else np.nan)
        bv = np.asarray(bv, dtype=float)
        cf = [(pd.Timestamp(t), -1.0) for t in ah]
        cf.append((eval_at, float(np.sum(1 + fv))))
        c = xirr(cf)
        cf_spy = [(pd.Timestamp(t), -1.0) for t in ah]
        cf_spy.append((eval_at, float(np.sum(1 + bv[np.isfinite(bv)]))))
        c_spy = xirr(cf_spy)
        rows.append({
            "year": int(y),
            "n_picks": int(len(fv)),
            "win_rate": float((fv > 0).mean()),
            "beat_spy_rate": float((fv > bv).mean()),
            "median_ret": float(np.nanmedian(fv)),
            "cagr_dca_picks": c,
            "cagr_dca_spy": c_spy,
            "edge": c - c_spy,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Ensemble strategies
# ---------------------------------------------------------------------------
def ensemble_intersection(score_fns: list, k_per_strategy: int = 30,
                          top_k_final: int = 5) -> pd.DataFrame:
    """For each month, score by each function, pick top-k per strategy,
    return the intersection. If intersection too small, fall back to the
    weighted-rank score."""
    feats = load_features_long()
    rows = []
    for asof, sub in feats.groupby(level="asof"):
        scores_per: list[pd.Series] = []
        for fn in score_fns:
            sub2 = sub.copy()
            sub2.index = sub2.index.droplevel("asof")
            s = fn(sub2).dropna()
            s = s[~s.index.isin(BENCH_EXCLUDED)]
            scores_per.append(s)
        if not scores_per:
            continue
        # Compute average rank across strategies (descending = better)
        rank_avg = None
        for s in scores_per:
            r = s.rank(ascending=False)
            r = r / r.max() if r.max() > 0 else r
            rank_avg = r if rank_avg is None else (rank_avg.add(r, fill_value=2.0))
        if rank_avg is None:
            continue
        rank_avg = rank_avg / len(scores_per)
        # Lower rank = better
        top = rank_avg.sort_values().head(top_k_final)
        for tkr, score in top.items():
            rows.append({"asof": asof, "ticker": tkr, "score": -float(score)})
    return pd.DataFrame(rows)


def ensemble_consensus(score_fns: list, top_per_strategy: int = 30,
                       min_votes: int = 3, top_k_final: int = 5) -> pd.DataFrame:
    """Stocks that appear in top-N of at least `min_votes` strategies.

    Score = number of strategies + average rank (lower rank=better)
    """
    feats = load_features_long()
    rows = []
    for asof, sub in feats.groupby(level="asof"):
        sub2 = sub.copy()
        sub2.index = sub2.index.droplevel("asof")
        votes: dict[str, int] = {}
        rank_sum: dict[str, float] = {}
        for fn in score_fns:
            s = fn(sub2).dropna()
            s = s[~s.index.isin(BENCH_EXCLUDED)]
            top_n = s.sort_values(ascending=False).head(top_per_strategy)
            for i, tkr in enumerate(top_n.index):
                votes[tkr] = votes.get(tkr, 0) + 1
                rank_sum[tkr] = rank_sum.get(tkr, 0) + i
        elig = [(t, v, rank_sum[t] / v) for t, v in votes.items() if v >= min_votes]
        # rank by votes desc, then by avg rank asc
        elig.sort(key=lambda x: (-x[1], x[2]))
        for tkr, v, ar in elig[:top_k_final]:
            rows.append({"asof": asof, "ticker": tkr, "score": v - ar / max(top_per_strategy, 1)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
def main() -> None:
    panel = load_panel()
    eval_at = panel.index.max()

    print("\n=== DEEP DIVE: TOP HONEST STRATEGIES ===")
    candidates = {
        "pullback_in_winner_k1_hold": (pullback_in_winner, 1, "ret__hold_forever"),
        "pullback_in_winner_k1_3y": (pullback_in_winner, 1, "ret__fixed_3y"),
        "pullback_in_winner_k3_hold": (pullback_in_winner, 3, "ret__hold_forever"),
        "pullback_in_winner_k5_hold": (pullback_in_winner, 5, "ret__hold_forever"),
        "quality_pullback_k1_hold": (quality_pullback, 1, "ret__hold_forever"),
        "quality_pullback_k3_hold": (quality_pullback, 3, "ret__hold_forever"),
        "dual_momentum_k1_3y": (dual_momentum, 1, "ret__fixed_3y"),
        "dual_momentum_k3_3y": (dual_momentum, 3, "ret__fixed_3y"),
        "explosive_winners_k1_hold": (explosive_winners, 1, "ret__hold_forever"),
    }
    for name, (fn, k, rc) in candidates.items():
        picks = picks_for(fn, top_k=k)
        merged = merge_fwd(picks)
        stats = cagr_dca(merged, rc, eval_at, panel, delist_iters=200)
        print(f"\n{name}")
        for kk, vv in stats.items():
            print(f"  {kk:>30s}: {vv:.4f}" if isinstance(vv, float) else f"  {kk:>30s}: {vv}")
        # Per-year breakdown
        yb = per_year_breakdown(merged, rc, panel, eval_at)
        print("  per-year:")
        print(yb.to_string(index=False))

    # ---- ENSEMBLES ----
    print("\n=== ENSEMBLES ===")
    fns_quality = [pullback_in_winner, quality_pullback, dual_momentum,
                   explosive_winners, winner_only]
    for k in (1, 3, 5):
        picks = ensemble_intersection(fns_quality, top_k_final=k)
        merged = merge_fwd(picks)
        for rc in ("ret__hold_forever", "ret__fixed_3y", "ret__fixed_5y", "ret__tp200"):
            stats = cagr_dca(merged, rc, eval_at, panel, delist_iters=100)
            print(f"  intersection_avg_rank k={k} rule={rc[5:]:14s} "
                  f"n={stats.get('n',0):4d} win={stats.get('win_rate',0):.3f}/{stats.get('win_rate_bias_corr_median',0):.3f}  "
                  f"CAGR={stats.get('cagr_dca',0):.3f} (bias-corr median={stats.get('cagr_dca_bias_corr_median',0):.3f}) "
                  f"vs SPY={stats.get('cagr_spy_dca',0):.3f} edge={stats.get('edge',0):+.3f}")

    print()
    for top_per in (20, 30, 50):
        for min_votes in (2, 3, 4):
            for k in (1, 3, 5):
                picks = ensemble_consensus(fns_quality, top_per_strategy=top_per,
                                           min_votes=min_votes, top_k_final=k)
                if picks.empty:
                    continue
                merged = merge_fwd(picks)
                stats = cagr_dca(merged, "ret__hold_forever", eval_at, panel, delist_iters=100)
                print(f"  consensus top_per={top_per:2d} min_votes={min_votes} k={k} hold_forever: "
                      f"n={stats.get('n',0):4d} win={stats.get('win_rate',0):.3f}/{stats.get('win_rate_bias_corr_median',0):.3f} "
                      f"CAGR={stats.get('cagr_dca',0):.3f} bias-corr={stats.get('cagr_dca_bias_corr_median',0):.3f} "
                      f"edge={stats.get('edge',0):+.3f}")


if __name__ == "__main__":
    main()
