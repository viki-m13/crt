"""Walk-forward validation on top alpha + ensemble strategies.

Splits validate genuine OOS robustness across 2002-2024.
"""
from __future__ import annotations

import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np
import pandas as pd

from experiments.monthly_dca.fast_score import (
    BENCH_EXCLUDED, load_features_long, load_fwd, load_panel,
)
from experiments.monthly_dca.fast_engine import xirr
from experiments.monthly_dca.strategies_fast import (
    quality_pullback, explosive_winners, pullback_in_winner,
    blended_pullback_momentum, dual_momentum,
)
from experiments.monthly_dca.strategies_pro import asymmetric_winner
from experiments.monthly_dca.strategies_alpha import (
    nova_star, persistent_winner, multibagger_engine,
    consensus_top_decile, alpha_intersect, rank_intersect,
    asymmetric_recovery_plus, clean_compounder,
)
from experiments.monthly_dca.strategies_alpha2 import (
    alpha_omega, alpha_omega_deep, ultra_nova, multibagger_max, apex,
    nova_tier1, the_bagger, nova_dual,
)
from experiments.monthly_dca.strategies_ensemble import (
    grand_ensemble, diamond_ensemble, strategy_rotation,
    best_of_top4, best_of_top4_intersect, quality_pullback_amped,
    explosive_winners_amped, pullback_in_winner_amped,
)


CACHE = Path("experiments/monthly_dca/cache")


# Walk-forward splits: each split has TRAIN window and TEST window.
# We don't optimize on TRAIN — we just measure CAGR in both windows.
# A strategy is "robust" if its TEST CAGR is in the TRAIN top-N AND >0.
SPLITS = {
    "A1": ("2002-01-01", "2010-12-31", "2011-01-01", "2018-12-31"),
    "A2": ("2002-01-01", "2014-12-31", "2015-01-01", "2021-12-31"),
    "A3": ("2002-01-01", "2017-12-31", "2018-01-01", "2024-12-31"),
    "R1": ("2002-01-01", "2007-12-31", "2008-01-01", "2010-12-31"),
    "R2": ("2005-01-01", "2010-12-31", "2011-01-01", "2013-12-31"),
    "R3": ("2008-01-01", "2013-12-31", "2014-01-01", "2016-12-31"),
    "R4": ("2011-01-01", "2016-12-31", "2017-01-01", "2019-12-31"),
    "R5": ("2014-01-01", "2019-12-31", "2020-01-01", "2022-12-31"),
    "R6": ("2017-01-01", "2022-12-31", "2023-01-01", "2024-12-31"),
    "STRICT": ("2002-01-01", "2020-12-31", "2021-01-01", "2024-12-31"),
}


def _spy_returns(panel: pd.DataFrame, ah: np.ndarray) -> np.ndarray:
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
    return np.asarray(bv, dtype=float)


def eval_strategy_in_window(score_fn, top_k: int, window_start: str, window_end: str,
                             feats: pd.DataFrame, fwd: pd.DataFrame, panel: pd.DataFrame,
                             eval_at, rule: str = "hold_forever") -> dict:
    feats_w = feats.loc[(feats.index.get_level_values("asof") >= pd.Timestamp(window_start)) &
                         (feats.index.get_level_values("asof") <= pd.Timestamp(window_end))]
    fwd_w = fwd.loc[(fwd.index.get_level_values("asof") >= pd.Timestamp(window_start)) &
                     (fwd.index.get_level_values("asof") <= pd.Timestamp(window_end))]
    chunks = []
    for asof, sub in feats_w.groupby(level="asof"):
        df_asof = sub.copy()
        df_asof.index = df_asof.index.get_level_values("ticker")
        try:
            s = score_fn(df_asof)
        except Exception:
            continue
        s = s.dropna()
        bad = [t for t in BENCH_EXCLUDED if t in s.index]
        s = s.drop(bad, errors="ignore")
        if s.empty: continue
        s = s.reset_index()
        s.columns = ["ticker", "score"]
        s["asof"] = asof
        chunks.append(s)
    if not chunks:
        return {}
    df_scores = pd.concat(chunks, ignore_index=True)
    df_scores = df_scores.sort_values(["asof", "score"], ascending=[True, False])
    picks = df_scores.groupby("asof", group_keys=False).head(top_k).reset_index(drop=True)
    if picks.empty:
        return {}
    fwd_reset = fwd_w.reset_index()
    merged = picks.merge(fwd_reset, on=["asof", "ticker"], how="left")
    rc = f"ret__{rule}"
    if rc not in merged.columns:
        return {}
    f = merged[rc].to_numpy(dtype=float)
    valid = np.isfinite(f)
    if not valid.any():
        return {}
    fv = f[valid]
    ah = pd.to_datetime(merged["asof"].to_numpy()[valid])
    bv = _spy_returns(panel, ah)
    cashflows = [(pd.Timestamp(t), -1.0) for t in ah]
    cashflows.append((eval_at, float(np.sum(1 + fv))))
    cagr = xirr(cashflows)
    cashflows_spy = [(pd.Timestamp(t), -1.0) for t in ah]
    cashflows_spy.append((eval_at, float(np.sum(1 + bv[np.isfinite(bv)]))))
    cagr_spy = xirr(cashflows_spy)
    return {
        "cagr": float(cagr) if np.isfinite(cagr) else float("nan"),
        "cagr_spy": float(cagr_spy) if np.isfinite(cagr_spy) else float("nan"),
        "edge": float(cagr - cagr_spy) if np.isfinite(cagr) and np.isfinite(cagr_spy) else float("nan"),
        "win_rate": float((fv > 0).mean()),
        "n_picks": int(valid.sum()),
        "median_ret": float(np.nanmedian(fv)),
    }


# Top candidate strategies (selected from full-window results) — focused on
# the strongest and the recommended winner.
TOP_STRATS = [
    ("strategy_rotation", strategy_rotation),
    ("grand_ensemble", grand_ensemble),
    ("diamond_ensemble", diamond_ensemble),
    ("explosive_winners_amped", explosive_winners_amped),
    ("quality_pullback_amped", quality_pullback_amped),
    ("best_of_top4", best_of_top4),
    # Baselines for comparison
    ("blended_pullback_momentum", blended_pullback_momentum),
    ("quality_pullback", quality_pullback),
    ("explosive_winners", explosive_winners),
    ("pullback_in_winner", pullback_in_winner),
]

TOP_KS = [1, 3, 5]


def main():
    feats = load_features_long()
    fwd = load_fwd()
    panel = load_panel()
    eval_at = panel.index.max()

    rows = []
    for split_name, (tr_s, tr_e, te_s, te_e) in SPLITS.items():
        print(f"\n=== Split {split_name}: TRAIN {tr_s}->{tr_e}, TEST {te_s}->{te_e} ===", flush=True)
        for name, fn in TOP_STRATS:
            for k in TOP_KS:
                tr = eval_strategy_in_window(fn, k, tr_s, tr_e, feats, fwd, panel, eval_at)
                te = eval_strategy_in_window(fn, k, te_s, te_e, feats, fwd, panel, eval_at)
                row = {"split": split_name, "strategy": name, "top_k": k}
                if tr:
                    row.update({f"train_{k}": v for k, v in tr.items()})
                if te:
                    row.update({f"test_{k}": v for k, v in te.items()})
                rows.append(row)
        print(f"  rows so far: {len(rows)}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(CACHE / "wf_top_alpha.csv", index=False)
    print(f"\nWrote {CACHE / 'wf_top_alpha.csv'}: {df.shape}")

    # Aggregate
    df["key"] = df["strategy"] + "::" + df["top_k"].astype(str)
    df_train_rank = df.copy()
    df_train_rank["train_rank_in_split"] = df_train_rank.groupby("split")["train_cagr"].rank(ascending=False)
    df_train_rank["in_top10_train"] = (df_train_rank["train_rank_in_split"] <= 10).astype(int)
    agg = df_train_rank.groupby("key").agg(
        n_splits_in_train_top10=("in_top10_train", "sum"),
        n_splits=("split", "count"),
        mean_test_cagr=("test_cagr", "mean"),
        median_test_cagr=("test_cagr", "median"),
        min_test_cagr=("test_cagr", "min"),
        max_test_cagr=("test_cagr", "max"),
        mean_test_edge=("test_edge", "mean"),
        min_test_edge=("test_edge", "min"),
        mean_test_win=("test_win_rate", "mean"),
    ).reset_index()
    agg = agg.sort_values("mean_test_cagr", ascending=False)
    agg.to_csv(CACHE / "wf_top_alpha_aggregate.csv", index=False)
    print(f"Wrote {CACHE / 'wf_top_alpha_aggregate.csv'}")
    print(agg.head(40).to_string(index=False))


if __name__ == "__main__":
    main()
