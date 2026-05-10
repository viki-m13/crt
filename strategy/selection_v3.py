"""V3: CRT as a tie-breaker on top of the baseline strategy_rotation.

Hypothesis: the existing regime classifier is well-tuned; the legacy
score formulas are well-tuned; but selecting the TOP-5 among the legacy
score's top-15 candidates leaves room for the novel CRT signal to add
value.

Variants:
  v3_topn_crt:  legacy score's top-N (N=10/15/20), narrow to top-5 by
                CRT score.
  v3_topn_comp: same, narrow by full composite.
  v3_blend_mul: legacy_score * (1 + alpha * z(crt_6m)).
  v3_blend_add: rank(legacy) + 0.5 * rank(composite).
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from experiments.monthly_dca.strategies_ensemble import strategy_rotation
from strategy.selection import zscore
from strategy.selection_v2 import composite_score


def v3_topn_crt(df: pd.DataFrame, top_n: int = 15) -> pd.Series:
    """Pick top-N by legacy strategy_rotation, narrow to final by crt_6m."""
    legacy = strategy_rotation(df)
    if legacy.dropna().empty:
        return pd.Series(np.nan, index=df.index)
    # Drop benchmarks
    excl = ["SPY", "QQQ", "IWM", "VTI", "RSP", "DIA", "BTC-USD", "ETH-USD"]
    legacy = legacy.drop(labels=[t for t in excl if t in legacy.index], errors="ignore")
    top_idx = legacy.dropna().sort_values(ascending=False).head(top_n).index
    out = pd.Series(np.nan, index=df.index)
    if "crt_6m" in df.columns:
        out.loc[top_idx] = df.loc[top_idx, "crt_6m"]
    else:
        out.loc[top_idx] = legacy.loc[top_idx]  # fallback
    return out


def v3_topn_composite(df: pd.DataFrame, top_n: int = 15) -> pd.Series:
    legacy = strategy_rotation(df)
    if legacy.dropna().empty:
        return pd.Series(np.nan, index=df.index)
    excl = ["SPY", "QQQ", "IWM", "VTI", "RSP", "DIA", "BTC-USD", "ETH-USD"]
    legacy = legacy.drop(labels=[t for t in excl if t in legacy.index], errors="ignore")
    top_idx = legacy.dropna().sort_values(ascending=False).head(top_n).index
    out = pd.Series(np.nan, index=df.index)
    comp = composite_score(df)
    out.loc[top_idx] = comp.loc[top_idx]
    return out


def v3_blend_mul(df: pd.DataFrame, alpha: float = 0.3) -> pd.Series:
    """Legacy score multiplied by a CRT booster."""
    legacy = strategy_rotation(df)
    if legacy.dropna().empty or "crt_6m" not in df.columns:
        return legacy
    excl = ["SPY", "QQQ", "IWM", "VTI", "RSP", "DIA", "BTC-USD", "ETH-USD"]
    legacy = legacy.drop(labels=[t for t in excl if t in legacy.index], errors="ignore")
    z_crt = zscore(df["crt_6m"]).reindex(legacy.index).fillna(0)
    boost = (1.0 + alpha * z_crt).clip(lower=0.0)
    return legacy * boost


def v3_blend_add(df: pd.DataFrame, beta: float = 0.5) -> pd.Series:
    """Rank-blended: rank(legacy) + beta * rank(composite)."""
    legacy = strategy_rotation(df)
    if legacy.dropna().empty:
        return pd.Series(np.nan, index=df.index)
    excl = ["SPY", "QQQ", "IWM", "VTI", "RSP", "DIA", "BTC-USD", "ETH-USD"]
    legacy = legacy.drop(labels=[t for t in excl if t in legacy.index], errors="ignore")
    legacy_rank = legacy.rank(pct=True)
    comp = composite_score(df)
    comp_rank = comp.rank(pct=True)
    out = pd.Series(np.nan, index=df.index)
    out.loc[legacy.index] = legacy_rank.loc[legacy.index] + beta * comp_rank.reindex(legacy.index).fillna(0.5)
    return out


def make_v3(name: str, top_k: int = 5, **kw):
    from experiments.monthly_dca.compound_engine import Strategy
    fn_map = {
        "topn_crt_10":  lambda df: v3_topn_crt(df, top_n=10),
        "topn_crt_15":  lambda df: v3_topn_crt(df, top_n=15),
        "topn_crt_20":  lambda df: v3_topn_crt(df, top_n=20),
        "topn_crt_25":  lambda df: v3_topn_crt(df, top_n=25),
        "topn_comp_10": lambda df: v3_topn_composite(df, top_n=10),
        "topn_comp_15": lambda df: v3_topn_composite(df, top_n=15),
        "topn_comp_20": lambda df: v3_topn_composite(df, top_n=20),
        "blend_mul_03": lambda df: v3_blend_mul(df, alpha=0.3),
        "blend_mul_05": lambda df: v3_blend_mul(df, alpha=0.5),
        "blend_add_05": lambda df: v3_blend_add(df, beta=0.5),
        "blend_add_10": lambda df: v3_blend_add(df, beta=1.0),
    }
    fn = fn_map[name]
    return Strategy(name=f"v3_{name}", score_fn=fn, top_k=top_k,
                    description="FHtzX v3 hybrid")


if __name__ == "__main__":
    from experiments.monthly_dca.compound_engine import ExitSpec, run_compound, benchmark_spy_dca
    from experiments.monthly_dca.fast_engine import load_panel

    panel = load_panel()
    rule = ExitSpec("monthly_rebalance", monthly_rebalance=True)

    rows = []
    for name in ["topn_crt_10", "topn_crt_15", "topn_crt_20", "topn_crt_25",
                  "topn_comp_10", "topn_comp_15", "topn_comp_20",
                  "blend_mul_03", "blend_mul_05", "blend_add_05", "blend_add_10"]:
        for k in [5]:
            strat = make_v3(name, top_k=k)
            try:
                res = run_compound(panel, strat, rule,
                                     start="2002-01-31", end="2024-12-31", cost_bps=5.0)
                spy = benchmark_spy_dca(panel, "2002-01-31", "2024-12-31")
                edge = res.cagr_money_weighted - spy["cagr_xirr"]
                rows.append({"name": name, "k": k, "cagr": res.cagr_money_weighted,
                             "edge": edge, "trades": res.n_trades})
                print(f"  v3_{name} k={k}: CAGR={res.cagr_money_weighted:.2%}, "
                       f"edge={edge:.2%}, trades={res.n_trades}")
            except Exception as e:
                print(f"  v3_{name} k={k}: ERROR {e}")
    pd.DataFrame(rows).to_csv("backtests/v3_sweep.csv", index=False)
