"""Save the winning strategy's full picks (with feature snapshot per pick) to CSV/JSON."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from experiments.monthly_dca.deepdive import (
    cagr_dca,
    merge_fwd,
    per_year_breakdown,
    picks_for,
)
from experiments.monthly_dca.fast_score import load_features_long, load_panel
from experiments.monthly_dca.strategies_fast import (
    pullback_in_winner,
    quality_pullback,
    explosive_winners,
)


OUT = Path("experiments/monthly_dca/cache")
OUT.mkdir(parents=True, exist_ok=True)


def save_strategy(name: str, fn, top_k: int, panel) -> None:
    eval_at = panel.index.max()
    # Use the FULL extended history window (2002-2024 picks held to today)
    picks = picks_for(fn, top_k=top_k, start="2002-01-31", end="2024-12-31")
    feats = load_features_long().reset_index()
    merged = picks.merge(feats, on=["asof", "ticker"], how="left")

    # Add forward returns for context
    merged_fwd = merge_fwd(picks)
    cols_to_keep = ["asof", "ticker", "score", "price", "pullback_1y", "trend_health_5y",
                    "recovery_rate", "mom_12_1", "mom_3y", "rsi_14", "d_sma200",
                    "ret__hold_forever", "ret__fixed_1y", "ret__fixed_3y", "ret__fixed_5y",
                    "days__hold_forever"]
    cols_present = [c for c in cols_to_keep if c in merged.columns or c in merged_fwd.columns]
    out_df = merged.merge(merged_fwd[["asof", "ticker"] + [c for c in cols_present if c in merged_fwd.columns and c not in merged.columns]],
                           on=["asof", "ticker"], how="left")
    out_df.to_csv(OUT / f"picks_full_{name}_k{top_k}.csv", index=False)
    print(f"Saved {OUT / f'picks_full_{name}_k{top_k}.csv'} ({len(out_df)} rows)")

    # Year-by-year + summary stats
    stats = cagr_dca(merge_fwd(picks), "ret__hold_forever", eval_at, panel, delist_iters=200)
    yb = per_year_breakdown(merge_fwd(picks), "ret__hold_forever", panel, eval_at)
    yb.to_csv(OUT / f"yb_{name}_k{top_k}.csv", index=False)

    # Summary JSON for posterity
    sd = {
        "strategy": name,
        "top_k": top_k,
        "exit_rule": "hold_forever",
        "eval_at": str(eval_at.date()),
        "stats": {k: (None if pd.isna(v) else float(v)) for k, v in stats.items()},
        "year_by_year": yb.to_dict(orient="records"),
        "n_picks": len(picks),
        "first_pick_date": str(pd.to_datetime(picks["asof"]).min().date()) if not picks.empty else None,
        "last_pick_date": str(pd.to_datetime(picks["asof"]).max().date()) if not picks.empty else None,
        "unique_tickers": int(picks["ticker"].nunique()) if not picks.empty else 0,
        "ticker_frequency": picks["ticker"].value_counts().head(20).to_dict() if not picks.empty else {},
    }
    with open(OUT / f"summary_{name}_k{top_k}.json", "w") as f:
        json.dump(sd, f, indent=2)
    print(f"Saved {OUT / f'summary_{name}_k{top_k}.json'}")


def main() -> None:
    panel = load_panel()
    save_strategy("pullback_in_winner", pullback_in_winner, 1, panel)
    save_strategy("pullback_in_winner", pullback_in_winner, 3, panel)
    save_strategy("pullback_in_winner", pullback_in_winner, 5, panel)
    save_strategy("quality_pullback", quality_pullback, 1, panel)
    save_strategy("quality_pullback", quality_pullback, 5, panel)
    save_strategy("explosive_winners", explosive_winners, 1, panel)
    save_strategy("explosive_winners", explosive_winners, 5, panel)
    # NEW recommended strategy
    from experiments.monthly_dca.strategies_fast import blended_pullback_momentum
    save_strategy("blended_pullback_momentum", blended_pullback_momentum, 5, panel)
    save_strategy("blended_pullback_momentum", blended_pullback_momentum, 1, panel)
    save_strategy("blended_pullback_momentum", blended_pullback_momentum, 10, panel)


if __name__ == "__main__":
    main()
