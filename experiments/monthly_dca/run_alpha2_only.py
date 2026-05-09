"""Quick eval of just the late-added alpha2 strategies (multibagger_max, apex, nova_tier1)."""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

warnings.filterwarnings("ignore", category=RuntimeWarning)
import pandas as pd

from experiments.monthly_dca.fast_score import load_features_long, load_fwd, load_panel
from experiments.monthly_dca.run_alpha import evaluate_strategy_fast
from experiments.monthly_dca.strategies_alpha2 import multibagger_max, apex, nova_tier1


def main():
    print("Loading data...")
    feats = load_features_long()
    fwd = load_fwd()
    panel = load_panel()

    rules = ["hold_forever", "fixed_3y", "fixed_5y"]
    top_ks = [1, 2, 3, 5, 10]

    rows = []
    for name, fn in [
        ("multibagger_max", multibagger_max),
        ("apex", apex),
        ("nova_tier1", nova_tier1),
    ]:
        for k in top_ks:
            print(f"  {name} k={k}", flush=True)
            r = evaluate_strategy_fast(fn, k, name, rules=rules,
                                        feats_long=feats, fwd=fwd, panel=panel)
            rows.extend(r)

    df = pd.DataFrame(rows)
    out = Path("experiments/monthly_dca/cache/sweep_alpha2_only.csv")
    df.to_csv(out, index=False)
    print(f"Wrote {out}")
    print(df.sort_values("cagr_dca_portfolio", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
