"""Test the LGB model — head-to-head and blended."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from runner import StratSpec, benchmark


def specs():
    out = []
    # Pure LGB at various K, hold, weighting
    for k in [1, 2, 3, 4, 5]:
        for w in ["ew", "invvol"]:
            for hold in [3, 6, 12]:
                out.append(StratSpec(
                    f"lgb_k{k}_{w}_h{hold}",
                    "lgb",
                    k_normal=k, k_recovery=k, k_bull=k,
                    weighting=w,
                    hold_months=hold,
                ))
    # LGB + ML blends
    for w_lgb in [0.3, 0.5, 0.7]:
        for k in [2, 3]:
            for hold in [6, 12]:
                out.append(StratSpec(
                    f"ml_plus_lgb_w{w_lgb}_k{k}_h{hold}",
                    "ml_plus_lgb",
                    weights={"lgb": w_lgb},
                    k_normal=k, k_recovery=k, k_bull=k,
                    weighting="invvol",
                    hold_months=hold,
                ))
    return out


if __name__ == "__main__":
    df = benchmark(specs(), save_prefix="lgb_sweep")
    print("\nTop 10 by WF mean:")
    print(df.sort_values("wf_mean_cagr", ascending=False).head(10)[
        ["name", "wf_mean_cagr", "cagr_full", "sharpe", "max_dd", "wf_n_beats_spy"]].to_string())
