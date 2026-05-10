"""Test conviction-stack and banger blends."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from runner import StratSpec, benchmark


def specs():
    out = []
    # Banger only
    for k in [2, 3, 5]:
        for w in ["ew", "invvol"]:
            for hold in [6, 12]:
                out.append(StratSpec(
                    f"banger_k{k}_{w}_h{hold}",
                    "banger",
                    k_normal=k, k_recovery=k, k_bull=k,
                    weighting=w, hold_months=hold,
                ))
    # ML + banger
    for w_b in [0.3, 0.5, 0.7]:
        for k in [2, 3]:
            for hold in [6, 12]:
                out.append(StratSpec(
                    f"ml_plus_banger_w{w_b}_k{k}_h{hold}",
                    "ml_plus_banger",
                    weights={"banger": w_b},
                    k_normal=k, k_recovery=k, k_bull=k,
                    weighting="invvol", hold_months=hold,
                ))
    # Conviction stack (3-way)
    for k in [2, 3, 5]:
        for w in ["ew", "invvol"]:
            for hold in [6, 12]:
                out.append(StratSpec(
                    f"conv_stack_k{k}_{w}_h{hold}",
                    "conviction_stack",
                    k_normal=k, k_recovery=k, k_bull=k,
                    weighting=w, hold_months=hold,
                ))
    return out


if __name__ == "__main__":
    df = benchmark(specs(), save_prefix="stacked_sweep")
    print("\nTop 12 by WF mean:")
    print(df.sort_values("wf_mean_cagr", ascending=False).head(12)[
        ["name", "wf_mean_cagr", "cagr_full", "sharpe", "max_dd", "wf_n_beats_spy"]].to_string())
