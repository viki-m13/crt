"""Basic parameter sweep around the strong ML baseline."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from runner import StratSpec, benchmark


def specs():
    out = []
    # Smaller k → higher CAGR, higher variance
    for k in [1, 2, 3, 4, 5]:
        for w in ["ew", "invvol"]:
            for hold in [3, 6, 12]:
                out.append(StratSpec(
                    f"ml_k{k}_{w}_h{hold}",
                    "ml_3plus6",
                    k_normal=k, k_recovery=k, k_bull=k,
                    weighting=w,
                    hold_months=hold,
                ))
    # Stronger filtering on ML picks
    for k in [3, 5]:
        for hold in [3, 6]:
            out.append(StratSpec(
                f"ml_filtered_pull50_mom30_k{k}_h{hold}",
                "ml_3plus6",
                k_normal=k, k_recovery=k, k_bull=k,
                hold_months=hold,
                pullback_filter=0.5,
                min_pick_mom=0.3,
            ))
    return out


if __name__ == "__main__":
    benchmark(specs(), save_prefix="sweep_basic")
