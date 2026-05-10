"""Run bias sensitivity only on the winners."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from validate_winner import bias_sensitivity, WINNERS, RESULTS
import pandas as pd

bias_rows = []
for name in ["v8_moderate", "v8_safe", "v8_max_cagr"]:
    cfg = WINNERS[name]
    print(f"\n=== {name} bias sensitivity ===", flush=True)
    bdf = bias_sensitivity(name, cfg, "ml_3plus6", alphas=(0, 0.02, 0.04), seeds=(1, 2, 3))
    bias_rows.append(bdf)
    print(bdf.to_string())
pd.concat(bias_rows, ignore_index=True).to_csv(RESULTS / "winners_bias.csv", index=False)
print(f"Saved bias to {RESULTS}/winners_bias.csv")
