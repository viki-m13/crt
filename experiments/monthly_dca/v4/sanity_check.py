"""Sanity check: v4 engine should reproduce v3 baseline when no extras enabled."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from v4_engine import (
    V4Variant, simulate_v4, evaluate_v4, load_spy_features, build_panel_with_score,
    build_spy_aligned, PIT,
)
import pandas as pd

V2 = Path('experiments/monthly_dca/cache/v2')

print("Loading inputs...")
monthly_returns = pd.read_parquet(V2 / 'monthly_returns_clean.parquet')
spy = load_spy_features()

print("Building panel with ml_3plus6 score...")
panel = build_panel_with_score("ml_3plus6")
spy_aligned = build_spy_aligned(PIT / "sp500_pit_panel.parquet", monthly_returns)

# v3 baseline
v3 = V4Variant(
    name="v3_baseline",
    scorer="ml_3plus6",
    k_normal=3, k_recovery=3, k_bull=3,
    weighting="ew", regime_gate="tight",
    hold_months=6, cap_per_pick=1.0,
    score_threshold=-1e9, stop_loss=-1.0, take_profit=1e9,
    cost_bps=10.0,
)
print("Simulating v3...")
eq = simulate_v4(panel, monthly_returns, spy, v3)
res = evaluate_v4(eq, spy_aligned, v3.name)
print(f"V3 reproduction: cagr_full={res['cagr_full']:.4f} wf_mean={res['wf_mean_cagr']:.4f}")
print(f"  Expected: cagr_full=0.3977 wf_mean=0.4280")
