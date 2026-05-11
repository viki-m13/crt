"""Phase 5b: run the deployed v3-winner config on augmented inputs.

Deployed v3-winner = `ml_3plus6|k3_3_3|ew|tight|h6|cap1.0`. This is what
the production v3 strategy actually uses (k=3 across all regimes, hold 6
months, equal-weight, tight regime gate). It is the apples-to-apples
comparator to experiments/monthly_dca/cache/v2/sp500_pit/v3_winner_summary.json
(WF mean CAGR 42.80%, Full CAGR 39.77%).

We monkey-patch the path constants in the shared sweep module so it
reads the augmented sp500_pit_panel, augmented ml_preds, and augmented
monthly_returns. Same simulation code as the original validate.

Inputs (augmented):
  augmented/sp500_pit_panel.parquet
  augmented/ml_preds.parquet
  augmented/monthly_returns_clean.parquet
  augmented/features/*.parquet (for SPY regime features)

Outputs:
  augmented/v3_winner_summary.json
  augmented/v3_winner_walkforward.csv
  augmented/v3_winner_yearly.csv
  augmented/v3_winner_equity.csv
"""
from __future__ import annotations

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
CACHE = ROOT / "experiments" / "monthly_dca" / "cache"
V2 = CACHE / "v2"
PIT = V2 / "sp500_pit"
AUG = PIT / "augmented"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "monthly_dca" / "v2"))

# Import the sweep module
import sp500_pit_extended_sweep as eswp  # noqa: E402

# Monkey-patch paths so it reads augmented inputs.
eswp.PIT = AUG
eswp.V2 = AUG
eswp.FEATURES_DIR = AUG / "features"
eswp.CACHE = AUG

# Also patch the V2 module-level constant used in the load functions
_orig_build_panel = eswp.build_panel_with_score


def _patched_build_panel_with_score(scorer: str) -> pd.DataFrame:
    """Build the scored panel using augmented sp500_pit_panel + augmented preds."""
    panel = pd.read_parquet(AUG / "sp500_pit_panel.parquet")
    if scorer == "ml_avg":
        ml = pd.read_parquet(AUG / "ml_preds.parquet")[
            ["asof", "ticker", "pred", "pred_1m", "pred_3m", "pred_6m"]
        ]
    elif scorer == "ml_1m":
        ml = pd.read_parquet(AUG / "ml_preds.parquet")[["asof", "ticker", "pred_1m"]]
        ml = ml.rename(columns={"pred_1m": "score"})
    elif scorer == "ml_3m":
        ml = pd.read_parquet(AUG / "ml_preds.parquet")[["asof", "ticker", "pred_3m"]]
        ml = ml.rename(columns={"pred_3m": "score"})
    elif scorer == "ml_6m":
        ml = pd.read_parquet(AUG / "ml_preds.parquet")[["asof", "ticker", "pred_6m"]]
        ml = ml.rename(columns={"pred_6m": "score"})
    elif scorer == "ml_3plus6":
        ml = pd.read_parquet(AUG / "ml_preds.parquet")[
            ["asof", "ticker", "pred_3m", "pred_6m"]
        ]
        ml["score"] = (ml["pred_3m"] + ml["pred_6m"]) / 2
    elif scorer == "ml_filter":
        ml = pd.read_parquet(AUG / "ml_preds.parquet")[["asof", "ticker", "pred"]]
        ml = ml.rename(columns={"pred": "score"})
    else:
        ml = pd.read_parquet(AUG / "ml_preds.parquet")[["asof", "ticker", "pred"]]
        ml = ml.rename(columns={"pred": "score"})
    panel["asof"] = pd.to_datetime(panel["asof"])
    ml["asof"] = pd.to_datetime(ml["asof"])
    panel = panel.merge(ml, on=["asof", "ticker"], how="left")
    return panel


eswp.build_panel_with_score = _patched_build_panel_with_score


# Patch load_spy_features to read augmented features
def _patched_load_spy_features() -> pd.DataFrame:
    rows = []
    for f in sorted((AUG / "features").glob("*.parquet")):
        d = pd.Timestamp(f.stem)
        df = pd.read_parquet(f)
        if "SPY" not in df.index:
            continue
        spy = df.loc["SPY"]
        rows.append({
            "asof": d,
            "spy_dsma200": float(spy.get("d_sma200", 0.0)),
            "spy_rsi14": float(spy.get("rsi_14", 50.0)),
            "spy_mom_12_1": float(spy.get("mom_12_1", 0.0)),
            "spy_mom_6_1": float(spy.get("mom_6_1", 0.0)),
            "spy_ret_21d": float(spy.get("ret_21d", 0.0)),
            "spy_below_200_streak": float(spy.get("max_below_200_streak", 0.0)),
        })
    return pd.DataFrame(rows).set_index("asof")


eswp.load_spy_features = _patched_load_spy_features


from sp500_pit_extended_sweep import (  # noqa: E402
    Variant, simulate_variant, evaluate,
)
from sp500_pit_v3_validate import (  # noqa: E402
    parse_variant_name, per_split_eval,
)


def main():
    variant_name = "ml_3plus6|k3_3_3|ew|tight|h6|cap1.0"
    print(f"=== Validating: {variant_name} on AUGMENTED inputs ===")
    v = parse_variant_name(variant_name)

    monthly_returns = pd.read_parquet(AUG / "monthly_returns_clean.parquet")
    spy_features = _patched_load_spy_features()
    print(f"[1] monthly_returns: {monthly_returns.shape}, SPY features: {spy_features.shape}")

    panel = _patched_build_panel_with_score(v.scorer)
    print(f"[2] panel (with score): {panel.shape}")

    eq = simulate_variant(panel, monthly_returns, spy_features, v)
    print(f"[3] simulation: {len(eq)} months, final ${eq['equity'].iloc[-1]:.2f}")

    full_dates = pd.DatetimeIndex(eq["date"])
    next_month = full_dates + pd.offsets.MonthEnd(1)
    spy_aligned = pd.DataFrame({
        "date": full_dates,
        "spy_ret_m": [
            float(monthly_returns["SPY"].loc[nxt]) if nxt in monthly_returns["SPY"].index else 0.0
            for nxt in next_month
        ],
    })

    metrics = evaluate(eq, spy_aligned, variant_name)
    splits = per_split_eval(eq, spy_aligned)

    print(f"\n[headline]")
    for k, v_ in metrics.items():
        print(f"  {k}: {v_}")

    print("\n[per-split walk-forward]")
    print(splits.round(3).to_string(index=False))

    eq.to_csv(AUG / "v3_winner_equity.csv", index=False)
    splits.to_csv(AUG / "v3_winner_walkforward.csv", index=False)

    summary = {
        "variant_name": variant_name,
        "n_months": int(len(eq)),
        "final_equity": float(eq["equity"].iloc[-1]),
        "cagr_full": metrics["cagr_full"],
        "spy_cagr_full": metrics["spy_cagr_full"],
        "edge_full_pp": metrics["edge_full_pp"],
        "sharpe": metrics["sharpe"],
        "max_dd": metrics["max_dd"],
        "n_cash_months": metrics["n_cash"],
        "wf_mean_cagr": metrics["wf_mean_cagr"],
        "wf_median_cagr": metrics["wf_median_cagr"],
        "wf_min_cagr": metrics["wf_min_cagr"],
        "wf_max_cagr": metrics["wf_max_cagr"],
        "wf_mean_edge_pp": metrics["wf_mean_edge_pp"],
        "wf_n_positive": metrics["wf_n_pos"],
        "wf_n_beats_spy": metrics["wf_n_beats"],
    }
    (AUG / "v3_winner_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[saved] {AUG / 'v3_winner_summary.json'}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
