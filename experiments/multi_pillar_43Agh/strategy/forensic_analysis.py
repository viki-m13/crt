"""Phase 1 — Forensic Analysis.

For each forensic feature, compute discriminating power:
- KS statistic between event group and control group
- AUROC of single-feature predictor
- Mean / median / std for each group

Also produces archetype centroids for Pillar 4 (winner archetype) and
the failure-feature ranking for Pillar 1.

Outputs:
  data/discriminating_features_winners.csv
  data/discriminating_features_failures.csv
  data/winner_archetype_centroid.parquet
  data/failure_archetype_centroid.parquet
  research/forensics/discriminating_features.md
  research/forensics/archetypes.md

Run from repo root.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "experiments" / "multi_pillar_43Agh" / "data"
RESEARCH = ROOT / "experiments" / "multi_pillar_43Agh" / "research" / "forensics"
RESEARCH.mkdir(parents=True, exist_ok=True)


FEATURE_COLS = [
    # Trend/momentum
    "mom_12_1", "mom_6_1", "mom_3", "mom_3y", "mom_2y", "mom_5y", "idio_mom_12_1",
    "mom_accel", "mom_consistency_12m", "mom_per_unit_vol_12", "acceleration_2y",
    # Pullback / position
    "pullback_1y", "pullback_3y", "pullback_5y", "pullback_all", "range_pos_1y",
    "dd_from_52wh", "dist_from_low_1y", "below_52wh", "new_52wh", "near_52wh_60d",
    "drawdown_age_days", "max_dd_5y", "min_dd_60d",
    # Trend health
    "trend_health_5y", "excess_5y_logret", "d_sma200", "d_sma50",
    "sma50_above_200", "frac_above_50dma_1y", "max_below_200_streak",
    "trend_r2_12m", "trend_slope_252", "fip_score",
    # Vol / risk
    "vol_1y", "vol_12m", "vol_3m", "vol_6m", "sharpe_1y", "sharpe_12m", "sharpe_5y",
    "vol_contraction", "vol_expansion_24m", "tail_ratio_24m", "best_month_24m",
    "worst_month_24m",
    # Quality
    "recovery_rate", "quality_score_5y", "multibagger_ratio_24m",
    # Relative strength vs SPY
    "rs_3m_spy", "rs_6m_spy", "rs_12m_spy", "rs_3m_zscore",
    # Microstructure / consolidation
    "tight_consolidation_60", "breakout_strength_60", "bb_width_pct",
    "bb_width_contraction", "rsi_14", "rsi_zone_score",
    # Price level
    "log_price",
    # Earnings drift proxy (non-fundamental)
    "earnings_drift_proxy",
]


def discriminate(events: pd.DataFrame, controls: pd.DataFrame,
                 feature_cols: list[str]) -> pd.DataFrame:
    rows = []
    for c in feature_cols:
        if c not in events.columns or c not in controls.columns:
            continue
        e = events[c].dropna().values
        ctl = controls[c].dropna().values
        if len(e) < 30 or len(ctl) < 30:
            continue
        try:
            ks_stat, ks_p = stats.ks_2samp(e, ctl)
        except Exception:
            ks_stat, ks_p = np.nan, np.nan
        # AUROC
        y = np.concatenate([np.ones(len(e)), np.zeros(len(ctl))])
        x = np.concatenate([e, ctl])
        order = np.argsort(x)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(1, len(x) + 1)
        n_pos = int(y.sum())
        n_neg = int((1 - y).sum())
        sum_pos_ranks = float(ranks[y == 1].sum())
        # Wilcoxon-Mann-Whitney equivalence
        u = sum_pos_ranks - n_pos * (n_pos + 1) / 2
        auroc = u / (n_pos * n_neg) if (n_pos * n_neg) > 0 else 0.5
        rows.append({
            "feature": c,
            "n_event": len(e),
            "n_ctrl": len(ctl),
            "event_mean": float(np.mean(e)),
            "ctrl_mean": float(np.mean(ctl)),
            "event_med": float(np.median(e)),
            "ctrl_med": float(np.median(ctl)),
            "event_std": float(np.std(e)),
            "ctrl_std": float(np.std(ctl)),
            "ks_stat": float(ks_stat),
            "ks_p": float(ks_p),
            "auroc": float(auroc),
            "auroc_signed": float(abs(auroc - 0.5)) + 0.5,  # absolute discrimination
        })
    df = pd.DataFrame(rows).sort_values("ks_stat", ascending=False)
    return df


def archetype_centroid(events: pd.DataFrame, feature_cols: list[str]) -> pd.Series:
    """Median per feature across all events (more robust than mean)."""
    cols = [c for c in feature_cols if c in events.columns]
    return events[cols].median()


def main():
    print("[load] event + control snapshots ...")
    win = pd.read_parquet(DATA / "winner_features.parquet")
    win_ctrl = pd.read_parquet(DATA / "winner_controls.parquet")
    fail = pd.read_parquet(DATA / "failure_features.parquet")
    fail_ctrl = pd.read_parquet(DATA / "failure_controls.parquet")
    print(f"  winners: {len(win)} events / {len(win_ctrl)} controls")
    print(f"  failures: {len(fail)} events / {len(fail_ctrl)} controls")

    print("[discriminate] winners vs controls ...")
    win_disc = discriminate(win, win_ctrl, FEATURE_COLS)
    win_disc.to_csv(DATA / "discriminating_features_winners.csv", index=False)
    print("  top-10 by KS:")
    print(win_disc.head(10)[["feature", "event_med", "ctrl_med", "ks_stat", "auroc"]].to_string(index=False))

    print("\n[discriminate] failures vs controls ...")
    fail_disc = discriminate(fail, fail_ctrl, FEATURE_COLS)
    fail_disc.to_csv(DATA / "discriminating_features_failures.csv", index=False)
    print("  top-10 by KS:")
    print(fail_disc.head(10)[["feature", "event_med", "ctrl_med", "ks_stat", "auroc"]].to_string(index=False))

    print("\n[centroid] winner / failure archetype centroids ...")
    win_cent = archetype_centroid(win, FEATURE_COLS)
    fail_cent = archetype_centroid(fail, FEATURE_COLS)
    win_cent.to_frame("winner_centroid").to_parquet(DATA / "winner_archetype_centroid.parquet")
    fail_cent.to_frame("failure_centroid").to_parquet(DATA / "failure_archetype_centroid.parquet")
    print("  saved.")

    # Markdown reports
    md_disc = ["# Discriminating features\n",
               "\n## Winners vs controls (top-25 by KS)\n",
               win_disc.head(25).to_markdown(index=False, floatfmt=".4f"),
               "\n\n## Failures vs controls (top-25 by KS)\n",
               fail_disc.head(25).to_markdown(index=False, floatfmt=".4f"),
               "\n"]
    (RESEARCH / "discriminating_features.md").write_text("\n".join(md_disc))

    md_arch = ["# Archetypes\n",
               "\n## Winner pre-window archetype (median feature values, 3m before base)\n",
               win_cent.to_frame("winner_centroid").to_markdown(floatfmt=".4f"),
               "\n\n## Failure pre-window archetype (median feature values, 3m before peak)\n",
               fail_cent.to_frame("failure_centroid").to_markdown(floatfmt=".4f"),
               "\n"]
    (RESEARCH / "archetypes.md").write_text("\n".join(md_arch))

    print("\n[done] reports in", RESEARCH)


if __name__ == "__main__":
    main()
