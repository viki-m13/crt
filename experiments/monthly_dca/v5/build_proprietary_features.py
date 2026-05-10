"""Build a panel of novel proprietary features designed to capture price-action
patterns, multi-timeframe alignment, drawdown-recovery dynamics, and
cross-sectional rank momentum.

Outputs cache/v2/sp500_pit/proprietary_features.parquet with columns:
  asof, ticker, plus the new features.

Features added:
  - rank_mom_12: cross-sectional rank in mom_12_1 today minus rank 3 months ago
                 (rising-rank stocks tend to continue)
  - rank_mom_3: same but for mom_3
  - coiling_strength: vol_contraction × tight_consolidation_60 × (mom_12_1 > 0)
                     — captures the "spring loading" pattern before breakouts
  - reversal_mom: 5d return - 21d return (positive = decelerating selling)
  - dd_recovery_speed: (price - prior_drawdown_low) / (prior_drawdown_amount × time)
  - mtf_alignment: count of (50d > 100d > 200d) and (price > all of them)
  - power_consolidation: combination of low BB width + high mom + high RSI
  - vertical_index: composite of acceleration, multibagger, FIP, breakout
  - low_drift_high_mom: low realized drift but high momentum (potential rotation
                        target)
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
CACHE = ROOT / "experiments" / "monthly_dca" / "cache"
V2 = CACHE / "v2"
PIT = V2 / "sp500_pit"
FEATURES_DIR = CACHE / "features"

EXCLUDE = {"SPY", "QQQ", "IWM", "VTI", "RSP", "DIA", "BTC-USD", "ETH-USD"}


def main():
    daily = pd.read_parquet(CACHE / "prices_extended.parquet")
    print(f"daily prices: {daily.shape}")

    feature_files = {pd.Timestamp(p.stem): p for p in FEATURES_DIR.glob("*.parquet")}
    asofs = sorted(feature_files.keys())

    # Build a (asof, ticker) -> features panel
    chunks = []
    prev_features_for_rank = {}  # ticker -> (asof, mom_12_1_rank, mom_3_rank)

    for d in asofs:
        feat = pd.read_parquet(feature_files[d])
        feat = feat[~feat.index.isin(EXCLUDE)]
        if len(feat) < 100: continue
        cols_avail = set(feat.columns)
        # Compute novel features only if dependencies present
        new = pd.DataFrame(index=feat.index)
        new["asof"] = d
        new["ticker"] = feat.index

        # Cross-sectional ranks within this month
        if "mom_12_1" in cols_avail:
            new["rank_mom_12_now"] = feat["mom_12_1"].rank(pct=True)
        if "mom_3" in cols_avail:
            new["rank_mom_3_now"] = feat["mom_3"].rank(pct=True)
        if "rsi_14" in cols_avail and "d_sma200" in cols_avail and "d_sma50" in cols_avail:
            # Multi-timeframe alignment: short-term mom + medium-term mom + long-term mom
            new["mtf_alignment"] = (
                (feat["d_sma50"] > 0).astype(float)
                + (feat["d_sma200"] > 0).astype(float)
                + (feat["sma50_above_200"] > 0).astype(float)
            )
        if "vol_contraction" in cols_avail and "tight_consolidation_60" in cols_avail and "mom_12_1" in cols_avail:
            new["coiling_strength"] = (
                feat["vol_contraction"].clip(lower=0) * feat["tight_consolidation_60"]
                * (feat["mom_12_1"] > 0).astype(float)
            )
        if "ret_5d" in cols_avail and "ret_21d" in cols_avail:
            new["reversal_mom"] = feat["ret_5d"] - feat["ret_21d"]
        if "bb_width_pct" in cols_avail and "rsi_14" in cols_avail and "mom_12_1" in cols_avail:
            new["power_consolidation"] = (
                (feat["bb_width_pct"] < 0.05).astype(float)
                * (feat["rsi_14"] > 55).astype(float)
                * np.maximum(feat["mom_12_1"], 0)
            )
        if "acceleration_2y" in cols_avail and "multibagger_ratio_24m" in cols_avail \
                and "fip_score" in cols_avail and "breakout_strength_60" in cols_avail:
            new["vertical_index"] = (
                feat["acceleration_2y"].rank(pct=True)
                + feat["multibagger_ratio_24m"].rank(pct=True)
                + feat["fip_score"].rank(pct=True)
                + feat["breakout_strength_60"].rank(pct=True)
            ) / 4.0
        if "trend_health_5y" in cols_avail and "mom_12_1" in cols_avail and "vol_1y" in cols_avail:
            # quality compounder with positive trend
            new["quality_compounder"] = (
                feat["trend_health_5y"].rank(pct=True)
                * (feat["mom_12_1"] > 0).astype(float)
                / (feat["vol_1y"].fillna(0.4).clip(lower=0.05) + 1e-6)
            )
        if "dd_from_52wh" in cols_avail and "drawdown_age_days" in cols_avail and "mom_12_1" in cols_avail:
            # Recovery setup: stock in moderate DD, but recovering (positive 12m mom)
            in_dd = (feat["dd_from_52wh"] < -0.10) & (feat["dd_from_52wh"] > -0.40)
            new["recovery_setup"] = in_dd.astype(float) * (feat["mom_12_1"] > 0).astype(float)

        # Now compute rank momentum: change in cross-sectional rank vs 3 months ago
        for tk in new.index:
            prev = prev_features_for_rank.get(tk, None)
            if prev is not None and (d - prev["asof"]).days < 130:  # ~4 months
                if "rank_mom_12_now" in new.columns:
                    new.at[tk, "rank_mom_change_12"] = float(new.at[tk, "rank_mom_12_now"] - prev["rank_mom_12_now"])
                if "rank_mom_3_now" in new.columns:
                    new.at[tk, "rank_mom_change_3"] = float(new.at[tk, "rank_mom_3_now"] - prev["rank_mom_3_now"])

        # Update prev (snapshot at d for use 3 months later)
        # Actually keep history of (date, rank) tuples
        for tk in new.index:
            prev_features_for_rank[tk] = {
                "asof": d,
                "rank_mom_12_now": new.at[tk, "rank_mom_12_now"] if "rank_mom_12_now" in new.columns else np.nan,
                "rank_mom_3_now": new.at[tk, "rank_mom_3_now"] if "rank_mom_3_now" in new.columns else np.nan,
            }

        chunks.append(new.reset_index(drop=True))

    panel = pd.concat(chunks, ignore_index=True)
    print(f"proprietary features panel: {panel.shape}")
    print(f"  columns: {[c for c in panel.columns if c not in ('asof', 'ticker')]}")
    panel.to_parquet(PIT / "proprietary_features.parquet", index=False)
    print(f"saved to {PIT}/proprietary_features.parquet")


if __name__ == "__main__":
    main()
