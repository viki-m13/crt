"""Phase 5a: build the augmented sp500_pit_panel.parquet.

Mirror of experiments/monthly_dca/v4/build_pit_panel.py but reads from
the augmented features dir and augmented monthly_returns_clean. Output
is what sp500_pit_extended_sweep.build_panel_with_score consumes.

Inputs:
  experiments/monthly_dca/cache/v2/sp500_pit/sp500_membership_monthly.parquet (unchanged)
  experiments/monthly_dca/cache/v2/sp500_pit/augmented/features/*.parquet
  experiments/monthly_dca/cache/v2/sp500_pit/augmented/monthly_returns_clean.parquet

Output:
  experiments/monthly_dca/cache/v2/sp500_pit/augmented/sp500_pit_panel.parquet
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
CACHE = ROOT / "experiments" / "monthly_dca" / "cache"
V2 = CACHE / "v2"
PIT = V2 / "sp500_pit"
AUG = PIT / "augmented"
FEATURES_DIR = AUG / "features"

EXCLUDE = {"SPY", "QQQ", "IWM", "VTI", "RSP", "DIA", "BTC-USD", "ETH-USD",
           "TQQQ", "SQQQ", "UPRO", "SPXL", "SPXS", "TZA", "TNA", "SOXL", "SOXS",
           "FAS", "FAZ", "TMF", "TMV", "UGL", "GLL", "BOIL", "KOLD"}


def build_pit_panel(members: pd.DataFrame, monthly_returns: pd.DataFrame) -> pd.DataFrame:
    members = members.copy()
    members["asof"] = pd.to_datetime(members["asof"])
    members_g = members.groupby("asof")["ticker"].apply(set)

    panel_chunks = []
    feature_files = {pd.Timestamp(p.stem): p for p in FEATURES_DIR.glob("*.parquet")}
    asofs = sorted(set(members_g.index) & set(feature_files.keys()))
    print(f"  building panel for {len(asofs)} months ({asofs[0].date()}..{asofs[-1].date()})")

    ref_d = pd.Timestamp("2010-12-31")
    if ref_d not in feature_files:
        ref_d = sorted(feature_files.keys())[len(feature_files) // 2]
    feature_cols = list(pd.read_parquet(feature_files[ref_d]).columns)
    print(f"  reference feature columns from {ref_d.date()}: {len(feature_cols)} cols")

    for d in asofs:
        feat = pd.read_parquet(feature_files[d])
        sp = members_g[d]
        feat = feat[feat.index.isin(sp)]
        feat = feat[~feat.index.isin(EXCLUDE)]
        if len(feat) < 50:
            continue
        if not set(feature_cols).issubset(feat.columns):
            continue
        feat = feat[feature_cols]
        for c in feature_cols:
            r = feat[c].rank(pct=True)
            feat[c + "_xs"] = (r - 0.5) * 2
        feat = feat.reset_index().rename(columns={"index": "ticker"})
        feat["asof"] = d
        panel_chunks.append(feat)
    panel = pd.concat(panel_chunks, axis=0, ignore_index=True)
    panel = panel[["asof", "ticker"] + [c + "_xs" for c in feature_cols] + feature_cols]
    print(f"  panel built: {panel.shape}, {panel['ticker'].nunique()} tickers")

    mr_dates = monthly_returns.index.sort_values()
    asof_idx = pd.DatetimeIndex(asofs)
    print("  computing forward returns 1m, 3m, 6m, 12m ...")
    log_mr = np.log1p(monthly_returns.fillna(0)).cumsum()
    asof_to_pos = {}
    for d in asof_idx:
        pos = mr_dates.searchsorted(d)
        cand = []
        for j in (pos - 1, pos):
            if 0 <= j < len(mr_dates):
                cand.append((j, abs((mr_dates[j] - d).days)))
        cand.sort(key=lambda x: x[1])
        if cand and cand[0][1] <= 7:
            asof_to_pos[d] = cand[0][0]

    fwd_dict = {1: [], 3: [], 6: [], 12: []}
    for h in (1, 3, 6, 12):
        for _, row in panel[["asof", "ticker"]].iterrows():
            d = row["asof"]
            tk = row["ticker"]
            pos = asof_to_pos.get(d, None)
            if pos is None or pos + h >= len(mr_dates) or tk not in monthly_returns.columns:
                fwd_dict[h].append(np.nan)
                continue
            d0 = mr_dates[pos]
            dh = mr_dates[pos + h]
            try:
                lr0 = log_mr.at[d0, tk]
                lrh = log_mr.at[dh, tk]
            except KeyError:
                fwd_dict[h].append(np.nan)
                continue
            if pd.isna(lr0) or pd.isna(lrh):
                fwd_dict[h].append(np.nan)
                continue
            fwd_dict[h].append(np.expm1(lrh - lr0))
    for h in (1, 3, 6, 12):
        panel[f"fwd_{h}m_ret"] = fwd_dict[h]
    for h in (1, 3, 6, 12):
        panel[f"rank_target_{h}m"] = panel.groupby("asof")[f"fwd_{h}m_ret"].rank(pct=True)
    return panel


def main():
    t0 = time.time()
    print("=" * 64)
    print("Phase 5a: build augmented sp500_pit_panel")
    print("=" * 64)

    members = pd.read_parquet(PIT / "sp500_membership_monthly.parquet")
    members["asof"] = pd.to_datetime(members["asof"])
    print(f"[1] members: {len(members)}, {members['asof'].nunique()} months, "
          f"{members['ticker'].nunique()} tickers")

    monthly_returns = pd.read_parquet(AUG / "monthly_returns_clean.parquet")
    print(f"[2] augmented monthly_returns: {monthly_returns.shape}")

    panel = build_pit_panel(members, monthly_returns)
    out = AUG / "sp500_pit_panel.parquet"
    panel.to_parquet(out, index=False)
    print(f"[3] saved -> {out}  ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
