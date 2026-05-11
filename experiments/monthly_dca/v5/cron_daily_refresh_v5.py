"""Cron-friendly daily refresh for the v5 PIT-S&P-500 strategy webapp.

End-to-end refresh (~3-5 min):
  1. Refresh iShares IVV holdings (current S&P 500 universe) + fetch any
     missing tickers via yfinance.
  2. Rebuild prices panel + monthly returns from docs/data/tickers/*.json.
  3. Compute base + extra features for any month-end not yet cached
     (and refresh the most recent two months, which may have grown).
  4. Refresh PIT S&P 500 membership panel (extends through latest live month).
  5. Score Chronos-bolt-tiny on the latest asof for the current S&P 500
     (incremental — appends to ml_preds_chronos.parquet).
  6. Rebuild experiments/docs/monthly-dca/data.json with the v5 strategy.

Designed to be idempotent and fail-soft.
"""
from __future__ import annotations

import subprocess
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np

from experiments.monthly_dca.backtester import month_end_dates
from experiments.monthly_dca.cache_features import main as run_features
from experiments.monthly_dca.extra_features import main as run_extras
from experiments.monthly_dca.load_data import main as run_load
from experiments.monthly_dca.fast_score import load_features_long

CACHE = ROOT / "experiments" / "monthly_dca" / "cache"
V2 = CACHE / "v2"
PIT = V2 / "sp500_pit"
CHRONOS_PRED_PATH = PIT / "ml_preds_chronos.parquet"


def refresh_recent_months(panel: pd.DataFrame, lookback_months: int = 3) -> None:
    me = month_end_dates(panel.index)
    target = me[-lookback_months:]
    if len(target) == 0:
        return
    start = target[0].strftime("%Y-%m-%d")
    end = target[-1].strftime("%Y-%m-%d")
    feat_dir = ROOT / "experiments" / "monthly_dca" / "cache" / "features"
    for d in target:
        p = feat_dir / f"{d.date()}.parquet"
        if p.exists():
            p.unlink()
    print(f"Refreshing features for {len(target)} month-ends: {start} → {end}")
    run_features(start=start, end=end)
    run_extras(start=start, end=end)


def refresh_pit_membership() -> None:
    print("Refreshing PIT S&P 500 membership panel...")
    script = ROOT / "experiments" / "monthly_dca" / "v2" / "build_sp500_pit_membership.py"
    subprocess.run([sys.executable, str(script)], check=True)


def refresh_ivv_holdings() -> bool:
    """Fetch latest IVV holdings + any missing tickers."""
    try:
        from experiments.monthly_dca.v5.load_ivv_holdings import main as run_ivv
        run_ivv()
        return True
    except Exception as e:
        print(f"  WARNING: IVV holdings refresh failed: {e}")
        traceback.print_exc()
        return False


def score_chronos_incremental(asof: pd.Timestamp) -> bool:
    """Score Chronos-bolt-tiny for the latest asof (current S&P 500 members)
    and append to ml_preds_chronos.parquet.  Idempotent: skips rows already
    in the parquet for this asof."""
    try:
        import torch
        from chronos import BaseChronosPipeline
    except ImportError:
        print(f"  WARNING: chronos-forecasting not installed; skipping Chronos scoring")
        return False
    try:
        members = pd.read_parquet(PIT / "sp500_membership_monthly.parquet")
        members["asof"] = pd.to_datetime(members["asof"])
        mem_at_asof = members[members["asof"] == asof]
        if len(mem_at_asof) == 0:
            # Use most recent membership snapshot
            latest_asof = members["asof"].max()
            mem_at_asof = members[members["asof"] == latest_asof]
            print(f"  Using membership from {latest_asof.date()} for {asof.date()}")

        existing = None
        if CHRONOS_PRED_PATH.exists():
            existing = pd.read_parquet(CHRONOS_PRED_PATH)
            existing["asof"] = pd.to_datetime(existing["asof"])
            if asof in existing["asof"].unique():
                print(f"  Chronos preds already exist for {asof.date()}, skipping")
                return True

        daily = pd.read_parquet(CACHE / "prices_extended.parquet")
        if asof not in daily.index:
            # Find most recent <= asof
            pos = daily.index.searchsorted(asof, side="right") - 1
            if pos < 0:
                print(f"  No daily prices at/before {asof.date()}")
                return False

        print(f"  Loading chronos-bolt-tiny...")
        pipe = BaseChronosPipeline.from_pretrained(
            "amazon/chronos-bolt-tiny", device_map="cpu", torch_dtype=torch.float32
        )

        WINDOW_DAYS = 252
        PRED_LEN = 64
        contexts, tickers = [], []
        for tk in mem_at_asof["ticker"]:
            if tk not in daily.columns: continue
            ts = daily[tk].dropna()
            pos = ts.index.searchsorted(asof, side="right") - 1
            if pos < WINDOW_DAYS: continue
            window = ts.iloc[pos - WINDOW_DAYS + 1: pos + 1]
            if len(window) < WINDOW_DAYS or window.isna().any(): continue
            arr = window.values.astype(np.float32)
            contexts.append(arr); tickers.append(tk)
        if not contexts:
            print(f"  No Chronos contexts at {asof.date()}")
            return False
        print(f"  Forecasting {len(contexts)} tickers...")
        rows = []
        for j in range(0, len(contexts), 500):
            batch = [torch.tensor(c, dtype=torch.float32) for c in contexts[j:j+500]]
            with torch.no_grad():
                out = pipe.predict(batch, prediction_length=PRED_LEN)
            preds = np.asarray(out)
            for i, tk in enumerate(tickers[j:j+500]):
                entry_px = contexts[j+i][-1]
                rows.append({
                    "asof": asof, "ticker": tk,
                    "chronos_p50_3m": float(preds[i, 4, -1] / entry_px - 1),
                    "chronos_p70_3m": float(preds[i, 6, -1] / entry_px - 1),
                    "chronos_p90_3m": float(preds[i, 8, -1] / entry_px - 1),
                    "chronos_p50_peak": float(np.max(preds[i, 4, :]) / entry_px - 1),
                })
        new_df = pd.DataFrame(rows)
        if existing is not None:
            combined = pd.concat([existing, new_df], ignore_index=True)
            # Drop duplicates (asof, ticker) keeping latest
            combined = combined.drop_duplicates(subset=["asof", "ticker"], keep="last")
        else:
            combined = new_df
        combined.to_parquet(CHRONOS_PRED_PATH, index=False)
        print(f"  Saved Chronos preds: {len(new_df)} new rows, {len(combined)} total")
        return True
    except Exception as e:
        print(f"  ERROR scoring Chronos: {e}")
        traceback.print_exc()
        return False


def main() -> int:
    try:
        print("=== Step 0: Refreshing IVV holdings ===")
        refresh_ivv_holdings()

        print("\n=== Step 1: Rebuilding prices panel ===")
        panel = run_load(force=True)
        print(f"  panel shape={panel.shape}  date range={panel.index.min().date()} → {panel.index.max().date()}")

        print("\n=== Step 2: Refreshing features for recent month-ends ===")
        run_features(start="2017-01-01", end="2099-01-01")
        run_extras(start="2017-01-01", end="2099-01-01")
        refresh_recent_months(panel, lookback_months=3)

        print("\n=== Step 3: Refreshing PIT S&P 500 membership ===")
        refresh_pit_membership()

        print("\n=== Step 4: Scoring Chronos-bolt-tiny for latest asof ===")
        # Use latest features asof
        feat_dir = ROOT / "experiments" / "monthly_dca" / "cache" / "features"
        feat_files = sorted(feat_dir.glob("*.parquet"))
        if feat_files:
            latest_asof = pd.Timestamp(feat_files[-1].stem)
            score_chronos_incremental(latest_asof)

        print("\n=== Step 5: Rebuilding webapp data.json (Mode B = 50% v5 + 50% trend sleeve) ===")
        load_features_long.cache_clear()
        # Mode B is the deployed production strategy. It internally runs the
        # v5 picker simulator (build_webapp_v5_pit's run_full_sim) AND overlays
        # the multi-asset trend rotation sleeve, then emits data.json with the
        # blended numbers. To return to Mode A (v5 picker only), swap the
        # import to build_webapp_v5_pit.
        from experiments.monthly_dca.v5.build_webapp_v5_mode_b import main as build_mode_b
        build_mode_b()

        print("\nDaily refresh complete (Mode B).")
        return 0
    except Exception as e:
        print(f"ERROR during daily refresh: {e}", file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
