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


def _step(label: str, fn) -> bool:
    """Run one refresh step in isolation. Steps 0-4 are *optional*
    data-freshening; a failure in any of them must NOT prevent Step 5
    (the E2 data.json rebuild) from running on whatever data is present.
    Returns True on success, False on handled failure."""
    print(f"\n=== {label} ===")
    try:
        fn()
        return True
    except Exception as e:  # noqa: BLE001 — deliberately fail-soft per step
        print(f"  WARN: '{label}' failed ({e}); continuing with existing "
              f"data.", file=sys.stderr)
        traceback.print_exc()
        return False


def main() -> int:
    panel_box = {}

    def _load():
        panel_box["panel"] = run_load(force=True)
        p = panel_box["panel"]
        print(f"  panel shape={p.shape}  range={p.index.min().date()} → "
              f"{p.index.max().date()}")

    def _features():
        run_features(start="2017-01-01", end="2099-01-01")
        run_extras(start="2017-01-01", end="2099-01-01")
        if "panel" in panel_box:
            refresh_recent_months(panel_box["panel"], lookback_months=3)

    def _chronos():
        feat_dir = ROOT / "experiments" / "monthly_dca" / "cache" / "features"
        ff = sorted(feat_dir.glob("*.parquet"))
        if ff:
            score_chronos_incremental(pd.Timestamp(ff[-1].stem))

    # Steps 0-4: optional, independently fail-soft.
    _step("Step 0: Refreshing IVV holdings", refresh_ivv_holdings)
    _step("Step 1: Rebuilding prices panel", _load)
    _step("Step 2: Refreshing features (recent month-ends)", _features)
    _step("Step 3: Refreshing PIT S&P 500 membership",
          refresh_pit_membership)
    _step("Step 4: Scoring Chronos-bolt-tiny for latest asof", _chronos)

    # Step 5: the deliverable — the E2 data.json the website renders.
    # This MUST run even if any optional step above failed, so the
    # public page is never left frozen on a stale build.
    print("\n=== Step 5: Rebuilding webapp data.json (v5 PIT-S&P-500, "
          "variant=E2) ===")
    try:
        load_features_long.cache_clear()
        from experiments.monthly_dca.v5.build_webapp_v5_pit import (
            main as build_v5)
        build_v5()
        print("\nDaily refresh complete (v5 / E2).")
        return 0
    except Exception as e:
        print(f"FATAL: E2 data.json rebuild failed: {e}", file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
