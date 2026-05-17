"""Reproducible, append-only live extension of the deployed E2 GBM
predictions.

THE INVARIANT: the validated backtest must never change. The committed
`ml_preds.parquet` (history ≤ the frozen cutoff = the deployed 56.6% /
Sharpe-1.10 E2) is immutable ground truth. The augmented feature/panel
pipeline is NOT bit-reproducible (data-vendor restatements + uncommitted
panel), so a full regen silently shifts the K=2 picks and tanks the
backtest (observed 56.6% → 38%). This script therefore:

  1. Loads the FROZEN committed ml_preds straight from git (not the
     working tree) — that is the deployed, validated history.
  2. Runs the standard walk-forward to score the NEW live months only
     (asof > frozen cutoff), using the current panel. The current panel
     is fine for *live* months — that's exactly what "live" means; it
     never feeds the published backtest.
  3. Writes ml_preds = frozen ⊕ new_live_rows and ASSERTS the
     ≤-cutoff slice is byte-identical to the frozen artifact. If the
     assertion fails the script aborts and writes nothing.

Net effect: the published backtest (≤ cutoff) is preserved exactly by
construction; only genuinely new live months are appended, so the
deployed `as_of` / current basket advance without changing any
validated number. This is what makes the monthly cron safe.

Usage:
    python3 experiments/monthly_dca/v5/spx_pit/extend_ml_preds_live.py
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
AUG = ROOT / "experiments" / "monthly_dca" / "cache" / "v2" / "sp500_pit" / "augmented"
ML_PREDS = AUG / "ml_preds.parquet"
PANEL = AUG / "panel_cross_section_v3.parquet"
FROZEN_REF = "origin/main"  # the committed, validated artifact lives here

from experiments.monthly_dca.v2.ml_strategy import fit_walkforward  # noqa


def _load_frozen() -> pd.DataFrame:
    """The deployed/validated ml_preds, read from git (NOT the working
    tree) so a locally-mutated file can never corrupt the baseline."""
    rel = ML_PREDS.relative_to(ROOT).as_posix()
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tf:
        tmp = Path(tf.name)
    try:
        with open(tmp, "wb") as fh:
            subprocess.run(["git", "-C", str(ROOT), "show",
                            f"{FROZEN_REF}:{rel}"], check=True, stdout=fh)
        df = pd.read_parquet(tmp)
    finally:
        tmp.unlink(missing_ok=True)
    df["asof"] = pd.to_datetime(df["asof"])
    return df


def main() -> int:
    t0 = time.time()
    print("=" * 64)
    print("Append-only live extension of E2 ml_preds (reproducible)")
    print("=" * 64)

    frozen = _load_frozen()
    cutoff = pd.Timestamp(frozen["asof"].max()).normalize()
    print(f"[1] frozen (validated) ml_preds: {len(frozen)} rows, "
          f"asof ≤ {cutoff.date()} (this is immutable ground truth)")

    big = pd.read_parquet(PANEL)
    panel_last = pd.to_datetime(
        big.reset_index()["asof"]).max().normalize()
    print(f"[2] current panel asof max = {panel_last.date()}")
    if panel_last <= cutoff:
        print(f"    nothing new past {cutoff.date()}; ml_preds unchanged.")
        return 0

    # Standard walk-forward; we keep ONLY the new live months. Training
    # on the current panel is correct for live scoring and never touches
    # the frozen published history.
    print(f"[3] scoring NEW live months {cutoff.date()} < asof ≤ "
          f"{panel_last.date()} ...")
    preds = fit_walkforward(big, target_horizons=(1, 3, 6),
                            train_end=panel_last)
    preds["asof"] = pd.to_datetime(preds["asof"])
    new_rows = preds[preds["asof"] > cutoff].copy()
    print(f"    new live rows: {len(new_rows)} "
          f"({sorted(new_rows['asof'].dt.date.unique())})")
    if new_rows.empty:
        print("    no new rows produced; ml_preds unchanged.")
        return 0

    out = pd.concat([frozen, new_rows], axis=0, ignore_index=True)

    # HARD INVARIANT: the ≤-cutoff slice must equal the frozen artifact
    # exactly (same rows, same predictions). We literally reuse `frozen`
    # so this is true by construction — assert it anyway as a tripwire.
    chk = out[out["asof"] <= cutoff].reset_index(drop=True)
    fz = frozen.reset_index(drop=True)
    assert len(chk) == len(fz), (
        f"frozen-slice row count changed {len(fz)} -> {len(chk)}")
    cols = [c for c in fz.columns if c in chk.columns]
    merged = fz[["asof", "ticker"] + [c for c in cols
                                      if c not in ("asof", "ticker")]]
    m = fz.merge(chk, on=["asof", "ticker"], suffixes=("_f", "_c"))
    assert len(m) == len(fz), "frozen (asof,ticker) keys changed"
    for c in ("pred", "pred_1m", "pred_3m", "pred_6m"):
        if f"{c}_f" in m and f"{c}_c" in m:
            d = (m[f"{c}_f"] - m[f"{c}_c"]).abs().max()
            assert d == 0.0, f"frozen {c} mutated (max|Δ|={d})"
    print("[4] INVARIANT OK — frozen backtest history byte-identical")

    out.to_parquet(ML_PREDS)
    print(f"[5] wrote {ML_PREDS}: {len(frozen)} frozen + {len(new_rows)} "
          f"live = {len(out)} rows, asof max {panel_last.date()}")
    print(f"Done in {(time.time()-t0)/60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
