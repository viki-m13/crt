"""Phase 3b: layer alpha + alpha2 + extra features onto the augmented base
features cached in Phase 3.

The original v2 feature pipeline is:
  1. cache_features.py        ->  21 base features
  2. alpha_features.py        -> +29 alpha features
  3. alpha2_features.py       -> +17 alpha2 features
  4. extra_features.py        -> +12 extra features
                              = 79 cols total per month

Our Phase 3 only ran step 1 on the augmented panel, producing 21-col files.
This script runs steps 2-4 using the augmented panel as the source for all
computations, merging into the augmented feature files until they match the
79-col schema downstream code expects.

We do NOT mutate the original cache/features/. We monkey-patch the module
constants used by the three feature scripts so they read+write only the
augmented dir.

Inputs:
  experiments/monthly_dca/cache/v2/sp500_pit/prices_extended_pit.parquet
  experiments/monthly_dca/cache/v2/sp500_pit/augmented/features/*.parquet (21 cols)

Outputs:
  experiments/monthly_dca/cache/v2/sp500_pit/augmented/features/*.parquet (~79 cols)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
CACHE = ROOT / "experiments" / "monthly_dca" / "cache"
PIT = CACHE / "v2" / "sp500_pit"
AUG_FEATURES = PIT / "augmented" / "features"
AUG_PANEL = PIT / "prices_extended_pit.parquet"

sys.path.insert(0, str(ROOT))

# Monkey-patch fast_engine constants BEFORE importing the feature scripts.
import experiments.monthly_dca.fast_engine as fast_engine  # noqa: E402
fast_engine.FEATURES_DIR = AUG_FEATURES
fast_engine.CACHE = PIT  # so any incidental CACHE / "..." paths land here

_orig_load_panel = fast_engine.load_panel
def _patched_load_panel():
    return pd.read_parquet(AUG_PANEL)
fast_engine.load_panel = _patched_load_panel

# Also force load_features to read from augmented
def _patched_load_features(asof):
    return pd.read_parquet(AUG_FEATURES / f"{asof.date()}.parquet")
fast_engine.load_features = _patched_load_features


def main():
    t0 = time.time()
    print("=" * 64)
    print("Phase 3b: layer alpha + alpha2 + extra onto augmented features")
    print("=" * 64)

    print(f"[0] augmented panel: {AUG_PANEL}")
    print(f"    augmented features dir: {AUG_FEATURES}")
    print(f"    feature files: {len(list(AUG_FEATURES.glob('*.parquet')))}")

    # Re-import each feature script AFTER monkey-patching so they pick up the
    # patched fast_engine constants. Each script's main() iterates the
    # features dir and merges its outputs back into each file.
    print("\n[1] alpha_features ...")
    import experiments.monthly_dca.alpha_features as af
    # Patch its FEATURES_DIR reference (was imported by name)
    af.FEATURES_DIR = AUG_FEATURES
    af.load_panel = _patched_load_panel
    af.main()
    print(f"    elapsed: {time.time()-t0:.1f}s")

    print("\n[2] alpha2_features ...")
    import experiments.monthly_dca.alpha2_features as a2f
    a2f.FEATURES_DIR = AUG_FEATURES
    a2f.CACHE = PIT
    a2f.load_panel = _patched_load_panel
    a2f.main(force=False)
    print(f"    elapsed: {time.time()-t0:.1f}s")

    print("\n[3] extra_features ...")
    import experiments.monthly_dca.extra_features as ef
    ef.FEATURES_DIR = AUG_FEATURES
    ef.load_panel = _patched_load_panel
    ef.load_features = _patched_load_features
    ef.main()
    print(f"    elapsed: {time.time()-t0:.1f}s")

    # Verify
    sample = pd.Timestamp("2010-12-31")
    sp = AUG_FEATURES / f"{sample.date()}.parquet"
    if sp.exists():
        df = pd.read_parquet(sp)
        print(f"\n[4] sample asof {sample.date()}: shape={df.shape}")
        orig = pd.read_parquet(CACHE / "features" / f"{sample.date()}.parquet")
        new_cols = set(df.columns)
        orig_cols = set(orig.columns)
        print(f"    matches original cols: {new_cols == orig_cols}")
        print(f"    aug-only:  {sorted(new_cols - orig_cols)}")
        print(f"    orig-only: {sorted(orig_cols - new_cols)}")
    print(f"\nDone in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
