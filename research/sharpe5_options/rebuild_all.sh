#!/bin/bash
# Full research pipeline on the complete panel. Run from sharpe5_options/.
# Each stage is resumable; structures checkpoints every 50 dates.
set -e
export USE_TIER2=1
cd "$(dirname "$0")"
echo "=== features $(date -u +%H:%M) ==="
python3 features.py
echo "=== structures $(date -u +%H:%M) ==="
python3 structures.py
echo "=== study1 predictability $(date -u +%H:%M) ==="
python3 study1_predictability.py
echo "=== study2 sleeves $(date -u +%H:%M) ==="
python3 study2_sleeves.py
echo "=== study4 dispersion $(date -u +%H:%M) ==="
python3 study4_dispersion.py
echo "=== rebuild done $(date -u +%H:%M) ==="
