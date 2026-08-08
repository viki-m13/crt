#!/bin/bash
# Full pipeline rebuild after fetch completion. Run from sharpe5_options/.
set -e
echo "=== features $(date -u +%H:%M) ==="
python3 features.py
echo "=== structures $(date -u +%H:%M) ==="
python3 structures.py
echo "=== exits $(date -u +%H:%M) ==="
python3 structures2_exits.py
echo "=== study1 $(date -u +%H:%M) ==="
python3 study1_predictability.py
echo "=== study2 $(date -u +%H:%M) ==="
python3 study2_sleeves.py
echo "=== rebuild done $(date -u +%H:%M) ==="
