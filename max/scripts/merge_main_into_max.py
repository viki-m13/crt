#!/usr/bin/env python3
"""
Fuse the main scanner's 1Y/3Y/5Y outcomes into the max dataset so the Max page
can render 10D/30D/60D + 1Y/3Y/5Y for every ticker the two scanners share.

Run once, or re-run whenever you want to refresh the fused data between
full max scanner runs. The GitHub Actions max scan workflow replaces these
with true analog-matched probabilities across all 8 horizons.
"""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MAIN_DATA = ROOT / "docs" / "data"
MAX_DATA = ROOT / "max" / "docs" / "data"

LONG_HORIZONS = ["1Y", "3Y", "5Y"]


def load_json(path):
    with open(path) as f:
        return json.load(f)


def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f)


def per_item_long_fields(main_item):
    """Extract prob/median/downside for 1Y/3Y/5Y from a main-scan item row."""
    out = {}
    for src_key, h in [("prob_1y", "1Y"), ("prob_3y", "3Y"), ("prob_5y", "5Y")]:
        v = main_item.get(src_key)
        if v is not None:
            out[src_key] = v
    for src_key in ["median_1y", "median_3y", "median_5y", "downside_1y"]:
        v = main_item.get(src_key)
        if v is not None:
            out[src_key] = v
    return out


def fuse_ticker_files():
    main_tickers_dir = MAIN_DATA / "tickers"
    max_tickers_dir = MAX_DATA / "tickers"
    max_files = sorted(max_tickers_dir.glob("*.json"))
    fused = 0
    for mp in max_files:
        ticker = mp.stem
        main_path = main_tickers_dir / f"{ticker}.json"
        if not main_path.exists():
            continue
        main_d = load_json(main_path)
        max_d = load_json(mp)

        # Merge long-horizon outcomes
        main_outcomes = main_d.get("outcomes", {})
        max_outcomes = max_d.get("outcomes", {})
        for h in LONG_HORIZONS:
            if h in main_outcomes and h not in max_outcomes:
                max_outcomes[h] = main_outcomes[h]
        max_d["outcomes"] = max_outcomes

        # Copy baseline context too
        main_base = main_d.get("evidence_baseline", {})
        max_base = max_d.get("evidence_baseline", {})
        for h in LONG_HORIZONS:
            if h in main_base and h not in max_base:
                max_base[h] = main_base[h]
        max_d["evidence_baseline"] = max_base

        save_json(mp, max_d)
        fused += 1
    return fused


def fuse_full_json():
    main_full = load_json(MAIN_DATA / "full.json")
    max_full = load_json(MAX_DATA / "full.json")

    # Index main items by ticker
    main_by_t = {it["ticker"]: it for it in main_full.get("items", [])}

    for item in max_full.get("items", []):
        m = main_by_t.get(item["ticker"])
        if not m:
            continue
        for k, v in per_item_long_fields(m).items():
            if item.get(k) is None:
                item[k] = v

    # Update details map for top-10
    main_details = main_full.get("details", {}) or {}
    for t, det in (max_full.get("details") or {}).items():
        m_det = main_details.get(t)
        if not m_det:
            continue
        for h in LONG_HORIZONS:
            if h in (m_det.get("outcomes") or {}) and h not in (det.get("outcomes") or {}):
                det.setdefault("outcomes", {})[h] = m_det["outcomes"][h]
            if h in (m_det.get("evidence_baseline") or {}) and h not in (det.get("evidence_baseline") or {}):
                det.setdefault("evidence_baseline", {})[h] = m_det["evidence_baseline"][h]

    # Record that the data was fused
    max_full.setdefault("model", {})["fused_from_main"] = True

    save_json(MAX_DATA / "full.json", max_full)


def main():
    n = fuse_ticker_files()
    fuse_full_json()
    print(f"[merge] Fused long-horizon outcomes into {n} ticker files and full.json")


if __name__ == "__main__":
    main()
