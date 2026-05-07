"""QSIT v2 — score-based intraday trigger.

v1 used a strict AND-gate of all 4 signals → 3 fires total. v2:

  * Loosens each individual filter to a "soft" condition.
  * Builds a composite SCORE (0-4) from the four signals.
  * Reports win-rate and ROI at every score threshold.
  * Tests multiple outcome thresholds (0.3%, 0.5%, 1%, 2%).

This lets us find the natural accuracy ceiling and the precise score
threshold that delivers 95% win rate (if any exists).

Key innovation: The SCORE is sum of (VCC_signal + VFA_dir_signal +
CACB_dir_signal + TODP_signal). Direction (UP/DN) is the SIGN of the
weighted average of (VFA + CACB).
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_HERE, "data", "intraday")


WINDOW_BARS = 15

# LOOSER thresholds for individual signals
VCC_PERCENTILES = [0.05, 0.10, 0.15, 0.25, 0.40]    # tested at multiple
VFA_VOL_RATIO_MIN = 1.2
VFA_CPIR_HIGH = 0.65
VFA_CPIR_LOW  = 0.35
CACB_AGREE_MIN = 1
CACB_TOP_K = 3
TODP_WINDOWS_UTC = [
    ("13:35", "14:30"),  # 09:35 - 10:30 ET
    ("17:30", "18:30"),  # 13:30 - 14:30 ET
    ("19:00", "19:55"),  # 15:00 - 15:55 ET
]
OUTCOME_BARS = 30
OUTCOME_THRESHOLDS = [0.003, 0.005, 0.01, 0.02]   # 0.3%, 0.5%, 1%, 2%

TARGET_TICKERS = ["QCOM", "INTC", "SLV", "GLD", "AVGO", "AMD",
                  "NVDA", "XLK", "SMH", "GDX", "PLTR", "XLE", "USO"]
PEER_POOL = TARGET_TICKERS + ["SPY", "QQQ", "IWM", "TLT"]


def _load(tk):
    path = os.path.join(DATA_DIR, f"{tk}.json")
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        b = json.load(fh)
    df = pd.DataFrame({
        "open": b["open"], "high": b["high"], "low": b["low"],
        "close": b["close"], "volume": b["volume"],
    }, index=pd.to_datetime(b["datetimes"], utc=True))
    return df.sort_index()


def _rv15(closes: np.ndarray) -> np.ndarray:
    n = len(closes)
    out = np.full(n, np.nan)
    if n < WINDOW_BARS + 2:
        return out
    log_ret = np.concatenate(([0.0], np.diff(np.log(np.maximum(closes, 1e-9)))))
    csum = np.concatenate(([0.0], np.cumsum(log_ret * log_ret)))
    var_w = (csum[WINDOW_BARS:] - csum[:-WINDOW_BARS]) / WINDOW_BARS
    out[WINDOW_BARS - 1:] = np.sqrt(np.maximum(var_w, 0.0))
    return out


def compute_features(df):
    f = df.copy()
    closes = f["close"].values
    f["rv_15"] = _rv15(closes)
    f["high_15"] = f["high"].rolling(WINDOW_BARS).max()
    f["low_15"]  = f["low"].rolling(WINDOW_BARS).min()
    f["vol_15"]  = f["volume"].rolling(WINDOW_BARS).sum()
    rng = f["high_15"] - f["low_15"]
    f["cpir"] = np.where(rng > 1e-9, (f["close"] - f["low_15"]) / rng, np.nan)
    # Time-of-day key (30-min slots)
    f["tod_slot"] = (f.index.hour * 60 + f.index.minute) // 30
    return f


def build_tod_dist(feat):
    """Per-slot empirical distributions of rv_15 and vol_15."""
    out = defaultdict(lambda: {"rv": [], "vol": []})
    for ts, row in feat.iterrows():
        if not (np.isfinite(row["rv_15"]) and np.isfinite(row["vol_15"])):
            continue
        out[int(row["tod_slot"])]["rv"].append(float(row["rv_15"]))
        out[int(row["tod_slot"])]["vol"].append(float(row["vol_15"]))
    return {k: {"rv": np.sort(np.array(v["rv"])),
                "vol": np.sort(np.array(v["vol"]))}
            for k, v in out.items()}


def percentile_in(value, sorted_arr):
    if len(sorted_arr) == 0:
        return float("nan")
    return np.searchsorted(sorted_arr, value) / len(sorted_arr)


def find_peers(target, all_data, k=CACB_TOP_K):
    base = np.log(all_data[target]["close"]).diff().dropna()
    base_5 = base.resample("5min").sum().dropna()
    corrs = []
    for tk, df in all_data.items():
        if tk == target:
            continue
        peer = np.log(df["close"]).diff().dropna().resample("5min").sum().dropna()
        joined = pd.concat([base_5, peer], axis=1, join="inner").dropna()
        if len(joined) < 100:
            continue
        c = joined.iloc[:, 0].corr(joined.iloc[:, 1])
        if not np.isnan(c):
            corrs.append((tk, c))
    corrs.sort(key=lambda x: -abs(x[1]))
    return [tk for tk, _ in corrs[:k]]


def compute_peer_vfa(feat, tod_dist):
    """Compute VFA_UP / VFA_DN columns on this feature frame."""
    vfa_up = []; vfa_dn = []
    for ts, row in feat.iterrows():
        if not (np.isfinite(row["vol_15"]) and np.isfinite(row["cpir"])):
            vfa_up.append(False); vfa_dn.append(False); continue
        slot = int(row["tod_slot"])
        base = tod_dist.get(slot, {}).get("vol")
        if base is None or len(base) == 0:
            vfa_up.append(False); vfa_dn.append(False); continue
        med = np.median(base)
        if med <= 0:
            vfa_up.append(False); vfa_dn.append(False); continue
        vr = row["vol_15"] / med
        vfa_up.append(bool(vr > VFA_VOL_RATIO_MIN and row["cpir"] > VFA_CPIR_HIGH))
        vfa_dn.append(bool(vr > VFA_VOL_RATIO_MIN and row["cpir"] < VFA_CPIR_LOW))
    feat["VFA_UP"] = vfa_up
    feat["VFA_DN"] = vfa_dn
    return feat


def compute_qsit_score(target, feat, peers, peer_feats, vcc_pct):
    """For each bar, compute (score_up, score_dn) ∈ {0..4}.
    Score is the count of agreed signals in that direction."""
    target_tod = build_tod_dist(feat)

    # S1: VCC — vol below percentile
    vcc = []
    for ts, row in feat.iterrows():
        if not np.isfinite(row["rv_15"]):
            vcc.append(False); continue
        slot = int(row["tod_slot"])
        base = target_tod.get(slot, {}).get("rv")
        if base is None or len(base) == 0:
            vcc.append(False); continue
        p = percentile_in(row["rv_15"], base)
        vcc.append(bool(p < vcc_pct))
    feat["VCC"] = vcc

    # S2: VFA already computed — VFA_UP / VFA_DN

    # S3: CACB — count peers with same-direction VFA
    peer_up = pd.DataFrame({p: peer_feats[p]["VFA_UP"].astype(int) for p in peers})
    peer_dn = pd.DataFrame({p: peer_feats[p]["VFA_DN"].astype(int) for p in peers})
    aligned_up = peer_up.reindex(feat.index, method="ffill", limit=2).fillna(0)
    aligned_dn = peer_dn.reindex(feat.index, method="ffill", limit=2).fillna(0)
    cacb_up = aligned_up.sum(axis=1) >= CACB_AGREE_MIN
    cacb_dn = aligned_dn.sum(axis=1) >= CACB_AGREE_MIN
    feat["CACB_UP"] = cacb_up
    feat["CACB_DN"] = cacb_dn

    # S4: TODP
    def _in(ts):
        hm = ts.strftime("%H:%M")
        return any(s <= hm <= e for s, e in TODP_WINDOWS_UTC)
    feat["TODP"] = feat.index.map(_in)

    # Score
    feat["score_up"] = (feat["VCC"].astype(int)
                       + feat["VFA_UP"].astype(int)
                       + feat["CACB_UP"].astype(int)
                       + feat["TODP"].astype(int))
    feat["score_dn"] = (feat["VCC"].astype(int)
                       + feat["VFA_DN"].astype(int)
                       + feat["CACB_DN"].astype(int)
                       + feat["TODP"].astype(int))
    return feat


def evaluate(feat, threshold_pct):
    """Walk through bars; for each (score, side), record whether
    target moved ≥ threshold_pct in side direction within OUTCOME_BARS."""
    n = len(feat)
    closes = feat["close"].values
    highs  = feat["high"].values
    lows   = feat["low"].values
    score_up_arr = feat["score_up"].values
    score_dn_arr = feat["score_dn"].values

    # For each (side, score) bucket, pool win/total
    buckets = {("UP", s): {"n": 0, "w": 0} for s in range(5)}
    buckets.update({("DN", s): {"n": 0, "w": 0} for s in range(5)})

    for i in range(n - OUTCOME_BARS):
        s_up = int(score_up_arr[i])
        s_dn = int(score_dn_arr[i])
        if s_up <= 0 and s_dn <= 0:
            continue
        entry = closes[i]
        if entry <= 0 or not np.isfinite(entry):
            continue
        win_high = float(np.max(highs[i + 1 : i + 1 + OUTCOME_BARS]))
        win_low  = float(np.min(lows[i + 1 : i + 1 + OUTCOME_BARS]))
        # Up direction
        if s_up > 0:
            hit = (win_high - entry) / entry >= threshold_pct
            buckets[("UP", s_up)]["n"] += 1
            if hit:
                buckets[("UP", s_up)]["w"] += 1
        # Down direction
        if s_dn > 0:
            hit = (entry - win_low) / entry >= threshold_pct
            buckets[("DN", s_dn)]["n"] += 1
            if hit:
                buckets[("DN", s_dn)]["w"] += 1
    return buckets


def main():
    t0 = time.time()
    print(f"[1/4] Loading intraday data…")
    all_data = {tk: _load(tk) for tk in PEER_POOL}
    all_data = {k: v for k, v in all_data.items() if v is not None and len(v) >= 200}
    print(f"  {len(all_data)} tickers loaded ({time.time()-t0:.1f}s)")

    # Pre-compute features + per-ticker ToD distributions + VFA
    print(f"[2/4] Pre-computing features + peer VFA…")
    all_feats = {}
    for tk, df in all_data.items():
        feat = compute_features(df)
        tod = build_tod_dist(feat)
        feat = compute_peer_vfa(feat, tod)
        all_feats[tk] = feat
    print(f"  done ({time.time()-t0:.1f}s)")

    # Run QSIT for each target × each VCC percentile × each outcome threshold
    print(f"[3/4] Running QSIT score evaluation across grid…")
    all_buckets = defaultdict(lambda: {"n": 0, "w": 0})
    target_universe = [t for t in TARGET_TICKERS if t in all_data]

    target_results = {}
    for target in target_universe:
        peers = find_peers(target, all_data, k=CACB_TOP_K)
        peer_feats_subset = {p: all_feats[p] for p in peers if p in all_feats}
        if len(peer_feats_subset) < 1:
            continue
        target_results[target] = {"peers": peers, "by_threshold": {}}
        for vcc_pct in VCC_PERCENTILES:
            for thr in OUTCOME_THRESHOLDS:
                feat = compute_qsit_score(target, all_feats[target].copy(),
                                          list(peer_feats_subset.keys()),
                                          peer_feats_subset, vcc_pct)
                buckets = evaluate(feat, thr)
                key = (vcc_pct, thr)
                if key not in target_results[target]["by_threshold"]:
                    target_results[target]["by_threshold"][key] = {}
                for (side, s), v in buckets.items():
                    target_results[target]["by_threshold"][key][f"{side}_{s}"] = v
                    pkey = (vcc_pct, thr, side, s)
                    all_buckets[pkey]["n"] += v["n"]
                    all_buckets[pkey]["w"] += v["w"]
    print(f"  done ({time.time()-t0:.1f}s)")

    # Report pooled results
    print()
    print(f"[4/4] Pooled results across {len(target_universe)} targets")
    print()

    # For each (vcc_pct, threshold), show win rate by score
    for vcc_pct in VCC_PERCENTILES:
        print(f"=== VCC percentile = {vcc_pct*100:.0f}%  ===")
        print(f"{'thresh':>7}  {'side':>4}  "
              f"{'s=1 n / win%':>15} "
              f"{'s=2 n / win%':>15} "
              f"{'s=3 n / win%':>15} "
              f"{'s=4 n / win%':>15}")
        for thr in OUTCOME_THRESHOLDS:
            for side in ["UP", "DN"]:
                cells = []
                for s in [1, 2, 3, 4]:
                    v = all_buckets[(vcc_pct, thr, side, s)]
                    if v["n"] > 0:
                        wr = v["w"] / v["n"] * 100
                        cells.append(f"{v['n']:>4} / {wr:>4.1f}%")
                    else:
                        cells.append("-")
                cells_padded = [c.rjust(15) for c in cells]
                print(f"  {thr*100:>4.1f}%  {side:>4}  "
                      f"{cells_padded[0]} {cells_padded[1]} "
                      f"{cells_padded[2]} {cells_padded[3]}")
        print()

    # Save
    out_path = os.path.join(_HERE, "results", "qsit_intraday_v2.json")
    # Convert all_buckets keys to strings for JSON
    serializable_buckets = {
        f"{k[0]:.2f}_{k[1]:.3f}_{k[2]}_{k[3]}": v
        for k, v in all_buckets.items()
    }
    with open(out_path, "w") as fh:
        json.dump({
            "config": {
                "vcc_percentiles": VCC_PERCENTILES,
                "outcome_thresholds": OUTCOME_THRESHOLDS,
                "outcome_bars": OUTCOME_BARS,
                "window_bars": WINDOW_BARS,
                "vfa_vol_ratio_min": VFA_VOL_RATIO_MIN,
                "vfa_cpir_high": VFA_CPIR_HIGH,
                "vfa_cpir_low": VFA_CPIR_LOW,
                "cacb_agree_min": CACB_AGREE_MIN,
                "cacb_top_k": CACB_TOP_K,
                "todp_windows_utc": TODP_WINDOWS_UTC,
                "target_tickers": target_universe,
            },
            "pooled_buckets": serializable_buckets,
        }, fh, separators=(",", ":"), default=str)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    sys.exit(main())
