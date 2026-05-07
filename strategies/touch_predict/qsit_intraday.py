"""QSIT — Quad-Stack Intraday Trigger.

A novel intraday strategy combining four orthogonal, proprietary
signals. ALL must fire together for a trade. Each signal addresses
a different microstructure phenomenon:

  S1. VOLATILITY COMPRESSION COIL (VCC).
      Compute 15-bar (30-min) rolling realized vol of log-returns.
      Compare to the *time-of-day-conditioned* historical
      distribution: for the same 30-min window of the trading day,
      what's been the vol over the last 30 sessions? Fires when
      current vol < 10th percentile.
      Theory: tight ranges before institutional flow → coiled spring.

  S2. VOLUME FOOTPRINT ASYMMETRY (VFA).
      Compute 15-bar volume ratio vs. time-of-day baseline.
      Compute 15-bar close-position-in-range (CPIR):
         CPIR = (last_close − period_low) / (period_high − period_low)
      Fires UP if vol_ratio > 1.5 AND CPIR > 0.75.
      Fires DOWN if vol_ratio > 1.5 AND CPIR < 0.25.
      Theory: above-avg volume PLUS extreme close = absorption.

  S3. CROSS-ASSET COORDINATION BURST (CACB).
      Identify top-3 correlated peers (using 5-min returns over
      lookback). Fire UP if ≥2 peers show same-direction VFA
      in same 30-min window; FIRE DOWN if ≥2 peers DOWN-VFA.
      Theory: sector rotation is more predictive than single name.

  S4. TIME-OF-DAY PROXIMITY (TODP).
      Trade only in 3 institutional-flow windows:
         09:35 – 10:30 ET  (post-open drive)
         13:30 – 14:30 ET  (afternoon trend)
         15:00 – 15:55 ET  (closing imbalance)
      Theory: signals outside these windows have lower follow-
      through because institutional desks aren't engaged.

ALL 4 must fire (with same direction on S2 & S3). Trade is "buy
direction for next 30 min." Profitability test: did ticker move
≥ 0.5% in fired direction within 60 min after fire?

Output: results/qsit_intraday.json
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_HERE, "data", "intraday")


# ---- config ------------------------------------------------------

# Each bar is 2 minutes; 15 bars = 30 minutes
WINDOW_BARS = 15

# Sliding-window distribution for VCC: how many sessions back to
# build the time-of-day percentile for "normal" vol at this time
TOD_LOOKBACK_DAYS = 20

# VCC compression threshold: vol below this percentile of recent
VCC_PERCENTILE = 0.10

# VFA thresholds
VFA_VOL_RATIO_MIN = 1.5
VFA_CPIR_HIGH = 0.75
VFA_CPIR_LOW = 0.25

# CACB: number of peers to consider, agreement count needed
CACB_TOP_K = 3
CACB_AGREE_MIN = 2

# TODP: trading windows (UTC times — yfinance returns UTC)
# Market open = 13:30 UTC, close = 20:00 UTC
TODP_WINDOWS_UTC = [
    ("13:35", "14:30"),  # 09:35 - 10:30 ET (post-open)
    ("17:30", "18:30"),  # 13:30 - 14:30 ET (afternoon)
    ("19:00", "19:55"),  # 15:00 - 15:55 ET (closing)
]

# Forward outcome: did ticker move ≥0.5% in predicted direction
# within 60 minutes (30 bars)?
OUTCOME_BARS = 30
OUTCOME_THRESHOLD_PCT = 0.005      # 0.5% move

# Universe — tickers we'll generate trades on
TARGET_TICKERS = ["QCOM", "INTC", "SLV", "GLD", "AVGO", "AMD",
                  "NVDA", "XLK", "SMH", "GDX", "PLTR", "XLE", "USO"]

# Tickers we'll use as potential peers (any of these can be a peer
# of any target ticker)
PEER_POOL = TARGET_TICKERS + ["SPY", "QQQ", "IWM", "TLT"]


# ---- data load ---------------------------------------------------

def _load_one(ticker: str) -> pd.DataFrame | None:
    path = os.path.join(DATA_DIR, f"{ticker}.json")
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        b = json.load(fh)
    df = pd.DataFrame({
        "open":   b["open"],
        "high":   b["high"],
        "low":    b["low"],
        "close":  b["close"],
        "volume": b["volume"],
    }, index=pd.to_datetime(b["datetimes"], utc=True))
    df = df.sort_index()
    return df


def load_all() -> dict[str, pd.DataFrame]:
    out = {}
    for tk in PEER_POOL:
        d = _load_one(tk)
        if d is not None and len(d) >= 200:
            out[tk] = d
    return out


# ---- correlations & peer discovery ------------------------------

def find_peers(target: str, all_data: dict[str, pd.DataFrame],
               top_k: int = CACB_TOP_K) -> list[str]:
    """Return top-K correlated tickers based on 5-min log-returns
    over the full available history."""
    base = all_data[target]
    base_log = np.log(base["close"]).diff().dropna()
    # Resample to 5-min for correlation
    base_5 = base_log.resample("5min").sum().dropna()
    corrs = []
    for tk, df in all_data.items():
        if tk == target:
            continue
        peer_log = np.log(df["close"]).diff().dropna().resample("5min").sum().dropna()
        joined = pd.concat([base_5, peer_log], axis=1, join="inner").dropna()
        if len(joined) < 100:
            continue
        c = joined.iloc[:, 0].corr(joined.iloc[:, 1])
        if not np.isnan(c):
            corrs.append((tk, c))
    corrs.sort(key=lambda x: -abs(x[1]))   # take the strongest correlations (positive or negative)
    return [tk for tk, _ in corrs[:top_k]]


# ---- per-ticker feature engineering -----------------------------

def _rolling_realized_vol(log_ret: np.ndarray, window: int) -> np.ndarray:
    """Rolling stdev of log returns over `window` bars."""
    n = len(log_ret)
    out = np.full(n, np.nan)
    if n < window + 1:
        return out
    csum = np.concatenate(([0.0], np.cumsum(log_ret * log_ret)))
    var_w = (csum[window:] - csum[:-window]) / window
    out[window - 1:] = np.sqrt(np.maximum(var_w, 0.0))
    return out


def _time_key(ts: pd.Timestamp) -> str:
    """Bucket time-of-day into 30-minute slots for ToD baselining."""
    minute_slot = (ts.hour * 60 + ts.minute) // 30
    return f"{minute_slot:02d}"


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute QSIT features on a single ticker's intraday bars."""
    f = df.copy()
    log_ret = np.log(f["close"]).diff().fillna(0.0).values
    rv_15 = _rolling_realized_vol(log_ret, WINDOW_BARS)
    f["rv_15"] = rv_15

    # Time-of-day key
    f["tod_key"] = f.index.map(_time_key)
    f["session_date"] = f.index.date

    # Rolling 15-bar high / low / volume sum
    f["high_15"] = f["high"].rolling(WINDOW_BARS).max()
    f["low_15"]  = f["low"].rolling(WINDOW_BARS).min()
    f["vol_15"]  = f["volume"].rolling(WINDOW_BARS).sum()

    # CPIR: where did 15-bar close land within the period's range?
    rng = f["high_15"] - f["low_15"]
    rng_safe = np.where(rng > 1e-9, rng, np.nan)
    f["cpir"] = (f["close"] - f["low_15"]) / rng_safe

    return f


def build_tod_distributions(features: pd.DataFrame) -> dict[str, dict[str, np.ndarray]]:
    """For each ToD slot, build distributions of {rv_15, vol_15} from
    the rolling sessions BEFORE the current session. Returns dict
    {tod_key: {'rv': sorted_array, 'vol': sorted_array}}.
    NOTE: This is a simplified static baseline (full history); a real
    walk-forward would refresh per session. For 60d data, statics are
    fine."""
    out: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"rv": [], "vol": []})
    for ts, row in features.iterrows():
        if not np.isfinite(row["rv_15"]) or not np.isfinite(row["vol_15"]):
            continue
        tod = row["tod_key"]
        out[tod]["rv"].append(float(row["rv_15"]))
        out[tod]["vol"].append(float(row["vol_15"]))
    # Convert to sorted np arrays for percentile lookups
    finalized = {}
    for tod, d in out.items():
        finalized[tod] = {
            "rv": np.sort(np.array(d["rv"])),
            "vol": np.sort(np.array(d["vol"])),
        }
    return finalized


def percentile_in(value: float, sorted_arr: np.ndarray) -> float:
    """Where does `value` sit in the sorted array? Returns 0..1."""
    if len(sorted_arr) == 0:
        return float("nan")
    idx = np.searchsorted(sorted_arr, value)
    return idx / len(sorted_arr)


# ---- per-ticker QSIT signals ------------------------------------

def compute_qsit_signals(target: str, target_df: pd.DataFrame,
                         peer_features: dict[str, pd.DataFrame],
                         peer_tod: dict[str, dict[str, dict[str, np.ndarray]]]
                         ) -> pd.DataFrame:
    """Run the four QSIT filters on `target` using `peer_features`
    for cross-asset confirmation. Returns the target's feature frame
    enriched with signal columns."""
    feat = compute_features(target_df)
    target_tod = build_tod_distributions(feat)

    # S1 — VCC: vol percentile vs ToD baseline
    vcc = []
    for ts, row in feat.iterrows():
        if not np.isfinite(row["rv_15"]):
            vcc.append(False); continue
        tod = row["tod_key"]
        if tod not in target_tod:
            vcc.append(False); continue
        p = percentile_in(row["rv_15"], target_tod[tod]["rv"])
        vcc.append(bool(p < VCC_PERCENTILE))
    feat["VCC"] = vcc

    # S2 — VFA: volume ratio AND CPIR
    vfa_up = []
    vfa_dn = []
    for ts, row in feat.iterrows():
        if not np.isfinite(row["vol_15"]) or not np.isfinite(row["cpir"]):
            vfa_up.append(False); vfa_dn.append(False); continue
        tod = row["tod_key"]
        baseline = target_tod.get(tod, {}).get("vol")
        if baseline is None or len(baseline) == 0:
            vfa_up.append(False); vfa_dn.append(False); continue
        median_vol = np.median(baseline)
        if median_vol <= 0:
            vfa_up.append(False); vfa_dn.append(False); continue
        vol_ratio = row["vol_15"] / median_vol
        vfa_up.append(bool(vol_ratio > VFA_VOL_RATIO_MIN and row["cpir"] > VFA_CPIR_HIGH))
        vfa_dn.append(bool(vol_ratio > VFA_VOL_RATIO_MIN and row["cpir"] < VFA_CPIR_LOW))
    feat["VFA_UP"] = vfa_up
    feat["VFA_DN"] = vfa_dn

    # S3 — CACB: cross-asset coordination
    # Look at peer VFA: in the same 2-min bucket, did ≥CACB_AGREE_MIN
    # peers register VFA_UP/DOWN?
    peer_vfa_up = pd.DataFrame(
        {p: peer_features[p]["VFA_UP"].astype(int) for p in peer_features})
    peer_vfa_dn = pd.DataFrame(
        {p: peer_features[p]["VFA_DN"].astype(int) for p in peer_features})
    # Align to target index
    aligned_up = peer_vfa_up.reindex(feat.index, method="ffill", limit=2).fillna(0)
    aligned_dn = peer_vfa_dn.reindex(feat.index, method="ffill", limit=2).fillna(0)
    n_peers_up = aligned_up.sum(axis=1)
    n_peers_dn = aligned_dn.sum(axis=1)
    feat["CACB_UP"] = n_peers_up >= CACB_AGREE_MIN
    feat["CACB_DN"] = n_peers_dn >= CACB_AGREE_MIN

    # S4 — TODP: time-of-day window
    def _in_todp(ts: pd.Timestamp) -> bool:
        hm = ts.strftime("%H:%M")
        for start, end in TODP_WINDOWS_UTC:
            if start <= hm <= end:
                return True
        return False
    feat["TODP"] = feat.index.map(_in_todp)

    # Composite trigger: all 4 fire (with consistent direction on S2/S3)
    feat["FIRE_UP"] = feat["VCC"] & feat["VFA_UP"] & feat["CACB_UP"] & feat["TODP"]
    feat["FIRE_DN"] = feat["VCC"] & feat["VFA_DN"] & feat["CACB_DN"] & feat["TODP"]
    return feat


# ---- outcome resolution -----------------------------------------

def evaluate_fires(feat: pd.DataFrame) -> dict:
    """For every fire bar, did target hit ≥0.5% directional move
    within OUTCOME_BARS bars? Report win rate."""
    n = len(feat)
    closes = feat["close"].values
    highs  = feat["high"].values
    lows   = feat["low"].values
    fires_up = np.where(feat["FIRE_UP"].values)[0]
    fires_dn = np.where(feat["FIRE_DN"].values)[0]

    # Drop fires too close to end (no forward window)
    fires_up = fires_up[fires_up + OUTCOME_BARS < n]
    fires_dn = fires_dn[fires_dn + OUTCOME_BARS < n]

    def _resolve(idxs, side: str):
        rows = []
        for i in idxs:
            entry = closes[i]
            window_high = float(np.max(highs[i + 1 : i + 1 + OUTCOME_BARS]))
            window_low  = float(np.min(lows[i + 1 : i + 1 + OUTCOME_BARS]))
            if side == "UP":
                hit = (window_high - entry) / entry >= OUTCOME_THRESHOLD_PCT
                max_excursion = (window_high - entry) / entry
                worst_excursion = (window_low - entry) / entry
            else:
                hit = (entry - window_low) / entry >= OUTCOME_THRESHOLD_PCT
                max_excursion = (entry - window_low) / entry
                worst_excursion = (entry - window_high) / entry
            rows.append({
                "i": int(i), "side": side, "entry": entry,
                "hit": bool(hit),
                "max_excursion": float(max_excursion),
                "worst_excursion": float(worst_excursion),
                "ts": str(feat.index[i]),
            })
        return rows

    up_rows = _resolve(fires_up, "UP")
    dn_rows = _resolve(fires_dn, "DN")

    n_up = len(up_rows); n_dn = len(dn_rows)
    n_total = n_up + n_dn
    wins_up = sum(1 for r in up_rows if r["hit"])
    wins_dn = sum(1 for r in dn_rows if r["hit"])
    win_total = wins_up + wins_dn
    summary = {
        "n_fires_total": n_total,
        "n_fires_up": n_up,
        "n_fires_dn": n_dn,
        "win_rate_total_pct": (win_total / n_total * 100) if n_total else 0.0,
        "win_rate_up_pct": (wins_up / n_up * 100) if n_up else 0.0,
        "win_rate_dn_pct": (wins_dn / n_dn * 100) if n_dn else 0.0,
        "avg_max_excursion_up_pct": (
            np.mean([r["max_excursion"] for r in up_rows]) * 100 if up_rows else 0.0),
        "avg_max_excursion_dn_pct": (
            np.mean([r["max_excursion"] for r in dn_rows]) * 100 if dn_rows else 0.0),
        "avg_worst_excursion_up_pct": (
            np.mean([r["worst_excursion"] for r in up_rows]) * 100 if up_rows else 0.0),
        "avg_worst_excursion_dn_pct": (
            np.mean([r["worst_excursion"] for r in dn_rows]) * 100 if dn_rows else 0.0),
        "fires_up_rows": up_rows[:30],   # sample
        "fires_dn_rows": dn_rows[:30],
    }
    return summary


# ---- main --------------------------------------------------------

def main() -> int:
    t0 = time.time()
    print(f"[1/4] Loading intraday data…")
    all_data = load_all()
    print(f"  {len(all_data)} tickers loaded ({time.time()-t0:.1f}s)")

    print(f"[2/4] Pre-computing features for all peers…")
    all_feats = {}
    for tk, df in all_data.items():
        all_feats[tk] = compute_features(df)
    print(f"  features ready ({time.time()-t0:.1f}s)")

    # Pre-compute peer VFA on each peer (needed for CACB)
    print(f"[3/4] Pre-computing peer VFA for cross-asset coordination…")
    for tk, feat in all_feats.items():
        target_tod = build_tod_distributions(feat)
        vfa_up_list = []; vfa_dn_list = []
        for ts, row in feat.iterrows():
            if not np.isfinite(row["vol_15"]) or not np.isfinite(row["cpir"]):
                vfa_up_list.append(False); vfa_dn_list.append(False); continue
            tod = row["tod_key"]
            base = target_tod.get(tod, {}).get("vol")
            if base is None or len(base) == 0:
                vfa_up_list.append(False); vfa_dn_list.append(False); continue
            med = np.median(base)
            if med <= 0:
                vfa_up_list.append(False); vfa_dn_list.append(False); continue
            vr = row["vol_15"] / med
            vfa_up_list.append(bool(vr > VFA_VOL_RATIO_MIN and row["cpir"] > VFA_CPIR_HIGH))
            vfa_dn_list.append(bool(vr > VFA_VOL_RATIO_MIN and row["cpir"] < VFA_CPIR_LOW))
        feat["VFA_UP"] = vfa_up_list
        feat["VFA_DN"] = vfa_dn_list
    print(f"  peer VFA ready ({time.time()-t0:.1f}s)")

    # For each target, identify peers and run QSIT
    print(f"[4/4] Running QSIT on each target ticker…")
    results = []
    target_universe = [t for t in TARGET_TICKERS if t in all_data]
    peer_tod_cache = {}
    for target in target_universe:
        peers = find_peers(target, all_data, top_k=CACB_TOP_K)
        peer_features = {p: all_feats[p] for p in peers if p in all_feats}
        if len(peer_features) < CACB_AGREE_MIN:
            continue
        # Compose peer subset for CACB
        target_df = all_data[target]
        feat = compute_qsit_signals(target, target_df, peer_features, peer_tod_cache)
        outcome = evaluate_fires(feat)
        outcome["target"] = target
        outcome["peers"] = peers
        results.append(outcome)
        print(f"  {target:<6} peers={peers}  "
              f"fires={outcome['n_fires_total']:>3}  "
              f"win%={outcome['win_rate_total_pct']:>5.1f}  "
              f"up={outcome['n_fires_up']}/{outcome['win_rate_up_pct']:.0f}%  "
              f"dn={outcome['n_fires_dn']}/{outcome['win_rate_dn_pct']:.0f}%")

    # Aggregate
    n_total = sum(r["n_fires_total"] for r in results)
    win_total = sum(int(r["n_fires_total"] * r["win_rate_total_pct"] / 100) for r in results)
    print()
    print(f"=== POOLED ACROSS ALL TARGETS ===")
    print(f"Total fires:    {n_total}")
    print(f"Pooled wins:    {win_total}")
    if n_total > 0:
        print(f"Pooled win-rate: {win_total/n_total*100:.1f}%")

    # Save
    out_path = os.path.join(_HERE, "results", "qsit_intraday.json")
    with open(out_path, "w") as fh:
        json.dump({
            "config": {
                "window_bars": WINDOW_BARS,
                "vcc_percentile": VCC_PERCENTILE,
                "vfa_vol_ratio_min": VFA_VOL_RATIO_MIN,
                "vfa_cpir_high": VFA_CPIR_HIGH,
                "vfa_cpir_low": VFA_CPIR_LOW,
                "cacb_top_k": CACB_TOP_K,
                "cacb_agree_min": CACB_AGREE_MIN,
                "todp_windows_utc": TODP_WINDOWS_UTC,
                "outcome_bars": OUTCOME_BARS,
                "outcome_threshold_pct": OUTCOME_THRESHOLD_PCT,
            },
            "results": results,
            "pooled_n_fires": n_total,
            "pooled_win_rate_pct": (win_total/n_total*100 if n_total > 0 else 0),
        }, fh, separators=(",", ":"), default=str)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    sys.exit(main())
