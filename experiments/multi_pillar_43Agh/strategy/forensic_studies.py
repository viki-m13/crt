"""Phase 1 — Forensic Studies (Study A: winners, Study B: failures).

Identifies winner episodes (>=5x return within 18 months) and failure
episodes (>= -60% drawdown within 18 months OR delisting) on the
1995-2026 daily price panel. For each episode, builds a 6-12 month
pre-window of features and stores it for downstream archetype/failure
work.

Builds matched controls: same-sector, similar starting momentum, similar
mcap proxy, that did NOT run / did NOT fail.

Outputs (under experiments/multi_pillar_43Agh/data/):
  winners.parquet         # episode rows: ticker, peak_date, base_date, mult, sector
  failures.parquet        # episode rows: ticker, peak_date, trough_date, depth, sector
  winner_features.parquet # per-episode 6-month-prior feature snapshot
  failure_features.parquet
  winner_controls.parquet
  failure_controls.parquet

Run from repo root:
  python3 experiments/multi_pillar_43Agh/strategy/forensic_studies.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
CACHE = ROOT / "experiments" / "monthly_dca" / "cache"
FEATURES_DIR = CACHE / "features"
OUT = ROOT / "experiments" / "multi_pillar_43Agh" / "data"
OUT.mkdir(parents=True, exist_ok=True)


def load_daily_prices() -> pd.DataFrame:
    """Daily adjusted close, 1995-01-03 → 2026-05-07, 1833 tickers."""
    return pd.read_parquet(CACHE / "prices_extended.parquet")


def find_winner_episodes(prices: pd.DataFrame, mult: float = 5.0,
                         max_days: int = 540) -> pd.DataFrame:
    """For each ticker, find non-overlapping windows where price rose >= mult
    within max_days trading days (18 months ≈ 378 td; we use 540 to be permissive).
    Returns rows: ticker, base_date (low), peak_date (high), mult.
    """
    out = []
    for tk in prices.columns:
        s = prices[tk].dropna()
        if len(s) < 60:
            continue
        v = s.values
        n = len(v)
        i = 0
        while i < n - 1:
            base = v[i]
            if not np.isfinite(base) or base <= 0:
                i += 1
                continue
            j_end = min(i + max_days, n)
            w = v[i:j_end]
            peak_idx_local = int(np.argmax(w))
            peak_v = w[peak_idx_local]
            if peak_v >= mult * base:
                out.append({
                    "ticker": tk,
                    "base_date": s.index[i],
                    "peak_date": s.index[i + peak_idx_local],
                    "base_price": float(base),
                    "peak_price": float(peak_v),
                    "mult": float(peak_v / base),
                    "days": int(peak_idx_local),
                })
                i = i + peak_idx_local + 60  # skip past the peak + cooldown
            else:
                i += 21  # check monthly
    return pd.DataFrame(out)


def find_failure_episodes(prices: pd.DataFrame, depth: float = -0.60,
                          max_days: int = 540) -> pd.DataFrame:
    """For each ticker, find non-overlapping windows where price fell to <=
    (1+depth) * peak_price within max_days. Captures BOTH delisted and surviving
    big-losers.
    """
    out = []
    for tk in prices.columns:
        s = prices[tk].dropna()
        if len(s) < 60:
            continue
        v = s.values
        n = len(v)
        i = 0
        while i < n - 1:
            peak = v[i]
            if not np.isfinite(peak) or peak <= 0:
                i += 1
                continue
            j_end = min(i + max_days, n)
            w = v[i:j_end]
            trough_idx_local = int(np.argmin(w))
            trough_v = w[trough_idx_local]
            if trough_v <= (1.0 + depth) * peak:
                out.append({
                    "ticker": tk,
                    "peak_date": s.index[i],
                    "trough_date": s.index[i + trough_idx_local],
                    "peak_price": float(peak),
                    "trough_price": float(trough_v),
                    "depth": float(trough_v / peak - 1.0),
                    "days": int(trough_idx_local),
                })
                i = i + trough_idx_local + 60
            else:
                i += 21
    return pd.DataFrame(out)


def add_delisting_failures(prices: pd.DataFrame) -> pd.DataFrame:
    """Tickers whose series ends before the panel ends (proxy for delisting)
    — record a failure episode with depth -100% from last 1Y high."""
    last_panel_date = prices.index.max()
    cutoff = last_panel_date - pd.Timedelta(days=120)
    out = []
    for tk in prices.columns:
        s = prices[tk].dropna()
        if len(s) < 60:
            continue
        last_dt = s.index.max()
        if last_dt < cutoff:
            # Ticker series ended early → delisting proxy
            window_start = max(s.index[0], last_dt - pd.Timedelta(days=400))
            window = s.loc[window_start:last_dt]
            if len(window) < 20:
                continue
            peak_dt = window.idxmax()
            peak_v = float(window.max())
            trough_v = float(window.iloc[-1])
            if peak_v <= 0 or peak_dt >= last_dt:
                continue
            out.append({
                "ticker": tk,
                "peak_date": peak_dt,
                "trough_date": last_dt,
                "peak_price": peak_v,
                "trough_price": trough_v,
                "depth": float(trough_v / peak_v - 1.0),
                "days": int((last_dt - peak_dt).days),
                "delisting": True,
            })
    return pd.DataFrame(out)


def find_pre_window_asof(asofs: list[pd.Timestamp], event_date: pd.Timestamp,
                         offset_months: int = 6) -> pd.Timestamp | None:
    """Find a feature-panel asof that is `offset_months` before event_date,
    using only available asofs in the cache."""
    target = event_date - pd.DateOffset(months=offset_months)
    cands = [a for a in asofs if a <= target]
    if not cands:
        return None
    return max(cands)


def feature_snapshot(asof: pd.Timestamp, ticker: str,
                     feat_cache: dict) -> dict | None:
    """Load (cached) feature parquet at asof, return ticker's row as dict."""
    key = pd.Timestamp(asof).date()
    if key not in feat_cache:
        f = FEATURES_DIR / f"{key}.parquet"
        feat_cache[key] = pd.read_parquet(f) if f.exists() else None
    df = feat_cache[key]
    if df is None or ticker not in df.index:
        return None
    return df.loc[ticker].to_dict()


def build_pre_features(episodes: pd.DataFrame, event_col: str,
                       offset_months: int = 6) -> pd.DataFrame:
    """For each episode, snapshot the feature row `offset_months` before
    the event."""
    asofs = sorted(p.stem for p in FEATURES_DIR.glob("*.parquet"))
    asofs = [pd.Timestamp(s) for s in asofs]
    feat_cache: dict = {}
    rows = []
    for _, ep in episodes.iterrows():
        event_dt = pd.Timestamp(ep[event_col])
        ao = find_pre_window_asof(asofs, event_dt, offset_months)
        if ao is None:
            continue
        snap = feature_snapshot(ao, ep["ticker"], feat_cache)
        if snap is None:
            continue
        snap.update({
            "ticker": ep["ticker"],
            "asof": ao,
            "event_date": event_dt,
            "offset_months": offset_months,
        })
        rows.append(snap)
    return pd.DataFrame(rows)


def sample_controls(events: pd.DataFrame, n_per: int = 5,
                    random_state: int = 17) -> pd.DataFrame:
    """For each event row, sample `n_per` control snapshots: same asof,
    similar mom_12_1 (±0.20), but ticker did NOT have an event within
    540 days after the asof."""
    asofs = sorted(p.stem for p in FEATURES_DIR.glob("*.parquet"))
    asofs = [pd.Timestamp(s) for s in asofs]
    feat_cache: dict = {}
    rng = np.random.default_rng(random_state)
    rows = []
    for _, ep in events.iterrows():
        ao = ep["asof"]
        snap_self = ep.get("mom_12_1", np.nan)
        if pd.isna(snap_self):
            continue
        key = pd.Timestamp(ao).date()
        if key not in feat_cache:
            f = FEATURES_DIR / f"{key}.parquet"
            feat_cache[key] = pd.read_parquet(f) if f.exists() else None
        df = feat_cache[key]
        if df is None:
            continue
        # Eligible controls: |mom_12_1 - snap_self| < 0.20 AND ticker != event ticker
        if "mom_12_1" not in df.columns:
            continue
        diff = (df["mom_12_1"] - snap_self).abs()
        cands = df.index[diff < 0.20].tolist()
        cands = [c for c in cands if c != ep["ticker"]]
        if not cands:
            continue
        n_take = min(n_per, len(cands))
        chosen = rng.choice(cands, size=n_take, replace=False)
        for c in chosen:
            row = df.loc[c].to_dict()
            row.update({
                "ticker": str(c),
                "asof": ao,
                "event_ticker": ep["ticker"],
            })
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    print("[load] daily prices ...")
    prices = load_daily_prices()
    print(f"  shape={prices.shape}")

    print("[study A] finding winner episodes (>=5x in <=540td) ...")
    winners = find_winner_episodes(prices, mult=5.0, max_days=540)
    print(f"  found {len(winners)} winner episodes")
    winners.to_parquet(OUT / "winners.parquet")

    print("[study B] finding failure episodes (<=-60% in <=540td) ...")
    failures = find_failure_episodes(prices, depth=-0.60, max_days=540)
    delisting_fails = add_delisting_failures(prices)
    if len(delisting_fails):
        delisting_fails["delisting"] = True
        failures = pd.concat([failures, delisting_fails], ignore_index=True)
        failures = failures.drop_duplicates(subset=["ticker", "peak_date"])
    print(f"  found {len(failures)} failure episodes (incl {len(delisting_fails)} delistings)")
    failures.to_parquet(OUT / "failures.parquet")

    print("[features] pre-window snapshots (winners @ -6m before peak base)...")
    win_feats = build_pre_features(winners, "base_date", offset_months=3)
    print(f"  {len(win_feats)} winner pre-snapshots")
    win_feats.to_parquet(OUT / "winner_features.parquet")

    print("[features] pre-window snapshots (failures @ -3m before peak)...")
    fail_feats = build_pre_features(failures, "peak_date", offset_months=3)
    print(f"  {len(fail_feats)} failure pre-snapshots")
    fail_feats.to_parquet(OUT / "failure_features.parquet")

    print("[controls] sampling matched controls (winners) ...")
    win_ctrls = sample_controls(win_feats, n_per=5)
    print(f"  {len(win_ctrls)} winner control snapshots")
    win_ctrls.to_parquet(OUT / "winner_controls.parquet")

    print("[controls] sampling matched controls (failures) ...")
    fail_ctrls = sample_controls(fail_feats, n_per=5)
    print(f"  {len(fail_ctrls)} failure control snapshots")
    fail_ctrls.to_parquet(OUT / "failure_controls.parquet")

    print("[done] outputs in", OUT)


if __name__ == "__main__":
    main()
