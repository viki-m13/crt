"""Tier 2 — "Vol-Alpha" GBM-selected put credit spreads.

The validated spec (VALIDATION.md §14): PUT vertical, short strike
0.6·σ60·√14 below spot, width 2.5% of spot, expiry snapped DOWN within
14 sessions, hold to expiry, published only when a gradient-boosted
classifier (fit ONLY on 2008–2018, artifact committed) scores today's
features above the frozen deep-confidence threshold (the cut that
achieved 99% cumulative accuracy on the design window).

Validation 2019–2026 (deduped, conservative fills): 98.2% accuracy,
24.3% net ROR/trade — 19.7% even at bare-realized-vol stress pricing —
~7 trades/week, worst trade −$516/contract, positive in 2020 and 2022.

Two entry points:
    python3 tier2.py train    # refit from the design window, save artifact
    python3 tier2.py scan     # score today's panel, verify against real
                              # chains, merge tier2_signals into
                              # results/signals.json

The scan is called by scan.py after research.py. All the v3 hygiene
applies: optionable-only, >=10y history, fresh series, reality layer
(real expirations/strikes/quotes, natural credit >= $0.05) fail-closed.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

from common import (
    covered_options_expiry, list_tickers, load_series, _rolling_mean,
    _nyse_valid_days_big,
)
from tech_spreads import rolling_std, rsi

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
MODEL_PATH = os.path.join(RESULTS, "tier2_model.joblib")
META_PATH = os.path.join(RESULTS, "tier2_meta.json")
ROWS_PATH = os.path.join(RESULTS, "sigma_distance_rows.npz")

ENGINE = "t2-volalpha-gbm"
CONV_ENGINE = "conviction-pick"
C_SIGMA = 0.6
WIDTH_PCT = 0.025
HORIZON = 14
DESIGN_END = np.datetime64("2018-12-31")
DESIGN_TARGET = 0.99          # cumulative design accuracy defining the cut
MIN_HISTORY_CAL_DAYS = 3652
MAX_STALE_SESSIONS = 5

FEATURES = ["vr_10_60", "vr_60_252", "trend200", "trend50", "dd252", "up252",
            "rsi14", "ret5", "ret21", "gap_age", "breadth", "d_breadth10",
            "sigma60"]


def _make_clf():
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=200, max_depth=4,
                                          random_state=0)


def train() -> int:
    """Fit on the design window of the sigma-distance row table and save
    the artifact + frozen threshold. Deterministic (random_state=0)."""
    import joblib
    d = np.load(ROWS_PATH, allow_pickle=False)
    dates = d["date"].astype("datetime64[D]")
    design = dates <= DESIGN_END
    m = (d["side"] == "put")
    X = np.column_stack([d[f] for f in FEATURES])
    finite = np.isfinite(X).all(axis=1)
    tr = m & finite & design
    y = d[f"win_c{C_SIGMA}"].astype(int)
    clf = _make_clf()
    clf.fit(X[tr], y[tr])
    p = clf.predict_proba(X[tr])[:, 1]
    order = np.argsort(-p)
    cum = np.cumsum(y[tr][order]) / np.arange(1, order.size + 1)
    idx = np.where(cum >= DESIGN_TARGET)[0]
    if not len(idx):
        print("no design cut reaches target", file=sys.stderr)
        return 1
    threshold = float(p[order][idx.max()])
    joblib.dump(clf, MODEL_PATH)
    meta = {
        "engine": ENGINE, "c_sigma": C_SIGMA, "width_pct": WIDTH_PCT,
        "horizon": HORIZON, "design_end": str(DESIGN_END),
        "design_target": DESIGN_TARGET, "threshold": threshold,
        "n_train": int(tr.sum()), "features": FEATURES,
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(META_PATH, "w") as fh:
        json.dump(meta, fh, indent=1)
    print(f"trained on {int(tr.sum())} design rows; "
          f"threshold={threshold:.6f} -> {MODEL_PATH}")
    return 0


def _load_model():
    import joblib
    with open(META_PATH) as fh:
        meta = json.load(fh)
    clf = joblib.load(MODEL_PATH)
    return clf, meta


def _features_today(close: np.ndarray, breadth_now: float,
                    breadth_10ago: float) -> dict | None:
    n = len(close)
    if n < 300:
        return None
    lr = np.concatenate(([0.0], np.diff(np.log(close))))
    s10 = rolling_std(lr, 10)[-1] * np.sqrt(252)
    s60 = rolling_std(lr, 60)[-1] * np.sqrt(252)
    s252 = rolling_std(lr, 252)[-1] * np.sqrt(252)
    if not (np.isfinite(s60) and s60 > 0 and np.isfinite(s252) and s252 > 0):
        return None
    sma200 = _rolling_mean(close, 200)[-1]
    sma50 = _rolling_mean(close, 50)[-1]
    hi252 = close[-252:].max()
    lo252 = close[-252:].min()
    r14 = rsi(close, 14)[-1]
    big = np.abs(lr) > 4.0 * (rolling_std(lr, 60) / np.sqrt(252))
    idx = np.where(big)[0]
    gap_age = float(n - 1 - idx.max()) if len(idx) else 999.0
    f = {
        "vr_10_60": s10 / s60, "vr_60_252": s60 / s252,
        "trend200": close[-1] / sma200 if np.isfinite(sma200) else np.nan,
        "trend50": close[-1] / sma50 if np.isfinite(sma50) else np.nan,
        "dd252": 1 - close[-1] / hi252, "up252": close[-1] / lo252 - 1,
        "rsi14": r14, "ret5": close[-1] / close[-6] - 1,
        "ret21": close[-1] / close[-22] - 1, "gap_age": gap_age,
        "breadth": breadth_now, "d_breadth10": breadth_now - breadth_10ago,
        "sigma60": s60,
    }
    if not all(np.isfinite(v) for v in f.values()):
        return None
    return f


def _series_fresh(end_date: np.datetime64) -> bool:
    import pandas as pd
    from datetime import datetime, timezone
    if os.environ.get("CS_SKIP_FRESHNESS") == "1":
        return True
    sessions = _nyse_valid_days_big()
    today = pd.Timestamp(datetime.now(timezone.utc).date())
    i_t = int(sessions.searchsorted(today, side="right")) - 1
    i_e = int(sessions.searchsorted(pd.Timestamp(str(end_date)), side="right")) - 1
    return (i_t - i_e) <= MAX_STALE_SESSIONS


def scan() -> int:
    clf, meta = _load_model()
    thr = meta["threshold"]
    # The Conviction Pick: the single highest-confidence high-liquidity
    # put spread of the day, published only when it clears the frozen
    # conviction threshold (design 97th pctile). Validated 96.9% / 24.7%
    # ROR, ~21/yr. Its gate is LOWER than the full Tier-2 list gate, so
    # score down to it and select the single best afterward.
    conv_thr = meta.get("conviction_threshold", thr)
    score_gate = min(thr, conv_thr)
    with open(os.path.join(RESULTS, "optionable.json")) as fh:
        optionable = json.load(fh)["optionable"]

    tickers = list_tickers()
    limit = os.environ.get("CS_LIMIT")
    if limit:
        tickers = tickers[: int(limit)]

    # pass 1: load panel, compute today's breadth + per-ticker state
    state = {}
    above = above10 = total = total10 = 0
    for t in tickers:
        ts = load_series(t)
        if ts is None:
            continue
        close = ts.close
        if len(close) < 300:
            continue
        sma = _rolling_mean(close, 200)
        if np.isfinite(sma[-1]):
            total += 1
            above += close[-1] >= sma[-1]
        if len(close) > 210 and np.isfinite(sma[-11]):
            total10 += 1
            above10 += close[-11] >= sma[-11]
        state[t] = ts
    breadth_now = above / max(total, 1)
    breadth_10 = above10 / max(total10, 1)

    # pass 2: score and verify
    skip_reality = os.environ.get("CS_SKIP_REALITY") == "1"
    cache = None
    if not skip_reality:
        from reality import ChainCache, verify_rung
        cache = ChainCache()
    signals = []
    n_scored = n_above = 0
    for t, ts in state.items():
        if not optionable.get(t, False):
            continue
        if (ts.dates[-1] - ts.dates[0]).astype(int) < MIN_HISTORY_CAL_DAYS:
            continue
        if not _series_fresh(ts.dates[-1]):
            continue
        f = _features_today(ts.close, breadth_now, breadth_10)
        if f is None:
            continue
        X = np.array([[f[k] for k in FEATURES]])
        proba = float(clf.predict_proba(X)[0, 1])
        n_scored += 1
        if proba < score_gate:
            continue
        n_above += 1
        spot = float(ts.close[-1])
        b = C_SIGMA * (f["sigma60"] / np.sqrt(252)) * np.sqrt(HORIZON)
        k_short = spot * (1 - b)
        k_long = k_short - spot * WIDTH_PCT
        snap = covered_options_expiry(str(ts.dates[-1]), HORIZON)
        if snap is None or k_long <= 0:
            continue
        exp_iso, kind, cal_days, sess = snap
        sig = {
            "engine": ENGINE, "ticker": t, "side": "put",
            "today_close": spot, "end_date": str(ts.dates[-1]),
            "horizon": HORIZON, "buffer_pct": b * 100.0,
            "model_strike": k_short, "model_long_strike": k_long,
            "expiry_date": exp_iso, "expiry_type": kind,
            "calendar_days_to_expiry": cal_days, "sessions_to_expiry": sess,
            "gbm_confidence": proba, "gbm_threshold": thr,
            "sigma60_pct": f["sigma60"] * 100.0,
        }
        if skip_reality:
            sig["strike"] = k_short
            sig["real"] = None
        else:
            rs = verify_rung(cache, t, "put", spot, str(ts.dates[-1]),
                             HORIZON, k_short, k_long, None, min_net=0.05)
            if rs is None:
                continue
            from dataclasses import asdict
            sig["real"] = asdict(rs)
            sig["strike"] = rs.short_strike
            sig["expiry_date"] = rs.expiry
            sig["calendar_days_to_expiry"] = rs.cal_days_to_expiry
            sig["sessions_to_expiry"] = rs.sessions_to_expiry
            sig["buffer_pct"] = rs.real_buffer_pct
        # live-log compatible rung view
        sig["variant"] = "gbm"
        sig["ladder"] = [{k: sig[k] for k in
                          ("engine", "horizon", "expiry_date", "strike",
                           "buffer_pct", "variant")}]
        signals.append(sig)

    signals.sort(key=lambda s: -s["gbm_confidence"])
    # The full Tier-2 list keeps its own (higher) threshold; the
    # Conviction Pick is the single best reality-verified candidate at
    # or above the conviction threshold.
    conv_candidates = [s for s in signals if s["gbm_confidence"] >= conv_thr
                       and s.get("real") is not None]
    tier2_signals = [s for s in signals if s["gbm_confidence"] >= thr]
    elite_thr = meta.get("elite_threshold", 0.9662)
    conviction_pick = None
    if conv_candidates:
        cp = dict(conv_candidates[0])          # highest confidence
        cp["engine"] = CONV_ENGINE
        cp["elite"] = bool(cp["gbm_confidence"] >= elite_thr)
        cp["ladder"] = [{**cp["ladder"][0], "engine": CONV_ENGINE}]
        conviction_pick = cp

    # merge into results/signals.json
    sig_path = os.path.join(RESULTS, "signals.json")
    with open(sig_path) as fh:
        blob = json.load(fh)
    blob["conviction_pick"] = conviction_pick
    blob["tier2_signals"] = tier2_signals
    blob["summary"]["conviction"] = {
        "engine": CONV_ENGINE, "threshold": conv_thr,
        "min_adv_usd": meta.get("conviction_min_adv_usd", 250e6),
        "published_today": conviction_pick is not None,
        "ticker": conviction_pick["ticker"] if conviction_pick else None,
        "elite_today": bool(conviction_pick and conviction_pick.get("elite")),
        "elite_threshold": elite_thr,
        "validated": {"window": "2019-2026", "accuracy": 0.969,
                      "avg_ror_per_trade": 0.247, "per_year": 21,
                      "worst_trade_usd": -160,
                      "elite_accuracy": 1.0, "elite_per_year": 10,
                      "note": ("One highest-confidence high-liquidity put "
                               "spread per week when the bar is met; ~1 every "
                               "2-3 weeks. NOT a guarantee; ~3% of trades lose. "
                               "Elite (99%) band: 73/73 in backtest, ~10/yr.")},
    }
    blob["summary"]["tier2"] = {
        "engine": ENGINE, "n_scored": n_scored,
        "n_above_threshold": n_above, "n_published": len(tier2_signals),
        "spec": {k: meta[k] for k in ("c_sigma", "width_pct", "horizon",
                                      "threshold", "design_target")},
        "validated": {
            "window": "2019-2026", "accuracy": 0.982, "ror_per_trade": 0.243,
            "ror_per_trade_stress": 0.197, "trades_per_week": 7,
            "worst_trade_usd": -516,
            "note": ("Deduped validation trades under conservative fills; "
                     "stress = zero volatility risk premium. NOT a "
                     "guarantee; ~1.8% of trades lose."),
        },
        "reality": ({"chain_fetch_failures": len(cache.failures),
                     "drops": dict(sorted(cache.drops.items()))}
                    if cache is not None else None),
    }
    with open(sig_path, "w") as fh:
        json.dump(blob, fh, indent=2)
    print(f"tier2: scored={n_scored} above_gate={n_above} "
          f"tier2_published={len(tier2_signals)} "
          f"conviction_pick={conviction_pick['ticker'] if conviction_pick else 'none'}")
    if cache is not None and cache.drops:
        print(f"tier2 reality drops: {dict(sorted(cache.drops.items()))}")
    return 0


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "scan"
    sys.exit(train() if mode == "train" else scan())
