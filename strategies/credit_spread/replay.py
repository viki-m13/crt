"""Point-in-time daily replay of the CreditFloor live protocol.

The live loop is:  every trading day, run the walk-forward conformal
engine on all data available up to that close, publish every eligible
ladder rung as a (ticker, side, horizon, strike, expiry) signal, then
resolve each signal at its actual snapped options expiry using
close-at-expiry semantics (exactly what ``live_log.py`` does).

The walk-forward backtest in ``research.py`` is NOT the same thing: it
scores session-horizon path coverage inside calendar-year folds. The
live scoreboard showed 36 losses in 7,000 resolutions (99.49%) for an
engine whose fold backtest claims 100.000% — so any change to the
engine must be validated against THIS replay, which mirrors the live
protocol bit-for-bit:

  - as-of-date eligibility: folds, purge gaps, sample minima and the
    conformal buffer are recomputed from data <= the publish date only;
  - per-horizon variant selection (plain vs regime, smaller buffer,
    regime deployable only when today's gate matches) as in
    ``process_ticker_side``;
  - strike = spot * (1 -/+ b_final(asof));
  - expiry snapped to the real weekly/monthly options calendar
    (``actual_options_expiry``), which can sit 0-4 sessions past the
    session-count horizon;
  - resolution: close on the snapped expiry date vs strike.

Output: one row per published rung per publish day, with everything a
downstream gate experiment needs (realized move to expiry, history-max
buffer, realized vol, pooled OOS sample count as-of publish, etc.).

Usage:
    python3 replay.py                        # full universe, writes results/replay_rows.csv.gz
    CS_LIMIT=50 python3 replay.py            # smoke test
    CS_REPLAY_START=2021-01-01 python3 replay.py
"""
from __future__ import annotations

import csv
import gzip
import os
import sys
import time
from dataclasses import dataclass

import numpy as np

from common import (
    FOLD_YEARS,
    HORIZONS,
    MAX_BUFFER,
    MIN_TEST_SAMPLES,
    MIN_TRAIN_SAMPLES,
    SAFETY_EPS,
    WARMUP_DAYS,
    actual_options_expiry,
    covered_options_expiry,
    compute_features,
    list_tickers,
    load_series,
    regime_mask,
    regime_mask_call,
    worst_buffer_path,
    worst_buffer_path_up,
)

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

REPLAY_START = os.environ.get("CS_REPLAY_START", "2021-01-01")

# Engine-config overrides for deep-history replays:
#   CS_FOLD_START=2006  -> walk-forward folds 2006..2026 instead of 2020..2026
#   CS_HORIZONS=7,21,63 -> horizon ladder override
#   CS_SNAP=down|up     -> expiry snapping (down = covered_options_expiry)
if os.environ.get("CS_FOLD_START"):
    FOLD_YEARS = list(range(int(os.environ["CS_FOLD_START"]), FOLD_YEARS[-1] + 1))
if os.environ.get("CS_HORIZONS"):
    HORIZONS = [int(x) for x in os.environ["CS_HORIZONS"].split(",")]
SNAP_MODE = os.environ.get("CS_SNAP", "down")


@dataclass
class ReplayParams:
    """Engine parameters. Defaults replicate the legacy live engine."""
    safety_eps: float = SAFETY_EPS       # absolute additive margin
    eps_rel: float = 0.0                 # relative margin: b = hist_max*(1+eps_rel) + safety_eps
    max_buffer: float = MAX_BUFFER
    min_train: int = MIN_TRAIN_SAMPLES
    min_test: int = MIN_TEST_SAMPLES


def _rolling_sigma60(close: np.ndarray, window: int = 60) -> np.ndarray:
    """Annualized rolling stdev (ddof=1) of daily log returns, aligned so
    out[j] uses returns ending at close[j]. NaN until enough history."""
    n = len(close)
    out = np.full(n, np.nan)
    if n < window + 1:
        return out
    lr = np.diff(np.log(close))
    # rolling sums for mean/var
    c1 = np.concatenate(([0.0], np.cumsum(lr)))
    c2 = np.concatenate(([0.0], np.cumsum(lr * lr)))
    s1 = c1[window:] - c1[:-window]          # sum of window returns ending at k = window..len(lr)
    s2 = c2[window:] - c2[:-window]
    var = (s2 - s1 * s1 / window) / (window - 1)
    var = np.maximum(var, 0.0)
    out[window:] = np.sqrt(var) * np.sqrt(252.0)
    return out


def _cum_at_obs(n: int, sample_idx: np.ndarray, h: int) -> np.ndarray:
    """Given sample indices i (events observable at index i+h), return an
    array c of length n where c[j] = #(samples with i+h <= j)."""
    c = np.zeros(n, dtype=np.int64)
    if len(sample_idx):
        obs = sample_idx + h
        obs = obs[obs < n]
        np.add.at(c, obs, 1)
    return np.cumsum(c)


def _prefix_max_at_obs(n: int, sample_idx: np.ndarray, values: np.ndarray, h: int) -> np.ndarray:
    """m[j] = max(values[i] for samples i with i+h <= j), -inf if none."""
    m = np.full(n, -np.inf)
    if len(sample_idx):
        obs = sample_idx + h
        keep = obs < n
        np.maximum.at(m, obs[keep], values[sample_idx[keep]])
    return np.maximum.accumulate(m)


def replay_ticker(ticker: str, params: ReplayParams, replay_start: np.datetime64,
                  expiry_cache: dict) -> dict[str, list]:
    """Replay one ticker. Returns column lists for the row table."""
    cols: dict[str, list] = {k: [] for k in (
        "date", "ticker", "side", "horizon", "variant", "spot", "hist_max",
        "buffer", "strike", "expiry_date", "cal_days", "close_at_expiry",
        "move", "win", "n_pooled", "sigma60",
    )}
    ts = load_series(ticker)
    if ts is None:
        return cols
    close = ts.close
    dates = ts.dates
    n = len(dates)
    feats = compute_features(close)
    sigma60 = _rolling_sigma60(close)
    warmup = np.zeros(n, dtype=bool)
    warmup[WARMUP_DAYS:] = True

    # j-indexed: replay day range
    j_start = int(np.searchsorted(dates, replay_start))
    if j_start >= n:
        return cols

    year_starts = {y: np.datetime64(f"{y}-01-01") for y in
                   list(FOLD_YEARS) + [FOLD_YEARS[-1] + 1]}
    # year index of each date for fold membership
    date_year = dates.astype("datetime64[Y]").astype(int) + 1970

    # today-regime-ok arrays (vectorized _today_regime_ok)
    put_ok_today = (np.isfinite(feats.trend) & np.isfinite(feats.dd252)
                    & (feats.trend >= 1.00) & (feats.dd252 <= 0.20))
    call_ok_today = (np.isfinite(feats.trend) & np.isfinite(feats.up252)
                     & (feats.trend <= 1.00) & (feats.up252 <= 0.20))

    for side in ("put", "call"):
        for h in HORIZONS:
            buf = worst_buffer_path(close, h) if side == "put" else worst_buffer_path_up(close, h)
            finite = np.isfinite(buf)
            end_dates = np.full(n, np.datetime64("9999-12-31"), dtype="datetime64[D]")
            end_dates[: n - h] = dates[h:]

            # per-variant: (eligible_bool[j], b_final[j])
            variant_state = {}
            for variant in ("plain", "regime"):
                require = variant == "regime"
                rmask = (regime_mask(feats, require) if side == "put"
                         else regime_mask_call(feats, require))
                base = rmask & warmup & finite
                base_idx = np.where(base)[0]

                # b_final(j): prefix max of buf over base samples observable by j
                histmax = _prefix_max_at_obs(n, base_idx, buf, h)
                b_final = np.where(np.isfinite(histmax),
                                   np.minimum(histmax * (1.0 + params.eps_rel)
                                              + params.safety_eps, 0.99),
                                   np.nan)
                base_cnt = _cum_at_obs(n, base_idx, h)

                # fold machinery
                ok_all = np.ones(n, dtype=bool)      # all active folds clean as-of j
                any_fold = np.zeros(n, dtype=bool)   # >=1 active fold as-of j
                pooled = np.zeros(n, dtype=np.int64)
                for y in FOLD_YEARS:
                    y0 = year_starts[y]
                    tr = base & (dates < y0) & (end_dates < y0)
                    if int(tr.sum()) < params.min_train:
                        continue
                    b_train = float(buf[tr].max())
                    b_hat = min(b_train * (1.0 + params.eps_rel) + params.safety_eps, 0.99)
                    te = base & (date_year == y)
                    te_idx = np.where(te)[0]
                    if len(te_idx) == 0:
                        continue
                    loss = buf[te_idx] > b_hat
                    losses_cum = _cum_at_obs(n, te_idx[loss], h)
                    tests_cum = _cum_at_obs(n, te_idx, h)
                    wins_cum = tests_cum - losses_cum
                    active = tests_cum >= 1
                    fold_ok = ~active | ((losses_cum == 0) & (wins_cum >= 1))
                    ok_all &= fold_ok
                    any_fold |= active
                    pooled += tests_cum

                eligible = (any_fold & ok_all & (pooled >= params.min_test)
                            & (base_cnt >= params.min_train)
                            & np.isfinite(b_final) & (b_final <= params.max_buffer))
                variant_state[variant] = (eligible, b_final, pooled)

            # per-day variant pick (mirror process_ticker_side):
            pe, pb, pn = variant_state["plain"]
            re_, rb, rn = variant_state["regime"]
            today_ok = put_ok_today if side == "put" else call_ok_today
            r_deploy = re_ & today_ok
            # choose regime when deployable and (plain ineligible or regime buffer smaller)
            use_regime = r_deploy & (~pe | (rb < pb))
            use_plain = pe & ~use_regime
            published = use_regime | use_plain
            published[:j_start] = False
            pj = np.where(published)[0]
            if len(pj) == 0:
                continue
            b_used = np.where(use_regime[pj], rb[pj], pb[pj])
            n_pooled = np.where(use_regime[pj], rn[pj], pn[pj])
            variants = np.where(use_regime[pj], "regime", "plain")
            spot = close[pj]
            strike = spot * (1.0 - b_used) if side == "put" else spot * (1.0 + b_used)

            for k, j in enumerate(pj):
                dkey = (str(dates[j]), h)
                if dkey not in expiry_cache:
                    if SNAP_MODE == "down":
                        expiry_cache[dkey] = covered_options_expiry(str(dates[j]), h)
                    else:
                        expiry_cache[dkey] = actual_options_expiry(str(dates[j]), h)
                snap = expiry_cache[dkey]
                if snap is None:
                    continue  # no standard expiry inside the certified window
                exp_iso, _kind, cal_days = snap[0], snap[1], snap[2]
                exp_d = np.datetime64(exp_iso, "D")
                # resolve: need data through expiry
                if dates[-1] < exp_d:
                    close_exp = np.nan
                    win = -1  # pending
                else:
                    kk = int(np.searchsorted(dates, exp_d, side="right")) - 1
                    close_exp = float(close[kk])
                    if side == "put":
                        win = 1 if close_exp >= strike[k] else 0
                    else:
                        win = 1 if close_exp <= strike[k] else 0
                move = (close_exp / spot[k] - 1.0) if np.isfinite(close_exp) else np.nan
                cols["date"].append(str(dates[j]))
                cols["ticker"].append(ticker)
                cols["side"].append(side)
                cols["horizon"].append(h)
                cols["variant"].append(variants[k])
                cols["spot"].append(round(float(spot[k]), 6))
                cols["hist_max"].append(round(float(b_used[k] - params.safety_eps)
                                              / (1.0 + params.eps_rel), 6))
                cols["buffer"].append(round(float(b_used[k]), 6))
                cols["strike"].append(round(float(strike[k]), 6))
                cols["expiry_date"].append(exp_iso)
                cols["cal_days"].append(cal_days)
                cols["close_at_expiry"].append(round(close_exp, 6) if np.isfinite(close_exp) else "")
                cols["move"].append(round(move, 6) if np.isfinite(move) else "")
                cols["win"].append(win)
                cols["n_pooled"].append(int(n_pooled[k]))
                s = sigma60[j]
                cols["sigma60"].append(round(float(s), 6) if np.isfinite(s) else "")
    return cols


def main() -> int:
    tickers = list_tickers()
    limit = os.environ.get("CS_LIMIT")
    if limit:
        tickers = tickers[: int(limit)]
    params = ReplayParams()
    replay_start = np.datetime64(REPLAY_START, "D")
    out_path = os.path.join(OUTPUT_DIR, os.environ.get("CS_REPLAY_OUT", "replay_rows.csv.gz"))

    expiry_cache: dict = {}
    t0 = time.time()
    header = None
    n_rows = 0
    with gzip.open(out_path, "wt", newline="") as fh:
        w = csv.writer(fh)
        for i, t in enumerate(tickers, 1):
            try:
                cols = replay_ticker(t, params, replay_start, expiry_cache)
            except Exception as exc:  # noqa: BLE001
                print(f"[ERR] {t}: {exc}", file=sys.stderr)
                continue
            if header is None:
                header = list(cols.keys())
                w.writerow(header)
            m = len(cols["date"])
            for r in range(m):
                w.writerow([cols[k][r] for k in header])
            n_rows += m
            if i % 100 == 0:
                print(f"  {i}/{len(tickers)}  rows={n_rows}  elapsed={time.time()-t0:.1f}s",
                      flush=True)
    print(f"Done: {n_rows} rows -> {out_path}  ({time.time()-t0:.1f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
