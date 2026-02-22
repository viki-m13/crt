#!/usr/bin/env python3
# ============================================================
# Rebound Ledger — static daily scan generator
#
# Writes:
#   - docs/data/full.json                (ranking table + top10 embedded details)
#   - docs/data/tickers/{TICKER}.json    (detail payload for every scored ticker)
#
# Scheduling:
#   - GitHub Action runs hourly.
#   - This script self-gates to run once per day after 5pm America/New_York.
#   - Set FORCE_RUN=1 to bypass gating (manual runs).
#
# Model: CRT REBOUND SCANNER — v8
#   • Washout Meter (0–100): how washed-out/sold-off it looks (stock-specific + market-adjusted).
#   • Edge Score    (0–100): rebound edge from historical analogs (same stock, similar setups).
#   • Final Score   (0–100): Edge Score amplified by Washout (the ONE score used for ranking + charts).
#   • Evidence A: top 10% Washout days vs all days.
#   • Evidence B: top 10% Final Score days vs all days.
# ============================================================

import os
import math
import json
import warnings
warnings.filterwarnings("ignore")

from datetime import datetime
from zoneinfo import ZoneInfo
from io import StringIO

import numpy as np
import pandas as pd
import yfinance as yf
import requests

# =========================
# CONFIG (KEEP SIMPLE)
# =========================
INTERVAL = "1d"
PERIOD   = "max"
BENCH    = "SPY"

ISHARES_HOLDINGS_URL = (
    "https://www.ishares.com/us/products/239707/ishares-russell-1000-etf/"
    "1467271812596.ajax?fileType=csv&fileName=IWB_holdings&dataType=fund"
)

ALWAYS_PLOT = ["SPY", "BTC-USD", "ETH-USD", "QQQ", "IWM", "DIA", "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "COST", "BRK-A", "ARM"]

CHUNK_SIZE = 80
MAX_TICKERS = None          # e.g. 600 for speed, else None
TOP10_EMBED = 10
PLOT_LAST_DAYS = 365 * 6

# Lookbacks (days)
LB_LT = 252                 # ~1 year
LB_ST = 63                  # ~3 months
BETA_LB = 126               # ~6 months
ATR_N = 14

# “Bottom confirmation” parameters (soft; not a hard rule)
DD_THR  = 0.25              # down ~25%+ from recent peak (LT)
POS_THR = 0.20              # in bottom ~20% of LT range
GATE_DD_SCALE  = 0.12
GATE_POS_SCALE = 0.10

# Filters (quality)
MIN_MED_DVOL_USD = 5_000_000
MAX_MISSING_FRAC = 0.10
MIN_HISTORY_BARS = LB_LT + BETA_LB + LB_ST + 220

# Analogs
ANALOG_K = 250
ANALOG_MIN = 80
ANALOG_MIN_SEP_DAYS = 10

# Stability check
STAB_K_SET = [150, 250, 350]
STAB_SHIFT_STEPS = [0, -5, -10, -15]     # only backward (no peeking)
STAB_STD_SCALE_POINTS = 12.0
STAB_MIN_MEAN_RATIO = 0.60

# Horizons
HORIZONS_DAYS = {"1Y": 252, "3Y": 252*3, "5Y": 252*5}
HORIZON_WEIGHTS = {"1Y": 0.50, "3Y": 0.35, "5Y": 0.15}

# Verdict thresholds (simple, readable)
VERDICT = dict(
    STRONG_SCORE=72.0,
    OK_SCORE=60.0,
    HIGH_CONF=72.0,
    OK_CONF=60.0,
    HIGH_STAB=72.0,
    OK_STAB=60.0,
)

# Final score knobs (dominant washout but still requires edge)
MIN_WASHOUT_TODAY = 55.0      # gate for the scan (set 0 to disable). ALWAYS_PLOT bypasses.
FINAL_WASH_FLOOR  = 0.35      # baseline multiplier even if washout is low
FINAL_WASH_WEIGHT = 0.65      # how much washout amplifies the edge score

# Plot-time performance (FinalScore series is computed sparsely, then forward-filled)
PLOT_SCORE_STEP_BARS = 12
EVID_SCORE_STEP_BARS = 18

# Outputs
OUT_DIR = os.path.join("docs", "data")
TICKER_DIR = os.path.join(OUT_DIR, "tickers")

# =========================
# Time gate (after 5pm NY)
# =========================
def should_run_now() -> bool:
    if os.getenv("FORCE_RUN", "").strip() == "1":
        return True
    tz = ZoneInfo("America/New_York")
    now = datetime.now(tz)
    if now.hour < 17:
        return False
    stamp_path = os.path.join(OUT_DIR, "last_run.txt")
    today = now.strftime("%Y-%m-%d")
    if os.path.exists(stamp_path):
        prev = open(stamp_path, "r").read().strip()
        if prev == today:
            return False
    return True

def mark_ran_today() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    tz = ZoneInfo("America/New_York")
    today = datetime.now(tz).strftime("%Y-%m-%d")
    open(os.path.join(OUT_DIR, "last_run.txt"), "w").write(today)

# =========================
# Helpers
# =========================
def sigmoid(x: float) -> float:
    x = float(np.clip(x, -20, 20))
    return 1.0 / (1.0 + math.exp(-x))

def robust_z(x: pd.Series, win: int) -> pd.Series:
    med = x.rolling(win).median()
    mad = (x - med).abs().rolling(win).median()
    denom = (1.4826 * mad).replace(0, np.nan)
    z = (x - med) / denom
    return z.replace([np.inf, -np.inf], np.nan)

def safe_float(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else np.nan
    except Exception:
        return np.nan

def json_float(x):
    """Like safe_float but returns None (-> JSON null) instead of NaN."""
    try:
        v = float(x)
        return v if np.isfinite(v) else None
    except Exception:
        return None

def sanitize_for_json(obj):
    """Recursively replace NaN/Inf floats with None so json.dump produces valid JSON."""
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return sanitize_for_json(float(obj))
    return obj

def percentile_rank(series: pd.Series, value: float) -> float:
    s = series.dropna().values.astype(float)
    if len(s) < 30 or not np.isfinite(value):
        return np.nan
    return float(np.mean(s <= value))

def washout_top_pct(series: pd.Series, value: float) -> float:
    """Return 'top X% most washed-out' where higher washout is more extreme."""
    p = percentile_rank(series, value)
    if not np.isfinite(p):
        return np.nan
    return float((1.0 - p) * 100.0)

def final_score(edge_score: float, washout: float) -> float:
    """FinalScore = EdgeScore amplified by how washed-out it is today."""
    e = safe_float(edge_score)
    w = safe_float(washout)
    if not np.isfinite(e) or not np.isfinite(w):
        return np.nan
    ww = float(np.clip(w / 100.0, 0.0, 1.0))
    return float(e * (FINAL_WASH_FLOOR + FINAL_WASH_WEIGHT * ww))

# =========================
# QUALITY GATES (price-based)
# =========================

def trend_quality(px: pd.Series, sma_win: int = 200) -> float:
    """Fraction of last 5 years the stock spent above its 200-day SMA. 0-100."""
    if len(px) < sma_win + 10:
        return np.nan
    sma = px.rolling(sma_win).mean()
    # Use last 5Y (1260 bars) or full history if shorter
    lookback = min(len(px), 1260)
    recent_px = px.iloc[-lookback:]
    recent_sma = sma.iloc[-lookback:]
    valid = recent_sma.notna()
    if valid.sum() < 252:  # need at least 1 year
        return np.nan
    above = (recent_px[valid] > recent_sma[valid]).sum()
    return float(100.0 * above / valid.sum())


def recovery_track_record(px: pd.Series, dd_threshold: float = 0.20, recovery_window: int = 756) -> dict:
    """How often has this stock recovered from drawdowns >= dd_threshold within recovery_window bars?
    Returns dict with recovery_rate (0-1), n_drawdowns, n_recovered."""
    if len(px) < 504:  # need 2+ years
        return {"recovery_rate": np.nan, "n_drawdowns": 0, "n_recovered": 0}

    # Find all drawdown episodes
    cummax = px.cummax()
    dd = (px / cummax) - 1.0  # negative values

    # Find points where drawdown first crosses threshold
    crossed = dd < -dd_threshold
    # Group into episodes (require 63+ bars between episodes)
    episodes = []
    last_end = -100
    for i in range(len(crossed)):
        if crossed.iloc[i] and (i - last_end) > 63:
            # Start of a new drawdown episode
            trough_start = i
            # Find the trough (deepest point in next 126 bars)
            search_end = min(i + 126, len(dd))
            trough_idx = dd.iloc[trough_start:search_end].idxmin()
            trough_pos = px.index.get_loc(trough_idx)

            # Check if it recovered to prior peak within recovery_window
            prior_peak = cummax.iloc[trough_pos]
            recovery_end = min(trough_pos + recovery_window, len(px))
            future_px = px.iloc[trough_pos:recovery_end]
            recovered = (future_px >= prior_peak * 0.95).any()  # within 5% of prior peak

            episodes.append({
                "trough_date": px.index[trough_pos],
                "dd_depth": float(dd.iloc[trough_pos]),
                "recovered": bool(recovered),
                "has_forward_data": recovery_end > trough_pos + 252  # at least 1Y of forward data
            })
            last_end = trough_pos + 63

    # Only count episodes with enough forward data
    valid_episodes = [e for e in episodes if e["has_forward_data"]]
    if len(valid_episodes) == 0:
        return {"recovery_rate": np.nan, "n_drawdowns": 0, "n_recovered": 0}

    n_recovered = sum(1 for e in valid_episodes if e["recovered"])
    return {
        "recovery_rate": float(n_recovered / len(valid_episodes)),
        "n_drawdowns": len(valid_episodes),
        "n_recovered": n_recovered,
    }


def selling_deceleration(px: pd.Series) -> float:
    """Is selling slowing down? Returns 0-100 (higher = more deceleration = better entry).
    Checks: 5d return vs 20d return, and RSI turning up."""
    if len(px) < 30:
        return np.nan

    ret_5d = float(px.iloc[-1] / px.iloc[-6] - 1) if len(px) >= 6 else 0.0
    ret_20d = float(px.iloc[-1] / px.iloc[-21] - 1) if len(px) >= 21 else 0.0

    # Deceleration: 5d less negative than 20d (or positive)
    if ret_20d < -0.01:  # stock is in a pullback
        decel = float(np.clip((ret_5d - ret_20d) / abs(ret_20d), -1, 2))
    else:
        decel = 0.5  # not in pullback, neutral

    # RSI(14)
    delta = px.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi_now = float(rsi.iloc[-1]) if np.isfinite(rsi.iloc[-1]) else 50.0

    # RSI turning up from oversold
    rsi_5d_ago = float(rsi.iloc[-6]) if len(rsi) >= 6 and np.isfinite(rsi.iloc[-6]) else 50.0
    rsi_turn = 1.0 if (rsi_now > rsi_5d_ago and rsi_5d_ago < 35) else 0.0

    # Combine: 50% deceleration + 30% RSI level + 20% RSI turn
    rsi_score = float(np.clip((rsi_now - 20) / 40, 0, 1))  # 0 at RSI=20, 1 at RSI=60
    score = 0.50 * float(np.clip((decel + 0.5) / 1.5, 0, 1)) + 0.30 * rsi_score + 0.20 * rsi_turn
    return float(np.clip(score * 100, 0, 100))

# =========================
# Verdict
# =========================
def verdict_line(score: float, confidence: float, stability: float, fragile: bool) -> str:
    s = safe_float(score); c = safe_float(confidence); st = safe_float(stability)
    if not np.isfinite(s) or not np.isfinite(c) or not np.isfinite(st):
        return "Verdict: Insufficient data"

    strong = (s >= VERDICT["STRONG_SCORE"])
    ok     = (s >= VERDICT["OK_SCORE"])
    high_c = (c >= VERDICT["HIGH_CONF"])
    ok_c   = (c >= VERDICT["OK_CONF"])
    high_s = (st >= VERDICT["HIGH_STAB"])
    ok_s   = (st >= VERDICT["OK_STAB"])

    if strong and high_c and high_s and (not fragile):
        return "Verdict: Strong and consistent"
    if strong and (ok_c or high_c) and (not high_s or fragile):
        return "Verdict: Strong signal, less consistent"
    if ok and (not ok_c):
        return "Verdict: Interesting but limited data"
    if ok and ok_c and ok_s and fragile:
        return "Verdict: Worth watching"
    if ok and ok_c:
        return "Verdict: Moderate"
    return "Verdict: No clear opportunity today"

# =========================
# Holdings
# =========================
def fetch_ishares_holdings_tickers(url: str) -> list:
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    }
    resp = requests.get(url, headers=headers, timeout=45)
    if resp.status_code != 200 or not resp.content:
        raise RuntimeError(f"Holdings download failed (HTTP {resp.status_code}).")

    raw_text = resp.content.decode("utf-8", errors="ignore")
    lines = raw_text.splitlines()

    header_idx = None
    for i, line in enumerate(lines[:700]):
        if line.strip().startswith("Ticker,") or line.strip().startswith('"Ticker",'):
            header_idx = i
            break
    if header_idx is None:
        for i, line in enumerate(lines):
            first = line.strip().split(",")[0].replace('"', '')
            if first == "Ticker":
                header_idx = i
                break
    if header_idx is None:
        raise RuntimeError("Could not locate 'Ticker' header (CSV layout likely changed).")

    trimmed = "\n".join(lines[header_idx:])
    df = pd.read_csv(StringIO(trimmed))
    df.columns = [c.strip() for c in df.columns]
    if "Ticker" not in df.columns:
        raise RuntimeError("Ticker column missing after parsing holdings CSV.")

    if "Asset Class" in df.columns:
        df = df[df["Asset Class"].astype(str).str.contains("Equity", case=False, na=False)]

    tick = (
        df["Ticker"].astype(str)
        .str.strip()
        .str.replace(" ", "", regex=False)
        .str.replace(".", "-", regex=False)
        .str.upper()
    )
    keep = tick[
        tick.ne("") &
        tick.ne("NAN") &
        (~tick.str.contains("CASH", case=False, na=False)) &
        (~tick.str.contains("DERIV", case=False, na=False))
    ].dropna().unique().tolist()

    if len(keep) < 5:
        raise RuntimeError(f"Parsed too few tickers ({len(keep)}). CSV layout likely changed.")
    return sorted(keep)

# =========================
# yfinance download
# =========================
def _extract_field(raw: pd.DataFrame, field: str) -> pd.DataFrame:
    if raw is None or len(raw) == 0:
        return pd.DataFrame()

    if isinstance(raw.columns, pd.MultiIndex):
        for lvl in (1, -1, 0):
            try:
                out = raw.xs(field, axis=1, level=lvl)
                if isinstance(out, pd.Series):
                    out = out.to_frame()
                out.columns = out.columns.astype(str)
                return out
            except Exception:
                pass

        tmp = raw.copy()
        tmp.columns = ["|".join(map(str, c)) for c in tmp.columns]
        keep = [c for c in tmp.columns if c.endswith(f"|{field}")]
        if not keep:
            return pd.DataFrame()
        out = tmp[keep].copy()
        out.columns = [c.split("|")[0] for c in out.columns]
        return out

    if field in raw.columns:
        s = raw[field].copy()
        if isinstance(s, pd.Series):
            return s.to_frame()
        return s
    return pd.DataFrame()

def download_ohlcv_period(tickers, period="max", interval="1d", chunk_size=80):
    tickers = sorted(set([t for t in tickers if isinstance(t, str) and t.strip()]))

    O_list, H_list, L_list, C_list, V_list, A_list = [], [], [], [], [], []
    for i in range(0, len(tickers), chunk_size):
        batch = tickers[i:i + chunk_size]
        raw = yf.download(
            batch,
            period=period,
            interval=interval,
            auto_adjust=False,
            progress=False,
            group_by="ticker",
            threads=True,
        )
        O_list.append(_extract_field(raw, "Open"))
        H_list.append(_extract_field(raw, "High"))
        L_list.append(_extract_field(raw, "Low"))
        C_list.append(_extract_field(raw, "Close"))
        V_list.append(_extract_field(raw, "Volume"))
        A_list.append(_extract_field(raw, "Adj Close"))

    def _cat(xlist):
        xlist = [x for x in xlist if isinstance(x, pd.DataFrame) and not x.empty]
        if not xlist:
            return pd.DataFrame()
        x = pd.concat(xlist, axis=1)
        x = x.loc[:, ~x.columns.duplicated()].sort_index()
        x.index = pd.to_datetime(x.index, utc=True)
        return x

    return {
        "Open": _cat(O_list),
        "High": _cat(H_list),
        "Low":  _cat(L_list),
        "Close": _cat(C_list),
        "Volume": _cat(V_list),
        "AdjClose": _cat(A_list),
    }

# =========================
# Features
# =========================
def compute_core_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    d = ohlcv.copy()
    for c in ["open", "high", "low", "close", "volume"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    close = d["close"]; high = d["high"]; low = d["low"]; vol = d["volume"]; opn = d["open"]

    hi_lt = high.rolling(LB_LT).max()
    lo_lt = low.rolling(LB_LT).min()
    rng_lt = (hi_lt - lo_lt).replace(0, np.nan)
    pos_lt = ((close - lo_lt) / rng_lt).replace([np.inf, -np.inf], np.nan)
    dd_lt  = (1.0 - close / hi_lt).replace([np.inf, -np.inf], np.nan)

    hi_st = high.rolling(LB_ST).max()
    lo_st = low.rolling(LB_ST).min()
    rng_st = (hi_st - lo_st).replace(0, np.nan)
    pos_st = ((close - lo_st) / rng_st).replace([np.inf, -np.inf], np.nan)
    dd_st  = (1.0 - close / hi_st).replace([np.inf, -np.inf], np.nan)

    prev_close = close.shift(1)
    tr = pd.concat([(high-low).abs(), (high-prev_close).abs(), (low-prev_close).abs()], axis=1).max(axis=1)
    atr = tr.rolling(ATR_N).mean()
    atr_pct = (atr / close).replace([np.inf, -np.inf], np.nan)

    volu_z = robust_z(vol, LB_ST)
    gap = (opn / prev_close - 1.0).replace([np.inf, -np.inf], np.nan)
    trend_st = (close / close.shift(LB_ST) - 1.0).replace([np.inf, -np.inf], np.nan)

    out = d.copy()
    out["dd_lt"] = dd_lt; out["pos_lt"] = pos_lt
    out["dd_st"] = dd_st; out["pos_st"] = pos_st
    out["atr_pct"] = atr_pct
    out["volu_z"] = volu_z
    out["gap"] = gap
    out["trend_st"] = trend_st
    return out

def compute_bottom_confirmation(dd_lt: pd.Series, pos_lt: pd.Series) -> pd.Series:
    dd_comp = ((dd_lt - DD_THR) / GATE_DD_SCALE).clip(-5, 5).apply(sigmoid)
    pos_comp = ((POS_THR - pos_lt) / GATE_POS_SCALE).clip(-5, 5).apply(sigmoid)
    return (dd_comp * pos_comp).clip(0, 1)

def compute_idiosyncratic_features(stock_px: pd.Series, spy_px: pd.Series) -> pd.DataFrame:
    r_s = np.log(stock_px).diff()
    r_m = np.log(spy_px).diff()

    cov = r_s.rolling(BETA_LB).cov(r_m)
    var = r_m.rolling(BETA_LB).var()
    beta = (cov / var).replace([np.inf, -np.inf], np.nan)

    resid = (r_s - beta * r_m).replace([np.inf, -np.inf], np.nan)
    resid_price = np.exp(resid.fillna(0).cumsum())
    if resid_price.notna().any() and resid_price.iloc[0] != 0:
        resid_price = resid_price / resid_price.iloc[0]

    hi_lt = resid_price.rolling(LB_LT).max()
    lo_lt = resid_price.rolling(LB_LT).min()
    rng_lt = (hi_lt - lo_lt).replace(0, np.nan)
    idio_pos_lt = ((resid_price - lo_lt) / rng_lt).replace([np.inf, -np.inf], np.nan)
    idio_dd_lt = (1.0 - resid_price / hi_lt).replace([np.inf, -np.inf], np.nan)

    hi_st = resid_price.rolling(LB_ST).max()
    lo_st = resid_price.rolling(LB_ST).min()
    rng_st = (hi_st - lo_st).replace(0, np.nan)
    idio_pos_st = ((resid_price - lo_st) / rng_st).replace([np.inf, -np.inf], np.nan)
    idio_dd_st = (1.0 - resid_price / hi_st).replace([np.inf, -np.inf], np.nan)

    out = pd.DataFrame(index=stock_px.index)
    out["beta"] = beta
    out["idio_dd_lt"] = idio_dd_lt
    out["idio_pos_lt"] = idio_pos_lt
    out["idio_dd_st"] = idio_dd_st
    out["idio_pos_st"] = idio_pos_st
    return out

def compute_market_regime(spy_px: pd.Series, spy_high: pd.Series, spy_low: pd.Series) -> pd.DataFrame:
    trend = (spy_px / spy_px.shift(LB_ST) - 1.0).replace([np.inf, -np.inf], np.nan)
    r = np.log(spy_px).diff()
    vol = r.rolling(LB_ST).std().replace([np.inf, -np.inf], np.nan)

    hi = spy_high.rolling(LB_LT).max()
    dd = (1.0 - spy_px / hi).replace([np.inf, -np.inf], np.nan)

    prev = spy_px.shift(1)
    tr = pd.concat([(spy_high-spy_low).abs(), (spy_high-prev).abs(), (spy_low-prev).abs()], axis=1).max(axis=1)
    atr = tr.rolling(ATR_N).mean()
    atr_pct = (atr / spy_px).replace([np.inf, -np.inf], np.nan)

    out = pd.DataFrame(index=spy_px.index)
    out["mkt_trend"] = trend
    out["mkt_vol"] = vol
    out["mkt_dd"] = dd
    out["mkt_atr_pct"] = atr_pct
    return out

def make_regime_bucket(mkt_row: pd.Series) -> str:
    tr = safe_float(mkt_row.get("mkt_trend", np.nan))
    dd = safe_float(mkt_row.get("mkt_dd", np.nan))
    vol= safe_float(mkt_row.get("mkt_vol", np.nan))
    if not np.isfinite(tr) or not np.isfinite(dd) or not np.isfinite(vol):
        return "UNK"

    tr_b = "UP" if tr >= 0 else "DN"
    dd_b = "DDH" if dd >= 0.15 else "DDL"
    if vol < 0.010: vol_b = "VLO"
    elif vol < 0.018: vol_b = "VMD"
    else: vol_b = "VHI"
    return f"{tr_b}_{dd_b}_{vol_b}"

def forward_returns(px: pd.Series) -> pd.DataFrame:
    out = pd.DataFrame(index=px.index)
    for name, days in HORIZONS_DAYS.items():
        out[f"fwd_{name}"] = (px.shift(-days) / px - 1.0).replace([np.inf, -np.inf], np.nan)
    return out

# =========================
# Washout Meter
# =========================
def compute_washout_meter(feat: pd.DataFrame) -> pd.Series:
    struct = (0.6 * feat["dd_lt"].clip(0, 1) + 0.4 * (1 - feat["pos_lt"]).clip(0, 1)).clip(0, 1)
    idio = (0.6 * feat["idio_dd_lt"].clip(0, 1) + 0.4 * (1 - feat["idio_pos_lt"]).clip(0, 1)).clip(0, 1)
    cap = (
        0.6 * robust_z(feat["atr_pct"], max(63, LB_ST)).clip(-3, 3).fillna(0) / 3.0 +
        0.4 * feat["volu_z"].clip(-3, 3).fillna(0) / 3.0
    )
    cap = ((cap + 1) / 2).clip(0, 1)
    confirm = feat["bottom_confirm"].clip(0, 1).fillna(0)

    wash = (0.55 * struct + 0.30 * idio + 0.15 * cap).clip(0, 1)
    wash = (wash * confirm).clip(0, 1)
    return 100.0 * wash

# =========================
# Analog engine
# =========================
def build_feature_matrix(feat: pd.DataFrame, feature_cols: list, zwin: int) -> pd.DataFrame:
    X = pd.DataFrame(index=feat.index)
    for c in feature_cols:
        X[c] = robust_z(feat[c], zwin)
    return X

def summarize(vals: np.ndarray) -> dict:
    if vals is None or len(vals) == 0:
        return {"n": 0}
    vals = np.array(vals, dtype=float)
    # Survivorship fix: treat NaN (delisted/missing) as -100% loss
    n_total = len(vals)
    n_missing = int(np.isnan(vals).sum())
    vals = np.where(np.isnan(vals), -1.0, vals)
    return {
        "n": int(len(vals)),
        "n_missing": n_missing,
        "win": float(np.mean(vals > 0)),
        "median": float(np.median(vals)),
        "p10": float(np.quantile(vals, 0.10)),
        "p90": float(np.quantile(vals, 0.90)),
    }

def horizon_unit(confirm: float, stats: dict, n_target: int) -> float:
    n = stats.get("n", 0)
    if n <= 0:
        return 0.0
    win = stats["win"]; med = stats["median"]; p10 = stats["p10"]
    sample_conf = min(1.0, n / max(1, n_target))
    med_term = sigmoid(med / 0.25)
    tail_pen = sigmoid(max(0.0, -p10) / 0.25)
    raw = 0.62 * win + 0.38 * med_term - 0.35 * tail_pen
    raw = float(np.clip(raw, 0, 1))
    return float(np.clip(confirm * sample_conf * raw, 0, 1))

def select_analogs_regime_balanced(X: pd.DataFrame, y: pd.Series, regimes: pd.Series,
                                  now_idx: pd.Timestamp, k: int, min_sep_days: int,
                                  eligible: pd.Series | None = None):
    cand = X.index[(X.notna().all(axis=1)) & (y.notna()) & (X.index < now_idx)]
    if eligible is not None:
        em = eligible.reindex(cand).fillna(False).values
        cand = cand[em]

    if len(cand) == 0:
        return []

    if now_idx not in X.index or not X.loc[now_idx].notna().all():
        return []

    x0 = X.loc[now_idx].values.astype(float)
    Xc = X.loc[cand].values.astype(float)
    d = np.sqrt(((Xc - x0) ** 2).sum(axis=1))
    order = np.argsort(d)

    cand_sorted = cand[order]
    reg_sorted  = regimes.reindex(cand_sorted).fillna("UNK").values

    buckets = {}
    for t, r in zip(cand_sorted, reg_sorted):
        buckets.setdefault(r, []).append(t)

    regs = list(buckets.keys())
    counts = np.array([len(buckets[r]) for r in regs], dtype=float)
    w = np.sqrt(counts)
    w = w / w.sum() if w.sum() > 0 else np.ones_like(w) / max(1, len(w))
    quota = {r: max(3, int(round(k * wi))) for r, wi in zip(regs, w)}

    chosen = []
    last_t = None

    def try_add(t):
        nonlocal last_t
        if last_t is None or abs((t - last_t).days) >= min_sep_days:
            chosen.append(t); last_t = t
            return True
        return False

    active = True
    while active and len(chosen) < k:
        active = False
        for r in regs:
            if len(chosen) >= k:
                break
            if quota.get(r, 0) <= 0 or not buckets[r]:
                continue
            t = buckets[r].pop(0)
            if try_add(t):
                quota[r] -= 1
                active = True

    if len(chosen) < k:
        remaining = []
        for r in regs:
            remaining.extend(buckets[r])
        rank = {t: i for i, t in enumerate(cand_sorted)}
        remaining = sorted(set(remaining), key=lambda t: rank.get(t, 10**9))
        for t in remaining:
            if len(chosen) >= k:
                break
            try_add(t)

    return chosen[:k]

def regime_entropy_score(regimes_for_analogs: list) -> float:
    if not regimes_for_analogs:
        return 0.0
    s = pd.Series(regimes_for_analogs)
    counts = s.value_counts().values.astype(float)
    p = counts / counts.sum()
    ent = -np.sum(p * np.log(p + 1e-12))
    ent_max = np.log(len(counts) + 1e-12)
    return float(np.clip(ent / ent_max if ent_max > 0 else 0.0, 0, 1))

# =========================
# Stability / Fragility
# =========================
def stability_metrics(feat: pd.DataFrame, X: pd.DataFrame, regimes: pd.Series, now_idx: pd.Timestamp):
    ok_idx = X.index[X.notna().all(axis=1) & feat["bottom_confirm"].notna() & feat["px"].notna()]
    if len(ok_idx) < 30:
        return 0.0, True, []

    try:
        pos = ok_idx.get_loc(now_idx)
    except Exception:
        pos = len(ok_idx) - 1
        now_idx = ok_idx[pos]

    samples = []
    for shift in STAB_SHIFT_STEPS:
        p2 = pos + shift
        if p2 < 0 or p2 >= len(ok_idx):
            continue
        idx2 = ok_idx[p2]
        confirm2 = safe_float(feat.loc[idx2, "bottom_confirm"])
        if not np.isfinite(confirm2):
            continue

        for k in STAB_K_SET:
            units = []
            weights = []
            for h, w in HORIZON_WEIGHTS.items():
                y = feat.get(f"fwd_{h}", None)
                if y is None:
                    continue
                analog_idx = select_analogs_regime_balanced(X, y, regimes, idx2, k=k, min_sep_days=ANALOG_MIN_SEP_DAYS)
                vals = y.loc[analog_idx].dropna().values.astype(float)
                s = summarize(vals)
                if s["n"] < ANALOG_MIN:
                    continue
                units.append(horizon_unit(confirm2, s, n_target=k))
                weights.append(w)

            if len(units) == 0:
                continue

            wsum = float(np.sum(weights))
            comp = float(np.dot(units, weights) / wsum) if wsum > 0 else float(np.mean(units))
            samples.append(100.0 * comp)

    if len(samples) < 6:
        return 0.0, True, samples

    samples = np.array(samples, dtype=float)
    mean = float(np.mean(samples))
    std  = float(np.std(samples))
    mn   = float(np.min(samples))

    disp = np.clip(std / STAB_STD_SCALE_POINTS, 0, 1)
    worst_pen = 0.0 if mean <= 1e-9 else np.clip((STAB_MIN_MEAN_RATIO - (mn / mean)) / STAB_MIN_MEAN_RATIO, 0, 1)
    stab = 100.0 * (1.0 - 0.75*disp - 0.25*worst_pen)
    stab = float(np.clip(stab, 0, 100))

    fragile = (stab < 60) or (std > STAB_STD_SCALE_POINTS) or (mean > 1e-9 and (mn / mean) < STAB_MIN_MEAN_RATIO)
    return stab, fragile, samples.tolist()

# =========================
# Confidence + Risk labels
# =========================
def compute_confidence(n_eff: int, k_target: int, market_variety: float, stability: float) -> float:
    conf_n = np.clip(n_eff / max(1, k_target), 0, 1)
    conf_m = np.clip(market_variety, 0, 1)
    conf_s = np.clip(stability / 100.0, 0, 1)
    conf = 100.0 * (0.45*conf_n + 0.30*conf_m + 0.25*conf_s)
    return float(np.clip(conf, 0, 100))

def risk_label(h_summaries: dict, beta: float):
    p10s = []
    for h, s in h_summaries.items():
        if s.get("n", 0) > 0:
            p10s.append(s["p10"])
    if not p10s:
        return "Unknown"

    worst_p10 = float(np.min(p10s))
    if np.isfinite(worst_p10) and worst_p10 < -0.45:
        return "Big downside possible"
    if np.isfinite(beta) and beta > 1.35:
        return "Moves a lot with the market"
    if np.isfinite(worst_p10) and worst_p10 > -0.12:
        return "Historically more resilient"
    if np.isfinite(worst_p10) and worst_p10 < -0.30:
        return "Choppy / volatile"
    return "Moderate"

# =========================
# Baseline stats (ALL days)
# =========================
def baseline_stats(feat: pd.DataFrame):
    """Per-horizon baseline stats over ALL valid historical days.
    Returns keys used by the website evidence block: win_norm / med_norm / p10_norm / n_norm.
    """
    out = {}
    for h in HORIZONS_DAYS.keys():
        y = feat.get(f"fwd_{h}", None)
        if y is None:
            continue
        valid = y.notna()
        vals = y.loc[valid].values.astype(float)
        if len(vals) == 0:
            continue
        s = summarize(vals)
        out[h] = {
            "win_norm": float(np.mean(vals > 0)),
            "med_norm": float(np.median(vals)),
            "p10_norm": float(np.quantile(vals, 0.10)),
            "n_norm": int(len(vals)),
        }
    return out

# =========================
# Evidence A: pure washout
# =========================
def evidence_washout_light(feat: pd.DataFrame):
    """Top 10% Washout Meter days vs ALL days (baseline)."""
    out = {}
    wm = feat["washout_meter"].dropna()
    wm = wm[wm > 0]
    if len(wm) < 250:
        return out

    thr = float(wm.quantile(0.90))
    wash_mask = feat["washout_meter"] >= thr

    for h in HORIZONS_DAYS.keys():
        y = feat.get(f"fwd_{h}", None)
        if y is None:
            continue
        valid = y.notna()

        wash = wash_mask & valid
        normal = valid

        n_wash = int(wash.sum())
        n_norm = int(normal.sum())
        if n_wash < 50 or n_norm < 200:
            continue

        yw = y[wash].astype(float).values
        yn = y[normal].astype(float).values

        out[h] = {
            "thr_wash": thr,
            "n_wash": n_wash,
            "win_wash": float(np.mean(yw > 0)),
            "med_wash": float(np.median(yw)),
            "p10_wash": float(np.quantile(yw, 0.10)),
            "n_norm": n_norm,
            "win_norm": float(np.mean(yn > 0)),
            "med_norm": float(np.median(yn)),
            "p10_norm": float(np.quantile(yn, 0.10)),
        }
    return out

# =========================
# Evidence B: top FinalScore days
# =========================
def evidence_finalscore(feat: pd.DataFrame, final_score_series: pd.Series):
    """Top 10% FinalScore days vs ALL days (baseline)."""
    out = {}
    s = final_score_series.dropna()
    if len(s) < 250:
        return out

    thr = float(s.quantile(0.90))
    top_mask = final_score_series >= thr

    for h in HORIZONS_DAYS.keys():
        y = feat.get(f"fwd_{h}", None)
        if y is None:
            continue
        valid = y.notna()

        top = top_mask.reindex(feat.index).fillna(False) & valid
        normal = valid

        n_top = int(top.sum())
        n_norm = int(normal.sum())
        if n_top < 50 or n_norm < 200:
            continue

        yt = y[top].astype(float).values
        yn = y[normal].astype(float).values

        out[h] = {
            "thr_final": thr,
            "n_top": n_top,
            "win_top": float(np.mean(yt > 0)),
            "med_top": float(np.median(yt)),
            "p10_top": float(np.quantile(yt, 0.10)),
            "n_norm": n_norm,
            "win_norm": float(np.mean(yn > 0)),
            "med_norm": float(np.median(yn)),
            "p10_norm": float(np.quantile(yn, 0.10)),
        }
    return out

# =========================
# Explain (simple)
# =========================
def fmt_rank(p01: float) -> str:
    if not np.isfinite(p01):
        return "nan"
    p = p01 * 100.0
    if p < 1.0:
        return f"{p:.1f}%"
    return f"{p:.0f}%"

def build_explain(feat: pd.DataFrame, now_idx: pd.Timestamp) -> list:
    hist = feat.loc[:now_idx].copy()
    if len(hist) < 200:
        return ["Not enough price history to generate insights."]

    dd = safe_float(hist["dd_lt"].iloc[-1])        # drawdown from 1Y high (negative or zero)
    pos = safe_float(hist["pos_lt"].iloc[-1])       # position in 1Y range (0=bottom, 1=top)
    idio_dd = safe_float(hist["idio_dd_lt"].iloc[-1])  # stock-specific drawdown after removing market
    volz = safe_float(hist["volu_z"].iloc[-1])      # volume z-score
    atrp = safe_float(hist["atr_pct"].iloc[-1])     # ATR as % of price

    at_p  = percentile_rank(hist["atr_pct"], atrp)

    lines = []

    # 1. How far it has dropped — the single most important fact
    if np.isfinite(dd):
        pct = abs(dd)
        if pct >= 0.40:
            lines.append(f"Down <strong>{pct:.0%}</strong> from its 1-year high — a steep decline.")
        elif pct >= 0.20:
            lines.append(f"Down <strong>{pct:.0%}</strong> from its 1-year high — a significant pullback.")
        elif pct >= 0.05:
            lines.append(f"Down <strong>{pct:.0%}</strong> from its 1-year high.")
        else:
            lines.append(f"Only <strong>{pct:.0%}</strong> off its 1-year high — a small dip.")

    # 2. Stock-specific vs. market — is this the stock's problem or everything?
    if np.isfinite(idio_dd) and np.isfinite(dd) and abs(dd) >= 0.03:
        stock_pct = abs(idio_dd)
        total_pct = max(abs(dd), 0.01)
        ratio = stock_pct / total_pct
        if ratio > 0.70:
            lines.append("Most of this drop is <strong>specific to the stock</strong>, not the broader market.")
        elif ratio < 0.30:
            lines.append("The <strong>broader market</strong> is driving most of this decline — the stock itself is holding up.")
        else:
            lines.append("The decline is a <strong>mix</strong> of market-wide selling and stock-specific weakness.")

    # 3. Where it sits in its range — only mention if it's notable
    if np.isfinite(pos):
        if pos <= 0.10:
            lines.append("Trading near the <strong>lowest price</strong> of the past year.")
        elif pos <= 0.25:
            lines.append("In the <strong>lower quarter</strong> of its 1-year price range.")

    # 4. Volume spike — plain language, only if notable
    if np.isfinite(volz) and volz > 1.5:
        lines.append("<strong>Elevated trading volume</strong>, which can signal selling pressure is peaking.")

    # 5. Volatility — only if extreme
    if np.isfinite(at_p) and at_p > 0.90:
        lines.append("<strong>Unusually large daily price swings</strong> — high uncertainty, but also potential opportunity.")

    out = lines[:3]
    if not out:
        out = ["No major pullback signals right now — a mild or routine dip."]
    return out

# =========================
# FinalScore series (for charts/evidence)
# =========================
def compute_edge_score_at(feat: pd.DataFrame, X: pd.DataFrame, regimes: pd.Series, now_idx: pd.Timestamp, eligible=None) -> float:
    confirm = safe_float(feat.loc[now_idx, "bottom_confirm"])
    if not np.isfinite(confirm):
        return np.nan
    if now_idx not in X.index or (not X.loc[now_idx].notna().all()):
        return np.nan

    h_units = {}
    for h, _w in HORIZON_WEIGHTS.items():
        y = feat.get(f"fwd_{h}", None)
        if y is None:
            continue
        analog_idx = select_analogs_regime_balanced(X, y, regimes, now_idx, k=ANALOG_K, min_sep_days=ANALOG_MIN_SEP_DAYS, eligible=eligible)
        # Fallback: if the restricted pool is too thin, allow all eligible past days.
        if len(analog_idx) < ANALOG_MIN:
            analog_idx = select_analogs_regime_balanced(X, y, regimes, now_idx, k=ANALOG_K, min_sep_days=ANALOG_MIN_SEP_DAYS)
        vals = y.loc[analog_idx].dropna().values.astype(float)
        s = summarize(vals)
        if s.get("n", 0) >= ANALOG_MIN:
            h_units[h] = horizon_unit(confirm, s, n_target=ANALOG_K)

    if not h_units:
        return np.nan

    weights, units = [], []
    for h, unit in h_units.items():
        weights.append(HORIZON_WEIGHTS.get(h, 0.0))
        units.append(unit)
    wsum = float(np.sum(weights))
    comp_unit = float(np.dot(units, weights) / wsum) if wsum > 0 else float(np.mean(units))
    return float(100.0 * comp_unit)

def compute_final_score_series(
    feat: pd.DataFrame,
    X: pd.DataFrame,
    regimes: pd.Series,
    start_idx: pd.Timestamp,
    end_idx: pd.Timestamp,
    step_bars: int,
    eligible=None,
) -> pd.Series:
    ok_idx = X.index[X.notna().all(axis=1) & feat["bottom_confirm"].notna() & feat["px"].notna()]
    idx = ok_idx[(ok_idx >= start_idx) & (ok_idx <= end_idx)]
    if len(idx) < 50:
        return pd.Series(index=feat.index[(feat.index >= start_idx) & (feat.index <= end_idx)], dtype=float)

    sample_idx = idx[::max(1, int(step_bars))]
    vals = {}
    for t in sample_idx:
        edge = compute_edge_score_at(feat, X, regimes, t, eligible=eligible)
        w = safe_float(feat.loc[t, "washout_meter"])
        vals[t] = final_score(edge, w)

    s = pd.Series(vals).sort_index()
    full = pd.Series(index=feat.index[(feat.index >= start_idx) & (feat.index <= end_idx)], dtype=float)
    full.loc[s.index] = s.values
    return full.ffill()


def _compute_1y_winrate_at(feat: pd.DataFrame, X: pd.DataFrame, regimes: pd.Series,
                           now_idx: pd.Timestamp, eligible=None) -> float:
    """1-year win probability from analog matching at a specific historical point."""
    if now_idx not in X.index or (not X.loc[now_idx].notna().all()):
        return np.nan
    y = feat.get("fwd_1Y", None)
    if y is None:
        return np.nan
    analog_idx = select_analogs_regime_balanced(
        X, y, regimes, now_idx, k=ANALOG_K, min_sep_days=ANALOG_MIN_SEP_DAYS, eligible=eligible)
    if len(analog_idx) < ANALOG_MIN:
        analog_idx = select_analogs_regime_balanced(
            X, y, regimes, now_idx, k=ANALOG_K, min_sep_days=ANALOG_MIN_SEP_DAYS)
    vals = y.loc[analog_idx].dropna().values.astype(float)
    vals = np.where(np.isnan(vals), -1.0, vals)  # survivorship fix
    if len(vals) < ANALOG_MIN:
        return np.nan
    return float(np.mean(vals > 0))


def compute_conviction_series(
    feat: pd.DataFrame,
    X: pd.DataFrame,
    regimes: pd.Series,
    start_idx: pd.Timestamp,
    end_idx: pd.Timestamp,
    step_bars: int,
    quality: float,
    eligible=None,
) -> pd.Series:
    """Historical Opportunity Score (quality × 1Y prob) for chart shading.

    Uses today's quality (stable over chart window) and the 1Y win rate
    from analog matching at each sampled historical point — same formula
    as the Opportunity Score shown in the badge/table.
    """
    if not np.isfinite(quality):
        return pd.Series(index=feat.index[(feat.index >= start_idx) & (feat.index <= end_idx)], dtype=float)

    ok_idx = X.index[X.notna().all(axis=1) & feat["px"].notna()]
    idx = ok_idx[(ok_idx >= start_idx) & (ok_idx <= end_idx)]
    if len(idx) < 50:
        return pd.Series(index=feat.index[(feat.index >= start_idx) & (feat.index <= end_idx)], dtype=float)

    sample_idx = idx[::max(1, int(step_bars))]
    vals = {}
    for t in sample_idx:
        win_1y = _compute_1y_winrate_at(feat, X, regimes, t, eligible=eligible)
        if np.isfinite(win_1y):
            vals[t] = float(quality * win_1y)  # conviction = quality × 1Y win rate

    s = pd.Series(vals).sort_index()
    full = pd.Series(index=feat.index[(feat.index >= start_idx) & (feat.index <= end_idx)], dtype=float)
    full.loc[s.index] = s.values
    return full.ffill()

# =========================
# Build one ticker payload
# =========================
def score_one_ticker(t: str, O: pd.DataFrame, H: pd.DataFrame, L: pd.DataFrame, C: pd.DataFrame, V: pd.DataFrame, PX: pd.DataFrame,
                     spy_px: pd.Series, mkt: pd.DataFrame,
                     feature_cols: list, zwin: int):
    if t not in O.columns or t not in H.columns or t not in L.columns or t not in C.columns or t not in V.columns or t not in PX.columns:
        return None

    df = pd.DataFrame({
        "open": O[t],
        "high": H[t],
        "low":  L[t],
        "close": C[t],
        "volume": V[t],
        "px": PX[t],
    }).dropna(subset=["open", "high", "low", "close", "volume", "px"])

    if len(df) < MIN_HISTORY_BARS:
        return None

    # Liquidity filter
    med_dvol = (df["px"] * df["volume"]).rolling(LB_ST).median()
    if med_dvol.dropna().empty or float(med_dvol.dropna().iloc[-1]) < MIN_MED_DVOL_USD:
        return None

    # Missing check (strict recent window)
    recent = df.tail(MIN_HISTORY_BARS)
    miss_frac = 1.0 - (len(recent) / float(MIN_HISTORY_BARS))
    if miss_frac > MAX_MISSING_FRAC:
        return None

    feat = compute_core_features(df[["open", "high", "low", "close", "volume"]])
    feat["px"] = df["px"]

    # Align to SPY + market regime
    idx = feat.index.intersection(spy_px.index).intersection(mkt.index)
    if len(idx) < MIN_HISTORY_BARS:
        return None
    feat = feat.reindex(idx)
    spy_aligned = spy_px.reindex(idx)

    # Idiosyncratic features + market
    idio = compute_idiosyncratic_features(feat["px"], spy_aligned)
    feat = feat.join(idio, how="left")
    feat["bottom_confirm"] = compute_bottom_confirmation(feat["dd_lt"], feat["pos_lt"])
    feat = feat.join(mkt.reindex(idx), how="left")
    feat = feat.join(forward_returns(feat["px"]), how="left")
    feat["washout_meter"] = compute_washout_meter(feat)

    X = build_feature_matrix(feat, feature_cols, zwin=zwin)
    ok_now = X.notna().all(axis=1) & feat["bottom_confirm"].notna() & feat["px"].notna()
    if ok_now.sum() == 0:
        return None
    now_idx = ok_now[ok_now].index[-1]

    regimes = feat.apply(make_regime_bucket, axis=1)

    confirm_today = safe_float(feat.loc[now_idx, "bottom_confirm"])
    wash_today = safe_float(feat.loc[now_idx, "washout_meter"])
    beta_today = safe_float(feat.loc[now_idx, "beta"])

    # Gate: this scan is about washed-out setups (ALWAYS_PLOT + SPY bypasses)
    if MIN_WASHOUT_TODAY > 0 and (t not in ALWAYS_PLOT) and (t != BENCH):
        if (not np.isfinite(wash_today)) or (wash_today < MIN_WASHOUT_TODAY):
            return None

    # Quality gates
    stock_px = PX[t].dropna()
    tq = trend_quality(stock_px)
    rtr = recovery_track_record(stock_px)
    sd = selling_deceleration(stock_px)

    quality = 0.0
    quality_parts = {}
    if np.isfinite(tq):
        quality_parts["trend"] = tq
    if np.isfinite(rtr.get("recovery_rate", np.nan)):
        quality_parts["recovery"] = rtr["recovery_rate"] * 100
    if np.isfinite(sd):
        quality_parts["momentum"] = sd

    if quality_parts:
        weights = {"trend": 0.45, "recovery": 0.35, "momentum": 0.20}
        total_w = sum(weights[k] for k in quality_parts)
        quality = sum(quality_parts[k] * weights[k] for k in quality_parts) / total_w
    else:
        quality = np.nan

    # Candidate pool for analogs: restrict to this stock's top-decile historical Final Score days.
    # We compute a (sparsely sampled) historical Final Score series first, then take its 90th percentile cutoff.
    final_series_hist = compute_final_score_series(
        feat, X, regimes, start_idx=feat.index.min(), end_idx=now_idx, step_bars=EVID_SCORE_STEP_BARS
    )
    eligible_topdecile = None
    fs_vals = final_series_hist.dropna()
    if len(fs_vals) >= 100:
        thr90 = float(fs_vals.quantile(0.90))
        eligible_topdecile = (final_series_hist >= thr90)

# Analogs -> horizon summaries + units
    h_summaries = {}
    h_units = {}
    h_n = []
    analog_regimes_for_conf = None

    for h, _w in HORIZON_WEIGHTS.items():
        y = feat.get(f"fwd_{h}", None)
        if y is None:
            continue

        analog_idx = select_analogs_regime_balanced(X, y, regimes, now_idx, k=ANALOG_K, min_sep_days=ANALOG_MIN_SEP_DAYS, eligible=eligible_topdecile)
        # Fallback: if the restricted pool is too thin, allow all eligible past days.
        if len(analog_idx) < ANALOG_MIN:
            analog_idx = select_analogs_regime_balanced(X, y, regimes, now_idx, k=ANALOG_K, min_sep_days=ANALOG_MIN_SEP_DAYS)
        vals = y.loc[analog_idx].dropna().values.astype(float)
        s = summarize(vals)
        if s["n"] >= ANALOG_MIN:
            h_summaries[h] = s
            h_units[h] = horizon_unit(confirm_today, s, n_target=ANALOG_K)
            h_n.append(s["n"])
            if analog_regimes_for_conf is None and h == "1Y":
                analog_regimes_for_conf = regimes.loc[analog_idx].fillna("UNK").tolist()

    if not h_units:
        return None

    # Edge Score (0–100)
    weights, units = [], []
    for h, unit in h_units.items():
        weights.append(HORIZON_WEIGHTS.get(h, 0.0))
        units.append(unit)
    wsum = float(np.sum(weights))
    comp_unit = float(np.dot(units, weights) / wsum) if wsum > 0 else float(np.mean(units))
    edge_score = 100.0 * comp_unit

    # Final Score (0–100)
    final_today = final_score(edge_score, wash_today)
    if not np.isfinite(final_today):
        return None

    # Stability (on EdgeScore behavior)
    stab_score, fragile, stab_samples = stability_metrics(feat, X, regimes, now_idx)

    # Market variety (entropy of analog regimes)
    if analog_regimes_for_conf is None:
        h0 = list(h_summaries.keys())[0]
        y0 = feat[f"fwd_{h0}"]
        idx0 = select_analogs_regime_balanced(X, y0, regimes, now_idx, k=ANALOG_K, min_sep_days=ANALOG_MIN_SEP_DAYS)
        analog_regimes_for_conf = regimes.loc[idx0].fillna("UNK").tolist()
    market_variety = regime_entropy_score(analog_regimes_for_conf)

    # Confidence
    n_eff = int(np.median(h_n)) if len(h_n) else 0
    confidence = compute_confidence(n_eff, ANALOG_K, market_variety, stab_score)

    # Risk
    risk = risk_label(h_summaries, beta_today)

    # Verdict (FinalScore)
    verdict = verdict_line(final_today, confidence, stab_score, fragile)

    # Explain
    explain_lines = build_explain(feat, now_idx)

    # Baseline stats (ALL days) used by the website evidence block (A = analogs, B = normal baseline).
    ev_base = baseline_stats(feat)

    # Series payload for chart window
    cutoff = now_idx - pd.Timedelta(days=PLOT_LAST_DAYS)
    win = feat[(feat.index >= cutoff) & (feat.index <= now_idx)].copy()
    if win.empty:
        win = feat.loc[:now_idx].copy()
    px = win["px"].astype(float)
    wash = win["washout_meter"].astype(float)

    # Opportunity Score (conviction) series for chart shading — same formula as the badge.
    conv_series_window = compute_conviction_series(
        feat, X, regimes, start_idx=cutoff, end_idx=now_idx, step_bars=PLOT_SCORE_STEP_BARS,
        quality=quality, eligible=eligible_topdecile,
    )
    conv_win = conv_series_window.reindex(win.index).ffill().fillna(0).astype(float)

    # Top-% washed-out (readable): "Top X%" of days by Washout Meter
    wtop = washout_top_pct(feat["washout_meter"].dropna(), wash_today)

    detail = {
        "ticker": t,
        "as_of": str(now_idx),
        "verdict": verdict.replace("Verdict: ", ""),
        "final_score": float(final_today),
        "edge_score": float(edge_score),
        "washout_today": float(wash_today) if np.isfinite(wash_today) else None,
        "confidence": float(confidence),
        "stability": float(stab_score),
        "fragile": bool(fragile),
        "risk": risk,
        "similar_cases": int(n_eff),
        "explain": explain_lines,
        "outcomes": {h: h_summaries.get(h, {"n": 0}) for h in ["1Y", "3Y", "5Y"]},
        "evidence_baseline": ev_base,
        
        "series": {
            "dates": [str(x.date()) for x in px.index],
            "prices": [json_float(v) for v in px.values],
            "wash": [json_float(v) for v in wash.values],
            "final": [json_float(v) for v in conv_win.values],
        },
        "stability_samples": stab_samples,
        "quality": float(quality) if np.isfinite(quality) else None,
        "quality_parts": sanitize_for_json(quality_parts),
        "recovery_history": sanitize_for_json(rtr),
    }

    row = {
        "ticker": t,
        "verdict": detail["verdict"],
        "final_score": float(final_today),
        "edge_score": float(edge_score),
        "washout_today": float(wash_today) if np.isfinite(wash_today) else None,
        "washout_top_pct": float(wtop) if np.isfinite(wtop) else None,
        "confidence": float(confidence),
        "stability": float(stab_score),
        "fragile": bool(fragile),
        "risk": risk,
        "similar_cases": int(n_eff),
        "typical": {
            "1Y": h_summaries.get("1Y", {}).get("median", None),
            "3Y": h_summaries.get("3Y", {}).get("median", None),
            "5Y": h_summaries.get("5Y", {}).get("median", None),
        },
        "prob_1y": float(h_summaries.get("1Y", {}).get("win", 0) * 100) if "1Y" in h_summaries else None,
        "prob_3y": float(h_summaries.get("3Y", {}).get("win", 0) * 100) if "3Y" in h_summaries else None,
        "prob_5y": float(h_summaries.get("5Y", {}).get("win", 0) * 100) if "5Y" in h_summaries else None,
        "quality": float(quality) if np.isfinite(quality) else None,
        "conviction": float(quality * h_summaries.get("1Y", {}).get("win", 0)) if (np.isfinite(quality) and "1Y" in h_summaries) else None,
        "median_1y": h_summaries.get("1Y", {}).get("median", None),
        "median_3y": h_summaries.get("3Y", {}).get("median", None),
        "median_5y": h_summaries.get("5Y", {}).get("median", None),
        "downside_1y": h_summaries.get("1Y", {}).get("p10", None),
        "n_analogs": int(n_eff),
    }

    return row, detail

# =========================
# MAIN
# =========================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(TICKER_DIR, exist_ok=True)

    if not should_run_now():
        print("[gate] Not time to run yet (or already ran today). Exiting.")
        return

    # Optional manual add-ons: EXTRA_TICKERS="TSM,UBER" etc.
    extra = os.getenv("EXTRA_TICKERS", "").strip()
    extra_tickers = []
    if extra:
        extra_tickers = [x.strip().upper().replace(".", "-") for x in extra.split(",") if x.strip()]

    print("=" * 110)
    print("REBOUND SCANNER (v8) — FinalScore everywhere + two evidence blocks")
    print("=" * 110)

    # Universe
    tickers = fetch_ishares_holdings_tickers(ISHARES_HOLDINGS_URL)
    if MAX_TICKERS is not None:
        tickers = tickers[:int(MAX_TICKERS)]
    universe = sorted(set(tickers + ALWAYS_PLOT + [BENCH] + extra_tickers))
    print(f"[UNIVERSE] holdings={len(tickers)} | extra={len(extra_tickers)} | universe={len(universe)}")

    print(f"[DATA] Downloading {INTERVAL} OHLCV for {len(universe)} tickers...")
    data = download_ohlcv_period(universe, period=PERIOD, interval=INTERVAL, chunk_size=CHUNK_SIZE)
    O, H, L, C, V, A = data["Open"], data["High"], data["Low"], data["Close"], data["Volume"], data["AdjClose"]
    if C.empty or BENCH not in C.columns:
        raise RuntimeError("Missing price data or SPY missing.")

    PX = A if (not A.empty and BENCH in A.columns) else C

    # Market regime features (SPY)
    spy_px = PX[BENCH].dropna()
    spy_h = H[BENCH].reindex(spy_px.index).dropna()
    spy_l = L[BENCH].reindex(spy_px.index).dropna()
    spy_px = spy_px.reindex(spy_h.index).reindex(spy_l.index).dropna()
    mkt = compute_market_regime(spy_px, spy_h, spy_l)

    usable = [t for t in universe if t in O.columns and t in H.columns and t in L.columns and t in V.columns and t in PX.columns]
    print(f"[DATA] usable tickers={len(usable)}/{len(universe)}")

    feature_cols = [
        "dd_lt","pos_lt","dd_st","pos_st","atr_pct","volu_z","gap","trend_st",
        "idio_dd_lt","idio_pos_lt","idio_dd_st","idio_pos_st",
        "mkt_trend","mkt_vol","mkt_dd","mkt_atr_pct",
    ]
    zwin = max(63, LB_ST)

    rows = []
    details = {}

    # Clear previous ticker details for a clean publish
    for fn in os.listdir(TICKER_DIR):
        if fn.endswith(".json"):
            try:
                os.remove(os.path.join(TICKER_DIR, fn))
            except Exception:
                pass

    for i, t in enumerate(usable, start=1):
        out = score_one_ticker(t, O, H, L, C, V, PX, spy_px, mkt, feature_cols, zwin)
        if out is None:
            continue
        row, det = out
        rows.append(row)

        # Write per-ticker detail
        with open(os.path.join(TICKER_DIR, f"{t}.json"), "w") as f:
            json.dump(sanitize_for_json(det), f)

        if i % 50 == 0:
            print(f"[PROGRESS] processed {i}/{len(usable)} | scored={len(rows)}")

    if not rows:
        print("No tickers scored. Try lowering MIN_MED_DVOL_USD or MIN_WASHOUT_TODAY.")
        return

    res = pd.DataFrame(rows)
    res = res.sort_values(["conviction", "prob_1y"], ascending=[False, False]).reset_index(drop=True)

    # Embed top10 details in full.json (fewer network calls)
    top10 = res.head(TOP10_EMBED)["ticker"].tolist()
    for t in top10:
        try:
            with open(os.path.join(TICKER_DIR, f"{t}.json"), "r") as f:
                details[t] = json.load(f)
        except Exception:
            pass

    # Full payload
    as_of = str(datetime.now(ZoneInfo("America/New_York")))
    payload = {
        "as_of": as_of,
        "model": {
            "version": "v8",
            "bench": BENCH,
            "interval": INTERVAL,
            "universe": "iShares Top 20 U.S. Stocks ETF holdings + ALWAYS_PLOT",
            "min_washout_today": MIN_WASHOUT_TODAY,
            "final_score": {
                "wash_floor": FINAL_WASH_FLOOR,
                "wash_weight": FINAL_WASH_WEIGHT,
            },
        },
        "items": res.to_dict(orient="records"),
        "details": details,
    }

    with open(os.path.join(OUT_DIR, "full.json"), "w") as f:
        json.dump(sanitize_for_json(payload), f)

    mark_ran_today()
    print(f"[OK] Wrote {len(rows)} tickers -> docs/data (as_of={as_of})")


if __name__ == "__main__":
    main()
