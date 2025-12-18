#!/usr/bin/env python3
# ============================================================
# CRT REBOUND SCANNER — v7 (website generator)
# ------------------------------------------------------------
# Website-facing outputs (no plotting):
#   - docs/data/full.json                (ranking table + top10 embedded details)
#   - docs/data/tickers/<TICKER>.json    (per-ticker details)
#
# Model notes (matches the user's baseline v7):
#   - Washout Meter is a stock-specific "how washed-out" score (0..100)
#   - Rebound Score is based on what happened after similar past setups
#   - Evidence compares:
#       • Washout days = top 10% Washout Meter days for THIS stock
#       • Normal days  = ALL days for THIS stock (baseline)
# ============================================================

import os
import json
import math
import warnings
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf
import requests
from io import StringIO

warnings.filterwarnings("ignore")

# =========================
# CONFIG (KEEP SIMPLE)
# =========================
INTERVAL = "1d"
PERIOD   = "max"
BENCH = "SPY"

ISHARES_HOLDINGS_URL = (
    "https://www.ishares.com/us/products/339779/ishares-top-20-u-s-stocks-etf/"
    "1467271812596.ajax?fileType=csv&fileName=holdings&dataType=fund"
)

ALWAYS_PLOT = ["SPY", "QQQ", "IWM", "DIA", "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA"]

CHUNK_SIZE = 80
MAX_TICKERS = None          # e.g. 600 for speed, else None

# Top10 cards embed (reduces network calls)
TOP10_EMBED = 10

# Store up to 6 years of daily points for the website chart
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

# Verdict thresholds (single-line)
VERDICT = dict(
    STRONG_SCORE=72.0,
    OK_SCORE=60.0,
    HIGH_CONF=72.0,
    OK_CONF=60.0,
    HIGH_STAB=72.0,
    OK_STAB=60.0,
)

# Outputs
OUT_DIR = os.path.join("docs", "data")
TICKER_DIR = os.path.join(OUT_DIR, "tickers")

# =========================
# Time gate (after 5pm NY)
# =========================
def should_run_now() -> bool:
    # Manual runs can set FORCE_RUN=1
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


def fmt_rank(p01: float) -> str:
    """Percentile formatting that never shows confusing '0%'."""
    if not np.isfinite(p01):
        return "nan"
    p = p01 * 100.0
    if p < 1.0:
        return f"{p:.1f}%"
    return f"{p:.0f}%"


def percentile_rank(series: pd.Series, value: float) -> float:
    s = series.dropna().values.astype(float)
    if len(s) < 30 or not np.isfinite(value):
        return np.nan
    return float(np.mean(s <= value))


def top_pct_most_extreme_high(series: pd.Series, value: float) -> float:
    """Return "top X%" where *higher* is more extreme; smaller is better (e.g. 2.3 means Top 2.3%)."""
    p = percentile_rank(series, value)
    if not np.isfinite(p):
        return np.nan
    return float((1.0 - p) * 100.0)


def md_bold_to_html(s: str) -> str:
    # Minimal conversion for the explain lines ("**bold**" -> "<strong>bold</strong>")
    out = str(s)
    while "**" in out:
        a = out.find("**")
        b = out.find("**", a + 2)
        if b < 0:
            break
        inner = out[a + 2 : b]
        out = out[:a] + f"<strong>{inner}</strong>" + out[b + 2 :]
    return out


# =========================
# Verdict (single-line)
# =========================
def verdict_line(score: float, confidence: float, stability: float, fragile: bool) -> str:
    s = safe_float(score)
    c = safe_float(confidence)
    st = safe_float(stability)

    if not np.isfinite(s) or not np.isfinite(c) or not np.isfinite(st):
        return "Verdict: Not enough data"

    strong = (s >= VERDICT["STRONG_SCORE"])
    ok     = (s >= VERDICT["OK_SCORE"])
    high_c = (c >= VERDICT["HIGH_CONF"])
    ok_c   = (c >= VERDICT["OK_CONF"])
    high_s = (st >= VERDICT["HIGH_STAB"])
    ok_s   = (st >= VERDICT["OK_STAB"])

    if strong and high_c and high_s and (not fragile):
        return "Verdict: Strong + stable"
    if strong and (ok_c or high_c) and (not high_s or fragile):
        return "Verdict: High-score but unstable"
    if ok and (not ok_c):
        return "Verdict: Promising but low confidence"
    if ok and ok_c and ok_s and fragile:
        return "Verdict: Mixed (watch stability)"
    if ok and ok_c:
        return "Verdict: Mixed"
    return "Verdict: Not compelling today"


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
    for c in ["open","high","low","close","volume"]:
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
    vals = vals.astype(float)
    return {
        "n": int(len(vals)),
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


def select_analogs_regime_balanced(
    X: pd.DataFrame,
    y: pd.Series,
    regimes: pd.Series,
    now_idx: pd.Timestamp,
    k: int,
    min_sep_days: int,
):
    cand = X.index[(X.notna().all(axis=1)) & (y.notna()) & (X.index < now_idx)]
    if len(cand) == 0:
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
# Evidence (explicit)
# =========================
def evidence_light(feat: pd.DataFrame):
    """
    Evidence compares:
      A) "Washout days" = TOP 10% of Washout Meter days for THIS stock
      B) "Normal days"  = ALL days (baseline)
    """
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
# Explain (simple)
# =========================
def build_explain(feat: pd.DataFrame, now_idx: pd.Timestamp) -> list:
    hist = feat.loc[:now_idx].copy()
    if len(hist) < 200:
        return ["Not enough history to explain clearly (need ~1 year+ of daily data)."]

    dd = safe_float(hist["dd_lt"].iloc[-1])
    pos = safe_float(hist["pos_lt"].iloc[-1])
    idio_dd = safe_float(hist["idio_dd_lt"].iloc[-1])
    volz = safe_float(hist["volu_z"].iloc[-1])
    atrp = safe_float(hist["atr_pct"].iloc[-1])

    dd_p  = percentile_rank(hist["dd_lt"], dd)
    pos_p = percentile_rank(hist["pos_lt"], pos)
    id_p  = percentile_rank(hist["idio_dd_lt"], idio_dd)
    vz_p  = percentile_rank(hist["volu_z"], volz)
    at_p  = percentile_rank(hist["atr_pct"], atrp)

    lines = []

    if np.isfinite(dd) and np.isfinite(dd_p):
        lines.append(
            (dd_p, f"Price is **{dd:.0%} below** its 1-year high (more extreme than **{fmt_rank(dd_p)}** of past days for this stock).")
        )

    if np.isfinite(pos) and np.isfinite(pos_p):
        lines.append(
            (1.0 - pos_p, f"Price sits in the **bottom {pos*100:.0f}%** of its 1-year range (only about **{fmt_rank(pos_p)}** of past days were this low-in-range or lower).")
        )

    if np.isfinite(idio_dd) and np.isfinite(id_p):
        lines.append(
            (id_p, f"Even after removing market moves, the stock looks weak: **stock-specific drawdown ~{idio_dd:.0%}** (more extreme than **{fmt_rank(id_p)}** of past days).")
        )

    if np.isfinite(volz) and np.isfinite(vz_p) and volz > 1.0:
        lines.append((vz_p, f"Trading volume is unusually high (higher than **{fmt_rank(vz_p)}** of past days)."))

    if np.isfinite(atrp) and np.isfinite(at_p) and at_p > 0.85:
        lines.append((at_p, f"Daily price swings are unusually large (bigger than **{fmt_rank(at_p)}** of past days)."))

    lines = sorted(lines, key=lambda x: x[0], reverse=True)
    out = [md_bold_to_html(txt) for _, txt in lines[:3]]
    if not out:
        out = ["Nothing is extremely washed-out today; this looks like a mild setup rather than a dramatic selloff."]
    return out


# =========================
# Scoring one ticker
# =========================
def score_one_ticker(t: str, O, H, L, C, V, PX, spy_px: pd.Series, mkt: pd.DataFrame, feature_cols: list, zwin: int):
    df = pd.DataFrame({
        "open": O[t],
        "high": H[t],
        "low":  L[t],
        "close": C[t],
        "volume": V[t],
        "px": PX[t],
    }).dropna(subset=["open","high","low","close","volume","px"])

    if len(df) < MIN_HISTORY_BARS:
        return None

    # Liquidity filter
    med_dvol = (df["px"] * df["volume"]).rolling(LB_ST).median()
    if med_dvol.dropna().empty or float(med_dvol.dropna().iloc[-1]) < MIN_MED_DVOL_USD:
        return None

    # Missing check
    recent = df.tail(MIN_HISTORY_BARS)
    miss_frac = 1.0 - (len(recent) / float(MIN_HISTORY_BARS))
    if miss_frac > MAX_MISSING_FRAC:
        return None

    feat = compute_core_features(df[["open","high","low","close","volume"]])
    feat["px"] = df["px"]

    # Align to SPY + market regime
    idx = feat.index.intersection(spy_px.index).intersection(mkt.index)
    if len(idx) < MIN_HISTORY_BARS:
        return None
    feat = feat.reindex(idx)
    spy_aligned = spy_px.reindex(idx)

    # Idiosyncratic features
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

    h_summaries = {}
    h_units = {}
    h_n = []
    analog_regimes_for_conf = None

    for h, w in HORIZON_WEIGHTS.items():
        y = feat.get(f"fwd_{h}", None)
        if y is None:
            continue

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

    # Composite rebound score
    weights, units = [], []
    for h, unit in h_units.items():
        weights.append(HORIZON_WEIGHTS.get(h, 0.0))
        units.append(unit)
    wsum = float(np.sum(weights))
    comp_unit = float(np.dot(units, weights) / wsum) if wsum > 0 else float(np.mean(units))
    rebound_score = 100.0 * comp_unit

    # Stability
    stab_score, fragile, stab_samples = stability_metrics(feat, X, regimes, now_idx)

    # Market variety
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

    # Verdict
    verdict = verdict_line(rebound_score, confidence, stab_score, fragile)

    # Explain + Evidence
    explain_lines = build_explain(feat, now_idx)
    ev = evidence_light(feat)

    # Washout rank + top10 range
    wm_hist = feat["washout_meter"].dropna()
    wash_top_pct = top_pct_most_extreme_high(wm_hist, wash_today)
    w90 = float(wm_hist.quantile(0.90)) if len(wm_hist) >= 60 else np.nan
    wmx = float(wm_hist.max()) if len(wm_hist) >= 60 else np.nan

    # Series for chart (last ~6y)
    cutoff = now_idx - pd.Timedelta(days=PLOT_LAST_DAYS)
    dplot = feat.loc[feat.index >= cutoff].copy()
    if dplot.empty:
        dplot = feat.copy()
    dplot = dplot.dropna(subset=["px"]).copy()
    series = {
        "dates": [str(x.date()) for x in dplot.index],
        "prices": [float(x) if np.isfinite(x) else None for x in dplot["px"].astype(float).values],
        "wash": [float(x) if np.isfinite(x) else 0.0 for x in pd.to_numeric(dplot["washout_meter"], errors="coerce").fillna(0.0).clip(0,100).values],
    }

    # Outcomes block (analog summaries)
    outcomes = {}
    for hh in ["1Y","3Y","5Y"]:
        if hh in h_summaries:
            s = h_summaries[hh]
            outcomes[hh] = {
                "n": int(s["n"]),
                "win": float(s["win"]),
                "median": float(s["median"]),
                "p10": float(s["p10"]),
                "p90": float(s["p90"]),
            }

    row = {
        "ticker": t,
        "verdict": verdict.replace("Verdict: ", ""),
        "rebound_score": float(rebound_score),
        "confidence": float(confidence),
        "stability": float(stab_score),
        "fragile": bool(fragile),
        "risk": str(risk),
        "washout_today": float(wash_today) if np.isfinite(wash_today) else None,
        "washout_top_pct": float(wash_top_pct) if np.isfinite(wash_top_pct) else None,
        "washout_top10_lo": float(w90) if np.isfinite(w90) else None,
        "washout_top10_hi": float(wmx) if np.isfinite(wmx) else None,
        "similar_cases": int(n_eff),
        "as_of": str(now_idx),
        # convenient table fields
        "y1_typical": float(h_summaries.get("1Y", {}).get("median", np.nan)),
        "y3_typical": float(h_summaries.get("3Y", {}).get("median", np.nan)),
        "y5_typical": float(h_summaries.get("5Y", {}).get("median", np.nan)),
    }

    det = {
        "ticker": t,
        "as_of": str(now_idx),
        "verdict": row["verdict"],
        "rebound_score": row["rebound_score"],
        "confidence": row["confidence"],
        "stability": row["stability"],
        "fragile": row["fragile"],
        "risk": row["risk"],
        "washout_today": row["washout_today"],
        "washout_top_pct": row["washout_top_pct"],
        "washout_top10_lo": row["washout_top10_lo"],
        "washout_top10_hi": row["washout_top10_hi"],
        "similar_cases": row["similar_cases"],
        "explain": explain_lines,
        "outcomes": outcomes,
        "evidence": ev,
        "stability_samples": stab_samples,
        "series": series,
    }

    # Most recent historical "washout day" for this ticker (top 10% washout), requiring >=1Y realized
    hist_signal = None
    try:
        wm = feat["washout_meter"].dropna()
        if len(wm) >= 260:
            thr = float(wm.quantile(0.90))
            px = feat["px"].astype(float)
            # search backwards for last day with washout>=thr and 1Y forward price exists
            idxs = wm.index
            # map index position for forward lookup
            pos_map = {ts: i for i, ts in enumerate(px.index)}
            for ts in reversed(list(idxs)):
                if not (np.isfinite(wm.loc[ts]) and wm.loc[ts] >= thr):
                    continue
                i0 = pos_map.get(ts, None)
                if i0 is None:
                    continue
                p0 = safe_float(px.iloc[i0])
                if not np.isfinite(p0) or p0 <= 0:
                    continue
                # require >=1Y forward
                if i0 + HORIZONS_DAYS["1Y"] >= len(px):
                    continue
                p1 = safe_float(px.iloc[i0 + HORIZONS_DAYS["1Y"]])
                r1 = (p1 / p0 - 1.0) if np.isfinite(p1) else np.nan
                r3 = np.nan
                r5 = np.nan
                if i0 + HORIZONS_DAYS["3Y"] < len(px):
                    p3 = safe_float(px.iloc[i0 + HORIZONS_DAYS["3Y"]])
                    if np.isfinite(p3):
                        r3 = (p3 / p0 - 1.0)
                if i0 + HORIZONS_DAYS["5Y"] < len(px):
                    p5 = safe_float(px.iloc[i0 + HORIZONS_DAYS["5Y"]])
                    if np.isfinite(p5):
                        r5 = (p5 / p0 - 1.0)

                top_pct = top_pct_most_extreme_high(wm, safe_float(wm.loc[ts]))
                hist_signal = {
                    "date": str(ts.date()),
                    "ticker": t,
                    "wash": float(wm.loc[ts]),
                    "wash_rank": (f"Top {top_pct:.1f}%" if np.isfinite(top_pct) and top_pct < 1.0 else (f"Top {top_pct:.0f}%" if np.isfinite(top_pct) else "—")),
                    "r1": float(r1) if np.isfinite(r1) else None,
                    "r3": float(r3) if np.isfinite(r3) else None,
                    "r5": float(r5) if np.isfinite(r5) else None,
                }
                break
    except Exception:
        hist_signal = None

    return row, det, hist_signal


def main():
    if not should_run_now():
        print("[SKIP] Not after 5pm New York time, or already ran today. Set FORCE_RUN=1 to override.")
        return

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(TICKER_DIR, exist_ok=True)

    tickers = fetch_ishares_holdings_tickers(ISHARES_HOLDINGS_URL)
    if MAX_TICKERS is not None:
        tickers = tickers[:int(MAX_TICKERS)]
    universe = sorted(set(tickers + ALWAYS_PLOT + [BENCH]))

    print(f"[UNIVERSE] holdings={len(tickers)} | universe={len(universe)}")
    print(f"[DATA] Downloading {INTERVAL} OHLCV for {len(universe)} tickers ...")
    data = download_ohlcv_period(universe, period=PERIOD, interval=INTERVAL, chunk_size=CHUNK_SIZE)
    O, H, L, C, V, A = data["Open"], data["High"], data["Low"], data["Close"], data["Volume"], data["AdjClose"]
    if C.empty or BENCH not in C.columns:
        raise RuntimeError("Missing price data or SPY missing.")

    PX = A if (not A.empty and BENCH in A.columns) else C

    # Market regime features
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
    embedded_details = {}
    hist_best_by_ticker = {}

    # Clear previous per-ticker details for a clean publish
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
        row, det, hist_sig = out
        rows.append(row)

        # Write per-ticker detail JSON
        with open(os.path.join(TICKER_DIR, f"{t}.json"), "w") as f:
            json.dump(det, f)

        if hist_sig is not None:
            hist_best_by_ticker[t] = hist_sig

        if i % 50 == 0:
            print(f"[PROGRESS] processed {i}/{len(usable)} | scored={len(rows)}")

    if not rows:
        print("No tickers scored. Try lowering MIN_MED_DVOL_USD.")
        return

    res = pd.DataFrame(rows)
    res = res.sort_values(["rebound_score", "confidence", "stability"], ascending=False).reset_index(drop=True)

    # Embed top10 details
    top10 = res.head(TOP10_EMBED)["ticker"].tolist()
    for t in top10:
        try:
            with open(os.path.join(TICKER_DIR, f"{t}.json"), "r") as f:
                embedded_details[t] = json.load(f)
        except Exception:
            pass

    # Historical signals: last 10 most recent (1 per ticker)
    hist = list(hist_best_by_ticker.values())
    hist.sort(key=lambda x: (x.get("date", ""), x.get("ticker", "")), reverse=True)
    hist = hist[:10]

    as_of = str(datetime.now(ZoneInfo("America/New_York")))
    payload = {
        "as_of": as_of,
        "model": {
            "version": "v7",
            "bench": BENCH,
            "interval": INTERVAL,
            "universe": "iShares Top 20 U.S. Stocks ETF holdings + ALWAYS_PLOT",
            "evidence": {
                "washout_days": "Top 10% Washout Meter days for this stock",
                "normal_days": "All historical days for this stock",
            },
        },
        "items": res.to_dict(orient="records"),
        "details": embedded_details,
        "historical_signals": hist,
    }

    with open(os.path.join(OUT_DIR, "full.json"), "w") as f:
        json.dump(payload, f)

    mark_ran_today()
    print(f"[OK] Wrote {len(res)} tickers -> docs/data (as_of={as_of})")


if __name__ == "__main__":
    main()
