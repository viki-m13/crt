#!/usr/bin/env python3
# ============================================================
# CRT REBOUND SCANNER — Static Precompute (GitHub Pages)
# Option A: precompute daily universe + embed chart series for all
# Output: docs/data/full.json
# Runs once/day after ~5PM New York time (when RUN_MODE=scheduled)
# ============================================================

import os, math, json, warnings
warnings.filterwarnings("ignore")

from io import StringIO
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf
import requests

# =========================
# CONFIG
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
MAX_TICKERS = None          # keep None to run full holdings universe
PLOT_LAST_DAYS = 365 * 6

# Lookbacks (days)
LB_LT = 252
LB_ST = 63
BETA_LB = 126
ATR_N = 14

DD_THR  = 0.25
POS_THR = 0.20
GATE_DD_SCALE  = 0.12
GATE_POS_SCALE = 0.10

MIN_MED_DVOL_USD = 5_000_000
MAX_MISSING_FRAC = 0.10
MIN_HISTORY_BARS = LB_LT + BETA_LB + LB_ST + 220

ANALOG_K = 250
ANALOG_MIN = 80
ANALOG_MIN_SEP_DAYS = 10

STAB_K_SET = [150, 250, 350]
STAB_SHIFT_STEPS = [0, -5, -10, -15]
STAB_STD_SCALE_POINTS = 12.0
STAB_MIN_MEAN_RATIO = 0.60

HORIZONS_DAYS = {"1Y": 252, "3Y": 252*3, "5Y": 252*5}
HORIZON_WEIGHTS = {"1Y": 0.50, "3Y": 0.35, "5Y": 0.15}

VERDICT = dict(
    STRONG_SCORE=72.0,
    OK_SCORE=60.0,
    HIGH_CONF=72.0,
    OK_CONF=60.0,
    HIGH_STAB=72.0,
    OK_STAB=60.0,
)

NY = ZoneInfo("America/New_York")

# =========================
# Time gate
# =========================
def should_run_today():
    run_mode = os.getenv("RUN_MODE", "scheduled").lower().strip()
    if run_mode == "manual":
        return True

    now_ny = datetime.now(tz=NY)
    if now_ny.hour < 17:
        return False

    stamp_path = "docs/data/last_run.json"
    if os.path.exists(stamp_path):
        try:
            j = json.load(open(stamp_path, "r"))
            if j.get("date_ny") == now_ny.strftime("%Y-%m-%d"):
                return False
        except Exception:
            pass
    return True

def write_run_stamp():
    now_ny = datetime.now(tz=NY)
    os.makedirs("docs/data", exist_ok=True)
    json.dump(
        {"date_ny": now_ny.strftime("%Y-%m-%d"), "asof": now_ny.strftime("%Y-%m-%d %H:%M")},
        open("docs/data/last_run.json", "w"),
        indent=2
    )

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
    if not np.isfinite(p01):
        return "nan"
    p = p01 * 100.0
    return f"{p:.1f}%" if p < 1.0 else f"{p:.0f}%"

def percentile_rank(series: pd.Series, value: float) -> float:
    s = series.dropna().values.astype(float)
    if len(s) < 30 or not np.isfinite(value):
        return np.nan
    return float(np.mean(s <= value))

def verdict_line(score: float, confidence: float, stability: float, fragile: bool) -> str:
    s = safe_float(score); c = safe_float(confidence); st = safe_float(stability)
    if not np.isfinite(s) or not np.isfinite(c) or not np.isfinite(st):
        return "Not enough data"

    strong = (s >= VERDICT["STRONG_SCORE"])
    ok     = (s >= VERDICT["OK_SCORE"])
    high_c = (c >= VERDICT["HIGH_CONF"])
    ok_c   = (c >= VERDICT["OK_CONF"])
    high_s = (st >= VERDICT["HIGH_STAB"])
    ok_s   = (st >= VERDICT["OK_STAB"])

    if strong and high_c and high_s and (not fragile):
        return "Strong + stable"
    if strong and (ok_c or high_c) and (not high_s or fragile):
        return "High-score but unstable"
    if ok and (not ok_c):
        return "Promising but low confidence"
    if ok and ok_c and ok_s and fragile:
        return "Mixed (watch stability)"
    if ok and ok_c:
        return "Mixed"
    return "Not compelling today"

# =========================
# Holdings
# =========================
def fetch_ishares_holdings_tickers(url: str) -> list:
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(url, headers=headers, timeout=60)
    if resp.status_code != 200 or not resp.content:
        raise RuntimeError(f"Holdings download failed (HTTP {resp.status_code}).")

    raw_text = resp.content.decode("utf-8", errors="ignore")
    lines = raw_text.splitlines()

    header_idx = None
    for i, line in enumerate(lines[:1200]):
        if line.strip().replace('"', '').startswith("Ticker,"):
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

    if len(keep) < 10:
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
# Feature engineering
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

def select_analogs_regime_balanced(X: pd.DataFrame, y: pd.Series, regimes: pd.Series,
                                  now_idx: pd.Timestamp, k: int, min_sep_days: int):
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
# Confidence + Risk
# =========================
def compute_confidence(n_eff: int, k_target: int, market_variety: float, stability: float) -> float:
    conf_n = np.clip(n_eff / max(1, k_target), 0, 1)
    conf_m = np.clip(market_variety, 0, 1)
    conf_s = np.clip(stability / 100.0, 0, 1)
    conf = 100.0 * (0.45*conf_n + 0.30*conf_m + 0.25*conf_s)
    return float(np.clip(conf, 0, 100))

def risk_label(h_summaries: dict, beta: float):
    p10s = []
    for _, s in h_summaries.items():
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
# Explain
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
        lines.append((dd_p, f"Price is {dd:.0%} below its 1-year high (more extreme than {fmt_rank(dd_p)} of past days)."))
    if np.isfinite(pos) and np.isfinite(pos_p):
        lines.append((1.0 - pos_p, f"Price sits in the bottom {pos*100:.0f}% of its 1-year range (only about {fmt_rank(pos_p)} of past days were this low-in-range or lower)."))
    if np.isfinite(idio_dd) and np.isfinite(id_p):
        lines.append((id_p, f"After removing market moves: stock-specific drawdown ~{idio_dd:.0%} (more extreme than {fmt_rank(id_p)} of past days)."))
    if np.isfinite(volz) and np.isfinite(vz_p) and volz > 1.0:
        lines.append((vz_p, f"Volume is unusually high (higher than {fmt_rank(vz_p)} of past days)."))
    if np.isfinite(atrp) and np.isfinite(at_p) and at_p > 0.85:
        lines.append((at_p, f"Daily price swings are unusually large (bigger than {fmt_rank(at_p)} of past days)."))

    lines = sorted(lines, key=lambda x: x[0], reverse=True)
    out = [txt for _, txt in lines[:3]]
    return out if out else ["Nothing is extremely washed-out today; this looks mild rather than dramatic."]

# =========================
# MAIN
# =========================
def main():
    if not should_run_today():
        print("Skip: not in run window or already ran today.")
        return

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

    results = []
    for i, t in enumerate(usable, start=1):
        df = pd.DataFrame({
            "open": O[t],
            "high": H[t],
            "low":  L[t],
            "close": C[t],
            "volume": V[t],
            "px": PX[t],
        }).dropna(subset=["open","high","low","close","volume","px"])

        if len(df) < MIN_HISTORY_BARS:
            continue

        med_dvol = (df["px"] * df["volume"]).rolling(LB_ST).median()
        if med_dvol.dropna().empty or float(med_dvol.dropna().iloc[-1]) < MIN_MED_DVOL_USD:
            continue

        recent = df.tail(MIN_HISTORY_BARS)
        miss_frac = 1.0 - (len(recent) / float(MIN_HISTORY_BARS))
        if miss_frac > MAX_MISSING_FRAC:
            continue

        feat = compute_core_features(df[["open","high","low","close","volume"]])
        feat["px"] = df["px"]

        idx = feat.index.intersection(spy_px.index).intersection(mkt.index)
        if len(idx) < MIN_HISTORY_BARS:
            continue

        feat = feat.reindex(idx)
        spy_aligned = spy_px.reindex(idx)

        idio = compute_idiosyncratic_features(feat["px"], spy_aligned)
        feat = feat.join(idio, how="left")

        feat["bottom_confirm"] = compute_bottom_confirmation(feat["dd_lt"], feat["pos_lt"])
        feat = feat.join(mkt.reindex(idx), how="left")
        feat = feat.join(forward_returns(feat["px"]), how="left")
        feat["washout_meter"] = compute_washout_meter(feat)

        X = build_feature_matrix(feat, feature_cols, zwin=zwin)

        ok_now = X.notna().all(axis=1) & feat["bottom_confirm"].notna() & feat["px"].notna()
        if ok_now.sum() == 0:
            continue
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
            continue

        weights, units = [], []
        for hh, unit in h_units.items():
            weights.append(HORIZON_WEIGHTS.get(hh, 0.0))
            units.append(unit)
        wsum = float(np.sum(weights))
        comp_unit = float(np.dot(units, weights) / wsum) if wsum > 0 else float(np.mean(units))
        rebound_score = 100.0 * comp_unit

        stab_score, fragile, _ = stability_metrics(feat, X, regimes, now_idx)

        if analog_regimes_for_conf is None:
            h0 = list(h_summaries.keys())[0]
            y0 = feat[f"fwd_{h0}"]
            idx0 = select_analogs_regime_balanced(X, y0, regimes, now_idx, k=ANALOG_K, min_sep_days=ANALOG_MIN_SEP_DAYS)
            analog_regimes_for_conf = regimes.loc[idx0].fillna("UNK").tolist()
        market_variety = regime_entropy_score(analog_regimes_for_conf)

        n_eff = int(np.median(h_n)) if len(h_n) else 0
        confidence = compute_confidence(n_eff, ANALOG_K, market_variety, stab_score)

        risk = risk_label(h_summaries, beta_today)
        verdict = verdict_line(rebound_score, confidence, stab_score, fragile)
        explain_lines = build_explain(feat, now_idx)

        cutoff = now_idx - pd.Timedelta(days=PLOT_LAST_DAYS)
        d = feat[feat.index >= cutoff].copy()
        if d.empty:
            d = feat.copy()

        px = pd.to_numeric(d["px"], errors="coerce").astype(float)
        wm = pd.to_numeric(d["washout_meter"], errors="coerce").clip(0,100).astype(float)

        ok = px.notna()
        px = px[ok]
        wm = wm.reindex(px.index).fillna(0.0)

        dates = [ts.tz_convert("UTC").strftime("%Y%m%d") for ts in px.index]

        out_row = {
            "Ticker": t,
            "Verdict": verdict,
            "ReboundScore": float(rebound_score),
            "Confidence": float(confidence),
            "Stability": float(stab_score),
            "Fragile": bool(fragile),
            "Risk": risk,
            "WashoutToday": float(wash_today) if np.isfinite(wash_today) else None,
            "SimilarCases": int(np.median(h_n)) if h_n else 0,
            "AsOf": now_idx.tz_convert("UTC").strftime("%Y-%m-%d"),
            "Explain": explain_lines,
            "Outcomes": {k: h_summaries.get(k, {"n":0}) for k in ["1Y","3Y","5Y"]},
            "Series": {
                "d": dates,
                "px": [float(x) if np.isfinite(x) else None for x in px.values.tolist()],
                "wm": [float(x) if np.isfinite(x) else 0.0 for x in wm.values.tolist()],
            },
        }
        results.append(out_row)

        if i % 100 == 0:
            print(f"[PROGRESS] processed {i}/{len(usable)} | scored={len(results)}")

    if not results:
        raise RuntimeError("No tickers scored. Consider lowering MIN_MED_DVOL_USD or ANALOG_MIN.")

    res = sorted(results, key=lambda r: (r.get("ReboundScore", -1e9), r.get("Confidence", -1e9), r.get("Stability", -1e9)), reverse=True)
    now_ny = datetime.now(tz=NY)

    payload = {
        "meta": {
            "asof": now_ny.strftime("%Y-%m-%d %H:%M"),
            "note": "Universe is iShares Russell 1000 holdings (IWB) plus a small always-list. Search only matches this universe.",
            "interval": INTERVAL,
            "period": PERIOD,
            "bench": BENCH,
            "count_scored": len(res),
        },
        "rows": res
    }

    os.makedirs("docs/data", exist_ok=True)
    with open("docs/data/full.json", "w") as f:
        json.dump(payload, f, separators=(",", ":"))

    write_run_stamp()
    print(f"[DONE] wrote docs/data/full.json with {len(res)} tickers.")

if __name__ == "__main__":
    main()
