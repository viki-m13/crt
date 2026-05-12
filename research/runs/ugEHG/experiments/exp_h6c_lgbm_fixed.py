"""H6c — LightGBM walk-forward ranker (fixed labels) + enhanced crash gate.

Fixes LightGBM label type error: use quintile integer labels (0-4).
Also tests enhanced crash gate with breadth + portfolio-vol signals.
"""

from __future__ import annotations

import json, sys, time, warnings
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

REPO = Path("/home/user/crt")
sys.path.insert(0, str(REPO / "strategy/YLOka"))
import harness as H  # noqa

DATA = REPO / "data/YLOka"
CACHE = REPO / "experiments/monthly_dca/cache"
QR = REPO / "quant_research"
HYPS = QR / "state/hypotheses_tested.jsonl"
EXP_DIR = QR / "experiments/exp_h6c_lgbm_fixed"
EXP_DIR.mkdir(parents=True, exist_ok=True)

print("Loading data...")
panel_full = H.load_panel_full()
mr = H.load_monthly_returns()
spy = H.load_spy_features()
print(f"Panel: {panel_full.shape}, MR: {mr.shape}")

RESEARCH_END = H.RESEARCH_END
hyp_count = 0
all_results = []


def metrics_detail(eq):
    met = H.metrics(eq)
    n = len(eq)
    chunk = n // 3
    sharpes = []
    for i in range(3):
        sub_r = eq["ret_m"].iloc[i * chunk:(i + 1) * chunk].astype(float)
        s = float(sub_r.mean() / sub_r.std() * np.sqrt(12)) if sub_r.std() > 0 else 0
        sharpes.append(s)
    met["sub_sharpes"] = sharpes
    met["sub_sharpe_min"] = min(sharpes)
    return met


def record(name, met):
    all_results.append(met)
    cagr = met.get("cagr", 0)
    sharpe = met.get("sharpe", 0)
    mdd = met.get("max_dd", 0)
    sub_min = met.get("sub_sharpe_min", 0)
    print(f"  {name:50s} CAGR={cagr:5.1%}  Sharpe={sharpe:.2f}  "
          f"MDD={mdd:.1%}  SubMin={sub_min:.2f}  ({met.get('wall_time_s', 0):.1f}s)")


# ════════════════════════════════════════════════════════════════════════════
# A. LightGBM walk-forward (fixed: quintile integer labels)
# ════════════════════════════════════════════════════════════════════════════
print("\n=== A. LightGBM Walk-Forward (Fixed) ===")

FEATURES = [
    "mom_12_1", "mom_6_1", "mom_3", "mom_2y", "idio_mom_12_1",
    "mom_per_unit_vol_12", "mom_consistency_12m", "trend_health_5y",
    "trend_r2_12m", "frac_above_50dma_1y", "vol_1y", "vol_12m", "vol_3m",
    "dd_from_52wh", "multibagger_ratio_24m", "acceleration_2y",
    "fip_score", "tight_consolidation_60", "breakout_strength_60",
    "min_dd_60d", "rbi_60", "rbi_120", "vol_asym_60", "vol_asym_126",
    "cst_score", "prerunner_dist", "crt_3m", "crt_6m", "rsi_14",
    "rsi_zone_score", "near_52wh_60d", "range_pos_1y", "log_price",
    "d_sma200", "d_sma50", "mom_5y", "sharpe_5y", "max_dd_5y",
]
FEATURES = [f for f in FEATURES if f in panel_full.columns]
print(f"  Using {len(FEATURES)} features")

panel_full["asof"] = pd.to_datetime(panel_full["asof"])
months = sorted(panel_full["asof"].unique())
months = [pd.Timestamp(m) for m in months]
mr_idx = mr.index
training_months = [m for m in months if m <= RESEARCH_END]

# Build training frame
print("  Building training frame...")
t0 = time.time()
rows_train = []
for m in months:
    sub = panel_full[panel_full["asof"] == m]
    pos = mr_idx.searchsorted(m)
    next_d = None
    for p in (pos - 1, pos):
        if 0 <= p < len(mr_idx) and abs((mr_idx[p] - m).days) <= 7:
            if p + 1 < len(mr_idx):
                next_d = mr_idx[p + 1]
                break
    if next_d is None:
        continue
    fwd = mr.loc[next_d]
    for _, row in sub.iterrows():
        tk = row["ticker"]
        fwd_r = fwd.get(tk, np.nan) if tk in fwd.index else np.nan
        if pd.isna(fwd_r):
            continue
        feat_row = {"asof": m, "ticker": tk, "fwd_1m": float(fwd_r)}
        for f in FEATURES:
            feat_row[f] = row.get(f, np.nan)
        rows_train.append(feat_row)

train_full = pd.DataFrame(rows_train)
print(f"  Training frame: {train_full.shape} in {time.time() - t0:.0f}s")

# Walk-forward LightGBM
RETRAIN_FREQ = 12
TRAIN_WINDOW = 36
N_BINS = 5  # quintile labels: 0, 1, 2, 3, 4

lgb_params = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "n_estimators": 200,
    "num_leaves": 31,
    "learning_rate": 0.05,
    "min_child_samples": 10,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 1.0,
    "reg_lambda": 1.0,
    "n_jobs": -1,
    "verbose": -1,
    "label_gain": list(range(N_BINS)),  # 0,1,2,3,4 gains for quintile labels
}

lgbm_scores = {}
model = None
last_retrain = None
model_trained = False

print("  Training LightGBM walk-forward...")
for i, m in enumerate(training_months):
    need_retrain = (
        last_retrain is None or
        (m.year - last_retrain.year) * 12 + (m.month - last_retrain.month) >= RETRAIN_FREQ
    )
    if need_retrain:
        train_end = pd.Timestamp(m.year, m.month, 1) - pd.DateOffset(months=1)
        train_start = train_end - pd.DateOffset(months=TRAIN_WINDOW - 1)
        mask = (train_full["asof"] >= train_start) & (train_full["asof"] <= train_end)
        sub = train_full[mask].dropna(subset=FEATURES + ["fwd_1m"])
        if len(sub) < 500:
            lgbm_scores[m] = None
            continue
        sub = sub.copy()
        # Quintile labels: 0=worst, 4=best, PER MONTH (cross-sectional)
        sub["label"] = sub.groupby("asof")["fwd_1m"].transform(
            lambda x: pd.qcut(x.rank(method="first"), q=N_BINS,
                               labels=False, duplicates="drop").astype(int)
        )
        sub = sub.dropna(subset=["label"])
        X = sub[FEATURES].values.astype(np.float32)
        y = sub["label"].values.astype(np.int32)
        groups = sub.groupby("asof").size().values
        # Fill NaN
        medians = np.nanmedian(X, axis=0)
        for j in range(X.shape[1]):
            X[np.isnan(X[:, j]), j] = medians[j]
        try:
            model = lgb.LGBMRanker(**lgb_params)
            model.fit(X, y, group=groups)
            last_retrain = m
            model_trained = True
            if i % 24 == 0:
                print(f"    Retrained at {m.strftime('%Y-%m')}")
        except Exception as e:
            print(f"    Training failed at {m}: {e}")
            model = None

    if not model_trained or model is None:
        lgbm_scores[m] = None
        continue

    sub_m = panel_full[panel_full["asof"] == m]
    X = sub_m[FEATURES].values.astype(np.float32)
    medians = np.nanmedian(X, axis=0)
    for j in range(X.shape[1]):
        X[np.isnan(X[:, j]), j] = medians[j]
    try:
        preds = model.predict(X)
        lgbm_scores[m] = dict(zip(sub_m["ticker"], preds))
    except Exception:
        lgbm_scores[m] = None

n_scored = sum(1 for v in lgbm_scores.values() if v is not None)
print(f"  Scored {n_scored}/{len(training_months)} months")

# Compute IC
lgbm_ics = []
for m in training_months:
    scores = lgbm_scores.get(m)
    if scores is None:
        continue
    pos = mr_idx.searchsorted(m)
    for p in (pos - 1, pos):
        if 0 <= p < len(mr_idx) and abs((mr_idx[p] - m).days) <= 7:
            if p + 1 < len(mr_idx):
                next_d = mr_idx[p + 1]
                score_s = pd.Series(scores)
                actual_s = mr.loc[next_d].reindex(score_s.index)
                valid = score_s.notna() & actual_s.notna()
                if valid.sum() < 20:
                    break
                ic, _ = spearmanr(score_s[valid], actual_s[valid])
                lgbm_ics.append(ic)
                break

if lgbm_ics:
    ic_arr = np.array(lgbm_ics)
    print(f"  LightGBM IC vs 1m: Mean={ic_arr.mean():.4f} "
          f"Std={ic_arr.std():.4f} "
          f"t={ic_arr.mean()/ic_arr.std()*np.sqrt(len(ic_arr)):.2f} "
          f"(n={len(lgbm_ics)})")

# Baseline IC comparison
by_asof = {pd.Timestamp(d): g for d, g in panel_full.groupby("asof")}
baseline_ics = []
for m in training_months:
    sub = by_asof.get(m)
    if sub is None or sub.empty:
        continue
    base_score = H.score_ml_3plus6(sub)
    pos = mr_idx.searchsorted(m)
    for p in (pos - 1, pos):
        if 0 <= p < len(mr_idx) and abs((mr_idx[p] - m).days) <= 7:
            if p + 1 < len(mr_idx):
                next_d = mr_idx[p + 1]
                actual = mr.loc[next_d].reindex(base_score.index)
                valid = base_score.notna() & actual.notna()
                if valid.sum() >= 20:
                    ic, _ = spearmanr(base_score[valid], actual[valid])
                    baseline_ics.append(ic)
                break

b_arr = np.array(baseline_ics)
print(f"  Baseline IC vs 1m: Mean={b_arr.mean():.4f} "
      f"Std={b_arr.std():.4f} "
      f"t={b_arr.mean()/b_arr.std()*np.sqrt(len(b_arr)):.2f}")


# Simulate LightGBM
def simulate_lgbm(K: int, fallback_fn=None) -> pd.DataFrame:
    cf = 5.0 / 10000.0
    equity = 1.0
    cur_picks, cur_weights = [], np.array([])
    held_for = 0
    cash_flag = False
    rows = []
    if fallback_fn is None:
        fallback_fn = H.score_ml_3plus6

    for i, m in enumerate(training_months):
        do_reb = (i == 0) or (held_for >= 6) or cash_flag
        spy_now = spy.loc[m].to_dict() if m in spy.index else {}
        regime = H.regime_tight(spy_now)

        if do_reb:
            scores = lgbm_scores.get(m)
            if regime == "crash":
                cur_picks, cur_weights, cash_flag = [], np.array([]), True
            elif scores is None:
                # Fallback to baseline
                sub = by_asof.get(m, pd.DataFrame()).copy()
                if not sub.empty:
                    sub["score"] = fallback_fn(sub)
                    picks = sub.nlargest(K, "score")
                    cur_picks = picks["ticker"].tolist()
                    cur_weights = np.ones(K) / K
                    cash_flag = False
            else:
                top_k = sorted(scores, key=scores.get, reverse=True)[:K]
                cur_picks = top_k
                cur_weights = np.ones(K) / K
                cash_flag = False
            held_for = 0

        pos1 = mr_idx.searchsorted(m)
        if cash_flag or not cur_picks:
            ret_m = 0.0
        else:
            cands = [(j, abs((mr_idx[j] - m).days)) for j in (pos1 - 1, pos1)
                      if 0 <= j < len(mr_idx)]
            cands.sort(key=lambda x: x[1])
            if not cands or cands[0][1] > 7 or cands[0][0] + 1 >= len(mr_idx):
                ret_m = 0.0
            else:
                next_d = mr_idx[cands[0][0] + 1]
                pick_rets = []
                for tk in cur_picks:
                    r = mr.at[next_d, tk] if tk in mr.columns and not pd.isna(mr.at[next_d, tk]) else -1.0
                    pick_rets.append(float(r))
                ret_m = float(np.dot(pick_rets, cur_weights))

        if do_reb and not cash_flag and cur_picks:
            equity *= (1 + ret_m) * (1 - cf)
        else:
            equity *= 1 + ret_m
        held_for += 1
        rows.append({"date": m, "equity": equity, "ret_m": ret_m,
                      "regime": "cash" if cash_flag else regime,
                      "n_picks": len(cur_picks), "picks": ",".join(cur_picks)})
    return pd.DataFrame(rows)


print("\n  LightGBM backtest results:")
for K in [3, 5, 7, 10, 15]:
    hyp_count += 1
    t0 = time.time()
    eq = simulate_lgbm(K=K)
    met = metrics_detail(eq)
    met.update({"exp_name": f"lgbm_fixed_K{K}", "K": K,
                 "wall_time_s": round(time.time() - t0, 2)})
    record(f"lgbm_fixed_K{K}", met)
    run_dir = EXP_DIR / f"lgbm_fixed_K{K}"
    run_dir.mkdir(exist_ok=True)
    eq.to_parquet(run_dir / "equity.parquet")
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(met, f, indent=2)


# ════════════════════════════════════════════════════════════════════════════
# B. LightGBM with risk-adjusted (Sharpe) label
# ════════════════════════════════════════════════════════════════════════════
print("\n=== B. LightGBM Sharpe-label (quintile) ===")

# Build RA training frame
rows_ra = []
for m in months:
    sub = panel_full[panel_full["asof"] == m]
    pos = mr_idx.searchsorted(m)
    next_d = None
    for p in (pos - 1, pos):
        if 0 <= p < len(mr_idx) and abs((mr_idx[p] - m).days) <= 7:
            if p + 1 < len(mr_idx):
                next_d = mr_idx[p + 1]
                break
    if next_d is None:
        continue
    fwd = mr.loc[next_d]
    for _, row in sub.iterrows():
        tk = row["ticker"]
        fwd_r = fwd.get(tk, np.nan) if tk in fwd.index else np.nan
        if pd.isna(fwd_r):
            continue
        vol3m = row.get("vol_3m", row.get("vol_1y", 0.3))
        if pd.isna(vol3m) or vol3m < 0.01:
            vol3m = 0.3
        # Monthly Sharpe of next month (return / monthly_vol)
        monthly_vol = float(vol3m) / np.sqrt(12)
        sharpe_1m = float(fwd_r) / monthly_vol
        feat_row = {"asof": m, "ticker": tk, "sharpe_1m": sharpe_1m, "fwd_1m": float(fwd_r)}
        for f in FEATURES:
            feat_row[f] = row.get(f, np.nan)
        rows_ra.append(feat_row)

train_ra = pd.DataFrame(rows_ra)
print(f"  RA training frame: {train_ra.shape}")

# Walk-forward LightGBM on Sharpe label
lgbm_ra_scores = {}
model_ra = None
last_retrain_ra = None
trained_ra = False

for i, m in enumerate(training_months):
    need_retrain = (
        last_retrain_ra is None or
        (m.year - last_retrain_ra.year) * 12 + (m.month - last_retrain_ra.month) >= RETRAIN_FREQ
    )
    if need_retrain:
        train_end = pd.Timestamp(m.year, m.month, 1) - pd.DateOffset(months=1)
        train_start = train_end - pd.DateOffset(months=TRAIN_WINDOW - 1)
        mask = (train_ra["asof"] >= train_start) & (train_ra["asof"] <= train_end)
        sub = train_ra[mask].dropna(subset=FEATURES + ["sharpe_1m"])
        if len(sub) < 500:
            lgbm_ra_scores[m] = None
            continue
        sub = sub.copy()
        sub["label"] = sub.groupby("asof")["sharpe_1m"].transform(
            lambda x: pd.qcut(x.rank(method="first"), q=N_BINS,
                               labels=False, duplicates="drop").astype(int)
        )
        sub = sub.dropna(subset=["label"])
        X = sub[FEATURES].values.astype(np.float32)
        y = sub["label"].values.astype(np.int32)
        groups = sub.groupby("asof").size().values
        medians = np.nanmedian(X, axis=0)
        for j in range(X.shape[1]):
            X[np.isnan(X[:, j]), j] = medians[j]
        try:
            model_ra = lgb.LGBMRanker(**lgb_params)
            model_ra.fit(X, y, group=groups)
            last_retrain_ra = m
            trained_ra = True
        except Exception:
            model_ra = None

    if not trained_ra or model_ra is None:
        lgbm_ra_scores[m] = None
        continue
    sub_m = panel_full[panel_full["asof"] == m]
    X = sub_m[FEATURES].values.astype(np.float32)
    medians = np.nanmedian(X, axis=0)
    for j in range(X.shape[1]):
        X[np.isnan(X[:, j]), j] = medians[j]
    try:
        preds = model_ra.predict(X)
        lgbm_ra_scores[m] = dict(zip(sub_m["ticker"], preds))
    except Exception:
        lgbm_ra_scores[m] = None

n_ra = sum(1 for v in lgbm_ra_scores.values() if v is not None)
print(f"  RA-label scored {n_ra}/{len(training_months)} months")


def simulate_lgbm_with_scores(scores_dict, K: int) -> pd.DataFrame:
    cf = 5.0 / 10000.0
    equity = 1.0
    cur_picks, cur_weights = [], np.array([])
    held_for = 0
    cash_flag = False
    rows = []
    for i, m in enumerate(training_months):
        do_reb = (i == 0) or (held_for >= 6) or cash_flag
        spy_now = spy.loc[m].to_dict() if m in spy.index else {}
        regime = H.regime_tight(spy_now)
        if do_reb:
            scores = scores_dict.get(m)
            if regime == "crash":
                cur_picks, cur_weights, cash_flag = [], np.array([]), True
            elif scores is None:
                sub = by_asof.get(m, pd.DataFrame()).copy()
                if not sub.empty:
                    sub["score"] = H.score_ml_3plus6(sub)
                    picks = sub.nlargest(K, "score")
                    cur_picks = picks["ticker"].tolist()
                    cur_weights = np.ones(K) / K
                    cash_flag = False
            else:
                top_k = sorted(scores, key=scores.get, reverse=True)[:K]
                cur_picks = top_k
                cur_weights = np.ones(K) / K
                cash_flag = False
            held_for = 0
        pos1 = mr_idx.searchsorted(m)
        if cash_flag or not cur_picks:
            ret_m = 0.0
        else:
            cands = [(j, abs((mr_idx[j] - m).days)) for j in (pos1 - 1, pos1)
                      if 0 <= j < len(mr_idx)]
            cands.sort(key=lambda x: x[1])
            if not cands or cands[0][1] > 7 or cands[0][0] + 1 >= len(mr_idx):
                ret_m = 0.0
            else:
                next_d = mr_idx[cands[0][0] + 1]
                pick_rets = [float(mr.at[next_d, tk]) if tk in mr.columns
                              and not pd.isna(mr.at[next_d, tk]) else -1.0
                              for tk in cur_picks]
                ret_m = float(np.dot(pick_rets, cur_weights))
        if do_reb and not cash_flag and cur_picks:
            equity *= (1 + ret_m) * (1 - cf)
        else:
            equity *= 1 + ret_m
        held_for += 1
        rows.append({"date": m, "equity": equity, "ret_m": ret_m,
                      "regime": "cash" if cash_flag else regime,
                      "n_picks": len(cur_picks), "picks": ",".join(cur_picks)})
    return pd.DataFrame(rows)


for K in [3, 5, 7, 10]:
    hyp_count += 1
    t0 = time.time()
    eq = simulate_lgbm_with_scores(lgbm_ra_scores, K=K)
    met = metrics_detail(eq)
    met.update({"exp_name": f"lgbm_ra_K{K}", "K": K,
                 "wall_time_s": round(time.time() - t0, 2)})
    record(f"lgbm_ra_K{K}", met)


# ════════════════════════════════════════════════════════════════════════════
# C. Enhanced crash gate (breadth + portfolio vol)
# ════════════════════════════════════════════════════════════════════════════
print("\n=== C. Enhanced crash gate ===")

# Compute cross-sectional breadth per month (frac above d_sma200 > 0)
breadth_map = {}
for m in months:
    sub = panel_full[panel_full["asof"] == m]
    if "d_sma200" in sub.columns:
        frac_above = float((sub["d_sma200"].fillna(0) > 0).mean())
        breadth_map[pd.Timestamp(m)] = frac_above
    else:
        breadth_map[pd.Timestamp(m)] = 0.5


def regime_enhanced(spy_now: dict, breadth: float = 0.5,
                     portfolio_vol_monthly: float = 0.05) -> str:
    """Enhanced crash gate: original + breadth + portfolio vol."""
    base = H.regime_tight(spy_now)
    if base == "crash":
        return "crash"
    # Breadth crash: if < 35% of universe above 200-day MA
    if breadth < 0.35:
        return "crash"
    # Portfolio vol crash: if recent portfolio vol is extreme (handled externally)
    if portfolio_vol_monthly > 0.12:  # > 12% monthly = ~42% annual
        return "crash"
    return base


def simulate_enhanced_gate(K: int, score_fn_name: str = "ml_3plus6",
                             breadth_threshold: float = 0.35,
                             port_vol_threshold: float = 0.12) -> pd.DataFrame:
    """Simulate with enhanced crash gate."""
    score_fn = H.SCORERS[score_fn_name]
    cf = 5.0 / 10000.0
    equity = 1.0
    cur_picks, cur_weights = [], np.array([])
    held_for = 0
    cash_flag = False
    rows = []
    recent_rets = []  # for portfolio vol calculation

    for i, m in enumerate(training_months):
        do_reb = (i == 0) or (held_for >= 6) or cash_flag
        spy_now = spy.loc[m].to_dict() if m in spy.index else {}
        breadth = breadth_map.get(m, 0.5)
        port_vol = float(np.std(recent_rets[-6:], ddof=1)) if len(recent_rets) >= 3 else 0.05
        regime = regime_enhanced(spy_now, breadth, port_vol)

        if do_reb:
            sub = by_asof.get(m, pd.DataFrame()).copy()
            if not sub.empty:
                sub["score"] = score_fn(sub)
                sub = sub.dropna(subset=["score"])
            if regime == "crash":
                cur_picks, cur_weights, cash_flag = [], np.array([]), True
            elif sub.empty:
                cur_picks, cur_weights, cash_flag = [], np.array([]), True
            else:
                picks = sub.nlargest(K, "score")
                cur_picks = picks["ticker"].tolist()
                cur_weights = np.ones(K) / K
                cash_flag = False
            held_for = 0

        pos1 = mr_idx.searchsorted(m)
        if cash_flag or not cur_picks:
            ret_m = 0.0
        else:
            cands = [(j, abs((mr_idx[j] - m).days)) for j in (pos1 - 1, pos1)
                      if 0 <= j < len(mr_idx)]
            cands.sort(key=lambda x: x[1])
            if not cands or cands[0][1] > 7 or cands[0][0] + 1 >= len(mr_idx):
                ret_m = 0.0
            else:
                next_d = mr_idx[cands[0][0] + 1]
                pick_rets = [float(mr.at[next_d, tk]) if tk in mr.columns
                              and not pd.isna(mr.at[next_d, tk]) else -1.0
                              for tk in cur_picks]
                ret_m = float(np.dot(pick_rets, cur_weights))

        recent_rets.append(ret_m)
        if len(recent_rets) > 12:
            recent_rets.pop(0)

        if do_reb and not cash_flag and cur_picks:
            equity *= (1 + ret_m) * (1 - cf)
        else:
            equity *= 1 + ret_m
        held_for += 1
        rows.append({"date": m, "equity": equity, "ret_m": ret_m,
                      "regime": "cash" if cash_flag else regime,
                      "n_picks": len(cur_picks), "picks": ",".join(cur_picks)})
    return pd.DataFrame(rows)


for K, bt, pvt in [(3, 0.35, 0.12), (3, 0.40, 0.12), (3, 0.35, 0.10),
                    (5, 0.35, 0.12), (5, 0.40, 0.12), (7, 0.35, 0.12),
                    (3, 0.45, 0.12), (3, 0.35, 0.15)]:
    hyp_count += 1
    t0 = time.time()
    eq = simulate_enhanced_gate(K=K, breadth_threshold=bt, port_vol_threshold=pvt)
    met = metrics_detail(eq)
    met.update({"exp_name": f"enh_gate_K{K}_bt{bt}_pvt{pvt}", "K": K,
                 "wall_time_s": round(time.time() - t0, 2)})
    record(f"enh_gate_K{K}_bt{bt:.2f}_pvt{pvt:.2f}", met)


# ════════════════════════════════════════════════════════════════════════════
# D. LightGBM + Enhanced Gate
# ════════════════════════════════════════════════════════════════════════════
print("\n=== D. LightGBM + Enhanced Gate ===")


def simulate_lgbm_enhanced_gate(K: int) -> pd.DataFrame:
    cf = 5.0 / 10000.0
    equity = 1.0
    cur_picks, cur_weights = [], np.array([])
    held_for = 0
    cash_flag = False
    rows = []
    recent_rets = []

    for i, m in enumerate(training_months):
        do_reb = (i == 0) or (held_for >= 6) or cash_flag
        spy_now = spy.loc[m].to_dict() if m in spy.index else {}
        breadth = breadth_map.get(m, 0.5)
        port_vol = float(np.std(recent_rets[-6:], ddof=1)) if len(recent_rets) >= 3 else 0.05
        regime = regime_enhanced(spy_now, breadth, port_vol)

        if do_reb:
            scores = lgbm_scores.get(m)
            if regime == "crash":
                cur_picks, cur_weights, cash_flag = [], np.array([]), True
            elif scores is None:
                sub = by_asof.get(m, pd.DataFrame()).copy()
                if not sub.empty:
                    sub["score"] = H.score_ml_3plus6(sub)
                    picks = sub.nlargest(K, "score")
                    cur_picks = picks["ticker"].tolist()
                    cur_weights = np.ones(K) / K
                    cash_flag = False
            else:
                top_k = sorted(scores, key=scores.get, reverse=True)[:K]
                cur_picks = top_k
                cur_weights = np.ones(K) / K
                cash_flag = False
            held_for = 0

        recent_rets.append(0.0 if cash_flag or not cur_picks else 0.0)

        pos1 = mr_idx.searchsorted(m)
        if cash_flag or not cur_picks:
            ret_m = 0.0
        else:
            cands = [(j, abs((mr_idx[j] - m).days)) for j in (pos1 - 1, pos1)
                      if 0 <= j < len(mr_idx)]
            cands.sort(key=lambda x: x[1])
            if not cands or cands[0][1] > 7 or cands[0][0] + 1 >= len(mr_idx):
                ret_m = 0.0
            else:
                next_d = mr_idx[cands[0][0] + 1]
                pick_rets = [float(mr.at[next_d, tk]) if tk in mr.columns
                              and not pd.isna(mr.at[next_d, tk]) else -1.0
                              for tk in cur_picks]
                ret_m = float(np.dot(pick_rets, cur_weights))

        recent_rets[-1] = ret_m

        if do_reb and not cash_flag and cur_picks:
            equity *= (1 + ret_m) * (1 - cf)
        else:
            equity *= 1 + ret_m
        held_for += 1
        rows.append({"date": m, "equity": equity, "ret_m": ret_m,
                      "regime": "cash" if cash_flag else regime,
                      "n_picks": len(cur_picks), "picks": ",".join(cur_picks)})
    return pd.DataFrame(rows)


for K in [3, 5, 7]:
    hyp_count += 1
    t0 = time.time()
    eq = simulate_lgbm_enhanced_gate(K=K)
    met = metrics_detail(eq)
    met.update({"exp_name": f"lgbm_enh_K{K}", "K": K,
                 "wall_time_s": round(time.time() - t0, 2)})
    record(f"lgbm_enh_K{K}", met)


# ════════════════════════════════════════════════════════════════════════════
# Summary
# ════════════════════════════════════════════════════════════════════════════

results_df = pd.DataFrame(all_results)
results_df.to_csv(EXP_DIR / "summary.csv", index=False)

print("\n\n=== ALL RESULTS — TOP BY SHARPE ===")
cols = ["exp_name", "K", "cagr", "sharpe", "max_dd", "sub_sharpe_min"]
print(results_df.sort_values("sharpe", ascending=False).head(20)[cols].to_string(index=False))

print("\n=== ALL RESULTS — TOP BY CAGR ===")
print(results_df.sort_values("cagr", ascending=False).head(20)[cols].to_string(index=False))

# Log hypotheses
with open(HYPS, "a") as f:
    f.write(json.dumps({"ts": datetime.utcnow().isoformat() + "Z",
                         "run_id": "h6c", "n_hparams": hyp_count}) + "\n")
print(f"\nTotal hypotheses this batch: {hyp_count}")
