"""H6b — Donchian + quality combos + LightGBM walk-forward ranker.

Key insight from H6: Quality filter dramatically cuts CAGR (40% -> 20%).
Donchian filter gets best CAGR (48.73%).

Here we:
1. Combine Donchian with mild quality/vol filter
2. Apply invvol portfolio weights to Donchian picks
3. Train a LightGBM walk-forward ranker on 47 features
4. Test risk-adjusted scoring (pred / vol)
5. Run a correlation-diversity picker (decorrelation)
"""

from __future__ import annotations

import json, os, sys, time, warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

REPO = Path("/home/user/crt")
sys.path.insert(0, str(REPO / "strategy/YLOka"))
import harness as H  # noqa

DATA = REPO / "data/YLOka"
CACHE = REPO / "experiments/monthly_dca/cache"
QR = REPO / "quant_research"
JOURNAL = QR / "state/journal.jsonl"
HYPS = QR / "state/hypotheses_tested.jsonl"
EXP_DIR = QR / "experiments/exp_h6b_donchian_quality"
EXP_DIR.mkdir(parents=True, exist_ok=True)

print("Loading data...")
panel_full = H.load_panel_full()
mr = H.load_monthly_returns()
spy = H.load_spy_features()
prices = H.load_prices()
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


def log_hyps(n):
    with open(HYPS, "a") as f:
        f.write(json.dumps({"ts": datetime.utcnow().isoformat() + "Z",
                             "run_id": "h6b", "n_hparams": n}) + "\n")


def record(name, met):
    all_results.append(met)
    cagr = met.get("cagr", 0)
    sharpe = met.get("sharpe", 0)
    mdd = met.get("max_dd", 0)
    sub_min = met.get("sub_sharpe_min", 0)
    print(f"  {name:55s} CAGR={cagr:5.1%}  Sharpe={sharpe:.2f}  "
          f"MDD={mdd:.1%}  SubMin={sub_min:.2f}  ({met.get('wall_time_s', 0):.1f}s)")
    run_dir = EXP_DIR / name.replace("/", "_").replace(" ", "_")
    run_dir.mkdir(exist_ok=True)
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(met, f, indent=2)


print("\n=== H6b Experiments ===")
print(f"{'Name':55s} CAGR    Sharpe  MDD    SubMin")
print("-" * 90)

# ════════════════════════════════════════════════════════════════════════════
# A. Risk-adjusted scoring: (pred_3m + pred_6m) / vol_3m
# ════════════════════════════════════════════════════════════════════════════
print("\n-- A. Risk-adjusted scoring --")

def score_ra_3m_vol(panel_at):
    base = (panel_at["pred_3m"] + panel_at["pred_6m"]) / 2
    vol = panel_at.get("vol_3m")
    if vol is None:
        return base
    vol_safe = vol.clip(lower=0.05)
    return base / vol_safe


def score_ra_rank(panel_at):
    """rank(pred/vol) + 0.3*rank(pred): blended risk-adjusted."""
    base = (panel_at["pred_3m"] + panel_at["pred_6m"]) / 2
    vol = panel_at.get("vol_3m", panel_at.get("vol_1y"))
    if vol is None:
        return base
    ra = base / vol.clip(lower=0.05)
    ra_r = H._safe_rank(ra)
    base_r = H._safe_rank(base)
    return 0.7 * ra_r + 0.3 * base_r


for sfn, sname in [(score_ra_3m_vol, "ra_pred_div_vol"),
                    (score_ra_rank, "ra_rank_blend")]:
    H.SCORERS[sname] = sfn
    for K in [3, 5, 7, 10]:
        hyp_count += 1
        t0 = time.time()
        cfg = H.StratConfig(name=sname, K=K, hold_months=6,
                             weighting="ew", cost_bps_per_leg=5.0,
                             crash_gate=True, score_fn_name=sname)
        eq = H.simulate(cfg, panel_full, mr, spy, start=None, end=RESEARCH_END)
        met = metrics_detail(eq)
        met.update({"exp_name": f"{sname}_K{K}", "K": K,
                     "wall_time_s": round(time.time() - t0, 2)})
        record(f"{sname}_K{K}", met)


# ════════════════════════════════════════════════════════════════════════════
# B. Donchian + mild quality filter combos
# ════════════════════════════════════════════════════════════════════════════
print("\n-- B. Donchian + quality/vol filter --")


def score_donchian_qual(vol_ceiling: float = 0.70):
    """ml_3plus6 gated by Donchian130 nearness AND vol ceiling.

    The Donchian filter is a pick-time filter applied post-scoring.
    Here we apply vol ceiling at scoring time.
    """
    def scorer(panel_at):
        base = (panel_at["pred_3m"] + panel_at["pred_6m"]) / 2
        vol = panel_at.get("vol_1y", panel_at.get("vol_12m"))
        if vol is not None:
            high_vol = vol.fillna(0.5) > vol_ceiling
            base = base.where(~high_vol, -1e9)
        return base
    scorer.__name__ = f"donchian_qual_vc{vol_ceiling:.2f}"
    return scorer


for vc in [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
    sname = f"dq_vc{vc:.2f}"
    sfn = score_donchian_qual(vc)
    H.SCORERS[sname] = sfn
    for K in [3, 5]:
        hyp_count += 1
        t0 = time.time()
        cfg = H.StratConfig(name=sname, K=K, hold_months=6, weighting="ew",
                             cost_bps_per_leg=5.0, crash_gate=True,
                             pick_filter="donchian130",
                             score_fn_name=sname)
        eq = H.simulate(cfg, panel_full, mr, spy, prices=prices,
                         start=None, end=RESEARCH_END)
        met = metrics_detail(eq)
        met.update({"exp_name": f"{sname}_K{K}", "K": K,
                     "wall_time_s": round(time.time() - t0, 2)})
        record(f"{sname}_K{K}", met)


# ════════════════════════════════════════════════════════════════════════════
# C. Donchian + invvol portfolio weights
# ════════════════════════════════════════════════════════════════════════════
print("\n-- C. Donchian + invvol portfolio weights --")


def simulate_donchian_invvol(K: int, vol_ceiling: float = 1.0,
                              score_fn_name: str = "ml_3plus6"):
    """Custom simulate: Donchian filter + invvol portfolio weights."""
    score_fn = H.SCORERS[score_fn_name]
    months = sorted(panel_full["asof"].unique())
    months = [pd.Timestamp(m) for m in months if pd.Timestamp(m) <= RESEARCH_END]
    by_asof = {pd.Timestamp(d): g for d, g in panel_full.groupby("asof")}
    mr_idx = mr.index
    cf = 5.0 / 10000.0

    equity = 1.0
    cur_picks = []
    cur_weights = np.array([])
    held_for = 0
    cash_flag = False
    rows = []

    for i, m in enumerate(months):
        do_reb = (i == 0) or (held_for >= 6) or cash_flag
        spy_now = spy.loc[m].to_dict() if m in spy.index else {}
        regime = H.regime_tight(spy_now)

        if do_reb:
            sub = by_asof.get(m, pd.DataFrame()).copy()
            if not sub.empty:
                sub["score"] = score_fn(sub)
                sub = sub.dropna(subset=["score"])
                if vol_ceiling < 1.0:
                    vol = sub.get("vol_1y", sub.get("vol_12m"))
                    if vol is not None:
                        sub = sub[vol.fillna(0.5) <= vol_ceiling]
            if regime == "crash":
                cur_picks, cur_weights, cash_flag = [], np.array([]), True; held_for = 0
            elif sub.empty:
                cur_picks, cur_weights, cash_flag = [], np.array([]), True; held_for = 0
            else:
                # Top K by ML score, then apply Donchian filter
                top = sub.nlargest(K * 3, "score")  # oversample before filter
                filtered = H.filter_donchian130(top, prices, m)
                picks = filtered.head(K)
                if picks.empty:
                    picks = top.head(K)
                # Invvol weights on filtered picks
                vol_col = sub.set_index("ticker").get("vol_3m")
                if vol_col is not None:
                    pick_vols = picks["ticker"].map(vol_col)
                    inv_vol = 1.0 / pick_vols.replace(0, 0.01).fillna(0.25)
                    w = inv_vol.values / inv_vol.sum()
                else:
                    w = np.ones(len(picks)) / len(picks)
                cur_picks = picks["ticker"].tolist()
                cur_weights = w
                cash_flag = False
                held_for = 0

        pos1 = mr_idx.searchsorted(m)
        if cash_flag or len(cur_picks) == 0:
            ret_m = 0.0
        elif pos1 + 1 >= len(mr_idx) or pos1 - 1 < 0:
            ret_m = 0.0
        else:
            cands = [(j, abs((mr_idx[j] - m).days)) for j in
                      (pos1 - 1, pos1) if 0 <= j < len(mr_idx)]
            cands.sort(key=lambda x: x[1])
            if not cands or cands[0][1] > 7 or cands[0][0] + 1 >= len(mr_idx):
                ret_m = 0.0
            else:
                next_d = mr_idx[cands[0][0] + 1]
                pick_rets = [float(mr.at[next_d, tk]) if tk in mr.columns
                              and not pd.isna(mr.at[next_d, tk]) else -1.0
                              for tk in cur_picks]
                ret_m = float(np.dot(np.array(pick_rets), cur_weights))

        if do_reb and not cash_flag and len(cur_picks) > 0:
            equity *= (1 + ret_m) * (1 - cf)
        else:
            equity *= 1 + ret_m

        held_for += 1
        rows.append({"date": m, "equity": equity, "ret_m": ret_m,
                      "regime": "cash" if cash_flag else regime,
                      "n_picks": len(cur_picks), "picks": ",".join(cur_picks)})
    return pd.DataFrame(rows)


for K, vc in [(3, 1.0), (3, 0.65), (3, 0.70), (3, 0.75),
               (5, 1.0), (5, 0.70), (5, 0.75), (7, 1.0), (7, 0.75)]:
    hyp_count += 1
    t0 = time.time()
    eq = simulate_donchian_invvol(K=K, vol_ceiling=vc)
    met = metrics_detail(eq)
    met.update({"exp_name": f"don_invvol_K{K}_vc{vc}", "K": K,
                 "wall_time_s": round(time.time() - t0, 2)})
    record(f"don_invvol_K{K}_vc{vc}", met)


# ════════════════════════════════════════════════════════════════════════════
# D. LightGBM walk-forward ranker
# ════════════════════════════════════════════════════════════════════════════
print("\n-- D. LightGBM walk-forward cross-sectional ranker --")

import lightgbm as lgb
from sklearn.preprocessing import QuantileTransformer

# Features to use (exclude identifiers and forward-looking)
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
print(f"  Using {len(FEATURES)} features for LightGBM")

# Prepare data: compute actual 1m forward returns for each panel observation
panel_full2 = panel_full.copy()
panel_full2["asof"] = pd.to_datetime(panel_full2["asof"])

months = sorted(panel_full2["asof"].unique())
months = [pd.Timestamp(m) for m in months]
mr_idx = mr.index


def get_fwd_return(asof: pd.Timestamp, ticker: str) -> float:
    """Next month actual return after asof."""
    pos = mr_idx.searchsorted(asof)
    # Find nearest month-end to asof
    for p in (pos - 1, pos):
        if 0 <= p < len(mr_idx) and abs((mr_idx[p] - asof).days) <= 7:
            if p + 1 < len(mr_idx):
                v = mr.at[mr_idx[p + 1], ticker] if ticker in mr.columns else np.nan
                return float(v) if not pd.isna(v) else np.nan
    return np.nan


print("  Computing forward returns for LightGBM training (this may take ~60s)...")
t_fwd = time.time()

# Build training frame: for each (asof, ticker), get features + 1m fwd return
rows_train = []
for m in months:
    sub = panel_full2[panel_full2["asof"] == m].copy()
    pos = mr_idx.searchsorted(m)
    # Find next return date
    next_ret_date = None
    for p in (pos - 1, pos):
        if 0 <= p < len(mr_idx) and abs((mr_idx[p] - m).days) <= 7:
            if p + 1 < len(mr_idx):
                next_ret_date = mr_idx[p + 1]
                break
    if next_ret_date is None:
        continue
    fwd = mr.loc[next_ret_date] if next_ret_date in mr.index else pd.Series(dtype=float)
    for _, row in sub.iterrows():
        tk = row["ticker"]
        if tk not in mr.columns:
            continue
        fwd_r = fwd.get(tk, np.nan)
        if pd.isna(fwd_r):
            continue
        feat_row = {"asof": m, "ticker": tk, "fwd_1m": float(fwd_r)}
        for f in FEATURES:
            feat_row[f] = row.get(f, np.nan)
        rows_train.append(feat_row)

train_full = pd.DataFrame(rows_train)
print(f"  Training frame built: {train_full.shape} in {time.time() - t_fwd:.0f}s")


# Walk-forward LightGBM: retrain every 12 months using 36m rolling window
RETRAIN_FREQ = 12   # retrain every 12 months
TRAIN_WINDOW = 36   # use 36 months of history

lgbm_scores = {}  # asof -> {ticker: score}
model_trained = False
model = None

lgb_params = {
    "objective": "rank_xendcg",
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
}

print("  Running walk-forward LightGBM training...")
training_months = [m for m in months if m <= RESEARCH_END]

last_retrain = None
for i, m in enumerate(training_months):
    need_retrain = (
        last_retrain is None or
        (m.year - last_retrain.year) * 12 + (m.month - last_retrain.month) >= RETRAIN_FREQ
    )
    if need_retrain:
        # Training data: previous TRAIN_WINDOW months
        train_end = pd.Timestamp(m.year, m.month, 1) - pd.DateOffset(months=1)
        train_start = train_end - pd.DateOffset(months=TRAIN_WINDOW - 1)
        train_mask = (train_full["asof"] >= train_start) & (train_full["asof"] <= train_end)
        train_sub = train_full[train_mask].dropna(subset=FEATURES + ["fwd_1m"])

        if len(train_sub) < 500:
            # Not enough data yet
            lgbm_scores[m] = None
            continue

        # Label = rank within each month (cross-sectional rank)
        train_sub = train_sub.copy()
        train_sub["label"] = train_sub.groupby("asof")["fwd_1m"].rank(pct=True)

        X_train = train_sub[FEATURES].values.astype(np.float32)
        y_train = train_sub["label"].values.astype(np.float32)
        groups = train_sub.groupby("asof").size().values

        # Fill NaN with median per column
        col_medians = np.nanmedian(X_train, axis=0)
        for j in range(X_train.shape[1]):
            mask = np.isnan(X_train[:, j])
            X_train[mask, j] = col_medians[j]

        # Train LightGBM ranker
        try:
            model = lgb.LGBMRanker(**lgb_params)
            model.fit(X_train, y_train, group=groups)
            last_retrain = m
            model_trained = True
        except Exception as e:
            print(f"  LightGBM training failed at {m}: {e}")
            model = None

    if not model_trained or model is None:
        lgbm_scores[m] = None
        continue

    # Score current month
    sub = panel_full2[panel_full2["asof"] == m].copy()
    X = sub[FEATURES].values.astype(np.float32)
    # Fill NaN
    col_medians_score = np.nanmedian(X, axis=0)
    for j in range(X.shape[1]):
        mask = np.isnan(X[:, j])
        X[mask, j] = col_medians_score[j]

    try:
        preds = model.predict(X)
        lgbm_scores[m] = dict(zip(sub["ticker"], preds))
    except Exception:
        lgbm_scores[m] = None

n_scored = sum(1 for v in lgbm_scores.values() if v is not None)
print(f"  LightGBM scored {n_scored}/{len(training_months)} months")


# Now simulate using LightGBM scores
def simulate_lgbm(K: int = 3) -> pd.DataFrame:
    """Simulate using walk-forward LightGBM scores."""
    by_asof = {pd.Timestamp(d): g for d, g in panel_full2.groupby("asof")}
    mr_idx = mr.index
    cf = 5.0 / 10000.0

    equity = 1.0
    cur_picks = []
    cur_weights = np.array([])
    held_for = 0
    cash_flag = False
    rows = []

    for i, m in enumerate(training_months):
        do_reb = (i == 0) or (held_for >= 6) or cash_flag
        spy_now = spy.loc[m].to_dict() if m in spy.index else {}
        regime = H.regime_tight(spy_now)

        if do_reb:
            scores = lgbm_scores.get(m)
            if regime == "crash" or scores is None:
                if regime == "crash":
                    cur_picks, cur_weights, cash_flag = [], np.array([]), True
                elif scores is None:
                    # Fall back to ml_3plus6
                    sub = by_asof.get(m, pd.DataFrame()).copy()
                    if not sub.empty:
                        sub["score"] = H.score_ml_3plus6(sub)
                        sub = sub.dropna(subset=["score"])
                        picks = sub.nlargest(K, "score")
                        cur_picks = picks["ticker"].tolist()
                        cur_weights = np.ones(K) / K
                        cash_flag = False
                held_for = 0
            else:
                # Pick top K by LightGBM rank
                sorted_tickers = sorted(scores, key=scores.get, reverse=True)
                picks = sorted_tickers[:K]
                cur_picks = picks
                cur_weights = np.ones(K) / K
                cash_flag = False
                held_for = 0

        pos1 = mr_idx.searchsorted(m)
        if cash_flag or len(cur_picks) == 0:
            ret_m = 0.0
        elif pos1 + 1 >= len(mr_idx) or pos1 - 1 < 0:
            ret_m = 0.0
        else:
            cands = [(j, abs((mr_idx[j] - m).days)) for j in
                      (pos1 - 1, pos1) if 0 <= j < len(mr_idx)]
            cands.sort(key=lambda x: x[1])
            if not cands or cands[0][1] > 7 or cands[0][0] + 1 >= len(mr_idx):
                ret_m = 0.0
            else:
                next_d = mr_idx[cands[0][0] + 1]
                pick_rets = [float(mr.at[next_d, tk]) if tk in mr.columns
                              and not pd.isna(mr.at[next_d, tk]) else -1.0
                              for tk in cur_picks]
                ret_m = float(np.mean(pick_rets))

        if do_reb and not cash_flag and len(cur_picks) > 0:
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
    eq = simulate_lgbm(K=K)
    met = metrics_detail(eq)
    met.update({"exp_name": f"lgbm_wf_K{K}", "K": K,
                 "wall_time_s": round(time.time() - t0, 2)})
    record(f"lgbm_wf_K{K}", met)
    # Also save equity
    run_dir = EXP_DIR / f"lgbm_wf_K{K}"
    run_dir.mkdir(exist_ok=True)
    eq.to_parquet(run_dir / "equity.parquet")


# LightGBM IC analysis
print("\n  LightGBM IC vs actual 1m return:")
lgbm_ics = []
for m in training_months:
    scores = lgbm_scores.get(m)
    if scores is None:
        continue
    # Compute IC against actual return
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
                from scipy.stats import spearmanr
                ic, _ = spearmanr(score_s[valid], actual_s[valid])
                lgbm_ics.append(ic)
                break

if lgbm_ics:
    lgbm_ic_arr = np.array(lgbm_ics)
    print(f"  LightGBM 1m IC: Mean={lgbm_ic_arr.mean():.4f} "
          f"Std={lgbm_ic_arr.std():.4f} "
          f"t-stat={lgbm_ic_arr.mean()/lgbm_ic_arr.std()*np.sqrt(len(lgbm_ic_arr)):.2f}")


# ════════════════════════════════════════════════════════════════════════════
# E. LightGBM on risk-adjusted label (fwd_ret / realized_vol)
# ════════════════════════════════════════════════════════════════════════════
print("\n-- E. LightGBM on Sharpe label (fwd_ret / vol_1m) --")

# Build Sharpe-label training frame
rows_sharpe = []
for m in months:
    sub = panel_full2[panel_full2["asof"] == m].copy()
    pos = mr_idx.searchsorted(m)
    next_ret_date = None
    for p in (pos - 1, pos):
        if 0 <= p < len(mr_idx) and abs((mr_idx[p] - m).days) <= 7:
            if p + 1 < len(mr_idx):
                next_ret_date = mr_idx[p + 1]
                break
    if next_ret_date is None:
        continue
    fwd = mr.loc[next_ret_date] if next_ret_date in mr.index else pd.Series(dtype=float)
    for _, row in sub.iterrows():
        tk = row["ticker"]
        fwd_r = fwd.get(tk, np.nan)
        if pd.isna(fwd_r):
            continue
        vol3m = row.get("vol_3m", row.get("vol_1y", 0.3))
        if pd.isna(vol3m) or vol3m < 0.01:
            vol3m = 0.3
        sharpe_label = float(fwd_r) / (vol3m / np.sqrt(12))  # scale vol to monthly
        feat_row = {"asof": m, "ticker": tk, "sharpe_label": sharpe_label,
                     "fwd_1m": float(fwd_r)}
        for f in FEATURES:
            feat_row[f] = row.get(f, np.nan)
        rows_sharpe.append(feat_row)

train_sharpe = pd.DataFrame(rows_sharpe)
print(f"  Sharpe-label frame: {train_sharpe.shape}")

# Walk-forward LightGBM on Sharpe label
lgbm_sharpe_scores = {}
model_sh = None
last_retrain_sh = None
model_trained_sh = False

for i, m in enumerate(training_months):
    need_retrain = (
        last_retrain_sh is None or
        (m.year - last_retrain_sh.year) * 12 + (m.month - last_retrain_sh.month) >= RETRAIN_FREQ
    )
    if need_retrain:
        train_end = pd.Timestamp(m.year, m.month, 1) - pd.DateOffset(months=1)
        train_start = train_end - pd.DateOffset(months=TRAIN_WINDOW - 1)
        mask = (train_sharpe["asof"] >= train_start) & (train_sharpe["asof"] <= train_end)
        train_sub = train_sharpe[mask].dropna(subset=FEATURES + ["sharpe_label"])
        if len(train_sub) < 500:
            lgbm_sharpe_scores[m] = None
            continue
        train_sub = train_sub.copy()
        train_sub["label"] = train_sub.groupby("asof")["sharpe_label"].rank(pct=True)
        X = train_sub[FEATURES].values.astype(np.float32)
        y = train_sub["label"].values.astype(np.float32)
        groups = train_sub.groupby("asof").size().values
        for j in range(X.shape[1]):
            mask_nan = np.isnan(X[:, j])
            X[mask_nan, j] = np.nanmedian(X[:, j])
        try:
            model_sh = lgb.LGBMRanker(**lgb_params)
            model_sh.fit(X, y, group=groups)
            last_retrain_sh = m
            model_trained_sh = True
        except Exception as e:
            model_sh = None
    if not model_trained_sh or model_sh is None:
        lgbm_sharpe_scores[m] = None
        continue
    sub = panel_full2[panel_full2["asof"] == m].copy()
    X = sub[FEATURES].values.astype(np.float32)
    for j in range(X.shape[1]):
        mask_nan = np.isnan(X[:, j])
        X[mask_nan, j] = np.nanmedian(X[:, j])
    try:
        preds = model_sh.predict(X)
        lgbm_sharpe_scores[m] = dict(zip(sub["ticker"], preds))
    except Exception:
        lgbm_sharpe_scores[m] = None


def simulate_lgbm_sharpe(K: int = 3, scores_dict=None) -> pd.DataFrame:
    """Simulate using walk-forward LightGBM Sharpe-label scores."""
    if scores_dict is None:
        scores_dict = lgbm_sharpe_scores
    by_asof = {pd.Timestamp(d): g for d, g in panel_full2.groupby("asof")}
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
                cur_picks, cur_weights, cash_flag = [], np.array([]), True; held_for = 0
            elif scores is None:
                sub = by_asof.get(m, pd.DataFrame()).copy()
                if not sub.empty:
                    sub["score"] = H.score_ml_3plus6(sub)
                    picks = sub.nlargest(K, "score")
                    cur_picks = picks["ticker"].tolist()
                    cur_weights = np.ones(K) / K
                    cash_flag = False
                held_for = 0
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
                ret_m = float(np.mean(pick_rets))

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
    eq = simulate_lgbm_sharpe(K=K)
    met = metrics_detail(eq)
    met.update({"exp_name": f"lgbm_sharpe_K{K}", "K": K,
                 "wall_time_s": round(time.time() - t0, 2)})
    record(f"lgbm_sharpe_K{K}", met)


# ════════════════════════════════════════════════════════════════════════════
# Summary
# ════════════════════════════════════════════════════════════════════════════

results_df = pd.DataFrame(all_results)
results_df.to_csv(EXP_DIR / "summary.csv", index=False)

print("\n\n=== TOP BY SHARPE ===")
cols = ["exp_name", "K", "cagr", "sharpe", "max_dd", "sub_sharpe_min"]
print(results_df.sort_values("sharpe", ascending=False).head(15)[cols].to_string(index=False))

print("\n=== TOP BY CAGR ===")
print(results_df.sort_values("cagr", ascending=False).head(15)[cols].to_string(index=False))

log_hyps(hyp_count)
print(f"\nTotal hypotheses this batch: {hyp_count}")
