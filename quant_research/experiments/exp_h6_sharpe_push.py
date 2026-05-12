"""Session 6 — Sharpe Push Experiments.

Objective: Close the gap from Sharpe=1.0 to ≥2.0.

Approach:
  1. Quality filter: exclude distressed/high-vol names
  2. Invvol weighting: weight by 1/vol_3m at stock level
  3. K sensitivity with quality: K=3,5,7,10,15 quality-filtered
  4. Monthly rebalance (hold=1): faster crash exit at higher cost
  5. Vol-targeting overlay: portfolio-level exposure scaling
  6. Composite: quality + invvol + crash gate + K sweep

All experiments use the research window (2003-09 → 2024-04).
Hypotheses tracked: see hypotheses_tested.jsonl.
"""

from __future__ import annotations

import hashlib, json, os, sys, time, warnings
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── paths ───────────────────────────────────────────────────────────────────
REPO = Path("/home/user/crt")
HARNESS = REPO / "strategy/YLOka/harness.py"
DATA = REPO / "data/YLOka"
CACHE = REPO / "experiments/monthly_dca/cache"
RUNS = REPO / "backtests/YLOka/runs"
QR = REPO / "quant_research"
JOURNAL = QR / "state/journal.jsonl"
HYPS = QR / "state/hypotheses_tested.jsonl"
EXP_DIR = QR / "experiments/exp_h6_sharpe_push"

EXP_DIR.mkdir(parents=True, exist_ok=True)

# Load the existing harness
sys.path.insert(0, str(REPO / "strategy/YLOka"))
import harness as H  # noqa

# ── data loading ────────────────────────────────────────────────────────────
print("Loading data...")
panel_full = H.load_panel_full()
mr = H.load_monthly_returns()
spy = H.load_spy_features()
prices = H.load_prices()
print(f"Panel: {panel_full.shape}, MR: {mr.shape}, SPY: {spy.shape}")

RESEARCH_END = H.RESEARCH_END

# ── helpers ─────────────────────────────────────────────────────────────────

def metrics_detail(eq: pd.DataFrame) -> dict:
    """Extended metrics."""
    met = H.metrics(eq)
    # Sub-period stationarity (3 equal chunks)
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


def log_hypotheses(n: int, run_id: str):
    with open(HYPS, "a") as f:
        f.write(json.dumps({"ts": datetime.utcnow().isoformat() + "Z",
                             "run_id": run_id, "n_hparams": n}) + "\n")


def log_journal(entry: dict):
    with open(JOURNAL, "a") as f:
        f.write(json.dumps(entry) + "\n")


# ── new scorer: quality-gated ML ────────────────────────────────────────────

def make_quality_gate_scorer(vol_ceiling: float = 0.50, quality_floor: float = 0.0):
    """Factory: returns a scorer that excludes high-vol / low-quality names.

    quality_composite = 0.4 * trend_health_5y + 0.3 * frac_above_50dma_1y
                      + 0.2 * sharpe_5y_scaled + 0.1 * (1 - dd_floor_scaled)
    Then gate: exclude names with vol_1y > vol_ceiling OR quality < quality_floor.
    """
    def scorer(panel_at: pd.DataFrame) -> pd.Series:
        base = (panel_at["pred_3m"] + panel_at["pred_6m"]) / 2
        # Vol gate
        vol = panel_at.get("vol_1y", panel_at.get("vol_12m"))
        if vol is not None:
            high_vol = vol.fillna(0.5) > vol_ceiling
            base = base.where(~high_vol, -1e9)
        # Quality floor
        if quality_floor > 0:
            th5 = panel_at.get("trend_health_5y")
            frac50 = panel_at.get("frac_above_50dma_1y")
            if th5 is not None and frac50 is not None:
                qc = (0.5 * th5.fillna(0.5) + 0.5 * frac50.fillna(0.5))
                low_q = qc < quality_floor
                base = base.where(~low_q, -1e9)
        return base
    scorer.__name__ = f"quality_gate_v{vol_ceiling:.2f}_qf{quality_floor:.2f}"
    return scorer


def make_invvol_scorer(base_scorer_fn, vol_col: str = "vol_3m", w: float = 0.30):
    """Blend rank(base_score) + w * rank(1/vol)."""
    def scorer(panel_at: pd.DataFrame) -> pd.Series:
        base = base_scorer_fn(panel_at)
        base_r = H._safe_rank(base.replace(-1e9, np.nan))
        vol = panel_at.get(vol_col)
        if vol is None:
            return base_r
        inv_vol_r = H._safe_rank(1.0 / vol.replace(0, np.nan).fillna(0.5))
        # propagate exclusions from base
        excl = base <= -1e8
        combined = (1 - w) * base_r + w * inv_vol_r
        return combined.where(~excl, -1e9)
    scorer.__name__ = f"invvol_{base_scorer_fn.__name__}_{vol_col}_w{w}"
    return scorer


def make_vol_target_scorer(base_scorer_fn, target_vol: float = 0.10,
                             lookback: int = 3):
    """Wrapper that adds a portfolio-vol-targeting note to the equity curve.
    Actually implemented as a post-processing step in simulate_with_vol_target.
    This scorer is just the base scorer.
    """
    return base_scorer_fn


# ── vol-targeting overlay ────────────────────────────────────────────────────

def simulate_with_vol_target(cfg: H.StratConfig, panel: pd.DataFrame,
                              mr: pd.DataFrame, spy_features: pd.DataFrame,
                              prices: Optional[pd.DataFrame] = None,
                              target_vol_annual: float = 0.12,
                              lookback_months: int = 3,
                              start=None, end=None) -> pd.DataFrame:
    """Run strategy then apply vol-targeting overlay.

    At each month t, realized portfolio vol over lookback_months is computed.
    Equity exposure = min(1.0, target_vol / realized_vol).
    Cash makes up the rest.
    """
    # Run raw simulation first
    eq = H.simulate(cfg, panel, mr, spy_features, prices=prices,
                     start=start, end=end)

    # Apply vol-targeting overlay
    r = eq["ret_m"].astype(float).values
    cash_m = (1 + cfg.cash_yield_apr) ** (1 / 12) - 1
    cf = cfg.cost_bps_per_leg / 10000.0
    target_vol_monthly = target_vol_annual / np.sqrt(12)

    new_equity = 1.0
    new_rows = []
    prev_scale = 1.0

    for i, row in enumerate(eq.itertuples()):
        raw_ret = float(row.ret_m)

        if row.regime == "cash":
            # Already in cash
            new_rows.append({"scale": 0.0, "ret_m_scaled": raw_ret})
            new_equity *= (1 + raw_ret)
            continue

        # Compute recent portfolio vol
        if i < lookback_months:
            scale = 1.0
        else:
            past_r = np.array([new_rows[j]["ret_m_scaled"] for j in
                                range(max(0, i - lookback_months), i)])
            if len(past_r) >= 2:
                realized_vol = float(np.std(past_r, ddof=1))
                if realized_vol > 1e-6:
                    scale = min(1.0, target_vol_monthly / realized_vol)
                else:
                    scale = 1.0
            else:
                scale = 1.0

        # Scale the return
        # scale = equity weight, (1-scale) = cash
        scaled_ret = scale * raw_ret + (1 - scale) * cash_m
        # Transaction cost if scale changed significantly
        if abs(scale - prev_scale) > 0.05:
            scaled_ret -= cf * abs(scale - prev_scale)  # proportional cost

        new_rows.append({"scale": scale, "ret_m_scaled": scaled_ret})
        new_equity *= (1 + scaled_ret)
        prev_scale = scale

    # Build new equity dataframe
    eq2 = eq.copy()
    scales = pd.DataFrame(new_rows)
    eq2["ret_m"] = scales["ret_m_scaled"].values
    eq2["vol_scale"] = scales["scale"].values
    # Recompute equity curve
    eq2["equity"] = (1 + eq2["ret_m"]).cumprod()
    return eq2


# ── experiment runner ────────────────────────────────────────────────────────

all_results = []
hyp_count = 0

def run_exp(name: str, scorer_fn, K: int = 3, hold: int = 6,
            weighting: str = "ew", vol_target: Optional[float] = None,
            use_invvol: bool = False, invvol_w: float = 0.30):
    """Run one experiment, log, return metrics."""
    global hyp_count
    hyp_count += 1

    # Register scorer
    fn_name = f"exp_h6_{name}"
    H.SCORERS[fn_name] = scorer_fn

    cfg = H.StratConfig(
        name=fn_name,
        K=K,
        hold_months=hold,
        weighting=weighting,
        cost_bps_per_leg=5.0,
        crash_gate=True,
        soft_cash=False,
        score_fn_name=fn_name,
    )

    t0 = time.time()
    if vol_target is not None:
        eq = simulate_with_vol_target(cfg, panel_full, mr, spy,
                                       prices=prices if "donchian" in name else None,
                                       target_vol_annual=vol_target,
                                       start=None, end=RESEARCH_END)
    else:
        eq = H.simulate(cfg, panel_full, mr, spy,
                         prices=prices if "donchian" in name else None,
                         start=None, end=RESEARCH_END)

    met = metrics_detail(eq)
    met["wall_time_s"] = round(time.time() - t0, 2)
    met["window"] = "research"
    met["exp_name"] = name
    met["K"] = K
    met["hold"] = hold
    met["vol_target"] = vol_target

    wall = met["wall_time_s"]
    cagr = met.get("cagr", 0)
    sharpe = met.get("sharpe", 0)
    mdd = met.get("max_dd", 0)
    sub_min = met.get("sub_sharpe_min", 0)
    print(f"  {name:55s} K={K} h={hold}  "
          f"CAGR={cagr:5.1%}  Sharpe={sharpe:.2f}  MDD={mdd:.1%}  "
          f"SubMin={sub_min:.2f}  ({wall:.1f}s)")

    all_results.append(met)

    # Save equity curve
    run_dir = EXP_DIR / f"{name}_K{K}_h{hold}"
    run_dir.mkdir(exist_ok=True)
    eq.to_parquet(run_dir / "equity.parquet")
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(met, f, indent=2)

    return met


# ════════════════════════════════════════════════════════════════════════════
# EXPERIMENTS
# ════════════════════════════════════════════════════════════════════════════

print("\n=== H6 Sharpe Push Experiments ===")
print(f"{'Name':55s} K    h    CAGR    Sharpe  MDD    SubMin")
print("-" * 100)

# ── Baseline reproduction ────────────────────────────────────────────────────
print("\n-- Baseline --")
run_exp("baseline", H.score_ml_3plus6, K=3, hold=6)

# ── Quality filter experiments ───────────────────────────────────────────────
print("\n-- Quality filter (vol ceiling) --")
for vc in [0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
    qg = make_quality_gate_scorer(vol_ceiling=vc, quality_floor=0.0)
    run_exp(f"qualvol{vc:.2f}", qg, K=3, hold=6)

# Quality + floor
print("\n-- Quality filter (vol + quality floor) --")
for vc, qf in [(0.45, 0.30), (0.45, 0.40), (0.50, 0.30), (0.50, 0.40), (0.55, 0.30)]:
    qg = make_quality_gate_scorer(vol_ceiling=vc, quality_floor=qf)
    run_exp(f"qualvol{vc:.2f}_qf{qf:.2f}", qg, K=3, hold=6)

# ── Invvol weighting ─────────────────────────────────────────────────────────
print("\n-- Invvol weighting (scorer blend) --")
for w in [0.15, 0.25, 0.35, 0.50]:
    iv = make_invvol_scorer(H.score_ml_3plus6, vol_col="vol_3m", w=w)
    run_exp(f"invvol_w{w:.2f}", iv, K=3, hold=6)

# Best quality + invvol
print("\n-- Quality + invvol combo --")
for vc in [0.45, 0.50, 0.55]:
    qg = make_quality_gate_scorer(vol_ceiling=vc, quality_floor=0.0)
    iv_qg = make_invvol_scorer(qg, vol_col="vol_3m", w=0.25)
    run_exp(f"qualinvvol_vc{vc:.2f}", iv_qg, K=3, hold=6)

# ── K sweep with quality filter ──────────────────────────────────────────────
print("\n-- K sweep with quality filter (vol<=0.50) --")
qg50 = make_quality_gate_scorer(vol_ceiling=0.50, quality_floor=0.0)
for K in [2, 3, 5, 7, 10, 15, 20]:
    run_exp(f"qualK{K}", qg50, K=K, hold=6)

# ── K sweep invvol + quality ─────────────────────────────────────────────────
print("\n-- K sweep: quality + invvol weighting --")
iv_qg50 = make_invvol_scorer(qg50, vol_col="vol_3m", w=0.25)
for K in [3, 5, 7, 10, 15]:
    H.SCORERS[f"invvol_qualK{K}"] = iv_qg50
    run_exp(f"invqK{K}", iv_qg50, K=K, hold=6)

# ── Hold period with quality filter ──────────────────────────────────────────
print("\n-- Hold period sweep (quality filtered) --")
for hold in [1, 2, 3, 6, 9, 12]:
    run_exp(f"qualH{hold}", qg50, K=3, hold=hold)

# ── Vol-targeting overlay ────────────────────────────────────────────────────
print("\n-- Vol-targeting overlay --")
for vt in [0.08, 0.10, 0.12, 0.15, 0.18]:
    run_exp(f"voltgt{vt:.2f}", H.score_ml_3plus6, K=3, hold=6, vol_target=vt)

# Quality + vol target
print("\n-- Quality + vol target combo --")
for K, vt in [(3, 0.10), (5, 0.10), (3, 0.12), (5, 0.12), (7, 0.12), (10, 0.12)]:
    run_exp(f"qual_voltgt{vt:.2f}_K{K}", qg50, K=K, hold=6, vol_target=vt)

# ── Invvol WEIGHTING (at portfolio level, not score) ─────────────────────────
print("\n-- Invvol portfolio weighting --")

def make_invvol_cfg(K: int, vol_target: Optional[float] = None):
    """invvol weighting = weight each pick by 1/vol_3m, normalized."""
    cfg = H.StratConfig(
        name=f"invvol_port_K{K}",
        K=K,
        hold_months=6,
        weighting="ew",  # overridden below
        cost_bps_per_leg=5.0,
        crash_gate=True,
        score_fn_name="ml_3plus6",
    )
    return cfg


# We need a custom simulate for invvol portfolio weighting
def simulate_invvol_port(K: int, score_fn_name: str = "ml_3plus6",
                          vol_col: str = "vol_3m",
                          quality_vol_ceiling: float = 1.0,
                          vol_target: Optional[float] = None):
    """Simulate with invvol portfolio weights (1/vol_col, normalized).

    This requires a custom loop since the harness only supports ew/conv.
    """
    cfg = H.StratConfig(name=f"invvol_K{K}_vc{quality_vol_ceiling:.2f}",
                         K=K, hold_months=6, weighting="ew",
                         cost_bps_per_leg=5.0, crash_gate=True,
                         score_fn_name=score_fn_name)
    score_fn = H.SCORERS[score_fn_name]

    months = sorted(panel_full["asof"].unique())
    months = [pd.Timestamp(m) for m in months if pd.Timestamp(m) <= RESEARCH_END]
    by_asof = {pd.Timestamp(d): g for d, g in panel_full.groupby("asof")}
    mr_idx = mr.index
    cf = cfg.cost_bps_per_leg / 10000.0
    cash_m = 0.0

    equity = 1.0
    cur_picks = []
    cur_weights = np.array([])
    held_for = 0
    cash_flag = False
    rows = []

    for i, m in enumerate(months):
        do_reb = (i == 0) or (held_for >= cfg.hold_months) or cash_flag
        spy_now = spy.loc[m].to_dict() if m in spy.index else {}
        regime = H.regime_tight(spy_now) if cfg.crash_gate else "normal"

        if do_reb:
            sub = by_asof.get(m, pd.DataFrame()).copy()
            if not sub.empty:
                sub["score"] = score_fn(sub)
                sub = sub.dropna(subset=["score"])
                sub = sub[sub["score"] > -1e8]  # exclude filtered
            if regime == "crash":
                cur_picks, cur_weights, cash_flag = [], np.array([]), True
                held_for = 0
            elif sub.empty:
                cur_picks, cur_weights, cash_flag = [], np.array([]), True
                held_for = 0
            else:
                # Apply quality vol ceiling
                if quality_vol_ceiling < 1.0 and vol_col in sub.columns:
                    sub = sub[sub[vol_col].fillna(0.5) <= quality_vol_ceiling]
                if sub.empty:
                    cur_picks, cur_weights, cash_flag = [], np.array([]), True
                    held_for = 0
                else:
                    picks = sub.nlargest(K, "score")
                    # Invvol weights
                    if vol_col in picks.columns:
                        inv_vol = 1.0 / picks[vol_col].replace(0, 0.01).fillna(0.25)
                        w = inv_vol.values / inv_vol.sum()
                    else:
                        w = np.ones(len(picks)) / len(picks)
                    cur_picks = picks["ticker"].tolist()
                    cur_weights = w
                    cash_flag = False
                    held_for = 0

        # Return computation
        pos1 = mr_idx.searchsorted(m)
        if cash_flag or len(cur_picks) == 0:
            ret_m = cash_m
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
                pick_rets = []
                for tk in cur_picks:
                    r = mr.at[next_d, tk] if tk in mr.columns else np.nan
                    pick_rets.append(-1.0 if pd.isna(r) else float(r))
                ret_m = float(np.dot(np.array(pick_rets), cur_weights))

        if do_reb and not cash_flag and len(cur_picks) > 0:
            equity *= (1 + ret_m) * (1 - cf)
        else:
            equity *= 1 + ret_m

        held_for += 1
        rows.append({
            "date": m, "equity": equity, "ret_m": ret_m,
            "regime": "cash" if cash_flag else regime,
            "n_picks": len(cur_picks),
            "picks": ",".join(cur_picks),
        })

    eq = pd.DataFrame(rows)
    if vol_target is not None:
        eq = _apply_vol_target(eq, vol_target, cash_m, cf)
    return eq


def _apply_vol_target(eq: pd.DataFrame, target_vol_annual: float,
                       cash_m: float, cf: float) -> pd.DataFrame:
    """Post-process equity curve with vol-targeting overlay."""
    target_vol_monthly = target_vol_annual / np.sqrt(12)
    lookback = 3
    new_rets = []
    prev_scale = 1.0
    eq2 = eq.copy()
    r = eq["ret_m"].astype(float).values

    for i in range(len(r)):
        if eq["regime"].iloc[i] == "cash":
            new_rets.append(r[i])
            continue
        if i < lookback:
            scale = 1.0
        else:
            past = np.array(new_rets[max(0, i - lookback):i])
            rv = np.std(past, ddof=1) if len(past) >= 2 else target_vol_monthly
            scale = min(1.0, target_vol_monthly / rv) if rv > 1e-6 else 1.0
        scaled = scale * r[i] + (1 - scale) * cash_m
        if abs(scale - prev_scale) > 0.05:
            scaled -= cf * abs(scale - prev_scale)
        new_rets.append(scaled)
        prev_scale = scale

    eq2["ret_m"] = new_rets
    eq2["equity"] = (1 + eq2["ret_m"]).cumprod()
    return eq2


print("\n-- Invvol portfolio weights (custom sim) --")
for K, vc, vt in [
    (3, 1.0, None),   # baseline invvol weights
    (3, 0.50, None),  # quality filtered
    (5, 0.50, None),
    (7, 0.50, None),
    (10, 0.50, None),
    (10, 0.50, 0.12),  # + vol target
    (15, 0.50, None),
    (15, 0.50, 0.12),
    (5, 0.45, None),
    (5, 0.45, 0.10),
    (3, 0.50, 0.10),
]:
    t0 = time.time()
    eq = simulate_invvol_port(K=K, vol_col="vol_3m", quality_vol_ceiling=vc, vol_target=vt)
    met = metrics_detail(eq)
    met.update({
        "exp_name": f"invvol_port_K{K}_vc{vc}_vt{vt}",
        "K": K, "hold": 6, "vol_target": vt,
        "wall_time_s": round(time.time() - t0, 2)
    })
    hyp_count += 1
    all_results.append(met)
    cagr = met.get("cagr", 0)
    sharpe = met.get("sharpe", 0)
    mdd = met.get("max_dd", 0)
    sub_min = met.get("sub_sharpe_min", 0)
    print(f"  {'invvol_port K=' + str(K) + ' vc=' + str(vc) + ' vt=' + str(vt):55s} "
          f"CAGR={cagr:5.1%}  Sharpe={sharpe:.2f}  MDD={mdd:.1%}  SubMin={sub_min:.2f}  "
          f"({met['wall_time_s']:.1f}s)")

    run_dir = EXP_DIR / f"invvol_port_K{K}_vc{vc}_vt{vt}"
    run_dir.mkdir(exist_ok=True)
    eq.to_parquet(run_dir / "equity.parquet")
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(met, f, indent=2)


# ── Summary ──────────────────────────────────────────────────────────────────

results_df = pd.DataFrame(all_results)
results_df = results_df.sort_values("sharpe", ascending=False)

print("\n\n=== TOP 20 by SHARPE ===")
cols = ["exp_name", "K", "hold", "vol_target", "cagr", "sharpe", "max_dd",
        "sub_sharpe_min"]
print(results_df[cols].head(20).to_string(index=False))

print("\n=== TOP 20 by CAGR ===")
results_df2 = results_df.sort_values("cagr", ascending=False)
print(results_df2[cols].head(20).to_string(index=False))

# Save summary
results_df.to_csv(EXP_DIR / "summary.csv", index=False)

# Log journal and hypotheses
log_hypotheses(hyp_count, "exp_h6_sharpe_push")
log_journal({
    "ts": datetime.utcnow().isoformat() + "Z",
    "exp_id": "h6_sharpe_push",
    "hypothesis": "Quality filter + invvol weighting + vol targeting can close Sharpe gap from 1.0 to target",
    "what_i_did": f"Ran {hyp_count} experiments: quality gates, invvol weighting, K sweep, hold sweep, vol targeting",
    "result": {
        "n_experiments": hyp_count,
        "best_sharpe": float(results_df["sharpe"].max()),
        "best_sharpe_exp": str(results_df.iloc[0]["exp_name"]),
        "best_sharpe_cagr": float(results_df.iloc[0]["cagr"]),
        "best_cagr": float(results_df2.iloc[0]["cagr"]),
        "best_cagr_exp": str(results_df2.iloc[0]["exp_name"]),
    },
    "hparams_tried": hyp_count,
    "next_action": "Analyze results, pick top configurations, run gauntlet if Sharpe >=1.5",
})

print(f"\nTotal hypotheses this session: {hyp_count}")
print(f"Results saved to: {EXP_DIR}")
