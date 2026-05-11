"""Tactical (signal-decay-triggered) rebalance simulator.

Current production: rebalance every 6 months on a fixed schedule. Picker's
between-rebalance output is ignored.

Tactical approach: re-evaluate the picker EVERY month. Trigger a rebalance
when the current basket has "drifted out of conviction" by some principled
rule. Otherwise hold. Min hold = 3 months (avoid churn), max hold = 12
months (avoid stale dead-money). Plus the regime gate stays untouched.

Variants tested (each is a different drift detector, all principle-based
not curve-fit):

  T1_strict_any20   — rebalance if ANY held stock falls below score-rank 80%
  T2_any30          — rebalance if ANY held stock falls below score-rank 70%
  T3_loose_avg50    — rebalance if AVG held-rank falls below 50%
  T4_swap_individual — every month, swap individual stocks that fall out of
                       top 30% (partial rotation, not full rebalance)
  T5_decay_combined — rebalance if either (a) avg held-rank < 50% OR
                       (b) any stock falls below 80% (combination)

All variants use the same K=3 GBM+Chronos picker, inv-vol cap 0.40,
PIT S&P 500 universe, look-ahead-fixed return application.
"""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import numpy as np

from experiments.monthly_dca.v5.validations.harness import (
    HarnessData, load_all, classify_regime_tight, invvol_weights,
    CHRONOS_FILTER_Q, CAP_PER_PICK, COST_BPS, K_PICKS,
)
from experiments.monthly_dca.v2.ml_strategy import EXCLUDE

RES = Path(__file__).resolve().parent / "results"


def _score_at(asof, eligible, data: HarnessData) -> pd.DataFrame:
    """Return DataFrame of (ticker, score, score_rank) for eligible cohort,
    after Chronos p70 q=0.45 filter."""
    sub = data.ml_v2[data.ml_v2["asof"] == asof].copy()
    sub = sub[sub["ticker"].isin(eligible)]
    if len(sub) == 0:
        return pd.DataFrame(columns=["ticker", "score", "score_rank"])
    sub["score"] = (sub["pred_3m"] + sub["pred_6m"]) / 2
    # Chronos gate
    ch = data.chronos.get(asof, {})
    if ch:
        sub["chr"] = sub["ticker"].map(ch)
        sub["chr_rk"] = sub["chr"].rank(pct=True)
        sub = sub[sub["chr_rk"] >= CHRONOS_FILTER_Q]
    sub["score_rank"] = sub["score"].rank(pct=True)
    return sub[["ticker", "score", "score_rank"]].reset_index(drop=True)


def _form_basket(scored: pd.DataFrame, data: HarnessData, asof,
                  k: int = K_PICKS) -> tuple[list[str], np.ndarray]:
    """Form top-K basket and inv-vol weights."""
    if len(scored) < k:
        return [], np.array([])
    top = scored.sort_values("score", ascending=False).head(k)
    picks = top["ticker"].tolist()
    weights = invvol_weights(picks, data.mret, asof, cap=CAP_PER_PICK)
    return picks, weights


def run_tactical_sim(data: HarnessData,
                     drift_detector,        # callable(scored, held) → bool
                     start: pd.Timestamp,
                     end: pd.Timestamp,
                     k: int = K_PICKS,
                     min_hold: int = 3,
                     max_hold: int = 12,
                     cost_bps: float = COST_BPS,
                     swap_individual: bool = False) -> dict:
    """Tactical rebalance simulator. The drift_detector decides whether to
    rebalance. If swap_individual=True, individual stocks are swapped each
    month based on `drift_detector` per-stock instead of full rebalance.
    """
    cf = cost_bps / 1e4
    asofs = [m for m in data.asofs
             if start <= m <= end
             and m in data.spy_features.index
             and m in data.mret.index
             and m in data.members_g]
    asofs = sorted(asofs)

    cur_picks: list[str] = []
    cur_weights = np.array([])
    cash = False
    months_since_entry = 0
    n_rebalances = 0
    n_swaps = 0
    last_regime = "normal"
    equity = 1.0
    log = []

    for i, m in enumerate(asofs):
        spy_now = data.spy_features.loc[m].to_dict() if m in data.spy_features.index else {}
        regime = classify_regime_tight(spy_now)

        # Apply CURRENT month's return on basket from prior iteration
        if cash or not cur_picks:
            ret_m = 0.0
        else:
            r = 0.0
            for tk, w in zip(cur_picks, cur_weights):
                rt = (float(data.mret.at[m, tk])
                      if (tk in data.mret.columns and m in data.mret.index
                          and pd.notna(data.mret.at[m, tk]))
                      else 0.0)
                r += w * rt
            ret_m = r

        # Regime crash → exit to cash
        if regime == "crash" and cur_picks:
            cur_picks, cur_weights = [], np.array([])
            cash = True
            months_since_entry = 0

        eligible = data.members_g.get(m, set()) - set(EXCLUDE)
        scored = _score_at(m, eligible, data)
        do_full_reb = False
        action = "hold"

        if regime != "crash":
            if not cur_picks or cash:
                # Initial or post-crash re-entry → rebalance
                do_full_reb = True
                action = "init"
            elif months_since_entry >= max_hold:
                # Force rebalance at max hold
                do_full_reb = True
                action = "max_hold"
            elif months_since_entry < min_hold:
                # Skip rotation check, holding
                action = "hold_min"
            elif swap_individual:
                # Per-stock swap logic
                if len(scored) > 0:
                    swap_mask = drift_detector(scored, cur_picks)  # bool per pick
                    if any(swap_mask):
                        # Replace each "drift" pick with the next-best unheld stock
                        held_set = set(cur_picks)
                        ranked = scored.sort_values("score", ascending=False)
                        candidates = ranked[~ranked["ticker"].isin(held_set)]["ticker"].tolist()
                        new_picks = list(cur_picks)
                        cidx = 0
                        for j, drift in enumerate(swap_mask):
                            if drift and cidx < len(candidates):
                                new_picks[j] = candidates[cidx]
                                cidx += 1
                                n_swaps += 1
                        cur_picks = new_picks
                        cur_weights = invvol_weights(cur_picks, data.mret, m, cap=CAP_PER_PICK)
                        # Cost on swaps: proportional to fraction rotated
                        rotate_frac = sum(swap_mask) / len(cur_picks)
                        ret_m -= cf * rotate_frac
                        action = f"swap_{sum(swap_mask)}"
            else:
                # Full rebalance threshold
                if len(scored) > 0:
                    do_full_reb = drift_detector(scored, cur_picks)
                    if do_full_reb:
                        action = "tactical_reb"

        if do_full_reb and regime != "crash":
            picks, weights = _form_basket(scored, data, m, k)
            if picks and len(picks) >= 1:
                cur_picks = list(picks)
                cur_weights = np.array(weights, dtype=float)
                if cur_weights.sum() == 0:
                    cur_weights = np.ones(len(cur_picks)) / len(cur_picks)
                else:
                    cur_weights = cur_weights / cur_weights.sum()
                cash = False
                months_since_entry = 0
                n_rebalances += 1
                if log:
                    ret_m -= cf

        if cur_picks:
            months_since_entry += 1
        equity *= (1 + ret_m)
        log.append({"date": str(m.date()), "regime": regime,
                     "action": action,
                     "ret_m": ret_m, "equity": equity,
                     "picks": ",".join(cur_picks) if cur_picks else "",
                     "n_picks": len(cur_picks),
                     "months_held": months_since_entry})
        last_regime = regime

    return {"log": log, "n_rebalances": n_rebalances, "n_swaps": n_swaps}


# Drift detectors --------------------------------------------------------

def detect_any_below(rank_threshold: float):
    """Return True if any held stock has score-rank below `rank_threshold`."""
    def fn(scored, held):
        rk_map = dict(zip(scored["ticker"], scored["score_rank"]))
        ranks = [rk_map.get(tk, 0.0) for tk in held]
        return any(r < rank_threshold for r in ranks)
    return fn


def detect_avg_below(rank_threshold: float):
    """Return True if AVG held-rank below threshold."""
    def fn(scored, held):
        rk_map = dict(zip(scored["ticker"], scored["score_rank"]))
        ranks = [rk_map.get(tk, 0.0) for tk in held]
        return (sum(ranks) / max(len(ranks), 1)) < rank_threshold
    return fn


def detect_combined(any_threshold, avg_threshold):
    """Either rule fires → rebalance."""
    a = detect_any_below(any_threshold)
    b = detect_avg_below(avg_threshold)
    def fn(scored, held):
        return a(scored, held) or b(scored, held)
    return fn


def detect_per_stock(rank_threshold: float):
    """Return bool list per held stock: True if its rank < threshold."""
    def fn(scored, held):
        rk_map = dict(zip(scored["ticker"], scored["score_rank"]))
        return [rk_map.get(tk, 0.0) < rank_threshold for tk in held]
    return fn


# Driver -----------------------------------------------------------------

def main():
    RES.mkdir(parents=True, exist_ok=True)
    data = load_all()
    spy = data.mret["SPY"].copy()
    spy.index = pd.to_datetime(spy.index)
    start = data.asofs[0]
    end = data.spy_features.index.max()
    print(f"Loaded. {start.date()} → {end.date()}\n")

    variants = [
        # All variants now have min_hold=6 to match production signal horizon
        ("T1_strict_any5",  detect_any_below(0.05), False, 6,
            "Min-hold 6m, full reb if ANY held drops below rank 5% (very strict)"),
        ("T2_strict_any10", detect_any_below(0.10), False, 6,
            "Min-hold 6m, full reb if ANY held drops below rank 10%"),
        ("T3_strict_any20", detect_any_below(0.20), False, 6,
            "Min-hold 6m, full reb if ANY held drops below rank 20%"),
        ("T4_strict_avg30", detect_avg_below(0.30), False, 6,
            "Min-hold 6m, full reb if AVG held-rank drops below 30%"),
        ("T5_strict_avg50", detect_avg_below(0.50), False, 6,
            "Min-hold 6m, full reb if AVG held-rank drops below 50%"),
        ("T6_combined_10_40", detect_combined(0.10, 0.40), False, 6,
            "Min-hold 6m, combined ANY<10% OR AVG<40%"),
        ("T7_swap_10", detect_per_stock(0.10), True, 6,
            "Min-hold 6m, swap individual stocks if drop below 10%"),
        ("T8_swap_20", detect_per_stock(0.20), True, 6,
            "Min-hold 6m, swap individual stocks if drop below 20%"),
        # Plus a "extend-only" variant: production schedule BUT extend hold
        # if picker still likes the basket at the scheduled rebalance moment
        ("T9_extend_only", detect_any_below(0.50), False, 6,
            "Min-hold 6m, extend if ANY ≥ rank 50% (else rebalance)"),
    ]

    summary = []
    for name, detector, swap, min_h, desc in variants:
        print(f"\n{'='*70}\n  {name}\n  {desc}\n{'='*70}")
        sim = run_tactical_sim(data, detector, start, end,
                                k=K_PICKS, min_hold=min_h, max_hold=12,
                                swap_individual=swap)
        log = sim["log"]
        df = pd.DataFrame(log)
        df["date"] = pd.to_datetime(df["date"])
        df["year"] = df["date"].dt.year
        n_months = len(log)
        years = n_months / 12
        final_eq = df["equity"].iloc[-1]
        cagr = (final_eq ** (1/years) - 1) * 100

        # SPY same window
        spy_w = spy.loc[df["date"].iloc[0]:df["date"].iloc[-1]]
        cagr_spy = ((1 + spy_w.fillna(0)).cumprod().iloc[-1] ** (12/len(spy_w)) - 1) * 100

        # Sharpe & MaxDD
        rr = df["ret_m"]
        sh = float(rr.mean() / rr.std() * np.sqrt(12)) if rr.std() > 0 else 0.0
        peak = df["equity"].cummax()
        mdd = float(((df["equity"] - peak) / peak).min() * 100)

        # Year-by-year
        yr = df.groupby("year")["ret_m"].apply(lambda x: (1+x).prod()-1) * 100
        spy_yr = spy_w.groupby(spy_w.index.year).apply(lambda x: (1+x).prod()-1) * 100
        edges = []
        for y in sorted(yr.index):
            if y in spy_yr.index:
                edges.append(yr[y] - spy_yr[y])
        edges_arr = np.array(edges)
        e2024 = yr.get(2024, 0) - spy_yr.get(2024, 0) if 2024 in yr.index and 2024 in spy_yr.index else 0
        e2025 = yr.get(2025, 0) - spy_yr.get(2025, 0) if 2025 in yr.index and 2025 in spy_yr.index else 0

        print(f"  CAGR {cagr:.2f}% vs SPY {cagr_spy:.2f}% (edge {cagr-cagr_spy:+.2f}pp)")
        print(f"  Sharpe {sh:.2f}  MaxDD {mdd:.1f}%")
        print(f"  Rebalances: {sim['n_rebalances']} (full) + {sim['n_swaps']} (swap) "
              f"over {years:.1f}y (avg hold {years*12/max(sim['n_rebalances'],1):.1f}m)")
        print(f"  Year-edge: mean {edges_arr.mean():+.2f}pp  std {edges_arr.std():.2f}pp  "
              f"min {edges_arr.min():+.2f}pp  max {edges_arr.max():+.2f}pp")
        print(f"  2024 edge {e2024:+.2f}pp  2025 edge {e2025:+.2f}pp")

        df.to_csv(RES / f"{name}_equity.csv", index=False)
        summary.append({
            "variant": name, "description": desc,
            "cagr_pct": float(cagr), "spy_cagr_pct": float(cagr_spy),
            "edge_pp": float(cagr - cagr_spy),
            "sharpe": sh, "max_dd_pct": mdd,
            "n_rebalances": sim["n_rebalances"], "n_swaps": sim["n_swaps"],
            "avg_hold_months": years*12/max(sim['n_rebalances'], 1),
            "year_edge_mean_pp": float(edges_arr.mean()),
            "year_edge_std_pp": float(edges_arr.std()),
            "year_edge_min_pp": float(edges_arr.min()),
            "year_edge_max_pp": float(edges_arr.max()),
            "y2024_edge_pp": float(e2024),
            "y2025_edge_pp": float(e2025),
        })

    df_sum = pd.DataFrame(summary)
    df_sum.to_csv(RES / "tactical_summary.csv", index=False)
    print("\n\n=== TACTICAL REBALANCE SWEEP ===")
    print(f"{'Variant':<22} {'CAGR':>7} {'edge':>8} {'Sharpe':>7} {'MDD':>7} {'reb':>4} {'avgHold':>8} {'std':>8} {'2024':>8}")
    for r in summary:
        print(f"{r['variant']:<22} {r['cagr_pct']:>6.2f}% {r['edge_pp']:>+6.2f}pp {r['sharpe']:>7.2f} {r['max_dd_pct']:>6.1f}% {r['n_rebalances']:>4} {r['avg_hold_months']:>6.1f}m {r['year_edge_std_pp']:>+6.2f}pp {r['y2024_edge_pp']:>+6.2f}pp")
    # Reference: production K=3 H=6 = 43.79% CAGR, std 51.84pp, 2024 -14.76pp
    print(f"\n{'PRODUCTION (ref)':<22} {'43.79%':>7} {'+32.0pp':>8} {'1.00':>7} {'-51.4%':>7} {'~44':>4} {'~6.0m':>8} {'+51.84pp':>8} {'-14.76pp':>8}")


if __name__ == "__main__":
    main()
