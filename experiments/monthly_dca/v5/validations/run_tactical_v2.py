"""Tactical rebalance V2: novel rotation triggers, K=3 PIT S&P 500.

After the V1 rank-decay variants underperformed production (CAGR 22-32% vs
production 44%), here we test signal-AGE based triggers and regime-adaptive
hold periods.

Variants:
  V1  picker_overlap       — rebalance only when new top-K overlap with
                              held < threshold (e.g., 1 of 3)
  V2  dispersion_gate      — rebalance at H=6 schedule BUT skip if cross-
                              sectional score dispersion is LOW (uncertain)
  V3  regime_hold          — Bull H=9, Recovery H=6, Normal H=3
  V4  conviction_skip      — H=6 schedule, but skip rebalance if avg held-
                              rank > 70% (high conviction → extend hold)
  V5  kelly_weighting      — H=6 schedule, weight picks by score/vol_1y
                              instead of inv-vol
  V6  opportunistic_swap   — every month, if best unheld score - worst
                              held score > buffer, swap one stock
"""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import numpy as np

from experiments.monthly_dca.v5.validations.harness import (
    HarnessData, load_all, classify_regime_tight, invvol_weights,
    CHRONOS_FILTER_Q, CAP_PER_PICK, COST_BPS, K_PICKS, HOLD_MONTHS,
)
from experiments.monthly_dca.v5.validations.run_tactical_rebalance import (
    _score_at, _form_basket,
)
from experiments.monthly_dca.v2.ml_strategy import EXCLUDE

RES = Path(__file__).resolve().parent / "results"


def kelly_weights(scored_subset: pd.DataFrame, data: HarnessData,
                   asof: pd.Timestamp, cap: float = CAP_PER_PICK):
    """Weight picks by score/vol_1y proxy for Kelly. Inputs: rows with
    `ticker` and `score` columns."""
    picks = scored_subset["ticker"].tolist()
    # Load vol_1y from features at asof
    fp = Path(__file__).resolve().parents[3] / "experiments" / "monthly_dca" / "cache" / "features" / f"{asof.strftime('%Y-%m-%d')}.parquet"
    if fp.exists():
        feat = pd.read_parquet(fp)
    else:
        # Fall back to inv-vol
        return invvol_weights(picks, data.mret, asof, cap=cap)
    raw = []
    for _, r in scored_subset.iterrows():
        tk = r["ticker"]
        score = float(r["score"])
        vol = float(feat.loc[tk, "vol_1y"]) if tk in feat.index and "vol_1y" in feat.columns else 0.30
        # Kelly-like: signal strength normalised by vol
        # Map score (0..1) → expected excess return (small)
        # Then Kelly fraction = (score - 0.5) / vol² as a rough proxy
        kelly = max(score - 0.5, 0.0) / max(vol, 0.05) ** 2
        raw.append(max(kelly, 0.001))
    w = np.array(raw)
    w = w / w.sum() if w.sum() > 0 else np.ones(len(w)) / len(w)
    # Apply cap
    for _ in range(20):
        over = w > cap
        if not over.any(): break
        excess = (w[over] - cap).sum()
        w[over] = cap
        if (~over).any():
            w[~over] += excess * w[~over] / w[~over].sum()
        else: break
    s = w.sum()
    return w / s if s > 0 else np.ones_like(w) / len(w)


def run_tactical_v2(data: HarnessData, mode: str,
                     start: pd.Timestamp, end: pd.Timestamp,
                     k: int = K_PICKS,
                     min_hold: int = 6,
                     max_hold: int = 12,
                     cost_bps: float = COST_BPS,
                     **kwargs) -> dict:
    """Unified driver for tactical V2 modes:
      'overlap_<n>': rebalance when overlap of new top-K with held < n
      'dispersion_<thr>': skip scheduled rebalance if score std < threshold
      'regime_hold': hold periods bull=9, recovery=6, normal=3
      'conviction_<thr>': skip scheduled if avg held-rank > thr (extend)
      'kelly': production H=6 but Kelly-weighted
      'opp_swap_<buf>': every month, swap if best_unheld - worst_held > buf
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
    n_rebs = 0
    n_swaps = 0
    equity = 1.0
    log = []

    for i, m in enumerate(asofs):
        spy_now = data.spy_features.loc[m].to_dict() if m in data.spy_features.index else {}
        regime = classify_regime_tight(spy_now)

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

        # Crash → cash
        if regime == "crash" and cur_picks:
            cur_picks, cur_weights = [], np.array([])
            cash = True
            months_since_entry = 0

        eligible = data.members_g.get(m, set()) - set(EXCLUDE)
        scored = _score_at(m, eligible, data)

        # Determine effective hold period for this iteration
        if mode == "regime_hold":
            eff_max = {"bull": 9, "recovery": 6, "normal": 3, "crash": 6}.get(regime, 6)
        else:
            eff_max = max_hold

        action = "hold"
        do_reb = False
        if regime != "crash":
            if not cur_picks or cash:
                do_reb = True; action = "init"
            elif months_since_entry >= eff_max:
                do_reb = True; action = "max_hold"
            elif months_since_entry < min_hold:
                action = "hold_min"
            else:
                # Mode-specific rebalance trigger
                if mode.startswith("overlap_"):
                    n_threshold = int(mode.split("_")[1])
                    new_top = scored.sort_values("score", ascending=False).head(k)["ticker"].tolist()
                    overlap = len(set(new_top) & set(cur_picks))
                    if overlap < n_threshold:
                        do_reb = True; action = f"overlap_{overlap}"
                elif mode.startswith("dispersion_"):
                    # Production schedule equivalent (H=6) BUT skip if low conviction
                    thr = float(mode.split("_")[1])
                    # Trigger every 6 months only
                    if months_since_entry >= 6:
                        disp = scored["score"].std() if len(scored) > 5 else 0
                        if disp >= thr:
                            do_reb = True; action = "disp_ok"
                        else:
                            action = "disp_skip"
                elif mode == "regime_hold":
                    # No additional trigger beyond eff_max (already handled)
                    pass
                elif mode.startswith("conviction_"):
                    thr = float(mode.split("_")[1])
                    # Trigger every 6 months only
                    if months_since_entry >= 6:
                        rk_map = dict(zip(scored["ticker"], scored["score_rank"]))
                        avg_rk = np.mean([rk_map.get(tk, 0.0) for tk in cur_picks])
                        if avg_rk < thr:  # held basket no longer top-conviction
                            do_reb = True; action = "low_conv"
                        else:
                            action = "extend"
                elif mode == "kelly":
                    if months_since_entry >= 6:
                        do_reb = True; action = "kelly_sched"
                elif mode.startswith("opp_swap_"):
                    buf = float(mode.split("_")[2])
                    # Always allow swap if buffer met
                    if len(scored) > k:
                        held_scores = scored[scored["ticker"].isin(cur_picks)]["score"].values
                        unheld = scored[~scored["ticker"].isin(cur_picks)]
                        if len(held_scores) and len(unheld):
                            best_unheld = unheld["score"].max()
                            worst_held = held_scores.min()
                            if best_unheld - worst_held > buf:
                                # Swap: replace worst held with best unheld
                                worst_tk = scored[scored["ticker"].isin(cur_picks)].sort_values("score").iloc[0]["ticker"]
                                best_tk = unheld.sort_values("score", ascending=False).iloc[0]["ticker"]
                                new_picks = [best_tk if tk == worst_tk else tk for tk in cur_picks]
                                cur_picks = new_picks
                                cur_weights = invvol_weights(cur_picks, data.mret, m, cap=CAP_PER_PICK)
                                ret_m -= cf / k  # cost on 1/k of capital
                                n_swaps += 1
                                action = "swapped"

        if do_reb and regime != "crash":
            picks, weights = _form_basket(scored, data, m, k)
            if mode == "kelly" and picks:
                # Recompute with kelly weights
                top_df = scored[scored["ticker"].isin(picks)].copy()
                top_df = top_df.set_index("ticker").loc[picks].reset_index()
                weights = kelly_weights(top_df, data, m, cap=CAP_PER_PICK)
            if picks and len(picks) >= 1:
                cur_picks = list(picks)
                cur_weights = np.array(weights, dtype=float)
                if cur_weights.sum() == 0:
                    cur_weights = np.ones(len(cur_picks)) / len(cur_picks)
                else:
                    cur_weights = cur_weights / cur_weights.sum()
                cash = False
                months_since_entry = 0
                n_rebs += 1
                if log:
                    ret_m -= cf

        if cur_picks:
            months_since_entry += 1
        equity *= (1 + ret_m)
        log.append({"date": str(m.date()), "regime": regime,
                     "action": action, "ret_m": ret_m, "equity": equity,
                     "picks": ",".join(cur_picks) if cur_picks else "",
                     "n_picks": len(cur_picks),
                     "months_held": months_since_entry})

    return {"log": log, "n_rebalances": n_rebs, "n_swaps": n_swaps}


def main():
    RES.mkdir(parents=True, exist_ok=True)
    data = load_all()
    spy = data.mret["SPY"].copy()
    spy.index = pd.to_datetime(spy.index)
    start = data.asofs[0]
    end = data.spy_features.index.max()

    variants = [
        ("V1_overlap_2",      "overlap_2",       "Reb when new top-3 overlap with held < 2"),
        ("V1_overlap_1",      "overlap_1",       "Reb when new top-3 overlap with held < 1"),
        ("V2_disp_005",       "dispersion_0.05", "H=6 schedule, skip if score std < 0.05"),
        ("V2_disp_010",       "dispersion_0.10", "H=6 schedule, skip if score std < 0.10"),
        ("V3_regime_hold",    "regime_hold",     "Bull H=9, Recovery H=6, Normal H=3"),
        ("V4_conv_80",        "conviction_0.80", "H=6 sched, skip rebalance if avg held-rank > 80%"),
        ("V4_conv_70",        "conviction_0.70", "H=6 sched, skip rebalance if avg held-rank > 70%"),
        ("V5_kelly",          "kelly",           "H=6 schedule, Kelly-weighted (score/vol²)"),
        ("V6_opp_005",        "opp_swap_0.05",   "Monthly swap if best_unheld - worst_held > 0.05"),
        ("V6_opp_010",        "opp_swap_0.10",   "Monthly swap if best_unheld - worst_held > 0.10"),
    ]

    rows = []
    for name, mode, desc in variants:
        print(f"\n{'='*70}\n  {name}: {desc}\n{'='*70}")
        try:
            sim = run_tactical_v2(data, mode, start, end)
        except Exception as e:
            print(f"  ERROR: {e}")
            continue
        log = sim["log"]
        df = pd.DataFrame(log); df["date"] = pd.to_datetime(df["date"]); df["year"] = df["date"].dt.year
        n_months = len(log); years = n_months / 12
        final_eq = df["equity"].iloc[-1]
        cagr = (final_eq ** (1/years) - 1) * 100
        spy_w = spy.loc[df["date"].iloc[0]:df["date"].iloc[-1]]
        cagr_spy = ((1 + spy_w.fillna(0)).cumprod().iloc[-1] ** (12/len(spy_w)) - 1) * 100
        rr = df["ret_m"]
        sh = float(rr.mean()/rr.std()*np.sqrt(12)) if rr.std() > 0 else 0.0
        peak = df["equity"].cummax()
        mdd = float(((df["equity"]-peak)/peak).min() * 100)
        yr = df.groupby("year")["ret_m"].apply(lambda x: (1+x).prod()-1)*100
        spy_yr = spy_w.groupby(spy_w.index.year).apply(lambda x: (1+x).prod()-1)*100
        edges = []
        for y in sorted(yr.index):
            if y in spy_yr.index: edges.append(yr[y]-spy_yr[y])
        e_arr = np.array(edges)
        e2024 = yr.get(2024, 0)-spy_yr.get(2024, 0) if 2024 in yr.index else 0
        e2025 = yr.get(2025, 0)-spy_yr.get(2025, 0) if 2025 in yr.index else 0
        avg_hold = years*12/max(sim['n_rebalances'], 1)
        print(f"  CAGR {cagr:.2f}% (edge {cagr-cagr_spy:+.2f}pp)  Sharpe {sh:.2f}  MDD {mdd:.1f}%")
        print(f"  Reb {sim['n_rebalances']} (full) + {sim['n_swaps']} (swap), avg hold {avg_hold:.1f}m")
        print(f"  Year-edge std {e_arr.std():.1f}pp min {e_arr.min():+.1f}pp  2024 {e2024:+.1f}pp  2025 {e2025:+.1f}pp")
        rows.append({"variant": name, "mode": mode, "desc": desc,
                     "cagr_pct": cagr, "spy_cagr_pct": cagr_spy,
                     "edge_pp": cagr - cagr_spy, "sharpe": sh, "max_dd_pct": mdd,
                     "n_rebs": sim["n_rebalances"], "n_swaps": sim["n_swaps"],
                     "avg_hold_m": avg_hold,
                     "year_edge_std_pp": float(e_arr.std()),
                     "year_edge_min_pp": float(e_arr.min()),
                     "y2024_edge_pp": float(e2024),
                     "y2025_edge_pp": float(e2025)})
        df.to_csv(RES / f"{name}_equity.csv", index=False)

    df_s = pd.DataFrame(rows)
    df_s.to_csv(RES / "tactical_v2_summary.csv", index=False)
    print("\n\n=== TACTICAL V2 SWEEP (PIT SP500) ===")
    print(f"{'Variant':<18} {'CAGR':>7} {'edge':>9} {'Sharpe':>7} {'MDD':>7} {'hold':>6} {'std':>8} {'2024':>8} {'2025':>8}")
    for r in rows:
        print(f"{r['variant']:<18} {r['cagr_pct']:>6.2f}% {r['edge_pp']:>+7.2f}pp {r['sharpe']:>7.2f} {r['max_dd_pct']:>6.1f}% {r['avg_hold_m']:>5.1f}m {r['year_edge_std_pp']:>+6.1f}pp {r['y2024_edge_pp']:>+6.1f}pp {r['y2025_edge_pp']:>+6.1f}pp")
    print(f"{'PRODUCTION':<18} {'43.79%':>7} {'+32.0pp':>9} {'1.00':>7} {'-51.4%':>7} {'6.0m':>6} {'+51.8pp':>8} {'-14.8pp':>8} {'+8.6pp':>8}")


if __name__ == "__main__":
    main()
