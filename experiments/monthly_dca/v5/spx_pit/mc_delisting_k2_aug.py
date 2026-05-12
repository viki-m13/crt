"""Phase 7d: Monte-Carlo synthetic-delisting overlay for K=2 lump-sum v5 on augmented PIT.

The deployed v5 K=3 has a bias-sensitivity table at
`experiments/monthly_dca/cache/v2/sp500_pit/v3_winner_bias_sensitivity.csv`
showing median CAGR drops to ~32 % at α=4 %/yr synthetic delisting.

K=2 is more concentrated — a single delist hits 50 % of the basket
instead of 33 %. The bias-MC needs to be re-run before we can claim
K=2 is deployment-safe.

For each α ∈ {0, 2, 4, 6, 8, 12, 16, 20} %/yr, run 30 iterations:
each pick has p_per_month = 1 - (1-α)^(1/12) probability of being
synthetically -100 % in the next month.

Compares K=3 (deployed) vs K=2 (proposed) under identical MC seeds.

Output:
  augmented/v5_k2_bias_sensitivity.csv      median/p10/p90 by alpha for K=2
  augmented/v5_k3_bias_sensitivity.csv      same for K=3 (sanity check)
  augmented/v5_k2_vs_k3_bias.csv            joined comparison
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sweep_v5_aug import (  # noqa: E402
    AUG, PIT, EXCLUDE, COST_BPS,
    classify_regime_tight, load_spy_features, calc_invvol_weights,
)


def simulate_with_delist(months, panel_by_asof, ml_by_asof, chr_by_asof,
                         members_g, monthly_returns, spy_features,
                         k, chr_q, hold, cap,
                         alpha_yr, seed) -> float:
    """Same picker as sweep run_one but with synthetic per-pick delisting.

    Returns final equity (so caller can compute CAGR).
    """
    rng = np.random.default_rng(seed)
    p_per_month = 1.0 - (1.0 - alpha_yr) ** (1.0 / 12.0) if alpha_yr > 0 else 0.0
    cf = COST_BPS / 1e4

    cur_picks = []
    cur_weights = np.array([])
    cash = False
    held_for = 0
    equity = 1.0

    for i, m in enumerate(months):
        regime = classify_regime_tight(
            spy_features.loc[m].to_dict() if m in spy_features.index else {})
        do_reb = (i == 0) or (held_for >= hold) or (cash != (regime == "crash"))

        if not cash and cur_picks:
            mr_pos = monthly_returns.index.searchsorted(m)
            if mr_pos + 1 < len(monthly_returns.index):
                next_d = monthly_returns.index[mr_pos + 1]
                pick_rets = []
                for tk in cur_picks:
                    # SYNTHETIC DELIST?
                    if p_per_month > 0 and rng.random() < p_per_month:
                        pick_rets.append(-1.0)
                        continue
                    if tk in monthly_returns.columns and next_d in monthly_returns.index:
                        r = monthly_returns.at[next_d, tk]
                        pick_rets.append(0.0 if pd.isna(r) else float(r))
                    else:
                        pick_rets.append(0.0)
                ret_m = float((np.array(pick_rets) * cur_weights).sum())
                equity *= (1 + ret_m)

        if do_reb:
            equity *= (1 - cf)
            if regime == "crash":
                cur_picks = []; cur_weights = np.array([]); cash = True
            else:
                sub_panel = panel_by_asof.get(m)
                sub_ml = ml_by_asof.get(m)
                sub_chr = chr_by_asof.get(m)
                if sub_panel is None or sub_ml is None:
                    cur_picks = []; cur_weights = np.array([])
                else:
                    sp_set = members_g.get(m, set())
                    sub = sub_panel[sub_panel["ticker"].isin(sp_set)]
                    sub = sub[~sub["ticker"].isin(EXCLUDE)]
                    sub = sub.merge(sub_ml[["ticker", "ml_score"]], on="ticker", how="left")
                    sub = sub.dropna(subset=["ml_score"])
                    if chr_q > 0 and sub_chr is not None and not sub_chr.empty:
                        sub = sub.merge(sub_chr[["ticker", "chronos_p70_3m"]],
                                        on="ticker", how="left")
                        sub = sub.dropna(subset=["chronos_p70_3m"])
                        sub["chr_p70_rk"] = sub["chronos_p70_3m"].rank(pct=True)
                        sub = sub[sub["chr_p70_rk"] >= chr_q]
                    sub = sub.sort_values("ml_score", ascending=False)
                    top = sub.head(k)
                    if len(top) < k:
                        cur_picks = []; cur_weights = np.array([])
                    else:
                        cur_picks = top["ticker"].tolist()
                        cur_weights = calc_invvol_weights(cur_picks, monthly_returns, m, cap=cap)
                cash = False
            held_for = 0
        else:
            held_for += 1

    n_months = len(months)
    return equity, n_months


def main():
    t0 = time.time()
    print("Loading augmented data ...")
    panel = pd.read_parquet(AUG / "sp500_pit_panel.parquet")
    panel["asof"] = pd.to_datetime(panel["asof"])
    ml = pd.read_parquet(AUG / "ml_preds.parquet")[["asof", "ticker", "pred_3m", "pred_6m"]]
    ml["asof"] = pd.to_datetime(ml["asof"])
    ml["ml_score"] = (ml["pred_3m"] + ml["pred_6m"]) / 2
    chr_ = pd.read_parquet(AUG / "ml_preds_chronos.parquet")[["asof", "ticker", "chronos_p70_3m"]]
    chr_["asof"] = pd.to_datetime(chr_["asof"])
    spy = load_spy_features()
    mr = pd.read_parquet(AUG / "monthly_returns_clean.parquet").fillna(0.0)
    if not isinstance(mr.index, pd.DatetimeIndex):
        mr.index = pd.to_datetime(mr.index)
    members = pd.read_parquet(PIT / "sp500_membership_monthly.parquet")
    members["asof"] = pd.to_datetime(members["asof"])
    members_g = members.groupby("asof")["ticker"].apply(set).to_dict()

    panel_by_asof = {a: g for a, g in panel.groupby("asof")}
    ml_by_asof = {a: g for a, g in ml.groupby("asof")}
    chr_by_asof = {a: g for a, g in chr_.groupby("asof")}
    months = sorted(set(panel["asof"]).intersection(set(spy.index)))
    months = [pd.Timestamp(m) for m in months]

    ALPHAS = [0.0, 0.02, 0.04, 0.06, 0.08, 0.12, 0.16, 0.20]
    N_ITERS = 30

    print(f"\nMC overlay: K ∈ {{2, 3}} × α ∈ {ALPHAS} × {N_ITERS} seeds = "
          f"{2 * len(ALPHAS) * N_ITERS} simulations")

    rows = []
    for k in (2, 3):
        for alpha in ALPHAS:
            cagrs = []
            for seed in range(N_ITERS):
                eq_final, n_months = simulate_with_delist(
                    months, panel_by_asof, ml_by_asof, chr_by_asof,
                    members_g, mr, spy,
                    k=k, chr_q=0.45, hold=6, cap=0.40,
                    alpha_yr=alpha, seed=seed,
                )
                years = max(n_months / 12.0, 1 / 12.0)
                cagr_v = eq_final ** (1.0 / years) - 1.0
                cagrs.append(cagr_v * 100)
            cagrs = np.array(cagrs)
            rows.append({
                "k": k,
                "alpha_yr": alpha,
                "p10": float(np.percentile(cagrs, 10)),
                "median": float(np.median(cagrs)),
                "p90": float(np.percentile(cagrs, 90)),
                "mean": float(np.mean(cagrs)),
                "n_iters": N_ITERS,
            })
            elapsed = time.time() - t0
            print(f"  k={k}  α={alpha:>5.2f}  "
                  f"p10={np.percentile(cagrs, 10):>7.2f}%  "
                  f"med={np.median(cagrs):>7.2f}%  "
                  f"p90={np.percentile(cagrs, 90):>7.2f}%  "
                  f"mean={np.mean(cagrs):>7.2f}%   "
                  f"({elapsed:.0f}s)")

    df = pd.DataFrame(rows)
    df.to_csv(AUG / "v5_k2_vs_k3_bias.csv", index=False)
    df[df["k"] == 2].to_csv(AUG / "v5_k2_bias_sensitivity.csv", index=False)
    df[df["k"] == 3].to_csv(AUG / "v5_k3_bias_sensitivity.csv", index=False)

    # Side-by-side
    print(f"\n{'alpha_yr':>8} {'K=3 med':>10} {'K=2 med':>10} {'Δ (K2-K3)':>11}")
    for alpha in ALPHAS:
        k3 = df[(df.k == 3) & (df.alpha_yr == alpha)]["median"].iloc[0]
        k2 = df[(df.k == 2) & (df.alpha_yr == alpha)]["median"].iloc[0]
        print(f"{alpha:>8.2f} {k3:>9.2f}% {k2:>9.2f}% {k2-k3:>10.2f}pp")

    print(f"\nSaved -> {AUG / 'v5_k2_vs_k3_bias.csv'}")


if __name__ == "__main__":
    main()
