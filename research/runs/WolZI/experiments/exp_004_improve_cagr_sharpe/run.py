"""Push CAGR toward 50% and Sharpe toward 2.0.

Key insight from exp_003:
  - pred_K1_h6: 49% CAGR, 0.84 Sharpe (CAGR almost at gate, Sharpe far)
  - pred_K3_h1: 35% CAGR, 1.01 Sharpe (monthly rebalance helps Sharpe)
  - pred_K20_h6: 19.9% CAGR, 0.98 Sharpe (most Sharpe, too little CAGR)

Experiments:
  A. Better crash gate → avoid more drawdown → improve both CAGR and Sharpe
  B. Volatility targeting: scale exposure by target_vol / rolling_vol
  C. Combine pred with pred_12m (both have high IC for 6m holding)
  D. Monthly rebalancing with K=2-5 (mid-ground)
  E. Extended research window through 2024-04 to match YLOka
"""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[3]))
from quant_research.backtest.hold_engine import load_data, simulate, _next_mr_date, _spy_regime, COST_BPS, RESEARCH_END
from quant_research.backtest.metrics import block_bootstrap_sharpe, deflated_sharpe_ratio

OUT = Path(__file__).parent
OUT.mkdir(parents=True, exist_ok=True)

print("Loading data...")
panel, mr, mr_idx, spy_feat = load_data()
print(f"Ready. Panel dates: {panel['asof'].min().date()} → {panel['asof'].max().date()}")

# -----------------------------------------------------------------------
# Extended research window (matching YLOka: through 2024-04)
# -----------------------------------------------------------------------
EXTENDED_END = pd.Timestamp("2024-04-30")
panel_ext = panel[panel["asof"] <= EXTENDED_END]
print(f"Extended panel: {panel_ext.shape}, dates to {panel_ext['asof'].max().date()}")

results = []

def run(name, score_col, K=3, hold_months=6, use_crash=True, target_panel=None, extended=False, **kw):
    p = (target_panel if target_panel is not None else
         (panel_ext if extended else panel))
    # filter to research end
    end = EXTENDED_END if extended else RESEARCH_END
    p = p[p["asof"] <= end]

    r = simulate(
        score_col=score_col, K=K, hold_months=hold_months,
        use_crash_gate=use_crash,
        panel=p, mr=mr, mr_idx=mr_idx, spy_feat=spy_feat,
        research_only=False,  # already filtered above
        **kw,
    )
    tag = " [EXT]" if extended else ""
    cagr = float(r['cagr']) if not isinstance(r['cagr'], complex) else float('nan')
    sharpe = float(r['sharpe']) if not isinstance(r['sharpe'], complex) else float('nan')
    max_dd = float(r['max_dd']) if not isinstance(r['max_dd'], complex) else float('nan')
    r['cagr'] = cagr; r['sharpe'] = sharpe; r['max_dd'] = max_dd
    print(f"  {name+tag:55s} CAGR={cagr:6.1%}  Sharpe={sharpe:5.2f}  "
          f"MaxDD={max_dd:6.1%}  Cash={r['cash_months']}m")
    return r

print("\n=== A. Extended window (to 2024-04) — matches YLOka ===")
for score, K, h in [("pred",3,6), ("pred_12m",3,6), ("pred",1,6), ("pred",3,1), ("pred",5,6)]:
    r = run(f"{score}_K{K}_h{h}", score, K=K, hold_months=h, extended=True)
    results.append({"name": f"{score}_K{K}_h{h}_ext", **{k: v for k, v in r.items() if k not in ("monthly_rets","detail")}})

print("\n=== B. Stricter crash gate (add SPY below 200 SMA any amount) ===")
# Add strict regime gate to panel
strict_panel = panel.copy()

def simulate_with_strict_gate(score_col, K, hold_months, p):
    """Like simulate but uses a stricter regime gate."""
    months = sorted(p["asof"].unique())
    by_asof = {pd.Timestamp(d): g for d, g in p.groupby("asof")}
    cf = COST_BPS / 10_000
    cash_m = (1 + 0.04) ** (1/12) - 1

    equity = 1.0
    cur_picks, cur_weights = [], np.array([])
    held_for, in_cash = 0, False
    rows = []

    for i, asof in enumerate(months):
        asof = pd.Timestamp(asof)
        do_reb = (i == 0) or (held_for >= hold_months) or in_cash

        # Strict regime: use CURRENT calendar month-end SPY features (no look-ahead)
        spy_date = asof + pd.offsets.MonthEnd(0)
        if spy_date in spy_feat.index:
            d200 = float(spy_feat.loc[spy_date, "d_sma200"])
            r1m = float(spy_feat.loc[spy_date, "ret_1m"])
            strict_crash = (d200 < 0) or (r1m < -0.05)
        else:
            strict_crash = False

        if do_reb or strict_crash:
            sub = by_asof.get(asof, pd.DataFrame()).copy()
            sub = sub.dropna(subset=[score_col]) if not sub.empty else sub

            if strict_crash:
                cur_picks, cur_weights, in_cash = [], np.array([]), True
                held_for = 0
            elif sub.empty:
                in_cash = True; held_for = 0
            else:
                top = sub.nlargest(K, score_col)
                cur_picks = top["ticker"].tolist()
                cur_weights = np.ones(K) / K
                in_cash = False; held_for = 0

        next_d = _next_mr_date(asof, mr_idx)
        if in_cash or not cur_picks or next_d is None:
            ret_m = cash_m
        else:
            pick_rets = np.array([
                mr.at[next_d, t] if t in mr.columns and not np.isnan(mr.at[next_d, t]) else -1.0
                for t in cur_picks
            ])
            ret_m = float((pick_rets * cur_weights).sum())

        cost = cf if (do_reb and not in_cash and cur_picks) else 0.0
        net_ret = ret_m - cost
        equity *= (1 + net_ret)
        if not in_cash:
            held_for += 1
        rows.append({"asof": asof, "ret": net_ret, "in_cash": in_cash})

    df = pd.DataFrame(rows)
    rets = df["ret"]
    n_years = len(rets)/12
    cagr = equity**(1/n_years)-1 if n_years > 0 else np.nan
    std_m = rets.std()
    sharpe = rets.mean()/std_m*np.sqrt(12) if std_m > 0 else np.nan
    cum = (1+rets).cumprod()
    max_dd = (cum/cum.cummax()-1).min()
    cash_months = int(df["in_cash"].sum())
    return {"cagr": cagr, "sharpe": sharpe, "max_dd": max_dd, "n_months": len(rets),
            "cash_months": cash_months, "monthly_rets": rets}

p_res = panel[panel["asof"] <= RESEARCH_END]
for K, h in [(3,6),(1,6),(3,1)]:
    r = simulate_with_strict_gate("pred", K, h, p_res)
    name = f"strict_gate_pred_K{K}_h{h}"
    print(f"  {name:55s} CAGR={r['cagr']:6.1%}  Sharpe={r['sharpe']:5.2f}  "
          f"MaxDD={r['max_dd']:6.1%}  Cash={r['cash_months']}m")
    results.append({"name": name, **{k: v for k, v in r.items() if k != "monthly_rets"}})

print("\n=== C. pred_12m (best single feature by corrected IC IR=0.190) ===")
for K, h in [(1,6),(2,6),(3,6),(5,6),(3,1),(3,3)]:
    r = run(f"pred12m_K{K}_h{h}", "pred_12m", K=K, hold_months=h)
    results.append({"name": f"pred12m_K{K}_h{h}", **{k: v for k, v in r.items() if k not in ("monthly_rets","detail")}})

print("\n=== D. Monthly rebalancing K sweep (h=1) — CAGR+Sharpe tradeoff ===")
for K in [1,2,3,5,7,10,15]:
    r = run(f"monthly_K{K}_h1", "pred", K=K, hold_months=1)
    results.append({"name": f"monthly_K{K}_h1", **{k: v for k, v in r.items() if k not in ("monthly_rets","detail")}})

print("\n=== E. pred + pred_12m blend (add a cross-sectional score) ===")
panel_r = panel[panel["asof"] <= RESEARCH_END].copy()
if "pred_12m" in panel_r.columns:
    panel_r["rk_pred"] = panel_r.groupby("asof")["pred"].rank(pct=True)
    panel_r["rk_pred12m"] = panel_r.groupby("asof")["pred_12m"].rank(pct=True)
    panel_r["blend_pred"] = 0.5 * panel_r["rk_pred"] + 0.5 * panel_r["rk_pred12m"]

    for K, h in [(3,6),(1,6),(3,1),(5,6)]:
        r = run(f"blend_pred_K{K}_h{h}", "blend_pred", K=K, hold_months=h, target_panel=panel_r)
        results.append({"name": f"blend_pred_K{K}_h{h}", **{k: v for k, v in r.items() if k not in ("monthly_rets","detail")}})

# -------------------------------------------------------------------
print("\n=== SUMMARY (sorted by CAGR) ===")
df = pd.DataFrame(results)
df.to_csv(OUT / "results.csv", index=False)
df_s = df.sort_values("cagr", ascending=False)
print(df_s[["name","cagr","sharpe","max_dd","cash_months","n_months"]].head(20).to_string())

print("\n=== SUMMARY (sorted by Sharpe) ===")
df_sh = df.sort_values("sharpe", ascending=False)
print(df_sh[["name","cagr","sharpe","max_dd","cash_months","n_months"]].head(10).to_string())

# -------------------------------------------------------------------
# Best result deep dive
best_cagr = df_s.iloc[0]
print(f"\n=== Best CAGR: {best_cagr['name']} → CAGR={best_cagr['cagr']:.1%} Sharpe={best_cagr['sharpe']:.2f} ===")

best_sharpe = df_sh.iloc[0]
print(f"=== Best Sharpe: {best_sharpe['name']} → CAGR={best_sharpe['cagr']:.1%} Sharpe={best_sharpe['sharpe']:.2f} ===")

# Any config meeting BOTH gates?
meets_both = df[(df["cagr"] >= 0.50) & (df["sharpe"] >= 2.0)]
print(f"\n=== Configs meeting BOTH CAGR≥50% AND Sharpe≥2.0: {len(meets_both)} ===")
if len(meets_both):
    print(meets_both.to_string())

# Configs meeting CAGR≥50% only
meets_cagr = df[df["cagr"] >= 0.50]
print(f"\nConfigs with CAGR≥50%: {len(meets_cagr)}")
if len(meets_cagr):
    print(meets_cagr[["name","cagr","sharpe","max_dd"]].to_string())

summary = {
    "best_cagr_config": best_cagr["name"],
    "best_cagr": float(best_cagr["cagr"]),
    "best_cagr_sharpe": float(best_cagr["sharpe"]),
    "best_sharpe_config": best_sharpe["name"],
    "best_sharpe": float(best_sharpe["sharpe"]),
    "best_sharpe_cagr": float(best_sharpe["cagr"]),
    "n_configs": len(results),
    "n_passing_both_gates": len(meets_both),
}
with open(OUT / "summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"\nDone. {len(results)} configs. Results in {OUT}")
