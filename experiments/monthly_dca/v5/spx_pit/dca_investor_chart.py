"""Investor-facing chart for the DCA evaluation: win-rate vs SPY-DCA by
horizon, and the v5 terminal-MOIC distribution (median / p05 / worst)."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

AUG = Path(__file__).resolve().parents[2] / "cache" / "v2" / "sp500_pit" / "augmented"
d = json.load(open(AUG / "dca_investor_eval.json"))

hk = ["H12", "H24", "H36", "H60", "H120"]
yrs = [1, 2, 3, 5, 10]
v5_win = [d["rolling"][k]["v5"]["win_rate_vs_spy"] * 100 for k in hk]
bl_win = [d["rolling"][k]["blend60_40"]["win_rate_vs_spy"] * 100 for k in hk]
med = [d["rolling"][k]["v5"]["median_moic"] for k in hk]
p05 = [d["rolling"][k]["v5"]["p05_moic"] for k in hk]
mn_ = [d["rolling"][k]["v5"]["min_moic"] for k in hk]
spy = [d["rolling"][k]["spy_median_moic"] for k in hk]

fig, ax = plt.subplots(1, 2, figsize=(13, 5))

ax[0].plot(yrs, v5_win, "o-", lw=2.5, label="v5 picker", color="#1a7f37")
ax[0].plot(yrs, bl_win, "s--", lw=2, label="60/40 v5+SPY", color="#0969da")
ax[0].axhline(100, color="grey", ls=":", lw=1)
for x, y in zip(yrs, v5_win):
    ax[0].annotate(f"{y:.0f}%", (x, y), textcoords="offset points",
                   xytext=(0, 8), ha="center", fontsize=9)
ax[0].set_title("Rolling monthly-DCA win rate vs DCA-into-SPY\n(PIT data, 2003-2026, no tuning)")
ax[0].set_xlabel("DCA horizon (years)")
ax[0].set_ylabel("% of rolling windows that beat SPY-DCA")
ax[0].set_ylim(40, 105)
ax[0].set_xticks(yrs)
ax[0].legend(loc="lower right")
ax[0].grid(alpha=0.3)

ax[1].fill_between(yrs, p05, med, alpha=0.18, color="#1a7f37", label="p05–median band")
ax[1].plot(yrs, med, "o-", lw=2.5, color="#1a7f37", label="v5 median MOIC")
ax[1].plot(yrs, mn_, "v--", lw=1.8, color="#cf222e", label="v5 WORST window MOIC")
ax[1].plot(yrs, spy, "s-", lw=1.8, color="#57606a", label="SPY-DCA median MOIC")
ax[1].axhline(1.0, color="grey", ls=":", lw=1)
for x, y in zip(yrs, med):
    ax[1].annotate(f"{y:.1f}x", (x, y), textcoords="offset points",
                   xytext=(0, 8), ha="center", fontsize=9)
ax[1].set_title("What $1/mo turns into (× total contributed)\nGreen = picker, grey = SPY, red = worst case ever")
ax[1].set_xlabel("DCA horizon (years)")
ax[1].set_ylabel("Terminal value ÷ money contributed")
ax[1].set_xticks(yrs)
ax[1].legend(loc="upper left")
ax[1].grid(alpha=0.3)

fig.tight_layout()
out = AUG / "dca_investor_eval.png"
fig.savefig(out, dpi=130)
print("wrote", out)
