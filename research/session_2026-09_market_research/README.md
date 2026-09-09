# Market research session, 2026-09-04 → 09
**48 pre-registered hypotheses. 2 with measured edge. 46 killed, most by their own controls.**

Every hypothesis was frozen in writing (`prereg/`) *before* its data was computed, with kill
criteria stated in advance. Where a result died, the pre-registration records why. Where I made a
reasoning error, that is recorded too rather than quietly dropped.

---

## The two things with measured edge

### 1. Leveraged-ETF variance harvest — `prereg/PREREG17.md`, `leveraged_etf/`
Short $1 of a 3× ETF, long $L of the underlying. Beta cancels **algebraically**; what remains is
realized variance: `-(L·R - L(L-1)/2·σ²T) + L·R = L(L-1)/2·σ²T`.

Verified across 7 pairs, 2006–2023 — measured regression slopes **2.94 / 2.97 / 3.11 / 3.17 / 3.66**
against a theory of 3.0, and 1.21 / 1.31 against 1.0. Intercepts all positive (+1.5% to +8.2%/yr):
the ETFs' expense ratio and leverage financing, also captured by the short.

| window | ann (per $1 short notional) | Sharpe | maxDD |
|---|---:|---:|---:|
| TRAIN 2010–19 | +2.45% | 0.64 | −3.4% |
| VALID 2020–23 | +5.95% | 2.19 | −1.5% |
| **TEST 2024–26 (sealed until last)** | **+11.79%** | **+3.35** | −4.3% |

All 7 pairs positive on TEST. Measured rebalance turnover 1.36%/day → trading cost **0.09%/yr**.
Worst month −1.28%, worst quarter −1.78%, 0.34% of days lose >1%.

**Open question, unresolved:** break-even borrow is 2.45%/yr (TRAIN) to 11.79%/yr (TEST). I have no
borrow-rate data. Below ~3%/yr it is clearly viable; above ~8% it dies in the VALID regime. This is
a broker quote, not a research question. Return on Reg-T capital is 0.9–4.4%; the Sharpe is
excellent, the unlevered return is small.
**A "clever" addition failed:** scaling by a HAR-RV variance forecast *reduced* Sharpe 2.19 → 1.14.
The identity does the work; the forecast does not.

### 2. Peer Breadth on Hyperliquid — `peer_breadth_hl/`
Third-party idea (`source_material/`), recreated from spec with its own audit's blockers fixed:
one open position per coin, pro-rata allocation at **open-time** equity (removes the look-ahead),
entry cost debited immediately, target live from the entry bar, cost reserve in the risk denominator.

| variant | CAGR | Sharpe | maxDD |
|---|---:|---:|---:|
| published 17 coins | +6.5% | 1.40 | −3.8% |
| **ex-ante oldest 17 (2021 knowledge)** | +5.5% | **1.18** | −5.6% |
| all 51 HL coins | +1.1% | 0.41 | −3.7% |

The ex-ante universe overlaps the published one by only 12 of 17 and still works — **the edge is not
coin-picking hindsight**, which is the test the ETF-rotation idea failed.
Correlation with StormGlide-v2 (live on HL): **+0.085**, stable at +0.015 / +0.092 / +0.123 across
sub-periods. A 50/50 blend lifts Sharpe **1.67 → 2.04** and cuts maxDD **−45% → −28%**.
Sharpe is flat across the risk-scaling frontier (1.39–1.47), so it is under-deployed, not under-edged.

**Caveats:** my 2025+ numbers are well below the author's (+6.4% / PF 1.31 vs +16.6% / PF 2.12);
clear decay (full-period Sharpe 1.18–1.40 → 2025+ 0.61–0.76); I control for listing but **not
delisting** — my HL set is all survivors. No forward evidence for either version.

---

## Findings that generalize (the more durable output)

1. **A large, replicating, de-overlapped IC is not evidence of tradable edge.** Wall-clock reversion
   IC −0.23 (t −24) → gross **+0.07 bp/day**. Volume-clock IC t +15.5 → edge/cost **0.17**. Harness
   verified against perfect foresight (+299.6 bp/day). This retracted an earlier claim of mine.
2. **Maker price improvement and adverse selection cancel one-for-one.** Post at +δ and the
   fill-conditional move is ≈ −1 bp for δ = 5, 10, 20 bp — measured against real high/low paths, not
   an assumed fill rate. ASE = 1.005 on Binance majors. Explains the account's earlier pt-7/pt-8
   maker failure mechanistically.
3. **Effective breadth ≈ 2×, not 20×.** Widening a cross-section 20 → 400 names raised t by 1.43×,
   not √20 = 4.5×. Daily equity returns are common-factor dominated; widening does not rescue a
   thin result. This refuted my own reason for pivoting to a wide universe.
4. **Power is the binding constraint.** t = Sharpe·√years, bar t>3: confirming Sharpe 1.2 needs 6.2
   years, Sharpe 0.6 needs 25. With 8 years, anything under ~1.2 is *undecidable* — it looks
   identical to the 46 nulls. Short horizons have power but no economics; long horizons the reverse.
5. **Markets are time-irreversible** (48.3% excess over phase-randomized surrogates, day-clustered
   t +80.7) — invisible to any spectral method, since |FFT|² discards phase. But it is a *symbol*
   property (per-day SNR 0.59), and it is **anti-correlated with exploitability** (ρ = −0.394):
   entropy production measures information *arrival*, which is priced by definition.
6. **Shared-price artifacts dominate naive tests.** Swapping which price a signal and target share
   flipped a result from +7.65 bp to −190 bp — a 25× contamination.

## Ideas killed, with the control that killed each
`prereg/` holds all 17 pre-registrations. Headlines:
- **Spectral timbre** — white-noise null matched to 4 decimals.
- **Cross-spectral lead-lag** — the follower's *own* oscillator beat the leader's (t +6 to +13.6).
- **Resonance tuning** — a *randomly* chosen passband beat the tuned one (t −37).
- **Thermodynamic alpha ceiling (TUR)** — the inequality held (0/20 violations, ~8× headroom) but
  ranked assets **backwards** (ρ −0.394). Physics fine; economics inverted.
- **Dissipation per unit order flow** — added nothing beyond Kyle's λ; and the base-rate trade it
  conditioned did not exist (5-min reversal ≈ 0 gross). *Check the base rate before building a
  conditioner for it.*
- **ASE inversion (breakout)** — worse than random entries. My mechanism reasoning was wrong: the
  maker's loss is not the taker's gain, because both sides pay their own frictions.
- **Weekly reversal / 12-1 momentum** — sign flips at the 2022 regime break; cost was *not* binding.
- **ETF momentum rotation** — +20.3% CAGR collapsed to +7.85% (vs SPY +8.35%) once nine
  hindsight-picked mega-caps were removed.
- **Unlevered long-only vs SPY** — 200d-MA, vol-target and dual-momentum all buy drawdown reduction
  with CAGR. None beats SPY's +10.89%/31.8x over 33 years.

## Analog pattern-matching — `analog/`, and why the tool says what it says
Live tool: https://claude.ai/code/artifact/9d4f3c4d-a602-4738-9e9d-c7310d191c2b
Requested at "99% accuracy". Measured over **121,287** out-of-sample projections:

| | analogs | benchmark |
|---|---:|---:|
| directional hit rate | 55.5% | **65.3%** (always say "up") |
| corr(projected, actual) | **−0.006** | — |
| "80%" band real coverage | **46.6%** | **79.5%** (trailing vol + drift) |

When 14+ of 20 analogs agreed, the hit rate was 64.5% and the unconditional up-rate was 65.4% —
agreement carried nothing. Two further attempts to raise it also lost to the base rate: a parameter
sweep (−7.2pp excess) and a drift+vol probability model (57.7% vs 64.9%, Brier skill −12.3%, flat
calibration across the whole predicted range). Direction is not forecastable here.
The tool therefore ships the **volatility-calibrated** band (79.5% measured coverage) as its risk
estimate and labels the analog spread as understating risk ~2×.

## Reproducing
Scripts expect the research-data caches under `/tmp/databranch/data/` (Binance perps, HL daily,
`daily_multiasset`, `equity_daily_broad`) pulled from the `vol` repo's `claude/research-data`
branch. Nothing in `vol` was modified — it was read through `git show` only.
PIT universe data used here is consolidated in `data/pit_archive/`.
