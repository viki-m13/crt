# Current Focus

**Status**: PAUSED — waiting for V input on goal framing

## What we know (Session 1 summary)

After exhaustive experimentation (116 new hypotheses this session, 224 total):

The target (Sharpe ≥ 2.0 AND CAGR ≥ 50%) is structurally blocked by the fundamental
law of active management with price-only monthly long-only equities:
- Signal IC ≈ 0.025 (real but modest)
- Oracle bound (K=3): CAGR=1632%, Sharpe=6.32 — so there IS alpha in the data
- Our strategy captures ~15% of oracle alpha, consistent with IC=0.025
- The crash gate adds extra alpha beyond the fundamental law bound
- Maximum Sharpe seen: 1.12 (at CAGR=13%)
- Maximum CAGR: 48.73% (Sharpe=1.00)

## Session 1 experiments run (all this invocation)

1. H6 Sharpe Push (59 experiments): quality filters, invvol weighting, K sweeps, hold sweeps, vol targeting
2. H6b Donchian Quality (37 experiments): risk-adjusted scoring, Donchian+quality, LightGBM ranker (failed), LightGBM Sharpe-label
3. H6c LightGBM Fixed (20 experiments): LightGBM with correct quintile labels (IC=-0.009, worse than baseline), enhanced crash gate, LightGBM+enhanced gate

## Key findings

- Quality filter: cuts CAGR proportionally more than Sharpe improvement
- Invvol weighting: marginal Sharpe improvement (+0.05-0.10 vs baseline)
- Vol targeting: does NOT improve Sharpe (by construction, scales both mean and std equally)
- LightGBM: cannot beat existing pred_3m/pred_6m (uses same features, less data)
- Enhanced crash gate (breadth): over-filters, hurts CAGR without Sharpe gain
- Risk-adjusted scoring (pred/vol): best Sharpe 1.12 but CAGR only 12.7%

## If continuing, next experiments would be

Priority 1: Regime-conditional quality (recovery → quality scoring, bull/normal → baseline)
Priority 2: MVO portfolio optimization with quality filter
Priority 3: If fundamentals available, add quality screens (gross profit, ROE)
Priority 4: Comprehensive ablation of the best-so-far config (Donchian K=3 h=6)

## Time budget used this session

~45 min execution + 15 min journaling/setup = ~60 min total
