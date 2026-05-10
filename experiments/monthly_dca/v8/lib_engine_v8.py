"""v8 engine: thin wrapper that injects v8 ml_preds into the v6 engine.

Allows running the same v6 simulator with a different prediction parquet
(ml_preds_v8 instead of ml_preds_v2).
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

import experiments.monthly_dca.v6.lib_engine as ve

V8 = Path('experiments/monthly_dca/v8')
PIT = Path('experiments/monthly_dca/cache/v2/sp500_pit')
FEATURES_DIR = Path('experiments/monthly_dca/cache/features')

EXCLUDE_TICKERS = ve.EXCLUDE_TICKERS


def load_score_panel_v8(scorer: str = 'ml_3plus6', universe: str = 'sp500_pit',
                        attach_pullback: bool = False) -> pd.DataFrame:
    """Load v8 ml_preds and assemble score panel matching v6 schema."""
    ml = pd.read_parquet(V8 / 'ml_preds_v8.parquet')
    ml['asof'] = pd.to_datetime(ml['asof'])
    if scorer == 'ml_3plus6':
        ml['score'] = (ml['pred_3m'] + ml['pred_6m']) / 2
    elif scorer == 'ml_filter':
        ml['score'] = ml['pred']
    elif scorer == 'ml_h6':
        ml['score'] = ml['pred_6m']
    elif scorer == 'ml_h3':
        ml['score'] = ml['pred_3m']
    elif scorer == 'ml_h12':
        ml['score'] = ml['pred_12m']
    else:
        raise ValueError(scorer)

    if universe == 'sp500_pit':
        mem = pd.read_parquet(PIT / 'sp500_membership_monthly.parquet')
        mem['asof'] = pd.to_datetime(mem['asof'])
        ml = ml.merge(mem, on=['asof', 'ticker'], how='inner')
    elif universe == 'broader':
        pass
    elif universe == 'non_sp500':
        mem = pd.read_parquet(PIT / 'sp500_membership_monthly.parquet')
        mem['asof'] = pd.to_datetime(mem['asof'])
        mem['in_sp500'] = True
        ml = ml.merge(mem, on=['asof', 'ticker'], how='left')
        ml = ml[ml['in_sp500'].isna()].drop(columns=['in_sp500'])
    else:
        raise ValueError(universe)

    ml = ml[~ml['ticker'].isin(EXCLUDE_TICKERS)]
    ml = ml.dropna(subset=['score'])

    asofs = sorted(ml['asof'].unique())
    vol_rows = []
    for asof in asofs:
        f = FEATURES_DIR / f'{pd.Timestamp(asof).date()}.parquet'
        if not f.exists():
            continue
        df = pd.read_parquet(f)
        if 'vol_1y' not in df.columns:
            continue
        vr = df['vol_1y'].rename('vol_1y').to_frame()
        vr['asof'] = pd.Timestamp(asof)
        vr['ticker'] = vr.index
        vol_rows.append(vr.reset_index(drop=True))
    if vol_rows:
        vols = pd.concat(vol_rows, ignore_index=True)
        ml = ml.merge(vols, on=['asof', 'ticker'], how='left')
    else:
        ml['vol_1y'] = np.nan

    extras = []
    if attach_pullback:
        feat_rows = []
        for asof in asofs:
            f = FEATURES_DIR / f'{pd.Timestamp(asof).date()}.parquet'
            if not f.exists():
                continue
            df = pd.read_parquet(f)
            cols = [c for c in ('pullback_1y', 'mom_12_1', 'trend_health_5y') if c in df.columns]
            if not cols:
                continue
            pr = df[cols].copy()
            pr['asof'] = pd.Timestamp(asof)
            pr['ticker'] = pr.index
            feat_rows.append(pr.reset_index(drop=True))
        if feat_rows:
            extras_df = pd.concat(feat_rows, ignore_index=True)
            ml = ml.merge(extras_df, on=['asof', 'ticker'], how='left')
        extras = ['pullback_1y', 'mom_12_1', 'trend_health_5y']

    if 'vol_1y' in ml.columns:
        ml['vol_rank'] = ml.groupby('asof')['vol_1y'].rank(pct=True)
        extras.append('vol_rank')
    return ml[['asof', 'ticker', 'score', 'vol_1y'] + extras]


def load_score_panel_blend(weight_v8: float = 0.5, scorer: str = 'ml_3plus6',
                           universe: str = 'sp500_pit') -> pd.DataFrame:
    """Blend v2 and v8 scores at given weight."""
    sp_v2 = ve.load_score_panel(scorer, universe, attach_pullback=False)
    sp_v8 = load_score_panel_v8(scorer, universe, attach_pullback=False)
    sp_v2 = sp_v2.rename(columns={'score': 'score_v2'})
    sp_v8 = sp_v8.rename(columns={'score': 'score_v8'})
    merged = sp_v2.merge(sp_v8[['asof', 'ticker', 'score_v8']],
                         on=['asof', 'ticker'], how='inner')
    merged['score'] = (1 - weight_v8) * merged['score_v2'] + weight_v8 * merged['score_v8']
    cols = ['asof', 'ticker', 'score', 'vol_1y']
    if 'vol_rank' in merged.columns:
        cols.append('vol_rank')
    return merged[cols]
