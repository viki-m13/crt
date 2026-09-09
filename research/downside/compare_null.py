"""Like-for-like recent real/null comparison with both position locks reset in 2024.

This replays already-fitted real forecasts; it neither retrains nor tunes on results.
Only a single fixed shuffled-training realization exists, not a permutation p-value.
"""
import argparse,json
from pathlib import Path
import pandas as pd
from .model import Config,METHODS
from .policy import replay,controls,metrics
from .run import dump


def recent_replay(full, null, output, first_year=2024):
    meta=json.loads((full/'metadata.json').read_text());kw=meta['config'].copy()
    for k in ('horizons','thresholds'):kw[k]=tuple(kw[k])
    cfg=Config(**kw);f=pd.read_parquet(full/'features.parquet')
    first=int(f.i.loc[pd.to_datetime(f.date).dt.year>=first_year].min())
    keep=['row_id','i','date','ticker','horizon','exit_i','fit_i','reference_price',
          'vol63','vol63_rank','rel63_rank','regime',*METHODS,'gate_rebound','gate_squeeze_veto']
    forecasts=[];outcomes=[]
    for h in cfg.horizons:
        r=pd.read_parquet(full/f'forecasts_{h}.parquet')
        if len(r):forecasts.append(r.loc[r.i>=first,keep])
        y=pd.read_parquet(full/f'outcomes_{h}.parquet');outcomes.append(y.loc[y.i>=first])
    pred=pd.concat(forecasts,ignore_index=True);y=pd.concat(outcomes,ignore_index=True)
    picks,ds=replay(pred,y,sorted(f.i.loc[f.i>=first].unique()),cfg)
    picks=controls(picks,y,f);summary=metrics(picks,ds,meta['cutoff_i'])
    output.mkdir(parents=True,exist_ok=True)
    picks.to_csv(output/'picks.csv',index=False);ds.to_csv(output/'decisions.csv',index=False)
    dump(output/'summary.json',summary)
    nulls={r['policy']:r for r in json.loads((null/'summary.json').read_text())}
    comparison=[]
    for r in summary:
        n=nulls[r['policy']]
        comparison.append({'policy':r['policy'],'real_matured':r['matured'],'real_wins':r['wins'],
            'real_precision':r['precision'],'real_random':r.get('random_expected'),
            'null_matured':n['matured'],'null_wins':n['wins'],'null_precision':n['precision'],
            'null_random':n.get('random_expected')})
    dump(output/'comparison.json',comparison);pd.DataFrame(comparison).to_csv(output/'comparison.csv',index=False)
    dump(output/'comparison_design.json',{'first_year':first_year,'universe':meta['inputs']['universe'],
        'both_ticker_locks_reset':True,'outcome_retraining_or_tuning':False,
        'fixed_null_realizations':1,'independent_significance_test':False})
    print('Recent comparison complete:',output,flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--full',type=Path,required=True)
    ap.add_argument('--null',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args();recent_replay(a.full,a.null,a.output)
