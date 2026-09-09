"""Real-data reproduction, locked-decision replay, and future-label attack."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from research.failure_first.data import load_archive
from research.failure_first.evaluate import clean_json
from .engine import Config,METHODS,annual_predictions,candidate_rows
from .evaluate import choose


def validate(inputs,root,cache=None,repeat=True):
    cache=cache or root;cfg=Config();meta=json.loads((root/'metadata.json').read_text())
    p,m,bm,notes=load_archive(inputs,meta['inputs']['universe'])
    f=pd.read_parquet(cache/'features.parquet')
    cols=json.loads((cache/'data_metadata.json').read_text())['feature_columns']
    ys={h:pd.read_parquet(cache/f'labels_{h}.parquet') for h in cfg.horizons}
    chunks=[pd.read_parquet(root/f'predictions_{year}.parquet') for year in meta['years']]
    pred=pd.concat(chunks,ignore_index=True)
    expected_ids=set(candidate_rows(f).row_id)
    assert set(pred.row_id)<=expected_ids
    assert not pred.duplicated(['row_id','horizon']).any()
    assert (pred.horizon>=30).all() and (pred.exit_i==pred.i+pred.horizon).all() and (pred.fit_i<=pred.i).all()
    price=p.to_numpy();checked=0
    for h,g in pred.groupby('horizon'):
        y=ys[h].set_index('row_id').loc[g.row_id]
        i=g.i.to_numpy(int);j=p.columns.get_indexer(g.ticker);end=i+h
        mature=end<len(p);ret=np.full(len(g),np.nan);ret[mature]=price[end[mature],j[mature]]/price[i[mature],j[mature]]-1
        down=np.where(mature,np.isfinite(ret)&(ret<0),np.nan)
        np.testing.assert_array_equal(y.success.to_numpy(),down)
        np.testing.assert_allclose(y['return'].to_numpy(),ret,equal_nan=True)
        checked+=len(g)
    audits=json.loads((root/'fit_audit.json').read_text())
    for a in audits:
        assert a['reference_max_i']<a['fit_i']-30
        if a['status']=='fitted':
            assert a['train_max_exit_i']<a['calibration_start_i']
            assert a['calibration_max_exit_i']<a['fit_i']
    picks=pd.read_csv(root/'picks.csv');dec=pd.read_csv(root/'decisions.csv')
    groups={int(i):g for i,g in pred.groupby('i')};by_policy={s:g for s,g in picks.groupby('policy')}
    replayed=0
    for policy,ds in dec.groupby('policy'):
        if policy=='adaptive95':continue # Synthetic tests target this screen's future filtering.
        method,threshold=policy.split('@');busy={}
        chosen=by_policy.get(policy,pd.DataFrame());by_date={int(r.i):r for r in chosen.itertuples()} if len(chosen) else {}
        for d in ds.sort_values('i').itertuples():
            r=choose(groups.get(int(d.i),pd.DataFrame()),method,float(threshold),busy)
            assert (r is not None)==bool(d.issued)
            if r is not None:
                old=by_date[int(d.i)]
                assert (r['ticker'],r['horizon'],r['exit_i'])==(old.ticker,old.horizon,old.exit_i)
                busy[r['ticker']]=r['exit_i']
            replayed+=1
    report={'input_hash_verification':True,'endpoint_rows_checked':checked,
            'full_decision_replay_count':replayed,'candidate_and_purge_checks':True,'repeated_rows':0,
            'not_a_production_certificate':True,'code_hashes':{x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in Path(__file__).parent.glob('*.py')}}
    if repeat and not meta['within_date_null']:
        year=2024;asof=int(p.index.searchsorted(pd.Timestamp(year,1,1)))
        until=int(p.index.searchsorted(pd.Timestamp(year+1,1,1)))
        subset={h:ys[h] for h in (30,252)}
        again,_=annual_predictions(f,subset,cols,asof,until,cfg)
        prior=pred[(pred.fit_i==asof)&pred.horizon.isin(subset)].sort_values(['row_id','horizon']).reset_index(drop=True)
        again=again.sort_values(['row_id','horizon']).reset_index(drop=True)
        pd.testing.assert_frame_equal(prior,again,check_exact=True)
        corrupted={h:y.copy() for h,y in subset.items()}
        for h,y in corrupted.items():
            future=y.exit_i>=asof
            y.loc[future,'success']=np.arange(int(future.sum()))%2
        attack,_=annual_predictions(f,corrupted,cols,asof,until,cfg)
        attack=attack.sort_values(['row_id','horizon']).reset_index(drop=True)
        pd.testing.assert_frame_equal(prior,attack,check_exact=True)
        report.update(repeated_rows=len(prior),future_label_attack_identical=True,repeat_year=year,repeat_horizons=[30,252])
    return report


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--inputs',type=Path,required=True)
    ap.add_argument('--directory',type=Path,required=True);ap.add_argument('--cache',type=Path)
    ap.add_argument('--no-repeat',action='store_true');a=ap.parse_args()
    report=validate(a.inputs,a.directory,a.cache,not a.no_repeat)
    (a.directory/'validation.json').write_text(json.dumps(clean_json(report),indent=2,allow_nan=False))
    print(json.dumps(clean_json(report),indent=2))

if __name__=='__main__':main()
