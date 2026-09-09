"""Mechanical audits plus exact repeated real-data folds and future-label attacks."""
from __future__ import annotations
import argparse,json,hashlib,time
from pathlib import Path
import numpy as np
import pandas as pd
from .archive import load_archive
from .model import Config,METHODS,fit_fold
from .evaluate import clean_json


def validate_directory(root,inputs,rerun=True):
    start=time.time();meta=json.loads((root/'metadata.json').read_text())
    cfg=Config();universe=meta['inputs']['universe'];checks={}
    # Re-reading the loader validates all nine byte hashes and NYSE alignment.
    p,m,b,notes=load_archive(inputs,universe)
    checks['input_hashes_verified']=len(notes['source_files'])
    checks['calendar_monotone_unique']=bool(p.index.is_monotonic_increasing and not p.index.has_duplicates)
    assert checks['calendar_monotone_unique']
    f=pd.read_parquet(root/'features.parquet');cols=meta['feature_columns']
    assert not f.duplicated(['i','ticker']).any()
    assert not any(any(s in c for s in ['forward','future','label','outcome','failure']) for c in cols)
    checks['feature_rows_unique']=True;checks['feature_label_names_separate']=True
    audits=json.loads((root/'fit_audit.json').read_text())
    fits=[a for a in audits if a['status']=='fitted']
    assert all(a['train_max_exit_i']<a['calibration_start_i'] for a in fits)
    assert all(a['calibration_max_exit_i']<a['asof_i'] for a in fits)
    checks['purged_model_fits']=len(fits)
    checks['insufficient_history_folds']=len(audits)-len(fits)
    total=0;pending=0;classes={str(h):{} for h in cfg.horizons};rlist=[]
    for h in cfg.horizons:
        y=pd.read_parquet(root/f'outcomes_{h}.parquet');r=pd.read_parquet(root/f'risks_{h}.parquet')
        d=y.loc[y.matured]
        assert np.array_equal((d.failure_class==0).to_numpy(),(d.success==1).to_numpy())
        assert y.loc[~y.matured,'success'].isna().all()
        assert (y.loc[~y.matured,'failure_class']==-1).all()
        assert (d.loc[~d.resolved,'failure_class']==1).all()
        classes[str(h)]={str(int(k)):int(v) for k,v in d.failure_class.value_counts().items()}
        if not r.empty:
            assert np.isfinite(r[list(METHODS)].to_numpy()).all()
            assert (r[list(METHODS)].to_numpy()>=0).all() and (r[list(METHODS)].to_numpy()<=1).all()
            assert (r.failure_veto>=r.binary).all() and (r.failure_veto>=r.failure_sum).all()
            assert (r.failure_veto_path>=r.failure_veto).all()
            assert (r.raw_sum+1e-12>=r.naive_channels).all()
            assert (r.fit_i<=r.i).all() and (r.exit_i==r.i+h).all()
            rlist.append(r[['row_id','i','ticker','horizon',*METHODS]])
        total+=len(r);pending+=int((~y.matured).sum())
    checks['scored_stock_horizon_rows']=total;checks['pending_labels_not_losses']=pending
    checks['exhaustive_failure_classes']=classes
    picks=pd.read_csv(root/'picks.csv');summary=json.loads((root/'summary.json').read_text())
    decisions=pd.read_csv(root/'decisions.csv');risk=pd.concat(rlist,ignore_index=True)
    by_stock=risk.sort_values(['i','ticker','horizon']).set_index(['i','ticker'],drop=False)
    for policy,g in picks.groupby('policy'):
        assert not g.i.duplicated().any()
        for ticker,seq in g.groupby('ticker'):
            seq=seq.sort_values('i')
            if len(seq)>1: assert (seq.i.to_numpy()[1:]>seq.exit_i.to_numpy()[:-1]).all()
        for row in g.itertuples():
            assert row.estimated_success>row.threshold-1e-12
            pool=by_stock.loc[[(row.i,row.ticker)]]
            eligible=pool.loc[pool[row.method]<1-row.threshold]
            assert row.horizon==eligible.horizon.min()
    for s in summary:
        g=picks.loc[picks.policy==s['policy']]
        d=g.loc[g.matured]
        assert len(g)==s['issued'] and len(d)==s['matured'] and int(d.success.sum())==s['wins']
        assert s['precision'] is None if len(d)==0 else abs(s['precision']-d.success.mean())<1e-12
    checks['joint_policy_locks_shortest_horizon_and_no_quota']=True
    checks['summary_recomputed_from_selected_forecasts']=True
    del by_stock,risk,rlist
    if rerun:
        # Four real folds: both a short and a longer horizon, late and earlier regime.
        chosen=[(30,2024),(180,2021)] if universe=='sp500' else [(30,2024),(126,2023)]
        tests=[]
        for h,year in chosen:
            asof=int(p.index.searchsorted(pd.Timestamp(year,1,1)))
            stop=int(p.index.searchsorted(pd.Timestamp(year+1,1,1)))
            mask=(f.i.to_numpy()>=asof)&(f.i.to_numpy()<stop)
            y=pd.read_parquet(root/f'outcomes_{h}.parquet')
            pred,a=fit_fold(f,y,cols,asof,h,cfg,mask)
            original=pd.read_parquet(root/f'risks_{h}.parquet')
            original=original.loc[original.fit_i==asof].reset_index(drop=True)
            pd.testing.assert_frame_equal(pred,original,check_exact=True)
            mutated=y.copy();future=mutated.exit_i>=asof
            mutated.loc[future,'failure_class']=2;mutated.loc[future,'path_failure']=1
            attacked,_=fit_fold(f,mutated,cols,asof,h,cfg,mask)
            pd.testing.assert_frame_equal(pred,attacked,check_exact=True)
            tests.append({'horizon':h,'year':year,'rows_identical':len(pred),
                          'exact_rerun':True,'future_label_attack_invariant':True})
        checks['real_fold_reruns_and_future_label_attacks']=tests
    checks['elapsed_seconds']=round(time.time()-start,3)
    checks['data_quality_status']='research_only_unverified_adjustments_and_terminal_actions'
    return checks


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--directory',type=Path,required=True)
    ap.add_argument('--inputs',type=Path,required=True);ap.add_argument('--skip-rerun',action='store_true')
    a=ap.parse_args();result=validate_directory(a.directory,a.inputs,not a.skip_rerun)
    (a.directory/'validation.json').write_text(json.dumps(clean_json(result),indent=2,allow_nan=False))
    print(json.dumps(clean_json(result),indent=2))

if __name__=='__main__': main()
