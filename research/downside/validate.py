"""Mechanical audit, independent endpoint computation and exact fold reproduction."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from .features import load_archive,make_features
from .model import Config,METHODS,labels_for_horizon,fit_fold
from .policy import choose
from .run import dump


def audit(inputs,root,repeat=True):
    meta=json.loads((root/'metadata.json').read_text());kw=meta['config'].copy()
    for k in ('horizons','thresholds'):kw[k]=tuple(kw[k])
    cfg=Config(**kw);p,m,b,notes=load_archive(inputs,meta['inputs']['universe'])
    f=pd.read_parquet(root/'features.parquet');cols=meta['feature_columns'];a=p.to_numpy()
    checks={};n=0
    forecasts=[];labels=[]
    for h in cfg.horizons:
        y=pd.read_parquet(root/f'outcomes_{h}.parquet');r=pd.read_parquet(root/f'forecasts_{h}.parquet')
        i=f.i.to_numpy(int);j=p.columns.get_indexer(f.ticker);done=i+h<len(p)
        end=np.full(len(f),np.nan);end[done]=a[i[done]+h,j[done]]
        known=done & np.isfinite(end)&(end>0)
        expected=np.where(done,(known & (end<a[i,j])).astype(float),np.nan)
        np.testing.assert_allclose(y.down,expected,equal_nan=True)
        assert not (y.down.eq(1)&(~y.resolved|y.flat)).any()
        assert (y.loc[~y.matured,'class']==-1).all()
        assert (y.exit_i==f.i+h).all()
        if len(r):
            assert not r.duplicated(['row_id','horizon']).any()
            assert np.isfinite(r[list(METHODS)]).all().all()
            assert r[list(METHODS)].ge(0).all().all() and r[list(METHODS)].le(1).all().all()
            cl=r[[c for c in r.columns if c.startswith('class_')]]
            np.testing.assert_allclose(cl.sum(axis=1),1,atol=1e-12)
            np.testing.assert_allclose(r.consensus,r[['direct','competing','recent']].min(axis=1))
            assert (r.fit_i<=r.i).all() and (r.exit_i==r.i+h).all()
            assert (r.gate_rebound==((r.ma200<0)&(r.rel63<0)&(r.r21>0)&(r.lower_high63<0))).all()
            forecasts.append(r);n+=len(r)
        labels.append(y)
    checks['independent_price_endpoint_labels']=sum(len(y) for y in labels)
    checks['prediction_rows_checked']=n
    audits=json.loads((root/'fit_audit.json').read_text())
    for s in audits:
        if s['status']=='fitted':
            assert s['train_max_exit_i']<s['calibration_start_i']
            assert s['calibration_max_exit_i']<s['asof_i']
    checks['fitted_purged_folds']=sum(x['status']=='fitted' for x in audits)
    pred=pd.concat(forecasts,ignore_index=True) if forecasts else pd.DataFrame()
    picks=pd.read_csv(root/'picks.csv') if (root/'picks.csv').stat().st_size>1 else pd.DataFrame()
    bydate={i:g for i,g in pred.groupby('i')} if len(pred) else {}
    checked=0
    for name,rows in picks.groupby('policy') if len(picks) else []:
        busy={}
        for row in rows.sort_values('i').itertuples():
            method=row.method;t=row.threshold
            chosen=choose(bydate[row.i],method,t,busy)
            assert chosen is not None
            for k in ('row_id','i','ticker','horizon','exit_i','fit_i'):
                assert chosen[k]==getattr(row,k),(name,k)
            np.testing.assert_allclose(chosen['reference_price'],a[int(row.i),p.columns.get_loc(row.ticker)],rtol=1e-12)
            assert row.i>busy.get(row.ticker,-1)
            busy[row.ticker]=row.exit_i;checked+=1
    checks['selected_forecast_replays']=checked
    if repeat:
        h=cfg.horizons[0]; year=max(meta['years'][0],2024)
        y=pd.read_parquet(root/f'outcomes_{h}.parquet')
        asof=int(p.index.searchsorted(pd.Timestamp(year,1,1)));end=int(p.index.searchsorted(pd.Timestamp(year+1,1,1)))
        test=(f.i>=asof)&(f.i<end)
        new,s=fit_fold(f,y,cols,asof,h,cfg,test,shuffle=meta['null'])
        stored=pred.loc[(pred.horizon==h)&(pred.fit_i==asof)].reset_index(drop=True)
        pd.testing.assert_frame_equal(stored,new,check_exact=False,atol=1e-12,rtol=1e-12)
        poison=y.copy();future=poison.exit_i>=asof
        for k in ['down','up','persistent_down','squeeze_proxy']:poison.loc[future,k]=1
        poison.loc[future,'return']=-.999;poison.loc[future,'class']=1
        attacked,_=fit_fold(f,poison,cols,asof,h,cfg,test,shuffle=meta['null'])
        pd.testing.assert_frame_equal(new,attacked,check_exact=True)
        checks['exact_real_fold_reproduced']=len(new);checks['future_label_attack_unchanged']=True
        # Price/benchmark and membership corruption after a cutoff must not affect prior states.
        k=len(p)-100;pm=p.copy();mm=m.copy();bm=b.copy()
        pm.iloc[k+1:]*=19;bm.iloc[k+1:]*=.01;mm.iloc[k+1:]=False
        alter,_,_=make_features(pm,mm,bm)
        pd.testing.assert_frame_equal(f.loc[f.i<=k],alter.loc[alter.i<=k],check_exact=True)
        checks['future_price_benchmark_membership_attack_unchanged']=True
    checks['status']='passed';dump(root/'validation.json',checks)
    print(json.dumps(checks,indent=2),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--inputs',type=Path,required=True)
    ap.add_argument('--directory',type=Path,required=True);ap.add_argument('--skip-repeat',action='store_true')
    a=ap.parse_args();audit(a.inputs,a.directory,not a.skip_repeat)
