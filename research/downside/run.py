"""Reproducible direct-downside experiments. All forecasts remain research-only."""
from __future__ import annotations
import argparse,hashlib,json,platform,time
from pathlib import Path
from dataclasses import asdict
import numpy as np
import pandas as pd
from .features import load_archive,make_features
from .model import Config,labels_for_horizon,fit_fold
from .policy import replay,controls,metrics,calibration,clean_json


def dump(path,obj):
    path.write_text(json.dumps(clean_json(obj),indent=2,allow_nan=False)+'\n')


def evaluate(root):
    meta=json.loads((root/'metadata.json').read_text());kw=meta['config']
    for k in ('horizons','thresholds'):kw[k]=tuple(kw[k])
    cfg=Config(**kw);f=pd.read_parquet(root/'features.parquet')
    preds=[];outcomes=[]
    for h in cfg.horizons:
        r=pd.read_parquet(root/f'forecasts_{h}.parquet');y=pd.read_parquet(root/f'outcomes_{h}.parquet')
        if len(r):preds.append(r)
        outcomes.append(y)
    r=pd.concat(preds,ignore_index=True) if preds else pd.DataFrame(columns=['row_id','horizon','i','exit_i'])
    y=pd.concat(outcomes,ignore_index=True)
    first=f.i.loc[pd.to_datetime(f.date).dt.year>=meta['years'][0]].min()
    y=y.loc[y.i>=first];dates=sorted(f.i.loc[(f.i>=first)&pd.to_datetime(f.date).dt.year.isin(meta['years'])].unique())
    picks,decisions=replay(r,y,dates,cfg)
    picks=controls(picks,y,f)
    picks.to_csv(root/'picks.csv',index=False);decisions.to_csv(root/'decisions.csv',index=False)
    summary=metrics(picks,decisions,meta['cutoff_i']);dump(root/'summary.json',summary)
    pd.DataFrame([{k:v for k,v in x.items() if not isinstance(v,(dict,list))} for x in summary]).to_csv(root/'summary.csv',index=False)
    dump(root/'calibration.json',calibration(r,y) if len(r) else [])
    meta['outer_predictions']=len(r);meta['all_policy_issued_rows']=len(picks)
    dump(root/'metadata.json',meta)
    print('EVALUATED',meta['inputs']['universe'],'policies',len(summary),'picks all policies',len(picks),flush=True)


def validate_resume(meta,cfg,universe,years,null):
    canonical=lambda x:json.dumps(x,sort_keys=True)
    if canonical(meta['config'])!=canonical(asdict(cfg)) or meta['years']!=years or meta['null']!=null:
        raise ValueError('Resume configuration differs from original experiment')
    if meta['inputs']['universe']!=universe:
        raise ValueError('Resume universe differs')
    for name in ('model.py','features.py','policy.py'):
        current=hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        if meta['source_hashes'].get(name)!=current:
            raise ValueError('Resume numerical source changed: '+name)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--inputs',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--universe',choices=['sp500','ndx'],required=True)
    ap.add_argument('--null',action='store_true');ap.add_argument('--years',type=int,nargs='+')
    ap.add_argument('--horizons',type=int,nargs='+');ap.add_argument('--evaluate-only',action='store_true')
    ap.add_argument('--resume',action='store_true')
    a=ap.parse_args();out=a.output;out.mkdir(parents=True,exist_ok=True)
    if a.evaluate_only:evaluate(out);return
    start=time.time();cfg=Config(**({'horizons':tuple(a.horizons)} if a.horizons else {}))
    p,m,market,notes=load_archive(a.inputs,a.universe)
    print('INPUT',a.universe,p.shape,notes['price_cutoff'],flush=True)
    first=2013 if a.universe=='sp500' else 2018;years=a.years or list(range(first,p.index[-1].year+1))
    if a.resume:
        meta=json.loads((out/'metadata.json').read_text())
        validate_resume(meta,cfg,a.universe,years,bool(a.null))
        if meta['inputs']['manifest_sha256']!=notes['manifest_sha256']:
            raise ValueError('Resume input manifest changed')
        f=pd.read_parquet(out/'features.parquet');cols=meta['feature_columns']
        audits=json.loads((out/'fit_audit.json').read_text())
        meta['resumed_after_resource_interruption']=True
    else:
        f,coverage,cols=make_features(p,m,market)
        f.to_parquet(out/'features.parquet',index=False);coverage.to_csv(out/'coverage.csv',index=False)
        meta=dict(config=asdict(cfg),years=years,null=bool(a.null),inputs=notes,cutoff_i=len(p)-1,
            eligible_rows=len(f),tickers=f.ticker.nunique(),feature_columns=cols,
            python=platform.python_version(),source_hashes={x.name:hashlib.sha256(x.read_bytes()).hexdigest()
                for x in Path(__file__).parent.glob('*.py')})
        audits=[]
    dump(out/'metadata.json',meta)
    for h in cfg.horizons:
        if a.resume and (out/f'forecasts_{h}.parquet').exists():
            completed=[v for v in audits if v['horizon']==h]
            if sorted(v['year'] for v in completed)==sorted(years):
                print('RESUME verified completed horizon',h,flush=True);continue
        audits=[v for v in audits if v['horizon']!=h]
        y=labels_for_horizon(f,p,market,h);y.to_parquet(out/f'outcomes_{h}.parquet',index=False);chunks=[]
        for year in years:
            asof=int(p.index.searchsorted(pd.Timestamp(year,1,1)))
            until=int(p.index.searchsorted(pd.Timestamp(year+1,1,1)))
            test=(f.i.to_numpy()>=asof)&(f.i.to_numpy()<until)
            if not test.any():continue
            pred,audit=fit_fold(f,y,cols,asof,h,cfg,test,shuffle=a.null)
            audit.update(year=year,universe=a.universe);audits.append(audit)
            if len(pred):chunks.append(pred)
            print(a.universe,'null' if a.null else 'real',h,year,audit['status'],len(pred),round(time.time()-start,1),flush=True)
        pred=pd.concat(chunks,ignore_index=True) if chunks else pd.DataFrame()
        pred.to_parquet(out/f'forecasts_{h}.parquet',index=False);dump(out/'fit_audit.json',audits)
    meta['training_seconds']=time.time()-start;dump(out/'metadata.json',meta)
    evaluate(out);meta=json.loads((out/'metadata.json').read_text());meta['total_seconds']=time.time()-start;dump(out/'metadata.json',meta)
    print('DONE',a.universe,round(meta['total_seconds'],2),flush=True)

if __name__=='__main__':main()
