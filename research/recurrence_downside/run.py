"""Reproducible experiment runner; no brokerage or recommendation output."""
import argparse,json,time,hashlib,platform
from dataclasses import asdict
from pathlib import Path
import numpy as np
import pandas as pd
from research.failure_first.data import load_archive,make_features
from research.failure_first.evaluate import clean_json
from .engine import Config,downside_labels,annual_predictions,METHODS,candidate_rows
from .evaluate import policy_replay,controls,summary


def write(path,obj):path.write_text(json.dumps(clean_json(obj),indent=2,allow_nan=False))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--inputs',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--universe',choices=['sp500','ndx'],required=True)
    ap.add_argument('--years',nargs='*',type=int);ap.add_argument('--null',action='store_true')
    ap.add_argument('--cache',type=Path);ap.add_argument('--evaluate-only',action='store_true')
    a=ap.parse_args();o=a.output;o.mkdir(parents=True,exist_ok=True);cfg=Config();t=time.time()
    cache=a.cache or o
    if a.evaluate_only:
        f=pd.read_parquet(cache/'features.parquet');notes=json.loads((cache/'data_metadata.json').read_text())
        p=None;cols=notes['feature_columns']
    else:
        p,m,market,notes=load_archive(a.inputs,a.universe)
        print('inputs',a.universe,p.shape,notes['price_cutoff'],flush=True)
        if (cache/'features.parquet').exists():f=pd.read_parquet(cache/'features.parquet');cols=json.loads((cache/'data_metadata.json').read_text())['feature_columns']
        else:
            f,coverage,cols=make_features(p,m,market)
            cache.mkdir(parents=True,exist_ok=True);f.to_parquet(cache/'features.parquet',index=False)
            coverage.to_csv(cache/'coverage.csv',index=False);notes['feature_columns']=cols
            write(cache/'data_metadata.json',notes)
    labels={}
    for h in cfg.horizons:
        fn=cache/f'labels_{h}.parquet'
        if fn.exists():labels[h]=pd.read_parquet(fn)
        else:
            labels[h]=downside_labels(f,p,market,h);labels[h].to_parquet(fn,index=False)
    first=2013 if a.universe=='sp500' else 2018
    years=a.years or list(range(first,pd.Timestamp(notes['price_cutoff']).year+1))
    metadata={'config':asdict(cfg),'inputs':notes,'candidate_rule':'frozen top5 relative weakness/acceleration/failed recovery',
        'feature_rows':len(f),'eligible_tickers':int(f.ticker.nunique()),'candidate_rows':len(candidate_rows(f)),
        'years':years,'within_date_null':a.null,'python':platform.python_version(),
        'code_hashes':{x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in Path(__file__).parent.glob('*.py')}}
    write(o/'metadata.json',metadata)
    audits=[];chunks=[]
    for year in years:
        target=o/f'predictions_{year}.parquet';au=o/f'audit_{year}.json'
        if a.evaluate_only:
            pred=pd.read_parquet(target);audit=json.loads(au.read_text())
        else:
            asof=int(p.index.searchsorted(pd.Timestamp(year,1,1)));until=int(p.index.searchsorted(pd.Timestamp(year+1,1,1)))
            pred,audit=annual_predictions(f,labels,cols,asof,until,cfg,shuffle=a.null)
            pred.to_parquet(target,index=False);write(au,audit)
        chunks.append(pred);audits.extend(audit)
        print(a.universe,'null' if a.null else 'real',year,len(pred),'seconds',round(time.time()-t,1),flush=True)
    pred=pd.concat(chunks,ignore_index=True)
    # Controls use all eligible stocks, not only the deliberately restricted candidate pond.
    ys=pd.concat([y for y in labels.values()],ignore_index=True)
    active_years=pd.to_datetime(f.date).dt.year.isin(years)
    dates=f.loc[active_years,'i'].drop_duplicates().sort_values().tolist()
    picks,dec=policy_replay(pred,ys,dates,cfg);picks=controls(picks,ys,f,cfg)
    picks.to_csv(o/'picks.csv',index=False);dec.to_csv(o/'decisions.csv',index=False)
    result=summary(picks,dec);write(o/'summary.json',result);write(o/'fit_audit.json',audits)
    flat=pd.DataFrame([{k:v for k,v in d.items() if not isinstance(v,(dict,list))} for d in result])
    flat.to_csv(o/'summary.csv',index=False)
    reli=[];joined=pred.merge(ys[['row_id','horizon','success','matured']],on=['row_id','horizon'],validate='one_to_one')
    for h,g in joined[joined.matured].groupby('horizon'):
        for method in METHODS:
            s=g.loc[g[method].notna()]
            if not len(s):continue
            reli.append({'horizon':int(h),'method':method,'n':len(s),'brier':float(((s[method]-s.success)**2).mean()),
                         'maximum_score':float(s[method].max()),'mean_score':float(s[method].mean()),
                         'base_rate':float(s.success.mean()),
                         'bins':[{'threshold':q,'n':int((s[method]>q).sum()),
                                  'precision':float(s.loc[s[method]>q,'success'].mean()) if (s[method]>q).any() else None}
                                  for q in cfg.thresholds]})
    write(o/'reliability.json',reli);metadata['seconds']=round(time.time()-t,3);metadata['forecast_rows']=len(pred)
    write(o/'metadata.json',metadata)
    print(flat[['policy','matured','wins','precision','random_expected','candidate_expected','nonoverlap_n']].to_string(index=False),flush=True)
    print('DONE',a.universe,metadata['seconds'],flush=True)

if __name__=='__main__':main()
