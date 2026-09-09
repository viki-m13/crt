"""Reproducible prequential training. Current outputs cannot authorize live buys."""
from __future__ import annotations
import argparse,json,hashlib,time,platform
from pathlib import Path
from dataclasses import asdict
import numpy as np
import pandas as pd
from .data import make_features,load_archive
from .model import Config,label_panel,fit_fold,fingerprint


def json_default(x):
    if isinstance(x,(np.integer,)): return int(x)
    if isinstance(x,(np.floating,)): return float(x)
    if isinstance(x,(np.bool_,)): return bool(x)
    if isinstance(x,Path): return str(x)
    raise TypeError(type(x).__name__)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--inputs',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--universe',choices=['sp500','ndx'],required=True)
    ap.add_argument('--null',action='store_true')
    ap.add_argument('--years',nargs='*',type=int)
    args=ap.parse_args(); out=args.output; out.mkdir(parents=True,exist_ok=True)
    cfg=Config(); start=time.time()
    p,m,market,notes=load_archive(args.inputs,args.universe)
    print('archive',args.universe,p.shape,notes['price_cutoff'],flush=True)
    f,coverage,cols=make_features(p,m,market)
    f.to_parquet(out/'features.parquet',index=False); coverage.to_csv(out/'coverage.csv',index=False)
    metadata={'config':asdict(cfg),'config_hash':fingerprint(cfg),'inputs':notes,'feature_columns':cols,
       'eligible_rows':len(f),'symbols':int(f.ticker.nunique()),'first_eligible_date':str(f.date.min()),
       'last_eligible_date':str(f.date.max()),'python':platform.python_version(),
       'source_hashes':{str(x.name):hashlib.sha256(x.read_bytes()).hexdigest() for x in Path(__file__).parent.glob('*.py')}}
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2,default=json_default,allow_nan=False))
    first=2013 if args.universe=='sp500' else 2018
    years=args.years or list(range(first,p.index[-1].year+1))
    audits=[]
    for h in cfg.horizons:
        y=label_panel(f,p,market,h)
        y.to_parquet(out/f'outcomes_{h}.parquet',index=False)
        chunks=[]
        for year in years:
            asof=int(p.index.searchsorted(pd.Timestamp(year,1,1)))
            until=int(p.index.searchsorted(pd.Timestamp(year+1,1,1)))
            mask=(f.i.to_numpy()>=asof)&(f.i.to_numpy()<until)
            if not mask.any(): continue
            prediction,audit=fit_fold(f,y,cols,asof,h,cfg,mask,shuffle=args.null)
            audit.update(year=year,universe=args.universe)
            audits.append(audit)
            chunks.append(prediction)
            print(args.universe,'null' if args.null else 'real',h,year,audit['status'],len(prediction),
                  'elapsed',round(time.time()-start,1),flush=True)
        pred=pd.concat(chunks,ignore_index=True) if chunks else pd.DataFrame()
        pred.to_parquet(out/f'risks_{h}.parquet',index=False)
        (out/'fit_audit.json').write_text(json.dumps(audits,indent=2,default=json_default,allow_nan=False))
    metadata['elapsed_seconds']=round(time.time()-start,3)
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2,default=json_default,allow_nan=False))
    print('DONE',args.universe,metadata['elapsed_seconds'],flush=True)

if __name__=='__main__': main()
