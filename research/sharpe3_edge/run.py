"""Execute fixed IPD variants, actual share ledgers, sensitivities and falsification."""
from pathlib import Path
from dataclasses import asdict
import argparse,hashlib,json,gc
import numpy as np
import pandas as pd
from .data import load
from .signals import features,labels,fit_horizons,horizon_map,HORIZONS,METHODS
from .portfolio import simulate,Settings
from .metrics import describe,bootstrap


def safe(obj):
    if isinstance(obj,dict):return {str(k):safe(v) for k,v in obj.items()}
    if isinstance(obj,(list,tuple)):return [safe(v) for v in obj]
    if isinstance(obj,(np.bool_,bool)):return bool(obj)
    if isinstance(obj,(np.integer,)):return int(obj)
    if isinstance(obj,(float,np.floating)):return float(obj) if np.isfinite(obj) else None
    return obj


def write(path,obj):Path(path).write_text(json.dumps(safe(obj),indent=2,allow_nan=False))


def source_hash():
    root=Path(__file__).parent
    return {f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(root.glob('*.py'))}


def compute_maps(f,labs,p,start_i):
    maps={};strict={};details=[]
    for year in range(p.index[start_i].year,p.index[-1].year+1):
        asof=int(p.index.searchsorted(pd.Timestamp(f'{year}-01-01')))
        s=fit_horizons(f,labs,asof)
        if not s.empty:
            if not (s.max_exit_i<asof).all():raise AssertionError('Future labels in fit')
            s['year']=year;details.append(s)
        maps[year]=horizon_map(s);strict[year]=horizon_map(s,True)
    return maps,strict,pd.concat(details,ignore_index=True) if details else pd.DataFrame()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--inputs',required=True);ap.add_argument('--output',required=True)
    ap.add_argument('--universe',choices=['sp500','ndx'],required=True);ap.add_argument('--quick',action='store_true')
    args=ap.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    p,m,b,v,notes=load(args.inputs,args.universe)
    print('loaded',args.universe,p.shape,flush=True)
    f,cov=features(p,m,b,v);f.to_pickle(out/'features.pkl');cov.to_csv(out/'coverage.csv',index=False)
    start_year=2013 if args.universe=='sp500' else 2018
    start_i=int(p.index.searchsorted(pd.Timestamp(f'{start_year}-01-01')))
    print('features',len(f),'eligible symbols',f.ticker.nunique(),flush=True)
    labs={};edge=[]
    for h in HORIZONS:
        y=labels(f,p,b,h);labs[h]=y
        for name,mask in [('all',f.family.notna()),('information',f.family.eq('information')),('transient',f.family.eq('transient'))]:
            for era,em in [('full',f.i>=start_i),('2013_2017',f.date.between('2013','2018',inclusive='left')),
                         ('2018_2021',f.date.between('2018','2022',inclusive='left')),('2022_on',f.date>='2022'),('2024_on',f.date>='2024')]:
                z=y.loc[mask&em&y.matured];d=z.groupby('i').excess.mean();bb=d.groupby(d.index//h).mean()
                if len(z):edge.append(dict(family=name,h=h,era=era,rows=len(z),dates=len(d),blocks=len(bb),
                    mean_return=float(z.ret.mean()),mean_paired_excess=float(d.mean()),
                    block_se=float(bb.std(ddof=1)/np.sqrt(len(bb))) if len(bb)>1 else None,
                    positive_rate=float((z.ret>0).mean()),missing_paths=int((~z.complete).sum())))
    pd.DataFrame(edge).to_csv(out/'mechanism_edge.csv',index=False)
    maps,strict,fit=compute_maps(f,labs,p,start_i);fit.to_csv(out/'horizon_fits.csv',index=False)
    write(out/'horizon_maps.json',{'primary':maps,'selective':strict})
    # State permutation: identical economic labels but randomized assignment of states within each origin.
    rng=np.random.default_rng(77);null=f.copy()
    for _,inds in f.groupby('i').groups.items():
        perm=rng.permutation(np.asarray(inds));null.loc[inds,['family','strength']]=f.loc[perm,['family','strength']].to_numpy()
    nullmaps,_,_=compute_maps(null,labs,p,start_i)
    del labs;gc.collect()
    frames={int(i):g.copy() for i,g in f.groupby('i',sort=True)}
    nulframes={int(i):g.copy() for i,g in null.groupby('i',sort=True)}
    # Real-data prefix audit of all computed features, including PIT membership and market mutation.
    cut=int(p.index.searchsorted(pd.Timestamp('2024-01-01')))
    pm=p.copy();pm.iloc[cut:]*=5;mm=m.copy();mm.iloc[cut:]=False;bm=b.copy();bm.iloc[cut:]*=.2
    fm,_=features(pm,mm,bm,v)
    pd.testing.assert_frame_equal(f[f.i<cut].reset_index(drop=True),fm[fm.i<cut].reset_index(drop=True))
    del pm,mm,bm,fm;gc.collect()
    jobs=[]
    for method in METHODS:
        if method=='ipd_volume60' and args.universe=='sp500':continue
        jobs.append((method,Settings(),'base'))
    if not args.quick:
        for method in ('ipd_adaptive','ipd_fixed60','ipd_selective','momentum60'):
            jobs.extend((method,Settings(fee_bps=c),'cost') for c in (0,10,50))
            jobs.extend((method,Settings(entry_lag=l),'delay') for l in (2,5))
            jobs.append((method,Settings(hedge=True),'hedged_assumption'))
        jobs.extend([('ipd_adaptive',Settings(missing_grace=5),'missing5'),
                     ('ipd_fixed60',Settings(missing_grace=5),'missing5'),
                     ('ipd_adaptive',Settings(),'state_null')])
    summaries=[];kept={};audit=[];eras=[]
    for method,cfg,kind in jobs:
        tag=f'{method}_{kind}_c{cfg.fee_bps:g}_l{cfg.entry_lag}'
        local=out/tag;local.mkdir(exist_ok=True)
        usemaps=nullmaps if kind=='state_null' else strict if method=='ipd_selective' else maps
        d,tr,orders=simulate(p,b,nulframes if kind=='state_null' else frames,method,start_i,usemaps,cfg)
        met=describe(d,b);met.update(method=method,kind=kind,tag=tag,**asdict(cfg))
        met.update(trades=len(tr),closed=int(tr.closed.sum()) if len(tr) else 0,
                   entry_dates=int(tr.entry_i.nunique()) if len(tr) else 0,writeoffs=int(tr.written_off.sum()) if len(tr) else 0,
                   horizons=tr.h.value_counts().to_dict() if len(tr) else {},
                   annual_turnover=float(d.turnover_notional.diff().fillna(0).div(d.nav.shift().fillna(1)).sum()*252/len(d)))
        d.to_csv(local/'daily.csv',index=False);tr.to_csv(local/'trades.csv',index=False);orders.to_csv(local/'orders.csv',index=False)
        write(local/'metrics.json',met);summaries.append(met)
        if kind=='base':kept[method]=d
        for era,lower,upper in [('2013_2017','2013','2018'),('2018_2021','2018','2022'),('2022_on','2022','2099'),('2024_on','2024','2099')]:
            q=d[d.date.between(lower,upper,inclusive='left')]
            if len(q)>20:eras.append(dict(tag=tag,era=era,**describe(q,b)))
        if method=='ipd_adaptive' and kind=='base':
            # Recompute ledger, not merely arithmetic, with arbitrary future corruption.
            pm=p.copy();pm.iloc[cut:]*=7;bm=b.copy();bm.iloc[cut:]*=2
            dm,_,_=simulate(pm,bm,frames,method,start_i,maps,cfg)
            pd.testing.assert_frame_equal(d[d.i<cut],dm[dm.i<cut]);del pm,bm,dm
            audit.append(dict(check='future_price_ledger_prefix',passed=True,days=int((d.i<cut).sum())))
        print(tag,'Sharpe',met['sharpe_zero'],'CAGR',met['cagr'],'DD',met['max_drawdown'],'trades',len(tr),flush=True)
    benchmark=kept['ipd_fixed60'].copy();benchmark['return']=0.
    values=b.iloc[start_i:].to_numpy();r=np.r_[0.,values[1:]/values[:-1]-1];r[0]=1/(1+.0025)-1
    benchmark['return']=r;benchmark['nav']=np.cumprod(1+r);benchmark['exposure']=1.;benchmark['gross']=1.
    benchmark['positions']=1;benchmark['borrow']=0.;benchmark['costs']=.0025;benchmark['turnover_notional']=1.;benchmark['margin_breach']=False
    benchmark.to_csv(out/'benchmark_daily.csv',index=False)
    bmmet=describe(benchmark,b);bmmet.update(method='SPY',kind='benchmark',tag='SPY',fee_bps=25.,entry_lag=1)
    summaries.append(bmmet)
    for era,lo,hi in [('2013_2017','2013','2018'),('2018_2021','2018','2022'),('2022_on','2022','2099'),('2024_on','2024','2099')]:
        q=benchmark[benchmark.date.between(lo,hi,inclusive='left')]
        if len(q)>20:eras.append(dict(tag='SPY',era=era,**describe(q,b)))
    uncertainty={name:bootstrap(d['return'],benchmark['return'],126) for name,d in kept.items()}
    # Deleting eras is a diagnostic, not training on the future held-out era.
    for name,d in kept.items():
        for lo,hi in [('2013','2018'),('2018','2022'),('2022','2099')]:
            mask=~d.date.between(lo,hi,inclusive='left')
            if mask.sum()>252:
                from .metrics import sharpe
                uncertainty[name]['without_'+lo+'_'+hi]=sharpe(d.loc[mask,'return'])
    pd.DataFrame(summaries).drop(columns=['annual_returns','horizons'],errors='ignore').to_csv(out/'summary.csv',index=False)
    pd.DataFrame(eras).drop(columns=['annual_returns'],errors='ignore').to_csv(out/'eras.csv',index=False)
    write(out/'all_metrics.json',summaries);write(out/'uncertainty.json',uncertainty)
    audit.append(dict(check='real_features_future_mutation',passed=True,rows=int((f.i<cut).sum())))
    write(out/'validation.json',audit)
    notes.update(source=source_hash(),feature_rows=len(f),eligible_tickers=int(f.ticker.nunique()),
                 start_year=start_year,experiment='A',fixed_variants=list(METHODS),completed_ledger_runs=len(jobs),
                 caveat='Costed hypothetical archived-series P&L, not fill-verified trading performance')
    write(out/'metadata.json',notes)
    print('DONE',out,flush=True)

if __name__=='__main__':main()
