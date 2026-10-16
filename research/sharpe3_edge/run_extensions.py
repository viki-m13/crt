from pathlib import Path
from dataclasses import asdict
import argparse,json
import numpy as np
import pandas as pd
from .data import load
from .extensions import enrich,proposals,NEW_METHODS
from . import portfolio
from .portfolio import Settings
from .metrics import describe,bootstrap
from .run import write,source_hash


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--inputs',required=True);ap.add_argument('--base',required=True)
    ap.add_argument('--output',required=True);ap.add_argument('--universe',required=True);args=ap.parse_args()
    root=Path(args.output);root.mkdir(parents=True,exist_ok=True);base=Path(args.base)
    p,m,b,v,notes=load(args.inputs,args.universe);f=pd.read_pickle(base/'features.pkl');g=enrich(f,p,b)
    cut=int(p.index.searchsorted(pd.Timestamp('2024-01-01')));pm=p.copy();pm.iloc[cut:]*=.1;bm=b.copy();bm.iloc[cut:]*=5
    gm=enrich(f,pm,bm)
    pd.testing.assert_frame_equal(g[g.i<cut],gm[gm.i<cut]);del pm,bm,gm
    frames={int(i):q for i,q in g.groupby('i',sort=True)}
    maps={int(y):x for y,x in json.loads((base/'horizon_maps.json').read_text())['primary'].items()}
    start=int(p.index.searchsorted(pd.Timestamp('2013-01-01' if args.universe=='sp500' else '2018-01-01')))
    portfolio.proposals=proposals
    summaries=[];annual=[];uncertainty={};bmret=pd.read_csv(base/'benchmark_daily.csv')['return']
    for method in (*NEW_METHODS,'ipd_adaptive','ipd_fixed60','momentum60','equal60'):
        for cfg,kind in [(Settings(work_conserving=True),'base'),(Settings(work_conserving=True,fee_bps=0),'zero_cost'),
                         (Settings(work_conserving=True,fee_bps=50),'cost50'),(Settings(work_conserving=True,entry_lag=5),'delay5'),
                         (Settings(work_conserving=True,hedge=True),'hedge')]:
            d,tr,o=portfolio.simulate(p,b,frames,method,start,maps,cfg);tag=method+'_'+kind
            folder=root/tag;folder.mkdir(exist_ok=True)
            met=describe(d,b);met.update(method=method,kind=kind,tag=tag,**asdict(cfg))
            met.update(trades=len(tr),closed=int(tr.closed.sum()) if len(tr) else 0,
                       writeoffs=int(tr.written_off.sum()) if len(tr) else 0,
                       annual_turnover=float(d.turnover_notional.diff().fillna(0).div(d.nav.shift().fillna(1)).sum()*252/len(d)))
            summaries.append(met);d.to_csv(folder/'daily.csv',index=False);tr.to_csv(folder/'trades.csv',index=False);o.to_csv(folder/'orders.csv',index=False)
            for era,lo,hi in [('2013_2017','2013','2018'),('2018_2021','2018','2022'),('2022_on','2022','2099'),('2024_on','2024','2099')]:
                q=d[d.date.between(lo,hi,inclusive='left')]
                if len(q)>20:annual.append(dict(tag=tag,era=era,**describe(q,b)))
            if kind=='base':uncertainty[tag]=bootstrap(d['return'],bmret,126)
            print(args.universe,tag,met['sharpe_zero'],met['cagr'],met['max_drawdown'],len(tr),flush=True)
    pd.DataFrame(summaries).drop(columns='annual_returns').to_csv(root/'summary.csv',index=False)
    pd.DataFrame(annual).drop(columns='annual_returns').to_csv(root/'eras.csv',index=False)
    write(root/'all_metrics.json',summaries);write(root/'uncertainty.json',uncertainty)
    notes.update(source=source_hash(),real_future_feature_rows_checked=int((g.i<cut).sum()),experiment='B',ledger_runs=len(summaries))
    write(root/'metadata.json',notes);print('DONE B',args.universe,flush=True)

if __name__=='__main__':main()
