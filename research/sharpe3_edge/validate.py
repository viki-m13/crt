"""Independent reconstruction from archived prices, quantities and locked trade dates."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
from .data import load
from .run import write
from .metrics import sharpe


def reconcile(p,b,trades,start,end,fee_bps=25,borrow_rate=.01,hedge=False):
    delta=np.zeros(end-start+1);a=p.to_numpy(float);bm=b.to_numpy(float)
    dates=p.index;checked=0
    for row in trades.itertuples(index=False):
        entry=int(row.entry_i);exit_i=int(row.exit_i);stop=min(end,exit_i)
        assert exit_i-entry>=30 and exit_i-entry==row.h
        col=p.columns.get_loc(row.ticker)
        px=a[entry:stop+1,col].copy()
        assert np.isfinite(px[0]) and px[0]>0
        assert np.isclose(px[0],row.entry_price,rtol=0,atol=1e-10)
        assert np.isclose(row.qty*px[0],row.notional,rtol=1e-12)
        bad=~np.isfinite(px)|(px<=0);px[np.cumsum(bad)>0]=0
        st=int(entry-start);en=int(stop-start)
        delta[st]-=float(row.entry_cost)
        changes=row.qty*np.diff(px)-row.hedge_qty*np.diff(bm[entry:stop+1])
        days=np.diff(dates[entry:stop+1].to_numpy()).astype('timedelta64[D]').astype(float)
        changes-=row.hedge_qty*bm[entry:stop]*borrow_rate*days/365.25
        delta[st+1:en+1]+=changes
        if bool(row.closed):
            assert exit_i<=end
            cost=fee_bps/10000*(row.qty*px[-1]+row.hedge_qty*bm[stop])
            assert np.isclose(row.exit_cost,cost,rtol=1e-12,atol=1e-12)
            assert np.isclose(row.exit_price,px[-1],rtol=1e-12,atol=1e-12)
            delta[en]-=cost
        else:assert exit_i>end
        checked+=1
    return 1+np.cumsum(delta),checked


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--inputs',required=True);ap.add_argument('--directory',required=True)
    ap.add_argument('--universe',required=True);ap.add_argument('--all-costs',action='store_true');args=ap.parse_args()
    root=Path(args.directory);p,m,b,v,notes=load(args.inputs,args.universe);summary=pd.read_csv(root/'summary.csv')
    rows=[]
    for row in summary.itertuples(index=False):
        if row.kind not in ('base','hedge','hedged_assumption') and not args.all_costs:continue
        if getattr(row,'missing_grace',0)>0:continue
        folder=root/row.tag
        if not folder.exists():continue
        d=pd.read_csv(folder/'daily.csv')
        try:tr=pd.read_csv(folder/'trades.csv')
        except pd.errors.EmptyDataError:tr=pd.DataFrame()
        nav,n=reconcile(p,b,tr,int(d.i.iloc[0]),int(d.i.iloc[-1]),row.fee_bps)
        discrepancy=float(np.max(np.abs(nav-d.nav.to_numpy())))
        if discrepancy>1e-9:raise AssertionError(f'Independent NAV discrepancy {row.tag}: {discrepancy}')
        r=np.r_[nav[0]-1,nav[1:]/nav[:-1]-1]
        if np.max(np.abs(r-d['return']))>1e-10:raise AssertionError('Daily-return mismatch')
        sr=sharpe(r)
        if sr is not None and abs(sr-row.sharpe_zero)>1e-9:raise AssertionError('Sharpe mismatch')
        if n:
            # Known-at-signal membership must hold; no current-universe filtering.
            idx=tr.signal_i.to_numpy(int);cols=p.columns.get_indexer(tr.ticker)
            assert m.to_numpy()[idx,cols].all()
            assert (tr.entry_i-tr.signal_i==row.entry_lag).all()
        rows.append(dict(tag=row.tag,lots_checked=n,days_checked=len(d),max_nav_error=discrepancy,passed=True))
    write(root/'independent_ledger_audit.json',rows);print('INDEPENDENT LEDGER AUDIT',len(rows),'runs',sum(x['lots_checked'] for x in rows),'lots',flush=True)

if __name__=='__main__':main()
