"""As-filed cash-confirmed operating improvement; no latest-frame backfill."""
from __future__ import annotations
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np
import pandas as pd
from research.edge3.data import load_bonds
from research.edge3.account import Costs, simulate, metrics, bootstrap

REVENUE = ('RevenueFromContractWithCustomerExcludingAssessedTax','Revenues',
           'SalesRevenueNet','RevenueFromContractWithCustomerIncludingAssessedTax')
OI='OperatingIncomeLoss'
CFO='NetCashProvidedByUsedInOperatingActivities'
METHODS=('profitability','cash_confirmed','disclosure_gap','accrual_warning')


def verify_facts(root: Path) -> pd.DataFrame:
    manifest=json.loads((root/'export_hashes.json').read_text())
    for name,digest in manifest.items():
        p=root/name
        if p.parent.resolve()!=root.resolve():raise ValueError('unsafe manifest path')
        if hashlib.sha256(p.read_bytes()).hexdigest()!=digest:raise ValueError(f'bad hash {name}')
    status=json.loads((root/'download_state.json').read_text())
    if status['status']!='complete':raise ValueError('SEC retrieval incomplete; do not silently backtest partial data')
    return pd.read_parquet(root/'as_filed_facts.parquet')


def _number(state,tag,period):
    value=state.get((tag,*period),np.nan)
    return float(value) if value is not None else np.nan


def _revenues(state,current,previous):
    # Do not compare tax-including current sales to tax-excluding historical sales.
    for tag in REVENUE:
        a,b=_number(state,tag,current),_number(state,tag,previous)
        if np.isfinite(a) and np.isfinite(b) and min(a,b)>0:return a,b,tag
    return None


def build_events(facts: pd.DataFrame) -> tuple[pd.DataFrame,dict]:
    """Update an issuer's knowledge only on filing dates; never use future revisions.

    Historical identity resolution remains an upstream limitation. Context-ambiguous
    same-tag/same-period/same-filing-date values are withheld, not chosen by hindsight.
    Quarterly profitability and year-to-date cash flow retain their own durations.
    """
    required={'cik','symbols','tag','start','end','val','accn','filed','form'}
    if not required.issubset(facts):raise ValueError('facts lack timestamps or provenance')
    f=facts.loc[facts.tag.isin({OI,CFO,*REVENUE})].copy()
    for c in ('start','end','filed'):f[c]=pd.to_datetime(f[c],errors='coerce')
    f['val']=pd.to_numeric(f.val,errors='coerce')
    f=f[f.form.isin(['10-K','10-Q','10-K/A','10-Q/A']) & f.filed.notna() & f.end.notna()]
    if (f.end>f.filed).any():raise ValueError('future reporting period in filed fact')
    out=[]; stats={'ambiguous_fact_updates':0,'quarter_reports_without_comparison':0,
                   'comparable_reports':0,'comparable_cash_reports':0}
    for cik,g in f.groupby('cik',sort=True):
        state={};reported=set()
        for filed,block in g.groupby('filed',sort=True):
            valid=block.dropna(subset=['start'])
            updates=valid.groupby(['tag','start','end'],sort=True)
            for key,v in updates:
                values=v.val.dropna().unique()
                if len(values)>1:
                    state[key]=np.nan;stats['ambiguous_fact_updates']+=1
                elif len(values)==1:state[key]=float(values[0])
                else:state[key]=np.nan
            # Only recent, first-reported true quarters are trade events. A later
            # amendment cannot resurrect an already reported quarter as a new trade.
            periods=sorted({(a,b) for tag,a,b in state if tag==OI and 70<=(b-a).days+1<=105
                            and 0<=(filed-b).days<=150},key=lambda z:z[1])
            for current in periods:
                end=current[1]
                if end in reported:continue
                if not np.isfinite(_number(state,OI,current)):continue
                if not any(np.isfinite(_number(state,t,current)) for t in REVENUE):continue
                reported.add(end)
                past=sorted({(a,b) for tag,a,b in state if tag==OI and 70<=(b-a).days+1<=105
                             and 330<=(end-b).days<=400},key=lambda z:abs((end-z[1]).days-365))
                comparable=None
                for previous in past:
                    rev=_revenues(state,current,previous)
                    oldoi=_number(state,OI,previous)
                    if rev is not None and np.isfinite(oldoi):comparable=(previous,rev,oldoi);break
                if comparable is None:
                    stats['quarter_reports_without_comparison']+=1;continue
                previous,(rev,oldrev,tag),oldoi=comparable
                margin=_number(state,OI,current)/rev; oldmargin=oldoi/oldrev
                cash_margin=cash_change=np.nan;cash_days=0
                periods_cf=sorted({(a,b) for t,a,b in state if t==CFO and b==end
                                   and 70<=(b-a).days+1<=380},key=lambda z:(z[1]-z[0]).days,reverse=True)
                for curcf in periods_cf:
                    dur=(curcf[1]-curcf[0]).days+1
                    olds=sorted({(a,b) for t,a,b in state if t==CFO and 330<=(end-b).days<=400
                                 and abs(((b-a).days+1)-dur)<=20},key=lambda z:abs((end-z[1]).days-365))
                    found=False
                    for oldcf in olds:
                        a,b=_number(state,CFO,curcf),_number(state,CFO,oldcf)
                        denom=_revenues(state,curcf,oldcf)
                        if denom is not None and np.isfinite(a) and np.isfinite(b):
                            cash_margin=a/denom[0];cash_change=cash_margin-b/denom[1]
                            cash_days=dur;found=True;break
                    if found:break
                symbols=sorted(set('|'.join(block.symbols.dropna().astype(str)).split('|')))
                out.append({'cik':int(cik),'symbols':'|'.join(symbols),'filed':filed,
                            'period_start':current[0],'period_end':end,'previous_period_end':previous[1],
                            'revenue_tag':tag,'revenue_growth':rev/oldrev-1,
                            'margin':margin,'margin_change':margin-oldmargin,
                            'cash_margin':cash_margin,'cash_change':cash_change,'cash_duration_days':cash_days,
                            'source_accessions':'|'.join(sorted(set(block.accn.dropna().astype(str))))})
                stats['comparable_reports']+=1
                stats['comparable_cash_reports']+=int(np.isfinite(cash_margin))
    columns=['cik','symbols','filed','period_start','period_end','previous_period_end','revenue_tag',
             'revenue_growth','margin','margin_change','cash_margin','cash_change','cash_duration_days',
             'source_accessions']
    return pd.DataFrame(out,columns=columns),stats


def attach_events(events,px,members):
    lp=np.log(px);r=px.pct_change(fill_method=None)
    past_ok=r.rolling(252,min_periods=252).count().eq(252).to_numpy()
    cols=px.columns;source=[];dropped={};bm=lp['SPY'].diff(63)
    for e in events.to_dict('records'):
        i=int(px.index.searchsorted(pd.Timestamp(e['filed']),side='right'))
        if i>=len(px):continue
        aliases=[]
        for s in str(e['symbols']).split('|'):
            if s in cols:aliases.append(s)
            elif s.replace('-','.') in cols:aliases.append(s.replace('-','.'))
        eligible=sorted({s for s in aliases if members[i,cols.get_loc(s)] and past_ok[i,cols.get_loc(s)]})
        if not eligible:
            dropped['no_eligible_identity_history']=dropped.get('no_eligible_identity_history',0)+1;continue
        symbol=eligible[0] # one share class per issuer, deterministic before any outcome
        e.update(i=i,ticker=symbol,signal_date=px.index[i],relative63=float(lp[symbol].diff(63).iloc[i]-bm.iloc[i]))
        source.append(e)
    return pd.DataFrame(source),dropped,past_ok


def event_signals(events,px,members,past_ok,cadence=1):
    result={k:np.zeros(px.shape) for k in METHODS};baskets=[]
    if events.empty:return result,baskets
    ev=events.copy();ev['batch']=((ev.i+cadence-1)//cadence)*cadence
    for t,g in ev.groupby('batch',sort=True):
        t=int(t)
        if t>=len(px):continue
        g=g.sort_values(['i','filed','ticker']).drop_duplicates('ticker',keep='last').copy()
        g=g[[members[t,px.columns.get_loc(s)] and past_ok[t,px.columns.get_loc(s)] for s in g.ticker]]
        if g.empty:continue
        good=(g.margin_change>0)&(g.revenue_growth>0)
        cash=good&(g.cash_margin>0)&(g.cash_change>0)
        warning=(g.revenue_growth>0)&(g.margin_change<=0)&(g.cash_change<0)
        g['score']=g.margin_change.rank(pct=True)+g.revenue_growth.rank(pct=True)
        pools={'profitability':good,'cash_confirmed':cash,'disclosure_gap':cash&(g.relative63<=0)}
        for method,mask in pools.items():
            take=g.loc[mask].sort_values(['score','ticker'],ascending=[False,True]).head(10)
            for symbol in take.ticker:result[method][t,px.columns.get_loc(symbol)]=.1
            baskets.append({'i':t,'method':method,'long_names':'|'.join(take.ticker),'short_names':''})
        longs=g.loc[cash].sort_values(['score','ticker'],ascending=[False,True]).head(5)
        shorts=g.loc[warning].sort_values(['score','ticker'],ascending=[True,True]).head(5)
        if not longs.empty and not shorts.empty:
            # Symmetric budget is no more than 10% per name; sparse baskets retain cash.
            budget=.1*min(len(longs),len(shorts))
            for s in longs.ticker:result['accrual_warning'][t,px.columns.get_loc(s)]=budget/len(longs)
            for s in shorts.ticker:result['accrual_warning'][t,px.columns.get_loc(s)]=-budget/len(shorts)
            baskets.append({'i':t,'method':'accrual_warning','long_names':'|'.join(longs.ticker),'short_names':'|'.join(shorts.ticker)})
    return result,baskets


def run(inputs,facts,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);started=time.time()
    px,members,rf,coverage=load_bonds(inputs)
    f=verify_facts(Path(facts))
    bad=f[pd.to_datetime(f.end)>pd.to_datetime(f.filed)]
    bad.to_csv(out/'source_timestamp_anomalies.csv',index=False)
    events,stats=build_events(f)
    events.to_parquet(out/'filing_events.parquet',index=False)
    events,dropped,past_ok=attach_events(events,px,members)
    events.to_csv(out/'eligible_events.csv',index=False)
    configs=[];validation=[];baskets=[]
    cut=int(px.index.searchsorted('2021'));best=None
    for cadence in (1,5):
        signals,b=event_signals(events,px,members,past_ok,cadence);baskets+=b
        for method,s in signals.items():
            for hold in (30,60):
                code=f'{method}__c{cadence}__h{hold}'
                d=simulate(px.iloc[:cut],s[:cut],hold,cadence,rf_returns=rf.iloc[:cut])
                st=metrics(d.loc['2016':'2020']);row={'code':code,'method':method,'cadence':cadence,'hold':hold,**st}
                validation.append({'stage':'validation',**row});configs.append((row,s))
                if st.get('sharpe') is not None and st['unknown_liquidations']==0 and st['max_gross']<=2:
                    if best is None or st['sharpe']>best['sharpe']:best=row
    selection={'selected':best,'rules':'validation only, not selection using 2021+ returns'}
    text=json.dumps(selection,indent=2);(out/'pretest_selection.json').write_text(text)
    (out/'pretest_selection.sha256').write_text(hashlib.sha256(text.encode()).hexdigest())
    for row,s in configs:
        code=row['code'];h=row['hold'];c=row['cadence'];p=px.iloc[cut:];rft=rf.iloc[cut:];st=s[cut:]
        d=simulate(p,st,h,c,rf_returns=rft);stat=metrics(d)
        d.to_csv(out/f'{code}.csv');validation.append({'stage':'test',**{k:row[k] for k in ('code','method','cadence','hold')},**stat})
        print('TEST',code,stat['sharpe'],flush=True)
        if best is not None and best['code']==code:
            audit={'base':stat,'block_sharpe_interval':bootstrap(d.excess),
                   'yearly':{str(y):metrics(g) for y,g in d.groupby(d.index.year)}}
            for key,cost,delay in [('low_cost',Costs(1,.01),1),('high_cost',Costs(10,.10),1),('extra_delay',Costs(),2)]:
                audit[key]=metrics(simulate(p,st,h,c,costs=cost,delay=delay,rf_returns=rft))
            null=[]
            for seed in range(10):
                rng=np.random.default_rng(3300+seed);fake=np.zeros_like(st)
                for t in np.flatnonzero(np.any(st!=0,axis=1)):
                    pool=np.flatnonzero(members[t+cut]&past_ok[t+cut]);values=st[t,st[t]!=0]
                    if len(pool)<len(values):raise ValueError('control pool too small')
                    fake[t,rng.choice(pool,len(values),replace=False)]=rng.permutation(values)
                null.append(metrics(simulate(p,fake,h,c,rf_returns=rft)))
            audit['matched_schedule_random_controls']=null
            (out/'selected_audit.json').write_text(json.dumps(audit,indent=2))
    pd.DataFrame(validation).to_csv(out/'all_candidates.csv',index=False)
    pd.DataFrame(baskets).to_csv(out/'baskets.csv',index=False)
    (out/'run_summary.json').write_text(json.dumps({'stats':stats,'dropped':dropped,'eligible_events':len(events),
        'unique_tickers':int(events.ticker.nunique()) if not events.empty else 0,'configs':len(configs),
        'elapsed':time.time()-started,'live_certified':False,'identity_caveat':'current SEC resolver, historical issuer mapping incomplete'},indent=2))

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--inputs',required=True);a.add_argument('--facts',required=True);a.add_argument('--output',required=True)
    x=a.parse_args();run(x.inputs,x.facts,x.output)
