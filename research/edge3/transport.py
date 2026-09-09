"""Cash-disclosure transport: precommitted residual-return learners and real books."""
from __future__ import annotations
import argparse,json,hashlib,time
from pathlib import Path
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from threadpoolctl import threadpool_limits
from research.edge3.data import load_bonds
from research.edge3.disclosure import verify_facts,build_events,attach_events
from research.edge3.account import Costs,simulate,metrics,bootstrap

FAMILIES=('P','F','D','T')
OWN=('revenue_growth','margin','margin_change','cash_margin','cash_change','cash_duration_days')


def make_panel(px,members,events,cadence=5):
    """First-report state, available only after filing, with past-correlation peers."""
    lp=np.log(px);ret=px.pct_change(fill_method=None);rm=ret['SPY'];n,k=px.shape
    v=ret.rolling(63,min_periods=63).std();beta=ret.rolling(126,min_periods=126).cov(rm).div(rm.rolling(126,min_periods=126).var(),axis=0)
    feats={f'r{h}':lp.diff(h) for h in (5,21,63,126,252)}
    feats.update(vol63=v,beta=beta,dd252=lp-lp.rolling(252,min_periods=252).max(),
                 ma200=lp-lp.rolling(200,min_periods=200).mean(),
                 vol_ratio=ret.rolling(21).std()/ret.rolling(126).std(),
                 up_fraction=ret.gt(0).where(ret.notna()).rolling(63,min_periods=63).mean(),
                 relative63=lp.diff(63).sub(lp['SPY'].diff(63),axis=0))
    state={name:np.full((n,k),np.nan) for name in OWN};date_state=np.full((n,k),np.nan)
    for e in events.to_dict('records'):
        i=int(e['i']);j=px.columns.get_loc(e['ticker'])
        for name in OWN:state[name][i,j]=float(e[name])
        date_state[i,j]=pd.Timestamp(e['filed']).to_datetime64().astype('datetime64[D]').astype(int)
    # Forward carrying never crosses backward over an unknown first observation.
    event_mask=np.isfinite(date_state)
    for j in range(k):
        ids=np.flatnonzero(event_mask[:,j])
        for pos,i in enumerate(ids):
            stop=ids[pos+1] if pos+1<len(ids) else n
            for name in OWN:state[name][i:stop,j]=state[name][i,j]
            date_state[i:stop,j]=date_state[i,j]
    day=px.index.to_numpy().astype('datetime64[D]').astype(int)
    age=day[:,None]-date_state
    historic=ret.rolling(252,min_periods=252).count().eq(252).to_numpy()
    eligible=members&historic&np.isfinite(state['margin_change'])&np.isfinite(state['revenue_growth'])&(age<=180)&(age>=0)
    rv=ret.to_numpy();frames=[];graph_audit=[];ref=-1;neighbors={}
    feature_arrays={name:d.to_numpy() for name,d in feats.items()};base_names=list(feats)
    for t in range(255,n,cadence):
        origin=(t//21)*21
        if origin!=ref:
            ref=origin;neighbors={};idx=np.flatnonzero(eligible[ref])
            if len(idx)>=10:
                h=rv[ref-251:ref+1,idx]
                with threadpool_limits(limits=2):corr=np.corrcoef(h,rowvar=False)
                np.fill_diagonal(corr,-np.inf)
                for jj,j in enumerate(idx):
                    order=np.argsort(-corr[jj],kind='stable')[:8]
                    neighbors[j]=idx[order[corr[jj,order]>0]]
                graph_audit.append({'ref_i':int(ref),'last_graph_return_i':int(ref),'names':len(idx)})
        ids=np.flatnonzero(eligible[t])
        if len(ids)==0:continue
        d={'i':np.full(len(ids),t),'j':ids,'date':np.repeat(px.index[t],len(ids)),'ticker':px.columns[ids],
           **{name:a[t,ids] for name,a in feature_arrays.items()},
           **{name:a[t,ids] for name,a in state.items()},'disclosure_age':age[t,ids]}
        d['market63']=np.full(len(ids),float(lp['SPY'].diff(63).iloc[t]))
        d['market_vol']=np.full(len(ids),float(rm.rolling(63).std().iloc[t]))
        d['cash_credible']=((d['cash_margin']>0)&(d['cash_change']>0)).astype(float)
        d['cash_price_gap']=d['cash_change']-d['relative63']
        d['margin_price_gap']=d['margin_change']-d['relative63']
        d['credibility_disagreement']=d['margin_change']*d['cash_credible']*(-d['relative63'])
        for name in ('margin_change','cash_change','revenue_growth'):
            vals=[]
            for j in ids:
                nbr=neighbors.get(j,np.array([],int));nbr=nbr[eligible[t,nbr]]
                a=state[name][t,nbr];a=a[np.isfinite(a)]
                vals.append(float(np.mean(a)) if len(a)>=2 else np.nan)
            d['peer_'+name]=vals
        frame=pd.DataFrame(d);frame['row_id']=np.arange(len(frame))
        frames.append(frame)
    out=pd.concat(frames,ignore_index=True) if frames else pd.DataFrame()
    if out.empty:raise ValueError('no supported feature observations')
    out['row_id']=np.arange(len(out));price=base_names+['market63','market_vol']
    own=price+list(OWN)+['disclosure_age']
    disagreement=own+['cash_credible','cash_price_gap','margin_price_gap','credibility_disagreement']
    groups={'P':price,'F':own,'D':disagreement,'T':disagreement+['peer_'+x for x in ('margin_change','cash_change','revenue_growth')]}
    return out,groups,pd.DataFrame(graph_audit)


def targets(frame,px,h):
    i=frame.i.to_numpy(int);j=frame.j.to_numpy(int);entry=i+1;end=entry+h;valid=end<len(px)
    p=px.to_numpy(float);target=np.full(len(frame),np.nan)
    z=np.flatnonzero(valid);a=p[entry[z],j[z]];b=p[end[z],j[z]]
    bidx=px.columns.get_loc('SPY');br=p[end[z],bidx]/p[entry[z],bidx]-1
    scale=np.maximum(frame.vol63.to_numpy()[z]*np.sqrt(h)*(1+np.abs(frame.beta.to_numpy()[z])),1e-6)
    values=((b/a-1)-frame.beta.to_numpy()[z]*br)/scale
    target[z]=np.where(np.isfinite(a)&np.isfinite(b)&(a>0)&(b>0)&np.isfinite(values),values,np.nan)
    return target,end


def predict(frame,groups,px,h,null=False,start_year=2013):
    y,ends=targets(frame,px,h);i=frame.i.to_numpy(int);years=frame.date.dt.year.to_numpy()
    out={name:np.full(len(frame),np.nan) for name in groups};audits=[]
    for year in range(start_year,int(years.max())+1):
        cutoff=int(px.index.searchsorted(f'{year}-01-01'));test=np.flatnonzero(years==year)
        train=np.flatnonzero((ends<cutoff)&(i>=cutoff-2520)&np.isfinite(y))
        if len(train)<2000 or len(np.unique(i[train]))<52 or len(test)==0:continue
        yy=y[train].copy();orig=i[train]
        if null:
            rng=np.random.default_rng(20260911+year+h)
            for d in np.unique(orig):
                which=np.flatnonzero(orig==d);yy[which]=rng.permutation(yy[which])
        _,iv,cnt=np.unique(orig,return_inverse=True,return_counts=True)
        weight=np.exp2(-(cutoff-orig)/756)/cnt[iv];weight/=weight.mean()
        audit={'year':year,'horizon':h,'train_rows':len(train),'test_rows':len(test),
               'max_train_endpoint_i':int(ends[train].max()),'first_test_i':int(i[test].min()),
               'fit_i':cutoff,'null':bool(null)}
        assert audit['max_train_endpoint_i']<cutoff<=audit['first_test_i']
        audits.append(audit)
        for name,cols in groups.items():
            x=frame[cols].replace([np.inf,-np.inf],np.nan).to_numpy(float)
            model=LGBMRegressor(objective='huber',alpha=.9,n_estimators=100,num_leaves=7,max_depth=3,
                learning_rate=.04,min_child_samples=200,reg_lambda=25,max_bin=63,n_jobs=2,
                random_state=20260911,verbosity=-1,deterministic=True,force_col_wise=True)
            with threadpool_limits(limits=2):model.fit(x[train],yy,sample_weight=weight)
            out[name][test]=model.predict(x[test])
        print('FIT',year,h,'null',null,len(train),flush=True)
    return out,pd.DataFrame(audits)


def construct(frame,prediction,px,cutoff=.05,book='hedged'):
    s=np.zeros(px.shape);bench=px.columns.get_loc('SPY')
    for t,g in frame.groupby('i',sort=True):
        ids=g.index.to_numpy();p=prediction[ids];valid=np.isfinite(p)&np.isfinite(g.vol63.to_numpy())&(g.vol63.to_numpy()>0)
        order=np.argsort(-p,kind='stable');long=order[valid[order]&(p[order]>=cutoff)][:20]
        if len(long)<5:continue
        names=g.j.to_numpy(int);vol=g.vol63.to_numpy();beta=g.beta.to_numpy()
        w=np.zeros(len(g));lw=1/vol[long];w[long]=lw/lw.sum()
        if book=='hedged':
            order=np.argsort(p,kind='stable');short=order[valid[order]&(p[order]<=-cutoff)][:20]
            if len(short)<5:continue
            sw=1/vol[short];w[long]*=.5;w[short]=-.5*sw/sw.sum()
            s[t,names]=w;s[t,bench]-=float(np.nansum(w*beta))
        elif book=='long':s[t,names]=w
        else:raise ValueError('unknown book')
        gross=np.abs(s[t]).sum()
        if gross:s[t]/=gross
    return s


def safe_run(px,s,h,rf,costs=Costs(),delay=1):
    try:
        d=simulate(px,s,h,5,costs=costs,delay=delay,rf_returns=rf)
        return d,metrics(d)
    except RuntimeError as e:return None,{'sharpe':None,'bankrupt':True,'error':str(e)}


def run(inputs,facts,out):
    started=time.time();out=Path(out);out.mkdir(exist_ok=True,parents=True)
    px,m,rf,cov=load_bonds(inputs);events,stats=build_events(verify_facts(Path(facts)))
    ev,drop,_=attach_events(events,px,m);frame,groups,ga=make_panel(px,m,ev)
    frame.to_parquet(out/'features.parquet',index=False);ga.to_csv(out/'graph_audit.csv',index=False)
    (out/'feature_groups.json').write_text(json.dumps(groups,indent=2));cut=int(px.index.searchsorted('2021'))
    rows=[];configs=[];best=None;all_audits=[]
    for h in (30,60):
        forecasts,fa=predict(frame,groups,px,h);all_audits.append(fa)
        pd.DataFrame({'row_id':frame.row_id,**forecasts}).to_parquet(out/f'predictions_h{h}.parquet',index=False)
        for family,p in forecasts.items():
            for th in (.05,.15):
                for book in ('long','hedged'):
                    code=f'{family}__h{h}__s{th}__{book}';s=construct(frame,p,px,th,book)
                    d,st=safe_run(px.iloc[:cut],s[:cut],h,rf.iloc[:cut])
                    if d is not None:st=metrics(d.loc['2016':'2020'])
                    row={'code':code,'family':family,'hold':h,'cutoff':th,'book':book,**st}
                    rows.append({'stage':'validation',**row});configs.append((row,s))
                    if st.get('sharpe') is not None and st['sharpe']>0 and st['unknown_liquidations']==0 and st['max_gross']<=2:
                        if best is None or st['sharpe']>best['sharpe']:best=row
    text=json.dumps({'selected':best},indent=2);(out/'pretest_selection.json').write_text(text)
    (out/'pretest_selection.sha256').write_text(hashlib.sha256(text.encode()).hexdigest())
    pd.concat(all_audits,ignore_index=True).to_csv(out/'fit_audit.csv',index=False)
    print('PRETEST SELECTION LOCKED',None if best is None else best['code'],flush=True)
    for row,s in configs:
        code=row['code'];h=row['hold'];d,st=safe_run(px.iloc[cut:],s[cut:],h,rf.iloc[cut:])
        rows.append({'stage':'test',**{k:row[k] for k in ('code','family','hold','cutoff','book')},**st})
        print('TEST',code,st.get('sharpe'),flush=True)
        if d is None:continue
        d.to_csv(out/f'{code}.csv')
        if best is not None and best['code']==code:
            a={'base':st,'block63':bootstrap(d.excess),'yearly':{str(y):metrics(g) for y,g in d.groupby(d.index.year)}}
            for key,cost,delay in [('low_cost',Costs(1,.01),1),('high_cost',Costs(10,.10),1),('extra_delay',Costs(),2)]:
                _,a[key]=safe_run(px.iloc[cut:],s[cut:],h,rf.iloc[cut:],cost,delay)
            _,a['sign_reversal']=safe_run(px.iloc[cut:],-s[cut:],h,rf.iloc[cut:])
            (out/'selected_audit.json').write_text(json.dumps(a,indent=2))
    # Fixed recent null: real and permuted models start identical cash-only 2024 books.
    ncut=int(px.index.searchsorted('2024'));nullrows=[]
    for h in (30,60):
        fake,fa=predict(frame,groups,px,h,null=True,start_year=2024)
        fa.to_csv(out/f'null_fit_audit_h{h}.csv',index=False)
        for family,p in fake.items():
            for th in (.05,.15):
                for book in ('long','hedged'):
                    s=construct(frame,p,px,th,book);_,st=safe_run(px.iloc[ncut:],s[ncut:],h,rf.iloc[ncut:])
                    nullrows.append({'variant':'null','family':family,'hold':h,'cutoff':th,'book':book,**st})
        for row,s in configs:
            if row['hold']!=h:continue
            _,st=safe_run(px.iloc[ncut:],s[ncut:],h,rf.iloc[ncut:])
            nullrows.append({'variant':'real',**{k:row[k] for k in ('family','hold','cutoff','book')},**st})
    pd.DataFrame(nullrows).to_csv(out/'recent_real_null.csv',index=False)
    pd.DataFrame(rows).to_csv(out/'all_candidates.csv',index=False)
    (out/'run_summary.json').write_text(json.dumps({'rows':len(frame),'tickers':int(frame.ticker.nunique()),'configs':len(configs),
        'elapsed':time.time()-started,'live_certified':False,'data_coverage':drop,'event_stats':stats,
        'observed_test_sharpe3':[r['code'] for r in rows if r['stage']=='test' and r.get('sharpe') is not None and r['sharpe']>=3]},indent=2))

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--inputs',required=True);a.add_argument('--facts',required=True);a.add_argument('--output',required=True)
    x=a.parse_args();run(x.inputs,x.facts,x.output)
