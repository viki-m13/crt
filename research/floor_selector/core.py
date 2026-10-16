"""Joint horizon forecasts with downside-first selection. Research, not a guarantee."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from scipy.stats import beta
from .data import FEATURES, features, load, sessions, targets

HORIZONS=(30,60,90,126,180,252,504,756)
POLICIES={'rank_one':(0.,False),'p80':(.8,False),'p90':(.9,False),
 'p95':(.95,False),'p975':(.975,False),'p99':(.99,False),
 'floor':(0.,True),'floor_p95':(.95,True)}
PARAMS=dict(n_estimators=100,num_leaves=15,max_depth=5,min_child_samples=100,
 learning_rate=.05,reg_lambda=10.,verbosity=-1,n_jobs=2,random_state=20260909,
 deterministic=True,force_col_wise=True)
OUTPUT=['i','date','ticker','horizon','exit_i','reference','scale','matured',
 'resolved','return','up','z','market_ma_gap','breadth']


def weights(frame,boundary):
    counts=frame.groupby('i')['i'].transform('size').to_numpy()
    w=np.exp2(-(boundary-frame.i.to_numpy())/1260)/counts
    return w/w.mean()


def training_rows(frame,boundary):
    return frame.loc[frame.exit_i.lt(boundary)&frame.i.ge(boundary-12*252)
                     &frame.matured].copy()


def fit(train,boundary):
    if len(train)<1500 or train.i.nunique()<24 or train.up.nunique()<2:
        return None
    x=train[FEATURES];w=weights(train,boundary)
    classifier=LGBMClassifier(**PARAMS).fit(x,train.up.astype(int),sample_weight=w)
    regressors=[]
    for q in (.05,.50,.95):
        model=LGBMRegressor(objective='quantile',alpha=q,**PARAMS)
        # Winsorization is a training stabilizer only, not a scored return edit.
        model.fit(x,train.z.clip(-100,100),sample_weight=w)
        regressors.append(model)
    return classifier,regressors


def predict(models,frame):
    classifier,regressors=models;x=frame[FEATURES];out=frame[OUTPUT].copy()
    out['p_model']=classifier.predict_proba(x)[:,1]
    q=np.column_stack([model.predict(x) for model in regressors])
    out['q05']=np.minimum(q[:,0],q[:,1]);out['q50']=q[:,1]
    out['q95']=np.maximum(q[:,2],q[:,1])  # crossing repair expands the band
    out['agreement']=out.q50.gt(0)
    return out


def walk_forward(frame,first_year=2013):
    output=[]
    for year in sorted(frame.date.dt.year.unique()):
        if year<first_year:continue
        test=frame.loc[frame.date.dt.year.eq(year)];boundary=int(test.i.min())
        train=training_rows(frame,boundary);models=fit(train,boundary)
        if models is None:continue
        out=predict(models,test);out['fit_boundary']=boundary
        out['max_training_exit']=int(train.exit_i.max());output.append(out)
        print(f'{year}: train={len(train):,}; forecasts={len(test):,}',flush=True)
    if not output:raise ValueError('Insufficient walk-forward training history')
    return pd.concat(output,ignore_index=True)


def correction(history,horizon):
    """Empirical global AND upper-score-tail error penalty; no coverage guarantee."""
    d=history.loc[history.horizon.eq(horizon)]
    if d.i.nunique()<24:return None
    global_shift=max(0.,float((d.q05-d.z).quantile(.95)))
    tail=d.loc[d.p_model.ge(.8)]
    if tail.i.nunique()<12:return None
    tail_shift=max(0.,float((tail.q05-tail.z).quantile(.95)))
    return max(global_shift,tail_shift)


def calibrate(raw):
    output=[]
    for i,now in raw.groupby('i',sort=True):
        history=raw.loc[raw.exit_i.lt(i)&raw.i.ge(i-10*252)&raw.matured]
        now=now.copy();now['floor_z']=np.nan
        for h in sorted(now.horizon.unique()):
            shift=correction(history,int(h))
            if shift is not None:
                mask=now.horizon.eq(h)
                now.loc[mask,'floor_z']=now.loc[mask,'q05']-shift
        output.append(now)
    return pd.concat(output,ignore_index=True)


def choose(now,name,open_until):
    """A complete causal stock/horizon/ranking policy; no future outcome access."""
    threshold,require_floor=POLICIES[name]
    d=now.loc[now.p_model.ge(threshold)&now.agreement].copy()
    d=d.loc[[int(r.i)>open_until.get(r.ticker,-1) for r in d.itertuples()]]
    if require_floor:d=d.loc[d.floor_z.gt(0)]
    if d.empty:return None
    d=d.sort_values(['ticker','horizon'],kind='stable').drop_duplicates('ticker')
    d['rank']=d.floor_z*d.scale/np.sqrt(d.horizon) if require_floor else d.p_model
    return d.sort_values(['rank','horizon','ticker'],ascending=[False,True,True],
                         kind='stable').iloc[0].to_dict()


def replay(predictions):
    rows=[];calendars={name:{} for name in POLICIES}
    for _,now in predictions.groupby('i',sort=True):
        for name in POLICIES:
            pick=choose(now,name,calendars[name])
            if pick is not None:
                pick['policy']=name;rows.append(pick)
                calendars[name][pick['ticker']]=int(pick['exit_i'])
    out=pd.DataFrame(rows,columns=list(predictions.columns)+['rank','policy'])
    out['date']=pd.to_datetime(out['date'])
    for flag in ['matured','resolved','up','agreement']:out[flag]=out[flag].astype(bool)
    return out


def nonoverlap(frame):
    """Time thinning does NOT establish financial independence/stationarity."""
    rows=[];next_i=-1
    for r in frame.sort_values(['i','ticker'],kind='stable').itertuples():
        if r.i>next_i:rows.append(r);next_i=r.exit_i
    return rows


def evidence(frame,asof_i,family_size=len(POLICIES)):
    d=frame.loc[frame.exit_i.lt(asof_i)&frame.matured].copy()
    d['up']=d.resolved.fillna(False)&d.up.fillna(False)
    n=len(d);success=int(d.up.sum());thin=nonoverlap(d)
    k=sum(bool(r.up) for r in thin);nt=len(thin)
    lower=float(beta.ppf(.05/family_size,k,nt-k+1)) if k else 0.
    result=dict(n=n,successes=success,precision=success/n if n else None,
      nonoverlap_n=nt,nonoverlap_successes=k,lower_diagnostic=lower)
    result['pass']=bool(n>=60 and nt>=20 and success/n>.95 and lower>.95)
    result['interval_status']='time_thinned_binomial_diagnostic_not_a_market_guarantee'
    return result


def strict_replay(picks,predictions):
    rows=[];audit=[];open_until={}
    for i,now in predictions.groupby('i',sort=True):
        admissible=[]
        for name in POLICIES:
            hist=picks.loc[picks.policy.eq(name)];e=evidence(hist,int(i))
            if e['pass']:
                current=picks.loc[picks.policy.eq(name)&picks.i.eq(i)]
                p=current.iloc[0].to_dict() if len(current) else None
                if p is not None and int(i)<=open_until.get(p['ticker'],-1):
                    p=None
                if p is not None:
                    p.update(policy=name,prior_precision=e['precision'],
                             prior_lower=e['lower_diagnostic']);admissible.append(p)
        if admissible:
            best=sorted(admissible,key=lambda p:(p['horizon'],-p['prior_lower'],
                         -p['p_model'],p['ticker'],p['policy']))[0]
            rows.append(best);open_until[best['ticker']]=int(best['exit_i'])
        audit.append(dict(i=int(i),qualifying_policies=len(admissible)))
    return pd.DataFrame(rows,columns=list(picks.columns)+['prior_precision','prior_lower']),audit


def metrics(picks,all_dates,cutoff,market=None):
    n=len(picks)
    if n==0:return dict(issued=0,matured=0,precision=None,coverage=0.,pending=0)
    d=picks.loc[picks.matured].copy()
    d['up']=d.resolved.fillna(False)&d.up.fillna(False)
    resolved=d.loc[d.resolved]
    active=set(picks.i.astype(int));run=longest=0
    for i in sorted(all_dates):
        run=0 if i in active else run+1;longest=max(longest,run)
    result=dict(issued=n,matured=len(d),successes=int(d.up.sum()),
      failures=int((~d.up).sum()),unresolved=int((~d.resolved).sum()),
      pending=int((~picks.matured).sum()),precision=float(d.up.mean()) if len(d) else None,
      coverage=n/len(all_dates),unique_tickers=picks.ticker.nunique(),
      longest_no_pick_decisions=longest,
      horizon_counts={str(k):int(v) for k,v in picks.horizon.value_counts().items()},
      observed_return_median=float(resolved['return'].median()) if len(resolved) else None,
      observed_return_p05=float(resolved['return'].quantile(.05)) if len(resolved) else None,
      observed_worst_return=float(resolved['return'].min()) if len(resolved) else None,
      positive_after_20bps=float((d['return'].fillna(-1)>.002).mean()) if len(d) else None,
      precision_by_issue_year={str(y):float(g.up.mean()) for y,g in d.groupby(d.date.dt.year)},
      evidence=evidence(picks,cutoff+1))
    if market is not None and len(d):
        m=market.to_numpy();b=m[d.exit_i.astype(int)]/m[d.i.astype(int)]-1
        result['same_date_horizon_spy_positive_rate']=float((b>0).mean())
    return result


def json_clean(v):
    if isinstance(v,dict):return {str(k):json_clean(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [json_clean(x) for x in v]
    if isinstance(v,np.integer):return int(v)
    if isinstance(v,(np.floating,float)):return float(v) if np.isfinite(v) else None
    if isinstance(v,np.bool_):return bool(v)
    if isinstance(v,(pd.Timestamp,np.datetime64)):return str(pd.Timestamp(v).date())
    return v


def write_json(path,value):
    Path(path).write_text(json.dumps(json_clean(value),indent=2,allow_nan=False)+'\n')


def append_forecast(path,record):
    """An issued identity cannot be edited, including its reference and deadline."""
    import fcntl
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);record=json_clean(record)
    identity='|'.join(str(record[k]) for k in ['model_id','as_of','ticker'])
    record=dict(record,forecast_id=hashlib.sha256(identity.encode()).hexdigest())
    with path.open('a+',encoding='utf-8') as f:
        fcntl.flock(f,fcntl.LOCK_EX);f.seek(0)
        for line in f:
            old=json.loads(line)
            if old['forecast_id']==record['forecast_id']:
                if old!=record:raise ValueError('Immutable forecast conflict')
                return old
        f.seek(0,2);f.write(json.dumps(record,allow_nan=False)+'\n');f.flush()
    return record


def scan_status(meta,today,latest_forecast_date,certificate_pass=False):
    reasons=[]
    if meta.get('price_basis')!='split_adjusted_price_only_verified':
        reasons.append('unverified_price_only_target')
    if not meta.get('data_quality_certified',False):
        reasons.append('uncertified_data_and_terminal_actions')
    age=(pd.Timestamp(today).normalize()-pd.Timestamp(meta['feed_end'])).days
    if age<0 or age>4:reasons.append('stale_or_future_dated_prices')
    fa=(pd.Timestamp(today).normalize()-pd.Timestamp(latest_forecast_date)).days
    if fa<0 or fa>4:reasons.append('stale_or_future_dated_forecast')
    if not certificate_pass:reasons.append('no_validated_95_percent_policy')
    return dict(status='NO_PICK' if reasons else 'ELIGIBLE_FOR_REVIEW',reasons=reasons,
      as_of=str(pd.Timestamp(today).date()),data_as_of=meta['feed_end'],recommendations=[])


def make_scan(meta,today,latest,picks,strict):
    # A live claim needs evidence for the COMPLETE adaptive policy, not merely
    # for one component bucket. No substitute second-best stock on cooldown.
    candidate=choose(latest,'rank_one',{})
    i=int(latest.i.max())
    if len(strict):
        valid=evidence(strict,i)
        current=strict.loc[strict.i.eq(i)]
        authorized=current.iloc[0].to_dict() if valid['pass'] and len(current) else None
    else:
        authorized=None
    status=scan_status(meta,today,latest.date.max(),authorized is not None)
    status['research_candidates_not_recommendations']=[]
    if candidate:
        fan=latest.loc[latest.ticker.eq(candidate['ticker'])].sort_values('horizon')
        c=dict(ticker=candidate['ticker'],as_of=candidate['date'],
          horizon=int(candidate['horizon']),reference=candidate['reference'],
          p_model=candidate['p_model'],label='UNVALIDATED_RESEARCH_ONLY',
          price_basis=meta['price_basis'],model_id='return-floor-v1',
          band_label='raw_nominal_90_percent_checkpoint_band_not_validated',
          checkpoint_fan=[dict(horizon=int(r.horizon),
            low_ratio=float(np.exp(np.clip(r.q05*r.scale,-20,20))),
            median_ratio=float(np.exp(np.clip(r.q50*r.scale,-20,20))),
            high_ratio=float(np.exp(np.clip(r.q95*r.scale,-20,20)))) for r in fan.itertuples()])
        status['research_candidates_not_recommendations'].append(c)
    if status['status']=='ELIGIBLE_FOR_REVIEW':
        status['recommendations']=[{k:authorized[k] for k in
          ['ticker','date','reference','horizon','p_model']}]
    return status


def finish(raw,p,market,meta,outdir,today):
    outdir=Path(outdir);outdir.mkdir(parents=True,exist_ok=True)
    evaluated=calibrate(raw);evaluated.to_parquet(outdir/'calibrated_forecasts.parquet',index=False)
    picks=replay(evaluated)
    calendar=sessions(p.index[0],p.index[-1]+pd.DateOffset(years=4))
    picks['evaluation_date']=pd.Series(index=picks.index,dtype='datetime64[ns]')
    picks['worst_close_dip']=np.nan
    if len(picks):
        picks['evaluation_date']=calendar[picks.exit_i.astype(int)].to_numpy()
        drawdowns=[]
        for row in picks.itertuples():
            path=p[row.ticker].iloc[row.i:row.exit_i+1]
            complete=row.matured and path.notna().all()
            drawdowns.append(float(path.min()/row.reference-1) if complete else np.nan)
        picks['worst_close_dip']=drawdowns
    picks.to_csv(outdir/'shadow_picks.csv',index=False)
    strict,audit=strict_replay(picks,evaluated);strict.to_csv(outdir/'strict_picks.csv',index=False)
    report=dict(meta=meta,forecasts=len(raw),dates=raw.i.nunique(),
      historical_only=True,goal_achieved=False,policies={},
      horizons=list(HORIZONS),eligible_stock_dates=len(raw[['i','ticker']].drop_duplicates()))
    for period,start in [('all_prequential',2013),('post2019',2019)]:
        dates=sorted(evaluated.loc[evaluated.date.dt.year.ge(start),'i'].unique())
        report['policies'][period]={name:metrics(picks.loc[picks.policy.eq(name)&
          picks.date.dt.year.ge(start)],dates,len(p)-1,market) for name in POLICIES}
    report['strict_policy']=metrics(strict,sorted(raw.i.unique()),len(p)-1,market)
    valid=evaluated.loc[evaluated.resolved]
    report['checkpoint_diagnostics']=dict(n=len(valid),
      median_log_mae=float(((valid.q50-valid.z)*valid.scale).abs().mean()),
      flat_log_mae=float((valid.z*valid.scale).abs().mean()),
      nominal_90_band_coverage=float(((valid.z>=valid.q05)&(valid.z<=valid.q95)).mean()))
    for period,start in [('all_prequential',2013),('post2019',2019)]:
        eligible=len(raw.loc[raw.date.dt.year.ge(start),['i','ticker']].drop_duplicates())
        for name,v in report['policies'][period].items():
            v['eligible_stock_dates']=eligible
            v['stock_date_coverage']=v['issued']/eligible if eligible else 0.
            v['coverage_unit']='fraction_of_monthly_decisions'
            group=picks.loc[picks.policy.eq(name)&picks.date.dt.year.ge(start)]
            observed=group.loc[group.matured,'worst_close_dip'].dropna()
            v['observed_worst_interim_close_dip']=float(observed.min()) if len(observed) else None
    report['checkpoint_diagnostics']['unresolved_matured']=int((evaluated.matured&~evaluated.resolved).sum())
    report['checkpoint_diagnostics']['pessimistic_nominal_90_coverage']=float(((valid.z>=valid.q05)&(valid.z<=valid.q95)).sum()/evaluated.matured.sum())
    report['code_sha256']={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in ['core.py','data.py']}
    write_json(outdir/'report.json',report)
    latest=evaluated.loc[evaluated.i.eq(evaluated.i.max())]
    scan=make_scan(meta,today,latest,picks,strict)
    for record in scan['research_candidates_not_recommendations']:
        record['evaluation_date']=str(calendar[int(latest.i.max())+record['horizon']].date())
        for point in record['checkpoint_fan']:
            point['evaluation_date']=str(calendar[int(latest.i.max())+point['horizon']].date())
    write_json(outdir/'scan.json',scan)
    for c in scan['research_candidates_not_recommendations']:
        append_forecast(outdir/'research_ledger.jsonl',c)
    print(json.dumps(json_clean(report['policies']['post2019']),indent=2),flush=True)
    return report


def run(root,universe,outdir,today):
    outdir=Path(outdir);outdir.mkdir(parents=True,exist_ok=True)
    p,requests,market,meta=load(Path(root),universe)
    f,c=features(p,requests,market);c.to_csv(outdir/'coverage.csv',index=False)
    frame=targets(f,p,HORIZONS);raw=walk_forward(frame)
    raw.to_parquet(outdir/'raw_forecasts.parquet',index=False)
    return finish(raw,p,market,meta,outdir,today)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--inputs',type=Path,required=True)
    ap.add_argument('--universe',choices=['sp500','ndx'],default='sp500')
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--today',default=str(pd.Timestamp.now(tz='UTC').date()))
    a=ap.parse_args();run(a.inputs,a.universe,a.out,a.today)
