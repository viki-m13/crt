"""Independent-data replication and preregistered netted-inventory cadence ablation."""
import argparse,json,time,hashlib
from pathlib import Path
import numpy as np,pandas as pd
from research.edge3.data import load_bonds
from research.edge3.signals import factor_signals,twin_signals,fund_signals,TWIN_PAIRS,FUND_PAIRS
from research.edge3.account import Costs,simulate,metrics,bootstrap

def safe(px,s,h,c,rf,costs=Costs(),delay=1):
    try:
        d=simulate(px,s,hold=h,cadence=c,costs=costs,rf_returns=rf,delay=delay)
        return d,metrics(d)
    except RuntimeError as e:return None,{'sharpe':None,'bankrupt':True,'error':str(e),'unknown_liquidations':None}

def main(root,out):
    out=Path(out);out.mkdir(exist_ok=True,parents=True);ts=time.time()
    px,m,rf,cov=load_bonds(root);pd.DataFrame(cov).to_csv(out/'coverage.csv',index=False)
    px.to_parquet(out/'prices.parquet');rf.to_csv(out/'actual_rf.csv')
    risk=[];r=px.pct_change(fill_method=None)
    for i,j in np.argwhere((abs(r.to_numpy())>1)&m):risk.append({'date':str(px.index[i].date()),'ticker':px.columns[j],'return':float(r.iloc[i,j])})
    pd.DataFrame(risk).to_csv(out/'unresolved_large_returns.csv',index=False)
    signals={};audits=[]
    for c in (5,1):
        f,a=factor_signals(px,m,cadence=c);audits+=a
        twins,_=twin_signals(px,cadence=c);funds,tracking=fund_signals(px,cadence=c)
        for name,s in f.items():signals[(name,c)]=s
        for name,s in (twins|funds).items():signals[(name,c)]=s
    pd.DataFrame(audits).to_csv(out/'factor_fit_audit.csv',index=False)
    tracking.to_csv(out/'tracking_diagnostic.csv',index=False)
    valend=px.index.searchsorted('2021');rows=[];choices={};runs=[]
    for (name,c),s in signals.items():
        for h in ((1,5,30,60) if c==5 else (5,30)):
            code=f'{name}__c{c}__h{h}';print('VALID',code,flush=True)
            d,stat=safe(px.iloc[:valend],s[:valend],h,c,rf.iloc[:valend])
            if d is not None:stat=metrics(d.loc['2016':'2020'])
            group='twin' if name.startswith('twin') else 'fund' if name.startswith('fund') else 'factor'
            row={'code':code,'group':group,'method':name,'cadence':c,'hold':h,'primary_30plus':h>=30,**stat}
            rows.append({'stage':'validation',**row});runs.append((code,name,c,h,s))
            if h>=30 and stat.get('sharpe') is not None and not stat['unknown_liquidations'] and stat['max_gross']<=2:
                if group not in choices or stat['sharpe']>choices[group]['sharpe']:choices[group]=row
    pl=json.dumps(choices,indent=2);(out/'pretest_selection.json').write_text(pl);(out/'pretest_selection.sha256').write_text(hashlib.sha256(pl.encode()).hexdigest())
    print('PRETEST CHOICES LOCKED',[(g,r['code']) for g,r in choices.items()],flush=True)
    cut=valend;pxt=px.iloc[cut:];rft=rf.iloc[cut:]
    for code,name,c,h,s in runs:
        st=s[cut:];d,stat=safe(pxt,st,h,c,rft)
        rows.append({'stage':'test','code':code,'method':name,'cadence':c,'hold':h,'primary_30plus':h>=30,**stat})
        print('TEST',code,stat.get('sharpe'),flush=True)
        if d is None:continue
        d.to_csv(out/f'{code}.csv')
        if any(r['code']==code for r in choices.values()):
            audit={'base':stat,'bootstrap_63':bootstrap(d.excess),'yearly':{str(y):metrics(g) for y,g in d.groupby(d.index.year)}}
            for n,cost,delay in [('low_cost',Costs(1,.01),1),('high_cost',Costs(10,.10),1),('extra_delay',Costs(),2),('gross_zero_carry',Costs(0,0,0,0,1),1)]:
                _,audit[n]=safe(pxt,st,h,c,rft if n!='gross_zero_carry' else np.zeros(len(rft)),cost,delay)
            _,audit['inverse_signal']=safe(pxt,-st,h,c,rft)
            # Ten fixed random within-active-support sign/assignment controls.
            null=[]
            for seed in range(10):
                rng=np.random.default_rng(1200+seed);fake=st.copy()
                for t in np.flatnonzero(np.any(st!=0,axis=1)):
                    idx=np.flatnonzero(st[t]!=0);fake[t,idx]=rng.permutation(st[t,idx])
                _,x=safe(pxt,fake,h,c,rft);null.append(x.get('sharpe'))
            audit['random_assignment_sharpes']=null
            (out/f'{code}_audit.json').write_text(json.dumps(audit,indent=2))
    pd.DataFrame(rows).to_csv(out/'all_candidates.csv',index=False)
    (out/'run_summary.json').write_text(json.dumps({'elapsed_seconds':time.time()-ts,'tested_configurations':len(runs),'pretest_choices':choices,'qualified_live':False,'sources':'export_hashes.json','observed_test_sharpe3':[r['code'] for r in rows if r['stage']=='test' and r['hold']>=30 and r.get('sharpe') is not None and r['sharpe']>=3],'target_met':False,'acceptance_blockers':['Borrow/financing/slippage are scenarios rather than executed quotes','Corporate actions and historical identities incomplete','No prospective independent market evidence']},indent=2))
    print('COMPLETE',time.time()-ts,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--inputs',required=True);p.add_argument('--output',required=True);a=p.parse_args();main(a.inputs,a.output)
