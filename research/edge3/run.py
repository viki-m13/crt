"""Hypotheses -> self-financing accounts -> pretest selection -> complete report."""
import argparse,json,hashlib,time
from pathlib import Path
from dataclasses import asdict
import numpy as np,pandas as pd
from research.edge3.data import load
from research.edge3.signals import factor_signals,twin_signals,fund_signals
from research.edge3.account import Costs,simulate,metrics,bootstrap

def main(root,out):
    out=Path(out);out.mkdir(exist_ok=True,parents=True)
    start=time.time();allrows=[];selected=[];bundles={}
    for uni in ['sp500','ndx','structural']:
        print('LOAD',uni,flush=True);px,mem,meta=load(root,uni)
        if uni=='structural':
            from research.edge3.signals import TWIN_PAIRS,FUND_PAIRS
            cols=sorted(set(x for a,b,*_ in TWIN_PAIRS+FUND_PAIRS for x in (a,b))&set(px.columns))
            px=px[cols];twins,details=twin_signals(px);funds,diag=fund_signals(px);signals=twins|funds
            pd.DataFrame(details).to_csv(out/'twin_inputs.csv',index=False)
            diag.to_csv(out/'tracking_arithmetic_vs_log.csv',index=False)
        else:
            signals,audit=factor_signals(px,mem)
            pd.DataFrame(audit).to_csv(out/f'{uni}_factor_fit_audit.csv',index=False)
        np.savez_compressed(out/f'{uni}_signals.npz',**signals)
        (out/f'{uni}_data.json').write_text(json.dumps(meta,indent=2));px.to_parquet(out/f'{uni}_prices.parquet')
        bundles[uni]=(px,signals)
        vstart='2019' if uni=='ndx' else '2016';stop=px.index.searchsorted('2021-01-01');local=[]
        for name,s in signals.items():
            for h in (1,5,30,60):
                code=f'{uni}__{name}__h{h}'
                try:
                    d=simulate(px.iloc[:stop],s[:stop],hold=h,costs=Costs())
                    value=metrics(d.loc[vstart:'2020'])
                except RuntimeError as exc:
                    value={'sharpe':None,'bankrupt':True,'error':str(exc),'unknown_liquidations':None,'max_gross':None}
                row={'code':code,'universe':uni,'method':name,'hold':h,'primary_30plus':h>=30,**value}
                allrows.append({'stage':'validation',**row});local.append(row)
                print('VALID',code,round(row.get('sharpe') or 0,3),'unknown',row.get('unknown_liquidations'),flush=True)
        eligible=[r for r in local if r['hold']>=30 and r.get('sharpe') is not None and not r['unknown_liquidations'] and r['max_gross']<=2]
        best=max(eligible,key=lambda r:r['sharpe']) if eligible else None
        selected.append({'universe':uni,'choice':best,'choice_is_validated_edge':False})
    payload=json.dumps(selected,indent=2);(out/'pretest_selection.json').write_text(payload)
    (out/'pretest_selection.sha256').write_text(hashlib.sha256(payload.encode()).hexdigest())
    print('PRETEST CHOICES LOCKED',[(x['universe'],x['choice']['code'] if x['choice'] else None) for x in selected],flush=True)
    for uni,(px,signals) in bundles.items():
        cut=px.index.searchsorted('2021-01-01');testpx=px.iloc[cut:]
        choices=[x['choice'] for x in selected if x['universe']==uni and x['choice']]
        for name,s in signals.items():
            for h in (1,5,30,60):
                code=f'{uni}__{name}__h{h}'
                try:d=simulate(testpx,s[cut:],hold=h,costs=Costs())
                except RuntimeError as exc:
                    allrows.append({'stage':'test','code':code,'universe':uni,'method':name,'hold':h,'primary_30plus':h>=30,'sharpe':None,'bankrupt':True,'error':str(exc)})
                    continue
                d.to_csv(out/f'{code}_test.csv')
                row={'code':code,'universe':uni,'method':name,'hold':h,'primary_30plus':h>=30,**metrics(d)}
                allrows.append({'stage':'test',**row})
                print('TEST',code,round(row.get('sharpe') or 0,3),'unknown',row.get('unknown_liquidations'),flush=True)
                if choices and choices[0]['code']==code:
                    diagnostics={'base':metrics(d),'bootstrap63':bootstrap(d.excess),'yearly':{str(y):metrics(g) for y,g in d.groupby(d.index.year)}}
                    for label,cost,delay in [('borrow1',Costs(borrow_apr=.01),1),('borrow10',Costs(borrow_apr=.10),1),('cost10',Costs(side_bps=10),1),('rf0',Costs(rf_apr=0),1),('rf5',Costs(rf_apr=.05),1),('extra_delay',Costs(),2)]:
                        diagnostics[label]=metrics(simulate(testpx,s[cut:],hold=h,costs=cost,delay=delay))
                    diagnostics['sign_reversed']=metrics(simulate(testpx,-s[cut:],hold=h,costs=Costs()))
                    (out/f'{uni}_selected_audit.json').write_text(json.dumps(diagnostics,indent=2))
    pd.DataFrame(allrows).to_csv(out/'all_candidates.csv',index=False)
    (out/'run_summary.json').write_text(json.dumps({'elapsed_seconds':time.time()-start,'costs':asdict(Costs()),'selection':selected,'target_met':False},indent=2))
    print('COMPLETE',time.time()-start,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--inputs',required=True);p.add_argument('--output',required=True);a=p.parse_args();main(a.inputs,a.output)
