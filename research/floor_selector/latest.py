"""Latest available session ranking. Off-cadence inference is research-only."""
import argparse
from pathlib import Path
import pandas as pd
from . import core
from .data import features,load,sessions,targets


def latest(root,universe,replay_directory,out,today):
    p,requests,market,meta=load(Path(root),universe)
    raw=pd.read_parquet(Path(replay_directory)/'raw_forecasts.parquet')
    # Rebuild only past-eligible features and fit the same annual training cutoff
    # as the latest completed replay. No update may ingest a pending label.
    historical,_=features(p,requests,market)
    frame=targets(historical,p,core.HORIZONS)
    boundary=int(raw.fit_boundary.max())
    model=core.fit(core.training_rows(frame,boundary),boundary)
    if model is None:raise ValueError('No trained annual model available')
    current=requests.loc[requests.i.eq(requests.i.max())].copy();current['i']=len(p)-1
    f,_=features(p,current,market);now=core.predict(model,targets(f,p,core.HORIZONS))
    history=raw.loc[raw.exit_i.lt(len(p)-1)&raw.matured&raw.i.ge(len(p)-1-10*252)]
    now['floor_z']=float('nan')
    for h in core.HORIZONS:
        penalty=core.correction(history,h)
        if penalty is not None:
            mask=now.horizon.eq(h);now.loc[mask,'floor_z']=now.loc[mask,'q05']-penalty
    result=core.make_scan(meta,today,now,pd.DataFrame(),pd.DataFrame())
    result['reasons'].append('latest_session_ranking_not_validated_as_daily_selection_policy')
    result['validation_scope']='Monthly replay; latest-session research ranking is separate'
    calendar=sessions(p.index[0],p.index[-1]+pd.DateOffset(years=4))
    for r in result['research_candidates_not_recommendations']:
        r['evaluation_date']=str(calendar[len(p)-1+r['horizon']].date())
        for point in r['checkpoint_fan']:
            point['evaluation_date']=str(calendar[len(p)-1+point['horizon']].date())
    core.write_json(out,result)
    return result

if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--inputs',type=Path,required=True);a.add_argument('--replay',type=Path,required=True)
    a.add_argument('--universe',choices=['sp500','ndx'],default='sp500')
    a.add_argument('--out',type=Path,required=True)
    a.add_argument('--today',default=str(pd.Timestamp.now(tz='UTC').date()))
    x=a.parse_args();latest(x.inputs,x.universe,x.replay,x.out,x.today)
