"""Synthetic fixtures test mechanics, never financial predictability."""
from dataclasses import replace
import numpy as np
import pandas as pd
import pytest
from research.recurrence_downside.engine import (
    Config,METHODS,DISTANCE_COLS,recovery_state,candidate_rows,downside_labels,
    within_date_permutation,state_normalization,transform,recurrence_estimate,
    neighbor_scores,reference_neighbors,direct_predictions)
from research.recurrence_downside.evaluate import choose,evidence_choice,episodes,policy_replay
from research.failure_first.data import make_features


def fixture(n=950,stocks=4):
    rng=np.random.default_rng(731)
    ix=pd.bdate_range('2000-01-03',periods=n)
    bm=pd.Series(100*np.exp(np.cumsum(rng.normal(0,.01,n))),index=ix)
    p=pd.DataFrame(100*np.exp(np.cumsum(rng.normal(0,.015,(n,stocks)),axis=0)),
                   index=ix,columns=[f'S{x}' for x in range(stocks)])
    m=pd.DataFrame(True,index=ix,columns=p.columns)
    f,c,cols=make_features(p,m,bm)
    return p,m,bm,f,cols


def rows(i=100):
    return pd.DataFrame([{'row_id':j//2,'i':i,'ticker':s,'date':'2020-01-01','horizon':h,
          'exit_i':i+h,'fit_i':i-1,**{m:q for m in METHODS}}
          for j,(s,h,q) in enumerate([('A',30,.91),('A',60,.99),('B',30,.92),('B',60,.99)])])

@pytest.mark.parametrize('kwargs',[{'horizons':(29,)},{'min_dates':0},{'neighbors':10},{'candidate_k':0}])
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):Config(**kwargs)

@pytest.mark.parametrize('v,kind,win',[(0.,'flat',0),(-.1,'decline',1),(.1,'rise',0),(np.nan,'unobserved',0)])
def test_endpoint_labels(v,kind,win):
    p,m,bm,f,_=fixture();one=f.iloc[[0]];i=int(one.i.iloc[0]);s=one.ticker.iloc[0]
    p.loc[p.index[i+30],s]=one.reference_price.iloc[0]*(1+v)
    y=downside_labels(one,p,bm,30)
    assert y.outcome_kind.iloc[0]==kind and y.success.iloc[0]==win


def test_recovered_crash_not_decline():
    p,m,bm,f,_=fixture();one=f.iloc[[0]];i=int(one.i.iloc[0]);s=one.ticker.iloc[0];v=one.reference_price.iloc[0]
    p.loc[p.index[i+1],s]=v*.2;p.loc[p.index[i+30],s]=v*1.2
    assert downside_labels(one,p,bm,30).success.iloc[0]==0


def test_pending_is_not_loss_or_success():
    p,m,bm,f,_=fixture();y=downside_labels(f.tail(1),p,bm,756)
    assert y.outcome_kind.iloc[0]=='pending' and np.isnan(y.success.iloc[0])


def test_future_mutation_does_not_change_eligible_features():
    p,m,bm,f,_=fixture(); p.iloc[700:]*=100;bm.iloc[700:]*=.2
    g,_,_=make_features(p,m,bm)
    pd.testing.assert_frame_equal(f[f.i<700].reset_index(drop=True),g[g.i<700].reset_index(drop=True))


def test_candidate_rule_no_label_access_and_determinism():
    *_,f,_=fixture();c=candidate_rows(f,2)
    g=f.assign(success=np.random.default_rng(1).integers(0,2,len(f)))
    assert candidate_rows(g,2).row_id.tolist()==c.row_id.tolist()
    assert candidate_rows(f.sample(frac=1,random_state=2),2).row_id.tolist()==c.row_id.tolist()
    assert c.groupby('i').size().max()<=6


def test_missing_reference_symbol_cannot_change_candidates():
    p,m,bm,f,_=fixture();p['OUTSIDE']=np.geomspace(1,1e5,len(p));m['OUTSIDE']=False
    g,_,_=make_features(p,m,bm)
    pd.testing.assert_frame_equal(candidate_rows(f),candidate_rows(g))


def test_within_date_permutation_preserves_base_rates():
    i=np.repeat(np.arange(20),8);y=np.arange(len(i))%3
    perm=within_date_permutation(i,34)
    assert sorted(perm)==list(range(len(i))) and np.array_equal(i[perm],i)
    for d in set(i):assert sorted(y[i==d])==sorted(y[perm][i==d])

@pytest.mark.parametrize('n,dates',[(31,31),(100,10)])
def test_unsupported_neighborhoods_abstain(n,dates):
    p,_,_=recurrence_estimate(np.ones(n),np.arange(n)%dates,1000,Config())
    assert np.isnan(p)


def test_repeated_stocks_do_not_multiply_date_support():
    cfg=Config();i=np.repeat(np.arange(20),2);y=(i%3>0).astype(float)
    a=recurrence_estimate(y,i,100,cfg)
    b=recurrence_estimate(np.repeat(y,4),np.repeat(i,4),100,cfg)
    np.testing.assert_allclose(a,b)


def test_pseudodates_prevent_perfect_estimate():
    p,n,e=recurrence_estimate(np.ones(128),np.arange(128),1000,Config())
    assert 0<p<1 and n==128 and e<=n


def test_recency_not_cancelled():
    i=np.repeat(np.arange(40)*20,2);y=(i>400).astype(float)
    short=recurrence_estimate(y,i,1000,Config(half_life=50))[0]
    long=recurrence_estimate(y,i,1000,Config(half_life=10000))[0]
    assert short>long


def test_neighbor_outcome_maturity_and_future_mutation():
    *_,f,cols=fixture();ref=f.iloc[:160].copy().reset_index(drop=True)
    y=pd.DataFrame({'success':np.arange(len(f))%2,'matured':True},index=f.index)
    asof=int(ref.i.median())+60;q=f.iloc[400].to_dict();ix=np.arange(len(ref))
    a=neighbor_scores(ix,ref,f,y,q,asof,30,Config())
    bad=y.copy();bad.loc[f.i+30>=asof,'success']=1-bad.loc[f.i+30>=asof,'success']
    b=neighbor_scores(ix,ref,f,bad,q,asof,30,Config())
    for k in a:assert (np.isnan(a[k]) and np.isnan(b[k])) or a[k]==b[k]


def test_cross_era_abstains_without_old_era():
    *_,f,cols=fixture();ref=f.iloc[:200].copy().reset_index(drop=True)
    y=pd.DataFrame({'success':1.,'matured':True},index=f.index)
    scores=neighbor_scores(np.arange(len(ref)),ref,f,y,f.iloc[1].to_dict(),950,30,Config())
    assert np.isnan(scores['cross_era'])


def test_normalization_uses_only_reference_and_is_finite():
    x=np.array([[1,2],[2,2],[3,2]],float);med,s=state_normalization(x)
    assert (s>0).all();z=transform([[1e10,-1e10]],med,s)
    np.testing.assert_allclose(z,[[8,-8]])


def test_reference_tree_excludes_future_features():
    *_,f,_=fixture();c=candidate_rows(f[f.i>=800]);cfg=Config()
    ref,a,meta=reference_neighbors(f,c,800,cfg)
    g=f.copy();g.loc[g.i>=800,list(DISTANCE_COLS)]*=12
    ref2,b,meta2=reference_neighbors(g,c,800,cfg)
    assert ref.i.max()<770 and meta==meta2;np.testing.assert_array_equal(a,b)


def test_choose_shortest_horizon_then_best_stock():
    r=choose(rows(),'consensus',.90,{})
    assert r['ticker']=='B' and r['horizon']==30


def test_busy_and_equality_and_unsupported_abstention():
    assert choose(rows(),'direct',.995,{}) is None
    assert choose(rows(),'direct',.90,{'A':1000,'B':1000}) is None
    r=rows();r['direct']=np.nan;assert choose(r,'direct',.90,{}) is None
    r['direct']=.95;assert choose(r,'direct',.95,{}) is None


def test_choose_ignores_hidden_outcomes():
    r=rows();a=choose(r,'direct',.90,{})
    r['success']=[0,1,0,1];b=choose(r,'direct',.90,{})
    assert (a['ticker'],a['horizon'])==(b['ticker'],b['horizon'])

@pytest.mark.parametrize('field,value',[('fit_i',9999),('exit_i',9999),('horizon',20),('direct',1.1)])
def test_invalid_forecast_rejected(field,value):
    r=rows();r[field]=value
    with pytest.raises(ValueError):choose(r,'direct',.90,{})


def test_evidence_ignores_future_successes():
    hist=[{'i':i*31,'exit_i':i*31+30,'horizon':30,'ticker':str(i),'success':1} for i in range(500)]
    assert evidence_choice({'direct@.95':hist},20,1) is None


def test_episode_overlap_transitive():
    r=pd.DataFrame({'i':[1,20,40,150],'exit_i':[30,50,60,180]})
    assert episodes(r)==2


def test_direct_future_label_attack():
    p,m,bm,f,cols=fixture(1900,6);cfg=Config(n_estimators=5,minimum_training_dates=5,
        minimum_calibration_dates=5,calibration_window=100)
    y=downside_labels(f,p,bm,30);asof=1400;mask=(f.i>=1400)&(f.i<1500)
    a,audit=direct_predictions(f,y,cols,asof,30,cfg,mask)
    bad=y.copy();bad.loc[bad.exit_i>=asof,'success']=0
    b,_=direct_predictions(f,bad,cols,asof,30,cfg,mask)
    np.testing.assert_array_equal(a,b)
    assert audit['train_max_exit_i']<audit['calibration_start_i']<asof


def test_controls_float32_decile_boundary_consistent():
    from research.recurrence_downside.evaluate import controls
    f=pd.DataFrame({'row_id':[0,1],'ticker':['A','B'],'i':[100,100],
        'vol63_rank':np.array([.7,.8],np.float32),'rel63_rank':np.array([.6,.7],np.float32),
        'rel63':[-.1,-.2],'trend_accel':[-.1,-.2],'ma200':[-.1,-.1],'r21':[.1,.1],'r5':[-.1,-.1]})
    y=pd.DataFrame({'row_id':[0,1],'i':[100,100],'horizon':[30,30],'success':[1.,0.]})
    p=f[['row_id','i','vol63_rank','rel63_rank']].assign(horizon=30)
    g=controls(p,y,f,Config())
    assert np.isfinite(g[['random_expected','candidate_expected','volmatched_expected','peer_expected']]).all().all()


def test_completely_empty_policy_reports_undefined_precision():
    from research.recurrence_downside.evaluate import summary
    r=rows();r[list(METHODS)]=np.nan
    y=r[['row_id','i','horizon','exit_i']].copy();y['success']=0.;y['matured']=True;y['resolved']=True
    p,d=policy_replay(r,y,[100],Config())
    assert p.empty and 'policy' in p and not d.issued.any()
    report=summary(p,d)
    assert len(report)==50 and all(x['precision'] is None for x in report)
