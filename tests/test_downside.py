"""Deterministic validation fixtures are not evidence of market predictability."""
import copy,json
import numpy as np
import pandas as pd
import pytest
from research.downside.model import (Config,METHODS,outcome_classes,labels_for_horizon,
    strict_weighted_cdf,combine,CompetingOutcomes,fit_fold)
from research.downside.features import make_features
from research.downside.policy import (choose,replay,controls,metrics,nonoverlap,evidence_score)
from research.failure_first.archive import checked_prices,sessions
from research.failure_first.model import split_masks,date_weights,fit_head


def fixture(n=1800,names=4):
    rng=np.random.default_rng(663)
    idx=pd.bdate_range('2001-01-01',periods=n)
    rm=rng.normal(.0002,.009,n)
    bm=pd.Series(100*np.exp(np.cumsum(rm)),index=idx)
    arr=rng.normal(.0001,.017,(n,names))+.5*rm[:,None]
    p=pd.DataFrame(100*np.exp(np.cumsum(arr,axis=0)),index=idx,columns=[f'S{k}' for k in range(names)])
    m=pd.DataFrame(True,index=idx,columns=p.columns)
    return p,m,bm

@pytest.fixture(scope='module')
def data():
    p,m,b=fixture();f,c,cols=make_features(p,m,b);return p,m,b,f,c,cols

@pytest.mark.parametrize('r,b,k',[(.1,-.1,0),(0.,-.1,0),(-.01,-.1,2),(-.01,.1,3),(-.2,.1,1),(-.3,-.1,1),(-.1,np.nan,3)])
def test_observed_classes(r,b,k):
    assert outcome_classes([r],[b],[True],[True])[0]==k

@pytest.mark.parametrize('m,k',[(True,4),(False,-1)])
def test_missing_is_never_decline(m,k):
    assert outcome_classes([np.nan],[0.],[False],[m])[0]==k

@pytest.mark.parametrize('r,k',[(np.nan,True),(.1,False)])
def test_invalid_known_outcome(r,k):
    with pytest.raises(ValueError):outcome_classes([r],[0],[True],[k])

@pytest.mark.parametrize('h',[1,29,0,-30])
def test_minimum_horizon(h):
    with pytest.raises(ValueError):Config(horizons=(h,))


def test_partition_exhaustive_and_pending(data):
    p,m,b,f,_,_=data;y=labels_for_horizon(f,p,b,60)
    d=y.loc[y.matured]
    assert (d.down+d.up+d.flat+d.unknown==1).all()
    assert (d.down==(d['class'].isin([1,2,3])).astype(int)).all()
    assert y.loc[~y.matured,'down'].isna().all()
    assert (y.loc[~y.matured,'class']==-1).all()

@pytest.mark.parametrize('return_value,expect',[(0.,0),(.2,0),(-.2,1)])
def test_endpoint_not_touch(data,return_value,expect):
    p,_,b,f,_,_=data;p=p.copy();row=f.iloc[[0]].copy();i=int(row.i.iloc[0]);s=row.ticker.iloc[0];start=row.reference_price.iloc[0]
    p.loc[p.index[i+1],s]=start*.1
    p.loc[p.index[i+30],s]=start*(1+return_value)
    y=labels_for_horizon(row,p,b,30)
    assert y.down.iloc[0]==expect
    assert y.exit_i.iloc[0]==i+30


def test_future_endpoint_nan_preserved(data):
    p,_,b,f,_,_=data;p=p.copy();row=f.iloc[[0]].copy();i=int(row.i.iloc[0]);s=row.ticker.iloc[0]
    p.loc[p.index[i+30],s]=np.nan;y=labels_for_horizon(row,p,b,30)
    assert y.unknown.iloc[0] and y.down.iloc[0]==0 and y.matured.iloc[0]


def test_positive_jump_then_down_is_down(data):
    p,_,b,f,_,_=data;p=p.copy();row=f.iloc[[0]].copy();i=int(row.i.iloc[0]);s=row.ticker.iloc[0];ref=row.reference_price.iloc[0]
    p.loc[p.index[i+1],s]=ref*2;p.loc[p.index[i+30],s]=ref*.9
    y=labels_for_horizon(row,p,b,30)
    assert y.down.iloc[0]==1 and y.squeeze_proxy.iloc[0]==1


def test_persistence_requires_every_checkpoint(data):
    p,_,b,f,_,_=data;p=p.copy();row=f.iloc[[0]].copy();i=int(row.i.iloc[0]);s=row.ticker.iloc[0];ref=row.reference_price.iloc[0]
    p.loc[p.index[i+15],s]=ref*1.01;p.loc[p.index[i+22],s]=ref*.9;p.loc[p.index[i+30],s]=ref*.8
    y=labels_for_horizon(row,p,b,30);assert y.down.iloc[0]==1 and y.persistent_down.iloc[0]==0
    p.loc[p.index[i+15],s]=ref*.99;y=labels_for_horizon(row,p,b,30);assert y.persistent_down.iloc[0]==1


def test_feature_prefix_invariant(data):
    p,m,b,f,c,cols=data;i=1200;q=p.copy();q.iloc[i+1:]*=14;bm=b.copy();bm.iloc[i+1:]*=.2
    other,_,_=make_features(q,m,bm)
    pd.testing.assert_frame_equal(f.loc[f.i<=i],other.loc[other.i<=i])


def test_outside_members_cannot_change_state(data):
    p,m,b,_,_,_=data;p=p.copy();m=m.copy();m.iloc[:,-1]=False
    a,_,_=make_features(p,m,b);p.iloc[:,-1]=np.geomspace(1,1e6,len(p));z,_,_=make_features(p,m,b)
    pd.testing.assert_frame_equal(a,z)


def test_future_membership_cannot_change_past(data):
    p,m,b,f,_,_=data;m=m.copy();m.iloc[1200:]=False;g,_,_=make_features(p,m,b)
    pd.testing.assert_frame_equal(f.loc[f.i<1200],g.loc[g.i<1200])

@pytest.mark.parametrize('h',[30,126,252])
def test_full_horizon_purging(data,h):
    p,m,b,f,_,_=data;y=labels_for_horizon(f,p,b,h);tr,ca,start=split_masks(f,y,1500,h,Config())
    assert y.loc[tr,'exit_i'].max()<f.loc[ca,'i'].min()
    assert y.loc[ca,'exit_i'].max()<1500


def test_recency_does_not_cancel():
    w=date_weights([0,0,100,100],100,100);assert w[2]>w[0] and w[1]==w[0]


def test_cdf_strict_equality():
    np.testing.assert_allclose(strict_weighted_cdf([-1,0,1],[1,2,1],[-1,0,.1,2]),[0,.25,.75,1])

@pytest.mark.parametrize('v,w,q',[([],[],[0]),([np.nan],[1],[0]),([0],[-1],[0]),([0],[1],[np.nan])])
def test_cdf_bad_input(v,w,q):
    with pytest.raises(ValueError):strict_weighted_cdf(v,w,q)


def test_multiclass_unknown_exclusion():
    rng=np.random.default_rng(99);x=rng.normal(size=(1000,3));y=np.arange(1000)%5
    cfg=Config(n_estimators=5);model=CompetingOutcomes().fit(x,y,np.arange(1000)//10,1000,cfg)
    p=model.predict(x[:30]);np.testing.assert_allclose(p.sum(axis=1),1)
    assert np.all(p[:,1:4].sum(axis=1)<1-p[:,4]+1e-14)


def test_multiclass_single_unknown_class():
    model=CompetingOutcomes().fit(np.ones((30,2)),np.full(30,4),np.arange(30),40,Config())
    p=model.predict(np.ones((2,2)));assert p[0,4]>.95 and p[0,1:4].sum()<.05


def test_consensus_not_independent_votes():
    np.testing.assert_allclose(combine([.9],[.95],[.7]),[.7])
    with pytest.raises(ValueError):combine([1.2],[.9],[.9])


def forecast_rows():
    rows=[]
    for s in ['A','B']:
        for h in [30,60]:
            row=dict(row_id=0 if s=='A' else 1,i=1000,date='2020-01-01',ticker=s,horizon=h,exit_i=1000+h,fit_i=999,
                reference_price=10.,vol63=.02,vol63_rank=.5,rel63_rank=.5,regime='bull',gate_rebound=True,gate_squeeze_veto=True)
            for method in METHODS:row[method]=.96 if h==30 else .99
            rows.append(row)
    return pd.DataFrame(rows)


def test_earliest_horizon_not_hindsight_max():
    row=choose(forecast_rows(),'direct',.95,{})
    assert row['horizon']==30 and row['ticker']=='A'


def test_no_forced_pick():
    assert choose(forecast_rows(),'direct',.999,{}) is None


def test_locked_ticker_and_deadline():
    f=forecast_rows();p=choose(f,'direct',.95,{'A':1030});assert p['ticker']=='B'
    assert choose(f,'direct',.95,{'A':1030,'B':1030}) is None

@pytest.mark.parametrize('method,gate',[('rebound','gate_rebound'),('squeeze_veto','gate_squeeze_veto')])
def test_explicit_veto(method,gate):
    f=forecast_rows();f[gate]=False;assert choose(f,method,.95,{}) is None


def test_labels_cannot_affect_selection():
    f=forecast_rows();a=choose(f,'direct',.95,{})
    f['down']=[0,1,1,0];f['return']=[100,-100,.9,-.9]
    assert choose(f,'direct',.95,{})==a

@pytest.mark.parametrize('error',['future_fit','bad_horizon','duplicate','future_deadline'])
def test_reject_invalid_forecasts(error):
    f=forecast_rows()
    if error=='future_fit':f['fit_i']=1001
    if error=='bad_horizon':f['horizon']=29;f['exit_i']=1029
    if error=='duplicate':f=pd.concat([f,f])
    if error=='future_deadline':f['exit_i']+=1
    with pytest.raises(ValueError):choose(f,'direct',.95,{})


def test_small_perfect_history_not_certified():
    history=[dict(i=k*100,exit_i=k*100+30,ticker='A',horizon=30,down=1) for k in range(10)]
    assert evidence_score(history,2000,1,56)==0.
    # Future successes must never help the gate.
    history += [dict(i=3000+k*100,exit_i=3030+k*100,ticker='A',horizon=30,down=1) for k in range(300)]
    assert evidence_score(history,2000,1,56)==0.


def test_correct_next_reference_not_up_complement(data):
    p,_,b,f,_,_=data;p=p.copy();row=f.iloc[[0]];i=int(row.i.iloc[0]);s=row.ticker.iloc[0];ref=row.reference_price.iloc[0]
    p.loc[p.index[i+1],s]=ref*.8;p.loc[p.index[i+30],s]=ref*.9
    y=labels_for_horizon(row,p,b,30);assert y.down.iloc[0]==1 and y.next_close_down.iloc[0]==0


def test_unobserved_is_failed_prediction_in_metrics():
    p=forecast_rows().iloc[[0]].copy();p['policy']='direct@0.95';p['estimated_down']=.96
    for key,val in dict(down=0.,unknown=True,resolved=False,up=0.,flat=False,next_close_down=0.,
        random_expected=.4,peer_expected=.4,weakest_momentum_down=0.,benchmark_down=0.,max_return=np.nan).items():p[key]=val
    p['return']=np.nan
    d=pd.DataFrame([dict(policy='direct@0.95',i=1000,issued=True,available=True)])
    r=metrics(p,d,1100)[0];assert r['precision']==0 and r['unknown']==1 and r['precision_if_all_unknown_down']==1


def test_empty_precision_not_perfect():
    d=pd.DataFrame([dict(policy='adaptive95',i=1000,issued=False,available=True)])
    assert metrics(pd.DataFrame(),d,1100)[0]['precision'] is None


def test_exchange_calendar_no_holiday_padding():
    idx=sessions('2001-09-01','2001-09-20')
    assert not pd.Timestamp('2001-09-11') in idx and not pd.Timestamp('2001-09-14') in idx
    assert pd.Timestamp('2001-09-17') in idx


def test_planted_negative_signal_detectable_and_complement():
    rng=np.random.default_rng(332);x=rng.normal(size=(3000,3));y=(x[:,0]<0).astype(int)
    c=Config(n_estimators=30);i=np.arange(len(x))//10
    a=fit_head(x,y,i,300,c);b=fit_head(x,1-y,i,300,c)
    qa=a.predict_proba(x)[:,1];qb=b.predict_proba(x)[:,1]
    assert ((qa>.5)==y).mean()>.95
    np.testing.assert_allclose(qa+qb,1,atol=1e-8)


def test_full_fold_future_label_attack_and_repeat(data):
    p,m,b,f,_,cols=data;y=labels_for_horizon(f,p,b,30)
    cfg=Config(n_estimators=5,minimum_training_dates=10,minimum_calibration_dates=10)
    test=(f.i>=1500)&(f.i<1550)
    a,audit=fit_fold(f,y,cols,1500,30,cfg,test)
    assert len(a)>0 and audit['status']=='fitted'
    altered=y.copy();future=y.exit_i>=1500
    altered.loc[future,['down','persistent_down','squeeze_proxy']]=1
    altered.loc[future,'class']=1;altered.loc[future,'return']=-.99
    b,_=fit_fold(f,altered,cols,1500,30,cfg,test)
    pd.testing.assert_frame_equal(a,b)
