"""Synthetic fixtures test correctness, not the market's predictability."""
import json
import numpy as np
import pandas as pd
import pytest
from dataclasses import replace
from research.failure_first.model import (Config,CHANNELS,aggregate_risk,date_weights,
    partition_failures,label_panel,split_masks,fit_fold,fit_head,Calibrator)
from research.failure_first.data import make_features
from research.failure_first.evaluate import (select,nonoverlap,cp_lower,historical_screen,
    joint_policy,add_controls,clean_json,block_bootstrap)


def market_fixture(n=1100,stocks=5):
    rng=np.random.default_rng(25)
    ix=pd.bdate_range('2000-01-03',periods=n)
    market=pd.Series(100*np.exp(np.cumsum(rng.normal(.0003,.008,n))),index=ix)
    p=pd.DataFrame(100*np.exp(np.cumsum(rng.normal(.0003,.015,(n,stocks)),axis=0)),index=ix,
                   columns=[f'T{k}' for k in range(stocks)])
    m=pd.DataFrame(True,index=ix,columns=p.columns)
    return p,m,market


def risk_rows(i=100):
    return pd.DataFrame([{'row_id':k//2,'i':i,'date':'2020-01-01','ticker':s,'horizon':h,
        'exit_i':i+h,'fit_i':i-10,'vol63_rank':.5,'vol63':.02,
        **{x:q for x in ('binary','failure_sum','failure_veto','failure_veto_path','naive_channels')}}
        for k,(s,h,q) in enumerate([('A',30,.04),('A',60,.01),('B',30,.03),('B',60,.02)])])

@pytest.mark.parametrize('h',[(0,),(29,),(30.,),(60,30),(30,30),()])
def test_bad_horizons(h):
    with pytest.raises(ValueError): Config(horizons=h)

@pytest.mark.parametrize('t',[(0.,),(1.,),(-1.,),()])
def test_bad_thresholds(t):
    with pytest.raises(ValueError):Config(thresholds=t)


def test_partition_is_exhaustive_and_price_only_not_causal():
    r=np.array([.01,np.nan,-.3,-.01,-.02,-.01,0.,np.nan])
    b=np.array([-.1,0,.1,-.1,.1,.1,.1,0]);peak=np.array([.1]*8)
    resolved=np.isfinite(r);matured=np.array([True]*7+[False])
    c=partition_failures(r,b,peak,resolved,matured)
    assert c.tolist()==[0,1,2,3,4,4,4,-1]
    c=partition_failures(np.array([-.01]),np.array([np.nan]),np.array([np.nan]),[True],[True])
    assert c[0]==5


def test_pending_never_labeled_success_or_failure():
    assert partition_failures([np.nan],[np.nan],[np.nan],[False],[False])[0]==-1
    with pytest.raises(ValueError):partition_failures([.1],[0],[.1],[True],[False])


def test_individual_small_risks_do_not_imply_small_union():
    summed,naive=aggregate_risk(np.full((1,5),.03))
    assert naive[0]<.05 and summed[0]>.05
    assert np.isclose(summed[0],.15)

@pytest.mark.parametrize('bad',[np.zeros((2,4)),np.full((2,5),np.nan),np.full((2,5),-1),np.full((2,5),1.1)])
def test_reject_malformed_risk(bad):
    with pytest.raises(ValueError):aggregate_risk(bad)


def test_conservative_sum_clipped_not_multiplied():
    a,b=aggregate_risk(np.full((1,5),.4))
    assert a[0]==1 and b[0]==.4


def test_date_weights_balance_stocks_but_keep_recency():
    w=date_weights([0,0,100],100,100)
    assert np.isclose(w[0],w[1])
    assert np.isclose(w[:2].sum()*2,w[2])


def test_future_prices_cannot_change_past_features():
    p,m,b=market_fixture(); f,_,cols=make_features(p,m,b)
    q=p.copy();q.iloc[650:]*=17
    c=b.copy();c.iloc[650:]*=.01
    g,_,cols2=make_features(q,m,c)
    pd.testing.assert_frame_equal(f[f.i<650].reset_index(drop=True),g[g.i<650].reset_index(drop=True))
    assert cols==cols2


def test_nonmembers_cannot_change_cross_section():
    p,m,b=market_fixture();m['T4']=False
    f,_,_=make_features(p,m,b);p['T4']=np.geomspace(1,1e9,len(p))
    g,_,_=make_features(p,m,b)
    pd.testing.assert_frame_equal(f,g)


def test_no_future_endpoint_availability_selection():
    p,m,b=market_fixture();f,_,_=make_features(p,m,b)
    q=p.copy();q.loc[q.index[700]:,'T0']=np.nan
    g,_,_=make_features(q,m,b)
    pd.testing.assert_frame_equal(f[f.i<700].reset_index(drop=True),g[g.i<700].reset_index(drop=True))


def test_label_endpoint_not_intervening_touch():
    p,m,b=market_fixture(); f,_,_=make_features(p,m,b);f=f.iloc[[0]]
    i=int(f.i.iloc[0]);s=f.ticker.iloc[0];entry=f.reference_price.iloc[0]
    p.loc[p.index[i+1],s]=entry*1.3;p.loc[p.index[i+30],s]=entry*.99
    y=label_panel(f,p,b,30)
    assert y.success.iloc[0]==0 and y.exit_i.iloc[0]==i+30


def test_recovered_crash_is_endpoint_success_but_path_failure():
    p,m,b=market_fixture();f,_,_=make_features(p,m,b);f=f.iloc[[0]]
    i=int(f.i.iloc[0]);s=f.ticker.iloc[0];v=f.reference_price.iloc[0]
    p.loc[p.index[i+1],s]=v*.6;p.loc[p.index[i+30],s]=v*1.01
    y=label_panel(f,p,b,30)
    assert y.success.iloc[0]==1 and y.failure_class.iloc[0]==0 and y.path_failure.iloc[0]==1


def test_missing_endpoint_versus_pending_and_flat():
    p,m,b=market_fixture();f,_,_=make_features(p,m,b)
    row=f.iloc[[0]];i=int(row.i.iloc[0]);s=row.ticker.iloc[0]
    p.loc[p.index[i+30],s]=np.nan;y=label_panel(row,p,b,30)
    assert y.success.iloc[0]==0 and y.failure_class.iloc[0]==1
    y=label_panel(f.iloc[[-1]],p,b,30)
    assert np.isnan(y.success.iloc[0]) and y.failure_class.iloc[0]==-1
    p.loc[p.index[i+30],s]=row.reference_price.iloc[0]
    assert label_panel(row,p,b,30).success.iloc[0]==0


def test_reference_mismatch_raises():
    p,m,b=market_fixture();f,_,_=make_features(p,m,b); f=f.iloc[[0]].copy()
    f.reference_price*=2
    with pytest.raises(ValueError):label_panel(f,p,b,30)


def test_next_close_has_distinct_entry():
    p,m,b=market_fixture();f,_,_=make_features(p,m,b);f=f.iloc[[0]]
    i=int(f.i.iloc[0]);s=f.ticker.iloc[0];v=f.reference_price.iloc[0]
    p.loc[p.index[i+1],s]=v*1.2;p.loc[p.index[i+30],s]=v*1.1
    y=label_panel(f,p,b,30)
    assert y.success.iloc[0]==1 and y.next_entry_success.iloc[0]==0


def test_exact_train_calibration_and_outer_purge():
    p,m,b=market_fixture();f,_,_=make_features(p,m,b);y=label_panel(f,p,b,60)
    tr,ca,start=split_masks(f,y,900,60,Config())
    assert (y.loc[tr,'exit_i']<start).all()
    assert (y.loc[ca,'exit_i']<900).all()
    assert set(f.loc[tr,'i']).isdisjoint(set(f.loc[ca,'i']))
    assert not (tr & ~y.matured).any()


def test_shortest_horizon_then_stock_ranking():
    today=risk_rows();r=select(today,'failure_veto',.95,{})
    assert r['ticker']=='B' and r['horizon']==30
    r=select(today,'failure_veto',.95,{'B':130})
    assert r['ticker']=='A' and r['horizon']==30 # Not A60 with lower risk.


def test_ticker_blocked_through_endpoint():
    assert select(risk_rows(),'failure_veto',.95,{'A':100,'B':200}) is None
    assert select(risk_rows(101),'failure_veto',.95,{'A':100,'B':200})['ticker']=='A'


def test_no_quota_or_threshold_lowering():
    assert select(risk_rows(),'failure_veto',.999,{}) is None


def test_shuffle_candidate_order_is_invariant():
    a=risk_rows();r=select(a,'failure_veto',.95,{})
    assert select(a.sample(frac=1,random_state=3),'failure_veto',.95,{})==r


def test_future_trained_model_rejected():
    t=risk_rows();t.fit_i=200
    with pytest.raises(ValueError):select(t,'failure_veto',.95,{})


def test_deadline_change_rejected():
    t=risk_rows();t.exit_i+=1
    with pytest.raises(ValueError):select(t,'failure_veto',.95,{})


def test_duplicate_estimates_rejected():
    t=risk_rows()
    with pytest.raises(ValueError):select(pd.concat([t,t]),'failure_veto',.95,{})


def test_nonoverlap_crosses_calendar_bucket_boundaries():
    d=pd.DataFrame({'i':[59,61,150],'exit_i':[119,121,180], 'ticker':['A','B','C'],'horizon':[60,60,30]})
    assert nonoverlap(d).ticker.tolist()==['A','C']


def test_small_perfect_sample_not_95_certificate():
    assert cp_lower(5,5)<.95
    assert cp_lower(65,79)<.95
    assert cp_lower(0,20)==0


def test_evidence_ignores_future_outcomes():
    d=pd.DataFrame({'i':np.arange(25)*40,'exit_i':np.arange(25)*40+30,
                    'ticker':[f'T{x}' for x in range(25)],'horizon':30,'success':1.})
    assert not historical_screen(d,50,1)
    z=d.copy();z.loc[z.exit_i>=50,'success']=0
    assert historical_screen(d,50,1)==historical_screen(z,50,1)


def test_json_no_nan_or_numpy_scalars():
    assert json.dumps(clean_json({'x':np.nan,'n':np.int64(3)}),allow_nan=False)=='{"x": null, "n": 3}'


def test_planted_signal_learnable_and_label_inversion_same_information():
    rng=np.random.default_rng(741);x=rng.normal(size=(4000,3));y=(x[:,0]>0).astype(int)
    i=np.repeat(np.arange(40)*20,100);cfg=replace(Config(),n_estimators=30,threads=1)
    a=fit_head(x,y,i,1000,cfg);c=fit_head(x,1-y,i,1000,cfg)
    q=rng.normal(size=(1000,3));p=a.predict_proba(q)[:,1];r=c.predict_proba(q)[:,1]
    assert ((p>.5)==(q[:,0]>0)).mean()>.97
    np.testing.assert_allclose(p,1-r,atol=1e-7)


def test_shuffled_label_positive_control_loses_discrimination():
    rng=np.random.default_rng(741);x=rng.normal(size=(4000,3));y=(x[:,0]>0).astype(int);rng.shuffle(y)
    i=np.repeat(np.arange(40)*20,100);cfg=replace(Config(),n_estimators=30,threads=1)
    a=fit_head(x,y,i,1000,cfg);q=rng.normal(size=(1000,3));p=a.predict_proba(q)[:,1]
    # Test against independent null labels. A chance weak split can correlate
    # with a deterministic planted boundary; that is not an IID null test.
    null_y=rng.integers(0,2,len(q))
    assert .44<((p>.5)==null_y).mean()<.56
    assert (p>.95).sum()==0


def test_future_label_mutation_cannot_change_model_forecasts():
    p,m,b=market_fixture(stocks=8);f,_,cols=make_features(p,m,b);y=label_panel(f,p,b,30)
    cfg=replace(Config(),n_estimators=8,minimum_training_dates=10,threads=1)
    mask=(f.i>=900)&(f.i<925)
    a,audit=fit_fold(f,y,cols,900,30,cfg,mask)
    assert len(a)>0
    z=y.copy();z.loc[z.exit_i>=900,'failure_class']=2;z.loc[z.exit_i>=900,'path_failure']=1
    c,_=fit_fold(f,z,cols,900,30,cfg,mask)
    pd.testing.assert_frame_equal(a,c)


def test_calibrator_never_declares_absent_class_zero_risk():
    c=Calibrator().fit(np.full(100,.01),np.zeros(100),np.repeat(np.arange(10),10))
    assert c.predict([.01])[0]>0


def test_empty_bootstrap_is_undefined_not_perfect():
    assert block_bootstrap(pd.DataFrame(),[1,2]) is None


def outcome_rows(risks):
    d=risks[['row_id','i','horizon','exit_i']].copy()
    d['matured']=True;d['resolved']=True;d['success']=1.
    return d


def test_joint_policy_current_future_label_does_not_choose_stock():
    r=risk_rows();y=outcome_rows(r);cfg=Config()
    a,_=joint_policy(r,y,[100],cfg)
    y['success']=0.;b,_=joint_policy(r,y,[100],cfg)
    cols=['i','ticker','horizon','policy','estimated_success']
    pd.testing.assert_frame_equal(a[cols],b[cols])


def test_joint_policy_misaligned_endpoint_is_rejected():
    r=risk_rows();y=outcome_rows(r);y.exit_i+=1
    with pytest.raises(ValueError):joint_policy(r,y,[100],Config())


def test_joint_policy_duplicate_outcome_is_rejected():
    r=risk_rows();y=outcome_rows(r)
    with pytest.raises(ValueError):joint_policy(r,pd.concat([y,y]),[100],Config())


def test_exact_random_control_not_one_lucky_random_draw():
    r=risk_rows();y=outcome_rows(r);y.loc[y.row_id==1,'success']=0
    picks,_=joint_policy(r,y,[100],Config())
    f=pd.DataFrame({'row_id':[0,1],'ticker':['A','B'],'vol63':[.01,.02],'vol63_rank':[.5,.5]})
    controls=add_controls(picks,y,f)
    assert (controls.random_expected==.5).all()
    assert (controls.volmatched_expected==.5).all()
    assert (controls.lowvol_success==1).all()


def test_zero_picks_has_undefined_accuracy():
    from research.failure_first.evaluate import summarize
    ds=pd.DataFrame({'policy':['none','none'],'i':[100,105],'issued':[False,False],'model_available':[True,True]})
    row=summarize(pd.DataFrame(),ds,200)[0]
    assert row['precision'] is None and row['wins']==0 and row['longest_no_pick_decisions']==2


def test_real_exchange_calendar_does_not_count_911_closure_as_sessions():
    from research.failure_first.archive import sessions
    idx=sessions('2001-09-10','2001-09-18')
    assert pd.Timestamp('2001-09-11') not in idx
    assert pd.Timestamp('2001-09-14') not in idx
    assert pd.Timestamp('2001-09-17') in idx
