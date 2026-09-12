import numpy as np
import pandas as pd
import pytest
from research.sharpe3_edge.data import clean
from research.sharpe3_edge.signals import features,labels,fit_horizons,horizon_map,proposals
from research.sharpe3_edge.portfolio import Settings,capped_weights,simulate
from research.sharpe3_edge.metrics import describe,sharpe


def fixture(n=650,k=8):
    rng=np.random.default_rng(43);idx=pd.bdate_range('2001-01-01',periods=n)
    bm=pd.Series(100*np.exp(np.cumsum(rng.normal(.0002,.009,n))),index=idx)
    p=pd.DataFrame(100*np.exp(np.cumsum(rng.normal(.0003,.012,(n,k)),axis=0)),index=idx,columns=[f'T{i}' for i in range(k)])
    mem=pd.DataFrame(True,index=idx,columns=p.columns)
    return p,mem,bm


def plain_frames(p,step=5):
    return {i:pd.DataFrame({'i':i,'ticker':p.columns,'vol':.01,'beta':1.,'family':'information',
                 'strength':1.,'e63':.2,'ma200':.1,'z63':1.,'z21':1.,'volume_rank':.8,'vol_rank':.2})
            for i in range(252,len(p),step)}


def test_future_feature_mutation():
    p,m,b=fixture();f,_=features(p,m,b);q=p.copy();q.iloc[470:]*=7;m2=m.copy();m2.iloc[470:]=False;b2=b.copy();b2.iloc[470:]*=.2
    g,_=features(q,m2,b2)
    pd.testing.assert_frame_equal(f[f.i<470].reset_index(drop=True),g[g.i<470].reset_index(drop=True))


def test_nonmember_cannot_move_scores():
    p,m,b=fixture();m['T7']=False;f,_=features(p,m,b);p['T7']=np.geomspace(1,1e8,len(p));g,_=features(p,m,b)
    pd.testing.assert_frame_equal(f,g)

@pytest.mark.parametrize('h',[30,60,90,126,180,252])
def test_horizon_entry_endpoint(h):
    p,m,b=fixture();f,_=features(p,m,b);y=labels(f,p,b,h);r=y.iloc[0];c=p.columns.get_loc(f.ticker.iloc[0])
    assert r.entry_i==f.i.iloc[0]+1 and r.exit_i-r.entry_i==h
    assert r.ret==pytest.approx(p.iloc[int(r.exit_i),c]/p.iloc[int(r.entry_i),c]-1)


def test_unknown_is_writeoff_not_drop_and_pending_distinct():
    p,m,b=fixture();f,_=features(p,m,b);q=p.copy();col=p.columns.get_loc(f.ticker.iloc[0]);q.iloc[260,col]=np.nan
    y=labels(f,q,b,60);assert y.iloc[0].ret==-1 and not y.iloc[0].complete
    assert y.iloc[-1].matured==False and np.isnan(y.iloc[-1].ret)


def test_training_future_labels_invisible():
    p,m,b=fixture(1300);f,_=features(p,m,b);ls={h:labels(f,p,b,h) for h in (30,60)}
    s=fit_horizons(f,ls,1000);corrupt={}
    for h,x in ls.items():
        x=x.copy();x.loc[x.exit_i>=1000,'excess']=10;corrupt[h]=x
    pd.testing.assert_frame_equal(s,fit_horizons(f,corrupt,1000))
    if not s.empty:assert (s.max_exit_i<1000).all()

@pytest.mark.parametrize('k',[1,2,3,4,5])
def test_caps_and_sparse_cash(k):
    w=capped_weights(np.geomspace(.001,.9,k));assert w.sum()<=k/5+1e-12;assert max(w)<=.3
    assert w.sum()==pytest.approx(k/5)

@pytest.mark.parametrize('hedge',[False,True])
@pytest.mark.parametrize('lag',[1,2,5])
def test_accounting_hold_and_causality(hedge,lag):
    p,m,b=fixture();fr=plain_frames(p);cfg=Settings(entry_lag=lag,hedge=hedge)
    d,tr,o=simulate(p,b,fr,'ipd_fixed60',252,cfg=cfg)
    assert not tr.empty;assert (tr.exit_i-tr.entry_i>=30).all();assert (tr.entry_i-tr.signal_i==lag).all()
    assert d.nav.iloc[-1]==pytest.approx(1+tr.net_pnl.sum())
    if not hedge:assert (d.cash>=-1e-12).all() and d.gross.max()<=1+1e-12
    q=p.copy();q.iloc[480:]*=8;b2=b.copy();b2.iloc[480:]*=3
    z,_,_=simulate(q,b2,fr,'ipd_fixed60',252,cfg=cfg)
    pd.testing.assert_frame_equal(d[d.i<480],z[z.i<480])


def test_missing_position_stays_zero_even_recovered():
    p,m,b=fixture();fr=plain_frames(p);d,t,o=simulate(p,b,fr,'ipd_fixed60',252)
    selected=t.iloc[0];q=p.copy();q.iloc[int(selected.entry_i)+3,int(selected.col)]=np.nan
    d2,t2,_=simulate(q,b,fr,'ipd_fixed60',252);bad=t2.loc[t2.id==selected.id].iloc[0]
    assert bad.written_off and bad.exit_price==0 and bad.exit_i==selected.exit_i


def test_no_successful_future_entry_replacement():
    p,m,b=fixture();fr=plain_frames(p);p.iloc[253,:]=np.nan
    _,tr,o=simulate(p,b,fr,'ipd_fixed60',252)
    assert len(o[o.status=='unavailable_entry'])==5
    assert not (tr.entry_i==253).any()


def test_constant_market_exact_fee_conservation():
    p,m,b=fixture(320);p[:]=100.;b[:]=100.;fr={252:plain_frames(p)[252]}
    d,t,_=simulate(p,b,fr,'ipd_fixed60',252,cfg=Settings(fee_bps=25))
    assert d.nav.iloc[-1]==pytest.approx(5/6+(1/6)*(.9975/1.0025))
    assert all(t.h==60) and t.closed.all()


def test_zero_sharpe_undefined():
    assert sharpe(np.zeros(200)) is None


def test_zero_candidate_stays_in_cash():
    p,m,b=fixture();fr={i:x.assign(family='none') for i,x in plain_frames(p).items()}
    d,t,o=simulate(p,b,fr,'ipd_fixed60',252)
    assert t.empty and np.allclose(d.nav,1) and (d.positions==0).all();assert len(o)>0


def test_higher_cost_worse_constant_price():
    p,m,b=fixture();p[:]=100.;b[:]=100.;fr=plain_frames(p)
    low=simulate(p,b,fr,'ipd_fixed60',252,cfg=Settings(fee_bps=10))[0]
    high=simulate(p,b,fr,'ipd_fixed60',252,cfg=Settings(fee_bps=50))[0]
    assert high.nav.iloc[-1]<low.nav.iloc[-1]


def test_validation_errors():
    p,m,b=fixture()
    with pytest.raises(ValueError):clean(p.iloc[::-1])
    with pytest.raises(ValueError):Settings(entry_lag=0)
    with pytest.raises(ValueError):capped_weights([0])
    with pytest.raises(ValueError):labels(pd.DataFrame(),p,b,29)
    m.iloc[0,0]=None
    with pytest.raises(ValueError):features(p,m,b)


def test_rank_tie_is_deterministic():
    p,m,b=fixture();f=plain_frames(p)[252]
    a=proposals(f,'ipd_fixed60');z=proposals(f.sample(frac=1,random_state=2),'ipd_fixed60')
    assert a.ticker.tolist()==z.ticker.tolist()


def test_adaptive_abstains_when_no_positive_supported_payoff():
    s=pd.DataFrame([dict(family='information',h=30,dates=30,blocks=15,lower=-.01,all_eras_positive=True)])
    assert not horizon_map(s)


def test_shortest_tie_and_era_veto():
    s=pd.DataFrame([dict(family='information',h=h,dates=30,blocks=15,lower=.02,all_eras_positive=False) for h in (60,30)])
    assert horizon_map(s)['information']==30
    assert not horizon_map(s,strict=True)


def test_extended_features_future_invariance():
    from research.sharpe3_edge.extensions import enrich
    p,m,b=fixture();f,_=features(p,m,b);g=enrich(f,p,b);p.iloc[500:]*=8;b.iloc[500:]*=.2
    q=enrich(f,p,b);pd.testing.assert_frame_equal(g[g.i<500],q[q.i<500])


def test_work_conserving_preserves_duration_and_accounting():
    p,m,b=fixture();fr=plain_frames(p)
    d,t,o=simulate(p,b,fr,'ipd_fixed60',252,cfg=Settings(work_conserving=True))
    assert (t.exit_i-t.entry_i==60).all();assert d.nav.iloc[-1]==pytest.approx(1+t.net_pnl.sum())
    assert not (d.cash< -1e-12).any()


def test_work_conserving_does_not_change_disabled_default():
    p,m,b=fixture();fr=plain_frames(p)
    a=simulate(p,b,fr,'ipd_fixed60',252)[0]
    z=simulate(p,b,fr,'ipd_fixed60',252,cfg=Settings(work_conserving=False))[0]
    pd.testing.assert_frame_equal(a,z)


def test_breadth_lag_only_uses_available_origin():
    from research.sharpe3_edge.extensions import enrich
    p,m,b=fixture();f,_=features(p,m,b);g=enrich(f,p,b)
    first=g[g.i==g.i.min()];assert first.breadth_change.isna().all()
    i=g.i.unique()[10];now=g[g.i==i].iloc[0];old=f[f.i<=i-21].iloc[-1]
    assert now.breadth_change==pytest.approx(now.breadth-old.breadth)

@pytest.mark.parametrize('hedge',[False,True])
@pytest.mark.parametrize('work',[False,True])
def test_independent_price_reconstruction(hedge,work):
    from research.sharpe3_edge.validate import reconcile
    p,m,b=fixture();fr=plain_frames(p);cfg=Settings(hedge=hedge,work_conserving=work)
    d,tr,o=simulate(p,b,fr,'ipd_fixed60',252,cfg=cfg)
    nav,n=reconcile(p,b,tr,252,len(p)-1)
    np.testing.assert_allclose(nav,d.nav,rtol=0,atol=1e-12);assert n==len(tr)


def test_independent_reconstruction_catches_quantity_corruption():
    from research.sharpe3_edge.validate import reconcile
    p,m,b=fixture();d,tr,_=simulate(p,b,plain_frames(p),'ipd_fixed60',252)
    tr.loc[0,'qty']*=1.1
    with pytest.raises(AssertionError):reconcile(p,b,tr,252,len(p)-1)
