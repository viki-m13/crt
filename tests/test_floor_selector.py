import copy
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from research.floor_selector import core as c
from research.floor_selector import data as d


def forecasts():
    return pd.DataFrame([dict(i=300,date=pd.Timestamp('2020-01-02'),ticker=s,
      horizon=h,exit_i=300+h,reference=100.,scale=.15,matured=True,resolved=True,
      up=True,**{'return':.03},z=.2,p_model=p,q05=.1,q50=.3,q95=1.,
      floor_z=.08,agreement=True) for s,h,p in
      [('AAA',30,.94),('AAA',60,.98),('BBB',30,.96),('BBB',60,.99)]] )


class FloorTests(unittest.TestCase):
    def test_unresolved_cannot_be_counted_as_success_even_if_flag_is_true(self):
        f=forecasts();f['resolved']=False;f['up']=True
        self.assertEqual(c.metrics(f,[300],1000)['successes'],0)
        self.assertEqual(c.evidence(f,1000)['successes'],0)

    def test_finish_with_only_abstentions(self):
        f=forecasts();f['agreement']=False
        idx=d.sessions('2018-01-01','2020-01-02')
        p=pd.DataFrame(100.,index=idx,columns=['AAA','BBB'])
        meta={'universe':'test','feed_end':'2020-01-02','price_basis':'unverified',
              'data_quality_certified':False}
        with tempfile.TemporaryDirectory() as td:
            import contextlib,io
            with contextlib.redirect_stdout(io.StringIO()):
                r=c.finish(f,p,p.AAA,meta,td,'2020-01-02')
            self.assertEqual(r['strict_policy']['issued'],0)
            self.assertEqual(json.loads((Path(td)/'scan.json').read_text())['status'],'NO_PICK')

    def test_all_policy_abstentions_are_valid_empty_frames(self):
        f=forecasts();f['agreement']=False
        picks=c.replay(f);strict,audit=c.strict_replay(picks,f)
        self.assertEqual(len(picks),0);self.assertEqual(len(strict),0)
        self.assertEqual(len(audit),1)

    def test_failure_rank_batch_multiple_comparisons(self):
        from research.floor_selector.failure_rank import bh_mask
        np.testing.assert_array_equal(bh_mask([.01,.02,.05,.9]),[True,True,False,False])
        self.assertEqual(len(bh_mask([])),0)

    def test_failure_rank_purges_future_and_counts_ties(self):
        from research.floor_selector.failure_rank import failure_ranks
        now=forecasts().iloc[[0]].copy();now['i']=2000;now['p_model']=.94
        h=pd.concat([now]*1000,ignore_index=True);h['i']=np.arange(1000)
        h['exit_i']=h.i+30;h['up']=True;h['p_model']=.1
        h.loc[:9,'up']=False;h.loc[:9,'p_model']=.94
        result=failure_ranks(h,now)
        self.assertAlmostEqual(result.failure_rank.iloc[0],11/1001)
        future=h.copy();future['exit_i']=3000;future['up']=False;future['p_model']=1.
        mixed=pd.concat([h,future],ignore_index=True)
        pd.testing.assert_frame_equal(failure_ranks(mixed,now),result)

    def test_real_model_fit_predict_and_future_label_purge(self):
        rng=np.random.default_rng(13);n=1800
        frame=pd.DataFrame(rng.normal(size=(n,len(d.FEATURES))),columns=d.FEATURES)
        frame['i']=np.repeat(np.arange(60)*31,30)
        frame['date']=pd.Timestamp('2000-01-01')+pd.to_timedelta(frame.i,unit='D')
        frame['ticker']=['AAA']*n;frame['horizon']=30;frame['exit_i']=frame.i+30
        frame['reference']=100.;frame['scale']=.1;frame['matured']=True
        frame['resolved']=True;frame['z']=rng.normal(size=n)
        frame['return']=np.expm1(frame.z*.1);frame['up']=frame.z>0
        train=c.training_rows(frame,2000)
        model=c.fit(train,2000);out=c.predict(model,frame.tail(5))
        self.assertTrue(out.p_model.between(0,1).all())
        self.assertTrue((out.q05<=out.q50).all())
        self.assertTrue((out.q95>=out.q50).all())
        future=frame.tail(5).copy();future['i']=2100;future['exit_i']=2130
        future['z']=-100.;future['up']=False
        augmented=pd.concat([frame,future],ignore_index=True)
        pd.testing.assert_frame_equal(c.training_rows(augmented,2000),train)

    def test_earliest_qualifying_horizon(self):
        p=c.choose(forecasts(),'p95',{})
        self.assertEqual((p['ticker'],p['horizon']),('AAA',60))
        self.assertNotEqual(p['horizon'],756)

    def test_no_pick_below_cutoff(self):
        f=forecasts();f['p_model']=.94
        self.assertIsNone(c.choose(f,'p95',{}))

    def test_no_pick_when_no_supported_floor(self):
        f=forecasts();f['floor_z']=np.nan
        self.assertIsNone(c.choose(f,'floor',{}))

    def test_median_disagreement_veto(self):
        f=forecasts();f['agreement']=False
        self.assertIsNone(c.choose(f,'p95',{}))

    def test_cooldown_no_repeated_ticker(self):
        p=c.choose(forecasts(),'p95',{'AAA':360})
        self.assertEqual(p['ticker'],'BBB')

    def test_future_outcome_mutation_cannot_change_selection(self):
        a=forecasts();b=a.copy();b['up']=False;b['return']=-.99
        b['resolved']=False;b['z']=-100
        x=c.choose(a,'p95',{});y=c.choose(b,'p95',{})
        for k in ['ticker','horizon','reference','p_model']:
            self.assertEqual(x[k],y[k])

    def test_training_maturity_strict(self):
        f=forecasts();f.loc[0,'exit_i']=300;f.loc[1,'exit_i']=299
        rows=c.training_rows(f,300)
        self.assertEqual(rows.index.tolist(),[1])

    def test_date_weight_decay_survives_normalization(self):
        f=pd.DataFrame({'i':[0,0,1260,1260]})
        w=c.weights(f,1260)
        self.assertAlmostEqual(w[2]/w[0],2.)

    def test_empty_metrics_not_perfect(self):
        m=c.metrics(pd.DataFrame(),[1,2,3],100)
        self.assertIsNone(m['precision'])
        self.assertEqual(m['issued'],0)

    def test_pending_does_not_count_as_success(self):
        f=forecasts();f['matured']=False
        m=c.metrics(f,[300],1000)
        self.assertIsNone(m['precision']);self.assertEqual(m['pending'],4)

    def test_missing_matured_failure(self):
        f=forecasts();f['resolved']=False;f['up']=False;f['return']=np.nan
        m=c.metrics(f,[300],1000)
        self.assertEqual(m['precision'],0.);self.assertEqual(m['unresolved'],4)

    def test_tiny_perfect_record_not_certificate(self):
        rows=[]
        for i in range(10):
            f=forecasts().iloc[[0]].copy();f['i']=i*40;f['exit_i']=i*40+30
            rows.append(f)
        e=c.evidence(pd.concat(rows),1000)
        self.assertFalse(e['pass']);self.assertLess(e['lower_diagnostic'],.95)

    def test_time_thinning_actual_overlap(self):
        f=forecasts();f['i']=[0,20,40,60];f['exit_i']=[30,50,70,90]
        self.assertEqual([r.i for r in c.nonoverlap(f)],[0,40])

    def test_evidence_future_outcomes_not_used(self):
        a=forecasts();b=a.copy();b['up']=False
        self.assertEqual(c.evidence(a,301),c.evidence(b,301))

    def test_stale_scan_refuses_even_with_statistical_pass(self):
        meta={'feed_end':'2020-01-02','price_basis':'split_adjusted_price_only_verified',
              'data_quality_certified':True}
        s=c.scan_status(meta,'2020-02-01','2020-01-02',True)
        self.assertEqual(s['status'],'NO_PICK');self.assertEqual(s['recommendations'],[])

    def test_unverified_price_basis_refuses(self):
        meta={'feed_end':'2020-01-02','price_basis':'adjustment_not_verified',
              'data_quality_certified':True}
        self.assertEqual(c.scan_status(meta,'2020-01-02','2020-01-02',True)['status'],'NO_PICK')

    def test_stale_forecast_refuses_fresh_prices(self):
        meta={'feed_end':'2020-02-01','price_basis':'split_adjusted_price_only_verified',
              'data_quality_certified':True}
        self.assertIn('stale_or_future_dated_forecast',
                      c.scan_status(meta,'2020-02-01','2020-01-02',True)['reasons'])

    def test_no_certificate_means_no_recommendation(self):
        meta={'feed_end':'2020-01-02','price_basis':'split_adjusted_price_only_verified',
              'data_quality_certified':True}
        self.assertEqual(c.scan_status(meta,'2020-01-02','2020-01-02')['status'],'NO_PICK')

    def test_ledger_is_idempotent_and_immutable(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'ledger.jsonl'
            r=dict(model_id='test',as_of='2020-01-01',ticker='AAA',horizon=30,reference=100.)
            c.append_forecast(path,r);c.append_forecast(path,r)
            self.assertEqual(len(path.read_text().splitlines()),1)
            for field,value in [('horizon',60),('reference',99.)]:
                modified=dict(r);modified[field]=value
                with self.assertRaises(ValueError):c.append_forecast(path,modified)

    def test_future_prices_dont_change_features(self):
        rng=np.random.default_rng(731);idx=pd.bdate_range('2010-01-01',periods=360)
        p=pd.DataFrame(100*np.exp(np.cumsum(rng.normal(.0003,.01,(360,5)),axis=0)),
                       index=idx,columns=list('ABCDE'))
        market=p.mean(axis=1);requests=pd.DataFrame({'i':[300]*5,'ticker':list('ABCDE')})
        f,_=d.features(p,requests,market)
        p.iloc[301:]*=17;market.iloc[301:]*=30
        g,_=d.features(p,requests,market)
        pd.testing.assert_frame_equal(f,g)

    def test_endpoint_not_intermediate_touch(self):
        idx=pd.bdate_range('2020-01-01',periods=40)
        p=pd.DataFrame({'AAA':[100.]+[120.]*29+[99.]*10},index=idx)
        f=pd.DataFrame({'i':[0],'ticker':['AAA'],'reference':[100.],'vol63':[.01]})
        t=d.targets(f,p,(30,));self.assertFalse(t.iloc[0].up)
        p.iloc[30]=100.;self.assertFalse(d.targets(f,p,(30,)).iloc[0].up)

    def test_missing_vs_pending_targets(self):
        idx=pd.bdate_range('2020-01-01',periods=40)
        p=pd.DataFrame({'AAA':[100.]*40},index=idx);p.iloc[30]=np.nan
        f=pd.DataFrame({'i':[0,20],'ticker':['AAA']*2,'reference':[100.]*2,'vol63':[.01]*2})
        t=d.targets(f,p,(30,))
        self.assertTrue(t.iloc[0].matured);self.assertFalse(t.iloc[0].resolved)
        self.assertFalse(t.iloc[1].matured);self.assertTrue(np.isnan(t.iloc[1].z))

    def test_invalid_horizons(self):
        for hs in [(29,),(30,30),(30.5,),()]:
            with self.assertRaises(ValueError):d.targets(pd.DataFrame(),pd.DataFrame(),hs)

    def test_real_exchange_closures(self):
        dates=d.sessions('2001-09-10','2001-09-17')
        self.assertEqual([str(x.date()) for x in dates],['2001-09-10','2001-09-17'])

    def test_duplicate_symbols_fail(self):
        p=pd.DataFrame([[1,2]],columns=['BRK.B','BRK-B'])
        with self.assertRaises(ValueError):d.clean_prices(p)

    def test_json_rejects_nonfinite_numbers_cleanly(self):
        s=c.json_clean({'a':np.nan,'b':np.inf,'c':np.float64(.1)})
        self.assertEqual(json.loads(json.dumps(s,allow_nan=False)),{'a':None,'b':None,'c':.1})

if __name__=='__main__':unittest.main()
