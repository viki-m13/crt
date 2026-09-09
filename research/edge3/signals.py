"""Economic residual hypotheses; coefficients estimated exclusively from prior returns."""
import numpy as np
import pandas as pd
from scipy.linalg import eigh
from threadpoolctl import threadpool_limits

TWIN_PAIRS=(('GOOG','GOOGL','2014-04-03'),('FOX','FOXA','2019-03-19'),('NWS','NWSA','2013-07-01'),('Z','ZG','2015-08-17'),('UA','UAA','2016-04-08'),('DISCA','DISCK','2008-09-18'),('LBTYA','LBTYK','2005-09-08'),('BRK-A','BRK.B','2000-01-03'))
FUND_PAIRS=(('TQQQ','QQQ',3),('UPRO','SPY',3),('SPXL','SPY',3),('SSO','SPY',2),('TNA','IWM',3),('URTY','IWM',3))

def normalize(w):
    den=np.abs(w).sum()
    return w/den if den>1e-12 else np.zeros_like(w)

def neutralize(score,loadings):
    A=np.column_stack([np.ones(len(score)),loadings])
    # Economic weights, not untradeable residual-return pseudo-assets.
    w=score-A@np.linalg.lstsq(A,score,rcond=1e-10)[0]
    return normalize(w)

def factor_signals(px,member,cadence=5,window=252):
    p=px.to_numpy(float); r=np.full_like(p,np.nan);r[1:]=p[1:]/p[:-1]-1
    kinds=('rev5','rev21','momentum','disagreement','consensus','long_momentum')
    out={k:np.zeros_like(p) for k in kinds};audit=[];fit=-1;cols=np.array([],int)
    for t in range(((window+cadence-1)//cadence)*cadence,len(p),cadence):
        ref=t//21*21
        if ref<window:continue
        if ref!=fit:
            fit=ref
            cols=np.flatnonzero(member[ref]&np.isfinite(r[ref-window+1:ref+1]).all(0))
            if len(cols)<30:continue
            hist=r[ref-window+1:ref+1,cols]
            vol=hist.std(0,ddof=1);good=vol>1e-5;cols=cols[good];vol=vol[good];hist=hist[:,good]
            if len(cols)<30:continue
            R=hist/vol;Rc=R-R.mean(0)
            with threadpool_limits(limits=2):
                cov=Rc.T@Rc/(len(R)-1)
                _,V=eigh(cov,subset_by_index=[len(cols)-5,len(cols)-1],check_finite=False)
            B=vol[:,None]*V
            audit.append({'fit':str(px.index[ref].date()),'last_fit_return':str(px.index[ref].date()),'count':len(cols)})
        if len(cols)<30:continue
        ok=member[t,cols]&np.isfinite(r[t-125:t+1,cols]).all(0)&np.isfinite(p[t,cols])
        if ok.sum()<30:continue
        history=r[t-125:t+1,cols]
        if not np.isfinite(history).all():
            # Do not silently fill unavailable peer returns during the frozen factor month.
            continue
        std=history/vol
        residual=std-(std@V)@V.T
        a=residual[-5:].sum(0)/np.sqrt(5);b=residual[-21:].sum(0)/np.sqrt(21)
        slow=residual[:-21].sum(0)/np.sqrt(105)
        scores={'rev5':-a,'rev21':-b,'momentum':slow,'disagreement':-a*np.exp(-np.maximum(a*slow,0)), 'consensus':np.where(a*b>0,-np.sign(a)*np.minimum(abs(a),abs(b)),0),'long_momentum':slow}
        for k,v in scores.items():
            v=np.clip(v[ok],-3,3)/vol[ok]
            if k=='long_momentum': w=normalize(np.maximum(v,0))
            else:w=neutralize(v,B[ok])
            out[k][t,cols[ok]]=w
    return out,audit

def twin_signals(px,cadence=5):
    out={k:np.zeros(px.shape) for k in ('twin_revert','twin_turn','twin_veto')};details=[]
    for a,b,birth in TWIN_PAIRS:
        if a not in px or b not in px:continue
        # Mask pre-issuance cloned/backfilled histories before rolling features.
        A=px[a].where(px.index>=birth);B=px[b].where(px.index>=birth)
        v=np.log(A/B)
        center=v.shift().rolling(126,min_periods=126).median()
        sd=v.shift().rolling(126,min_periods=126).std()
        z=(v-center)/sd; recent=v.diff(5);slow=v.diff(63)
        valid=(sd>1e-5)&z.notna()&A.notna()&B.notna()
        sig=np.where(valid&(abs(z)>=1.5),-np.sign(z),0.)
        j=px.columns.get_loc(a);k=px.columns.get_loc(b)
        base={'twin_revert':sig,'twin_turn':sig*(z*recent<0),'twin_veto':sig*(z*slow<=0)}
        for name,x in base.items():
            x=np.nan_to_num(x)
            out[name][:,j]+=x/(2*len(TWIN_PAIRS));out[name][:,k]-=x/(2*len(TWIN_PAIRS))
        details.append({'pair':a+'/'+b,'available_after':birth,'signal_count':int((sig[::cadence]!=0).sum())})
    for v in out.values():v[np.arange(len(px))%cadence!=0]=0
    return out,details

def fund_signals(px,cadence=5):
    p=px.to_numpy();out={k:np.zeros_like(p) for k in ('fund_constant','fund_track_revert','fund_track_scaled')};diag=[]
    for a,b,L in FUND_PAIRS:
        if a not in px or b not in px:continue
        ra=px[a].pct_change(fill_method=None);rb=px[b].pct_change(fill_method=None)
        err=ra-L*rb
        displacement=err.rolling(5).sum()-5*err.shift().rolling(63).mean()
        sd=err.shift().rolling(63).std();z=displacement/(sd*np.sqrt(5))
        known=z.notna()&(sd>1e-7)&px[a].notna()&px[b].notna()
        # + means long lev/short L underlying; negative is the expense-capture side.
        sigs={'fund_constant':np.where(known,-1.,0.),'fund_track_revert':np.where(known,-np.sign(z),0.),'fund_track_scaled':np.where(known,np.clip(-z,-1,1),0.)}
        j=px.columns.get_loc(a);k=px.columns.get_loc(b)
        for name,s in sigs.items():
            out[name][:,j]+=s/(len(FUND_PAIRS)*(L+1));out[name][:,k]-=L*s/(len(FUND_PAIRS)*(L+1))
        diag.append(pd.DataFrame({'pair':a+'/'+b,'date':px.index,'arithmetic_hedge':L*rb-ra,'log_spread_not_pnl':L*np.log1p(rb)-np.log1p(ra),'variance':rb*rb}))
    for v in out.values():v[np.arange(len(px))%cadence!=0]=0
    return out,pd.concat(diag,ignore_index=True)
