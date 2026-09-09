"""Self-financing, lagged static-lot total-return-unit accounting; no log-P&L."""
from dataclasses import dataclass
import numpy as np
import pandas as pd

@dataclass(frozen=True)
class Costs:
    side_bps: float=5.
    borrow_apr: float=.03
    rf_apr: float=.03
    funding_spread: float=.02
    rebate_fraction: float=0.
    def __post_init__(self):
        if not np.isfinite([self.side_bps,self.borrow_apr,self.rf_apr,self.funding_spread,self.rebate_fraction]).all() or min(self.side_bps,self.borrow_apr,self.rf_apr,self.funding_spread)<0 or not 0<=self.rebate_fraction<=1:
            raise ValueError('invalid costs')

def simulate(prices, signals, hold=30, cadence=5, costs=Costs(), initial=1., delay=1, end=None, rf_returns=None):
    """Signal at t; pre-specified quantities fill at t+delay close; exit hold sessions later.

    'shares' are units of archived adjusted-return series. This avoids log-spread
    pseudo-P&L, but is NOT a claim of raw-price/dividend/fill completeness.
    Missing execution means no fill for that leg. A held missing mark is forced to
    an adverse price (long zero, short twice last mark); each event blocks certification.
    Every scheduled, unexpired cohort remains marked in daily NAV, not dropped.
    """
    p=prices.to_numpy(float); s=np.asarray(signals,float)
    if s.shape!=p.shape or not np.isfinite(s).all(): raise ValueError('invalid signals')
    if any(type(v) is not int or v<1 for v in (hold,cadence,delay)): raise ValueError('invalid clocks')
    if not np.isfinite(initial) or initial<=0 or len(prices)==0 or prices.columns.has_duplicates:raise ValueError('invalid account/input size')
    if end is not None and (type(end) is not int or not 1<=end<=len(prices)):raise ValueError('invalid end')
    if not prices.index.is_monotonic_increasing or prices.index.has_duplicates: raise ValueError('bad dates')
    if np.max(np.sum(np.abs(s),axis=1),initial=0)>1+1e-8: raise ValueError('gross >1 at signal')
    T,N=p.shape; T=min(T,end) if end else T
    rf_input=None if rf_returns is None else np.asarray(rf_returns,float)
    if rf_input is not None and (len(rf_input)!=len(p) or not np.isfinite(rf_input).all()):raise ValueError('invalid risk-free returns')
    q=np.zeros(N); marks=np.where(np.isfinite(p[0]),p[0],0.); cash=initial; nav=initial
    cohorts=[]; records=[]; trades=[]; capacity=max(1,int(np.ceil(hold/cadence)))
    days=np.r_[1,np.diff(prices.index.to_numpy()).astype('timedelta64[D]').astype(float)]/365
    # Quantities generated at the preceding signal close, using that close's NAV.
    queue={}
    for d in range(T):
        oldnav=nav
        prevshort=float((-np.minimum(q,0)*marks).sum())
        free=cash-prevshort
        rf=costs.rf_apr*days[d] if rf_input is None else float(rf_input[d])
        interest=(max(free,0)+costs.rebate_fraction*prevshort)*rf+min(free,0)*(rf+costs.funding_spread*days[d])
        borrow=prevshort*costs.borrow_apr*days[d]
        cash+=interest-borrow
        valid=np.isfinite(p[d])&(p[d]>0)
        q[np.abs(q*marks)<1e-12*max(nav,1.)]=0. # floating-point dust, not a tradable holding
        unknown=(q!=0)&~valid
        missing_count=int(unknown.sum()); trade=0.; fees=0.; partial=0
        if missing_count:
            # Conservative stress liquidation, not a known realized delisting payout.
            stress=np.where(q>0,0,2*marks)
            cash+=float((q[unknown]*stress[unknown]).sum())
            for lot in cohorts: lot['q'][unknown]=0
            q[unknown]=0
        marks=np.where(valid,p[d],marks)
        nav=cash+float(q@marks)
        remaining=[]; netorder=np.zeros(N)
        for lot in cohorts:
            if lot['exit']<=d:
                delta=-lot['q']; dollars=delta*marks
                amt=float(np.abs(dollars).sum()); netorder+=delta; q+=delta
                trades.append((lot['signal'],lot['entry'],d,'exit',amt))
            else: remaining.append(lot)
        cohorts=remaining
        if d in queue and nav>0:
            order,sg=queue.pop(d);partial=int(((order!=0)&~valid).sum())
            qty=np.where(valid,order,0);dollars=qty*marks
            amt=float(np.abs(dollars).sum());netorder+=qty;q+=qty
            cohorts.append({'q':qty.copy(),'entry':d,'signal':sg,'exit':d+hold})
            trades.append((sg,d,d+hold,'entry',amt))
        trade=float(np.abs(netorder*marks).sum());fees=trade*costs.side_bps/1e4
        cash-=float(netorder@marks)+fees
        nav=cash+float(q@marks)
        gross=float(np.abs(q*marks).sum())/max(nav,1e-12)
        short=float((-np.minimum(q,0)*marks).sum())/max(nav,1e-12)
        rtn=nav/oldnav-1 if oldnav>0 else np.nan
        if nav<=0:
            top=np.argsort(np.abs(q*marks))[-8:]
            raise RuntimeError(f'bankruptcy {prices.index[d]} NAV={nav}, previous={oldnav}, positions={[(prices.columns[j],float(q[j]),float(marks[j])) for j in top]}')
        records.append((nav,rtn,rtn-rf,rf,gross,short,trade/oldnav,fees/oldnav,borrow/oldnav,missing_count,partial,interest/oldnav))
        # Do not condition orders on availability of future execution/exit prices.
        if d+delay<T and np.any(s[d]):
            if np.any((s[d]!=0)&~valid):raise ValueError('signal on unavailable quote')
            queue[d+delay]=(np.divide(s[d]*nav/capacity,marks,out=np.zeros(N),where=valid),d)
    result=pd.DataFrame(records,index=prices.index[:T],columns=['equity','return','excess','rf','gross','short','turnover','cost','borrow','unknown_liquidations','unfilled_legs','cash_interest'])
    result.attrs['trades']=trades
    return result

def metrics(d):
    if len(d)<2:return {'n':len(d)}
    x=d.excess.to_numpy(); mu=x.mean(); sd=x.std(ddof=1)
    z=x-mu; gamma=np.dot(z,z)/len(z); lrv=gamma
    for k in range(1,min(63,len(x)-1)+1): lrv+=2*(1-k/64)*np.dot(z[k:],z[:-k])/len(z)
    eq=np.r_[1.,np.cumprod(1+d['return'].to_numpy())];dd=eq/np.maximum.accumulate(eq)-1
    return {'n':len(d),'sharpe':float(mu/sd*np.sqrt(252)) if sd>1e-14 else None,'hac_sharpe':float(mu/np.sqrt(lrv)*np.sqrt(252)) if lrv>1e-14 else None,'cagr':float(eq[-1]**(252/len(d))-1),'max_drawdown':float(dd.min()),'annual_excess':float(252*mu),'annual_vol':float(sd*np.sqrt(252)),'worst_day':float(d['return'].min()),'avg_gross':float(d.gross.mean()),'max_gross':float(d.gross.max()),'annual_turnover':float(d.turnover.mean()*252),'unknown_liquidations':int(d.unknown_liquidations.sum()),'unfilled_legs':int(d.unfilled_legs.sum()),'trading_drag':float(d.cost.mean()*252),'borrow_drag':float(d.borrow.mean()*252)}

def bootstrap(x,seed=91,reps=1000,block=63):
    x=np.asarray(x,float);rng=np.random.default_rng(seed);n=len(x);out=[]
    for _ in range(reps):
        starts=rng.integers(0,n,int(np.ceil(n/block)))
        a=x[((starts[:,None]+np.arange(block))%n).ravel()[:n]]
        sd=a.std(ddof=1)
        if sd>1e-14:out.append(a.mean()/sd*np.sqrt(252))
    return np.quantile(out,[.025,.5,.975]).tolist() if out else None
