"""Share/cash mark-to-market ledger with immutable, minimum-30-session lots."""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from .signals import proposals

@dataclass(frozen=True)
class Settings:
    fee_bps: float=25.
    entry_lag: int=1
    sleeves: int=6
    names: int=5
    hedge: bool=False
    borrow_rate: float=.01
    cash_rate: float=0.
    missing_grace: int=0
    def __post_init__(self):
        if self.entry_lag<1 or self.fee_bps<0 or self.sleeves!=6 or self.names<1 or self.missing_grace<0:
            raise ValueError('Invalid ledger settings')


def capped_weights(vol,names=5,all_members=False):
    vol=np.asarray(vol,float)
    if not len(vol):return vol
    if not np.isfinite(vol).all() or (vol<=0).any():raise ValueError('Bad volatility')
    if all_members:return np.repeat(1/len(vol),len(vol))
    target=min(1.,len(vol)/names); w=np.zeros(len(vol)); free=np.ones(len(vol),bool)
    for _ in range(len(vol)+1):
        left=target-w.sum()
        if left<1e-14 or not free.any():break
        x=1/vol[free]; alloc=left*x/x.sum(); ids=np.flatnonzero(free)
        over=alloc>.3
        if not over.any():w[ids]=alloc;break
        w[ids[over]]=.3;free[ids[over]]=False
    assert w.sum()<=1+1e-12 and (w<=.3+1e-12).all()
    return w


def simulate(p,market,feature_frames,method,start_i,maps=None,cfg=Settings()):
    if not p.index.equals(market.index):raise ValueError('Calendar mismatch')
    a=p.to_numpy(float); bm=market.to_numpy(float); columns=p.columns.get_indexer
    if not np.isfinite(bm).all() or (bm<=0).any():raise ValueError('Missing benchmark')
    fee=cfg.fee_bps/10000; cash=np.full(6,1/6.); cash[-1]=1-cash[:-1].sum(); books=[[] for _ in range(6)]
    pending={}; trades=[]; orders=[]; daily=[]; cost_total=borrow_total=0.; turnover_total=0.; nextid=0
    maps=maps or {}; all_lots=[]
    for t in range(start_i,len(p)):
        days=(p.index[t]-p.index[t-1]).days if t>start_i else 0
        for s,book in enumerate(books):
            restricted=sum(l['short_proceeds'] for l in book)
            cash[s]+=max(0.,cash[s]-restricted)*((1+cfg.cash_rate)**(days/365.25)-1)
            for l in book:
                bcost=l['hedge_qty']*bm[max(t-1,0)]*cfg.borrow_rate*days/365.25
                cash[s]-=bcost;borrow_total+=bcost;l['borrow']+=bcost
                price=a[t,l['col']]
                if l['written_off']:l['mark']=0.
                elif np.isfinite(price) and price>0:
                    l['mark']=price;l['missing_streak']=0
                else:
                    l['missing_streak']+=1
                    if l['missing_streak']>cfg.missing_grace:
                        l['written_off']=True;l['mark']=0.;l['writeoff_i']=t
                l['min_price']=min(l['min_price'],l['mark']);l['max_price']=max(l['max_price'],l['mark'])
            ending=[l for l in book if l['exit_i']==t]
            for l in ending:
                value=l['qty']*l['mark']; cover=l['hedge_qty']*bm[t]
                cost=fee*(value+cover);cash[s]+=value-cover-cost
                turnover_total+=value+cover;cost_total+=cost
                l['exit_price']=l['mark'];l['exit_cost']=cost;l['closed']=True
                l['net_pnl']=value-l['notional']+l['short_proceeds']-cover-l['entry_cost']-cost-l['borrow']
                trades.append(l.copy());book.remove(l)
        # Signal selected earlier; execution may fail but is NEVER silently replaced.
        for s,selected in pending.pop(t,[]):
            if books[s]:raise AssertionError('Overlapping sleeve reservation')
            if selected.empty:continue
            w=capped_weights(selected.vol.to_numpy(),cfg.names,method=='equal60')
            betas=selected.beta.clip(0,1).to_numpy() if cfg.hedge else np.zeros(len(selected))
            fraction=.5 if cfg.hedge else 1.
            budget=cash[s]*fraction/(1+fee*float(np.sum(w*(1+betas))))
            for (_,row),weight,beta in zip(selected.iterrows(),w,betas):
                col=int(columns([row.ticker])[0]);price=a[t,col]
                if not np.isfinite(price) or price<=0:
                    orders.append(dict(i=int(row.i),entry_i=t,sleeve=s,ticker=row.ticker,h=int(row.h),status='unavailable_entry'))
                    continue
                notion=budget*weight;short=notion*beta;cost=fee*(notion+short)
                cash[s]-=notion+cost;cash[s]+=short
                cost_total+=cost;turnover_total+=notion+short
                l=dict(id=nextid,sleeve=s,ticker=row.ticker,col=col,signal_i=int(row.i),entry_i=t,
                    exit_i=t+int(row.h),h=int(row.h),entry_price=price,qty=notion/price,notional=notion,
                    hedge_qty=short/bm[t],short_proceeds=short,entry_cost=cost,borrow=0.,mark=price,
                    min_price=price,max_price=price,missing_streak=0,written_off=False,writeoff_i=-1,closed=False,
                    family=str(row.family),score=float(row.score))
                nextid+=1;books[s].append(l);all_lots.append(l)
                orders.append(dict(i=int(row.i),entry_i=t,sleeve=s,ticker=row.ticker,h=int(row.h),status='filled'))
        if (t-252)%5==0 and t in feature_frames:
            s=((t-252)//5)%6
            busy=bool(books[s]) or any(any(x[0]==s for x in v) for v in pending.values())
            if not busy and cash[s]>1e-12:
                held={l['ticker'] for book in books for l in book}
                hm=maps.get(p.index[t].year,{})
                choices=proposals(feature_frames[t],method,hm,() if method=='equal60' else held)
                if method!='equal60':choices=choices.head(cfg.names)
                if not choices.empty and t+cfg.entry_lag<len(p):pending.setdefault(t+cfg.entry_lag,[]).append((s,choices.copy()))
                else:orders.append(dict(i=t,entry_i=t+cfg.entry_lag,sleeve=s,ticker='',h=0,status='abstain'))
        longs=sum(l['qty']*l['mark'] for book in books for l in book)
        shorts=sum(l['hedge_qty']*bm[t] for book in books for l in book)
        nav=cash.sum()+longs-shorts
        if nav<=0:raise ValueError('Insolvent backtest')
        if not cfg.hedge and (cash< -1e-10).any():raise AssertionError('Borrowed cash in long-only ledger')
        daily.append(dict(date=str(p.index[t].date()),i=t,nav=nav,cash=float(cash.sum()),long=longs,short=shorts,
            exposure=longs/nav,gross=(longs+shorts)/nav,net=(longs-shorts)/nav,positions=sum(map(len,books)),
            costs=cost_total,borrow=borrow_total,turnover_notional=turnover_total,
            margin_breach=bool(cfg.hedge and (longs+shorts)>0 and nav<.3*(longs+shorts))))
    # Final holdings remain open and are marked, not artificially liquidated.
    for book in books:
        for l in book:
            l=l.copy();l['net_pnl']=l['qty']*l['mark']-l['notional']+l['short_proceeds']-l['hedge_qty']*bm[-1]-l['entry_cost']-l['borrow'];trades.append(l)
    d=pd.DataFrame(daily); d['return']=d.nav.pct_change();d.loc[0,'return']=d.loc[0,'nav']-1
    tr=pd.DataFrame(trades);od=pd.DataFrame(orders)
    if not tr.empty:
        # Each lot's economically realized/marked P&L plus initial cash reconciles NAV at zero cash yield.
        if cfg.cash_rate==0 and abs(1+tr.net_pnl.sum()-d.nav.iloc[-1])>1e-9:raise AssertionError('P&L reconciliation')
        if (tr.h<30).any() or not (tr.exit_i-tr.entry_i==tr.h).all():raise AssertionError('Holding duration')
    return d,tr,od
