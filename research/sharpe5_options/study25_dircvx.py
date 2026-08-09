#!/usr/bin/env python3
"""Study 25: express the ONE directional signal convexly.

Study 24 killed non-directional convexity: buying straddles/strangles has ~zero
Kelly growth, and vol-of-vol fails as a buy signal because it raises implied
vol as much as it raises realized movement.

But skew25 predicts FORWARD STOCK RETURNS with t=+5.6, monotone in horizon
(study 10) — the only genuinely directional edge found anywhere here. It was
expressed in stock, earning Sharpe 0.44. Stock is linear: a 2.8% IC converts to
a small Sharpe. Options are convex: the same signal, expressed in OTM calls and
puts, converts a modest hit-rate edge into an asymmetric payoff, which is
exactly the structure Kelly rewards.

So: buy OTM CALLS on the highest-skew names, OTM PUTS on the lowest-skew names,
at the ask, held to expiry. If the directional edge is real and large enough to
clear the premium, Kelly growth will be positive and large. If the market has
already priced it, growth will be zero — the same answer as everything else.
"""
import math, os
import numpy as np, pandas as pd
import engine as E
from structures import settle
from study24_convex import kelly_growth

HERE=os.path.dirname(os.path.abspath(__file__))
LIQUID=set("SPY DIA AAPL MSFT NVDA AMZN GOOGL META TSLA AMD MU QCOM NFLX BA XOM JPM BAC C GM F INTC CSCO PYPL AVGO ORCL CRM ADBE TXN COST WMT DIS CAT GE PFE CVX WFC MS GS".split())

def main(step=2):
    dates=E.available_dates()[::step]
    sp=pd.read_parquet(os.path.join(HERE,"cache","spots.parquet"))
    panel=sp.pivot(index="date",columns="act_symbol",values="spot").sort_index()
    hist={s:panel[s].dropna() for s in panel.columns}
    f=pd.read_parquet(os.path.join(HERE,"cache","features.parquet"))
    f["date"]=pd.to_datetime(f.date)
    fmap={(r.date.strftime("%Y-%m-%d"),r.act_symbol):r for r in f.itertuples(index=False)}
    def se_of(sym,exp):
        c=hist.get(sym)
        if c is None: return None
        b=c[c.index<=exp]; a=c[c.index>exp]
        if len(b) and (pd.Timestamp(exp)-pd.Timestamp(b.index[-1])).days<=4: return float(b.iloc[-1])
        if len(a) and (pd.Timestamp(a.index[0])-pd.Timestamp(exp)).days<=4: return float(a.iloc[0])
        return None
    rows=[]
    for di,day in enumerate(dates):
        ch=E.load_chain(day); ch=ch[(ch.dte>=20)&(ch.dte<=60)&(ch.ask>0)]
        if ch.empty or day not in panel.index: continue
        # rank skew cross-sectionally today
        sk={}
        for sym in LIQUID:
            r=fmap.get((day,sym))
            if r is not None and np.isfinite(getattr(r,"skew25",np.nan)): sk[sym]=r.skew25
        if len(sk)<15: continue
        vals=np.array(list(sk.values())); hi=np.quantile(vals,.8); lo=np.quantile(vals,.2)
        for sym,g in ch.groupby("act_symbol"):
            if sym not in sk or sym not in panel.columns: continue
            s=panel.at[day,sym]
            if not np.isfinite(s): continue
            exp=g.expiration.min(); ge=g[g.expiration==exp]
            se=se_of(sym,exp)
            if se is None or abs(math.log(se/s))>0.8: continue
            side=None
            if sk[sym]>=hi: side="call"      # high skew -> bullish (study 10 sign)
            elif sk[sym]<=lo: side="put"
            if side is None: continue
            for otm,lab in ((0.03,"3pct"),(0.06,"6pct"),(0.10,"10pct")):
                if side=="call":
                    cand=ge[(ge.call_put=="Call")&(ge.ask>0)]
                    if cand.empty: continue
                    o=cand.loc[(cand.strike-s*(1+otm)).abs().idxmin()]
                    pay=settle(se,float(o.strike),"Call")*100.0
                else:
                    cand=ge[(ge.call_put=="Put")&(ge.ask>0)]
                    if cand.empty: continue
                    o=cand.loc[(cand.strike-s*(1-otm)).abs().idxmin()]
                    pay=settle(se,float(o.strike),"Put")*100.0
                cost=float(o.ask)*100.0
                if cost<=0: continue
                rows.append((day,sym,side,lab,cost,pay,(pay-cost)/cost,int(ge.dte.iloc[0])))
        if (di+1)%100==0: print(f"dircvx {di+1}/{len(dates)} rows={len(rows):,}",flush=True)
    d=pd.DataFrame(rows,columns=["date","sym","side","otm","cost","payoff","ret","dte"])
    d.to_parquet(os.path.join(HERE,"cache","dircvx.parquet"),index=False)
    print("\n"+"="*86)
    print("STUDY 25 — the directional skew signal expressed in OTM options")
    print("="*86)
    for otm in ("3pct","6pct","10pct"):
        g=d[d.otm==otm]
        if len(g)<200: continue
        rounds=365/max(g.dte.mean(),1)
        fk,gr=kelly_growth(g.ret.values)
        print(f"\n--- {otm} OTM (n={len(g):,}) ---")
        print(f"  mean={g.ret.mean():+.3f} median={g.ret.median():+.3f} "
              f"win%={(g.ret>0).mean():.1%} max={g.ret.max():+.1f}x")
        print(f"  Kelly f*={fk:.3f} g/bet={gr:+.5f} -> ann {math.exp(gr*rounds)-1:+.1%}")
        for sd in ("call","put"):
            s2=g[g.side==sd]
            if len(s2)<150: continue
            f2,g2=kelly_growth(s2.ret.values)
            print(f"    {sd:>4}s only: mean={s2.ret.mean():+.3f} win%={(s2.ret>0).mean():.1%} "
                  f"Kelly f*={f2:.3f} -> ann {math.exp(g2*rounds)-1:+.1%}")

if __name__=="__main__": main()
