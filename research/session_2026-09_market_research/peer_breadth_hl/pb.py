"""Peer Breadth, recreated from spec, on the Hyperliquid daily universe.
Causal by construction: signal at close of day t -> enter at OPEN of day t+1.
Fixes the package's own audited blockers: one open position per coin, pro-rata
batch allocation at OPEN-time equity, entry cost debited immediately,
target live from the entry bar."""
import numpy as np, pandas as pd, glob, json, sys
from pathlib import Path

def load(minhist=30):
    O,H,L,C={},{},{},{}
    for f in sorted(glob.glob('/tmp/ase/hl_*.parquet')):
        s=Path(f).stem[3:]
        d=pd.read_parquet(f)
        d['dt']=pd.to_datetime(d['d']) if 'd' in d.columns else pd.to_datetime(d['t'],unit='ms')
        d=d.set_index('dt').sort_index()
        O[s],H[s],L[s],C[s]=d.o,d.h,d.l,d.c
    return (pd.DataFrame(O),pd.DataFrame(H),pd.DataFrame(L),pd.DataFrame(C))

O,H,L,C=load()
C=C.sort_index(); O=O.reindex(C.index); H=H.reindex(C.index); L=L.reindex(C.index)

def run(universe=None, breadth_bars=5, frac=0.70, atr_n=14, stop_atr=2.5,
        min_stop=0.005, max_stop=0.20, target_r=3.0, hold_days=14,
        risk_frac=0.0025, max_risk=0.015, gross_cap=2.0, cost_bp=20.0,
        fund_bp_per_8h=1.0, start='2021-01-01', end='2026-08-11', seed_hist=30,
        perturb_future=False, equity0=100000.0):
    cols=[c for c in C.columns if (universe is None or c in universe)]
    c=C[cols]; o=O[cols]; h=H[cols]; l=L[cols]
    if perturb_future:                      # causality probe: corrupt the last 20% of prices
        k=int(len(c)*0.8); c=c.copy(); c.iloc[k:]*=1.5
    r5=c.pct_change(breadth_bars)
    ok=r5.notna()&r5.shift(1).notna()&c.notna()
    tr=pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()]).groupby(level=0).max()
    atr=tr.rolling(atr_n).mean()
    stop_pct=(stop_atr*atr/c).clip(lower=min_stop)
    listed=c.notna().cumsum()               # PIT listing guard
    idx=c.index; idx=idx[(idx>=start)&(idx<=end)]
    eq=equity0; cash=equity0; pos={}; trades=[]; curve=[]
    for ti,t in enumerate(idx):
        i=c.index.get_loc(t)
        if i<max(atr_n,breadth_bars)+2: continue
        # ---- 1. mark & manage OPEN positions on today's bar (exits first)
        for s in list(pos):
            p=pos[s]
            hi=h[s].iloc[i]; lo=l[s].iloc[i]; op=o[s].iloc[i]
            if not np.isfinite(hi): continue
            px=None; why=None
            if p['side']>0:
                if lo<=p['stop']: px,why=p['stop'],'stop'
                elif hi>p['target']: px,why=p['target'],'target'
            else:
                if hi>=p['stop']: px,why=p['stop'],'stop'
                elif lo<p['target']: px,why=p['target'],'target'
            if px is None and (ti-p['ti'])>=hold_days: px,why=op,'time'
            if px is not None:
                gross=p['side']*(px-p['entry'])*p['qty']
                fee=abs(px*p['qty'])*cost_bp/2/1e4
                fund=abs(p['entry']*p['qty'])*(fund_bp_per_8h*3*(ti-p['ti']))/1e4
                cash+=gross-fee-fund
                trades.append(dict(sym=s,side=p['side'],why=why,pnl=gross-fee-p['entry_fee']-fund,
                                   entry=p['entry'],exit=px,days=ti-p['ti'],t=t))
                del pos[s]
        # ---- 2. equity at OPEN-time (no info from later today)
        mtm=sum(p['side']*(c[s].iloc[i-1]-p['entry'])*p['qty'] for s,p in pos.items() if np.isfinite(c[s].iloc[i-1]))
        eq=cash+mtm
        curve.append((t,eq))
        # ---- 3. signal from YESTERDAY's close (i-1) -> enter at TODAY's open
        j=i-1
        el=ok.iloc[j]&(listed.iloc[j]>=seed_hist)
        elig=[s for s in cols if el.get(s,False)]
        if len(elig)<max(8,int(np.ceil(0.8*(len(cols)-1)))): continue
        cands=[]
        for s in elig:
            peers=[q for q in elig if q!=s]
            if len(peers)<8: continue
            pr=r5[peers].iloc[j]; prv=r5[peers].iloc[j-1]
            shr_p=(pr>0).sum()/len(peers); shr_p_y=(prv>0).sum()/len(peers)
            shr_n=(pr<0).sum()/len(peers); shr_n_y=(prv<0).sum()/len(peers)
            own=r5[s].iloc[j]; up=c[s].iloc[j]>c[s].iloc[j-1]
            sp=stop_pct[s].iloc[j]
            if not np.isfinite(sp) or sp>max_stop: continue
            if s in pos: continue
            if shr_p>=frac and shr_p_y<frac and own>0 and up: cands.append((s,+1,sp))
            elif shr_n>=frac and shr_n_y<frac and own<0 and not up: cands.append((s,-1,sp))
        if not cands: continue
        open_risk=sum(abs(p['entry']-p['stop'])*p['qty'] for p in pos.values())
        room=max(0.0,max_risk*eq-open_risk)
        want=[min(risk_frac*eq, room) for _ in cands]
        tot=sum(want)
        if tot>room and tot>0: want=[w*room/tot for w in want]
        gross_now=sum(abs(p['entry']*p['qty']) for p in pos.values())
        for (s,side,sp),wr in zip(cands,want):
            if wr<=0: continue
            entry=o[s].iloc[i]
            if not np.isfinite(entry) or entry<=0: continue
            qty=wr/(entry*(sp+cost_bp/1e4))          # cost reserve in denominator
            notional=entry*qty
            if notional<100: continue
            if gross_now+notional>gross_cap*eq: continue
            gross_now+=notional
            fee=notional*cost_bp/2/1e4
            cash-=fee
            pos[s]=dict(side=side,entry=entry,qty=qty,ti=ti,entry_fee=fee,
                        stop=entry*(1-side*sp),target=entry*(1+side*target_r*sp))
    E=pd.Series(dict(curve)).sort_index()
    T=pd.DataFrame(trades)
    return E,T

def stats(E,T,lab):
    if len(E)<50: return None
    r=E.pct_change().dropna()
    dd=(E/E.cummax()-1).min()*100
    yrs=len(E)/365
    wins=(T.pnl>0).sum() if len(T) else 0
    pf=(T.pnl[T.pnl>0].sum()/abs(T.pnl[T.pnl<0].sum())) if len(T) and (T.pnl<0).any() else np.nan
    return dict(lab=lab,ret=(E.iloc[-1]/E.iloc[0]-1)*100,cagr=((E.iloc[-1]/E.iloc[0])**(1/yrs)-1)*100,
                sh=r.mean()/r.std()*np.sqrt(365) if r.std()>0 else np.nan,dd=dd,
                n=len(T),wr=100*wins/max(len(T),1),pf=pf)
if __name__=='__main__':
    import itertools
    PB20={'AAVE','ADA','AVAX','BCH','BNB','BTC','DOGE','ETH','FIL','LINK','LTC','NEAR','SOL','SUI','TRX','UNI','WLD','XRP','ZEC','PEPE','kPEPE'}
    have=set(C.columns)
    print(f"HL universe: {len(have)} syms. Overlap with the published 20: {len(PB20&have)} -> {sorted(PB20&have)}")
    out=[]
    E,T=run(universe=sorted(PB20&have)); out.append(stats(E,T,f'published-20 subset ({len(PB20&have)} on HL)'))
    E2,T2=run(universe=None);            out.append(stats(E2,T2,f'ALL {len(have)} HL coins'))
    print(f"\n{'variant':34s} {'ret%':>8s} {'CAGR%':>7s} {'Sharpe':>7s} {'maxDD%':>7s} {'n':>5s} {'WR%':>6s} {'PF':>5s}")
    for s in out:
        if s: print(f"{s['lab']:34s} {s['ret']:+8.1f} {s['cagr']:+7.1f} {s['sh']:+7.2f} {s['dd']:+7.1f} {s['n']:5d} {s['wr']:6.1f} {s['pf']:5.2f}")
    E.to_frame('eq').to_parquet('/tmp/pbhl/eq20.parquet'); E2.to_frame('eq').to_parquet('/tmp/pbhl/eqall.parquet')
    T.to_parquet('/tmp/pbhl/tr20.parquet'); T2.to_parquet('/tmp/pbhl/trall.parquet')
