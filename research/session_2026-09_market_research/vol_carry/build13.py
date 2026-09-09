"""PREREG13: build sigma (entropy production) + ATM IV + forward RV panel."""
import numpy as np, pandas as pd, glob, itertools
from pathlib import Path
M=3
def code_of(tr):
    o=np.argsort(np.asarray(tr),kind='stable'); return int((o*(M**np.arange(M))).sum())
uniq=sorted({code_of(p) for p in itertools.permutations([0,1,2])})
IDX={c:i for i,c in enumerate(uniq)}
REV=np.zeros(len(uniq),dtype=int)
for p in itertools.permutations([0.0,1.0,2.0]): REV[IDX[code_of(p)]]=IDX[code_of(p[::-1])]
def kl_irrev(r):
    if len(r)<200: return np.nan
    W=np.lib.stride_tricks.sliding_window_view(r,M)
    c=(np.argsort(W,axis=1,kind='stable')*(M**np.arange(M))).sum(1)
    c=np.array([IDX[x] for x in c])
    p=np.bincount(c,minlength=len(uniq)).astype(float); p/=p.sum()
    q=p[REV]; k=(p>0)&(q>0)
    return float((p[k]*np.log(p[k]/q[k])).sum())

TK=['AAPL','AMD','AMZN','GOOGL','META','MSFT','NFLX','NVDA','SPY','TSLA','XLE','XLF']
D=Path('/tmp/databranch/data/equity_1m_alpaca')
sig_rows=[]
for t in TK:
    d=pd.read_parquet(D/f'{t}.parquet',columns=['t','c'])
    d['t']=pd.to_datetime(d.t,utc=True).dt.tz_convert('America/New_York')
    d=d[(d.t.dt.time>=pd.Timestamp('09:30').time())&(d.t.dt.time<=pd.Timestamp('16:00').time())]
    d['day']=d.t.dt.normalize().dt.tz_localize(None)
    per={}
    for day,g in d.groupby('day',sort=True):
        c=g.c.to_numpy(float)
        if len(c)<300 or c.min()<=0: continue
        r=np.diff(np.log(c))
        if np.all(np.isfinite(r)) and r.std()>0: per[day]=r/r.std()
    days=sorted(per)
    for i in range(10,len(days)):
        w=np.concatenate([per[days[j]] for j in range(i-10,i)])   # trailing 10d, CAUSAL
        sig_rows.append(dict(sym=t,day=days[i],sigma=kl_irrev(w)))
    print(f'sigma {t} ok',flush=True)
S=pd.DataFrame(sig_rows)

# daily closes for realized vol
px=[]
for t in TK:
    d=pd.read_parquet(D/f'{t}.parquet',columns=['t','c'])
    d['t']=pd.to_datetime(d.t,utc=True).dt.tz_convert('America/New_York')
    d=d[(d.t.dt.time>=pd.Timestamp('09:30').time())&(d.t.dt.time<=pd.Timestamp('16:00').time())]
    d['day']=d.t.dt.normalize().dt.tz_localize(None)
    px.append(d.groupby('day').c.last().rename(t))
PX=pd.concat(px,axis=1).sort_index()
LR=np.log(PX).diff()

# chains -> ATM IV nearest 30d
rows=[]
for f in sorted(glob.glob('/home/user/crt/research/sharpe5_options/cache/chains/*.parquet')):
    ch=pd.read_parquet(f)
    ch=ch[ch.act_symbol.isin(TK)]
    if not len(ch): continue
    ch['date']=pd.to_datetime(ch.date); ch['expiration']=pd.to_datetime(ch.expiration)
    ch['dte']=(ch.expiration-ch.date).dt.days
    d0=ch.date.iloc[0]
    if d0 not in PX.index: continue
    for t,g in ch.groupby('act_symbol'):
        spot=PX.loc[d0,t]
        if not np.isfinite(spot): continue
        gg=g[(g.dte>=20)&(g.dte<=45)]
        if not len(gg): continue
        exp=gg.iloc[(gg.dte-30).abs().argsort()].expiration.iloc[0]
        gg=gg[gg.expiration==exp]
        k=gg.iloc[(gg.strike-spot).abs().argsort()].strike.iloc[0]
        atm=gg[gg.strike==k]
        cal=atm[atm.call_put=='Call']; put=atm[atm.call_put=='Put']
        if not len(cal) or not len(put): continue
        iv=float(np.nanmean([cal.vol.iloc[0],put.vol.iloc[0]]))
        if not np.isfinite(iv) or iv<=0: continue
        rows.append(dict(sym=t,day=d0,expiry=exp,dte=int((exp-d0).days),strike=float(k),spot=float(spot),iv=iv,
            c_bid=float(cal.bid.iloc[0]),c_ask=float(cal.ask.iloc[0]),
            p_bid=float(put.bid.iloc[0]),p_ask=float(put.ask.iloc[0]),
            c_delta=float(cal.delta.iloc[0]),p_delta=float(put.delta.iloc[0])))
C=pd.DataFrame(rows)
print(f'\nchain rows {len(C):,}, {C.sym.nunique()} syms, {C.day.min().date()}..{C.day.max().date()}')

# forward realized vol to expiry + HAR-RV terms
def rvann(sym,a,b):
    s=LR[sym].loc[(LR.index>a)&(LR.index<=b)].dropna()
    return float(s.std()*np.sqrt(252)) if len(s)>=5 else np.nan
C['rv_fwd']=[rvann(r.sym,r.day,r.expiry) for r in C.itertuples()]
for w,nm in ((1,'rv1'),(5,'rv5'),(22,'rv22')):
    C[nm]=[float(LR[r.sym].loc[LR.index<=r.day].tail(w).std()*np.sqrt(252)) if w>1 else
           float(abs(LR[r.sym].loc[LR.index<=r.day].iloc[-1])*np.sqrt(252)) for r in C.itertuples()]
P=C.merge(S,on=['sym','day'],how='inner').dropna(subset=['sigma','iv','rv_fwd','rv22'])
P['vrp']=P.iv-P.rv_fwd
P.to_parquet('/tmp/vrp/panel13.parquet',index=False)
print(f"panel {len(P):,} rows, {P.sym.nunique()} syms, {P.day.min().date()}..{P.day.max().date()}")
print(P[['iv','rv_fwd','vrp','sigma','rv22']].describe().round(4).to_string())
