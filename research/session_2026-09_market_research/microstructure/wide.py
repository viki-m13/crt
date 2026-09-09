"""PREREG7: wide-universe breadth test. PIT universe, survivorship-free."""
import numpy as np, pandas as pd
W='/tmp/wide/'
cl=pd.read_parquet(W+'close.parquet'); op=pd.read_parquet(W+'open.parquet'); dv=pd.read_parquet(W+'dvol.parquet')
cl.index=pd.to_datetime(cl.index); op.index=pd.to_datetime(op.index); dv.index=pd.to_datetime(dv.index)
cols=sorted(set(cl.columns)&set(op.columns)&set(dv.columns))
cl,op,dv=cl[cols],op[cols],dv[cols]
cl=cl[cl.index<'2024-01-01']; op=op.reindex(cl.index); dv=dv.reindex(cl.index)
print(f"panel {cl.shape}, {cl.index.min().date()}..{cl.index.max().date()}")

# PIT membership
pit=pd.read_csv(W+'sp500_pit.csv'); pit['date']=pd.to_datetime(pit.date)
pit=pit[(pit.date>=cl.index.min())&(pit.date<=cl.index.max())].set_index('date').tickers
memb=pd.DataFrame(False,index=cl.index,columns=cols)
pit=pit.reindex(cl.index).ffill()
for d,s in pit.items():
    if isinstance(s,str):
        t=[x for x in s.split(',') if x in memb.columns]
        memb.loc[d,t]=True
print(f"PIT members/day: mean {memb.sum(1).mean():.0f}, min {memb.sum(1).min()}, max {memb.sum(1).max()}")

# returns
intra = cl/op - 1.0                       # close_D / open_D
onit  = op.shift(-1)/cl - 1.0             # open_{D+1} / close_D   (target of W1)
on_in = op/cl.shift(1) - 1.0              # open_D / close_{D-1}   (signal of W3)
dv20  = dv.shift(1).rolling(20,min_periods=10).mean()   # TRAILING only

valid = memb & cl.notna() & op.notna() & (dv20 > 2e7)   # >$20M ADV, trailing
print(f"tradable/day after PIT + $20M trailing ADV: mean {valid.sum(1).mean():.0f}")

def xs_ls(sig, tgt, mask, nq=5, nmin=20, sub=None, rng=None):
    """per-day quintile L/S gross return. sub=int -> random subsample of universe."""
    out={}
    for d in sig.index:
        m=mask.loc[d]
        s=sig.loc[d][m]; t=tgt.loc[d][m]
        ok=s.notna()&t.notna(); s=s[ok]; t=t[ok]
        if sub is not None and len(s)>sub:
            pick=rng.choice(len(s),sub,replace=False); s=s.iloc[pick]; t=t.iloc[pick]
        if len(s)<nmin: continue
        r=s.rank(method='first'); k=max(len(s)//nq,1)
        lo=t[r<=k].mean(); hi=t[r>len(s)-k].mean()
        if np.isfinite(lo) and np.isfinite(hi): out[d]=hi-lo
    return pd.Series(out)

def rep(name,ser,split):
    s=ser[(ser.index>=split[0])&(ser.index<=split[1])].dropna()
    if len(s)<30: return None
    t=s.mean()/(s.std(ddof=1)/np.sqrt(len(s)))
    return dict(n=len(s), bp=s.mean()*1e4, t=t, be=s.mean()*1e4, sharpe=s.mean()/s.std()*np.sqrt(252))

TR=('2016-01-01','2021-12-31'); VA=('2022-01-01','2023-12-31')
res={}
w1=xs_ls(-intra, onit, valid); res['W1']=(rep('W1',w1,TR),rep('W1',w1,VA))
w3=xs_ls(-on_in, intra, valid); res['W3']=(rep('W3',w3,TR),rep('W3',w3,VA))
disp=intra[valid].std(1); dmed=disp[(disp.index>=TR[0])&(disp.index<=TR[1])].median()
w2=w1[disp.reindex(w1.index)>dmed];  res['W2']=(rep('W2',w2,TR),rep('W2',w2,VA))

print(f"\n{'test':4s} {'split':6s} {'ndays':>6s} {'gross bp/day':>13s} {'t':>7s} {'ann Sharpe':>11s} {'break-even cost bp':>19s}")
for k,(tr,va) in res.items():
    for lab,r in (('TRAIN',tr),('VALID',va)):
        if r: print(f"{k:4s} {lab:6s} {r['n']:6d} {r['bp']:+13.2f} {r['t']:+7.2f} {r['sharpe']:+11.2f} {r['be']:19.2f}")
w1.to_frame('r').to_parquet(W+'w1.parquet'); disp.to_frame('d').to_parquet(W+'disp.parquet')
