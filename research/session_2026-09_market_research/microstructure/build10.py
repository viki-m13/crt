"""PREREG10 event builder. All causal. Uses P_rev(pi)=P_fwd(reverse(pi)) so the
rolling irreversibility is O(n) via cumulative pattern counts."""
import numpy as np, pandas as pd, glob, itertools
from pathlib import Path
M=3
# pattern encoding + the reversal map, derived by enumeration
def code_of(tr):
    o=np.argsort(np.asarray(tr),kind='stable'); return int((o*(M**np.arange(M))).sum())
uniq=sorted({code_of(p) for p in itertools.permutations([0,1,2])})
IDX={c:i for i,c in enumerate(uniq)}                      # dense 0..5
REV=np.zeros(len(uniq),dtype=int)
for p in itertools.permutations([0.0,1.0,2.0]):
    REV[IDX[code_of(p)]]=IDX[code_of(p[::-1])]
print("pattern reversal map:",REV)

def codes_dense(r):
    W=np.lib.stride_tricks.sliding_window_view(r,M)
    c=(np.argsort(W,axis=1,kind='stable')*(M**np.arange(M))).sum(1)
    return np.array([IDX[x] for x in c])

def rolling_irrev(r,win):
    """KL(P_fwd||P_rev) over a trailing window, at every index. O(n)."""
    c=codes_dense(r); n=len(c); K=len(uniq)
    oh=np.zeros((n,K)); oh[np.arange(n),c]=1.0
    cs=np.vstack([np.zeros(K),np.cumsum(oh,axis=0)])
    out=np.full(len(r),np.nan)
    for i in range(win,n+1):
        cnt=cs[i]-cs[i-win]
        tot=cnt.sum()
        if tot<win*0.5: continue
        p=cnt/tot; q=p[REV]
        k=(p>0)&(q>0)
        out[i+M-2]=float((p[k]*np.log(p[k]/q[k])).sum())
    return out

rows=[]
for f in sorted(glob.glob('/tmp/venues/c1m_*.parquet')):
    sym=Path(f).stem[4:]
    d=pd.read_parquet(f,columns=['t','c','v','tbv','n'])
    d['t']=pd.to_datetime(d.t,unit='ms',utc=True)
    d=d[d.t<'2026-01-01']                                  # TEST LOCKED
    c=d.c.to_numpy(float); v=d.v.to_numpy(float); tb=d.tbv.to_numpy(float); ntr=d.n.to_numpy(float)
    ok=np.isfinite(c)&(c>0)&np.isfinite(v)
    c,v,tb,ntr,ts=c[ok],v[ok],tb[ok],ntr[ok],d.t.to_numpy()[ok]
    lr=np.diff(np.log(c)); lr=np.concatenate([[0.0],lr])
    irr=rolling_irrev(lr,120)
    sflow=2*tb-v
    vbar=pd.Series(v).shift(1).rolling(100,min_periods=50).mean().to_numpy()
    rv=pd.Series(lr).shift(1).rolling(120,min_periods=60).std().to_numpy()
    tsz=pd.Series(v/np.maximum(ntr,1)).shift(1).rolling(120,min_periods=60).mean().to_numpy()
    lc=np.log(c)
    idx=np.arange(240,len(c)-65,5)
    F=np.array([sflow[i-4:i+1].sum() for i in idx])/np.maximum(vbar[idx],1e-9)
    dP=lc[idx]-lc[idx-5]
    aF=np.abs(F)
    keep=(aF>1e-6)&np.isfinite(irr[idx])&np.isfinite(rv[idx])&np.isfinite(tsz[idx])
    i2=idx[keep]; F=F[keep]; dP=dP[keep]; aF=aF[keep]
    rows.append(pd.DataFrame(dict(sym=sym,t=ts[i2],F=F,dP=dP,
        lam=np.abs(dP)/aF, sig=irr[i2], D=irr[i2]/aF,
        vol=rv[i2], tsize=tsz[i2], volume=v[i2],
        f15=lc[i2+15]-lc[i2], f30=lc[i2+30]-lc[i2], f60=lc[i2+60]-lc[i2])))
    print(f"{sym}: {len(i2):,} events",flush=True)
E=pd.concat(rows,ignore_index=True)
E['day']=pd.to_datetime(E.t).dt.tz_convert('UTC').dt.normalize()
for k in (15,30,60): E[f'rev{k}']=-np.sign(E.dP)*E[f'f{k}']
E.to_parquet('/tmp/venues/events10.parquet',index=False)
print(f"\n{len(E):,} events, {E.day.nunique()} days, {E.sym.nunique()} symbols, {E.day.min().date()}..{E.day.max().date()}")
