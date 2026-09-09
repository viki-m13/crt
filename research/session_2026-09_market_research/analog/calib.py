"""Is the analog cone honest? Measure coverage of the 10-90 band, then fix it."""
import numpy as np, pandas as pd, json
D=json.load(open('/tmp/analog/data.json')); SY=sorted(D)
LP={s:np.log(np.array(D[s]['p'])) for s in SY}
W,H,K=120,63,20
rows=[]
for s in SY:
    lp=LP[s]; n=len(lp)
    if n<W+H+800: continue
    M=np.lib.stride_tricks.sliding_window_view(lp,W); Mn=M-M[:,[0]]
    for t in range(800,n-H,5):
        cur=lp[t-W+1:t+1]-lp[t-W+1]
        ehi=t-W-1
        if ehi<=W: continue
        rows_i=np.arange(W-1,ehi+1)-W+1
        d=np.sqrt(((Mn[rows_i]-cur)**2).mean(axis=1))
        ends=np.arange(W-1,ehi+1)
        ok=ends+H<n
        d=d[ok]; ends=ends[ok]
        if len(ends)<K: continue
        sel=ends[np.argsort(d)[:K]]
        fwd=np.array([lp[e+H]-lp[e] for e in sel])
        a=lp[t+H]-lp[t]
        rows.append(dict(sym=s,lo10=np.quantile(fwd,.10),hi90=np.quantile(fwd,.90),
                         lo25=np.quantile(fwd,.25),hi75=np.quantile(fwd,.75),
                         med=np.median(fwd),sd=fwd.std(),actual=a))
R=pd.DataFrame(rows); R.to_parquet('/tmp/analog/calib.parquet')
print(f"{len(R):,} out-of-sample cones\n")
print("BAND COVERAGE  (what fraction of actual outcomes landed inside the stated band)")
for lab,lo,hi,target in (('10-90 band','lo10','hi90',80),('25-75 band','lo25','hi75',50)):
    cov=((R.actual>=R[lo])&(R.actual<=R[hi])).mean()*100
    print(f"  {lab}: stated {target}%  ->  ACTUAL {cov:5.1f}%   {'TOO NARROW' if cov<target-3 else ('too wide' if cov>target+3 else 'calibrated')}")
print("\nWHY: the analogs resemble each other, so their spread understates real uncertainty.")
print(f"  median analog spread (90-10): {(np.exp(R.hi90)-np.exp(R.lo10)).median()*100:5.1f} pts")
print(f"  actual |outcome - median|  : {(np.abs(np.exp(R.actual)-np.exp(R.med))).median()*100:5.1f} pts")
# find the widening factor that calibrates it
print("\nCALIBRATION FIX: widen the band about its median by factor f")
for f in (1.0,1.25,1.5,1.75,2.0,2.5,3.0):
    lo=R.med+(R.lo10-R.med)*f; hi=R.med+(R.hi90-R.med)*f
    cov=((R.actual>=lo)&(R.actual<=hi)).mean()*100
    print(f"  f={f:4.2f} -> 10-90 coverage {cov:5.1f}%")
# unconditional (drift) benchmark band from trailing vol
print("\nBENCHMARK: a band built only from trailing volatility + drift (no analogs at all)")
allc=[]
for s in SY:
    lp=LP[s]; n=len(lp)
    if n<W+H+800: continue
    r=np.diff(lp)
    for t in range(800,n-H,5):
        sd=r[t-252:t].std()*np.sqrt(H); mu=r[t-252:t].mean()*H
        a=lp[t+H]-lp[t]
        allc.append((mu-1.2816*sd<=a<=mu+1.2816*sd))
print(f"  10-90 coverage from vol+drift alone: {np.mean(allc)*100:5.1f}%")
