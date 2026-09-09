"""Search for ANY analog configuration that beats the base rate.
Metric that matters: excess hit rate (hit - base) and corr(proj, actual)."""
import numpy as np, pandas as pd, json, itertools
D=json.load(open('/tmp/analog/data.json'))
SY=sorted(D)
LP={s:np.log(np.array(D[s]['p'])) for s in SY}
DT={s:np.array(D[s]['d']) for s in SY}
rng=np.random.default_rng(7)

def eval_cfg(W,H,K,pool,volnorm,agree_gate=None,dist_gate=None,step=7):
    hits=[];proj=[];act=[];base=[]
    # pre-build normalized window banks
    bank={}
    for s in SY:
        lp=LP[s]
        if len(lp)<W+H+400: continue
        M=np.lib.stride_tricks.sliding_window_view(lp,W)
        Mn=M-M[:,[0]]
        if volnorm:
            sd=Mn.std(axis=1,keepdims=True); sd[sd==0]=1; Mn=Mn/sd
        bank[s]=Mn
    for s in SY:
        if s not in bank: continue
        lp=LP[s]; n=len(lp); Mn=bank[s]
        for t in range(max(W+H+300,750),n-H,step):
            cur=lp[t-W+1:t+1]-lp[t-W+1]
            if volnorm:
                sd=cur.std(); cur=cur/(sd if sd>0 else 1)
            cands=[]
            src=[s] if pool=='self' else SY
            for u in src:
                if u not in bank: continue
                lu=LP[u]; Mu=bank[u]
                # window ending index e maps to row e-W+1 ; need e+H < len and (u!=s or e < t-W)
                emax=len(lu)-H-1
                elo=W-1
                ehi=(t-W-1) if u==s else emax
                ehi=min(ehi,emax)
                if ehi<=elo: continue
                rows=np.arange(elo,ehi+1)-W+1
                d=np.sqrt(((Mu[rows]-cur)**2).mean(axis=1))
                for e,dd in zip(np.arange(elo,ehi+1),d):
                    cands.append((dd,u,e))
            if len(cands)<K: continue
            cands.sort(key=lambda x:x[0])
            top=cands[:K]
            fwd=np.array([LP[u][e+H]-LP[u][e] for _,u,e in top])
            npos=(fwd>0).sum()
            dmin=top[0][0]
            if agree_gate is not None and not (npos>=agree_gate*K or npos<=(1-agree_gate)*K): continue
            if dist_gate is not None and dmin>dist_gate: continue
            p=fwd.mean(); a=lp[t+H]-lp[t]
            proj.append(p);act.append(a);hits.append(np.sign(p)==np.sign(a));base.append(a>0)
    if len(hits)<300: return None
    h=np.mean(hits)*100; b=np.mean(base)*100
    c=np.corrcoef(proj,act)[0,1] if np.std(proj)>0 else np.nan
    return dict(W=W,H=H,K=K,pool=pool,vn=volnorm,ag=agree_gate,dg=dist_gate,
                n=len(hits),hit=h,base=b,excess=h-b,corr=c)
res=[]
for W,H,K,pool,vn in itertools.product([60,120],[21,63],[20,50],['self','cross'],[False,True]):
    r=eval_cfg(W,H,K,pool,vn)
    if r: res.append(r); print(f"W{W} H{H} K{K} {pool:5s} vn={int(vn)}  n={r['n']:6d} hit {r['hit']:5.2f}% base {r['base']:5.2f}% excess {r['excess']:+5.2f}pp corr {r['corr']:+.3f}",flush=True)
pd.DataFrame(res).to_csv('/tmp/analog/sweep.csv',index=False)
R=pd.DataFrame(res).sort_values('excess',ascending=False)
print("\nBEST BY EXCESS HIT RATE:"); print(R.head(5).to_string(index=False))
print(f"\nconfigs with excess > 0: {(R.excess>0).sum()} of {len(R)}")
print(f"mean excess across all configs: {R.excess.mean():+.2f}pp")
