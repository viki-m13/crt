import numpy as np, pandas as pd
E=pd.read_parquet('/tmp/venues/events10.parquet')
E['split']=np.where(E.day<'2025-01-01','TRAIN','VALID')
COST=0.0010          # 10 bp round trip, taker both ways
RNG=np.random.default_rng(31337)
for k in (15,30,60): E[f'net{k}']=E[f'rev{k}']-COST
E['Dr']=E.groupby(['sym','day']).D.transform(lambda s: s.rank(pct=True))
E['q']=pd.cut(E.Dr,[0,.2,.4,.6,.8,1.0],labels=[1,2,3,4,5])
def dayt(df,col):
    dm=df.groupby('day')[col].mean()
    return dm.mean(), dm.mean()/(dm.std(ddof=1)/np.sqrt(len(dm))), len(dm)
print(f"{len(E):,} events | {E.day.nunique()} days | {E.sym.nunique()} symbols")
print("\nH1  REVERSAL PAYOFF BY DISSIPATION-PER-FLOW QUINTILE (net of 10bp), bp/trade")
print(f"{'split':6s} {'D quintile':11s} {'net15':>9s} {'net30':>9s} {'net60':>9s} {'n':>9s}")
for sp in ('TRAIN','VALID'):
    for qq in (1,2,3,4,5):
        g=E[(E.split==sp)&(E.q==qq)]
        lab='1 LOW-D' if qq==1 else ('5 HIGH-D' if qq==5 else str(qq))
        print(f"{sp:6s} {lab:11s} "+" ".join(f"{g[f'net{k}'].mean()*1e4:+9.2f}" for k in (15,30,60))+f" {len(g):9,}")
print("\nH3  SPREAD: low-D quintile minus high-D quintile (gross, cost cancels)")
for sp in ('TRAIN','VALID'):
    for k in (15,30,60):
        lo=E[(E.split==sp)&(E.q==1)].groupby('day')[f'rev{k}'].mean()
        hi=E[(E.split==sp)&(E.q==5)].groupby('day')[f'rev{k}'].mean()
        d=(lo-hi).dropna()
        t=d.mean()/(d.std(ddof=1)/np.sqrt(len(d)))
        print(f"  {sp} k={k:3d}  spread={d.mean()*1e4:+7.2f} bp  t={t:+6.2f}  ndays={len(d)}")
print("\nH2  ORTHOGONALITY: does D survive residualizing on lambda, vol, trade size, volume?")
V=E[E.split=='VALID'].copy()
for c in ('lam','vol','tsize','volume'):
    V['r_'+c]=V.groupby(['sym','day'])[c].transform(lambda s: s.rank(pct=True))
X=V[['r_lam','r_vol','r_tsize','r_volume']].to_numpy()
X=np.column_stack([X,np.ones(len(X))])
y=V.Dr.to_numpy()
b,*_=np.linalg.lstsq(X,y,rcond=None); V['Dres']=y-X@b
print(f"  R^2 of D on the four controls: {1-V.Dres.var()/V.Dr.var():.4f}")
V['qres']=pd.cut(V.groupby(['sym','day']).Dres.transform(lambda s:s.rank(pct=True)),[0,.2,.4,.6,.8,1.0],labels=[1,2,3,4,5])
for k in (15,30,60):
    lo=V[V.qres==1].groupby('day')[f'rev{k}'].mean(); hi=V[V.qres==5].groupby('day')[f'rev{k}'].mean()
    d=(lo-hi).dropna(); t=d.mean()/(d.std(ddof=1)/np.sqrt(len(d)))
    print(f"  residualized D, k={k:3d}: spread={d.mean()*1e4:+7.2f} bp  t={t:+6.2f}")
print("\n  control: lambda alone (the standard discriminator), VALID")
V['ql']=pd.cut(V.r_lam,[0,.2,.4,.6,.8,1.0],labels=[1,2,3,4,5])
for k in (15,30,60):
    lo=V[V.ql==1].groupby('day')[f'rev{k}'].mean(); hi=V[V.ql==5].groupby('day')[f'rev{k}'].mean()
    d=(lo-hi).dropna(); t=d.mean()/(d.std(ddof=1)/np.sqrt(len(d)))
    print(f"  lambda,      k={k:3d}: spread={d.mean()*1e4:+7.2f} bp  t={t:+6.2f}")
print("\nH5  PER-SYMBOL (VALID, k=30, low-D minus high-D)")
for s,g in E[E.split=='VALID'].groupby('sym'):
    lo=g[g.q==1].groupby('day').rev30.mean(); hi=g[g.q==5].groupby('day').rev30.mean()
    d=(lo-hi).dropna()
    if len(d)>30: print(f"  {s:10s} {d.mean()*1e4:+7.2f} bp  t={d.mean()/(d.std(ddof=1)/np.sqrt(len(d))):+6.2f}")
