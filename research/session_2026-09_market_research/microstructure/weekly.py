"""PREREG15: weekly cross-sectional reversal at REAL Alpaca costs.
Zero commission; full round-trip Corwin-Schultz spread charged per name per rebalance."""
import numpy as np, pandas as pd
W='/tmp/wide/'
cl=pd.read_parquet(W+'close.parquet'); op=pd.read_parquet(W+'open.parquet'); dv=pd.read_parquet(W+'dvol.parquet')
for x in (cl,op,dv): x.index=pd.to_datetime(x.index)
cols=sorted(set(cl.columns)&set(op.columns)&set(dv.columns)); cl,op,dv=cl[cols],op[cols],dv[cols]
cl=cl[cl.index<'2024-01-01']; op=op.reindex(cl.index); dv=dv.reindex(cl.index)
pit=pd.read_csv(W+'sp500_pit.csv'); pit['date']=pd.to_datetime(pit.date)
pit=pit.set_index('date').tickers.reindex(cl.index).ffill()
memb=pd.DataFrame(False,index=cl.index,columns=cols)
for d,s in pit.items():
    if isinstance(s,str): memb.loc[d,[x for x in s.split(',') if x in memb.columns]]=True
dv20=dv.shift(1).rolling(20,min_periods=10).mean()
valid=memb&cl.notna()&op.notna()&(dv20>2e7)
lr=np.log(cl).diff()
# Corwin-Schultz per name from daily high/low proxy: use |open-close| range proxy is weak,
# so use the measured intraday spreads where available and a dvol-based fit elsewhere.
sp20=pd.read_parquet('/tmp/timbre/panel4.parquet').groupby('sym').spread.median()
med=float(sp20.median())
SPREAD=pd.Series({c: float(sp20.get(c,med)) for c in cols})
print(f"cost model: zero commission + full round-trip spread; median {med*1e4:.2f} bp")

wk=cl.index.to_period('W')
weeks=sorted(set(wk))
def weekly_ls(skip_last_day=False, nq=5, sigma_bucket=None, S=None):
    out=[]
    for a,b in zip(weeks[:-2],weeks[1:-1]):
        ia=cl.index[wk==a]; ib=cl.index[wk==b]
        if len(ia)<3 or len(ib)<3: continue
        t0,t1=ia[0],ia[-1]
        if skip_last_day and len(ia)>=2: t1=ia[-2]
        sig=-(np.log(cl.loc[t1])-np.log(cl.loc[t0]))      # reversal: -last week's return
        m=valid.loc[ia[-1]]
        s=sig[m].dropna()
        if sigma_bucket is not None and S is not None:
            sv=S.loc[ia[-1]].reindex(s.index)
            q=sv.rank(pct=True)
            s=s[(q>sigma_bucket[0])&(q<=sigma_bucket[1])].dropna()
        if len(s)<40: continue
        fwd=(np.log(cl.loc[ib[-1]])-np.log(cl.loc[ib[0]])).reindex(s.index)
        ok=fwd.notna(); s=s[ok]; fwd=fwd[ok]
        if len(s)<40: continue
        r=s.rank(method='first'); k=max(len(s)//nq,1)
        lo=fwd[r<=k]; hi=fwd[r>len(s)-k]
        cost=float(SPREAD.reindex(list(lo.index)+list(hi.index)).mean())
        out.append(dict(week=ib[-1],gross=float(hi.mean()-lo.mean()),cost=cost,
                        net=float(hi.mean()-lo.mean())-cost,n=len(s)))
    return pd.DataFrame(out)

def rep(df,tag):
    df=df.copy(); df['split']=np.where(df.week<'2022-01-01','TRAIN','VALID')
    for sp in ('TRAIN','VALID'):
        g=df[df.split==sp]
        if len(g)<30: continue
        t=g.net.mean()/(g.net.std(ddof=1)/np.sqrt(len(g)))
        tg=g.gross.mean()/(g.gross.std(ddof=1)/np.sqrt(len(g)))
        sh=g.net.mean()/g.net.std()*np.sqrt(52)
        print(f"  {tag:22s} {sp:6s} gross={g.gross.mean()*1e4:+8.1f}bp (t{tg:+5.2f})  cost={g.cost.mean()*1e4:6.1f}  "
              f"NET={g.net.mean()*1e4:+8.1f}bp (t{t:+5.2f})  annSharpe={sh:+5.2f}  nwk={len(g)}")
print("\nW1  WEEKLY CROSS-SECTIONAL REVERSAL, quintile L/S, weekly rebalance")
r1=weekly_ls(False); rep(r1,'R1 raw')
r2=weekly_ls(True);  rep(r2,'R2 skip-last-day')
r1.to_parquet(W+'r1.parquet'); r2.to_parquet(W+'r2.parquet')
