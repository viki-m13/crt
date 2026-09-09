import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
D='/tmp/databranch/data/daily_multiasset/'
def ld(s):
    d=pd.read_parquet(D+f'{s}.parquet'); d['d']=pd.to_datetime(d['d']); return d.set_index('d')['adj'].sort_index()
spy=ld('SPY'); tlt=ld('TLT'); ief=ld('IEF'); gld=ld('GLD'); qqq=ld('QQQ')
vix=ld('IDX_VIX'); v3m=ld('IDX_VIX3M'); irx=ld('IDX_IRX')
px=pd.concat([spy,tlt,ief,gld,qqq],axis=1); px.columns=['SPY','TLT','IEF','GLD','QQQ']
px=px[px.index>='1993-02-01']
r=np.log(px).diff()
cash=(irx.reindex(px.index).ffill()/100/252).fillna(0)     # T-bill daily
COST=0.00027
def stats(e,lab):
    e=e.dropna(); eq=(1+np.expm1(e)).cumprod(); yrs=len(e)/252
    dd=(eq/eq.cummax()-1)
    return dict(lab=lab,cagr=(eq.iloc[-1]**(1/yrs)-1)*100,sh=e.mean()/e.std()*np.sqrt(252),
                dd=dd.min()*100,eq=eq,fin=eq.iloc[-1])
out={}
# 1 SPY buy & hold
out['SPY buy & hold']=r.SPY
# 2 200d MA timing on SPY, else T-bills
ma=px.SPY.rolling(200).mean(); sig=(px.SPY>ma).shift(1).fillna(False)
out['SPY 200d MA + T-bills']=np.where(sig,r.SPY,cash)-np.abs(sig.astype(float).diff().fillna(0))*COST
# 3 vol-target (unlevered, cap 100%)
rv=r.SPY.rolling(20).std()*np.sqrt(252)
w=(0.15/rv).clip(0,1.0).shift(1).fillna(0)
out['SPY vol-target (cap 1x)']=w*r.SPY+(1-w)*cash-np.abs(w.diff().fillna(0))*COST
# 4 dual momentum SPY/TLT/cash, monthly
me=px.index.to_series().groupby(px.index.to_period('M')).last()
pos=pd.Series(0.0,index=px.index); hold=None; ser=[]
mom=px.pct_change(252)
choice=pd.Series(index=px.index,dtype=object)
for i in range(1,len(me)):
    t=me.iloc[i]; prev=me.iloc[i-1]
    m_s=mom.SPY.get(prev,np.nan); m_t=mom.TLT.get(prev,np.nan)
    pick='SPY' if (m_s>0 and (np.isnan(m_t) or m_s>=m_t)) else ('TLT' if (not np.isnan(m_t) and m_t>0) else 'CASH')
    nxt=me.iloc[i+1] if i+1<len(me) else px.index[-1]
    choice.loc[(px.index>t)&(px.index<=nxt)]=pick
choice=choice.ffill().fillna('CASH')
dm=np.where(choice=='SPY',r.SPY,np.where(choice=='TLT',r.TLT,cash))
turn=(choice!=choice.shift()).astype(float)
out['dual momentum SPY/TLT/cash']=pd.Series(dm,index=px.index)-turn*COST
res=[stats(pd.Series(v,index=px.index) if not isinstance(v,pd.Series) else v,k) for k,v in out.items()]
print(f"{'strategy':30s} {'CAGR%':>7s} {'Sharpe':>7s} {'maxDD%':>8s} {'$1 ->':>9s}")
for s in res: print(f"{s['lab']:30s} {s['cagr']:+7.2f} {s['sh']:+7.2f} {s['dd']:+8.1f} {s['fin']:9.1f}x")
fig,ax=plt.subplots(2,1,figsize=(13,9),gridspec_kw={'height_ratios':[3,1]},sharex=True)
cols={'SPY buy & hold':'#888','SPY 200d MA + T-bills':'#0b7','SPY vol-target (cap 1x)':'#07b','dual momentum SPY/TLT/cash':'#b30'}
for s in res:
    ax[0].plot(s['eq'].index,s['eq'],lw=1.6,color=cols[s['lab']],label=f"{s['lab']}  CAGR {s['cagr']:.1f}%  Sh {s['sh']:.2f}  DD {s['dd']:.0f}%")
    dd=s['eq']/s['eq'].cummax()-1
    ax[1].plot(dd.index,dd*100,lw=1.1,color=cols[s['lab']])
ax[0].set_yscale('log'); ax[0].set_ylabel('growth of $1 (log)'); ax[0].legend(fontsize=9,loc='upper left')
ax[0].set_title('Unlevered, long-only, no shorting/borrow — vs SPY buy & hold, 1993-2026 (costs 2.7bp/trade)')
ax[0].grid(alpha=.3); ax[1].grid(alpha=.3); ax[1].set_ylabel('drawdown %'); ax[1].set_xlabel('')
plt.tight_layout(); plt.savefig('/tmp/lo/curve.png',dpi=130)
print('\nsaved /tmp/lo/curve.png')
