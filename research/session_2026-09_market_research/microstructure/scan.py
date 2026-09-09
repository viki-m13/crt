"""PREREG12: Adverse-Selection Efficiency scanner. Identical measurement across
venues at matched 15-min bars. Fills simulated against actual high/low."""
import numpy as np, pandas as pd, glob
from pathlib import Path
DELTAS=[10e-4,25e-4,50e-4]; HOLDS=[1,2,4]

def ase_for(o,h,l,c,label,venue,cls=''):
    c=np.asarray(c,float); h=np.asarray(h,float); l=np.asarray(l,float)
    ok=np.isfinite(c)&np.isfinite(h)&np.isfinite(l)&(c>0)
    c,h,l=c[ok],h[ok],l[ok]
    n=len(c)
    if n<400: return []
    out=[]
    half=n//2
    for d in DELTAS:
        Ls=c*(1+d); Lb=c*(1-d)
        i=np.arange(n-6)
        fs=h[i+1]>Ls[i]            # sell filled
        fb=l[i+1]<Lb[i]            # buy filled
        for k in HOLDS:
            ex=c[i+1+k]
            pnl_s=(Ls[i]-ex)/Ls[i]; pnl_b=(ex-Lb[i])/Lb[i]
            filled=np.concatenate([pnl_s[fs],pnl_b[fb]])
            uncond=np.concatenate([pnl_s,pnl_b])
            if len(filled)<200: continue
            fm=filled.mean(); um=uncond.mean()
            adv=um-fm; ase=adv/d
            # E4 stability: first vs second half
            m1=i<half
            f1=np.concatenate([pnl_s[fs&m1],pnl_b[fb&m1]]); u1=np.concatenate([pnl_s[m1],pnl_b[m1]])
            f2=np.concatenate([pnl_s[fs&~m1],pnl_b[fb&~m1]]); u2=np.concatenate([pnl_s[~m1],pnl_b[~m1]])
            a1=(u1.mean()-f1.mean())/d if len(f1)>100 else np.nan
            a2=(u2.mean()-f2.mean())/d if len(f2)>100 else np.nan
            se=filled.std()/np.sqrt(len(filled))
            out.append(dict(venue=venue,sym=label,cls=cls,delta=d*1e4,k=k,
                n=n,nfill=len(filled),fillrate=(fs.mean()+fb.mean())/2,
                filled_bp=fm*1e4,uncond_bp=um*1e4,adv_bp=adv*1e4,ASE=ase,
                t_filled=fm/se if se>0 else np.nan,ase_h1=a1,ase_h2=a2))
    return out

rows=[]
# --- builder DEX (native 15m)
for f in sorted(glob.glob('/tmp/ase/bx_*_15m.parquet')):
    d=pd.read_parquet(f); sym=Path(f).stem.replace('bx_xyz_','').replace('_15m','')
    rows+=ase_for(d.o,d.h,d.l,d.c,sym,'builder-dex',d.cls.iloc[0])
# --- Binance perps 1m -> 15m  (CALIBRATION NULL)
for f in sorted(glob.glob('/tmp/venues/c1m_*.parquet')):
    d=pd.read_parquet(f,columns=['t','o','h','l','c']); sym=Path(f).stem[4:]
    d['t']=pd.to_datetime(d.t,unit='ms',utc=True); d=d.set_index('t')
    g=d.resample('15min').agg(o=('o','first'),h=('h','max'),l=('l','min'),c=('c','last')).dropna()
    rows+=ase_for(g.o,g.h,g.l,g.c,sym,'binance-perp','crypto')
# --- US equities 1m -> 15m
for f in sorted(Path('/tmp/databranch/data/equity_1m_alpaca').glob('*.parquet')):
    d=pd.read_parquet(f,columns=['t','o','h','l','c'])
    d['t']=pd.to_datetime(d.t,utc=True).dt.tz_convert('America/New_York')
    d=d[(d.t.dt.time>=pd.Timestamp('09:30').time())&(d.t.dt.time<=pd.Timestamp('16:00').time())]
    d=d.set_index('t').resample('15min').agg(o=('o','first'),h=('h','max'),l=('l','min'),c=('c','last')).dropna()
    rows+=ase_for(d.o,d.h,d.l,d.c,f.stem,'us-equity','equity')
T=pd.DataFrame(rows); T.to_parquet('/tmp/ase/ase.parquet',index=False)
print(f"{len(T)} rows, {T.sym.nunique()} assets\n")
print("E1  CALIBRATION -- does the scanner recover the known null on Binance majors?")
b=T[T.venue=='binance-perp']
print(b.groupby('delta').ASE.agg(['mean','median','std']).round(3).to_string())
print(f"  overall median ASE = {b.ASE.median():.3f}   (E1 needs 0.85-1.15)")
print()
print("E2  ASE BY VENUE (median across assets, all deltas/holds)")
print(T.groupby('venue').ASE.agg(n=('size'),median=('median'),mean=('mean'),
      q10=lambda s:s.quantile(.10)).round(3).to_string())
print()
print("     by builder-dex asset class:")
print(T[T.venue=='builder-dex'].groupby('cls').ASE.agg(n='size',median='median',q10=lambda s:s.quantile(.10)).round(3).to_string())
