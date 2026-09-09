import numpy as np, pandas as pd, importlib.util, sys
spec=importlib.util.spec_from_file_location('pb','/tmp/pbhl/pb.py'); pb=importlib.util.module_from_spec(spec)
sys.argv=['x']; spec.loader.exec_module(pb)
C=pb.C
PB20={'AAVE','ADA','AVAX','BCH','BNB','BTC','DOGE','ETH','FIL','LINK','LTC','NEAR','SOL','SUI','TRX','UNI','WLD','XRP','ZEC','kPEPE'}
U=sorted(PB20&set(C.columns))
print(f"{'risk/trade':>10s} {'maxRisk':>8s} {'CAGR%':>7s} {'Sharpe':>7s} {'maxDD%':>7s} {'n':>5s}",flush=True)
for rf,mr in ((0.0025,0.015),(0.0075,0.045),(0.0125,0.075),(0.02,0.12),(0.03,0.18)):
    E,T=pb.run(universe=U,start='2021-01-01',risk_frac=rf,max_risk=mr,gross_cap=6.0)
    s=pb.stats(E,T,'x')
    print(f"{rf:10.4f} {mr:8.3f} {s['cagr']:+7.1f} {s['sh']:+7.2f} {s['dd']:+7.1f} {s['n']:5d}",flush=True)
