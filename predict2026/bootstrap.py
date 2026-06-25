import numpy as np, pandas as pd, copy
from predict2026.openf1 import OpenF1Client
from predict2026 import ingest, predictor
ds=ingest.build_dataset(OpenF1Client(cache_dir='cache/openf1_2026'))
sch=ds['schedule']; NR=int(((sch.kind=='race')&(~sch.done)).sum()); NS=int(((sch.kind=='sprint')&(~sch.done)).sum())
ROUNDS=sorted(ds['quali']['round'].unique())

def resample_ds(ds, samp):
    new=dict(ds)
    def remap(df):
        parts=[]
        for nr,orig in enumerate(samp,1):
            d=df[df['round']==orig].copy(); d['round']=nr; parts.append(d)
        return pd.concat(parts,ignore_index=True)
    new['quali']=remap(ds['quali']); new['pace']=remap(ds['pace']); new['results']=remap(ds['results'])
    return new

# reliability tornado on the SHIPPED model
prof=predictor.build_profiles(ds)
base=predictor.simulate(prof,NR,NS,n_sims=20000)[0]
b_ant=base[base.driver=='ANT']['WDC_prob_%'].values[0]
print(f"Baseline (new reliability model): ANT {b_ant:.0f}%")
# vary Antonelli's reliability marginal +/-
for mult in [0.5,2.0]:
    p=prof.copy(); p.loc[p.driver=='ANT','dnf_rate']=np.clip(p.loc[p.driver=='ANT','dnf_rate']*mult,0.02,0.6)
    a=predictor.simulate(p,NR,NS,n_sims=20000)[0]
    print(f"  Antonelli DNF x{mult}: ANT {a[a.driver=='ANT']['WDC_prob_%'].values[0]:.0f}%")

print("\nBOOTSTRAP (resample 7 races, 120x) - NEW reliability model:")
ant=[]
rng=np.random.default_rng(0)
for it in range(120):
    samp=list(rng.choice(ROUNDS,size=len(ROUNDS),replace=True))
    try:
        nds=resample_ds(ds,samp); pf=predictor.build_profiles(nds)
        r=predictor.simulate(pf,NR,NS,n_sims=4000,seed=it)[0]
        ant.append(r[r.driver=='ANT']['WDC_prob_%'].values[0])
    except Exception as e:
        pass
ant=np.array(ant)
print(f"  Antonelli WDC%: median {np.median(ant):.0f}, 10-90th pct [{np.percentile(ant,10):.0f} - {np.percentile(ant,90):.0f}]  (n={len(ant)})")
print(f"  OLD model band was [69 - 98] (spread {98-69}); NEW spread {np.percentile(ant,90)-np.percentile(ant,10):.0f}")
