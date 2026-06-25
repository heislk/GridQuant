import numpy as np, pandas as pd
from predict2026.openf1 import OpenF1Client
from predict2026 import ingest
from predict2026.config import RACE_POINTS,SPRINT_POINTS,RECENT_ROUNDS
ds=ingest.build_dataset(OpenF1Client(cache_dir='predict2026/cache_snapshot/openf1_2026'))
g=pd.DataFrame(ds["grid"].values()); num2a=dict(zip(g.driver_number,g.acronym)); num2t=dict(zip(g.driver_number,g.team))
q=ds["quali"]; pace=ds["pace"]; rr=ds["results"][ds["results"].kind=="race"].copy(); rr["driver"]=rr.driver_number.map(num2a)
wdc=ds["wdc"].copy(); wdc["driver"]=wdc.driver_number.map(num2a); pts=dict(zip(wdc.driver,wdc.points_current))
sch=ds["schedule"]; NR=int(((sch.kind=='race')&(~sch.done)).sum()); NS=int(((sch.kind=='sprint')&(~sch.done)).sum())
ROUNDS=sorted(q["round"].unique())

def build(P_rounds, sw, rw, wq,wr,wres, ppr_scale=0.05, dnf_mult=None):
    qd=q[q.driver.notna() & q["round"].isin(P_rounds)]; pd_=pace[pace.driver.notna() & pace["round"].isin(P_rounds)]; rd=rr[rr["round"].isin(P_rounds)]
    def bl(df,c):
        s=df.groupby("driver")[c].mean(); rec=sorted(df["round"].unique())[-RECENT_ROUNDS:]; r=df[df["round"].isin(rec)].groupby("driver")[c].mean()
        return pd.Series({i:sw*s[i]+rw*r.get(i,s[i]) for i in s.index})
    qg=bl(qd,"gap_pole"); rg=bl(pd_,"driver_gap_to_best_s")
    nd=rd["round"].nunique() or 1; ppr={d:pts.get(d,0)/nd for d in num2a.values()}; mppr=max(ppr.values())
    gm=max(0.03,rd.dnf.mean()); st=rd.groupby("driver").apply(lambda x:(~x.dns).sum(),include_groups=False)
    dnf=((rd.groupby("driver").dnf.sum()+5*gm)/(st+5)).to_dict()
    rows=[]
    for d in sorted(set(qg.index)|set(rg.index)):
        a=qg.get(d,np.nan); b=rg.get(d,np.nan)
        if np.isnan(a) and np.isnan(b): continue
        a=b if np.isnan(a) else a; b=a if np.isnan(b) else b
        resg=(mppr-ppr.get(d,0))*ppr_scale; num=next((n for n,x in num2a.items() if x==d),None)
        dr=dnf.get(d,gm)
        if dnf_mult and d in dnf_mult: dr=min(0.6,dr*dnf_mult[d])
        rows.append({"driver":d,"team":num2t.get(num),"qg":a,"strength":wq*a+wr*b+wres*resg,"dnf":dr,"pts":pts.get(d,0)})
    return pd.DataFrame(rows)

def sim(P,k,dev,grid,sigr,teamr,sigq,nsims=12000,seed=7):
    D=len(P); rng=np.random.default_rng(seed); teams=P.team.values; ut=sorted(set(teams)); tix={t:i for i,t in enumerate(ut)}; ts=np.array([tix[t] for t in teams])
    rpv=np.zeros(D); [rpv.__setitem__(p-1,v) for p,v in RACE_POINTS.items()]; spv=np.zeros(D); [spv.__setitem__(p-1,v) for p,v in SPRINT_POINTS.items()]
    tot=np.tile(P.pts.values.astype(float),(nsims,1)); cw=np.zeros((nsims,len(ut)))
    def ev(n,sg,pv):
        nonlocal tot,cw
        for _ in range(n):
            cw=cw+rng.normal(0,dev,(nsims,len(ut))); dv=cw[:,ts]
            qs=-(P.qg.values[None,:]+dv+rng.normal(0,teamr,(nsims,len(ut)))[:,ts])/sigq+rng.normal(0,1,(nsims,D)); gr=np.argsort(-qs,1).argsort(1)+1
            sc=-k*(P.strength.values[None,:]+dv)-grid*gr+rng.normal(0,teamr,(nsims,len(ut)))[:,ts]+rng.normal(0,sg,(nsims,D))
            sc=np.where(rng.random((nsims,D))<P.dnf.values[None,:],-1e9,sc)
            o=np.argsort(-sc,1); rk=np.empty_like(o); np.put_along_axis(rk,o,np.broadcast_to(np.arange(D),o.shape),1); tot+=pv[rk]
    ev(NR,sigr,rpv); ev(NS,sigr+0.15,spv)
    ch=np.argmax(tot+rng.uniform(0,1e-6,tot.shape),1); w=np.bincount(ch,minlength=D)/nsims
    return dict(zip(P.driver,100*w))

# baseline (shipped config)
base=dict(sw=0.50,rw=0.50,wq=0.50,wr=0.30,wres=0.20,k=2.1,dev=0.07,grid=0.12,sigr=1.0,teamr=0.30,sigq=0.12)
P=build(ROUNDS,base['sw'],base['rw'],base['wq'],base['wr'],base['wres'])
b=sim(P,base['k'],base['dev'],base['grid'],base['sigr'],base['teamr'],base['sigq'])
print(f"BASELINE: ANT {b['ANT']:.0f}  HAM {b['HAM']:.0f}  RUS {b['RUS']:.0f}\n")
print("TORNADO - how far the headline (Antonelli %) swings per assumption:")
print(f"{'assumption swept':40} {'ANT% range':16} swing")
def run_var(**kw):
    p=dict(base,**kw)
    P=build(ROUNDS,p['sw'],p['rw'],p['wq'],p['wr'],p['wres'])
    return sim(P,p['k'],p['dev'],p['grid'],p['sigr'],p['teamr'],p['sigq'])['ANT']
tests={
 "Development uncertainty (dev .04->.14)":[dict(dev=0.04),dict(dev=0.14)],
 "Pace determinism k (1.6->2.6)":[dict(k=1.6),dict(k=2.6)],
 "Strength: quali-heavy vs results-heavy":[dict(wq=0.7,wr=0.2,wres=0.1),dict(wq=0.3,wr=0.2,wres=0.5)],
 "Recency of pace (season-only vs recent-heavy)":[dict(sw=0.8,rw=0.2),dict(sw=0.2,rw=0.8)],
 "Race-day noise sigma (0.7->1.4)":[dict(sigr=0.7),dict(sigr=1.4)],
 "Grid stickiness (0.0->0.25)":[dict(grid=0.0),dict(grid=0.25)],
}
for name,(lo,hi) in tests.items():
    a=run_var(**lo); c=run_var(**hi); rng=abs(c-a)
    print(f"  {name:40} {min(a,c):4.0f} - {max(a,c):4.0f}      {rng:4.0f}")

# Antonelli reliability sensitivity
P_lo=build(ROUNDS,base['sw'],base['rw'],base['wq'],base['wr'],base['wres'],dnf_mult={'ANT':0.3})
P_hi=build(ROUNDS,base['sw'],base['rw'],base['wq'],base['wr'],base['wres'],dnf_mult={'ANT':2.0})
alo=sim(P_lo,base['k'],base['dev'],base['grid'],base['sigr'],base['teamr'],base['sigq'])['ANT']
ahi=sim(P_hi,base['k'],base['dev'],base['grid'],base['sigr'],base['teamr'],base['sigq'])['ANT']
print(f"  {'Antonelli reliability (3x better/2x worse)':40} {min(alo,ahi):4.0f} - {max(alo,ahi):4.0f}      {abs(ahi-alo):4.0f}")

# ---- RACE BOOTSTRAP: how much does having only 7 races limit certainty? ----
print("\nRACE BOOTSTRAP (resample the 7 completed races w/ replacement, 200x):")
ant=[]; ham=[]
rng0=np.random.default_rng(0)
for it in range(200):
    samp=list(rng0.choice(ROUNDS,size=len(ROUNDS),replace=True))
    P=build(samp,base['sw'],base['rw'],base['wq'],base['wr'],base['wres'])
    o=sim(P,base['k'],base['dev'],base['grid'],base['sigr'],base['teamr'],base['sigq'],nsims=4000,seed=it)
    ant.append(o.get('ANT',0)); ham.append(o.get('HAM',0))
ant=np.array(ant); ham=np.array(ham)
print(f"  Antonelli WDC%: median {np.median(ant):.0f}, 10-90th pct [{np.percentile(ant,10):.0f} - {np.percentile(ant,90):.0f}]")
print(f"  Hamilton  WDC%: median {np.median(ham):.0f}, 10-90th pct [{np.percentile(ham,10):.0f} - {np.percentile(ham,90):.0f}]")
print("  => the point estimate (77%) carries a wide band purely from 7-race sampling noise.")
