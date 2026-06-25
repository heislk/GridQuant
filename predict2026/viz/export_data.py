"""Run the model and export per-simulation data for the visualization."""

import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_DATA = _os.path.join(_ROOT, "predict2026", "data", "sim_viz.json")
_MEDIA = _os.path.join(_ROOT, "media"); _os.makedirs(_os.path.join(_MEDIA, "previews"), exist_ok=True)
_CACHE = _os.path.join(_ROOT, "predict2026", "cache_snapshot", "openf1_2026")

import numpy as np, pandas as pd, json
from scipy.stats import norm
from predict2026.openf1 import OpenF1Client
from predict2026 import ingest, predictor
from predict2026.config import (RACE_POINTS, SPRINT_POINTS, PACE_TO_UTILITY, SIGMA_RACE,
    SIGMA_QUALI_S, SIGMA_SPRINT_EXTRA, GRID_STICKINESS, TEAM_RACE_SIGMA, DEV_STEP_S)
from predict2026.reliability import TEAMMATE_RHO

ds=ingest.build_dataset(OpenF1Client(cache_dir=_CACHE))
sch=ds['schedule'].sort_values('date')
prof=predictor.build_profiles(ds).reset_index(drop=True)
D=len(prof); idx={d:i for i,d in enumerate(prof.driver)}
mu=(-PACE_TO_UTILITY*prof.strength_s.values).astype(float)  # base; dev added in loop
quali_gap=prof.quali_gap_s.values.astype(float); strength=prof.strength_s.values.astype(float)
dnf=prof.dnf_rate.values.astype(float); start=prof.points_now.values.astype(float)
teams=prof.team.values; ut=sorted(set(teams)); tix={t:i for i,t in enumerate(ut)}; tsel=np.array([tix[t] for t in teams])
thr=norm.ppf(1-np.clip(dnf,1e-4,0.99)); rho=TEAMMATE_RHO
rp=np.zeros(D); [rp.__setitem__(p-1,v) for p,v in RACE_POINTS.items()]
spv=np.zeros(D); [spv.__setitem__(p-1,v) for p,v in SPRINT_POINTS.items()]

# remaining events in chronological order
rem=sch[~sch.done][['country','location','kind','round']].to_dict('records')
N=20000; rng=np.random.default_rng(20260623)
tot=np.tile(start,(N,1)); cw=np.zeros((N,len(ut)))
ia,ih=idx['ANT'],idx['HAM']
CODE={"Austria":"AUT","United Kingdom":"GBR","Belgium":"BEL","Hungary":"HUN","Netherlands":"NED",
      "Italy":"ITA","Spain":"ESP","Azerbaijan":"AZE","Singapore":"SGP","United States":"USA",
      "Mexico":"MEX","Brazil":"BRA","Qatar":"QAT","United Arab Emirates":"UAE"}
LOC={"Austin":"USA-Austin","Las Vegas":"USA-Vegas","Miami":"USA-Miami"}
def label(ev):
    base=LOC.get(ev.get("location","")) or CODE.get(ev["country"], ev["country"][:3].upper())
    return base+("\u00b7S" if ev["kind"]=="sprint" else "")
event_labels=[]; lead_traj=[ (tot[:,ia]-tot[:,ih]).copy() ]
def run(ev):
    global tot,cw
    sig = SIGMA_RACE + (SIGMA_SPRINT_EXTRA if ev['kind']=='sprint' else 0)
    pv = spv if ev['kind']=='sprint' else rp
    cw = cw + rng.normal(0,DEV_STEP_S,(N,len(ut))); dev=cw[:,tsel]
    qs=-(quali_gap[None,:]+dev+rng.normal(0,TEAM_RACE_SIGMA,(N,len(ut)))[:,tsel])/SIGMA_QUALI_S+rng.normal(0,1,(N,D))
    grid=(-qs).argsort(1).argsort(1)+1
    sc=-PACE_TO_UTILITY*(strength[None,:]+dev)-GRID_STICKINESS*grid+rng.normal(0,TEAM_RACE_SIGMA,(N,len(ut)))[:,tsel]+rng.normal(0,sig,(N,D))
    zt=rng.normal(0,1,(N,len(ut)))[:,tsel]; u=rng.normal(0,1,(N,D)); lat=np.sqrt(rho)*zt+np.sqrt(1-rho)*u
    sc=np.where(lat>thr[None,:],-1e9,sc)
    o=np.argsort(-sc,1); rk=np.empty_like(o); np.put_along_axis(rk,o,np.broadcast_to(np.arange(D),o.shape),1)
    tot+=pv[rk]
for ev in rem:
    run(ev); event_labels.append(label(ev))
    lead_traj.append((tot[:,ia]-tot[:,ih]).copy())

champ=np.argmax(tot+rng.uniform(0,1e-6,tot.shape),1)
lead_traj=np.array(lead_traj).T  # (N, n_events+1)

# winner code per sim: 0=ANT,1=HAM,2=other
wcode=np.where(champ==ia,0,np.where(champ==ih,1,2)).astype(int)
# export: all champions/winners + final pts; sample of trajectories
traj_all=np.rint(lead_traj).astype(int)
out={
  "n_sims": int(N),
  "events": event_labels,
  "start_lead": float(start[ia]-start[ih]),
  "winners": wcode.tolist(),
  "final_ant": tot[:,ia].astype(int).tolist(),
  "final_ham": tot[:,ih].astype(int).tolist(),
  "champ_name": [prof.driver.values[c] for c in champ[:0]], # placeholder
  "traj_all": traj_all.tolist(),
  "counts": {"ANT":int((wcode==0).sum()),"HAM":int((wcode==1).sum()),"other":int((wcode==2).sum())},
  "ant_final_med": int(np.median(tot[:,ia])), "ham_final_med": int(np.median(tot[:,ih])),
}
# top-line: distribution of who wins among 'other'
other_mask=wcode==2
if other_mask.any():
    oc=pd.Series(champ[other_mask]).map({i:d for d,i in idx.items()}).value_counts().to_dict()
    out["other_breakdown"]={k:int(v) for k,v in oc.items()}
json.dump(out, open(_DATA,"w"))
print("exported", N, "sims |", len(event_labels), "events | winners ANT/HAM/other:", out["counts"])
print("events:", event_labels)
