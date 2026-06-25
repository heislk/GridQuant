import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_DATA = _os.path.join(_ROOT, "predict2026", "data", "sim_viz.json")
_MEDIA = _os.path.join(_ROOT, "media"); _os.makedirs(_os.path.join(_MEDIA, "previews"), exist_ok=True)

import json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
d=json.load(open(_DATA))
N=d["n_sims"]; TR=np.array(d["traj_all"]); WIN=np.array(d["winners"]); FA=np.array(d["final_ant"]); FH=np.array(d["final_ham"]); EV=d["events"]
ANT=(22/255,184/255,148/255); HAM=(226/255,75/255,74/255); OTH=(154/255,152/255,140/255)
def lerp(c1,c2,t): return tuple(c1[i]+(c2[i]-c1[i])*t for i in range(3))
antl=(159/255,225/255,203/255); antd=(15/255,110/255,86/255); haml=(247/255,193/255,193/255); hamd=(163/255,45/255,45/255)

margin=np.where(WIN==0,FA-FH,np.where(WIN==1,FH-FA,0))
order=np.lexsort((-margin, WIN))  # winner asc, margin desc
COLS,ROWS=200,100
img=np.ones((ROWS,COLS,3))
mm={0:max(1,margin[WIN==0].max()),1:max(1,margin[WIN==1].max())}
for k,sim in enumerate(order[:COLS*ROWS]):
    r,c=k//COLS,k%COLS; w=WIN[sim]
    if w==0: img[r,c]=lerp(antl,antd,0.15+0.85*min(1,margin[sim]/mm[0]))
    elif w==1: img[r,c]=lerp(haml,hamd,0.15+0.85*min(1,margin[sim]/mm[1]))
    else: img[r,c]=(0.81,0.80,0.77)

fig=plt.figure(figsize=(11,8.4),dpi=120); gs=GridSpec(2,1,height_ratios=[1,1.15],hspace=0.28)
ax1=fig.add_subplot(gs[0]); ax1.imshow(img,aspect='auto',interpolation='nearest')
ax1.set_title("Every one of the 20,000 simulated seasons - one cell each (sorted by champion & margin)",fontsize=12,fontweight='bold')
ax1.set_xticks([]); ax1.set_yticks([])
a,h,o=(WIN==0).mean()*100,(WIN==1).mean()*100,(WIN==2).mean()*100
ax1.text(0.5,-0.06,f"Antonelli {a:.1f}%   ·   Hamilton {h:.1f}%   ·   field {o:.1f}%",transform=ax1.transAxes,ha='center',fontsize=11)

ax2=fig.add_subplot(gs[1]); nE=len(EV); x=np.arange(nE+1)
samp=np.random.default_rng(1).choice(N,3000,replace=False)
for i in samp:
    c=ANT if WIN[i]==0 else HAM if WIN[i]==1 else OTH
    ax2.plot(x,TR[i],color=c,alpha=0.035,lw=0.8)
p10=np.percentile(TR,10,0); p50=np.percentile(TR,50,0); p90=np.percentile(TR,90,0)
ax2.fill_between(x,p10,p90,color='k',alpha=0.07); ax2.plot(x,p50,'k-',lw=2,label='median')
ax2.axhline(0,color='k',ls='--',lw=1.2)
ax2.text(0.3,3,"Antonelli ahead",color=antd,fontsize=9); ax2.text(0.3,-14,"Hamilton ahead",color=hamd,fontsize=9)
ax2.scatter([0],[d["start_lead"]],color='k',zorder=5,s=25)
ax2.set_xticks(x[1::2]); ax2.set_xticklabels(EV[::2],rotation=45,ha='right',fontsize=8)
ax2.set_ylabel("Antonelli's points lead over Hamilton"); ax2.set_xlim(-0.3,nE+0.3); ax2.set_ylim(max(-160,TR.min()),min(320,TR.max()))
ax2.set_title("Each season is itself 18 race simulations - 3,000 of the 20,000 lead-trajectories",fontsize=12,fontweight='bold')
ax2.legend(loc='upper left',fontsize=9)
fig.savefig(_os.path.join(_MEDIA, "previews", "simulations_overview.png"),bbox_inches='tight',facecolor='white'); print("saved sim_preview.png")
