"""Generate a self-contained interactive HTML visualization of all 20k simulations."""

import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_DATA = _os.path.join(_ROOT, "predict2026", "data", "sim_viz.json")
_MEDIA = _os.path.join(_ROOT, "media"); _os.makedirs(_os.path.join(_MEDIA, "previews"), exist_ok=True)
_CACHE = _os.path.join(_ROOT, "predict2026", "cache_snapshot", "openf1_2026")

import json

d = json.load(open(_DATA))
DATA = json.dumps(d, separators=(",", ":"))

HTML = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>20,000 simulated 2026 F1 seasons</title>
<style>
:root{--ant:#16B894;--ant-d:#0F6E56;--ant-l:#9FE1CB;--ham:#E24B4A;--ham-d:#A32D2D;--ham-l:#F7C1C1;
 --oth:#9A988F;--ink:#14161c;--muted:#6b7280;--line:#e6e6ea;--bg:#ffffff;--panel:#f7f8fa;--gold:#E8A317;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;line-height:1.5}
.wrap{max-width:1060px;margin:0 auto;padding:28px 20px 60px}
h1{font-size:25px;font-weight:600;margin:0 0 2px}
.sub{color:var(--muted);font-size:14px;margin:0 0 20px}
h2{font-size:18px;font-weight:600;margin:34px 0 4px}
.hint{color:var(--muted);font-size:13px;margin:0 0 12px}
.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:6px 0 8px}
.card{background:var(--panel);border-radius:12px;padding:14px 16px}
.card .lab{font-size:13px;color:var(--muted)}
.card .val{font-size:26px;font-weight:600;margin-top:2px}
.canvas-box{position:relative;width:100%}
canvas{display:block;width:100%;height:auto;border-radius:10px}
.layer{position:absolute;left:0;top:0;pointer-events:none;border-radius:0}
#mtip{position:absolute;pointer-events:none;background:#14161c;color:#fff;font-size:12px;padding:6px 9px;
 border-radius:7px;opacity:0;transform:translate(-50%,-115%);white-space:nowrap;z-index:5}
.controls{display:flex;flex-wrap:wrap;gap:10px;align-items:center;margin:10px 0}
.chip{display:inline-flex;align-items:center;gap:6px;font-size:13px;border:1px solid var(--line);
 border-radius:20px;padding:5px 11px;cursor:pointer;user-select:none;background:#fff}
.chip.off{opacity:.4}
.sw{width:11px;height:11px;border-radius:3px;display:inline-block}
button{font:inherit;font-size:13px;border:1px solid var(--line);background:#fff;border-radius:20px;
 padding:6px 13px;cursor:pointer}
button:hover{background:var(--panel)}
.readout{background:var(--panel);border-radius:10px;padding:12px 14px;font-size:13.5px;margin-top:12px;min-height:42px}
.readout b{font-weight:600}
.axis{color:var(--muted);font-size:11px}
.foot{color:var(--muted);font-size:12px;margin-top:30px;border-top:1px solid var(--line);padding-top:12px}
.sr-only{position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0,0,0,0)}
</style></head><body><div class="wrap">
<h2 class="sr-only">Interactive visualization of 20,000 Monte-Carlo simulations of the 2026 F1 championship, each shown individually.</h2>

<h1>20,000 simulated 2026 seasons</h1>
<p class="sub">Every cell and every line below is one full Monte-Carlo season, the remaining 15 Grands&nbsp;Prix + 3 sprints, run from today's standings. As&nbsp;of 2026-06-23.</p>

<div class="cards" id="cards"></div>

<h2>Every simulation, one cell</h2>
<p class="hint">All 20,000 seasons, sorted by who wins and by final margin. Hover to inspect a season &middot; click a cell to trace it on the chart below.</p>
<div class="canvas-box" id="mosaicBox">
  <canvas id="mosaic" role="img" aria-label="Mosaic of 20,000 simulated seasons coloured by champion"></canvas>
  <canvas id="mosaicHi" class="layer"></canvas>
  <div id="mtip"></div>
</div>

<h2>Each season is 18 races deep</h2>
<p class="hint">Antonelli's points lead over Hamilton as each remaining round is simulated. Above the line Antonelli leads; cross below and Hamilton's in front. Each faint line is one season.</p>
<div class="controls" id="legend"></div>
<div class="canvas-box" id="spagBox">
  <canvas id="spag" role="img" aria-label="Ensemble of simulated championship-lead trajectories across the remaining races"></canvas>
  <canvas id="spagHi" class="layer"></canvas>
</div>
<div class="controls">
  <button id="randBtn"><i></i>Trace a random season</button>
  <button id="clearBtn">Clear trace</button>
</div>
<div class="readout" id="readout">Click any cell above, or trace a random season, to follow one simulation race by race.</div>

<p class="foot">GridQuant &middot; 20,000 Monte-Carlo simulations of the model in <code>predict2026/</code> &middot; OpenF1 data (unofficial). The exact percentages are uncertain (see MODEL_LIMITATIONS.md); this shows the full spread, not a single number.</p>
</div>
<script id="D" type="application/json">__DATA__</script>
<script>
const D=JSON.parse(document.getElementById('D').textContent);
const N=D.n_sims, EV=D.events, TR=D.traj_all, WIN=D.winners, FA=D.final_ant, FH=D.final_ham;
const COL={0:{l:'#9FE1CB',d:'#0F6E56',name:'Antonelli'},1:{l:'#F7C1C1',d:'#A32D2D',name:'Hamilton'},2:{l:'#cfcdc4',d:'#7d7b73',name:'Other'}};
const pct=x=>Math.round(x*1000/N)/10;

document.getElementById('cards').innerHTML=[
 ['Antonelli',D.counts.ANT,'var(--ant)'],['Hamilton',D.counts.HAM,'var(--ham)'],['Field',D.counts.other,'var(--oth)']
].map(([n,c,col])=>`<div class="card"><div class="lab"><span class="sw" style="background:${col}"></span> ${n} wins</div><div class="val">${pct(c)}%</div><div class="lab">${c.toLocaleString()} of ${N.toLocaleString()} seasons</div></div>`).join('');

function dpr(cv,w,h){const r=Math.max(1,window.devicePixelRatio||1);cv.width=w*r;cv.height=h*r;const x=cv.getContext('2d');x.setTransform(r,0,0,r,0,0);return x;}
function lerp(a,b,t){a=parseInt(a.slice(1),16);b=parseInt(b.slice(1),16);const ar=a>>16,ag=(a>>8)&255,ab=a&255,br=b>>16,bg=(b>>8)&255,bb=b&255;const r=Math.round(ar+(br-ar)*t),g=Math.round(ag+(bg-ag)*t),bl=Math.round(ab+(bb-ab)*t);return`rgb(${r},${g},${bl})`;}

const margin=i=>WIN[i]===0?FA[i]-FH[i]:WIN[i]===1?FH[i]-FA[i]:0;
const order=[...Array(N).keys()].sort((a,b)=> WIN[a]!==WIN[b]?WIN[a]-WIN[b]:margin(b)-margin(a));
const cellOf=new Int32Array(N); order.forEach((sim,k)=>cellOf[sim]=k);

const COLS=200, ROWS=Math.ceil(N/COLS), MW=1000, MH=Math.round(ROWS*MW/COLS);
const mc=document.getElementById('mosaic'), mhi=document.getElementById('mosaicHi');
const mx=dpr(mc,MW,MH); const mhx=dpr(mhi,MW,MH);
const cw=MW/COLS, ch=MH/ROWS;
const maxMarg={0:1,1:1};[...Array(N).keys()].forEach(i=>{if(WIN[i]<2)maxMarg[WIN[i]]=Math.max(maxMarg[WIN[i]],margin(i))});
function drawMosaic(){
  mx.clearRect(0,0,MW,MH);
  for(let k=0;k<N;k++){const sim=order[k];const col=k%COLS,row=(k/COLS)|0;const w=WIN[sim];
    let fill; if(w<2){const t=Math.min(1,margin(sim)/maxMarg[w]);fill=lerp(COL[w].l,COL[w].d,0.15+0.85*t);}else fill=COL[2].l;
    mx.fillStyle=fill; mx.fillRect(col*cw,row*ch,cw+0.5,ch+0.5);}
}
drawMosaic();
const mtip=document.getElementById('mtip');
function simAtEvent(e){const rect=mc.getBoundingClientRect();const sx=MW/rect.width,sy=MH/rect.height;
  const px=(e.clientX-rect.left)*sx,py=(e.clientY-rect.top)*sy;const col=Math.floor(px/cw),row=Math.floor(py/ch);
  const k=row*COLS+col; if(col<0||col>=COLS||k<0||k>=N)return null; return {sim:order[k],col,row};}
mhi.parentElement.style.cursor='crosshair';
mc.parentElement.addEventListener('mousemove',e=>{const h=simAtEvent(e);mhx.clearRect(0,0,MW,MH);if(!h){mtip.style.opacity=0;return;}
  mhx.strokeStyle='#14161c';mhx.lineWidth=1.5;mhx.strokeRect(h.col*cw,h.row*ch,cw,ch);
  const s=h.sim,w=WIN[s];const rect=mc.getBoundingClientRect();
  mtip.style.left=(e.clientX-rect.left)+'px';mtip.style.top=(e.clientY-rect.top)+'px';mtip.style.opacity=1;
  mtip.innerHTML=`<b style="color:${w===0?'#9FE1CB':w===1?'#F7C1C1':'#cfcdc4'}">${COL[w].name} champion</b> &middot; ANT ${FA[s]} - HAM ${FH[s]}`;});
mc.parentElement.addEventListener('mouseleave',()=>{mhx.clearRect(0,0,MW,MH);mtip.style.opacity=0;});
mc.parentElement.addEventListener('click',e=>{const h=simAtEvent(e);if(h)traceSim(h.sim);});

const SW=1000, SH=430, PADL=46, PADR=14, PADT=16, PADB=34;
const sc=document.getElementById('spag'), shi=document.getElementById('spagHi');
const sx=dpr(sc,SW,SH); const shx=dpr(shi,SW,SH);
const nE=EV.length;
let lo=Infinity,hi=-Infinity; for(let i=0;i<N;i++){const v0=TR[i][0],vN=TR[i][nE];if(vN<lo)lo=vN;if(vN>hi)hi=vN;}
lo=Math.min(lo,-40); hi=Math.max(hi,80); lo=Math.max(lo,-200); hi=Math.min(hi,360);
const X=e=>PADL+e*(SW-PADL-PADR)/nE;
const Y=v=>PADT+(hi-v)*(SH-PADT-PADB)/(hi-lo);
const env=[];for(let e=0;e<=nE;e++){const col=new Float64Array(N);for(let i=0;i<N;i++)col[i]=TR[i][e];col.sort();
  env.push([col[(N*0.1)|0],col[(N*0.5)|0],col[(N*0.9)|0]]);}
let show={0:true,1:true,2:true};
const SAMPLE=6000; const samp=[];{const step=N/SAMPLE;for(let i=0;i<N;i+=step)samp.push((i)|0);}
function drawSpag(){
  sx.clearRect(0,0,SW,SH);
  sx.fillStyle='#f7f8fa';sx.fillRect(0,0,SW,SH);
  sx.fillStyle='rgba(226,75,74,0.06)';sx.fillRect(PADL,Y(0),SW-PADL-PADR,SH-PADB-Y(0));
  for(const i of samp){const w=WIN[i];if(!show[w])continue;
    sx.strokeStyle=w===0?'rgba(22,184,148,0.05)':w===1?'rgba(226,75,74,0.06)':'rgba(150,148,140,0.08)';
    sx.lineWidth=1;sx.beginPath();for(let e=0;e<=nE;e++){const xx=X(e),yy=Y(TR[i][e]);e?sx.lineTo(xx,yy):sx.moveTo(xx,yy);}sx.stroke();}
  sx.setLineDash([5,4]);sx.strokeStyle='#14161c';sx.lineWidth=1.2;sx.beginPath();sx.moveTo(PADL,Y(0));sx.lineTo(SW-PADR,Y(0));sx.stroke();sx.setLineDash([]);
  sx.strokeStyle='rgba(20,22,28,0.55)';sx.lineWidth=2;sx.beginPath();for(let e=0;e<=nE;e++){const xx=X(e),yy=Y(env[e][1]);e?sx.lineTo(xx,yy):sx.moveTo(xx,yy);}sx.stroke();
  sx.strokeStyle='rgba(20,22,28,0.25)';sx.lineWidth=1;sx.setLineDash([2,3]);
  [0,2].forEach(j=>{sx.beginPath();for(let e=0;e<=nE;e++){const xx=X(e),yy=Y(env[e][j]);e?sx.lineTo(xx,yy):sx.moveTo(xx,yy);}sx.stroke();});sx.setLineDash([]);
  sx.fillStyle='#14161c';sx.beginPath();sx.arc(X(0),Y(D.start_lead),4,0,7);sx.fill();
  sx.fillStyle='#6b7280';sx.font='11px sans-serif';sx.textAlign='right';
  [lo,0,hi,Math.round((hi+lo)/2)].forEach(v=>{sx.fillText((v>0?'+':'')+Math.round(v),PADL-6,Y(v)+3);});
  sx.save();sx.translate(13,SH/2);sx.rotate(-Math.PI/2);sx.textAlign='center';sx.fillText('Antonelli lead (pts)',0,0);sx.restore();
  sx.textAlign='center';sx.fillStyle='#6b7280';
  for(let e=1;e<=nE;e++){if(e%2===0||e===nE){sx.fillText(EV[e-1],X(e),SH-10);}}
  sx.textAlign='left';sx.fillText('now',X(0)-6,SH-10);
  sx.fillStyle='#A32D2D';sx.font='11px sans-serif';sx.fillText('Hamilton ahead',PADL+6,Y(0)+15);
  sx.fillStyle='#0F6E56';sx.fillText('Antonelli ahead',PADL+6,Y(0)-7);
}
drawSpag();
function traceSim(sim){shx.clearRect(0,0,SW,SH);const w=WIN[sim];const gold='#E8A317';
  shx.strokeStyle=gold;shx.lineWidth=2.6;shx.beginPath();for(let e=0;e<=nE;e++){const xx=X(e),yy=Y(TR[sim][e]);e?shx.lineTo(xx,yy):shx.moveTo(xx,yy);}shx.stroke();
  shx.fillStyle=gold;for(let e=0;e<=nE;e++){shx.beginPath();shx.arc(X(e),Y(TR[sim][e]),2.4,0,7);shx.fill();}
  const lead0=TR[sim][0],leadN=TR[sim][nE];const lc=Math.min(...TR[sim]);
  const flips=(()=>{let f=0;for(let e=1;e<=nE;e++)if((TR[sim][e-1]>=0)!==(TR[sim][e]>=0))f++;return f;})();
  document.getElementById('readout').innerHTML=
   `<b style="color:${w===0?'#0F6E56':w===1?'#A32D2D':'#7d7b73'}">${COL[w].name} wins</b> this season &middot; final points: Antonelli <b>${FA[sim]}</b>, Hamilton <b>${FH[sim]}</b> &middot; `+
   `lead went ${lead0>0?'+':''}${lead0} to ${leadN>0?'+':''}${leadN}, dipped to ${lc} &middot; the lead changed hands <b>${flips}</b> time${flips===1?'':'s'}.`;
}
document.getElementById('legend').innerHTML=[0,1,2].map(w=>`<span class="chip" data-w="${w}"><span class="sw" style="background:${w===0?'#16B894':w===1?'#E24B4A':'#9A988F'}"></span>${COL[w].name}</span>`).join('');
document.querySelectorAll('.chip').forEach(c=>c.addEventListener('click',()=>{const w=+c.dataset.w;show[w]=!show[w];c.classList.toggle('off',!show[w]);drawSpag();}));
document.getElementById('randBtn').addEventListener('click',()=>{traceSim((Math.random()*N)|0);});
document.getElementById('clearBtn').addEventListener('click',()=>{shx.clearRect(0,0,SW,SH);document.getElementById('readout').textContent='Click any cell above, or trace a random season, to follow one simulation race by race.';});
</script></body></html>"""

out = HTML.replace("__DATA__", DATA)
open(_os.path.join(_MEDIA, "2026_simulations_explorer.html"), "w").write(out)
print("wrote championship_simulations_2026.html", len(out), "bytes")
