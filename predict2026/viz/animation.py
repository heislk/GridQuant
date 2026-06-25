"""Clean 4K explainer: who wins the 2026 F1 title?"""

import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_DATA = _os.path.join(_ROOT, "predict2026", "data", "sim_viz.json")
_MEDIA = _os.path.join(_ROOT, "media"); _os.makedirs(_os.path.join(_MEDIA, "previews"), exist_ok=True)
_CACHE = _os.path.join(_ROOT, "predict2026", "cache_snapshot", "openf1_2026")

import json, os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
plt.rcParams["font.family"] = "Lato"

d = json.load(open(_DATA))
N = d["n_sims"]
pA = d["counts"]["ANT"] / N; pH = d["counts"]["HAM"] / N; pF = d["counts"]["other"] / N
gA, gH, gF = 82, 16, 2

BG, INK, MUT, FAINT = "#0B0E11", "#F4F7FA", "#8A95A1", "#2A323C"
TEAL = "#19D6B4"; RED = "#FF3B3B"; SIL = "#9AA7B1"; GREY = "#5A6470"; PLATE_INK = "#0B0E11"
FOOT = "#5A6470"
STAND = [("ANTONELLI", 12, 156, TEAL, "5 WINS"), ("HAMILTON", 44, 115, RED, None), ("RUSSELL", 63, 106, SIL, None)]

ASPECT = 16 / 9
FPS = 24
S1, S2, S3, S4, S5, S6 = 72, 108, 156, 96, 96, 72
BNDS = np.cumsum([0, S1, S2, S3, S4, S5, S6]); TOTAL = int(BNDS[-1])
W_IN, H_IN, DPI = 19.2, 10.8, 200
fig = plt.figure(figsize=(W_IN, H_IN), dpi=DPI)

def eo(t): return 0 if t <= 0 else 1 if t >= 1 else 1 - (1 - t) ** 3

def ax_frame():
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis("off"); ax.set_facecolor(BG); return ax

def plate(ax, num, cx, cy, h, color, fs):
    w = h / ASPECT
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.012",
                                fc=color, ec="none", mutation_aspect=ASPECT, zorder=4))
    ax.text(cx, cy - 0.002, str(num), color=PLATE_INK, fontsize=fs, ha="center", va="center", weight="bold", zorder=5)

def footer(ax, a=1.0):
    ax.text(0.5, 0.05, "20,000-simulation forecast   ·   2026 F1, after round 7   ·   GridQuant",
            color=FOOT, fontsize=15, ha="center", alpha=a)

NDOT = 700; PERROW = 14; H_COL = 0.50; BASEY = 0.16
nA = round(pA * NDOT); nH = round(pH * NDOT); nF = NDOT - nA - nH
COLX = {0: 0.235, 1: 0.5, 2: 0.765}; COLC = {0: TEAL, 1: RED, 2: GREY}
ROWSA = int(np.ceil(nA / PERROW)); DY = H_COL / ROWSA; DX = DY * (H_IN / W_IN); DOTS_S = 28
def slot(c, k):
    r, col = divmod(k, PERROW)
    return COLX[c] + (col - (PERROW - 1) / 2) * DX, BASEY + r * DY
arrival = np.array([0] * nA + [1] * nH + [2] * nF); np.random.default_rng(3).shuffle(arrival)
cum = np.zeros((NDOT + 1, 3), int)
for i, w in enumerate(arrival): cum[i + 1] = cum[i]; cum[i + 1][w] += 1

def draw_columns(ax, nrev):
    cnt = cum[nrev]
    for c in (0, 1, 2):
        xs, ys = [], []
        for k in range(cnt[c]):
            x, y = slot(c, k); xs.append(x); ys.append(y)
        if xs: ax.scatter(xs, ys, s=DOTS_S, c=COLC[c], edgecolors="none", zorder=3)
    return cnt

def update(f):
    fig.clf(); fig.patch.set_facecolor(BG); ax = ax_frame()
    # 1. question
    if f < BNDS[1]:
        a = eo(f / 20)
        ax.text(0.5, 0.625, "WHO WINS THE 2026 F1 TITLE?", color=INK, fontsize=62, ha="center", va="center", weight="bold", alpha=a)
        ax.text(0.5, 0.53, "7 of 22 races are done.", color=MUT, fontsize=27, ha="center", va="center", alpha=a)
        a2 = eo((f - 12) / 22)
        for i in range(22):
            x = 0.315 + i * (0.37 / 21)
            ax.scatter([x], [0.43], s=130, c=(TEAL if i < 7 else FAINT), edgecolors="none", alpha=a2, zorder=3)
        ax.text(0.5, 0.365, "So we played the rest out, 20,000 times.", color=INK, fontsize=24, ha="center", alpha=a2)
        return
    # 2. standings
    if f < BNDS[2]:
        g = f - BNDS[1]
        ax.text(0.085, 0.875, "THE STANDINGS NOW", color=INK, fontsize=34, weight="bold")
        ax.text(0.085, 0.83, "after 7 of 22 races", color=MUT, fontsize=22)
        x0 = 0.30; xmax = 0.90; maxpts = 156; rows = [0.64, 0.475, 0.31]
        for i, (name, num, pts, col, badge) in enumerate(STAND):
            y = rows[i]; gr = eo((g - i * 6) / 34)
            plate(ax, num, 0.115, y, 0.058, col, 22)
            ax.text(0.16, y, name, color=INK, fontsize=30, va="center", weight="bold")
            bw = (xmax - x0) * (pts / maxpts) * gr
            ax.add_patch(Rectangle((x0, y - 0.034), bw, 0.068, fc=col, ec="none", zorder=2))
            ax.text(x0 + bw + 0.012, y, f"{int(round(pts * gr))}", color=INK, fontsize=30, va="center", weight="bold")
            if badge and gr > 0.98:
                ax.text(x0 + 0.014, y, badge, color=PLATE_INK, fontsize=16, va="center", weight="bold")
        footer(ax)
        return
    # 3. the 20,000 pile-up
    if f < BNDS[3]:
        g = f - BNDS[2]
        ax.text(0.5, 0.90, "WE PLAYED THE REST OF THE SEASON 20,000 TIMES", color=INK, fontsize=29, ha="center", weight="bold")
        ax.text(0.5, 0.855, "stacked by who won each season", color=MUT, fontsize=22, ha="center")
        p = eo(min(1, g / (S3 - 18))); nrev = int(p * NDOT)
        cnt = draw_columns(ax, nrev); tot = max(1, cnt.sum())
        for name, c, col in [("ANTONELLI", 0, TEAL), ("HAMILTON", 1, RED), ("FIELD", 2, GREY)]:
            ax.text(COLX[c], 0.78, f"{100*cnt[c]/tot:.0f}%", color=col, fontsize=50, ha="center", weight="bold")
            ax.text(COLX[c], 0.10, name, color=col, fontsize=23, ha="center", weight="bold")
        ax.text(0.5, 0.05, f"{int(p*20000):,} of 20,000 seasons", color=FOOT, fontsize=16, ha="center")
        return
    # 4. out of 100
    if f < BNDS[4]:
        g = f - BNDS[3]
        ax.text(0.5, 0.90, "OUT OF EVERY 100 SEASONS", color=INK, fontsize=36, ha="center", weight="bold")
        p = eo(min(1, g / (S4 - 22))); nrev = int(round(p * 100))
        gx0, gy0, cell = 0.345, 0.21, 0.030
        seq = [TEAL] * gA + [RED] * gH + [GREY] * gF
        xs, ys, cs = [], [], []
        for k in range(nrev):
            r, c = divmod(k, 10); xs.append(gx0 + c * cell); ys.append(gy0 + (9 - r) * cell * ASPECT); cs.append(seq[k])
        if xs: ax.scatter(xs, ys, s=250, c=cs, edgecolors=BG, linewidths=1.2, marker="s", zorder=3)
        ax.text(0.76, 0.66, f"{min(nrev, gA)}", color=TEAL, fontsize=150, ha="center", va="center", weight="bold")
        ax.text(0.76, 0.50, "won by", color=MUT, fontsize=24, ha="center")
        ax.text(0.76, 0.45, "ANTONELLI", color=TEAL, fontsize=40, ha="center", va="top", weight="bold")
        if p > 0.96:
            ax.text(0.76, 0.31, "Hamilton 16   ·   Field 2", color=MUT, fontsize=24, ha="center")
        footer(ax)
        return
    # 5. how sure
    if f < BNDS[5]:
        g = f - BNDS[4]
        ax.text(0.5, 0.82, "HOW SURE IS THIS?", color=INK, fontsize=36, ha="center", weight="bold")
        tx0, tx1, ty = 0.14, 0.86, 0.55
        def TX(v): return tx0 + (tx1 - tx0) * v / 100
        ax.add_patch(Rectangle((tx0, ty - 0.006), tx1 - tx0, 0.012, fc=FAINT, ec="none"))
        for v in (0, 50, 100):
            ax.text(TX(v), ty - 0.075, f"{v}%", color=MUT, fontsize=18, ha="center")
        bw = eo(min(1, g / 30)); lo, hi = 60, 95
        L = TX(82 - (82 - lo) * bw); Rr = TX(82 + (hi - 82) * bw)
        ax.add_patch(Rectangle((L, ty - 0.045), Rr - L, 0.09, fc=TEAL, ec="none", alpha=0.22))
        ax.scatter([TX(82)], [ty], s=430, c=TEAL, edgecolors=BG, linewidths=2, zorder=5)
        ax.text(TX(82), ty + 0.08, "82%", color=TEAL, fontsize=42, ha="center", weight="bold")
        if bw > 0.6:
            ax.text(L, ty - 0.075, "60%", color=TEAL, fontsize=19, ha="center")
            ax.text(Rr, ty - 0.075, "95%", color=TEAL, fontsize=19, ha="center")
        ax.text(0.5, 0.31, "It is still early, so this is a forecast, not a guarantee.", color=INK, fontsize=24, ha="center")
        ax.text(0.5, 0.265, "The likeliest answer is about 82%, but the real range runs from 60% to 95%.", color=MUT, fontsize=21, ha="center")
        footer(ax)
        return
    # 6. stamp
    g = f - BNDS[5]; a = eo(min(1, g / 16)); sc = 1 + 0.04 * (1 - a)
    plate(ax, 12, 0.5, 0.66, 0.10 * sc, TEAL, 40)
    ax.text(0.5, 0.545, "ANTONELLI", color=TEAL, fontsize=int(76 * sc), ha="center", va="center", weight="bold", alpha=a)
    ax.add_patch(Rectangle((0.5 - 0.14 * a, 0.475), 0.28 * a, 0.004, fc=TEAL, ec="none"))
    ax.text(0.5, 0.435, "82% TITLE FAVOURITE", color=INK, fontsize=40, ha="center", va="center", weight="bold", alpha=a)
    ax.text(0.5, 0.355, "Hamilton 16%      ·      Field 2%", color=MUT, fontsize=27, ha="center", alpha=a)
    footer(ax, a)
    return

if os.environ.get("RENDER"):
    from matplotlib.animation import FuncAnimation, FFMpegWriter
    anim = FuncAnimation(fig, update, frames=TOTAL, interval=1000 / FPS)
    w = FFMpegWriter(fps=FPS, bitrate=16000, codec="libx264", extra_args=["-pix_fmt", "yuv420p", "-preset", "medium"])
    anim.save(_os.path.join(_MEDIA, "2026_title_forecast.mp4"), writer=w, dpi=DPI, savefig_kwargs={"facecolor": BG})
    print("saved", TOTAL, "frames")
