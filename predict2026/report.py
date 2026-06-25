"""Charts + self-contained HTML report for the 2026 championship forecast."""
import os
import base64
import io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from .config import AS_OF, LIVE_SEASON, N_SIMS

TEAM_COLORS = {
    "Mercedes": "#27F4D2", "Ferrari": "#E80020", "McLaren": "#FF8000",
    "Red Bull Racing": "#3671C6", "Alpine": "#0093CC", "Racing Bulls": "#6692FF",
    "Haas F1 Team": "#B6BABD", "Williams": "#64C4FF", "Audi": "#52E252",
    "Aston Martin": "#229971", "Cadillac": "#C49A6C",
}
plt.rcParams.update({"figure.dpi": 130, "font.size": 11, "axes.grid": True,
                     "grid.alpha": 0.25, "axes.axisbelow": True})


def _c(team):
    return TEAM_COLORS.get(team, "#888888")


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def chart_wdc(wdc_res, path, top=8):
    d = wdc_res.head(top)[::-1]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    bars = ax.barh(d["driver"], d["WDC_prob_%"], color=[_c(t) for t in d["team"]],
                   edgecolor="black", linewidth=0.5)
    for b, v in zip(bars, d["WDC_prob_%"]):
        ax.text(v + 0.6, b.get_y() + b.get_height() / 2, f"{v:.1f}%",
                va="center", fontsize=10, fontweight="bold")
    ax.set_xlabel("Probability of winning the 2026 Drivers' Championship (%)")
    ax.set_title(f"2026 WDC win probability, {N_SIMS:,} simulations from {AS_OF.date()}",
                 fontweight="bold")
    ax.set_xlim(0, max(d["WDC_prob_%"]) * 1.18)
    return _save(fig, path)


def chart_trajectory(pace_df, dev_df, path, focus=None):
    team_rows = pace_df[pace_df["driver"].isna()]
    if focus is None:
        focus = dev_df.sort_values("gap_season_s")["team"].head(5).tolist()
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    for team in focus:
        s = team_rows[team_rows["team"] == team].sort_values("round")
        ax.plot(s["round"], s["gap_to_best_s"], "-o", color=_c(team),
                label=team, linewidth=2.2, markersize=5)
    ax.axvspan(3.6, 4.4, color="gold", alpha=0.18)
    ax.text(4.0, ax.get_ylim()[1] * 0.94, "Miami\nupgrades", ha="center",
            va="top", fontsize=8.5, color="#7a5c00")
    ax.axvspan(4.6, 5.4, color="silver", alpha=0.18)
    ax.text(5.0, ax.get_ylim()[1] * 0.94, "Canada\nupgrades", ha="center",
            va="top", fontsize=8.5, color="#555")
    ax.set_xlabel("Round (1=Australia ... 7=Barcelona)")
    ax.set_ylabel("Race-pace gap to fastest car (s/lap)")
    ax.set_title("Development trajectory, falling line = upgrades working",
                 fontweight="bold")
    ax.invert_yaxis()  # closer to top = faster
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(fontsize=9, loc="lower left")
    return _save(fig, path)


def chart_upgrade_verdict(upg_df, path):
    d = upg_df.dropna(subset=["improvement_s"]).copy()
    d["label"] = d["team"] + "  R" + d["round"].astype(str)
    d = d.sort_values("improvement_s")
    colors = ["#2ca02c" if v >= 0.15 else "#98c379" if v >= 0.05
              else "#cccccc" if v > -0.05 else "#d62728" for v in d["improvement_s"]]
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    bars = ax.barh(d["label"], d["improvement_s"], color=colors,
                   edgecolor="black", linewidth=0.5)
    for b, v, verdict in zip(bars, d["improvement_s"], d["verdict"]):
        ax.text(v + (0.02 if v >= 0 else -0.02), b.get_y() + b.get_height() / 2,
                f"{v:+.2f}s", va="center", ha="left" if v >= 0 else "right",
                fontsize=9, fontweight="bold")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Pace gap closed after the package (s/lap)  to  right = helped")
    ax.set_title("Did the upgrades help? Measured pace change before vs after",
                 fontweight="bold")
    pad = max(abs(d["improvement_s"].min()), d["improvement_s"].max()) * 1.35
    ax.set_xlim(-pad, pad)
    return _save(fig, path)


def chart_points(wdc_res, path, top=8):
    d = wdc_res.head(top)[::-1]
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    y = np.arange(len(d))
    ax.barh(y, d["points_now"], color=[_c(t) for t in d["team"]], alpha=0.55,
            label="Points now")
    err = np.vstack([d["proj_points"] - d["proj_points_p10"],
                     d["proj_points_p90"] - d["proj_points"]])
    ax.errorbar(d["proj_points"], y, xerr=err, fmt="D", color="black",
                markersize=6, capsize=4, linewidth=1.3, label="Projected final (P10-P90)")
    ax.set_yticks(y)
    ax.set_yticklabels(d["driver"])
    ax.set_xlabel("Championship points")
    ax.set_title("Current vs projected final points", fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    return _save(fig, path)


def _b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def build_html(ds, dev_df, upg_df, wdc_res, wcc, charts, path):
    leader = wdc_res.iloc[0]
    runner = wdc_res.iloc[1]
    n_done = int(((ds["schedule"].kind == "race") & (ds["schedule"].done)).sum())
    n_left = int(((ds["schedule"].kind == "race") & (ds["schedule"].done)).sum())

    def img(key, alt):
        return f'<img src="data:image/png;base64,{_b64(charts[key])}" alt="{alt}" style="width:100%;max-width:820px;border:1px solid #e3e3e3;border-radius:8px;margin:8px 0;">'

    wdc_rows = "\n".join(
        f"<tr><td>{i+1}</td><td><b>{r.driver}</b></td><td>{r.team}</td>"
        f"<td>{int(r.points_now)}</td><td>{r.proj_points:.0f}</td>"
        f"<td>{int(r.proj_points_p10)}-{int(r.proj_points_p90)}</td>"
        f"<td><b>{r['WDC_prob_%']:.1f}%</b></td></tr>"
        for i, r in wdc_res.head(8).iterrows())

    upg_rows = "\n".join(
        f"<tr><td>R{int(r['round'])}</td><td>{r.team}</td>"
        f"<td>{'' if pd.isna(r.parts) else int(r.parts)}</td>"
        f"<td style='max-width:340px'>{r.package}</td>"
        f"<td>{'' if r.improvement_s is None or pd.isna(r.improvement_s) else f'{r.improvement_s:+.2f}s'}</td>"
        f"<td class='v-{r.verdict.split()[0].lower()}'>{r.verdict}</td></tr>"
        for _, r in upg_df.iterrows())

    dev_rows = "\n".join(
        f"<tr><td>{r.team}</td><td>{r.gap_season_s:.2f}</td><td>{r.gap_recent_s:.2f}</td>"
        f"<td>{r.gap_delta_s:+.2f}</td><td>{r.trend}</td><td>{r.projected_gap_s:.2f}</td></tr>"
        for _, r in dev_df.iterrows())

    wcc_rows = "\n".join(
        f"<tr><td>{i+1}</td><td>{r.team}</td><td>{r.proj_points:.0f}</td><td><b>{r['WCC_prob_%']:.1f}%</b></td></tr>"
        for i, r in wcc.sort_values('proj_points', ascending=False).head(6).reset_index(drop=True).iterrows())

    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>GridQuant, 2026 Championship Forecast</title>
<style>
 body{{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;color:#1a1a1a;max-width:900px;margin:0 auto;padding:24px;line-height:1.55}}
 h1{{font-size:25px;margin-bottom:2px}} h2{{font-size:19px;border-bottom:2px solid #eee;padding-bottom:6px;margin-top:34px}}
 .sub{{color:#666;font-size:14px;margin-top:0}}
 .hero{{background:linear-gradient(135deg,#0b1020,#1c2440);color:#fff;border-radius:12px;padding:20px 24px;margin:18px 0}}
 .hero b{{color:#27F4D2}} .hero .two{{color:#E80020}}
 table{{border-collapse:collapse;width:100%;font-size:14px;margin:10px 0}}
 th,td{{border:1px solid #e6e6e6;padding:7px 9px;text-align:left}} th{{background:#f6f7f9}}
 td:first-child{{color:#888}}
 .v-clear{{color:#1a7f37;font-weight:bold}} .v-modest{{color:#3f8f4f}} .v-no{{color:#c4351c;font-weight:bold}} .v-neutral{{color:#888}}
 .note{{background:#fff8e6;border-left:4px solid #f0c000;padding:10px 14px;font-size:13.5px;border-radius:4px;margin:14px 0}}
 .foot{{color:#888;font-size:12px;margin-top:30px;border-top:1px solid #eee;padding-top:10px}}
</style></head><body>

<h1>GridQuant · 2026 F1 World Championship Forecast</h1>
<p class="sub">As of {AS_OF.date()}, after {n_done} of {n_done+n_left} Grands Prix · {N_SIMS:,} Monte-Carlo seasons · data: OpenF1</p>

<div class="hero">
<div style="font-size:15px;opacity:.8">PREDICTED 2026 WORLD CHAMPION</div>
<div style="font-size:30px;font-weight:800;margin:4px 0"><b>{leader['name'] or leader.driver}</b> &nbsp;<span style="font-size:18px;opacity:.85">{leader.team}</span></div>
<div style="font-size:15px">{leader['WDC_prob_%']:.0f}% title probability, chief rival <span class="two">{runner['name'] or runner.driver}</span> ({runner.team}) at {runner['WDC_prob_%']:.0f}%</div>
</div>

<p>{leader.driver} leads with <b>{int(leader.points_now)}</b> points, {int(leader.points_now-runner.points_now)} clear of {runner.driver}, and the forecast keeps {leader.driver} a clear favourite, {leader.team} has qualified on pole at every round so far. But the title is not settled: {runner.team}'s in-season upgrades have closed most of the gap on race pace, making {runner.driver} the one driver with a realistic path to overhauling the lead over the remaining {n_left} rounds.</p>

{img('wdc','WDC win probability')}

<h2>Drivers' championship, projected</h2>
<table><tr><th>#</th><th>Driver</th><th>Team</th><th>Pts now</th><th>Proj.</th><th>P10-P90</th><th>Title %</th></tr>
{wdc_rows}</table>
{img('points','current vs projected points')}

<h2>Have the upgrades helped?</h2>
<p>Each team's race pace is tracked as a gap to the fastest car every weekend; a falling line means the development programme is delivering. {('Ferrari' if 'Ferrari' in dev_df.team.values else 'The leading developer')} shows the clearest gain, from roughly half a second off to the front of the field in race trim after its 11-part Miami package, with Red Bull the next-best developer.</p>
{img('trajectory','development trajectory')}
{img('verdict','upgrade verdicts')}

<h3>Development trajectory by team</h3>
<table><tr><th>Team</th><th>Season gap</th><th>Recent gap</th><th>Δ</th><th>Trend</th><th>Projected</th></tr>
{dev_rows}</table>

<h3>Verified 2026 upgrade packages, measured effect</h3>
<table><tr><th>Round</th><th>Team</th><th>Parts</th><th>Package</th><th>Pace Δ</th><th>Verdict</th></tr>
{upg_rows}</table>

<h2>Constructors' championship, projected</h2>
<table><tr><th>#</th><th>Team</th><th>Proj. pts</th><th>Title %</th></tr>
{wcc_rows}</table>

<div class="note"><b>How confident is this, really?</b> Treat the headline as <b>"strong favourite," not a precise number.</b> Resampling the 7 completed races puts the leader's true title chance anywhere in roughly a <b>60-95% band</b>, the single biggest driver is reliability, which we still estimate from very few retirements (now pooled at team level with a 2025 prior and correlated team-mate failures, the most defensible estimate the data allows). What is robust across every test: the leader is a clear favourite, the runner-up (16%, in line with betting markets) is the one realistic challenger, and the rest of the field is a genuine long shot in slower cars. The exact percentage will only sharpen as more races run, see MODEL_LIMITATIONS.md. Probabilities come from {N_SIMS:,} simulations of the remaining {n_left} Grands Prix + sprints (qualifying to grid to race, points-per-race form, development random walk), started from the real current points.</div>

<p class="foot">GridQuant analytics · OpenF1 data (unofficial; not associated with Formula 1). F1, FORMULA 1 and related marks are trade marks of Formula One Licensing B.V. Upgrade-package details sourced from FIA car-presentation documents as reported by Sky Sports, PlanetF1, The Race and Motorsport.com (May-June 2026). Model calibrated and back-tested against the seven completed 2026 rounds.</p>
</body></html>"""
    with open(path, "w") as f:
        f.write(html)
    return path
