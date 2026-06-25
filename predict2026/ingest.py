"""Pull the 2026 season from OpenF1, up to the as-of cutoff."""
from datetime import datetime
import numpy as np
import pandas as pd

from .config import LIVE_SEASON, AS_OF


def _dt(s):
    return datetime.fromisoformat(s)


def load_schedule(client):
    """Race/sprint sessions; cancelled meetings dropped; rounds keyed by meeting."""
    meetings = client.get("meetings", year=LIVE_SEASON)
    cancelled = {m["meeting_key"] for m in meetings if m.get("is_cancelled")}

    sessions = client.get("sessions", year=LIVE_SEASON)
    rows = []
    for s in sessions:
        if s["meeting_key"] in cancelled:
            continue  # did not take place (e.g. Bahrain/Saudi 2026)
        name, stype = s.get("session_name", ""), s.get("session_type", "")
        is_race = stype == "Race" and name == "Race"
        is_sprint = name == "Sprint"
        if not (is_race or is_sprint):
            continue
        start = _dt(s["date_start"])
        rows.append({
            "session_key": s["session_key"], "meeting_key": s["meeting_key"],
            "country": s["country_name"], "location": s.get("location", ""),
            "kind": "sprint" if is_sprint else "race", "date": start,
            "done": start <= AS_OF,
        })
    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    # round numbering by meeting: one round per GP weekend, sprint inherits it
    race_order = (df[df["kind"] == "race"].sort_values("date")["meeting_key"].tolist())
    round_map = {mk: i + 1 for i, mk in enumerate(race_order)}
    df["round"] = df["meeting_key"].map(round_map)
    return df


def load_grid(client, ref_session_key):
    drivers = client.get("drivers", session_key=ref_session_key)
    grid = {}
    for d in drivers:
        num = d["driver_number"]
        if num in grid:
            continue
        grid[num] = {"driver_number": num, "acronym": d.get("name_acronym"),
                     "full_name": d.get("full_name"), "team": d.get("team_name"),
                     "colour": d.get("team_colour")}
    return grid


def load_standings(client, ref_session_key):
    wdc = client.get("championship_drivers", session_key=ref_session_key)
    wcc = client.get("championship_teams", session_key=ref_session_key)
    wdc_df, wcc_df = pd.DataFrame(wdc), pd.DataFrame(wcc)
    # normalise the points column name (defensive against schema drift)
    if "points_current" not in wdc_df.columns:
        for alt in ("points", "points_total"):
            if alt in wdc_df.columns:
                wdc_df = wdc_df.rename(columns={alt: "points_current"})
                break
    return wdc_df, wcc_df


def load_results(client, schedule):
    rows = []
    for _, s in schedule[schedule["done"]].iterrows():
        for r in client.get("session_result", session_key=int(s["session_key"])):
            if r.get("driver_number") is None:
                continue
            rows.append({"session_key": s["session_key"], "round": s["round"],
                         "country": s["country"], "kind": s["kind"],
                         "driver_number": r.get("driver_number"), "position": r.get("position"),
                         "dnf": bool(r.get("dnf")), "dns": bool(r.get("dns")),
                         "dsq": bool(r.get("dsq")), "laps": r.get("number_of_laps")})
    return pd.DataFrame(rows)


def load_pace(client, schedule, grid):
    """Per-round race-pace gap to the fastest car (the upgrade/development signal)."""
    num_to_team = {n: g["team"] for n, g in grid.items()}
    num_to_acr = {n: g["acronym"] for n, g in grid.items()}
    rows = []
    races = schedule[(schedule["done"]) & (schedule["kind"] == "race")]
    for _, s in races.iterrows():
        laps = client.get("laps", session_key=int(s["session_key"]))
        if not laps:
            continue
        ldf = pd.DataFrame(laps)
        if "lap_duration" not in ldf.columns:
            continue
        ldf = ldf.dropna(subset=["lap_duration"])
        ldf = ldf[(ldf["lap_duration"] > 60) & (ldf["lap_duration"] < 200)]
        if ldf.empty:
            continue
        driver_pace = {}
        for num, grp in ldf.groupby("driver_number"):
            d = grp["lap_duration"].sort_values()
            keep = d.iloc[: max(3, int(len(d) * 0.6))]
            if len(keep) >= 3:
                driver_pace[num] = float(keep.median())
        if not driver_pace:
            continue
        team_pace = {}
        for num, p in driver_pace.items():
            t = num_to_team.get(num)
            if t:
                team_pace.setdefault(t, []).append(p)
        team_pace = {t: float(np.median(v)) for t, v in team_pace.items()}
        best = min(team_pace.values())
        for t, p in team_pace.items():
            rows.append({"round": s["round"], "country": s["country"], "team": t,
                         "driver": None, "driver_number": None, "team_pace_s": p,
                         "gap_to_best_s": p - best, "driver_pace_s": None,
                         "driver_gap_to_best_s": None})
        best_d = min(driver_pace.values())
        for num, p in driver_pace.items():
            rows.append({"round": s["round"], "country": s["country"],
                         "team": num_to_team.get(num), "driver": num_to_acr.get(num),
                         "driver_number": num, "team_pace_s": None, "gap_to_best_s": None,
                         "driver_pace_s": p, "driver_gap_to_best_s": p - best_d})
    return pd.DataFrame(rows)


def _q_best(duration):
    """Best valid lap from an OpenF1 qualifying 'duration' [Q1,Q2,Q3] (or scalar)."""
    if isinstance(duration, list):
        vals = [x for x in duration if isinstance(x, (int, float)) and x and x > 30]
        return min(vals) if vals else None
    return duration if isinstance(duration, (int, float)) and duration > 30 else None


def load_quali(client, schedule, grid):
    """Per-round driver & team qualifying gap to pole (clean pace signal)."""
    num_to_team = {n: g["team"] for n, g in grid.items()}
    num_to_acr = {n: g["acronym"] for n, g in grid.items()}
    rows = []
    races = schedule[(schedule["done"]) & (schedule["kind"] == "race")]
    for _, s in races.iterrows():
        qs = client.get("sessions", meeting_key=int(s["meeting_key"]),
                        session_name="Qualifying")
        if not qs:
            continue
        qr = client.get("session_result", session_key=int(qs[0]["session_key"]))
        times = {}
        for d in qr:
            n, b = d.get("driver_number"), _q_best(d.get("duration"))
            if n is not None and b is not None:
                times[n] = b
        if not times:
            continue
        pole = min(times.values())
        for n, t in times.items():
            rows.append({"round": s["round"], "country": s["country"],
                         "driver": num_to_acr.get(n), "driver_number": n,
                         "team": num_to_team.get(n), "q": t, "gap_pole": t - pole})
    return pd.DataFrame(rows)


def build_dataset(client):
    schedule = load_schedule(client)
    done_races = schedule[(schedule["done"]) & (schedule["kind"] == "race")]
    ref = int(done_races.iloc[-1]["session_key"])
    grid = load_grid(client, ref)
    wdc, wcc = load_standings(client, ref)
    return {"schedule": schedule, "grid": grid, "ref_session_key": ref,
            "wdc": wdc, "wcc": wcc,
            "results": load_results(client, schedule),
            "pace": load_pace(client, schedule, grid),
            "quali": load_quali(client, schedule, grid)}
