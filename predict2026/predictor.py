"""Monte Carlo championship predictor.

Build a driver profile from the season so far (qualifying pace, race pace and
points per race), then play out the remaining rounds many times to get title odds.
Each race is a two-stage draw: qualifying sets the grid, then the race runs from
grid position plus pace. Team pace also drifts a little over the rest of the year.
"""
import numpy as np
import pandas as pd
from scipy.stats import norm

from .reliability import compute_car_dnf, TEAMMATE_RHO

from .config import (RACE_POINTS, SPRINT_POINTS, N_SIMS, DEFAULT_DNF_RATE,
                     DNF_PRIOR_WEIGHT, PACE_TO_UTILITY, SIGMA_RACE, SIGMA_QUALI_S,
                     SIGMA_SPRINT_EXTRA, GRID_STICKINESS, TEAM_RACE_SIGMA, DEV_STEP_S,
                     STRENGTH_QUALI, STRENGTH_RACE, STRENGTH_RESULTS,
                     RESULTS_PPR_TO_S,
                     SEASON_WEIGHT, RECENT_WEIGHT, RECENT_ROUNDS)


def _points_vector(points_map, n):
    v = np.zeros(n)
    for pos, pts in points_map.items():
        if pos - 1 < n:
            v[pos - 1] = pts
    return v


def _blended(df, value_col, key="driver"):
    """Per-key season+recent blend of a metric (lower = faster/better)."""
    if df.empty:
        return pd.Series(dtype=float)
    season = df.groupby(key)[value_col].mean()
    recent_rounds = sorted(df["round"].unique())[-RECENT_ROUNDS:]
    recent = df[df["round"].isin(recent_rounds)].groupby(key)[value_col].mean()
    return pd.Series({k: SEASON_WEIGHT * season[k] + RECENT_WEIGHT * recent.get(k, season[k])
                      for k in season.index})


def build_profiles(ds):
    grid = pd.DataFrame(ds["grid"].values()) if isinstance(ds["grid"], dict) else ds["grid"]
    num2acr = dict(zip(grid["driver_number"], grid["acronym"]))
    num2team = dict(zip(grid["driver_number"], grid["team"]))
    num2name = dict(zip(grid["driver_number"], grid["full_name"]))

    q_gap = _blended(ds["quali"][ds["quali"]["driver"].notna()], "gap_pole")
    r_gap = _blended(ds["pace"][ds["pace"]["driver"].notna()], "driver_gap_to_best_s")

    # results signal: points per race (championship currency) -> gap in seconds.
    # Robust to a single bad result and properly rewards wins.
    races = ds["results"][ds["results"]["kind"] == "race"].copy()
    races["driver"] = races["driver_number"].map(num2acr)
    n_done = max(1, races["round"].nunique())

    # reliability: team-pooled 2026 + 2025 historical prior + driver adjustment
    # (far more stable than one DNF per driver; see reliability.py)
    dnf_rate, grid_dnf = compute_car_dnf(ds, num2acr, num2team)

    wdc = ds["wdc"].copy()
    wdc["driver"] = wdc["driver_number"].map(num2acr)
    pts_now = dict(zip(wdc["driver"], wdc["points_current"]))
    ppr = {d: pts_now.get(d, 0.0) / n_done for d in num2acr.values()}
    best_ppr = max(ppr.values()) if ppr else 0.0
    res_gap = pd.Series({d: (best_ppr - v) * RESULTS_PPR_TO_S for d, v in ppr.items()})

    rows = []
    for d in set(q_gap.index) | set(r_gap.index):
        qg = q_gap.get(d, np.nan)
        rg = r_gap.get(d, np.nan)
        if np.isnan(qg) and np.isnan(rg):
            continue
        if np.isnan(qg):
            qg = rg
        if np.isnan(rg):
            rg = qg
        rsg = res_gap.get(d, np.nan)
        if np.isnan(rsg):
            rsg = (qg + rg) / 2
        num = next((n for n, a in num2acr.items() if a == d), None)
        team = num2team.get(num)
        if team is None:
            continue
        strength = STRENGTH_QUALI * qg + STRENGTH_RACE * rg + STRENGTH_RESULTS * rsg
        rows.append({
            "driver_number": num, "driver": d, "name": num2name.get(num), "team": team,
            "points_now": float(pts_now.get(d, 0.0)),
            "quali_gap_s": round(float(qg), 3), "race_gap_s": round(float(rg), 3),
            "results_gap_s": round(float(rsg), 3), "strength_s": round(float(strength), 3),
            "pace_gap_s": round(float(strength), 3), # alias kept for the report
            "dnf_rate": round(float(dnf_rate.get(d, grid_dnf)), 3),
        })
    return pd.DataFrame(rows).sort_values("strength_s").reset_index(drop=True)


def simulate(profiles, n_races, n_sprints, n_sims=N_SIMS, seed=20260623):
    rng = np.random.default_rng(seed)
    D = len(profiles)
    quali_gap = profiles["quali_gap_s"].values.astype(float)
    strength = profiles["strength_s"].values.astype(float)
    dnf = profiles["dnf_rate"].values.astype(float)
    start = profiles["points_now"].values.astype(float)
    teams = profiles["team"].values
    uniq_teams = sorted(set(teams))
    tix = {t: i for i, t in enumerate(uniq_teams)}
    tsel = np.array([tix[t] for t in teams])

    race_pts = _points_vector(RACE_POINTS, D)
    sprint_pts = _points_vector(SPRINT_POINTS, D)
    # retirement thresholds for the Gaussian-copula DNF model (teammate-correlated)
    dnf_thresh = norm.ppf(1.0 - np.clip(dnf, 1e-4, 0.99))
    rho = TEAMMATE_RHO

    totals = np.tile(start, (n_sims, 1))
    car_walk = np.zeros((n_sims, len(uniq_teams)))  # cumulative development drift

    def run_events(n, sigma, pts_vec):
        nonlocal totals, car_walk
        for _ in range(n):
            car_walk = car_walk + rng.normal(0, DEV_STEP_S, size=(n_sims, len(uniq_teams)))
            dev = car_walk[:, tsel]
            # stage 1: qualifying -> grid (1 = pole)
            q_team = rng.normal(0, TEAM_RACE_SIGMA, size=(n_sims, len(uniq_teams)))[:, tsel]
            q_score = -(quali_gap[None, :] + dev + q_team) / SIGMA_QUALI_S \
                + rng.normal(0, 1.0, size=(n_sims, D))
            grid_pos = (-q_score).argsort(axis=1).argsort(axis=1) + 1
            # stage 2: race finish from strength + sticky grid position
            team_form = rng.normal(0, TEAM_RACE_SIGMA, size=(n_sims, len(uniq_teams)))[:, tsel]
            r_util = -PACE_TO_UTILITY * (strength[None, :] + dev) - GRID_STICKINESS * grid_pos + team_form
            scores = r_util + rng.normal(0, sigma, size=(n_sims, D))
            # correlated retirements: shared per-team mechanical "bad day" (z_team)
            # plus an independent per-driver component (u)
            z_team = rng.normal(0, 1, size=(n_sims, len(uniq_teams)))[:, tsel]
            u = rng.normal(0, 1, size=(n_sims, D))
            latent = np.sqrt(rho) * z_team + np.sqrt(1 - rho) * u
            retired = latent > dnf_thresh[None, :]
            scores = np.where(retired, -1e9, scores)
            order = np.argsort(-scores, axis=1)
            ranks = np.empty_like(order)
            np.put_along_axis(ranks, order, np.broadcast_to(np.arange(D), order.shape), axis=1)
            totals += pts_vec[ranks]

    run_events(n_races, SIGMA_RACE, race_pts)
    run_events(n_sprints, SIGMA_RACE + SIGMA_SPRINT_EXTRA, sprint_pts)

    jitter = rng.uniform(0, 1e-6, size=totals.shape)
    champ_idx = np.argmax(totals + jitter, axis=1)
    wdc_wins = np.bincount(champ_idx, minlength=D)
    finish_rank = (-totals).argsort(axis=1).argsort(axis=1) + 1

    res = profiles.copy()
    res["WDC_prob_%"] = (100.0 * wdc_wins / n_sims).round(3)
    res["proj_points"] = totals.mean(axis=0).round(1)
    res["proj_points_p10"] = np.percentile(totals, 10, axis=0).round(0)
    res["proj_points_p90"] = np.percentile(totals, 90, axis=0).round(0)
    res["avg_finish_rank"] = finish_rank.mean(axis=0).round(2)
    res["top3_prob_%"] = (100.0 * (finish_rank <= 3).mean(axis=0)).round(1)

    Tmat = np.zeros((D, len(uniq_teams)))
    for d, t in enumerate(teams):
        Tmat[d, tix[t]] = 1.0
    team_totals = totals @ Tmat
    team_champ = np.argmax(team_totals + rng.uniform(0, 1e-6, team_totals.shape), axis=1)
    wcc_wins = np.bincount(team_champ, minlength=len(uniq_teams))
    wcc = pd.DataFrame({
        "team": uniq_teams,
        "WCC_prob_%": (100.0 * wcc_wins / n_sims).round(1),
        "proj_points": team_totals.mean(axis=0).round(1),
    }).sort_values("proj_points", ascending=False).reset_index(drop=True)

    res = res.sort_values("WDC_prob_%", ascending=False).reset_index(drop=True)
    return res, wcc
