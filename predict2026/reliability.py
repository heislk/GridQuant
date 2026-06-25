"""Per-car retirement model.

One DNF per driver is too little to go on, so we pool retirements at the team
level, lean on last season as a prior, and let team-mates fail together.
"""
import numpy as np

# 2025 team DNF rate (sampled), mapped onto 2026 entrant names. New entrants
# (Cadillac) have no history and fall back to the grid mean.
HIST_2025_TEAM_DNF = {
    "Mercedes": 0.05, "McLaren": 0.03, "Ferrari": 0.10, "Red Bull Racing": 0.15,
    "Williams": 0.10, "Haas F1 Team": 0.05, "Aston Martin": 0.15, "Alpine": 0.16,
    "Audi": 0.15,           # ex-Kick Sauber
    "Racing Bulls": 0.03,
}

PRIOR_STRENGTH = 6.0    # pseudo car-races of prior
PRIOR_HIST_W = 0.5      # within the prior: last-season vs current grid mean
DRIVER_WEIGHT = 0.35    # how much a driver's own deviation from the team counts
TEAMMATE_RHO = 0.40     # mechanical correlation between team-mates (copula)
DNF_FLOOR, DNF_CEIL = 0.02, 0.60


def compute_car_dnf(ds, num2acr, num2team):
    races = ds["results"][ds["results"]["kind"] == "race"]
    total_dnf = races["dnf"].sum()
    total_cars = (~races["dns"]).sum()
    grid_mean = float(total_dnf / total_cars) if total_cars else 0.12

    # team-pooled 2026
    team_dnf, team_n = {}, {}
    for num, g in races.groupby("driver_number"):
        t = num2team.get(num)
        if not t:
            continue
        team_dnf[t] = team_dnf.get(t, 0) + int(g["dnf"].sum())
        team_n[t] = team_n.get(t, 0) + int((~g["dns"]).sum())

    out = {}
    for num, g in races.groupby("driver_number"):
        d = num2acr.get(num)
        t = num2team.get(num)
        if not t:
            continue
        t25 = HIST_2025_TEAM_DNF.get(t, grid_mean)
        prior = PRIOR_HIST_W * t25 + (1 - PRIOR_HIST_W) * grid_mean
        td, tn = team_dnf.get(t, 0), team_n.get(t, 0)
        team_base = (td + PRIOR_STRENGTH * prior) / (tn + PRIOR_STRENGTH)
        t26_rate = td / tn if tn else grid_mean
        started = int((~g["dns"]).sum())
        drv_rate = int(g["dnf"].sum()) / started if started else t26_rate
        driver_dev = drv_rate - t26_rate
        out[d] = float(np.clip(team_base + DRIVER_WEIGHT * driver_dev, DNF_FLOOR, DNF_CEIL))
    return out, grid_mean
