import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional
import random

# ---------------------------------------------------------------------------
# Year-aware season configuration.
#
# Previously this module was hard-coded to 2024 (GRID_2024 / STRENGTH_2024).
# It is now keyed by season so the pipeline can be rolled forward. 2025 is the
# completed baseline and 2026 is the live/predicted season. For a full,
# data-driven 2026 forecast that also models in-season upgrades, use the
# dedicated package in ``predict2026/`` (run ``run_2026_prediction.py``); the
# maps below remain for the legacy parquet pipeline.
# ---------------------------------------------------------------------------

SEASON_GRIDS = {
    2024: {
        'VER': 'Red Bull Racing', 'PER': 'Red Bull Racing',
        'HAM': 'Mercedes', 'RUS': 'Mercedes',
        'LEC': 'Ferrari', 'SAI': 'Ferrari',
        'NOR': 'McLaren', 'PIA': 'McLaren',
        'ALO': 'Aston Martin', 'STR': 'Aston Martin',
        'GAS': 'Alpine', 'OCO': 'Alpine',
        'ALB': 'Williams', 'SAR': 'Williams', 'COL': 'Williams',
        'TSU': 'RB', 'RIC': 'RB', 'LAW': 'RB',
        'BOT': 'Kick Sauber', 'ZHO': 'Kick Sauber',
        'HUL': 'Haas F1 Team', 'MAG': 'Haas F1 Team', 'BEA': 'Haas F1 Team',
    },
    # 2026 grid (new regulations: Audi replaces Sauber, Cadillac joins,
    # Hamilton at Ferrari, Antonelli at Mercedes). Source: OpenF1 2026 entry list.
    2026: {
        'NOR': 'McLaren', 'PIA': 'McLaren',
        'VER': 'Red Bull Racing', 'HAD': 'Red Bull Racing',
        'ANT': 'Mercedes', 'RUS': 'Mercedes',
        'LEC': 'Ferrari', 'HAM': 'Ferrari',
        'ALO': 'Aston Martin', 'STR': 'Aston Martin',
        'GAS': 'Alpine', 'COL': 'Alpine',
        'ALB': 'Williams', 'SAI': 'Williams',
        'LAW': 'Racing Bulls', 'LIN': 'Racing Bulls',
        'OCO': 'Haas F1 Team', 'BEA': 'Haas F1 Team',
        'BOR': 'Audi', 'HUL': 'Audi',
        'PER': 'Cadillac', 'BOT': 'Cadillac',
    },
}

# Approximate car performance in Plackett-Luce utility units. Higher = faster.
SEASON_STRENGTHS = {
    2024: {
        'Red Bull Racing': 2.5, 'McLaren': 2.3, 'Ferrari': 2.2, 'Mercedes': 2.0,
        'Aston Martin': 1.2, 'RB': 0.8, 'AlphaTauri': 0.8, 'Haas F1 Team': 0.7,
        'Williams': 0.5, 'Alpine': 0.4, 'Kick Sauber': 0.1,
    },
    # 2026 strengths derived from GridQuant's as-of pace analysis (recent
    # race-pace gap to the fastest car, upgrade-adjusted). Mercedes & Ferrari
    # at the front after Ferrari's mid-season upgrade surge.
    2026: {
        'Mercedes': 2.5, 'Ferrari': 2.45, 'Red Bull Racing': 1.8, 'McLaren': 1.4,
        'Alpine': 0.7, 'Racing Bulls': 0.6, 'Haas F1 Team': 0.45,
        'Williams': 0.4, 'Audi': 0.4, 'Cadillac': 0.05, 'Aston Martin': 0.0,
    },
}


@dataclass
class DriverProfile:
    name: str
    team: str
    base_elo: float
    war_value: float
    consistency: float  # 0.0 to 1.0 (inverse of std dev)
    dnf_rate: float     # 0.0 to 1.0
    team_strength: float = 0.0  # Utility bonus for car performance


class ChampionshipSimulator:
    def __init__(self, drivers: List[DriverProfile], n_sims: int = 10000,
                 calendar_length: int = 24, fastest_lap_point: bool = False):
        self.drivers = drivers
        self.n_sims = n_sims
        # Current F1 points system. The fastest-lap point was removed for 2025+,
        # so it is off by default (set fastest_lap_point=True for <=2024).
        self.points_system = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10,
                              6: 8, 7: 6, 8: 4, 9: 2, 10: 1}
        self.calendar_length = calendar_length
        self.fastest_lap_point = fastest_lap_point

    def _simulate_race_plackett_luce(self, rng: np.random.Generator) -> Dict[str, int]:
        active_drivers = []
        for d in self.drivers:
            if rng.random() > d.dnf_rate:
                active_drivers.append(d)

        utilities = []
        for d in active_drivers:
            mu = (d.war_value * 2.0) + d.team_strength
            elo_adj = (d.base_elo - 1500) / 400.0
            mu += elo_adj * 0.3
            sigma = 0.8 + ((1.1 - d.consistency) * 0.5)
            score = rng.normal(mu, sigma)
            utilities.append((d.name, score))

        utilities.sort(key=lambda x: x[1], reverse=True)

        points_map = {}
        for pos, (driver_name, _) in enumerate(utilities, 1):
            points = self.points_system.get(pos, 0)
            if self.fastest_lap_point and pos <= 10 and rng.random() < 0.1:
                points += 1
            points_map[driver_name] = points
        return points_map

    def run_season(self, rng: np.random.Generator) -> Dict[str, float]:
        season_points = {d.name: 0 for d in self.drivers}
        for _ in range(self.calendar_length):
            race_res = self._simulate_race_plackett_luce(rng)
            for driver, pts in race_res.items():
                if driver in season_points:
                    season_points[driver] += pts
        return season_points

    def simulate(self) -> pd.DataFrame:
        print(f"  [Championship Sim] Simulating {self.n_sims:,} seasons (Plackett-Luce + Car Strength)...")
        rng = np.random.default_rng()
        wdc_wins = {d.name: 0 for d in self.drivers}
        total_points = {d.name: 0 for d in self.drivers}
        avg_pos = {d.name: 0 for d in self.drivers}

        for _ in range(self.n_sims):
            final_standings = self.run_season(rng)
            winner = max(final_standings, key=final_standings.get)
            wdc_wins[winner] += 1
            sorted_standing_names = sorted(final_standings, key=final_standings.get, reverse=True)
            for rank, name in enumerate(sorted_standing_names, 1):
                total_points[name] += final_standings[name]
                avg_pos[name] += rank

        results = []
        for d in self.drivers:
            name = d.name
            results.append({
                'Driver': name, 'Team': d.team,
                'WDC_Probability': (wdc_wins[name] / self.n_sims) * 100,
                'AvgPoints': total_points[name] / self.n_sims,
                'AvgRank': avg_pos[name] / self.n_sims,
            })
        return pd.DataFrame(results).sort_values('WDC_Probability', ascending=False)


def create_driver_profiles_v2(war_df: pd.DataFrame, results_df: pd.DataFrame,
                              target_year: int = 2026) -> List[DriverProfile]:
    """Build driver profiles for any configured season (year-aware).

    Uses the per-year grid + car-strength registry above, falling back to the
    teams found in ``results_df`` when a season isn't pre-registered.
    """
    res_year = results_df[results_df['Year'] == target_year].copy() if not results_df.empty else pd.DataFrame()
    driver_team_map = {}
    if not res_year.empty:
        if 'Round' in res_year.columns:
            res_year = res_year.sort_values('Round')
        for _, row in res_year.iterrows():
            driver_team_map[row['Abbreviation']] = row['TeamName']

    grid = SEASON_GRIDS.get(target_year, {})
    strengths = SEASON_STRENGTHS.get(target_year, {})

    profiles = []
    for _, row in war_df.iterrows():
        driver = row['Driver']
        current_team = grid.get(driver) or driver_team_map.get(driver)
        if not current_team:
            continue
        car_score = strengths.get(current_team, 1.0)
        const = float(row.get('Consistency', 0.5))
        profiles.append(DriverProfile(
            name=driver, team=current_team,
            base_elo=float(row.get('ELO', 1500)),
            war_value=float(row.get('AdjustedWAR', 0)),
            consistency=const, dnf_rate=0.05, team_strength=car_score,
        ))
    return profiles


def create_driver_profiles(war_df: pd.DataFrame, target_year: int = 2026) -> List[DriverProfile]:
    """Deprecated legacy wrapper."""
    return []
