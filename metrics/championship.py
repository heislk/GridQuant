import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional
import random

@dataclass
class DriverProfile:
    name: str
    team: str
    base_elo: float
    war_value: float
    consistency: float  # 0.0 to 1.0 (inverse of std dev)
    dnf_rate: float     # 0.0 to 1.0
    team_strength: float = 0.0 # Utility bonus for car performance

class ChampionshipSimulator:
    def __init__(self, drivers: List[DriverProfile], n_sims: int = 10000):
        self.drivers = drivers
        self.n_sims = n_sims
        # Standard F1 Points System (2024)
        self.points_system = {
            1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 
            6: 8, 7: 6, 8: 4, 9: 2, 10: 1
        }
        self.calendar_length = 24  # Standard season length
        
    def _simulate_race_plackett_luce(self, rng: np.random.Generator) -> Dict[str, int]:
        """
        Simulates one race using Plackett-Luce Model for rankings.
        P(i wins) propto exp(skill_i)
        We allow DNF chance before ranking.
        """
        # 1. Determine active drivers for this race (DnF check)
        active_drivers = []
        for d in self.drivers:
            # Base DNF rate + Random bad luck
            if rng.random() > d.dnf_rate:
                active_drivers.append(d)
                
        # 2. Calculate "Utilities" or "Strengths" for Plackett-Luce
        utilities = []
        for d in active_drivers:
            # Mean Performance = Driver Skill (WAR) + Car Strength + ELO
            
            # WAR: -0.5 to +0.5 typically (deltas). We scaled * 2.0 -> -1.0 to +1.0.
            # Car: 0 to 2.5 (from points normalization).
            # This balances Car vs Driver roughly 50/50 or 60/40 favoring Car.
            
            mu = (d.war_value * 2.0) + d.team_strength
            
            # ELO Adjustment (Long term skill)
            elo_adj = (d.base_elo - 1500) / 400.0 
            mu += elo_adj * 0.3 
            
            # Consistency affects sigma
            # Base sigma ~0.8 for randomness
            sigma = 0.8 + ((1.1 - d.consistency) * 0.5)
            
            score = rng.normal(mu, sigma)
            utilities.append( (d.name, score) )
            
        # 3. Sort by Score (Descending)
        utilities.sort(key=lambda x: x[1], reverse=True)
        
        # 4. Assign Points
        points_map = {}
        for pos, (driver_name, _) in enumerate(utilities, 1):
            points = self.points_system.get(pos, 0)
            
            # FL Point check (Top 10 only)
            if pos <= 10 and rng.random() < 0.1: 
                points += 1
                
            points_map[driver_name] = points
            
        return points_map

    def run_season(self, rng: np.random.Generator) -> Dict[str, float]:
        """
        Simulates one full season. Returns standings (points).
        """
        season_points = {d.name: 0 for d in self.drivers}
        
        for _ in range(self.calendar_length):
            race_res = self._simulate_race_plackett_luce(rng)
            for driver, pts in race_res.items():
                if driver in season_points:
                    season_points[driver] += pts
                    
        return season_points

    def simulate(self) -> pd.DataFrame:
        """
        Runs Monte Carlo simulation for the championship.
        """
        print(f"  [Championship Sim] Simulating {self.n_sims:,} seasons (Plackett-Luce + Car Strength)...")
        
        rng = np.random.default_rng()
        
        wdc_wins = {d.name: 0 for d in self.drivers}
        total_points = {d.name: 0 for d in self.drivers}
        avg_pos = {d.name: 0 for d in self.drivers}
        
        for _ in range(self.n_sims):
            final_standings = self.run_season(rng)
            
            # Determine Champion
            winner = max(final_standings, key=final_standings.get)
            wdc_wins[winner] += 1
            
            # Aggregate Stats
            sorted_standing_names = sorted(final_standings, key=final_standings.get, reverse=True)
            for rank, name in enumerate(sorted_standing_names, 1):
                total_points[name] += final_standings[name]
                avg_pos[name] += rank
        
        # Compile Results
        results = []
        for d in self.drivers:
            name = d.name
            results.append({
                'Driver': name,
                'Team': d.team,
                'WDC_Probability': (wdc_wins[name] / self.n_sims) * 100,
                'AvgPoints': total_points[name] / self.n_sims,
                'AvgRank': avg_pos[name] / self.n_sims
            })
            
        return pd.DataFrame(results).sort_values('WDC_Probability', ascending=False)

def create_driver_profiles_v2(war_df: pd.DataFrame, results_df: pd.DataFrame, target_year: int = 2024) -> List[DriverProfile]:
    """
    v2: Requires results_df for accurate team mapping and strength calc.
    Includes robust fallback for 2024 Grid.
    """
    # 1. Dynamic Map Attempts
    res_year = results_df[results_df['Year'] == target_year].copy()
    driver_team_map = {}
    
    if not res_year.empty:
        if 'Round' in res_year.columns:
            res_year = res_year.sort_values('Round')
        for _, row in res_year.iterrows():
            driver_team_map[row['Abbreviation']] = row['TeamName']

    # 2. Hardcoded Fallback / Overlay for 2024
    # To fix data issues (e.g. Russell -> Williams in raw data)
    GRID_2024 = {
        'VER': 'Red Bull Racing', 'PER': 'Red Bull Racing',
        'HAM': 'Mercedes', 'RUS': 'Mercedes',
        'LEC': 'Ferrari', 'SAI': 'Ferrari',
        'NOR': 'McLaren', 'PIA': 'McLaren',
        'ALO': 'Aston Martin', 'STR': 'Aston Martin',
        'GAS': 'Alpine', 'OCO': 'Alpine',
        'ALB': 'Williams', 'SAR': 'Williams', 'COL': 'Williams',
        'TSU': 'RB', 'RIC': 'RB', 'LAW': 'RB',
        'BOT': 'Kick Sauber', 'ZHO': 'Kick Sauber',
        'HUL': 'Haas F1 Team', 'MAG': 'Haas F1 Team', 'BEA': 'Haas F1 Team'
    }
    
    # 3. Strength Map (Approximate Car Performance in Utility Units)
    STRENGTH_2024 = {
        'Red Bull Racing': 2.5,
        'McLaren': 2.3,
        'Ferrari': 2.2,
        'Mercedes': 2.0,
        'Aston Martin': 1.2,
        'RB': 0.8, 'AlphaTauri': 0.8,
        'Haas F1 Team': 0.7,
        'Williams': 0.5,
        'Alpine': 0.4,
        'Kick Sauber': 0.1
    }

    profiles = []
    
    for _, row in war_df.iterrows():
        driver = row['Driver'] # Abbreviation e.g. VER
        
        # Determine Team
        current_team = None
        if target_year == 2024 and driver in GRID_2024:
            current_team = GRID_2024[driver]
        elif driver in driver_team_map:
            current_team = driver_team_map[driver]
            
        if not current_team:
            continue
            
        # Determine Strength
        car_score = 0.0
        if target_year == 2024:
            car_score = STRENGTH_2024.get(current_team, 0.5)
        else:
            # Basic fallback if calculating historically
            car_score = 1.0 
        
        const = float(row.get('Consistency', 0.5))
        
        profiles.append(DriverProfile(
            name=driver,
            team=current_team,
            base_elo=float(row.get('ELO', 1500)),
            war_value=float(row.get('AdjustedWAR', 0)),
            consistency=const,
            dnf_rate=0.05,
            team_strength=car_score
        ))
        
    return profiles

def create_driver_profiles(war_df: pd.DataFrame, target_year: int = 2024) -> List[DriverProfile]:
    """Deprecated legacy wrapper."""
    return []
