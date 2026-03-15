import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple
from dataclasses import dataclass

@dataclass
class DriverPaceProfile:
    name: str
    s1_params: Tuple[float, float, float] # a, loc, scale
    s2_params: Tuple[float, float, float]
    s3_params: Tuple[float, float, float]
    valid: bool = True

class QualifyingSimulator:
    def __init__(self, profiles: List[DriverPaceProfile], n_sims: int = 1000):
        self.profiles = {p.name: p for p in profiles if p.valid}
        self.drivers = list(self.profiles.keys())
        self.n_sims = n_sims
        
    def _get_lap_time(self, driver: str, rng: np.random.Generator) -> float:
        p = self.profiles[driver]
        
        # Sim 3 sectors
        s1 = stats.skewnorm.rvs(*p.s1_params, random_state=rng)
        s2 = stats.skewnorm.rvs(*p.s2_params, random_state=rng)
        s3 = stats.skewnorm.rvs(*p.s3_params, random_state=rng)
        
        return s1 + s2 + s3
        
    def simulate_session(self, rng: np.random.Generator) -> Dict[str, int]:
        """
        Simulates one full Quali session (Q1->Q2->Q3).
        Returns {driver: grid_position}
        """
        # 1. Q1: All Drivers
        q1_times = []
        for d in self.drivers:
            q1_times.append((d, self._get_lap_time(d, rng)))
            
        q1_times.sort(key=lambda x: x[1]) # Fastest first (lowest time)
        
        # Grid 16-20 eliminated
        grid_positions = {}
        q2_drivers = []
        
        # If fewer than 20 drivers, adjust logic dynamically
        n_drivers = len(q1_times)
        n_q2 = min(15, max(0, n_drivers - 5)) # Usually 15 proceed
        n_q3 = min(10, max(0, n_q2 - 5))      # Usually 10 proceed
        
        # Q1 Eliminations
        for i in range(n_q2, n_drivers):
            driver_name = q1_times[i][0]
            grid_positions[driver_name] = i + 1
            
        q2_candidates = [x[0] for x in q1_times[:n_q2]]
        
        # 2. Q2
        if not q2_candidates:
            return grid_positions
            
        q2_times = []
        for d in q2_candidates:
            q2_times.append((d, self._get_lap_time(d, rng)))
        q2_times.sort(key=lambda x: x[1])
        
        # Q2 Eliminations (11-15)
        for i in range(n_q3, len(q2_times)):
            driver_name = q2_times[i][0]
            grid_positions[driver_name] = i + 1
            
        q3_candidates = [x[0] for x in q2_times[:n_q3]]
        
        # 3. Q3
        if not q3_candidates:
            return grid_positions
            
        q3_times = []
        for d in q3_candidates:
            q3_times.append((d, self._get_lap_time(d, rng)))
        q3_times.sort(key=lambda x: x[1])
        
        # Q3 Ranking (1-10)
        for i, (driver, _) in enumerate(q3_times):
            grid_positions[driver] = i + 1
            
        return grid_positions

    def simulate(self) -> pd.DataFrame:
        print(f"  [Quali Sim] Simulating {self.n_sims} qualifying sessions...")
        rng = np.random.default_rng()
        
        pole_positions = {d: 0 for d in self.drivers}
        avg_grid = {d: 0 for d in self.drivers}
        q3_appearances = {d: 0 for d in self.drivers}
        
        for _ in range(self.n_sims):
            grid = self.simulate_session(rng)
            
            for driver, pos in grid.items():
                if pos == 1:
                    pole_positions[driver] += 1
                if pos <= 10:
                    q3_appearances[driver] += 1
                avg_grid[driver] += pos
                
        results = []
        for d in self.drivers:
            results.append({
                'Driver': d,
                'PoleProb': (pole_positions[d] / self.n_sims) * 100,
                'Q3Prob': (q3_appearances[d] / self.n_sims) * 100,
                'AvgGridPos': avg_grid[d] / self.n_sims
            })
            
        return pd.DataFrame(results).sort_values('PoleProb', ascending=False)

def build_profiles_from_laps(laps_df: pd.DataFrame, target_year: int = 2024) -> List[DriverPaceProfile]:
    """
    Analyzes historical data to build pace profiles for simulations.
    FILTERS: Only drivers active in the target_year.
    """
    from .sector_sim import fit_skewnorm # Reuse fitting logic
    
    # 1. Identify Active Drivers for the target year
    active_drivers = laps_df[laps_df['Year'] == target_year]['Driver'].unique()
    
    # Ensure times are floats (handle timedeltas)
    laps_df = laps_df.copy() # Avoid SettingWithCopy warning on slice
    for s in ['Sector1Time', 'Sector2Time', 'Sector3Time']:
        if s in laps_df.columns and pd.api.types.is_timedelta64_dtype(laps_df[s]):
            laps_df[s] = laps_df[s].dt.total_seconds()

    # We can use historical data for these drivers, but only return profiles for THEM.
    drivers = laps_df['Driver'].unique()
    
    profiles = []
    
    for driver in drivers:
        # SKIP if not in active list
        if driver not in active_drivers:
            continue
            
        d_laps = laps_df[laps_df['Driver'] == driver]
        
        # Require enough data
        if len(d_laps) < 3: continue # Lower threshold for Quali data which is sparse
            
        try:
            params = {}
            valid = True
            for sec in ['Sector1Time', 'Sector2Time', 'Sector3Time']:
                p = fit_skewnorm(d_laps[sec].dropna())
                if p:
                    params[sec] = p
                else:
                    valid = False
                    break
            
            if valid:
                profiles.append(DriverPaceProfile(
                    name=driver,
                    s1_params=params['Sector1Time'],
                    s2_params=params['Sector2Time'],
                    s3_params=params['Sector3Time']
                ))
        except:
            continue
            
    return profiles
