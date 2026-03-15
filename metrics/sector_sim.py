import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Optional
import warnings

warnings.filterwarnings('ignore')

def fit_skewnorm(data: pd.Series) -> Optional[tuple]:
    if len(data) < 3:
        return None
    try:
        a, loc, scale = stats.skewnorm.fit(data)
        if scale <= 1e-4 or scale > 10.0: 
            return None
        return a, loc, scale
    except Exception:
        return None

def simulate_sector_performance(laps_df: pd.DataFrame, year: int = 2024, n_sims: int = 1000000) -> pd.DataFrame:
    print(f"  [Sector Sim] Initializing Monte Carlo Simulation ({n_sims:,} laps/driver)...")
    clean_laps = laps_df[
        (laps_df['Year'] == year) &
        (laps_df['TrackStatus'] == '1') &
        (laps_df['PitOutTime'].isna()) & 
        (laps_df['PitInTime'].isna())
    ].copy()
    
    for s in ['Sector1Time', 'Sector2Time', 'Sector3Time']:
        if clean_laps[s].dtype == 'object': # Handle potential timedelta strings if not already converted
             # assuming simple conversion needed or already handled upstream, 
             # but ensuring it's float seconds here is good practice.
             # In this pipeline, it seems they come as timedeltas or floats.
             pass
        if pd.api.types.is_timedelta64_dtype(clean_laps[s]):
            clean_laps[s] = clean_laps[s].dt.total_seconds()
            
    if 'LapTimeSeconds' not in clean_laps.columns:
        if 'LapTime' in clean_laps.columns:
            if pd.api.types.is_timedelta64_dtype(clean_laps['LapTime']):
                clean_laps['LapTimeSeconds'] = clean_laps['LapTime'].dt.total_seconds()
            else:
                 # Assume it might be float or try to coerce
                 clean_laps['LapTimeSeconds'] = clean_laps['LapTime']
            
    clean_laps = clean_laps.dropna(subset=['Sector1Time', 'Sector2Time', 'Sector3Time'])

    results = []
    tracks = clean_laps['Event'].unique()
    
    for track in tracks:
        track_df = clean_laps[clean_laps['Event'] == track]
        
        if len(track_df) < 100: 
            continue
            
        drivers = track_df['Driver'].unique()
        print(f"    Processing {track} ({len(drivers)} drivers)...")
        
        driver_sims: Dict[str, Dict[str, np.ndarray]] = {}
        
        for driver in drivers:
            d_laps = track_df[track_df['Driver'] == driver].copy()
            
            best_lap = d_laps['LapTimeSeconds'].min()
            if pd.isna(best_lap) or best_lap == 0: 
                continue
            
            d_laps = d_laps[d_laps['LapTimeSeconds'] <= best_lap * 1.05]
            
            if len(d_laps) < 5: 
                continue
            
            sim_sectors = []
            valid_driver = True
            
            for sec in ['Sector1Time', 'Sector2Time', 'Sector3Time']:
                data = d_laps[sec].dropna()
                
                params = fit_skewnorm(data)
                
                if params is None:
                    valid_driver = False
                    break
                
                a, loc, scale = params
                
                try:
                    sim = stats.skewnorm.rvs(a, loc=loc, scale=scale, size=n_sims)
                    
                    physical_limit = data.min() * 0.95
                    sim = np.maximum(sim, physical_limit)
                    
                    sim_sectors.append(sim)
                except ValueError:
                    valid_driver = False
                    break
            
            if valid_driver:
                driver_sims[driver] = {
                    's1': sim_sectors[0],
                    's2': sim_sectors[1],
                    's3': sim_sectors[2],
                    'total': sim_sectors[0] + sim_sectors[1] + sim_sectors[2]
                }
        
        if not driver_sims:
            continue
            
        all_s1 = np.concatenate([d['s1'] for d in driver_sims.values()])
        all_s2 = np.concatenate([d['s2'] for d in driver_sims.values()])
        all_s3 = np.concatenate([d['s3'] for d in driver_sims.values()])
        
        best_s1 = np.percentile(all_s1, 0.01)
        best_s2 = np.percentile(all_s2, 0.01)
        best_s3 = np.percentile(all_s3, 0.01)
        
        global_ultimate = best_s1 + best_s2 + best_s3
        
        for driver, sims in driver_sims.items():
            driver_potential = np.percentile(sims['total'], 1.0)
            
            gap_pct = (driver_potential - global_ultimate) / global_ultimate * 100
            
            results.append({
                'Track': track,
                'Driver': driver,
                'TotalLapsSimulated': n_sims,
                'GapToUltimate': gap_pct,
                'TheoreticalLap': global_ultimate,
                'DriverPotential': driver_potential
            })
            
    return pd.DataFrame(results)
