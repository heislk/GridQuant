import pandas as pd
import numpy as np

def calculate_theoretical_best_laps(laps_df):
    valid_laps = laps_df[(laps_df['TrackStatus'] == '1') & laps_df['Sector1Time'].notna() & laps_df['Sector2Time'].notna() & laps_df['Sector3Time'].notna()].copy()
    for sec in ['Sector1Time', 'Sector2Time', 'Sector3Time']:
        valid_laps[sec] = valid_laps[sec].dt.total_seconds()
    pace_data = []
    for ((year, event), event_group) in valid_laps.groupby(['Year', 'Event']):
        if 'Session' in event_group.columns and 'Qualifying' in event_group['Session'].values:
            group = event_group[event_group['Session'] == 'Qualifying']
        else:
            group = event_group
        best_s1 = group['Sector1Time'].min()
        best_s2 = group['Sector2Time'].min()
        best_s3 = group['Sector3Time'].min()
        ultimate_lap = best_s1 + best_s2 + best_s3
        if ultimate_lap == 0:
            continue
        for (driver, driver_laps) in group.groupby('Driver'):
            driver_best = driver_laps['LapTime'].dt.total_seconds().min()
            d_s1 = driver_laps['Sector1Time'].min()
            d_s2 = driver_laps['Sector2Time'].min()
            d_s3 = driver_laps['Sector3Time'].min()
            driver_ideal = d_s1 + d_s2 + d_s3
            if np.isnan(driver_best) or driver_best == 0:
                continue
            gap_pct = (driver_best - ultimate_lap) / ultimate_lap * 100
            execution_gap = driver_best - driver_ideal if driver_ideal > 0 else 0
            pace_data.append({'Year': year, 'Event': event, 'Driver': driver, 'UltimateLap': ultimate_lap, 'DriverBest': driver_best, 'DriverIdeal': driver_ideal, 'GapToUltimatePct': gap_pct, 'ExecutionGap': execution_gap, 'SessionType': 'Qualifying' if 'Qualifying' in group['Session'].unique() else 'Race'})
    return pd.DataFrame(pace_data)

def aggregate_qualifying_performance(pace_df):
    if pace_df.empty:
        return pd.DataFrame()
    stats = pace_df.groupby('Driver').agg({'GapToUltimatePct': ['mean', 'std', 'count'], 'ExecutionGap': 'mean'})
    stats.columns = ['MeanGapPct', 'StdGap', 'Races', 'AvgExecutionLoss']
    stats = stats[stats['Races'] >= 10].sort_values('MeanGapPct')
    return stats.reset_index()