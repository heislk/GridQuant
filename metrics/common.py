import pandas as pd
import numpy as np
import glob
import os

def load_all_data(data_dir):
    laps_files = glob.glob(os.path.join(data_dir, 'f1_laps_*.parquet'))
    results_files = glob.glob(os.path.join(data_dir, 'f1_results_*.parquet'))
    laps_dfs = [pd.read_parquet(f) for f in laps_files]
    results_dfs = [pd.read_parquet(f) for f in results_files]
    if not laps_dfs:
        raise ValueError(f'No lap data found in {data_dir}')
    laps_df = pd.concat(laps_dfs, ignore_index=True)
    results_df = pd.concat(results_dfs, ignore_index=True) if results_dfs else pd.DataFrame()
    return (laps_df, results_df)

def get_team_driver_mapping(results_df):
    mapping = {}
    if results_df.empty:
        return mapping
    for (_, row) in results_df.iterrows():
        key = (row['Year'], row['Event'], row['TeamName'])
        if key not in mapping:
            mapping[key] = []
        driver = row['Abbreviation']
        if driver not in mapping[key]:
            mapping[key].append(driver)
    return mapping

def calculate_gini_coefficient(values):
    values = np.array(sorted(values))
    n = len(values)
    if n == 0 or np.sum(values) == 0:
        return 0
    cumulative = np.cumsum(values)
    return (n + 1 - 2 * np.sum(cumulative) / cumulative[-1]) / n

def calculate_fuel_corrected_pace(laps_df, fuel_penalty_per_lap=0.06):
    df = laps_df.copy()
    if 'LapNumber' not in df.columns: return df
    total_laps = df['LapNumber'].max()
    df['FuelMsg'] = (total_laps - df['LapNumber']) * fuel_penalty_per_lap
    if 'LapTimeSeconds' in df.columns:
        df['FuelCorrectedLapTime'] = df['LapTimeSeconds'] - df['FuelMsg']
    elif 'LapTime' in df.columns and pd.api.types.is_timedelta64_dtype(df['LapTime']):
        df['FuelCorrectedLapTime'] = df['LapTime'].dt.total_seconds() - df['FuelMsg']
    return df