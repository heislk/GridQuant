import pandas as pd
import numpy as np

def calculate_lap1_performance(laps_df, results_df):
    lap1 = laps_df[laps_df['LapNumber'] == 1].copy()
    start_pos = results_df[['Year', 'Event', 'Session', 'Abbreviation', 'GridPosition']].copy()
    start_pos = start_pos.rename(columns={'Abbreviation': 'Driver'})
    end_pos = lap1[['Year', 'Event', 'Session', 'Driver', 'Position']]
    merged = pd.merge(start_pos, end_pos, on=['Year', 'Event', 'Session', 'Driver'])
    merged = merged[merged['GridPosition'] > 0]
    merged['PositionsGained'] = merged['GridPosition'] - merged['Position']
    stats = merged.groupby('Driver')['PositionsGained'].agg(['mean', 'sum', 'count', 'std']).reset_index()
    stats.columns = ['Driver', 'AvgGain', 'TotalGain', 'Races', 'Consistency']
    stats = stats[stats['Races'] >= 10].sort_values('AvgGain', ascending=False)
    return stats