import pandas as pd
import numpy as np

def calculate_telemetry_signature(telemetry_df, laps_df, results_df):
    if telemetry_df.empty:
        return pd.DataFrame()
    valid_laps = laps_df[laps_df['IsAccurate'] == True][['Session', 'Driver', 'LapNumber', 'Year', 'Event']]
    stats = []
    groups = telemetry_df.groupby(['Year', 'Driver'])
    for ((year, driver), group) in groups:
        speed = group['Speed']
        if len(speed) < 1000:
            continue
        top_speed = speed.quantile(0.98)
        corner_speed = speed[speed > 50].quantile(0.15)
        stats.append({'Year': year, 'Driver': driver, 'TopSpeed': top_speed, 'CorneringSpeed': corner_speed, 'DataPoints': len(speed)})
    sig_df = pd.DataFrame(stats)
    team_map = results_df[['Year', 'Abbreviation', 'TeamName']].drop_duplicates()
    sig_df = sig_df.merge(team_map, left_on=['Year', 'Driver'], right_on=['Year', 'Abbreviation'], how='left')
    return sig_df