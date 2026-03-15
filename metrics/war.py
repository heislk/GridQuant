import pandas as pd
import numpy as np
from collections import defaultdict
from .common import get_team_driver_mapping

def calculate_raw_deltas(laps_df, results_df):
    laps_df = laps_df.copy()
    laps_df['LapTimeSeconds'] = laps_df['LapTime'].dt.total_seconds()
    clean_laps = laps_df[(laps_df['TrackStatus'] == '1') & laps_df['PitOutTime'].isna() & laps_df['PitInTime'].isna() & (laps_df['LapTimeSeconds'] > 60) & (laps_df['LapTimeSeconds'] < 200)].copy()
    team_mapping = get_team_driver_mapping(results_df)
    driver_to_team = {}
    for ((year, event, team), drivers) in team_mapping.items():
        for driver in drivers:
            driver_to_team[year, event, driver] = (team, drivers)
    raw_deltas = []
    for ((year, event), group) in clean_laps.groupby(['Year', 'Event']):
        drivers_in_race = group['Driver'].unique()
        for driver in drivers_in_race:
            key = (year, event, driver)
            if key not in driver_to_team:
                continue
            (team, teammates) = driver_to_team[key]
            if len(teammates) < 2:
                continue
            teammate = [t for t in teammates if t != driver]
            if not teammate:
                continue
            teammate = teammate[0]
            if teammate not in drivers_in_race:
                continue
            driver_laps = group[group['Driver'] == driver]['LapTimeSeconds']
            teammate_laps = group[group['Driver'] == teammate]['LapTimeSeconds']
            if len(driver_laps) < 5 or len(teammate_laps) < 5:
                continue
            driver_median = driver_laps.median()
            teammate_median = teammate_laps.median()
            delta = teammate_median - driver_median
            raw_deltas.append({'year': year, 'event': event, 'driver': driver, 'teammate': teammate, 'team': team, 'delta': delta, 'driver_median': driver_median, 'teammate_median': teammate_median})
    return pd.DataFrame(raw_deltas)

def calculate_elo_ratings(raw_deltas_df, k_factor=32, initial_elo=1500):
    elo_ratings = defaultdict(lambda : initial_elo)
    elo_history = defaultdict(list)
    raw_deltas_df = raw_deltas_df.sort_values(['year', 'event'])
    processed_races = set()
    for (_, row) in raw_deltas_df.iterrows():
        race_key = (row['year'], row['event'], row['driver'], row['teammate'])
        reverse_key = (row['year'], row['event'], row['teammate'], row['driver'])
        if race_key in processed_races or reverse_key in processed_races:
            continue
        processed_races.add(race_key)
        driver_a = row['driver']
        driver_b = row['teammate']
        delta = row['delta']
        elo_a = elo_ratings[driver_a]
        elo_b = elo_ratings[driver_b]
        expected_a = 1 / (1 + 10 ** ((elo_b - elo_a) / 400))
        if delta > 0.1:
            score_a = 1.0
        elif delta < -0.1:
            score_a = 0.0
        else:
            score_a = 0.5
        margin_factor = min(abs(delta) / 0.5, 2.0)
        adjusted_k = k_factor * (1 + margin_factor * 0.5)
        elo_ratings[driver_a] += adjusted_k * (score_a - expected_a)
        elo_ratings[driver_b] += adjusted_k * (1 - score_a - (1 - expected_a))
        elo_history[driver_a].append({'year': row['year'], 'event': row['event'], 'elo': elo_ratings[driver_a]})
        elo_history[driver_b].append({'year': row['year'], 'event': row['event'], 'elo': elo_ratings[driver_b]})
    return (dict(elo_ratings), dict(elo_history))

def calculate_adjusted_war(raw_deltas_df, elo_ratings):
    driver_stats = defaultdict(lambda : {'raw_deltas': [], 'adjusted_deltas': [], 'teammates': set(), 'teams': set(), 'years': set()})
    for (_, row) in raw_deltas_df.iterrows():
        driver = row['driver']
        teammate = row['teammate']
        delta = row['delta']
        teammate_elo = elo_ratings.get(teammate, 1500)
        elo_weight = (teammate_elo - 1400) / 200
        elo_weight = max(0.5, min(elo_weight, 2.0))
        adjusted_delta = delta * elo_weight
        driver_stats[driver]['raw_deltas'].append(delta)
        driver_stats[driver]['adjusted_deltas'].append(adjusted_delta)
        driver_stats[driver]['teammates'].add(teammate)
        driver_stats[driver]['teams'].add(row['team'])
        driver_stats[driver]['years'].add(row['year'])
    war_data = []
    for (driver, stats) in driver_stats.items():
        if len(stats['raw_deltas']) < 10:
            continue
        raw_mean = np.mean(stats['raw_deltas'])
        raw_std = np.std(stats['raw_deltas'])
        adjusted_mean = np.mean(stats['adjusted_deltas'])
        war_data.append({'Driver': driver, 'RawDelta': raw_mean, 'AdjustedWAR': adjusted_mean, 'StdDelta': raw_std, 'Consistency': 1 / (1 + raw_std), 'RaceCount': len(stats['raw_deltas']), 'ELO': elo_ratings.get(driver, 1500), 'Teams': list(stats['teams']), 'Years': list(stats['years']), 'Teammates': list(stats['teammates'])})
    war_df = pd.DataFrame(war_data)
    war_df = war_df.sort_values('AdjustedWAR', ascending=False)
    return war_df