
import pandas as pd
import metrics.dna as dna
from metrics.common import load_all_data
from metrics.war import calculate_raw_deltas, calculate_elo_ratings, calculate_adjusted_war
from metrics.racecraft import calculate_lap1_performance
from metrics.qualifying import calculate_theoretical_best_laps

print("Loading data...")
laps_df, results_df = load_all_data('../f1_data/f1_data_parquet')
laps_df['LapTimeSeconds'] = laps_df['LapTime'].dt.total_seconds()

print("Calc WAR...")
raw = calculate_raw_deltas(laps_df, results_df)
elo_r, _ = calculate_elo_ratings(raw)
war_df = calculate_adjusted_war(raw, elo_r)

print("Calc Lap 1...")
lap1 = calculate_lap1_performance(laps_df, results_df)

print("Calc Pace...")
pace = calculate_theoretical_best_laps(laps_df)

print("Calc DNA...")
dna_df = dna.calculate_bayesian_dna(war_df, lap1, pace, laps_df)

print(dna_df.sort_values('BattleRating', ascending=False).head(10))
