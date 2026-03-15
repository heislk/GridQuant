import pandas as pd
from metrics.common import load_all_data, get_team_driver_mapping
from metrics.war import calculate_raw_deltas, calculate_elo_ratings, calculate_adjusted_war

data_dir = "../f1_data/f1_data_parquet"
print(f"Loading from {data_dir}")
laps_df, results_df = load_all_data(data_dir)

print(f"Laps: {len(laps_df)}")
print(f"Results: {len(results_df)}")

if not results_df.empty:
    print("Sample Results:")
    print(results_df.head())
    print("Unique Sessions in Results:", results_df['Session'].unique() if 'Session' in results_df.columns else "No Session Col")

mapping = get_team_driver_mapping(results_df)
print(f"Team Mapping Keys: {len(mapping)}")
if len(mapping) > 0:
    print(f"Sample Mapping: {list(mapping.items())[0]}")

deltas = calculate_raw_deltas(laps_df, results_df)
print(f"Raw Deltas: {len(deltas)}")
if not deltas.empty:
    print(deltas.head())

if deltas.empty:
    print("Deltas are empty! Check team mapping or laps data.")
