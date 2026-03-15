from metrics.common import load_all_data
import os
import pandas as pd

script_dir = os.path.dirname(os.path.abspath('f1_sabermetrics/run_pipeline.py'))
project_root = os.path.dirname(script_dir)
data_dir = os.path.join(project_root, 'f1_data', 'f1_data_parquet')

laps, res = load_all_data(data_dir)
res_2024 = res[res['Year'] == 2024]

print(f"2024 Rows: {len(res_2024)}")
rus = res_2024[res_2024['Abbreviation'] == 'RUS']
alb = res_2024[res_2024['Abbreviation'] == 'ALB']

print("\nRUSSELL 2024:")
if not rus.empty:
    print(rus[['Event', 'TeamName', 'Position']].head())
else:
    print("No RUS in 2024")

print("\nALBON 2024:")
if not alb.empty:
    print(alb[['Event', 'TeamName', 'Position']].head())
else:
    print("No ALB in 2024")
    
# Check Unique Teams in 2024
print("\nTEAMS 2024:", res_2024['TeamName'].unique())
