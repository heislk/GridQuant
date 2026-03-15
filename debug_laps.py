from metrics.common import load_all_data
import os
import pandas as pd

script_dir = os.path.dirname(os.path.abspath('f1_sabermetrics/run_pipeline.py'))
project_root = os.path.dirname(script_dir)
data_dir = os.path.join(project_root, 'f1_data', 'f1_data_parquet')

laps, res = load_all_data(data_dir)
print("Laps Columns:", laps.columns.tolist())
# Check first row to see time format
print("Laps Sample:\n", laps.head(1).T)
