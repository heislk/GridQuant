from metrics.common import load_all_data
import os
import pandas as pd

script_dir = os.path.dirname(os.path.abspath('f1_sabermetrics/run_pipeline.py'))
project_root = os.path.dirname(script_dir)
data_dir = os.path.join(project_root, 'f1_data', 'f1_data_parquet')

print("Loading data...")
laps, res = load_all_data(data_dir)
print("Results Columns:", res.columns.tolist())

# Check telemetry sample if possible, effectively repeating Phase 1 logic
telemetry_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if 'telemetry' in f]
if telemetry_files:
    t_file = telemetry_files[0]
    print(f"Reading sample telemetry: {t_file}")
    t_df = pd.read_parquet(t_file) # read all columns to see what's there
    print("Telemetry Columns:", t_df.columns.tolist())
    print("Telemetry Sample:\n", t_df.head(1).T)
else:
    print("No telemetry files found.")
