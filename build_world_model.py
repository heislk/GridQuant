import pandas as pd
import numpy as np
import pickle
import glob
import os
from scipy import stats

class WorldModel:

    def __init__(self):
        self.tyre_degradation = {}
        self.pit_loss_distributions = {}
        self.safety_car_rates = {}
        self.base_lap_times = {}

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            return pickle.load(f)

def load_laps_data(data_dir):
    files = glob.glob(os.path.join(data_dir, 'f1_laps_*.parquet'))
    dfs = []
    for f in files:
        df = pd.read_parquet(f)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def calculate_fuel_adjustment(lap_number, total_laps, fuel_effect_per_lap=0.055):
    laps_remaining = total_laps - lap_number
    fuel_load_fraction = laps_remaining / total_laps
    return fuel_load_fraction * fuel_effect_per_lap * total_laps

def extract_tyre_degradation(laps_df):
    degradation_params = {}
    laps_df = laps_df.dropna(subset=['LapTime', 'Compound', 'TyreLife', 'Event'])
    laps_df = laps_df[laps_df['TrackStatus'] == '1']
    laps_df['LapTimeSeconds'] = laps_df['LapTime'].dt.total_seconds()
    laps_df = laps_df[laps_df['LapTimeSeconds'] < 200]
    laps_df = laps_df[laps_df['LapTimeSeconds'] > 60]
    for event in laps_df['Event'].unique():
        event_df = laps_df[laps_df['Event'] == event]
        degradation_params[event] = {}
        for compound in ['SOFT', 'MEDIUM', 'HARD']:
            compound_df = event_df[event_df['Compound'] == compound]
            if len(compound_df) < 50:
                continue
            grouped = compound_df.groupby('TyreLife')['LapTimeSeconds'].agg(['mean', 'std', 'count'])
            grouped = grouped[grouped['count'] >= 5]
            if len(grouped) < 5:
                continue
            tyre_ages = grouped.index.values
            mean_times = grouped['mean'].values
            std_times = grouped['std'].fillna(0.3).values
            try:
                (slope, intercept, r_value, p_value, std_err) = stats.linregress(tyre_ages, mean_times)
                avg_volatility = np.mean(std_times[std_times > 0]) if len(std_times[std_times > 0]) > 0 else 0.3
                cliff_threshold = None
                if len(tyre_ages) > 10:
                    time_diffs = np.diff(mean_times)
                    if len(time_diffs) > 0:
                        cliff_candidates = np.where(time_diffs > 2 * np.std(time_diffs))[0]
                        if len(cliff_candidates) > 0:
                            cliff_threshold = tyre_ages[cliff_candidates[0] + 1]
                degradation_params[event][compound] = {'mu': slope, 'sigma': avg_volatility, 'intercept': intercept, 'cliff_lap': cliff_threshold, 'r_squared': r_value ** 2, 'sample_size': len(compound_df)}
            except Exception:
                continue
    return degradation_params

def extract_pit_loss_distributions(laps_df):
    pit_distributions = {}
    pit_laps = laps_df[laps_df['PitOutTime'].notna() | laps_df['PitInTime'].notna()].copy()
    if 'PitOutTime' in pit_laps.columns and 'PitInTime' in pit_laps.columns:
        pass
    for event in laps_df['Event'].unique():
        event_laps = laps_df[laps_df['Event'] == event]
        clean_laps = event_laps[(event_laps['TrackStatus'] == '1') & event_laps['PitOutTime'].isna() & event_laps['PitInTime'].isna()]
        if len(clean_laps) < 10:
            continue
        clean_lap_times = clean_laps['LapTime'].dt.total_seconds()
        clean_lap_times = clean_lap_times[(clean_lap_times > 60) & (clean_lap_times < 200)]
        if len(clean_lap_times) < 10:
            continue
        median_clean_time = clean_lap_times.median()
        pit_in_laps = event_laps[event_laps['PitInTime'].notna()]
        pit_out_laps = event_laps[event_laps['PitOutTime'].notna()]
        all_pit_laps = pd.concat([pit_in_laps, pit_out_laps]).drop_duplicates()
        if len(all_pit_laps) < 5:
            pit_distributions[event] = {'mean': 22.0, 'std': 2.0, 'min': 18.0, 'max': 30.0}
            continue
        pit_lap_times = all_pit_laps['LapTime'].dt.total_seconds()
        pit_lap_times = pit_lap_times[(pit_lap_times > 60) & (pit_lap_times < 250)]
        if len(pit_lap_times) < 5:
            pit_distributions[event] = {'mean': 22.0, 'std': 2.0, 'min': 18.0, 'max': 30.0}
            continue
        pit_deltas = pit_lap_times - median_clean_time
        pit_deltas = pit_deltas[(pit_deltas > 10) & (pit_deltas < 50)]
        if len(pit_deltas) < 3:
            pit_distributions[event] = {'mean': 22.0, 'std': 2.0, 'min': 18.0, 'max': 30.0}
            continue
        pit_distributions[event] = {'mean': pit_deltas.mean(), 'std': pit_deltas.std() if pit_deltas.std() > 0 else 2.0, 'min': pit_deltas.min(), 'max': pit_deltas.max()}
    return pit_distributions

def extract_safety_car_rates(laps_df):
    sc_rates = {}
    for event in laps_df['Event'].unique():
        event_laps = laps_df[laps_df['Event'] == event]
        years = event_laps['Year'].unique()
        total_races = len(years)
        sc_laps = event_laps[event_laps['TrackStatus'].isin(['4', '6', '7'])]
        races_with_sc = sc_laps.groupby('Year').ngroups if len(sc_laps) > 0 else 0
        total_laps_in_event = event_laps.groupby('Year')['LapNumber'].max().mean()
        total_laps_in_event = total_laps_in_event if not np.isnan(total_laps_in_event) else 55
        if total_races > 0:
            sc_probability_per_race = races_with_sc / total_races
            lambda_per_lap = sc_probability_per_race / total_laps_in_event
        else:
            lambda_per_lap = 0.005
        sc_rates[event] = {'lambda_per_lap': lambda_per_lap, 'historical_sc_races': races_with_sc, 'total_races_analyzed': total_races, 'avg_race_laps': total_laps_in_event}
    return sc_rates

def extract_base_lap_times(laps_df):
    base_times = {}
    clean_laps = laps_df[(laps_df['TrackStatus'] == '1') & laps_df['PitOutTime'].isna() & laps_df['PitInTime'].isna()].copy()
    clean_laps['LapTimeSeconds'] = clean_laps['LapTime'].dt.total_seconds()
    clean_laps = clean_laps[(clean_laps['LapTimeSeconds'] > 60) & (clean_laps['LapTimeSeconds'] < 200)]
    for event in clean_laps['Event'].unique():
        event_laps = clean_laps[clean_laps['Event'] == event]
        fastest_laps = event_laps.nsmallest(20, 'LapTimeSeconds')
        if len(fastest_laps) > 0:
            base_times[event] = {'fastest': fastest_laps['LapTimeSeconds'].min(), 'top20_mean': fastest_laps['LapTimeSeconds'].mean(), 'top20_std': fastest_laps['LapTimeSeconds'].std()}
    return base_times

def build_world_model(data_dir, output_path):
    print('Loading lap data...')
    laps_df = load_laps_data(data_dir)
    print(f'Loaded {len(laps_df):,} laps')
    model = WorldModel()
    print('Extracting tyre degradation parameters...')
    model.tyre_degradation = extract_tyre_degradation(laps_df)
    print('Extracting pit loss distributions...')
    model.pit_loss_distributions = extract_pit_loss_distributions(laps_df)
    print('Extracting safety car rates...')
    model.safety_car_rates = extract_safety_car_rates(laps_df)
    print('Extracting base lap times...')
    model.base_lap_times = extract_base_lap_times(laps_df)
    print(f'Saving world model to {output_path}...')
    model.save(output_path)
    print('\n' + '=' * 50)
    print('WORLD MODEL SUMMARY')
    print('=' * 50)
    print(f'Circuits with degradation data: {len(model.tyre_degradation)}')
    print(f'Circuits with pit data: {len(model.pit_loss_distributions)}')
    print(f'Circuits with SC data: {len(model.safety_car_rates)}')
    print(f'Circuits with base times: {len(model.base_lap_times)}')
    return model
if __name__ == '__main__':
    data_dir = '../f1_data/f1_data_parquet'
    output_path = 'world_model.pkl'
    model = build_world_model(data_dir, output_path)
    print('\nSample Degradation Data (Silverstone):')
    if 'British Grand Prix' in model.tyre_degradation:
        for (compound, params) in model.tyre_degradation['British Grand Prix'].items():
            print(f"  {compound}: mu={params['mu']:.4f}s/lap, sigma={params['sigma']:.3f}s")