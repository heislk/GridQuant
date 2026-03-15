import numpy as np
import pandas as pd
from scipy import stats

def calculate_bayesian_dna(war_df, lap1_df, qualifying_df, laps_df):
    dna = {}
    
    prior_mu = 0
    prior_sigma = 0.5
    prior_precision = 1 / (prior_sigma ** 2)
    
    paces = {}
    for _, row in war_df.iterrows():
        driver = row['Driver']
        obs_mean = row['AdjustedWAR']
        obs_std = row['StdDelta'] if row['StdDelta'] > 0 else 0.5
        n = row['RaceCount']
        
        data_precision = n / (obs_std ** 2)
        
        post_precision = prior_precision + data_precision
        post_mu = (prior_precision * prior_mu + data_precision * obs_mean) / post_precision
        
        paces[driver] = post_mu

    pace_vals = list(paces.values())
    min_p, max_p = min(pace_vals), max(pace_vals)
    for d in paces:
        if max_p > min_p:
            paces[d] = 50 + ((paces[d] - min_p) / (max_p - min_p)) * 50
        else:
            paces[d] = 75

    aggressions = {}
    if not lap1_df.empty:
        for _, row in lap1_df.iterrows():
            driver = row['Driver']
            net_gain = row['TotalGain']
            races = row['Races']
            
            wins = (net_gain + races) / 2
            wins = max(0, min(races, wins))
            losses = races - wins
            
            alpha_post = 2 + wins
            beta_post = 2 + losses
            post_mean = alpha_post / (alpha_post + beta_post)
            
            aggressions[driver] = post_mean * 100
    
    consistencies = {}
    
    if 'LapTimeSeconds' not in laps_df.columns:
        laps_df['LapTimeSeconds'] = laps_df['LapTime'].dt.total_seconds()
        
    grouped_laps = laps_df[laps_df['IsAccurate']].groupby('Driver')['LapTimeSeconds']
    
    for driver, times in grouped_laps:
        if len(times) < 50: continue
        
        s2 = np.var(times)
        if s2 == 0: s2 = 0.001
        
        score = 1.0 / np.sqrt(s2) 
        consistencies[driver] = score

    c_vals = list(consistencies.values())
    if c_vals:
        min_c, max_c = min(c_vals), max(c_vals)
        for d in consistencies:
            consistencies[d] = 50 + ((consistencies[d] - min_c) / (max_c - min_c)) * 50
    qualis = {}
    if not qualifying_df.empty:
        for _, row in qualifying_df.iterrows():
            driver = row['Driver']
            gap = row['GapToUltimatePct']
            score = max(50, 100 - (gap * 25))
            qualis[driver] = score
    drivers = set(paces.keys())
    dna_list = []
    
    for d in drivers:
        dna_list.append({
            'Driver': d,
            'RacePace': paces.get(d, 50),
            'Aggression': aggressions.get(d, 50),
            'Consistency': consistencies.get(d, 50),
            'QualiPace': qualis.get(d, 50),
            'BattleRating': (paces.get(d, 50) * 0.45) + (qualis.get(d, 50) * 0.35) + (aggressions.get(d, 50) * 0.10) + (consistencies.get(d, 50) * 0.10)
        })
        
    return pd.DataFrame(dna_list)
