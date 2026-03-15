import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from metrics.common import load_all_data
from metrics.war import calculate_raw_deltas, calculate_elo_ratings, calculate_adjusted_war
from metrics.simulation import RaceSimulator, compare_strategies, get_default_strategies, TyreModel
from metrics.qualifying import calculate_theoretical_best_laps
from metrics.optimization import StrategyOptimizer
from metrics.championship import ChampionshipSimulator, create_driver_profiles
from metrics.qualifying_sim import QualifyingSimulator, build_profiles_from_laps
import metrics.telemetry as telemetry
import metrics.racecraft as racecraft
import charts.advanced_charts as advanced_charts
from metrics.overtake import run_overtake_simulation_grid
import metrics.dna as dna
import metrics.sector_sim as sector_sim
from build_world_model import WorldModel
from charts.driver_charts import plot_war_rankings, plot_elo_timeline, plot_dominance, plot_risk_return
from charts.team_charts import plot_competitive_balance, plot_team_history
from charts.tech_charts import plot_tyre_model, plot_strategy_analysis
from charts.qualifying_charts import plot_qualifying_pace, plot_pace_evolution
from charts.simulation_charts import plot_championship_probabilities, plot_qualifying_probabilities

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    data_dir = os.path.join(project_root, 'f1_data', 'f1_data_parquet')
    base_results = os.path.join(script_dir, 'results')

    dirs = {
        'drivers': os.path.join(base_results, 'drivers'),
        'teams': os.path.join(base_results, 'teams'),
        'simulation': os.path.join(base_results, 'simulation'),
        'data': os.path.join(base_results, 'data'),
        'rankings': os.path.join(base_results, 'drivers', 'rankings'),
        'dna': os.path.join(base_results, 'drivers', 'dna'),
        'performance': os.path.join(base_results, 'drivers', 'performance'),
        'sectors': os.path.join(base_results, 'simulation', 'sectors'),
        'strategies': os.path.join(base_results, 'simulation', 'strategies'),
        'championship': os.path.join(base_results, 'simulation', 'championship'),
        'qualifying': os.path.join(base_results, 'simulation', 'qualifying')
    }
    for d in dirs.values():
        if not os.path.exists(d):
            os.makedirs(d)
    print('=' * 70)
    print('F1 SABERMETRICS PIPELINE')
    print('=' * 70)
    print('\n[PHASE 1] Loading Data...')
    (laps_df, results_df) = load_all_data(data_dir)
    print(f'  Loaded {len(laps_df):,} laps, {len(results_df):,} results')
    print('  Loading Telemetry (this may take a moment)...')
    try:
        import pandas as pd
        telemetry_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if 'telemetry' in f]
        telemetry_dfs = []
        for f in sorted(telemetry_files):
            df = pd.read_parquet(f, columns=['Session', 'Driver', 'Date', 'Speed', 'RPM', 'nGear', 'Throttle', 'Brake', 'DRS', 'Event'])
            year = int(f.split('_')[-1].split('.')[0])
            df['Year'] = year
            telemetry_dfs.append(df)
        telemetry_df = pd.concat(telemetry_dfs, ignore_index=True)
        print(f'  Loaded {len(telemetry_df):,} telemetry points')
    except Exception as e:
        print(f'Warning: Telemetry load failed: {e}')
        telemetry_df = pd.DataFrame()
    
    print('\n[PHASE 2] Calculating Metrics...')
    print('  Calculating driver deltas...')
    raw_deltas_df = calculate_raw_deltas(laps_df, results_df)
    print('  Updating ELO ratings...')
    (elo_ratings, elo_history) = calculate_elo_ratings(raw_deltas_df)
    print('  Computing WAR...')
    war_df = calculate_adjusted_war(raw_deltas_df, elo_ratings)
    war_csv_path = os.path.join(dirs['data'], 'driver_war_rankings.csv')
    war_df.to_csv(war_csv_path, index=False)
    print(f'  Saved: {war_csv_path}')
    
    # -------------------------------------------------------------
    # NEW: Championship Simulation
    # -------------------------------------------------------------
    print('\n[PHASE 2b] Running Championship Simulation (Plackett-Luce + Car Strength)...')
    try:
        # Filter for active 2024 drivers using V2 (Points-based Car Strength)
        from metrics.championship import create_driver_profiles_v2
        profiles = create_driver_profiles_v2(war_df, results_df, target_year=2024)
        
        if profiles:
            champ_sim = ChampionshipSimulator(profiles, n_sims=10000)
            champ_results = champ_sim.simulate()
            champ_path = os.path.join(dirs['championship'], 'season_simulation.csv')
            champ_results.to_csv(champ_path, index=False)
            print(f'  Saved: {champ_path}')
            
            from charts.simulation_charts import plot_championship_probabilities
            plot_championship_probabilities(champ_results, os.path.join(dirs['championship'], 'championship_win_probs.png'))
            
        else:
            print('  Skipping Championship Sim (No profiles for 2024).')
    except Exception as e:
        print(f'  Error in Championship Sim: {e}')

    print('\n[PHASE 3] Generating Visualizations...')
    print('  [1/7] WAR Rankings...')
    plot_war_rankings(war_df, elo_ratings, os.path.join(dirs['rankings'], 'war_rankings.png'))
    print('  [2/7] ELO Timeline...')
    plot_elo_timeline(elo_history, os.path.join(dirs['rankings'], 'elo_timeline.png'))
    print('  [3/7] Risk/Return...')
    plot_risk_return(war_df, os.path.join(dirs['rankings'], 'risk_return.png'))
    print('  [4/7] Driver Dominance...')
    plot_dominance(raw_deltas_df, os.path.join(dirs['performance'], 'driver_dominance.png'))
    print('  [5/7] Competitive Balance...')
    plot_competitive_balance(results_df, os.path.join(dirs['teams'], 'competitive_balance.png'))
    print('  [6/7] Team History...')
    plot_team_history(elo_history, os.path.join(dirs['teams'], 'team_history.png'))
    print('  [7/7] Tyre Model...')
    try:
        model = WorldModel.load('world_model.pkl')
        plot_tyre_model(model, os.path.join(dirs['strategies'], 'tyre_model.png'))
    except Exception as e:
        print(f'  Skipping Tyre Model (missing pickle): {e}')
    print('\n[PHASE 4] Running Strategy Simulation...')
    print('  Initializing Monte Carlo Simulator (Silverstone model)...')
    simulator = RaceSimulator(total_laps=52, base_lap_time=92.5, pit_loss=22.0)
    strategies = get_default_strategies()
    print('  Running 10,000 simulations per strategy...')
    analysis = compare_strategies(simulator, strategies, n_sims=10000)
    print('  Generating strategy_comparison.png...')
    plot_strategy_analysis(analysis, os.path.join(dirs['strategies'], 'strategy_comparison.png'))
    
    print('\n[PHASE 5] Analyzing Qualifying Pace...')
    print('  Calculating theoretical best laps (Gap to Ultimate)...')
    pace_df = calculate_theoretical_best_laps(laps_df)
    if not pace_df.empty:
        print('  Generating qualifying_pace.png...')
        plot_qualifying_pace(pace_df, os.path.join(dirs['performance'], 'qualifying_pace.png'))
        print('  Generating pace_evolution.png...')
        plot_pace_evolution(pace_df, os.path.join(dirs['performance'], 'pace_evolution.png'))
    else:
        print('  Skipping Quaifying charts (no data).')

    # -------------------------------------------------------------
    # NEW: Qualifying Simulation (Knockout)
    # -------------------------------------------------------------
    print('\n[PHASE 5b] Running Qualifying Knockout Simulation (All Tracks)...')
    try:
        # Get all events from 2024
        target_year = 2024
        if 2024 not in laps_df['Year'].values:
             target_year = laps_df['Year'].max()
             
        # List of events
        events_2024 = laps_df[laps_df['Year'] == target_year]['Event'].unique()
        print(f"  Found {len(events_2024)} events for {target_year}. Simulating each...")
        
        for event in events_2024:
            print(f"    Simulating Qualifying for {event}...")
            q_laps = laps_df[(laps_df['Year'] == target_year) & (laps_df['Event'] == event)].copy()
            
            if len(q_laps) < 100: # Skip if barely any data 
                continue
                
            q_profiles = build_profiles_from_laps(q_laps, target_year=target_year)
            
            if q_profiles:
                q_sim = QualifyingSimulator(q_profiles, n_sims=1000)
                q_results = q_sim.simulate()
                q_path = os.path.join(dirs['qualifying'], f'qualifying_sim_{target_year}_{event.replace(" ", "_")}.csv')
                q_results.to_csv(q_path, index=False)
                
                # Plot
                from charts.simulation_charts import plot_qualifying_probabilities
                plot_qualifying_probabilities(q_results, os.path.join(dirs['qualifying'], f'qualifying_probs_{target_year}_{event.replace(" ", "_")}.png'))
            else:
                pass # Silently skip empty profiles
                
    except Exception as e:
        print(f'  Error in Qualifying Sim: {e}')

    print('\n[PHASE 6] Running Deterministic Strategy Optimizer...')
    print('  Solving Bellman Equation for Optimal Strategy (Silverstone)...')
    tyre_models = {'SOFT': TyreModel(base_deg_per_lap=0.065, volatility=0.15, cliff_lap=22, cliff_penalty=0.25), 'MEDIUM': TyreModel(base_deg_per_lap=0.04, volatility=0.12, cliff_lap=32, cliff_penalty=0.2), 'HARD': TyreModel(base_deg_per_lap=0.025, volatility=0.1, cliff_lap=45, cliff_penalty=0.15)}
    optimizer = StrategyOptimizer(total_laps=52, pit_loss=22.0, tyre_models=tyre_models, base_lap_time=92.5)
    optimal_result = optimizer.solve()
    opt_strategy = optimal_result['strategy']
    print(f"  Optimal Time: {optimal_result['optimal_time']:.2f}s")
    print('  Optimal Path found by DP:')
    for step in opt_strategy:
        if step['action'].startswith('box'):
            print(f"    Lap {step['lap']}: BOX for {step['action'].split('_')[1]} (Stint Time: {step['time']:.1f}s)")
        else:
            pass
    current_compound = opt_strategy[0]['compound']
    stint_start = 1
    for (i, step) in enumerate(opt_strategy):
        if step['action'].startswith('box'):
            print(f"    Stint: {current_compound} (Laps {stint_start}-{step['lap']})")
            current_compound = step['action'].split('_')[1]
            stint_start = step['lap'] + 1
    print(f'    Stint: {current_compound} (Laps {stint_start}-52)')
    print('\n[PHASE 7] Analyzing Telemetry Signatures...')
    try:
        sig_df = telemetry.calculate_telemetry_signature(telemetry_df, laps_df, results_df)
        advanced_charts.plot_telemetry_signature(sig_df, os.path.join(dirs['teams'], 'telemetry_signature.png'))
    except Exception as e:
        print(f'Skipping Telemetry Signature due to error: {e}')
    print('\n[PHASE 8] Analyzing Start Performance...')
    try:
        lap1_df = racecraft.calculate_lap1_performance(laps_df, results_df)
        advanced_charts.plot_lap1_performance(lap1_df, os.path.join(dirs['performance'], 'lap1_performance.png'))
    except Exception as e:
        print(f'Skipping Lap 1 Analysis due to error: {e}')
    print('\n[PHASE 9] Running Overtake Simulation (3D)...')
    try:
        overtake_grid = run_overtake_simulation_grid()
        advanced_charts.plot_overtake_3d(overtake_grid, os.path.join(dirs['simulation'], 'overtake_probability_3d.png'))
    except Exception as e:
        print(f'Skipping Overtake Sim due to error: {e}')
    except Exception as e:
        print(f'Skipping Overtake Sim due to error: {e}')
        print('\n[PHASE 10] Analyzing Driver DNA (Bayesian)...')
    try:
        dna_df = dna.calculate_bayesian_dna(war_df, lap1_df, pace_df, laps_df)
        advanced_charts.plot_driver_dna(dna_df, os.path.join(dirs['dna'], 'driver_dna.png'))
        
        # New: DNA Battle Comparisons
        if not dna_df.empty:
            sorted_dna = dna_df.sort_values('BattleRating', ascending=False)
            if len(sorted_dna) >= 2:
                # 1 vs 2 Battle
                d1 = sorted_dna.iloc[0]['Driver']
                d2 = sorted_dna.iloc[1]['Driver']
                advanced_charts.plot_dna_comparison(dna_df, d1, d2, os.path.join(dirs['dna'], f'dna_battle_{d1}_vs_{d2}.png'))
                
            if len(sorted_dna) >= 3:
                advanced_charts.plot_dna_battle_royale(dna_df, os.path.join(dirs['dna'], 'dna_battle_royale.png'))
                
    except Exception as e:
        print(f"Skipping Driver DNA due to error: {e}")

    except Exception as e:
        print(f"Skipping Driver DNA due to error: {e}")

    # ---------------------------------------------------------
    # 11. RECENT FORM ANALYSIS (2023-2024)
    # ---------------------------------------------------------
    print('\n[PHASE 11] Running Recent Form Analysis (2023-2024)...')
    try:
        # Filter Data
        recent_years = [2023, 2024]
        laps_recent = laps_df[laps_df['Year'].isin(recent_years)].copy()
        results_recent = results_df[results_df['Year'].isin(recent_years)].copy()
        
        if not laps_recent.empty:
            print(f"  Analyzing {len(laps_recent):,} laps from 2023-2024...")
            
            # 1. WAR & Risk/Return
            raw_deltas_recent = calculate_raw_deltas(laps_recent, results_recent)
            elo_r_recent, _ = calculate_elo_ratings(raw_deltas_recent) # Recalc ELO from scratch or use history? Scratch for pure form.
            war_recent = calculate_adjusted_war(raw_deltas_recent, elo_r_recent)
            
            plot_risk_return(war_recent, os.path.join(dirs['rankings'], 'risk_return_2023_2024.png'))
            plot_war_rankings(war_recent, elo_r_recent, os.path.join(dirs['rankings'], 'war_rankings_2023_2024.png'))
            
            # 2. Driver DNA (Recent)
            # Need recent qualifying pace and lap1 too
            pace_recent = calculate_theoretical_best_laps(laps_recent)
            lap1_recent = racecraft.calculate_lap1_performance(laps_recent, results_recent)
            
            dna_recent = dna.calculate_bayesian_dna(war_recent, lap1_recent, pace_recent, laps_recent)
            advanced_charts.plot_driver_dna(dna_recent, os.path.join(dirs['dna'], 'driver_dna_2023_2024.png'))
            
            # Battles (Recent)
            if not dna_recent.empty:
                sorted_dna = dna_recent.sort_values('BattleRating', ascending=False)
                if len(sorted_dna) >= 2:
                    d1 = sorted_dna.iloc[0]['Driver']
                    d2 = sorted_dna.iloc[1]['Driver']
                    advanced_charts.plot_dna_comparison(dna_recent, d1, d2, os.path.join(dirs['dna'], f'dna_battle_{d1}_vs_{d2}_2023_2024.png'))
                
                if len(sorted_dna) >= 3:
                     advanced_charts.plot_dna_battle_royale(dna_recent, os.path.join(dirs['dna'], 'dna_battle_royale_2023_2024.png'))
                     
    except Exception as e:
        print(f"Skipping Recent Form Analysis due to error: {e}")

    # ---------------------------------------------------------
    # 12. STOCHASTIC SECTOR SIMULATION (1M LAPS)
    # ---------------------------------------------------------
    print('\n[PHASE 12] Running Stochastic Sector Simulation (1,000,000 Laps)...')
    try:
        # Use recent data for relevance (2024)
        sim_year = 2024
        print(f"  Simulating 1,000,000 laps per driver for {sim_year} tracks...")
        sector_results = sector_sim.simulate_sector_performance(laps_df, year=sim_year, n_sims=1000000)
        
        if not sector_results.empty:
            advanced_charts.plot_sector_dominance(sector_results, os.path.join(dirs['sectors'], f'sector_dominance_{sim_year}.png'))
    except Exception as e:
        print(f"Skipping Sector Simulation due to error: {e}")

    # ---------------------------------------------------------
    # 13. RIVALRY TELEMETRY (GHOST CAR)
    # ---------------------------------------------------------
    print('\n[PHASE 13] Running Ghost Car Telemetry Analysis...')
    try:
        from charts.telemetry_charts import plot_ghost_car_comparison
        
        # 1. Detect tightest Quali battle in 2024
        # We need Qualifying Results: session type 'Qualifying' or 'Q3'
        q_res = results_df[ (results_df['Year'] == 2024) & (results_df['Session'].isin(['Qualifying', 'Q3'])) ].copy()
        
        # If 'Session' isn't reliable, assume GridPosition is a proxy (not perfect due to penalties)
        # Better: Look at 'Q3' column if exists
        
        best_battle = None
        min_delta = 999.0
        
        # Group by event
        if not q_res.empty:
            for event, grp in q_res.groupby('Event'):
                # Sort by position
                grp = grp.sort_values('Position')
                if len(grp) < 2: continue
                
                p1 = grp.iloc[0]
                p2 = grp.iloc[1]
                
                # Check time delta
                # We need Q3 time. 'Q3' column?
                # results_df usually has Q1, Q2, Q3 as timedelta strings or objects
                # Or 'Time' is the relevant time.
                
                # If 'Time' is race time, this won't work.
                # Assuming 'Q3' column exists or 'Time' for Quali session
                
                # Let's try Q3 column first
                t1 = p1.get('Q3')
                t2 = p2.get('Q3')
                
                # Convert to float
                def to_sec(t):
                    if pd.isna(t): return None
                    if isinstance(t, pd.Timedelta): return t.total_seconds()
                    # Parse string? skipping complexity for now
                    return None
                    
                s1 = to_sec(t1)
                s2 = to_sec(t2)
                
                if s1 and s2:
                    delta = abs(s1 - s2)
                    if delta < min_delta:
                        min_delta = delta
                        best_battle = (event, p1['Abbreviation'], p2['Abbreviation'])
        
        # Fallback if detection fails: VER vs NOR at random track
        if not best_battle:
            print("  No Q3 data found for auto-battle. Defaulting to VER vs NOR.")
            best_battle = ('British Grand Prix', 'VER', 'NOR')
            
        event_name, d1, d2 = best_battle
        print(f"  Tightest Battle Detected: {d1} vs {d2} at {event_name} (Delta: {min_delta:.3f}s)")
        
        # Get Telemetry
        # We need to filter telemetry_df by Event and Driver
        # Telemetry load was Phase 1... do we have 'Event' in telemetry?
        # The telemetry loader used filenames: 'telemetry_2024.parquet'. 
        # It has 'Session' and 'Date'. It might NOT have 'Event' explicitly if filename didn't have it.
        # But 'Date' helps match result?
        
        # Let's assume we can match by Date? 
        # Actually telemetry loader regex was: year.
        # We need to filter by Event. 
        # common.py's telem loader just loaded everything.
        # We need to intersect with laps_df to find the Session Key or Date range.
        
        # SIMPLIFICATION:
        # Just plot VER vs NOR data if available, assuming valid timestamps.
        # Ideally we map Event -> Date Range.
        
        # For this demo, let's just pass the whole DF and filter by Driver. 
        # Telemetry Charts needs to be smart about selecting the "Fastest Lap" slice.
        # We will update plot_ghost_car_comparison to handle extracting the lap.
        
        # Updating logic to pass filtered telemetry
        # Filter telemetry by Event and Driver directly
        # (Event column added to loader in Phase 1)
        telem_full = telemetry_df[ 
            (telemetry_df['Driver'].isin([d1, d2])) & 
            (telemetry_df['Year'] == 2024) &
            (telemetry_df['Event'] == event_name)
        ].copy()

        # SLICE FASTEST LAPS
        # 1. Get Fastest Lap info from laps_df
        laps_battle = laps_df[ 
            (laps_df['Event'] == event_name) & 
            (laps_df['Year'] == 2024) & 
            (laps_df['Driver'].isin([d1, d2])) & 
            (laps_df['IsAccurate'] == True) # Ensure valid lap
        ].copy()
        
        # Sort by LapTime to find fastest
        fastest_laps = laps_battle.sort_values('LapTime').groupby('Driver').first()
        
        telem_slices = []
        
        for driver in [d1, d2]:
            if driver not in fastest_laps.index:
                print(f"  Warning: No valid fastest lap found for {driver}")
                continue
                
            lap = fastest_laps.loc[driver]
            start_time = lap['LapStartTime']
            lap_duration = lap['LapTime']
            
            # StartTime is SessionTime (Timedelta).
            # Telemetry has 'Date' (Timestamp) and sometimes 'SessionTime' (Timedelta).
            # If 'SessionTime' exists in telemetry, use that.
            
            if 'SessionTime' in telem_full.columns:
                end_time = start_time + lap_duration
                # Buffer 2s
                slice_df = telem_full[
                    (telem_full['Driver'] == driver) & 
                    (telem_full['SessionTime'] >= start_time - pd.Timedelta(seconds=1)) &
                    (telem_full['SessionTime'] <= end_time + pd.Timedelta(seconds=1))
                ].copy()
            else:
                # Fallback to Date if we can sync? 
                # Laps has LapStartDate.
                start_date = lap['LapStartDate']
                end_date = start_date + lap_duration
                slice_df = telem_full[
                    (telem_full['Driver'] == driver) & 
                    (telem_full['Date'] >= start_date - pd.Timedelta(seconds=1)) &
                    (telem_full['Date'] <= end_date + pd.Timedelta(seconds=1))
                ].copy()
                
            telem_slices.append(slice_df)
            
        if telem_slices:
            telem_battle = pd.concat(telem_slices)
            plot_ghost_car_comparison(telem_battle, d1, d2, event_name, os.path.join(dirs['performance'], f'ghost_car_{d1}_vs_{d2}.png'))
            
    except Exception as e:
        print(f"Skipping Ghost Car Analysis due to error: {e}")

    print('\nPipeline Complete.')
if __name__ == '__main__':
    main()