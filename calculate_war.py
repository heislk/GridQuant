import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from metrics.common import load_all_data
from metrics.war import calculate_raw_deltas, calculate_elo_ratings, calculate_adjusted_war
from charts.driver_charts import plot_war_rankings

def print_war_report(war_df, elo_ratings):
    print('\n' + '=' * 75)
    print('DRIVER VALUATION: ELO-WEIGHTED WINS ABOVE REPLACEMENT (WAR)')
    print('Methodology: Teammate Comparison with ELO Quality Adjustment')
    print('=' * 75)
    print('\nTOP 15 DRIVERS (2019-2025):')
    print('-' * 75)
    print(f"{'Rank':<5} {'Driver':<6} {'Adj WAR':>10} {'Raw':>8} {'ELO':>7} {'Races':>7} {'Primary Team':<20}")
    print('-' * 75)
    for (i, (_, row)) in enumerate(war_df.head(15).iterrows(), 1):
        primary_team = row['Teams'][0] if row['Teams'] else 'N/A'
        if len(primary_team) > 18:
            primary_team = primary_team[:18] + '..'
        print(f"{i:<5} {row['Driver']:<6} {row['AdjustedWAR']:>+.3f}s {row['RawDelta']:>+.3f}s {row['ELO']:>7.0f} {row['RaceCount']:>7} {primary_team:<20}")

def main():
    data_dir = '../f1_data/f1_data_parquet'
    print('Loading F1 data...')
    (laps_df, results_df) = load_all_data(data_dir)
    print(f'Loaded {len(laps_df):,} laps, {len(results_df):,} results')
    print('\nCalculating raw teammate deltas...')
    raw_deltas_df = calculate_raw_deltas(laps_df, results_df)
    print('\nBuilding ELO rating system...')
    (elo_ratings, _) = calculate_elo_ratings(raw_deltas_df)
    print('\nComputing ELO-adjusted WAR...')
    war_df = calculate_adjusted_war(raw_deltas_df, elo_ratings)
    print_war_report(war_df, elo_ratings)
    plot_war_rankings(war_df, elo_ratings, 'war_rankings.png')
    war_df.to_csv('driver_war_rankings.csv', index=False)
    print('\nSaved: driver_war_rankings.csv')
if __name__ == '__main__':
    main()