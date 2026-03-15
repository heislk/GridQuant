import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from collections import defaultdict
from metrics.common import calculate_gini_coefficient

def plot_competitive_balance(results_df, output_path):
    plt.style.use('dark_background')
    results_df = results_df.copy()
    results_df['Points'] = pd.to_numeric(results_df['Points'], errors='coerce').fillna(0)
    yearly_gini = []
    yearly_std = []
    years = sorted(results_df['Year'].unique())
    for year in years:
        year_results = results_df[results_df['Year'] == year]
        driver_points = year_results.groupby('Abbreviation')['Points'].sum()
        if len(driver_points) > 5:
            gini = calculate_gini_coefficient(driver_points.values)
            std = driver_points.std()
            yearly_gini.append({'year': year, 'gini': gini})
            yearly_std.append({'year': year, 'std': std})
    (fig, axes) = plt.subplots(1, 2, figsize=(16, 6))
    gini_df = pd.DataFrame(yearly_gini)
    colors = ['#da3633' if y < 2021 else '#238636' for y in gini_df['year']]
    axes[0].bar(range(len(gini_df)), gini_df['gini'], color=colors, edgecolor='white', linewidth=0.5)
    axes[0].set_xticks(range(len(gini_df)))
    axes[0].set_xticklabels(gini_df['year'], fontsize=10)
    axes[0].axvline(x=1.5, color='#f0883e', linewidth=2, linestyle='--', label='Budget Cap (2021)')
    axes[0].set_xlabel('Season', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Gini Coefficient', fontsize=12, fontweight='bold')
    axes[0].set_title('COMPETITIVE INEQUALITY\n(Lower = More Equal)', fontsize=14, fontweight='bold', color='#58a6ff')
    axes[0].legend(fontsize=9, facecolor='#161b22', edgecolor='#30363d')
    std_df = pd.DataFrame(yearly_std)
    axes[1].plot(range(len(std_df)), std_df['std'], color='#58a6ff', linewidth=3, marker='o', markersize=8)
    axes[1].fill_between(range(len(std_df)), std_df['std'], alpha=0.3, color='#58a6ff')
    axes[1].set_xticks(range(len(std_df)))
    axes[1].set_xticklabels(std_df['year'], fontsize=10)
    axes[1].axvline(x=1.5, color='#f0883e', linewidth=2, linestyle='--', label='Budget Cap (2021)')
    axes[1].set_xlabel('Season', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Points Std Dev', fontsize=12, fontweight='bold')
    axes[1].set_title('POINTS SPREAD\n(Lower = Closer Competition)', fontsize=14, fontweight='bold', color='#58a6ff')
    axes[1].legend(fontsize=9, facecolor='#161b22', edgecolor='#30363d')
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='#0d1117')
    plt.close()
    print(f'Saved: {output_path}')

def plot_team_history(elo_history, output_path):
    team_drivers = {'Red Bull': ['VER', 'PER', 'ALB', 'GAS'], 'Mercedes': ['HAM', 'RUS', 'BOT'], 'Ferrari': ['LEC', 'SAI', 'VET'], 'McLaren': ['NOR', 'PIA', 'RIC', 'SAI']}
    (fig, ax) = plt.subplots(figsize=(16, 8))
    team_colors = {'Red Bull': '#1e41ff', 'Mercedes': '#00d2be', 'Ferrari': '#dc0000', 'McLaren': '#ff8700'}
    for (team, drivers) in team_drivers.items():
        team_elos = defaultdict(list)
        for driver in drivers:
            if driver in elo_history:
                for entry in elo_history[driver]:
                    key = f"{entry['year']} {entry['event']}"
                    team_elos[key].append(entry['elo'])
        x_labels = []
        y_values = []
        sorted_races = sorted(team_elos.keys())
        for race in sorted_races:
            elos = team_elos[race]
            if elos:
                y_values.append(np.mean(elos))
                x_labels.append(race)
        if len(y_values) > 5:
            y_smooth = pd.Series(y_values).rolling(window=5, center=True).mean()
            ax.plot(range(len(y_values)), y_smooth, label=team, color=team_colors[team], linewidth=3)
    ax.set_ylabel('Average Driver ELO', fontsize=12, fontweight='bold')
    ax.set_title('TEAM PERFORMANCE HISTORY\n(Aggregated Driver ELO)', fontsize=14, fontweight='bold', color='#58a6ff')
    ax.legend(fontsize=10, facecolor='#161b22', edgecolor='#30363d')
    ax.set_xlabel('Race Sequence (2019-2025)', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='#0d1117')
    plt.close()
    print(f'Saved: {output_path}')