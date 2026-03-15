import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from collections import defaultdict

def plot_war_rankings(war_df, elo_ratings, output_path):
    plt.style.use('dark_background')
    top_drivers = war_df.head(20).copy()
    (fig, axes) = plt.subplots(1, 2, figsize=(16, 10))
    colors = []
    for delta in top_drivers['AdjustedWAR']:
        if delta > 0.3:
            colors.append('#238636')
        elif delta > 0.1:
            colors.append('#3fb950')
        elif delta > -0.1:
            colors.append('#f0883e')
        elif delta > -0.3:
            colors.append('#da3633')
        else:
            colors.append('#8b0000')
    bars = axes[0].barh(range(len(top_drivers)), top_drivers['AdjustedWAR'], color=colors, edgecolor='#30363d', linewidth=0.5)
    axes[0].set_yticks(range(len(top_drivers)))
    axes[0].set_yticklabels(top_drivers['Driver'], fontsize=11, fontweight='bold')
    axes[0].axvline(x=0, color='#8b949e', linewidth=1, linestyle='--')
    axes[0].set_xlabel('ELO-Adjusted WAR (seconds/race)', fontsize=12, fontweight='bold')
    axes[0].set_title('DRIVER WAR: ELO-Weighted Performance\n(2019-2025)', fontsize=14, fontweight='bold', color='#58a6ff')
    axes[0].invert_yaxis()
    for (i, (bar, delta, races, elo)) in enumerate(zip(bars, top_drivers['AdjustedWAR'], top_drivers['RaceCount'], top_drivers['ELO'])):
        label = f'{delta:+.3f}s | ELO: {elo:.0f}'
        x_pos = max(bar.get_width() + 0.02, 0.05)
        axes[0].text(x_pos, i, label, va='center', ha='left', fontsize=9, color='#8b949e')
    sorted_elo = sorted(elo_ratings.items(), key=lambda x: x[1], reverse=True)[:25]
    drivers_elo = [x[0] for x in sorted_elo]
    elos = [x[1] for x in sorted_elo]
    elo_colors = []
    for e in elos:
        if e > 1600:
            elo_colors.append('#238636')
        elif e > 1500:
            elo_colors.append('#3fb950')
        elif e > 1400:
            elo_colors.append('#f0883e')
        else:
            elo_colors.append('#da3633')
    axes[1].barh(range(len(drivers_elo)), elos, color=elo_colors, edgecolor='#30363d', linewidth=0.5)
    axes[1].set_yticks(range(len(drivers_elo)))
    axes[1].set_yticklabels(drivers_elo, fontsize=10, fontweight='bold')
    axes[1].axvline(x=1500, color='#8b949e', linewidth=1, linestyle='--', label='Average (1500)')
    axes[1].set_xlabel('ELO Rating', fontsize=12, fontweight='bold')
    axes[1].set_title('DRIVER ELO RATINGS\n(Chess-Style Ranking)', fontsize=14, fontweight='bold', color='#58a6ff')
    axes[1].invert_yaxis()
    axes[1].legend(fontsize=9, facecolor='#161b22', edgecolor='#30363d')
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='#0d1117')
    plt.close()
    print(f'Saved: {output_path}')

def plot_elo_timeline(elo_history, output_path):
    top_drivers = ['VER', 'HAM', 'LEC', 'NOR', 'RUS', 'ALO', 'PER', 'SAI']
    colors = {'VER': '#1e41ff', 'HAM': '#00d2be', 'LEC': '#dc0000', 'NOR': '#ff8700', 'RUS': '#00d2be', 'ALO': '#006f62', 'PER': '#1e41ff', 'SAI': '#dc0000'}
    (fig, ax) = plt.subplots(figsize=(16, 8))
    for driver in top_drivers:
        if driver not in elo_history:
            continue
        history = elo_history[driver]
        elos = [h['elo'] for h in history]
        x = range(len(elos))
        color = colors.get(driver, '#888888')
        ax.plot(x, elos, label=driver, color=color, linewidth=2.5, alpha=0.9)
        if len(elos) > 0:
            ax.annotate(driver, (len(elos) - 1, elos[-1]), textcoords='offset points', xytext=(5, 0), fontsize=10, fontweight='bold', color=color)
    ax.axhline(y=1500, color='#8b949e', linewidth=1, linestyle='--', alpha=0.7, label='Average (1500)')
    ax.set_xlabel('Race Number (Chronological)', fontsize=12, fontweight='bold')
    ax.set_ylabel('ELO Rating', fontsize=12, fontweight='bold')
    ax.set_title('DRIVER ELO EVOLUTION (2019-2025)\nChess-Style Rating Over Time', fontsize=14, fontweight='bold', color='#58a6ff')
    ax.legend(loc='upper left', fontsize=9, facecolor='#161b22', edgecolor='#30363d', ncol=2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='#0d1117')
    plt.close()
    print(f'Saved: {output_path}')

def plot_risk_return(war_df, output_path):
    (fig, ax) = plt.subplots(figsize=(12, 10))
    colors = []
    for delta in war_df['AdjustedWAR']:
        if delta > 0.2:
            colors.append('#238636')
        elif delta > 0:
            colors.append('#3fb950')
        elif delta > -0.2:
            colors.append('#f0883e')
        else:
            colors.append('#da3633')
    sizes = war_df['RaceCount'] * 4
    scatter = ax.scatter(war_df['StdDelta'], war_df['AdjustedWAR'], s=sizes, c=colors, alpha=0.7, edgecolors='white', linewidth=1)
    for (i, row) in war_df.iterrows():
        ax.annotate(row['Driver'], (row['StdDelta'], row['AdjustedWAR']), textcoords='offset points', xytext=(5, 5), fontsize=9, color='#c9d1d9', fontweight='bold')
    ax.axhline(y=0, color='#8b949e', linewidth=1, linestyle='--', alpha=0.7)
    ax.axvline(x=war_df['StdDelta'].median(), color='#8b949e', linewidth=1, linestyle=':', alpha=0.5)
    ax.set_xlabel('Volatility (Std Dev of Performance)', fontsize=12, fontweight='bold')
    ax.set_ylabel('ELO-Adjusted WAR (seconds/race)', fontsize=12, fontweight='bold')
    ax.set_title('RISK-RETURN PROFILE OF F1 DRIVERS\n(Bubble Size = Race Count)', fontsize=14, fontweight='bold', color='#58a6ff')
    ax.annotate('HIGH RETURN\nLOW RISK\n(Elite)', xy=(0.02, 0.98), xycoords='axes fraction', fontsize=10, color='#238636', fontweight='bold', ha='left', va='top')
    ax.annotate('HIGH RETURN\nHIGH RISK\n(Volatile Star)', xy=(0.98, 0.98), xycoords='axes fraction', fontsize=10, color='#3fb950', fontweight='bold', ha='right', va='top')
    ax.annotate('LOW RETURN\nLOW RISK\n(Consistent Loser)', xy=(0.02, 0.02), xycoords='axes fraction', fontsize=10, color='#f0883e', fontweight='bold', ha='left', va='bottom')
    ax.annotate('LOW RETURN\nHIGH RISK\n(Liability)', xy=(0.98, 0.02), xycoords='axes fraction', fontsize=10, color='#da3633', fontweight='bold', ha='right', va='bottom')
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='#0d1117')
    plt.close()
    print(f'Saved: {output_path}')

def plot_dominance(raw_deltas_df, output_path):
    stats = defaultdict(lambda : {'wins': 0, 'races': 0})
    for (_, row) in raw_deltas_df.iterrows():
        stats[row['driver']]['races'] += 1
        if row['delta'] > 0:
            stats[row['driver']]['wins'] += 1
    data = []
    for (driver, s) in stats.items():
        if s['races'] >= 20:
            win_rate = s['wins'] / s['races']
            data.append({'Driver': driver, 'WinRate': win_rate, 'Races': s['races']})
    df = pd.DataFrame(data).sort_values('WinRate', ascending=False)
    (fig, ax) = plt.subplots(figsize=(14, 8))
    colors = []
    for rate in df['WinRate']:
        if rate > 0.7:
            colors.append('#238636')
        elif rate > 0.5:
            colors.append('#3fb950')
        elif rate > 0.3:
            colors.append('#da3633')
        else:
            colors.append('#8b0000')
    bars = ax.bar(df['Driver'], df['WinRate'] * 100, color=colors, edgecolor='#30363d')
    ax.axhline(y=50, color='#8b949e', linestyle='--', alpha=0.5)
    ax.set_ylabel('Head-to-Head Win Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('DRIVER DOMINANCE: Frequency of Beating Teammate\n(Min 20 Races)', fontsize=14, fontweight='bold', color='#58a6ff')
    ax.tick_params(axis='x', rotation=45)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, height + 1, f'{height:.0f}%', ha='center', va='bottom', fontsize=9, color='#c9d1d9')
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='#0d1117')
    plt.close()
    print(f'Saved: {output_path}')