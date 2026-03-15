import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_qualifying_pace(pace_df, output_path):
    plt.style.use('dark_background')
    if pace_df.empty:
        print('No pace data available to plot.')
        return
    stats = pace_df.groupby('Driver')['GapToUltimatePct'].agg(['mean', 'std', 'count']).reset_index()
    stats = stats[stats['count'] >= 5].sort_values('mean')
    (fig, ax) = plt.subplots(figsize=(14, 8))
    colors = []
    for gap in stats['mean']:
        if gap < 0.2:
            colors.append('#238636')
        elif gap < 0.5:
            colors.append('#3fb950')
        elif gap < 1.0:
            colors.append('#f0883e')
        else:
            colors.append('#da3633')
    bars = ax.bar(stats['Driver'], stats['mean'], yerr=stats['std'], capsize=5, color=colors, edgecolor='#30363d', alpha=0.9)
    ax.set_ylabel('Gap to Theoretical Best (%)', fontsize=12, fontweight='bold')
    ax.set_title('QUALIFYING PACE: Gap to Ultimate Lap\n(Theoretical Best Sectors)', fontsize=14, fontweight='bold', color='#58a6ff')
    ax.tick_params(axis='x', rotation=45)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, height + 0.05, f'{height:.2f}%', ha='center', va='bottom', fontsize=9, color='#c9d1d9')
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='#0d1117')
    plt.close()
    print(f'Saved: {output_path}')

def plot_pace_evolution(pace_df, output_path):
    top_drivers = ['VER', 'HAM', 'LEC', 'NOR', 'ALO', 'RUS', 'SAI', 'PIA']
    (fig, ax) = plt.subplots(figsize=(16, 8))
    colors = {'VER': '#1e41ff', 'HAM': '#00d2be', 'LEC': '#dc0000', 'NOR': '#ff8700', 'RUS': '#00d2be', 'ALO': '#006f62', 'SAI': '#dc0000', 'PIA': '#ff8700'}
    for driver in top_drivers:
        driver_data = pace_df[pace_df['Driver'] == driver].sort_values(['Year', 'Event'])
        if len(driver_data) < 5:
            continue
        x = range(len(driver_data))
        y = driver_data['GapToUltimatePct'].rolling(window=3, center=True).mean()
        ax.plot(x, y, label=driver, color=colors.get(driver, 'white'), linewidth=2)
    ax.set_ylabel('Gap to Ultimate (%) - Smoothed', fontsize=12, fontweight='bold')
    ax.set_xlabel('Race Sequence', fontsize=12, fontweight='bold')
    ax.set_title('PACE EVOLUTION: Gap to Ultimate Lap over Time', fontsize=14, fontweight='bold', color='#58a6ff')
    ax.legend(fontsize=10, facecolor='#161b22', edgecolor='#30363d', ncol=2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='#0d1117')
    plt.close()
    print(f'Saved: {output_path}')