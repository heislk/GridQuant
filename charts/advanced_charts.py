import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

def plot_telemetry_signature(sig_df, output_path):
    plt.style.use('dark_background')
    if sig_df.empty:
        print('No telemetry signature data found.')
        return
    latest_year = sig_df['Year'].max()
    data = sig_df[sig_df['Year'] == latest_year].copy()
    data = data[data['TopSpeed'] > 280]
    data = data[data['CorneringSpeed'] > 60]
    (fig, ax) = plt.subplots(figsize=(14, 10))
    team_colors = {'Red Bull': '#1e41ff', 'Mercedes': '#00d2be', 'Ferrari': '#dc0000', 'McLaren': '#ff8700', 'Aston Martin': '#229971', 'Alpine': '#0090ff', 'Williams': '#64c4ff', 'Haas': '#b6babd', 'Kick Sauber': '#52e252', 'RB': '#6692ff'}
    for (idx, row) in data.iterrows():
        team = row['TeamName']
        color = '#888888'
        for (t_key, t_col) in team_colors.items():
            if isinstance(team, str) and t_key in team:
                color = t_col
                break
        ax.scatter(row['TopSpeed'], row['CorneringSpeed'], color=color, s=150, alpha=0.8, edgecolors='white')
        ax.text(row['TopSpeed'] + 0.5, row['CorneringSpeed'], row['Driver'], fontsize=9, color=color, fontweight='bold')
    avg_speed = data['TopSpeed'].mean()
    avg_corner = data['CorneringSpeed'].mean()
    ax.axvline(x=avg_speed, color='#8b949e', linestyle='--', alpha=0.5)
    ax.axhline(y=avg_corner, color='#8b949e', linestyle='--', alpha=0.5)
    ax.text(data['TopSpeed'].max(), data['CorneringSpeed'].max(), 'EFFICIENT / BALANCED\n(Fast Everywhere)', ha='right', va='top', color='#3fb950', fontsize=11, fontweight='bold')
    ax.text(data['TopSpeed'].min(), data['CorneringSpeed'].max(), 'HIGH DOWNFORCE / DRAGGY\n(Fast Corners, Slow Straights)', ha='left', va='top', color='#da3633', fontsize=11, fontweight='bold')
    ax.text(data['TopSpeed'].max(), data['CorneringSpeed'].min(), 'LOW DRAG / SLIPPERY\n(Fast Straights, Slow Corners)', ha='right', va='bottom', color='#f0883e', fontsize=11, fontweight='bold')
    ax.set_xlabel('Top Speed (98th %ile, km/h)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cornering Speed (15th %ile, km/h)', fontsize=12, fontweight='bold')
    ax.set_title(f'TELEMETRY SIGNATURE ({latest_year})\nCar Concept Analysis: Drag vs Downforce', fontsize=14, fontweight='bold', color='#58a6ff')
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='#0d1117')
    plt.close()
    print(f'Saved: {output_path}')

def plot_lap1_performance(lap1_df, output_path):
    plt.style.use('dark_background')
    if lap1_df.empty:
        print('No Lap 1 data found.')
        return
    df = lap1_df.head(20).copy()
    (fig, ax) = plt.subplots(figsize=(14, 8))
    colors = []
    for gain in df['AvgGain']:
        if gain > 1.0:
            colors.append('#238636')
        elif gain > 0:
            colors.append('#3fb950')
        elif gain > -1.0:
            colors.append('#f0883e')
        else:
            colors.append('#da3633')
    bars = ax.bar(df['Driver'], df['AvgGain'], color=colors, edgecolor='white')
    ax.axhline(y=0, color='#8b949e', linewidth=1)
    ax.set_ylabel('Avg Positions Gained (Lap 1)', fontsize=12, fontweight='bold')
    ax.set_title('THE LAUNCH KINGS: Lap 1 Performance\n(Avg Positions Gained/Lost Start to Lap 1)', fontsize=14, fontweight='bold', color='#58a6ff')
    for bar in bars:
        height = bar.get_height()
        label_y = height + 0.1 if height > 0 else height - 0.2
        ax.text(bar.get_x() + bar.get_width() / 2.0, label_y, f'{height:+.1f}', ha='center', va='center', fontsize=9, color='#c9d1d9')
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='#0d1117')
    plt.close()
    print(f'Saved: {output_path}')

def plot_overtake_3d(grid_df, output_path):
    plt.style.use('dark_background')
    if grid_df.empty:
        return
    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(111, projection='3d')
    pivot = grid_df.pivot(index='TyreDelta', columns='StraightLength', values='OvertakeProb')
    X = pivot.columns.values
    Y = pivot.index.values
    (X, Y) = np.meshgrid(X, Y)
    Z = pivot.values
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.9, antialiased=True)
    ax.contourf(X, Y, Z, zdir='z', offset=0, cmap='viridis', alpha=0.3)
    ax.set_xlabel('Straight Length (m)', fontsize=11, fontweight='bold', labelpad=10)
    ax.set_ylabel('Tyre Delta (s)', fontsize=11, fontweight='bold', labelpad=10)
    ax.set_zlabel('Overtake Probability', fontsize=11, fontweight='bold', labelpad=10)
    ax.set_zlim(0, 1.0)
    ax.set_title('OVERTAKE PROBABILITY SURFACE\nPhysics-Based Simulation (Sigmoid Prob Model)', fontsize=14, fontweight='bold', color='#58a6ff')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(color='#30363d', linestyle='--', alpha=0.3)
    ax.xaxis.pane.set_edgecolor('#30363d')
    ax.yaxis.pane.set_edgecolor('#30363d')
    ax.zaxis.pane.set_edgecolor('#30363d')
    cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, pad=0.1)
    cbar.set_label('Probability', fontsize=10, fontweight='bold')
    ax.view_init(elev=25, azim=135)
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='#0d1117')
    plt.close()
    print(f'Saved: {output_path}')

def plot_driver_dna(dna_df, output_path):
    plt.style.use('dark_background')
    
    if dna_df.empty:
        return
        
    drivers_to_plot = dna_df.sort_values('RacePace', ascending=False).head(6)['Driver'].tolist()
    
    categories = ['RacePace', 'QualiPace', 'Aggression', 'Consistency', 'BattleRating']
    N = len(categories)
    
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), subplot_kw=dict(polar=True))
    axes = axes.flatten()
    
    colors = ['#1e41ff', '#00d2be', '#dc0000', '#ff8700', '#229971', '#0090ff']
    
    for idx, driver in enumerate(drivers_to_plot):
        ax = axes[idx]
        row = dna_df[dna_df['Driver'] == driver].iloc[0]
        values = [row[cat] for cat in categories]
        values += values[:1]
        
        ax.plot(angles, values, linewidth=2, linestyle='solid', color=colors[idx])
        ax.fill(angles, values, color=colors[idx], alpha=0.4)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=9, fontweight='bold', color='#c9d1d9')
        
        ax.set_ylim(0, 100)
        ax.set_yticks([25, 50, 75, 100])
        ax.set_yticklabels([]) 
        
        ax.set_title(driver, size=14, color=colors[idx], weight='bold', y=1.1)
        
        ax.grid(color='#30363d', linestyle='--', alpha=0.5)
        ax.spines['polar'].set_visible(False)
        ax.set_facecolor('#0d1117')
        
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='#0d1117')
    plt.close()
    print(f"Saved: {output_path}")

def plot_dna_comparison(dna_df, driver1, driver2, output_path):
    plt.style.use('dark_background')
    
    if dna_df.empty:
        return
        
    d1_data = dna_df[dna_df['Driver'] == driver1]
    d2_data = dna_df[dna_df['Driver'] == driver2]
    
    if d1_data.empty or d2_data.empty:
        print(f"Drivers {driver1} or {driver2} not found for comparison.")
        return

    categories = ['RacePace', 'QualiPace', 'Aggression', 'Consistency', 'BattleRating']
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    # Driver 1
    v1 = d1_data.iloc[0][categories].tolist()
    v1 += v1[:1]
    ax.plot(angles, v1, linewidth=3, linestyle='solid', color='#1e41ff', label=driver1)
    ax.fill(angles, v1, color='#1e41ff', alpha=0.3)
    
    # Driver 2
    v2 = d2_data.iloc[0][categories].tolist()
    v2 += v2[:1]
    ax.plot(angles, v2, linewidth=3, linestyle='solid', color='#ff8700', label=driver2)
    ax.fill(angles, v2, color='#ff8700', alpha=0.3)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12, fontweight='bold', color='#c9d1d9')
    ax.set_ylim(0, 100)
    ax.set_yticks([25, 50, 75, 100])
    ax.set_yticklabels([])
    
    ax.set_title(f'DNA BATTLE: {driver1} vs {driver2}', size=16, color='#58a6ff', weight='bold', y=1.1)
    ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), fontsize=12, facecolor='#161b22', edgecolor='white')
    
    ax.grid(color='#30363d', linestyle='--', alpha=0.5)
    ax.spines['polar'].set_visible(False)
    ax.set_facecolor('#0d1117')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='#0d1117')
    plt.close()
    print(f"Saved: {output_path}")

def plot_dna_battle_royale(dna_df, output_path):
    plt.style.use('dark_background')
    
    top_3 = dna_df.sort_values('RacePace', ascending=False).head(3)['Driver'].tolist()
    if len(top_3) < 3: return
    
    categories = ['RacePace', 'QualiPace', 'Aggression', 'Consistency', 'BattleRating']
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(polar=True))
    
    colors = ['#1e41ff', '#ff8700', '#dc0000'] # Red Bull, McLaren, Ferrari usually
    
    for idx, driver in enumerate(top_3):
        row = dna_df[dna_df['Driver'] == driver].iloc[0]
        values = [row[cat] for cat in categories]
        values += values[:1]
        
        ax.plot(angles, values, linewidth=2, linestyle='solid', color=colors[idx], label=driver)
        ax.fill(angles, values, color=colors[idx], alpha=0.15)
        
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12, fontweight='bold', color='#c9d1d9')
    ax.set_ylim(0, 100)
    ax.set_yticks([50, 75, 100])
    ax.set_yticklabels([50, 75, 100], color='#8b949e', fontsize=10)
    
    ax.set_title(f'TITAN BATTLE: Top 3 Drivers Comparison', size=18, color='#58a6ff', weight='bold', y=1.1)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.1), fontsize=12, ncol=3, facecolor='#161b22', edgecolor='white')
    
    ax.grid(color='#30363d', linestyle='--', alpha=0.5)
    ax.spines['polar'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='#0d1117')
    plt.close()
    print(f"Saved: {output_path}")

def plot_sector_dominance(sector_df, output_path):
    plt.style.use('dark_background')
    
    if sector_df.empty:
        return
        
    drivers_to_plot = sector_df.groupby('Driver')['GapToUltimate'].mean().sort_values().head(12).index
    pivot = sector_df[sector_df['Driver'].isin(drivers_to_plot)].pivot(index='Driver', columns='Track', values='GapToUltimate')
    
    pivot.columns = [c.replace(' Grand Prix', '') for c in pivot.columns]
    
    fig, ax = plt.subplots(figsize=(20, 10))
    
    # Custom cmap: Green (Low Gap) -> Red (High Gap)
    cmap = sns.color_palette("RdYlGn_r", as_cmap=True)
    
    sns.heatmap(pivot, annot=True, fmt='.2f', cmap=cmap, ax=ax, 
                linewidths=.5, linecolor='#0d1117', cbar_kws={'label': 'Gap to Theoretical Limit (%)'})
    
    ax.set_title('SECTOR DOMINANCE: GAP TO THEORETICAL PERFECTION\n(Based on 1,000,000 Simulated Laps per Driver per Track)', 
                 fontsize=16, fontweight='bold', color='#58a6ff', pad=20)
    
    ax.set_ylabel('Driver', fontsize=12, fontweight='bold')
    ax.set_xlabel('Track', fontsize=12, fontweight='bold')
    
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='#0d1117')
    plt.close()
    print(f"Saved: {output_path}")