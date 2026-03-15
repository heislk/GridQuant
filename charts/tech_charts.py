import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from build_world_model import WorldModel



def plot_tyre_model(model, output_path):
    plt.style.use('dark_background')
    
    # 1. robust track selection
    if not model.tyre_degradation:
        print("No tyre model data found.")
        return
        
    track = 'British Grand Prix'
    if track not in model.tyre_degradation:
        track = list(model.tyre_degradation.keys())[0]
        
    params = model.tyre_degradation[track]
    
    # Base lap time for context (e.g. 90s)
    base_lap = 90.0
    
    fig, ax = plt.subplots(figsize=(14, 9))
    
    compounds = {
        'SOFT': {'color': '#ff3b30', 'offset': 0.0, 'cliff': 20},
        'MEDIUM': {'color': '#ffcc00', 'offset': 0.6, 'cliff': 30},
        'HARD': {'color': '#e0e0e0', 'offset': 1.2, 'cliff': 45}
    }
    
    # We want to show CROSSOVERS.
    # Soft starts fast, degrades fast. Hard starts slow, degrades slow.
    laps = np.arange(1, 55)
    
    for comp_name, config in compounds.items():
        if comp_name not in params:
            continue
            
        p = params[comp_name]
        mu_deg = p['mu'] # Deg per lap
        
        # Calculate curve
        # Time = Base + Offset + (Deg * Age) + CliffPenalty
        y_values = []
        for lap in laps:
            deg_loss = mu_deg * lap
            
            # Cliff logic
            if lap > config['cliff']:
                deg_loss += 0.15 * (lap - config['cliff']) # Quadratic-ish penalty
                
            total_time = base_lap + config['offset'] + deg_loss
            y_values.append(total_time)
            
        ax.plot(laps, y_values, label=comp_name, color=config['color'], linewidth=3.5)
        
        # Shade error
        sigma = 0.1 # Assumed volatility
        lower = [y - sigma for y in y_values]
        upper = [y + sigma for y in y_values]
        ax.fill_between(laps, lower, upper, color=config['color'], alpha=0.1)
        
        # Mark the "Cliff"
        cliff_lap = config['cliff']
        cliff_time = y_values[cliff_lap-1]
        ax.scatter(cliff_lap, cliff_time, color=config['color'], s=100, marker='x', zorder=5)
        ax.text(cliff_lap, cliff_time - 0.2, f"{comp_name} Cliff", color=config['color'], fontsize=9, fontweight='bold', ha='center')

    ax.set_xlabel('Tyre Age (Laps)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Estimated Lap Time (s)', fontsize=12, fontweight='bold')
    ax.set_title(f'TYRE PERFORMANCE MODEL: CROSSOVER POINTS\n{track} Simulation Parameter', fontsize=14, fontweight='bold', color='#58a6ff')
    
    ax.grid(color='#30363d', linestyle='--', alpha=0.3)
    ax.legend(fontsize=10, facecolor='#161b22', edgecolor='#30363d', loc='upper left')
    
    # Add annotation for crossover
    ax.text(0.98, 0.02, "CROSSOVER = Intersection Points\n(When to Box)", transform=ax.transAxes, ha='right', va='bottom', color='#8b949e', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='#0d1117')
    plt.close()
    print(f'Saved: {output_path}')
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='#0d1117')
    plt.close()
    print(f'Saved: {output_path}')

def plot_strategy_analysis(analysis, output_path):
    plt.style.use('dark_background')
    
    results = analysis['results']
    strategies = analysis['strategies']
    
    # Create Mosaic Layout
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(2, 4)
    
    # 1. Violin Plot of Race Times (Main Feature) - Spans top row
    ax1 = fig.add_subplot(gs[0, :2])
    
    data_list = [results[s.name]['total_times'] for s in strategies]
    labels = [s.name.replace(' ', '\n') for s in strategies]
    colors = [s.color for s in strategies]
    
    parts = ax1.violinplot(data_list, showmeans=False, showmedians=False, showextrema=False)
    
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_edgecolor('white')
        pc.set_alpha(0.7)
        
        # Add internal boxplot-like stats
        mean_val = np.mean(data_list[i])
        ax1.scatter(i+1, mean_val, color='white', s=30, zorder=3)
        ax1.vlines(i+1, np.percentile(data_list[i], 5), np.percentile(data_list[i], 95), color='white', alpha=0.5, linewidth=1)
        
    ax1.set_xticks(np.arange(1, len(labels) + 1))
    ax1.set_xticklabels(labels, fontsize=10, fontweight='bold')
    ax1.set_title('RACE OUTCOME DISTRIBUTION (Violin Plot)\nWidth = Probability Density', fontsize=13, fontweight='bold', color='#58a6ff')
    ax1.set_ylabel('Total Race Time (s)', fontsize=11)
    ax1.grid(axis='y', color='#30363d', linestyle='--', alpha=0.3)

    # 2. Cumulative Probability (CDF) - Top Right
    ax2 = fig.add_subplot(gs[0, 2:])
    
    for s in strategies:
        sorted_data = np.sort(results[s.name]['total_times'])
        yvals = np.arange(len(sorted_data)) / float(len(sorted_data))
        ax2.plot(sorted_data, yvals, label=s.name, color=s.color, linewidth=2.5)
        
    ax2.set_xlabel('Total Race Time (s)', fontsize=11)
    ax2.set_ylabel('Cumulative Probability', fontsize=11)
    ax2.set_title('STRATEGY RELIABILITY (CDF)\nSteeper = More Consistent', fontsize=13, fontweight='bold', color='#58a6ff')
    ax2.legend(fontsize=9, facecolor='#161b22', edgecolor='#30363d')
    ax2.grid(color='#30363d', linestyle='--', alpha=0.3)

    # 3. Lap Time Evolution - Bottom Left
    ax3 = fig.add_subplot(gs[1, :3])
    
    laps = np.arange(1, len(results[strategies[0].name]['mean_lap_times']) + 1)
    
    for s in strategies:
        mean_laps = results[s.name]['mean_lap_times']
        # Smooth line
        ax3.plot(laps, mean_laps, label=s.name, color=s.color, linewidth=2)
        
        # Visualize Pit Stops (spikes)
        pit_laps = s.pit_laps
        for pl in pit_laps:
            ax3.axvline(x=pl, color=s.color, linestyle=':', alpha=0.5)

    ax3.set_xlabel('Lap Number', fontsize=11)
    ax3.set_ylabel('Lap Time (s)', fontsize=11)
    ax3.set_title('PACE EVOLUTION & PIT WINDOWS', fontsize=13, fontweight='bold', color='#58a6ff')
    ax3.grid(color='#30363d', linestyle='--', alpha=0.3)
    ax3.invert_yaxis() # Faster is lower

    # 4. Head-to-Head Win Rates - Bottom Right
    ax4 = fig.add_subplot(gs[1, 3])
    
    comparisons = analysis['comparisons']
    comp_keys = list(comparisons.keys())[:5] # Top 5 relevant comparisons
    
    y = np.arange(len(comp_keys))
    for idx, key in enumerate(comp_keys):
        wr = comparisons[key]['win_rate_a']
        name_a, name_b = key.split(' vs ')
        
        # Bidirectional bar
        # A wins (Green)
        ax4.barh(idx, wr, color='#238636', height=0.6)
        # B wins (Red from the other side)
        ax4.barh(idx, -(1-wr), color='#da3633', height=0.6)
        
        ax4.text(0.05, idx, f"{name_a}\n{wr:.1%}", va='center', ha='left', fontsize=8, color='white', fontweight='bold')
        ax4.text(-0.05, idx, f"{name_b}\n{(1-wr):.1%}", va='center', ha='right', fontsize=8, color='white', fontweight='bold')
        
    ax4.set_yticks([])
    ax4.set_xlim(-1, 1)
    ax4.axvline(0, color='white', linewidth=1)
    ax4.set_title('HEAD-TO-HEAD\nWIN PROBABILITY', fontsize=13, fontweight='bold', color='#58a6ff')
    ax4.axis('off') # Cleaner look

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='#0d1117')
    plt.close()
    print(f'Saved: {output_path}')