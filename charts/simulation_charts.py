import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.patheffects as path_effects
from typing import Dict

# =============================================================================
# DESIGN SYSTEM
# =============================================================================
STYLE = {
    'bg_color': '#101014',       # Very dark, slightly blue-ish charcoal
    'text_primary': '#FFFFFF',   # Pure white
    'text_secondary': '#8b949e', # Muted grey
    'accent_line': '#30363d',    # Subtle grid lines
    'font': 'sans-serif'         # Fallback to system sans (clean)
}

TEAM_COLORS = {
    'Red Bull': '#1e41ff',
    'Mercedes': '#00d2be',
    'Ferrari': '#dc0000',
    'McLaren': '#ff8700',
    'Aston Martin': '#229971',
    'Alpine': '#0090ff',
    'Williams': '#64c4ff', 
    'Haas': '#b6babd',
    'Kick Sauber': '#52e252',
    'RB': '#6692ff',
    'Unknown': '#444444'
}

def setup_plot_style():
    plt.rcParams['font.family'] = STYLE['font']
    plt.rcParams['font.weight'] = 'medium'
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.titleweight'] = 'bold'

def draw_header(ax, title, subtitle):
    # Main Title
    ax.text(0, 1.08, title.upper(), transform=ax.transAxes, 
            fontsize=24, fontweight='heavy', color=STYLE['text_primary'], va='bottom')
    # Subtitle
    ax.text(0, 1.02, subtitle, transform=ax.transAxes, 
            fontsize=12, fontweight='medium', color=STYLE['text_secondary'], va='bottom')
            
    # Decorative line
    line = plt.Line2D([0, 0.1], [1.14, 1.14], transform=ax.transAxes, 
                      color='#ff0000', linewidth=3)
    ax.add_line(line)

def plot_championship_probabilities(sim_results: pd.DataFrame, output_path: str):
    """
    Plots the Championship Win Probabilities.
    Style: Horizontal Bar for easy reading, magazine style.
    """
    setup_plot_style()
    
    if sim_results.empty: return
        
    # Data Prep
    df = sim_results.sort_values('WDC_Probability', ascending=True).tail(8) # Top 8 only for impact
    
    fig, ax = plt.subplots(figsize=(16, 9), facecolor=STYLE['bg_color'])
    ax.set_facecolor(STYLE['bg_color'])
    
    # Header
    draw_header(ax, "World Championship", "Projected Winner Probability • Monte Carlo Season Simulation")
    
    # Bars
    y_range = np.arange(len(df))
    colors = [TEAM_COLORS.get(t, '#888888') for t in df['Team']]
    
    bars = ax.barh(y_range, df['WDC_Probability'], height=0.6, color=colors, edgecolor=STYLE['bg_color'], linewidth=0)
    
    # Effects: Glow / Shadow underneath? 
    # Simplified approach: Clean sharp bars against dark background is very modern.
    
    # Labels directly on/next to bars
    for i, (bar, driver, prob) in enumerate(zip(bars, df['Driver'], df['WDC_Probability'])):
        width = bar.get_width()
        
        # Driver Name (Left of start, or inside if wide enough?)
        # Let's put Driver Name Left aligned, hovering above bar or to the left
        ax.text(0, i + 0.45, driver.upper(), fontsize=14, fontweight='bold', color=STYLE['text_primary'], va='center')
        
        # Percentage Value (At the end of the bar)
        text_color = colors[i] # Match bar color for text
        # If bar is very small, place next to it.
        label_x = width + 1.5
        ax.text(label_x, i, f"{prob:.1f}%", fontsize=16, fontweight='bold', color=text_color, va='center')

    # Remove axes
    ax.set_yticks([])
    ax.set_xticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # Add subtle grid vertical? No, simpler is better for "Designer" look.
    
    # Footer
    ax.text(1, -0.05, "Source: F1 Sabermetrics • 10,000 Simulations", transform=ax.transAxes,
            ha='right', fontsize=9, color=STYLE['text_secondary'])

    plt.tight_layout()
    # Add padding for header since we drew outside axes
    plt.subplots_adjust(top=0.85, bottom=0.1)
    
    plt.savefig(output_path, dpi=250, bbox_inches='tight', facecolor=STYLE['bg_color'])
    plt.close()
    print(f"Saved: {output_path}")

def plot_qualifying_probabilities(sim_results: pd.DataFrame, output_path: str):
    """
    Plots Qualifying: Focused on POLE POSITION.
    Style: Minimalist, huge typography for the top stars.
    """
    setup_plot_style()
    
    if sim_results.empty: return
        
    # Sorting by Pole Prob
    df = sim_results.sort_values('PoleProb', ascending=False)
    # Filter: Top 5 + "Others" if needed, or just top 5-7 distinct
    df = df.head(7).iloc[::-1] # Reverse for Horizontal bars (Top at top)
    
    fig, ax = plt.subplots(figsize=(10, 10), facecolor=STYLE['bg_color'])
    ax.set_facecolor(STYLE['bg_color'])
    
    draw_header(ax, "Pole Position", "Probability of starting P1 • British GP Simulation")
    
    # Y Positions
    y_pos = np.arange(len(df))
    
    # We will use "Lollipops" or just thinner bars with big text?
    # Let's go with "Process Bar" style
    
    # Background rail
    ax.barh(y_pos, [100]*len(df), height=0.04, color=STYLE['accent_line'], align='center')
    
    # Foreground Data
    for i, (driver, prob) in enumerate(zip(df['Driver'], df['PoleProb'])):
        # Get team color
        # Need to find team from somewhere, assuming sim_results doesn't have it? 
        # Actually championship.py output had team, qualifying output might not.
        # Let's quick-lookup or default.
        # Ideally we pass team in sim_results. 
        # Assuming we can infer or it's not strictly critical to be perfect here, use a generic lookup or match Championship.
        # For robustness, default color if not found.
        c = '#ffffff'
        # Heuristic color mapping from known drivers if needed, 
        # or maybe we update sim output to include team. 
        # For now, let's use white for clean look or basic mapping.
        if 'Verstappen' in driver: c = TEAM_COLORS['Red Bull']
        elif 'Norris' in driver: c = TEAM_COLORS['McLaren']
        elif 'Leclerc' in driver: c = TEAM_COLORS['Ferrari']
        elif 'Russel' in driver or 'Hamilton' in driver: c = TEAM_COLORS['Mercedes']
        elif 'Alonso' in driver: c = TEAM_COLORS['Aston Martin']
        
        # Circle Marker at end
        ax.scatter(prob, i, s=500, color=c, zorder=3, edgecolors=STYLE['bg_color'], linewidth=2)
        
        # Value inside circle if fits? Or next to it.
        # Let's put Value next to it.
        ax.text(prob + 3.0, i, f"{prob:.1f}%", va='center', fontsize=14, fontweight='bold', color=c)
        
        # Name Label above the line
        ax.text(0, i + 0.25, driver.upper(), va='bottom', fontsize=12, fontweight='bold', color=STYLE['text_primary'])
        
        # Progress Bar active part
        ax.barh(i, prob, height=0.08, color=c, align='center', zorder=2)

    # Clean up
    ax.set_xlim(0, 115) # Leave space for text
    ax.set_ylim(-0.5, len(df)-0.5)
    ax.axis('off')
    
    # Footer
    ax.text(1, 0.02, "Based on 1,000 Qualifying Sessions", transform=ax.transAxes,
            ha='right', fontsize=8, color=STYLE['text_secondary'])
            
    plt.tight_layout()
    plt.subplots_adjust(top=0.85, bottom=0.05)
    
    plt.savefig(output_path, dpi=250, bbox_inches='tight', facecolor=STYLE['bg_color'])
    plt.close()
    print(f"Saved: {output_path}")
