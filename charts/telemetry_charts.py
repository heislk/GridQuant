import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from scipy.interpolate import interp1d

# =============================================================================
# DESIGN SYSTEM (Shared)
# =============================================================================
STYLE = {
    'bg_color': '#101014',       
    'text_primary': '#FFFFFF',   
    'text_secondary': '#8b949e', 
    'accent_line': '#30363d',    
    'font': 'sans-serif'         
}

TEAM_COLORS = {
    'Red Bull': '#1eff00', # Neon Green/Blue
    'Red Bull Racing': '#3671c6',
    'Mercedes': '#00d2be',
    'Ferrari': '#dc0000',
    'McLaren': '#ff8700',
    'Aston Martin': '#229971',
    'Alpine': '#0090ff',
    'Williams': '#64c4ff', 
    'Haas': '#b6babd',
    'Haas F1 Team': '#b6babd',
    'Kick Sauber': '#52e252',
    'RB': '#6692ff',
    'AlphaTauri': '#6692ff'
}

def get_color(team_name):
    return TEAM_COLORS.get(team_name, '#ffffff')

def plot_ghost_car_comparison(telemetry_df: pd.DataFrame, driver1: str, driver2: str, track_name: str, output_path: str):
    """
    Plots a Ghost Car comparison (Speed vs Distance) for two drivers.
    Interpolates data to distance to allow direct overlay.
    """
    plt.style.use('dark_background')
    
    # 1. Filter Data
    t1 = telemetry_df[telemetry_df['Driver'] == driver1].copy()
    t2 = telemetry_df[telemetry_df['Driver'] == driver2].copy()
    
    if t1.empty or t2.empty:
        print(f"  Warning: No telemetry for {driver1} or {driver2}. Skipping chart.")
        return

    # 2. Add Distance (Integration)
    # Speed is km/h. Distance = Speed * Time. 
    # Time is usually an index or Date. We need delta seconds.
    # Telemetry usually comes at a fixed rate (e.g. 20Hz implied, or we use index).
    # Ideally telemetry_df has 'Time' or 'Date'.
    # If standard FastF1 export, 'Date' is timestamp.
    
    # Sort by time
    t1 = t1.sort_values('Date')
    t2 = t2.sort_values('Date')
    
    # Calculate Distances
    for df in [t1, t2]:
        # Calculate time delta in hours
        df['TimeSec'] = (df['Date'] - df['Date'].iloc[0]).dt.total_seconds()
        df['dt'] = df['TimeSec'].diff().fillna(0)
        # Distance (km) = Speed (km/h) * Time (h)
        # Distance (m) = Speed / 3.6 * Time (s)
        df['ds'] = (df['Speed'] / 3.6) * df['dt']
        df['Distance'] = df['ds'].cumsum()
        
    # 3. Interpolate to common distance grid for comparison
    # Use the shorter lap as reference limit
    max_dist = min(t1['Distance'].max(), t2['Distance'].max())
    common_dist = np.linspace(0, max_dist, 2000)
    
    # Interpolators
    f_speed1 = interp1d(t1['Distance'], t1['Speed'], kind='linear', fill_value='extrapolate')
    f_speed2 = interp1d(t2['Distance'], t2['Speed'], kind='linear', fill_value='extrapolate')
    
    speed1_interp = f_speed1(common_dist)
    speed2_interp = f_speed2(common_dist)
    
    # Delta (Driver 1 - Driver 2)
    # Speed Delta is interesting, Time Delta involves re-integrating.
    # Let's stick to Speed Overlay + Throttle Subplot
    
    # Get Teams (assuming passed somewhere or inferred, taking first row)
    # We don't have Team in telemetry_df usually, we need to pass it or color manually.
    # We will guess colors based on Driver for now (hardcoded map or generic).
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), height_ratios=[3, 1], facecolor=STYLE['bg_color'])
    ax1.set_facecolor(STYLE['bg_color'])
    ax2.set_facecolor(STYLE['bg_color'])
    
    # --- Main Speed Trace ---
    l1, = ax1.plot(common_dist, speed1_interp, color='#00d2be', linewidth=2.5, label=driver1) # Mercerdes/Petronas Cyan generic
    l2, = ax1.plot(common_dist, speed2_interp, color='#ff8700', linewidth=2.5, label=driver2) # McLaren Orange generic
    
    # Fill gaps
    ax1.fill_between(common_dist, speed1_interp, speed2_interp, 
                     where=(speed1_interp > speed2_interp), interpolate=True, color='#00d2be', alpha=0.1)
    ax1.fill_between(common_dist, speed1_interp, speed2_interp, 
                     where=(speed2_interp > speed1_interp), interpolate=True, color='#ff8700', alpha=0.1)

    ax1.set_ylabel('Speed (km/h)', fontsize=12, fontweight='bold')
    ax1.set_title(f'GHOST CAR COMPARISON • {driver1} vs {driver2}\n{track_name} • Fastest Lap Telemetry', 
                  fontsize=18, fontweight='heavy', color=STYLE['text_primary'], pad=20)
    
    ax1.legend(facecolor=STYLE['bg_color'], edgecolor='white', fontsize=12)
    ax1.grid(color=STYLE['accent_line'], alpha=0.3, linestyle='--')
    
    # --- Throttle Trace (or Delta) ---
    # Interpolate throttle
    if 'Throttle' in t1.columns:
        f_thr1 = interp1d(t1['Distance'], t1['Throttle'], kind='linear', fill_value='extrapolate')
        f_thr2 = interp1d(t2['Distance'], t2['Throttle'], kind='linear', fill_value='extrapolate')
        
        thr1 = f_thr1(common_dist)
        thr2 = f_thr2(common_dist)
        
        ax2.plot(common_dist, thr1, color='#00d2be', linewidth=1.5, alpha=0.8)
        ax2.plot(common_dist, thr2, color='#ff8700', linewidth=1.5, alpha=0.8)
        ax2.set_ylabel('Throttle %', fontsize=12, fontweight='bold')
        ax2.set_ylim(-5, 105)
    
    ax2.set_xlabel('Lap Distance (m)', fontsize=12, fontweight='bold')
    ax2.grid(color=STYLE['accent_line'], alpha=0.3, linestyle='--')
    
    # Spines
    for ax in [ax1, ax2]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color(STYLE['accent_line'])
        ax.spines['left'].set_color(STYLE['accent_line'])
        ax.tick_params(colors=STYLE['text_secondary'])
        ax.yaxis.label.set_color(STYLE['text_secondary'])
        ax.xaxis.label.set_color(STYLE['text_secondary'])

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor=STYLE['bg_color'])
    plt.close()
    print(f"Saved: {output_path}")
