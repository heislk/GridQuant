"""Track each team's race-pace trend to see whether their upgrades are working."""
import numpy as np
import pandas as pd
from scipy import stats

from .config import UPGRADE_PROJECTION_WEIGHT


# --- Verified 2026 upgrade packages -----------------------------------------
# round index is on the post-cancellation 22-round calendar
# (1 Australia, 2 China, 3 Japan, 4 Miami, 5 Canada, 6 Monaco, 7 Barcelona...).
# Sourced from FIA car-presentation documents as reported in June 2026.
FIA_UPGRADE_LOG_2026 = [
    {"round": 4, "team": "Ferrari", "parts": 11,
     "package": "Grid-leading 11-part package (floor, bodywork); 'worked as expected'",
     "source": "Sky Sports / PlanetF1 (Miami 2026)"},
    {"round": 4, "team": "McLaren", "parts": 7,
     "package": "New floor (stage 1) - aero load & efficiency across all conditions",
     "source": "Sky Sports / The Race (Miami 2026)"},
    {"round": 4, "team": "Red Bull Racing", "parts": 7,
     "package": "'Definitive step forward' incl. rotating rear wing",
     "source": "Sky Sports (Miami 2026)"},
    {"round": 4, "team": "Mercedes", "parts": 2,
     "package": "Minor updates only - major package deliberately held for Canada",
     "source": "Sky Sports (Miami 2026)"},
    {"round": 5, "team": "Mercedes", "parts": None,
     "package": "First major aero package of 2026 (added downforce)",
     "source": "Motorsport.com (Canada 2026)"},
    {"round": 5, "team": "McLaren", "parts": None,
     "package": "Stage 2 of the two-part Miami/Canada upgrade",
     "source": "Motorsport.com (Canada 2026)"},
]


def _team_gap_series(pace_df):
    """team -> Series(round -> gap_to_best_s), team-level rows only."""
    team_rows = pace_df[pace_df["driver"].isna()].copy()
    out = {}
    for team, grp in team_rows.groupby("team"):
        s = grp.set_index("round")["gap_to_best_s"].sort_index()
        out[team] = s
    return out


def _robust_slope(rounds, gaps):
    """Theil-Sen slope (s per round); robust to one-off outlier weekends."""
    if len(rounds) < 3:
        return 0.0
    try:
        slope, _, _, _ = stats.theilslopes(gaps, rounds)
        return float(slope)
    except Exception:
        try:
            return float(np.polyfit(rounds, gaps, 1)[0])
        except Exception:
            return 0.0


def compute_team_development(pace_df, recent_n=3):
    """Per-team development summary + projected car gap for remaining races."""
    series = _team_gap_series(pace_df)
    rows = []
    for team, s in series.items():
        s = s.dropna()
        if s.empty:
            continue
        rounds = s.index.values.astype(float)
        # clip extreme one-off weekends (e.g. a wet/limited-data race) so the
        # trend stays honest without discarding the team
        gaps = np.clip(s.values, 0, np.nanpercentile(s.values, 90) + 1.0)
        slope = _robust_slope(rounds, gaps)
        gap_season = float(np.mean(s.values))
        gap_start = float(np.median(s.values[: min(3, len(s.values))]))
        # robust recent level: median of last N races (Monaco race-pace deltas
        # are notoriously distorted, so a median resists that single outlier)
        gap_recent = float(np.median(s.values[-recent_n:]))
        gap_delta = gap_recent - gap_start  # negative => improved
        # project the rest of the season: current level continued with a damped
        # slice of the established trend (clamped to non-negative)
        projected = max(0.0, gap_recent + slope * 2.0 * UPGRADE_PROJECTION_WEIGHT)

        if slope <= -0.08:
            trend = "Strongly improving"
        elif slope <= -0.03:
            trend = "Improving"
        elif slope < 0.03:
            trend = "Flat"
        elif slope < 0.08:
            trend = "Slipping"
        else:
            trend = "Regressing"

        rows.append({
            "team": team,
            "slope_s_per_round": round(slope, 4),
            "gap_season_s": round(gap_season, 3),
            "gap_start_s": round(gap_start, 3),
            "gap_recent_s": round(gap_recent, 3),
            "gap_delta_s": round(gap_delta, 3),
            "trend": trend,
            "projected_gap_s": round(projected, 3),
        })
    df = pd.DataFrame(rows).sort_values("projected_gap_s").reset_index(drop=True)
    return df


def evaluate_upgrade_log(pace_df, log=None):
    """Attach a data-driven 'did it help?' verdict to each known package."""
    if log is None:
        log = FIA_UPGRADE_LOG_2026
    series = _team_gap_series(pace_df)
    out = []
    for item in log:
        team, r = item["team"], item["round"]
        s = series.get(team)
        verdict, before, after, improvement = "Unknown", None, None, None
        if s is not None:
            s = s.dropna()
            before_vals = s[s.index < r].values[-2:]
            after_vals = s[s.index >= r].values[:3]
            if len(before_vals) and len(after_vals):
                before = float(np.mean(before_vals))
                after = float(np.mean(after_vals))
                improvement = before - after  # positive => gap closed => helped
                if improvement >= 0.15:
                    verdict = "Clear gain"
                elif improvement >= 0.05:
                    verdict = "Modest gain"
                elif improvement > -0.05:
                    verdict = "Neutral"
                else:
                    verdict = "No measurable gain"
        out.append({
            "round": r, "team": team, "parts": item["parts"],
            "package": item["package"],
            "gap_before_s": round(before, 3) if before is not None else None,
            "gap_after_s": round(after, 3) if after is not None else None,
            "improvement_s": round(improvement, 3) if improvement is not None else None,
            "verdict": verdict, "source": item["source"],
        })
    return pd.DataFrame(out)


def projected_car_gaps(dev_df):
    """team -> projected pace gap (s) to the best car for remaining races."""
    return dict(zip(dev_df["team"], dev_df["projected_gap_s"]))
