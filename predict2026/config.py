"""Settings for the 2026 championship model."""
from datetime import datetime, timezone

# Which season we're predicting, and the cutoff. Anything after AS_OF is
# treated as not-yet-run and gets simulated.
BASELINE_SEASON = 2025
LIVE_SEASON = 2026
AS_OF = datetime(2026, 6, 23, tzinfo=timezone.utc)

# Current F1 points (2025+, no fastest-lap point).
RACE_POINTS = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}
SPRINT_POINTS = {1: 8, 2: 7, 3: 6, 4: 5, 5: 4, 6: 3, 7: 2, 8: 1}

N_SIMS = 30000
DEFAULT_DNF_RATE = 0.06
DNF_PRIOR_WEIGHT = 5.0

# Driver strength = qualifying pace + race pace + points per race.
STRENGTH_QUALI = 0.50
STRENGTH_RACE = 0.30
STRENGTH_RESULTS = 0.20
RESULTS_PPR_TO_S = 0.05          # points-per-race deficit -> seconds
QUALI_WEIGHT = 0.65              # only used as a fallback when results are missing
RACE_WEIGHT = 0.35
SEASON_WEIGHT = 0.50
RECENT_WEIGHT = 0.50
RECENT_ROUNDS = 3

# Race model knobs (tuned on the completed 2026 races).
PACE_TO_UTILITY = 2.1
SIGMA_RACE = 1.0
SIGMA_QUALI_S = 0.12
GRID_STICKINESS = 0.12           # how much starting position matters
SIGMA_SPRINT_EXTRA = 0.15
TEAM_RACE_SIGMA = 0.30           # team-mates share a weekend
DEV_STEP_S = 0.07                # per-team pace drift across the rest of the season

UPGRADE_PROJECTION_WEIGHT = 0.5
CACHE_SUBDIR = "openf1_2026"
