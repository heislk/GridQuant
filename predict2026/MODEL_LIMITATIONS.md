# Where this model could be wrong

A rundown of the 2026 championship predictor's assumptions, fragilities,
and blind spots. The point is to know how much to trust the headline (currently
Antonelli 82%, Hamilton 16%), and where it could be flat wrong.

Everything below is measured, not asserted (`sensitivity.py`, `bootstrap.py`).

## Headline finding: the uncertainty is largely irreducible right now

We rebuilt the single most important input, reliability, from scratch (see §2),
about as good as the data allows. It barely changed the conclusion, and
**it did not shrink the uncertainty**. That is the key lesson: with only 7 races run,
the forecast is intrinsically wide, and who wins genuinely hinges on reliability we
cannot pin down yet. More modelling does not fix this, only more races will.

Resampling the 7 completed races (block bootstrap) puts Antonelli's true title
probability across roughly a **60-98% band** (median 85%). Read the headline as
"clear favourite," not as a precise number.

## 1. What the answer rides on (and what it doesn't)

One-at-a-time sweep of every assumption, by how far it moves Antonelli's probability:

| Assumption swept | Antonelli % range | Swing |
|---|---|---|
| **Reliability** (Antonelli DNF x0.5 ... x2) | **55 - 91** | **36** |
| Development uncertainty (how open the season is) | 67 - 82 | 15 |
| Recency weighting of pace | 69 - 83 | 14 |
| Strength mix (quali / race / results) | 76 - 79 | 3 |
| Race-day noise, pace to result scale, grid stickiness | <77 - 78 | 1-2 |

The structural machinery (pace to result mapping, race noise, grid model, the
two-stage qualifying to race) is **stable**. The forecast rides on **reliability** and
**how open we think the season is**, and reliability dominates.

## 2. Reliability, now properly modelled, still the biggest lever

Originally each driver's retirement risk came from 1 DNF in 7 races. It now uses
(`reliability.py`):

- **Team-level pooling** of 2026 retirements (both cars to 2x the sample), since
  mechanical failure is mostly a car/power-unit property.
- A **2025 historical prior** mapped across the 2026 entry changes (Sauber to Audi, etc.),
  blended with the current grid mean and weighted modestly (2026 is a regulation reset,
  so last year's reliability is only a weak guide).
- A small **driver adjustment** for personally crash-prone drivers.
- **Correlated team-mate failures** in the simulation (Gaussian copula): a Mercedes
  "bad day" can take out both cars together, instead of treating retirements as
  independent.

This made Antonelli's estimate *more reliable* (Mercedes ran 5% DNFs in 2025 and 14%
in 2026), which nudged him **up** to 82%, not down. It is the better estimate, but
because the *output* is so sensitive to reliability, and we still have few retirements
to learn from, the headline uncertainty did not shrink. In short: this input
is as good as the data allows, and it is still the thing most likely to be wrong.

It also still cannot see *why* a car retires (crash vs. engine vs. lapped), OpenF1
does not label cause, so the mechanical/driver split is an assumption, not data.

## 3. Seven races is thin, the decimal is false precision

Bootstrap (resample the 7 completed races, rerun the whole pipeline):
**Antonelli median 85%, 10th-90th percentile about  [60% - 98%]**, Hamilton roughly
[2% - 28%]. Two scheduled early rounds (Bahrain, Saudi) were cancelled, so we have
*fewer* data points than a normal third-of-season.

## 4. Structural blind spots, invisible at any setting

- **Track-type fit.** Every remaining race is generic; the back-half mixes street
  circuits (Singapore, Vegas) and power tracks that suit different cars.
- **Weather.** No wet-race modelling, pure unmodelled variance.
- **Driver intangibles.** No "rookie under title pressure" term for Antonelli; and it
  under-credits elite outliers, Verstappen sits near 0% because it reads the car, not
  the driver who has dragged midfield cars to wins.
- **Team orders & politics** (Mercedes backing Antonelli; Ferrari's Hamilton/Leclerc
  balance), not modelled.
- **Discrete upgrades.** Development is a smooth random walk; a single transformative
  (or failed) upgrade is exactly what it cannot anticipate, yet that is how Ferrari
  closed the gap this year.
- **Points double-use.** Current points set the start line *and* feed the strength
  signal (points-per-race), so the leader's standing is mildly counted twice.

## 5. Data-quality caveats

- Race "pace" is a noisy median-lap proxy that disagrees with results for race-managers
  (Russell). Blending in points-per-race only partly fixes it.
- OpenF1 is unofficial; a few 2026 sessions returned placeholder data; DNF cause is
  unlabelled.
- No tyre, stint, pit-stop, or safety-car modelling.

## 6. Bottom line, what to trust

**Trust the ranking, not the decimal.** Robust across every test: Antonelli is a clear
favourite (5 wins from 7, fastest qualifier, reliable team), Hamilton is the one
realistic challenger (16%, matching betting markets), and the rest are long shots in
measurably slower cars.

**Distrust the exact number.** 82% is one defensible point in a 60-98% band, and it
leans most on reliability, which is now estimated as well as 7 races permit, and is
still the likeliest thing to be wrong. The gap to the betting market (60%) is a
philosophy choice (trust demonstrated dominance vs. hedge for a long season), not a
bug. The only real cure for the uncertainty is more races.
