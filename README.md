# GridQuant

Python tools for digging into Formula 1 with stats instead of just the points table.

## 2026 title forecast

`predict2026/` forecasts the current championship. It pulls the season so far from
the OpenF1 API (up to a cutoff date), rates each driver from qualifying pace, race
pace and points per race, then plays out the rest of the season a few thousand times
to get win probabilities. It also models reliability and tracks whether each team's
upgrades are actually making the car faster.

As of round 7 it has Antonelli around 82% for the drivers' title, Hamilton around 16%,
everyone else a long shot. The number is wide though; `predict2026/MODEL_LIMITATIONS.md`
explains why and where it can be wrong.

### Run it

    python run_2026_prediction.py
    python run_2026_prediction.py --cache predict2026/cache_snapshot/openf1_2026   # offline, cached pull

The report, charts and CSVs land in `results/predictions/`.

To roll the forecast to a new round, change `AS_OF` (and the season) in
`predict2026/config.py`.

### Visuals

The video and the interactive explorer are built from the same simulation. They are
generated, not committed (see `.gitignore`), and land in `media/`:

    python predict2026/viz/export_data.py     # cache the simulation data
    python predict2026/viz/interactive.py      # interactive HTML explorer
    python predict2026/viz/chart.py            # static overview image
    python predict2026/viz/render.py           # the MP4 (needs ffmpeg)

### Layout

    predict2026/
      config.py          settings and tuning
      openf1.py          cached API client
      ingest.py          pull the season as of the cutoff
      predictor.py       build driver profiles, run the Monte Carlo
      reliability.py     per-car DNF model
      upgrades.py        team pace trend / upgrade tracker
      report.py          charts and the HTML report
      viz/               video + interactive build scripts
      MODEL_LIMITATIONS.md
    run_2026_prediction.py
    media/               generated video / explorer (gitignored)

The older `run_pipeline.py` and `metrics/` are the original historical-analysis tools
that run on local FastF1 parquet data.

---

OpenF1 is an unofficial project and is not associated with Formula 1. F1, FORMULA 1
and related marks are trade marks of Formula One Licensing B.V.
