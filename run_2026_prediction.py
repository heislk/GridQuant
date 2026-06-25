"""
Run the 2026 World Championship forecast end to end.

Pulls the as-of 2026 season from OpenF1, analyses each team's upgrade/development
trajectory, simulates the rest of the season, and writes a forecast report.

    python run_2026_prediction.py
"""
import os
import argparse
import pandas as pd

from predict2026.openf1 import OpenF1Client
from predict2026 import ingest, upgrades, predictor, report
from predict2026.config import AS_OF, N_SIMS, CACHE_SUBDIR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=os.path.join("cache", CACHE_SUBDIR))
    ap.add_argument("--out", default="results/predictions")
    ap.add_argument("--sims", type=int, default=N_SIMS)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    charts_dir = os.path.join(args.out, "charts")
    os.makedirs(charts_dir, exist_ok=True)

    print("=" * 64)
    print("GridQuant · 2026 World Championship Forecast")
    print(f"As-of {AS_OF.date()} · {args.sims:,} simulations")
    print("=" * 64)

    print("\n[1/4] Pulling as-of 2026 data from OpenF1 ...")
    client = OpenF1Client(cache_dir=args.cache)
    ds = ingest.build_dataset(client)
    sch = ds["schedule"]
    n_races = int(((sch.kind == "race") & (~sch.done)).sum())
    n_sprints = int(((sch.kind == "sprint") & (~sch.done)).sum())
    n_done = int(((sch.kind == "race") & (sch.done)).sum())
    print(f"      {n_done} GPs completed · {n_races} GP + {n_sprints} sprints remaining")

    print("[2/4] Analysing upgrade / development trajectories ...")
    dev = upgrades.compute_team_development(ds["pace"])
    upg = upgrades.evaluate_upgrade_log(ds["pace"])

    print("[3/4] Simulating the rest of the season ...")
    prof = predictor.build_profiles(ds)
    wdc_res, wcc = predictor.simulate(prof, n_races, n_sprints, n_sims=args.sims)

    print("[4/4] Writing report + charts ...")
    charts = {
        "wdc": report.chart_wdc(wdc_res, os.path.join(charts_dir, "wdc_probability.png")),
        "trajectory": report.chart_trajectory(ds["pace"], dev, os.path.join(charts_dir, "development_trajectory.png")),
        "verdict": report.chart_upgrade_verdict(upg, os.path.join(charts_dir, "upgrade_verdicts.png")),
        "points": report.chart_points(wdc_res, os.path.join(charts_dir, "projected_points.png")),
    }
    html = report.build_html(ds, dev, upg, wdc_res, wcc, charts,
                             os.path.join(args.out, "championship_forecast_2026.html"))

    # machine-readable outputs
    prof.to_csv(os.path.join(args.out, "driver_profiles.csv"), index=False)
    dev.to_csv(os.path.join(args.out, "team_development.csv"), index=False)
    upg.to_csv(os.path.join(args.out, "upgrade_verdicts.csv"), index=False)
    wdc_res.to_csv(os.path.join(args.out, "wdc_forecast.csv"), index=False)
    wcc.to_csv(os.path.join(args.out, "wcc_forecast.csv"), index=False)

    print("\nDone. Report:", html)
    print("\nPredicted 2026 World Champion: "
          f"{wdc_res.iloc[0]['name'] or wdc_res.iloc[0]['driver']} "
          f"({wdc_res.iloc[0]['team']}) {wdc_res.iloc[0]['WDC_prob_%']:.1f}%")
    print("Chief rival: "
          f"{wdc_res.iloc[1]['name'] or wdc_res.iloc[1]['driver']} "
          f"({wdc_res.iloc[1]['team']}) {wdc_res.iloc[1]['WDC_prob_%']:.1f}%")


if __name__ == "__main__":
    main()
