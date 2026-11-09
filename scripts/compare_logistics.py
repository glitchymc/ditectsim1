"""Compare the flowchart's truck-dispatching options on identical demand.

Every scenario runs on the same order streams, rider population and initial
battery states for each seed (common random numbers), so differences come
from the logistics policy, not from luck in the demand.

  1   hub trucks CCS -> BSS -> CCS, called when depleted packs reach a threshold
  2a  dedicated trucks BSS -> CCS -> BSS, Poisson departures
  2b  dedicated trucks, scheduled departures
  2c  ... each returning immediately or waiting for enough charged packs
"""
from __future__ import annotations

import argparse
import copy
import sys
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_integrated import add_args, config_from_args, station_sites  # noqa: E402
from common import city_from_args  # noqa: E402

from ebike_swap.integrated import IntegratedSim, summary  # noqa: E402

SCENARIOS = {
    "1 hub, threshold": dict(mode="hub"),
    "2a dedicated, Poisson, immediate": dict(mode="dedicated", departure="poisson", return_policy="immediate"),
    "2a dedicated, Poisson, wait": dict(mode="dedicated", departure="poisson", return_policy="wait"),
    "2b dedicated, schedule, immediate": dict(mode="dedicated", departure="schedule", return_policy="immediate"),
    "2b dedicated, schedule, wait": dict(mode="dedicated", departure="schedule", return_policy="wait"),
}
KPIS = ["mean_swap_wait_min", "p95_swap_wait_min", "mean_swap_trip_min", "bss_stockout_frac",
        "cancel_rate", "mean_delivery_min", "truck_trips", "truck_km", "truck_utilization",
        "packs_charged_at_ccs", "stranded_riders"]
_W: dict = {}


def _init(city, base_cfg, sites, jobs):
    _W.update(city=city, cfg=base_cfg, sites=sites, jobs=jobs)


def _run(name: str, seed: int):
    cfg = copy.deepcopy(_W["cfg"])
    for k, v in SCENARIOS[name].items():
        setattr(cfg.trucks, k, v)
    sim = IntegratedSim(_W["city"], cfg, _W["sites"], seed=seed,
                        jobs_by_platform=copy.deepcopy(_W["jobs"][seed])).run()
    return name, seed, summary(sim)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_args(p)
    p.add_argument("--seeds", type=int, default=4)
    p.add_argument("--out", default="out/logistics")
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    city = city_from_args(a)
    cfg = config_from_args(a)
    sites = station_sites(city, a.stations)
    seeds = list(range(1, a.seeds + 1))
    probe = IntegratedSim(city, cfg, sites, seed=0, jobs_by_platform=[[] for _ in cfg.platforms])
    jobs = {s: IntegratedSim.generate_jobs(city, cfg, probe.router, s) for s in seeds}
    tasks = [(n, s) for n in SCENARIOS for s in seeds]
    with ProcessPoolExecutor(mp_context=get_context("fork"), initializer=_init,
                             initargs=(city, cfg, sites, jobs)) as ex:
        rows = [{"scenario": n, "seed": s, **r} for n, s, r in ex.map(_run, *zip(*tasks))]
    df = pd.DataFrame(rows)
    df.to_csv(out / "per_seed.csv", index=False)
    mean = df.groupby("scenario", sort=False)[KPIS].mean()
    # paired difference in mean swap wait vs option 1 (same demand per seed)
    base = df[df["scenario"] == "1 hub, threshold"].set_index("seed")["mean_swap_wait_min"]
    diffs = {}
    for n in SCENARIOS:
        d = df[df["scenario"] == n].set_index("seed")["mean_swap_wait_min"] - base
        diffs[n] = f"{d.mean():+.1f} ± {d.std(ddof=1) / np.sqrt(len(d)):.1f}" if len(d) > 1 else f"{d.mean():+.1f}"
    mean["swap_wait_vs_1"] = pd.Series(diffs)
    pd.set_option("display.width", 250)
    print(mean.round(3).to_string())
    mean.to_csv(out / "comparison.csv")
    print("wrote", out)


if __name__ == "__main__":
    main()
