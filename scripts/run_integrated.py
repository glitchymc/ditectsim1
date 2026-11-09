"""Run one day of the integrated frontend + backend simulator and write tables and figures.

Examples
  python scripts/run_integrated.py                                  # option 1: hub trucks on threshold
  python scripts/run_integrated.py --truck-mode dedicated --departure schedule --return-policy wait
  python scripts/run_integrated.py --charger fast --gif
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import add_city_args, city_from_args  # noqa: E402

from ebike_swap.config import DemandParams  # noqa: E402
from ebike_swap.demand import ODGenerator  # noqa: E402
from ebike_swap.config import EnergyParams, TravelParams  # noqa: E402
from ebike_swap.geo import GridRouter  # noqa: E402
from ebike_swap.integrated import (IntegratedConfig, IntegratedSim, bss_table, platform_table,  # noqa: E402
                                   rider_table, summary, truck_table)
from ebike_swap.integrated import viz  # noqa: E402
from ebike_swap.optimize import candidate_sites, demand_design  # noqa: E402


def add_args(p: argparse.ArgumentParser):
    add_city_args(p)
    p.add_argument("--orders-per-day", type=float, default=8000, help="total over all platforms")
    p.add_argument("--riders", type=int, default=400)
    p.add_argument("--stations", type=int, default=10)
    p.add_argument("--slots", type=int, default=22)
    p.add_argument("--truck-mode", choices=["hub", "dedicated"], default="hub")
    p.add_argument("--departure", choices=["poisson", "schedule"], default="schedule")
    p.add_argument("--return-policy", choices=["immediate", "wait"], default="immediate")
    p.add_argument("--trucks", type=int, default=6, help="hub mode fleet size")
    p.add_argument("--truck-capacity", type=int, default=30)
    p.add_argument("--threshold", type=int, default=8)
    p.add_argument("--charger", choices=["standard", "fast"], default="standard")
    p.add_argument("--ccs-spares", type=int, default=250)
    p.add_argument("--seed", type=int, default=1)


def config_from_args(a) -> IntegratedConfig:
    from ebike_swap.integrated.config import default_platforms
    cfg = IntegratedConfig(platforms=default_platforms(a.orders_per_day))
    cfg.riders.n_riders = a.riders
    cfg.bss.n_stations, cfg.bss.slots = a.stations, a.slots
    cfg.bss.initial_charged = min(cfg.bss.initial_charged, a.slots)
    t = cfg.trucks
    t.mode, t.departure, t.return_policy = a.truck_mode, a.departure, a.return_policy
    t.n_trucks, t.capacity, t.threshold = a.trucks, a.truck_capacity, a.threshold
    cfg.ccs.charger.kind = a.charger
    cfg.ccs.initial_charged = a.ccs_spares
    return cfg


def station_sites(city, n: int, seed: int = 0) -> np.ndarray:
    """Place swap stations by demand k-means (the best simple layout in the earlier study)."""
    router = GridRouter(city.terrain, EnergyParams(), TravelParams())
    orders = ODGenerator(city, DemandParams(orders_per_day=3000), router).generate(seed + 500)
    cand = candidate_sites(city)
    return cand[demand_design(city, cand, n, n * 2, orders, seed).sites]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_args(p)
    p.add_argument("--gif", action="store_true")
    p.add_argument("--out", default="out/integrated")
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    city = city_from_args(a)
    cfg = config_from_args(a)
    cfg.snapshots = True
    sites = station_sites(city, a.stations)
    t = time.time()
    sim = IntegratedSim(city, cfg, sites, seed=a.seed).run()
    print(f"simulated {sum(len(j) for j in sim.jobs)} orders, {sim.k.n_events} events in {time.time() - t:.1f}s")
    s = summary(sim)
    for k, v in s.items():
        print(f"  {k:26s} {v:.3f}" if isinstance(v, float) else f"  {k:26s} {v}")
    pd.set_option("display.width", 200)
    for name, tbl in [("platforms", platform_table(sim)), ("stations", bss_table(sim)),
                      ("trucks", truck_table(sim)), ("riders", rider_table(sim))]:
        tbl.to_csv(out / f"{name}.csv", index=False)
        if name != "riders":
            print(f"\n{name}\n" + tbl.round(2).to_string(index=False))
    pd.DataFrame(sim.samples).fillna(0).to_csv(out / "inventory.csv", index=False)
    (out / "summary.json").write_text(json.dumps(s, indent=2))
    viz.inventory_figure(sim, str(out / "inventory.png"))
    peak = int(np.argmax([x["bss_queue"].sum() + 0.01 * x["bss_depleted"].sum() for x in sim.snaps]))
    viz.snapshot_figure(sim, peak, str(out / "snapshot_peak.png"))
    if a.gif:
        viz.animate(sim, str(out / "day.gif"), every=2)
    print("wrote", out)


if __name__ == "__main__":
    main()
