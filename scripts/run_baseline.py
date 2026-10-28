"""Run one simulated day for a given station layout and write tables + figures."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import add_city_args, city_from_args  # noqa: E402

from ebike_swap import (ODGenerator, SimConfig, Simulation, StationLayout,  # noqa: E402
                        station_table, summarize)
from ebike_swap.dispatch import LeastExpectedTime, NearestStation  # noqa: E402
from ebike_swap.geo import GridRouter  # noqa: E402
from ebike_swap.optimize import candidate_sites, uniform_design  # noqa: E402
from ebike_swap import viz  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    add_city_args(p)
    p.add_argument("--stations", type=int, default=15)
    p.add_argument("--batteries", type=int, default=150, help="total battery budget")
    p.add_argument("--riders", type=int, default=400)
    p.add_argument("--orders-per-day", type=float, default=8000)
    p.add_argument("--policy", choices=["nearest", "least_time"], default="nearest")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--gif", action="store_true", help="also write an animation")
    p.add_argument("--out", default="out/baseline")
    a = p.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    city = city_from_args(a)
    cfg = SimConfig(snapshot_every_min=10)
    cfg.rider.n_riders = a.riders
    cfg.demand.orders_per_day = a.orders_per_day
    router = GridRouter(city.terrain, cfg.energy, cfg.travel)
    orders = ODGenerator(city, cfg.demand, router).generate(a.seed)
    d = uniform_design(city, candidate_sites(city), a.stations, a.batteries)
    layout = d.layout(candidate_sites(city))
    policy = LeastExpectedTime() if a.policy == "least_time" else NearestStation()

    t = time.time()
    sim = Simulation(city, cfg, layout, orders, seed=a.seed, station_policy=policy).run()
    print(f"simulated {len(orders)} orders in {time.time() - t:.1f}s")
    s = summarize(sim)
    for k, v in s.items():
        print(f"  {k:24s} {v:.3f}" if isinstance(v, float) else f"  {k:24s} {v}")
    st = station_table(sim)
    st.to_csv(out / "stations.csv", index=False)
    (out / "summary.json").write_text(json.dumps(s, indent=2))
    peak = int(np.argmax([x["queue"].sum() for x in sim.snapshots]))
    viz.snapshot_figure(city, sim, peak, str(out / "snapshot_peak.png"))
    if a.gif:
        viz.animate(city, sim, str(out / "day.gif"))
    print("wrote", out)


if __name__ == "__main__":
    main()
