"""Simulation-based station location + battery allocation.

1. Build baseline layouts (uniform coverage, demand-weighted, simulation pilot).
2. Improve the best one with the deck's iterative queued-vs-unused loop and,
   for comparison, simulated annealing. Both use common random numbers on
   the training seeds.
3. Re-score every layout on held-out seeds and write tables + figures.
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

from ebike_swap import SimConfig  # noqa: E402
from ebike_swap import viz  # noqa: E402
from ebike_swap.dispatch import LeastExpectedTime  # noqa: E402
from ebike_swap.optimize import (Evaluator, IterativeRebalancer, Objective,  # noqa: E402
                                 SimulatedAnnealing, candidate_sites, demand_design,
                                 pilot_design, uniform_design)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    add_city_args(p)
    p.add_argument("--stations", type=int, default=15)
    p.add_argument("--batteries", type=int, default=150)
    p.add_argument("--riders", type=int, default=400)
    p.add_argument("--orders-per-day", type=float, default=8000)
    p.add_argument("--train-seeds", type=int, default=6)
    p.add_argument("--holdout-seeds", type=int, default=8)
    p.add_argument("--iters", type=int, default=25)
    p.add_argument("--sa-evals", type=int, default=100)
    p.add_argument("--z", type=float, default=1.0,
                   help="accept a move only if its paired gain exceeds z standard errors")
    p.add_argument("--policy", choices=["nearest", "least_time"], default="nearest")
    p.add_argument("--candidate-km", type=float, default=0.4)
    p.add_argument("--out", default="out/optimize")
    a = p.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    city = city_from_args(a)
    cfg = SimConfig()
    cfg.rider.n_riders = a.riders
    cfg.demand.orders_per_day = a.orders_per_day
    obj = Objective()
    policy = LeastExpectedTime if a.policy == "least_time" else None
    cand = candidate_sites(city, a.candidate_km)
    train = list(range(1, a.train_seeds + 1))
    hold = list(range(1001, 1001 + a.holdout_seeds))
    print(f"{len(cand)} candidate sites, K={a.stations}, B={a.batteries}, "
          f"train seeds {train}, holdout seeds {hold}")

    t0 = time.time()
    with Evaluator(city, cfg, train, obj, policy_factory=policy) as ev:
        print(f"generated training demand in {time.time() - t0:.0f}s")
        designs = {
            "uniform": uniform_design(city, cand, a.stations, a.batteries),
            "demand k-means": demand_design(city, cand, a.stations, a.batteries, ev.orders[train[0]]),
            "pilot k-means": pilot_design(ev, cand, a.stations, a.batteries),
        }
        base = {n: ev.evaluate(d.layout(cand)) for n, d in designs.items()}
        for n, r in base.items():
            print(f"  {n:16s} train cost {r.cost:8.1f} ± {r.cost_se:.1f}")
        start_name = min(base, key=lambda n: base[n].cost)
        print(f"iterative rebalancing from '{start_name}'")
        it = IterativeRebalancer(ev, cand, max_iters=a.iters, z=a.z).run(designs[start_name])
        print(f"  -> {it.best_eval.cost:.1f} after {len(it.history) - 1} iterations ({it.seconds:.0f}s)")
        print("simulated annealing from the same start")
        sa = SimulatedAnnealing(ev, cand, n_evals=a.sa_evals, z=a.z).run(designs[start_name])
        print(f"  -> {sa.best_eval.cost:.1f} ({sa.seconds:.0f}s)")
        designs["iterative (deck)"] = it.best
        designs["simulated annealing"] = sa.best
        train_evals = ev.n_evals

    print("held-out evaluation")
    with Evaluator(city, cfg, hold, obj, policy_factory=policy) as hv:
        held = {n: hv.evaluate(d.layout(cand)) for n, d in designs.items()}
    rows = []
    for n, r in held.items():
        s = r.summary
        rows.append({"layout": n, "cost": r.cost, "cost_se": r.cost_se,
                     "swap_h": s["swap_h"], "late_h": s["late_h"], "undelivered": s["undelivered"],
                     "mean_swap_downtime_min": s["mean_swap_downtime_min"],
                     "mean_station_wait_min": s["mean_station_wait_min"],
                     "p95_station_wait_min": s["p95_station_wait_min"],
                     "mean_delivery_min": s["mean_delivery_min"], "late_rate": s["late_rate"],
                     "idle_stations": s["idle_stations"], "station_max_queue": s["station_max_queue"]})
    table = pd.DataFrame(rows).sort_values("cost")
    ref = table.set_index("layout").loc["uniform", "cost"]
    table["vs_uniform"] = table["cost"] / ref - 1
    pd.set_option("display.width", 200)
    print(table.round(3).to_string(index=False))
    table.to_csv(out / "holdout_comparison.csv", index=False)

    # Paired difference best-vs-uniform per holdout seed (same demand streams).
    best = table.iloc[0]["layout"]
    diff = np.array(held[best].per_seed) - np.array(held["uniform"].per_seed)
    paired = {"best": best, "mean_diff": float(diff.mean()),
              "se_diff": float(diff.std(ddof=1) / np.sqrt(len(diff))) if len(diff) > 1 else 0.0,
              "seeds_better": int((diff < 0).sum()), "n_seeds": len(diff)}
    print("paired vs uniform:", paired)

    for n, d in designs.items():
        lay = d.layout(cand)
        st = held[n].stations.sort_values("station")
        st.assign(x=lay.xy[:, 0], y=lay.xy[:, 1], batteries=lay.batteries).to_csv(
            out / f"stations_{n.split()[0]}.csv", index=False)
    it.history.to_csv(out / "history_iterative.csv", index=False)
    sa.history.to_csv(out / "history_sa.csv", index=False)
    (out / "run.json").write_text(json.dumps({
        "args": vars(a), "provenance": city.provenance, "train_evals": train_evals,
        "start": start_name, "paired_vs_uniform": paired,
        "iterative_seconds": it.seconds, "sa_seconds": sa.seconds}, indent=2))
    viz.layout_figure(city, {n: d.layout(cand) for n, d in designs.items()},
                      {n: held[n].stations for n in designs}, str(out / "layouts.png"))
    viz.convergence_figure({"iterative (deck)": it.history, "simulated annealing": sa.history},
                           str(out / "convergence.png"))
    print("wrote", out)


if __name__ == "__main__":
    main()
