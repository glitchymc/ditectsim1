"""Simulation-based optimisation of station locations and battery inventory.

Decision: K station sites (chosen from a candidate grid) and the split of a
total battery budget B across them. Every candidate layout is scored by
running the simulator on the same set of pre-generated order streams
(common random numbers), so differences between layouts are not drowned out
by demand noise. The final comparison re-scores layouts on held-out seeds.

Optimisers
----------
* `IterativeRebalancer`: the deck's loop. Run, find the station with the most
  queueing and the most under-used one, then try moving batteries or the
  station itself toward the queued area; keep the best improving move.
* `SimulatedAnnealing`: generic local search over the same neighbourhood,
  mixing random and guided moves. A check that the guided loop is not stuck.
"""
from __future__ import annotations

import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from multiprocessing import get_context
from typing import Callable

import numpy as np
import pandas as pd

from .config import SimConfig
from .data.city import City
from .demand import ODGenerator
from .engine import Simulation, StationLayout
from .geo import MANHATTAN_GRID_DEG, GridRouter
from .metrics import station_table, summarize


# ---------------------------------------------------------------- objective
@dataclass
class Objective:
    """Cost in rider-hour equivalents (lower is better).

    swap time (detour + queue + service) + w_late x hours of lateness beyond
    `late_after_min` + w_cancel x hours per undelivered order.
    """

    w_swap: float = 1.0
    w_late: float = 1.0
    w_cancel: float = 1.0
    late_after_min: float = 45.0

    def components(self, sim: Simulation) -> dict:
        swap_h = sum(r.swap_min for r in sim.riders) / 60
        d = sim.t_delivered - np.array([o.t_created for o in sim.orders])
        late_h = np.nansum(np.clip(d - self.late_after_min, 0, None)) / 60
        undelivered = int(np.isnan(sim.t_delivered).sum())
        return {"swap_h": swap_h, "late_h": float(late_h), "undelivered": undelivered,
                "cost": self.w_swap * swap_h + self.w_late * late_h + self.w_cancel * undelivered}


# ---------------------------------------------------------------- evaluator
@dataclass
class EvalResult:
    cost: float
    cost_se: float
    per_seed: list[float]
    summary: dict
    stations: pd.DataFrame        # per-station stats averaged over seeds
    triggers: list[np.ndarray]    # per-station swap trigger points (all seeds)


_W: dict = {}


def _worker_init(city, cfg, orders_by_seed, objective, policy_factory, dispatcher_factory):
    _W.update(city=city, cfg=cfg, orders=orders_by_seed, obj=objective,
              policy=policy_factory, dispatcher=dispatcher_factory)


def _run_one(layout: StationLayout, seed: int):
    w = _W
    sim = Simulation(w["city"], w["cfg"], layout, w["orders"][seed], seed=seed,
                     dispatcher=w["dispatcher"]() if w["dispatcher"] else None,
                     station_policy=w["policy"]() if w["policy"] else None).run()
    comp = w["obj"].components(sim)
    summ = summarize(sim, w["obj"].late_after_min)
    summ.update(comp)
    st = station_table(sim)
    trig = [np.asarray(s.trigger_points).reshape(-1, 2) for s in sim.stations]
    return comp["cost"], summ, st, trig


class Evaluator:
    """Scores layouts on fixed order streams; optionally in parallel processes."""

    def __init__(self, city: City, cfg: SimConfig, seeds: list[int],
                 objective: Objective | None = None,
                 policy_factory: Callable | None = None,
                 dispatcher_factory: Callable | None = None,
                 n_jobs: int | None = None, orders_by_seed: dict | None = None):
        self.city, self.cfg, self.seeds = city, cfg, list(seeds)
        self.objective = objective or Objective()
        router = GridRouter(city.terrain, cfg.energy, cfg.travel)
        gen = ODGenerator(city, cfg.demand, router)
        self.orders = dict(orders_by_seed or {})
        for s in self.seeds:
            if s not in self.orders:
                self.orders[s] = gen.generate(s)
        self.n_evals = 0
        self.n_jobs = n_jobs if n_jobs is not None else min(len(self.seeds), os.cpu_count() or 1)
        args = (city, cfg, self.orders, self.objective, policy_factory, dispatcher_factory)
        if self.n_jobs > 1:
            self._pool = ProcessPoolExecutor(self.n_jobs, mp_context=get_context("fork"),
                                             initializer=_worker_init, initargs=args)
        else:
            self._pool = None
            _worker_init(*args)

    def close(self):
        if self._pool is not None:
            self._pool.shutdown()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def evaluate(self, layout: StationLayout) -> EvalResult:
        self.n_evals += 1
        if self._pool is not None:
            outs = list(self._pool.map(_run_one, [layout] * len(self.seeds), self.seeds))
        else:
            outs = [_run_one(layout, s) for s in self.seeds]
        costs = [o[0] for o in outs]
        summ = pd.DataFrame([o[1] for o in outs]).mean(numeric_only=True).to_dict()
        st = pd.concat([o[2] for o in outs]).groupby("station").mean(numeric_only=True).reset_index()
        trig = [np.concatenate([o[3][k] for o in outs]) for k in range(len(layout))]
        se = float(np.std(costs, ddof=1) / math.sqrt(len(costs))) if len(costs) > 1 else 0.0
        return EvalResult(float(np.mean(costs)), se, costs, summ, st, trig)


# ---------------------------------------------------------- layout helpers
def candidate_sites(city: City, spacing_km: float = 0.4) -> np.ndarray:
    return city.region.grid(spacing_km)


def _kmeans(pts: np.ndarray, k: int, rng: np.random.Generator, iters: int = 50,
            weights: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Weighted k-means with k-means++ seeding."""
    w = np.ones(len(pts)) if weights is None else np.asarray(weights, float)
    c = [pts[rng.choice(len(pts), p=w / w.sum())]]
    for _ in range(1, k):
        d2 = ((pts[:, None, :] - np.array(c)[None]) ** 2).sum(-1).min(1) * w
        c.append(pts[rng.choice(len(pts), p=d2 / d2.sum())])
    c = np.array(c)
    for _ in range(iters):
        lab = ((pts[:, None, :] - c[None]) ** 2).sum(-1).argmin(1)
        new = np.array([np.average(pts[lab == j], axis=0, weights=w[lab == j])
                        if (lab == j).any() else c[j] for j in range(k)])
        if np.allclose(new, c):
            break
        c = new
    lab = ((pts[:, None, :] - c[None]) ** 2).sum(-1).argmin(1)
    return c, lab


def snap(points: np.ndarray, sites: np.ndarray) -> np.ndarray:
    """Map each point to a distinct nearest candidate site index."""
    used: set[int] = set()
    out = []
    for p in points:
        order = np.argsort(((sites - p) ** 2).sum(1))
        j = next(int(i) for i in order if int(i) not in used)
        used.add(j)
        out.append(j)
    return np.array(out)


def split_budget(weights: np.ndarray, total: int, min_per: int = 2) -> np.ndarray:
    """Largest-remainder apportionment of `total` batteries with a floor."""
    weights = np.asarray(weights, float)
    k = len(weights)
    if total < k * min_per:
        raise ValueError("battery budget below the per-station minimum")
    w = weights / weights.sum() if weights.sum() > 0 else np.full(k, 1 / k)
    extra = total - k * min_per
    raw = w * extra
    b = np.floor(raw).astype(int)
    b[np.argsort(-(raw - b))[: extra - b.sum()]] += 1
    return b + min_per


@dataclass
class Design:
    """A layout expressed as candidate-site indices + batteries."""

    sites: np.ndarray
    batteries: np.ndarray

    def layout(self, cand: np.ndarray) -> StationLayout:
        return StationLayout(cand[self.sites], self.batteries)

    def copy(self) -> "Design":
        return Design(self.sites.copy(), self.batteries.copy())

    def key(self):
        o = np.argsort(self.sites)
        return tuple(self.sites[o]), tuple(self.batteries[o])


def uniform_design(city: City, cand: np.ndarray, k: int, total: int, seed: int = 0) -> Design:
    """Spatially even coverage: k-means over the service area, equal batteries."""
    rng = np.random.default_rng(seed)
    c, _ = _kmeans(city.region.grid(0.2), k, rng)
    return Design(snap(c, cand), split_budget(np.ones(k), total))


def demand_design(city: City, cand: np.ndarray, k: int, total: int, orders, seed: int = 0) -> Design:
    """k-means over delivery destinations; batteries proportional to cluster size."""
    rng = np.random.default_rng(seed)
    pts = np.array([o.dest for o in orders])
    c, lab = _kmeans(pts, k, rng)
    return Design(snap(c, cand), split_budget(np.bincount(lab, minlength=k), total))


def pilot_design(ev: Evaluator, cand: np.ndarray, k: int, total: int,
                 pilot_spacing_km: float = 0.8, seed: int = 0) -> Design:
    """Run once with dense, well-stocked stations to learn where riders want to
    swap, then cluster those swap-trigger points (a simulation-informed p-median)."""
    city = ev.city
    dense = city.region.grid(pilot_spacing_km)
    res = ev.evaluate(StationLayout(dense, np.full(len(dense), max(4, total // len(dense) * 2))))
    pts = np.concatenate([t for t in res.triggers if len(t)])
    rng = np.random.default_rng(seed)
    c, lab = _kmeans(pts, k, rng)
    return Design(snap(c, cand), split_budget(np.bincount(lab, minlength=k), total))


def random_design(cand: np.ndarray, k: int, total: int, seed: int = 0) -> Design:
    rng = np.random.default_rng(seed)
    return Design(rng.choice(len(cand), k, replace=False), split_budget(np.ones(k), total))


# --------------------------------------------------------------- moves
_TH = math.radians(MANHATTAN_GRID_DEG)
_UV = np.array([[math.sin(_TH), math.cos(_TH)], [math.cos(_TH), -math.sin(_TH)]])


def l1_median(pts: np.ndarray) -> np.ndarray:
    """Coordinate-wise median in the street-grid frame (minimises grid L1 distance)."""
    uv = pts @ _UV.T
    return np.median(uv, axis=0) @ _UV


class Moves:
    def __init__(self, cand: np.ndarray, min_batteries: int = 2, rng=None):
        self.cand = cand
        self.min_b = min_batteries
        self.rng = rng or np.random.default_rng()

    def _free_site_near(self, d: Design, p: np.ndarray, exclude: int | None = None) -> int | None:
        used = set(int(s) for s in d.sites)
        if exclude is not None:
            used.discard(int(d.sites[exclude]))
        for j in np.argsort(((self.cand - p) ** 2).sum(1)):
            if int(j) not in used:
                return int(j)
        return None

    def transfer(self, d: Design, src: int, dst: int, n: int) -> Design | None:
        n = min(n, d.batteries[src] - self.min_b)
        if n <= 0 or src == dst:
            return None
        e = d.copy()
        e.batteries[src] -= n
        e.batteries[dst] += n
        return e

    def relocate(self, d: Design, i: int, p: np.ndarray) -> Design | None:
        j = self._free_site_near(d, p, exclude=i)
        if j is None or j == d.sites[i]:
            return None
        e = d.copy()
        e.sites[i] = j
        return e

    def random_move(self, d: Design, radius_km: float = 1.0) -> Design | None:
        rng = self.rng
        k = len(d.sites)
        if rng.uniform() < 0.5:
            i = int(rng.integers(k))
            p = self.cand[d.sites[i]] + rng.normal(0, radius_km, 2)
            return self.relocate(d, i, p)
        src, dst = rng.choice(k, 2, replace=False)
        return self.transfer(d, int(src), int(dst), int(rng.integers(1, 4)))


def _guided_proposals(d: Design, res: EvalResult, mv: Moves, step: int) -> list[tuple[str, Design]]:
    """The deck's rule: shift capacity / stations from unused toward queued stations."""
    st = res.stations.sort_values("station")
    wait = st["total_wait_min"].to_numpy()
    use = st["swaps_per_battery"].to_numpy()
    spare = st["avg_charged_frac"].to_numpy()
    hot = int(np.argmax(wait))
    cold = int(np.argmin(use))
    surplus = int(np.argmax(spare - 0.5 * (np.arange(len(spare)) == hot)))
    props: list[tuple[str, Design]] = []
    if wait[hot] > 0:
        for src, tag in ((surplus, "surplus"), (cold, "cold")):
            e = mv.transfer(d, src, hot, step)
            if e is not None:
                props.append((f"batteries {tag}->hot x{step}", e))
        trig = res.triggers[hot]
        if len(trig) >= 4 and cold != hot:
            # split the hot station's catchment: send the cold station to the
            # half of the hot station's demand that is farther from it
            c, lab = _kmeans(trig, 2, mv.rng, iters=20)
            far = int(np.argmax(((c - mv.cand[d.sites[hot]]) ** 2).sum(1)))
            e = mv.relocate(d, cold, c[far])
            if e is not None:
                moved = min(step, e.batteries[hot] - mv.min_b)
                if moved > 0:
                    e.batteries[hot] -= moved
                    e.batteries[cold] += moved
                props.append(("relocate cold->hot catchment", e))
    # pull every station toward the L1-median of its own swap demand
    e = d.copy()
    for i, trig in enumerate(res.triggers):
        if len(trig) >= 3:
            j = mv._free_site_near(e, l1_median(trig), exclude=i)
            if j is not None:
                e.sites[i] = j
    if e.key() != d.key():
        props.append(("recentre all", e))
    return props


def paired_improvement(new: EvalResult, old: EvalResult, z: float) -> bool:
    """True if `new` beats `old` by more than z standard errors of the paired
    per-seed difference (both were run on the same seeds, so noise cancels)."""
    d = np.asarray(new.per_seed) - np.asarray(old.per_seed)
    if len(d) < 2:
        return d.mean() < 0
    return d.mean() + z * d.std(ddof=1) / math.sqrt(len(d)) < 0


@dataclass
class OptResult:
    best: Design
    best_eval: EvalResult
    history: pd.DataFrame
    evals: int
    seconds: float
    trace: list = field(default_factory=list)


class IterativeRebalancer:
    def __init__(self, ev: Evaluator, cand: np.ndarray, min_batteries: int = 2,
                 max_iters: int = 30, step0: int = 4, patience: int = 6, n_random: int = 4,
                 z: float = 1.0, seed: int = 0, verbose: bool = True):
        self.ev, self.cand = ev, cand
        self.mv = Moves(cand, min_batteries, np.random.default_rng(seed))
        self.max_iters, self.step0, self.verbose = max_iters, step0, verbose
        self.patience, self.n_random, self.z = patience, n_random, z

    def run(self, d0: Design) -> OptResult:
        t0 = time.time()
        cur, res = d0, self.ev.evaluate(d0.layout(self.cand))
        hist = [{"iter": 0, "move": "start", "cost": res.cost, "best": res.cost, "accepted": True}]
        step = self.step0
        seen = {cur.key()}
        stale = 0
        for it in range(1, self.max_iters + 1):
            props = [(n, e) for n, e in _guided_proposals(cur, res, self.mv, step)
                     if e.key() not in seen]
            # random fallbacks keep the search moving when the guided moves stall
            while len(props) < self.n_random:
                e = self.mv.random_move(cur)
                if e is not None and e.key() not in seen:
                    props.append(("random", e))
            scored = []
            for name, e in props:
                seen.add(e.key())
                r = self.ev.evaluate(e.layout(self.cand))
                scored.append((r.cost, name, e, r))
            c, name, e, r = min(scored, key=lambda s: s[0])
            accepted = paired_improvement(r, res, self.z)
            if accepted:
                cur, res = e, r
                stale = 0
            else:
                stale += 1
                step = max(1, step // 2)
            hist.append({"iter": it, "move": name, "cost": c, "best": res.cost, "accepted": accepted})
            if self.verbose:
                print(f"  iter {it:2d}  {'+' if accepted else ' '} {name:32s} cost {c:8.1f}  best {res.cost:8.1f}")
            if stale >= self.patience:
                break
        return OptResult(cur, res, pd.DataFrame(hist), self.ev.n_evals, time.time() - t0)


class SimulatedAnnealing:
    def __init__(self, ev: Evaluator, cand: np.ndarray, min_batteries: int = 2,
                 n_evals: int = 120, t0_frac: float = 0.02, guided_prob: float = 0.3,
                 z: float = 1.0, seed: int = 0, verbose: bool = True):
        self.z = z
        self.ev, self.cand = ev, cand
        self.rng = np.random.default_rng(seed)
        self.mv = Moves(cand, min_batteries, self.rng)
        self.n_evals, self.t0_frac, self.guided_prob, self.verbose = n_evals, t0_frac, guided_prob, verbose

    def run(self, d0: Design) -> OptResult:
        t_start = time.time()
        cur, res = d0, self.ev.evaluate(d0.layout(self.cand))
        best, best_res = cur, res
        T0 = self.t0_frac * res.cost
        hist = [{"iter": 0, "move": "start", "cost": res.cost, "best": res.cost, "accepted": True}]
        for it in range(1, self.n_evals):
            T = T0 * (1 - it / self.n_evals) + 1e-9
            name, e = "random", None
            if self.rng.uniform() < self.guided_prob:
                props = _guided_proposals(cur, res, self.mv, int(self.rng.integers(1, 4)))
                if props:
                    name, e = props[int(self.rng.integers(len(props)))]
            if e is None:
                e = self.mv.random_move(cur)
                if e is None:
                    continue
            r = self.ev.evaluate(e.layout(self.cand))
            accept = r.cost < res.cost or self.rng.uniform() < math.exp(-(r.cost - res.cost) / T)
            if accept:
                cur, res = e, r
                # the incumbent changes only on a statistically clear paired win
                if paired_improvement(r, best_res, self.z):
                    best, best_res = e, r
            hist.append({"iter": it, "move": name, "cost": r.cost, "best": best_res.cost, "accepted": accept})
            if self.verbose and it % 10 == 0:
                print(f"  SA {it:3d}  T {T:6.2f}  current {res.cost:8.1f}  best {best_res.cost:8.1f}")
        return OptResult(best, best_res, pd.DataFrame(hist), self.ev.n_evals, time.time() - t_start)
