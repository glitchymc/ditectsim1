"""Frontend: food-ordering platforms and multi-app riders.

Flow (left half of the flowchart):
  Platform k OD generator -> Platform k dispatch (independent algorithm)
  -> offer on the rider's phone -> rider accepts or declines (route / pay)
  -> declined: platform re-offers after `refresh_min`
  -> accepted: order joins the rider's batched active job queue
     [P1 order, P2 order, ...], served as a pickup/drop-off stop sequence.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from ..demand import Order
from .battery import Battery
from .config import PlatformSpec

if TYPE_CHECKING:
    from .model import IntegratedSim


@dataclass(eq=False)
class Job:
    """One order on one platform."""

    gid: int
    platform: int
    order: Order
    pay: float
    od_km: float
    t_created: float
    t_ready: float
    t_accepted: float = math.nan
    t_picked: float = math.nan
    t_delivered: float = math.nan
    cancelled: bool = False
    rider: int = -1
    offers: int = 0
    declined: set = field(default_factory=set)

    @property
    def open(self) -> bool:
        return self.rider < 0 and not self.cancelled


@dataclass(eq=False)
class Stop:
    kind: str          # "P" pickup at restaurant, "D" drop-off at customer
    job: Job

    @property
    def xy(self) -> np.ndarray:
        return self.job.order.origin if self.kind == "P" else self.job.order.dest


@dataclass(eq=False)
class Rider:
    id: int
    pos: np.ndarray
    battery: Battery
    apps: frozenset            # platform indices the rider is logged into
    wage: float                # $/min the rider needs to make an order worth it
    shift: tuple[float, float]
    state: str = "off"         # off | active | to_bss | queued | swapping
    stops: list = field(default_factory=list)
    head_locked: bool = False  # travelling to / waiting at stops[0]
    low: bool = False          # below swap threshold: finish tasks, take no offers
    off_after: bool = False
    pending: set = field(default_factory=set)   # platforms with an offer on the phone
    leg: tuple | None = None   # (t0, t1, p0, p1) for animation
    # stats
    km: float = 0.0
    earnings: float = 0.0
    delivered: dict = field(default_factory=dict)
    offers_seen: int = 0
    offers_accepted: int = 0
    swaps: int = 0
    swap_wait_min: float = 0.0
    swap_trip_min: float = 0.0
    t_swap_start: float = 0.0
    stranded: int = 0

    @property
    def n_jobs(self) -> int:
        return sum(1 for s in self.stops if s.kind == "D")

    def position(self, t: float) -> np.ndarray:
        if self.leg is None:
            return self.pos
        t0, t1, p0, p1 = self.leg
        if t >= t1 or t1 <= t0:
            return p1
        return p0 + (p1 - p0) * max(0.0, t - t0) / (t1 - t0)


def best_insertion(sim: "IntegratedSim", r: Rider, job: Job):
    """Cheapest place to insert the job's pickup and drop-off into the rider's
    stop list (pickup first). Returns (new_stops, added_km, total_km)."""
    start = 1 if r.head_locked and r.stops else 0
    fixed, rest = r.stops[:start], r.stops[start:]
    P, D = Stop("P", job), Stop("D", job)
    dist = sim.router.distance

    def route_km(seq):
        pts = [r.pos] + [s.xy for s in seq]
        return float(sum(dist(pts[i], pts[i + 1]) for i in range(len(pts) - 1)))

    base = route_km(r.stops)
    best = None
    for i in range(len(rest) + 1):
        for j in range(i, len(rest) + 1):
            seq = fixed + rest[:i] + [P] + rest[i:j] + [D] + rest[j:]
            km = route_km(seq)
            if best is None or km < best[2]:
                best = (seq, km - base, km)
    return best


class Dispatcher:
    """Base: a platform's own matching algorithm."""

    def __init__(self, platform: "Platform"):
        self.p = platform

    def candidates(self, job: Job, idle_only: bool) -> tuple[np.ndarray, np.ndarray]:
        sim, k = self.p.sim, self.p.idx
        m = sim.member[:, k] & sim.accepting & ~sim.pending[:, k]
        if idle_only:
            m &= sim.njobs == 0
        if job.declined:
            m[list(job.declined)] = False
        ids = np.flatnonzero(m)
        if len(ids) == 0:
            return ids, np.empty(0)
        d = sim.router.distance(sim.rider_pos[ids], job.order.origin)
        ok = d <= self.p.spec.max_pickup_km
        return ids[ok], d[ok]

    def pick(self, job: Job) -> int | None:
        raise NotImplementedError


class NearestIdle(Dispatcher):
    """Offer to the nearest rider with no active jobs."""

    def pick(self, job):
        ids, d = self.candidates(job, idle_only=True)
        return int(ids[d.argmin()]) if len(ids) else None


class Batching(Dispatcher):
    """Offer to the rider (idle or busy) whose route grows least; busy riders
    qualify if the stacked order adds at most `batch_detour_km`."""

    def pick(self, job):
        ids, d = self.candidates(job, idle_only=False)
        if not len(ids):
            return None
        sim = self.p.sim
        best, best_cost = None, math.inf
        for rid, dk in zip(ids, d):
            r = sim.riders[rid]
            if not r.stops:
                cost = dk + job.od_km
            else:
                _, added, _ = best_insertion(sim, r, job)
                if added - job.od_km > self.p.spec.batch_detour_km:
                    continue
                cost = added
            if cost < best_cost:
                best, best_cost = int(rid), cost
        return best


DISPATCHERS = {"nearest": NearestIdle, "batching": Batching}


class Platform:
    def __init__(self, sim: "IntegratedSim", idx: int, spec: PlatformSpec):
        self.sim, self.idx, self.spec = sim, idx, spec
        self.algo = DISPATCHERS[spec.dispatch](self)
        self.jobs: list[Job] = []

    def submit(self, job: Job):
        self.jobs.append(job)
        self.sim.k.after(self.spec.order_timeout_min, self._timeout, job)
        self.try_offer(job)

    def _timeout(self, job: Job):
        if job.open:
            job.cancelled = True

    def try_offer(self, job: Job):
        if not job.open:
            return
        rid = self.algo.pick(job)
        if rid is None:
            self.sim.k.after(self.spec.refresh_min, self.try_offer, job)
            return
        job.offers += 1
        self.sim.send_offer(self, job, self.sim.riders[rid])

    def on_answer(self, job: Job, rider: Rider, accepted: bool):
        if not accepted:
            job.declined.add(rider.id)
            self.sim.k.after(self.spec.refresh_min, self.try_offer, job)
