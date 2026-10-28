"""Dispatch models and station-choice policies.

A dispatcher reacts to two events, a new order and a rider becoming free, and
calls `sim.assign(rider, order)` to match them. Station-choice policies decide
where a rider who needs a battery goes. Both are swappable.
"""
from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING, Protocol

import numpy as np

if TYPE_CHECKING:
    from .agents import Rider, Station
    from .demand import Order
    from .engine import Simulation


class Dispatcher(Protocol):
    def bind(self, sim: "Simulation") -> None: ...
    def on_order(self, t: float, order: "Order") -> None: ...
    def on_rider_free(self, t: float, rider: "Rider") -> None: ...
    def on_cancel(self, t: float, order: "Order") -> None: ...


class NaiveRegionalDispatcher:
    """Regional FIFO queues; each order goes to the nearest available rider.

    A rider is eligible if idle, within `max_pickup_km`, in the order's region
    (unless cross-region is allowed) and holding enough charge for pickup,
    delivery and the trip to the nearest station afterwards.
    """

    def bind(self, sim):
        self.sim = sim
        self.p = sim.cfg.dispatch
        self.queues: dict[int, deque] = {}

    def _eligible(self, order, cand: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        sim = self.sim
        d_pick = sim.router.distance(sim.rider_pos[cand], order.origin)
        need = sim.energy_bound(d_pick + sim.order_tail_km[order.id])
        ok = (d_pick <= self.p.max_pickup_km) & (sim.rider_soc[cand] * sim.cfg.energy.battery_wh >= need)
        if not self.p.allow_cross_region:
            ok &= sim.rider_region[cand] == sim.order_region(order)
        return ok, d_pick

    def _best_rider(self, order):
        sim = self.sim
        cand = np.flatnonzero(sim.rider_idle)
        if len(cand) == 0:
            return None
        ok, d = self._eligible(order, cand)
        if not ok.any():
            return None
        return sim.riders[int(cand[ok][d[ok].argmin()])]

    def on_order(self, t, order):
        r = self._best_rider(order)
        if r is not None:
            self.sim.assign(t, r, order)
        else:
            self.queues.setdefault(self.sim.order_region(order), deque()).append(order)

    def on_cancel(self, t, order):
        q = self.queues.get(self.sim.order_region(order))
        if q is not None and order in q:
            q.remove(order)

    def on_rider_free(self, t, rider):
        sim = self.sim
        own = int(sim.rider_region[rider.id])
        heads = [(reg, q[0]) for reg, q in self.queues.items() if q
                 and (self.p.allow_cross_region or reg == own)]
        if not heads:
            return
        ids = np.array([o.id for _, o in heads])
        d = sim.router.distance(sim.order_origin[ids], rider.pos)
        need = sim.energy_bound(d + sim.order_tail_km[ids])
        ok = (d <= self.p.max_pickup_km) & (rider.soc * sim.cfg.energy.battery_wh >= need)
        if not ok.any():
            return
        # own region first, then the oldest order, then the closest
        keys = [((reg != own), o.t_created, d[i]) for i, (reg, o) in enumerate(heads) if ok[i]]
        idx = [i for i in range(len(heads)) if ok[i]]
        reg, order = heads[idx[min(range(len(keys)), key=keys.__getitem__)]]
        self.queues[reg].popleft()
        sim.assign(t, rider, order)


class StationPolicy(Protocol):
    def choose(self, t: float, rider: "Rider", stations: list["Station"], sim: "Simulation") -> int: ...


class NearestStation:
    """Naive rider: go to the closest station regardless of its queue."""

    def choose(self, t, rider, stations, sim):
        return int(sim.router.distance(sim.station_xy, rider.pos).argmin())


class LeastExpectedTime:
    """Informed rider: minimise travel time + expected wait, among reachable stations."""

    def __init__(self, candidates: int = 4):
        self.k = candidates

    def choose(self, t, rider, stations, sim):
        d = sim.router.distance(sim.station_xy, rider.pos)
        reach = rider.soc * sim.cfg.energy.battery_wh >= sim.energy_bound(d)
        order = np.argsort(np.where(reach, d, np.inf))[: self.k]
        best, best_cost = int(d.argmin()), np.inf
        for s in order:
            if not reach[s]:
                continue
            travel = sim.router.route(rider.pos, stations[s].xy).minutes
            cost = travel + stations[s].estimated_wait(t + travel)
            if cost < best_cost:
                best, best_cost = int(s), cost
        return best
