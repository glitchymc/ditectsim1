"""Backend: battery swap stations (BSS), trucks and the central charging station (CCS).

Flow (right half of the flowchart):
  BSS (swap, no on-site charging) --truck--> CCS unloading buffer -> testing
  (discard bad packs) -> van -> charging hall (constant current, then constant
  voltage) -> van -> loading buffer --truck--> BSS

Truck options:
  1. "hub":       trucks wait at the CCS; a BSS whose depleted count reaches
                  `threshold` calls one (CCS -> BSS -> CCS).
  2. "dedicated": one truck per BSS (BSS -> CCS -> BSS), leaving on a
                  (a) Poisson clock or (b) fixed schedule, and at the CCS
                  (c) returning immediately or waiting for enough charged packs.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from .battery import Battery, apply_charge, charge_minutes

if TYPE_CHECKING:
    from .model import IntegratedSim


# --------------------------------------------------------------------- BSS
@dataclass(eq=False)
class BSS:
    id: int
    xy: np.ndarray
    slots: int
    charged: deque = field(default_factory=deque)
    depleted: list = field(default_factory=list)
    queue: deque = field(default_factory=deque)       # (rider, t_arrive)
    in_swap: int = 0
    truck_called: bool = False
    # stats
    swaps: int = 0
    waits: list = field(default_factory=list)
    max_queue: int = 0
    truck_visits: int = 0
    _t: float = 0.0
    area_charged: float = 0.0
    area_depleted: float = 0.0
    area_queue: float = 0.0
    stockout_min: float = 0.0

    def advance(self, t: float):
        dt = t - self._t
        if dt > 0:
            self.area_charged += dt * len(self.charged)
            self.area_depleted += dt * len(self.depleted)
            self.area_queue += dt * len(self.queue)
            if not self.charged:
                self.stockout_min += dt
            self._t = t

    @property
    def free_slots(self) -> int:
        return self.slots - len(self.charged) - len(self.depleted) - self.in_swap


# ------------------------------------------------------------------- trucks
@dataclass(eq=False)
class Truck:
    id: int
    home: str                     # "ccs" or "bss<k>"
    bss: int | None = None        # dedicated truck's station
    cargo_charged: list = field(default_factory=list)
    cargo_depleted: list = field(default_factory=list)
    busy: bool = False
    pos: np.ndarray | None = None
    leg: tuple | None = None
    trips: int = 0
    km: float = 0.0
    packs_moved: int = 0
    missed_departures: int = 0
    wait_until: float = 0.0
    busy_min: float = 0.0
    _t_busy: float = 0.0

    def position(self, t):
        if self.leg is None:
            return self.pos
        t0, t1, p0, p1 = self.leg
        if t >= t1 or t1 <= t0:
            return p1
        return p0 + (p1 - p0) * max(0.0, t - t0) / (t1 - t0)


class Fleet:
    """Truck dispatching for both options in the flowchart."""

    def __init__(self, sim: "IntegratedSim"):
        self.sim = sim
        self.p = sim.cfg.trucks
        self.trucks: list[Truck] = []
        self.requests: deque = deque()   # hub mode: BSS ids waiting for a truck
        self.waiting: deque = deque()    # dedicated "wait" policy: trucks at the CCS
        ccs = sim.ccs_xy
        if self.p.mode == "hub":
            for i in range(self.p.n_trucks):
                self.trucks.append(Truck(i, "ccs", pos=ccs.copy()))
        elif self.p.mode == "dedicated":
            for s in sim.stations:
                t = Truck(len(self.trucks), f"bss{s.id}", bss=s.id, pos=s.xy.copy())
                self.trucks.append(t)
        else:
            raise ValueError(f"unknown truck mode {self.p.mode!r}")

    # ------------------------------------------------------------ helpers
    def _drive(self, tr: Truck, dest, then, *args):
        sim = self.sim
        km = float(sim.router.distance(tr.pos, dest))
        minutes = km / self.p.speed_kmh * 60
        tr.km += km
        tr.leg = (sim.k.now, sim.k.now + minutes, tr.pos.copy(), np.asarray(dest, float).copy())
        tr.pos = np.asarray(dest, float).copy()
        sim.k.after(minutes, then, tr, *args)

    def _handle(self, n: int) -> float:
        return n * self.p.handle_min_per_pack

    def _set_busy(self, tr: Truck, busy: bool):
        now = self.sim.k.now
        if tr.busy and not busy:
            tr.busy_min += now - tr._t_busy
        if busy and not tr.busy:
            tr._t_busy = now
        tr.busy = busy

    def start(self):
        if self.p.mode == "dedicated":
            for tr in self.trucks:
                self._schedule_next(tr, first=True)

    # ---------------------------------------------------- option 1: hub
    def on_depleted(self, bss: BSS):
        """Option 1a: call a truck when depleted packs reach the threshold. A
        station that has run dry with riders waiting also calls one, or it could
        never recover (no swaps -> no new depleted packs -> no trigger)."""
        if self.p.mode != "hub" or bss.truck_called:
            return
        starving = not bss.charged and bss.queue
        if len(bss.depleted) >= self.p.threshold or starving:
            bss.truck_called = True
            self.requests.append(bss.id)
            self._dispatch_hub()

    def _dispatch_hub(self):
        sim = self.sim
        while self.requests:
            tr = next((t for t in self.trucks if not t.busy), None)
            if tr is None:
                return
            s = sim.stations[self.requests[0]]
            # nothing to bring and nothing worth collecting yet: wait for charged packs
            if sim.ccs.n_loading == 0 and len(s.depleted) < self.p.threshold:
                return
            self.requests.popleft()
            self._set_busy(tr, True)
            tr.trips += 1
            target = max(0, s.slots - len(s.charged))
            tr.cargo_charged = sim.ccs.take_charged(min(self.p.capacity, target), "truck")
            sim.k.after(self._handle(len(tr.cargo_charged)), self._drive, tr, s.xy, self._at_bss, s)

    def _at_bss(self, tr: Truck, s: BSS):
        """Hub trucks swap loads (take depleted, drop charged); dedicated trucks
        arrive home and only drop charged packs (they collect at departure)."""
        sim = self.sim
        hub = self.p.mode == "hub"
        s.advance(sim.k.now)
        s.truck_visits += 1
        take = []
        if hub:
            room = self.p.capacity - len(tr.cargo_charged) - len(tr.cargo_depleted)
            take = s.depleted[:room]
            del s.depleted[:len(take)]
            for b in take:
                b.where = "truck"
            tr.cargo_depleted.extend(take)
            s.truck_called = False
        put = tr.cargo_charged[:max(0, s.free_slots)]
        del tr.cargo_charged[:len(put)]
        for b in put:
            b.where = "bss_charged"
        s.charged.extend(put)
        tr.packs_moved += len(take) + len(put)
        sim.serve_queue(s)
        if hub:
            sim.k.after(self._handle(len(take) + len(put)), self._drive, tr, sim.ccs_xy, self._back_to_ccs)
        else:
            sim.k.after(self._handle(len(put)), self._dedicated_return_done, tr)

    def _back_to_ccs(self, tr: Truck):
        sim = self.sim
        n = len(tr.cargo_depleted) + len(tr.cargo_charged)
        sim.ccs.receive_depleted(tr.cargo_depleted)
        sim.ccs.return_charged(tr.cargo_charged)
        tr.cargo_depleted, tr.cargo_charged = [], []
        tr.leg = None

        def done(tr=tr):
            self._set_busy(tr, False)
            # a BSS may have crossed the threshold again while we were out
            for s in sim.stations:
                self.on_depleted(s)
            self._dispatch_hub()

        sim.k.after(self._handle(n), done)

    # ---------------------------------------------- option 2: dedicated
    def _schedule_next(self, tr: Truck, first: bool = False):
        sim, p = self.sim, self.p
        if p.departure == "poisson":
            dt = sim.rng.exponential(p.mean_headway_min)
            sim.k.after(dt, self._depart, tr)
        elif p.departure == "schedule":
            now = sim.k.now
            if first:
                t = p.schedule_start_min
                while t < now:
                    t += p.schedule_every_min
            else:
                t = now + p.schedule_every_min
            sim.k.at(t, self._depart, tr)
        else:
            raise ValueError(f"unknown departure rule {p.departure!r}")

    def _depart(self, tr: Truck):
        sim = self.sim
        self._schedule_next(tr)
        if tr.busy:
            tr.missed_departures += 1
            return
        s = sim.stations[tr.bss]
        s.advance(sim.k.now)
        self._set_busy(tr, True)
        tr.trips += 1
        take = s.depleted[:self.p.capacity - len(tr.cargo_charged)]
        del s.depleted[:len(take)]
        for b in take:
            b.where = "truck"
        tr.cargo_depleted.extend(take)
        tr.packs_moved += len(take)
        sim.k.after(self._handle(len(take)), self._drive, tr, sim.ccs_xy, self._dedicated_at_ccs)

    def _dedicated_at_ccs(self, tr: Truck):
        sim = self.sim
        n = len(tr.cargo_depleted) + len(tr.cargo_charged)
        sim.ccs.receive_depleted(tr.cargo_depleted)
        sim.ccs.return_charged(tr.cargo_charged)
        tr.cargo_depleted, tr.cargo_charged = [], []
        tr.leg = None
        sim.k.after(self._handle(n), self._dedicated_load, tr, sim.k.now + self._handle(n))

    def _target(self, tr: Truck) -> int:
        s = self.sim.stations[tr.bss]
        return max(0, min(self.p.capacity, s.slots - len(s.charged) - s.in_swap))

    def _dedicated_load(self, tr: Truck, t_arrived: float):
        sim = self.sim
        target = self._target(tr)
        if self.p.return_policy == "wait" and sim.ccs.n_loading < target:
            tr.wait_until = t_arrived + self.p.max_wait_min
            self.waiting.append(tr)
            sim.k.at(tr.wait_until, self._wait_timeout, tr)
            return
        self._leave_ccs(tr, target)

    def _wait_timeout(self, tr: Truck):
        if tr in self.waiting:
            self.waiting.remove(tr)
            self._leave_ccs(tr, self._target(tr))

    def on_charged_available(self):
        """Called when the loading buffer grows: release waiting trucks / requests."""
        if self.p.mode == "hub":
            self._dispatch_hub()
            return
        for tr in list(self.waiting):
            if self.sim.ccs.n_loading >= self._target(tr):
                self.waiting.remove(tr)
                self._leave_ccs(tr, self._target(tr))

    def _leave_ccs(self, tr: Truck, target: int):
        sim = self.sim
        tr.cargo_charged = sim.ccs.take_charged(target, "truck")
        s = sim.stations[tr.bss]
        sim.k.after(self._handle(len(tr.cargo_charged)), self._drive, tr, s.xy, self._at_bss, s)

    def _dedicated_return_done(self, tr: Truck):
        tr.leg = None
        self._set_busy(tr, False)


# ---------------------------------------------------------------------- CCS
class CCS:
    """Central charging station: unloading buffer -> testing -> van -> charging
    hall (CC then CV) -> van -> loading buffer."""

    def __init__(self, sim: "IntegratedSim"):
        self.sim = sim
        self.p = sim.cfg.ccs
        self.unloading: deque = deque()
        self.testing = 0
        self.tested: deque = deque()          # passed, waiting for the van
        self.van_load: list = []
        self.van_busy = False
        self.hall_queue: deque = deque()      # at the charging hall, waiting for a charger
        self.charging = 0
        self.hall_ready: deque = deque()      # charged, waiting for the van back
        self.loading: deque = deque()         # charged, ready for trucks
        self.discarded: list[Battery] = []
        self.purchased = 0
        self.charged_count = 0
        self.charge_min_total = 0.0
        self.charger_busy_min = 0.0
        self._charging_battery: list = []

    @property
    def n_loading(self) -> int:
        return len(self.loading)

    # ----------------------------------------------------------- in / out
    def receive_depleted(self, packs: list[Battery]):
        for b in packs:
            b.where = "ccs_unloading"
            self.unloading.append(b)
        self._try_test()

    def return_charged(self, packs: list[Battery]):
        for b in packs:
            b.where = "ccs_loading"
            self.loading.append(b)
        if packs:
            self.sim.fleet.on_charged_available()

    def take_charged(self, n: int, where: str) -> list[Battery]:
        out = []
        while self.loading and len(out) < n:
            b = self.loading.popleft()
            b.where = where
            out.append(b)
        return out

    # ------------------------------------------------------------ testing
    def _try_test(self):
        while self.unloading and self.testing < self.p.n_testers:
            b = self.unloading.popleft()
            b.where = "ccs_testing"
            self.testing += 1
            self.sim.k.after(self.p.test_min, self._test_done, b)

    def _test_done(self, b: Battery):
        sim = self.sim
        self.testing -= 1
        if b.soh < self.p.soh_min or sim.rng.uniform() < self.p.defect_prob:
            b.where = "discarded"
            self.discarded.append(b)
            if self.p.replace_discarded:
                self.purchased += 1
                self.return_charged([sim.new_battery(soc=1.0, soh=sim.rng.uniform(0.97, 1.0))])
        else:
            b.where = "ccs_tested"
            self.tested.append(b)
        self._try_test()

    # ---------------------------------------------------------------- van
    def start(self):
        self.sim.k.after(self.p.van_every_min, self._van_tick)

    def _van_tick(self):
        self.sim.k.after(self.p.van_every_min, self._van_tick)
        if self.van_busy or not (self.tested or self.hall_ready):
            return
        self.van_busy = True
        self.van_load = [self.tested.popleft() for _ in range(min(self.p.van_capacity, len(self.tested)))]
        for b in self.van_load:
            b.where = "ccs_van"
        self.sim.k.after(self.p.van_transit_min, self._van_at_hall)

    def _van_at_hall(self):
        for b in self.van_load:
            b.where = "ccs_hall_queue"
            self.hall_queue.append(b)
        self._try_charge()
        self.van_load = [self.hall_ready.popleft() for _ in range(min(self.p.van_capacity, len(self.hall_ready)))]
        for b in self.van_load:
            b.where = "ccs_van"
        self.sim.k.after(self.p.van_transit_min, self._van_back)

    def _van_back(self):
        packs, self.van_load = self.van_load, []
        self.van_busy = False
        self.return_charged(packs)

    # ----------------------------------------------------------- chargers
    def _try_charge(self):
        sim = self.sim
        while self.hall_queue and self.charging < self.p.n_chargers:
            b = self.hall_queue.popleft()
            b.where = "ccs_charging"
            self.charging += 1
            m = charge_minutes(b, sim.cfg.battery, self.p.charger)
            self.charge_min_total += m
            self.charger_busy_min += m
            sim.k.after(m, self._charge_done, b)

    def _charge_done(self, b: Battery):
        apply_charge(b, self.sim.cfg.battery, self.p.charger)
        b.where = "ccs_hall_ready"
        self.charging -= 1
        self.charged_count += 1
        self.hall_ready.append(b)
        self._try_charge()
