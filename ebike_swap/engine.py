"""Discrete-event engine wiring OD generator -> dispatcher -> riders <-> stations."""
from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass

import numpy as np

from .agents import Rider, RiderState, Station
from .config import SimConfig
from .data.city import City
from .demand import Order
from .dispatch import Dispatcher, NaiveRegionalDispatcher, NearestStation, StationPolicy
from .geo import GridRouter, Router



@dataclass
class StationLayout:
    xy: np.ndarray          # (K, 2) km
    batteries: np.ndarray   # (K,) packs per station

    def __post_init__(self):
        self.xy = np.asarray(self.xy, float).reshape(-1, 2)
        self.batteries = np.asarray(self.batteries, int).reshape(-1)
        if len(self.xy) != len(self.batteries):
            raise ValueError("xy and batteries must have the same length")
        if (self.batteries < 1).any():
            raise ValueError("every station needs at least one battery")

    def __len__(self):
        return len(self.batteries)


(ORDER, TIMEOUT, ARRIVE_PICKUP, DEPART_PICKUP, ARRIVE_DROPOFF, ARRIVE_STATION,
 SWAP_DONE, CHARGE_DONE, SHIFT_START, SHIFT_END, SNAPSHOT) = range(11)


class Simulation:
    def __init__(self, city: City, cfg: SimConfig, layout: StationLayout,
                 orders: list[Order], seed: int = 0,
                 dispatcher: Dispatcher | None = None,
                 station_policy: StationPolicy | None = None,
                 router: Router | None = None):
        self.city = city
        self.cfg = cfg
        self.orders = orders
        self.rng = np.random.default_rng(seed + 1_000_003)
        self.router = router or GridRouter(city.terrain, cfg.energy, cfg.travel)
        self.dispatcher = dispatcher or NaiveRegionalDispatcher()
        self.policy = station_policy or NearestStation()

        chargers = cfg.station.chargers_per_station
        self.stations = [Station(i, layout.xy[i].copy(), int(b), int(chargers or b))
                         for i, b in enumerate(layout.batteries)]
        self.station_xy = layout.xy.copy()
        self.minutes_full_charge = cfg.energy.battery_wh / (cfg.station.charge_kw * 1000) * 60

        self._init_energy_bounds()
        self._init_riders()
        self.dispatcher.bind(self)

        n = len(orders)
        self.t_assigned = np.full(n, np.nan)
        self.t_pickup = np.full(n, np.nan)
        self.t_delivered = np.full(n, np.nan)
        self.cancelled = np.zeros(n, bool)
        self._order_region = np.array([o.dest_zone for o in orders], int) if n else np.empty(0, int)
        # Per-order energy need beyond the pickup leg: delivery + trip to nearest station.
        if n:
            org = np.array([o.origin for o in orders])
            dst = np.array([o.dest for o in orders])
            st_km = self.router.distance(dst[:, None, :], self.station_xy[None, :, :]).min(1)
            self.order_tail_km = self.router.distance(org, dst) + st_km
            self.order_origin = org
        else:
            self.order_tail_km = np.empty(0)
            self.order_origin = np.empty((0, 2))
        self.snapshots: list[dict] = []
        self._heap: list = []
        self._seq = itertools.count()
        self.now = 0.0

    # ----------------------------------------------------------------- setup
    def _init_energy_bounds(self):
        cfg, city = self.cfg, self.city
        probe = np.vstack([city.region.grid(0.25), city.restaurant_xy])
        z = city.terrain.elevation(probe[:, 0], probe[:, 1])
        self.max_climb_m = float(z.max() - z.min())
        e = cfg.energy
        self._wh_per_km_bound = e.wh_per_km + e.lights_per_km * e.p_stop_at_light * e.wh_per_stop
        self._wh_climb = self.max_climb_m * e.total_mass_kg * 9.81 / e.climb_efficiency / 3600
        # The deck's assumption: the minimal level covers the longest job plus
        # the trip to a station from anywhere in the service area.
        d_st = self.router.distance(probe[:, None, :], self.station_xy[None, :, :]).min(1)
        self.max_station_km = float(d_st.max())
        worst = cfg.dispatch.max_pickup_km + cfg.demand.max_od_km + self.max_station_km
        thr = cfg.rider.swap_threshold
        if thr == "auto":
            thr = self.energy_bound(worst) / e.battery_wh + cfg.rider.threshold_margin
        self.swap_threshold = float(min(thr, 0.95))
        self.threshold_capped = thr > 0.95

    def energy_bound(self, km):
        """Upper bound on Wh for a trip of `km` (includes worst-case climb)."""
        return np.asarray(km) * self._wh_per_km_bound + self._wh_climb

    def _init_riders(self):
        cfg, rng = self.cfg, self.rng
        n = cfg.rider.n_riders
        rxy = self.city.restaurant_xy
        pos = rxy[rng.integers(len(rxy), size=n)] + rng.normal(0, 0.1, (n, 2))
        soc = rng.uniform(*cfg.rider.init_soc_range, size=n)
        t0, t1 = cfg.demand.start_hour * 60, cfg.demand.end_hour * 60
        if cfg.rider.shift_mode == "all_day":
            shifts = [(t0, t1)] * n
        elif cfg.rider.shift_mode == "peaks":
            prof = np.roll(np.asarray(self.city.hourly_profile, float), -1)
            starts = rng.choice(24, size=n, p=prof / prof.sum()) * 60 + rng.uniform(0, 60, n)
            dur = rng.uniform(4, 9, n) * 60
            shifts = list(zip(np.clip(starts, t0, t1), np.clip(starts + dur, t0, t1)))
        else:
            raise ValueError(f"unknown shift_mode {cfg.rider.shift_mode!r}")
        self.riders = [Rider(i, pos[i].copy(), float(soc[i]), tuple(shifts[i])) for i in range(n)]
        self.rider_pos = pos.copy()
        self.rider_soc = soc.copy()
        self.rider_idle = np.zeros(n, bool)
        self.rider_region = self.city.zone_of(pos)

    # --------------------------------------------------------------- helpers
    def push(self, t, kind, a=None, b=None):
        heapq.heappush(self._heap, (t, next(self._seq), kind, a, b))

    def region_of(self, pts):
        return self.city.zone_of(pts)

    def order_region(self, order: Order) -> int:
        return int(self._order_region[order.id])

    def od_km(self, order: Order) -> float:
        return float(self.router.distance(order.origin, order.dest))

    def nearest_station_km(self, pt) -> float:
        return float(self.router.distance(self.station_xy, pt).min())

    def _move(self, t, rider: Rider, dest) -> float:
        r = self.router.route(rider.pos, dest)
        wh_left = rider.soc * self.cfg.energy.battery_wh - r.wh
        if wh_left < 0:
            rider.stranded += 1
            wh_left = 0.0
        rider.soc = wh_left / self.cfg.energy.battery_wh
        rider.km += r.km
        rider.wh += r.wh
        t1 = t + r.minutes
        rider.leg = (t, t1, rider.pos.copy(), np.asarray(dest, float).copy())
        rider.pos = np.asarray(dest, float).copy()
        self.rider_soc[rider.id] = rider.soc
        self.rider_pos[rider.id] = rider.pos
        return t1

    def _set_idle(self, t, rider: Rider):
        rider.state = RiderState.IDLE
        rider.leg = None
        self.rider_idle[rider.id] = True
        self.rider_region[rider.id] = int(self.region_of(rider.pos[None, :])[0])
        self.dispatcher.on_rider_free(t, rider)

    def _go_off(self, rider: Rider):
        rider.state = RiderState.OFF
        rider.leg = None
        self.rider_idle[rider.id] = False

    # ------------------------------------------------------------- dispatch API
    def assign(self, t, rider: Rider, order: Order):
        assert rider.state is RiderState.IDLE
        self.rider_idle[rider.id] = False
        rider.state = RiderState.TO_PICKUP
        rider.order = order
        self.t_assigned[order.id] = t
        self.push(self._move(t, rider, order.origin), ARRIVE_PICKUP, rider)

    # ---------------------------------------------------------------- events
    def _after_job(self, t, rider: Rider):
        if rider.off_after_job:
            self._go_off(rider)
        elif rider.soc < self.swap_threshold:
            self._go_swap(t, rider)
        else:
            self._set_idle(t, rider)

    def _go_swap(self, t, rider: Rider):
        s = self.policy.choose(t, rider, self.stations, self)
        self.stations[s].trigger_points.append(rider.pos.copy())
        rider.swap_started = t
        rider.state = RiderState.TO_STATION
        self.push(self._move(t, rider, self.stations[s].xy), ARRIVE_STATION, rider, s)

    def _start_swap(self, t, rider: Rider, st: Station, t_arrive: float):
        st.charged -= 1
        st.waits.append(t - t_arrive)
        rider.state = RiderState.SWAPPING
        rider.leg = None
        self.push(t + self.cfg.station.swap_service_min, SWAP_DONE, rider, st.id)

    def _handle(self, t, kind, a, b):
        if kind == ORDER:
            order: Order = a
            self.dispatcher.on_order(t, order)
            if np.isnan(self.t_assigned[order.id]):
                self.push(t + self.cfg.dispatch.order_timeout_min, TIMEOUT, order)
        elif kind == TIMEOUT:
            order = a
            if np.isnan(self.t_assigned[order.id]) and not self.cancelled[order.id]:
                self.cancelled[order.id] = True
                self.dispatcher.on_cancel(t, order)
        elif kind == ARRIVE_PICKUP:
            rider: Rider = a
            rider.leg = None
            ready = rider.order.t_ready
            if ready > t:
                rider.state = RiderState.WAIT_FOOD
                self.push(ready, DEPART_PICKUP, rider)
            else:
                self._handle(t, DEPART_PICKUP, rider, None)
        elif kind == DEPART_PICKUP:
            rider = a
            self.t_pickup[rider.order.id] = t
            rider.state = RiderState.TO_DROPOFF
            self.push(self._move(t, rider, rider.order.dest), ARRIVE_DROPOFF, rider)
        elif kind == ARRIVE_DROPOFF:
            rider = a
            self.t_delivered[rider.order.id] = t
            rider.n_orders += 1
            rider.order = None
            rider.leg = None
            self._after_job(t, rider)
        elif kind == ARRIVE_STATION:
            rider, st = a, self.stations[b]
            rider.leg = None
            st.advance(t)
            st.n_arrivals += 1
            if st.charged > 0 and not st.queue:
                self._start_swap(t, rider, st, t)
            else:
                rider.state = RiderState.QUEUED
                st.queue.append((rider.id, t))
                st.max_queue = max(st.max_queue, len(st.queue))
        elif kind == SWAP_DONE:
            rider, st = a, self.stations[b]
            st.advance(t)
            finish = st.start_charge(t, rider.soc, self.minutes_full_charge)
            if finish is not None:
                self.push(finish, CHARGE_DONE, None, st.id)
            rider.soc = 1.0
            self.rider_soc[rider.id] = 1.0
            rider.n_swaps += 1
            rider.swap_min += t - rider.swap_started
            st.n_swaps += 1
            if rider.off_after_job:
                self._go_off(rider)
            else:
                self._set_idle(t, rider)
        elif kind == CHARGE_DONE:
            st = self.stations[b]
            st.advance(t)
            nxt = st.finish_charge(t)
            if nxt is not None:
                self.push(nxt, CHARGE_DONE, None, st.id)
            if st.queue and st.charged > 0:
                rid, t_arr = st.queue.popleft()
                self._start_swap(t, self.riders[rid], st, t_arr)
        elif kind == SHIFT_START:
            rider = a
            if rider.soc < self.swap_threshold:
                self._go_swap(t, rider)
            else:
                self._set_idle(t, rider)
        elif kind == SHIFT_END:
            rider = a
            if rider.state in (RiderState.IDLE, RiderState.OFF):
                self._go_off(rider)
            else:
                rider.off_after_job = True
        elif kind == SNAPSHOT:
            self._snapshot(t)

    def _snapshot(self, t):
        self.snapshots.append({
            "t": t,
            "pos": np.array([r.position(t) for r in self.riders]),
            "soc": np.array([r.soc for r in self.riders]),
            "state": [r.state.value for r in self.riders],
            "charged": np.array([s.charged for s in self.stations]),
            "charging": np.array([len(s.charging) + len(s.charge_backlog) for s in self.stations]),
            "queue": np.array([len(s.queue) for s in self.stations]),
        })

    # ------------------------------------------------------------------ run
    def run(self) -> "Simulation":
        cfg = self.cfg
        t0, t1 = cfg.demand.start_hour * 60, cfg.demand.end_hour * 60
        for st in self.stations:
            st._t_last = t0
            st.t_stats_end = t1
        for o in self.orders:
            self.push(o.t_created, ORDER, o)
        for r in self.riders:
            self.push(r.shift[0], SHIFT_START, r)
            self.push(r.shift[1], SHIFT_END, r)
        if cfg.snapshot_every_min:
            for ts in np.arange(t0, t1 + 1e-9, cfg.snapshot_every_min):
                self.push(float(ts), SNAPSHOT)
        # No new orders arrive after t1, so the queue drains: in-flight
        # deliveries, swaps and charging finish and the heap empties.
        while self._heap:
            t, _, kind, a, b = heapq.heappop(self._heap)
            self.now = t
            self._handle(t, kind, a, b)
        for st in self.stations:
            st.advance(t1)
        return self
