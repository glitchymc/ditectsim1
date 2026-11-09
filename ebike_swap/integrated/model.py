"""Integrated simulator: frontend (platforms, multi-app riders) wired to the
backend (swap stations, trucks, central charging station) on one event clock."""
from __future__ import annotations

import math
from collections import Counter

import numpy as np

from ..config import DemandParams, EnergyParams, TravelParams
from ..data.city import City
from ..demand import ODGenerator
from ..geo import GridRouter, lonlat_to_xy
from .backend import BSS, CCS, Fleet
from .battery import Battery
from .config import IntegratedConfig
from .frontend import Job, Platform, Rider, best_insertion
from .kernel import Kernel


class IntegratedSim:
    def __init__(self, city: City, cfg: IntegratedConfig, bss_xy: np.ndarray, seed: int = 0,
                 jobs_by_platform: list | None = None):
        self.city, self.cfg, self.seed = city, cfg, seed
        self.rng = np.random.default_rng(seed + 7_777)
        self.k = Kernel(cfg.start_hour * 60)
        self.energy = EnergyParams(battery_wh=cfg.battery.nominal_wh)
        self.travel = TravelParams()
        self.router = GridRouter(city.terrain, self.energy, self.travel)
        self._init_energy_bound()
        self.batteries: list[Battery] = []
        self.ccs_xy = np.array(lonlat_to_xy(*cfg.ccs.lonlat), float)

        bss_xy = np.asarray(bss_xy, float).reshape(-1, 2)
        self.stations = [BSS(i, xy.copy(), cfg.bss.slots) for i, xy in enumerate(bss_xy)]
        self.bss_xy = bss_xy
        for s in self.stations:
            for _ in range(min(cfg.bss.initial_charged, s.slots)):
                s.charged.append(self.new_battery(1.0, where="bss_charged"))
        self.ccs = CCS(self)
        for _ in range(cfg.ccs.initial_charged):
            self.ccs.loading.append(self.new_battery(1.0, where="ccs_loading"))
        self.fleet = Fleet(self)

        self.platforms = [Platform(self, i, sp) for i, sp in enumerate(cfg.platforms)]
        self._init_riders()
        self.jobs = jobs_by_platform if jobs_by_platform is not None else self.generate_jobs(city, cfg, self.router, seed)
        self.samples: list[dict] = []
        self.snaps: list[dict] = []

    # ------------------------------------------------------------- setup
    @staticmethod
    def generate_jobs(city: City, cfg: IntegratedConfig, router, seed: int) -> list[list[Job]]:
        """Each platform has its own OD generator (own volume, own random stream)."""
        out, gid = [], 0
        for k, sp in enumerate(cfg.platforms):
            dp = DemandParams(orders_per_day=sp.orders_per_day,
                              start_hour=cfg.start_hour, end_hour=cfg.end_hour)
            orders = ODGenerator(city, dp, router).generate(seed * 1000 + k)
            jobs = []
            for o in orders:
                od = float(router.distance(o.origin, o.dest))
                jobs.append(Job(gid, k, o, sp.pay_base + sp.pay_per_km * od, od,
                                o.t_created, o.t_ready))
                gid += 1
            out.append(jobs)
        return out

    def _init_energy_bound(self):
        e = self.energy
        probe = self.city.region.grid(0.25)
        z = self.city.terrain.elevation(probe[:, 0], probe[:, 1])
        climb = float(z.max() - z.min())
        self._wh_km = e.wh_per_km + e.lights_per_km * e.p_stop_at_light * e.wh_per_stop
        self._wh_climb = climb * e.total_mass_kg * 9.81 / e.climb_efficiency / 3600
        self._min_per_km = 60 / self.travel.cruise_kmh + e.lights_per_km * e.p_stop_at_light * self.travel.light_delay_min

    def energy_bound(self, km):
        return np.asarray(km) * self._wh_km + self._wh_climb

    def new_battery(self, soc: float, soh: float | None = None, where: str = "?") -> Battery:
        bp = self.cfg.battery
        b = Battery(len(self.batteries), bp.nominal_wh,
                    soh if soh is not None else float(self.rng.uniform(*bp.init_soh)), soc, where=where)
        self.batteries.append(b)
        return b

    def _init_riders(self):
        cfg, rng = self.cfg.riders, self.rng
        n, P = cfg.n_riders, len(self.platforms)
        rxy = self.city.restaurant_xy
        pos = rxy[rng.integers(len(rxy), size=n)] + rng.normal(0, 0.1, (n, 2))
        probs = np.asarray(cfg.apps_per_rider, float)[:P]
        probs = probs / probs.sum()
        share = np.array([p.orders_per_day for p in self.cfg.platforms], float)
        t0, t1 = self.cfg.start_hour * 60, self.cfg.end_hour * 60
        self.riders = []
        self.member = np.zeros((n, P), bool)
        for i in range(n):
            n_apps = int(rng.choice(len(probs), p=probs)) + 1
            apps = rng.choice(P, size=n_apps, replace=False, p=share / share.sum())
            self.member[i, apps] = True
            if cfg.shift_mode == "peaks":
                prof = np.roll(np.asarray(self.city.hourly_profile, float), -1)
                st = float(np.clip(rng.choice(24, p=prof / prof.sum()) * 60 + rng.uniform(0, 60), t0, t1))
                shift = (st, float(min(t1, st + rng.uniform(4, 9) * 60)))
            else:
                shift = (t0, t1)
            b = self.new_battery(float(rng.uniform(*cfg.init_soc)), where="rider")
            self.riders.append(Rider(i, pos[i].copy(), b, frozenset(int(a) for a in apps),
                                     float(rng.uniform(*cfg.wage_per_min)), shift))
        self.rider_pos = pos.copy()
        self.accepting = np.zeros(n, bool)
        self.pending = np.zeros((n, P), bool)
        self.njobs = np.zeros(n, int)

    def _sync(self, r: Rider):
        self.rider_pos[r.id] = r.pos
        self.njobs[r.id] = r.n_jobs
        self.accepting[r.id] = (r.state == "active" and not r.low and not r.off_after
                                and r.n_jobs < self.cfg.riders.max_jobs)

    # ------------------------------------------------------------ offers
    def send_offer(self, platform: Platform, job: Job, r: Rider):
        r.pending.add(platform.idx)
        self.pending[r.id, platform.idx] = True
        r.offers_seen += 1
        self.k.after(self.cfg.riders.decide_min, self._decide, platform, job, r)

    def _decide(self, platform: Platform, job: Job, r: Rider):
        r.pending.discard(platform.idx)
        self.pending[r.id, platform.idx] = False
        if not job.open:
            return
        accepted = False
        if self.accepting[r.id]:
            seq, added_km, total_km = best_insertion(self, r, job)
            last = seq[-1].xy
            reserve = self.energy_bound(total_km + float(self.router.distance(self.bss_xy, last).min()))
            if r.battery.wh >= reserve:
                value = job.pay - r.wage * added_km * self._min_per_km
                p_accept = 1.0 / (1.0 + math.exp(-value / self.cfg.riders.accept_noise))
                accepted = self.rng.uniform() < p_accept
        if accepted:
            job.rider = r.id
            job.t_accepted = self.k.now
            r.stops = seq
            r.offers_accepted += 1
            self._sync(r)
            if not r.head_locked:
                self._advance(r)
        platform.on_answer(job, r, accepted)

    # --------------------------------------------------------- rider legs
    def _move(self, r: Rider, dest, then, *args):
        rt = self.router.route(r.pos, dest)
        if not r.battery.drain(rt.wh):
            r.stranded += 1
        r.km += rt.km
        r.leg = (self.k.now, self.k.now + rt.minutes, r.pos.copy(), np.asarray(dest, float).copy())
        r.pos = np.asarray(dest, float).copy()
        self.rider_pos[r.id] = r.pos
        self.k.after(rt.minutes, then, r, *args)

    def _advance(self, r: Rider):
        if r.stops:
            r.head_locked = True
            self._move(r, r.stops[0].xy, self._arrive)
            return
        r.head_locked = False
        r.leg = None
        if r.off_after:
            r.state = "off"
        elif r.low or r.battery.soc < self.cfg.riders.swap_threshold:
            self._go_bss(r)
        else:
            r.state = "active"
        self._sync(r)

    def _arrive(self, r: Rider):
        r.leg = None
        s = r.stops[0]
        if s.kind == "P":
            wait = s.job.t_ready - self.k.now
            if wait > 0:
                self.k.after(wait, self._picked, r)
            else:
                self._picked(r)
            return
        job = s.job
        job.t_delivered = self.k.now
        r.earnings += job.pay
        r.delivered[job.platform] = r.delivered.get(job.platform, 0) + 1
        r.stops.pop(0)
        # flowchart: battery < threshold? no -> keep working; yes -> finish tasks, then swap
        if r.battery.soc < self.cfg.riders.swap_threshold:
            r.low = True
        self._sync(r)
        self._advance(r)

    def _picked(self, r: Rider):
        r.stops[0].job.t_picked = self.k.now
        r.stops.pop(0)
        self._advance(r)

    # -------------------------------------------------------------- swaps
    def _choose_bss(self, r: Rider) -> BSS:
        d = self.router.distance(self.bss_xy, r.pos)
        if self.cfg.riders.bss_choice == "avoid_empty":
            # prefer a station with a pack free, but only among those in range
            reach = self.energy_bound(d) <= r.battery.wh
            short = np.array([len(s.charged) - len(s.queue) <= 0 for s in self.stations])
            score = np.where(reach, d + 2.0 * short, np.inf)
            if np.isfinite(score).any():
                return self.stations[int(score.argmin())]
        return self.stations[int(d.argmin())]

    def _go_bss(self, r: Rider):
        s = self._choose_bss(r)
        r.state = "to_bss"
        r.t_swap_start = self.k.now
        self._move(r, s.xy, self._at_bss, s)

    def _at_bss(self, r: Rider, s: BSS):
        r.leg = None
        s.advance(self.k.now)
        if s.charged and not s.queue:
            self._swap(r, s, self.k.now)
        else:
            r.state = "queued"
            s.queue.append((r, self.k.now))
            s.max_queue = max(s.max_queue, len(s.queue))
            self.fleet.on_depleted(s)

    def serve_queue(self, s: BSS):
        s.advance(self.k.now)
        while s.queue and s.charged:
            r, t_arr = s.queue.popleft()
            self._swap(r, s, t_arr)

    def _swap(self, r: Rider, s: BSS, t_arr: float):
        wait = self.k.now - t_arr
        s.waits.append(wait)
        r.swap_wait_min += wait
        new = s.charged.popleft()
        new.where = "bss_swapping"
        s.in_swap += 1
        r.state = "swapping"
        self.k.after(self.cfg.bss.swap_min, self._swap_done, r, s, new)

    def _swap_done(self, r: Rider, s: BSS, new: Battery):
        s.advance(self.k.now)
        old = r.battery
        old.where = "bss_depleted"
        s.depleted.append(old)
        s.in_swap -= 1
        s.swaps += 1
        new.where = "rider"
        r.battery = new
        r.swaps += 1
        r.swap_trip_min += self.k.now - r.t_swap_start
        r.low = False
        r.state = "off" if r.off_after else "active"
        self._sync(r)
        self.fleet.on_depleted(s)

    # -------------------------------------------------------------- shifts
    def _shift_start(self, r: Rider):
        r.state = "active"
        r.off_after = False
        if r.battery.soc < self.cfg.riders.swap_threshold:
            self._go_bss(r)
        self._sync(r)

    def _shift_end(self, r: Rider):
        if r.state == "active" and not r.stops:
            r.state = "off"
        else:
            r.off_after = True
        self._sync(r)

    # ------------------------------------------------------------ sampling
    def _sample(self):
        t = self.k.now
        c = Counter(b.where for b in self.batteries)
        row = {"t": t, **c}
        for s in self.stations:
            s.advance(t)
        row["bss_queue"] = sum(len(s.queue) for s in self.stations)
        row["riders_active"] = int(sum(r.state != "off" for r in self.riders))
        row["riders_busy"] = int(sum(bool(r.stops) for r in self.riders))
        row["trucks_busy"] = int(sum(tr.busy for tr in self.fleet.trucks))
        self.samples.append(row)
        if self.cfg.snapshots:
            self.snaps.append({
                "t": t,
                "pos": np.array([r.position(t) for r in self.riders]),
                "soc": np.array([r.battery.soc for r in self.riders]),
                "state": [r.state for r in self.riders],
                "apps": [len(r.apps) for r in self.riders],
                "bss_charged": np.array([len(s.charged) for s in self.stations]),
                "bss_depleted": np.array([len(s.depleted) for s in self.stations]),
                "bss_queue": np.array([len(s.queue) for s in self.stations]),
                "trucks": np.array([tr.position(t) for tr in self.fleet.trucks]),
                "truck_load": np.array([len(tr.cargo_charged) + len(tr.cargo_depleted) for tr in self.fleet.trucks]),
                "ccs": {k: c.get(k, 0) for k in CCS_STAGES},
            })
        if t + self.cfg.sample_every_min <= self.horizon:
            self.k.after(self.cfg.sample_every_min, self._sample)

    # ----------------------------------------------------------------- run
    @property
    def horizon(self) -> float:
        return self.cfg.end_hour * 60 + self.cfg.drain_min

    def run(self) -> "IntegratedSim":
        k = self.k
        for s in self.stations:
            s._t = k.now
        for p, jobs in zip(self.platforms, self.jobs):
            for j in jobs:
                k.at(j.t_created, p.submit, j)
        for r in self.riders:
            k.at(r.shift[0], self._shift_start, r)
            k.at(r.shift[1], self._shift_end, r)
        self.ccs.start()
        self.fleet.start()
        k.at(k.now, self._sample)
        k.run(self.horizon)
        for s in self.stations:
            s.advance(self.horizon)
        for tr in self.fleet.trucks:
            if tr.busy:
                tr.busy_min += self.horizon - tr._t_busy
        return self


CCS_STAGES = ["ccs_unloading", "ccs_testing", "ccs_tested", "ccs_van", "ccs_hall_queue",
              "ccs_charging", "ccs_hall_ready", "ccs_loading"]
