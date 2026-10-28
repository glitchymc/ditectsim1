"""Rider and station agents (the deck's "naive" models).

The agents hold state and bookkeeping only; the event engine drives them.
"""
from __future__ import annotations

import heapq
from collections import deque
from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class RiderState(str, Enum):
    OFF = "off"
    IDLE = "idle"
    TO_PICKUP = "to_pickup"
    WAIT_FOOD = "wait_food"
    TO_DROPOFF = "to_dropoff"
    TO_STATION = "to_station"
    QUEUED = "queued"
    SWAPPING = "swapping"


@dataclass
class Rider:
    id: int
    pos: np.ndarray
    soc: float
    shift: tuple[float, float]
    state: RiderState = RiderState.OFF
    order: object | None = None
    # current leg for position interpolation: (t0, t1, p0, p1)
    leg: tuple[float, float, np.ndarray, np.ndarray] | None = None
    # bookkeeping
    km: float = 0.0
    wh: float = 0.0
    n_orders: int = 0
    n_swaps: int = 0
    swap_min: float = 0.0          # detour + queue + service time spent on swaps
    swap_started: float = 0.0
    stranded: int = 0              # times the battery would have gone below zero
    off_after_job: bool = False

    @property
    def available(self) -> bool:
        return self.state is RiderState.IDLE

    def position(self, t: float) -> np.ndarray:
        if self.leg is None:
            return self.pos
        t0, t1, p0, p1 = self.leg
        if t1 <= t0 or t >= t1:
            return p1
        if t <= t0:
            return p0
        return p0 + (p1 - p0) * (t - t0) / (t1 - t0)


@dataclass
class Station:
    """Swap cabinet with `n_batteries` packs and `n_chargers` charging slots.

    Charged packs are handed out first-come-first-served. Riders queue when
    none are charged. A returned pack charges at full power for a time set by
    its state of charge; if every charger is busy it waits for one.
    """

    id: int
    xy: np.ndarray
    n_batteries: int
    n_chargers: int
    charged: int = 0
    charging: list = field(default_factory=list)       # heap of finish times
    charge_backlog: deque = field(default_factory=deque)   # socs waiting for a charger
    queue: deque = field(default_factory=deque)        # (rider_id, t_arrive)
    # time-weighted statistics
    _t_last: float = 0.0
    area_queue: float = 0.0
    area_charged: float = 0.0
    stockout_min: float = 0.0
    n_swaps: int = 0
    n_arrivals: int = 0
    waits: list = field(default_factory=list)
    max_queue: int = 0
    trigger_points: list = field(default_factory=list)  # where riders decided to come here

    def __post_init__(self):
        self.charged = self.n_batteries

    t_stats_end: float = float("inf")   # time-weighted stats stop accruing here

    def advance(self, t: float):
        t = min(t, self.t_stats_end)
        dt = t - self._t_last
        if dt > 0:
            self.area_queue += dt * len(self.queue)
            self.area_charged += dt * self.charged
            if self.charged == 0:
                self.stockout_min += dt
            self._t_last = t

    def estimated_wait(self, t_arrive: float) -> float:
        """Wait a rider arriving at `t_arrive` would face, from current state.

        Assumes no other arrivals in between; ignores packs in the charger backlog.
        """
        ahead = len(self.queue)
        if self.charged > ahead:
            return 0.0
        need = ahead - self.charged
        finishes = sorted(self.charging)
        if need < len(finishes):
            return max(0.0, finishes[need] - t_arrive)
        return 240.0   # effectively unavailable

    def start_charge(self, t: float, soc: float, minutes_full: float) -> float | None:
        """Accept a depleted pack. Returns its finish time if a charger is free."""
        dur = max(0.0, 1.0 - soc) * minutes_full
        if len(self.charging) < self.n_chargers:
            finish = t + dur
            heapq.heappush(self.charging, finish)
            return finish
        self.charge_backlog.append(dur)
        return None

    def finish_charge(self, t: float) -> float | None:
        """A pack finished charging; returns finish time of the next backlog pack, if any."""
        heapq.heappop(self.charging)
        self.charged += 1
        if self.charge_backlog:
            finish = t + self.charge_backlog.popleft()
            heapq.heappush(self.charging, finish)
            return finish
        return None
