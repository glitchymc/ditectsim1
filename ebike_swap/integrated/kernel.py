"""Minimal discrete-event kernel: a time-ordered queue of callbacks."""
from __future__ import annotations

import heapq
import itertools


class Kernel:
    def __init__(self, t0: float = 0.0):
        self.now = t0
        self._q: list = []
        self._seq = itertools.count()
        self.n_events = 0

    def at(self, t: float, fn, *args):
        heapq.heappush(self._q, (max(t, self.now), next(self._seq), fn, args))

    def after(self, dt: float, fn, *args):
        self.at(self.now + dt, fn, *args)

    def run(self, until: float):
        while self._q and self._q[0][0] <= until:
            t, _, fn, args = heapq.heappop(self._q)
            self.now = t
            self.n_events += 1
            fn(*args)
        self.now = until
