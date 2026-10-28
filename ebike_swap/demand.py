"""OD generator: time-dependent arrivals -> demographic destination -> restaurant.

Each stage is a small class with one method, so any stage can be replaced by
a user-supplied object with the same signature (e.g. a destination sampler fit
to platform data) without touching the rest of the simulator.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .config import DemandParams
from .data.city import City
from .geo import Router


@dataclass
class Order:
    id: int
    t_created: float          # minutes since midnight
    restaurant: int
    origin: np.ndarray        # restaurant xy
    dest: np.ndarray
    dest_zone: int
    cuisine: str
    prep_min: float

    @property
    def t_ready(self) -> float:
        return self.t_created + self.prep_min


def day_type(dow: int) -> str:
    return "weekend" if dow >= 5 else "weekday"


class ArrivalProcess(Protocol):
    def sample(self, rng: np.random.Generator) -> np.ndarray: ...


class DestinationSampler(Protocol):
    def sample(self, t: float, rng: np.random.Generator) -> tuple[int, np.ndarray]: ...


class RestaurantSampler(Protocol):
    def sample(self, dest: np.ndarray, t: float, rng: np.random.Generator) -> tuple[int, str]: ...


class TimeDependentGenerator:
    """Piecewise-constant (hourly) non-homogeneous Poisson arrivals."""

    def __init__(self, city: City, p: DemandParams):
        self.p = p
        prof = np.asarray(city.hourly_profile, float)
        mult = float(city.dow_multiplier[p.day_of_week])
        self.rate_per_hour = p.orders_per_day * mult * prof / prof.sum()

    def sample(self, rng):
        times = []
        h0, h1 = self.p.start_hour, self.p.end_hour
        for h in range(int(np.floor(h0)), int(np.ceil(h1))):
            lo, hi = max(h, h0), min(h + 1, h1)
            if hi <= lo:
                continue
            n = rng.poisson(self.rate_per_hour[h % 24] * (hi - lo))
            times.append(rng.uniform(lo, hi, n) * 60.0)
        return np.sort(np.concatenate(times)) if times else np.empty(0)


class DemographicDestinationSampler:
    """Zone weight = activity(t) * exp(beta_young * dyoung) * (income/median)^beta_income.

    activity(t) blends residents and workers by the hourly office share, so
    lunch demand leans to office districts and dinner demand to homes. The
    young-adult and income effects follow the studies cited in the deck.
    """

    def __init__(self, city: City, p: DemandParams):
        self.city = city
        z = city.zones
        young = z["share_young"].to_numpy(float)
        inc = z["median_income"].to_numpy(float)
        self.affinity = (np.exp(p.beta_young * (young - young.mean()))
                         * (inc / np.median(inc)) ** p.beta_income)
        self.pop = z["population"].to_numpy(float)
        self.jobs = z["jobs"].to_numpy(float)
        o = np.asarray(city.office_share, float)
        self.office = o if p.day_of_week < 5 else 0.3 * o
        self._cache: dict[int, np.ndarray] = {}

    def weights(self, hour: int) -> np.ndarray:
        if hour not in self._cache:
            o = self.office[hour % 24]
            w = ((1 - o) * self.pop + o * self.jobs) * self.affinity
            self._cache[hour] = w / w.sum()
        return self._cache[hour]

    def sample(self, t, rng):
        zi = int(rng.choice(len(self.pop), p=self.weights(int(t // 60))))
        return zi, self.city.sample_in_zone(zi, rng)


class RestaurantTypeSampler:
    """Cuisine from the day-type prior, then a nearby restaurant of that cuisine.

    Among restaurants of the chosen cuisine within `max_od_km` (network
    distance) the choice probability decays as exp(-d / od_decay_km). If none
    serve that cuisine nearby, the cuisine is redrawn among those that do.
    """

    def __init__(self, city: City, p: DemandParams, router: Router):
        self.p = p
        self.router = router
        self.xy = city.restaurant_xy
        self.cuisine = city.restaurants["cuisine"].to_numpy()
        self.prior = city.cuisine_prior[day_type(p.day_of_week)]

    def sample(self, dest, t, rng):
        d = self.router.distance(self.xy, dest)
        near = d <= self.p.max_od_km
        if not near.any():
            i = int(d.argmin())
            return i, str(self.cuisine[i])
        cand_cuisines = np.unique(self.cuisine[near])
        pw = np.array([self.prior.get(c, 0.0) for c in cand_cuisines])
        if pw.sum() == 0:
            pw = np.ones(len(cand_cuisines))
        c = cand_cuisines[rng.choice(len(cand_cuisines), p=pw / pw.sum())]
        idx = np.flatnonzero(near & (self.cuisine == c))
        w = np.exp(-d[idx] / self.p.od_decay_km)
        i = int(idx[rng.choice(len(idx), p=w / w.sum())])
        return i, str(c)


class ODGenerator:
    def __init__(self, city: City, p: DemandParams, router: Router,
                 arrivals: ArrivalProcess | None = None,
                 destinations: DestinationSampler | None = None,
                 restaurants: RestaurantSampler | None = None):
        self.city = city
        self.p = p
        self.arrivals = arrivals or TimeDependentGenerator(city, p)
        self.destinations = destinations or DemographicDestinationSampler(city, p)
        self.restaurants = restaurants or RestaurantTypeSampler(city, p, router)

    def _prep(self, rng):
        if self.city.prep_times is not None and len(self.city.prep_times):
            return float(rng.choice(self.city.prep_times))
        return float(rng.uniform(*self.p.prep_time_range))

    def generate(self, seed: int) -> list[Order]:
        rng = np.random.default_rng(seed)
        rxy = self.city.restaurant_xy
        orders = []
        for k, t in enumerate(self.arrivals.sample(rng)):
            z, dest = self.destinations.sample(t, rng)
            r, cu = self.restaurants.sample(dest, t, rng)
            orders.append(Order(k, float(t), r, rxy[r].copy(), dest, z, cu, self._prep(rng)))
        return orders
