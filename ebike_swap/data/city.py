"""City inputs consumed by the simulator.

A `City` bundles everything the demand and routing modules need. It can be
built from public datasets (see `loaders.py`) or from the synthetic Manhattan
stand-in (see `synthetic.py`). Mixing is allowed: e.g. real restaurants with
synthetic zones.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from matplotlib.path import Path

from ..geo import FLAT, Region, Terrain

# Cuisine order counts in the Kaggle "Food Ordering and Delivery App" dataset
# (1,898 NYC orders). Used as the default cuisine prior.
KAGGLE_CUISINE_COUNTS = {
    "American": 584, "Japanese": 470, "Italian": 298, "Chinese": 215,
    "Mexican": 77, "Indian": 73, "Middle Eastern": 49, "Mediterranean": 46,
    "Thai": 19, "French": 18, "Southern": 17, "Korean": 13, "Spanish": 12,
    "Vietnamese": 7,
}

# Relative order intensity by hour of day (lunch and dinner peaks).
DEFAULT_HOURLY_PROFILE = np.array([
    0.9, 0.6, 0.4, 0.2, 0.1, 0.1, 0.2, 0.5, 0.9, 1.1, 1.5, 3.0,
    4.2, 3.5, 2.0, 1.5, 1.8, 3.0, 4.6, 5.0, 4.0, 2.8, 1.9, 1.3,
])

# Relative daily volume, Mon..Sun. Moderate weekend uplift by default; call
# `loaders.calibrate_from_kaggle` to use the dataset's (much stronger) skew.
DEFAULT_DOW_MULTIPLIER = np.array([0.9, 0.9, 0.95, 1.0, 1.15, 1.2, 1.1])

# Share of destination weight that follows workplaces rather than homes,
# by hour, on weekdays (weekends use 30% of this).
DEFAULT_OFFICE_SHARE = np.array([
    0, 0, 0, 0, 0, 0, 0, 0.1, 0.3, 0.5, 0.6, 0.75,
    0.8, 0.75, 0.6, 0.5, 0.4, 0.25, 0.1, 0.05, 0, 0, 0, 0,
])


@dataclass
class City:
    region: Region
    zones: pd.DataFrame          # zone_id, x, y, population, jobs, median_income, share_young
    restaurants: pd.DataFrame    # name, cuisine, x, y
    terrain: Terrain = FLAT
    zone_paths: list[Path | None] | None = None   # polygons for destination sampling
    zone_half_km: float = 0.4    # used when a zone has no polygon
    cuisine_prior: dict[str, dict[str, float]] = field(default_factory=dict)  # day_type -> cuisine -> p
    hourly_profile: np.ndarray = field(default_factory=lambda: DEFAULT_HOURLY_PROFILE.copy())
    dow_multiplier: np.ndarray = field(default_factory=lambda: DEFAULT_DOW_MULTIPLIER.copy())
    office_share: np.ndarray = field(default_factory=lambda: DEFAULT_OFFICE_SHARE.copy())
    prep_times: np.ndarray | None = None   # empirical prep-time sample (min); None => uniform range
    provenance: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if not self.cuisine_prior:
            p = _normalise(KAGGLE_CUISINE_COUNTS)
            self.cuisine_prior = {"weekday": p, "weekend": dict(p)}
        for col in ("population", "jobs", "median_income", "share_young"):
            if col not in self.zones:
                raise ValueError(f"zones table missing column {col!r}")
        self.zones = self.zones.reset_index(drop=True)
        self.restaurants = self.restaurants.reset_index(drop=True)
        self._zone_xy = self.zones[["x", "y"]].to_numpy(float)
        self._restaurant_xy = self.restaurants[["x", "y"]].to_numpy(float)

    @property
    def restaurant_xy(self) -> np.ndarray:
        return self._restaurant_xy

    @property
    def zone_xy(self) -> np.ndarray:
        return self._zone_xy

    def zone_of(self, pts: np.ndarray) -> np.ndarray:
        """Nearest-centroid zone index for each point."""
        pts = np.atleast_2d(pts)
        d = ((pts[:, None, :] - self.zone_xy[None, :, :]) ** 2).sum(-1)
        return d.argmin(1)

    def sample_in_zone(self, z: int, rng: np.random.Generator) -> np.ndarray:
        path = self.zone_paths[z] if self.zone_paths else None
        cx, cy = self._zone_xy[z]
        for _ in range(50):
            if path is not None:
                (x0, y0), (x1, y1) = path.vertices.min(0), path.vertices.max(0)
                p = rng.uniform([x0, y0], [x1, y1])
                ok = path.contains_point(p)
            else:
                p = np.array([cx, cy]) + rng.uniform(-self.zone_half_km, self.zone_half_km, 2)
                ok = True
            if ok and self.region.contains(p)[0]:
                return p
        return np.array([cx, cy])


def _normalise(d: dict[str, float]) -> dict[str, float]:
    s = float(sum(d.values()))
    return {k: v / s for k, v in d.items()}
