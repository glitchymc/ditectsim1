"""Geometry, routing and terrain.

Coordinates are a local planar frame in km (x east, y north) anchored near the
southern tip of Manhattan. The default router measures L1 distance along the
Manhattan street grid (rotated ~29 degrees from true north). Any object with the
same `route()` signature can replace it, e.g. a router over an OSM street graph.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np
from matplotlib.path import Path

from .config import EnergyParams, TravelParams

ORIGIN_LAT = 40.70
ORIGIN_LON = -74.02
KM_PER_DEG_LAT = 110.95
KM_PER_DEG_LON = 111.32 * math.cos(math.radians(40.75))
MANHATTAN_GRID_DEG = 29.0


def lonlat_to_xy(lon, lat):
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    return (lon - ORIGIN_LON) * KM_PER_DEG_LON, (lat - ORIGIN_LAT) * KM_PER_DEG_LAT


def xy_to_lonlat(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return ORIGIN_LON + x / KM_PER_DEG_LON, ORIGIN_LAT + y / KM_PER_DEG_LAT


# Coarse Manhattan shoreline (lon, lat), counter-clockwise from the Battery.
MANHATTAN_OUTLINE_LONLAT = [
    (-74.0160, 40.7010), (-74.0190, 40.7110), (-74.0130, 40.7300),
    (-74.0100, 40.7480), (-74.0000, 40.7620), (-73.9900, 40.7760),
    (-73.9760, 40.7980), (-73.9600, 40.8200), (-73.9480, 40.8420),
    (-73.9380, 40.8600), (-73.9280, 40.8750), (-73.9160, 40.8780),
    (-73.9090, 40.8730), (-73.9170, 40.8590), (-73.9290, 40.8440),
    (-73.9340, 40.8270), (-73.9330, 40.8100), (-73.9280, 40.8000),
    (-73.9360, 40.7880), (-73.9420, 40.7760), (-73.9560, 40.7600),
    (-73.9680, 40.7450), (-73.9720, 40.7300), (-73.9740, 40.7180),
    (-73.9800, 40.7110), (-73.9960, 40.7070), (-74.0080, 40.7020),
]


class Region:
    """Service area polygon in the local xy frame."""

    def __init__(self, polygon_xy: np.ndarray):
        self.polygon = np.asarray(polygon_xy, dtype=float)
        self._path = Path(self.polygon)

    @classmethod
    def manhattan(cls) -> "Region":
        lon, lat = zip(*MANHATTAN_OUTLINE_LONLAT)
        x, y = lonlat_to_xy(lon, lat)
        return cls(np.column_stack([x, y]))

    @property
    def bounds(self):
        (x0, y0), (x1, y1) = self.polygon.min(0), self.polygon.max(0)
        return x0, y0, x1, y1

    def contains(self, pts: np.ndarray) -> np.ndarray:
        return self._path.contains_points(np.atleast_2d(pts))

    def sample_uniform(self, n: int, rng: np.random.Generator) -> np.ndarray:
        x0, y0, x1, y1 = self.bounds
        out = []
        while sum(len(o) for o in out) < n:
            cand = rng.uniform([x0, y0], [x1, y1], size=(max(2 * n, 64), 2))
            out.append(cand[self.contains(cand)])
        return np.concatenate(out)[:n]

    def grid(self, spacing_km: float) -> np.ndarray:
        x0, y0, x1, y1 = self.bounds
        xs = np.arange(x0 + spacing_km / 2, x1, spacing_km)
        ys = np.arange(y0 + spacing_km / 2, y1, spacing_km)
        pts = np.array([(x, y) for y in ys for x in xs])
        return pts[self.contains(pts)]


class Terrain(Protocol):
    def elevation(self, x: np.ndarray, y: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class FunctionTerrain:
    """Elevation (m) from an arbitrary vectorised function of (x, y)."""

    fn: Callable[[np.ndarray, np.ndarray], np.ndarray]

    def elevation(self, x, y):
        return self.fn(np.asarray(x, float), np.asarray(y, float))


@dataclass
class GridTerrain:
    """Elevation raster (m) on a regular xy grid, bilinear interpolation."""

    x0: float
    y0: float
    cell_km: float
    z: np.ndarray   # shape (ny, nx)

    def elevation(self, x, y):
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        ny, nx = self.z.shape
        fx = np.clip((x - self.x0) / self.cell_km, 0, nx - 1.0001)
        fy = np.clip((y - self.y0) / self.cell_km, 0, ny - 1.0001)
        i, j = fy.astype(int), fx.astype(int)
        dy, dx = fy - i, fx - j
        z = self.z
        return ((1 - dy) * ((1 - dx) * z[i, j] + dx * z[i, j + 1])
                + dy * ((1 - dx) * z[i + 1, j] + dx * z[i + 1, j + 1]))


FLAT = FunctionTerrain(lambda x, y: np.zeros_like(x))


@dataclass(frozen=True)
class Route:
    km: float
    minutes: float
    wh: float


class Router(Protocol):
    def route(self, a: tuple[float, float], b: tuple[float, float]) -> Route: ...
    def distance(self, a: np.ndarray, b: np.ndarray) -> np.ndarray: ...


_FRACTIONS: dict[int, np.ndarray] = {}


def _fractions(n: int) -> np.ndarray:
    f = _FRACTIONS.get(n)
    if f is None:
        f = _FRACTIONS[n] = np.linspace(0.0, 1.0, n)[:, None]
    return f


class GridRouter:
    """Rotated-L1 street-grid router with elevation and signal-aware energy.

    The path runs along the avenue axis first, then the street axis. Climb is
    integrated by sampling the terrain along that L-shaped path.
    """

    def __init__(self, terrain: Terrain, energy: EnergyParams, travel: TravelParams,
                 grid_deg: float = MANHATTAN_GRID_DEG, samples_per_km: float = 4.0):
        self.terrain = terrain
        self.e = energy
        self.t = travel
        th = math.radians(grid_deg)
        # unit vectors of the avenue (u) and street (v) directions in xy
        self._u = np.array([math.sin(th), math.cos(th)])
        self._v = np.array([math.cos(th), -math.sin(th)])
        self.samples_per_km = samples_per_km
        g = 9.81
        self._wh_per_m_climb = energy.total_mass_kg * g / energy.climb_efficiency / 3600.0
        self._wh_per_m_descent = energy.total_mass_kg * g * energy.regen_fraction / 3600.0

    def distance(self, a, b):
        """Vectorised network distance (km); a, b broadcast over (..., 2)."""
        d = np.asarray(b, float) - np.asarray(a, float)
        return (np.abs(d @ self._u) + np.abs(d @ self._v)) * self.t.detour_factor

    def route(self, a, b) -> Route:
        a = np.asarray(a, float)
        b = np.asarray(b, float)
        d = b - a
        du = float(d[0] * self._u[0] + d[1] * self._u[1])
        dv = float(d[0] * self._v[0] + d[1] * self._v[1])
        km = (abs(du) + abs(dv)) * self.t.detour_factor
        if km == 0.0:
            return Route(0.0, 0.0, 0.0)
        corner = a + du * self._u
        n1 = min(int(abs(du) * self.samples_per_km) + 2, 64)
        n2 = min(int(abs(dv) * self.samples_per_km) + 2, 64)
        f1, f2 = _fractions(n1), _fractions(n2)[1:]
        pts = np.concatenate([a + f1 * (corner - a), corner + f2 * (b - corner)])
        z = self.terrain.elevation(pts[:, 0], pts[:, 1])
        dz = np.diff(z)
        climb = dz[dz > 0].sum()
        descent = -dz[dz < 0].sum()
        e = self.e
        stops = km * e.lights_per_km * e.p_stop_at_light
        wh = (km * e.wh_per_km + climb * self._wh_per_m_climb
              - descent * self._wh_per_m_descent + stops * e.wh_per_stop)
        minutes = km / self.t.cruise_kmh * 60.0 + stops * self.t.light_delay_min
        return Route(km, minutes, max(wh, 0.0))

    def wh_upper_bound(self, km: float, max_climb_m: float) -> float:
        """Conservative energy bound for a trip of `km` with at most `max_climb_m` climb."""
        e = self.e
        return (km * e.wh_per_km + max_climb_m * self._wh_per_m_climb
                + km * e.lights_per_km * e.p_stop_at_light * e.wh_per_stop)
