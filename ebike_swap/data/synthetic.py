"""Synthetic Manhattan stand-in.

Produces a `City` with plausible *shapes* (dense Midtown/Downtown restaurants,
residential Upper East/West Side, hilly Washington Heights) so the simulator
and optimiser can be exercised without network access. Nothing here is
calibrated; swap in `loaders.py` outputs for any substantive study.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ..geo import MANHATTAN_GRID_DEG, FunctionTerrain, Region
from .city import KAGGLE_CUISINE_COUNTS, City, _normalise

_TH = math.radians(MANHATTAN_GRID_DEG)
_U = np.array([math.sin(_TH), math.cos(_TH)])   # along the island (uptown)
_V = np.array([math.cos(_TH), -math.sin(_TH)])  # across the island (east)


def along_island(x, y):
    """Distance (km) uptown from the Battery along the avenue axis."""
    return (np.asarray(x) - 0.3) * _U[0] + (np.asarray(y) - 0.1) * _U[1]


def _bump(s, centre, width):
    return np.exp(-0.5 * ((s - centre) / width) ** 2)


def _elevation(x, y):
    """Low downtown, gentle Midtown rise, Morningside and Washington Heights ridges (m)."""
    s = along_island(x, y)
    return 4 + 18 * _bump(s, 7.5, 3.0) + 30 * _bump(s, 13.5, 2.0) + 60 * _bump(s, 17.5, 1.8)


# (along-island km, relative density, width km) of restaurant clusters
_RESTAURANT_CLUSTERS = [
    (1.0, 1.0, 0.5), (2.6, 1.6, 0.6), (3.8, 1.4, 0.6), (5.2, 1.2, 0.6),
    (6.4, 2.2, 0.8), (8.0, 1.0, 0.7), (9.8, 0.9, 0.8), (12.5, 0.7, 0.9),
    (15.0, 0.4, 0.9), (18.0, 0.35, 1.0),
]
# Ethnic enclaves that skew cuisine mix: (along-island km, cuisine, weight)
_ENCLAVES = [(1.9, "Chinese", 4.0), (5.9, "Korean", 6.0), (4.6, "Indian", 2.0),
             (12.8, "Southern", 3.0), (17.5, "Spanish", 4.0), (2.4, "Italian", 1.5)]


def synthetic_manhattan(seed: int = 0, n_restaurants: int = 6000,
                        zone_km: float = 0.8) -> City:
    rng = np.random.default_rng(seed)
    region = Region.manhattan()

    # Zones: square cells clipped to the island.
    zxy = region.grid(zone_km)
    s = along_island(zxy[:, 0], zxy[:, 1])
    resid = 0.4 + 1.2 * _bump(s, 10.5, 2.5) + 0.9 * _bump(s, 15.5, 2.5) + 0.6 * _bump(s, 3.0, 1.5)
    jobs = 0.1 + 3.0 * _bump(s, 6.5, 1.6) + 2.0 * _bump(s, 0.9, 0.8)
    income = 45_000 + 110_000 * (_bump(s, 9.5, 3.5) + 0.5 * _bump(s, 1.5, 1.5)) * (1 - 0.6 * (s > 13))
    young = 0.18 + 0.14 * _bump(s, 3.0, 1.5) + 0.08 * _bump(s, 13.5, 2.0)
    noise = rng.lognormal(0, 0.15, len(zxy))
    zones = pd.DataFrame({
        "zone_id": np.arange(len(zxy)),
        "x": zxy[:, 0], "y": zxy[:, 1],
        "population": (resid * noise * 8000).round(),
        "jobs": (jobs * rng.lognormal(0, 0.2, len(zxy)) * 8000).round(),
        "median_income": income.round(-2),
        "share_young": np.clip(young * rng.lognormal(0, 0.1, len(zxy)), 0.05, 0.6),
    })

    # Restaurants: mixture of clusters along the island, rejected outside it.
    centres = np.array([c[0] for c in _RESTAURANT_CLUSTERS])
    wts = np.array([c[1] for c in _RESTAURANT_CLUSTERS])
    widths = np.array([c[2] for c in _RESTAURANT_CLUSTERS])
    pts = []
    while sum(len(p) for p in pts) < n_restaurants:
        cand = region.sample_uniform(n_restaurants, rng)
        k = rng.choice(len(centres), size=len(cand), p=wts / wts.sum())
        sc = along_island(cand[:, 0], cand[:, 1])
        keep = rng.uniform(size=len(cand)) < _bump(sc, centres[k], widths[k])
        pts.append(cand[keep])
    rxy = np.concatenate(pts)[:n_restaurants]

    prior = _normalise(KAGGLE_CUISINE_COUNTS)
    names = list(prior)
    base = np.array([prior[c] for c in names])
    rs = along_island(rxy[:, 0], rxy[:, 1])
    cuisines = []
    for si in rs:
        w = base.copy()
        for centre, cu, boost in _ENCLAVES:
            w[names.index(cu)] *= 1 + boost * _bump(si, centre, 0.4)
        cuisines.append(names[rng.choice(len(names), p=w / w.sum())])
    restaurants = pd.DataFrame({
        "name": [f"R{i:05d}" for i in range(len(rxy))],
        "cuisine": cuisines, "x": rxy[:, 0], "y": rxy[:, 1],
    })

    return City(
        region=region, zones=zones, restaurants=restaurants,
        terrain=FunctionTerrain(_elevation), zone_half_km=zone_km / 2,
        provenance={"zones": "synthetic", "restaurants": "synthetic",
                    "terrain": "synthetic", "cuisine_prior": "Kaggle counts (deck)"},
    )
