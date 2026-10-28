"""Loaders for the public datasets named in the deck.

Each loader returns plain pandas objects in the simulator's local xy frame so
they can be dropped into a `City` (see `build_city`). Download the files
manually; this module does no network I/O.

* DOHMH NYC Restaurant Inspection Results (data.cityofnewyork.us, 43nn-pn8j)
* Kaggle "Food Ordering and Delivery App Dataset" (ahsan81)
* NYC neighbourhood ACS 2008-12 (GeoDa Center), as GeoJSON or CSV
"""
from __future__ import annotations

import json
from pathlib import Path as FsPath

import numpy as np
import pandas as pd
from matplotlib.path import Path

from ..geo import Region, lonlat_to_xy
from .city import City, _normalise

# DOHMH "CUISINE DESCRIPTION" -> Kaggle cuisine category. Unmapped => "Other".
DOHMH_CUISINE_MAP = {
    "American": "American", "Hamburgers": "American", "Chicken": "American",
    "Hotdogs": "American", "Sandwiches": "American", "Steakhouse": "American",
    "Barbecue": "American", "New American": "American", "Pizza": "Italian",
    "Italian": "Italian", "Japanese": "Japanese", "Sushi": "Japanese",
    "Chinese": "Chinese", "Chinese/Japanese": "Chinese", "Chinese/Cuban": "Chinese",
    "Mexican": "Mexican", "Tex-Mex": "Mexican", "Latin American": "Mexican",
    "Indian": "Indian", "Pakistani": "Indian", "Bangladeshi": "Indian",
    "Middle Eastern": "Middle Eastern", "Turkish": "Middle Eastern",
    "Mediterranean": "Mediterranean", "Greek": "Mediterranean",
    "Thai": "Thai", "French": "French", "Soul Food": "Southern",
    "Southern": "Southern", "Cajun": "Southern", "Creole": "Southern",
    "Korean": "Korean", "Spanish": "Spanish", "Tapas": "Spanish",
    "Vietnamese/Cambodian/Malaysia": "Vietnamese", "Vietnamese": "Vietnamese",
}


def load_dohmh_restaurants(csv_path: str | FsPath, boro: str | None = "Manhattan",
                           region: Region | None = None,
                           keep_unmapped: bool = False) -> pd.DataFrame:
    """One row per establishment (CAMIS) with cuisine mapped to Kaggle categories."""
    cols = ["CAMIS", "DBA", "BORO", "CUISINE DESCRIPTION", "Latitude", "Longitude"]
    df = pd.read_csv(csv_path, usecols=lambda c: c in cols, low_memory=False)
    df = df.dropna(subset=["Latitude", "Longitude"])
    df = df[(df["Latitude"] != 0) & (df["Longitude"] != 0)]
    if boro is not None:
        df = df[df["BORO"].str.casefold() == boro.casefold()]
    df = df.drop_duplicates("CAMIS")
    x, y = lonlat_to_xy(df["Longitude"], df["Latitude"])
    out = pd.DataFrame({
        "name": df["DBA"].astype(str).to_numpy(),
        "cuisine": df["CUISINE DESCRIPTION"].map(DOHMH_CUISINE_MAP).fillna("Other").to_numpy(),
        "x": x, "y": y,
    })
    if not keep_unmapped:
        out = out[out["cuisine"] != "Other"]
    if region is not None:
        out = out[region.contains(out[["x", "y"]].to_numpy())]
    return out.reset_index(drop=True)


def calibrate_from_kaggle(csv_path: str | FsPath) -> dict:
    """Cuisine priors by day type, weekday/weekend volume ratio and prep-time sample.

    The dataset only labels orders "Weekday"/"Weekend", so the per-day ratio is
    (weekend orders / 2) / (weekday orders / 5).
    """
    df = pd.read_csv(csv_path)
    prior = {}
    for label, key in (("Weekday", "weekday"), ("Weekend", "weekend")):
        sub = df[df["day_of_the_week"] == label]
        prior[key] = _normalise(sub["cuisine_type"].value_counts().to_dict())
    n_we = (df["day_of_the_week"] == "Weekend").sum()
    n_wd = (df["day_of_the_week"] == "Weekday").sum()
    ratio = (n_we / 2) / (n_wd / 5)
    dow = np.array([1, 1, 1, 1, 1, ratio, ratio], dtype=float)
    return {"cuisine_prior": prior, "dow_multiplier": dow / dow.mean(),
            "prep_times": df["food_preparation_time"].to_numpy(float),
            "weekend_per_day_ratio": ratio}


DEFAULT_ZONE_FIELDS = {
    "population": "poptot", "median_income": "medianinco",
    "young": ["pop18_24", "pop25_34"],   # summed => population aged 18-34
    "name": "ntaname",
}


def load_neighborhoods(path: str | FsPath, fields: dict | None = None,
                       young_cols: list[str] | None = None) -> tuple[pd.DataFrame, list[Path]]:
    """Zones from a neighbourhood GeoJSON (Polygon/MultiPolygon) or CSV with lat/lon.

    `fields` maps simulator columns to source property names, e.g.
    {"population": "poptot", "median_income": "medianinco", "jobs": "jobs",
     "name": "ntaname"}. `young_cols` are summed and divided by population to
    get `share_young`. Field names vary by release, so check your file.
    Missing `jobs` defaults to population (no workplace signal).
    """
    fields = fields or {k: v for k, v in DEFAULT_ZONE_FIELDS.items() if k != "young"}
    young_cols = young_cols if young_cols is not None else DEFAULT_ZONE_FIELDS["young"]
    path = FsPath(path)
    paths: list[Path] = []
    if path.suffix.lower() in (".geojson", ".json"):
        gj = json.loads(path.read_text())
        rows = []
        for feat in gj["features"]:
            props = feat["properties"]
            ring = _largest_ring(feat["geometry"])
            x, y = lonlat_to_xy(ring[:, 0], ring[:, 1])
            poly = np.column_stack([x, y])
            paths.append(Path(poly))
            cx, cy = _centroid(poly)
            rows.append({**props, "x": cx, "y": cy})
        src = pd.DataFrame(rows)
    else:
        src = pd.read_csv(path)
        src["x"], src["y"] = lonlat_to_xy(src["lon"], src["lat"])
        paths = [None] * len(src)
    pop = pd.to_numeric(src[fields["population"]], errors="coerce").fillna(0)
    zones = pd.DataFrame({
        "zone_id": np.arange(len(src)),
        "name": src[fields["name"]] if fields.get("name") in src else np.arange(len(src)),
        "x": src["x"], "y": src["y"],
        "population": pop,
        "jobs": pd.to_numeric(src[fields["jobs"]], errors="coerce").fillna(0)
        if fields.get("jobs") in src else pop,
        "median_income": pd.to_numeric(src[fields["median_income"]], errors="coerce"),
    })
    zones["median_income"] = zones["median_income"].fillna(zones["median_income"].median())
    present = [c for c in young_cols if c in src]
    young = src[present].apply(pd.to_numeric, errors="coerce").fillna(0).sum(axis=1) if present else 0.25 * pop
    zones["share_young"] = np.where(pop > 0, young / pop.where(pop > 0, 1), 0.0)
    keep = zones["population"] > 0
    return zones[keep].reset_index(drop=True), [p for p, k in zip(paths, keep) if k]


def build_city(base: City, restaurants: pd.DataFrame | None = None,
               zones: tuple[pd.DataFrame, list] | None = None,
               kaggle: dict | None = None) -> City:
    """Replace parts of `base` (typically the synthetic city) with real data."""
    prov = dict(base.provenance)
    kw = dict(region=base.region, zones=base.zones, restaurants=base.restaurants,
              terrain=base.terrain, zone_paths=base.zone_paths, zone_half_km=base.zone_half_km,
              cuisine_prior=base.cuisine_prior, hourly_profile=base.hourly_profile,
              dow_multiplier=base.dow_multiplier, office_share=base.office_share,
              prep_times=base.prep_times)
    if restaurants is not None:
        kw["restaurants"] = restaurants
        prov["restaurants"] = "DOHMH"
    if zones is not None:
        kw["zones"], kw["zone_paths"] = zones
        prov["zones"] = "ACS neighbourhoods"
    if kaggle is not None:
        kw["cuisine_prior"] = kaggle["cuisine_prior"]
        kw["dow_multiplier"] = kaggle["dow_multiplier"]
        kw["prep_times"] = kaggle["prep_times"]
        prov["cuisine_prior"] = prov["dow"] = prov["prep"] = "Kaggle"
    return City(**kw, provenance=prov)


def _largest_ring(geom: dict) -> np.ndarray:
    if geom["type"] == "Polygon":
        rings = [geom["coordinates"][0]]
    elif geom["type"] == "MultiPolygon":
        rings = [poly[0] for poly in geom["coordinates"]]
    else:
        raise ValueError(f"unsupported geometry {geom['type']}")
    rings = [np.asarray(r, float) for r in rings]
    return max(rings, key=lambda r: abs(_area(r)))


def _area(p: np.ndarray) -> float:
    x, y = p[:, 0], p[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _centroid(p: np.ndarray) -> tuple[float, float]:
    x, y = p[:, 0], p[:, 1]
    xn, yn = np.roll(x, -1), np.roll(y, -1)
    cross = x * yn - xn * y
    a = cross.sum() / 2
    if abs(a) < 1e-12:
        return float(x.mean()), float(y.mean())
    return float(((x + xn) * cross).sum() / (6 * a)), float(((y + yn) * cross).sum() / (6 * a))
