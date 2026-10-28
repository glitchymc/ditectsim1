"""Shared CLI plumbing: build a City from real files where given, synthetic otherwise."""
from __future__ import annotations

import argparse

from ebike_swap.data import (build_city, calibrate_from_kaggle, load_dohmh_restaurants,
                             load_neighborhoods, synthetic_manhattan)


def add_city_args(p: argparse.ArgumentParser):
    g = p.add_argument_group("data (synthetic stand-ins are used for anything not given)")
    g.add_argument("--restaurants", help="DOHMH Restaurant Inspection Results CSV")
    g.add_argument("--neighborhoods", help="NYC neighbourhood ACS GeoJSON or CSV")
    g.add_argument("--kaggle", help="Kaggle food_order.csv")
    g.add_argument("--city-seed", type=int, default=0)


def city_from_args(a):
    city = synthetic_manhattan(a.city_seed)
    rest = load_dohmh_restaurants(a.restaurants, region=city.region) if a.restaurants else None
    zones = load_neighborhoods(a.neighborhoods) if a.neighborhoods else None
    kag = calibrate_from_kaggle(a.kaggle) if a.kaggle else None
    if rest is not None or zones is not None or kag is not None:
        city = build_city(city, restaurants=rest, zones=zones, kaggle=kag)
    print("data provenance:", city.provenance)
    return city
