import json

import numpy as np
import pandas as pd
import pytest

from ebike_swap import ODGenerator, SimConfig, Simulation, StationLayout, summarize
from ebike_swap.agents import RiderState, Station
from ebike_swap.data import (build_city, calibrate_from_kaggle, load_dohmh_restaurants,
                             load_neighborhoods, synthetic_manhattan)
from ebike_swap.dispatch import LeastExpectedTime
from ebike_swap.geo import FLAT, FunctionTerrain, GridRouter, lonlat_to_xy
from ebike_swap.optimize import (Evaluator, IterativeRebalancer, candidate_sites, split_budget,
                                 uniform_design)


@pytest.fixture(scope="module")
def city():
    return synthetic_manhattan(0, n_restaurants=600)


@pytest.fixture(scope="module")
def cfg():
    c = SimConfig()
    c.rider.n_riders = 40
    c.demand.orders_per_day = 900
    return c


@pytest.fixture(scope="module")
def orders(city, cfg):
    return ODGenerator(city, cfg.demand, GridRouter(city.terrain, cfg.energy, cfg.travel)).generate(3)


def small_layout(city, k=5, b=6):
    d = uniform_design(city, candidate_sites(city), k, k * b)
    return d.layout(candidate_sites(city))


def test_router_basics(cfg):
    r = GridRouter(FLAT, cfg.energy, cfg.travel)
    a, b = np.array([0.0, 0.0]), np.array([1.0, 2.0])
    assert r.route(a, a).km == 0
    assert r.route(a, b).km == pytest.approx(r.route(b, a).km)
    assert r.route(a, b).km == pytest.approx(float(r.distance(a, b)))
    hill = GridRouter(FunctionTerrain(lambda x, y: 20 * y), cfg.energy, cfg.travel)
    assert hill.route(a, b).wh > r.route(a, b).wh            # climbing costs energy
    assert hill.route(b, a).wh == pytest.approx(r.route(b, a).wh)  # no regen by default


def test_station_fifo_and_charging():
    s = Station(0, np.zeros(2), n_batteries=2, n_chargers=1)
    s.charged = 0
    assert s.start_charge(0.0, 0.5, 120.0) == pytest.approx(60.0)
    assert s.start_charge(0.0, 0.0, 120.0) is None      # one charger busy -> backlog
    assert s.estimated_wait(10.0) == pytest.approx(50.0)
    nxt = s.finish_charge(60.0)
    assert s.charged == 1 and nxt == pytest.approx(180.0)


def test_day_conserves_orders_and_batteries(city, cfg, orders):
    lay = small_layout(city)
    sim = Simulation(city, cfg, lay, orders, seed=3).run()
    delivered = ~np.isnan(sim.t_delivered)
    assert (delivered | sim.cancelled).all()
    assert not (delivered & sim.cancelled).any()
    for st in sim.stations:
        assert st.charged + len(st.charging) + len(st.charge_backlog) == st.n_batteries
        assert not st.queue
    assert sum(r.stranded for r in sim.riders) == 0
    assert all(r.state in (RiderState.IDLE, RiderState.OFF) for r in sim.riders)
    s = summarize(sim)
    assert s["swaps"] == sum(st.n_swaps for st in sim.stations) > 0
    # event ordering sanity
    ok = delivered
    assert (sim.t_assigned[ok] >= np.array([o.t_created for o in orders])[ok] - 1e-9).all()
    assert (sim.t_pickup[ok] >= np.array([o.t_ready for o in orders])[ok] - 1e-9).all()
    assert (sim.t_delivered[ok] >= sim.t_pickup[ok]).all()


def test_deterministic_given_seed(city, cfg, orders):
    lay = small_layout(city)
    a = summarize(Simulation(city, cfg, lay, orders, seed=3).run())
    b = summarize(Simulation(city, cfg, lay, orders, seed=3).run())
    assert a == b


def test_shift_mode_and_policy(city, cfg, orders):
    import copy
    c = copy.deepcopy(cfg)
    c.rider.shift_mode = "peaks"
    sim = Simulation(city, c, small_layout(city), orders, seed=3,
                     station_policy=LeastExpectedTime()).run()
    s = summarize(sim)
    assert s["stranded_events"] == 0 and s["delivered"] > 0


def test_nearest_rider_is_dispatched(city, cfg, orders):
    import copy
    c = copy.deepcopy(cfg)
    c.rider.n_riders = 3
    c.rider.init_soc_range = (1.0, 1.0)
    o = orders[0]
    sim = Simulation(city, c, small_layout(city), [o], seed=0)
    for r, off in zip(sim.riders, (2.0, 0.3, 1.0)):
        r.pos = o.origin + np.array([off, 0.0])
        sim.rider_pos[r.id] = r.pos
    o.id = 0
    sim.run()
    assert sim.riders[1].n_orders == 1


def test_split_budget():
    b = split_budget(np.array([1, 3, 0, 6]), 40, min_per=2)
    assert b.sum() == 40 and b.min() >= 2 and b[3] == b.max()


def test_optimizer_runs_and_keeps_budget(city, cfg):
    cand = candidate_sites(city, 0.8)
    with Evaluator(city, cfg, [1, 2], n_jobs=1) as ev:
        d0 = uniform_design(city, cand, 5, 30)
        res = IterativeRebalancer(ev, cand, max_iters=2, verbose=False).run(d0)
    assert res.best.batteries.sum() == 30
    assert len(set(res.best.sites.tolist())) == 5
    assert res.best_eval.cost <= res.history["cost"].iloc[0] + 1e-9


def test_loaders(tmp_path, city):
    lon, lat = -73.985, 40.758
    pd.DataFrame({
        "CAMIS": [1, 1, 2, 3, 4], "DBA": ["A", "A", "B", "C", "D"],
        "BORO": ["Manhattan", "Manhattan", "Manhattan", "Brooklyn", "Manhattan"],
        "CUISINE DESCRIPTION": ["Pizza", "Pizza", "Korean", "Thai", "Bakery"],
        "Latitude": [lat, lat, lat + .01, 40.65, lat], "Longitude": [lon, lon, lon, -73.95, lon],
    }).to_csv(tmp_path / "dohmh.csv", index=False)
    r = load_dohmh_restaurants(tmp_path / "dohmh.csv", region=city.region)
    assert list(r["cuisine"]) == ["Italian", "Korean"]

    pd.DataFrame({"cuisine_type": ["American"] * 3 + ["Thai"],
                  "day_of_the_week": ["Weekend", "Weekend", "Weekday", "Weekday"],
                  "food_preparation_time": [20, 25, 30, 35]}).to_csv(tmp_path / "k.csv", index=False)
    k = calibrate_from_kaggle(tmp_path / "k.csv")
    assert k["cuisine_prior"]["weekend"] == {"American": 1.0}
    assert k["weekend_per_day_ratio"] == pytest.approx((2 / 2) / (2 / 5))

    sq = [[lon - .01, lat - .01], [lon + .01, lat - .01], [lon + .01, lat + .01], [lon - .01, lat + .01],
          [lon - .01, lat - .01]]
    gj = {"type": "FeatureCollection", "features": [{"type": "Feature", "geometry": {
        "type": "Polygon", "coordinates": [sq]}, "properties": {
        "ntaname": "Midtown", "poptot": 1000, "medianinco": 90000, "pop18_24": 100, "pop25_34": 200}}]}
    (tmp_path / "n.geojson").write_text(json.dumps(gj))
    zones, paths = load_neighborhoods(tmp_path / "n.geojson")
    x, y = lonlat_to_xy(lon, lat)
    assert zones.loc[0, "share_young"] == pytest.approx(0.3)
    assert zones.loc[0, "x"] == pytest.approx(float(x), abs=1e-6)
    c2 = build_city(city, restaurants=r, zones=(zones, paths), kaggle=k)
    p = c2.sample_in_zone(0, np.random.default_rng(0))
    assert paths[0].contains_point(p)
