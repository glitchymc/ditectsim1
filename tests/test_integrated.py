import copy

import numpy as np
import pytest

from ebike_swap.data import synthetic_manhattan
from ebike_swap.integrated import IntegratedConfig, IntegratedSim, platform_table, rider_table, summary
from ebike_swap.integrated.battery import Battery, charge_minutes
from ebike_swap.integrated.config import BatteryParams, ChargerParams, default_platforms
from ebike_swap.integrated.frontend import Rider, best_insertion
from ebike_swap.optimize import candidate_sites, uniform_design

KNOWN = {"rider", "bss_charged", "bss_depleted", "bss_swapping", "truck", "discarded",
         "ccs_unloading", "ccs_testing", "ccs_tested", "ccs_van", "ccs_hall_queue",
         "ccs_charging", "ccs_hall_ready", "ccs_loading"}


@pytest.fixture(scope="module")
def city():
    return synthetic_manhattan(0, n_restaurants=600)


def small_cfg(**trucks):
    cfg = IntegratedConfig(platforms=default_platforms(1500))
    cfg.riders.n_riders = 70
    cfg.bss.n_stations, cfg.bss.slots, cfg.bss.initial_charged = 4, 12, 10
    cfg.ccs.initial_charged = 40
    cfg.trucks.threshold, cfg.trucks.capacity, cfg.trucks.n_trucks = 4, 15, 2
    for k, v in trucks.items():
        setattr(cfg.trucks, k, v)
    return cfg


def sites(city, k=4):
    c = candidate_sites(city)
    return c[uniform_design(city, c, k, 2 * k).sites]


@pytest.fixture(scope="module")
def hub_run(city):
    return IntegratedSim(city, small_cfg(), sites(city), seed=2).run()


def check_ledger(sim):
    where = [b.where for b in sim.batteries]
    assert set(where) <= KNOWN
    n0 = sim.cfg.riders.n_riders + len(sim.stations) * sim.cfg.bss.initial_charged + sim.cfg.ccs.initial_charged
    assert len(sim.batteries) == n0 + sim.ccs.purchased
    assert sum(w == "truck" for w in where) == sum(len(t.cargo_charged) + len(t.cargo_depleted)
                                                    for t in sim.fleet.trucks)
    assert sum(w == "rider" for w in where) == len(sim.riders)
    for s in sim.stations:
        assert len(s.charged) + len(s.depleted) + s.in_swap <= s.slots
        assert all(b.where == "bss_charged" for b in s.charged)
        assert all(b.where == "bss_depleted" for b in s.depleted)
    assert sum(w == "ccs_loading" for w in where) == len(sim.ccs.loading)


def test_charge_time_cc_cv():
    bp = BatteryParams()
    b = Battery(0, 720, 1.0, 0.2)
    std, fast = charge_minutes(b, bp, ChargerParams("standard")), charge_minutes(b, bp, ChargerParams("fast"))
    assert 60 < std < 150 and fast < std
    b.soc = 0.9                    # only constant-voltage left
    assert 0 < charge_minutes(b, bp, ChargerParams()) < std
    b.soc = 1.0
    assert charge_minutes(b, bp, ChargerParams()) == 0


def test_hub_day_conserves_batteries_and_serves_orders(hub_run):
    sim = hub_run
    check_ledger(sim)
    s = summary(sim)
    assert s["stranded_riders"] == 0
    assert s["delivered"] / s["orders"] > 0.9
    assert s["swaps"] > 0 and s["truck_trips"] > 0 and s["packs_charged_at_ccs"] > 0
    for jobs in sim.jobs:
        for j in jobs:
            if not np.isnan(j.t_delivered):
                assert j.t_created <= j.t_accepted <= j.t_picked <= j.t_delivered
                assert j.t_picked >= j.t_ready - 1e-9
            assert not (j.cancelled and j.rider >= 0)
    assert all(not r.stops for r in sim.riders)


def test_multi_app_riders_serve_several_platforms(hub_run):
    rt = rider_table(hub_run)
    assert (rt["apps"] >= 2).any()
    assert rt["orders_from_2plus_platforms"].sum() > 0
    # riders only ever deliver for platforms they are logged into
    for r in hub_run.riders:
        assert set(r.delivered) <= set(r.apps)
    pt = platform_table(hub_run)
    assert set(pt["platform"]) == {"P1", "P2", "P3"}


@pytest.mark.parametrize("departure,ret", [("schedule", "immediate"), ("poisson", "wait")])
def test_dedicated_trucks(city, departure, ret):
    sim = IntegratedSim(city, small_cfg(mode="dedicated", departure=departure, return_policy=ret,
                                        schedule_every_min=90, mean_headway_min=90),
                        sites(city), seed=3).run()
    check_ledger(sim)
    assert len(sim.fleet.trucks) == len(sim.stations)
    assert sum(t.trips for t in sim.fleet.trucks) > 0
    assert summary(sim)["stranded_riders"] == 0


def test_same_jobs_same_result(city):
    cfg = small_cfg()
    probe = IntegratedSim(city, cfg, sites(city), seed=5)
    jobs = IntegratedSim.generate_jobs(city, cfg, probe.router, 5)
    a = summary(IntegratedSim(city, cfg, sites(city), seed=5, jobs_by_platform=copy.deepcopy(jobs)).run())
    b = summary(IntegratedSim(city, cfg, sites(city), seed=5, jobs_by_platform=copy.deepcopy(jobs)).run())
    assert a == b


def test_best_insertion_keeps_pickup_before_dropoff(city, hub_run):
    sim = hub_run
    r = Rider(999, np.array([3.0, 5.0]), Battery(0, 720), frozenset({0}), 0.3, (0, 1440))
    j1, j2 = sim.jobs[0][0], sim.jobs[0][1]
    r.stops, _, _ = best_insertion(sim, r, j1)
    r.head_locked = True
    seq, added, total = best_insertion(sim, r, j2)
    assert seq[0] is r.stops[0]                       # committed head stop stays first
    for j in (j1, j2):
        kinds = [s.kind for s in seq if s.job is j]
        assert kinds == ["P", "D"]
    assert added >= 0 and total >= added
