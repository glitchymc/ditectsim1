"""Parameters for the integrated frontend (platforms + riders) and backend
(swap stations, trucks, central charging station) simulator.

Every row of the flowchart's PARAMETERS / OPTIONS table maps to a field here.
Units: minutes, km, Wh, A, V. Defaults are placeholders to calibrate.
"""
from __future__ import annotations

from dataclasses import dataclass, field


# ------------------------------------------------------------------ frontend
@dataclass
class PlatformSpec:
    """One delivery platform with its own demand and dispatch algorithm."""

    name: str
    orders_per_day: float
    dispatch: str = "nearest"          # "nearest" | "batching"
    pay_base: float = 3.0              # $ per order
    pay_per_km: float = 1.2            # $ per km restaurant -> customer
    refresh_min: float = 1.0           # re-offer interval after a decline / no candidate
    order_timeout_min: float = 45.0    # unassigned orders are cancelled after this
    max_pickup_km: float = 3.0
    batch_detour_km: float = 1.5       # "batching": max extra km for a stacked order


def default_platforms(total_orders_per_day: float = 8000.0) -> list[PlatformSpec]:
    # Generic stand-ins for the diagram's "Platform 1 (UberEats)", "Platform 2 (DoorDash)", ...
    return [
        PlatformSpec("P1", 0.45 * total_orders_per_day, "nearest", 3.0, 1.2),
        PlatformSpec("P2", 0.35 * total_orders_per_day, "batching", 2.5, 1.5),
        PlatformSpec("P3", 0.20 * total_orders_per_day, "nearest", 3.5, 1.0),
    ]


@dataclass
class RiderBehavior:
    n_riders: int = 400
    # share of riders logged into 1, 2, 3, ... apps at once (multi-apping)
    apps_per_rider: tuple[float, ...] = (0.4, 0.4, 0.2)
    max_jobs: int = 2                  # batched active job queue length
    wage_per_min: tuple[float, float] = (0.25, 0.45)   # reservation value of time, $/min
    accept_noise: float = 1.0          # logit scale ($) of the accept/decline decision
    decide_min: float = 0.2            # time to look at an offer
    swap_threshold: float = 0.25       # go swap once below this (after finishing tasks)
    bss_choice: str = "avoid_empty"    # "nearest" | "avoid_empty" (app shows which stations have packs)
    init_soc: tuple[float, float] = (0.4, 1.0)
    shift_mode: str = "all_day"        # "all_day" | "peaks"


# ------------------------------------------------------------------ backend
@dataclass
class BatteryParams:
    nominal_wh: float = 720.0
    voltage: float = 48.0
    init_soh: tuple[float, float] = (0.82, 1.0)
    fade_per_cycle: float = 0.0004     # SoH lost per full equivalent cycle
    fast_charge_fade: float = 0.5      # extra fade multiplier when charged above 0.75C


@dataclass
class BSSParams:
    n_stations: int = 10
    slots: int = 22                    # BSS capacity (charged + depleted packs)
    initial_charged: int = 20
    swap_min: float = 1.0


@dataclass
class TruckParams:
    # Option 1: "hub"       truck CCS -> BSS -> CCS when a BSS's depleted count hits `threshold`
    # Option 2: "dedicated" truck BSS -> CCS -> BSS, departures by `departure`
    mode: str = "hub"
    n_trucks: int = 6                  # hub mode fleet; dedicated mode uses one per BSS
    capacity: int = 30
    speed_kmh: float = 16.0
    handle_min_per_pack: float = 0.3   # load or unload one pack
    threshold: int = 8                 # option 1a: depleted packs that trigger a pickup
    departure: str = "schedule"        # option 2a "poisson" | 2b "schedule"
    mean_headway_min: float = 120.0    # 2a: mean time between departures
    schedule_every_min: float = 120.0  # 2b: departure every N minutes ...
    schedule_start_min: float = 360.0  # ... starting at this time of day
    return_policy: str = "immediate"   # 2c: "immediate" | "wait" (for enough charged packs)
    max_wait_min: float = 90.0         # cap on "wait"


@dataclass
class ChargerParams:
    kind: str = "standard"             # "standard" | "fast" (sets current)
    current_a: float | None = None     # override; None -> preset
    cv_start_soc: float = 0.8          # constant current up to here, then constant voltage
    term_current_a: float = 1.0        # CV phase ends at this taper current

    @property
    def current(self) -> float:
        if self.current_a is not None:
            return self.current_a
        return {"standard": 10.0, "fast": 20.0}[self.kind]


@dataclass
class CCSParams:
    lonlat: tuple[float, float] = (-73.940, 40.745)   # Long Island City, Queens
    initial_charged: int = 250         # spare pool in the loading buffer
    n_testers: int = 2
    test_min: float = 2.0
    soh_min: float = 0.80              # discard below this state of health
    defect_prob: float = 0.005         # random test failure
    replace_discarded: bool = True     # buy a new pack for each discard
    n_chargers: int = 80
    charger: ChargerParams = field(default_factory=ChargerParams)
    van_capacity: int = 40             # shuttle between testing and the charging hall
    van_transit_min: float = 8.0
    van_every_min: float = 15.0


@dataclass
class IntegratedConfig:
    platforms: list[PlatformSpec] = field(default_factory=default_platforms)
    riders: RiderBehavior = field(default_factory=RiderBehavior)
    battery: BatteryParams = field(default_factory=BatteryParams)
    bss: BSSParams = field(default_factory=BSSParams)
    trucks: TruckParams = field(default_factory=TruckParams)
    ccs: CCSParams = field(default_factory=CCSParams)
    start_hour: float = 0.0
    end_hour: float = 24.0
    drain_min: float = 180.0           # keep running after the last order
    sample_every_min: float = 10.0     # inventory time series / snapshots
    snapshots: bool = False
