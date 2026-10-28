"""Simulation parameters.

All times are in minutes, distances in km, energy in Wh unless noted.
Defaults are order-of-magnitude placeholders for a NYC delivery e-bike and
should be replaced with calibrated values when better data is available.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EnergyParams:
    battery_wh: float = 720.0          # 48 V x 15 Ah pack
    wh_per_km: float = 15.0            # flat-ground cruise with cargo
    total_mass_kg: float = 120.0       # rider + bike + cargo
    climb_efficiency: float = 0.75     # drivetrain efficiency when climbing
    regen_fraction: float = 0.0        # share of descent energy recovered
    lights_per_km: float = 12.0        # Manhattan street grid ~ every 80 m
    p_stop_at_light: float = 0.5
    wh_per_stop: float = 1.0           # re-accelerating 120 kg to ~25 km/h


@dataclass
class TravelParams:
    cruise_kmh: float = 18.0
    light_delay_min: float = 5.0 / 60.0   # mean delay per signal encountered
    detour_factor: float = 1.05           # network distance / grid L1 distance


@dataclass
class StationParams:
    charge_kw: float = 0.36            # per-slot charge power (0.5C for 720 Wh)
    swap_service_min: float = 1.0      # time to physically swap a pack
    chargers_per_station: int | None = None   # None => one charger per slot


@dataclass
class RiderParams:
    n_riders: int = 400
    init_soc_range: tuple[float, float] = (0.4, 1.0)
    # "auto" => minimal level that covers the longest job plus the trip to the
    # farthest-nearest station (the deck's assumption); a float overrides it.
    swap_threshold: float | str = "auto"
    threshold_margin: float = 0.03
    # Shift model: "all_day" or "peaks" (riders log on around meal peaks).
    shift_mode: str = "all_day"


@dataclass
class DispatchParams:
    max_pickup_km: float = 3.0
    allow_cross_region: bool = True
    order_timeout_min: float = 60.0    # unassigned orders are cancelled after this


@dataclass
class DemandParams:
    orders_per_day: float = 8000.0
    day_of_week: int = 5               # 0=Mon ... 6=Sun
    start_hour: float = 0.0
    end_hour: float = 24.0
    max_od_km: float = 3.0             # restaurants considered for a destination
    od_decay_km: float = 1.0           # distance-decay scale for restaurant choice
    prep_time_range: tuple[float, float] = (20.0, 35.0)   # Kaggle dataset range
    # Demographic destination weights (see demand.DemographicDestinationSampler).
    beta_young: float = 1.5
    beta_income: float = 0.5


@dataclass
class SimConfig:
    energy: EnergyParams = field(default_factory=EnergyParams)
    travel: TravelParams = field(default_factory=TravelParams)
    station: StationParams = field(default_factory=StationParams)
    rider: RiderParams = field(default_factory=RiderParams)
    dispatch: DispatchParams = field(default_factory=DispatchParams)
    demand: DemandParams = field(default_factory=DemandParams)
    snapshot_every_min: float | None = None   # record rider/station state for viz
