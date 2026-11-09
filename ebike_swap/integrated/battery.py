"""Battery packs as individual entities that move between riders, swap
stations, trucks and the central charging station."""
from __future__ import annotations

import math
from dataclasses import dataclass

from .config import BatteryParams, ChargerParams


@dataclass(eq=False)
class Battery:
    id: int
    nominal_wh: float
    soh: float = 1.0               # state of health: usable share of nominal capacity
    soc: float = 1.0               # state of charge, 0..1 of usable capacity
    cycles: float = 0.0            # full equivalent cycles
    where: str = "?"               # current location label (for the ledger)

    @property
    def usable_wh(self) -> float:
        return self.nominal_wh * self.soh

    @property
    def wh(self) -> float:
        return self.soc * self.usable_wh

    def drain(self, wh: float) -> bool:
        """Use `wh`; returns False if the pack ran flat."""
        need = wh / self.usable_wh
        ok = need <= self.soc + 1e-12
        used = min(need, self.soc)
        self.soc -= used
        self.cycles += used
        return ok


def charge_minutes(b: Battery, bp: BatteryParams, cp: ChargerParams) -> float:
    """CC-CV charge time from b.soc to full.

    Constant current I up to `cv_start_soc`, then constant voltage with the
    current decaying exponentially to the termination current. With charge
    Q left in the CV phase, tau = Q / (I0 - I_term) and t = tau ln(I0 / I_term).
    """
    ah = b.usable_wh / bp.voltage
    I, It, s = cp.current, cp.term_current_a, cp.cv_start_soc
    t_cc = max(0.0, s - b.soc) * ah / I
    q_cv = (1.0 - max(b.soc, s)) * ah
    if q_cv <= 0:
        return t_cc * 60
    I0 = I if b.soc <= s else I * (1.0 - b.soc) / (1.0 - s)
    if I0 <= It:
        return t_cc * 60
    tau = q_cv / (I0 - It)
    return (t_cc + tau * math.log(I0 / It)) * 60


def apply_charge(b: Battery, bp: BatteryParams, cp: ChargerParams):
    """Fill the pack and age it; charging faster than 0.75C wears it faster."""
    c_rate = cp.current / (b.usable_wh / bp.voltage)
    fade = bp.fade_per_cycle * (1.0 - b.soc)
    if c_rate > 0.75:
        fade *= 1.0 + bp.fast_charge_fade
    b.soh = max(0.0, b.soh - fade)
    b.soc = 1.0
