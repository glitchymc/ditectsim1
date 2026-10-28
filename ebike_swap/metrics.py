"""Turn a finished `Simulation` into tables and headline KPIs."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .engine import Simulation


def order_table(sim: Simulation) -> pd.DataFrame:
    o = sim.orders
    df = pd.DataFrame({
        "id": [x.id for x in o],
        "t_created": [x.t_created for x in o],
        "cuisine": [x.cuisine for x in o],
        "dest_zone": [x.dest_zone for x in o],
        "t_assigned": sim.t_assigned,
        "t_pickup": sim.t_pickup,
        "t_delivered": sim.t_delivered,
        "cancelled": sim.cancelled,
    })
    df["delivery_min"] = df["t_delivered"] - df["t_created"]
    df["assign_wait_min"] = df["t_assigned"] - df["t_created"]
    return df


def station_table(sim: Simulation) -> pd.DataFrame:
    rows = []
    for s in sim.stations:
        span = max((sim.cfg.demand.end_hour - sim.cfg.demand.start_hour) * 60, 1e-9)
        w = np.asarray(s.waits) if s.waits else np.zeros(1)
        tp = np.asarray(s.trigger_points) if s.trigger_points else s.xy[None, :]
        rows.append({
            "station": s.id, "x": s.xy[0], "y": s.xy[1], "batteries": s.n_batteries,
            "swaps": s.n_swaps, "arrivals": s.n_arrivals,
            "swaps_per_battery": s.n_swaps / s.n_batteries,
            "mean_wait_min": float(w.mean()) if s.waits else 0.0,
            "p90_wait_min": float(np.quantile(w, 0.9)) if s.waits else 0.0,
            "total_wait_min": float(w.sum()) if s.waits else 0.0,
            "avg_queue": s.area_queue / span, "max_queue": s.max_queue,
            "stockout_frac": s.stockout_min / span,
            "avg_charged_frac": s.area_charged / span / s.n_batteries,
            "trigger_cx": float(tp[:, 0].mean()), "trigger_cy": float(tp[:, 1].mean()),
        })
    return pd.DataFrame(rows)


def rider_table(sim: Simulation) -> pd.DataFrame:
    return pd.DataFrame([{
        "rider": r.id, "orders": r.n_orders, "km": r.km, "wh": r.wh, "swaps": r.n_swaps,
        "swap_min": r.swap_min, "stranded": r.stranded, "soc_end": r.soc,
        "shift_start": r.shift[0], "shift_end": r.shift[1],
    } for r in sim.riders])


def summarize(sim: Simulation, late_after_min: float = 45.0) -> dict:
    od = order_table(sim)
    st = station_table(sim)
    rd = rider_table(sim)
    delivered = od["t_delivered"].notna()
    n = max(len(od), 1)
    swaps = int(rd["swaps"].sum())
    waits = np.concatenate([np.asarray(s.waits) for s in sim.stations]) if swaps else np.zeros(0)
    dmin = od.loc[delivered, "delivery_min"]
    return {
        "orders": len(od),
        "delivered": int(delivered.sum()),
        "cancel_rate": float(od["cancelled"].mean()) if len(od) else 0.0,
        "undelivered_rate": float(1 - delivered.sum() / n),
        "mean_delivery_min": float(dmin.mean()) if len(dmin) else np.nan,
        "p90_delivery_min": float(dmin.quantile(0.9)) if len(dmin) else np.nan,
        "late_rate": float((dmin > late_after_min).sum() / n),
        "mean_assign_wait_min": float(od["assign_wait_min"].mean()),
        "swaps": swaps,
        "swap_hours": float(rd["swap_min"].sum() / 60),
        "mean_swap_downtime_min": float(rd["swap_min"].sum() / swaps) if swaps else 0.0,
        "mean_station_wait_min": float(waits.mean()) if len(waits) else 0.0,
        "p95_station_wait_min": float(np.quantile(waits, 0.95)) if len(waits) else 0.0,
        "station_max_queue": int(st["max_queue"].max()) if len(st) else 0,
        "idle_stations": int((st["swaps"] == 0).sum()),
        "stranded_events": int(rd["stranded"].sum()),
        "swap_threshold": sim.swap_threshold,
    }
