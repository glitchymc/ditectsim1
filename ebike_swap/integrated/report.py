"""Tables and headline numbers for a finished IntegratedSim."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .model import IntegratedSim


def job_table(sim: IntegratedSim) -> pd.DataFrame:
    rows = []
    for k, jobs in enumerate(sim.jobs):
        for j in jobs:
            rows.append({"platform": sim.cfg.platforms[k].name, "gid": j.gid, "t_created": j.t_created,
                         "t_accepted": j.t_accepted, "t_picked": j.t_picked, "t_delivered": j.t_delivered,
                         "cancelled": j.cancelled, "offers": j.offers, "pay": j.pay, "od_km": j.od_km,
                         "rider": j.rider})
    df = pd.DataFrame(rows)
    df["delivery_min"] = df["t_delivered"] - df["t_created"]
    df["accept_min"] = df["t_accepted"] - df["t_created"]
    return df


def platform_table(sim: IntegratedSim) -> pd.DataFrame:
    jt = job_table(sim)
    g = jt.groupby("platform", sort=False)
    out = pd.DataFrame({
        "dispatch": [p.dispatch for p in sim.cfg.platforms],
        "orders": g.size(),
        "delivered": g["t_delivered"].count(),
        "cancel_rate": g["cancelled"].mean(),
        "offers_per_order": g["offers"].mean(),
        "mean_accept_min": g["accept_min"].mean(),
        "mean_delivery_min": g["delivery_min"].mean(),
        "p90_delivery_min": g["delivery_min"].quantile(0.9),
        "riders_logged_in": sim.member.sum(0),
    })
    out["acceptance_rate"] = out["delivered"].clip(lower=0) / (out["offers_per_order"] * out["orders"])
    return out.reset_index()


def rider_table(sim: IntegratedSim) -> pd.DataFrame:
    return pd.DataFrame([{
        "rider": r.id, "apps": len(r.apps), "orders": sum(r.delivered.values()),
        "orders_from_2plus_platforms": int(len(r.delivered) >= 2), "earnings": r.earnings,
        "km": r.km, "offers_seen": r.offers_seen, "offers_accepted": r.offers_accepted,
        "swaps": r.swaps, "swap_wait_min": r.swap_wait_min, "swap_trip_min": r.swap_trip_min,
        "stranded": r.stranded, "soc_end": r.battery.soc,
    } for r in sim.riders])


def bss_table(sim: IntegratedSim) -> pd.DataFrame:
    span = sim.horizon - sim.cfg.start_hour * 60
    return pd.DataFrame([{
        "bss": s.id, "x": s.xy[0], "y": s.xy[1], "slots": s.slots, "swaps": s.swaps,
        "mean_wait_min": float(np.mean(s.waits)) if s.waits else 0.0,
        "p95_wait_min": float(np.quantile(s.waits, 0.95)) if s.waits else 0.0,
        "max_queue": s.max_queue, "stockout_frac": s.stockout_min / span,
        "avg_charged": s.area_charged / span, "avg_depleted": s.area_depleted / span,
        "truck_visits": s.truck_visits,
    } for s in sim.stations])


def truck_table(sim: IntegratedSim) -> pd.DataFrame:
    span = sim.horizon - sim.cfg.start_hour * 60
    return pd.DataFrame([{
        "truck": t.id, "home": t.home, "trips": t.trips, "km": t.km, "packs_moved": t.packs_moved,
        "utilization": t.busy_min / span, "missed_departures": t.missed_departures,
    } for t in sim.fleet.trucks])


def summary(sim: IntegratedSim) -> dict:
    jt, rt, bt, tt = job_table(sim), rider_table(sim), bss_table(sim), truck_table(sim)
    waits = np.concatenate([s.waits for s in sim.stations if s.waits]) if rt["swaps"].sum() else np.zeros(1)
    alive = [b for b in sim.batteries if b.where != "discarded"]
    ccs = sim.ccs
    return {
        "orders": len(jt), "delivered": int(jt["t_delivered"].notna().sum()),
        "cancel_rate": float(jt["cancelled"].mean()),
        "mean_delivery_min": float(jt["delivery_min"].mean()),
        "offers_per_order": float(jt["offers"].mean()),
        "riders_multi_app": float((rt["apps"] >= 2).mean()),
        "mean_earnings_per_rider": float(rt["earnings"].mean()),
        "swaps": int(rt["swaps"].sum()),
        "mean_swap_wait_min": float(waits.mean()),
        "p95_swap_wait_min": float(np.quantile(waits, 0.95)),
        "mean_swap_trip_min": float(rt["swap_trip_min"].sum() / max(rt["swaps"].sum(), 1)),
        "bss_stockout_frac": float(bt["stockout_frac"].mean()),
        "truck_trips": int(tt["trips"].sum()), "truck_km": float(tt["km"].sum()),
        "truck_utilization": float(tt["utilization"].mean()),
        "packs_charged_at_ccs": ccs.charged_count,
        "packs_discarded": len(ccs.discarded), "packs_purchased": ccs.purchased,
        "battery_fleet": len(alive),
        "mean_soh": float(np.mean([b.soh for b in alive])),
        "stranded_riders": int(rt["stranded"].sum()),
        "events": sim.k.n_events,
    }
