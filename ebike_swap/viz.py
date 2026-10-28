"""Plots following the deck's sketch: map of riders coloured by battery level,
next to per-station battery racks and queue bars."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from .data.city import City  # noqa: E402

SOC_CMAP = LinearSegmentedColormap.from_list("soc", ["#c0392b", "#e6a817", "#2e8b57"])
CHARGED, CHARGING, EMPTY_SLOT, QUEUE = "#2e8b57", "#e6a817", "#d9d9d9", "#c0392b"


def _hhmm(t):
    return f"{int(t // 60) % 24:02d}:{int(t % 60):02d}"


def _draw_frame(fig, axes, city: City, station_xy, batteries, snap, max_queue):
    ax_map, ax_rack = axes
    ax_map.clear()
    ax_rack.clear()
    poly = city.region.polygon
    ax_map.fill(poly[:, 0], poly[:, 1], color="#f2f2f2", ec="#999", lw=0.8, zorder=0)
    active = np.array([s != "off" for s in snap["state"]])
    p = snap["pos"][active]
    sc = ax_map.scatter(p[:, 0], p[:, 1], c=snap["soc"][active], cmap=SOC_CMAP, vmin=0, vmax=1,
                        s=9, lw=0, zorder=2)
    q = snap["queue"]
    ax_map.scatter(station_xy[:, 0], station_xy[:, 1], marker="s", s=40 + 25 * q,
                   c=np.where(q > 0, QUEUE, "#1f4e79"), ec="white", lw=0.8, zorder=3)
    for i, (x, y) in enumerate(station_xy):
        ax_map.annotate(f"S{i}", (x, y), xytext=(3, 3), textcoords="offset points", fontsize=6)
    ax_map.set_aspect("equal")
    ax_map.set_axis_off()
    ax_map.set_title(f"{_hhmm(snap['t'])}  ·  {active.sum()} riders on shift", fontsize=9)

    n = len(station_xy)
    bmax = int(batteries.max())
    for i in range(n):
        y = n - 1 - i
        b, ch, cg = int(batteries[i]), int(snap["charged"][i]), int(snap["charging"][i])
        colors = [CHARGED] * ch + [CHARGING] * min(cg, b - ch) + [EMPTY_SLOT] * max(0, b - ch - cg)
        ax_rack.barh([y] * b, [0.8] * b, left=np.arange(b), color=colors[:b], height=0.6, lw=0)
        if q[i]:
            ax_rack.barh(y, q[i] * (bmax / max(max_queue, 1)) * 0.6, left=bmax + 1.5,
                         color=QUEUE, height=0.6)
            ax_rack.text(bmax + 1.6 + q[i] * (bmax / max(max_queue, 1)) * 0.6, y, str(q[i]),
                         va="center", fontsize=6)
    ax_rack.set_yticks(range(n))
    ax_rack.set_yticklabels([f"S{i}" for i in reversed(range(n))], fontsize=6)
    ax_rack.set_xticks([])
    ax_rack.set_xlim(-0.5, bmax * 1.75 + 3)
    for s in ("top", "right", "bottom"):
        ax_rack.spines[s].set_visible(False)
    ax_rack.set_title("batteries: charged / charging      queue", fontsize=9, loc="left")
    return sc


def snapshot_figure(city: City, sim, k: int = -1, path: str | None = None):
    snaps = sim.snapshots
    fig, axes = plt.subplots(1, 2, figsize=(10, 7), gridspec_kw={"width_ratios": [1, 1.1]})
    mq = max(int(s["queue"].max()) for s in snaps)
    bats = np.array([s.n_batteries for s in sim.stations])
    sc = _draw_frame(fig, axes, city, sim.station_xy, bats, snaps[k], mq)
    fig.colorbar(sc, ax=axes[0], fraction=0.03, pad=0.01, label="battery level")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=130)
    return fig


def animate(city: City, sim, path: str, fps: int = 6):
    snaps = sim.snapshots
    fig, axes = plt.subplots(1, 2, figsize=(10, 7), gridspec_kw={"width_ratios": [1, 1.1]})
    mq = max(int(s["queue"].max()) for s in snaps)
    bats = np.array([s.n_batteries for s in sim.stations])
    sc = _draw_frame(fig, axes, city, sim.station_xy, bats, snaps[0], mq)
    fig.colorbar(sc, ax=axes[0], fraction=0.03, pad=0.01, label="battery level")
    fig.tight_layout()
    anim = FuncAnimation(fig, lambda i: _draw_frame(fig, axes, city, sim.station_xy, bats, snaps[i], mq),
                         frames=len(snaps))
    anim.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)


def layout_figure(city: City, layouts: dict, stations: dict, path: str | None = None):
    """One map per layout; marker area = batteries, colour = mean queue wait."""
    n = len(layouts)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 6.2), squeeze=False)
    vmax = max(float(st["mean_wait_min"].max()) for st in stations.values()) or 1.0
    poly = city.region.polygon
    for ax, (name, lay) in zip(axes[0], layouts.items()):
        st = stations[name].sort_values("station")
        ax.fill(poly[:, 0], poly[:, 1], color="#f2f2f2", ec="#999", lw=0.8)
        s = ax.scatter(lay.xy[:, 0], lay.xy[:, 1], s=6 * lay.batteries, c=st["mean_wait_min"],
                       cmap="magma_r", vmin=0, vmax=vmax, ec="k", lw=0.4)
        ax.set_title(name, fontsize=9)
        ax.set_aspect("equal")
        ax.set_axis_off()
    fig.colorbar(s, ax=axes[0].tolist(), fraction=0.02, label="mean queue wait (min)")
    if path:
        fig.savefig(path, dpi=130, bbox_inches="tight")
    return fig


def convergence_figure(histories: dict, path: str | None = None):
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    for name, h in histories.items():
        ax.plot(np.arange(len(h)), h["best"], label=name, lw=1.6)
        ax.scatter(np.arange(len(h)), h["cost"], s=6, alpha=0.35)
    ax.set_xlabel("iteration")
    ax.set_ylabel("cost (rider-hour equivalents)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=130)
    return fig
