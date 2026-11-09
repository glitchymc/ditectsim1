"""Figures laid out like the flowchart: riders and swap stations on the left,
trucks in the middle, the central charging station pipeline on the right."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

from ..viz import SOC_CMAP  # noqa: E402
from .model import CCS_STAGES, IntegratedSim  # noqa: E402

GREEN, RED, AMBER, BLUE, GREY = "#2e8b57", "#c0392b", "#e6a817", "#1f4e79", "#9aa5ad"
STAGE_LABELS = {
    "ccs_unloading": "unloading buffer", "ccs_testing": "testing", "ccs_tested": "passed, awaiting van",
    "ccs_van": "van", "ccs_hall_queue": "waiting for charger", "ccs_charging": "charging (CC→CV)",
    "ccs_hall_ready": "charged, awaiting van", "ccs_loading": "loading buffer",
}
LEDGER = [  # (column, label, colour) in flow order
    ("rider", "on riders' bikes", BLUE), ("bss_charged", "BSS charged", GREEN),
    ("bss_depleted", "BSS depleted", RED), ("truck", "on trucks", "#6c5b7b"),
    ("ccs_unloading", "CCS unloading", "#e07a5f"), ("ccs_testing", "CCS testing", "#f2a65a"),
    ("ccs_tested", "CCS awaiting van", "#f6c28b"), ("ccs_van", "CCS van", "#c9b18a"),
    ("ccs_hall_queue", "CCS waiting for charger", "#f1d58a"), ("ccs_charging", "CCS charging", AMBER),
    ("ccs_hall_ready", "CCS charged, awaiting van", "#a3c9a8"), ("ccs_loading", "CCS loading buffer", "#69a37b"),
]


def _hhmm(t):
    return f"{int(t // 60) % 24:02d}:{int(t % 60):02d}" + (" (+1d)" if t >= 1440 else "")


def inventory_figure(sim: IntegratedSim, path: str | None = None):
    """Where every battery pack is over the day, plus riders queued at stations."""
    df = pd.DataFrame(sim.samples).fillna(0)
    h = df["t"] / 60
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(10, 6.4), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    ys = [df[c] if c in df else np.zeros(len(df)) for c, _, _ in LEDGER]
    ax.stackplot(h, ys, labels=[lab for _, lab, _ in LEDGER], colors=[c for _, _, c in LEDGER], lw=0)
    ax.set_ylabel("battery packs")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=8)
    ax.set_title("Where the packs are", loc="left", fontsize=10)
    ax2.plot(h, df["bss_queue"], color=RED, lw=1.6, label="riders queued at BSS")
    ax2.plot(h, df["riders_busy"], color=BLUE, lw=1.2, label="riders on a job")
    ax2.set_xlabel("hour of day")
    ax2.legend(frameon=False, fontsize=8, loc="upper left")
    for a in (ax, ax2):
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=130, bbox_inches="tight")
    return fig


def _frame(fig, axes, sim: IntegratedSim, snap: dict, qmax: int):
    ax_map, ax_bss, ax_ccs = axes
    for a in axes:
        a.clear()
    poly = sim.city.region.polygon
    ax_map.fill(poly[:, 0], poly[:, 1], color="#f2f2f2", ec="#999", lw=0.8, zorder=0)
    on = np.array([s != "off" for s in snap["state"]])
    p = snap["pos"][on]
    ax_map.scatter(p[:, 0], p[:, 1], c=snap["soc"][on], cmap=SOC_CMAP, vmin=0, vmax=1, s=8, lw=0, zorder=2)
    q = snap["bss_queue"]
    ax_map.scatter(sim.bss_xy[:, 0], sim.bss_xy[:, 1], marker="s", s=40 + 12 * q,
                   c=np.where(q > 0, RED, BLUE), ec="white", lw=0.8, zorder=3)
    for i, (x, y) in enumerate(sim.bss_xy):
        ax_map.annotate(f"BSS{i}", (x, y), xytext=(3, 3), textcoords="offset points", fontsize=6)
    tr = snap["trucks"]
    ax_map.scatter(tr[:, 0], tr[:, 1], marker="^", s=55, c="#6c5b7b", ec="white", lw=0.6, zorder=4)
    ax_map.scatter(*sim.ccs_xy, marker="*", s=220, c=AMBER, ec="k", lw=0.6, zorder=4)
    ax_map.annotate("CCS", sim.ccs_xy, xytext=(6, -2), textcoords="offset points", fontsize=8, weight="bold")
    ax_map.set_aspect("equal")
    ax_map.set_axis_off()
    ax_map.set_title(f"{_hhmm(snap['t'])} · {on.sum()} riders · ▲ trucks", fontsize=9)

    n, slots = len(sim.stations), sim.cfg.bss.slots
    for i in range(n):
        y = n - 1 - i
        ch, dp = int(snap["bss_charged"][i]), int(snap["bss_depleted"][i])
        colors = [GREEN] * ch + [RED] * dp + ["#e3e3e3"] * max(0, slots - ch - dp)
        ax_bss.barh([y] * slots, [0.8] * slots, left=np.arange(slots), color=colors[:slots], height=0.6, lw=0)
        if q[i]:
            w = q[i] / max(qmax, 1) * slots * 0.5
            ax_bss.barh(y, w, left=slots + 1, color=RED, height=0.6, alpha=0.6)
            ax_bss.text(slots + 1.3 + w, y, str(q[i]), va="center", fontsize=6)
    ax_bss.set_yticks(range(n))
    ax_bss.set_yticklabels([f"BSS{i}" for i in reversed(range(n))], fontsize=6)
    ax_bss.set_xticks([])
    ax_bss.set_xlim(-0.5, slots * 1.6 + 2)
    ax_bss.spines[["top", "right", "bottom"]].set_visible(False)
    ax_bss.set_title("swap stations: charged / depleted · queue", fontsize=9, loc="left")

    vals = [snap["ccs"].get(k, 0) for k in CCS_STAGES]
    ys = np.arange(len(CCS_STAGES))[::-1]
    ax_ccs.barh(ys, vals, color=["#e07a5f", "#f2a65a", "#f6c28b", "#c9b18a", "#f1d58a", AMBER, "#a3c9a8", GREEN],
                height=0.6)
    for y, v in zip(ys, vals):
        ax_ccs.text(v + 1, y, str(v), va="center", fontsize=7)
    ax_ccs.set_yticks(ys)
    ax_ccs.set_yticklabels([STAGE_LABELS[k] for k in CCS_STAGES], fontsize=7)
    ax_ccs.set_xlim(0, max(10, max(vals) * 1.25))
    ax_ccs.spines[["top", "right"]].set_visible(False)
    ax_ccs.tick_params(axis="x", labelsize=7)
    ax_ccs.set_title("central charging station", fontsize=9, loc="left")


def _setup():
    return plt.subplots(1, 3, figsize=(14, 6.5), gridspec_kw={"width_ratios": [1, 1.05, 0.9]})


def snapshot_figure(sim: IntegratedSim, k: int = -1, path: str | None = None):
    fig, axes = _setup()
    qmax = max(int(s["bss_queue"].max()) for s in sim.snaps)
    _frame(fig, axes, sim, sim.snaps[k], qmax)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=120)
    return fig


def animate(sim: IntegratedSim, path: str, fps: int = 6, every: int = 1):
    fig, axes = _setup()
    snaps = sim.snaps[::every]
    qmax = max(int(s["bss_queue"].max()) for s in snaps)
    _frame(fig, axes, sim, snaps[0], qmax)
    fig.tight_layout()
    FuncAnimation(fig, lambda i: _frame(fig, axes, sim, snaps[i], qmax), frames=len(snaps)).save(
        path, writer=PillowWriter(fps=fps))
    plt.close(fig)
