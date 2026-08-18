"""Classical DoE figures: contour, 3D surface, traces, interaction, Pareto,
half-normal.

Plots lead, tables support. These answer the questions a formulator brings --
which lever matters, where the sweet spot sits, whether the levers interact --
in a form readable at a glance rather than decoded from a coefficient table.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm  # noqa: F401

from pipeline.doe.analysis import ResponseAnalysis

GRADE_COLOURS = {"K100LV": "#3b7dd8", "K4M": "#d9822b", "K100M": "#8e5bb5"}
_FALLBACK = ["#3b7dd8", "#d9822b", "#8e5bb5", "#3aa17e", "#c0504d"]


def _colour(grade: str, i: int = 0) -> str:
    return GRADE_COLOURS.get(grade, _FALLBACK[i % len(_FALLBACK)])


def _save(fig: plt.Figure, out: Path, name: str, banner: str | None) -> None:
    if banner:
        # Above the title, not below the axes: with constrained layout the bottom
        # of the figure belongs to the x-labels and the banner lands on top of them.
        # bbox_inches="tight" pulls anything past y=1 back into the saved image.
        fig.text(
            0.5, 1.02, banner, ha="center", va="bottom",
            fontsize=7.5, color="#a03030", weight="bold",
        )
    fig.savefig(out / f"{name}.svg", format="svg", bbox_inches="tight")
    fig.savefig(out / f"{name}.png", format="png", bbox_inches="tight")
    plt.close(fig)


def _z_array(grid_rows: tuple[tuple[float | None, ...], ...]) -> np.ndarray:
    return np.array(
        [[np.nan if v is None else v for v in row] for row in grid_rows], dtype=float
    )


def contour_panel(ra: ResponseAnalysis, out: Path, banner: str | None) -> str:
    """One contour per grade, on a shared colour scale.

    The shared scale is deliberate. Scaling each panel to its own range makes
    every grade look equally variable and hides the thing worth seeing: that one
    grade spans far more of the response than another.
    """
    grids = ra.grids
    lo, hi = ra.scale
    # constrained layout, not the global autolayout: tight_layout and colorbars
    # fight each other and the bar ends up drawn over the last panel.
    fig, axes = plt.subplots(
        1, len(grids), figsize=(4.6 * len(grids), 3.9), sharey=True,
        layout="constrained",
    )
    axes = np.atleast_1d(axes)
    levels = np.linspace(lo, hi, 12) if np.isfinite(lo) and hi > lo else 10

    mesh = None
    for ax, g in zip(axes, grids, strict=False):
        z = _z_array(g.z)
        mesh = ax.contourf(g.api_axis, g.hpmc_axis, z, levels=levels, cmap="viridis")
        lines = ax.contour(
            g.api_axis, g.hpmc_axis, z, levels=levels,
            colors="white", linewidths=0.5, alpha=0.6,
        )
        ax.clabel(lines, inline=True, fontsize=6, fmt="%.0f")
        ax.scatter(
            [p[0] for p in g.design_points], [p[1] for p in g.design_points],
            s=26, c="white", edgecolor="#1c2330", linewidth=0.8, zorder=5,
        )
        ax.set_title(f"{g.grade}  ({g.viscosity_cp:,.0f} cP)", fontsize=10)
        ax.set_xlabel("API (wt%)")
    axes[0].set_ylabel("HPMC (wt%)")
    if mesh is not None:
        bar = fig.colorbar(mesh, ax=axes.tolist(), shrink=0.9, pad=0.02)
        bar.set_label(
            f"{ra.response.spec.label} ({ra.response.spec.units})", fontsize=9
        )
    fig.suptitle(
        f"{ra.response.spec.label} — lactose is the balance to 100 wt%",
        fontsize=11,
    )
    name = f"doe_contour_{ra.response.spec.key}"
    _save(fig, out, name, banner)
    return name


def surface_3d(ra: ResponseAnalysis, out: Path, banner: str | None) -> str:
    """The same surface in relief, where curvature is easier to see."""
    grids = ra.grids
    lo, hi = ra.scale
    fig = plt.figure(figsize=(4.4 * len(grids), 3.8), layout="constrained")
    for i, g in enumerate(grids, start=1):
        # mypy sees the 2-D Axes signature; a 3-D projection adds
        # plot_surface and set_zlabel at runtime.
        ax: Any = fig.add_subplot(1, len(grids), i, projection="3d")
        z = _z_array(g.z)
        mesh_a, mesh_h = np.meshgrid(g.api_axis, g.hpmc_axis, indexing="xy")
        ax.plot_surface(
            mesh_a, mesh_h, z, cmap="viridis", vmin=lo, vmax=hi,
            linewidth=0, antialiased=True, rstride=2, cstride=2,
        )
        ax.set_xlabel("API (wt%)", fontsize=8)
        ax.set_ylabel("HPMC (wt%)", fontsize=8)
        ax.set_zlabel(ra.response.spec.units, fontsize=8)
        ax.set_title(g.grade, fontsize=10)
        ax.tick_params(labelsize=7)
    fig.suptitle(f"{ra.response.spec.label} — fitted surface", fontsize=11)
    name = f"doe_surface3d_{ra.response.spec.key}"
    _save(fig, out, name, banner)
    return name


def traces(ra: ResponseAnalysis, out: Path, banner: str | None) -> str:
    """Cox response traces: the mixture analogue of a main-effects plot."""
    fig, ax = plt.subplots(figsize=(5.4, 3.7))
    for i, tr in enumerate(ra.traces):
        ax.plot(
            tr.x_values, tr.y_values, lw=2,
            color=_FALLBACK[i % len(_FALLBACK)], label=tr.label,
        )
    ax.set_xlabel("component (wt%)")
    ax.set_ylabel(f"{ra.response.spec.label} ({ra.response.spec.units})")
    ax.set_title("Cox response traces")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.25)
    name = f"doe_traces_{ra.response.spec.key}"
    _save(fig, out, name, banner)
    return name


def interaction(ra: ResponseAnalysis, out: Path, banner: str | None) -> str:
    """Non-parallel lines are the interaction, read straight off the picture."""
    prof = ra.interactions[0]
    fig, ax = plt.subplots(figsize=(5.4, 3.7))
    for i, (label, ys) in enumerate(prof.series):
        ax.plot(prof.x_values, ys, lw=2, color=_colour(label, i), label=label)
    ax.set_xlabel(f"{prof.factor.upper()} (wt%)")
    ax.set_ylabel(f"{ra.response.spec.label} ({ra.response.spec.units})")
    suffix = "" if prof.parallel else " — not parallel"
    ax.set_title(f"{prof.label} interaction{suffix}")
    ax.legend(fontsize=8, frameon=False, title="grade")
    ax.grid(alpha=0.25)
    name = f"doe_interaction_{ra.response.spec.key}"
    _save(fig, out, name, banner)
    return name


def pareto(ra: ResponseAnalysis, out: Path, banner: str | None) -> str:
    """Standardised effects against the significance and Bonferroni lines."""
    eff = ra.ranking.effects
    fig, ax = plt.subplots(figsize=(5.4, max(2.6, 0.28 * len(eff) + 1.1)))
    names = [e.term for e in eff][::-1]
    vals = [e.abs_t for e in eff][::-1]
    colours = ["#2f6fd0" if e.significant else "#b9c2d0" for e in eff][::-1]
    ax.barh(names, vals, color=colours, height=0.68)
    if np.isfinite(ra.ranking.t_critical):
        ax.axvline(
            ra.ranking.t_critical, color="#c0504d", ls="--", lw=1.1,
            label=f"p=0.05  (|t|={ra.ranking.t_critical:.2f})",
        )
    if np.isfinite(ra.ranking.bonferroni_t):
        ax.axvline(
            ra.ranking.bonferroni_t, color="#7d1f1c", ls=":", lw=1.2,
            label=f"Bonferroni  (|t|={ra.ranking.bonferroni_t:.2f})",
        )
    ax.set_xlabel("|standardised effect|")
    ax.set_title(f"Pareto of effects — {ra.response.spec.label}")
    ax.legend(fontsize=7.5, frameon=False, loc="lower right")
    ax.tick_params(axis="y", labelsize=7.5)
    name = f"doe_pareto_{ra.response.spec.key}"
    _save(fig, out, name, banner)
    return name


def half_normal(ra: ResponseAnalysis, out: Path, banner: str | None) -> str:
    """Inert terms fall on a line through the origin; real ones leave it.

    This plot needs no error estimate, which makes it the one to trust when the
    design has few residual degrees of freedom.
    """
    eff = sorted(ra.ranking.effects, key=lambda e: e.abs_t)
    quantiles = list(ra.ranking.half_normal_quantiles)[::-1]
    n = min(len(eff), len(quantiles))

    fig, ax = plt.subplots(figsize=(4.8, 4.0))
    for i in range(n):
        colour = "#c0504d" if eff[i].significant else "#5d6879"
        ax.scatter(quantiles[i], eff[i].abs_t, s=26, color=colour, zorder=3)
        if eff[i].significant:
            ax.annotate(
                eff[i].term, (quantiles[i], eff[i].abs_t), fontsize=6.5,
                xytext=(4, -1), textcoords="offset points",
            )

    inert = [(quantiles[i], eff[i].abs_t) for i in range(n) if not eff[i].significant]
    if len(inert) >= 2:
        xs = np.array([p[0] for p in inert])
        ys = np.array([p[1] for p in inert])
        denominator = float(np.sum(xs**2))
        slope = float(np.sum(xs * ys) / denominator) if denominator > 0 else 0.0
        line = np.linspace(0.0, max(quantiles[:n]) if n else 1.0, 20)
        ax.plot(
            line, slope * line, color="#b9c2d0", ls="--", lw=1,
            label="line through the inert terms",
        )
        ax.legend(fontsize=7.5, frameon=False)

    ax.set_xlabel("half-normal quantile")
    ax.set_ylabel("|standardised effect|")
    ax.set_title(f"Half-normal plot — {ra.response.spec.label}")
    ax.grid(alpha=0.25)
    name = f"doe_halfnormal_{ra.response.spec.key}"
    _save(fig, out, name, banner)
    return name


#: Which responses get the full figure set. Rendering all six for all nine
#: responses would produce 54 figures, most never opened; these are the ones the
#: storyline rests on.
FEATURED: tuple[str, ...] = ("pct_12h", "t50", "pct_24h")


def render_doe_figures(
    analyses: tuple[ResponseAnalysis, ...], out_dir: Path, banner: str | None
) -> list[tuple[str, int, str]]:
    """Render the DoE figure set. Returns (name, rank, caption)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    made: list[tuple[str, int, str]] = []

    for ra in analyses:
        if ra.response.spec.key not in FEATURED or not ra.usable:
            continue
        label = ra.response.spec.label
        made.append((
            contour_panel(ra, out_dir, banner), 2,
            f"Fitted {label.lower()} across the tested composition region at each "
            "grade, on a shared colour scale. White points are the formulations "
            "actually run; blank area is outside the tested region and is not "
            "predicted.",
        ))
        made.append((
            surface_3d(ra, out_dir, banner), 4,
            f"The {label.lower()} surface in relief, where curvature and saddle "
            "regions are easier to see than in contour spacing.",
        ))
        made.append((
            interaction(ra, out_dir, banner), 1,
            f"{label} against HPMC content, drawn once per grade. Lines that fan "
            "apart are the interaction: the effect of polymer content depends on "
            "which grade carries it.",
        ))
        made.append((
            traces(ra, out_dir, banner), 3,
            f"Cox response traces for {label.lower()}. Each varies one component "
            "and lets the others absorb the change in their existing proportions, "
            "which is what happens when a formulation is adjusted.",
        ))
        made.append((
            pareto(ra, out_dir, banner), 5,
            f"Standardised effects on {label.lower()}, against the 5% and "
            "Bonferroni significance lines.",
        ))
        made.append((
            half_normal(ra, out_dir, banner), 6,
            f"Half-normal plot for {label.lower()}. Inert terms lie on the line; "
            "real effects depart from it. Needs no error estimate, so it is the "
            "more trustworthy read when residual degrees of freedom are few.",
        ))
    return made
