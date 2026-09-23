"""Classical DoE figures: contour, 3D surface, traces, interaction, Pareto,
half-normal.

Plots lead, tables support. These answer the questions a formulator brings --
which lever matters, where the sweet spot sits, whether the levers interact --
in a form readable at a glance rather than decoded from a coefficient table.

Each figure is split into a ``draw_*`` function that paints onto axes it is
given and a thin wrapper that makes and saves the standalone figure, so the
headline composites reuse exactly the same drawing code.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from pipeline.doe.analysis import ResponseAnalysis
from pipeline.figures import publication as pub
from pipeline.figures.publication import FigureRecord


def _z_array(grid_rows: tuple[tuple[float | None, ...], ...]) -> np.ndarray:
    return np.array(
        [[np.nan if v is None else v for v in row] for row in grid_rows], dtype=float
    )


def _response_label(ra: ResponseAnalysis) -> str:
    spec = ra.response.spec
    return f"{spec.label} ({spec.units})" if spec.units else spec.label


#: How many of the largest significant effects the half-normal plot names.
HALF_NORMAL_LABELS = 6


# --- drawing primitives ------------------------------------------------------


def draw_contours(fig: Figure, axes: Sequence[Axes], ra: ResponseAnalysis) -> None:
    """One contour per grade on a shared colour scale, with one colourbar.

    The shared scale is deliberate. Scaling each panel to its own range makes
    every grade look equally variable and hides the thing worth seeing: that one
    grade spans far more of the response than another.
    """
    lo, hi = ra.scale
    levels: Any = (
        ticker.MaxNLocator(nbins=10).tick_values(lo, hi)
        if np.isfinite(lo) and hi > lo else 10
    )
    mesh = None
    for ax, g in zip(axes, ra.grids, strict=False):
        z = _z_array(g.z)
        mesh = ax.contourf(g.api_axis, g.hpmc_axis, z, levels=levels, cmap="viridis")
        lines = ax.contour(
            g.api_axis, g.hpmc_axis, z, levels=levels,
            colors="white", linewidths=0.4, alpha=0.7,
        )
        ax.clabel(lines, inline=True, fontsize=5.5, fmt=lambda v: f"{v:g}")
        ax.scatter(
            [p[0] for p in g.design_points], [p[1] for p in g.design_points],
            s=14, facecolor="white", edgecolor="black", linewidth=0.5, zorder=5,
            clip_on=False,
        )
        pub.header_note(ax, f"{g.grade} ({g.viscosity_cp:,.0f} cP)")
        ax.set_xlabel("API (wt%)")
        pub.auto_minor(ax)
    axes[0].set_ylabel("HPMC (wt%)")
    for ax in axes[1:]:
        ax.tick_params(labelleft=False)
    if mesh is not None:
        bar = fig.colorbar(mesh, ax=list(axes), shrink=0.95, pad=0.02, aspect=25)
        bar.set_label(_response_label(ra))
        bar.ax.tick_params(which="both", direction="in")


def draw_pareto(ax: Axes, ra: ResponseAnalysis) -> None:
    """Standardised effects against the 5 % and Bonferroni lines."""
    eff = ra.ranking.effects
    names = [e.term for e in eff][::-1]
    vals = [e.abs_t for e in eff][::-1]
    colours = [pub.OKABE_ITO[0] if e.significant else "#C8C8C8" for e in eff][::-1]
    ax.barh(names, vals, color=colours, height=0.65, edgecolor="black", linewidth=0.4)
    if np.isfinite(ra.ranking.t_critical):
        ax.axvline(
            ra.ranking.t_critical, color=pub.HIGHLIGHT, ls="--", lw=0.8,
            label=f"p = 0.05 (|t| = {ra.ranking.t_critical:.2f})",
        )
    if np.isfinite(ra.ranking.bonferroni_t):
        ax.axvline(
            ra.ranking.bonferroni_t, color=pub.INK, ls=":", lw=0.9,
            label=f"Bonferroni (|t| = {ra.ranking.bonferroni_t:.2f})",
        )
    ax.set_xlim(0, max([*vals, ra.ranking.bonferroni_t if np.isfinite(ra.ranking.bonferroni_t)
                        else 0.0]) * 1.08)
    ax.set_ylim(-0.6, len(names) - 0.4)
    ax.set_xlabel("|Standardised effect|")
    pub.categorical(ax, "y")
    ax.legend(loc="lower right")


def draw_interaction(ax: Axes, ra: ResponseAnalysis) -> None:
    """Non-parallel lines are the interaction, read straight off the picture."""
    prof = ra.interactions[0]
    n = len(prof.x_values)
    for i, (label, ys) in enumerate(prof.series):
        st = pub.grade_style(label, i)
        ax.plot(
            prof.x_values, ys, **st.line(), markevery=max(1, n // 6),
            markerfacecolor="white", markeredgewidth=0.8, label=label,
        )
    ax.set_xlabel(f"{prof.factor.upper()} (wt%)")
    ax.set_ylabel(_response_label(ra))
    pub.auto_minor(ax)
    ax.legend(title="Grade")


def draw_traces(ax: Axes, ra: ResponseAnalysis) -> None:
    """Cox response traces: the mixture analogue of a main-effects plot."""
    styles = ("-", "--", "-.", ":")
    for i, tr in enumerate(ra.traces):
        ax.plot(
            tr.x_values, tr.y_values, color=pub.series_colour(i + 3),
            linestyle=styles[i % len(styles)], label=tr.label.replace("Hpmc", "HPMC"),
        )
    ax.set_xlabel("Component (wt%)")
    ax.set_ylabel(_response_label(ra))
    pub.auto_minor(ax)
    ax.legend()


def draw_half_normal(ax: Axes, ra: ResponseAnalysis) -> None:
    """Inert terms fall on a line through the origin; real ones leave it."""
    eff = sorted(ra.ranking.effects, key=lambda e: e.abs_t)
    quantiles = list(ra.ranking.half_normal_quantiles)[::-1]
    n = min(len(eff), len(quantiles))
    sig = [i for i in range(n) if eff[i].significant]
    inert_idx = [i for i in range(n) if not eff[i].significant]

    ax.scatter(
        [quantiles[i] for i in inert_idx], [eff[i].abs_t for i in inert_idx],
        s=14, facecolor="white", edgecolor=pub.INK, linewidth=0.7, zorder=3,
        label="not significant",
    )
    ax.scatter(
        [quantiles[i] for i in sig], [eff[i].abs_t for i in sig],
        s=16, color=pub.HIGHLIGHT, marker="s", zorder=3, label="significant (p < 0.05)",
    )
    # Only the largest effects are named; below them the labels would pile up.
    # Their vertical positions are spread so neighbours never overprint.
    named = sorted(sig, key=lambda i: eff[i].abs_t)[-HALF_NORMAL_LABELS:]
    top = max((e.abs_t for e in eff[:n]), default=1.0) * 1.15
    ys = pub.spread_labels([eff[i].abs_t for i in named], 0.055 * top)
    for i, ly in zip(named, ys, strict=True):
        ax.annotate(
            eff[i].term, (quantiles[i], eff[i].abs_t), xytext=(quantiles[i] + 0.06, ly),
            fontsize=6, va="center",
            arrowprops={"arrowstyle": "-", "color": pub.MUTED, "lw": 0.3,
                        "shrinkA": 0, "shrinkB": 2},
        )
    if len(inert_idx) >= 2:
        qx = np.array([quantiles[i] for i in inert_idx])
        qy = np.array([eff[i].abs_t for i in inert_idx])
        denominator = float(np.sum(qx**2))
        slope = float(np.sum(qx * qy) / denominator) if denominator > 0 else 0.0
        line = np.linspace(0.0, max(quantiles[:n]) if n else 1.0, 20)
        ax.plot(line, slope * line, color=pub.MUTED, ls="--", lw=0.8,
                label="fit through inert terms")
    ax.set_xlim(0, (max(quantiles[:n]) if n else 1.0) * 1.35)
    ax.set_ylim(0, top)
    ax.set_xlabel("Half-normal quantile")
    ax.set_ylabel("|Standardised effect|")
    pub.auto_minor(ax)
    ax.legend(loc="upper left")


# --- standalone figures ------------------------------------------------------


def contour_panel(ra: ResponseAnalysis, out: Path, banner: str | None) -> str:
    n = len(ra.grids)
    fig, axes = pub.new_figure(pub.DOUBLE, 2.5, ncols=n, sharey=True)
    axes = list(np.atleast_1d(axes))
    draw_contours(fig, axes, ra)
    pub.label_panels(axes)
    return pub.save(fig, out, f"doe_contour_{ra.response.spec.key}", banner=banner)


def surface_3d(ra: ResponseAnalysis, out: Path, banner: str | None) -> str:
    """The same surface in relief, where curvature is easier to see."""
    pub.apply_style()
    lo, hi = ra.scale
    n = len(ra.grids)
    fig = plt.figure(figsize=(pub.DOUBLE, 2.6), layout="constrained")
    axes = []
    for i, g in enumerate(ra.grids, start=1):
        ax: Any = fig.add_subplot(1, n, i, projection="3d")
        z = _z_array(g.z)
        mesh_a, mesh_h = np.meshgrid(g.api_axis, g.hpmc_axis, indexing="xy")
        ax.plot_surface(
            mesh_a, mesh_h, z, cmap="viridis", vmin=lo, vmax=hi,
            linewidth=0, antialiased=True, rstride=2, cstride=2,
        )
        ax.set_xlabel("API (wt%)", labelpad=-2)
        ax.set_ylabel("HPMC (wt%)", labelpad=-2)
        ax.set_zlabel(_response_label(ra), labelpad=-1)
        ax.tick_params(labelsize=6, pad=-1)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.set_minor_locator(ticker.NullLocator())
            axis.set_major_locator(ticker.MaxNLocator(5))
            axis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            axis._axinfo["grid"]["linewidth"] = 0.3
            axis._axinfo["grid"]["color"] = (0.8, 0.8, 0.8, 1.0)
        pub.corner_note(ax, g.grade, "upper right")
        axes.append(ax)
    pub.label_panels(axes)
    return pub.save(fig, out, f"doe_surface3d_{ra.response.spec.key}", banner=banner)


def traces(ra: ResponseAnalysis, out: Path, banner: str | None) -> str:
    fig, ax = pub.new_figure(pub.SINGLE)
    draw_traces(ax, ra)
    return pub.save(fig, out, f"doe_traces_{ra.response.spec.key}", banner=banner)


def interaction(ra: ResponseAnalysis, out: Path, banner: str | None) -> str:
    fig, ax = pub.new_figure(pub.SINGLE)
    draw_interaction(ax, ra)
    return pub.save(fig, out, f"doe_interaction_{ra.response.spec.key}", banner=banner)


def pareto(ra: ResponseAnalysis, out: Path, banner: str | None) -> str:
    eff = ra.ranking.effects
    fig, ax = pub.new_figure(pub.SINGLE, max(2.2, 0.2 * len(eff) + 0.9))
    draw_pareto(ax, ra)
    return pub.save(fig, out, f"doe_pareto_{ra.response.spec.key}", banner=banner)


def half_normal(ra: ResponseAnalysis, out: Path, banner: str | None) -> str:
    fig, ax = pub.new_figure(pub.SINGLE, 3.0)
    draw_half_normal(ax, ra)
    return pub.save(fig, out, f"doe_halfnormal_{ra.response.spec.key}", banner=banner)


#: Which responses get the full figure set. Rendering all six for all nine
#: responses would produce 54 figures, most never opened; these are the ones the
#: storyline rests on.
FEATURED: tuple[str, ...] = ("pct_12h", "t50", "pct_24h")


def _significant_terms(ra: ResponseAnalysis) -> str:
    sig = [e.term for e in ra.ranking.effects if e.significant]
    if not sig:
        return "No term clears the 5 % line."
    return f"Terms clearing the 5 % line: {', '.join(sig)}."


def render_doe_figures(
    analyses: tuple[ResponseAnalysis, ...], out_dir: Path, banner: str | None
) -> list[FigureRecord]:
    """Render the DoE figure set."""
    out_dir.mkdir(parents=True, exist_ok=True)
    made: list[FigureRecord] = []

    for ra in analyses:
        key = ra.response.spec.key
        if key not in FEATURED or not ra.usable:
            continue
        label = ra.response.spec.label
        prof = ra.interactions[0] if ra.interactions else None
        fan = (
            "The lines are close to parallel, so the HPMC effect barely depends on grade."
            if prof is not None and prof.parallel
            else "The lines are not parallel: the HPMC effect depends on which grade "
            "carries it."
        )
        made += [
            FigureRecord(
                f"DOE-interaction-{key}", "doe", 10,
                f"{label} against HPMC content predicted by the fitted model, one line "
                f"per grade. {fan}",
                interaction(ra, out_dir, banner),
            ),
            FigureRecord(
                f"DOE-contour-{key}", "doe", 11,
                f"Fitted {label.lower()} across the tested composition region at each "
                "grade (panels), on a shared colour scale. Open circles are the "
                "formulations actually run; blank area lies outside the tested region "
                "and is not predicted. Lactose is the balance to 100 wt%.",
                contour_panel(ra, out_dir, banner),
            ),
            FigureRecord(
                f"DOE-traces-{key}", "doe", 12,
                f"Cox response traces for {label.lower()}. Each varies one component "
                "and lets the others absorb the change in their existing proportions, "
                "which is what happens when a formulation is adjusted.",
                traces(ra, out_dir, banner),
            ),
            FigureRecord(
                f"DOE-pareto-{key}", "doe", 13,
                f"Standardised effects on {label.lower()}, against the 5 % (dashed) and "
                f"Bonferroni (dotted) significance lines. {_significant_terms(ra)}",
                pareto(ra, out_dir, banner),
            ),
            FigureRecord(
                f"DOE-halfnormal-{key}", "doe", 14,
                f"Half-normal plot for {label.lower()}. Inert terms lie on the dashed "
                "line through the origin; real effects depart from it (the "
                f"{HALF_NORMAL_LABELS} largest are named). Needs no error "
                "estimate, so it is the more trustworthy read when residual degrees of "
                "freedom are few.",
                half_normal(ra, out_dir, banner),
            ),
            FigureRecord(
                f"DOE-surface3d-{key}", "supplementary", 30,
                f"The fitted {label.lower()} surface in relief for each grade, where "
                "curvature and saddle regions are easier to see than in contour spacing.",
                surface_3d(ra, out_dir, banner),
            ),
        ]
    return made
