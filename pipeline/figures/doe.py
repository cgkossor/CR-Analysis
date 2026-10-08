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
import pandas as pd
from matplotlib import ticker
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from pipeline import config
from pipeline.doe import ternary, views
from pipeline.doe.analysis import ResponseAnalysis
from pipeline.doe.measured import MeasuredPoint
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


def draw_interaction(
    ax: Axes, ra: ResponseAnalysis, measured: Sequence[MeasuredPoint] | None = None
) -> None:
    """Model lines per grade, over the measured formulations they were fitted to.

    Lines are predictions along one slice through the reference composition.
    Points are measured formulation means (+/- 1 SD over replicates) at their
    own composition, so they scatter about the slice; a line that runs away
    from its grade's points is a model that does not describe the data.
    """
    prof = ra.interactions[0]
    factor = f"{prof.factor}_wt"
    grades = [label for label, _ in prof.series]
    for i, grade in enumerate(grades):
        st = pub.grade_style(grade, i)
        pts = [m for m in (measured or ()) if m.grade == grade]
        if pts:
            xs = [getattr(m, factor) for m in pts]
            means = [m.mean for m in pts]
            err = [m.sd if np.isfinite(m.sd) else 0.0 for m in pts]
            ax.errorbar(xs, means, yerr=err, fmt=st.marker, color=st.colour, ms=3.2,
                        alpha=0.55, elinewidth=0.6, capsize=1.5, zorder=2)
    for i, (label, ys) in enumerate(prof.series):
        st = pub.grade_style(label, i)
        ax.plot(prof.x_values, ys, color=st.colour, linestyle=st.linestyle, lw=1.4,
                label=label, zorder=3)
    ax.set_xlabel(f"{prof.factor.upper()} (wt%)")
    ax.set_ylabel(_response_label(ra))
    pub.auto_minor(ax)
    handles, labels = ax.get_legend_handles_labels()
    if measured:
        handles += [Line2D([], [], color=pub.INK, lw=1.4),
                    Line2D([], [], color=pub.INK, marker="o", ls="none", ms=3.2, alpha=0.55)]
        labels += ["model prediction", "measured mean ± SD"]
    ax.legend(handles, labels, title="Grade", fontsize=6)


def draw_traces(ax: Axes, traces: Sequence[Any], ra: ResponseAnalysis,
                legend: bool = True,
                measured: Sequence[MeasuredPoint] | None = None) -> None:
    """Cox response traces: the mixture analogue of a main-effects plot.

    An open circle marks the reference formulation on each line, the one point
    all three traces share.
    """
    styles = ("-", "--", "-.", ":")
    # The grade's measured range, behind the model slice. One composition axis
    # cannot place every formulation for three components at once, so the
    # band shows how far the data run at this grade instead.
    if measured:
        vals = [m.mean for m in measured]
        ax.axhspan(min(vals), max(vals), color=pub.MUTED, alpha=0.15, lw=0, zorder=0,
                   label="measured range, this grade")
    for i, tr in enumerate(traces):
        colour = pub.series_colour(i + 3)
        ax.plot(
            tr.x_values, tr.y_values, color=colour,
            linestyle=styles[i % len(styles)], label=tr.label.replace("Hpmc", "HPMC"),
        )
        ref = tr.reference_point.get(tr.factor)
        if ref is not None:
            x0 = float(ref)
            ax.plot(x0, float(np.interp(x0, tr.x_values, tr.y_values)), "o", color=colour,
                    markerfacecolor="white", markeredgewidth=0.8, markersize=3.5, zorder=4)
    ax.set_xlabel("Component (wt%)")
    ax.set_ylabel(_response_label(ra))
    pub.auto_minor(ax)
    if legend:
        ax.legend(title="Model prediction, varying", fontsize=6)


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


def traces(ra: ResponseAnalysis, out: Path, banner: str | None,
           measured: Sequence[MeasuredPoint] | None = None) -> str:
    """One panel per grade on a shared axis, so no grade's slice stands for all."""
    by_grade = ra.traces_by_grade or ((ra.trace_grade, ra.traces),)
    fig, axes = pub.new_figure(pub.DOUBLE, 2.5, ncols=len(by_grade), sharey=True)
    axes = list(np.atleast_1d(axes))
    for i, (ax, (grade, trs)) in enumerate(zip(axes, by_grade, strict=True)):
        draw_traces(ax, trs, ra, legend=i == 0,
                    measured=[m for m in (measured or ()) if m.grade == grade])
        pub.header_note(ax, grade)
        if i:
            ax.set_ylabel("")
    if len(axes) > 1:
        pub.label_panels(axes)
    return pub.save(fig, out, f"doe_traces_{ra.response.spec.key}", banner=banner)


def _trace_caption(ra: ResponseAnalysis) -> str:
    label = ra.response.spec.label
    ref = ra.traces[0].reference_point if ra.traces else {}
    blend = (
        f"{float(ref['api']):.3g} / {float(ref['hpmc']):.3g} / {float(ref['lactose']):.3g} "
        "wt% API / HPMC / lactose"
        if {"api", "hpmc", "lactose"} <= set(ref) else "the design centroid"
    )
    values = ra.response.values[ra.response.available]
    measured = (
        f" For comparison, measured {label.lower()} spans {float(values.min()):.3g} to "
        f"{float(values.max()):.3g} {ra.response.spec.units} across the "
        f"{values.size} formulations in the model."
        if values.size else ""
    )
    return (
        f"Cox response traces for {label.lower()}, one panel per grade. How to read: these "
        f"are model predictions, not measured data. Each line starts from one reference "
        f"formulation ({blend}; open circle) and varies one component while the other two "
        "keep their ratio, which is what happens when a formulation is adjusted. A flat "
        "trace means that component barely moves the response at that grade; it does not "
        f"mean every run behaved that way.{measured}"
    )


def interaction(ra: ResponseAnalysis, out: Path, banner: str | None,
                measured: Sequence[MeasuredPoint] | None = None) -> str:
    fig, ax = pub.new_figure(pub.SINGLE)
    draw_interaction(ax, ra, measured)
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


#: Which responses get the figure set: t50 and every fixed-time % released.
FEATURED: tuple[str, ...] = (
    "t50", "pct_1h", "pct_2h", "pct_4h", "pct_8h", "pct_12h", "pct_24h",
)


def _significant_terms(ra: ResponseAnalysis) -> str:
    sig = [e.term for e in ra.ranking.effects if e.significant]
    if not sig:
        return "No term clears the 5 % line."
    return f"Terms clearing the 5 % line: {', '.join(sig)}."


def _levels(lo: float, hi: float) -> Any:
    return (ticker.MaxNLocator(nbins=10).tick_values(lo, hi)
            if np.isfinite(lo) and np.isfinite(hi) and hi > lo else 10)


def _draw_triangle(ax: Axes, reg: ternary.Region) -> None:
    """Outline, real-wt% grid lines and corner labels of the pseudocomponent triangle."""
    c = ternary.CORNERS
    ax.plot([*c[:, 0], c[0, 0]], [*c[:, 1], c[0, 1]], color=pub.INK, lw=0.8, zorder=4)
    centre = c.mean(axis=0)
    for t in ternary.ticks(reg):
        (x0, y0), (x1, y1) = t.ends
        ax.plot([x0, x1], [y0, y1], color="white", lw=0.35, alpha=0.6, zorder=3)
        # One edge per component: lactose on the left, API along the bottom,
        # HPMC on the right.
        if t.component == "hpmc":
            x0, y0 = x1, y1
        # Value at the edge where the line leaves the triangle, pushed outward,
        # in the component's own colour so the three families can be told apart.
        out = np.array([x0, y0]) - centre
        out = out / (np.linalg.norm(out) or 1.0) * 0.045
        i = ternary.COMPONENTS.index(t.component)
        ax.text(x0 + out[0], y0 + out[1], f"{t.value:g}",
                ha="center", va="center", fontsize=5, color=pub.series_colour(i + 3))
    for i, name in enumerate(ternary.COMPONENTS):
        hi = (reg.lower[i] + reg.span) * 100.0
        x, y = c[i]
        ha = {0: "right", 1: "left", 2: "center"}[i]
        dy = {0: -0.05, 1: -0.05, 2: 0.03}[i]
        ax.text(x, y + dy, f"{ternary.LABELS[name]} {hi:.0f}%", ha=ha, va="center",
                fontsize=6.5, color=pub.series_colour(i + 3))
    ax.set_aspect("equal")
    ax.set_xlim(-0.12, 1.12)
    ax.set_ylim(-0.1, 0.95)
    ax.axis("off")


def draw_ternary(fig: Figure, axes: Sequence[Axes], ra: ResponseAnalysis,
                 view: views.TernaryView) -> None:
    """Ternary contours per grade; measured blends coloured on the same scale."""
    import matplotlib.colors as mcolors
    from matplotlib.tri import Triangulation

    finite = [g.value[np.isfinite(g.value)] for g in view.grids]
    measured = [v for *_, v in view.points]
    every = np.concatenate([*finite, np.asarray(measured, dtype=float)])
    lo, hi = (float(np.nanmin(every)), float(np.nanmax(every))) if every.size else (0, 1)
    levels = _levels(lo, hi)
    norm = mcolors.Normalize(lo, hi)
    mesh = None
    for ax, g in zip(axes, view.grids, strict=False):
        if len(g.triangles):
            tri = Triangulation(g.xy[:, 0], g.xy[:, 1], g.triangles)
            z = np.where(np.isfinite(g.value), g.value, np.nanmean(g.value))
            mesh = ax.tricontourf(tri, z, levels=levels, cmap="viridis", norm=norm)
            ax.tricontour(tri, z, levels=levels, colors="white", linewidths=0.3, alpha=0.6)
        _draw_triangle(ax, view.region)
        pts = [(x, y, v) for x, y, grade, v in view.points if grade == g.grade]
        if pts:
            ax.scatter([q[0] for q in pts], [q[1] for q in pts], c=[q[2] for q in pts],
                       cmap="viridis", norm=norm, s=22, edgecolor="black", linewidth=0.6,
                       zorder=6)
        pub.header_note(ax, g.grade)
    if mesh is not None:
        bar = fig.colorbar(mesh, ax=list(axes), shrink=0.85, pad=0.02, aspect=25)
        bar.set_label(_response_label(ra))
        bar.ax.tick_params(which="both", direction="in")


def draw_pred_actual(ax: Axes, ra: ResponseAnalysis,
                     pairs: Sequence[tuple[str, int, float, float]],
                     order: Sequence[str] = (), compact: bool = False) -> None:
    """Measured against fitted, by grade, with the 1:1 line.

    ``compact`` shortens the labels for a small panel inside a composite
    figure: the response is named by the neighbouring panels' colour bar.
    """
    present = set(g for g, *_ in pairs)
    grades = [g for g in order if g in present] + sorted(present - set(order))
    for i, grade in enumerate(grades):
        st = pub.grade_style(grade, i)
        pts = [(m, f) for g, _, m, f in pairs if g == grade]
        ax.plot([f for _, f in pts], [m for m, _ in pts], ls="none", marker=st.marker,
                color=st.colour, ms=4, label=grade)
    vals = [v for *_, m, f in pairs for v in (m, f)]
    if vals:
        lo, hi = min(vals), max(vals)
        pad = (hi - lo) * 0.05 or 1.0
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=pub.INK, lw=0.7, ls="--")
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
    units = ra.response.spec.units
    if compact:
        ax.set_xlabel(f"Predicted ({units})" if units else "Predicted")
        ax.set_ylabel(f"Measured ({units})" if units else "Measured")
        pub.corner_note(ax, f"R² {ra.fit.r_squared:.2f}\nQ² {ra.fit.pred_r_squared:.2f}",
                        "upper left")
    else:
        ax.set_xlabel(f"Predicted {_response_label(ra)}")
        ax.set_ylabel(f"Measured {_response_label(ra)}")
        pub.corner_note(ax, f"R² = {ra.fit.r_squared:.2f}, predicted R² = "
                            f"{ra.fit.pred_r_squared:.2f}", "upper left")
    ax.set_aspect("equal", adjustable="box")
    pub.auto_minor(ax)
    ax.legend(loc="lower right", title=None if compact else "Grade", fontsize=6)


def draw_piepel(axes: Sequence[Axes], ra: ResponseAnalysis,
                traces_by_grade: dict[str, list[views.PiepelTrace]],
                on_trace: Sequence[views.TracePoint] = ()) -> None:
    """Piepel traces per grade, inside the tested region, with the measured
    blends that lie on each trace line (and only those)."""
    styles = ("-", "--", "-.")
    markers = ("o", "s", "^")
    for k, (ax, (grade, trs)) in enumerate(zip(axes, traces_by_grade.items(), strict=False)):
        for i, tr in enumerate(trs):
            y = np.where(tr.inside, tr.y, np.nan)
            colour = pub.series_colour(i + 3)
            ax.plot(tr.x_wt, y, color=colour, ls=styles[i],
                    label=ternary.LABELS[tr.component])
            pts = [p for p in on_trace if p.grade == grade and p.component == tr.component]
            if pts:
                ax.errorbar([p.x_wt for p in pts], [p.mean for p in pts],
                            yerr=[p.sd if np.isfinite(p.sd) else 0.0 for p in pts],
                            fmt=markers[i], color=colour, ms=4, mfc="white", mew=1.0,
                            elinewidth=0.7, capsize=2, zorder=5)
        pub.header_note(ax, grade)
        ax.set_xlabel("Component (wt%)")
        if k == 0:
            ax.set_ylabel(_response_label(ra))
            handles, labels = ax.get_legend_handles_labels()
            if on_trace:
                handles.append(Line2D([], [], color=pub.INK, marker="o", mfc="white", ls="none",
                                      ms=4))
                labels.append("measured blend on the line")
            ax.legend(handles, labels, title="Model, varying", fontsize=6)
        pub.auto_minor(ax)


def ternary_figure(ra: ResponseAnalysis, design_points: pd.DataFrame, out: Path,
                   banner: str | None) -> str:
    view = views.ternary_view(ra, design_points)
    fig, axes = pub.new_figure(pub.DOUBLE, 2.6, ncols=len(view.grids))
    axes = list(np.atleast_1d(axes))
    draw_ternary(fig, axes, ra, view)
    pub.label_panels(axes)
    return pub.save(fig, out, f"doe_ternary_{ra.response.spec.key}", banner=banner)


def pred_actual_figure(ra: ResponseAnalysis, design_points: pd.DataFrame, out: Path,
                       banner: str | None) -> str:
    fig, ax = pub.new_figure(pub.SINGLE, 3.0)
    draw_pred_actual(ax, ra, views.fitted_pairs(ra, design_points),
                     [g for g, _ in views.grade_levels(design_points)])
    return pub.save(fig, out, f"doe_predicted_actual_{ra.response.spec.key}", banner=banner)


def piepel_figure(ra: ResponseAnalysis, design_points: pd.DataFrame, out: Path,
                  banner: str | None, replicates: pd.DataFrame | None = None) -> str:
    traces_by_grade = views.piepel_traces(ra, design_points)
    on_trace = views.points_on_traces(ra, design_points, replicates,
                                      config.TRACE_POINT_TOL_WT)
    fig, axes = pub.new_figure(pub.DOUBLE, 2.4, ncols=len(traces_by_grade), sharey=True)
    axes = list(np.atleast_1d(axes))
    draw_piepel(axes, ra, traces_by_grade, on_trace)
    pub.label_panels(axes)
    return pub.save(fig, out, f"doe_piepel_{ra.response.spec.key}", banner=banner)


def render_doe_figures(
    analyses: tuple[ResponseAnalysis, ...], out_dir: Path, banner: str | None,
    design_points: pd.DataFrame | None = None, replicates: pd.DataFrame | None = None,
) -> list[FigureRecord]:
    """The standard mixture-DoE figure set for each featured response.

    Ternary contours with the measured blends, predicted vs actual, and the
    Pareto chart of effects; Piepel traces as a supporting (methods) view.
    ``replicates`` is accepted for callers that pass it and is not needed here.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    made: list[FigureRecord] = []
    if design_points is None:
        return made
    for ra in analyses:
        key = ra.response.spec.key
        if key not in FEATURED or not ra.usable:
            continue
        label = ra.response.spec.label
        made += [
            FigureRecord(
                f"DOE-ternary-{key}", "doe", 10,
                f"{label} across the tested blends, one triangle per grade. Each corner "
                "is one component at its highest possible level with the other two at "
                "their lowest tested levels (L-pseudocomponents); every blend is a point "
                "inside, and the coloured numbers on each edge give that component's wt% "
                "along the white grid lines. "
                "Background: the fitted model, on one colour scale shared by all panels "
                "and the points. Points: the measured blends, filled with their "
                "measured value, so a point that stands out from its surroundings is "
                "one the model does not fit. Blank: outside the tested region.",
                ternary_figure(ra, design_points, out_dir, banner),
            ),
            FigureRecord(
                f"DOE-predicted-actual-{key}", "doe", 11,
                f"Measured against model-predicted {label.lower()} for every "
                "formulation, by grade. Points on the dashed 1:1 line are predicted "
                "exactly; the spread about it is the model's error. R² describes the "
                "fit; predicted R² (leave-one-out) describes how well it predicts a "
                "formulation it has not seen.",
                pred_actual_figure(ra, design_points, out_dir, banner),
            ),
            FigureRecord(
                f"DOE-pareto-{key}", "doe", 13,
                f"Standardised effects on {label.lower()}, against the 5 % (dashed) and "
                f"Bonferroni (dotted) significance lines. {_significant_terms(ra)}",
                pareto(ra, out_dir, banner),
            ),
            FigureRecord(
                f"DOE-piepel-{key}", "doe", 12,
                f"Piepel response traces for {label.lower()}, one panel per grade. Each "
                "line is the model's prediction as one component rises from the reference "
                "blend (the average tested blend) along its Piepel direction, the other "
                "two keeping their ratio above their lowest tested levels; a steep line is "
                "a component that moves the response. Open symbols (+/-1 SD) are the "
                "measured blends lying on that trace line, within "
                f"{config.TRACE_POINT_TOL_WT:g} wt%; blends off the line are not shown, "
                "because their value belongs to a different blend.",
                piepel_figure(ra, design_points, out_dir, banner, replicates),
            ),
        ]
    return made
