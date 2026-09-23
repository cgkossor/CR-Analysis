"""Journal-style figures for the disintegration section (DT-01 … DT-08).

Built on ``pipeline.figures.publication``, so they match the dissolution figures:
closed box, inward mirrored ticks, no titles, (A)/(B) panel letters, grade as
colour + marker + line style. The DoE panels reuse the dissolution drawing code
(``pipeline.figures.doe``) rather than copying it.

Captions are written from the computed numbers, with wording that follows them.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import ticker
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from scipy import stats

from pipeline.disintegration import report, settings
from pipeline.disintegration.analysis import DisintegrationAnalysis
from pipeline.disintegration.correlation import METRICS, _pair
from pipeline.figures import publication as pub
from pipeline.figures.doe import draw_contours, draw_pareto
from pipeline.figures.publication import FigureRecord

SECTION = "disintegration"


def _grade_frame(
    r: DisintegrationAnalysis, grade: str, censored: bool | None = False
) -> pd.DataFrame:
    m = r.matched[r.matched["grade"] == grade]
    if censored is None:
        return m
    return m[m["dt_censored"] == censored]


def _band(x: np.ndarray, y: np.ndarray, grid: np.ndarray) -> tuple[np.ndarray, ...] | None:
    """OLS line and 95% confidence band for the mean, on ``grid``."""
    n = x.size
    if n < 3 or np.ptp(x) == 0:
        return None
    b, a = np.polyfit(x, y, 1)
    resid = y - (a + b * x)
    s = np.sqrt(np.sum(resid**2) / (n - 2))
    sxx = np.sum((x - x.mean()) ** 2)
    half = stats.t.ppf(0.975, n - 2) * s * np.sqrt(1 / n + (grid - x.mean()) ** 2 / sxx)
    fit = a + b * grid
    return fit, fit - half, fit + half


def _plain_log(ax: Axes, which: str = "y") -> None:
    """Log-axis labels as plain numbers (0.5, 1, 2, 10), with 1-2-5 majors on short ranges.

    Call after the limits are final.
    """
    axis = ax.yaxis if which == "y" else ax.xaxis
    lo, hi = ax.get_ylim() if which == "y" else ax.get_xlim()
    if hi / lo < 30:
        axis.set_major_locator(ticker.LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    axis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    axis.set_minor_formatter(ticker.NullFormatter())


def _censored_legend(ax: Axes) -> None:
    ax.scatter([], [], marker="^", facecolor="white", edgecolor=pub.INK, s=16,
               label="censored (still intact at test end)")


# --- DT-01 ------------------------------------------------------------------
def fig_dt_vs_td(r: DisintegrationAnalysis, out: Path, banner: str | None,
                 formats: Sequence[str]) -> FigureRecord:
    fig, ax = pub.new_figure(pub.SINGLE, aspect=0.9)
    xs_all: list[float] = []
    for i, g in enumerate(r.grade_order):
        st = pub.grade_style(g, i)
        m = _grade_frame(r, g)
        ln_td = m["ln_td"].to_numpy(dtype=float)
        ln_dt = m["ln_dt"].to_numpy(dtype=float)
        td_sd = (m["log10_td_sd"] * np.log(10)).to_numpy(dtype=float)
        dt_sd = m["ln_sd"].to_numpy(dtype=float)
        x, y = np.exp(ln_td), np.exp(ln_dt)
        ax.errorbar(
            x, y,
            xerr=[x - np.exp(ln_td - td_sd), np.exp(ln_td + td_sd) - x],
            yerr=[y - np.exp(ln_dt - dt_sd), np.exp(ln_dt + dt_sd) - y],
            fmt=st.marker, color=st.colour, markerfacecolor=st.colour, markeredgecolor="black",
            markeredgewidth=0.4, elinewidth=0.5, capsize=1.2, ms=4, label=g, zorder=3,
        )
        xs_all += list(x)
        if len(ln_td) >= 3:
            grid = np.linspace(ln_td.min(), ln_td.max(), 40)
            band = _band(ln_td, ln_dt, grid)
            if band is not None:
                fit, lo, hi = band
                ax.plot(np.exp(grid), np.exp(fit), color=st.colour, ls=st.linestyle, lw=1.0)
                ax.fill_between(np.exp(grid), np.exp(lo), np.exp(hi), color=st.colour,
                                alpha=0.15, lw=0)
        cens = _grade_frame(r, g, censored=True)
        if len(cens):
            cx = np.exp(cens["ln_td"].to_numpy(dtype=float))
            xs_all += list(cx)
            ax.scatter(cx, np.full(len(cx), r.data.test_end_h), marker="^", s=18,
                       facecolor="white", edgecolor=st.colour, linewidth=0.8, zorder=3)
    if len(r.censored_points):
        _censored_legend(ax)
    pub.log_axis(ax, "x")
    pub.log_axis(ax, "y")
    lo_lim = min(min(xs_all), float(r.matched["dt_h"].min())) / 1.5
    hi_lim = max(max(xs_all), r.data.test_end_h) * 1.5
    ax.plot([lo_lim, hi_lim], [lo_lim, hi_lim], color=pub.MUTED, ls=":", lw=0.8,
            label="DT = Td", zorder=1)
    ax.set_xlim(lo_lim, hi_lim)
    ax.set_ylim(lo_lim, hi_lim)
    _plain_log(ax, "x")
    _plain_log(ax, "y")
    ax.set_xlabel("Dissolution time scale, Weibull Td (h)")
    ax.set_ylabel("Disintegration time, DT (h)")
    ax.legend(loc="upper left", fontsize=6.5)

    a = r.grades.ancova
    within = r.correlation.within_grade
    cap = (
        "Disintegration time against the Weibull dissolution time scale, one point per "
        "formulation (geometric mean; bars ±1 replicate SD on the log scale). Lines are "
        "per-grade least-squares fits with 95% confidence bands; the dotted line is DT = Td. "
    )
    if within:
        cap += (
            "Within-grade Pearson r = "
            + ", ".join(f"{w.grade} {w.pearson:.2f}" for w in within) + ". "
        )
    if a is not None:
        cap += (
            "ANCOVA: grades "
            f"{'share a slope' if a.selected == 'common slope' else 'differ in slope'} "
            f"(p = {a.p_common_slope:.3g}) and differ in offset (p = {a.p_common_offset:.2g}). "
        )
    if len(r.censored_points):
        cap += (f"Open triangles: {len(r.censored_points)} formulations still intact at the "
                f"{r.data.test_end_h:g} h test end (excluded from fits).")
    file = pub.save(fig, out, "DT-01_dt_vs_td", formats=formats, banner=banner)
    return FigureRecord("DT-01", SECTION, 1, cap, file)


# --- DT-02 ------------------------------------------------------------------
def fig_correlation(r: DisintegrationAnalysis, out: Path, banner: str | None,
                    formats: Sequence[str]) -> FigureRecord:
    pub.apply_style()
    fig = plt.figure(figsize=(pub.ONEHALF, 3.0), layout="constrained")
    ax_a, ax_b = fig.subplots(1, 2, width_ratios=[1.0, 1.15])
    ranked = r.correlation.ranked[::-1]
    ys = np.arange(len(ranked))
    for k, (attr, ci_attr, marker, off, col, lab) in enumerate((
        ("spearman", "spearman_ci", "o", 0.15, pub.OKABE_ITO[0], "Spearman ρ"),
        ("pearson", "pearson_ci", "s", -0.15, pub.OKABE_ITO[1], "Pearson r (log scale)"),
    )):
        vals = np.array([getattr(c, attr) for c in ranked])
        cis = np.array([getattr(c, ci_attr) for c in ranked])
        ax_a.errorbar(vals, ys + off, xerr=[vals - cis[:, 0], cis[:, 1] - vals], fmt=marker,
                      color=col, ms=3.5, elinewidth=0.7, capsize=1.2, label=lab, zorder=3 - k)
    ax_a.axvline(0, color=pub.MUTED, lw=0.6)
    for sgn in (-1, 1):
        ax_a.axvline(sgn * settings.RHO_EXPECTED, color=pub.MUTED, ls=":", lw=0.7,
                     label=f"|ρ| = {settings.RHO_EXPECTED}" if sgn == 1 else None)
    ax_a.set_yticks(ys, [c.metric.label for c in ranked])
    ax_a.set_ylim(-0.7, len(ranked) - 0.3)
    ax_a.set_xlim(-1.05, 1.05)
    ax_a.set_xlabel("Correlation with ln DT (95% bootstrap CI)")
    pub.categorical(ax_a, "y")
    ax_a.legend(loc="center", fontsize=6)

    # B: Spearman pooled and within each grade, metric x grade.
    usable = r.matched[~r.matched["dt_censored"]]
    cols = ["All", *r.grade_order]
    metrics = [m for m in METRICS if m.key in usable.columns]
    grid = np.full((len(metrics), len(cols)), np.nan)
    for i, metric in enumerate(metrics):
        for j, col in enumerate(cols):
            sub = usable if col == "All" else usable[usable["grade"] == col]
            x, y = _pair(sub, metric)
            if x.size >= 4 and np.ptp(x) > 0 and np.ptp(y) > 0:
                grid[i, j] = stats.spearmanr(x, y).statistic
    im = ax_b.imshow(grid, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            v = grid[i, j]
            ax_b.text(j, i, "—" if not np.isfinite(v) else f"{v:.2f}", ha="center",
                      va="center", fontsize=5.8, color="white" if abs(v) > 0.6 else "black")
    ax_b.set_xticks(range(len(cols)), cols)
    ax_b.set_yticks(range(len(metrics)), [m.label for m in metrics])
    ax_b.tick_params(which="both", top=False, right=False, length=0)
    ax_b.minorticks_off()
    bar = fig.colorbar(im, ax=ax_b, shrink=0.9, pad=0.02, aspect=25)
    bar.set_label("Spearman ρ with ln DT")
    bar.ax.tick_params(which="both", direction="in")
    pub.label_panels([ax_a, ax_b])

    best = r.correlation.ranked[0] if r.correlation.ranked else None
    cap = (
        "(A) Correlation of formulation-mean disintegration time with each dissolution "
        "metric, ranked by |Spearman ρ|, with 95% formulation-bootstrap intervals; dotted "
        f"lines mark |ρ| = {settings.RHO_EXPECTED}. (B) Spearman ρ pooled and within each "
        "grade; '—' marks fewer than four points. "
    )
    if best is not None:
        cap += f"Strongest pooled association: {best.metric.label} (ρ = {best.spearman:+.2f})."
    file = pub.save(fig, out, "DT-02_correlation", formats=formats, banner=banner)
    return FigureRecord("DT-02", SECTION, 2, cap, file)


# --- DT-03 ------------------------------------------------------------------
def fig_matched_composition(r: DisintegrationAnalysis, out: Path, banner: str | None,
                            formats: Sequence[str]) -> FigureRecord:
    pub.apply_style()
    fig = plt.figure(figsize=(pub.DOUBLE, 2.7), layout="constrained")
    axes = fig.subplots(1, 3, width_ratios=[1, 1, 1.15])
    ax_a, ax_b, ax_c = axes
    usable = r.matched[~r.matched["dt_censored"]]
    order = list(r.grade_order)
    xpos = {g: i for i, g in enumerate(order)}
    for ax, col in ((ax_a, "dt_h"), (ax_b, "td_h")):
        for _, case in usable.groupby("case"):
            case = case.sort_values("viscosity_cp")
            ax.plot([xpos[g] for g in case["grade"]], case[col], color=pub.MUTED, lw=0.5,
                    alpha=0.8, zorder=1)
        for i, g in enumerate(order):
            st = pub.grade_style(g, i)
            v = usable.loc[usable["grade"] == g, col].to_numpy(dtype=float)
            ax.scatter(np.full(v.size, xpos[g]), v, marker=st.marker, color=st.colour,
                       edgecolor="black", linewidth=0.4, s=16, zorder=3)
        if col == "dt_h":
            cz = r.censored_points
            for i, g in enumerate(order):
                n_c = int((cz["grade"] == g).sum())
                if n_c:
                    ax.scatter(np.full(n_c, xpos[g]), np.full(n_c, r.data.test_end_h),
                               marker="^", s=16, facecolor="white",
                               edgecolor=pub.grade_style(g, i).colour, linewidth=0.8, zorder=3)
        ax.set_xticks(range(len(order)), order)
        ax.set_xlim(-0.4, len(order) - 0.6)
        pub.log_axis(ax, "y")
        pub.categorical(ax, "x")
        ax.set_xlabel("HPMC grade")
    lo = float(min(usable["dt_h"].min(), usable["td_h"].min())) / 1.4
    hi = float(max(usable["dt_h"].max(), usable["td_h"].max(), r.data.test_end_h)) * 1.4
    for ax in (ax_a, ax_b):
        ax.set_ylim(lo, hi)
        _plain_log(ax, "y")
    ax_a.set_ylabel("Time (h)")
    pub.corner_note(ax_a, "Disintegration time DT")
    pub.corner_note(ax_b, "Dissolution time scale Td")
    ax_b.tick_params(labelleft=False)

    effects = r.grades
    if effects.block_dt is not None and effects.block_td is not None:
        tdc = {(c.a, c.b): c for c in effects.block_td.contrasts}
        rows = [(c, tdc.get((c.a, c.b))) for c in effects.block_dt.contrasts]
        ys = np.arange(len(rows))[::-1]
        for k, (idx, lab, col, marker, off) in enumerate((
            (0, "DT", pub.OKABE_ITO[0], "o", 0.14), (1, "Td", pub.OKABE_ITO[4], "s", -0.14),
        )):
            for y, pair in zip(ys, rows, strict=True):
                c = pair[idx]
                if c is None:
                    continue
                ax_c.errorbar(c.ratio, y + off, xerr=[[c.ratio - c.ci[0]], [c.ci[1] - c.ratio]],
                              fmt=marker, color=col, ms=3.8, elinewidth=0.8, capsize=1.5,
                              label=lab if y == ys[0] else None, zorder=3 - k)
        ax_c.axvline(1.0, color=pub.MUTED, lw=0.6, ls=":", label="no grade effect")
        ax_c.set_xscale("log")
        pub.log_axis(ax_c, "x")
        ax_c.set_yticks(ys, [f"{c.a} / {c.b}" for c, _ in rows])
        ax_c.set_ylim(-0.6, len(rows) - 0.4)
        _plain_log(ax_c, "x")
        ax_c.set_xlabel("Grade ratio, same composition")
        pub.categorical(ax_c, "y")
        ax_c.legend(loc="lower right", fontsize=6.5)
    pub.label_panels(axes)

    cap = (
        "Same composition, different grade. (A) Disintegration time and (B) dissolution "
        "time scale for each composition (grey lines join one case across grades), on the "
        "same log axis. (C) Grade ratios at matched composition from the randomised-block "
        "model ln y ~ grade + case, with Tukey-adjusted 95% intervals. Open triangles in (A): "
        f"formulations still intact at the {r.data.test_end_h:g} h test end, excluded from the "
        f"models. {report.grade_sentence(r)}"
    )
    file = pub.save(fig, out, "DT-03_matched_composition", formats=formats, banner=banner)
    return FigureRecord("DT-03", SECTION, 3, cap, file)


# --- DT-04 ------------------------------------------------------------------
def fig_erosion_lag(r: DisintegrationAnalysis, out: Path, banner: str | None,
                    formats: Sequence[str]) -> FigureRecord:
    fig, axes = pub.new_figure(pub.DOUBLE, 2.6, ncols=2, sharey=True)
    ax_a, ax_b = axes
    for i, g in enumerate(r.grade_order):
        st = pub.grade_style(g, i)
        m = _grade_frame(r, g)
        lag = m["erosion_lag"].to_numpy(dtype=float)
        h = m["hpmc_wt"].to_numpy(dtype=float)
        ax_a.scatter(h, lag, marker=st.marker, color=st.colour, edgecolor="black",
                     linewidth=0.4, s=16, label=g, zorder=3)
        if len(h) >= 3 and np.ptp(h) > 0:
            grid = np.linspace(h.min(), h.max(), 30)
            band = _band(h, np.log(lag), grid)
            if band is not None:
                ax_a.plot(grid, np.exp(band[0]), color=st.colour, ls=st.linestyle, lw=1.0)
                ax_a.fill_between(grid, np.exp(band[1]), np.exp(band[2]), color=st.colour,
                                  alpha=0.15, lw=0)
        if "peppas_n" in m:
            ax_b.scatter(m["peppas_n"], lag, marker=st.marker, color=st.colour,
                         edgecolor="black", linewidth=0.4, s=16, zorder=3)
    for ax in axes:
        ax.axhline(1.0, color=pub.MUTED, ls=":", lw=0.8, label="DT = Td")
        pub.auto_minor(ax)
    pub.log_axis(ax_a, "y")
    ax_a.set_xlabel("HPMC (wt%)")
    ax_a.set_ylabel("Erosion lag R = DT / Td")
    ax_b.set_xlabel("Peppas release exponent n")
    _plain_log(ax_a, "y")
    ax_a.legend(loc="lower right", fontsize=6.5)
    pub.label_panels(axes)
    effects = r.grades
    trend = "; ".join(
        f"{lt.grade} {lt.slope_per_10wt:+.2f} ({lt.ci[0]:+.2f} to {lt.ci[1]:+.2f})"
        for lt in effects.lag
    )
    cap = (
        "(A) Erosion lag, disintegration time over the dissolution time scale, against HPMC "
        "content by grade, with per-grade log-linear fits and 95% bands. R > 1: the matrix "
        "outlasts its release time scale; R < 1: it is gone first. Change in ln R per +10 wt% "
        f"HPMC: {trend}. (B) R against the Peppas exponent (Spearman ρ = "
        f"{effects.lag_vs_peppas_rho:+.2f}, p = {effects.lag_vs_peppas_p:.2g})."
    )
    file = pub.save(fig, out, "DT-04_erosion_lag", formats=formats, banner=banner)
    return FigureRecord("DT-04", SECTION, 4, cap, file)


# --- DT-05 ------------------------------------------------------------------
def fig_doe(r: DisintegrationAnalysis, out: Path, banner: str | None,
            formats: Sequence[str]) -> FigureRecord | None:
    ra = r.dt_fit
    if ra is None or not ra.usable or not ra.grids:
        return None
    pub.apply_style()
    fig = plt.figure(figsize=(pub.DOUBLE, 2.7), layout="constrained")
    left, right = fig.subfigures(1, 2, width_ratios=[3.0, 1.35])
    contour_axes = list(np.atleast_1d(left.subplots(1, len(ra.grids), sharey=True)))
    draw_contours(left, contour_axes, ra)
    ax_p = right.subplots(1, 1)
    draw_pareto(ax_p, ra)
    pub.label_panels([*contour_axes, ax_p])
    t = ra.anova
    cap = (
        f"Classical DoE on disintegration time: predicted DT (h) over the API–HPMC design "
        f"space for each grade on one colour scale (white points: design points), and the "
        f"Pareto of standardised effects. Model {ra.spec.label}, {len(ra.kept_terms)} terms, "
        f"R² = {t.r_squared:.2f}, R²(pred) = {t.pred_r_squared:.2f}. {ra.takeaway_contour}"
    )
    if ra.response.n_missing:
        cap += f" {ra.response.n_missing} censored formulations excluded."
    file = pub.save(fig, out, "DT-05_doe", formats=formats, banner=banner)
    return FigureRecord("DT-05", SECTION, 5, cap, file)


# --- DT-06 ------------------------------------------------------------------
def fig_precision(r: DisintegrationAnalysis, out: Path, banner: str | None,
                  formats: Sequence[str]) -> FigureRecord:
    pub.apply_style()
    fig = plt.figure(figsize=(pub.ONEHALF, 2.6), layout="constrained")
    ax_a, ax_b = fig.subplots(1, 2, width_ratios=[2.3, 1.0])
    pts = r.precision.points.copy()
    pts = pts.sort_values("gmean_h").reset_index(drop=True)
    long = r.data.long
    flagged = {(f.case, f.grade, f.replicate) for f in r.precision.flags}
    for x, p in enumerate(pts.itertuples()):
        i = r.grade_order.index(p.grade) if p.grade in r.grade_order else 0
        st = pub.grade_style(str(p.grade), i)
        reps = long[(long["case"] == p.case) & (long["grade"] == p.grade)]
        ok = reps[~reps["censored"]]
        ax_a.scatter(np.full(len(ok), x), ok["dt_h"], s=5, color=pub.MUTED, zorder=2,
                     linewidth=0)
        cz = reps[reps["censored"]]
        if len(cz):
            ax_a.scatter(np.full(len(cz), x), cz["dt_h"], s=12, marker="^",
                         facecolor="white", edgecolor=pub.INK, linewidth=0.5, zorder=2)
        ax_a.scatter([x], [p.gmean_h], marker=st.marker, color=st.colour, edgecolor="black",
                     linewidth=0.4, s=14, zorder=3)
        for rep in ok.itertuples():
            if (int(p.case), str(p.grade), int(rep.replicate)) in flagged:
                ax_a.scatter([x], [rep.dt_h], s=40, facecolor="none", edgecolor=pub.HIGHLIGHT,
                             linewidth=0.9, zorder=4)
    for i, g in enumerate(r.grade_order):
        st = pub.grade_style(g, i)
        ax_a.scatter([], [], marker=st.marker, color=st.colour, edgecolor="black",
                     linewidth=0.4, s=14, label=f"{g} (geometric mean)")
    ax_a.scatter([], [], s=5, color=pub.MUTED, label="replicate")
    if r.precision.flags:
        ax_a.scatter([], [], s=40, facecolor="none", edgecolor=pub.HIGHLIGHT,
                     label="Dixon Q outlier (kept)")
    if long["censored"].any():
        ax_a.scatter([], [], s=12, marker="^", facecolor="white", edgecolor=pub.INK,
                     label="censored replicate")
    pub.log_axis(ax_a, "y")
    ax_a.set_xlim(-1, len(pts))
    _plain_log(ax_a, "y")
    ax_a.set_xticks([])
    ax_a.tick_params(axis="x", which="both", bottom=False, top=False)
    ax_a.set_xlabel("Formulations, ordered by DT")
    ax_a.set_ylabel("Disintegration time (h)")
    ax_a.legend(loc="upper left", fontsize=5.8)

    ok_pts = pts[~pts["censored"]]
    for i, g in enumerate(r.grade_order):
        st = pub.grade_style(g, i)
        cv = ok_pts.loc[ok_pts["grade"] == g, "cv"].to_numpy(dtype=float) * 100
        if cv.size == 0:
            continue
        ax_b.boxplot([cv], positions=[i], widths=0.5, showfliers=False,
                     medianprops={"color": st.colour, "lw": 1.2},
                     boxprops={"lw": 0.6}, whiskerprops={"lw": 0.6}, capprops={"lw": 0.6})
        jitter = (np.arange(cv.size) - (cv.size - 1) / 2) * 0.04
        ax_b.scatter(i + jitter, cv, marker=st.marker, color=st.colour, edgecolor="black",
                     linewidth=0.3, s=10, zorder=3)
    ax_b.axhline(settings.DT_CV_WARN * 100, color=pub.HIGHLIGHT, ls="--", lw=0.8)
    ax_b.set_xticks(range(len(r.grade_order)), list(r.grade_order))
    ax_b.set_xlim(-0.6, len(r.grade_order) - 0.4)
    ax_b.set_ylim(0, None)
    pub.categorical(ax_b, "x")
    ax_b.set_ylabel("Replicate CV (%)")
    ax_b.legend(handles=[Line2D([], [], color=pub.HIGHLIGHT, ls="--", lw=0.8,
                                label=f"{settings.DT_CV_WARN:.0%} limit")],
                loc="upper left", fontsize=6)
    pub.label_panels([ax_a, ax_b])

    p = r.precision
    cap = (
        "(A) Every replicate (grey) and the geometric mean (grade symbol) per formulation, "
        f"ordered by DT; ICC(1) = {p.icc:.3f}. Formulations at the top edge were still intact "
        f"at the {r.data.test_end_h:g} h test end. "
        + (f"Circled: {len(p.flags)} replicate(s) flagged by Dixon's Q (α = 0.05), kept in "
           "the analysis. " if p.flags else "No replicate flagged by Dixon's Q. ")
        + "(B) Replicate CV per formulation by grade; dashed line at the "
        f"{settings.DT_CV_WARN:.0%} warning limit. Brown–Forsythe p = {p.brown_forsythe_p:.2g}."
    )
    file = pub.save(fig, out, "DT-06_precision", formats=formats, banner=banner)
    return FigureRecord("DT-06", SECTION, 6, cap, file)


# --- DT-07 ------------------------------------------------------------------
def fig_prediction(r: DisintegrationAnalysis, out: Path, banner: str | None,
                   formats: Sequence[str]) -> FigureRecord | None:
    if not r.loo:
        return None
    n = len(r.loo)
    fig, axes = pub.new_figure(pub.DOUBLE, 2.2, ncols=n, sharex=True, sharey=True)
    axes = list(np.atleast_1d(axes))
    allv = np.concatenate([np.exp(m.observed) for m in r.loo] + [np.exp(m.predicted)
                                                                 for m in r.loo])
    lo, hi = float(allv.min()) / 1.4, float(allv.max()) * 1.4
    for ax, m in zip(axes, r.loo, strict=True):
        for i, g in enumerate(r.grade_order):
            st = pub.grade_style(g, i)
            sel = np.array([lab == g for lab in m.grades])
            ax.scatter(np.exp(m.predicted[sel]), np.exp(m.observed[sel]), marker=st.marker,
                       color=st.colour, edgecolor="black", linewidth=0.4, s=13,
                       label=g, zorder=3)
        ax.plot([lo, hi], [lo, hi], color=pub.MUTED, ls=":", lw=0.8, label="1:1")
        pub.log_axis(ax, "x")
        pub.log_axis(ax, "y")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        _plain_log(ax, "x")
        _plain_log(ax, "y")
        ax.set_xlabel("LOO-predicted DT (h)")
        pub.header_note(ax, m.label)
        pub.corner_note(ax, f"Q² = {m.q2:.2f}\n×{m.fold_error:.2f}", "lower right")
    axes[0].set_ylabel("Observed DT (h)")
    axes[0].legend(loc="upper left", fontsize=6)
    pub.label_panels(axes)
    cap = (
        "Leave-one-formulation-out prediction of disintegration time from nested models, all "
        "on the same formulations; dotted line 1:1, with Q² and the typical multiplicative "
        f"error in each panel. {report.prediction_sentence(r)}"
    )
    file = pub.save(fig, out, "DT-07_prediction", formats=formats, banner=banner)
    return FigureRecord("DT-07", SECTION, 7, cap, file)


# --- DT-08 ------------------------------------------------------------------
def fig_model_diagnostics(r: DisintegrationAnalysis, out: Path, banner: str | None,
                          formats: Sequence[str]) -> FigureRecord | None:
    a = r.grades.ancova
    if a is None:
        return None
    fig, axes = pub.new_figure(pub.DOUBLE, 2.3, ncols=3)
    ax_a, ax_b, ax_c = axes
    grades = [lab.split("/")[-1] for lab in a.labels]
    for i, g in enumerate(r.grade_order):
        st = pub.grade_style(g, i)
        sel = np.array([x == g for x in grades])
        ax_a.scatter(a.fitted[sel], a.residuals[sel], marker=st.marker, color=st.colour,
                     edgecolor="black", linewidth=0.4, s=13, label=g, zorder=3)
    ax_a.axhline(0, color=pub.MUTED, lw=0.7, ls=":")
    ax_a.set_xlabel("Fitted ln DT")
    ax_a.set_ylabel("Residual (ln)")
    ax_a.legend(loc="best", fontsize=6)
    pub.auto_minor(ax_a)

    (osm, osr), (slope, intercept, _) = stats.probplot(a.residuals, dist="norm")
    ax_b.scatter(osm, osr, s=12, facecolor="white", edgecolor=pub.INK, linewidth=0.6, zorder=3)
    line = np.array([osm.min(), osm.max()])
    ax_b.plot(line, intercept + slope * line, color=pub.MUTED, ls="--", lw=0.8,
              label="normal reference")
    ax_b.set_xlabel("Normal quantile")
    ax_b.set_ylabel("Ordered residual")
    ax_b.legend(loc="upper left", fontsize=6)
    pub.auto_minor(ax_b)

    idx = np.arange(a.n)
    limit = 4.0 / a.n
    ax_c.vlines(idx, 0, a.cooks, color=pub.INK, lw=0.6)
    ax_c.scatter(idx, a.cooks, s=6, color=pub.INK, zorder=3)
    ax_c.axhline(limit, color=pub.HIGHLIGHT, ls="--", lw=0.8, label=f"4/n = {limit:.2f}")
    for i in np.flatnonzero(a.cooks > limit):
        ax_c.annotate(a.labels[i], (i, a.cooks[i]), xytext=(2, 2), textcoords="offset points",
                      fontsize=5.5)
    ax_c.set_xlim(-1, a.n)
    ax_c.set_ylim(0, max(float(a.cooks.max()), limit) * 1.2)
    ax_c.set_xlabel("Formulation index")
    ax_c.set_ylabel("Cook's distance")
    ax_c.legend(loc="upper right", fontsize=6)
    pub.categorical(ax_c, "x")
    pub.label_panels(axes)
    n_infl = int((a.cooks > limit).sum())
    cap = (
        f"Diagnostics for the ANCOVA of ln DT on ln Td by grade ({a.selected}, n = {a.n}). "
        f"(A) Residuals against fitted values (Breusch–Pagan p = {a.breusch_pagan_p:.2g}). "
        f"(B) Normal Q–Q plot (Shapiro–Wilk p = {a.shapiro_p:.2g}). (C) Cook's distance; "
        f"{n_infl} point(s) exceed 4/n."
    )
    file = pub.save(fig, out, "DT-08_model_diagnostics", formats=formats, banner=banner)
    return FigureRecord("DT-08", SECTION, 8, cap, file)


def render_figures(
    r: DisintegrationAnalysis, out_dir: Path, formats: Sequence[str] = ("png",)
) -> list[FigureRecord]:
    """Render DT-01 … DT-08 into ``out_dir`` and write the caption manifest."""
    pub.apply_style()
    out_dir.mkdir(parents=True, exist_ok=True)
    banner = pub.SYNTHETIC_BANNER if r.is_synthetic else None
    makers = (
        fig_dt_vs_td, fig_correlation, fig_matched_composition, fig_erosion_lag,
        fig_doe, fig_precision, fig_prediction, fig_model_diagnostics,
    )
    records = [rec for make in makers if (rec := make(r, out_dir, banner, formats)) is not None]
    pub.write_captions(records, out_dir, "Disintegration figures")
    return records
