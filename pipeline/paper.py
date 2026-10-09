"""The manuscript's main figures, across every API in the run.

The figure gallery holds every diagnostic figure for every response; a paper
needs a handful, in reading order, each carrying one part of the story:

    Fig 1  The design: the tested blends in the mixture triangle.
    Fig 2  Measured release profiles, API x grade.
    Fig 3  Which lever controls release: composition, grade, interaction.
    Fig 4  The response surface for the headline response, with model fit.
    Fig 5  Release mechanism: the Weibull shape parameter across the design.
    Fig 6  Where the HPMC grade can be interchanged.
    Fig 7  Disintegration against dissolution.
    Fig 8  How few runs reproduce the conclusions.

Every figure is laid out with one row (or column) per API, so the same
function draws the one-API paper of today and the four-API paper to come.
Figures whose data an API lacks (no disintegration file, say) leave that API
out and say so in the caption. Written to ``paper/`` with ``captions.json``,
``captions.md``, the design table, and an index of the supplementary figures.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import numpy as np

from pipeline import config
from pipeline.doe import ternary, views
from pipeline.figures import doe as doefig
from pipeline.figures import publication as pub
from pipeline.figures import render
from pipeline.figures.publication import FigureRecord
from pipeline.manuscript import disintegration_link, grade_swap

if TYPE_CHECKING:
    from pipeline.analysis import Analysis
    from pipeline.disintegration.analysis import DisintegrationAnalysis
    from pipeline.manuscript import Manuscript
    from pipeline.stress.subsets import StressTest

#: The response the surface figure is drawn for: the first of these that every
#: API in the run can model. t50 goes undefined for formulations that never
#: reach half their dose, which a slow (low-solubility) API can do; MDT and
#: % at 8 h are always defined.
HEADLINE_PREFERENCE: tuple[str, ...] = ("t50", "mdt_h", "pct_8h")


def headline_response(inputs: list[PaperInput]) -> str | None:
    for key in HEADLINE_PREFERENCE:
        if all(_usable(p.analysis.doe.by_key(key)) is not None for p in inputs):
            return key
    return None

#: Height of one API row, inches.
ROW_H = 2.25
#: A row of three ternaries across the full width: each triangle is about 2.2 in
#: wide, so the row needs about 2.3 in including its grade labels.
SURFACE_ROW_H = 2.35
#: The row of predicted-vs-actual panels under the ternaries in Fig 4.
FIT_ROW_H = 2.6


#: Background for each main figure: why it matters, beside its caption.
CONTEXT: dict[str, str] = {
    "Scheme1": (
        "Every result in the paper is stated in one of these metrics, and each "
        "describes a different part of the curve: the t-values and fixed-time "
        "percentages describe timing, MDT the whole curve, the slopes the rate in "
        "each phase, and the Weibull and Peppas parameters its shape. Defining "
        "them once, on one curve, lets the reader interpret every later figure "
        "without returning to the methods."
    ),
    "Fig1": (
        "In a mixture the components sum to 100%, so no component can change on its own: "
        "raising HPMC must lower API, lactose or both. The tested blends therefore define a "
        "constrained region, and every conclusion in the paper is an interpolation inside it. "
        "Showing that region first tells the reader where the results hold, and using the "
        "same blends for every API is what makes the cross-API comparisons direct."
    ),
    "Fig2": (
        "In a hydrophilic HPMC matrix the polymer hydrates into a gel layer that controls "
        "release, by drug diffusion through the gel and by erosion of the gel itself. The "
        "measured profiles are the primary evidence for everything that follows: they show "
        "how large the composition and grade effects are, and the replicate spread shows how "
        "precisely they were measured, before any model is involved."
    ),
    "Fig3": (
        "A formulator has two practical handles on release rate: how much HPMC to use and "
        "which viscosity grade. Knowing which dominates tells which to adjust first, and the "
        "interaction share tells whether rules learned in one grade carry over to another. A "
        "large interaction means the effect of HPMC content depends on the grade, so neither "
        "lever can be set without the other."
    ),
    "Fig4": (
        "The fitted response surface turns 33 measured formulations into a prediction for any "
        "blend in the tested region, which is the basis of a formulation design space in the "
        "sense of ICH Q8. The ternary plot is the standard way to show a three-component "
        "response, and the predicted-vs-actual panel, with the leave-one-out Q², is the "
        "evidence that the surface predicts formulations it was not fitted to rather than "
        "merely reproducing them."
    ),
    "Fig4A": (
        "t50 does not exist for a blend that never releases half its dose, so the "
        "slowest formulations drop out of a t50 surface and its slow corner is "
        "extrapolated from faster blends. Mean dissolution time summarises the whole "
        "curve and exists for every blend, so this version shows the full design "
        "space, at the cost of reading as a lower bound where release is unfinished."
    ),
    "Fig6A": (
        "An f2 value is abstract until two curves are seen side by side. The example "
        "shows what regulatory similarity looks like for formulations made with "
        "different grades, and the count map shows how general that is: which "
        "formulations have a ready substitute in another grade and which have none, "
        "which is the practical question when a grade is unavailable or changes supplier."
    ),
    "Fig5": (
        "Release rate alone does not say how the drug leaves the matrix. The Weibull shape "
        "parameter is commonly read as a mechanism indicator, from diffusion-dominated "
        "release at low values to erosion-influenced and sigmoidal release at higher ones. "
        "Mechanism matters for robustness: erosion-controlled release tends to be more "
        "sensitive to hydrodynamics and gastrointestinal conditions than diffusion-controlled "
        "release, and it is also where API solubility is expected to act."
    ),
    "Fig6": (
        "HPMC grades are often treated as interchangeable when a supplier or grade changes, "
        "yet a change in the release-controlling excipient is a significant post-approval "
        "change. The f2 similarity factor (f2 >= 50) is the regulatory criterion for "
        "comparing dissolution profiles, so mapping where grade pairs pass it shows where a "
        "substitution is likely to be defensible and where it would alter release. A "
        "regulatory case would still need its own batches."
    ),
    "Fig7": (
        "Disintegration testing takes minutes to hours, against a day for a full dissolution "
        "profile. If disintegration time tracks the dissolution time scale, it could screen "
        "formulations or support quality control. The pooled fit tests the relationship "
        "across grades; whether it also holds within each grade decides whether "
        "disintegration ranks formulations or merely separates grades."
    ),
    "Fig8": (
        "Each new API would otherwise need the full 33-run design. Showing how prediction "
        "error grows as runs are removed identifies the smallest design that reaches the same "
        "conclusions, which sets the experimental cost of extending the study to further "
        "APIs, including the low-solubility ones still in progress."
    ),
}


@dataclass(frozen=True)
class PaperInput:
    """One API's results, as the paper figures need them."""

    api: str
    analysis: Analysis
    stress: StressTest
    disintegration: DisintegrationAnalysis | None
    manuscript: Manuscript


def _banner(inputs: list[PaperInput]) -> str | None:
    return pub.SYNTHETIC_BANNER if any(p.analysis.quality.is_synthetic for p in inputs) else None


def _row_note(ax: Any, api: str) -> None:
    """The API name at the left of a row, outside the axes."""
    ax.annotate(api, xy=(0, 0.5), xycoords="axes fraction", xytext=(-46, 0),
                textcoords="offset points", rotation=90, ha="center", va="center",
                fontsize=8, fontweight="bold")


# --- Scheme 1: what each metric measures -------------------------------------

#: The illustrative curve: a Weibull profile, not data.
SCHEME_F_INF, SCHEME_TD, SCHEME_BETA = 92.0, 6.0, 0.75


def scheme1_metrics(out: Path) -> FigureRecord:
    """One model curve, annotated with every release metric the analysis uses."""
    from pipeline.profiles.fits import weibull

    t = np.linspace(0.0, 24.0, 2401)
    y = weibull(t, SCHEME_F_INF, SCHEME_TD, SCHEME_BETA)

    def t_at(pct: float) -> float:
        return float(np.interp(pct, y, t))

    fig, (ax, bx) = pub.new_figure(pub.DOUBLE, 3.3, ncols=2,
                                   gridspec_kw={"width_ratios": [2.3, 1]})
    # Peppas window: the part of the curve below 60 % released.
    t60 = t_at(60.0)
    ax.axvspan(0, t60, color=pub.OKABE_ITO[5], alpha=0.12, lw=0)
    ax.text(t60 / 2, 3, "Peppas fit\n(first 60 %)", ha="center", fontsize=6.5,
            color=pub.INK)
    ax.plot(t, y, color=pub.INK, lw=1.6)
    # Plateau.
    ax.axhline(SCHEME_F_INF, color=pub.MUTED, lw=0.8, ls="--")
    ax.text(23.8, SCHEME_F_INF + 1.5, "F∞ (Weibull plateau)", ha="right", fontsize=6.5)
    # t10 ... t80.
    for pct, colour in ((10, 0), (25, 1), (50, 2), (80, 4)):
        tx = t_at(pct)
        c = pub.OKABE_ITO[colour]
        ax.plot([0, tx], [pct, pct], color=c, lw=0.7, ls=":")
        ax.plot([tx, tx], [0, pct], color=c, lw=0.7, ls=":")
        ax.plot(tx, pct, "o", color=c, ms=3.5)
        ax.text(tx + 0.25, pct - 4.5, f"t{pct}", fontsize=7, color=c, fontweight="bold")
    # Td: where the curve reaches 63.2 % of its plateau.
    y_td = SCHEME_F_INF * (1 - np.exp(-1.0))
    ax.plot(SCHEME_TD, y_td, "D", color=pub.OKABE_ITO[3], ms=4)
    ax.annotate("Td (time scale:\n63.2 % of F∞)", (SCHEME_TD, y_td), xytext=(9.5, 52),
                fontsize=6.5, color=pub.OKABE_ITO[3],
                arrowprops={"arrowstyle": "-", "color": pub.OKABE_ITO[3], "lw": 0.6})
    # % released at the fixed times.
    fixed = (1, 2, 4, 8, 12, 24)
    ax.plot(fixed, np.interp(fixed, t, y), "s", mfc="white", mec=pub.INK, ms=4, mew=0.8)
    ax.text(8.6, 60.0, "% released at\n1, 2, 4, 8, 12, 24 h", fontsize=6.5)
    # Mean dissolution time.
    dm = np.diff(y)
    mid = 0.5 * (t[1:] + t[:-1])
    mdt = float(np.sum(mid * dm) / np.sum(dm))
    ax.axvline(mdt, color=pub.HIGHLIGHT, lw=0.8, ls="-.")
    ax.text(mdt + 0.25, 8, "MDT", fontsize=7, color=pub.HIGHLIGHT, fontweight="bold")
    # Early and late slopes.
    # Each slope's label sits clear of the curve: right of the early one,
    # under the late one.
    for (a, b), label, (lx, ly, ha) in (
        ((0.0, 2.0), "early slope (0–2 h)", (2.4, 24.0, "left")),
        ((8.0, 24.0), "late slope (8–24 h)", (24.0, 74.0, "right")),
    ):
        m = (t >= a) & (t <= b)
        k, c0 = np.polyfit(t[m], y[m], 1)
        xs = np.array([a, b])
        ax.plot(xs, k * xs + c0, color=pub.OKABE_ITO[0], lw=1.0, ls="--")
        ax.text(lx, ly, label, fontsize=6.5, color=pub.OKABE_ITO[0], ha=ha)
    pub.time_axis(ax)
    pub.percent_axis(ax)

    # Inset panel: what beta does to the shape at the same Td.
    for beta, ls in ((0.5, "-"), (1.0, "--"), (1.8, ":")):
        bx.plot(t, weibull(t, 100.0, SCHEME_TD, beta), color=pub.INK, ls=ls,
                label=f"β = {beta:g}")
    bx.set_xlim(0, 24)
    bx.set_ylim(0, 105)
    bx.set_xlabel("Time (h)")
    bx.set_ylabel("Drug released (%)")
    bx.legend(title="Same Td", fontsize=6)
    pub.auto_minor(bx)
    pub.label_panels([ax, bx])
    file = pub.save(fig, out, "Scheme1_metrics")
    return FigureRecord(
        "Scheme1", "main", 0,
        "Release metrics used in this work, shown on an illustrative Weibull profile "
        f"(F∞ = {SCHEME_F_INF:g} %, Td = {SCHEME_TD:g} h, β = {SCHEME_BETA:g}); a schematic, "
        "not measured data. (A) Times to 10, 25, 50 and 80 % released (t10 to t80); "
        "% released at fixed times (squares); the Weibull plateau F∞ and time scale Td; "
        "the mean dissolution time (MDT); least-squares release rates over 0 to 2 h and "
        "8 to 24 h; and the region below 60 % released used for the Peppas exponent. "
        "(B) The Weibull shape parameter β at a fixed Td: β < 1 gives a fast initial "
        "release that slows, β = 1 a first-order curve, β > 1 a sigmoidal curve with an "
        "initial lag.",
        file,
    )


# --- Fig 1: the design --------------------------------------------------------


def fig1_design(inputs: list[PaperInput], out: Path, banner: str | None) -> FigureRecord:
    a = inputs[0].analysis
    dp = a.design_points.drop_duplicates("case").sort_values("case")
    comp = dp[["api_wt", "hpmc_wt", "lactose_wt"]].to_numpy(dtype=float) / 100.0
    reg = ternary.region(views.design_comp(a.design_points))
    xy = reg.to_xy(comp)
    fig, ax = pub.new_figure(pub.SINGLE, 3.0)
    doefig._draw_triangle(ax, reg)
    ax.scatter(xy[:, 0], xy[:, 1], s=30, color=pub.INK, zorder=6)
    for (x, y), case in zip(xy, dp["case"], strict=True):
        # Below-right near the top corner, where the corner label sits above.
        offset = (6, -7) if y > 0.75 else (4, 4)
        ax.annotate(str(int(case)), (x, y), xytext=offset, textcoords="offset points",
                    fontsize=6.5, zorder=7)
    file = pub.save(fig, out, "Fig1_design", banner=banner)

    with (out / "Table1_design.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["case", "API_wt", "HPMC_wt", "lactose_wt"])
        for r in dp.itertuples(index=False):
            w.writerow([int(r.case), r.api_wt, r.hpmc_wt, r.lactose_wt])

    grades = render.grades_by_viscosity(a)
    same = all(
        set(map(tuple, p.analysis.design_points[["case", "api_wt", "hpmc_wt", "lactose_wt"]]
                .drop_duplicates().to_numpy().tolist()))
        == set(map(tuple, a.design_points[["case", "api_wt", "hpmc_wt", "lactose_wt"]]
                   .drop_duplicates().to_numpy().tolist()))
        for p in inputs
    )
    return FigureRecord(
        "Fig1", "main", 1,
        f"The mixture design: the {len(dp)} API / HPMC / lactose blends (numbered by case) "
        "in the constrained region of the mixture triangle. Each corner is one component at "
        "its highest possible level with the other two at their lowest tested levels; edge "
        "numbers are wt%. Every blend was made with each HPMC grade "
        f"({', '.join(grades)})" + (f", for each API ({', '.join(p.api for p in inputs)})"
                                   if len(inputs) > 1 and same else "")
        + ". Compositions are listed in Table1_design.csv.",
        file,
    )


# --- Fig 2: release profiles -------------------------------------------------


def fig2_profiles(inputs: list[PaperInput], out: Path, banner: str | None) -> FigureRecord:
    grades = render.grades_by_viscosity(inputs[0].analysis)
    n = len(inputs)
    fig, axes = pub.new_figure(pub.DOUBLE, ROW_H * n + 0.9, nrows=n, ncols=len(grades),
                               sharex=True, sharey=True, squeeze=False)
    handles: dict[int, Any] = {}
    styles = render.case_styles(inputs[0].analysis)
    for r, p in enumerate(inputs):
        profiles = render.measured_profiles(p.analysis)
        for c, grade in enumerate(grades):
            ax = axes[r][c]
            for key in sorted(k for k in profiles if k[1] == grade):
                if key[0] not in styles:
                    continue
                st = styles[key[0]].style
                render.draw_measured(ax, profiles[key], st.colour, st.marker)
                handles.setdefault(key[0], render.case_handle(st))
            render.full_release_line(ax)
            pub.time_axis(ax)
            pub.percent_axis(ax)
            if r == 0:
                pub.header_note(ax, grade)
            if c:
                ax.set_ylabel("")
            if r < n - 1:
                ax.set_xlabel("")
        if n > 1:
            _row_note(axes[r][0], p.api)
    pub.label_panels([ax for row in axes for ax in row])
    render.case_legend(fig, handles, styles)
    return FigureRecord(
        "Fig2", "main", 2,
        "Measured release profiles by HPMC grade (columns, ordered by viscosity)"
        + (" and API (rows)" if n > 1 else "")
        + f". {render.MEASURED_CAPTION} Each case keeps its colour and marker in every "
        "panel; the legend gives each composition. Dotted grey: 100 % release.",
        pub.save(fig, out, "Fig2_profiles", banner=banner),
    )


# --- Fig 3: which lever ------------------------------------------------------


def fig3_levers(inputs: list[PaperInput], out: Path, banner: str | None) -> FigureRecord | None:
    usable = [p for p in inputs if p.manuscript.decompositions]
    if not usable:
        return None
    n = len(usable)
    fig, axes = pub.new_figure(pub.DOUBLE if n > 1 else pub.ONEHALF, 2.9, ncols=n,
                               sharey=True, squeeze=False)
    parts = (("Composition", "share_composition", pub.OKABE_ITO[0]),
             ("Grade", "share_grade", pub.OKABE_ITO[1]),
             ("Interaction", "share_interaction", pub.OKABE_ITO[2]))
    for c, p in enumerate(usable):
        ax = axes[0][c]
        decs = p.manuscript.decompositions
        y = np.arange(len(decs))[::-1]
        left = np.zeros(len(decs))
        for name, attr, colour in parts:
            v = 100 * np.array([getattr(d, attr) for d in decs])
            ax.barh(y, v, left=left, color=colour, height=0.66, label=name,
                    edgecolor="white", linewidth=0.5)
            left += v
        ax.set_yticks(y, [d.label for d in decs])
        ax.set_xlim(0, 100)
        ax.set_xlabel("Share of variation (%)")
        pub.categorical(ax, "y")
        if n > 1:
            pub.header_note(ax, p.api)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncols=3, frameon=False)
    pub.label_panels(axes[0])
    return FigureRecord(
        "Fig3", "main", 3,
        "Which lever controls release. For each response, the share of the variation across "
        "the case x grade design explained by composition, by HPMC grade, and by their "
        "interaction (eta-squared on formulation means; the three add to 100 %)"
        + (", for each API (panels)" if n > 1 else "") + ".",
        pub.save(fig, out, "Fig3_levers", banner=banner),
    )


# --- Fig 4 and 5: ternary surfaces -------------------------------------------


def _surface_rows(
    inputs: list[PaperInput], pick: Any, out: Path, stem: str, banner: str | None,
    with_fit: bool,
) -> tuple[str, list[str]] | None:
    """One row of ternaries per API at the full figure width.

    An equal-aspect triangle shrinks in height whenever it loses width, so the
    triangles get the whole row: sharing it with the fit panel had squeezed
    them to a third of their space. The predicted-vs-actual panels, when
    wanted, form one final row, one panel per API.
    """
    rows = [(p, pick(p)) for p in inputs]
    rows = [(p, ra) for p, ra in rows if ra is not None]
    if not rows:
        return None
    grades = render.grades_by_viscosity(rows[0][0].analysis)
    n = len(rows)
    pub.apply_style()
    import matplotlib.pyplot as plt

    heights = [SURFACE_ROW_H] * n + ([FIT_ROW_H] if with_fit else [])
    fig = plt.figure(figsize=(pub.DOUBLE, sum(heights) + 0.2), layout="constrained")
    subs = fig.subfigures(len(heights), 1, squeeze=False, height_ratios=heights)
    letters: list[Any] = []
    for (p, ra), sub in zip(rows, subs[:n, 0], strict=True):
        axes = list(np.atleast_1d(sub.subplots(1, len(grades))))
        view = views.ternary_view(ra, p.analysis.design_points)
        doefig.draw_ternary(sub, axes, ra, view)
        if n > 1:
            sub.suptitle(p.api, x=0.01, ha="left", fontsize=8, fontweight="bold")
        letters += axes
    if with_fit:
        fit_axes = list(np.atleast_1d(subs[n, 0].subplots(1, max(n, 2))))
        for fax, (p, ra) in zip(fit_axes, rows, strict=False):
            doefig.draw_pred_actual(fax, ra, views.fitted_pairs(ra, p.analysis.design_points),
                                    grades, compact=True)
            if n > 1:
                pub.header_note(fax, p.api)
        for spare in fit_axes[len(rows):]:
            spare.set_visible(False)
        letters += fit_axes[:len(rows)]
    pub.label_panels(letters)
    return pub.save(fig, out, stem, banner=banner), [p.api for p, _ in rows]


def fig4_surface(inputs: list[PaperInput], out: Path, banner: str | None) -> FigureRecord | None:
    key = headline_response(inputs)
    if key is None:
        return None
    return _surface_figure(inputs, out, banner, key, "Fig4", "Fig4_surface")


def fig4a_mdt(inputs: list[PaperInput], out: Path, banner: str | None) -> FigureRecord | None:
    """Fig 4 on mean dissolution time, which every profile has.

    t50 is undefined for a blend that never reaches 50 %, so Fig 4 drops those
    blends from its triangles. MDT keeps every blend; for one still releasing
    at the end of the run it is a lower bound, and the caption says how many.
    """
    if headline_response(inputs) == "mdt_h":
        return None  # Fig 4 already shows MDT.
    rec = _surface_figure(inputs, out, banner, "mdt_h", "Fig4A", "Fig4A_surface_mdt")
    if rec is None:
        return None
    counts = []
    for p in inputs:
        bounds = mdt_lower_bounds(p)
        if bounds is None:
            continue
        counts.append(f"{p.api}: {bounds[0]} of {bounds[1]}"
                      if len(inputs) > 1 else f"{bounds[0]} of {bounds[1]}")
    note = (" Alternative to Fig 4 on mean dissolution time (MDT), which is defined for "
            "every blend, including those that never reach 50% released. MDT is computed "
            f"over the {config.ANALYSIS_WINDOW_H:g} h run; for a profile still rising at the "
            "end it is a lower bound on the true value"
            + (f" (formulations affected: {'; '.join(counts)})" if counts else "") + ".")
    from dataclasses import replace

    return replace(rec, caption=rec.caption + note)


def mdt_lower_bounds(p: PaperInput) -> tuple[int, int] | None:
    """(formulations whose MDT is a lower bound, all formulations), or None.

    A formulation counts when any of its vessels was still releasing at the end
    of the run, so its MDT over the run understates the true value.
    """
    reps = p.analysis.replicates
    if "mdt_truncated" not in reps:
        return None
    flagged = reps.groupby(["case", "grade"])["mdt_truncated"].any()
    return int(flagged.sum()), len(flagged)


def _surface_figure(inputs: list[PaperInput], out: Path, banner: str | None, key: str,
                    fid: str, stem: str) -> FigureRecord | None:
    made = _surface_rows(inputs, lambda p: _usable(p.analysis.doe.by_key(key)),
                         out, stem, banner, with_fit=True)
    if made is None:
        return None
    file, apis = made
    label = inputs[0].analysis.doe.by_key(key)
    name = label.response.spec.label.lower() if label else key
    return FigureRecord(
        fid, "main", 4,
        f"Response surface for {name}" + (f", by API (rows: {', '.join(apis)})"
                                          if len(apis) > 1 else "")
        + ". Triangles, one per HPMC grade: the fitted mixture model over the tested blends "
        "(background) with each measured blend filled with its measured value on the same "
        "colour scale; a blend that stands out from its surroundings is one the model does "
        "not fit. Bottom row: measured against predicted for every formulation"
        + (", one panel per API" if len(apis) > 1 else "")
        + ", with the 1:1 line, R² and the leave-one-out predicted R² (Q²).",
        file,
    )


def _usable(ra: Any) -> Any:
    return ra if ra is not None and getattr(ra, "usable", False) else None


def _beta_response(a: Analysis) -> Any:
    """The Weibull shape surface, dressed as a DoE response for the ternary drawer."""
    surface = a.surfaces.get("weibull_beta")
    if surface is None or not surface.fit.estimable:
        return None
    dp = a.design_points
    if "weibull_beta_mean" not in dp:
        return None
    values = dp["weibull_beta_mean"].to_numpy(dtype=float)
    spec = SimpleNamespace(label="Weibull shape β", units="", key="weibull_beta")
    return SimpleNamespace(
        spec=surface.spec, kept_terms=surface.kept_terms, fit=surface.fit,
        response=SimpleNamespace(spec=spec, values=values, available=np.isfinite(values)),
    )


def fig5_mechanism(inputs: list[PaperInput], out: Path, banner: str | None
                   ) -> FigureRecord | None:
    made = _surface_rows(inputs, lambda p: _beta_response(p.analysis), out,
                         "Fig5_mechanism", banner, with_fit=False)
    if made is None:
        return None
    file, apis = made
    return FigureRecord(
        "Fig5", "main", 5,
        "Release mechanism: the Weibull shape parameter β across the tested blends, one "
        "triangle per HPMC grade" + (f", by API (rows: {', '.join(apis)})"
                                    if len(apis) > 1 else "")
        + ". β below about 0.75 indicates Fickian diffusion-controlled release; between "
        "0.75 and 1, combined diffusion and erosion; above 1, a sigmoidal profile with an "
        "initial lag. Background: fitted model; points: measured blends.",
        file,
    )


# --- Fig 6: grade interchangeability -----------------------------------------


def fig6_grade_swap(inputs: list[PaperInput], out: Path, banner: str | None
                    ) -> FigureRecord | None:
    from itertools import combinations

    usable = [p for p in inputs if any(s.valid for s in p.manuscript.swaps)]
    if not usable:
        return None
    grades = render.grades_by_viscosity(usable[0].analysis)
    pairs = list(combinations(grades, 2))
    n = len(usable)
    fig, axes = pub.new_figure(pub.DOUBLE, ROW_H * n + 0.3, nrows=n, ncols=len(pairs),
                               sharex=True, sharey=True, squeeze=False)
    for r, p in enumerate(usable):
        for c, (ga, gb) in enumerate(pairs):
            grade_swap.draw_pair(axes[r][c], p.manuscript.swaps, ga, gb)
            if r < n - 1:
                axes[r][c].set_xlabel("")
        axes[r][0].set_ylabel("HPMC (wt%)")
        if n > 1:
            _row_note(axes[r][0], p.api)
    pub.label_panels([ax for row in axes for ax in row])
    from pipeline import config

    return FigureRecord(
        "Fig6", "main", 6,
        "Where the HPMC grade can be interchanged. Each panel compares two grades at every "
        "tested blend (API and HPMC content; lactose is the balance)"
        + (", for each API (rows)" if n > 1 else "")
        + f". Filled: the measured mean profiles are similar (f2 ≥ "
        f"{config.F2_SIMILAR_THRESHOLD:g}); open: not similar. Numbers are the f2 values.",
        pub.save(fig, out, "Fig6_grade_swap", banner=banner),
    )



def fig6a_equivalence(inputs: list[PaperInput], out: Path, banner: str | None
                      ) -> FigureRecord | None:
    """Example equivalent profiles beside the count of equivalents, one row per API."""
    from pipeline.figures import headlines

    rows = []
    for p in inputs:
        best = render.best_cross_grade_set(p.analysis)
        if best is not None:
            rows.append((p, best))
    if not rows:
        return None
    n = len(rows)
    fig, axes = pub.new_figure(pub.DOUBLE, 2.9 * n, nrows=n, ncols=2, squeeze=False,
                               width_ratios=[1.6, 1.0])
    parts = []
    for (p, best), (ax_a, ax_b) in zip(rows, axes, strict=True):
        grades = render.grades_by_viscosity(p.analysis)
        cases, counts = headlines.equivalence_counts(p.analysis, grades)
        render.draw_equivalence(ax_a, p.analysis, best)
        headlines.draw_equivalence_counts(fig, ax_b, counts, cases, grades)
        if n > 1:
            pub.header_note(ax_a, p.api)
        summary = p.analysis.equivalence_summary
        parts.append(
            (f"{p.api}: " if n > 1 else "")
            + f"example target case {best.target_case} ({best.target_grade}), with "
            f"equivalents in {len(best.grades_spanned)} grades; "
            f"{len(summary.cross_grade_targets)} of {summary.n_targets} targets have at "
            "least one equivalent in another grade")
    pub.label_panels(list(axes.flat))
    return FigureRecord(
        "Fig6A", "main", 6,
        "Grade equivalence in practice" + (" (rows: one per API)" if n > 1 else "")
        + ". Left: mean measured profiles of formulations made with different HPMC grades "
        f"that are f2-similar (f2 ≥ {config.F2_SIMILAR_THRESHOLD:g}) to one target "
        "formulation, with ±SD bars. Right: for every target formulation (case × grade), "
        "the number of formulations in another grade that are f2-similar to it. "
        + "; ".join(parts) + ".",
        pub.save(fig, out, "Fig6A_equivalence_examples", banner=banner),
    )

# --- Fig 7: disintegration ---------------------------------------------------


def fig7_disintegration(inputs: list[PaperInput], out: Path, banner: str | None
                        ) -> FigureRecord | None:
    usable = [p for p in inputs
              if p.disintegration is not None and disintegration_link.drawable(p.disintegration)]
    if not usable:
        return None
    n = len(usable)
    fig, axes = pub.new_figure(pub.DOUBLE if n > 1 else pub.SINGLE, 2.8, ncols=n,
                               squeeze=False)
    notes = []
    for c, p in enumerate(usable):
        assert p.disintegration is not None
        pooled = disintegration_link.draw_dt(axes[0][c], p.disintegration, legend=c == 0)
        if n > 1:
            pub.header_note(axes[0][c], p.api)
        if c:
            axes[0][c].set_ylabel("")
        if pooled is not None:
            notes.append(f"{p.api}: slope {pooled.slope:.2f} (95% CI {pooled.slope_ci[0]:.2f} "
                         f"to {pooled.slope_ci[1]:.2f})")
    pub.label_panels(axes[0])
    missing = [p.api for p in inputs if p not in usable]
    return FigureRecord(
        "Fig7", "main", 7,
        "Disintegration time against the Weibull dissolution time scale Td, both on log "
        "axes, by grade" + (" and API (panels)" if n > 1 else "")
        + ". Lines: pooled Deming fits, which allow for error in both measurements ("
        + "; ".join(notes) + "). Tablets intact at the end of the test are omitted."
        + (f" No disintegration data for {', '.join(missing)}." if missing else ""),
        pub.save(fig, out, "Fig7_disintegration", banner=banner),
    )


# --- Fig 8: reduced design ---------------------------------------------------


def fig8_reduced(inputs: list[PaperInput], out: Path, banner: str | None) -> FigureRecord:
    n = len(inputs)
    fig, axes = pub.new_figure(pub.DOUBLE if n > 1 else pub.SINGLE, 2.6, ncols=n,
                               squeeze=False, sharey=False)
    details = []
    for c, p in enumerate(inputs):
        render.draw_stress(axes[0][c], p.stress)
        if n > 1:
            pub.header_note(axes[0][c], p.api)
        rec = p.stress.recommended
        if rec:
            details.append(f"{p.api}: {rec.size} of {len(p.analysis.design_points)} runs, "
                           f"{rec.profile_rmse_pct:.2f} % RMSE")
    pub.label_panels(axes[0])
    return FigureRecord(
        "Fig8", "main", 8,
        "How few runs reproduce the conclusions: profile prediction error of D-optimal "
        "reduced designs against the number of runs"
        + (" for each API (panels)" if n > 1 else "") + ". Recommended: "
        + ("; ".join(details) if details else "none met the criteria") + ".",
        pub.save(fig, out, "Fig8_reduced_design", banner=banner),
    )


# --- the set -----------------------------------------------------------------


def render_paper(
    inputs: list[PaperInput], out: Path, supplementary: dict[str, Path] | None = None
) -> list[FigureRecord]:
    """Draw the main figures into ``out`` and write their captions and indices."""
    out.mkdir(parents=True, exist_ok=True)
    for stale in list(out.glob("*.png")) + list(out.glob("*.svg")):
        stale.unlink()
    banner = _banner(inputs)
    makers = (fig1_design, fig2_profiles, fig3_levers, fig4_surface, fig4a_mdt, fig5_mechanism,
              fig6_grade_swap, fig6a_equivalence, fig7_disintegration, fig8_reduced)
    from dataclasses import replace

    made = [scheme1_metrics(out)] + [m(inputs, out, banner) for m in makers]
    records = [replace(r, context=CONTEXT.get(r.id, "")) for r in made if r is not None]
    pub.write_captions(records, out, "Manuscript figures (main)")
    _write_supplementary_index(out, supplementary or {})
    from pipeline import deck_brief

    deck_brief.write(inputs, records, out, headline_response(inputs))
    return records


def _write_supplementary_index(out: Path, figures_dirs: dict[str, Path]) -> None:
    """``supplementary.md``: every other figure, per API, with its caption."""
    lines = ["# Supplementary figures", "",
             "Every figure not in the main set, by API, with its file and caption.", ""]
    for api, folder in figures_dirs.items():
        lines += [f"## {api}", ""]
        for cap in sorted(folder.rglob("captions.json")):
            if cap.parent == out:
                continue
            for entry in json.loads(cap.read_text(encoding="utf-8")):
                rel = (cap.parent / entry["png"]).relative_to(folder)
                lines.append(f"- **{entry['id']}** ({rel.as_posix()}): {entry['caption']}")
        lines.append("")
    (out / "supplementary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
