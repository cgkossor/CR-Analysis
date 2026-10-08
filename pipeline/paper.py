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
#: Ternary rows need more height: an equal-aspect triangle is as tall as it is wide.
SURFACE_ROW_H = 2.7


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
    rows = [(p, pick(p)) for p in inputs]
    rows = [(p, ra) for p, ra in rows if ra is not None]
    if not rows:
        return None
    grades = render.grades_by_viscosity(rows[0][0].analysis)
    n = len(rows)
    pub.apply_style()
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(pub.DOUBLE, SURFACE_ROW_H * n + 0.3), layout="constrained")
    subs = fig.subfigures(n, 1, squeeze=False)
    letters: list[Any] = []
    for (p, ra), sub in zip(rows, subs[:, 0], strict=True):
        # Ternaries (with their own colour bar) on the left; the fit panel, when
        # wanted, in a column of its own so the colour bar never crowds it.
        if with_fit:
            left, right = sub.subfigures(1, 2, width_ratios=[len(grades) + 0.6, 1.25])
        else:
            left, right = sub, None
        axes = list(np.atleast_1d(left.subplots(1, len(grades))))
        view = views.ternary_view(ra, p.analysis.design_points)
        doefig.draw_ternary(left, axes, ra, view)
        if right is not None:
            fax = right.subplots(1, 1)
            doefig.draw_pred_actual(fax, ra, views.fitted_pairs(ra, p.analysis.design_points),
                                    grades, compact=True)
            axes.append(fax)
        if n > 1:
            sub.suptitle(p.api, x=0.01, ha="left", fontsize=8, fontweight="bold")
        letters += axes
    pub.label_panels(letters)
    return pub.save(fig, out, stem, banner=banner), [p.api for p, _ in rows]


def fig4_surface(inputs: list[PaperInput], out: Path, banner: str | None) -> FigureRecord | None:
    key = headline_response(inputs)
    if key is None:
        return None
    made = _surface_rows(inputs, lambda p: _usable(p.analysis.doe.by_key(key)),
                         out, "Fig4_surface", banner, with_fit=True)
    if made is None:
        return None
    file, apis = made
    label = inputs[0].analysis.doe.by_key(key)
    name = label.response.spec.label.lower() if label else key
    return FigureRecord(
        "Fig4", "main", 4,
        f"Response surface for {name}" + (f", by API (rows: {', '.join(apis)})"
                                          if len(apis) > 1 else "")
        + ". Triangles, one per HPMC grade: the fitted mixture model over the tested blends "
        "(background) with each measured blend filled with its measured value on the same "
        "colour scale; a blend that stands out from its surroundings is one the model does "
        "not fit. Right: measured against predicted for every formulation, with the 1:1 "
        "line, R² and the leave-one-out predicted R² (Q²).",
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
    makers = (fig1_design, fig2_profiles, fig3_levers, fig4_surface, fig5_mechanism,
              fig6_grade_swap, fig7_disintegration, fig8_reduced)
    records = [r for r in (m(inputs, out, banner) for m in makers) if r is not None]
    pub.write_captions(records, out, "Manuscript figures (main)")
    _write_supplementary_index(out, supplementary or {})
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
