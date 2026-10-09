"""The headline figures: a short, fixed set of composites for a slide deck.

Each is a multi-panel figure that carries one message. They are rendered into
``figures/headlines/`` with their own caption files, so the folder can be handed
over as-is. The set is deliberately hard-coded; revisit it once the audit and
diagnostics have run on real data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from pipeline.analysis import Analysis
from pipeline.figures import doe as doefig
from pipeline.figures import publication as pub
from pipeline.figures import render
from pipeline.figures.publication import FigureRecord
from pipeline.stress.subsets import StressTest

#: The headline set, in deck order.
HEADLINES: tuple[str, ...] = (
    "H1_release_by_grade",
    "H2_composition_lever",
    "H3_doe_contours",
    "H4_grade_equivalence",
    "H5_reduced_design",
)

#: The response the DoE headlines are drawn for.
HEADLINE_RESPONSE = "t50"


def _h1(analysis: Analysis, out: Path, banner: str | None) -> FigureRecord:
    grades = render.grades_by_viscosity(analysis)
    profiles = render.measured_profiles(analysis)
    styles = render.case_styles(analysis)

    fig, axes = pub.new_figure(pub.DOUBLE, 3.4, ncols=len(grades), sharey=True)
    axes = list(np.atleast_1d(axes))
    handles: dict[int, Any] = {}
    for i, (ax, grade) in enumerate(zip(axes, grades, strict=False)):
        for key in sorted(k for k in profiles if k[1] == grade):
            case = key[0]
            if case not in styles:
                continue
            st = styles[case].style
            render.draw_measured(ax, profiles[key], st.colour, st.marker)
            handles.setdefault(case, render.case_handle(st))
        render.full_release_line(ax)
        pub.time_axis(ax)
        pub.percent_axis(ax)
        pub.corner_note(ax, grade, "lower right")
        if i:
            ax.set_ylabel("")
    pub.label_panels(axes)
    render.case_legend(fig, handles, styles)
    n_rep = int(analysis.quality.n_replicates)
    return FigureRecord(
        "H1", "headline", 1,
        f"Measured release profiles by HPMC grade (panels, ordered by viscosity; n = {n_rep} "
        f"replicates). {render.MEASURED_CAPTION} Each case keeps the same colour and marker "
        "in every panel; marker shape shows the API level and the legend gives each "
        "composition. Dotted grey: 100 % release.",
        pub.save(fig, out, HEADLINES[0], banner=banner),
    )


def _h2(analysis: Analysis, out: Path, banner: str | None) -> FigureRecord | None:
    """The composition lever, beside how well the headline model fits the data."""
    ra = analysis.doe.by_key(HEADLINE_RESPONSE)
    usable = ra is not None and ra.usable
    fig, axes = pub.new_figure(pub.ONEHALF, 2.6, ncols=2 if usable else 1)
    axes = list(np.atleast_1d(axes))
    render.draw_lever(axes[0], analysis)
    caption = render.lever_caption(analysis)
    if usable and ra is not None:
        from pipeline.doe import views

        doefig.draw_pred_actual(
            axes[1], ra, views.fitted_pairs(ra, analysis.design_points),
            [g for g, _ in views.grade_levels(analysis.design_points)],
        )
        caption = (
            f"(A) {caption} (B) Measured against model-predicted "
            f"{ra.response.spec.label.lower()} for every formulation; the dashed line is "
            "1:1, so the scatter about it is the model's error."
        )
    if len(axes) > 1:
        pub.label_panels(axes)
    return FigureRecord("H2", "headline", 2, caption,
                        pub.save(fig, out, HEADLINES[1], banner=banner))


def _h3(analysis: Analysis, out: Path, banner: str | None) -> FigureRecord | None:
    ra = analysis.doe.by_key(HEADLINE_RESPONSE)
    if ra is None or not ra.usable:
        return None
    from pipeline.doe import views

    pub.apply_style()
    view = views.ternary_view(ra, analysis.design_points)
    n_terms = len(ra.ranking.effects)
    fig = plt.figure(figsize=(pub.DOUBLE, 2.6 + max(1.8, 0.11 * n_terms + 0.5)),
                     layout="constrained")
    top, bottom = fig.subfigures(2, 1, height_ratios=[2.6, max(1.8, 0.11 * n_terms + 0.5)])
    caxes = list(np.atleast_1d(top.subplots(1, len(view.grids))))
    doefig.draw_ternary(top, caxes, ra, view)
    pax = bottom.subplots(1, 1)
    doefig.draw_pareto(pax, ra)
    pub.label_panels([*caxes, pax])
    sig = [e.term for e in ra.ranking.effects if e.significant]
    sig_text = f"Significant terms: {', '.join(sig)}." if sig else "No term is significant."
    label = ra.response.spec.label.lower()
    letters = "".join(chr(ord("A") + i) for i in range(len(caxes)))
    return FigureRecord(
        "H3", "headline", 3,
        f"({letters[0]}–{letters[-1]}) {label.capitalize()} across the tested blends, one "
        "triangle per grade: background is the fitted model, points are the measured "
        "blends filled with their measured value on the same colour scale. "
        f"({chr(ord('A') + len(caxes))}) Standardised effects on {label} against the 5 % "
        f"(dashed) and Bonferroni (dotted) lines. {sig_text}",
        pub.save(fig, out, HEADLINES[2], banner=banner),
    )


def equivalence_counts(analysis: Analysis, grades: list[str]) -> tuple[list[int], np.ndarray]:
    """Rows: cases. Columns: grades. Value: equivalents in a *different* grade."""
    cases = sorted({int(c) for c in analysis.design_points["case"]})
    counts = np.full((len(cases), len(grades)), np.nan)
    for s in analysis.equivalence:
        if s.target_case not in cases or s.target_grade not in grades:
            continue
        n = sum(1 for m in s.members if m.grade != s.target_grade)
        counts[cases.index(s.target_case), grades.index(s.target_grade)] = n
    return cases, counts


def draw_equivalence_counts(fig: Any, ax: Any, counts: np.ndarray, cases: list[int],
                            grades: list[str]) -> None:
    """Heatmap: for each target (case, grade), its f2 equivalents in another grade."""
    vmax = max(1.0, float(np.nanmax(counts)) if np.isfinite(counts).any() else 1.0)
    im = ax.imshow(counts, cmap="Blues", vmin=0, vmax=vmax, aspect="auto")
    for r in range(counts.shape[0]):
        for c in range(counts.shape[1]):
            v = counts[r, c]
            text = "–" if not np.isfinite(v) else f"{int(v)}"
            ax.text(c, r, text, ha="center", va="center", fontsize=6,
                    color="white" if np.isfinite(v) and v > 0.6 * vmax else "black")
    ax.set_xticks(range(len(grades)), grades)
    ax.set_yticks(range(len(cases)), [str(c) for c in cases])
    ax.tick_params(which="both", top=False, right=False, length=0)
    pub.categorical(ax, "both")
    ax.set_xlabel("Target grade")
    ax.set_ylabel("Target case")
    bar = fig.colorbar(im, ax=ax, shrink=0.95, aspect=25, pad=0.02)
    bar.set_label("Equivalents in another grade")
    bar.ax.tick_params(which="both", direction="in")


def _h4(analysis: Analysis, out: Path, banner: str | None) -> FigureRecord | None:
    best = render.best_cross_grade_set(analysis)
    if best is None:
        return None
    grades = render.grades_by_viscosity(analysis)
    cases, counts = equivalence_counts(analysis, grades)
    fig, axes = pub.new_figure(pub.DOUBLE, 2.9, ncols=2, width_ratios=[1.6, 1.0])
    render.draw_equivalence(axes[0], analysis, best)

    draw_equivalence_counts(fig, axes[1], counts, cases, grades)
    pub.label_panels(axes)

    summary = analysis.equivalence_summary
    isolated = len(summary.isolated_targets)
    return FigureRecord(
        "H4", "headline", 4,
        f"(A) Mean measured profiles f2-similar to case {best.target_case} "
        f"({best.target_grade}), spanning {len(best.grades_spanned)} grades. (B) For each "
        "target formulation, the number of f2-equivalent formulations made with a "
        f"different grade. {len(summary.cross_grade_targets)} of {summary.n_targets} "
        f"targets have at least one; {isolated} ha{'s' if isolated == 1 else 've'} none.",
        pub.save(fig, out, HEADLINES[3], banner=banner),
    )


def _h5(analysis: Analysis, stress: StressTest, out: Path, banner: str | None) -> FigureRecord:
    fig, axes = pub.new_figure(pub.ONEHALF, 2.6, ncols=2)
    render.draw_stress(axes[0], stress)
    selected = set(stress.recommended.selected) if stress.recommended else None
    render.draw_design_plane(axes[1], analysis, selected)
    pub.label_panels(axes)
    n_full = len(analysis.design_points)
    rec = stress.recommended
    detail = (
        f"The recommended {rec.size}-run design reaches {rec.profile_rmse_pct:.2f} % RMSE "
        f"against {stress.full_profile_rmse_pct:.2f} % for all {n_full} runs."
        if rec else "No reduced design met the recommendation criteria."
    )
    return FigureRecord(
        "H5", "headline", 5,
        "(A) Profile prediction error of D-optimal reduced designs against design size. "
        "(B) The recommended runs (filled) among all formulations (open) in API–HPMC "
        f"space; grades are nudged ±{render.GRADE_NUDGE} wt% sideways to stay visible. "
        f"{detail}",
        pub.save(fig, out, HEADLINES[4], banner=banner),
    )


def render_headlines(analysis: Analysis, stress: StressTest, out: Path) -> list[FigureRecord]:
    """Render the headline set into ``out`` with ``captions.json`` and ``captions.md``."""
    out.mkdir(parents=True, exist_ok=True)
    for stale in out.glob("*.png"):
        stale.unlink()
    banner = render.banner_for(analysis)
    made = [
        _h1(analysis, out, banner),
        _h2(analysis, out, banner),
        _h3(analysis, out, banner),
        _h4(analysis, out, banner),
        _h5(analysis, stress, out, banner),
    ]
    records = [r for r in made if r is not None]
    pub.write_captions(records, out, "Headline figures")
    return records
