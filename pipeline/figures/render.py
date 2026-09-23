"""Render every figure as a 300 dpi PNG, with captions (AC9).

Figures are ranked by relevance to the storyline, not by the order they were
convenient to produce. Rank 1 is the finding a formulator does not already know;
the confirmatory ones rank last because confirming that more HPMC slows release
is a pipeline sanity check, not a result.

All styling comes from ``publication``: closed box, inward major and minor ticks,
panel letters, no titles. The takeaway sentence lives in the caption, which is
built from the computed numbers so it cannot claim something the data do not show.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from pipeline import config
from pipeline.analysis import Analysis
from pipeline.equivalence.sets import EquivalenceSet
from pipeline.figures import publication as pub
from pipeline.figures.doe import render_doe_figures
from pipeline.figures.publication import FigureRecord
from pipeline.profiles.grid import project_onto_grid
from pipeline.stress.subsets import StressTest

#: Sideways nudge per grade (wt%) when grades share a composition in one plane,
#: so three formulations at the same point do not print on top of each other.
GRADE_NUDGE = 1.6


# --- shared data helpers -------------------------------------------------------


def banner_for(analysis: Analysis) -> str | None:
    return pub.SYNTHETIC_BANNER if analysis.quality.is_synthetic else None


def grades_by_viscosity(analysis: Analysis) -> list[str]:
    pts = analysis.design_points.drop_duplicates("grade").sort_values("viscosity_cp")
    return [str(g) for g in pts["grade"]]


@dataclass(frozen=True)
class CaseStyle:
    """How one case is drawn in every panel, and where it sits in the legend."""

    style: pub.SeriesStyle
    label: str
    api_level: int
    api_wt: float


def _api_level(api: float, levels: list[float]) -> int:
    """0 / 1 / 2 for the lower, middle and upper third of the tested API range."""
    if len(levels) < 2:
        return 0
    position = levels.index(api) / (len(levels) - 1)
    if position < 1 / 3 - 1e-9:
        return 0
    if position > 2 / 3 + 1e-9:
        return 2
    return 1


def case_styles(analysis: Analysis) -> dict[int, CaseStyle]:
    """A fixed style and legend label per case, shared by every panel.

    Line style shows the API level; colour tells the cases within one level
    apart. The label carries the composition, so the legend alone says which
    recipe a line is: "Case 3: 22.5 / 29.5 / 48" is API / HPMC / lactose, wt%.
    """
    comp = analysis.design_points.drop_duplicates("case").sort_values("case")
    levels = sorted(float(v) for v in comp["api_wt"].unique())
    used: dict[int, int] = {}
    out: dict[int, CaseStyle] = {}
    for r in comp.itertuples():
        level = _api_level(float(r.api_wt), levels)
        k = used.get(level, 0)
        used[level] = k + 1
        style = pub.SeriesStyle(
            pub.CASE_COLOURS[k % len(pub.CASE_COLOURS)], "o", pub.API_LEVEL_LINESTYLES[level]
        )
        label = f"Case {int(r.case)}: {r.api_wt:g} / {r.hpmc_wt:g} / {r.lactose_wt:g}"
        out[int(r.case)] = CaseStyle(style, label, level, float(r.api_wt))
    return out


CASE_LEGEND_TITLE = "Case: API / HPMC / lactose (wt%). Line style shows the API level."


def case_legend(fig: Figure, handles: dict[int, Line2D], styles: dict[int, CaseStyle]) -> None:
    """One legend below the panels, one column per API level, each with a header.

    Matplotlib fills legend columns top to bottom, so each level's block is
    padded to the same height to keep it in its own column.
    """
    by_level: dict[int, list[int]] = {}
    for case in sorted(handles):
        by_level.setdefault(styles[case].api_level, []).append(case)
    order = sorted(by_level)
    height = max(len(v) for v in by_level.values())
    api = {lvl: sorted({styles[c].api_wt for c in cases}) for lvl, cases in by_level.items()}

    def blank() -> Line2D:
        return Line2D([], [], linestyle="none")

    entries: list[Line2D] = []
    labels: list[str] = []
    headers: list[int] = []
    for lvl in order:
        lo, hi = api[lvl][0], api[lvl][-1]
        span = f"{lo:g} wt%" if lo == hi else f"{lo:g} to {hi:g} wt%"
        headers.append(len(labels))
        entries.append(blank())
        labels.append(f"{pub.API_LEVEL_NAMES[lvl]}, {span}")
        for case in by_level[lvl]:
            entries.append(handles[case])
            labels.append(styles[case].label)
        for _ in range(height - len(by_level[lvl])):
            entries.append(blank())
            labels.append("")
    legend = fig.legend(
        entries, labels, loc="outside lower center", ncol=len(order),
        title=CASE_LEGEND_TITLE, handlelength=3.2, columnspacing=2.4, borderaxespad=0.2,
    )
    for i in headers:
        legend.get_texts()[i].set_fontweight("bold")


def replicate_bands(
    analysis: Analysis,
) -> dict[tuple[int, str], tuple[np.ndarray, np.ndarray]]:
    """Mean and SD across replicates per design point, on the analysis time grid.

    Each replicate is projected onto the grid on its own clock first, exactly as
    ``analysis._mean_profiles`` does, so the band and the mean line agree.
    """
    grid = analysis.time_grid_info
    out: dict[tuple[int, str], tuple[np.ndarray, np.ndarray]] = {}
    for (case, grade), group in analysis.db.profiles.groupby(["case", "grade"]):
        stacked = np.vstack([
            project_onto_grid(
                rep["time_h"].to_numpy(dtype=float),
                rep["pct_released"].to_numpy(dtype=float),
                grid.times_h,
                grid.max_gap_h,
            )
            for _, rep in group.groupby("replicate")
        ])
        ok = np.isfinite(stacked).all(axis=0)
        mean = np.where(ok, stacked.mean(axis=0), np.nan)
        sd = np.where(ok, stacked.std(axis=0, ddof=1) if len(stacked) > 1 else 0.0, np.nan)
        out[(int(case), str(grade))] = (mean, sd)
    return out


def best_cross_grade_set(analysis: Analysis) -> EquivalenceSet | None:
    cross = [s for s in analysis.equivalence if s.spans_multiple_grades]
    if not cross:
        return None
    return max(cross, key=lambda s: (len(s.grades_spanned), s.n_members))


# --- drawing primitives (shared with headlines) -------------------------------


def draw_lever(ax: Axes, analysis: Analysis) -> None:
    levers = analysis.lever_effects
    grades = [str(g) for g in levers["grade"]]
    values = levers["fold_change_td"].to_numpy(dtype=float)
    x = np.arange(len(grades))
    for i, (g, v) in enumerate(zip(grades, values, strict=True)):
        st = pub.grade_style(g, i)
        ax.bar(i, v, width=0.6, color=st.colour, edgecolor="black", linewidth=0.5)
        ax.text(i, v, f"×{v:.2f}", ha="center", va="bottom", fontsize=7,
                transform=ax.transData)
    ax.axhline(1.0, color=pub.INK, lw=0.7, ls="--", label="no change (×1)")
    ax.set_xticks(x, grades)
    pub.categorical(ax, "x")
    ax.set_xlim(-0.6, len(grades) - 0.4)
    ax.set_ylim(0, float(np.nanmax(values)) * 1.18)
    pub.auto_minor(ax)
    ax.set_xlabel("HPMC grade")
    ax.set_ylabel("Td multiplier, +10 wt% HPMC")
    ax.legend(loc="upper right")


def lever_caption(analysis: Analysis) -> str:
    levers = analysis.lever_effects
    parts = [f"{g} ×{v:.2f}" for g, v in zip(levers["grade"], levers["fold_change_td"],
                                                strict=True)]
    values = levers["fold_change_td"].to_numpy(dtype=float)
    diffs = np.diff(values)
    if np.all(diffs < 0):
        trend = ("The effect shrinks as grade viscosity rises: the composition and grade "
                 "levers are partial substitutes, not independent.")
    elif np.all(diffs > 0):
        trend = "The effect grows as grade viscosity rises: the two levers reinforce."
    else:
        trend = "The effect does not change monotonically with grade viscosity."
    return (
        "Fold change in Weibull scale Td from substituting 10 wt% lactose with HPMC, "
        f"evaluated at the design centroid for each grade ({', '.join(parts)}); grades "
        f"ordered by viscosity. Dashed line: no change. {trend}"
    )


def draw_equivalence(ax: Axes, analysis: Analysis, best: EquivalenceSet) -> int:
    """Overlay the target and up to three f2-similar members. Returns curves drawn."""
    others = [
        (m.case, m.grade)
        for m in best.members
        if (m.case, m.grade) != (best.target_case, best.target_grade)
    ][:3]
    shown = [(best.target_case, best.target_grade), *others]
    markers = ("o", "s", "^", "D")
    lines = ("-", "--", "-.", ":")
    drawn = 0
    for i, key in enumerate(shown):
        curve = analysis.observed_profiles.get(key)
        if curve is None:
            continue
        f2 = next((m.f2 for m in best.members if (m.case, m.grade) == key), float("nan"))
        label = f"Case {key[0]}, {key[1]}" + (" (target)" if i == 0 else f", f2 = {f2:.0f}")
        ok = np.isfinite(curve)
        ax.plot(
            analysis.time_grid[ok], curve[ok], color=pub.grade_style(key[1], i).colour,
            marker=markers[i % 4], linestyle=lines[i % 4], markersize=3.2,
            markerfacecolor="white" if i else None, markeredgewidth=0.7, label=label,
        )
        drawn += 1
    pub.time_axis(ax)
    pub.percent_axis(ax)
    ax.legend(loc="lower right")
    return drawn


def draw_stress(ax: Axes, stress: StressTest) -> None:
    usable = [r for r in stress.results if r.estimable]
    ax.plot(
        [r.size for r in usable], [r.profile_rmse_pct for r in usable],
        color=pub.OKABE_ITO[0], marker="o", markerfacecolor="white",
        label="reduced design",
    )
    ax.axhline(stress.full_profile_rmse_pct, color=pub.INK, ls="--", lw=0.7,
               label=f"full design ({stress.full_profile_rmse_pct:.2f} %)")
    if stress.recommended:
        ax.axvline(stress.recommended.size, color=pub.HIGHLIGHT, ls=":", lw=1.0,
                   label=f"recommended ({stress.recommended.size} runs)")
    top = max([r.profile_rmse_pct for r in usable] + [stress.full_profile_rmse_pct])
    ax.set_ylim(0, top * 1.3)
    pub.auto_minor(ax)
    ax.set_xlabel("Runs in the reduced design")
    ax.set_ylabel("Profile RMSE (% released)")
    ax.legend(loc="lower right")


def draw_design_plane(
    ax: Axes,
    analysis: Analysis,
    selected: set[tuple[int, str]] | None = None,
) -> None:
    """Every formulation in API-HPMC space; grades nudged apart, selected filled."""
    grades = grades_by_viscosity(analysis)
    offset = {g: (i - (len(grades) - 1) / 2) * GRADE_NUDGE for i, g in enumerate(grades)}
    pts = analysis.design_points
    for i, g in enumerate(grades):
        sub = pts[pts["grade"] == g]
        st = pub.grade_style(g, i)
        chosen = [
            selected is None or (int(r.case), str(r.grade)) in selected
            for r in sub.itertuples()
        ]
        mask = np.array(chosen, dtype=bool)
        x = sub["api_wt"].to_numpy(dtype=float) + offset[g]
        y = sub["hpmc_wt"].to_numpy(dtype=float)
        ax.scatter(x[mask], y[mask], marker=st.marker, s=16, color=st.colour,
                   edgecolor="black", linewidth=0.4, label=g, zorder=3, clip_on=False)
        if selected is not None and (~mask).any():
            ax.scatter(x[~mask], y[~mask], marker=st.marker, s=16, facecolor="white",
                       edgecolor=st.colour, linewidth=0.7, zorder=2, clip_on=False)
    pub.auto_minor(ax)
    ax.set_xlabel("API (wt%)")
    ax.set_ylabel("HPMC (wt%)")
    handles = [
        Line2D([], [], marker=pub.grade_style(g, i).marker, color=pub.grade_style(g, i).colour,
               linestyle="none", markeredgecolor="black", markeredgewidth=0.4, label=g)
        for i, g in enumerate(grades)
    ]
    if selected is not None:
        handles.append(Line2D([], [], marker="o", color=pub.MUTED, markerfacecolor="white",
                              linestyle="none", label="not selected"))
    ax.margins(x=0.08, y=0.08)
    ax.legend(handles=handles, loc="upper center", ncols=2,
              bbox_to_anchor=(0.5, 1.0), handletextpad=0.2, columnspacing=0.8)
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.25 * (hi - lo))


# --- the figure set ----------------------------------------------------------


def render_all(analysis: Analysis, stress: StressTest, out_dir: Path) -> list[FigureRecord]:
    """Render the full figure set, the headline subset, and the caption manifests."""
    from pipeline.figures.headlines import render_headlines

    pub.apply_style()
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in [*out_dir.glob("*.svg"), *out_dir.glob("*.png")]:
        stale.unlink()
    banner = banner_for(analysis)
    records: list[FigureRecord] = []

    def emit(fid: str, name: str, rank: int, caption: str, fig: Figure,
             section: str = "main") -> None:
        records.append(FigureRecord(fid, section, rank, caption,
                                    pub.save(fig, out_dir, name, banner=banner)))

    # --- F01: the non-additive lever --------------------------------------
    fig, ax = pub.new_figure(pub.SINGLE, 2.6)
    draw_lever(ax, analysis)
    emit("F01", "01_lever_non_additivity", 1, lever_caption(analysis), fig)

    # --- F02: equivalence demonstration ----------------------------------
    best = best_cross_grade_set(analysis)
    if best is not None:
        fig, ax = pub.new_figure(pub.SINGLE, 2.8)
        draw_equivalence(ax, analysis, best)
        emit(
            "F02", "02_equivalence_demonstration", 2,
            f"Mean measured profiles of formulations f2-similar to case "
            f"{best.target_case} ({best.target_grade}), spanning "
            f"{len(best.grades_spanned)} grades ({', '.join(best.grades_spanned)}). "
            "Different composition and grade can give the same release, so formulation "
            "freedom for a given target can be spent on secondary criteria.",
            fig,
        )

    # --- F03: stress-test degradation ------------------------------------
    fig, ax = pub.new_figure(pub.SINGLE, 2.6)
    draw_stress(ax, stress)
    emit(
        "F03", "03_stress_degradation", 3,
        f"Profile prediction error of D-optimal reduced designs against the full "
        f"{len(analysis.design_points)}-run design (dashed). Designs below the "
        "recommended size (dotted) are excluded even where apparent error is low, "
        "because they cannot estimate their own uncertainty.",
        fig,
    )

    # --- F04: FDS --------------------------------------------------------
    fds_model = (
        "scheffe_linear" if "scheffe_linear" in analysis.design_diagnostics
        else "scheffe_quadratic"
    )
    diag = analysis.design_diagnostics.get(fds_model)
    if diag is not None and diag.fds_spv:
        spv = np.asarray(diag.fds_spv)
        fig, ax = pub.new_figure(pub.SINGLE, 2.6)
        ax.plot(np.linspace(0, 1, len(spv)), spv, color=pub.OKABE_ITO[3])
        ax.set_xlim(0, 1)
        ax.set_ylim(bottom=0)
        pub.auto_minor(ax)
        ax.set_xlabel("Fraction of design space")
        ax.set_ylabel("Scaled prediction variance")
        emit(
            "F04", "04_fds", 4,
            "Fraction-of-design-space plot: scaled prediction variance across the convex "
            "hull of tested compositions crossed with the tested viscosity range. The "
            "flatter and lower the curve, the more uniformly trustworthy predictions are.",
            fig,
        )

    # --- F05: observed vs predicted (CV) ---------------------------------
    folds = [f for f in analysis.cross_validation.folds if np.isfinite(f.profile_rmse_pct)]
    if folds:
        fig, ax = pub.new_figure(pub.SINGLE, 3.2)
        grades = grades_by_viscosity(analysis)
        allv: list[float] = []
        for i, g in enumerate(grades):
            sub = [f for f in folds if f.grade == g]
            obs = [f.observed.get("log10_td", np.nan) for f in sub]
            pred = [f.predicted.get("log10_td", np.nan) for f in sub]
            allv += obs + pred
            st = pub.grade_style(g, i)
            ax.scatter(obs, pred, marker=st.marker, s=18, color=st.colour,
                       edgecolor="black", linewidth=0.4, label=g, zorder=3)
        lo, hi = float(np.nanmin(allv)), float(np.nanmax(allv))
        pad = 0.05 * (hi - lo)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=pub.INK, ls="--", lw=0.7,
                label="y = x")
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_aspect("equal")
        pub.auto_minor(ax)
        ax.set_xlabel("Observed log$_{10}$ Td (h)")
        ax.set_ylabel("Predicted log$_{10}$ Td (h)")
        ax.legend(loc="upper left")
        rmse = analysis.cross_validation.rmse_by_response.get("log10_td", float("nan"))
        emit(
            "F05", "05_cv_observed_vs_predicted", 5,
            "Leave-one-formulation-out prediction of log10 Td: each point is predicted "
            "by a surface refitted without it, with all its replicates held out together "
            f"(RMSE {rmse:.3f} log10 units). Dashed: identity.",
            fig,
        )

    # --- F06: response correlation heatmap -------------------------------
    space = analysis.response_space
    fig, ax = pub.new_figure(pub.ONEHALF * 0.8, pub.ONEHALF * 0.7)
    matrix = space.pearson.to_numpy()
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1)
    n = len(space.metrics)
    ax.set_xticks(range(n), space.metrics, rotation=90)
    ax.set_yticks(range(n), space.metrics)
    ax.tick_params(which="both", top=False, right=False, length=0)
    pub.categorical(ax, "both")
    bar = fig.colorbar(im, ax=ax, shrink=0.85, aspect=25)
    bar.set_label("Pearson r")
    bar.ax.tick_params(which="both", direction="in")
    emit(
        "F06", "06_response_correlation", 6,
        f"Pearson correlation among the {n} AC2 metrics, which span only "
        f"{space.n_components_90} independent dimensions (90 % variance). The block "
        "structure is why agreement between these metrics is not independent "
        "confirmation, and why conclusions rest on the reduced key-response set.",
        fig,
    )

    # --- F07: PCA biplot -------------------------------------------------
    if space.scores.shape[1] > 1:
        fig, ax = pub.new_figure(pub.SINGLE * 1.15, 3.4)
        ax.axhline(0, color=pub.MUTED, lw=0.5, ls=":")
        ax.axvline(0, color=pub.MUTED, lw=0.5, ls=":")
        ax.scatter(space.scores[:, 0], space.scores[:, 1], s=12, facecolor="white",
                   edgecolor=pub.OKABE_ITO[0], linewidth=0.7, zorder=3)
        scale = float(np.abs(space.scores[:, :2]).max()) * 0.85
        lim = float(np.abs(space.scores[:, :2]).max()) * 1.25
        tips = {
            m: (float(space.loadings.loc[m, "PC1"]) * scale,
                float(space.loadings.loc[m, "PC2"]) * scale)
            for m in space.loadings.index
        }
        for x, y in tips.values():
            ax.annotate("", xy=(x, y), xytext=(0, 0),
                        arrowprops={"arrowstyle": "-|>", "color": pub.HIGHLIGHT,
                                    "lw": 0.7, "shrinkA": 0, "shrinkB": 0})
        # Labels on each side are spread vertically so near-parallel loadings
        # (t10, t25 and t50, say) stay legible; a thin leader joins label and tip.
        for side in (1, -1):
            names = [m for m, (x, _) in tips.items() if (x >= 0) == (side > 0)]
            ys = pub.spread_labels([tips[m][1] * 1.1 for m in names], 0.075 * lim)
            for m, ly in zip(names, ys, strict=True):
                x, y = tips[m]
                lx = x * 1.1 + side * 0.03 * lim
                ax.annotate(m, xy=(x, y), xytext=(lx, ly), fontsize=5.5,
                            color=pub.HIGHLIGHT, ha="left" if side > 0 else "right",
                            va="center",
                            arrowprops={"arrowstyle": "-", "color": pub.MUTED,
                                        "lw": 0.3, "shrinkA": 1, "shrinkB": 1})
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        pub.auto_minor(ax)
        ax.set_xlabel(f"PC1 ({space.explained_variance_ratio[0]:.0%})")
        ax.set_ylabel(f"PC2 ({space.explained_variance_ratio[1]:.0%})")
        emit(
            "F07", "07_pca_biplot", 7,
            "Principal components of the standardised response matrix: formulations "
            "(circles) and metric loadings (arrows, labels joined by thin leaders). "
            "Dotted lines mark the origin.",
            fig,
        )

    # --- F08: raw profiles by grade (confirmatory) -----------------------
    grades = grades_by_viscosity(analysis)
    styles = case_styles(analysis)
    fig, axes = pub.new_figure(pub.DOUBLE, 3.3, ncols=len(grades), sharey=True)
    axes = list(np.atleast_1d(axes))
    handles: dict[int, Line2D] = {}
    for i, (ax, grade) in enumerate(zip(axes, grades, strict=False)):
        for (case, g), curve in sorted(analysis.observed_profiles.items()):
            if g != grade or case not in styles:
                continue
            st = styles[case].style
            ok = np.isfinite(curve)
            (line,) = ax.plot(analysis.time_grid[ok], curve[ok], color=st.colour, lw=1.0,
                              linestyle=st.linestyle)
            handles.setdefault(case, line)
        ax.axhline(config.CENSORING_PCT, color=pub.MUTED, ls=":", lw=0.7)
        pub.time_axis(ax)
        pub.percent_axis(ax)
        pub.corner_note(ax, grade, "lower right")
        if i:
            ax.set_ylabel("")
    pub.label_panels(axes)
    case_legend(fig, handles, styles)
    emit(
        "F08", "08_profiles_by_grade", 8,
        "Mean measured profile of every formulation, one panel per grade (ordered by "
        "viscosity). Each case keeps the same colour and line style in every panel; line "
        "style shows the API level and the legend gives each composition. Dotted grey: "
        f"{config.CENSORING_PCT:.0f} % release.",
        fig,
    )

    # --- F09: design points coloured by Td -------------------------------
    pts = analysis.design_points
    vmin, vmax = float(pts["log10_td_mean"].min()), float(pts["log10_td_mean"].max())
    fig, axes = pub.new_figure(pub.DOUBLE, 2.4, ncols=len(grades), sharey=True)
    axes = list(np.atleast_1d(axes))
    sc = None
    for i, (ax, grade) in enumerate(zip(axes, grades, strict=False)):
        sub = pts[pts["grade"] == grade]
        sc = ax.scatter(sub["hpmc_wt"], sub["api_wt"], c=sub["log10_td_mean"], s=36,
                        cmap="viridis", vmin=vmin, vmax=vmax,
                        marker=pub.grade_style(grade, i).marker,
                        edgecolor="black", linewidth=0.4)
        pub.corner_note(ax, grade, "upper right")
        pub.auto_minor(ax)
        ax.set_xlabel("HPMC (wt%)")
    axes[0].set_ylabel("API (wt%)")
    if sc is not None:
        bar = fig.colorbar(sc, ax=axes, shrink=0.95, aspect=25, pad=0.02)
        bar.set_label("log$_{10}$ Td (h)")
        bar.ax.tick_params(which="both", direction="in")
    pub.label_panels(axes)
    emit(
        "F09", "09_design_space", 9,
        f"The {pts[['api_wt', 'hpmc_wt']].drop_duplicates().shape[0]} tested compositions "
        "in API–HPMC space, one panel per grade, coloured by the fitted Weibull scale on "
        "a shared scale. Lactose is the balance to 100 wt%.",
        fig,
    )

    # --- classical DoE ---------------------------------------------------
    records += render_doe_figures(analysis.doe.responses, out_dir, banner)

    pub.write_captions(records, out_dir, "Figures")
    render_headlines(analysis, stress, out_dir / "headlines")
    return records
