"""Render every figure as vector (SVG) and raster (PNG), with captions (AC9).

Figures are ranked by relevance to the storyline, not by the order they were
convenient to produce. Rank 1 is the finding a formulator does not already know;
the confirmatory ones rank last because confirming that more HPMC slows release
is a pipeline sanity check, not a result.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pipeline import config
from pipeline.analysis import Analysis
from pipeline.stress.subsets import StressTest

GRADE_COLOURS = {"K100LV": "#3b7dd8", "K4M": "#d9822b", "K100M": "#8e5bb5"}
_FALLBACK = ["#3b7dd8", "#d9822b", "#8e5bb5", "#3aa17e", "#c0504d"]


@dataclass(frozen=True)
class Figure:
    """One rendered figure and its place in the argument."""

    name: str
    rank: int
    caption: str
    svg: str
    png: str


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 200,
            "font.size": 9,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.autolayout": True,
        }
    )


def _colour(grade: str, index: int = 0) -> str:
    return GRADE_COLOURS.get(grade, _FALLBACK[index % len(_FALLBACK)])


def _save(fig: plt.Figure, out: Path, name: str, banner: str | None) -> tuple[str, str]:
    if banner:
        fig.text(
            0.5,
            0.985,
            banner,
            ha="center",
            va="top",
            fontsize=7.5,
            color="#a03030",
            weight="bold",
        )
    svg = out / f"{name}.svg"
    png = out / f"{name}.png"
    fig.savefig(svg, format="svg", bbox_inches="tight")
    fig.savefig(png, format="png", bbox_inches="tight")
    plt.close(fig)
    return svg.name, png.name


def render_all(analysis: Analysis, stress: StressTest, out_dir: Path) -> list[Figure]:
    """Render the full figure set and write a caption manifest."""
    _style()
    out_dir.mkdir(parents=True, exist_ok=True)
    banner = (
        "SYNTHETIC PLACEHOLDER DATA — NOT EXPERIMENTAL"
        if analysis.quality.is_synthetic
        else None
    )
    figures: list[Figure] = []

    def emit(name: str, rank: int, caption: str, fig: plt.Figure) -> None:
        svg, png = _save(fig, out_dir, name, banner)
        figures.append(Figure(name, rank, caption, svg, png))

    # --- Rank 1: the non-additive lever -----------------------------------
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    levers = analysis.lever_effects
    ax.bar(
        [str(g) for g in levers["grade"]],
        levers["fold_change_td"],
        color=[_colour(str(g)) for g in levers["grade"]],
        width=0.6,
    )
    ax.axhline(1.0, color="#444", lw=0.8, ls="--")
    for i, v in enumerate(levers["fold_change_td"]):
        ax.text(i, v + 0.02, f"×{v:.2f}", ha="center", fontsize=9, weight="bold")
    ax.set_ylabel("Td multiplier from +10 wt% HPMC")
    ax.set_xlabel("HPMC viscosity grade")
    ax.set_title("The composition lever pays less at higher grade")
    ax.set_ylim(0.9, float(levers["fold_change_td"].max()) * 1.15)
    emit(
        "01_lever_non_additivity",
        1,
        "Marginal effect of substituting 10 wt% lactose with HPMC, evaluated at the "
        "design centroid for each grade. The effect shrinks monotonically with grade "
        "viscosity: the two levers are substitutes, not complements. This is the "
        "headline finding — the relationship a formulator cannot read off the "
        "single-factor trends.",
        fig,
    )

    # --- Rank 2: equivalence demonstration --------------------------------
    cross = [s for s in analysis.equivalence if s.spans_multiple_grades]
    if cross:
        best = max(cross, key=lambda s: (len(s.grades_spanned), s.n_members))
        fig, ax = plt.subplots(figsize=(5.6, 3.8))
        others = [
            (m.case, m.grade)
            for m in best.members
            if (m.case, m.grade) != (best.target_case, best.target_grade)
        ][:3]
        shown = [(best.target_case, best.target_grade), *others]
        for i, key in enumerate(shown):
            curve = analysis.observed_profiles.get(key)
            if curve is None:
                continue
            f2 = next(
                (m.f2 for m in best.members if (m.case, m.grade) == key), float("nan")
            )
            label = f"case {key[0]} / {key[1]}"
            if i > 0:
                label += f"  (f2 {f2:.0f})"
            ax.plot(
                analysis.time_grid,
                curve,
                marker="o",
                ms=3,
                lw=1.6,
                color=_colour(key[1], i),
                label=label,
            )
        ax.set_xlim(0, config.PLOT_MAX_TIME_H)
        ax.set_xlabel("time (h)")
        ax.set_ylabel("% released")
        ax.set_title("Different compositions, different grades, same profile")
        ax.legend(fontsize=7.5, frameon=False)
        emit(
            "02_equivalence_demonstration",
            2,
            "Measured profiles of formulations that differ in both composition and "
            "HPMC grade yet are f2-similar. Formulation freedom for a given target is "
            "real and can be spent on secondary criteria.",
            fig,
        )

    # --- Rank 3: stress-test degradation ----------------------------------
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    usable = [r for r in stress.results if r.estimable]
    ax.plot(
        [r.size for r in usable],
        [r.profile_rmse_pct for r in usable],
        marker="o",
        color="#3b7dd8",
        label="profile RMSE",
    )
    ax.axhline(
        stress.full_profile_rmse_pct, color="#444", ls="--", lw=0.9, label="full design"
    )
    if stress.recommended:
        ax.axvline(
            stress.recommended.size,
            color="#c0504d",
            ls=":",
            lw=1.4,
            label=f"recommended ({stress.recommended.size} runs)",
        )
    ax.set_xlabel("runs in the reduced design")
    ax.set_ylabel("profile RMSE (% released)")
    ax.set_title("What a smaller design costs")
    ax.legend(fontsize=7.5, frameon=False)
    emit(
        "03_stress_degradation",
        3,
        "Prediction error of D-optimal reduced designs against the full 33-run design. "
        "Designs below the recommended size are excluded even where apparent error is "
        "low, because they cannot estimate their own uncertainty.",
        fig,
    )

    # --- Rank 4: FDS ------------------------------------------------------
    fds_model = (
        "scheffe_linear"
        if "scheffe_linear" in analysis.design_diagnostics
        else "scheffe_quadratic"
    )
    diag = analysis.design_diagnostics.get(fds_model)
    if diag is not None and diag.fds_spv:
        spv = np.asarray(diag.fds_spv)
        fig, ax = plt.subplots(figsize=(5.2, 3.5))
        ax.plot(np.linspace(0, 1, len(spv)), spv, color="#3aa17e", lw=1.8)
        ax.set_xlabel("fraction of design space")
        ax.set_ylabel("scaled prediction variance")
        ax.set_title("Fraction of design space (FDS)")
        emit(
            "04_fds",
            4,
            "Scaled prediction variance across the convex hull of tested compositions "
            "crossed with the tested viscosity range. The flatter and lower the curve, "
            "the more uniformly trustworthy predictions are across the region.",
            fig,
        )

    # --- Rank 5: observed vs predicted (CV) -------------------------------
    folds = [f for f in analysis.cross_validation.folds if np.isfinite(f.profile_rmse_pct)]
    if folds:
        fig, ax = plt.subplots(figsize=(4.8, 4.4))
        obs = [f.observed.get("log10_td", np.nan) for f in folds]
        pred = [f.predicted.get("log10_td", np.nan) for f in folds]
        colours = [_colour(f.grade) for f in folds]
        ax.scatter(obs, pred, c=colours, s=30, edgecolor="white", lw=0.6, zorder=3)
        lo = float(np.nanmin(obs + pred))
        hi = float(np.nanmax(obs + pred))
        ax.plot([lo, hi], [lo, hi], color="#444", ls="--", lw=0.9)
        ax.set_xlabel("observed log₁₀(Td)")
        ax.set_ylabel("cross-validated prediction")
        ax.set_title("Leave-one-formulation-out prediction")
        emit(
            "05_cv_observed_vs_predicted",
            5,
            "Each point is a formulation predicted by a surface refitted without it, "
            "with all of its replicates held out together. This is the error attached "
            "to every prediction in the tool.",
            fig,
        )

    # --- Rank 6: response correlation heatmap -----------------------------
    space = analysis.response_space
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    matrix = space.pearson.to_numpy()
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(space.metrics)))
    ax.set_yticks(range(len(space.metrics)))
    ax.set_xticklabels(space.metrics, rotation=90, fontsize=7)
    ax.set_yticklabels(space.metrics, fontsize=7)
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Pearson r")
    ax.set_title(f"{len(space.metrics)} metrics, {space.n_components_90} real dimensions")
    emit(
        "06_response_correlation",
        6,
        "Correlation among the AC2 metrics. The block structure is why agreement "
        "between these metrics is not independent confirmation, and why conclusions "
        "are drawn only on the reduced key-response set.",
        fig,
    )

    # --- Rank 7: PCA biplot ----------------------------------------------
    if space.scores.shape[1] > 1:
        fig, ax = plt.subplots(figsize=(5.2, 4.6))
        ax.scatter(space.scores[:, 0], space.scores[:, 1], s=26, color="#8e5bb5", alpha=0.75)
        scale = float(np.abs(space.scores[:, :2]).max()) * 0.9
        for metric in space.loadings.index:
            x = float(space.loadings.loc[metric, "PC1"]) * scale
            y = float(space.loadings.loc[metric, "PC2"]) * scale
            ax.arrow(0, 0, x, y, color="#c0504d", lw=0.8, head_width=scale * 0.02)
            ax.text(x * 1.08, y * 1.08, metric, fontsize=6.5, color="#c0504d")
        ax.set_xlabel(f"PC1 ({space.explained_variance_ratio[0]:.0%})")
        ax.set_ylabel(f"PC2 ({space.explained_variance_ratio[1]:.0%})")
        ax.set_title("Response space")
        emit(
            "07_pca_biplot",
            7,
            "Principal components of the standardised response matrix. The axes "
            "correspond to Weibull scale, asymptote and shape, which is the empirical "
            "justification for modelling those parameters rather than each metric.",
            fig,
        )

    # --- Rank 8: raw profiles by grade (confirmatory) ---------------------
    grades = sorted({g for _, g in analysis.observed_profiles})
    fig, axes = plt.subplots(1, len(grades), figsize=(3.4 * len(grades), 3.3), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, grade in zip(axes, grades, strict=False):
        for (_case, g), curve in sorted(analysis.observed_profiles.items()):
            if g != grade:
                continue
            ax.plot(analysis.time_grid, curve, lw=1.1, alpha=0.85, color=_colour(grade))
        ax.axhline(config.CENSORING_PCT, color="#888", ls=":", lw=0.9)
        ax.set_xlim(0, config.PLOT_MAX_TIME_H)
        ax.set_title(grade)
        ax.set_xlabel("time (h)")
    axes[0].set_ylabel("% released")
    fig.suptitle("Measured profiles by grade (confirmatory)", y=1.02, fontsize=10)
    emit(
        "08_profiles_by_grade",
        8,
        "All measured profiles, split by grade, with the 80% censoring threshold "
        "marked. Confirms the known direction — higher grade releases more slowly — "
        "and is a pipeline sanity check rather than a finding.",
        fig,
    )

    # --- Rank 9: surface contour ------------------------------------------
    fig, ax = plt.subplots(figsize=(5.0, 4.2))
    points = analysis.design_points
    sc = ax.scatter(
        points["hpmc_wt"],
        points["api_wt"],
        c=points["log10_td_mean"],
        s=90,
        cmap="viridis",
        edgecolor="white",
        lw=0.7,
    )
    fig.colorbar(sc, ax=ax, label="log₁₀(Td), h")
    ax.set_xlabel("HPMC (wt%)")
    ax.set_ylabel("API (wt%)")
    ax.set_title("Design points in composition space")
    emit(
        "09_design_space",
        9,
        "The 11 compositions in API–HPMC space, coloured by fitted Weibull scale. "
        "Lactose is the balance to 100 wt%, so this plane carries the whole mixture.",
        fig,
    )

    manifest = [
        {"name": f.name, "rank": f.rank, "caption": f.caption, "svg": f.svg, "png": f.png}
        for f in sorted(figures, key=lambda f: f.rank)
    ]
    (out_dir / "captions.json").write_text(
        json.dumps(manifest, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    return figures
