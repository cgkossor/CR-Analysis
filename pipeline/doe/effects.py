"""Effect estimates, response traces and interaction profiles.

A straight main-effects plot is not available on a mixture. Components sum to a
constant, so one cannot be varied while the others are held fixed -- something
has to absorb the change, and the answer depends on what. The mixture analogue is
the **Cox response trace**: vary one component and let the others take up the
difference in their existing proportions. That is the question a formulator is
actually asking ("what if I add polymer and take it out of everything else?"),
and it is what Minitab draws for a mixture design.

Grade is an ordinary process variable, not a mixture component, so it gets a
conventional main-effect profile.

The Pareto and half-normal plots rank standardised effects. Both answer "which
terms matter", but they fail differently: the Pareto shows magnitude against a
significance threshold and needs an error estimate to draw one; the half-normal
needs no error estimate at all, because inert effects fall on a straight line
through the origin and real ones depart from it. On a design with few residual
degrees of freedom the half-normal is the more trustworthy of the two.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats

from pipeline.design.matrix import ModelSpec, build_model_matrix


@dataclass(frozen=True)
class StandardisedEffect:
    """One term's effect, scaled by its own standard error."""

    term: str
    estimate: float
    std_error: float
    t_value: float
    abs_t: float
    p_value: float
    significant: bool


@dataclass(frozen=True)
class EffectRanking:
    """Standardised effects with the reference lines both plots need."""

    effects: tuple[StandardisedEffect, ...]
    t_critical: float
    bonferroni_t: float
    df_residual: int
    half_normal_quantiles: tuple[float, ...]
    note: str = ""


@dataclass(frozen=True)
class Trace:
    """A response trace: fitted response along one factor's direction."""

    factor: str
    label: str
    x_values: tuple[float, ...]
    y_values: tuple[float, ...]
    x_units: str
    within_hull: tuple[bool, ...]
    reference_point: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class InteractionProfile:
    """Fitted response against one composition factor, one line per grade."""

    factor: str
    label: str
    x_values: tuple[float, ...]
    series: tuple[tuple[str, tuple[float, ...]], ...]
    parallel: bool
    divergence: float
    interpretation: str


def standardised_effects(
    coefficients: list[tuple[str, float, float]], df_residual: int
) -> EffectRanking:
    """Rank terms by |t|, with the reference lines for Pareto and half-normal."""
    effects: list[StandardisedEffect] = []
    for name, estimate, se in coefficients:
        t_val = estimate / se if se > 0 else float("nan")
        p_val = (
            float(2.0 * (1.0 - stats.t.cdf(abs(t_val), df_residual)))
            if np.isfinite(t_val) and df_residual > 0
            else float("nan")
        )
        effects.append(
            StandardisedEffect(
                term=name,
                estimate=estimate,
                std_error=se,
                t_value=t_val,
                abs_t=abs(t_val) if np.isfinite(t_val) else 0.0,
                p_value=p_val,
                significant=bool(np.isfinite(p_val) and p_val < 0.05),
            )
        )
    effects.sort(key=lambda e: -e.abs_t)

    t_crit = float(stats.t.ppf(0.975, df_residual)) if df_residual > 0 else float("nan")
    # Bonferroni line: the threshold if every term were tested simultaneously.
    # With this many correlated terms the uncorrected line is optimistic.
    m = max(len(effects), 1)
    bonf = (
        float(stats.t.ppf(1.0 - 0.025 / m, df_residual)) if df_residual > 0 else float("nan")
    )

    # Half-normal plotting positions: inert effects fall on a line through the
    # origin, so departures identify the real ones without needing an error term.
    n = len(effects)
    quantiles = [
        float(stats.norm.ppf(0.5 + 0.5 * (i + 0.5) / n)) for i in range(n)
    ] if n else []

    note = ""
    if df_residual > 0 and df_residual < 5:
        note = (
            f"Only {df_residual} residual degrees of freedom, so the significance line "
            "is poorly determined. Read the half-normal plot, which needs no error "
            "estimate, in preference to the Pareto."
        )
    return EffectRanking(
        effects=tuple(effects),
        t_critical=t_crit,
        bonferroni_t=bonf,
        df_residual=df_residual,
        half_normal_quantiles=tuple(reversed(quantiles)),
        note=note,
    )


def _predict(
    beta: np.ndarray, keep: list[int], spec: ModelSpec,
    comps: np.ndarray, proc: np.ndarray,
) -> np.ndarray:
    return build_model_matrix(comps, proc, spec)[:, keep] @ beta


def cox_traces(
    beta: np.ndarray,
    keep: list[int],
    spec: ModelSpec,
    centroid: np.ndarray,
    hull_test: object,
    process_value: float,
    component_ranges: dict[str, tuple[float, float]],
    n_points: int = 41,
) -> list[Trace]:
    """Cox response trace for each mixture component.

    Varying component *i* by ``delta`` moves the others to
    ``x_j * (1 - x_i_new) / (1 - x_i_old)`` -- their existing proportions,
    rescaled to fill what is left. Any other convention answers a different
    question, so the convention is stated rather than assumed.
    """
    names = ("api", "hpmc", "lactose")
    traces: list[Trace] = []

    for i, name in enumerate(names):
        lo, hi = component_ranges[name]
        grid = np.linspace(lo / 100.0, hi / 100.0, n_points)
        comps = np.zeros((n_points, 3))
        inside: list[bool] = []

        for k, xi in enumerate(grid):
            remaining = 1.0 - centroid[i]
            row = np.zeros(3)
            row[i] = xi
            if remaining > 1e-9:
                scale = (1.0 - xi) / remaining
                for j in range(3):
                    if j != i:
                        row[j] = centroid[j] * scale
            comps[k] = row
            inside.append(bool(np.all(row >= -1e-9)) and bool(np.all(row <= 1.0 + 1e-9)))

        y = _predict(beta, keep, spec, comps, np.full(n_points, process_value))
        traces.append(
            Trace(
                factor=name,
                label=f"{name.upper() if name == 'api' else name.capitalize()} (wt%)",
                x_values=tuple(float(v * 100.0) for v in grid),
                y_values=tuple(float(v) for v in y),
                x_units="wt%",
                within_hull=tuple(inside),
                reference_point={
                    "api": f"{centroid[0] * 100:.1f}",
                    "hpmc": f"{centroid[1] * 100:.1f}",
                    "lactose": f"{centroid[2] * 100:.1f}",
                },
            )
        )
    return traces


def grade_trace(
    beta: np.ndarray,
    keep: list[int],
    spec: ModelSpec,
    centroid: np.ndarray,
    process_levels: np.ndarray,
    grade_labels: list[str],
) -> Trace:
    """Main-effect profile for grade, which is an ordinary process variable."""
    comps = np.tile(centroid, (len(process_levels), 1))
    y = _predict(beta, keep, spec, comps, np.asarray(process_levels, dtype=float))
    return Trace(
        factor="grade",
        label="HPMC viscosity grade",
        x_values=tuple(float(v) for v in process_levels),
        y_values=tuple(float(v) for v in y),
        x_units="coded log10(viscosity)",
        within_hull=tuple(True for _ in process_levels),
        reference_point={"grades": ", ".join(grade_labels)},
    )


def interaction_profile(
    beta: np.ndarray,
    keep: list[int],
    spec: ModelSpec,
    centroid: np.ndarray,
    component: str,
    component_range: tuple[float, float],
    process_levels: list[tuple[str, float]],
    n_points: int = 31,
) -> InteractionProfile:
    """Response against one component, drawn once per grade.

    Parallel lines mean the two levers are additive. Divergence is the
    interaction, and it is read off the picture rather than from a coefficient --
    which is the point of drawing it.
    """
    index = {"api": 0, "hpmc": 1, "lactose": 2}[component]
    lo, hi = component_range
    grid = np.linspace(lo / 100.0, hi / 100.0, n_points)

    comps = np.zeros((n_points, 3))
    for k, xi in enumerate(grid):
        remaining = 1.0 - centroid[index]
        row = np.zeros(3)
        row[index] = xi
        if remaining > 1e-9:
            scale = (1.0 - xi) / remaining
            for j in range(3):
                if j != index:
                    row[j] = centroid[j] * scale
        comps[k] = row

    series: list[tuple[str, tuple[float, ...]]] = []
    curves: list[np.ndarray] = []
    for label, v in process_levels:
        y = _predict(beta, keep, spec, comps, np.full(n_points, v))
        series.append((label, tuple(float(t) for t in y)))
        curves.append(y)

    # How far from parallel: compare the end-to-end change of each line.
    spans = [float(c[-1] - c[0]) for c in curves]
    divergence = float(max(spans) - min(spans)) if len(spans) > 1 else 0.0
    typical = float(np.mean(np.abs(spans))) if spans else 0.0
    relative = divergence / typical if typical > 1e-12 else 0.0
    parallel = relative < 0.15

    if parallel:
        interpretation = (
            f"The lines are close to parallel, so {component.upper()} and grade act "
            "roughly independently here: what you gain from one does not depend much "
            "on the other."
        )
    else:
        strongest = process_levels[int(np.argmax(np.abs(spans)))][0]
        weakest = process_levels[int(np.argmin(np.abs(spans)))][0]
        interpretation = (
            f"The lines are not parallel — the effect of {component.upper()} differs by "
            f"{relative:.0%} between grades, strongest at {strongest} and weakest at "
            f"{weakest}. The two levers are partial substitutes rather than additive, "
            "so reaching for both buys less than the sum of their separate effects."
        )

    return InteractionProfile(
        factor=component,
        label=f"{component.upper()} x grade",
        x_values=tuple(float(v * 100.0) for v in grid),
        series=tuple(series),
        parallel=parallel,
        divergence=relative,
        interpretation=interpretation,
    )
