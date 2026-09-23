"""Run the classical DoE analysis for every named response.

One entry point, one result object per response, so the reports, the figures and
the dashboard all read the same numbers and cannot disagree about what was found.

Model selection per response, rather than one model imposed on all of them: the
composition degree is chosen by sequential sums of squares against that response,
because the shape that fits release timing is not necessarily the shape that fits
completeness.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from pipeline.design.matrix import ModelSpec, build_model_matrix, term_names
from pipeline.doe import effects, interpret, surface
from pipeline.doe.anova import AnovaTable
from pipeline.doe.anova import build as build_anova
from pipeline.doe.responses import ResponseData, collect
from pipeline.surfaces.model import SurfaceFit
from pipeline.surfaces.reduce import choose_composition_degree, reduce_model


@dataclass(frozen=True)
class ResponseAnalysis:
    """Everything the classical DoE produced for one named response."""

    response: ResponseData
    spec: ModelSpec
    kept_terms: tuple[str, ...]
    fit: SurfaceFit
    anova: AnovaTable
    ranking: effects.EffectRanking
    traces: tuple[effects.Trace, ...]
    grade_trace: effects.Trace | None
    interactions: tuple[effects.InteractionProfile, ...]
    grids: tuple[surface.SurfaceGrid, ...]
    scale: tuple[float, float]

    takeaway_anova: str
    takeaway_effects: str
    takeaway_traces: str
    takeaway_contour: str
    notes: tuple[str, ...] = field(default=())

    @property
    def usable(self) -> bool:
        return self.fit.estimable and self.anova.residual_df > 0


@dataclass(frozen=True)
class DoeAnalysis:
    """The classical analysis across every response."""

    responses: tuple[ResponseAnalysis, ...]
    method_notes: dict[str, str]

    def by_key(self, key: str) -> ResponseAnalysis | None:
        for r in self.responses:
            if r.response.spec.key == key:
                return r
        return None


def _grade_levels(design_points: pd.DataFrame) -> list[tuple[str, float, float]]:
    seen: dict[str, tuple[float, float]] = {}
    for row in design_points.itertuples():
        seen.setdefault(str(row.grade), (float(row.v_coded), float(row.viscosity_cp)))
    return sorted(
        ((g, v, cp) for g, (v, cp) in seen.items()), key=lambda t: t[2]
    )


def run(
    design_points: pd.DataFrame,
    replicates: pd.DataFrame,
    max_process_power: int,
    responses: Sequence[ResponseData] | None = None,
) -> DoeAnalysis:
    """Fit and characterise every named response.

    ``responses`` defaults to the dissolution responses built by :func:`collect`.
    Another study measured on the same design points (disintegration, say) passes
    its own, aligned row for row with ``design_points``, and gets the identical
    treatment.
    """
    comp = design_points[["api_wt", "hpmc_wt", "lactose_wt"]].to_numpy(dtype=float) / 100.0
    proc = design_points["v_coded"].to_numpy(dtype=float)
    grades = _grade_levels(design_points)
    centroid = comp.mean(axis=0)

    ranges = {
        name: (
            float(design_points[f"{name}_wt"].min()),
            float(design_points[f"{name}_wt"].max()),
        )
        for name in ("api", "hpmc", "lactose")
    }

    out: list[ResponseAnalysis] = []
    if responses is None:
        responses = collect(design_points, replicates)
    for data in responses:
        mask = data.available
        if not data.usable:
            continue

        y = data.values[mask]
        comp_m, proc_m = comp[mask], proc[mask]

        # The process power must come from the levels this response actually has,
        # not from the design as a whole. Censoring is not random: t80 goes
        # undefined for the slowest formulations, which can remove an entire
        # viscosity grade. Fitting a quadratic in log-viscosity to two surviving
        # levels is not merely optimistic, it is inestimable -- and the response
        # would silently vanish from the analysis rather than say so.
        # Step the process power down until the surviving points can actually
        # estimate it. Counting distinct levels is not enough: censoring drops the
        # slowest formulations, so a grade can survive with too few points to
        # support composition terms crossed with v^2, leaving the matrix
        # rank-deficient while all three levels are still nominally present. That
        # is what made this response silently vanish from the analysis.
        levels_here = int(np.unique(np.round(proc_m, 9)).size)
        power = min(max(levels_here - 1, 0), max_process_power)
        while power > 0:
            probe = build_model_matrix(
                comp_m, proc_m, ModelSpec("scheffe", "linear", power)
            )
            if np.linalg.matrix_rank(probe) == probe.shape[1]:
                break
            power -= 1

        response_notes: list[str] = []
        if power < max_process_power:
            shape = {0: "constant", 1: "linear"}.get(power, f"degree-{power}")
            response_notes.append(
                f"Limited to a {shape} term in log-viscosity. The full design supports "
                f"degree {max_process_power}, but after censoring the {int(mask.sum())} "
                f"points that remain for this response cannot estimate it — grade "
                "effects are described more coarsely here than for the responses that "
                "keep every point."
            )

        degree, _ = choose_composition_degree(comp_m, proc_m, y, power)
        spec = ModelSpec("scheffe", degree, power)
        labels = term_names(spec)

        if build_model_matrix(comp_m, proc_m, spec).shape[1] >= len(y):
            # Not enough points for this shape; step back to linear.
            spec = ModelSpec("scheffe", "linear", power)
            labels = term_names(spec)

        reduced = reduce_model(comp_m, proc_m, y, spec, response_name=data.spec.key)
        fit = reduced.fit
        kept = list(reduced.kept_terms)
        keep_idx = [i for i, n in enumerate(labels) if n in set(kept)]
        matrix = build_model_matrix(comp_m, proc_m, spec)[:, keep_idx]

        table = build_anova(matrix, y, kept, response_name=data.spec.key)
        ranking = effects.standardised_effects(
            [(c.name, c.estimate, c.std_error) for c in fit.coefficients],
            fit.df_residual,
        )

        beta = np.array([c.estimate for c in fit.coefficients])
        traces = effects.cox_traces(
            beta, keep_idx, spec, centroid, None, float(np.median(proc)), ranges
        )
        g_trace = effects.grade_trace(
            beta, keep_idx, spec, centroid,
            np.array([g[1] for g in grades]), [g[0] for g in grades],
        )
        interactions = tuple(
            effects.interaction_profile(
                beta, keep_idx, spec, centroid, component, ranges[component],
                [(g[0], g[1]) for g in grades],
            )
            for component in ("hpmc", "api")
        )
        grids = surface.build_grids(beta, keep_idx, spec, comp, grades)

        out.append(
            ResponseAnalysis(
                response=data,
                spec=spec,
                kept_terms=tuple(kept),
                fit=fit,
                anova=table,
                ranking=ranking,
                traces=tuple(traces),
                grade_trace=g_trace,
                interactions=interactions,
                grids=tuple(grids),
                scale=surface.shared_scale(list(grids)),
                takeaway_anova=interpret.anova_takeaway(table, data),
                takeaway_effects=interpret.effects_takeaway(ranking, data),
                takeaway_traces=interpret.trace_takeaway(traces, data),
                takeaway_contour=interpret.contour_takeaway(
                    [(g.grade, g.z_min, g.z_max) for g in grids], data
                ),
                notes=(
                    (reduced.rationale,)
                    + tuple(response_notes)
                    + ((data.coverage_note,) if data.n_missing else ())
                ),
            )
        )

    return DoeAnalysis(responses=tuple(out), method_notes=dict(interpret.METHOD_NOTES))
