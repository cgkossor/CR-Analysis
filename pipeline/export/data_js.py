"""Serialise the analysis into ``dashboard/data.js`` (G7, G10).

Data ships as a plain assignment to a global, never as a fetched JSON file: an
``index.html`` opened straight from the filesystem cannot ``fetch()`` a sibling
file under any browser's file:// origin rules, so a fetch-based dashboard would
fail exactly in the situation G7 requires it to work.

Serialisation is sorted and fixed-precision throughout, so two runs of the
pipeline produce byte-identical output (G10).
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from pipeline import config
from pipeline.analysis import Analysis
from pipeline.diagnostics import Diagnostics
from pipeline.glossary import as_dict as glossary_dict
from pipeline.optimize.goals import as_payload as goal_payload
from pipeline.profiles.grid import project_onto_grid
from pipeline.stress.subsets import StressTest

if TYPE_CHECKING:
    from pipeline.disintegration.analysis import DisintegrationAnalysis
    from pipeline.doe.analysis import DoeAnalysis

#: Decimal places used for every exported float. Fixed so that two runs cannot
#: differ in the last bit of a repr.
PRECISION = 6


def _clean(value: Any) -> Any:
    """Convert numpy/pandas scalars into JSON-safe Python values, deterministically."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if math.isnan(number) or math.isinf(number):
            return None
        return round(number, PRECISION)
    if isinstance(value, (np.ndarray, list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    if value is None or isinstance(value, str):
        return value
    if pd.isna(value):
        return None
    return str(value)


def _frame(frame: pd.DataFrame, columns: list[str] | None = None) -> list[dict[str, Any]]:
    subset = frame[columns] if columns else frame
    return [_clean(record) for record in subset.to_dict(orient="records")]


def _replicate_sd(a: Analysis, case: int, grade: str) -> np.ndarray:
    """Replicate standard deviation at each canonical grid point.

    This is WITHIN-BATCH analytical repeatability -- vessels from one compression
    batch. It is a much smaller quantity than the cross-validated prediction
    error and must never be presented in its place; see the guidelines.
    """
    subset = a.db.profiles[
        (a.db.profiles["case"] == case) & (a.db.profiles["grade"] == grade)
    ]
    stacked = []
    for _, group in subset.groupby("replicate"):
        ordered = group.sort_values("time_h")
        stacked.append(
            project_onto_grid(
                ordered["time_h"].to_numpy(dtype=float),
                ordered["pct_released"].to_numpy(dtype=float),
                a.time_grid,
                a.time_grid_info.max_gap_h,
            )
        )
    if len(stacked) < 2:
        return np.full(len(a.time_grid), np.nan)
    matrix = np.vstack(stacked)
    # A grid point outside some replicates' measured window has fewer than two
    # values there, and a sample SD of one point is undefined rather than zero.
    # Compute only where it is defined; elsewhere the band is simply absent.
    counts = np.sum(np.isfinite(matrix), axis=0)
    out = np.full(matrix.shape[1], np.nan)
    usable = counts >= 2
    if np.any(usable):
        with np.errstate(invalid="ignore"):
            out[usable] = np.nanstd(matrix[:, usable], axis=0, ddof=1)
    return out


#: Contour grids are downsampled for the browser. Full resolution is for the
#: printed figures; the dashboard redraws on every interaction and a 60x60 grid
#: per response per grade would dominate the payload for no visible gain.
DASHBOARD_GRID = 26


def _downsample(rows: tuple[tuple[float | None, ...], ...], target: int) -> list[list[Any]]:
    if not rows:
        return []
    step_r = max(len(rows) // target, 1)
    step_c = max(len(rows[0]) // target, 1)
    return [
        [None if v is None else round(float(v), 2) for v in row[::step_c]]
        for row in rows[::step_r]
    ]


def _doe_payload(doe: DoeAnalysis) -> dict[str, Any]:
    """Serialise a classical DoE analysis for the dashboard."""
    responses = []
    unavailable = []
    for ra in doe.responses:
        if not ra.usable:
            # Say why rather than omitting it. A response that quietly disappears
            # looks like it was never asked for.
            unavailable.append({
                "key": ra.response.spec.key,
                "label": ra.response.spec.label,
                "reason": (
                    "The model is not estimable from the points this response has. "
                    + " ".join(ra.notes)
                ).strip(),
                "n_missing": ra.response.n_missing,
                "n_total": ra.response.n_total,
            })
            continue
        table = ra.anova
        responses.append({
            "key": ra.response.spec.key,
            "label": ra.response.spec.label,
            "units": ra.response.spec.units,
            "spec_note": ra.response.spec.note,
            "model": ra.spec.label,
            "terms": list(ra.kept_terms),
            # Coefficients travel with the response so the formulator tool can
            # search on the quantities the goals are actually written in --
            # t50 and % released -- rather than on Weibull parameters.
            "coefficients": [
                {"name": c.name, "estimate": _clean(c.estimate)}
                for c in ra.fit.coefficients
            ],
            "process_power": ra.spec.process_power,
            "n_obs": table.n_obs,
            "n_missing": ra.response.n_missing,
            "coverage_note": ra.response.coverage_note,
            "summary": {
                "s": _clean(table.s),
                "r2": _clean(table.r_squared),
                "adj_r2": _clean(table.adj_r_squared),
                "pred_r2": _clean(table.pred_r_squared),
                "residual_df": table.residual_df,
            },
            "anova": {
                "model": {
                    "source": table.model_row.source, "df": table.model_row.df,
                    "adj_ss": _clean(table.model_row.adj_ss),
                    "adj_ms": _clean(table.model_row.adj_ms),
                    "f": _clean(table.model_row.f_value),
                    "p": _clean(table.model_row.p_value),
                },
                "rows": [
                    {
                        "source": r.source, "df": r.df,
                        "adj_ss": _clean(r.adj_ss), "adj_ms": _clean(r.adj_ms),
                        "f": _clean(r.f_value), "p": _clean(r.p_value),
                        "group": r.is_group,
                    }
                    for r in table.rows
                ],
                "residual": {"df": table.residual_df, "ss": _clean(table.residual_ss)},
                "total": {"df": table.total_df, "ss": _clean(table.total_ss)},
                "notes": list(table.notes),
            },
            "effects": {
                "terms": [e.term for e in ra.ranking.effects],
                "abs_t": _clean([e.abs_t for e in ra.ranking.effects]),
                "significant": [e.significant for e in ra.ranking.effects],
                "quantiles": _clean(ra.ranking.half_normal_quantiles),
                "t_critical": _clean(ra.ranking.t_critical),
                "bonferroni": _clean(ra.ranking.bonferroni_t),
                "note": ra.ranking.note,
            },
            "traces": [
                {
                    "factor": t.factor, "label": t.label,
                    "x": _clean(t.x_values), "y": _clean(t.y_values),
                }
                for t in ra.traces
            ],
            "interaction": {
                "factor": ra.interactions[0].factor,
                "label": ra.interactions[0].label,
                "x": _clean(ra.interactions[0].x_values),
                "series": [
                    {"grade": g, "y": _clean(ys)} for g, ys in ra.interactions[0].series
                ],
                "parallel": ra.interactions[0].parallel,
                "divergence": _clean(ra.interactions[0].divergence),
            },
            "grids": [
                {
                    "grade": g.grade,
                    "viscosity_cp": _clean(g.viscosity_cp),
                    "api_axis": _clean(list(g.api_axis)[::max(len(g.api_axis)//DASHBOARD_GRID,1)]),
                    "hpmc_axis": _clean(
                        list(g.hpmc_axis)[::max(len(g.hpmc_axis)//DASHBOARD_GRID,1)]
                    ),
                    "z": _downsample(g.z, DASHBOARD_GRID),
                    "points": _clean([[p[0], p[1]] for p in g.design_points]),
                }
                for g in ra.grids
            ],
            "scale": _clean(list(ra.scale)),
            "takeaways": {
                "anova": ra.takeaway_anova,
                "effects": ra.takeaway_effects,
                "traces": ra.takeaway_traces,
                "contour": ra.takeaway_contour,
                "interaction": ra.interactions[0].interpretation,
            },
            "notes": list(ra.notes),
        })
    return {
        "responses": responses,
        "unavailable": unavailable,
        "method_notes": doe.method_notes,
    }


#: The disintegration responses offered in the DoE tab. log10_dt exists for
#: model comparison inside the section and would only duplicate dt_h there.
DT_DOE_KEYS = ("dt_h",)


def _disintegration_payload(r: DisintegrationAnalysis) -> dict[str, Any]:
    """Serialise the disintegration section for its dashboard tab."""
    m = r.matched
    td = r.correlation.by_key("td_h")
    block = r.grades.block_dt
    pts = r.precision.points
    return {
        "synthetic": r.is_synthetic,
        "test_end_h": _clean(r.data.test_end_h),
        "grade_order": list(r.grade_order),
        "summary": {
            "formulations": len(pts),
            "replicates_run": len(r.data.long),
            "censored_replicates": int(r.data.long["censored"].sum()),
            "censored_formulations": int(pts["censored"].sum()),
            "matched": r.matching.n_matched,
            "icc": _clean(r.precision.icc),
            "spearman_td": _clean(td.spearman) if td else None,
            "spearman_td_n": td.n if td else 0,
        },
        "points": [
            {
                "case": int(row.case),
                "grade": str(row.grade),
                "api_wt": _clean(row.api_wt),
                "hpmc_wt": _clean(row.hpmc_wt),
                "lactose_wt": _clean(row.lactose_wt),
                "dt_h": _clean(row.dt_h),
                "ci_lo_h": _clean(row.ci_lo_h),
                "ci_hi_h": _clean(row.ci_hi_h),
                "n": int(row.n),
                "cv": _clean(row.cv),
                "censored": bool(row.dt_censored),
                "td_h": _clean(row.td_h),
                "t50": _clean(row.t50),
                "t80": _clean(row.t80),
                "lag": _clean(row.erosion_lag),
            }
            for row in m.itertuples()
        ],
        "replicates": [
            {
                "case": int(row.case),
                "grade": str(row.grade),
                "replicate": int(row.replicate),
                "dt_h": _clean(row.dt_h),
                "censored": bool(row.censored),
            }
            for row in r.data.long.itertuples()
        ],
        "correlations": [
            {
                "key": c.metric.key,
                "label": c.metric.label,
                "n": c.n,
                "spearman": _clean(c.spearman),
                "lo": _clean(c.spearman_ci[0]),
                "hi": _clean(c.spearman_ci[1]),
            }
            for c in r.correlation.table
        ],
        "grade_ratios": [
            {
                "a": c.a,
                "b": c.b,
                "ratio": _clean(c.ratio),
                "lo": _clean(c.ci[0]),
                "hi": _clean(c.ci[1]),
                "p": _clean(c.p_adj),
            }
            for c in (block.contrasts if block is not None else ())
        ],
        "prediction": [
            {"key": x.key, "label": x.label, "n": x.n, "q2": _clean(x.q2)} for x in r.loo
        ],
        "doe_keys": [
            k for k in DT_DOE_KEYS if (fit := r.doe.by_key(k)) is not None and fit.usable
        ],
    }


def build_payload(
    analysis: Analysis,
    stress: StressTest | None = None,
    diagnostics: Diagnostics | None = None,
    disintegration: DisintegrationAnalysis | None = None,
) -> dict[str, Any]:
    """Assemble the dashboard payload."""
    a = analysis
    quality = a.quality

    profiles: list[dict[str, Any]] = []
    for (case, grade), curve in sorted(a.observed_profiles.items()):
        row = a.design_points[
            (a.design_points["case"] == case) & (a.design_points["grade"] == grade)
        ]
        replicate_curves = []
        subset = a.db.profiles[
            (a.db.profiles["case"] == case) & (a.db.profiles["grade"] == grade)
        ]
        for rep, group in subset.groupby("replicate"):
            ordered = group.sort_values("time_h")
            # Each replicate carries ITS OWN measured times. Exporting values
            # alone forced the dashboard to index them against the canonical
            # grid positionally, so once the two lengths diverged -- which is
            # exactly what reconciling ragged sampling times causes -- the curve
            # was drawn across the wrong x values and compressed into the first
            # few hours. x and y travel together now.
            replicate_curves.append(
                {
                    "replicate": int(rep),
                    "times_h": _clean(ordered["time_h"].to_numpy()),
                    "pct": _clean(ordered["pct_released"].to_numpy()),
                }
            )
        profiles.append(
            {
                "case": int(case),
                "grade": str(grade),
                "id": str(row["api"].iloc[0]) + f"_{grade}_C{case:02d}" if len(row) else "",
                "api_wt": _clean(row["api_wt"].iloc[0]) if len(row) else None,
                "hpmc_wt": _clean(row["hpmc_wt"].iloc[0]) if len(row) else None,
                "lactose_wt": _clean(row["lactose_wt"].iloc[0]) if len(row) else None,
                "viscosity_cp": _clean(row["viscosity_cp"].iloc[0]) if len(row) else None,
                "mean_pct": _clean(curve),
                "sd_pct": _clean(_replicate_sd(a, int(case), str(grade))),
                "replicates": replicate_curves,
                "censoring": str(row["censoring_status"].iloc[0]) if len(row) else "",
                "log10_td": _clean(row["log10_td_mean"].iloc[0]) if len(row) else None,
                "log10_td_sd": _clean(row["log10_td_sd"].iloc[0]) if len(row) else None,
                "beta": _clean(row["weibull_beta_mean"].iloc[0]) if len(row) else None,
                "f_inf": _clean(row["weibull_f_inf_mean"].iloc[0]) if len(row) else None,
                "asymptote_identified": bool(
                    row["n_asymptote_identified"].iloc[0] == row["n_replicates"].iloc[0]
                )
                if len(row)
                else False,
            }
        )

    surfaces: dict[str, Any] = {}
    for name, reduced in sorted(a.surfaces.items()):
        fit = reduced.fit
        surfaces[name] = {
            "model": fit.model_label,
            "kept_terms": list(reduced.kept_terms),
            "dropped_terms": list(reduced.dropped_terms),
            "rationale": reduced.rationale,
            "r2": _clean(fit.r_squared),
            "adj_r2": _clean(fit.adj_r_squared),
            "pred_r2": _clean(fit.pred_r_squared),
            "rmse": _clean(fit.rmse),
            "adequate_precision": _clean(fit.adequate_precision),
            "adequate_precision_flagged": fit.adequate_precision_flagged,
            "r2_gap_flagged": fit.r2_gap_flagged,
            "df_residual": int(fit.df_residual),
            "notes": list(fit.notes),
            "coefficients": [
                {
                    "name": c.name,
                    "estimate": _clean(c.estimate),
                    "std_error": _clean(c.std_error),
                    "t": _clean(c.t_value),
                    "p": _clean(c.p_value),
                    "ci_low": _clean(c.ci_low),
                    "ci_high": _clean(c.ci_high),
                }
                for c in fit.coefficients
            ],
            "residuals": _clean(fit.residuals),
            "fitted": _clean(fit.fitted),
            "studentised": _clean(fit.studentised),
            "leverage": _clean(fit.leverage),
            "cooks_distance": _clean(fit.cooks_distance),
            "box_cox": {
                "lambda": _clean(a.box_cox[name].lambda_hat),
                "ci_low": _clean(a.box_cox[name].ci_low),
                "ci_high": _clean(a.box_cox[name].ci_high),
                "recommendation": a.box_cox[name].recommendation,
                "applicable": a.box_cox[name].applicable,
                "note": a.box_cox[name].note,
            },
            "sequential": [
                {
                    "model": s.model,
                    "terms": int(s.n_terms),
                    "added": int(s.added_terms),
                    "ss": _clean(s.sum_squares),
                    "f": _clean(s.f_value),
                    "p": _clean(s.p_value),
                }
                for s in a.sequential.get(name, ())
            ],
        }

    space = a.response_space
    response_space = {
        "metrics": list(space.metrics),
        "key_responses": list(space.key_responses),
        "pearson": {
            "labels": list(space.pearson.columns),
            "matrix": _clean(space.pearson.to_numpy()),
        },
        "spearman": {
            "labels": list(space.spearman.columns),
            "matrix": _clean(space.spearman.to_numpy()),
        },
        "explained_variance": _clean(space.explained_variance_ratio),
        "cumulative_variance": _clean(space.cumulative_variance),
        "n_components_90": int(space.n_components_90),
        "loadings": {
            "metrics": list(space.loadings.index),
            "pc1": _clean(space.loadings["PC1"].to_numpy()),
            "pc2": _clean(space.loadings["PC2"].to_numpy())
            if space.loadings.shape[1] > 1
            else [],
        },
        "scores": _clean(space.scores[:, :2]) if space.scores.shape[1] > 1 else [],
        "groups": [
            {
                "members": list(g.members),
                "representative": g.representative,
                "justification": g.justification,
                "max_abs_correlation": _clean(g.max_abs_correlation),
                "structural": g.contains_structural,
            }
            for g in space.groups
        ],
        "structural_notes": list(space.structural_notes),
        "empirical_notes": list(space.empirical_notes),
        "consistency": space.weibull_consistency,
        "coverage_notes": list(space.notes),
    }

    cv = a.cross_validation
    validation = {
        "profile_rmse_pct": _clean(cv.profile_rmse_pct),
        "profile_rmse_worst_pct": _clean(cv.profile_rmse_pct_worst),
        "median_f2": _clean(cv.median_f2),
        "n_folds": int(cv.n_folds),
        "rmse_by_response": _clean(cv.rmse_by_response),
        "notes": list(cv.notes),
        "folds": [
            {
                "case": f.case,
                "grade": f.grade,
                "profile_rmse_pct": _clean(f.profile_rmse_pct),
                "f2": _clean(f.f2),
                "f2_valid": f.f2_valid,
                "observed": _clean(f.observed),
                "predicted": _clean(f.predicted),
            }
            for f in cv.folds
        ],
    }

    equivalence = {
        "summary": {
            "n_targets": a.equivalence_summary.n_targets,
            "max_set_size": a.equivalence_summary.max_set_size,
            "median_set_size": _clean(a.equivalence_summary.median_set_size),
            "cross_grade_count": len(a.equivalence_summary.cross_grade_targets),
            "isolated": [
                {"case": c, "grade": g} for c, g in a.equivalence_summary.isolated_targets
            ],
            "demonstration": a.equivalence_summary.demonstration,
        },
        "sets": [
            {
                "case": s.target_case,
                "grade": s.target_grade,
                "spans_multiple_grades": s.spans_multiple_grades,
                "grades": list(s.grades_spanned),
                "members": [
                    {
                        "case": m.case,
                        "grade": m.grade,
                        "f2": _clean(m.f2),
                        "api_wt": _clean(m.api_wt),
                        "hpmc_wt": _clean(m.hpmc_wt),
                        "lactose_wt": _clean(m.lactose_wt),
                        "cv_rmse_pct": _clean(m.cv_profile_rmse_pct),
                    }
                    for m in s.members
                ],
            }
            for s in a.equivalence
        ],
    }

    design_models = {
        key: {
            "terms": int(d.n_terms),
            "rank": int(d.rank),
            "estimable": d.estimable,
            "condition_number": _clean(d.condition_number),
            "d_criterion": _clean(d.d_efficiency),
            "a_criterion": _clean(d.a_efficiency),
            "g_efficiency": _clean(d.g_efficiency),
            "max_leverage": _clean(d.max_leverage),
            "leverage": _clean(d.leverage),
            "fds": _clean(d.fds_spv[:: max(len(d.fds_spv) // 200, 1)]) if d.fds_spv else [],
            "aliases": [
                {
                    "term": e.term,
                    "exact": e.exact,
                    "loads_on": _clean(
                        dict(
                            sorted(
                                ((k, v) for k, v in e.aliased_with.items() if abs(v) > 1e-8),
                                key=lambda kv: -abs(kv[1]),
                            )[:6]
                        )
                    ),
                }
                for e in d.aliases
            ],
            "notes": list(d.notes),
        }
        for key, d in sorted(a.design_diagnostics.items())
    }

    payload: dict[str, Any] = {
        "glossary": glossary_dict(),
        "meta": {
            "source_file": quality.source_name,
            "is_synthetic": quality.is_synthetic,
            "provenance_notes": list(quality.provenance_notes[:6]),
            "vessel_volume_ml": _clean(quality.vessel_volume_ml),
            "viscosity_source": quality.viscosity_source,
            "seed": config.PIPELINE_SEED,
            "f2_threshold": config.F2_SIMILAR_THRESHOLD,
            "censoring_pct": config.CENSORING_PCT,
            "model": a.model_spec.label,
            "plot_min_time_h": _clean(config.PLOT_MIN_TIME_H),
            "plot_max_time_h": _clean(config.PLOT_MAX_TIME_H),
            "plot_min_release_pct": _clean(config.PLOT_MIN_RELEASE_PCT),
            "plot_max_release_pct": _clean(config.PLOT_MAX_RELEASE_PCT),
        },
        "quality": {
            "apis": list(quality.apis),
            "n_ids": quality.n_ids,
            "n_replicates": quality.n_replicates,
            "cases_per_api": _clean(quality.cases_per_api),
            "grades_per_api": {k: list(v) for k, v in sorted(quality.grades_per_api.items())},
            "expected_cells": quality.expected_cells,
            "observed_cells": quality.observed_cells,
            "missing_cells": [
                {"api": a_, "case": c, "grade": g} for a_, c, g in quality.missing_cells
            ],
            "timepoints_h": _clean(quality.timepoints_h),
            "uniform_spacing": quality.uniform_spacing,
            "samples_within_2h": quality.samples_within_2h,
            "monotonicity_violations": quality.monotonicity_violations,
            "profiles_with_violation": quality.profiles_with_violation,
            "largest_backward_step_pct": _clean(quality.largest_backward_step_pct),
            "values_above_100": quality.values_above_100,
            "max_pct_released": _clean(quality.max_pct_released),
            "censored_profiles": quality.censored_profiles,
            "fully_censored_ids": list(quality.fully_censored_ids),
            "partially_censored_ids": list(quality.partially_censored_ids),
            "peppas_min_points": quality.peppas_min_points,
            "peppas_failing": list(quality.peppas_failing_profiles),
            "mass_range_mg": _clean(quality.mass_range_mg),
            "warnings": list(quality.warnings),
            "mixture": {
                "is_mixture": quality.mixture.is_mixture,
                "total": _clean(quality.mixture.constant_total),
                "rank": quality.mixture.rank,
                "singular_values": _clean(quality.mixture.singular_values),
                "max_deviation": _clean(quality.mixture.max_abs_deviation),
                "verdict": quality.mixture.verdict,
            },
        },
        "grid_h": _clean(a.time_grid),
        "time_grid": {
            "n_points": a.time_grid_info.n_points,
            "n_raw_times": a.time_grid_info.n_raw_times,
            "n_distinct_vectors": a.time_grid_info.n_distinct_vectors,
            "max_shift_min": _clean(a.time_grid_info.max_shift_h * 60.0),
            "reconciled": a.time_grid_info.collapsed,
            "notes": list(a.time_grid_info.notes),
        },
        "profiles": profiles,
        "design_points": _frame(
            a.design_points,
            [
                "case",
                "grade",
                "api_wt",
                "hpmc_wt",
                "lactose_wt",
                "viscosity_cp",
                "log10_visc",
                "v_coded",
                "log10_td_mean",
                "log10_td_sd",
                "weibull_beta_mean",
                "weibull_beta_sd",
                "weibull_f_inf_mean",
                "t50_mean",
                "mdt_h_mean",
                "censoring_status",
                "n_replicates",
                "peak_pct",
            ],
        ),
        "design_models": design_models,
        "lack_of_fit": {
            "estimable": a.lack_of_fit.estimable_at_replicate_level,
            "pure_error_df": a.lack_of_fit.pure_error_df_replicate,
            "n_distinct_points": a.lack_of_fit.n_distinct_points,
            "n_observations": a.lack_of_fit.n_observations,
            "description": a.lack_of_fit.describe(),
        },
        "surfaces": surfaces,
        "response_space": response_space,
        "validation": validation,
        "equivalence": equivalence,
        "levers": _frame(a.lever_effects),
        "doe": _doe_payload(a.doe),
        "goals": goal_payload(),
    }

    if diagnostics is not None:
        payload["diagnostics"] = {
            "verdict": diagnostics.verdict,
            "checks": [
                {
                    "section": c.section,
                    "name": c.name,
                    "status": c.status,
                    "value": c.display,
                    "note": c.note,
                }
                for c in diagnostics.checks
            ],
        }

    if disintegration is not None:
        payload["disintegration"] = _disintegration_payload(disintegration)
        # Disintegration time gets the full DoE treatment in the DoE tab, listed
        # after the dissolution responses and marked with the study it came from.
        extra = _doe_payload(disintegration.doe)
        for resp in extra["responses"]:
            if resp["key"] in DT_DOE_KEYS:
                resp["study"] = "disintegration"
                payload["doe"]["responses"].append(resp)

    if stress is not None:
        payload["stress"] = {
            "full_profile_rmse_pct": _clean(stress.full_profile_rmse_pct),
            "recommended_size": stress.recommended_size,
            "breakdown_size": stress.breakdown_size,
            "rationale": stress.rationale,
            "notes": list(stress.notes),
            "recommended_points": [
                {"case": c, "grade": g}
                for c, g in (stress.recommended.selected if stress.recommended else ())
            ],
            "curve": [
                {
                    "size": r.size,
                    "estimable": r.estimable,
                    "profile_rmse_pct": _clean(r.profile_rmse_pct),
                    "worst_pct": _clean(r.profile_rmse_worst_pct),
                    "d_criterion": _clean(r.d_criterion),
                    "direction_agrees": r.lever_direction_agrees,
                    "equivalence_jaccard": _clean(r.equivalence_jaccard),
                }
                for r in stress.results
            ],
        }

    return payload


#: Figure folders shown in the dashboard's Figures tab, in display order.
GALLERY_GROUPS: tuple[tuple[str, str], ...] = (
    ("headlines", "Headline figures"),
    ("", "Dissolution figures"),
    ("disintegration", "Disintegration figures"),
)


def figure_gallery(
    figures_dir: Path, dashboard_dir: Path, *, disintegration: bool = False
) -> list[dict[str, Any]]:
    """The rendered figures, read from each folder's ``captions.json``.

    Paths are relative to the dashboard, so the page shows the PNGs straight
    from ``outputs/figures`` when opened from disk. The disintegration folder
    is listed only when this run produced that section, so figures left over
    from an earlier run are never shown as current.
    """
    items: list[dict[str, Any]] = []
    for sub, group in GALLERY_GROUPS:
        if sub == "disintegration" and not disintegration:
            continue
        folder = figures_dir / sub if sub else figures_dir
        manifest = folder / "captions.json"
        if not manifest.exists():
            continue
        for entry in json.loads(manifest.read_text(encoding="utf-8")):
            png = folder / str(entry["png"])
            if not png.exists():
                continue
            items.append({
                "group": group,
                "id": str(entry["id"]),
                "caption": str(entry["caption"]),
                "src": os.path.relpath(png, dashboard_dir).replace(os.sep, "/"),
            })
    return items


def write_data_js(payload: dict[str, Any], path: str | Path) -> Path:
    """Write ``data.js`` as a global assignment, deterministically."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(payload, indent=1, sort_keys=True, ensure_ascii=True, allow_nan=False)
    destination.write_text(
        "// Generated by pipeline.run -- do not edit.\n"
        "// Assigned to a global rather than fetched: an index.html opened from the\n"
        "// filesystem cannot fetch() a sibling file, which is exactly the situation\n"
        "// guardrail G7 requires the dashboard to work in.\n"
        f"window.CR_DATA = {body};\n",
        encoding="utf-8",
        newline="\n",
    )
    return destination
