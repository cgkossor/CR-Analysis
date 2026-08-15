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
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pipeline import config
from pipeline.analysis import Analysis
from pipeline.stress.subsets import StressTest

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


def build_payload(analysis: Analysis, stress: StressTest | None = None) -> dict[str, Any]:
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
            replicate_curves.append(
                {
                    "replicate": int(rep),
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
            "plot_max_time_h": _clean(config.PLOT_MAX_TIME_H),
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
    }

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
