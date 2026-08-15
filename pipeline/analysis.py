"""End-to-end analysis orchestration.

Runs the whole chain in dependency order and returns one payload that the report
writers, the figure generators and the dashboard exporter all consume, so no two
outputs can disagree about what the analysis found.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from pipeline import config
from pipeline.design.matrix import ModelSpec, build_model_matrix, code_process, term_names
from pipeline.design.report import analyse as analyse_design
from pipeline.design.report import design_points
from pipeline.equivalence.sets import (
    EquivalenceSet,
    EquivalenceSummary,
    build_equivalence_sets,
    summarise,
)
from pipeline.io.load import Database
from pipeline.io.quality import QualityReport, build_quality_report
from pipeline.profiles.table import build_design_point_table, build_replicate_table
from pipeline.responses.space import ResponseSpace, characterise
from pipeline.surfaces.model import BoxCoxResult, box_cox_analysis, sequential_sum_of_squares
from pipeline.surfaces.reduce import ReducedModel, choose_composition_degree, reduce_model
from pipeline.validation.cv import CrossValidation, leave_one_formulation_out

#: The AC2 metrics offered to the AC5 response-space analysis.
CANDIDATE_METRICS: tuple[str, ...] = (
    "log10_td",
    "weibull_beta",
    "weibull_f_inf",
    "t10",
    "t25",
    "t50",
    "mdt_h",
    "pct_4h",
    "pct_12h",
    "pct_24h",
    "peppas_n",
    "early_slope",
    "late_slope",
    "slope_ratio",
)

#: Weibull parameters carried by the two-stage surface.
SURFACE_RESPONSES: tuple[str, ...] = ("log10_td", "weibull_beta", "weibull_f_inf")


@dataclass
class Analysis:
    """Everything the pipeline computed, in one object."""

    db: Database
    quality: QualityReport
    replicates: pd.DataFrame
    design_points: pd.DataFrame
    design_diagnostics: dict[str, Any]
    lack_of_fit: Any
    design_table: pd.DataFrame
    response_space: ResponseSpace
    surfaces: dict[str, ReducedModel]
    box_cox: dict[str, BoxCoxResult]
    sequential: dict[str, Any]
    cross_validation: CrossValidation
    equivalence: tuple[EquivalenceSet, ...]
    equivalence_summary: EquivalenceSummary
    lever_effects: pd.DataFrame
    time_grid: np.ndarray
    observed_profiles: dict[tuple[int, str], np.ndarray]
    model_spec: ModelSpec


def _mean_profiles(db: Database, grid: np.ndarray) -> dict[tuple[int, str], np.ndarray]:
    out: dict[tuple[int, str], np.ndarray] = {}
    for (case, grade), group in db.profiles.groupby(["case", "grade"]):
        series = group.groupby("time_h")["pct_released"].mean().reindex(grid)
        out[(int(case), str(grade))] = series.to_numpy(dtype=float)
    return out


def _lever_effects(
    surface: ReducedModel, spec: ModelSpec, design: pd.DataFrame, kept: list[str]
) -> pd.DataFrame:
    """Marginal effect of swapping lactose for HPMC, evaluated at each grade.

    This is the question a formulator actually asks -- "if I add ten points of
    HPMC, what happens?" -- and the answer depends on grade whenever the
    composition x grade interaction is real. Reporting the interaction as a
    coefficient answers a different, less useful question.
    """
    all_labels = term_names(spec)
    keep_idx = [i for i, name in enumerate(all_labels) if name in set(kept)]
    beta = np.array([c.estimate for c in surface.fit.coefficients])

    centre = design[["api_wt", "hpmc_wt", "lactose_wt"]].mean().to_numpy() / 100.0
    step = 0.10  # ten percentage points of HPMC, taken from lactose

    rows: list[dict[str, float | str]] = []
    for grade, sub in design.groupby("grade"):
        v = float(sub["v_coded"].iloc[0])
        base = centre.copy()
        moved = centre + np.array([0.0, step, -step])
        if moved[2] < 0:
            moved = centre + np.array([0.0, centre[2], -centre[2]])
        pair = np.vstack([base, moved])
        matrix = build_model_matrix(pair, np.array([v, v]), spec)[:, keep_idx]
        pred = matrix @ beta
        rows.append(
            {
                "grade": str(grade),
                "viscosity_cp": float(sub["viscosity_cp"].iloc[0]),
                "v_coded": v,
                "delta_log10_td": float(pred[1] - pred[0]),
                "fold_change_td": float(10.0 ** (pred[1] - pred[0])),
            }
        )
    return pd.DataFrame(rows).sort_values("viscosity_cp").reset_index(drop=True)


def run_analysis(db: Database) -> Analysis:
    """Run the full chain on a loaded database."""
    quality = build_quality_report(db)
    replicates = build_replicate_table(db)
    points = build_design_point_table(replicates)

    diagnostics, lof, design_table = analyse_design(db)
    design_table = design_points(db)

    # --- AC5 first: everything downstream is restricted to the key responses.
    metric_frame = (
        replicates.groupby(["case", "grade"])[
            [m for m in CANDIDATE_METRICS if m in replicates.columns]
        ]
        .mean()
        .reset_index()
    )
    space = characterise(metric_frame, [m for m in CANDIDATE_METRICS if m in metric_frame])

    merged = points.merge(
        design_table[["case", "grade", "v_coded", "viscosity_cp"]],
        on=["case", "grade"],
        how="left",
        suffixes=("", "_dup"),
    )
    comp = merged[["api_wt", "hpmc_wt", "lactose_wt"]].to_numpy(dtype=float) / 100.0
    proc = merged["v_coded"].to_numpy(dtype=float)

    n_levels = int(merged["v_coded"].nunique())
    power = min(max(n_levels - 1, 0), 2)

    responses = {
        name: merged[f"{name}_mean"].to_numpy(dtype=float)
        for name in SURFACE_RESPONSES
        if f"{name}_mean" in merged.columns
    }

    degree, _ = choose_composition_degree(comp, proc, responses["log10_td"], power)
    spec = ModelSpec("scheffe", degree, power)

    surfaces: dict[str, ReducedModel] = {}
    box_cox: dict[str, BoxCoxResult] = {}
    sequential: dict[str, Any] = {}
    for name, values in responses.items():
        surfaces[name] = reduce_model(comp, proc, values, spec, response_name=name)
        raw = 10.0**values if name == "log10_td" else values
        box_cox[name] = box_cox_analysis(np.asarray(raw, dtype=float))
        mats = {
            d: build_model_matrix(comp, proc, ModelSpec("scheffe", d, power))
            for d in ("linear", "quadratic", "special_cubic")
        }
        sequential[name] = sequential_sum_of_squares(
            mats, values, ["linear", "quadratic", "special_cubic"]
        )

    grid = np.sort(db.profiles["time_h"].unique())
    observed = _mean_profiles(db, grid)

    cv = leave_one_formulation_out(
        comp,
        proc,
        responses,
        spec,
        kept_terms=list(surfaces["log10_td"].kept_terms),
        time_grid=grid,
        observed_profiles=observed,
        case_ids=merged["case"].to_numpy(),
        grades=merged["grade"].to_numpy(),
    )

    cv_by_point = {
        (f.case, f.grade): f.profile_rmse_pct for f in cv.folds
    }
    compositions = {
        int(row.case): (float(row.api_wt), float(row.hpmc_wt), float(row.lactose_wt))
        for row in merged.drop_duplicates("case").itertuples()
    }
    eq_sets = build_equivalence_sets(grid, observed, compositions, cv_by_point)
    eq_summary = summarise(eq_sets)

    levers = _lever_effects(
        surfaces["log10_td"], spec, merged, list(surfaces["log10_td"].kept_terms)
    )

    return Analysis(
        db=db,
        quality=quality,
        replicates=replicates,
        design_points=merged,
        design_diagnostics=diagnostics,
        lack_of_fit=lof,
        design_table=design_table,
        response_space=space,
        surfaces=surfaces,
        box_cox=box_cox,
        sequential=sequential,
        cross_validation=cv,
        equivalence=eq_sets,
        equivalence_summary=eq_summary,
        lever_effects=levers,
        time_grid=grid,
        observed_profiles=observed,
        model_spec=spec,
    )


def predict_profile(
    analysis: Analysis,
    api_wt: float,
    hpmc_wt: float,
    lactose_wt: float,
    log10_visc: float,
    times_h: np.ndarray,
) -> dict[str, Any]:
    """Predict a release profile, with its guardrail status attached (G3, G6)."""
    from pipeline.profiles.fits import weibull

    total = api_wt + hpmc_wt + lactose_wt
    comp = np.array([[api_wt, hpmc_wt, lactose_wt]], dtype=float) / max(total, 1e-9)

    levels = np.log10(analysis.design_points["viscosity_cp"].to_numpy(dtype=float))
    v = code_process(np.array([log10_visc]), levels)

    params: dict[str, float] = {}
    for name, reduced in analysis.surfaces.items():
        labels = term_names(analysis.model_spec)
        keep = [i for i, lbl in enumerate(labels) if lbl in set(reduced.kept_terms)]
        matrix = build_model_matrix(comp, v, analysis.model_spec)[:, keep]
        params[name] = float(reduced.fit.predict(matrix)[0])

    td = 10.0 ** params["log10_td"]
    curve = weibull(times_h, params["weibull_f_inf"], td, params["weibull_beta"])
    curve = np.clip(np.maximum.accumulate(curve), 0.0, 100.0)

    in_visc_range = bool(levels.min() - 1e-9 <= log10_visc <= levels.max() + 1e-9)

    return {
        "parameters": params,
        "td_h": td,
        "profile": curve.tolist(),
        "times_h": np.asarray(times_h, dtype=float).tolist(),
        "in_viscosity_range": in_visc_range,
        "cv_profile_rmse_pct": analysis.cross_validation.profile_rmse_pct,
        "seed": config.PIPELINE_SEED,
    }
