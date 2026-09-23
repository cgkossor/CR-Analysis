"""Privacy-safe audit of the disintegration sheet: section ``T``.

The same contract as ``pipeline.audit``: every value is a bool or an int, and
every label is a fixed identifier written here. Nothing read from the workbook
can reach the report. It answers "does the real sheet meet this section's
assumptions?" without disclosing a time, a grade or an ID.

Registered as stage ``T`` in :func:`pipeline.audit.run_audit`.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from pipeline.disintegration import settings
from pipeline.disintegration.load import load_disintegration

if TYPE_CHECKING:
    from pipeline.analysis import Analysis
    from pipeline.audit import AuditReport

SECTION = "T"


def _flag(x: float, test: bool) -> bool:
    return bool(np.isfinite(x) and test)


def audit_disintegration(
    report: AuditReport, path: Path, analysis: Analysis | None, *, dedicated: bool = False
) -> None:
    """Add the T entries. Raises on failure, so ``_run_stage`` can locate it."""
    add = report.add
    add(SECTION, "separate_file", dedicated)
    data = load_disintegration(path, dedicated=dedicated)
    add(SECTION, "sheet_present", data is not None)
    if data is None:
        return
    add(SECTION, "per_tablet_rows", data.per_tablet_rows)

    long = data.long
    add(SECTION, "rows", data.n_rows)
    add(SECTION, "columns", data.n_columns)
    add(SECTION, "replicate_columns", data.n_replicate_columns)
    add(SECTION, "unit_code", data.unit_code)
    add(SECTION, "test_end_column", data.test_end_from_sheet)
    add(SECTION, "composition_columns", data.n_composition_columns)
    add(SECTION, "api_column", data.has_api_column)
    add(SECTION, "unrecognised_columns", data.n_unrecognised_columns)
    add(SECTION, "nonnumeric_cells", data.n_nonnumeric)
    add(SECTION, "nonpositive_times", data.n_nonpositive)
    add(SECTION, "replicates_run", len(long))
    add(SECTION, "censored_replicates", int(long["censored"].sum()))
    add(SECTION, "synthetic_flag", data.is_synthetic)
    add(SECTION, "truth_present", bool(data.truth))

    if analysis is None:
        return
    from pipeline.disintegration.analysis import run_disintegration

    r = run_disintegration(analysis, data)
    pts, m = r.precision.points, r.matching
    add(SECTION, "formulations", len(pts))
    add(SECTION, "censored_formulations", int(pts["censored"].sum()))
    uncensored_by_grade = {
        g: int(((r.matched["grade"] == g) & ~r.matched["dt_censored"]).sum())
        for g in r.grade_order
    }
    add(SECTION, "grades_fully_censored", sum(1 for v in uncensored_by_grade.values() if v == 0))
    add(SECTION, "grades_ge3_points", sum(1 for v in uncensored_by_grade.values() if v >= 3))
    add(SECTION, "reps_min", int(pts["n"].min()))
    add(SECTION, "reps_max", int(pts["n"].max()))
    add(SECTION, "formulations_lt_min_reps", int((pts["n"] < settings.DT_MIN_REPS).sum()))
    ok_pts = pts[~pts["censored"]]
    add(SECTION, "formulations_cv_high", int((ok_pts["cv"] > settings.DT_CV_WARN).sum()))
    add(SECTION, "dixon_flags", len(r.precision.flags))
    add(SECTION, "icc_ge_09", _flag(r.precision.icc, r.precision.icc >= 0.9))
    add(SECTION, "dt_without_dissolution", len(m.dt_without_dissolution))
    add(SECTION, "dissolution_without_dt", len(m.dissolution_without_dt))
    add(SECTION, "composition_mismatch", len(m.composition_mismatch))
    add(SECTION, "composition_sum_off", len(m.composition_sum_off))
    add(SECTION, "grades_without_viscosity", len(m.grades_without_viscosity))
    add(SECTION, "dt_lt_t50", len(r.plausibility.below_t50))
    add(SECTION, "dt_lt_t80", len(r.plausibility.below_t80))

    c = r.correlation
    td = c.by_key("td_h")
    t80 = c.by_key("t80")
    add(SECTION, "matched_points", m.n_matched)
    add(SECTION, "correlation_points", td.n if td else 0)
    rho = td.spearman if td else float("nan")
    add(SECTION, "rho_td_finite", bool(np.isfinite(rho)))
    add(SECTION, "rho_td_positive", _flag(rho, rho > 0))
    add(SECTION, "rho_td_ge_expected", _flag(rho, rho >= settings.RHO_EXPECTED))
    rho80 = t80.spearman if t80 else float("nan")
    add(SECTION, "rho_t80_finite", bool(np.isfinite(rho80)))
    add(SECTION, "rho_t80_ge_expected", _flag(rho80, rho80 >= settings.RHO_EXPECTED))
    add(SECTION, "within_grade_r_ge_09", sum(
        1 for w in c.within_grade if np.isfinite(w.pearson) and w.pearson >= 0.9))
    add(SECTION, "sign_disagreements", sum(
        1 for x in c.table if np.isfinite(x.pearson) and not x.sign_agrees))

    a = r.grades.ancova
    add(SECTION, "ancova_ran", a is not None)
    add(SECTION, "ancova_residual_df", a.residual_df if a is not None else -1)
    add(SECTION, "ancova_separate_slopes", bool(a is not None and a.selected == "separate slopes"))
    add(SECTION, "ancova_vif_gt_10", bool(a is not None and a.vif_ln_td > 10))
    add(SECTION, "complete_blocks", r.grades.n_complete_blocks)
    add(SECTION, "block_model_ran", r.grades.block_dt is not None)

    fit = r.dt_fit
    add(SECTION, "doe_dt_estimable", bool(fit is not None and fit.usable))
    add(SECTION, "doe_dt_r2_ge_05", bool(
        fit is not None and np.isfinite(fit.anova.r_squared) and fit.anova.r_squared >= 0.5))
    dt_resp = fit.response if fit is not None else None
    add(SECTION, "doe_dt_points_dropped", dt_resp.n_missing if dt_resp is not None else -1)

    add(SECTION, "loo_models_run", len(r.loo))
    add(SECTION, "loo_nan_predictions", sum(int((~np.isfinite(x.predicted)).sum()) for x in r.loo))
    add(SECTION, "recovery_total", len(r.recovery))
    add(SECTION, "recovery_covered", sum(1 for x in r.recovery if x.covered))
