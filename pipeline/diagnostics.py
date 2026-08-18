"""A single scannable health check for a pipeline run.

Six analysis reports are the right depth once you trust the run. This is the
thing to read first: every check that could invalidate what follows, as a
pass/warn/fail with the number behind it, on one page.

Emitted as Markdown for reading and JSON for scripting. A check is FAIL only
when it makes downstream output untrustworthy, WARN when it constrains how the
output may be read, and PASS otherwise -- so "any FAIL" is a genuine stop signal
rather than a style complaint.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

from pipeline import config
from pipeline.analysis import Analysis
from pipeline.stress.subsets import StressTest

Status = Literal["PASS", "WARN", "FAIL", "INFO"]

@dataclass(frozen=True)
class Check:
    """One diagnostic line."""

    section: str
    name: str
    status: Status
    value: Any
    note: str = ""

    @property
    def display(self) -> str:
        v = self.value
        if isinstance(v, bool):
            return "true" if v else "false"
        if v is None:
            return "n/a"
        if isinstance(v, float):
            if not np.isfinite(v):
                return "not computable"
            return f"{v:,.4g}"
        return str(v)


@dataclass
class Diagnostics:
    """The full set of checks, plus a one-line verdict."""

    checks: list[Check] = field(default_factory=list)

    def add(
        self, section: str, name: str, status: Status, value: Any, note: str = ""
    ) -> None:
        self.checks.append(Check(section, name, status, value, note))

    @property
    def failures(self) -> list[Check]:
        return [c for c in self.checks if c.status == "FAIL"]

    @property
    def warnings(self) -> list[Check]:
        return [c for c in self.checks if c.status == "WARN"]

    @property
    def verdict(self) -> str:
        if self.failures:
            return (
                f"NOT USABLE — {len(self.failures)} check(s) failed. Results below them "
                "cannot be relied on until these are resolved."
            )
        if self.warnings:
            return (
                f"USABLE WITH CAVEATS — {len(self.warnings)} warning(s). The analysis "
                "ran, but these constrain how the results may be read."
            )
        return "OK — every check passed."


def _finite(x: float | None) -> bool:
    return x is not None and bool(np.isfinite(x))


def collect(analysis: Analysis, stress: StressTest | None = None) -> Diagnostics:
    """Run every check against a completed analysis."""
    d = Diagnostics()
    q = analysis.quality
    grid = analysis.time_grid_info
    cv = analysis.cross_validation
    space = analysis.response_space

    # --- Provenance ---------------------------------------------------------
    d.add(
        "Provenance",
        "data is synthetic placeholder",
        "WARN" if q.is_synthetic else "PASS",
        q.is_synthetic,
        "no result is an experimental finding" if q.is_synthetic else "real database",
    )
    d.add("Provenance", "source file", "INFO", q.source_name)
    d.add(
        "Provenance",
        "grade viscosities from workbook",
        "PASS" if q.viscosity_source == "workbook" else "WARN",
        q.viscosity_source == "workbook",
        "" if q.viscosity_source == "workbook" else "using config fallback; confirm values",
    )
    d.add("Provenance", "vessel volume (mL)", "INFO", q.vessel_volume_ml)

    # --- Contents -----------------------------------------------------------
    d.add("Contents", "APIs", "INFO", len(q.apis))
    d.add("Contents", "formulations", "INFO", q.n_ids)
    d.add("Contents", "replicates each", "INFO", q.n_replicates)
    complete = q.observed_cells >= q.expected_cells
    d.add(
        "Contents",
        "design crossing complete",
        "PASS" if complete else "WARN",
        complete,
        f"{q.observed_cells} of {q.expected_cells} intended cells",
    )
    d.add(
        "Contents",
        "grades present",
        "PASS" if len(q.grades_per_api.get(q.apis[0], ())) >= 3 else "WARN",
        len(q.grades_per_api.get(q.apis[0], ())) if q.apis else 0,
        "3 grades identify a curved viscosity effect; 2 identify only a line",
    )

    # --- Timepoints ---------------------------------------------------------
    d.add("Timepoints", "distinct time vectors", "INFO", grid.n_distinct_vectors)
    d.add("Timepoints", "raw distinct timestamps", "INFO", grid.n_raw_times)
    d.add("Timepoints", "canonical grid points", "INFO", grid.n_points)
    d.add(
        "Timepoints",
        "schedule reconciled",
        "INFO",
        grid.collapsed,
        f"largest adjustment {grid.max_shift_h * 60:.2f} min" if grid.collapsed else "",
    )
    lost = any("WARNING" in n for n in grid.notes)
    d.add(
        "Timepoints",
        "time resolution preserved",
        "WARN" if lost else "PASS",
        not lost,
        "distinct nominal pulls were merged; lower TIME_CLUSTER_REL" if lost else "",
    )

    # --- Data quality -------------------------------------------------------
    d.add(
        "Data quality",
        "observations above 100%",
        "INFO",
        q.values_above_100,
        f"max {q.max_pct_released:.2f}% — preserved as measured, never clipped",
    )

    # A consistent plateau above 100% is usually not assay scatter. Scatter is
    # symmetric about the true value; a systematic offset means the denominator
    # is wrong -- actual tablet content exceeding mass x label API%. This
    # quantifies it and says so. It changes no calculation.
    plateaus = [
        float(np.nanmax(curve))
        for curve in analysis.observed_profiles.values()
        if np.any(np.isfinite(curve))
    ]
    complete = [p for p in plateaus if p >= config.COMPLETE_RELEASE_PCT]
    if complete:
        median_plateau = float(np.median(complete))
        offset = median_plateau - 100.0
        systematic = offset > 2.0
        d.add(
            "Data quality",
            "median plateau of completed profiles (%)",
            "WARN" if systematic else "PASS",
            median_plateau,
            (
                f"a systematic +{offset:.1f}% offset across {len(complete)} profiles. "
                "Random assay scatter is symmetric about the true value, so a "
                "one-sided offset of this size more often means the dose denominator "
                "is understated — actual tablet content above mass x label API% — "
                "than that the assay is noisy. Worth checking content uniformity or "
                "the assumed API fraction. Nothing has been rescaled."
            )
            if systematic
            else f"offset {offset:+.1f}% — consistent with ordinary assay scatter",
        )
    d.add("Data quality", "negative observations", "PASS" if not q.negative_values else "WARN",
          q.negative_values)
    d.add(
        "Data quality",
        "monotonicity violations",
        "INFO",
        q.monotonicity_violations,
        f"across {q.profiles_with_violation} profiles, "
        f"largest {q.largest_backward_step_pct:.2f} pp",
    )
    d.add("Data quality", "tablet mass range (mg)", "INFO",
          f"{q.mass_range_mg[0]:.1f}–{q.mass_range_mg[1]:.1f}")

    # --- Guardrails ---------------------------------------------------------
    d.add(
        "Guardrails",
        "G4 all profiles Peppas-reportable",
        "PASS" if not q.peppas_failing_profiles else "WARN",
        not q.peppas_failing_profiles,
        f"{len(q.peppas_failing_profiles)} profile(s) below {config.PEPPAS_MIN_POINTS} points "
        f"under {config.PEPPAS_MAX_PCT:g}%" if q.peppas_failing_profiles else
        f"minimum {q.peppas_min_points} points in window",
    )
    d.add("Guardrails", "G5 fully censored formulations", "INFO", len(q.fully_censored_ids),
          f"never reach {config.CENSORING_PCT:g}%")
    d.add(
        "Guardrails",
        "G5 partially censored",
        "WARN" if q.partially_censored_ids else "PASS",
        len(q.partially_censored_ids),
        "replicates straddle the threshold; not collapsible to one flag"
        if q.partially_censored_ids else "",
    )
    d.add(
        "Guardrails",
        "G1 solubility features gated",
        "PASS",
        len(q.apis) < 2,
        "single API — cross-API panels render as insufficient-data"
        if len(q.apis) < 2 else "2+ APIs; contrast still confounded with molecule identity",
    )

    # --- Design -------------------------------------------------------------
    m = q.mixture
    d.add("Design", "A1 mixture (constant sum)", "INFO", m.is_mixture,
          f"rank {m.rank}, max deviation {m.max_abs_deviation:.2e}")
    d.add(
        "Design",
        "composition model estimable",
        "PASS" if m.rank >= 2 else "FAIL",
        m.rank >= 2,
        "" if m.rank >= 2 else "composition space is degenerate; a surface cannot be fitted",
    )
    d.add("Design", "model form", "INFO", analysis.model_spec.label)
    lof = analysis.lack_of_fit
    d.add("Design", "lack-of-fit estimable", "INFO", lof.estimable_at_replicate_level,
          f"{lof.pure_error_df_replicate} pure-error df (within-batch only, "
          "anti-conservative)")

    # --- Surfaces -----------------------------------------------------------
    for name, reduced in analysis.surfaces.items():
        fit = reduced.fit
        d.add(f"Surface: {name}", "estimable", "PASS" if fit.estimable else "FAIL",
              fit.estimable)
        d.add(f"Surface: {name}", "terms / residual df", "INFO",
              f"{fit.n_terms} / {fit.df_residual}")
        d.add(f"Surface: {name}", "R2", "INFO", fit.r_squared)
        d.add(f"Surface: {name}", "predicted R2 (PRESS)", "INFO", fit.pred_r_squared)
        d.add(
            f"Surface: {name}",
            "adj-vs-pred R2 gap acceptable",
            "WARN" if fit.r2_gap_flagged else "PASS",
            not fit.r2_gap_flagged,
            f"gap {fit.adj_r_squared - fit.pred_r_squared:.3f} exceeds "
            f"{config.R2_GAP_FLAG}" if fit.r2_gap_flagged else "",
        )
        d.add(
            f"Surface: {name}",
            "adequate precision",
            "WARN" if fit.adequate_precision_flagged else "PASS",
            fit.adequate_precision,
            f"below {config.ADEQUATE_PRECISION_FLAG:g}; do not optimise on this surface"
            if fit.adequate_precision_flagged else "",
        )

    # --- Validation (G6) ----------------------------------------------------
    cv_ok = _finite(cv.profile_rmse_pct)
    d.add(
        "Validation (G6)",
        "cross-validated error computable",
        "PASS" if cv_ok else "FAIL",
        cv_ok,
        "" if cv_ok else "every prediction would be shown without an error estimate",
    )
    d.add("Validation (G6)", "profile RMSE (% released)", "INFO", cv.profile_rmse_pct)
    d.add("Validation (G6)", "worst fold RMSE", "INFO", cv.profile_rmse_pct_worst)
    d.add("Validation (G6)", "median f2 predicted vs observed", "INFO", cv.median_f2)
    d.add("Validation (G6)", "folds completed", "INFO", cv.n_folds)
    missing = sum(1 for f in cv.folds if not np.isfinite(f.profile_rmse_pct))
    d.add(
        "Validation (G6)",
        "folds with no profile error",
        "WARN" if missing else "PASS",
        missing,
        "excluded from the summary RMSE" if missing else "",
    )

    # --- Response space -----------------------------------------------------
    d.add("Response space", "metrics analysed", "INFO", len(space.metrics))
    d.add("Response space", "components to 90% variance", "INFO", space.n_components_90)
    d.add("Response space", "key responses", "INFO", len(space.key_responses),
          ", ".join(space.key_responses))
    consistent = "must be revisited" not in space.weibull_consistency
    d.add(
        "Response space",
        "dimensionality consistent with the model",
        "PASS" if consistent else "WARN",
        consistent,
        "" if consistent else "AC4's two-stage approach needs revisiting",
    )

    # --- Equivalence & design economy --------------------------------------
    es = analysis.equivalence_summary
    d.add("Equivalence", "formulations with a cross-grade match", "INFO",
          len(es.cross_grade_targets), f"of {es.n_targets}")
    d.add("Equivalence", "formulations with no alternative", "INFO",
          len(es.isolated_targets))
    if stress is not None:
        d.add("Design economy", "recommended minimum runs", "INFO",
              stress.recommended_size, f"of {len(analysis.design_points)}")
        d.add(
            "Design economy",
            "a reduced design is viable",
            "PASS" if stress.recommended is not None else "WARN",
            stress.recommended is not None,
            "" if stress.recommended is not None else "run the full design",
        )

    return d


def render_markdown(d: Diagnostics, analysis: Analysis) -> str:
    out: list[str] = ["# Run diagnostics\n"]
    add = out.append

    add(f"**{d.verdict}**\n")
    if analysis.quality.is_synthetic:
        add(
            "> **PLACEHOLDER DATA — SYNTHETIC, NOT EXPERIMENTAL.** "
            "No value here is a measurement.\n"
        )

    if d.failures:
        add("## Failures\n")
        for c in d.failures:
            add(f"- **{c.section} / {c.name}** = `{c.display}` — {c.note}")
        add("")
    if d.warnings:
        add("## Warnings\n")
        for c in d.warnings:
            tail = f" — {c.note}" if c.note else ""
            add(f"- **{c.section} / {c.name}** = `{c.display}`{tail}")
        add("")

    add("## All checks\n")
    current = ""
    for c in d.checks:
        if c.section != current:
            current = c.section
            add(f"\n### {current}\n")
            add("| check | value | status | note |")
            add("|---|---|---|---|")
        mark = "" if c.status == "INFO" else f"**{c.status}**"
        add(f"| {c.name} | `{c.display}` | {mark} | {c.note} |")

    return "\n".join(out) + "\n"


def render_json(d: Diagnostics) -> str:
    payload = {
        "verdict": d.verdict,
        "n_failures": len(d.failures),
        "n_warnings": len(d.warnings),
        "checks": [
            {
                "section": c.section,
                "name": c.name,
                "status": c.status,
                "value": (
                    None
                    if isinstance(c.value, float) and not np.isfinite(c.value)
                    else c.value
                ),
                "note": c.note,
            }
            for c in d.checks
        ],
    }
    return json.dumps(payload, indent=1, sort_keys=False) + "\n"


def render_console(d: Diagnostics) -> str:
    """Compact summary for the terminal, so a bad run is obvious immediately."""
    lines = [d.verdict]
    for c in d.failures + d.warnings:
        lines.append(f"  [{c.status}] {c.section} / {c.name} = {c.display}"
                     + (f"  ({c.note})" if c.note else ""))
    return "\n".join(lines)


def write(analysis: Analysis, stress: StressTest | None, reports: Path) -> Diagnostics:
    d = collect(analysis, stress)
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "diagnostics.md").write_text(
        render_markdown(d, analysis), encoding="utf-8", newline="\n"
    )
    (reports / "diagnostics.json").write_text(render_json(d), encoding="utf-8", newline="\n")
    return d
