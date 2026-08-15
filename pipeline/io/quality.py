"""AC1 data-quality report and the assumption checks that gate the modelling path.

This module characterises the database *before* anything is fitted. It reports
what is there, what is missing, and which structural facts decide the modelling
approach downstream -- in particular the A1 mixture test, whose outcome
determines whether a Scheffe/slack-variable form is required (AC3).

Nothing here repairs data. Anomalies are surfaced, counted and named (G2).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from pipeline import config
from pipeline.io.load import Database


@dataclass(frozen=True)
class MixtureTest:
    """Result of the A1 constant-sum test."""

    sums: tuple[float, ...]
    max_abs_deviation: float
    singular_values: tuple[float, ...]
    rank: int
    condition_ratio: float
    is_mixture: bool
    constant_total: float | None
    fixed_components: tuple[str, ...]
    n_independent_dims: int

    @property
    def verdict(self) -> str:
        if not self.is_mixture:
            return (
                "NOT a mixture: component totals vary across cases. Standard polynomial "
                "RSM is admissible; check VIF and condition number before fitting."
            )
        if self.n_independent_dims <= 1:
            return (
                f"MIXTURE with only {self.n_independent_dims} independent composition "
                f"dimension(s) -- component(s) {list(self.fixed_components)} are fixed. "
                "The composition axis is degenerate; a surface in three components "
                "cannot be fitted and the design must be treated as one-dimensional."
            )
        return (
            f"MIXTURE: components sum to {self.constant_total:g} on every case; the "
            "composition matrix has rank 2, so the three components are exactly "
            "collinear. Polynomial RSM on all three with an intercept is SINGULAR. "
            "A Scheffe canonical form or slack-variable reduction is required (AC3)."
        )


@dataclass(frozen=True)
class QualityReport:
    """Everything AC1 requires, plus the structural facts AC3/AC4 depend on."""

    source_name: str
    is_synthetic: bool
    provenance_notes: tuple[str, ...]
    vessel_volume_ml: float
    viscosity_source: str
    grade_viscosity_cp: dict[str, float]

    apis: tuple[str, ...]
    n_ids: int
    n_replicates: int
    cases_per_api: dict[str, int]
    grades_per_api: dict[str, tuple[str, ...]]
    replicate_counts: dict[int, int]

    n_timepoints: int
    timepoints_h: tuple[float, ...]
    distinct_time_vectors: int
    uniform_spacing: bool
    samples_within_2h: int

    missing_cells: tuple[tuple[str, int, str], ...]
    expected_cells: int
    observed_cells: int
    cases_missing_entirely: tuple[int, ...]

    mixture: MixtureTest

    monotonicity_violations: int
    profiles_with_violation: int
    largest_backward_step_pct: float
    values_above_100: int
    max_pct_released: float
    negative_values: int

    censored_profiles: int
    fully_censored_ids: tuple[str, ...]
    partially_censored_ids: tuple[str, ...]

    peppas_point_counts: dict[str, int]
    peppas_min_points: int
    peppas_failing_profiles: tuple[str, ...]

    mass_range_mg: tuple[float, float]
    mass_composition_corr: dict[str, float]

    run_dates: int
    replicates_share_run_date: bool

    warnings: tuple[str, ...] = field(default=())


def _mixture_test(design: pd.DataFrame) -> MixtureTest:
    comps = ["api_wt", "hpmc_wt", "lactose_wt"]
    matrix = design[comps].to_numpy(dtype=float)
    sums = matrix.sum(axis=1)
    max_dev = float(np.max(np.abs(sums - sums.mean())))

    centered = matrix - matrix.mean(axis=0)
    svals = np.linalg.svd(centered, compute_uv=False)
    rank = int(np.linalg.matrix_rank(centered))
    ratio = float(svals[-1] / svals[0]) if svals[0] > 0 else 0.0

    # Constant to machine precision, or to the precision the file actually reports.
    is_mixture = bool(max_dev <= 1e-9 or (max_dev <= 0.05 and rank < 3))
    fixed = tuple(c for c in comps if float(np.ptp(design[c].to_numpy(dtype=float))) == 0.0)

    return MixtureTest(
        sums=tuple(float(s) for s in np.unique(np.round(sums, 9))),
        max_abs_deviation=max_dev,
        singular_values=tuple(float(v) for v in svals),
        rank=rank,
        condition_ratio=ratio,
        is_mixture=is_mixture,
        constant_total=float(sums[0]) if is_mixture else None,
        fixed_components=fixed,
        n_independent_dims=rank,
    )


def build_quality_report(db: Database) -> QualityReport:
    """Characterise ``db`` without modifying or repairing it."""
    prof = db.profiles
    warnings: list[str] = []

    apis = tuple(sorted(prof["api"].astype(str).unique()))
    cases_per_api = {
        api: int(prof.loc[prof["api"] == api, "case"].nunique()) for api in apis
    }
    grades_per_api = {
        api: tuple(sorted(prof.loc[prof["api"] == api, "grade"].astype(str).unique()))
        for api in apis
    }
    replicate_counts = {
        int(r): int(n) for r, n in prof.groupby("replicate")["id"].nunique().items()
    }

    # --- timepoints (A7) --------------------------------------------------
    vectors = prof.groupby(["id", "replicate"])["time_h"].apply(
        lambda s: tuple(sorted(round(float(v), 9) for v in s.unique()))
    )
    distinct_vectors = int(vectors.nunique())
    times = tuple(vectors.iloc[0])
    diffs = np.diff(np.asarray(times)) if len(times) > 1 else np.array([0.0])
    uniform = bool(np.allclose(diffs, diffs[0])) if len(diffs) else True
    within_2h = int(sum(1 for t in times if 0 < t <= 2.0))
    if distinct_vectors > 1:
        warnings.append(
            f"{distinct_vectors} distinct time vectors across profiles: metric extraction "
            "must interpolate to a common grid before comparison (A7 violated)."
        )

    # --- design completeness ---------------------------------------------
    observed = set(
        map(tuple, prof[["api", "case", "grade"]].drop_duplicates().astype(object).to_numpy())
    )
    all_cases = sorted(int(c) for c in prof["case"].unique())
    all_grades = sorted(str(g) for g in prof["grade"].unique())
    missing: list[tuple[str, int, str]] = []
    for api in apis:
        for case in all_cases:
            for grade in all_grades:
                if (api, case, grade) not in observed:
                    missing.append((api, case, grade))

    expected_cells = len(apis) * config.EXPECTED_CASES * config.EXPECTED_GRADES
    observed_cells = len(observed)

    spec_cases = (
        set(int(c) for c in db.design_spec["case"].unique())
        if db.design_spec is not None
        else set()
    )
    missing_cases = tuple(sorted(spec_cases - set(all_cases))) if spec_cases else ()
    if spec_cases and missing_cases:
        warnings.append(
            f"Case(s) {list(missing_cases)} appear in the intended design specification "
            "but have no measured profiles."
        )
    if not spec_cases and len(all_cases) < config.EXPECTED_CASES:
        warnings.append(
            f"{len(all_cases)} of {config.EXPECTED_CASES} intended cases present, and the "
            "workbook carries no design specification, so the missing cases cannot be "
            "identified by composition -- only counted."
        )

    # --- A1 ---------------------------------------------------------------
    design_pts = (
        prof[["case", "api_wt", "hpmc_wt", "lactose_wt"]]
        .drop_duplicates("case")
        .sort_values("case")
        .reset_index(drop=True)
    )
    mixture = _mixture_test(design_pts)

    # --- profile-level anomalies -----------------------------------------
    grouped = prof.sort_values("time_h").groupby(["id", "replicate"], sort=False)

    steps = grouped["pct_released"].apply(lambda s: np.diff(s.to_numpy(dtype=float)))
    violations = int(sum(int((d < 0).sum()) for d in steps))
    profiles_with_violation = int(sum(1 for d in steps if (d < 0).any()))
    largest_back = float(min((d.min() if len(d) else 0.0) for d in steps))

    above_100 = int((prof["pct_released"] > 100.0).sum())
    max_pct = float(prof["pct_released"].max())
    negatives = int((prof["pct_released"] < 0).sum())
    if above_100:
        warnings.append(
            f"{above_100} observation(s) exceed 100% released (max {max_pct:.2f}%). These "
            "are assay noise near plateau and are preserved, not clipped (G2/G8). The "
            "monotonic-and-bounded requirement constrains PREDICTED profiles only."
        )
    if violations:
        warnings.append(
            f"{violations} backward step(s) across {profiles_with_violation} profile(s); "
            f"largest {largest_back:.2f} percentage points. Expected from assay noise; "
            "reported, not smoothed."
        )

    # --- G5 censoring ------------------------------------------------------
    peak = grouped["pct_released"].max().rename("peak").reset_index()
    peak["censored"] = peak["peak"] < config.CENSORING_PCT
    censored_profiles = int(peak["censored"].sum())
    by_id = peak.groupby("id")["censored"].agg(["sum", "count"])
    fully = tuple(sorted(by_id.index[by_id["sum"] == by_id["count"]].astype(str)))
    partial = tuple(
        sorted(by_id.index[(by_id["sum"] > 0) & (by_id["sum"] < by_id["count"])].astype(str))
    )
    if partial:
        warnings.append(
            f"{len(partial)} formulation(s) are PARTIALLY censored -- some replicates reach "
            f"{config.CENSORING_PCT:g}% and others do not: {list(partial)}. These sit on the "
            "censoring boundary and must not be collapsed to a single flag (G5)."
        )

    # --- G4 Peppas feasibility --------------------------------------------
    def usable(series: pd.Series) -> int:
        vals = series.to_numpy(dtype=float)
        return int(((vals > 0) & (vals <= config.PEPPAS_MAX_PCT)).sum())

    counts = grouped["pct_released"].apply(usable)
    peppas_counts = {f"{i}|rep{r}": int(v) for (i, r), v in counts.items()}
    min_points = int(counts.min()) if len(counts) else 0
    failing = tuple(
        sorted(f"{i}|rep{r}" for (i, r), v in counts.items() if v < config.PEPPAS_MIN_POINTS)
    )
    if failing:
        warnings.append(
            f"{len(failing)} profile(s) have fewer than {config.PEPPAS_MIN_POINTS} points at "
            f"or below {config.PEPPAS_MAX_PCT:g}% release. Peppas k and n are NOT reported "
            "for these (G4); this is a limitation of the sampling schedule, not of the fit."
        )

    # --- mass ---------------------------------------------------------------
    per_id = prof.drop_duplicates(["id", "replicate"])
    mass_range = (
        float(per_id["tablet_mass_mg"].min()),
        float(per_id["tablet_mass_mg"].max()),
    )
    id_mean = per_id.groupby("id").agg(
        mass=("tablet_mass_mg", "mean"),
        api_wt=("api_wt", "first"),
        hpmc_wt=("hpmc_wt", "first"),
        lactose_wt=("lactose_wt", "first"),
    )
    mass_corr = {
        comp: float(np.corrcoef(id_mean["mass"], id_mean[comp])[0, 1])
        for comp in ("api_wt", "hpmc_wt", "lactose_wt")
    }

    dates = prof["run_date"].dropna()
    n_dates = int(dates.nunique()) if len(dates) else 0
    same_day = True
    if len(dates):
        per_row = prof.dropna(subset=["run_date"]).groupby(["id", "time_h"])["run_date"].nunique()
        same_day = bool((per_row <= 1).all())

    if db.is_synthetic:
        warnings.append(
            "DATABASE IS SYNTHETIC PLACEHOLDER MATERIAL. No value represents an "
            "experimental measurement. Every figure, dashboard panel and guideline "
            "derived from it must carry the provenance banner."
        )
    if db.viscosity_source != "workbook":
        warnings.append(
            "Grade -> nominal viscosity came from the config fallback, not the workbook. "
            "log10(viscosity) is a model factor (AC4); confirm the values against the "
            "supplier specification for the grades actually used."
        )

    return QualityReport(
        source_name=db.source_path.name,
        is_synthetic=db.is_synthetic,
        provenance_notes=db.provenance_notes,
        vessel_volume_ml=db.vessel_volume_ml,
        viscosity_source=db.viscosity_source,
        grade_viscosity_cp=dict(db.grade_viscosity_cp),
        apis=apis,
        n_ids=int(prof["id"].nunique()),
        n_replicates=int(prof["replicate"].nunique()),
        cases_per_api=cases_per_api,
        grades_per_api=grades_per_api,
        replicate_counts=replicate_counts,
        n_timepoints=len(times),
        timepoints_h=times,
        distinct_time_vectors=distinct_vectors,
        uniform_spacing=uniform,
        samples_within_2h=within_2h,
        missing_cells=tuple(missing),
        expected_cells=expected_cells,
        observed_cells=observed_cells,
        cases_missing_entirely=missing_cases,
        mixture=mixture,
        monotonicity_violations=violations,
        profiles_with_violation=profiles_with_violation,
        largest_backward_step_pct=largest_back,
        values_above_100=above_100,
        max_pct_released=max_pct,
        negative_values=negatives,
        censored_profiles=censored_profiles,
        fully_censored_ids=fully,
        partially_censored_ids=partial,
        peppas_point_counts=peppas_counts,
        peppas_min_points=min_points,
        peppas_failing_profiles=failing,
        mass_range_mg=mass_range,
        mass_composition_corr=mass_corr,
        run_dates=n_dates,
        replicates_share_run_date=same_day,
        warnings=tuple(warnings),
    )


def render_markdown(report: QualityReport) -> str:
    """Render the quality report as Markdown for ``outputs/reports``."""
    r = report
    out: list[str] = []
    add = out.append

    add("# Data quality report (AC1)\n")
    add(f"Source: `{r.source_name}`\n")

    if r.is_synthetic:
        add(
            "> **PLACEHOLDER DATA — SYNTHETIC, NOT EXPERIMENTAL.**\n"
            "> No value below represents a measurement. Nothing derived from this "
            "database is an experimental finding.\n"
        )

    add("## Contents\n")
    add(f"- APIs present: {len(r.apis)} — {', '.join(r.apis)}")
    add(f"- Formulation IDs: {r.n_ids}")
    add(f"- Replicates per formulation: {r.n_replicates}")
    for api in r.apis:
        add(
            f"- `{api}`: {r.cases_per_api[api]} cases x "
            f"{len(r.grades_per_api[api])} grades ({', '.join(r.grades_per_api[api])})"
        )
    add(f"- Vessel volume used for dose normalisation: {r.vessel_volume_ml:g} mL")
    add(
        f"- Grade → nominal viscosity ({r.viscosity_source}): "
        + ", ".join(f"{g} = {v:g} cP" for g, v in sorted(r.grade_viscosity_cp.items()))
    )

    add("\n## Design completeness\n")
    add(f"- Observed design cells: {r.observed_cells} / {r.expected_cells} intended")
    if r.missing_cells:
        add(f"- **Missing cells ({len(r.missing_cells)}):**")
        for api, case, grade in r.missing_cells:
            add(f"  - `{api}` case {case} grade {grade}")
    else:
        add("- No missing cells: the crossing is complete.")
    if r.cases_missing_entirely:
        add(f"- Cases in the design spec with no data: {list(r.cases_missing_entirely)}")

    add("\n## A1 — mixture constraint\n")
    m = r.mixture
    add(f"- Component totals observed: {list(m.sums)}")
    add(f"- Max deviation from constant total: `{m.max_abs_deviation:.3e}`")
    add(f"- Composition matrix singular values: {[round(v, 6) for v in m.singular_values]}")
    add(f"- Rank: **{m.rank}** (σ_min/σ_max = `{m.condition_ratio:.3e}`)")
    add(f"\n**Verdict.** {m.verdict}\n")

    add("\n## Timepoint coverage (A7)\n")
    add(f"- Distinct time vectors across profiles: {r.distinct_time_vectors}")
    add(f"- Timepoints per profile: {r.n_timepoints}")
    add(f"- Spacing: {'uniform' if r.uniform_spacing else 'NON-uniform'}")
    add(f"- Samples inside the first 2 h: {r.samples_within_2h}")
    add(f"- Grid (h): {[round(t, 4) for t in r.timepoints_h]}")

    add("\n## Profile anomalies\n")
    add(
        f"- Monotonicity: {r.monotonicity_violations} backward step(s) across "
        f"{r.profiles_with_violation} profile(s); largest "
        f"{r.largest_backward_step_pct:.2f} pp"
    )
    add(f"- Observations above 100% released: {r.values_above_100} (max {r.max_pct_released:.2f}%)")
    add(f"- Negative observations: {r.negative_values}")

    add("\n## Censoring (G5)\n")
    add(f"- Replicate profiles never reaching {config.CENSORING_PCT:g}%: {r.censored_profiles}")
    add(f"- Formulations fully censored: {len(r.fully_censored_ids)}")
    for i in r.fully_censored_ids:
        add(f"  - `{i}`")
    add(f"- Formulations **partially** censored: {len(r.partially_censored_ids)}")
    for i in r.partially_censored_ids:
        add(f"  - `{i}` — replicates straddle the threshold")

    add("\n## Peppas feasibility (G4)\n")
    add(
        f"- Minimum usable points (0 < %released ≤ {config.PEPPAS_MAX_PCT:g}) across all "
        f"profiles: {r.peppas_min_points}"
    )
    add(f"- Profiles below the {config.PEPPAS_MIN_POINTS}-point threshold: "
        f"{len(r.peppas_failing_profiles)}")
    for i in r.peppas_failing_profiles:
        add(f"  - `{i}` — no Peppas parameters reported")

    add("\n## Tablet mass\n")
    add(f"- Range: {r.mass_range_mg[0]:.1f}–{r.mass_range_mg[1]:.1f} mg")
    add(
        "- Correlation with composition: "
        + ", ".join(f"{k} = {v:+.3f}" for k, v in r.mass_composition_corr.items())
    )
    add(f"- Distinct run dates: {r.run_dates}; replicates share a run date: "
        f"{r.replicates_share_run_date}")

    if r.warnings:
        add("\n## Warnings\n")
        for w in r.warnings:
            add(f"- {w}")

    return "\n".join(out) + "\n"
