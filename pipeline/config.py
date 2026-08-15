"""Pipeline configuration.

Everything here is a *method* or *design-intent* constant, not data. Nothing in
this module may encode the contents of a particular database file: compositions,
grades, timepoints, formulation IDs and replicate counts are all discovered at
ingest. Swapping the placeholder for the real database must require no edit here
beyond the method constants a formulator would genuinely re-specify.
"""

from __future__ import annotations

from typing import Final

# --- Dissolution method -----------------------------------------------------
# Confirmed by the user: USP Apparatus 2, 900 mL, in-situ measurement with no
# withdrawal (fibre-optic / flow-through probe). The vessel volume is required
# to convert the assay concentration column into % released and is NOT carried
# in the raw file, so it must live here. See docs: no cumulative-sampling
# correction is applied, and that is a deliberate consequence of in-situ
# measurement rather than an omission.
VESSEL_VOLUME_ML: Final[float] = 900.0

#: Sampling is in-situ; if a future study withdraws and replaces aliquots this
#: must become True and the cumulative correction applied in `profiles`.
SAMPLING_WITHDRAWS_ALIQUOTS: Final[bool] = False

# --- Design intent ----------------------------------------------------------
# Used ONLY to report shortfall against the intended design in the AC1 quality
# report. Never used to filter, pad, index or reindex observed data.
EXPECTED_CASES: Final[int] = 11
EXPECTED_GRADES: Final[int] = 3

# --- Grade -> nominal viscosity fallback ------------------------------------
# Preferred source is the workbook's own `Design` sheet. This mapping is a
# fallback for files that omit it, and is reported as such in the quality
# report so the provenance of log10(viscosity) is always visible.
FALLBACK_GRADE_VISCOSITY_CP: Final[dict[str, float]] = {
    "K100LV": 100.0,
    "K4M": 4_000.0,
    "K100M": 100_000.0,
}

# --- Guardrail thresholds ---------------------------------------------------
#: G4 - Peppas is fit on the <=60% released portion only, and needs >=3 points.
PEPPAS_MAX_PCT: Final[float] = 60.0
PEPPAS_MIN_POINTS: Final[int] = 3

#: G5 - a profile that never reaches this level has a censored t80.
CENSORING_PCT: Final[float] = 80.0

#: Physical ceiling on the Weibull asymptote. A formulation cannot release more
#: than the dose it contains; the headroom above 100 absorbs assay overshoot near
#: plateau. Without this bound, a heavily censored profile -- one that never
#: approaches its own asymptote -- lets the optimiser run F_inf off to
#: non-physical values and drag Td with it.
MAX_PHYSICAL_RELEASE_PCT: Final[float] = 105.0

#: A Weibull asymptote counts as identified only when the profile actually climbs
#: to this fraction of the fitted F_inf. Below it, F_inf and Td are extrapolated
#: rather than estimated, and every downstream consumer is told so.
ASYMPTOTE_IDENTIFIED_FRACTION: Final[float] = 0.80

#: A profile at or above this level has essentially finished releasing, so its
#: observed-window MDT is the true MDT rather than a lower bound. At >=95% at
#: most a twentieth of the dose remains, which cannot move MDT materially.
COMPLETE_RELEASE_PCT: Final[float] = 95.0

#: AC7 - f2 similarity threshold.
F2_SIMILAR_THRESHOLD: Final[float] = 50.0

#: f2 point selection (USP convention): at most one point beyond this level, so
#: the statistic is not dominated by the plateau. Critical here because the
#: sampling schedule is front-loaded (16 of 25 samples inside the first 2 h).
F2_PLATEAU_PCT: Final[float] = 85.0
F2_MIN_POINTS: Final[int] = 3

#: AC4 - adj-vs-predicted R^2 gap beyond this is flagged.
R2_GAP_FLAG: Final[float] = 0.20
#: AC4 - adequate precision below this is flagged.
ADEQUATE_PRECISION_FLAG: Final[float] = 4.0

# --- Determinism (G10) ------------------------------------------------------
#: Seed for every resample, CV split and optimiser start in this pipeline.
#: Distinct from any seed used to generate a placeholder database.
PIPELINE_SEED: Final[int] = 8_314_159

# --- Provenance -------------------------------------------------------------
#: Substrings that, found in a workbook's notes, mark the database as synthetic
#: placeholder material. Drives the dashboard/figure provenance banner. When the
#: real database arrives with no such marker the banner disappears on its own -
#: removing it must never require a code change.
SYNTHETIC_MARKERS: Final[tuple[str, ...]] = (
    "PLACEHOLDER",
    "SYNTHETIC",
    "NOT EXPERIMENTAL",
)
