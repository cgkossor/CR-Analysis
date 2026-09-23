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

#: Lower bound for the Weibull asymptote ceiling. The bound exists to stop a
#: heavily censored profile -- one that never approaches its own plateau -- from
#: letting the optimiser run F_inf off to non-physical values and drag Td with
#: it. It is NOT a statement that release cannot exceed this number.
#:
#: The ceiling actually applied is data-aware: max(this, observed peak x margin).
#: A flat 105 would clamp a profile that genuinely reached 110%, biasing both
#: F_inf and Td downward on exactly the formulations that released most fully.
MAX_PHYSICAL_RELEASE_PCT: Final[float] = 105.0

#: How far above a profile's own observed peak its asymptote may be fitted.
#: Applied when the peak exceeds MAX_PHYSICAL_RELEASE_PCT, so the fit follows
#: the data instead of a constant.
ASYMPTOTE_CEILING_MARGIN: Final[float] = 1.05

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

# --- Timepoint reconciliation -----------------------------------------------
# Real sampling does not happen at exact times: a nominal 60-minute pull lands
# at 59.5 or 61 depending on the operator. Two timestamps within this distance
# are treated as the same nominal sample when building the comparison grid.
#
# The tolerance is absolute near t=0 and proportional afterwards, because a
# one-minute slip at the 2-minute pull is a genuinely different timepoint while
# at the 24-hour pull it is nothing.
TIME_CLUSTER_ABS_H: Final[float] = 0.5 / 60.0
TIME_CLUSTER_REL: Final[float] = 0.01

# --- Analysis window ----------------------------------------------------------
# Every metric, fit and comparison uses 0 to this many hours. Runs extended past
# it for diagnostic reasons are trimmed at load. The first reading after the
# window is kept, so the value AT the window end is interpolated rather than
# lost to a reading that landed at 24.02 h.
ANALYSIS_WINDOW_H: Final[float] = 24.0

# --- Nominal schedule for densely sampled data ------------------------------
# Manual pulls cluster into a nominal schedule (above). An in-situ probe logging
# every few minutes does not: its thousands of distinct timestamps recover a
# grid of hundreds of points, different for every method revision. Once the
# recovered grid would exceed MAX_GRID_POINTS, profiles are instead resampled
# onto this fixed schedule. Resampling is used only for profile COMPARISONS
# (mean curves, f2, cross-validation, stress test); per-profile metrics and fits
# still use every measured point.
MAX_GRID_POINTS: Final[int] = 60
NOMINAL_SCHEDULE_H: Final[tuple[float, ...]] = (
    0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0,
    10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0,
)
#: A resampled point is left blank when the readings either side of it are
#: further apart than this, so a hole in the record is never bridged by a line.
MAX_INTERP_GAP_H: Final[float] = 1.0

# =============================================================================
# PLOTTING -- AXIS LIMITS
# =============================================================================
# >>> EDIT AXIS LIMITS HERE. <<<
#
# These four constants set the axes on EVERY plot: the matplotlib figures in
# outputs/figures/ and the interactive charts in the dashboard both read them,
# so the two cannot drift apart. Change a value, re-run the pipeline, and every
# figure and panel follows.
#
# They are fixed rather than derived from the data on purpose. Auto-scaled axes
# silently rescale between runs, so two profiles that look equally steep are not
# actually comparable -- and a formulation whose release collapsed would look
# normal because the axis shrank to fit it.

#: Time axis, hours. The upper limit sits past the 24 h endpoint so the final
#: measurement is not pinned against the frame edge.
PLOT_MIN_TIME_H: Final[float] = 0.0
PLOT_MAX_TIME_H: Final[float] = 25.0

#: Release axis, % of dose. The upper limit is deliberately above 100: real
#: assays overshoot near plateau, values of 105-110% are ordinary, and clipping
#: the axis at 100 would hide them rather than showing what was measured.
PLOT_MIN_RELEASE_PCT: Final[float] = 0.0
PLOT_MAX_RELEASE_PCT: Final[float] = 115.0

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
