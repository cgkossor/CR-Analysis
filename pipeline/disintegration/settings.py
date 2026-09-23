"""Tunable constants for the disintegration section.

Kept apart from ``pipeline.config`` so this section can evolve without touching
a file every other part of the pipeline depends on.
"""

from __future__ import annotations

from typing import Final

from pipeline import config

#: Sheet names looked for, case-insensitively, exact match first.
SHEET_NAMES: Final[tuple[str, ...]] = ("disintegration", "disintegration time", "dt")
NOTES_SHEET_NAMES: Final[tuple[str, ...]] = ("disintegration_notes", "disintegration notes")

#: Fewer replicates than this and a formulation's precision cannot be judged.
DT_MIN_REPS: Final[int] = 3

#: Replicate CV above this is flagged. Disintegration of a swollen gel is a
#: noisier end point than an assay, so this is looser than a dissolution limit.
DT_CV_WARN: Final[float] = 0.15

#: Test duration assumed when the sheet carries no ``Test_end`` column. A
#: replicate at or beyond it is right-censored: the tablet had not gone.
DT_DEFAULT_TEST_END_H: Final[float] = 24.0

#: Composition columns must sum to 100 within this (wt%).
COMP_SUM_TOL: Final[float] = 0.5

#: Dixon's Q is the outlier test that suits three or four replicates.
DIXON_ALPHA: Final[float] = 0.05

#: Formulation-level bootstrap resamples for correlation intervals.
BOOTSTRAP_N: Final[int] = 2000

#: Correlation at or above this is what "DT tracks dissolution" means here.
RHO_EXPECTED: Final[float] = 0.8

#: Fewer matched design points than this and correlations are anecdotes.
MIN_POINTS_CORRELATION: Final[int] = 8

SEED: Final[int] = config.PIPELINE_SEED

#: Seed of the synthetic generator (separate from the analysis seed).
SYNTHETIC_SEED: Final[int] = 20_260_922

#: SchemaError codes for this sheet. Offset by 100 so they never collide with
#: the dissolution codes in ``pipeline.io.schema.SCHEMA_ERROR_CODES``.
DT_ERROR_CODES: Final[dict[int, str]] = {
    101: "disintegration ID column missing or ambiguous",
    102: "disintegration case column missing or ambiguous",
    103: "disintegration grade column missing or ambiguous",
    104: "no disintegration replicate columns",
    105: "disintegration time unit not determinable",
    106: "disintegration time units inconsistent",
    107: "duplicate disintegration IDs",
}
