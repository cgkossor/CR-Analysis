"""Schema detection and validation for the dissolution workbook.

The raw layout is *hybrid*, not long: one row per (ID, timepoint), with the
replicates carried as parallel column families -- ``conc_1..N``, ``Mass_1..N_mg``,
``Date_1..N`` -- and a single shared elapsed-time column.

Columns are detected by pattern rather than by fixed position or exact literal
name, and the replicate count is discovered from the file. Units are parsed out
of the headers: a unit that cannot be established is a loud failure, never a
guess, because a microgram/milligram mix-up silently rescales every profile by
1000x (AC1: "malformed input fails loudly with a specific message").
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final

import pandas as pd


class SchemaError(ValueError):
    """Raised when the workbook does not match the expected structure.

    Carries a specific, actionable message naming the offending column or
    expectation. Never raised for merely surprising *data* -- only for a shape
    the pipeline cannot interpret.

    ``code`` identifies the rule that failed without quoting the message, which
    names columns and IDs. It is what ``pipeline.audit`` reports; the table is
    :data:`SCHEMA_ERROR_CODES`.
    """

    def __init__(self, message: str, code: int = 0) -> None:
        super().__init__(message)
        self.code = code


#: Every rule that can reject a workbook, by the integer ``SchemaError.code``.
SCHEMA_ERROR_CODES: Final[dict[int, str]] = {
    0: "unclassified",
    1: "input file not found",
    2: "vessel volume not positive",
    3: "no dissolution sheet",
    11: "ID column missing",
    12: "ID column ambiguous",
    13: "case column missing",
    14: "case column ambiguous",
    15: "API name column missing",
    16: "API name column ambiguous",
    17: "HPMC grade column missing",
    18: "HPMC grade column ambiguous",
    20: "composition column(s) missing",
    21: "no elapsed-time column",
    22: "time unit not determinable",
    23: "no concentration columns",
    24: "concentration unit not determinable",
    25: "concentration units inconsistent",
    26: "mass unit not determinable",
    27: "mass units inconsistent",
    28: "replicate has concentration but no mass column",
    30: "no tablet-mass columns",
    31: "non-positive dose",
    32: "grade without nominal viscosity",
}

_ROLE_CODES: Final[dict[str, int]] = {"ID": 11, "case": 13, "API name": 15, "HPMC grade": 17}


# --- unit vocabularies ------------------------------------------------------
# Values are the factor converting the stated unit to the canonical unit.
_CONC_UNITS: Final[dict[str, float]] = {
    "ug_ml": 1.0,
    "ug/ml": 1.0,
    "mcg_ml": 1.0,
    "mcg/ml": 1.0,
    "ugml": 1.0,
    "mg_ml": 1000.0,
    "mg/ml": 1000.0,
}
_MASS_UNITS: Final[dict[str, float]] = {"mg": 1.0, "g": 1000.0}
_TIME_UNITS: Final[dict[str, float]] = {
    "min": 1.0 / 60.0,
    "minute": 1.0 / 60.0,
    "minutes": 1.0 / 60.0,
    "h": 1.0,
    "hr": 1.0,
    "hour": 1.0,
    "hours": 1.0,
}

_CONC_RE = re.compile(r"^conc(?:entration)?[_\s]*(\d+)\b", re.IGNORECASE)
_MASS_RE = re.compile(r"^mass[_\s]*(\d+)", re.IGNORECASE)
_DATE_RE = re.compile(r"^date[_\s]*(\d+)", re.IGNORECASE)
# Longest alternatives first, and "not followed by a letter" rather than \b, so
# that headers like "Min_1" (underscore is a word character) still match.
_TIME_RE = re.compile(
    r"^(minutes|minute|min|hours|hour|elapsed|time|hr|h)(?![a-z])", re.IGNORECASE
)
_COMP_RE = re.compile(r"^(api|hpmc|lactose)\s*\[?\s*(wt\s*%|%|w/w)", re.IGNORECASE)


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def _extract_unit(header: str, vocab: dict[str, float]) -> float | None:
    """Return the conversion factor for the first recognised unit in ``header``."""
    lowered = _norm(header).lower()
    # Prefer a bracketed unit, which is unambiguous.
    bracketed = re.findall(r"[\[(]([^\])]*)[\])]", lowered)
    for token in [*bracketed, lowered]:
        cleaned = token.replace(" ", "").replace("-", "_")
        for unit, factor in vocab.items():
            if re.search(rf"(?<![a-z0-9]){re.escape(unit)}(?![a-z0-9])", cleaned):
                return factor
    return None


@dataclass(frozen=True)
class ReplicateColumns:
    """The column family belonging to one replicate index."""

    index: int
    conc: str
    mass: str | None
    date: str | None


@dataclass(frozen=True)
class DissolutionSchema:
    """Resolved column layout and unit conversions for a dissolution sheet."""

    id_col: str
    case_col: str
    api_col: str
    grade_col: str
    component_cols: dict[str, str]
    time_col: str
    time_to_hours: float
    conc_to_ug_per_ml: float
    mass_to_mg: float
    replicates: tuple[ReplicateColumns, ...]

    @property
    def n_replicates(self) -> int:
        return len(self.replicates)


def _require(columns: list[str], pattern: str, label: str) -> str:
    rx = re.compile(pattern, re.IGNORECASE)
    hits = [c for c in columns if rx.search(_norm(c))]
    if not hits:
        raise SchemaError(
            f"Required {label} column not found. Expected a header matching "
            f"/{pattern}/i. Columns present: {columns}",
            _ROLE_CODES.get(label, 0),
        )
    if len(hits) > 1:
        raise SchemaError(
            f"Ambiguous {label} column: {hits} all match /{pattern}/i. "
            "Rename so exactly one column identifies it.",
            _ROLE_CODES.get(label, 0) + 1,
        )
    return hits[0]


def detect_schema(frame: pd.DataFrame) -> DissolutionSchema:
    """Resolve the column layout of a dissolution sheet, or fail loudly."""
    cols = [str(c) for c in frame.columns]

    id_col = _require(cols, r"^id$", "ID")
    case_col = _require(cols, r"^case$", "case")
    api_col = _require(cols, r"^api$", "API name")
    grade_col = _require(cols, r"^(hpmc\s*)?grade$", "HPMC grade")

    component_cols: dict[str, str] = {}
    for col in cols:
        match = _COMP_RE.match(_norm(col))
        if match:
            component_cols[match.group(1).lower()] = col
    missing = {"api", "hpmc", "lactose"} - component_cols.keys()
    if missing:
        raise SchemaError(
            f"Missing composition column(s) for {sorted(missing)}. Expected headers like "
            f"'API [wt%]'. Columns present: {cols}",
            20,
        )

    # --- time -------------------------------------------------------------
    time_candidates = [c for c in cols if _TIME_RE.match(_norm(c))]
    if not time_candidates:
        raise SchemaError(
            f"No elapsed-time column found (expected a header starting with "
            f"min/time/hour). Columns present: {cols}",
            21,
        )
    time_col = time_candidates[0]
    time_factor = _extract_unit(time_col, _TIME_UNITS)
    if time_factor is None:
        raise SchemaError(
            f"Cannot determine the unit of the time column {time_col!r}. Rename it to "
            "carry an explicit unit, e.g. 'Time_min' or 'Time [h]'. Elapsed time is "
            "never assumed.",
            22,
        )

    # --- replicate families ----------------------------------------------
    conc_by_idx: dict[int, str] = {}
    for col in cols:
        match = _CONC_RE.match(_norm(col))
        if match:
            conc_by_idx[int(match.group(1))] = col
    if not conc_by_idx:
        raise SchemaError(
            f"No concentration columns found (expected 'conc_1', 'conc_2', ...). "
            f"Columns present: {cols}",
            23,
        )

    mass_by_idx = {
        int(m.group(1)): c for c in cols if (m := _MASS_RE.match(_norm(c))) is not None
    }
    date_by_idx = {
        int(m.group(1)): c for c in cols if (m := _DATE_RE.match(_norm(c))) is not None
    }

    conc_factors = {c: _extract_unit(c, _CONC_UNITS) for c in conc_by_idx.values()}
    unresolved = [c for c, f in conc_factors.items() if f is None]
    if unresolved:
        raise SchemaError(
            f"Cannot determine the concentration unit for {unresolved}. Expected the "
            "unit in the header, e.g. 'conc_1 [ug_ml]'. Guessing is not permitted: a "
            "ug/mL vs mg/mL error rescales every release profile by 1000x.",
            24,
        )
    distinct_conc = set(conc_factors.values())
    if len(distinct_conc) > 1:
        raise SchemaError(
            f"Concentration columns carry inconsistent units: {conc_factors}. "
            "All replicates must report in the same unit.",
            25,
        )
    conc_factor = distinct_conc.pop()
    assert conc_factor is not None  # narrowed by the `unresolved` check above

    mass_factor = 1.0
    if mass_by_idx:
        mass_factors = {c: _extract_unit(c, _MASS_UNITS) for c in mass_by_idx.values()}
        unresolved_mass = [c for c, f in mass_factors.items() if f is None]
        if unresolved_mass:
            raise SchemaError(
                f"Cannot determine the tablet-mass unit for {unresolved_mass}. Expected "
                "e.g. 'Mass_1_mg'. Mass sets the dose denominator, so it is never assumed.",
                26,
            )
        distinct_mass = set(mass_factors.values())
        if len(distinct_mass) > 1:
            raise SchemaError(f"Tablet-mass columns carry inconsistent units: {mass_factors}.", 27)
        resolved_mass = distinct_mass.pop()
        assert resolved_mass is not None
        mass_factor = resolved_mass

    replicates = tuple(
        ReplicateColumns(
            index=idx,
            conc=conc_by_idx[idx],
            mass=mass_by_idx.get(idx),
            date=date_by_idx.get(idx),
        )
        for idx in sorted(conc_by_idx)
    )

    without_mass = [r.index for r in replicates if r.mass is None]
    if without_mass and mass_by_idx:
        raise SchemaError(
            f"Replicate(s) {without_mass} have a concentration column but no tablet-mass "
            "column, while other replicates do. Dose cannot be computed for them. Supply "
            "the missing mass column or remove the partial replicate.",
            28,
        )

    return DissolutionSchema(
        id_col=id_col,
        case_col=case_col,
        api_col=api_col,
        grade_col=grade_col,
        component_cols=component_cols,
        time_col=time_col,
        time_to_hours=time_factor,
        conc_to_ug_per_ml=conc_factor,
        mass_to_mg=mass_factor,
        replicates=replicates,
    )
