"""Read the optional ``Disintegration`` sheet into a tidy long frame.

Layout (one row per formulation, replicates as columns, like the dissolution
sheet)::

    ID | Case | API | HPMC Grade | API [wt%] | HPMC [wt%] | Lactose [wt%] |
    DT_1 [min] | DT_2 [min] | DT_3 [min] | DT_4 [min] | Test_end [min]

* The unit is read from each replicate header (``s``, ``min`` or ``h``) and never
  assumed. Everything is converted to hours.
* A blank replicate cell means that replicate was not run, so three replicates
  are as valid as four.
* A cell written ``>1440``, or a value at or past the test end, is
  right-censored: the tablet had not finished disintegrating when the test
  stopped. Censored values are kept and flagged, never dropped and never read
  as the time itself.

A workbook without the sheet returns ``None``. The rest of the pipeline then
skips this section.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline import config
from pipeline.disintegration import settings
from pipeline.io.schema import SchemaError

#: Unit token -> factor to hours.
_UNITS: dict[str, float] = {
    "s": 1 / 3600, "sec": 1 / 3600, "secs": 1 / 3600, "second": 1 / 3600, "seconds": 1 / 3600,
    "min": 1 / 60, "mins": 1 / 60, "minute": 1 / 60, "minutes": 1 / 60,
    "h": 1.0, "hr": 1.0, "hrs": 1.0, "hour": 1.0, "hours": 1.0,
}
#: Integer code for the unit, as the audit reports it.
UNIT_CODES: dict[float, int] = {1 / 3600: 1, 1 / 60: 2, 1.0: 3}

_DT_RE = re.compile(r"^(?:dt|disintegration(?:[_\s]*time)?)[_\s]*(\d+)(?!\d)", re.IGNORECASE)
_END_RE = re.compile(r"^test[_\s]*end", re.IGNORECASE)
_COMP_RE = re.compile(r"^(api|hpmc|lactose)\s*\[?\s*(wt\s*%|%|w/w)", re.IGNORECASE)
_ROLES: dict[str, tuple[str, int]] = {
    "id": (r"^id$", 101),
    "case": (r"^case$", 102),
    "grade": (r"^(hpmc\s*)?grade$", 103),
}

#: Canonical long-format columns.
LONG_COLUMNS = (
    "id", "api", "case", "grade", "api_wt", "hpmc_wt", "lactose_wt",
    "replicate", "dt_h", "censored",
)


@dataclass(frozen=True)
class DisintegrationData:
    """The loaded sheet, plus what the diagnostics and audit need to judge it."""

    long: pd.DataFrame
    """One row per (id, replicate) that was run. See :data:`LONG_COLUMNS`."""

    unit_to_hours: float
    test_end_h: float
    test_end_from_sheet: bool
    is_synthetic: bool
    truth: dict[str, float] = field(default_factory=dict)
    """Ground-truth parameters from a synthetic generator's notes sheet."""

    # --- shape facts, all counts, for the diagnostics and the audit ---------
    n_rows: int = 0
    n_columns: int = 0
    n_replicate_columns: int = 0
    n_unrecognised_columns: int = 0
    n_composition_columns: int = 0
    n_nonnumeric: int = 0
    """Cells that were neither blank, a number, nor a ``>`` censoring mark."""
    n_nonpositive: int = 0
    has_api_column: bool = False

    @property
    def unit_code(self) -> int:
        return UNIT_CODES.get(self.unit_to_hours, 0)


def _norm(text: object) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def find_sheet(sheets: list[str], candidates: tuple[str, ...]) -> str | None:
    lowered = {s: s.strip().lower() for s in sheets}
    for c in candidates:
        for s, low in lowered.items():
            if low == c:
                return s
    return None


def _unit(header: str) -> float | None:
    low = _norm(header).lower()
    for token in [*re.findall(r"[\[(]([^\])]*)[\])]", low), low.split("_")[-1]]:
        key = token.strip().replace(" ", "")
        if key in _UNITS:
            return _UNITS[key]
    return None


def _role(cols: list[str], role: str) -> str:
    pattern, code = _ROLES[role]
    hits = [c for c in cols if re.search(pattern, _norm(c), re.IGNORECASE)]
    if len(hits) != 1:
        state = "missing" if not hits else f"ambiguous ({hits})"
        raise SchemaError(
            f"Disintegration sheet: {role} column {state}. Columns present: {cols}", code
        )
    return hits[0]


def _parse_cell(value: object) -> tuple[float, bool, bool]:
    """(value, censored, non-numeric) for one replicate cell."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan, False, False
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value), False, False
    text = str(value).strip()
    if not text:
        return np.nan, False, False
    censored = text.startswith(">")
    try:
        return float(text.lstrip(">").strip()), censored, False
    except ValueError:
        return np.nan, False, True


def _read_truth(path: Path, sheet: str) -> tuple[bool, dict[str, float]]:
    raw = pd.read_excel(path, sheet_name=sheet, header=None)
    synthetic = False
    truth: dict[str, float] = {}
    for _, row in raw.iterrows():
        cells = [c for c in row.tolist() if pd.notna(c)]
        if not cells:
            continue
        line = " ".join(str(c) for c in cells)
        if any(m.lower() in line.lower() for m in config.SYNTHETIC_MARKERS):
            synthetic = True
        if len(cells) >= 2 and isinstance(cells[1], (int, float)) and not isinstance(
            cells[1], bool
        ):
            truth[str(cells[0]).strip()] = float(cells[1])
    return synthetic, truth


def load_disintegration(path: str | Path) -> DisintegrationData | None:
    """Load the sheet, or return ``None`` when the workbook has none."""
    src = Path(path)
    sheets = list(pd.ExcelFile(src).sheet_names)
    sheet = find_sheet(sheets, settings.SHEET_NAMES)
    if sheet is None:
        return None

    frame = pd.read_excel(src, sheet_name=sheet)
    frame = frame.dropna(how="all")
    cols = [str(c) for c in frame.columns]
    frame.columns = cols

    id_col = _role(cols, "id")
    case_col = _role(cols, "case")
    grade_col = _role(cols, "grade")
    api_hits = [c for c in cols if re.fullmatch(r"api", _norm(c), re.IGNORECASE)]
    api_col = api_hits[0] if len(api_hits) == 1 else None

    comp_cols: dict[str, str] = {}
    for c in cols:
        m = _COMP_RE.match(_norm(c))
        if m:
            comp_cols[m.group(1).lower()] = c

    dt_cols: dict[int, str] = {}
    for c in cols:
        m = _DT_RE.match(_norm(c))
        if m:
            dt_cols[int(m.group(1))] = c
    if not dt_cols:
        raise SchemaError(
            "Disintegration sheet: no replicate columns (expected 'DT_1 [min]', "
            f"'DT_2 [min]', ...). Columns present: {cols}",
            104,
        )
    factors = {c: _unit(c) for c in dt_cols.values()}
    if any(f is None for f in factors.values()):
        raise SchemaError(
            "Disintegration sheet: cannot determine the time unit of "
            f"{[c for c, f in factors.items() if f is None]}. Put it in the header, "
            "e.g. 'DT_1 [min]'. The unit is never assumed.",
            105,
        )
    if len(set(factors.values())) > 1:
        raise SchemaError(
            f"Disintegration sheet: replicate columns carry different units {factors}.", 106
        )
    to_h = float(next(iter(factors.values())))  # type: ignore[arg-type]

    end_hits = [c for c in cols if _END_RE.match(_norm(c))]
    end_col = end_hits[0] if end_hits else None

    ids = frame[id_col].astype(str).str.strip()
    if ids.duplicated().any():
        raise SchemaError(
            f"Disintegration sheet: duplicate IDs {sorted(set(ids[ids.duplicated()]))}.", 107
        )

    known = {id_col, case_col, grade_col, *comp_cols.values(), *dt_cols.values()}
    known |= {c for c in (api_col, end_col) if c is not None}
    unrecognised = [c for c in cols if c not in known]

    default_end = settings.DT_DEFAULT_TEST_END_H
    records: list[dict[str, object]] = []
    n_nonnumeric = 0
    n_nonpositive = 0
    ends: list[float] = []
    for i, row in enumerate(frame.itertuples(index=False)):
        values = dict(zip(cols, row, strict=True))
        end_h = default_end
        if end_col is not None:
            end_val, _, bad = _parse_cell(values[end_col])
            if np.isfinite(end_val) and end_val > 0:
                end_unit = _unit(end_col) or to_h
                end_h = end_val * end_unit
            n_nonnumeric += int(bad)
        ends.append(end_h)
        base = {
            "id": ids.iloc[i],
            "api": str(values[api_col]).strip() if api_col else "",
            "case": int(values[case_col]),
            "grade": str(values[grade_col]).strip(),
            "api_wt": float(values[comp_cols["api"]]) if "api" in comp_cols else np.nan,
            "hpmc_wt": float(values[comp_cols["hpmc"]]) if "hpmc" in comp_cols else np.nan,
            "lactose_wt": (
                float(values[comp_cols["lactose"]]) if "lactose" in comp_cols else np.nan
            ),
        }
        for rep in sorted(dt_cols):
            val, cens, bad = _parse_cell(values[dt_cols[rep]])
            n_nonnumeric += int(bad)
            if not np.isfinite(val):
                continue
            if val <= 0:
                n_nonpositive += 1
                continue
            hours = val * to_h
            censored = cens or hours >= end_h - 1e-9
            records.append(
                {**base, "replicate": rep, "dt_h": min(hours, end_h) if censored else hours,
                 "censored": bool(censored)}
            )

    long = pd.DataFrame.from_records(records, columns=list(LONG_COLUMNS))
    long = long.sort_values(["case", "grade", "replicate"], kind="mergesort").reset_index(
        drop=True
    )

    synthetic = False
    truth: dict[str, float] = {}
    notes = find_sheet(sheets, settings.NOTES_SHEET_NAMES)
    if notes is not None:
        synthetic, truth = _read_truth(src, notes)

    return DisintegrationData(
        long=long,
        unit_to_hours=to_h,
        test_end_h=float(max(ends)) if ends else default_end,
        test_end_from_sheet=end_col is not None,
        is_synthetic=synthetic,
        truth=truth,
        n_rows=len(frame),
        n_columns=len(cols),
        n_replicate_columns=len(dt_cols),
        n_unrecognised_columns=len(unrecognised),
        n_composition_columns=len(comp_cols),
        n_nonnumeric=n_nonnumeric,
        n_nonpositive=n_nonpositive,
        has_api_column=api_col is not None,
    )
