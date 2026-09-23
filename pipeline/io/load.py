"""Load the dissolution workbook into a tidy long frame.

Responsibilities:

* melt the hybrid replicate-column layout into one row per (id, replicate, time);
* convert the assay concentration into ``pct_released`` using the per-replicate
  tablet mass and API fraction (the file carries concentration, not % released);
* recover the intended design specification and grade -> viscosity map when the
  workbook supplies them, recording which source was used;
* detect whether the database is synthetic placeholder material.

The source file is opened read-only and never written (G8).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from pipeline import config
from pipeline.io.schema import DissolutionSchema, SchemaError, detect_schema

#: Canonical long-format columns produced by :func:`load_database`.
LONG_COLUMNS = (
    "id",
    "api",
    "case",
    "grade",
    "replicate",
    "time_h",
    "conc_ug_ml",
    "tablet_mass_mg",
    "dose_mg",
    "pct_released",
    "run_date",
    "api_wt",
    "hpmc_wt",
    "lactose_wt",
)


@dataclass(frozen=True)
class Database:
    """A loaded database plus the provenance needed to report on it honestly."""

    profiles: pd.DataFrame
    """Long format, one row per (id, replicate, time_h). See :data:`LONG_COLUMNS`."""

    design_spec: pd.DataFrame | None
    """Intended composition of each case, when the workbook declares it."""

    grade_viscosity_cp: dict[str, float]
    """Grade -> nominal viscosity actually used."""

    viscosity_source: str
    """Where :attr:`grade_viscosity_cp` came from: ``workbook`` or ``config-fallback``."""

    is_synthetic: bool
    """True when the workbook declares itself placeholder/synthetic material."""

    provenance_notes: tuple[str, ...] = field(default=())
    """Lines from the workbook's notes that triggered the synthetic determination."""

    source_path: Path = Path()
    vessel_volume_ml: float = config.VESSEL_VOLUME_ML
    schema: DissolutionSchema | None = None


def _find_sheet(sheets: list[str], *candidates: str) -> str | None:
    for candidate in candidates:
        for sheet in sheets:
            if sheet.strip().lower() == candidate:
                return sheet
    for candidate in candidates:
        for sheet in sheets:
            if candidate in sheet.strip().lower():
                return sheet
    return None


def _read_notes(path: Path, sheet: str) -> tuple[bool, tuple[str, ...]]:
    """Detect the synthetic-placeholder marker and return the triggering lines."""
    raw = pd.read_excel(path, sheet_name=sheet, header=None)
    lines: list[str] = []
    for _, row in raw.iterrows():
        cells = [str(v).strip() for v in row.tolist() if pd.notna(v)]
        if cells:
            lines.append(" | ".join(cells))
    hits = tuple(
        line
        for line in lines
        if any(marker.lower() in line.lower() for marker in config.SYNTHETIC_MARKERS)
    )
    return bool(hits), hits


def _read_design_sheet(
    path: Path, sheet: str
) -> tuple[pd.DataFrame | None, dict[str, float]]:
    """Recover the intended case compositions and the grade -> viscosity map."""
    raw = pd.read_excel(path, sheet_name=sheet, header=None)
    design: pd.DataFrame | None = None
    viscosity: dict[str, float] = {}

    for i, row in raw.iterrows():
        cells = [str(v).strip().lower() if pd.notna(v) else "" for v in row.tolist()]

        if design is None and cells and cells[0] == "case":
            block: list[dict[str, float]] = []
            for j in range(int(i) + 1, len(raw)):
                vals = raw.iloc[j].tolist()
                if pd.isna(vals[0]) or not str(vals[0]).strip():
                    break
                try:
                    block.append(
                        {
                            "case": int(float(vals[0])),
                            "api_wt": float(vals[1]),
                            "hpmc_wt": float(vals[2]),
                            "lactose_wt": float(vals[3]),
                        }
                    )
                except (TypeError, ValueError, IndexError):
                    break
            if block:
                design = pd.DataFrame(block)

        if cells and cells[0] == "grade":
            for j in range(int(i) + 1, len(raw)):
                vals = raw.iloc[j].tolist()
                if pd.isna(vals[0]) or not str(vals[0]).strip():
                    break
                try:
                    viscosity[str(vals[0]).strip()] = float(vals[1])
                except (TypeError, ValueError, IndexError):
                    break

    return design, viscosity


def load_database(path: str | Path, *, vessel_volume_ml: float | None = None) -> Database:
    """Read ``path`` and return a :class:`Database`.

    Raises :class:`SchemaError` with a specific message when the workbook cannot
    be interpreted.
    """
    src = Path(path)
    if not src.exists():
        raise SchemaError(f"Input database not found: {src}", 1)

    volume = config.VESSEL_VOLUME_ML if vessel_volume_ml is None else vessel_volume_ml
    if volume <= 0:
        raise SchemaError(f"Vessel volume must be positive, got {volume}.", 2)

    book = pd.ExcelFile(src)
    sheets = list(book.sheet_names)

    diss_sheet = _find_sheet(sheets, "dissolution", "data", "profiles")
    if diss_sheet is None:
        raise SchemaError(
            f"No dissolution sheet found in {src.name}. Looked for a sheet named "
            f"'Dissolution'/'Data'/'Profiles'; sheets present: {sheets}",
            3,
        )

    frame = book.parse(diss_sheet)
    schema = detect_schema(frame)

    if not any(r.mass is not None for r in schema.replicates):
        raise SchemaError(
            "No tablet-mass columns found. Dose = mass x API fraction is required to "
            "convert the concentration column into % released; without it the profiles "
            "cannot be normalised. Supply 'Mass_<rep>_mg' columns.",
            30,
        )

    records: list[pd.DataFrame] = []
    for rep in schema.replicates:
        cols = {
            schema.id_col: "id",
            schema.api_col: "api",
            schema.case_col: "case",
            schema.grade_col: "grade",
            schema.time_col: "time_h",
            rep.conc: "conc_ug_ml",
            schema.component_cols["api"]: "api_wt",
            schema.component_cols["hpmc"]: "hpmc_wt",
            schema.component_cols["lactose"]: "lactose_wt",
        }
        if rep.mass:
            cols[rep.mass] = "tablet_mass_mg"
        if rep.date:
            cols[rep.date] = "run_date"

        part = frame[list(cols)].rename(columns=cols).copy()
        part["replicate"] = rep.index
        if "run_date" not in part.columns:
            part["run_date"] = pd.NaT
        records.append(part)

    long = pd.concat(records, ignore_index=True)
    long["time_h"] = long["time_h"].astype(float) * schema.time_to_hours
    long["conc_ug_ml"] = long["conc_ug_ml"].astype(float) * schema.conc_to_ug_per_ml
    long["tablet_mass_mg"] = long["tablet_mass_mg"].astype(float) * schema.mass_to_mg

    long["dose_mg"] = long["tablet_mass_mg"] * long["api_wt"] / 100.0
    if (long["dose_mg"] <= 0).any():
        bad = long.loc[long["dose_mg"] <= 0, "id"].unique()[:5]
        raise SchemaError(
            f"Non-positive dose computed for id(s) {list(bad)}. Check tablet mass and "
            "API wt% columns.",
            31,
        )

    # ug/mL * mL -> ug; /1000 -> mg; over dose in mg -> fraction; x100 -> percent.
    long["pct_released"] = long["conc_ug_ml"] * volume / 1000.0 / long["dose_mg"] * 100.0

    long = long.sort_values(["id", "replicate", "time_h"], kind="mergesort").reset_index(drop=True)
    long = long[list(LONG_COLUMNS)]

    design_spec: pd.DataFrame | None = None
    viscosity: dict[str, float] = {}
    design_sheet = _find_sheet(sheets, "design")
    if design_sheet is not None:
        design_spec, viscosity = _read_design_sheet(src, design_sheet)

    if viscosity:
        source = "workbook"
    else:
        viscosity = dict(config.FALLBACK_GRADE_VISCOSITY_CP)
        source = "config-fallback"

    observed_grades = set(long["grade"].astype(str).unique())
    unmapped = sorted(observed_grades - set(viscosity))
    if unmapped:
        raise SchemaError(
            f"No nominal viscosity for grade(s) {unmapped}. log10(viscosity) is a model "
            "factor (AC4), so an unmapped grade cannot be modelled. Add it to the "
            "workbook's Design sheet or to config.FALLBACK_GRADE_VISCOSITY_CP.",
            32,
        )

    is_synthetic = False
    notes: tuple[str, ...] = ()
    notes_sheet = _find_sheet(sheets, "notes", "readme", "about")
    if notes_sheet is not None:
        is_synthetic, notes = _read_notes(src, notes_sheet)

    return Database(
        profiles=long,
        design_spec=design_spec,
        grade_viscosity_cp=viscosity,
        viscosity_source=source,
        is_synthetic=is_synthetic,
        provenance_notes=notes,
        source_path=src,
        vessel_volume_ml=volume,
        schema=schema,
    )
