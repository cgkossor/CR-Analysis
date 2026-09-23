"""A privacy-safe audit of how a database meets the pipeline's assumptions.

The real database cannot leave the machine it lives on, but the question "is the
data shaped the way the code expects?" still has to be answered somewhere else.
This module answers it with a report that carries **only booleans and
integers** -- counts, flags, enum codes and line numbers -- so it can be pasted
out of a terminal without disclosing a single measurement, name or header.

The guarantee is structural, not a matter of care:

* :meth:`AuditReport.add` rejects any value that is not a ``bool`` or ``int``.
  A float, a string or ``None`` raises ``TypeError``; there is no path by which
  a data value reaches the output.
* Labels are fixed identifiers written in this file and must match
  ``[a-z][a-z0-9_]*``. Nothing read from the workbook is ever used as a label.
* An exception is reported as an integer kind plus the pipeline module index
  and line number where it was raised. Its message is never printed, because
  messages routinely quote column names and IDs.

Every stage runs in isolation, so one crash is located precisely instead of
blanking the rest of the report. Run with::

    python -m pipeline.audit --input <workbook.xlsx>

The report is printed and also saved to ``outputs/reports/audit.txt`` (or the
path given with ``--out``).
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
import traceback
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np
import pandas as pd

from pipeline import config

if TYPE_CHECKING:
    from pipeline.analysis import Analysis
    from pipeline.io.load import Database
    from pipeline.stress.subsets import StressTest

#: Bumped whenever a label changes meaning, so a pasted report can be read
#: against the right version of this file.
AUDIT_VERSION = 1

BEGIN = "=== CR-AUDIT"
END = "=== END CR-AUDIT"

SECTIONS: dict[str, str] = {
    "A": "environment",
    "B": "workbook preflight",
    "C": "ingest",
    "D": "raw values",
    "E": "time",
    "F": "release and metrics",
    "G": "design and quality",
    "R": "full analysis",
    "H": "weibull surfaces",
    "I": "cross-validation",
    "J": "equivalence",
    "K": "stress test",
    "L": "doe",
    "M": "formulator",
    "N": "response space",
    "T": "disintegration",
    "O": "dashboard payload",
}

_LABEL_RE = re.compile(r"^[a-z][a-z0-9_]*$")

#: The whole grammar of a rendered report, including the dashboard's ``[W]``
#: section. There is deliberately no production that admits free text, a float,
#: or anything read from the workbook; the tests hold every line to it.
REPORT_LINE = re.compile(
    r"^(?:"
    r"=== CR-AUDIT v\d+ commit=(?:[0-9a-f]{4,40}|unknown) ==="
    r"|\[[A-Z]\] [a-z -]+ (?:ok=[01]|skipped)(?: warn_runtime=\d+ warn_other=\d+)?"
    r"|[A-Z]\d{2,} [a-z][a-z0-9_]*=-?\d+"
    r"|=== END CR-AUDIT lines=\d+ ==="
    r")$"
)
_PKG_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = _PKG_ROOT.parent

#: Integer codes for exception kinds. The message is never reported.
ERROR_KINDS: dict[str, int] = {
    "KeyError": 1,
    "ValueError": 2,
    "IndexError": 3,
    "TypeError": 4,
    "ZeroDivisionError": 5,
    "FloatingPointError": 5,
    "LinAlgError": 6,
    "SchemaError": 7,
    "AttributeError": 8,
    "AssertionError": 10,
    "FileNotFoundError": 11,
    "MemoryError": 12,
}
_OTHER_ERROR = 9

T = TypeVar("T")


def module_index() -> list[str]:
    """Pipeline modules in a fixed order; an error location is an index into this."""
    return sorted(p.relative_to(_REPO_ROOT).as_posix() for p in _PKG_ROOT.rglob("*.py"))


def _coerce(value: object) -> bool | int:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    raise TypeError(
        f"audit values must be bool or int, got {type(value).__name__}; "
        "convert to a count or a flag first"
    )


@dataclass(frozen=True)
class Entry:
    """One audited fact."""

    code: str
    label: str
    value: bool | int

    @property
    def text(self) -> str:
        v = self.value
        return f"{self.code} {self.label}={int(v) if isinstance(v, bool) else v}"


@dataclass
class AuditReport:
    """Ordered, type-enforced collection of audit entries."""

    commit: str = "unknown"
    entries: list[Entry] = field(default_factory=list)
    stage_ok: dict[str, bool] = field(default_factory=dict)
    stage_warnings: dict[str, tuple[int, int]] = field(default_factory=dict)
    """Per stage: (RuntimeWarnings, other warnings) raised while it ran."""
    _counters: dict[str, int] = field(default_factory=dict)

    def add(self, section: str, label: str, value: object) -> None:
        if section not in SECTIONS:
            raise ValueError(f"unknown audit section {section!r}")
        if not _LABEL_RE.match(label):
            raise ValueError(f"audit label {label!r} is not a fixed identifier")
        coerced = _coerce(value)
        n = self._counters.get(section, 0) + 1
        self._counters[section] = n
        self.entries.append(Entry(f"{section}{n:02d}", label, coerced))

    def get(self, label: str) -> bool | int | None:
        for e in self.entries:
            if e.label == label:
                return e.value
        return None

    def lines(self) -> list[str]:
        body: list[str] = []
        for letter, title in SECTIONS.items():
            if letter not in self.stage_ok and not any(
                e.code.startswith(letter) for e in self.entries
            ):
                continue
            ok = self.stage_ok.get(letter)
            status = "skipped" if ok is None else f"ok={int(ok)}"
            if letter in self.stage_warnings:
                rw, ow = self.stage_warnings[letter]
                status += f" warn_runtime={rw} warn_other={ow}"
            body.append(f"[{letter}] {title} {status}")
            body.extend(e.text for e in self.entries if e.code[0] == letter)
        return body

    def render(self) -> str:
        body = self.lines()
        head = f"{BEGIN} v{AUDIT_VERSION} commit={self.commit} ==="
        tail = f"{END} lines={len(body)} ==="
        return "\n".join([head, *body, tail])

    def as_payload(self) -> dict[str, Any]:
        return {
            "version": AUDIT_VERSION,
            "commit": self.commit,
            # A list, not a dict: data.js is written with sorted keys, which
            # would scramble the stage order.
            "sections": [[k, v] for k, v in SECTIONS.items()],
            "stage_ok": {k: bool(v) for k, v in self.stage_ok.items()},
            # Warning counts are left out: libraries emit some warnings only
            # once per process, so they would differ between the two passes of
            # --check-determinism and break the byte-identical data.js (G10).
            # The terminal report keeps them.
            "entries": [{"code": e.code, "label": e.label, "value": e.value} for e in self.entries],
        }


def _error_frames(exc: BaseException) -> list[tuple[int, int]]:
    """(module index, line) for the innermost pipeline frames, innermost first."""
    modules = module_index()
    frames: list[tuple[int, int]] = []
    for fs in reversed(traceback.extract_tb(exc.__traceback__)):
        try:
            rel = Path(fs.filename).resolve().relative_to(_REPO_ROOT).as_posix()
        except ValueError:
            continue
        if rel in modules:
            frames.append((modules.index(rel), int(fs.lineno or 0)))
    return frames[:3]


def _run_stage(report: AuditReport, section: str, fn: Callable[[], T]) -> T | None:
    """Run one stage; on failure record where, never what."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            result = fn()
            ok = True
        except Exception as exc:
            result = None
            ok = False
            report.add(
                section, "stage_error_kind", ERROR_KINDS.get(type(exc).__name__, _OTHER_ERROR)
            )
            code = getattr(exc, "code", None)
            if isinstance(code, int) and not isinstance(code, bool):
                report.add(section, "schema_error_code", code)
            for depth, (mod, line) in enumerate(_error_frames(exc), start=1):
                report.add(section, f"error_frame{depth}_module", mod)
                report.add(section, f"error_frame{depth}_line", line)
    runtime = sum(1 for w in caught if issubclass(w.category, RuntimeWarning))
    report.stage_warnings[section] = (runtime, len(caught) - runtime)
    report.stage_ok[section] = ok
    return result


# --------------------------------------------------------------------- helpers
def _finite_count(values: Any) -> int:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    return int(np.isfinite(arr).sum())


def _nonnumeric(series: pd.Series) -> int:
    """Cells that hold something but do not parse as a number."""
    parsed = pd.to_numeric(series, errors="coerce")
    return int((series.notna() & parsed.isna()).sum())


def _vmajor_minor(module: str) -> int:
    try:
        mod = __import__(module)
        parts = str(mod.__version__).split(".")
        return int(parts[0]) * 100 + int(re.sub(r"\D.*", "", parts[1]) or 0)
    except Exception:
        return -1


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], cwd=_REPO_ROOT, capture_output=True, text=True, timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def _commit() -> str:
    sha = _git("rev-parse", "--short", "HEAD")
    return sha if sha and re.fullmatch(r"[0-9a-f]{4,40}", sha) else "unknown"


# ---------------------------------------------------------------------- stages
def _environment(r: AuditReport) -> None:
    r.add("A", "audit_version", AUDIT_VERSION)
    r.add("A", "python_version", sys.version_info.major * 100 + sys.version_info.minor)
    for name in ("numpy", "pandas", "scipy", "statsmodels", "openpyxl"):
        r.add("A", f"{name}_version", _vmajor_minor(name))
    status = _git("status", "--porcelain", "--untracked-files=no")
    r.add("A", "git_available", status is not None)
    r.add("A", "tracked_files_modified", bool(status))
    r.add("A", "pipeline_modules", len(module_index()))


def _preflight(r: AuditReport, path: Path) -> None:
    """Look at the raw workbook before any schema logic can reject it."""
    from pipeline.io import schema as sc
    from pipeline.io.load import _find_sheet

    r.add("B", "input_exists", path.exists())
    if not path.exists():
        return
    r.add("B", "input_is_xlsx", path.suffix.lower() in (".xlsx", ".xlsm"))

    book = pd.ExcelFile(path)
    sheets = [str(s) for s in book.sheet_names]
    r.add("B", "sheets", len(sheets))
    diss = _find_sheet(sheets, "dissolution", "data", "profiles")
    r.add("B", "data_sheet_found", diss is not None)
    r.add(
        "B",
        "data_sheet_exact_name",
        any(s.strip().lower() in ("dissolution", "data", "profiles") for s in sheets),
    )
    r.add("B", "data_sheet_is_first", diss is not None and sheets.index(diss) == 0)
    r.add("B", "design_sheet_found", _find_sheet(sheets, "design") is not None)
    r.add("B", "notes_sheet_found", _find_sheet(sheets, "notes", "readme", "about") is not None)
    if diss is None:
        return

    frame = book.parse(diss)
    cols = [str(c) for c in frame.columns]
    norm = [sc._norm(c) for c in cols]
    r.add("B", "data_rows", len(frame))
    r.add("B", "data_cols", len(cols))
    r.add("B", "blank_headers", sum(1 for c in norm if not c or c.startswith("Unnamed")))
    r.add(
        "B",
        "duplicate_headers",
        sum(1 for c in norm if re.search(r"\.\d+$", c) and c.rsplit(".", 1)[0] in norm),
    )
    r.add("B", "fully_blank_rows", int(frame.isna().all(axis=1).sum()))

    roles = {
        "id": r"^id$",
        "case": r"^case$",
        "api": r"^api$",
        "grade": r"^(hpmc\s*)?grade$",
    }
    found: dict[str, str | None] = {}
    for role, pattern in roles.items():
        hits = [c for c, n in zip(cols, norm, strict=True) if re.search(pattern, n, re.I)]
        r.add("B", f"{role}_col_matches", len(hits))
        found[role] = hits[0] if len(hits) == 1 else None

    comp = {
        m.group(1).lower(): c
        for c, n in zip(cols, norm, strict=True)
        if (m := sc._COMP_RE.match(n)) is not None
    }
    for name in ("api", "hpmc", "lactose"):
        r.add("B", f"composition_{name}_found", name in comp)
    conc = [c for c, n in zip(cols, norm, strict=True) if sc._CONC_RE.match(n)]
    mass = [c for c, n in zip(cols, norm, strict=True) if sc._MASS_RE.match(n)]
    date = [c for c, n in zip(cols, norm, strict=True) if sc._DATE_RE.match(n)]
    time = [c for c, n in zip(cols, norm, strict=True) if sc._TIME_RE.match(n)]
    recognised = set(comp.values()) | set(conc) | set(mass) | set(date) | set(time)
    recognised |= {c for c in found.values() if c is not None}

    looks_like_comp = [
        c
        for c, n in zip(cols, norm, strict=True)
        if c not in recognised and re.search(r"(wt\s*%|%|w/w|\bwt\b)", n, re.I)
    ]
    r.add("B", "composition_like_cols_unrecognised", len(looks_like_comp))
    r.add(
        "B",
        "component_named_cols_unrecognised",
        sum(
            1
            for c, n in zip(cols, norm, strict=True)
            if c not in recognised and re.match(r"^(api|hpmc|lactose)\b", n, re.I)
        ),
    )
    r.add("B", "unrecognised_cols", sum(1 for c in cols if c not in recognised))

    r.add("B", "time_col_candidates", len(time))
    time_factor = sc._extract_unit(time[0], sc._TIME_UNITS) if time else None
    r.add("B", "time_unit_recognised", time_factor is not None)
    r.add(
        "B", "time_unit_enum_1min_2h", 0 if time_factor is None else (1 if time_factor < 1 else 2)
    )

    conc_units = [sc._extract_unit(c, sc._CONC_UNITS) for c in conc]
    r.add("B", "conc_cols", len(conc))
    r.add("B", "conc_cols_unit_recognised", sum(1 for u in conc_units if u is not None))
    r.add("B", "conc_distinct_units", len({u for u in conc_units if u is not None}))
    first_unit = next((u for u in conc_units if u is not None), None)
    r.add(
        "B",
        "conc_unit_enum_1ugml_2mgml",
        0 if first_unit is None else (1 if first_unit == 1.0 else 2),
    )
    r.add("B", "mass_cols", len(mass))
    r.add(
        "B",
        "mass_cols_unit_recognised",
        sum(1 for c in mass if sc._extract_unit(c, sc._MASS_UNITS) is not None),
    )
    r.add("B", "date_cols", len(date))

    def idx(rx: re.Pattern[str], names: list[str]) -> list[int]:
        out: list[int] = []
        for c in names:
            m = rx.match(sc._norm(c))
            if m:
                out.append(int(m.group(1)))
        return sorted(out)

    ci, mi = idx(sc._CONC_RE, conc), idx(sc._MASS_RE, mass)
    r.add("B", "replicate_indices_contiguous", ci == list(range(1, len(ci) + 1)))
    r.add("B", "mass_indices_match_conc", mi == ci)
    ti = sorted(
        int(m.group(2)) for c in time if (m := sc._TIME_IDX_RE.match(sc._norm(c))) is not None
    )
    r.add("B", "time_cols_indexed", len(ti))
    r.add("B", "time_indices_match_conc", bool(ti) and set(ci) <= set(ti))
    if len(time) > 1:
        first = pd.to_numeric(frame[time[0]], errors="coerce")
        r.add(
            "B",
            "time_cols_identical_to_first",
            sum(1 for c in time[1:] if first.equals(pd.to_numeric(frame[c], errors="coerce"))),
        )

    # --- cell-level content, per role -------------------------------------
    id_col = found["id"]
    for role in ("id", "case", "api", "grade"):
        col = found[role]
        if col is None:
            continue
        r.add("B", f"{role}_blank_cells", int(frame[col].isna().sum()))
        r.add("B", f"{role}_distinct", int(frame[col].dropna().astype(str).nunique()))
    if found["case"] is not None:
        case = frame[found["case"]].dropna()
        parsed = pd.to_numeric(case, errors="coerce")
        non_int = parsed.isna() | (parsed.notna() & (parsed.round() != parsed))
        r.add("B", "case_non_integer_cells", int(non_int.sum()))
    if found["id"] is not None and time:
        # Merged cells in Excel read back as one value followed by blanks.
        r.add(
            "B",
            "rows_with_time_but_blank_id",
            int((frame[found["id"]].isna() & frame[time[0]].notna()).sum()),
        )

    numeric_cols = [*comp.values(), *conc, *mass] + ([time[0]] if time else [])
    r.add("B", "nonnumeric_cells_in_numeric_cols", sum(_nonnumeric(frame[c]) for c in numeric_cols))
    for name, c in comp.items():
        r.add("B", f"composition_{name}_blank_cells", int(frame[c].isna().sum()))
        r.add("B", f"composition_{name}_nonnumeric_cells", _nonnumeric(frame[c]))
    if time:
        t = pd.to_numeric(frame[time[0]], errors="coerce")
        r.add("B", "time_blank_cells", int(frame[time[0]].isna().sum()))
        r.add("B", "time_nonnumeric_cells", _nonnumeric(frame[time[0]]))
        r.add("B", "time_negative_cells", int((t < 0).sum()))
    r.add("B", "conc_blank_cells", sum(int(frame[c].isna().sum()) for c in conc))
    r.add("B", "conc_nonnumeric_cells", sum(_nonnumeric(frame[c]) for c in conc))
    r.add(
        "B",
        "conc_negative_cells",
        sum(int((pd.to_numeric(frame[c], errors="coerce") < 0).sum()) for c in conc),
    )
    r.add("B", "mass_blank_cells", sum(int(frame[c].isna().sum()) for c in mass))
    r.add("B", "mass_nonnumeric_cells", sum(_nonnumeric(frame[c]) for c in mass))

    if id_col is not None:
        grouped = frame.groupby(frame[id_col].astype(str), sort=False)
        empty_reps = 0
        partial_mass = 0
        varying_mass = 0
        for _, g in grouped:
            for c in conc:
                if g[c].isna().all():
                    empty_reps += 1
            for c in mass:
                blanks = int(g[c].isna().sum())
                if 0 < blanks < len(g):
                    partial_mass += 1
                if g[c].dropna().nunique() > 1:
                    varying_mass += 1
        r.add("B", "replicates_entirely_blank", empty_reps)
        r.add("B", "mass_partially_filled_id_reps", partial_mass)
        r.add("B", "mass_varies_within_id_rep", varying_mass)
        if time:
            dup = 0
            unsorted = 0
            for _, g in grouped:
                tt = pd.to_numeric(g[time[0]], errors="coerce").dropna().to_numpy()
                dup += int(len(tt) - len(np.unique(tt)))
                unsorted += int(bool(len(tt) > 1 and np.any(np.diff(tt) < 0)))
            r.add("B", "duplicate_times_within_id", dup)
            r.add("B", "ids_with_unsorted_time", unsorted)


def _ingest(r: AuditReport, path: Path, db: Database | None) -> Database:
    from pipeline.io.load import _find_sheet, _read_design_sheet, load_database

    if db is None:
        db = load_database(path)
    r.add("C", "loaded", True)
    r.add("C", "viscosity_from_workbook", db.viscosity_source == "workbook")
    r.add("C", "declared_synthetic", db.is_synthetic)
    r.add("C", "vessel_volume_is_default", db.vessel_volume_ml == config.VESSEL_VOLUME_ML)
    r.add("C", "replicates_in_schema", db.schema.n_replicates if db.schema else -1)
    r.add(
        "C",
        "time_per_replicate",
        bool(db.schema and all(rep.time for rep in db.schema.replicates)),
    )
    r.add("C", "analysis_window_h", math.floor(config.ANALYSIS_WINDOW_H))
    r.add("C", "rows_beyond_window_dropped", db.rows_beyond_window)
    r.add(
        "C",
        "time_unit_enum_1min_2h",
        0 if db.schema is None else (1 if db.schema.time_to_hours < 1 else 2),
    )
    sheets = list(pd.ExcelFile(path).sheet_names)
    design_sheet = _find_sheet(sheets, "design")
    if design_sheet is not None:
        spec, visc = _read_design_sheet(path, design_sheet)
        r.add("C", "design_spec_rows", 0 if spec is None else len(spec))
        r.add("C", "design_sheet_viscosity_entries", len(visc))
    grades = set(db.profiles["grade"].astype(str).unique())
    r.add("C", "grades_with_viscosity", sum(1 for g in grades if g in db.grade_viscosity_cp))
    r.add(
        "C",
        "grades_in_fallback_table",
        sum(1 for g in grades if g in config.FALLBACK_GRADE_VISCOSITY_CP),
    )
    return db


def _raw_values(r: AuditReport, db: Database) -> None:
    p = db.profiles
    r.add("D", "long_rows", len(p))
    for col in (
        "time_h",
        "conc_ug_ml",
        "tablet_mass_mg",
        "api_wt",
        "hpmc_wt",
        "lactose_wt",
        "dose_mg",
        "pct_released",
    ):
        r.add("D", f"nan_{col}", int(p[col].isna().sum()))
    r.add("D", "negative_conc", int((p["conc_ug_ml"] < 0).sum()))
    r.add("D", "duplicate_id_rep_time", int(p.duplicated(["id", "replicate", "time_h"]).sum()))

    r.add("D", "n_ids", int(p["id"].nunique()))
    r.add("D", "n_apis", int(p["api"].nunique()))
    r.add("D", "n_cases", int(p["case"].nunique()))
    r.add("D", "n_grades", int(p["grade"].nunique()))
    r.add("D", "n_replicate_indices", int(p["replicate"].nunique()))

    per_id = p.groupby("id")
    r.add("D", "ids_with_multiple_cases", int((per_id["case"].nunique() > 1).sum()))
    r.add("D", "ids_with_multiple_grades", int((per_id["grade"].nunique() > 1).sum()))
    r.add("D", "ids_with_multiple_apis", int((per_id["api"].nunique() > 1).sum()))
    comp_cols = ["api_wt", "hpmc_wt", "lactose_wt"]
    comps = p.drop_duplicates(["id", *comp_cols])
    r.add("D", "ids_with_multiple_compositions", int((comps.groupby("id").size() > 1).sum()))
    by_case = p.drop_duplicates(["case", *comp_cols]).groupby("case").size()
    r.add("D", "cases_composition_differs_across_rows", int((by_case > 1).sum()))
    cg = p.groupby(["case", "grade"])["id"].nunique()
    r.add("D", "case_grade_cells_with_multiple_ids", int((cg > 1).sum()))
    cga = p.groupby(["case", "grade"])["api"].nunique()
    r.add("D", "case_grade_cells_with_multiple_apis", int((cga > 1).sum()))

    uniq = p.drop_duplicates(comp_cols)[comp_cols].dropna()
    sums = uniq.sum(axis=1)
    r.add("D", "distinct_compositions", len(uniq))
    r.add("D", "compositions_sum_off_100_gt_0p05", int(((sums - 100).abs() > 0.05).sum()))
    r.add("D", "compositions_sum_off_100_gt_0p5", int(((sums - 100).abs() > 0.5).sum()))
    r.add("D", "compositions_sum_off_100_gt_5", int(((sums - 100).abs() > 5).sum()))
    r.add("D", "composition_sum_floor", math.floor(sums.min()) if len(sums) else -1)
    r.add("D", "composition_sum_ceil", math.ceil(sums.max()) if len(sums) else -1)
    for c in comp_cols:
        r.add("D", f"{c}_distinct_levels", int(p[c].nunique()))

    reps = per_id["replicate"].nunique()
    r.add("D", "replicates_per_id_min", int(reps.min()) if len(reps) else 0)
    r.add("D", "replicates_per_id_max", int(reps.max()) if len(reps) else 0)
    finite = p.assign(ok=np.isfinite(p["pct_released"])).groupby(["id", "replicate"])["ok"].sum()
    r.add("D", "profiles_total", len(finite))
    r.add("D", "profiles_with_zero_finite_points", int((finite == 0).sum()))
    r.add("D", "profiles_with_lt4_finite_points", int((finite < 4).sum()))
    mass_var = p.groupby(["id", "replicate"])["tablet_mass_mg"].nunique(dropna=False)
    r.add("D", "profiles_mass_not_constant", int((mass_var > 1).sum()))
    dose_bad = p.groupby(["id", "replicate"])["dose_mg"].apply(lambda s: bool(s.isna().any()))
    r.add("D", "profiles_with_any_nan_dose", int(dose_bad.sum()))


def _time(r: AuditReport, db: Database) -> None:
    from pipeline.profiles.grid import build_time_grid

    p = db.profiles[np.isfinite(db.profiles["pct_released"]) & np.isfinite(db.profiles["time_h"])]
    groups = p.groupby(["id", "replicate"])
    lengths = groups.size()
    last = groups["time_h"].max()
    first = groups["time_h"].min()
    r.add("E", "profiles_with_data", len(lengths))
    r.add("E", "timepoints_min", int(lengths.min()) if len(lengths) else 0)
    r.add("E", "timepoints_max", int(lengths.max()) if len(lengths) else 0)
    r.add("E", "profiles_starting_at_t0", int((first.abs() < 1e-9).sum()))
    r.add("E", "profiles_starting_after_1h", int((first > 1.0).sum()))
    r.add("E", "last_time_h_floor_min", math.floor(last.min()) if len(last) else -1)
    r.add("E", "last_time_h_ceil_max", math.ceil(last.max()) if len(last) else -1)
    for h in (8, 12, 24):
        r.add("E", f"profiles_ending_before_{h}h", int((last < h - 1e-9).sum()))
    r.add("E", "profiles_covering_1h", int((first <= 1.0).sum()))
    vectors = groups["time_h"].apply(lambda s: tuple(np.round(np.sort(s.to_numpy()), 6)))
    r.add("E", "distinct_time_vectors", int(vectors.nunique()))
    r.add(
        "E",
        "samples_within_2h_max",
        int(groups["time_h"].apply(lambda s: int(((s > 0) & (s <= 2)).sum())).max())
        if len(lengths)
        else 0,
    )

    grid = build_time_grid(
        [
            g["time_h"].to_numpy(dtype=float)
            for _, g in db.profiles.groupby(["case", "grade", "replicate"])
        ]
    )
    r.add("E", "grid_points", len(grid.times_h))
    r.add("E", "grid_raw_times", grid.n_raw_times)
    r.add("E", "grid_collapsed", grid.collapsed)
    r.add("E", "grid_nonfinite_points", int((~np.isfinite(grid.times_h)).sum()))
    r.add("E", "grid_covers_24h", bool(len(grid.times_h) and np.nanmax(grid.times_h) >= 24 - 1e-9))
    r.add("E", "grid_resampled_to_nominal", grid.resampled)

    # How far interpolation had to reach: the widest bracket around any grid
    # point, per replicate, and how many grid points the gap guard blanked.
    from pipeline.profiles.grid import bracket_gaps

    widest = 0.0
    blanked = 0
    clocks_differ = 0
    for _, g in db.profiles.groupby("id"):
        vectors = [
            np.sort(rep["time_h"].to_numpy(dtype=float)) for _, rep in g.groupby("replicate")
        ]
        clocks_differ += int(
            any(len(v) != len(vectors[0]) or not np.allclose(v, vectors[0]) for v in vectors[1:])
        )
        for v in vectors:
            gaps = bracket_gaps(v, grid.times_h)
            inside = np.isfinite(gaps)
            if inside.any():
                widest = max(widest, float(gaps[inside].max()))
            if grid.max_gap_h is not None:
                blanked += int((inside & (gaps > grid.max_gap_h + 1e-12)).sum())
    r.add("E", "interp_widest_bracket_min", math.ceil(widest * 60))
    r.add("E", "grid_points_blanked_by_gap_guard", blanked)
    r.add("E", "ids_where_replicate_clocks_differ", clocks_differ)


def _release(r: AuditReport, db: Database, replicates: pd.DataFrame | None) -> pd.DataFrame:
    from pipeline.profiles.table import build_replicate_table

    p = db.profiles
    groups = p.sort_values("time_h").groupby(["id", "replicate"])
    peak = groups["pct_released"].max()
    bins = [
        (-math.inf, 0),
        (0, 10),
        (10, 50),
        (50, 80),
        (80, 105),
        (105, 150),
        (150, 1000),
        (1000, math.inf),
    ]
    names = ["le0", "0_10", "10_50", "50_80", "80_105", "105_150", "150_1000", "gt1000"]
    for (lo, hi), name in zip(bins, names, strict=True):
        r.add("F", f"profiles_peak_pct_{name}", int(((peak > lo) & (peak <= hi)).sum()))
    r.add("F", "profiles_peak_nan", int(peak.isna().sum()))
    steps = groups["pct_released"].apply(lambda s: np.diff(s.dropna().to_numpy(dtype=float)))
    r.add("F", "backward_steps", int(sum(int((d < 0).sum()) for d in steps)))
    r.add("F", "backward_steps_gt_5pct", int(sum(int((d < -5).sum()) for d in steps)))
    r.add("F", "profiles_with_backward_step", int(sum(1 for d in steps if (d < 0).any())))
    r.add("F", "observations_above_100pct", int((p["pct_released"] > 100).sum()))
    r.add("F", "observations_negative_pct", int((p["pct_released"] < 0).sum()))

    table = replicates if replicates is not None else build_replicate_table(db)
    r.add("F", "replicate_rows", len(table))
    metric_cols = [
        "t10",
        "t25",
        "t50",
        "t80",
        "pct_1h",
        "pct_2h",
        "pct_4h",
        "pct_8h",
        "pct_12h",
        "pct_24h",
        "mdt_h",
        "early_slope",
        "late_slope",
        "slope_ratio",
        "log10_td",
        "weibull_beta",
        "weibull_f_inf",
        "peppas_n",
    ]
    for col in metric_cols:
        r.add("F", f"finite_{col}", _finite_count(table[col]) if col in table.columns else -1)
    for flag in (
        "t50_censored",
        "t80_censored",
        "weibull_valid",
        "weibull_asymptote_identified",
        "peppas_valid",
        "mdt_truncated",
    ):
        r.add(
            "F",
            f"count_{flag}",
            int(table[flag].fillna(False).astype(bool).sum()) if flag in table.columns else -1,
        )
    return table


def _design(r: AuditReport, db: Database) -> None:
    from pipeline.design.report import design_points
    from pipeline.io.quality import build_quality_report

    q = build_quality_report(db)
    m = q.mixture
    r.add("G", "is_mixture", m.is_mixture)
    r.add("G", "composition_rank", m.rank)
    r.add("G", "fixed_components", len(m.fixed_components))
    r.add("G", "observed_cells", q.observed_cells)
    r.add("G", "expected_cells_from_config", q.expected_cells)
    r.add("G", "missing_cells_in_crossing", len(q.missing_cells))
    r.add("G", "cases_missing_vs_design_sheet", len(q.cases_missing_entirely))
    r.add("G", "quality_warnings", len(q.warnings))
    r.add("G", "fully_censored_ids", len(q.fully_censored_ids))
    r.add("G", "partially_censored_ids", len(q.partially_censored_ids))
    r.add("G", "peppas_failing_profiles", len(q.peppas_failing_profiles))
    r.add(
        "G",
        "mass_corr_nonfinite",
        sum(1 for v in q.mass_composition_corr.values() if not np.isfinite(v)),
    )
    r.add("G", "run_dates", q.run_dates)

    pts = design_points(db)
    r.add("G", "design_points", len(pts))
    r.add("G", "process_levels", int(pts["v_coded"].nunique()))
    r.add("G", "design_points_nan_viscosity", int(pts["viscosity_cp"].isna().sum()))
    r.add("G", "design_points_nan_v_coded", int(pts["v_coded"].isna().sum()))
    r.add("G", "grades_per_case_min", int(pts.groupby("case")["grade"].nunique().min()))
    r.add("G", "cases_per_grade_min", int(pts.groupby("grade")["case"].nunique().min()))


def _surfaces(r: AuditReport, a: Analysis) -> None:
    from pipeline.analysis import SURFACE_RESPONSES

    r.add("H", "model_process_power", a.model_spec.process_power)
    r.add(
        "H",
        "model_degree_enum_1lin_2quad_3cubic",
        {"linear": 1, "quadratic": 2, "special_cubic": 3}.get(a.model_spec.composition_degree, 0),
    )
    for name in SURFACE_RESPONSES:
        s = a.surfaces.get(name)
        r.add("H", f"{name}_present", s is not None)
        col = f"{name}_mean"
        if col in a.design_points.columns:
            r.add("H", f"{name}_nan_design_points", int(a.design_points[col].isna().sum()))
        if s is None:
            continue
        f = s.fit
        stats_finite = all(np.isfinite(v) for v in (f.r_squared, f.rmse, f.adj_r_squared))
        r.add("H", f"{name}_estimable", f.estimable)
        r.add("H", f"{name}_n_obs", f.n_obs)
        r.add("H", f"{name}_terms_kept", len(s.kept_terms))
        r.add("H", f"{name}_terms_dropped", len(s.dropped_terms))
        r.add("H", f"{name}_df_residual", f.df_residual)
        r.add("H", f"{name}_stats_finite", stats_finite)
        r.add("H", f"{name}_estimable_but_nan", f.estimable and not stats_finite)
        r.add("H", f"{name}_r2_ge_0p5", bool(np.isfinite(f.r_squared) and f.r_squared >= 0.5))
        r.add("H", f"{name}_pred_r2_finite", bool(np.isfinite(f.pred_r_squared)))
        r.add(
            "H",
            f"{name}_nonfinite_coefficients",
            sum(1 for c in f.coefficients if not np.isfinite(c.estimate)),
        )
    r.add("H", "lever_rows", len(a.lever_effects))
    if len(a.lever_effects):
        r.add(
            "H",
            "lever_rows_nonfinite",
            int((~np.isfinite(a.lever_effects["delta_log10_td"].to_numpy(dtype=float))).sum()),
        )


def _cv(r: AuditReport, a: Analysis) -> None:
    cv = a.cross_validation
    r.add("I", "design_points", len(a.design_points))
    r.add("I", "folds_run", cv.n_folds)
    r.add("I", "folds_missing", len(a.design_points) - cv.n_folds)
    r.add("I", "profile_rmse_finite", bool(np.isfinite(cv.profile_rmse_pct)))
    r.add("I", "median_f2_finite", bool(np.isfinite(cv.median_f2)))
    r.add("I", "folds_rmse_nan", sum(1 for f in cv.folds if not np.isfinite(f.profile_rmse_pct)))
    r.add("I", "folds_f2_invalid", sum(1 for f in cv.folds if not f.f2_valid))
    r.add("I", "folds_out_of_hull", sum(1 for f in cv.folds if not f.in_hull))
    r.add("I", "notes", len(cv.notes))


def _equivalence(r: AuditReport, a: Analysis) -> None:
    from pipeline.equivalence.f2 import similarity_f2

    obs = a.observed_profiles
    r.add("J", "observed_profiles", len(obs))
    r.add("J", "grid_points", len(a.time_grid))
    r.add("J", "profiles_with_nan_grid_cells", sum(1 for v in obs.values() if np.isnan(v).any()))
    r.add("J", "nan_grid_cells_total", int(sum(int(np.isnan(v).sum()) for v in obs.values())))
    r.add(
        "J",
        "profiles_nan_at_last_grid_point",
        sum(1 for v in obs.values() if len(v) and np.isnan(v[-1])),
    )
    keys = sorted(obs)
    valid = invalid = lt3 = 0
    for i, k1 in enumerate(keys):
        for k2 in keys[i + 1 :]:
            res = similarity_f2(a.time_grid, obs[k1], obs[k2])
            if res.valid:
                valid += 1
            else:
                invalid += 1
                lt3 += int(res.n_points < config.F2_MIN_POINTS)
    r.add("J", "pairs_f2_valid", valid)
    r.add("J", "pairs_f2_invalid", invalid)
    r.add("J", "pairs_lt_min_points", lt3)
    sets = a.equivalence
    r.add("J", "sets", len(sets))
    r.add(
        "J",
        "sets_missing_self",
        sum(
            1
            for s in sets
            if not any((m.case, m.grade) == (s.target_case, s.target_grade) for m in s.members)
        ),
    )
    r.add("J", "sets_isolated", sum(1 for s in sets if s.freedom == 0))
    r.add("J", "sets_cross_grade", sum(1 for s in sets if s.spans_multiple_grades))
    r.add("J", "largest_set", max((s.n_members for s in sets), default=0))


def _stress(r: AuditReport, a: Analysis, stress: StressTest | None) -> StressTest:
    if stress is None:
        from pipeline.run import _stress as run_stress

        stress = run_stress(a)
    r.add("K", "full_rmse_finite", bool(np.isfinite(stress.full_profile_rmse_pct)))
    r.add("K", "candidate_sizes", len(stress.results))
    r.add("K", "estimable_sizes", sum(1 for s in stress.results if s.estimable))
    r.add(
        "K", "finite_rmse_sizes", sum(1 for s in stress.results if np.isfinite(s.profile_rmse_pct))
    )
    r.add("K", "direction_agrees_sizes", sum(1 for s in stress.results if s.lever_direction_agrees))
    r.add("K", "recommended_exists", stress.recommended is not None)
    r.add("K", "recommended_size", stress.recommended_size)
    return stress


def _doe(r: AuditReport, a: Analysis) -> None:
    from pipeline.doe.responses import RESPONSES, collect

    data = {d.spec.key: d for d in collect(a.design_points, a.replicates)}
    r.add("L", "responses_defined", len(RESPONSES))
    r.add("L", "responses_fitted", len(a.doe.responses))
    for spec in RESPONSES:
        k = spec.key
        d = data.get(k)
        fitted = a.doe.by_key(k)
        r.add("L", f"{k}_source_present", d is not None)
        r.add("L", f"{k}_points_available", int(d.available.sum()) if d is not None else 0)
        r.add("L", f"{k}_dropped_lt4_points", d is not None and not d.usable)
        r.add("L", f"{k}_fitted", fitted is not None)
        if fitted is None:
            continue
        r.add("L", f"{k}_usable", fitted.usable)
        r.add("L", f"{k}_residual_df", fitted.anova.residual_df)
        r.add("L", f"{k}_process_power", fitted.spec.process_power)
        r.add("L", f"{k}_interactions", len(fitted.interactions))
        r.add("L", f"{k}_stats_finite", bool(np.isfinite(fitted.fit.r_squared)))


def _formulator(r: AuditReport, a: Analysis) -> None:
    from pipeline.optimize.goals import GOALS

    fitted = {x.response.spec.key for x in a.doe.responses}
    r.add("M", "goals", len(GOALS))
    for g in GOALS:
        missing = sum(1 for c in g.components if c.response not in fitted)
        r.add("M", f"goal_{g.key}_responses_missing", missing)
    pts = a.design_points
    unique = pts.drop_duplicates("case")[["api_wt", "hpmc_wt"]].drop_duplicates()
    r.add("M", "unique_compositions_for_hull", len(unique))
    # The dashboard scans API and HPMC in 5-point steps and takes lactose as the
    # balance to 100. If the tested compositions do not sum to 100, no grid point
    # can land inside the lactose range and every goal comes back empty.
    lo = {c: float(pts[f"{c}_wt"].min()) for c in ("api", "hpmc", "lactose")}
    hi = {c: float(pts[f"{c}_wt"].max()) for c in ("api", "hpmc", "lactose")}
    n = 0
    a_val = lo["api"]
    while a_val <= hi["api"] + 1e-9:
        h_val = lo["hpmc"]
        while h_val <= hi["hpmc"] + 1e-9:
            lac = 100 - a_val - h_val
            if lo["lactose"] - 1e-9 <= lac <= hi["lactose"] + 1e-9:
                n += 1
            h_val += 5
        a_val += 5
    r.add("M", "scan_points_with_lactose_balance_in_range", n)


def _response_space(r: AuditReport, a: Analysis) -> None:
    s = a.response_space
    r.add("N", "metrics_offered", len(s.metrics))
    r.add("N", "key_responses", len(s.key_responses))
    r.add("N", "redundancy_groups", len(s.groups))
    r.add("N", "pca_rows", int(np.asarray(s.scores).shape[0]) if np.asarray(s.scores).ndim else 0)
    r.add("N", "two_dimensional", s.two_dimensional)


def _count_nulls(value: Any) -> tuple[int, int]:
    """(null leaves, total leaves) in a JSON-like structure."""
    if value is None:
        return 1, 1
    if isinstance(value, dict):
        pairs = [_count_nulls(v) for v in value.values()]
    elif isinstance(value, list):
        pairs = [_count_nulls(v) for v in value]
    else:
        return 0, 1
    return sum(p[0] for p in pairs), sum(p[1] for p in pairs)


def _payload(
    r: AuditReport, a: Analysis, stress: StressTest, payload: dict[str, Any] | None
) -> None:
    if payload is None:
        from pipeline.diagnostics import collect as collect_diagnostics
        from pipeline.export.data_js import build_payload

        payload = build_payload(a, stress, collect_diagnostics(a, stress))
    json.dumps(payload, allow_nan=False)
    r.add("O", "serialisable", True)
    for key in sorted(payload):
        if key == "audit" or not _LABEL_RE.match(key):
            continue
        nulls, total = _count_nulls(payload[key])
        r.add("O", f"{key}_leaves", total)
        r.add("O", f"{key}_nulls", nulls)
    profiles = payload.get("profiles", [])
    r.add("O", "profiles", len(profiles))
    diag = payload.get("diagnostics")
    if diag:
        r.add("O", "diagnostic_checks", len(diag["checks"]))
        r.add("O", "diagnostic_fail", sum(1 for c in diag["checks"] if c["status"] == "FAIL"))
        r.add("O", "diagnostic_warn", sum(1 for c in diag["checks"] if c["status"] == "WARN"))


# ------------------------------------------------------------------ entry point
def run_audit(
    path: str | Path,
    *,
    db: Database | None = None,
    analysis: Analysis | None = None,
    stress: StressTest | None = None,
    payload: dict[str, Any] | None = None,
    disintegration: str | Path | None = None,
) -> AuditReport:
    """Audit ``path``, reusing any objects a caller has already computed.

    ``disintegration`` is a separate workbook of disintegration data, when the
    data are not in a sheet of ``path``.
    """
    src = Path(path)
    report = AuditReport(commit=_commit())

    _run_stage(report, "A", lambda: _environment(report))
    _run_stage(report, "B", lambda: _preflight(report, src))

    if analysis is not None:
        db = analysis.db
    given = db
    db = _run_stage(report, "C", lambda: _ingest(report, src, given))
    if db is None:
        return report

    known_db: Database = db
    _run_stage(report, "D", lambda: _raw_values(report, known_db))
    _run_stage(report, "E", lambda: _time(report, known_db))
    reps = analysis.replicates if analysis is not None else None
    _run_stage(report, "F", lambda: _release(report, known_db, reps))
    _run_stage(report, "G", lambda: _design(report, known_db))

    if analysis is None:
        from pipeline.analysis import run_analysis

        analysis = _run_stage(report, "R", lambda: run_analysis(known_db))
    else:
        report.stage_ok["R"] = True
    if analysis is None:
        return report

    a: Analysis = analysis
    _run_stage(report, "H", lambda: _surfaces(report, a))
    _run_stage(report, "I", lambda: _cv(report, a))
    _run_stage(report, "J", lambda: _equivalence(report, a))
    stress = _run_stage(report, "K", lambda: _stress(report, a, stress))
    _run_stage(report, "L", lambda: _doe(report, a))
    _run_stage(report, "M", lambda: _formulator(report, a))
    _run_stage(report, "N", lambda: _response_space(report, a))
    # Imported here: the disintegration section imports this module's types.
    from pipeline.disintegration.audit import audit_disintegration

    dt_src = Path(disintegration) if disintegration is not None else src
    _run_stage(
        report,
        "T",
        lambda: audit_disintegration(report, dt_src, a, dedicated=disintegration is not None),
    )
    if stress is not None:
        s: StressTest = stress
        _run_stage(report, "O", lambda: _payload(report, a, s, payload))
    return report


#: Inside ``outputs/``, which .gitignore excludes, so a report about the real
#: database is never committed by accident.
DEFAULT_OUT = Path("outputs") / "reports" / "audit.txt"


def save(report: AuditReport, path: str | Path = DEFAULT_OUT) -> Path:
    """Write the rendered report to a text file, exactly as printed."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report.render() + "\n", encoding="utf-8", newline="\n")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m pipeline.audit",
        description=(
            "Privacy-safe audit: prints only booleans and integers, so the block "
            "can be shared without disclosing any data."
        ),
    )
    parser.add_argument("--input", required=True, help="path to the dissolution workbook")
    parser.add_argument(
        "--disintegration",
        metavar="FILE",
        help="a separate workbook holding the disintegration data, if any",
    )
    parser.add_argument(
        "--out",
        default=str(DEFAULT_OUT),
        help=f"text file to save the report to (default: {DEFAULT_OUT.as_posix()})",
    )
    args = parser.parse_args(argv)
    report = run_audit(args.input, disintegration=args.disintegration)
    print(report.render())
    print(f"\nSaved to {save(report, args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
