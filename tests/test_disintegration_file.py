"""Disintegration data in a separate workbook, one row per tablet.

The lab records disintegration in its own file, not in a sheet of the
dissolution workbook, and in a different shape: one row per tablet with a
``Replicate`` column, the time split across hour / minute / second columns
that add up, extra measurements (mass, thickness) alongside, and headers spelt
"Disentegration". This file is re-laid from the synthetic generator's
Disintegration sheet, so the per-tablet reader can be checked against the
original reader on identical data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pipeline.analysis import run_analysis
from pipeline.audit import REPORT_LINE, run_audit
from pipeline.diagnostics import collect
from pipeline.disintegration.load import load_disintegration
from pipeline.disintegration.synthetic import generate
from pipeline.io.load import load_database
from pipeline.io.schema import SchemaError
from pipeline.run import _disintegration, _payload, _stress

ROOT = Path(__file__).resolve().parents[1]

HR, MIN, SEC = (
    "Disentegration Time [hr]", "Disentegration Time [min]", "Disentegration Time [sec]"
)


def _database() -> Path:
    matches = sorted(ROOT.glob("*.xlsx"))
    if not matches:
        pytest.skip("no database workbook present")
    return matches[0]


def _per_tablet(wide_path: Path) -> pd.DataFrame:
    """The synthetic Disintegration sheet as one row per tablet, H:M:S split."""
    data = load_disintegration(wide_path)
    assert data is not None
    rows = []
    for r in data.long.itertuples():
        seconds = round(r.dt_h * 3600)
        h, rem = divmod(seconds, 3600)
        m, s = divmod(rem, 60)
        rows.append({
            "ID": r.id, "Case": r.case, "API": r.api, "HPMC Grade": r.grade,
            "API [wt%]": r.api_wt, "HPMC [wt%]": r.hpmc_wt, "Lactose [wt%]": r.lactose_wt,
            "Replicate": r.replicate, "Mass [mg]": 300.0, "Thickness [mm]": 4.2,
            HR: f">{h}" if r.censored else h, MIN: m, SEC: s,
        })
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def files(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    tmp = tmp_path_factory.mktemp("dtfile")
    wide = generate(_database(), tmp / "with_dt.xlsx")
    dt_file = tmp / "disintegration.xlsx"
    with pd.ExcelWriter(dt_file, engine="openpyxl") as writer:
        _per_tablet(wide).to_excel(writer, sheet_name="Sheet1", index=False)
    return wide, dt_file


class TestPerTabletLayout:
    def test_reads_the_first_sheet_of_a_dedicated_file(self, files: tuple[Path, Path]) -> None:
        _, dt_file = files
        assert load_disintegration(dt_file) is None  # not a Disintegration sheet name
        data = load_disintegration(dt_file, dedicated=True)
        assert data is not None and data.per_tablet_rows

    def test_same_times_as_the_original_layout(self, files: tuple[Path, Path]) -> None:
        wide, dt_file = files
        a = load_disintegration(wide)
        b = load_disintegration(dt_file, dedicated=True)
        assert a is not None and b is not None
        keys = ["id", "replicate"]
        left = a.long.sort_values(keys).reset_index(drop=True)
        right = b.long.sort_values(keys).reset_index(drop=True)
        assert list(left["id"]) == list(right["id"])
        # H:M:S is rounded to the second.
        np.testing.assert_allclose(right["dt_h"], left["dt_h"], atol=1 / 3600)
        assert list(right["censored"]) == list(left["censored"])

    def test_extra_measurements_are_not_flagged(self, files: tuple[Path, Path]) -> None:
        data = load_disintegration(files[1], dedicated=True)
        assert data is not None and data.n_unrecognised_columns == 0

    def test_two_columns_in_one_unit_are_refused(self, tmp_path: Path) -> None:
        frame = pd.DataFrame({
            "ID": ["a"], "Case": [1], "HPMC Grade": ["K4M"], "Replicate": [1],
            "Disintegration Time [min]": [5], "DT [min]": [6],
        })
        path = tmp_path / "bad.xlsx"
        frame.to_excel(path, index=False)
        with pytest.raises(SchemaError) as err:
            load_disintegration(path, dedicated=True)
        assert err.value.code == 106


class TestTwoFileRun:
    def test_payload_and_audit_take_the_separate_file(
        self, files: tuple[Path, Path]
    ) -> None:
        wide, dt_file = files
        # The dissolution data come from the workbook, the disintegration data
        # from their own file.
        analysis = run_analysis(load_database(wide))
        stress = _stress(analysis)
        dt = _disintegration(analysis, str(wide), str(dt_file))
        assert dt is not None
        payload, audit = _payload(
            analysis, stress, collect(analysis, stress), str(wide), dt, str(dt_file)
        )
        assert payload["disintegration"]["summary"]["formulations"] == 33
        assert any(r["key"] == "dt_h" for r in payload["doe"]["responses"])
        assert audit.get("separate_file") is True
        assert audit.get("per_tablet_rows") is True
        assert audit.stage_ok["T"] is True
        assert all(REPORT_LINE.match(line) for line in audit.render().splitlines())

    def test_audit_cli_path(self, files: tuple[Path, Path]) -> None:
        report = run_audit(_database(), disintegration=files[1])
        assert report.get("sheet_present") is True
        assert report.get("formulations") == 33
