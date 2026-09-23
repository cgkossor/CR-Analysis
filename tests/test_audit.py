"""The privacy-safe audit must never disclose data, and must survive bad data.

The audit exists so a report about the *real* database can leave the machine it
lives on. Two properties make that safe and useful, and both are tested here:

* **Nothing but integers and flags.** Every line of the rendered report must
  match a strict grammar with no room for a value, a name or a header, and the
  known sensitive strings of the workbook must not appear anywhere in it.
* **A crash is located, not propagated.** Workbooks broken in the ways real
  spreadsheets break -- merged cells, a blank replicate, text in a numeric cell,
  a short study -- must still produce a complete report that says where.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pipeline.audit import REPORT_LINE, AuditReport, main, module_index, run_audit

ROOT = Path(__file__).resolve().parents[1]


def _source() -> Path:
    matches = sorted(ROOT.glob("*.xlsx"))
    if not matches:
        pytest.skip("no database workbook present")
    return matches[0]


def _write(path: Path, dissolution: pd.DataFrame, design: pd.DataFrame) -> Path:
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        dissolution.to_excel(writer, sheet_name="Dissolution", index=False)
        design.to_excel(writer, sheet_name="Design", index=False, header=False)
    return path


def _assert_grammar(text: str) -> None:
    lines = text.splitlines()
    bad = [line for line in lines if not REPORT_LINE.match(line)]
    assert not bad, f"lines outside the audit grammar: {bad[:5]}"
    assert int(lines[-1].split("lines=")[1].split()[0]) == len(lines) - 2


@pytest.fixture(scope="module")
def raw() -> pd.DataFrame:
    return pd.read_excel(_source(), sheet_name="Dissolution")


@pytest.fixture(scope="module")
def design_sheet() -> pd.DataFrame:
    return pd.read_excel(_source(), sheet_name="Design", header=None)


@pytest.fixture(scope="module")
def report() -> AuditReport:
    return run_audit(_source())


class TestTypeGuard:
    """The privacy guarantee is structural: the report refuses non-integers."""

    @pytest.mark.parametrize("value", [1.5, float("nan"), "K4M", None, np.float64(2.0)])
    def test_rejects_anything_but_int_or_bool(self, value: object) -> None:
        with pytest.raises(TypeError):
            AuditReport().add("A", "x", value)

    @pytest.mark.parametrize("label", ["API_1", "has space", "1abc", "grade=K4M", ""])
    def test_rejects_labels_that_are_not_fixed_identifiers(self, label: str) -> None:
        with pytest.raises(ValueError):
            AuditReport().add("A", label, 1)

    def test_accepts_numpy_integers_and_bools(self) -> None:
        r = AuditReport()
        r.add("A", "n", np.int64(3))
        r.add("A", "flag", np.bool_(True))
        assert [e.value for e in r.entries] == [3, True]


class TestPlaceholderReport:
    def test_every_stage_completes(self, report: AuditReport) -> None:
        failed = [k for k, ok in report.stage_ok.items() if not ok]
        assert not failed, f"stages failed on the placeholder: {failed}"
        assert set(report.stage_ok) >= set("ABCDEFGRHIJKLMNO")

    def test_output_follows_the_grammar(self, report: AuditReport) -> None:
        _assert_grammar(report.render())

    def test_no_workbook_content_leaks(self, report: AuditReport, raw: pd.DataFrame) -> None:
        text = report.render()
        sensitive: set[str] = {_source().name, _source().stem}
        sensitive |= {str(c) for c in raw.columns}
        for col in ("ID", "API", "HPMC Grade"):
            sensitive |= {str(v) for v in raw[col].dropna().unique()}
        leaked = sorted(s for s in sensitive if len(s) >= 3 and s in text)
        assert not leaked, f"workbook content appears in the audit: {leaked}"

    def test_reflects_the_known_structure(self, report: AuditReport) -> None:
        assert report.get("n_ids") == 33
        assert report.get("n_grades") == 3
        assert report.get("replicates_per_id_min") == 3
        assert report.get("profiles_ending_before_24h") == 0
        assert report.get("composition_like_cols_unrecognised") == 0
        assert report.get("recommended_exists") is True

    def test_payload_is_json_ready_and_ordered(self, report: AuditReport) -> None:
        payload = report.as_payload()
        assert [s[0] for s in payload["sections"]][:3] == ["A", "B", "C"]
        assert all(isinstance(e["value"], (bool, int)) for e in payload["entries"])


class TestBrokenWorkbooks:
    """Real spreadsheets break in these ways; each must be reported, not fatal."""

    def test_merged_cells_are_counted_and_ingest_failure_located(
        self, tmp_path: Path, raw: pd.DataFrame, design_sheet: pd.DataFrame
    ) -> None:
        frame = raw.copy().astype(object)
        later = frame["ID"].duplicated()
        for col in ("ID", "Case", "API", "HPMC Grade", "Mass_1_mg"):
            frame.loc[later, col] = None
        rep = run_audit(_write(tmp_path / "merged.xlsx", frame, design_sheet))
        _assert_grammar(rep.render())
        assert rep.get("rows_with_time_but_blank_id") == int(later.sum())
        assert rep.get("mass_partially_filled_id_reps") == 0  # blank ID cannot group
        assert rep.stage_ok["B"] is True

    def test_blank_replicate_crash_is_located(
        self, tmp_path: Path, raw: pd.DataFrame, design_sheet: pd.DataFrame
    ) -> None:
        frame = raw.copy()
        frame.loc[frame["ID"] == frame["ID"].iloc[0], "conc_3 [ug_ml]"] = np.nan
        rep = run_audit(_write(tmp_path / "blank.xlsx", frame, design_sheet))
        _assert_grammar(rep.render())
        assert rep.get("replicates_entirely_blank") == 1
        assert rep.get("profiles_with_zero_finite_points") == 1
        if rep.stage_ok.get("R") is False:
            mod = rep.get("error_frame1_module")
            assert isinstance(mod, int) and 0 <= mod < len(module_index())
            assert isinstance(rep.get("error_frame1_line"), int)

    def test_text_in_a_numeric_cell(
        self, tmp_path: Path, raw: pd.DataFrame, design_sheet: pd.DataFrame
    ) -> None:
        frame = raw.copy().astype({"conc_1 [ug_ml]": object})
        frame.loc[5, "conc_1 [ug_ml]"] = "<LOQ"
        rep = run_audit(_write(tmp_path / "text.xlsx", frame, design_sheet))
        text = rep.render()
        _assert_grammar(text)
        assert "LOQ" not in text
        assert rep.get("conc_nonnumeric_cells") == 1

    def test_schema_rejection_reports_its_rule(
        self, tmp_path: Path, raw: pd.DataFrame, design_sheet: pd.DataFrame
    ) -> None:
        frame = raw.rename(columns={"Min_1": "Time"})
        rep = run_audit(_write(tmp_path / "unit.xlsx", frame, design_sheet))
        _assert_grammar(rep.render())
        assert rep.stage_ok["C"] is False
        assert rep.get("schema_error_code") == 22
        assert rep.get("time_unit_recognised") is False

    def test_extra_excipient_is_flagged(
        self, tmp_path: Path, raw: pd.DataFrame, design_sheet: pd.DataFrame
    ) -> None:
        frame = raw.copy()
        frame["Lactose [wt%]"] = frame["Lactose [wt%]"] - 5
        frame["MCC [wt%]"] = 5.0
        rep = run_audit(_write(tmp_path / "mcc.xlsx", frame, design_sheet))
        _assert_grammar(rep.render())
        assert rep.get("composition_like_cols_unrecognised") == 1
        assert rep.get("compositions_sum_off_100_gt_0p5") == rep.get("distinct_compositions")

    def test_short_study_is_flagged(
        self, tmp_path: Path, raw: pd.DataFrame, design_sheet: pd.DataFrame
    ) -> None:
        frame = raw[raw["Min_1"] <= 720].copy()
        rep = run_audit(_write(tmp_path / "short.xlsx", frame, design_sheet))
        _assert_grammar(rep.render())
        assert rep.get("profiles_ending_before_24h") == rep.get("profiles_with_data")
        assert rep.get("finite_pct_24h") == 0

    def test_missing_file(self, tmp_path: Path) -> None:
        rep = run_audit(tmp_path / "absent.xlsx")
        _assert_grammar(rep.render())
        assert rep.get("input_exists") is False
        assert rep.get("schema_error_code") == 1


def test_cli_saves_the_report_as_text(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "nested" / "audit.txt"
    assert main(["--input", str(_source()), "--out", str(out)]) == 0
    saved = out.read_text(encoding="utf-8")
    assert saved.startswith("=== CR-AUDIT") and saved.rstrip().endswith("===")
    assert saved.rstrip() in capsys.readouterr().out
    for line in saved.splitlines():
        assert REPORT_LINE.match(line), line
