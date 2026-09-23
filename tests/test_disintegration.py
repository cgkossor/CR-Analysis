"""Disintegration section: loads what it should, finds what the generator hid.

The synthetic generator writes its true parameters next to the data, so the
analysis is tested on *recovering* them, not merely on running. The audit
section is held to the same integers-only grammar as the rest of the audit, and
the figures to the same journal conventions as the dissolution figures.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import openpyxl
import pytest
from PIL import Image

from pipeline.analysis import run_analysis
from pipeline.audit import REPORT_LINE, AuditReport
from pipeline.disintegration.analysis import run_disintegration
from pipeline.disintegration.audit import audit_disintegration
from pipeline.disintegration.load import load_disintegration
from pipeline.disintegration.synthetic import build_frames, generate
from pipeline.doe.analysis import run as run_doe
from pipeline.figures import publication as pub
from pipeline.io.load import load_database
from pipeline.io.schema import SchemaError

ROOT = Path(__file__).resolve().parents[1]


def _database() -> Path:
    matches = sorted(ROOT.glob("*.xlsx"))
    if not matches:
        pytest.skip("no database workbook present")
    return matches[0]


@pytest.fixture(scope="module")
def workbook(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return generate(_database(), tmp_path_factory.mktemp("dt") / "with_dt.xlsx")


@pytest.fixture(scope="module")
def analysis(workbook: Path) -> Any:
    return run_analysis(load_database(workbook))


@pytest.fixture(scope="module")
def result(workbook: Path, analysis: Any) -> Any:
    data = load_disintegration(workbook)
    assert data is not None
    return run_disintegration(analysis, data)


def _sheet(tmp: Path, header: list[str], rows: list[list[object]]) -> Path:
    wb = openpyxl.Workbook()
    ws = wb.active
    assert ws is not None
    ws.title = "Disintegration"
    ws.append(header)
    for row in rows:
        ws.append(row)
    path = tmp / "dt.xlsx"
    wb.save(path)
    return path


_BASE = ["ID", "Case", "API", "HPMC Grade", "API [wt%]", "HPMC [wt%]", "Lactose [wt%]"]


# --- generator and loader ---------------------------------------------------
def test_generator_is_deterministic() -> None:
    a, truth_a = build_frames(_database())
    b, truth_b = build_frames(_database())
    assert a.equals(b) and truth_a == truth_b
    c, _ = build_frames(_database(), seed=1)
    assert not a.equals(c)


def test_synthetic_round_trips_through_the_loader(workbook: Path) -> None:
    data = load_disintegration(workbook)
    assert data is not None
    assert data.is_synthetic and data.truth
    assert data.unit_code == 2  # minutes
    assert data.test_end_from_sheet and data.test_end_h == pytest.approx(24.0)
    per = data.long.groupby(["case", "grade"]).size()
    assert per.min() >= 3 and per.max() <= 4
    assert data.long["censored"].any()


def test_workbook_without_the_sheet_returns_none() -> None:
    assert load_disintegration(_database()) is None


def test_units_blanks_and_censoring(tmp_path: Path) -> None:
    path = _sheet(
        tmp_path,
        [*_BASE, "DT_1 [s]", "DT_2 [s]", "DT_3 [s]", "Test_end [s]"],
        [
            ["A", 1, "X", "K4M", 20, 40, 40, 3600, 7200, None, 36000],
            ["B", 2, "X", "K4M", 30, 40, 30, ">36000", 36000, 1800, 36000],
        ],
    )
    data = load_disintegration(path)
    assert data is not None
    a = data.long[data.long["id"] == "A"]
    assert list(a["dt_h"]) == pytest.approx([1.0, 2.0])  # the blank third replicate is not run
    b = data.long[data.long["id"] == "B"]
    assert list(b["censored"]) == [True, True, False]  # '>' mark, and a value at the test end
    assert b["dt_h"].max() == pytest.approx(10.0)


@pytest.mark.parametrize(
    ("header", "rows", "code"),
    [
        ([*_BASE, "DT_1 [min]"], [["A", 1, "X", "K4M", 20, 40, 40, 5],
                                  ["A", 2, "X", "K4M", 20, 40, 40, 6]], 107),
        ([*_BASE, "DT_1"], [["A", 1, "X", "K4M", 20, 40, 40, 5]], 105),
        ([*_BASE, "DT_1 [min]", "DT_2 [h]"], [["A", 1, "X", "K4M", 20, 40, 40, 5, 1]], 106),
        (_BASE, [["A", 1, "X", "K4M", 20, 40, 40]], 104),
    ],
)
def test_schema_errors_carry_codes(
    tmp_path: Path, header: list[str], rows: list[list[object]], code: int
) -> None:
    with pytest.raises(SchemaError) as err:
        load_disintegration(_sheet(tmp_path, header, rows))
    assert err.value.code == code


# --- recovery of the planted structure ---------------------------------------
def test_structural_parameters_are_recovered(result: Any) -> None:
    assert len(result.recovery) == 7
    missed = [r.name for r in result.recovery if not r.covered]
    assert not missed, f"truth outside the 95% CI for {missed}"


def test_dt_tracks_dissolution(result: Any) -> None:
    c = result.correlation
    assert c.by_key("t80").spearman >= 0.8
    assert all(w.pearson >= 0.9 for w in c.within_grade)
    # Pooled correlation is lower than within-grade: grades sit on offset lines.
    assert c.by_key("td_h").pearson < min(w.pearson for w in c.within_grade)
    assert all(x.sign_as_expected in (None, True) for x in c.table)


def test_grade_compresses_disintegration_relative_to_release(result: Any) -> None:
    g = result.grades
    assert g.n_complete_blocks >= 5
    assert g.compression < 0.8
    assert g.divergence_index < 1.0
    # Ordered by viscosity for both DT and Td at matched composition.
    for block in (g.block_dt, g.block_td):
        e = [block.effects[k] for k in result.grade_order]
        assert e == sorted(e)
    a = g.ancova
    offsets = [ln.offset for ln in a.lines]
    assert offsets == sorted(offsets, reverse=True)
    assert a.p_common_offset < 0.001


def test_planted_outlier_is_flagged(result: Any) -> None:
    truth = result.data.truth
    hits = [
        f for f in result.precision.flags
        if f.case == int(truth["outlier_case"]) and f.replicate == int(truth["outlier_replicate"])
    ]
    assert hits
    assert result.precision.icc > 0.9


def test_recipe_predicts_better_than_dissolution_alone(result: Any) -> None:
    by = {m.key: m for m in result.loo}
    assert by["recipe"].q2 > by["dissolution"].q2
    assert by["dissolution_grade"].q2 > by["dissolution"].q2
    assert all(np.isfinite(m.predicted).all() for m in result.loo)


def test_censored_points_are_excluded_and_counted(result: Any) -> None:
    n_cens = len(result.censored_points)
    assert n_cens > 0
    assert result.dt_fit.response.n_missing == n_cens
    assert result.correlation.n_censored_excluded == n_cens


# --- the shared DoE entry point ----------------------------------------------
def test_doe_default_responses_unchanged(analysis: Any) -> None:
    dp = analysis.design_points
    power = min(max(int(dp["v_coded"].nunique()) - 1, 0), 2)
    again = run_doe(dp, analysis.replicates, power)
    assert [r.response.spec.key for r in again.responses] == [
        r.response.spec.key for r in analysis.doe.responses
    ]
    for a, b in zip(again.responses, analysis.doe.responses, strict=True):
        assert a.anova.r_squared == pytest.approx(b.anova.r_squared)


def test_doe_on_dt_is_estimable(result: Any) -> None:
    for key in ("dt_h", "log10_dt"):
        ra = result.doe.by_key(key)
        assert ra is not None and ra.usable


# --- audit ------------------------------------------------------------------
def test_audit_section_is_integers_only_and_leaks_nothing(workbook: Path, analysis: Any,
                                                          result: Any) -> None:
    report = AuditReport()
    audit_disintegration(report, workbook, analysis)
    assert report.get("sheet_present") == 1
    assert report.get("recovery_covered") == report.get("recovery_total") == 7
    text = report.render()
    for line in text.splitlines():
        assert REPORT_LINE.match(line), line
    forbidden = {*result.grade_order, *result.data.long["id"].unique(),
                 *result.data.long["api"].unique(), "Disintegration"}
    for word in forbidden:
        assert word not in text


def test_audit_section_without_the_sheet(analysis: Any) -> None:
    report = AuditReport()
    audit_disintegration(report, _database(), analysis)
    assert [e.label for e in report.entries] == ["sheet_present"]
    assert report.get("sheet_present") == 0


# --- figures ----------------------------------------------------------------
def test_figures_are_journal_styled(result: Any, tmp_path: Path) -> None:
    from pipeline.disintegration.figures import render_figures

    pub.STRICT = True  # a title or suptitle raises inside save()
    try:
        records = render_figures(result, tmp_path, formats=("png", "pdf"))
    finally:
        pub.STRICT = False
    ids = [r.id for r in records]
    assert ids == [f"DT-0{i}" for i in range(1, 9)]
    assert (tmp_path / "captions.json").exists() and (tmp_path / "captions.md").exists()
    for rec in records:
        png = tmp_path / rec.file
        assert png.exists() and png.with_suffix(".pdf").exists()
        with Image.open(png) as im:
            dpi = im.info.get("dpi", (0, 0))
        assert round(dpi[0]) == 300
        assert rec.caption and "nan" not in rec.caption.lower()


def test_run_section_skips_without_the_sheet(analysis: Any, tmp_path: Path) -> None:
    from pipeline.disintegration.__main__ import run_section

    assert run_section(analysis, _database(), tmp_path, figures=False) is None
    assert not (tmp_path / "reports").exists()
