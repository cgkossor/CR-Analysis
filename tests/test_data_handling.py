"""Data-handling diagnostics: every removed or skipped reading is accounted for.

Uses the probe-layout fixture from ``test_dense_sampling`` with spikes planted
at known readings, so the report can be checked against ground truth: the
counts must add up, each planted spike must be listed, and the diagnostics and
figures must say so.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from test_dense_sampling import _probe_layout, _source

from pipeline import config
from pipeline.analysis import run_analysis
from pipeline.data_handling import build, write
from pipeline.diagnostics import collect
from pipeline.io.load import load_database

#: (row, replicate, multiplier) — each inside the analysis window.
PLANTED = ((40, 1, 1.25), (300, 2, 0.8), (301, 2, 0.8), (650, 3, 1.3))


@pytest.fixture(scope="module")
def spiked(tmp_path_factory: pytest.TempPathFactory) -> Path:
    raw = pd.read_excel(_source(), sheet_name="Dissolution")
    design = pd.read_excel(_source(), sheet_name="Design", header=None)
    dense = _probe_layout(raw)
    for row, rep, factor in PLANTED:
        dense.loc[row, f"conc_{rep} [ug_ml]"] *= factor
    path = tmp_path_factory.mktemp("spiked") / "spiked.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        dense.to_excel(writer, sheet_name="Dissolution", index=False)
        design.to_excel(writer, sheet_name="Design", index=False, header=False)
    return path


@pytest.fixture(scope="module")
def analysis(spiked: Path) -> Any:
    return run_analysis(load_database(spiked))


def test_every_planted_spike_is_removed_and_logged(analysis: Any) -> None:
    db = analysis.db
    assert db.spikes_removed == len(PLANTED)
    assert len(db.spike_log) == len(PLANTED)
    assert (db.spike_log["deviation"].abs() > db.spike_log["threshold"]).all()
    assert (db.spike_log["time_h"] <= config.ANALYSIS_WINDOW_H + 1e-9).all()


def test_reading_counts_add_up(analysis: Any) -> None:
    log = analysis.db.readings_log
    assert (log["loaded"] == log["kept"] + log["beyond_window"] + log["spikes_removed"]).all()
    assert int(log["kept"].sum()) == len(analysis.db.profiles)
    totals = build(analysis)["totals"]
    assert totals["readings_loaded"] == (
        totals["readings_analysed"] + totals["spikes_removed"] + totals["beyond_window_dropped"]
    )


def test_report_and_diagnostics_name_the_spikes(analysis: Any, tmp_path: Path) -> None:
    info = write(analysis, tmp_path)
    md = (tmp_path / "data_handling.md").read_text(encoding="utf-8")
    assert f"{len(PLANTED)} reading(s) removed" in md
    for rid in analysis.db.spike_log["id"].unique():
        assert rid in md
    stored = json.loads((tmp_path / "data_handling.json").read_text(encoding="utf-8"))
    assert len(stored["spikes"]) == len(PLANTED) == len(info["spikes"])
    checks = {c.name: c for c in collect(analysis).checks if c.section == "Data handling"}
    assert checks["spike readings removed"].value == len(PLANTED)


def test_placeholder_reports_no_spikes() -> None:
    a = run_analysis(load_database(_source()))
    info = build(a)
    assert info["totals"]["spikes_removed"] == 0
    assert info["spikes"] == []
    assert all(p["error_bars"] == p["points_plotted"] for p in info["plotting"])


def test_cox_traces_are_given_for_every_grade_and_say_they_are_a_model_slice() -> None:
    a = run_analysis(load_database(_source()))
    ra = a.doe.by_key("pct_24h")
    grades = sorted(a.design_points["grade"].unique())
    assert sorted(g for g, _ in ra.traces_by_grade) == grades
    assert ra.trace_grade in grades
    assert "reference formulation" in ra.takeaway_traces
    assert "not measured data" in ra.takeaway_traces
    spans = {
        g: max(np.ptp(t.y_values) for t in trs) for g, trs in ra.traces_by_grade
    }
    assert len(set(np.round(list(spans.values()), 3))) > 1
