"""In-situ probe data: a clock per replicate, dense logging, runs past 24 h.

The placeholder was built like a manual-pull study: 25 shared timepoints, one
elapsed-time column, every run ending at 24 h. The real database comes off
fibre-optic probes, and differs in three ways that each broke something
silently:

* every replicate has its own time column (``Min_1``, ``Min_2``, ``Min_3``), and
  only the first was read, so replicates 2 and 3 were timed by replicate 1's
  clock;
* hundreds of readings per run, on schedules that changed with the method, so
  the recovered comparison grid had hundreds of points;
* some runs continue past 24 h for diagnostics, which stretched that grid and
  left most profiles blank at its end -- blanking the stress test.

The fixture below re-lays the placeholder into that shape. It is a test fixture
only: values are interpolated from the placeholder's own profiles (held at
their 24 h value beyond it) and never reach the pipeline's data path.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pipeline import config
from pipeline.analysis import Analysis, run_analysis
from pipeline.audit import REPORT_LINE, AuditReport, run_audit
from pipeline.io.load import Database, load_database
from pipeline.profiles.grid import project_onto_grid
from pipeline.run import _stress

ROOT = Path(__file__).resolve().parents[1]

STEP_MIN = 10.0
END_H = 30.0
#: Each probe logs a little later than the one before, so the clocks differ.
#: All start at t = 0, as every profile in the real database does.
OFFSET_MIN = (0.0, 0.7, 1.4)


def _source() -> Path:
    matches = sorted(ROOT.glob("*.xlsx"))
    if not matches:
        pytest.skip("no database workbook present")
    return matches[0]


def _probe_layout(raw: pd.DataFrame) -> pd.DataFrame:
    """The placeholder, re-laid as per-replicate probe logs out to 30 h."""
    base = np.arange(0.0, END_H * 60 + 1e-9, STEP_MIN)
    blocks: list[pd.DataFrame] = []
    for _, g in raw.groupby("ID", sort=False):
        g = g.sort_values("Min_1")
        block = pd.DataFrame(
            {
                c: g[c].iloc[0]
                for c in (
                    "ID",
                    "Case",
                    "API",
                    "HPMC Grade",
                    "API [wt%]",
                    "HPMC [wt%]",
                    "Lactose [wt%]",
                )
            },
            index=range(len(base)),
        )
        for r, offset in enumerate(OFFSET_MIN, start=1):
            t = np.where(base == 0, 0.0, base + offset)
            conc = np.interp(t, g["Min_1"].to_numpy(float), g[f"conc_{r} [ug_ml]"].to_numpy(float))
            block[f"Date_{r}"] = g[f"Date_{r}"].iloc[0]
            block[f"Probe_num_{r}"] = r
            block[f"Mass_{r}_mg"] = g[f"Mass_{r}_mg"].iloc[0]
            block[f"Basket_{r}"] = r
            block[f"Min_{r}"] = t
            block[f"conc_{r} [ug_ml]"] = conc
            block[f"AU_{r}"] = conc / 100.0
        blocks.append(block)
    return pd.concat(blocks, ignore_index=True)


@pytest.fixture(scope="module")
def workbook(tmp_path_factory: pytest.TempPathFactory) -> Path:
    raw = pd.read_excel(_source(), sheet_name="Dissolution")
    design = pd.read_excel(_source(), sheet_name="Design", header=None)
    path = tmp_path_factory.mktemp("probe") / "probe_layout.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        _probe_layout(raw).to_excel(writer, sheet_name="Dissolution", index=False)
        design.to_excel(writer, sheet_name="Design", index=False, header=False)
    return path


@pytest.fixture(scope="module")
def db(workbook: Path) -> Database:
    return load_database(workbook)


@pytest.fixture(scope="module")
def analysis(db: Database) -> Analysis:
    return run_analysis(db)


class TestPerReplicateClocks:
    def test_each_replicate_reads_its_own_time_column(self, db: Database) -> None:
        assert db.schema is not None
        assert [r.time for r in db.schema.replicates] == ["Min_1", "Min_2", "Min_3"]

    def test_replicate_times_are_not_replicate_ones(self, db: Database) -> None:
        first_id = db.profiles["id"].iloc[0]
        p = db.profiles[db.profiles["id"] == first_id]
        starts = p.groupby("replicate")["time_h"].min().to_numpy()
        np.testing.assert_allclose(starts, 0.0)
        second = p.groupby("replicate")["time_h"].apply(lambda s: sorted(s)[1]).to_numpy()
        np.testing.assert_allclose((second * 60) - STEP_MIN, OFFSET_MIN, atol=1e-9)

    def test_shared_clock_layout_is_unchanged(self) -> None:
        placeholder = load_database(_source())
        assert placeholder.schema is not None
        assert all(r.time is None for r in placeholder.schema.replicates)


class TestAnalysisWindow:
    def test_readings_past_the_window_are_dropped(self, db: Database) -> None:
        assert db.rows_beyond_window > 0
        last = db.profiles.groupby(["id", "replicate"])["time_h"].max()
        # At most one reading past the window survives, to bracket 24 h.
        assert (last <= config.ANALYSIS_WINDOW_H + STEP_MIN / 60 + 1e-9).all()

    def test_24h_value_survives_trimming(self, analysis: Analysis) -> None:
        assert np.isfinite(analysis.replicates["pct_24h"]).all()


class TestNominalSchedule:
    def test_dense_data_are_resampled_onto_the_nominal_schedule(self, analysis: Analysis) -> None:
        grid = analysis.time_grid_info
        assert grid.resampled
        np.testing.assert_allclose(analysis.time_grid, config.NOMINAL_SCHEDULE_H)

    def test_no_profile_has_gaps_on_the_grid(self, analysis: Analysis) -> None:
        for curve in analysis.observed_profiles.values():
            assert np.isfinite(curve).all()

    def test_mean_is_taken_after_resampling_each_replicate(
        self, db: Database, analysis: Analysis
    ) -> None:
        key = sorted(analysis.observed_profiles)[0]
        sub = db.profiles[(db.profiles["case"] == key[0]) & (db.profiles["grade"] == key[1])]
        expected = np.mean(
            [
                project_onto_grid(
                    g["time_h"].to_numpy(float),
                    g["pct_released"].to_numpy(float),
                    analysis.time_grid,
                )
                for _, g in sub.groupby("replicate")
            ],
            axis=0,
        )
        np.testing.assert_allclose(analysis.observed_profiles[key], expected)

    def test_gap_guard_blanks_rather_than_bridges(self) -> None:
        t = np.array([0.0, 1.0, 5.0])
        out = project_onto_grid(t, t * 10, np.array([0.5, 3.0]), max_gap_h=1.0)
        assert out[0] == pytest.approx(5.0)
        assert np.isnan(out[1])


class TestStressTest:
    def test_full_design_error_is_computed(self, analysis: Analysis) -> None:
        stress = _stress(analysis)
        assert np.isfinite(stress.full_profile_rmse_pct)
        assert any(np.isfinite(r.profile_rmse_pct) for r in stress.results)

    def test_missing_grid_cells_do_not_blank_the_error(self, analysis: Analysis) -> None:
        holed = dict(analysis.observed_profiles)
        key = next(iter(holed))
        curve = holed[key].copy()
        curve[-3:] = np.nan
        holed[key] = curve
        analysis_holed = Analysis(**{**analysis.__dict__, "observed_profiles": holed})
        assert np.isfinite(_stress(analysis_holed).full_profile_rmse_pct)


class TestAudit:
    @pytest.fixture(scope="class")
    def report(self, workbook: Path) -> AuditReport:
        return run_audit(workbook)

    def test_reports_the_probe_layout(self, report: AuditReport) -> None:
        assert all(REPORT_LINE.match(line) for line in report.render().splitlines())
        assert report.get("time_cols_indexed") == 3
        assert report.get("time_indices_match_conc") is True
        assert report.get("time_per_replicate") is True
        assert report.get("ids_where_replicate_clocks_differ") == 33
        assert report.get("grid_resampled_to_nominal") is True
        rows_dropped = report.get("rows_beyond_window_dropped")
        assert isinstance(rows_dropped, int) and rows_dropped > 0
        assert report.get("full_rmse_finite") is True
        assert report.get("profiles_with_nan_grid_cells") == 0
