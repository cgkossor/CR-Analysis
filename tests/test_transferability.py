"""The pipeline must survive a database swap without code changes.

The placeholder will be replaced by the real database, which will differ in
timepoint count, replicate count, API count, and possibly in column spelling and
units. This module builds structurally different workbooks and runs the whole
chain on each.

Two kinds of variant are used, and the distinction matters:

* **Subsets of the real workbook** -- fewer timepoints, fewer grades, fewer
  cases, renamed columns, rescaled units. These contain no invented values, only
  a re-shaped view of data that already exists.
* **Analytic fixtures** for the fit/metric layer, generated from a closed-form
  Weibull curve. These are test fixtures in the same sense as the hand-computed
  reference profiles AC2 mandates: they live only in ``tests/``, never reach the
  pipeline's data path, and never appear in any output.

Neither is used to fill a gap in the analysis dataset, which is what G2 forbids.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pipeline.analysis import run_analysis
from pipeline.io.load import load_database
from pipeline.io.schema import SchemaError, detect_schema
from pipeline.profiles.fits import fit_peppas, fit_weibull, weibull
from pipeline.profiles.metrics import extract_metrics

ROOT = Path(__file__).resolve().parents[1]


def _source() -> Path:
    matches = sorted(ROOT.glob("*.xlsx"))
    if not matches:
        pytest.skip("no database workbook present")
    return matches[0]


def _write(path: Path, dissolution: pd.DataFrame, design: pd.DataFrame | None = None) -> Path:
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        dissolution.to_excel(writer, sheet_name="Dissolution", index=False)
        if design is not None:
            design.to_excel(writer, sheet_name="Design", index=False, header=False)
    return path


@pytest.fixture(scope="module")
def raw() -> pd.DataFrame:
    return pd.read_excel(_source(), sheet_name="Dissolution")


@pytest.fixture(scope="module")
def design_sheet() -> pd.DataFrame:
    return pd.read_excel(_source(), sheet_name="Design", header=None)


class TestStructuralVariation:
    """Each variant must run end to end and self-describe correctly."""

    def test_fewer_timepoints(self, tmp_path: Path, raw: pd.DataFrame, design_sheet) -> None:
        keep = sorted(raw["Min_1"].unique())[::2]
        subset = raw[raw["Min_1"].isin(keep)].copy()
        path = _write(tmp_path / "fewer_times.xlsx", subset, design_sheet)

        analysis = run_analysis(load_database(path))
        assert analysis.quality.n_timepoints == len(keep)
        assert len(analysis.time_grid) == len(keep)
        assert analysis.quality.n_ids == 33
        assert np.isfinite(analysis.cross_validation.profile_rmse_pct)

    def test_two_grades_drops_the_quadratic_viscosity_term(
        self, tmp_path: Path, raw: pd.DataFrame, design_sheet
    ) -> None:
        """Three grades identify v^2; two identify only v. This must adapt itself."""
        subset = raw[raw["HPMC Grade"] != "K4M"].copy()
        path = _write(tmp_path / "two_grades.xlsx", subset, design_sheet)

        analysis = run_analysis(load_database(path))
        assert analysis.model_spec.process_power == 1, (
            "with two viscosity levels the quadratic process term is not identifiable "
            "and must be dropped automatically"
        )
        assert not any("v^2" in name for name in analysis.surfaces["log10_td"].kept_terms)
        assert analysis.quality.n_ids == 22

    def test_fewer_cases(self, tmp_path: Path, raw: pd.DataFrame, design_sheet) -> None:
        subset = raw[raw["Case"] <= 7].copy()
        path = _write(tmp_path / "fewer_cases.xlsx", subset, design_sheet)

        analysis = run_analysis(load_database(path))
        assert analysis.quality.cases_per_api["API_1"] == 7
        # The design-intent constant is used to report the shortfall, never to filter.
        assert analysis.quality.observed_cells == 21
        assert analysis.quality.expected_cells == 33
        assert analysis.quality.cases_missing_entirely == (8, 9, 10, 11)

    def test_two_replicates(self, tmp_path: Path, raw: pd.DataFrame, design_sheet) -> None:
        subset = raw.drop(columns=["Mass_3_mg", "conc_3 [ug_ml]", "Date_3"]).copy()
        path = _write(tmp_path / "two_reps.xlsx", subset, design_sheet)

        analysis = run_analysis(load_database(path))
        assert analysis.quality.n_replicates == 2
        assert len(analysis.replicates) == 33 * 2

    def test_renamed_columns_and_rescaled_units(
        self, tmp_path: Path, raw: pd.DataFrame, design_sheet
    ) -> None:
        """Different spelling and mg/mL instead of ug/mL must be handled, not guessed."""
        renamed = raw.rename(
            columns={
                "HPMC Grade": "Grade",
                "Min_1": "Time_min",
                "conc_1 [ug_ml]": "Concentration_1 [mg_ml]",
                "conc_2 [ug_ml]": "Concentration_2 [mg_ml]",
                "conc_3 [ug_ml]": "Concentration_3 [mg_ml]",
            }
        ).copy()
        for i in (1, 2, 3):
            renamed[f"Concentration_{i} [mg_ml]"] = renamed[f"Concentration_{i} [mg_ml]"] / 1000.0
        path = _write(tmp_path / "renamed.xlsx", renamed, design_sheet)

        analysis = run_analysis(load_database(path))
        # The unit conversion must cancel exactly: same concentrations, same answers.
        baseline = run_analysis(load_database(_source()))
        assert analysis.quality.max_pct_released == pytest.approx(
            baseline.quality.max_pct_released, rel=1e-9
        )

    def test_hours_instead_of_minutes(
        self, tmp_path: Path, raw: pd.DataFrame, design_sheet
    ) -> None:
        converted = raw.rename(columns={"Min_1": "Time [h]"}).copy()
        converted["Time [h]"] = converted["Time [h]"] / 60.0
        path = _write(tmp_path / "hours.xlsx", converted, design_sheet)

        analysis = run_analysis(load_database(path))
        baseline = run_analysis(load_database(_source()))
        assert np.allclose(analysis.time_grid, baseline.time_grid)


class TestLoudFailures:
    """AC1: malformed input fails with a specific message, never a silent guess."""

    def test_missing_concentration_unit_is_rejected(self, raw: pd.DataFrame) -> None:
        frame = raw.rename(columns={"conc_1 [ug_ml]": "conc_1"})
        with pytest.raises(SchemaError, match="concentration unit"):
            detect_schema(frame)

    def test_missing_time_unit_is_rejected(self, raw: pd.DataFrame) -> None:
        frame = raw.rename(columns={"Min_1": "Elapsed"})
        with pytest.raises(SchemaError, match="unit of the time column"):
            detect_schema(frame)

    def test_missing_composition_column_is_rejected(self, raw: pd.DataFrame) -> None:
        frame = raw.drop(columns=["Lactose [wt%]"])
        with pytest.raises(SchemaError, match="composition column"):
            detect_schema(frame)

    def test_unmapped_grade_is_rejected(self, tmp_path: Path, raw: pd.DataFrame) -> None:
        """A grade with no nominal viscosity cannot enter a log-viscosity model."""
        frame = raw.copy()
        frame.loc[frame["HPMC Grade"] == "K4M", "HPMC Grade"] = "K15M"
        path = _write(tmp_path / "unknown_grade.xlsx", frame)
        with pytest.raises(SchemaError, match="nominal viscosity"):
            load_database(path)


class TestHighResolutionProfiles:
    """A 200-point profile must work exactly like a 25-point one.

    Generated from a closed-form Weibull curve: a test fixture, not data.
    """

    @pytest.mark.parametrize("n_points", [25, 200, 1000])
    def test_fits_and_metrics_scale_with_resolution(self, n_points: int) -> None:
        t = np.linspace(0.0, 24.0, n_points)
        y = weibull(t, 92.0, 6.0, 0.68)

        fit = fit_weibull(t, y)
        assert fit.valid
        assert fit.params["f_inf"] == pytest.approx(92.0, rel=1e-3)
        assert fit.params["td"] == pytest.approx(6.0, rel=1e-3)
        assert fit.params["beta"] == pytest.approx(0.68, rel=1e-3)

        metrics = extract_metrics(t, y)
        assert metrics.n_points == n_points
        assert not metrics.release_times["t50"].censored
        # t50 is the time to 50% of the DOSE, not the Weibull median. With
        # F_inf = 92 those differ by ~20%, so the distinction is not academic:
        #   F(t) = 50  =>  t = Td * (-ln(1 - 50/F_inf))^(1/beta)
        expected = 6.0 * (-np.log(1 - 50.0 / 92.0)) ** (1 / 0.68)
        assert metrics.release_times["t50"].value == pytest.approx(expected, rel=5e-3)

    def test_peppas_window_scales_with_resolution(self) -> None:
        """More points below 60% must be used, not truncated to a fixed count."""
        t = np.linspace(0.01, 24.0, 400)
        y = weibull(t, 95.0, 8.0, 0.6)
        result = fit_peppas(t, y)
        assert result.valid
        assert result.n_points == int(((y > 0) & (y <= 60.0)).sum())
        assert result.n_points > 25, "the sub-60% window should grow with resolution"

    def test_mdt_converges_as_resolution_increases(self) -> None:
        """Unequal-spacing handling must not bias the integral."""
        values = []
        for n in (25, 200, 2000):
            t = np.linspace(0.0, 240.0, n)
            y = weibull(t, 100.0, 6.0, 1.0)
            values.append(extract_metrics(t, y).mdt_h)
        # For beta = 1 the true MDT is Td = 6 h.
        assert values[-1] == pytest.approx(6.0, rel=0.02)
        assert abs(values[-1] - 6.0) < abs(values[0] - 6.0)
