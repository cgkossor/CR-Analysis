"""Metric extraction checked against hand-computed reference profiles (AC2).

Every expected value below is derived by hand in the docstring or comment that
accompanies it, so a failure points at the implementation rather than at a
previously-recorded output. These are synthetic *test fixtures*, deliberately
confined to the test suite: they never enter the pipeline's data path (G2).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pipeline.profiles.censoring import summarise_censored, time_to_percent
from pipeline.profiles.fits import fit_higuchi, fit_peppas, fit_weibull, weibull
from pipeline.profiles.metrics import (
    extract_metrics,
    mean_dissolution_time,
    percent_at,
)

# --- Reference profile 1: exactly linear, 25 %/h ---------------------------
# t : 0  1   2   3   4
# y : 0  25  50  75  100
LINEAR_T = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
LINEAR_Y = np.array([0.0, 25.0, 50.0, 75.0, 100.0])

# --- Reference profile 2: censored, never reaches 80 % ---------------------
CENSORED_T = np.array([0.0, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0])
CENSORED_Y = np.array([0.0, 10.0, 18.0, 30.0, 45.0, 55.0, 70.0])

# --- Reference profile 3: immediate release, fails G4 ----------------------
FAST_T = np.array([0.0, 1.0, 2.0, 3.0])
FAST_Y = np.array([0.0, 95.0, 98.0, 99.0])


class TestReleaseTimes:
    def test_linear_t50_is_exactly_two_hours(self) -> None:
        # y = 25t, so y = 50 at t = 2 exactly.
        result = time_to_percent(LINEAR_T, LINEAR_Y, 50.0)
        assert not result.censored
        assert result.value == pytest.approx(2.0)

    def test_linear_t10_interpolates_inside_first_interval(self) -> None:
        # y = 25t -> t10 = 10/25 = 0.4
        assert time_to_percent(LINEAR_T, LINEAR_Y, 10.0).value == pytest.approx(0.4)

    def test_linear_t80(self) -> None:
        # 80/25 = 3.2
        assert time_to_percent(LINEAR_T, LINEAR_Y, 80.0).value == pytest.approx(3.2)

    def test_censored_t80_is_censored_not_missing(self) -> None:
        # Peak is 70 %, so t80 is right-censored at the last observation, 24 h.
        result = time_to_percent(CENSORED_T, CENSORED_Y, 80.0)
        assert result.censored
        assert result.bound == pytest.approx(24.0)
        assert result.peak_pct == pytest.approx(70.0)
        assert str(result) == ">24"
        # G5: a censored value must not masquerade as a usable number.
        assert math.isnan(result.numeric_or_nan)

    def test_censored_t50_interpolates_between_8h_and_12h(self) -> None:
        # Crosses 50 % between (8, 45) and (12, 55): 8 + 5*4/10 = 10.0
        assert time_to_percent(CENSORED_T, CENSORED_Y, 50.0).value == pytest.approx(10.0)

    def test_censored_t25_interpolates_between_2h_and_4h(self) -> None:
        # Between (2, 18) and (4, 30): 2 + 7*2/12 = 3.166666...
        assert time_to_percent(CENSORED_T, CENSORED_Y, 25.0).value == pytest.approx(
            2.0 + 7.0 * 2.0 / 12.0
        )


class TestPercentAt:
    def test_interpolates_within_range(self) -> None:
        assert percent_at(LINEAR_T, LINEAR_Y, 1.5) == (pytest.approx(37.5), True)

    def test_refuses_to_extrapolate_beyond_last_observation(self) -> None:
        # G3: no silent extrapolation. 8 h is outside a 0-4 h profile.
        value, valid = percent_at(LINEAR_T, LINEAR_Y, 8.0)
        assert not valid
        assert math.isnan(value)


class TestMeanDissolutionTime:
    def test_linear_profile_mdt_is_two_hours(self) -> None:
        # dM = 25 each, t_mid = 0.5, 1.5, 2.5, 3.5
        # MDT = 25*(0.5+1.5+2.5+3.5)/100 = 2.0
        mdt, truncated, _ = mean_dissolution_time(LINEAR_T, LINEAR_Y)
        assert mdt == pytest.approx(2.0)
        assert not truncated

    def test_respects_unequal_spacing(self) -> None:
        """t = [0, 1, 10], y = [0, 50, 100].

        dM = [50, 50], t_mid = [0.5, 5.5]
        MDT = (0.5*50 + 5.5*50)/100 = 3.0

        Treating the grid as uniform would give 1.0, so this test discriminates
        against an implicit-uniform-spacing implementation.
        """
        mdt, _, _ = mean_dissolution_time(
            np.array([0.0, 1.0, 10.0]), np.array([0.0, 50.0, 100.0])
        )
        assert mdt == pytest.approx(3.0)
        assert mdt != pytest.approx(1.0)

    def test_still_rising_profile_is_flagged_truncated(self) -> None:
        mdt, truncated, note = mean_dissolution_time(CENSORED_T, CENSORED_Y)
        assert truncated
        assert "lower bound" in note
        assert math.isfinite(mdt)


class TestSlopes:
    def test_linear_early_slope_is_25_per_hour(self) -> None:
        m = extract_metrics(LINEAR_T, LINEAR_Y)
        assert m.early_slope == pytest.approx(25.0)
        assert m.early_slope_n == 3  # t = 0, 1, 2


class TestPeppasGuardrail:
    def test_refuses_when_fewer_than_three_points_below_60pct(self) -> None:
        # G4: the fast profile has zero observations in (0, 60] %.
        result = fit_peppas(FAST_T, FAST_Y)
        assert not result.valid
        assert "n" not in result.params
        assert "k" not in result.params
        assert "G4 requires" in result.note

    def test_recovers_known_power_law_exactly(self) -> None:
        # y = 20 * t^0.5, all points at or below 60 %.
        t = np.array([1.0, 2.0, 4.0, 9.0])
        y = 20.0 * t**0.5
        result = fit_peppas(t, y)
        assert result.valid
        assert result.params["n"] == pytest.approx(0.5, abs=1e-9)
        assert result.params["k"] == pytest.approx(20.0, abs=1e-9)
        assert result.n_points == 4

    def test_uses_only_the_sub60_window(self) -> None:
        # Points above 60 % must not enter the fit: appending a wild plateau
        # point leaves k and n untouched.
        t = np.array([1.0, 2.0, 4.0, 9.0])
        y = 20.0 * t**0.5
        base = fit_peppas(t, y)
        extended = fit_peppas(np.append(t, 25.0), np.append(y, 99.0))
        assert extended.n_points == base.n_points
        assert extended.params["n"] == pytest.approx(base.params["n"])


class TestWeibull:
    def test_recovers_known_parameters_with_free_asymptote(self) -> None:
        # F_inf deliberately below 100: a fit that pins the asymptote at 100
        # cannot pass this test.
        t = np.array([0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 18.0, 24.0])
        y = weibull(t, 88.0, 3.0, 0.7)
        result = fit_weibull(t, y)
        assert result.valid
        assert result.params["f_inf"] == pytest.approx(88.0, rel=1e-3)
        assert result.params["td"] == pytest.approx(3.0, rel=1e-3)
        assert result.params["beta"] == pytest.approx(0.7, rel=1e-3)
        assert result.r_squared == pytest.approx(1.0, abs=1e-6)

    def test_is_deterministic(self) -> None:
        # G10: repeated fits of the same profile give identical numbers.
        a = fit_weibull(CENSORED_T, CENSORED_Y)
        b = fit_weibull(CENSORED_T, CENSORED_Y)
        assert a.params == b.params


class TestHiguchi:
    def test_recovers_known_constant(self) -> None:
        t = np.array([1.0, 4.0, 9.0, 16.0])
        result = fit_higuchi(t, 15.0 * np.sqrt(t))
        assert result.params["k"] == pytest.approx(15.0)
        assert result.r_squared == pytest.approx(1.0, abs=1e-12)


class TestCensoredSummary:
    def test_median_is_censored_when_most_values_are(self) -> None:
        values = [
            time_to_percent(CENSORED_T, CENSORED_Y, 80.0),
            time_to_percent(CENSORED_T, CENSORED_Y, 80.0),
            time_to_percent(LINEAR_T, LINEAR_Y, 80.0),
        ]
        summary = summarise_censored(values)
        assert summary["n_censored"] == 2
        assert summary["median_display"] == ">24"

    def test_reports_counts_rather_than_dropping(self) -> None:
        values = [
            time_to_percent(LINEAR_T, LINEAR_Y, 80.0),
            time_to_percent(CENSORED_T, CENSORED_Y, 80.0),
        ]
        summary = summarise_censored(values)
        assert summary["n"] == 2
        assert summary["n_observed"] == 1
        assert summary["n_censored"] == 1


class TestEndToEndMetrics:
    def test_censored_profile_marks_t80(self) -> None:
        m = extract_metrics(CENSORED_T, CENSORED_Y)
        assert m.censored_at_80
        assert m.release_times["t80"].censored
        assert m.peak_pct == pytest.approx(70.0)

    def test_out_of_range_sample_times_are_invalid_not_zero(self) -> None:
        m = extract_metrics(LINEAR_T, LINEAR_Y)
        assert m.pct_at_valid["pct_1h"]
        assert not m.pct_at_valid["pct_8h"]
        assert math.isnan(m.pct_at["pct_8h"])
